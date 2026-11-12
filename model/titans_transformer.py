"""Minimal memory-as-context Titans: window attention + a Titans LMM, sized for
verification rather than capability.

GOAL. A transformer with real trainable recurrence whose per-step backward cost
is CONSTANT in the horizon, so it can run in real time: read a page in context
while remembering the book title from far earlier, or watch a screen in context
while remembering earlier frames.

WHY THIS IS CONSTANT TIME. The memory WRITE (arXiv:2501.00663 eqs 10-13) is a
delta rule evaluated entirely in the forward pass, so storing costs no gradient
through time. Three parameter groups then have three different credit costs:

  read side  (W_Q, attention, lm_head) -- q_t is computed at THIS tick, so the
             current loss gives exact gradient. O(1), nothing special needed.
  write side (W_K, W_V) -- their effect only surfaces in LATER reads, so exact
             credit is BPTT. This is the only group that needs an approximation.
  memory M   -- not a parameter. Written by the local rule, never by gradient.

So the write side is the whole difficulty, and `MODE` selects how it is paid for:

  "none"     no memory at all. The control: must score well in-context and at
             chance on recurrent questions, or the task is not measuring what it
             claims.
  "frozen"   memory on, write side left random and frozen. Still trains the READ
             side exactly, so recurrence is trainable and strictly O(1) with no
             approximation anywhere. Worth testing first because Gate A showed
             the LMM reaches 0.9975 recall with random frozen projections -- if
             this already lifts recurrent accuracy without costing in-context
             accuracy, the engineering goal is met with no BPTT at all.
  "trace"    write side trained from an eligibility trace, still O(1). S_t in
             eq 10 already IS such a trace, and
             bend_proofs/synaptic_hold_bptt_approximation bounds the error:
             exact when intermediate ticks are inactive, and dependent only on
             the number of ACTIVE ticks, never on the horizon.

The LMM dimension is tied to `hidden` so a memory read can be injected as one
extra attention row -- that is what "memory as context" means, and MAC is the
variant the paper finds best on needle-in-a-haystack.

Read-then-write ordering is load-bearing: a query token's own value is derived
from its key, so writing first would let a tick corrupt the association it is
about to retrieve.
"""

from __future__ import annotations

import numpy as np
from sili.tensor import Tensor, exp, gather, gaussian_attention

from model.toy_recall_models import DenseTensorLinear, rmsnorm_tensor

MODES = ("none", "frozen", "trace")


class TitansMAC:
    def __init__(
        self,
        vocab: int,
        hidden: int = 32,
        window: int = 16,
        *,
        mode: str = "frozen",
        n_layers: int = 2,
        mem_lr: float = 0.5,
        trace_beta: float = 0.9,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
        rng: np.random.Generator | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        rng = rng or np.random.default_rng()
        self.vocab, self.hidden, self.window = vocab, hidden, window
        self.mode, self.mem_lr, self.trace_beta = mode, mem_lr, trace_beta
        self.num_cpus, self.rms_eps = num_cpus, rms_eps
        self.n_rows = window + (0 if mode == "none" else 1)

        s = 1.0 / np.sqrt(hidden)
        self.embed = Tensor((rng.standard_normal((vocab, hidden)) * 0.3).astype(np.float32))
        self.input_ln = Tensor(np.ones(hidden, dtype=np.float32))
        self.post_ln = Tensor(np.ones(hidden, dtype=np.float32))
        # TWO LAYERS MINIMUM. Associative recall here is induction -- match the
        # query key against its earlier occurrence, then emit the token that
        # FOLLOWED it -- and a single attention layer provably cannot do that: it
        # needs a previous-token head feeding a match head. A 1-layer version of
        # this model sat exactly at chance on the in-context score.
        self.n_layers = n_layers
        self.blocks = [
            {
                "q": DenseTensorLinear(hidden, hidden, scale=s),
                "k": DenseTensorLinear(hidden, hidden, scale=s),
                "v": DenseTensorLinear(hidden, hidden, scale=s),
                "o": DenseTensorLinear(hidden, hidden, scale=s),
                "ln": Tensor(np.ones(hidden, dtype=np.float32)),
            }
            for _ in range(n_layers)
        ]
        self.lm_head = DenseTensorLinear(hidden, vocab, scale=s)
        # One center/sigma PER ATTENTION ROW, matching ToyTileRecurrence. A
        # single shared pair yields NaN, since the kernel indexes these per row.
        self.centers = Tensor(np.array([i + 0.5 for i in range(self.n_rows)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(self.n_rows, dtype=np.float32))

        # Memory read side trains exactly; write side depends on mode.
        self.W_Q_mem = DenseTensorLinear(hidden, hidden, scale=s)
        self.W_K_mem = (rng.standard_normal((hidden, hidden)) * s).astype(np.float32)
        self.W_V_mem = (rng.standard_normal((hidden, hidden)) * s).astype(np.float32)
        self.M = np.zeros((hidden, hidden), dtype=np.float32)
        # eligibility traces for the write side, same shapes as W_K/W_V
        self._tr_K = np.zeros_like(self.W_K_mem)
        self._tr_V = np.zeros_like(self.W_V_mem)
        self.last = {}

    def parameters(self) -> list[Tensor]:
        """Only Tensors here. W_K/W_V are numpy and updated out-of-band (frozen,
        or by the trace rule), which is exactly why the backward stays O(1)."""
        params = [self.embed, self.input_ln, self.post_ln, self.centers, self.log_sigmas]
        for blk in self.blocks:
            params.append(blk["ln"])
            for key in ("q", "k", "v", "o"):
                params += blk[key].parameters()
        params += self.lm_head.parameters()
        if self.mode != "none":
            params += self.W_Q_mem.parameters()
        return params

    def reset(self) -> None:
        """Clear memory at an episode boundary. Traces persist across episodes on
        purpose: they estimate a statistic of the task, not of one sequence."""
        self.M[:] = 0.0

    def _window_ids(self, tokens: np.ndarray, i: int) -> np.ndarray:
        lo = max(0, i - self.window + 1)
        ids = tokens[lo : i + 1]
        if len(ids) < self.window:  # left-pad by repeating the first token
            ids = np.concatenate([np.full(self.window - len(ids), ids[0]), ids])
        return ids.astype(np.int64)

    def step(self, tokens: np.ndarray, i: int) -> Tensor:
        """Logits for position i. Builds the window, reads memory as an extra
        context row, then writes. Returns [vocab] logits as a 1-row Tensor."""
        ids = self._window_ids(tokens, i)
        x = _gather_rows(self.embed, ids, self.hidden)
        x_n = rmsnorm_tensor(x, self.input_ln, self.rms_eps)

        if self.mode == "none":
            rows = x_n
        else:
            # READ FIRST (causality): q from the newest token's embedding.
            cur = _gather_rows(self.embed, ids[-1:], self.hidden)
            q_mem = self.W_Q_mem.forward(rmsnorm_tensor(cur, self.input_ln, self.rms_eps))
            read = Tensor(self.M) @ q_mem.reshape((self.hidden, 1))
            read = read.reshape((1, self.hidden))
            rows = _concat_rows(read, x_n)

        sigmas = exp(self.log_sigmas)
        h = rows
        for blk in self.blocks:
            hn = rmsnorm_tensor(h, blk["ln"], self.rms_eps)
            attn = gaussian_attention(
                blk["q"].forward(hn),
                blk["k"].forward(hn),
                blk["v"].forward(hn),
                self.centers,
                sigmas,
                num_cpus=self.num_cpus,
                causal=False,
            )
            h = h + blk["o"].forward(attn)  # residual, so depth composes
        h = rmsnorm_tensor(h, self.post_ln, self.rms_eps)
        logits = self.lm_head.forward(h)

        if self.mode != "none":
            self._write(h.data[-1])
        return logits.reshape((self.n_rows, self.vocab))

    def _write(self, h_last: np.ndarray) -> None:
        """One delta-rule write, forward pass only (eqs 11-13 with alpha=eta=0).

        h_last is the POST-ATTENTION hidden state, not a raw embedding: eq 11
        derives k and v from the same x_t, so a raw token would store key->key
        and recall exactly nothing. The attention core has to do the binding
        first -- that is why Titans is a hybrid."""
        h = h_last.astype(np.float32)
        k = h @ self.W_K_mem
        v = h @ self.W_V_mem
        kk = float(k @ k)
        if kk < 1e-8:
            return
        err = self.M @ k - v
        th = self.mem_lr / (kk + 1e-6)
        self.M = self.M - th * 2.0 * np.outer(err, k)
        if self.mode == "trace":
            # Eligibility trace over the write side: accumulate the outer
            # products that WOULD carry credit, so a later read can be charged
            # against them in O(1) instead of unrolling. cf. the Bend proof.
            self._tr_K = self.trace_beta * self._tr_K + np.outer(h, k)
            self._tr_V = self.trace_beta * self._tr_V + np.outer(h, v)
        self.last = {
            "surprise": float(np.linalg.norm(err)),
            "mem_norm": float(np.linalg.norm(self.M)),
        }

    def apply_trace_credit(self, read_grad: np.ndarray, lr: float) -> None:
        """Charge the current read's gradient against the write-side traces.

        O(1) in the horizon: one outer-product-sized update, no unrolling. Called
        by the trainer after backward, using the gradient that arrived at the
        memory-read row."""
        if self.mode != "trace":
            return
        g = np.asarray(read_grad, dtype=np.float32).ravel()
        if g.size != self.hidden or not np.any(g):
            return
        scale = lr / (float(np.linalg.norm(g)) + 1e-8)
        self.W_K_mem -= scale * self._tr_K * float(np.dot(g, g)) ** 0.5
        self.W_V_mem += scale * self._tr_V * float(np.dot(g, g)) ** 0.5


def _gather_rows(table: Tensor, ids: np.ndarray, width: int) -> Tensor:
    """Differentiable row lookup. A numpy index into `.data` would detach, which
    silently freezes the embedding table -- caught here by asserting every
    parameter receives gradient."""
    flat = (np.asarray(ids, dtype=np.int64)[:, None] * width + np.arange(width)).ravel()
    return gather(table, flat).reshape((len(ids), width))


def _concat_rows(a: Tensor, b: Tensor) -> Tensor:
    """Row-concatenate two Tensors. Built by padding and adding because the
    Tensor layer has no concat primitive, and a numpy concat of .data would cut
    the graph -- which would silently stop training the memory read path."""
    n_a = a.data.shape[0]
    n_b = b.data.shape[0]
    width = a.data.shape[1]
    top = np.zeros((n_a + n_b, n_a), dtype=np.float32)
    top[:n_a, :n_a] = np.eye(n_a, dtype=np.float32)
    bot = np.zeros((n_a + n_b, n_b), dtype=np.float32)
    bot[n_a:, :n_b] = np.eye(n_b, dtype=np.float32)
    del width
    return Tensor(top) @ a + Tensor(bot) @ b
