"""Minimal memory-as-context Titans: window attention + a Titans LMM, sized for
verification rather than capability.

Target: real trainable recurrence whose per-step BACKWARD is constant in the
horizon, so it can run in real time -- a page in context, the book title in
recurrent memory.

The write (eqs 10-13) is a delta rule in the forward pass, so storing needs no
gradient through time. That splits the parameters by credit cost: the READ side
(W_Q, attention, head) gets exact gradient from the current tick at O(1); M is
not a parameter at all; only the WRITE side (W_K, W_V) acts through later reads
and so needs an approximation. Hence the modes -- "none" (control), "frozen"
(write side random+frozen, O(1) with no approximation anywhere), "trace" (write
side from an eligibility trace, still O(1); S_t already is such a trace, and
bend_proofs/synaptic_hold_bptt_approximation bounds the error by the number of
ACTIVE ticks, never the horizon).

Measured O(1): see scripts/measure_constant_time_backward.py. Rationale, bug
history and results: JOURNAL.md 2026-11-12.
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
        sigma_init: float | None = None,
        mem_lr: float = 0.5,
        trace_beta: float = 0.9,
        grad_clip: float = 1.0,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
        rng: np.random.Generator | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        rng = rng or np.random.default_rng()
        self.vocab, self.hidden, self.window = vocab, hidden, window
        self.mode, self.mem_lr, self.trace_beta = mode, mem_lr, trace_beta
        self.grad_clip = grad_clip
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
        # sigma is a Gaussian POSITIONAL prior and its unit is TOKENS, so it has
        # to match the distance over which a match must be found. sigma=1
        # (log_sigmas=0) confines every row to its neighbours and suppresses the
        # content matching induction needs: measured page accuracy 0.147 at
        # sigma=1 vs 0.256 at sigma=4. But the window width is NOT the right
        # scale either (0.206 at sigma=16) -- the query key recurs within its own
        # page, so the required reach is the page length, not the context length.
        # Default to a small multiple of that; override per task.
        sig = 4.0 if sigma_init is None else float(sigma_init)
        self.log_sigmas = Tensor(np.full(self.n_rows, np.log(sig), dtype=np.float32))

        # Memory read side trains exactly; write side depends on mode.
        self.W_Q_mem = DenseTensorLinear(hidden, hidden, scale=s)
        self.W_K_mem = (rng.standard_normal((hidden, hidden)) * s).astype(np.float32)
        self.W_V_mem = (rng.standard_normal((hidden, hidden)) * s).astype(np.float32)
        self.M = np.zeros((hidden, hidden), dtype=np.float32)
        # eligibility traces for the write side, same shapes as W_K/W_V
        self._tr_K = np.zeros_like(self.W_K_mem)
        self._tr_V = np.zeros_like(self.W_V_mem)
        self._last_read = None
        self._last_q = None
        self._adam_t = 0
        self._adam_m = {"K": np.zeros_like(self.W_K_mem), "V": np.zeros_like(self.W_V_mem)}
        self._adam_v = {"K": np.zeros_like(self.W_K_mem), "V": np.zeros_like(self.W_V_mem)}
        self.last = {}

    def read_grad(self):
        """Gradient arriving at the memory-read row, or None. The trainer feeds
        this to apply_trace_credit; without that call `trace` is byte-identical
        to `frozen`, which is a bug that already cost two full comparison runs."""
        r = self._last_read
        return None if r is None else getattr(r, "grad", None)

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
            # kept so the trainer can recover the gradient that ARRIVES at the
            # memory-read row after backward; that gradient is what the
            # write-side trace is charged against.
            self._last_read = read
            self._last_q = q_mem.data.ravel().astype(np.float32).copy()
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
            # DERIVED eligibility traces (see JOURNAL 2026-11-13). With
            # dL/dM = outer(g, q) at the read and M accumulating
            # -2 theta_s err_s k_s^T from each write, holding err fixed:
            #   dL/dW_K = -2 SUM_s theta_s (err_s . g) h_s (x) q
            #   dL/dW_V = +2 SUM_s theta_s (k_s . q)   h_s (x) g
            # so the traces that must be carried are h (x) err and h (x) k.
            # An earlier version stored h (x) k for BOTH and then dropped the
            # g/q contraction entirely, which left a monotone drift rather than
            # a gradient -- it moved |dW_K| by 0.545 and learned nothing.
            self._tr_K = self.trace_beta * self._tr_K + th * np.outer(h, err)
            self._tr_V = self.trace_beta * self._tr_V + th * np.outer(h, k)
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
        # Project the read gradient onto the stored traces. The traces hold the
        # outer products that WOULD have carried credit, so one trace-weighted
        # step charges the whole history in O(1) instead of unrolling it.
        q = self._last_q
        if q is None:
            return
        # The derived contraction. Direction comes from g and q; the traces
        # supply the accumulated history in O(1), with no unrolling.
        d_WK = -2.0 * np.outer(self._tr_K @ g, q)
        d_WV = 2.0 * np.outer(self._tr_V @ q, g)
        # ADAM, not raw SGD. Every Tensor parameter here is trained by
        # AdamOptimizer, and the write side was the only group getting plain SGD
        # -- with a derived gradient of order 1e-4, lr=0.003 moved |dW_K| by
        # 9.29e-04 over ~24k updates, i.e. the write side was effectively frozen
        # and learned nothing. Adam's 1/sqrt(v) scaling makes the effective step
        # ~lr regardless of gradient scale, which is the actual mismatch; it is
        # the same treatment the rest of the model already gets rather than a
        # hand-picked write-side learning rate.
        self._adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, d in (("K", d_WK), ("V", d_WV)):
            m_, v_ = self._adam_m[name], self._adam_v[name]
            m_ *= b1
            m_ += (1.0 - b1) * d
            v_ *= b2
            v_ += (1.0 - b2) * (d * d)
            mhat = m_ / (1.0 - b1**self._adam_t)
            vhat = v_ / (1.0 - b2**self._adam_t)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            if name == "K":
                self.W_K_mem -= step
            else:
                self.W_V_mem -= step


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
