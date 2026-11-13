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
from sili.tensor import Tensor, exp, gather, gaussian_attention, reduce_sum

from model.toy_recall_models import DenseTensorLinear, rmsnorm_tensor

MODES = ("none", "frozen", "trace", "tbptt")


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
        write_gate: float | None = None,
        tbptt_k: int = 8,
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
        # Relative-surprise threshold below which a tick does not write.
        self.write_gate = write_gate
        self.tbptt_k = tbptt_k
        self.n_writes = 0
        self.n_ticks = 0
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
        # TBPTT: the write becomes DIFFERENTIABLE, so W_K/W_V are ordinary
        # Tensor parameters trained by the same optimizer as everything else.
        # No hand-derived gradient, no separate write-side optimizer, and no
        # extra per-synapse state -- which is what two failed derivations and an
        # Adam detour were trying to work around. Memory stays in the graph for
        # tbptt_k steps then detaches, so per-step backward is O(k) = O(1) in
        # the horizon. Reach is infinite via the PERSISTENT STATE, not via the
        # gradient path: W_K is shared across ticks, so a write rule credited
        # inside the window governs writes at any distance.
        self.W_K_t = Tensor(self.W_K_mem.copy()) if mode == "tbptt" else None
        self.W_V_t = Tensor(self.W_V_mem.copy()) if mode == "tbptt" else None
        self.M_t = Tensor(np.zeros((hidden, hidden), dtype=np.float32)) if mode == "tbptt" else None
        self._since_detach = 0
        # eligibility traces for the write side, same shapes as W_K/W_V
        self._tr_K = np.zeros_like(self.W_K_mem)
        self._tr_V = np.zeros_like(self.W_V_mem)
        self._last_read = None
        self._last_q = None
        # ONE accumulator per write-side matrix (not Adam's two).
        self._imp = {"K": np.zeros_like(self.W_K_mem), "V": np.zeros_like(self.W_V_mem)}
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
        if self.mode == "tbptt":
            params += [self.W_K_t, self.W_V_t]
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
            mem_src = self.M_t if self.mode == "tbptt" else Tensor(self.M)
            read = mem_src @ q_mem.reshape((self.hidden, 1))
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

        if self.mode == "tbptt":
            self._write_t(h, self.n_rows - 1)
        elif self.mode != "none":
            self._write(h.data[-1])
        return logits.reshape((self.n_rows, self.vocab))

    def _write_t(self, h_all: Tensor, row: int) -> None:
        """Differentiable delta-rule write (eqs 11-13, alpha=eta=0), built from
        Tensor ops so autograd supplies the exact gradient within the window."""
        d = self.hidden
        sel = np.zeros((1, self.n_rows), dtype=np.float32)
        sel[0, row] = 1.0
        h = Tensor(sel) @ h_all  # [1, d], stays in the graph
        k = h @ self.W_K_t  # [1, d]
        v = h @ self.W_V_t  # [1, d]
        kk = reduce_sum(k * k, axis=-1).reshape((1, 1))  # [1,1]
        kt = k.reshape((d, 1))
        err = self.M_t @ kt - v.reshape((d, 1))  # [d,1]
        upd = (err @ k) * (self.mem_lr / (kk + 1e-6))  # [d,d]
        self.M_t = self.M_t - upd * 2.0
        self.n_ticks += 1
        self.n_writes += 1
        self._since_detach += 1
        if self._since_detach >= self.tbptt_k:
            # TRUNCATION: drop the graph beyond k steps. Cost per step stays
            # O(k); the memory CONTENT survives, only the gradient path is cut.
            self.M_t = Tensor(self.M_t.data.copy())
            self._since_detach = 0

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
        self.n_ticks += 1
        if kk < 1e-8:
            return
        err = self.M @ k - v
        # Surprise gate: skip uninformative ticks (the RSM idea). Measured to
        # cut writes 100% -> 26% here. NOTE it does NOT fix the trace gradient
        # -- FD cos stayed ~0 at every activity level. JOURNAL 2026-11-13.
        if self.write_gate is not None:
            rel = float(np.linalg.norm(err)) / (float(np.linalg.norm(v)) + 1e-8)
            if rel < self.write_gate:
                return
        self.n_writes += 1
        th = self.mem_lr / (kk + 1e-6)
        self.M = self.M - th * 2.0 * np.outer(err, k)
        if self.mode == "trace":
            # Eligibility traces h (x) err and h (x) k. WARNING: the gradient
            # built from these FAILS a finite-difference check (cos ~ 0.13, and
            # -0.25 at beta=0.9) -- it is near-orthogonal to the true gradient.
            # Do not train on it until re-derived. JOURNAL 2026-11-13.
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
        # SINGLE-STATE adaptive step, not Adam: sili's layers carry ONE
        # optimizer variable so per-synapse state stays at 1 byte at fp4.
        # cf. feedback_importance_is_already_the_optimizer; JOURNAL 2026-11-13.
        beta = 0.99
        for name, d in (("K", d_WK), ("V", d_WV)):
            imp = self._imp[name]
            imp *= beta
            imp += (1.0 - beta) * (d * d)
            step = lr * d / (np.sqrt(imp) + 1e-8)
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
