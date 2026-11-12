"""Titans Neural Long-Term Memory (LMM), ported from arXiv:2501.00663 s3.1.

    k_t = x_t W_K,  v_t = x_t W_V,  q_t = x_t W_Q                   (11, 14)
    l(M_{t-1}; x_t) = || M_{t-1}(k_t) - v_t ||_2^2                      (12)
    S_t = eta_t * S_{t-1} - theta_t * grad_M l(M_{t-1}; x_t)            (10)
    M_t = (1 - alpha_t) * M_{t-1} + S_t                                 (13)
    y_t = M*(q_t)       -- retrieval is a forward pass, no update       (15)

For a matrix-valued memory the inner gradient is closed form,
grad_M ||Mk - v||^2 = 2 (M k - v) k^T: a delta-rule outer-product update,
O(d^2) per step, entirely in the forward pass. No autograd, no BPTT -- and S_t
with its eta_t momentum is an eligibility trace over surprise (cf.
bend_proofs/synaptic_hold_bptt_approximation).

Context: implemented because RSM measured neutral-to-negative here and has no
independent replication, so there was no working reference to ablate from.
Titans is the validated descendant; establish it works, THEN ablate toward RSM.
See JOURNAL.md 2026-11-12.

GATES MUST NOT BE LEFT RANDOM. alpha_t, eta_t, theta_t are data-dependent and
LEARNED in the paper. With random projections alpha_t ~= sigmoid(0) ~= 0.5, so
eq 13 halves the memory every step (0.5^32 ~= 2e-10 across a 32-token gap) and
recall lands BELOW chance. Pass const_* to pin them when the projections are
frozen; const_theta=None uses the normalized delta-rule step 1/||k||^2, which
stores one pair in one shot.

A CONSTANT gate is a different mechanism, not a weaker learned one: one constant
alpha decays must-keep and junk writes at the same rate, so no value of it can be
right. `adaptive_gates=True` recovers the data-dependence with no outer loop and
no oracle, from forward-pass statistics -- alpha on memory saturation (no decay
until full), eta on agreement between the new gradient and the stored trace (so
a transient spike resets it instead of smearing in). `retrieval_gated_forget`
decays each column of M in inverse proportion to query traffic, a
use-it-or-lose-it rule the paper lacks (their alpha is input-driven, never
read-driven).

Measured verdict, 4 regimes x 6 arms: NO setting of forgetting or momentum beats
having neither, because a delta rule already carries implicit recency -- each
write overwrites the directions it touches. The gates' win is landing on the
right value unprompted (eta self-selects 0.003 where a 14-cell grid found 0), not
a higher number. See JOURNAL.md 2026-11-12 for the full table and the two
remaining infidelities (linear vs deep memory; scale).

Ablation flags momentum/forget are the paper's own Table 5 rungs; its published
ordering is weight decay (forgetting) > momentum > convolution > persistent
memory. Reproducing that ORDERING is a scale-free validation gate -- their
absolute numbers need 170M-760M params, the ordering does not.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class TitansMemory:
    """Matrix-valued Titans LMM with data-dependent gates.

    `momentum=False` and `forget=False` are the paper's own ablation rungs, not
    conveniences: removing them should degrade recall, forgetting more than
    momentum, if this port is faithful.
    """

    def __init__(
        self,
        d_in: int,
        d_mem: int | None = None,
        *,
        momentum: bool = True,
        forget: bool = True,
        theta_scale: float = 1.0,
        alpha_scale: float = 1.0,
        const_alpha: float | None = 0.0,
        const_eta: float | None = 0.9,
        const_theta: float | None = None,
        adaptive_gates: bool = False,
        alpha_max: float = 0.1,
        eta_max: float = 0.9,
        u_target: float = 1.0,
        gamma: float = 8.0,
        retrieval_gated_forget: bool = False,
        read_decay: float = 0.9,
        lag_adaptive: bool = False,
        lag_horizon: float = 48.0,
        lag_beta: float = 0.02,
        lag_buffer: int = 192,
        tie_query_to_key: bool = True,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng()
        d = d_mem or d_in
        self.d_in, self.d = d_in, d
        self.momentum, self.forget = momentum, forget
        self.theta_scale, self.alpha_scale = theta_scale, alpha_scale
        self.const_alpha, self.const_eta, self.const_theta = const_alpha, const_eta, const_theta
        self.adaptive_gates = adaptive_gates
        self.alpha_max, self.eta_max = alpha_max, eta_max
        self.u_target, self.gamma = u_target, gamma
        self.retrieval_gated_forget = retrieval_gated_forget
        self.read_decay = read_decay
        self.read_usage = np.zeros(d, dtype=np.float32)
        self.lag_adaptive = lag_adaptive
        self.lag_horizon, self.lag_beta, self.lag_buffer = lag_horizon, lag_beta, lag_buffer
        self.lag_ema = 0.0
        self._tick = 0
        self._kbuf: list[tuple[int, np.ndarray]] = []

        s = 1.0 / np.sqrt(d_in)
        self.W_K = (rng.standard_normal((d_in, d)) * s).astype(np.float32)
        self.W_V = (rng.standard_normal((d_in, d)) * s).astype(np.float32)
        # Querying BY KEY is the natural choice when the projections are frozen:
        # with an untrained W_Q there is nothing to align q with k, so recall
        # would fail for a reason unrelated to the memory mechanism.
        self.W_Q = self.W_K if tie_query_to_key else (rng.standard_normal((d_in, d)) * s).astype(np.float32)

        self.w_alpha = (rng.standard_normal(d_in) * s).astype(np.float32)
        self.w_eta = (rng.standard_normal(d_in) * s).astype(np.float32)
        self.w_theta = (rng.standard_normal(d_in) * s).astype(np.float32)

        self.M = np.zeros((d, d), dtype=np.float32)
        self.S = np.zeros((d, d), dtype=np.float32)
        self.last = {}

    def reset(self) -> None:
        """Clear memory and surprise momentum. Call at every sequence boundary."""
        self.M[:] = 0.0
        self.S[:] = 0.0
        self.read_usage[:] = 0.0
        self._tick = 0
        self._kbuf.clear()

    def _adaptive_alpha(self) -> float:
        """Forget only once there is a shortage of room: alpha stays ~0 until
        ||M|| reaches the capacity budget, then rises toward alpha_max."""
        u = float(np.linalg.norm(self.M)) / np.sqrt(self.d)
        return float(self.alpha_max * _sigmoid(self.gamma * (u / self.u_target - 1.0)))

    def _adaptive_eta(self, grad: np.ndarray) -> float:
        """Accumulate the surprise trace only while the new gradient agrees with
        it, so a transient spike resets the trace instead of smearing into it."""
        s_n = float(np.linalg.norm(self.S))
        g_n = float(np.linalg.norm(grad))
        if s_n < 1e-8 or g_n < 1e-8:
            return 0.0
        cos = float(np.sum(self.S * -grad)) / (s_n * g_n)
        return float(self.eta_max * max(0.0, cos))

    def _decayed(self, alpha: float) -> np.ndarray:
        """(1 - alpha) * M_{t-1}, the eq-13 decay -- except that retrieval-gated
        forgetting spares the columns queries actually read. Column j of M is the
        one key component j drives, so usage is tracked per query component."""
        if alpha <= 0.0:
            return self.M
        if not self.retrieval_gated_forget:
            return (1.0 - alpha) * self.M
        peak = float(self.read_usage.max())
        spare = self.read_usage / peak if peak > 0.0 else np.zeros(self.d, dtype=np.float32)
        return self.M * (1.0 - alpha * (1.0 - spare)).astype(np.float32)

    def _gates(self, grad: np.ndarray, *, x: np.ndarray | None = None) -> tuple[float, float]:
        """Resolve (alpha_t, eta_t) for this step. Precedence: the ablation
        switches win, then intrinsic adaptation, then a pinned constant, then the
        paper's learned projections (only meaningful once those are trained)."""
        if self.lag_adaptive:
            a_lag, e_lag = self._lag_gates()
            return (a_lag if self.forget else 0.0), (e_lag if self.momentum else 0.0)
        if not self.forget:
            alpha = 0.0
        elif self.adaptive_gates:
            alpha = self._adaptive_alpha()
        elif self.const_alpha is not None:
            alpha = self.const_alpha
        elif x is not None:
            alpha = float(self.alpha_scale * _sigmoid(x @ self.w_alpha))
        else:
            alpha = 0.0

        if not self.momentum:
            eta = 0.0
        elif self.adaptive_gates:
            eta = self._adaptive_eta(grad)
        elif self.const_eta is not None:
            eta = self.const_eta
        elif x is not None:
            eta = float(_sigmoid(x @ self.w_eta))
        else:
            eta = 0.9
        return float(alpha), float(eta)

    def _note_read(self, q: np.ndarray) -> None:
        self.read_usage *= self.read_decay
        self.read_usage += q * q
        if not self.lag_adaptive or not self._kbuf:
            return
        # How far back is this query reaching? Match it against the keys the
        # stream itself wrote; the lag of the best match is the task's retrieval
        # horizon, which is what decides whether to broaden or sharpen. The
        # stream supplies the queries, so this is not an oracle.
        qn = float(np.linalg.norm(q))
        if qn < 1e-8:
            return
        sims = [(float(kb @ q) / qn, tk) for tk, kb in self._kbuf]
        best, tick = max(sims)
        if best > 0.5:
            lag = float(self._tick - tick)
            self.lag_ema += self.lag_beta * (lag - self.lag_ema)

    def _lag_gates(self) -> tuple[float, float]:
        """Queries reaching deep call for a BROAD retention profile (high eta);
        queries that only touch recent writes call for a SHARP one (eta low,
        alpha carrying the decay). This is the temporal tradeoff the regime sweep
        measured, turned into a controller."""
        f = float(np.clip(self.lag_ema / self.lag_horizon, 0.0, 1.0))
        return self.alpha_max * (1.0 - f), self.eta_max * f

    def read(self, x: np.ndarray) -> np.ndarray:
        """y_t = M*(q_t), eq 15. Forward pass only; records retrieval traffic."""
        q = x.astype(np.float32) @ self.W_Q
        self._note_read(q)
        return self.M @ q

    def write(self, x: np.ndarray, *, theta: float | None = None) -> dict:
        """One inner-loop step: eqs 10-13. `theta` overrides the data-dependent
        step size, which is how a caller can hold the write budget fixed across
        ablation arms instead of letting an untrained gate confound them."""
        x = x.astype(np.float32)
        k = x @ self.W_K
        v = x @ self.W_V

        if theta is not None:
            th = theta
        elif self.const_theta is not None:
            th = self.const_theta
        else:
            # normalized delta-rule step: one-shot storage of this (k, v) pair
            th = 0.5 / (float(k @ k) + 1e-6)

        err = self.M @ k - v
        grad = 2.0 * np.outer(err, k)  # grad_M ||Mk - v||^2, closed form
        alpha, eta = self._gates(grad, x=x)
        self.S = eta * self.S - th * grad
        self.M = self._decayed(alpha) + self.S

        self.last = {
            "surprise": float(np.linalg.norm(err)),
            "grad_norm": float(np.linalg.norm(grad)),
            "alpha": float(alpha),
            "eta": float(eta),
            "theta": float(th),
            "mem_norm": float(np.linalg.norm(self.M)),
        }
        return self.last

    def write_kv(self, k: np.ndarray, v: np.ndarray, *, theta: float | None = None) -> dict:
        """eqs 10/12/13 on an EXPLICIT (k, v) pair, bypassing W_K/W_V.

        Needed because the LMM is not a standalone associative memory over raw
        tokens: eq 11 derives k_t AND v_t from the same x_t, so feeding a key
        token stores key->key. In a real Titans, x_t is a hidden state that has
        already attended over context, so the ATTENTION CORE does the key/value
        binding and the memory only stores it -- which is why Titans is a hybrid
        and why MAC (memory-as-context) wins on NIAH. This entry point isolates
        the memory update itself from that binding problem."""
        k = k.astype(np.float32)
        v = v.astype(np.float32)
        if theta is not None:
            th = theta
        elif self.const_theta is not None:
            th = self.const_theta
        else:
            th = self.theta_scale * 0.5 / (float(k @ k) + 1e-6)
        err = self.M @ k - v
        grad = 2.0 * np.outer(err, k)
        alpha, eta = self._gates(grad)
        if self.lag_adaptive and theta is None and self.const_theta is None:
            th *= 1.0 - eta  # hold the EFFECTIVE step fixed as eta varies
        self.S = eta * self.S - th * grad
        self.M = self._decayed(alpha) + self.S
        self._note_write(k)
        self.last = {
            "surprise": float(np.linalg.norm(err)),
            "alpha": float(alpha),
            "eta": float(eta),
            "theta": float(th),
            "mem_norm": float(np.linalg.norm(self.M)),
            "lag_ema": self.lag_ema,
        }
        return self.last

    def _note_write(self, k: np.ndarray) -> None:
        self._tick += 1
        if not self.lag_adaptive:
            return
        n = float(np.linalg.norm(k))
        if n > 1e-8:
            self._kbuf.append((self._tick, k / n))
            if len(self._kbuf) > self.lag_buffer:
                del self._kbuf[0]

    def read_q(self, q: np.ndarray) -> np.ndarray:
        """y = M q for an explicit query vector (eq 15, projections bypassed)."""
        q = q.astype(np.float32)
        self._note_read(q)
        return self.M @ q

    def step(self, x: np.ndarray, *, write: bool = True, theta: float | None = None) -> np.ndarray:
        """Read-then-write, so `y` reflects only the PAST (strictly causal). This
        ordering matters on recall tasks: a query token's own v_t is derived from
        the key, so writing before reading would let it corrupt the association it
        is trying to retrieve."""
        y = self.read(x)
        if write:
            self.write(x, theta=theta)
        return y


class TitansDeepMemory:
    """Titans LMM with a DEEP memory, M(k) = W_L sigma(... sigma(W_1 k)).

    `TitansMemory` above is a matrix memory, which Table 5 of the paper lists as
    the "Linear Memory" ABLATION ROW: it costs 7.34 points of long-context
    accuracy (92.68 -> 85.34), the largest single drop in that column and
    essentially tied with removing weight decay (85.60). Their base memory is an
    MLP with L_M >= 2. So the gate sweep on a matrix memory was measuring one
    ablation stacked on another, which also explains why forgetting could not pay
    there: a delta rule on a matrix already overwrites the directions it touches,
    so recency is free and alpha has nothing left to buy. An MLP has no such
    implicit recency.

    The inner gradient stays closed form (plain backprop through L layers, all in
    the forward pass -- no autograd, no BPTT), and the normalized delta-rule step
    generalizes exactly. For a step -th*g along the gradient, the output moves by
    about -2*th*J J^T err, so matching it to -err gives

        th = 2 ||err||^2 / ||g||^2        (summed over layers)

    which reduces to 0.5/||k||^2 for L=1, the one-shot storage step. That makes
    L=1 here a check against the validated matrix path rather than new code.
    """

    def __init__(
        self,
        d: int,
        *,
        n_layers: int = 2,
        hidden: int | None = None,
        momentum: bool = True,
        forget: bool = True,
        const_alpha: float | None = 0.0,
        const_eta: float | None = 0.0,
        const_theta: float | None = None,
        theta_scale: float = 1.0,
        adaptive_gates: bool = False,
        alpha_max: float = 0.1,
        eta_max: float = 0.9,
        u_target: float = 1.0,
        gamma: float = 8.0,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng()
        self.d = d
        self.theta_scale = theta_scale
        h = hidden or d
        dims = [d] + [h] * (n_layers - 1) + [d]
        self.W = [
            (rng.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i])).astype(np.float32) for i in range(n_layers)
        ]
        self.W0 = [w.copy() for w in self.W]
        self.S = [np.zeros_like(w) for w in self.W]
        self.n_layers = n_layers
        self.momentum, self.forget = momentum, forget
        self.const_alpha, self.const_eta, self.const_theta = const_alpha, const_eta, const_theta
        self.adaptive_gates = adaptive_gates
        self.alpha_max, self.eta_max = alpha_max, eta_max
        self.u_target, self.gamma = u_target, gamma
        self.last = {}

    def reset(self) -> None:
        """Restore the initial map and clear the surprise trace. Weight decay
        pulls toward this init, not toward zero: at zero the first layer's
        gradient vanishes (it is proportional to W_2^T err) and the memory is
        permanently dead, so zero is not an available resting point."""
        for w, w0, s in zip(self.W, self.W0, self.S, strict=True):
            w[:] = w0
            s[:] = 0.0

    def _forward(self, k):
        acts = [k]
        for i, w in enumerate(self.W):
            z = w @ acts[-1]
            acts.append(np.tanh(z) if i < self.n_layers - 1 else z)
        return acts

    def _grads(self, acts, err):
        """Backprop 2*err through the stack; returns one gradient per layer."""
        g = [None] * self.n_layers
        delta = 2.0 * err
        for i in range(self.n_layers - 1, -1, -1):
            g[i] = np.outer(delta, acts[i])
            if i > 0:
                delta = (self.W[i].T @ delta) * (1.0 - acts[i] ** 2)
        return g

    def read_q(self, q: np.ndarray) -> np.ndarray:
        """y = M(q), eq 15. Forward pass only."""
        return self._forward(q.astype(np.float32))[-1]

    def write_kv(self, k: np.ndarray, v: np.ndarray, *, theta: float | None = None) -> dict:
        """eqs 10/12/13 on an explicit (k, v) pair, with a deep memory."""
        k = k.astype(np.float32)
        v = v.astype(np.float32)
        acts = self._forward(k)
        err = acts[-1] - v
        g = self._grads(acts, err)

        gn2 = float(sum(float(np.sum(gi * gi)) for gi in g))
        if theta is not None:
            th = theta
        elif self.const_theta is not None:
            th = self.const_theta
        else:
            th = self.theta_scale * 2.0 * float(err @ err) / (gn2 + 1e-8)

        if not self.forget:
            alpha = 0.0
        elif self.adaptive_gates:
            dev = np.sqrt(sum(float(np.sum((w - w0) ** 2)) for w, w0 in zip(self.W, self.W0, strict=True)))
            u = float(dev) / np.sqrt(self.d)
            alpha = float(self.alpha_max * _sigmoid(self.gamma * (u / self.u_target - 1.0)))
        else:
            alpha = float(self.const_alpha or 0.0)

        if not self.momentum:
            eta = 0.0
        elif self.adaptive_gates:
            sn = np.sqrt(sum(float(np.sum(s * s)) for s in self.S))
            if sn < 1e-8 or gn2 < 1e-16:
                eta = 0.0
            else:
                dot = sum(float(np.sum(s * -gi)) for s, gi in zip(self.S, g, strict=True))
                eta = float(self.eta_max * max(0.0, dot / (sn * np.sqrt(gn2))))
        else:
            eta = float(self.const_eta or 0.0)

        for i in range(self.n_layers):
            self.S[i] = eta * self.S[i] - th * g[i]
            self.W[i] = self.W0[i] + (1.0 - alpha) * (self.W[i] - self.W0[i]) + self.S[i]

        self.last = {
            "surprise": float(np.linalg.norm(err)),
            "alpha": alpha,
            "eta": eta,
            "theta": float(th),
            "mem_norm": float(np.sqrt(sum(float(np.sum(w * w)) for w in self.W))),
        }
        return self.last
