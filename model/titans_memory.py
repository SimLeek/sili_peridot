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
        tie_query_to_key: bool = True,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng()
        d = d_mem or d_in
        self.d_in, self.d = d_in, d
        self.momentum, self.forget = momentum, forget
        self.theta_scale, self.alpha_scale = theta_scale, alpha_scale
        self.const_alpha, self.const_eta, self.const_theta = const_alpha, const_eta, const_theta

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

    def read(self, x: np.ndarray) -> np.ndarray:
        """y_t = M*(q_t), eq 15. Pure forward pass, no state change."""
        return self.M @ (x.astype(np.float32) @ self.W_Q)

    def write(self, x: np.ndarray, *, theta: float | None = None) -> dict:
        """One inner-loop step: eqs 10-13. `theta` overrides the data-dependent
        step size, which is how a caller can hold the write budget fixed across
        ablation arms instead of letting an untrained gate confound them."""
        x = x.astype(np.float32)
        k = x @ self.W_K
        v = x @ self.W_V

        if not self.forget:
            alpha = 0.0
        elif self.const_alpha is not None:
            alpha = self.const_alpha
        else:
            alpha = self.alpha_scale * _sigmoid(x @ self.w_alpha)

        if not self.momentum:
            eta = 0.0
        elif self.const_eta is not None:
            eta = self.const_eta
        else:
            eta = _sigmoid(x @ self.w_eta)

        if theta is not None:
            th = theta
        elif self.const_theta is not None:
            th = self.const_theta
        else:
            # normalized delta-rule step: one-shot storage of this (k, v) pair
            th = 0.5 / (float(k @ k) + 1e-6)

        err = self.M @ k - v
        grad = 2.0 * np.outer(err, k)  # grad_M ||Mk - v||^2, closed form
        self.S = eta * self.S - th * grad
        self.M = (1.0 - alpha) * self.M + self.S

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
        alpha = (self.const_alpha if self.const_alpha is not None else 0.0) if self.forget else 0.0
        eta = (self.const_eta if self.const_eta is not None else 0.9) if self.momentum else 0.0
        if theta is not None:
            th = theta
        elif self.const_theta is not None:
            th = self.const_theta
        else:
            th = 0.5 / (float(k @ k) + 1e-6)
        err = self.M @ k - v
        grad = 2.0 * np.outer(err, k)
        self.S = eta * self.S - th * grad
        self.M = (1.0 - alpha) * self.M + self.S
        self.last = {
            "surprise": float(np.linalg.norm(err)),
            "alpha": float(alpha),
            "eta": float(eta),
            "theta": float(th),
            "mem_norm": float(np.linalg.norm(self.M)),
        }
        return self.last

    def read_q(self, q: np.ndarray) -> np.ndarray:
        """y = M q for an explicit query vector (eq 15, projections bypassed)."""
        return self.M @ q.astype(np.float32)

    def step(self, x: np.ndarray, *, write: bool = True, theta: float | None = None) -> np.ndarray:
        """Read-then-write, so `y` reflects only the PAST (strictly causal). This
        ordering matters on recall tasks: a query token's own v_t is derived from
        the key, so writing before reading would let it corrupt the association it
        is trying to retrieve."""
        y = self.read(x)
        if write:
            self.write(x, theta=theta)
        return y
