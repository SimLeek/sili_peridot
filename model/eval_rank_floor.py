"""
sili_peridot/model/eval_rank_floor.py
───────────────────────────────────────────
Does a DISLDOLayer-family layer's SCALE-rank constraint (`scale_rank=1`
vs `2` vs higher -- the rank of the value_scale⊗output_scale envelope,
see TrueMultiDigitLayer/DISLDOLayer) actually cap what gradient descent
can reach in practice, given this project's own finding that the
discrete per-synapse FP4 codes are hard to move (often bit-exact
frozen, see eval_stuck_weights.py)?

IMPORTANT, worth understanding before reading results: `scale_rank=k`
does NOT bound the layer's REPRESENTABLE matrix rank the way a literal
low-rank factorization would. The codes themselves are per-synapse
independent and nominally full-rank-capable; a rank-k scale envelope
applied elementwise is equivalent to `diag(row_scale) @ codes @
diag(col_scale)` -- a similarity-like rescaling that leaves rank(codes)
untouched. So the Eckart-Young floor computed here is NOT a hard
representational ceiling for the real FP4 layer -- it's the floor for
gradient descent's easy-to-move part (the scale) alone. If the real
FP4 arm's loss converges near (or above) the rank-k floor and doesn't
beat it with real training, that's rigorous, quantitative evidence the
codes aren't contributing real rank in practice (a sharper version of
this project's earlier `mean_delta_w=0.0` finding). If it clearly beats
the floor, the codes ARE eventually contributing real rank, just
slower than the scale envelope.

Three-arm comparison, same target/training budget, only the layer
differs (see [[feedback_do_science_correctly]]):
  - LowRankDenseLayer(rank=k): plain `x @ U @ V` factorization, no
    quantization at all. Should land ~exactly on the EY floor with
    enough training -- this is the harness's own sanity check.
  - real FP4 DISLDOLayer/DISLDOLayerDeterministic at scale_rank=k:
    the actual question.
  - DISLDOLayer32 (float32 backend, same kernels/training dynamics,
    exact storage): upper bound, should approach ~0 loss regardless
    of scale_rank (nothing there to get stuck on).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from sili.tensor import Tensor, reduce_sum


def permutation_matrix(n: int, shift: int = 1) -> np.ndarray:
    """Cyclic shift by `shift` on `n` slots. Orthogonal (all n singular
    values exactly 1), so its best rank-k approximation (Eckart-Young)
    has a clean, closed-form squared-Frobenius floor of exactly n-k --
    no need to invoke `shift=0` (identity, already rank-deficient in a
    trivial way) or any non-orthogonal target."""
    if not (1 <= shift < n):
        raise ValueError(f"shift must be in [1, n-1], got shift={shift}, n={n}")
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        P[i, (i + shift) % n] = 1.0
    return P


def eckart_young_floor(target: np.ndarray, rank: int) -> float:
    """Best-achievable squared Frobenius error for approximating `target`
    at the given rank, via truncated SVD (Eckart-Young-Mirsky): exactly
    the sum of squared singular values beyond `rank`."""
    singular_values = np.linalg.svd(target, compute_uv=False)
    discarded = singular_values[rank:]
    return float(np.sum(discarded ** 2))


class LowRankDenseLayer:
    """Plain `x @ U @ V` factorization (U: n_in x rank, V: rank x n_out),
    no quantization -- the harness's own sanity-check arm. Should reach
    ~the exact Eckart-Young floor for `rank` with enough training,
    confirming the training loop itself is capable of hitting a known
    answer before trusting any FP4-arm comparison against it."""

    def __init__(self, n_in: int, n_out: int, rank: int, rng: np.random.Generator):
        scale = 1.0 / np.sqrt(max(rank, 1))
        self.U = Tensor((rng.standard_normal((n_in, rank)) * scale).astype(np.float32))
        self.V = Tensor((rng.standard_normal((rank, n_out)) * scale).astype(np.float32))

    def forward(self, x: Tensor, learning_rate: float = 0.0) -> Tensor:
        return x @ self.U @ self.V

    def trainable_params(self) -> List[Tensor]:
        return [self.U, self.V]


class FullRankDenseLayer:
    """Plain `x @ W`, no rank constraint, no quantization -- the
    "should reach ~0 loss" upper-bound arm alongside DISLDOLayer32."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        scale = 1.0 / np.sqrt(n_in)
        self.W = Tensor((rng.standard_normal((n_in, n_out)) * scale).astype(np.float32))

    def forward(self, x: Tensor, learning_rate: float = 0.0) -> Tensor:
        return x @ self.W

    def trainable_params(self) -> List[Tensor]:
        return [self.W]


@dataclass
class RankFloorReport:
    n: int
    rank: int
    ey_floor: float
    final_sse: float  # summed squared error at the END of training (stability check)
    best_sse: float  # lowest summed squared error seen at ANY point during training
    ratio_to_floor: float  # best_sse / ey_floor -- 1.0 = exactly matched the floor
    beat_floor: bool  # best_sse meaningfully below ey_floor (real rank escape)


def measure_rank_floor(layer, target: np.ndarray, n_steps: int, lr: float,
                        rank: int, opt=None, opt_step: Optional[Callable] = None,
                        clip_grad_norm: Optional[Callable] = None,
                        beat_floor_tol: float = 0.9, eval_every: int = 1,
                        lr_decay: float = 0.99) -> RankFloorReport:
    """Trains `layer` (any object with `.forward(x: Tensor, learning_rate)
    -> Tensor` and `.trainable_params() -> List[Tensor]`) to reproduce
    `target` (n x n) via full-batch regression against all n standard
    basis vectors every step -- exact, no sampling noise, `target` is
    small enough (permutation matrices tested at n<=32) that this is
    cheap. `opt`/`opt_step`/`clip_grad_norm` let the caller supply its
    own AdamOptimizer/clip_grad_norm_ (kept as injected dependencies,
    not imported here, so this module doesn't need to know which
    project convention is in use); pass opt=None for a layer whose
    weights update inline (real DISLDOLayer-family layers, `lr` is
    threaded through `forward`'s `learning_rate` instead).

    `beat_floor_tol`: `best_sse` must be below `beat_floor_tol * ey_floor`
    to count as `beat_floor=True` -- a real, meaningful escape from the
    rank-k floor, not just landing on it within numerical noise.

    `lr_decay`: per-outer-step exponential decay applied to `lr`
    (`effective_lr = lr * lr_decay**step`), default 0.99. Root-caused,
    not just papered over with best_sse tracking below: traced a real
    DISLDOLayer32 run step-by-step and found it converges cleanly to
    SSE below the rank-2 floor by step ~20, then destabilizes and
    diverges around step ~350-400 at a CONSTANT lr. Cause is the
    standard, well-known RMSprop-without-decay pathology --
    linear_disldo.hpp's own update is plain `ci = beta2*ci +
    (1-beta2)*(g^2+contrib^2)` (beta2=0.999, no bias correction on this
    per-synapse ci) then `delta_w = -lr*g/(sqrt(ci)+eps)`. Once the
    model is near-converged, `ci` (a SLOW ~1000-step EMA of the
    ONCE-larger gradients) stays elevated relative to the now-tiny
    residual gradient for a long time, which is what keeps the step
    naturally small during the long stable plateau. But `ci` eventually
    decays down to match the small residual gradient too -- once it
    does, `g/sqrt(ci)` stops shrinking with the error and returns to
    ~full lr-sized steps REGARDLESS of how small the actual error is,
    so it overshoots, the resulting larger gradient pushes ci back up,
    and the cycle repeats/compounds. This is exactly why RMSprop/Adam
    are essentially never run without an external lr schedule in
    practice -- nothing sili-specific, and `ci`'s own bias-correction
    (already fixed elsewhere for value_scale_importance/
    output_scale_importance) wouldn't fix THIS failure mode -- that
    fixes ci being too SMALL early on; this is ci becoming well-matched
    to a small gradient LATE, which is the opposite problem. `lr_decay`
    keeps the effective step shrinking faster than ci can "catch up",
    confirmed directly to eliminate the divergence entirely (flat
    SSE=3.0000 from step 100 through 410 at lr_decay in {0.985, 0.99,
    0.995}, vs. spiking past 150+ with no decay). `best_sse`/`final_sse`
    are both still tracked and reported separately -- lr_decay makes
    them converge to the same value in practice, but the distinction
    stays meaningful for any caller who passes lr_decay=1.0 (undecayed)
    or a decay too slow for their own step budget.

    Per-example online updates, NOT one accumulated-then-batched update
    across all n basis vectors -- real DISLDOLayer-family layers update
    inline as a side effect of each individual `forward(..., learning_rate)`
    call's backward, using that call's own LOCAL gradient. Summing all n
    columns' losses into one `total_loss` and calling `.backward()` once
    would still fire n separate full-lr inline updates (one per forward
    call in the graph, each seeing only its own local gradient, not an
    n-way-averaged one) -- silently amplifying the DISLDO arms' effective
    step size ~n-fold relative to the dense arms' single batched Adam
    step, which reliably diverged in practice. Per-example updates for
    every arm keep the comparison apples-to-apples (same update count,
    same per-update gradient scale) regardless of which arm owns the
    update mechanism."""
    n = target.shape[0]
    target = target.astype(np.float32)
    basis = np.eye(n, dtype=np.float32)
    params = layer.trainable_params() if hasattr(layer, "trainable_params") else []

    def _evaluate() -> float:
        sse = 0.0
        for i in range(n):
            x = Tensor(basis[i:i + 1])
            y_pred = layer.forward(x, 0.0)
            sse += float(np.sum((y_pred.data.reshape(-1) - target[i]) ** 2))
        return sse

    best_sse = float("inf")
    for step in range(n_steps):
        effective_lr = lr * (lr_decay ** step)
        for i in range(n):
            x = Tensor(basis[i:i + 1])
            y_pred = layer.forward(x, effective_lr)
            y_true = Tensor(target[i:i + 1])
            diff = y_pred - y_true
            sq_err = reduce_sum(diff ** 2)
            sq_err.backward()
            if params:
                if clip_grad_norm is not None:
                    clip_grad_norm(params, 1.0)
                if opt is not None and opt_step is not None:
                    opt_step(opt, params, effective_lr)
            for p in params:
                p.zero_grad()
        if step % eval_every == 0:
            best_sse = min(best_sse, _evaluate())

    final_sse = _evaluate()
    best_sse = min(best_sse, final_sse)

    floor = eckart_young_floor(target, rank)
    ratio = best_sse / floor if floor > 0 else float("inf")
    return RankFloorReport(
        n=n, rank=rank, ey_floor=floor, final_sse=final_sse, best_sse=best_sse,
        ratio_to_floor=ratio, beat_floor=(floor > 0 and best_sse < beat_floor_tol * floor),
    )
