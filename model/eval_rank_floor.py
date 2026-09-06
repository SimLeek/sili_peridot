"""
sili_peridot/model/eval_rank_floor.py
───────────────────────────────────────────
Does a DISLDOLayer-family layer's scale-rank constraint (`scale_rank=1`
vs `2` vs higher) actually cap what gradient descent can reach in
practice? Three-arm comparison (LowRankDenseLayer sanity check, real FP4
DISLDOLayer, DISLDOLayer32 upper bound). See
docs/research/eval_rank_floor.rst:eval_rank_floor.module_overview for the
full design and why scale_rank is not a hard representational ceiling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sili.tensor import Tensor, reduce_sum


def permutation_matrix(n: int, shift: int = 1) -> np.ndarray:
    """Cyclic shift by `shift` on `n` slots, giving a clean closed-form
    EY floor of n-k. See
    docs/research/eval_rank_floor.rst:eval_rank_floor.permutation_matrix_shift_choice."""
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
    return float(np.sum(discarded**2))


class LowRankDenseLayer:
    """Plain `x @ U @ V` factorization, no quantization -- the harness's
    own sanity-check arm. See
    docs/research/eval_rank_floor.rst:eval_rank_floor.low_rank_layer_sanity_check."""

    def __init__(self, n_in: int, n_out: int, rank: int, rng: np.random.Generator):
        scale = 1.0 / np.sqrt(max(rank, 1))
        self.U = Tensor((rng.standard_normal((n_in, rank)) * scale).astype(np.float32))
        self.V = Tensor((rng.standard_normal((rank, n_out)) * scale).astype(np.float32))

    def forward(self, x: Tensor, learning_rate: float = 0.0) -> Tensor:
        return x @ self.U @ self.V

    def trainable_params(self) -> list[Tensor]:
        return [self.U, self.V]


class FullRankDenseLayer:
    """Plain `x @ W`, no rank constraint, no quantization -- the
    dense upper-bound arm alongside DISLDOLayer32. See
    docs/research/eval_rank_floor.rst:eval_rank_floor.module_overview."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        scale = 1.0 / np.sqrt(n_in)
        self.W = Tensor((rng.standard_normal((n_in, n_out)) * scale).astype(np.float32))

    def forward(self, x: Tensor, learning_rate: float = 0.0) -> Tensor:
        return x @ self.W

    def trainable_params(self) -> list[Tensor]:
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


def measure_rank_floor(
    layer,
    target: np.ndarray,
    n_steps: int,
    lr: float,
    rank: int,
    opt=None,
    opt_step: Callable | None = None,
    clip_grad_norm: Callable | None = None,
    beat_floor_tol: float = 0.9,
    eval_every: int = 1,
    lr_decay: float = 0.99,
) -> RankFloorReport:
    """Trains `layer` (`.forward(x: Tensor, learning_rate) -> Tensor` and
    `.trainable_params() -> List[Tensor]`) to reproduce `target` (n x n)
    via full-batch regression against all n standard basis vectors every
    step -- exact, no sampling noise.

    `opt`/`opt_step`/`clip_grad_norm`: injected optimizer dependencies,
    not imported here (pass opt=None for a layer that updates inline).
    `beat_floor_tol`: best_sse must be below beat_floor_tol * ey_floor to
    count as beat_floor=True. `lr_decay`: per-step exponential decay on
    `lr`, fixes a real RMSprop-without-decay divergence. Updates are
    per-example/online, not one batched loss across all n columns. See
    docs/research/eval_rank_floor.rst:
      - eval_rank_floor.measure_rank_floor_injected_optimizer
      - eval_rank_floor.measure_rank_floor_per_example_updates
      - eval_rank_floor.measure_rank_floor_lr_decay_rmsprop_divergence
    """
    n = target.shape[0]
    target = target.astype(np.float32)
    basis = np.eye(n, dtype=np.float32)
    params = layer.trainable_params() if hasattr(layer, "trainable_params") else []

    def _evaluate() -> float:
        sse = 0.0
        for i in range(n):
            x = Tensor(basis[i : i + 1])
            y_pred = layer.forward(x, 0.0)
            sse += float(np.sum((y_pred.data.reshape(-1) - target[i]) ** 2))
        return sse

    best_sse = float("inf")
    for step in range(n_steps):
        effective_lr = lr * (lr_decay**step)
        for i in range(n):
            x = Tensor(basis[i : i + 1])
            y_pred = layer.forward(x, effective_lr)
            y_true = Tensor(target[i : i + 1])
            diff = y_pred - y_true
            sq_err = reduce_sum(diff**2)
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
        n=n,
        rank=rank,
        ey_floor=floor,
        final_sse=final_sse,
        best_sse=best_sse,
        ratio_to_floor=ratio,
        beat_floor=(floor > 0 and best_sse < beat_floor_tol * floor),
    )
