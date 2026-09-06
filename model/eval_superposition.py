"""
sili_peridot/model/eval_superposition.py
───────────────────────────────────────────
Does FP4 quantization structurally cap superposition (Elhage et al. 2022's
Toy Models of Superposition) below what float32 achieves at the same
bottleneck width -- distinct from eval_rank_floor.py's rank-reachability
axis and from a temporal/MQAR-style capacity test.
See docs/research/eval_superposition.rst:eval_superposition.module_overview.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sili.tensor import Tensor, reduce_sum


def sample_sparse_features(rng: np.random.Generator, n_features: int, density: float) -> np.ndarray:
    """One sparse feature vector: each active w.p. `density`, magnitude ~ Uniform(0,1).
    See docs/research/eval_superposition.rst:eval_superposition.sample_sparse_features_convention."""
    active = rng.random(n_features) < density
    magnitude = rng.uniform(0.0, 1.0, n_features)
    return (active * magnitude).astype(np.float32)


def feature_importance(n_features: int, decay: float = 0.9) -> np.ndarray:
    """Geometric per-feature importance weighting (I_i = decay^i).
    See docs/research/eval_superposition.rst:eval_superposition.feature_importance_geometric_decay."""
    return np.array([decay**i for i in range(n_features)], dtype=np.float32)


def weighted_mse(x: np.ndarray, x_hat: np.ndarray, importance: np.ndarray) -> float:
    return float(np.sum(importance * (x - x_hat) ** 2))


def no_superposition_baseline(importance: np.ndarray, hidden_width: int, density: float) -> float:
    """Closed-form expected weighted loss for the best NO-superposition strategy
    (perfectly represent the hidden_width most-important features, drop the rest).
    See docs/research/eval_superposition.rst:eval_superposition.no_superposition_baseline_derivation."""
    dropped = importance[hidden_width:]
    return float(np.sum(dropped) * density / 3.0)


@dataclass
class SuperpositionReport:
    n_features: int
    hidden_width: int
    density: float
    final_weighted_loss: float
    best_weighted_loss: float  # lowest weighted loss seen at any evaluation checkpoint


def measure_superposition(
    encoder,
    decoder,
    n_features: int,
    hidden_width: int,
    density: float,
    n_steps: int,
    lr: float,
    seed: int,
    importance_decay: float = 0.9,
    opt=None,
    opt_step: Callable | None = None,
    clip_grad_norm: Callable | None = None,
    lr_decay: float = 1.0,
    eval_every: int = 20,
    eval_batch: int = 200,
    log_fn: Callable[[int, int, float, float], None] | None = None,
) -> SuperpositionReport:
    """Trains `encoder` (n_features -> hidden_width) and `decoder`
    (hidden_width -> n_features) -- each any object with `.forward(x:
    Tensor, learning_rate) -> Tensor` and (for a plain, non-quantized
    layer) `.trainable_params() -> List[Tensor]` -- to reconstruct
    randomly sampled sparse feature vectors through a ReLU decoder,
    matching Toy Models of Superposition's exact architecture. Pass
    opt=None (both encoder and decoder ignore trainable_params, i.e. real
    DISLDOLayer-family layers) for arms whose weights update inline
    during backward(); pass a real optimizer + opt_step closure for the
    plain-float sanity arm (FullRankDenseLayer encoder/decoder).
    See docs/research/eval_superposition.rst:eval_superposition.measure_superposition_single_forward_no_amplification.
    See docs/research/eval_superposition.rst:eval_superposition.measure_superposition_lr_decay_reuse.
    """
    rng = np.random.default_rng(seed)
    importance = feature_importance(n_features, importance_decay)
    enc_params = encoder.trainable_params() if hasattr(encoder, "trainable_params") else []
    dec_params = decoder.trainable_params() if hasattr(decoder, "trainable_params") else []
    params = enc_params + dec_params

    def _evaluate() -> float:
        eval_rng = np.random.default_rng(seed + 999)
        total = 0.0
        for _ in range(eval_batch):
            x_np = sample_sparse_features(eval_rng, n_features, density)
            x = Tensor(x_np.reshape(1, -1))
            hidden = encoder.forward(x, 0.0)
            x_hat = decoder.forward(hidden, 0.0).relu()
            total += weighted_mse(x_np, x_hat.data.reshape(-1), importance)
        return total / eval_batch

    importance_t = Tensor(importance.reshape(1, -1))
    best = float("inf")
    for step in range(n_steps):
        effective_lr = lr * (lr_decay**step)
        x_np = sample_sparse_features(rng, n_features, density)
        x = Tensor(x_np.reshape(1, -1))
        hidden = encoder.forward(x, effective_lr)
        x_hat = decoder.forward(hidden, effective_lr).relu()
        diff = x_hat - x
        loss = reduce_sum(importance_t * diff * diff)
        loss.backward()
        if params:
            if clip_grad_norm is not None:
                clip_grad_norm(params, 1.0)
            if opt is not None and opt_step is not None:
                opt_step(opt, params, effective_lr)
        for p in params:
            p.zero_grad()
        if step % eval_every == 0:
            current = _evaluate()
            best = min(best, current)
            if log_fn is not None:
                log_fn(step, n_steps, current, best)

    final = _evaluate()
    best = min(best, final)
    if log_fn is not None:
        log_fn(n_steps, n_steps, final, best)
    return SuperpositionReport(
        n_features=n_features,
        hidden_width=hidden_width,
        density=density,
        final_weighted_loss=final,
        best_weighted_loss=best,
    )
