"""
sili_peridot/model/eval_superposition.py
───────────────────────────────────────────
Can FP4-quantized weights pack more independent SPARSE features than they
have raw dimensions -- the same "superposition" phenomenon Anthropic's Toy
Models of Superposition (Elhage et al. 2022) found in dense float weights
-- or does quantization structurally cap this below what float32 achieves
at the same width? Real dense transformer weights are known to rely
heavily on superposition to represent far more features/circuits than
their raw dimension count, so this is directly relevant to the actual
MiniCPM5-FP4 conversion goal, more so than a synthetic rank/recall test.

Distinct axis from eval_rank_floor.py (does gradient descent reach a rank
the architecture is theoretically capable of) and from a recurrent/MQAR-
style capacity test (temporal capacity over TIME, a separate planned
test) -- this is about SPATIAL packing precision: can a bottleneck of
width `hidden_width` represent `n_features` independent SPARSE
directions at an acceptable reconstruction cost, and does that cost
degrade under FP4 relative to float32 at the SAME width/sparsity.

Setup follows the original Toy Models of Superposition paper directly:
linear encoder (n_features -> hidden_width, no nonlinearity), ReLU
decoder (hidden_width -> n_features), trained to reconstruct sparse input
vectors under an importance-weighted MSE loss (earlier features matter
more -- feature_importance's own geometric decay). Superposition is the
STRATEGY of packing multiple features into non-orthogonal directions in
the bottleneck, tolerable because features are sparse (they rarely
co-activate, so the resulting interference is rare too) -- dense input
(density=1.0) gives the model no reason to ever attempt it, sparser input
makes it increasingly worthwhile, which is why the real comparison sweeps
density rather than testing at one fixed sparsity level.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sili.tensor import Tensor, reduce_sum


def sample_sparse_features(rng: np.random.Generator, n_features: int, density: float) -> np.ndarray:
    """One sparse feature vector: each of n_features independently active
    with probability `density`, magnitude ~ Uniform(0,1) when active, 0
    otherwise -- matches Toy Models of Superposition's own convention."""
    active = rng.random(n_features) < density
    magnitude = rng.uniform(0.0, 1.0, n_features)
    return (active * magnitude).astype(np.float32)


def feature_importance(n_features: int, decay: float = 0.9) -> np.ndarray:
    """Geometric per-feature importance weighting (I_i = decay^i) -- earlier
    features matter more, matching Toy Models of Superposition's own
    convention (without some importance gradient, every feature is
    interchangeable and there's nothing to prioritize when interference
    under a tight bottleneck is unavoidable)."""
    return np.array([decay**i for i in range(n_features)], dtype=np.float32)


def weighted_mse(x: np.ndarray, x_hat: np.ndarray, importance: np.ndarray) -> float:
    return float(np.sum(importance * (x - x_hat) ** 2))


def no_superposition_baseline(importance: np.ndarray, hidden_width: int, density: float) -> float:
    """Closed-form expected weighted loss for the best NO-superposition
    strategy: with only `hidden_width` orthogonal directions available,
    perfectly represent the `hidden_width` most-important features (their
    own importance-sorted order -- `importance` is assumed already sorted
    descending, matching feature_importance's own geometric-decay
    convention) and give up entirely on the rest (predict 0 for them,
    ReLU's own natural "no signal" output). A dropped feature i is 0 with
    probability (1-density) and Uniform(0,1) with probability density, so
    E[x_i^2] = density * E[U^2] = density/3 (U~Uniform(0,1)). This is the
    EY-floor-style reference number for this module: a real, principled
    lower bound on achievable loss WITHOUT superposition, so "did training
    beat this" is a rigorous claim, not just "did the loss go down"."""
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

    Each step samples ONE sparse feature vector, calls encoder.forward
    then decoder.forward exactly once each (a normal single-pass 2-layer
    network, unlike eval_rank_floor.py's per-basis-vector regression --
    no risk of the "same layer's forward called N times within one
    accumulated backward()" amplification bug that harness had to work
    around, since each layer here is genuinely only touched once per
    step).

    `lr_decay`: see eval_rank_floor.py's own measure_rank_floor docstring
    for why this matters for DISLDOLayer-family arms specifically (late-
    training RMSprop divergence once per-synapse ci decays to match a
    small residual) -- same mechanism, same fix, reused here."""
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
