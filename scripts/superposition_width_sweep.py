"""At what width does FP4 superposition packing start to work, if ever?
Direct instruction: n_synapses in a fully-connected encoder/decoder pair
scales with hidden_width^2 (holding the n_features/hidden_width
compression ratio fixed), so "does FP4 beat the no-superposition
baseline" might genuinely be a LARGER-SCALE-ONLY property, not something
a small toy test (n_features=20, hidden_width=5) could ever show -- we
need the actual curve, not a single small data point.

Sweeps hidden_width (n_features = 4*hidden_width, holding the
compression ratio fixed) across an increasing range, comparing
no_superposition_baseline against float_dense (Adam, harness sanity),
fp4_deterministic, fp4_stochastic, and fp32_reference at each width --
same 4 arms as tests/test_eval_superposition.py's real-model test, just
repeated across widths instead of at one fixed size.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/superposition_width_sweep.py
"""

import numpy as np
from sili.sparse_rnn import DISLDOLayer, DISLDOLayer32, DISLDOLayerDeterministic

from model.eval_rank_floor import FullRankDenseLayer
from model.eval_superposition import (
    feature_importance,
    measure_superposition,
    no_superposition_baseline,
)
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_

DENSITY = 0.05
BASE_N_STEPS = 800
BASE_HIDDEN_WIDTH = 5  # the already-validated single-point config this sweep extends
MAX_N_STEPS = 6000
LR = 0.02
COMPRESSION_RATIO = 4  # n_features = COMPRESSION_RATIO * hidden_width
WIDTHS = [5, 10, 20, 40, 80]


def _importance_decay_for_width(hidden_width: int) -> float:
    """Holds decay^hidden_width constant across widths (== 0.9**BASE_HIDDEN_WIDTH,
    the already-validated single-point config's own value) instead of using a
    FIXED decay=0.9 for every width. With decay fixed, the tail mass beyond
    hidden_width (what no_superposition_baseline scores against) collapses
    toward zero as hidden_width grows regardless of anything FP4-related --
    confirmed directly: an initial sweep with fixed decay=0.9 saw baseline
    fall from 0.0782 (hw=5) to ~0.00004 (hw=80), making "beats baseline"
    meaningless at large width. Scaling decay keeps the SHAPE of the
    packing problem (how much importance sits past the bottleneck, in units
    of hidden_width) comparable at every scale."""
    return (0.9**BASE_HIDDEN_WIDTH) ** (1.0 / hidden_width)


def _n_steps_for_width(hidden_width: int) -> int:
    """Scales training budget with width -- confirmed directly (see
    conversation): at a FIXED 800-step budget, even the plain float+Adam
    harness-sanity arm (which reliably beats the baseline at hw=5) stopped
    beating it by hw=10 and got monotonically worse from there, meaning
    larger networks simply hadn't converged in the same step budget rather
    than sparsity/packing genuinely getting harder. Linear scaling with
    width, capped at MAX_N_STEPS to keep the sweep tractable."""
    return min(MAX_N_STEPS, BASE_N_STEPS * max(1, hidden_width // BASE_HIDDEN_WIDTH))


def run_width(hidden_width: int, seed: int = 4000):
    n_features = COMPRESSION_RATIO * hidden_width
    decay = _importance_decay_for_width(hidden_width)
    n_steps = _n_steps_for_width(hidden_width)
    baseline = no_superposition_baseline(feature_importance(n_features, decay), hidden_width, DENSITY)

    rng = np.random.default_rng(seed)
    float_encoder = FullRankDenseLayer(n_features, hidden_width, rng)
    float_decoder = FullRankDenseLayer(hidden_width, n_features, rng)
    float_report = measure_superposition(
        float_encoder,
        float_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed,
        importance_decay=decay,
        opt=AdamOptimizer(),
        opt_step=lambda o, p, lr: o.step(p, lr=lr),
        clip_grad_norm=clip_grad_norm_,
    )

    fp4_det_encoder = DISLDOLayerDeterministic(
        n_features, hidden_width, n_features * hidden_width, num_cpus=1, rng=np.random.default_rng(seed + 1), dense=True
    )
    fp4_det_decoder = DISLDOLayerDeterministic(
        hidden_width, n_features, hidden_width * n_features, num_cpus=1, rng=np.random.default_rng(seed + 2), dense=True
    )
    fp4_det_report = measure_superposition(
        fp4_det_encoder,
        fp4_det_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed + 3,
        importance_decay=decay,
    )

    fp4_stoch_encoder = DISLDOLayer(
        n_features, hidden_width, n_features * hidden_width, num_cpus=1, rng=np.random.default_rng(seed + 4), dense=True
    )
    fp4_stoch_decoder = DISLDOLayer(
        hidden_width, n_features, hidden_width * n_features, num_cpus=1, rng=np.random.default_rng(seed + 5), dense=True
    )
    fp4_stoch_report = measure_superposition(
        fp4_stoch_encoder,
        fp4_stoch_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed + 6,
        importance_decay=decay,
    )

    fp32_encoder = DISLDOLayer32(
        n_features, hidden_width, n_features * hidden_width, num_cpus=1, rng=np.random.default_rng(seed + 7)
    )
    fp32_decoder = DISLDOLayer32(
        hidden_width, n_features, hidden_width * n_features, num_cpus=1, rng=np.random.default_rng(seed + 8)
    )
    fp32_report = measure_superposition(
        fp32_encoder,
        fp32_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed + 9,
        importance_decay=decay,
    )

    return {
        "hidden_width": hidden_width,
        "n_features": n_features,
        "n_steps": n_steps,
        "baseline": baseline,
        "float_adam": float_report.best_weighted_loss,
        "fp4_det": fp4_det_report.best_weighted_loss,
        "fp4_stoch": fp4_stoch_report.best_weighted_loss,
        "fp32": fp32_report.best_weighted_loss,
    }


if __name__ == "__main__":
    print(
        f"{'hidden_w':>9} {'n_steps':>8} {'baseline':>9} {'float_adam':>11} "
        f"{'fp4_det':>9} {'fp4_stoch':>10} {'fp32':>9} "
        f"{'det/fp32':>9} {'stoch/fp32':>11} "
        f"{'det_beats_base':>15} {'stoch_beats_base':>17}",
        flush=True,
    )
    for hw in WIDTHS:
        r = run_width(hw)
        print(
            f"{r['hidden_width']:>9} {r['n_steps']:>8} {r['baseline']:>9.4f} "
            f"{r['float_adam']:>11.4f} {r['fp4_det']:>9.4f} {r['fp4_stoch']:>10.4f} "
            f"{r['fp32']:>9.4f} {r['fp4_det'] / r['fp32']:>9.3f} {r['fp4_stoch'] / r['fp32']:>11.3f} "
            f"{r['fp4_det'] < r['baseline']!s:>15} "
            f"{r['fp4_stoch'] < r['baseline']!s:>17}",
            flush=True,
        )
