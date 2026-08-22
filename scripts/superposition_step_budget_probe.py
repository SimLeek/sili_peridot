"""Isolates the "just needs more steps" hypothesis from width entirely.

The width sweep (scripts/superposition_width_sweep.py) already scaled
n_steps with width and the sanity check (float+Adam should beat
no_superposition_baseline) still failed past hidden_width=5 -- but
n_steps and width were both changing together there, so it's not a
clean test of "was the step budget just too small." Fixes width at 10
(the smallest width where the sanity check already failed at 1600
steps) and sweeps ONLY n_steps, holding everything else -- including
the width-matched importance decay -- fixed, to see whether float+Adam
eventually clears the baseline given enough steps alone.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/superposition_step_budget_probe.py
"""
import numpy as np

from model.eval_rank_floor import FullRankDenseLayer
from model.eval_superposition import (
    feature_importance, measure_superposition, no_superposition_baseline,
)
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_
from sili.sparse_rnn import DISLDOLayer, DISLDOLayer32, DISLDOLayerDeterministic

HIDDEN_WIDTH = 10
N_FEATURES = 40
DENSITY = 0.05
LR = 0.02
DECAY = (0.9 ** 5) ** (1.0 / HIDDEN_WIDTH)  # matches width_sweep.py's own hw=10 value
STEP_BUDGETS = [1600, 4000, 8000, 16000, 32000]


def run_steps(n_steps: int, seed: int = 5000):
    baseline = no_superposition_baseline(feature_importance(N_FEATURES, DECAY), HIDDEN_WIDTH, DENSITY)

    rng = np.random.default_rng(seed)
    float_encoder = FullRankDenseLayer(N_FEATURES, HIDDEN_WIDTH, rng)
    float_decoder = FullRankDenseLayer(HIDDEN_WIDTH, N_FEATURES, rng)
    float_report = measure_superposition(
        float_encoder, float_decoder, N_FEATURES, HIDDEN_WIDTH, DENSITY, n_steps, LR, seed=seed,
        importance_decay=DECAY,
        opt=AdamOptimizer(), opt_step=lambda o, p, l: o.step(p, lr=l), clip_grad_norm=clip_grad_norm_)

    fp4_det_encoder = DISLDOLayerDeterministic(N_FEATURES, HIDDEN_WIDTH, N_FEATURES * HIDDEN_WIDTH,
                                               num_cpus=1, rng=np.random.default_rng(seed + 1), dense=True)
    fp4_det_decoder = DISLDOLayerDeterministic(HIDDEN_WIDTH, N_FEATURES, HIDDEN_WIDTH * N_FEATURES,
                                               num_cpus=1, rng=np.random.default_rng(seed + 2), dense=True)
    fp4_det_report = measure_superposition(fp4_det_encoder, fp4_det_decoder, N_FEATURES, HIDDEN_WIDTH,
                                           DENSITY, n_steps, LR, seed=seed + 3, importance_decay=DECAY)

    fp4_stoch_encoder = DISLDOLayer(N_FEATURES, HIDDEN_WIDTH, N_FEATURES * HIDDEN_WIDTH,
                                    num_cpus=1, rng=np.random.default_rng(seed + 4), dense=True)
    fp4_stoch_decoder = DISLDOLayer(HIDDEN_WIDTH, N_FEATURES, HIDDEN_WIDTH * N_FEATURES,
                                    num_cpus=1, rng=np.random.default_rng(seed + 5), dense=True)
    fp4_stoch_report = measure_superposition(fp4_stoch_encoder, fp4_stoch_decoder, N_FEATURES, HIDDEN_WIDTH,
                                              DENSITY, n_steps, LR, seed=seed + 6, importance_decay=DECAY)

    fp32_encoder = DISLDOLayer32(N_FEATURES, HIDDEN_WIDTH, N_FEATURES * HIDDEN_WIDTH,
                                 num_cpus=1, rng=np.random.default_rng(seed + 7))
    fp32_decoder = DISLDOLayer32(HIDDEN_WIDTH, N_FEATURES, HIDDEN_WIDTH * N_FEATURES,
                                 num_cpus=1, rng=np.random.default_rng(seed + 8))
    fp32_report = measure_superposition(fp32_encoder, fp32_decoder, N_FEATURES, HIDDEN_WIDTH,
                                        DENSITY, n_steps, LR, seed=seed + 9, importance_decay=DECAY)

    return {
        "n_steps": n_steps, "baseline": baseline,
        "float_adam": float_report.best_weighted_loss,
        "fp4_det": fp4_det_report.best_weighted_loss,
        "fp4_stoch": fp4_stoch_report.best_weighted_loss,
        "fp32": fp32_report.best_weighted_loss,
    }


if __name__ == "__main__":
    print(f"hidden_width={HIDDEN_WIDTH} n_features={N_FEATURES} density={DENSITY} decay={DECAY:.4f}")
    print(f"{'n_steps':>8} {'baseline':>9} {'float_adam':>11} {'fp4_det':>9} {'fp4_stoch':>10} {'fp32':>9} "
          f"{'adam_beats_base':>16}", flush=True)
    for n_steps in STEP_BUDGETS:
        r = run_steps(n_steps)
        print(f"{r['n_steps']:>8} {r['baseline']:>9.4f} {r['float_adam']:>11.4f} {r['fp4_det']:>9.4f} "
              f"{r['fp4_stoch']:>10.4f} {r['fp32']:>9.4f} {str(r['float_adam'] < r['baseline']):>16}",
              flush=True)
