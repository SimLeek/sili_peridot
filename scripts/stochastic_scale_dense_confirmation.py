"""See docs/research/stochastic_scale_dense_confirmation.rst:module_overview."""

import numpy as np
from sili.sparse_rnn import DISLDOLayer, DISLDOLayerDeterministic

from scripts.stochastic_stability_vs_scale_sparsity import _train_and_eval

EMBED_WIDTH = 32
COLUMN_NEURONS = 16
INPUT_DENSITY = 1.0
N_SEEDS = 12
SEEDS = list(range(2000, 2000 + N_SEEDS))


def sem(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


if __name__ == "__main__":
    det_accs, stoch_accs = [], []
    for i, seed in enumerate(SEEDS):
        d = _train_and_eval(EMBED_WIDTH, COLUMN_NEURONS, INPUT_DENSITY, DISLDOLayerDeterministic, seed)
        s = _train_and_eval(EMBED_WIDTH, COLUMN_NEURONS, INPUT_DENSITY, DISLDOLayer, seed)
        det_accs.append(d)
        stoch_accs.append(s)
        print(f"seed={seed} ({i + 1}/{N_SEEDS}) det={d:.4f} stoch={s:.4f}", flush=True)

    det_accs = np.array(det_accs)
    stoch_accs = np.array(stoch_accs)
    gap = det_accs.mean() - stoch_accs.mean()
    # See docs/research/stochastic_scale_dense_confirmation.rst:paired_t_stat.
    diffs = det_accs - stoch_accs
    diff_sem = sem(diffs)
    t_stat = float(diffs.mean() / diff_sem) if diff_sem > 0 else float("nan")

    print(f"\nn_seeds={N_SEEDS} state_width={EMBED_WIDTH * COLUMN_NEURONS}")
    print(f"det:    mean={det_accs.mean():.4f} std={det_accs.std(ddof=1):.4f} sem={sem(det_accs):.4f}")
    print(f"stoch:  mean={stoch_accs.mean():.4f} std={stoch_accs.std(ddof=1):.4f} sem={sem(stoch_accs):.4f}")
    print(f"gap (det-stoch): mean={gap:.4f} paired_sem={diff_sem:.4f} paired_t={t_stat:.2f}")
