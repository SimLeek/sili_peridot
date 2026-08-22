"""Statistical follow-up, not a seed-pinning fix: both prior grid runs
(weight-sparsity version and input-sparsity version) agree the most
interesting cell is state_width=512, full weight connectivity density,
full input density -- weight-sparsity grid saw stochastic BEAT
deterministic there (0.811 vs 0.767, gap=-0.044); input-sparsity grid's
identical cell saw them roughly TIE (0.756 vs 0.767, gap=+0.011). Both
are n=3 seeds -- not enough to tell a real effect from seed noise. Per
direct correction: the fix for "is this a real effect" is MORE
independent trials, not fixing/pinning the stochastic-rounding RNG to
make a single run reproducible (that would just answer "what does seed
X do", not "is the population-level gap actually near zero or not").

Reuses _train_and_eval from stochastic_stability_vs_scale_sparsity.py
unmodified (same task, same training procedure) at exactly that one
grid cell (embed_width=32, column_neurons=16 -> state_width=512,
input_density=1.0), just with many more seeds, so this is a real
statistical estimate of the deterministic-vs-stochastic gap at this
specific scale/density point, not a fresh comparison on different
grounds.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/stochastic_scale_dense_confirmation.py
"""
import numpy as np

from scripts.stochastic_stability_vs_scale_sparsity import _train_and_eval
from sili.sparse_rnn import DISLDOLayer, DISLDOLayerDeterministic

EMBED_WIDTH = 32
COLUMN_NEURONS = 16
INPUT_DENSITY = 1.0
N_SEEDS = 12
SEEDS = list(range(2000, 2000 + N_SEEDS))  # fresh seed block, no overlap with either prior grid


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
    # paired (same seed -> same task instances/init for both arms) t-test-style stat --
    # gives a rough sense of whether the paired gap is distinguishable from 0.
    diffs = det_accs - stoch_accs
    diff_sem = sem(diffs)
    t_stat = float(diffs.mean() / diff_sem) if diff_sem > 0 else float("nan")

    print(f"\nn_seeds={N_SEEDS} state_width={EMBED_WIDTH * COLUMN_NEURONS}")
    print(f"det:    mean={det_accs.mean():.4f} std={det_accs.std(ddof=1):.4f} sem={sem(det_accs):.4f}")
    print(f"stoch:  mean={stoch_accs.mean():.4f} std={stoch_accs.std(ddof=1):.4f} sem={sem(stoch_accs):.4f}")
    print(f"gap (det-stoch): mean={gap:.4f} paired_sem={diff_sem:.4f} paired_t={t_stat:.2f}")
