"""Real multi-seed validation of EnergyDynamics (use_energy=True) as a
fix for dense connectivity's chance-level collapse (JOURNAL.md
2026-08-10/11 -- "pathological attractor" investigation). The
magnitude_penalty_coef fix (pure L2 on activation magnitude) cleanly
eliminated the correlated non-finite-gradient skip storms but did NOT
restore real learning over a full run. Direct user hypothesis: this
looks like a regularization-strength/kind issue, not just a numerical
-safety one -- EnergyDynamics' activation_cost term is L1-flavored
(`new_energy -= activation_cost * abs(h)`, not L2), plus it brings
real homeostatic machinery (density-targeted KL sparsity, refractory
drain, forced-firing bootstrap) magnitude_penalty_coef doesn't have at
all -- worth testing at the SAME already-tuned-low ENERGY_KWARGS this
project uses elsewhere (drive=0.00535, activation_cost=0.005,
precision=0.001, density=0.005, p=0.995, reactivity=0.0001), not a
new/untried config.

A quick 5-seed, 1500-step smoke test showed a modest, noisy signal
(use_energy=1 beat use_energy=0 on 3/5 seeds, mean last-3-checkpoint
accuracy 0.236 vs 0.206) -- inconclusive on its own (the
magnitude_penalty smoke test looked similarly promising short-run and
did NOT hold up at full scale), hence this full validation.

    SILI_RUN_BASE_SWEEP=1 pytest tests/test_dense_energy_rl_sweep.py -s
"""

import os
import statistics
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.learning_slope import analyze
from tests.test_residual_base_sweep import (
    CHANCE_RATE,
    POST_SEED_ARGS,
    REPO_ROOT,
    RUN_ENV_VAR,
    SCRIPT,
    _parse_checkpoints_from_text,
    _seeds,
)
from tests.test_residual_base_sweep_dense import ARMS, SPARSE_REFERENCE

# use_energy=1 (position 1), everything else matching FIXED_ARGS/POST_SEED_ARGS
# exactly, positions 14-16 explicit so ENERGY_KWARGS (train_tile_curriculum.py's
# own already-tuned-low default) is what actually applies -- magnitude_penalty
# _coef=0.0 (position 16) so this isolates use_energy alone, not combined.
ENERGY_FIXED_ARGS = ["1", "1", "15000", "750"]  # use_energy=1 use_attention train_steps checkpoint_every
ENERGY_TAIL = ["0", "6.0", "0.0"]  # use_synaptogenesis, clip_range, magnitude_penalty_coef

# No-fix dense reference (JOURNAL.md 2026-08-10/11, this session's own
# full sweep -- every arm collapsed to chance):
DENSE_NOFIX_REFERENCE = {
    "base4_dense": {"mean": 0.0938, "std": 0.0097},
    "base6_dense": {"mean": 0.1171, "std": 0.0310},
    "base12_dense": {"mean": 0.1050, "std": 0.0163},
    "base24_dense": {"mean": 0.0979, "std": 0.0240},
}


@pytest.mark.skipif(
    not os.environ.get(RUN_ENV_VAR), reason=f"expensive multi-seed real training run, opt in via {RUN_ENV_VAR}=1"
)
def test_dense_energy_rl_vs_no_fix_and_sparse():
    seeds = _seeds()
    results = {label: {} for label in ARMS}
    for label, arm in ARMS.items():
        for seed in seeds:
            args = [*ENERGY_FIXED_ARGS, str(seed), *POST_SEED_ARGS, *ENERGY_TAIL]
            proc = subprocess.run(
                [sys.executable, SCRIPT, arm, *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            assert proc.returncode == 0, f"{arm} seed={seed} failed:\n{proc.stderr}"
            steps, accs = _parse_checkpoints_from_text(proc.stdout)
            analyzed = analyze(steps, accs, CHANCE_RATE, window=8)
            results[label][seed] = analyzed["mean_acc"]
            print(
                f"{label:<12} seed={seed}  mean_acc={analyzed['mean_acc']:.4f}  status={analyzed['status']}", flush=True
            )

    print("\narm            mean    std     per-seed  (use_energy=True, default ENERGY_KWARGS)")
    for label in ("base4_dense", "base6_dense", "base12_dense", "base24_dense"):
        per_seed = [results[label][s] for s in seeds]
        print(
            f"{label:<14} {statistics.mean(per_seed):.4f}  "
            f"{statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0:.4f}  "
            f"{[round(v, 4) for v in per_seed]}"
        )

    print("\nenergy-rl-dense vs no-fix-dense vs sparse-echo (same base/seeds/config):")
    print(f"{'base':<8} {'energy mean':<12} {'energy std':<12} {'nofix mean':<12} {'sparse mean':<12}")
    for base_label, dense_label in (
        ("base4", "base4_dense"),
        ("base6", "base6_dense"),
        ("base12", "base12_dense"),
        ("base24", "base24_dense"),
    ):
        per_seed = [results[dense_label][s] for s in seeds]
        e_mean = statistics.mean(per_seed)
        e_std = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        nofix = DENSE_NOFIX_REFERENCE[dense_label]
        sparse = SPARSE_REFERENCE[base_label]
        print(f"{base_label:<8} {e_mean:<12.4f} {e_std:<12.4f} {nofix['mean']:<12.4f} {sparse['mean']:<12.4f}")
