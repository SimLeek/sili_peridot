"""Real multi-seed validation of ToyTileRecurrenceRealFP4's
magnitude_penalty_coef fix (JOURNAL.md 2026-08-10, later still --
"pathological attractor" diagnosis): a short-run, multi-seed smoke test
already showed a very clean effect (non-finite-gradient skip rate
4.67%->~0% even at coef=0.001; short-run accuracy improved 4/5 seeds
at coef=0.01). This runs the SAME full config as
test_residual_base_sweep_dense.py (15000 steps, 5 seeds, 4 dense arms)
but with magnitude_penalty_coef=0.01, to see whether the fix actually
restores dense connectivity to competitive accuracy vs sparse-echo, not
just "no longer collapses to chance in a short run."

    SILI_RUN_BASE_SWEEP=1 pytest tests/test_dense_magnitude_penalty_sweep.py -s
"""
import os
import statistics
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.learning_slope import analyze
from tests.test_residual_base_sweep import (
    _parse_checkpoints_from_text, _seeds, FIXED_ARGS, POST_SEED_ARGS,
    CHANCE_RATE, RUN_ENV_VAR, SCRIPT, REPO_ROOT,
)
from tests.test_residual_base_sweep_dense import ARMS, SPARSE_REFERENCE

MAGNITUDE_PENALTY_COEF = "0.01"
# position 14 (use_synaptogenesis=0), 15 (clip_range=6.0) filled explicitly
# so position 16 (magnitude_penalty_coef) lands correctly.
DENSE_MAGPEN_TAIL = ["0", "6.0", MAGNITUDE_PENALTY_COEF]

# Dense, no-penalty reference (this session's earlier full sweep,
# JOURNAL.md 2026-08-10 -- every arm/seed collapsed to chance):
DENSE_NOPEN_REFERENCE = {
    "base4_dense":  {"mean": 0.0938, "std": 0.0097},
    "base6_dense":  {"mean": 0.1171, "std": 0.0310},
    "base12_dense": {"mean": 0.1050, "std": 0.0163},
    "base24_dense": {"mean": 0.0979, "std": 0.0240},
}


@pytest.mark.skipif(not os.environ.get(RUN_ENV_VAR),
                    reason=f"expensive multi-seed real training run, opt in via {RUN_ENV_VAR}=1")
def test_dense_magnitude_penalty_vs_no_penalty_and_sparse():
    seeds = _seeds()
    results = {label: {} for label in ARMS}
    for label, arm in ARMS.items():
        for seed in seeds:
            args = [*FIXED_ARGS, str(seed), *POST_SEED_ARGS, *DENSE_MAGPEN_TAIL]
            proc = subprocess.run(
                [sys.executable, SCRIPT, arm, *args],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
            assert proc.returncode == 0, f"{arm} seed={seed} failed:\n{proc.stderr}"
            steps, accs = _parse_checkpoints_from_text(proc.stdout)
            analyzed = analyze(steps, accs, CHANCE_RATE, window=8)
            results[label][seed] = analyzed["mean_acc"]
            print(f"{label:<12} seed={seed}  mean_acc={analyzed['mean_acc']:.4f}  "
                  f"status={analyzed['status']}", flush=True)

    print(f"\narm            mean    std     per-seed  (magnitude_penalty_coef={MAGNITUDE_PENALTY_COEF})")
    for label in ("base4_dense", "base6_dense", "base12_dense", "base24_dense"):
        per_seed = [results[label][s] for s in seeds]
        print(f"{label:<14} {statistics.mean(per_seed):.4f}  "
              f"{statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0:.4f}  "
              f"{[round(v, 4) for v in per_seed]}")

    print("\nmagpen-dense vs no-penalty-dense vs sparse-echo (same base/seeds/config):")
    print(f"{'base':<8} {'magpen mean':<12} {'magpen std':<12} {'nopen mean':<12} {'sparse mean':<12}")
    for base_label, dense_label in (("base4", "base4_dense"), ("base6", "base6_dense"),
                                     ("base12", "base12_dense"), ("base24", "base24_dense")):
        per_seed = [results[dense_label][s] for s in seeds]
        m_mean = statistics.mean(per_seed)
        m_std = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        nopen = DENSE_NOPEN_REFERENCE[dense_label]
        sparse = SPARSE_REFERENCE[base_label]
        print(f"{base_label:<8} {m_mean:<12.4f} {m_std:<12.4f} {nopen['mean']:<12.4f} {sparse['mean']:<12.4f}")
