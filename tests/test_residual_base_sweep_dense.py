"""Dense-connectivity counterpart to test_residual_base_sweep.py --
answers the ORIGINAL question the block4 dense loader was built for
(does dense connectivity reduce seed-to-seed variance vs the sparse
echo network), now that the permanent-NaN divergence bug blocking it
is fixed (JOURNAL.md 2026-08-10, later still; project_hybrid_precision
_plan / project_sili_block4_dense_loader memory). Identical config to
the sparse sweep (same FIXED_ARGS/POST_SEED_ARGS/seeds) so the two
result tables are directly comparable -- only `dense=True` differs.

    SILI_RUN_BASE_SWEEP=1 pytest tests/test_residual_base_sweep_dense.py -s
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

ARMS = {
    "base4_dense": "true_multi_digit_deterministic_base4_dense",
    "base6_dense": "true_multi_digit_deterministic_base6_dense",
    "base12_dense": "true_multi_digit_deterministic_dense",  # project default base, dense connectivity
    "base24_dense": "true_multi_digit_deterministic_base24_dense",
}

# Sparse echo-network reference (base=4/6/12/24, same 5 seeds, same
# fixed config) -- JOURNAL.md / project_hybrid_precision_plan memory,
# 2026-08-10. Recorded here so this run's own output can report a
# direct side-by-side without needing to re-run the sparse sweep too.
SPARSE_REFERENCE = {
    "base4":  {"mean": 0.6417, "std": 0.1006, "per_seed": [0.723, 0.750, 0.633, 0.498, 0.604]},
    "base6":  {"mean": 0.6929, "std": 0.0786, "per_seed": [0.750, 0.652, 0.725, 0.575, 0.762]},
    "base12": {"mean": 0.7296, "std": 0.0429, "per_seed": [0.744, 0.746, 0.785, 0.681, 0.692]},
    "base24": {"mean": 0.6775, "std": 0.0614, "per_seed": [0.733, 0.700, 0.665, 0.577, 0.713]},
}


@pytest.mark.skipif(not os.environ.get(RUN_ENV_VAR),
                    reason=f"expensive multi-seed real training run, opt in via {RUN_ENV_VAR}=1")
def test_dense_base_sweep_vs_sparse_reference():
    seeds = _seeds()
    results = {label: {} for label in ARMS}
    for label, arm in ARMS.items():
        for seed in seeds:
            args = [*FIXED_ARGS, str(seed), *POST_SEED_ARGS]
            proc = subprocess.run(
                [sys.executable, SCRIPT, arm, *args],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
            assert proc.returncode == 0, f"{arm} seed={seed} failed:\n{proc.stderr}"
            steps, accs = _parse_checkpoints_from_text(proc.stdout)
            analyzed = analyze(steps, accs, CHANCE_RATE, window=8)
            results[label][seed] = analyzed["mean_acc"]
            print(f"{label:<12} seed={seed}  mean_acc={analyzed['mean_acc']:.4f}  "
                  f"status={analyzed['status']}", flush=True)

    print("\narm            mean    std     per-seed")
    for label in ("base4_dense", "base6_dense", "base12_dense", "base24_dense"):
        per_seed = [results[label][s] for s in seeds]
        print(f"{label:<14} {statistics.mean(per_seed):.4f}  "
              f"{statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0:.4f}  "
              f"{[round(v, 4) for v in per_seed]}")

    print("\ndense vs sparse-echo reference (same base, same seeds/config):")
    print(f"{'base':<8} {'dense mean':<12} {'dense std':<12} {'sparse mean':<12} {'sparse std':<12}")
    for base_label, dense_label in (("base4", "base4_dense"), ("base6", "base6_dense"),
                                     ("base12", "base12_dense"), ("base24", "base24_dense")):
        per_seed = [results[dense_label][s] for s in seeds]
        d_mean = statistics.mean(per_seed)
        d_std = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        ref = SPARSE_REFERENCE[base_label]
        print(f"{base_label:<8} {d_mean:<12.4f} {d_std:<12.4f} {ref['mean']:<12.4f} {ref['std']:<12.4f}")
