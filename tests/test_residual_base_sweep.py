"""Optional integration test: reproduces the base=4/6/12/24 residual
-scale sweep that justified switching TrueMultiDigitLayer's default
`base` from 4.0 to 12.0 (see JOURNAL.md 2026-08-10 and the
project_hybrid_precision_plan memory). NOT part of the default test
run -- a real multi-seed training comparison (default 5 seeds x 4
arms, ~35 minutes), not a unit test. Opt in with:

    SILI_RUN_BASE_SWEEP=1 pytest tests/test_residual_base_sweep.py -s
    SILI_BASE_SWEEP_SEEDS=1000,1001,1002 SILI_RUN_BASE_SWEEP=1 pytest ... -s

Prints the same result table recorded in JOURNAL.md and asserts the
one claim this test exists to guard: base=12 (the current default)
beats base=4 (the old default) on a MAJORITY of seeds, not just one.

Direct correction that shaped this test's design, worth keeping in
mind when reading it: the FIRST version of this comparison (a single
seed=1000 run per arm) was itself confounded -- ToyTileRecurrenceRealFP4
never passed `rng=` down to its layers, so every layer's initial
connectivity/weight values were genuinely unseeded regardless of the
`seed` CLI arg (confirmed directly: same seed, same command, gave
0.70 then 0.65 final-step accuracy across two runs). That bug is now
fixed (model/toy_tile_precision_models.py, model/toy_precision_models.py),
so `seed` is real now -- but a single seed only ever shows "there
exists a draw where X wins," never "X usually wins." Hence the
multi-seed, paired (same seed set across every arm) design here.
base=24 is reported but not asserted on -- it was still LEARNING (not
plateaued) at 15000 steps in the original single-seed sweep, so a
pass/fail threshold on it would be testing noise, not a real
regression.
"""
import os
import re
import statistics
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.learning_slope import analyze

RUN_ENV_VAR = "SILI_RUN_BASE_SWEEP"
SEEDS_ENV_VAR = "SILI_BASE_SWEEP_SEEDS"
DEFAULT_SEEDS = [1000, 1001, 1002, 1003, 1004]
SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "train_tile_curriculum.py")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

# Same config the original sweep used (JOURNAL.md 2026-08-10): use_energy=0,
# use_attention=1, 15000 steps, checkpoint every 750, steps_per_stage=1500,
# embed_width=16, column_neurons=8, max_weights=1500, seq_len_max=8
# (out-of-context), peak_lr=0.002, o_proj_depth=1. `seed` is inserted
# per-run below.
FIXED_ARGS = ["0", "1", "15000", "750"]  # use_energy use_attention train_steps checkpoint_every
POST_SEED_ARGS = ["1500", "16", "8", "1500", "8", "0.002", "1"]
CHANCE_RATE = 0.1  # VOCAB=10 in train_tile_curriculum.py

ARMS = {
    "base4": "true_multi_digit_deterministic_base4",
    "base6": "true_multi_digit_deterministic_base6",
    "base12": "true_multi_digit_deterministic",  # current default
    "base24": "true_multi_digit_deterministic_base24",
}


def _seeds():
    raw = os.environ.get(SEEDS_ENV_VAR)
    if not raw:
        return DEFAULT_SEEDS
    return [int(s) for s in raw.split(",")]


@pytest.mark.skipif(not os.environ.get(RUN_ENV_VAR),
                    reason=f"expensive multi-seed real training run, opt in via {RUN_ENV_VAR}=1 "
                           f"(control seed count with {SEEDS_ENV_VAR}=1000,1001,...)")
def test_residual_base_sweep_reports_results():
    seeds = _seeds()
    # results[label][seed] = mean_acc for that (arm, seed) run
    results = {label: {} for label in ARMS}
    for label, arm in ARMS.items():
        for seed in seeds:
            args = [*FIXED_ARGS, str(seed), *POST_SEED_ARGS]
            proc = subprocess.run(
                [sys.executable, SCRIPT, arm, *args],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
            assert proc.returncode == 0, f"{arm} seed={seed} failed:\n{proc.stderr}"
            steps, accs = _parse_checkpoints_from_text(proc.stdout)
            analyzed = analyze(steps, accs, CHANCE_RATE, window=8)
            results[label][seed] = analyzed["mean_acc"]
            print(f"{label:<6} seed={seed}  mean_acc={analyzed['mean_acc']:.4f}  "
                  f"status={analyzed['status']}", flush=True)

    print("\nbase   mean(mean_acc)  std(mean_acc)  per-seed")
    for label in ("base4", "base6", "base12", "base24"):
        per_seed = [results[label][s] for s in seeds]
        print(f"{label:<6} {statistics.mean(per_seed):.4f}          "
              f"{statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0:.4f}         "
              f"{[round(v, 4) for v in per_seed]}")

    # Paired comparison (same seed set for every arm): count how many
    # seeds base12 beats base4 on, rather than trusting a single
    # cross-seed mean -- more robust at this small N, and directly
    # answers "does base=12 usually win," matching what this test
    # exists to check.
    base4_wins = sum(1 for s in seeds if results["base12"][s] > results["base4"][s])
    print(f"\nbase12 beats base4 on {base4_wins}/{len(seeds)} seeds")

    assert base4_wins > len(seeds) / 2, (
        f"base=12 (current default) only beat base=4 (old default) on "
        f"{base4_wins}/{len(seeds)} seeds -- the switch to base=12 as the "
        f"project default needs revisiting.")


def _parse_checkpoints_from_text(text: str):
    """parse_checkpoints (scripts/learning_slope.py) reads from a file
    path; the subprocess run here only has stdout in memory, so
    replicate its regex directly against text instead of round
    -tripping through a temp file."""
    pat = re.compile(r"step=\s*(\d+).*?acc=([\d.]+)")
    steps, accs = [], []
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            accs.append(float(m.group(2)))
    return np.array(steps, dtype=np.float64), np.array(accs, dtype=np.float64)
