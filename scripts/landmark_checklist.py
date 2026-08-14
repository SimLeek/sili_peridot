"""Pre-commit regression checklist for training-relevant changes to
sili_peridot/model, sili_peridot/scripts, or sili__new's sili/.

Runs the 4-config dense-init L1-sparsity sweep (baseline x {energy, zero_init}
x {on, off}) and prints a table against the current best-known reference for
each config. Each of the 4 configs tracks its OWN "best so far" -- baseline
is not the only thing worth comparing against; a change meant to fix energy_rl
should be judged against baseline_energy's own current number, not baseline's.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/landmark_checklist.py [--seeds N] [--steps N]
"""
import argparse
import statistics
from scripts.l1_sparsity_probe import OriginalArchModel, run

SEEDS = [1000, 1001, 1002, 1003, 1004]
N_STEPS = 15000
COEF = 0.05

# Current best-known reference (2026-08-12, post -ffast-math NaN fix,
# sili__new setup.py's -fno-finite-math-only). Update these when a real,
# deliberate improvement lands -- not silently, and not from a single run
# (see feedback_always_regression_test_before_commit memory: run-to-run
# variance on this hardware is real, ~0.80-0.93 for baseline across
# repeats).
REFERENCE = {
    "baseline":          {"mean": 0.8667, "note": "avg of 3 runs: 0.8000, 0.8667, 0.9333"},
    "baseline_energy":   {"mean": 0.1333, "note": "skip_rate 0.000% post -ffast-math fix (was 46.8%, see JOURNAL.md)"},
    "baseline_zeroinit": {"mean": 0.0000, "note": "total failure, all seeds"},
    "zeroinit_energy":   {"mean": 0.0000, "note": "total failure, all seeds"},
}

CONFIGS = [
    ("baseline",          dict(use_energy=False, all_zero_init=False)),
    ("baseline_energy",   dict(use_energy=True,  all_zero_init=False)),
    ("baseline_zeroinit", dict(use_energy=False, all_zero_init=True)),
    ("zeroinit_energy",   dict(use_energy=True,  all_zero_init=True)),
]


def run_config(name, kwargs, seeds, n_steps):
    per_seed = []
    tot_skips = tot_calls = 0
    step_times = []
    for seed in seeds:
        model = OriginalArchModel(
            seed, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
            l1_sparsity_coef=COEF, **kwargs,
        )
        print(f"[{name}] starting seed={seed} ({n_steps} steps)...", flush=True)
        accs, skips, total, avg_step_time = run(model, n_steps, seed, verbose=True)
        per_seed.append(statistics.mean(accs[-3:]))
        tot_skips += skips
        tot_calls += total
        step_times.append(avg_step_time)
    mean = statistics.mean(per_seed)
    skip_rate = tot_skips / tot_calls if tot_calls else 0.0
    avg_step_time_overall = statistics.mean(step_times)
    return mean, per_seed, skip_rate, avg_step_time_overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--steps", type=int, default=N_STEPS)
    args = ap.parse_args()
    seeds = SEEDS[: args.seeds]

    print(f"landmark_checklist: {len(seeds)} seeds x {args.steps} steps, coef={COEF}\n")
    rows = []
    for name, kwargs in CONFIGS:
        mean, per_seed, skip_rate, avg_step_time = run_config(name, kwargs, seeds, args.steps)
        ref = REFERENCE[name]["mean"]
        delta = mean - ref
        flag = "OK" if delta >= -0.05 else "REGRESSED"
        rows.append((name, mean, ref, delta, flag, skip_rate, per_seed, avg_step_time))

    print(f"{'config':<18} {'mean':>7} {'ref':>7} {'delta':>7}  {'flag':<10} {'skip_rate':>10} {'avg_step':>10}")
    for name, mean, ref, delta, flag, skip_rate, per_seed, avg_step_time in rows:
        print(f"{name:<18} {mean:>7.4f} {ref:>7.4f} {delta:>+7.4f}  {flag:<10} {skip_rate:>9.3%} {avg_step_time*1000:>8.1f}ms")
        print(f"    per_seed={[round(v, 4) for v in per_seed]}")

    regressed = [r for r in rows if r[4] == "REGRESSED"]
    if regressed:
        print(f"\nREGRESSION in: {', '.join(r[0] for r in regressed)} -- do not commit without understanding why.")
    else:
        print("\nNo regressions vs. current reference table.")


if __name__ == "__main__":
    main()
