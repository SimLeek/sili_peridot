"""Isolated probe: does scale_rank=2 help the NON-zero-init baseline arms
too, not just the zero-init escape-from-0 scenario it was built for?

Run separately from landmark_checklist.py (per direct request -- "isolated
runs") rather than folded into that suite's CONFIGS, so it doesn't disturb
an already-running invocation of that script (CONFIGS is read once at
import time; editing the file after launch has no effect on a process
already running). Reuses landmark_checklist's own run_config() for
identical methodology (same seeds, same steps, same eval() metric, same
per-seed incremental printing) so the numbers are directly comparable to
that script's baseline/baseline_energy rows.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/rank2_baseline_probe.py [--seeds N] [--steps N]
"""
import argparse
import statistics
from scripts.landmark_checklist import run_config, SEEDS, N_STEPS, N_EVAL

CONFIGS = [
    ("baseline_rank2",        dict(use_energy=False, all_zero_init=False, scale_rank=2)),
    ("baseline_energy_rank2", dict(use_energy=True,  all_zero_init=False, scale_rank=2)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--eval", type=int, default=N_EVAL)
    args = ap.parse_args()
    seeds = SEEDS[: args.seeds]

    print(f"rank2_baseline_probe: {len(seeds)} seeds x {args.steps} steps x {args.eval}-eval\n", flush=True)
    rows = []
    for name, kwargs in CONFIGS:
        mean_old, mean_eval, per_seed_old, per_seed_eval, skip_rate, avg_step_time = run_config(
            name, kwargs, seeds, args.steps, args.eval)
        rows.append((name, mean_old, mean_eval, skip_rate, per_seed_eval, avg_step_time))

    # landmark_checklist.py's own REFERENCE for direct comparison --
    # rank=1 (implicit, the scale_rank default) numbers for the SAME two
    # arms, so the rank2-vs-rank1 delta is readable at a glance.
    rank1_ref = {"baseline_rank2": 0.8667, "baseline_energy_rank2": 0.1333}

    print(f"{'config':<24} {'old_style':>10} {'eval_acc':>9} {'rank1_old_ref':>14} {'delta':>8} {'skip_rate':>10} {'avg_step':>10}")
    for name, mean_old, mean_eval, skip_rate, per_seed_eval, avg_step_time in rows:
        ref = rank1_ref[name]
        delta = mean_old - ref
        print(f"{name:<24} {mean_old:>10.4f} {mean_eval:>9.4f} {ref:>14.4f} {delta:>+8.4f} {skip_rate:>9.3%} {avg_step_time*1000:>8.1f}ms")
        print(f"    eval_per_seed={[round(v, 4) for v in per_seed_eval]}")

    print("\nrank1_old_ref is landmark_checklist.py's REFERENCE for baseline/")
    print("baseline_energy (old_style metric, rank=1 implicit) -- delta shows")
    print("whether scale_rank=2 helps even without zero-init.")


if __name__ == "__main__":
    main()
