"""
scripts/mqar_multi_seed_validation.py
──────────────────────────────────────
Reusable multi-seed A/B harness for train_mqar_curriculum.py's
train_curriculum(). Built for task #366 (does per-layer nucleus input
selection change quality/speed vs the old fixed-fraction top-k?) and
meant to be reused for #370's full-mechanism validation afterward --
same script, different ARMS/BASE_KWARGS.

Exists because the real engine has confirmed run-to-run
nondeterminism (backward_sparse threading nondeterminism, see
JOURNAL.md "Follow-up run result" entry: an exact rerun of arm F --
same seed, same code, same config -- produced 5 LEVEL_UPs vs the
original's 1, flipping a "late-run relapse" read into "healthy").
A single-seed comparison between two configs cannot tell a real
effect from that noise floor. The fix is MORE seeds, not RNG pinning
(see feedback_statistical_power_not_seeding memory) -- this script
makes "how many seeds does it take to tell" directly answerable and
repeatable instead of ad hoc per-task reasoning.

Mirrors lr_per_row_nnz_ab_test.py's CONFIGS/run_arm/print-table
convention rather than inventing a new one.

Define ARMS/BASE_KWARGS below as plain train_curriculum() kwarg
dicts and edit directly per validation question -- simpler and more
reliable than threading yet more args through this project's
positional-argv CLI convention (unwieldy past ~20 positions,
confirmed directly while testing task #369).

Usage: PYTHONPATH=<repo root> python3 scripts/mqar_multi_seed_validation.py [num_seeds] [base_seed]
"""

import statistics
import sys

from scripts.train_mqar_curriculum import train_curriculum

BASE_SEED = 1
NUM_SEEDS = 5
MAX_STEPS = 3000

# Matches arm F's own config (JOURNAL.md "nucleus/energy-threshold top-k
# math" entries): embed_width=48, fp32, same peak_lr/num_tiles/k_max as
# every other arm-F-derived comparison this session. additive_rank=0 and
# dynamic_rank_control=False are required, not optional, for
# precision="fp32" -- DISLDOLayer32 doesn't accept either kwarg at all
# (confirmed directly: train_curriculum's own defaults of additive_rank=1/
# dynamic_rank_control=True TypeError on layer construction for fp32),
# unlike fp4/fp8's DISLDOLayer/DISLDOLayer8.
BASE_KWARGS = {
    "precision": "fp32",
    "max_steps": MAX_STEPS,
    "peak_lr": 0.02,
    "num_tiles": 8,
    "k_max": 4,
    "embed_width": 48,
    "additive_rank": 0,
    "dynamic_rank_control": False,
}

# task #366: fixed-fraction top-k (today's baseline) vs per-layer nucleus
# selection (x_r_target), matched at ~10% density on input, dense (1.0)
# on grad -- isolates the INPUT axis only, same as arm F itself did.
ARMS = [
    ("fixed_fraction_p10", {"input_sparsity_p": 0.10, "dy_sparsity_p": 1.0}),
    ("nucleus_r_target", {"x_r_target": 0.90, "x_k_min": 1, "dy_sparsity_p": 1.0}),
]


def run_arm(name, kwargs, seeds):
    per_seed = []
    for seed in seeds:
        print(f"[{name}] starting seed={seed} ({MAX_STEPS} steps)...", flush=True)
        r = train_curriculum(seed=seed, **{**BASE_KWARGS, **kwargs})
        row = {
            "seed": seed,
            "peak_vocab": r["peak_stage"]["vocab"],
            "peak_k": r["peak_stage"]["k"],
            "final_vocab": r["final_vocab"],
            "final_k": r["final_k"],
            "graduated": r["graduated"],
            "steps_per_sec": r["steps_per_sec"],
            "elapsed_s": r["elapsed_s"],
        }
        per_seed.append(row)
        print(
            f"[{name}] seed={seed} done: peak_vocab={row['peak_vocab']} "
            f"peak_k={row['peak_k']} sps={row['steps_per_sec']:.3f} "
            f"elapsed={row['elapsed_s']:.0f}s",
            flush=True,
        )
    return per_seed


def summarize(per_seed, field):
    vals = [row[field] for row in per_seed]
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, stdev, vals


def noise_floor_verdict(a_vals, b_vals):
    """Rough two-sample check: |mean diff| vs combined stderr. NOT a
    real t-distribution p-value (no scipy dependency available) --
    z>=2 means 'worth trusting / worth more seeds to nail down', not
    a formal significance proof."""
    if len(a_vals) < 2 or len(b_vals) < 2:
        return "need >=2 seeds per arm to compare"
    ma, mb = statistics.mean(a_vals), statistics.mean(b_vals)
    sa, sb = statistics.stdev(a_vals), statistics.stdev(b_vals)
    se = (sa**2 / len(a_vals) + sb**2 / len(b_vals)) ** 0.5
    if se == 0:
        return f"delta={ma - mb:+.3f}, zero combined spread"
    z = abs(ma - mb) / se
    verdict = "likely real, not just noise" if z >= 2.0 else "indistinguishable from noise at this seed count"
    return f"delta={ma - mb:+.3f}, z={z:.2f} -> {verdict}"


def main():
    num_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_SEEDS
    base_seed = int(sys.argv[2]) if len(sys.argv) > 2 else BASE_SEED
    seeds = [base_seed + i for i in range(num_seeds)]

    print(f"multi-seed validation: {len(ARMS)} arms x {num_seeds} seeds (seeds={seeds}), max_steps={MAX_STEPS}\n")

    results = {name: run_arm(name, kwargs, seeds) for name, kwargs in ARMS}

    print("\n" + "=" * 100)
    print(f"{'arm':<24} {'peak_vocab':>14} {'peak_k':>10} {'sps':>10} {'graduated':>10}")
    for name, _ in ARMS:
        per_seed = results[name]
        pv_mean, pv_std, pv_vals = summarize(per_seed, "peak_vocab")
        pk_mean, _, _ = summarize(per_seed, "peak_k")
        sps_mean, _, _ = summarize(per_seed, "steps_per_sec")
        grad_rate = sum(1 for r in per_seed if r["graduated"]) / len(per_seed)
        print(f"{name:<24} {pv_mean:>8.2f}±{pv_std:<5.2f} {pk_mean:>10.2f} {sps_mean:>10.3f} {grad_rate:>9.1%}")
        print(f"    peak_vocab_per_seed={pv_vals}")

    if len(ARMS) == 2:
        (name_a, _), (name_b, _) = ARMS
        a_vals = [r["peak_vocab"] for r in results[name_a]]
        b_vals = [r["peak_vocab"] for r in results[name_b]]
        print(f"\n{name_a} vs {name_b} (peak_vocab): {noise_floor_verdict(a_vals, b_vals)}")
        a_sps = [r["steps_per_sec"] for r in results[name_a]]
        b_sps = [r["steps_per_sec"] for r in results[name_b]]
        print(f"{name_a} vs {name_b} (steps_per_sec): {noise_floor_verdict(a_sps, b_sps)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
