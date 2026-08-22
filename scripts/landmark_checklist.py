"""Pre-commit regression checklist for training-relevant changes to
sili_peridot/model, sili_peridot/scripts, or sili__new's sili/.

Runs the 4-config dense-init L1-sparsity sweep (baseline x {energy, zero_init}
x {on, off}) and prints results AS EACH SEED/CONFIG COMPLETES (not just at
the very end) -- lets a caller pull partial numbers from the log mid-run
instead of only getting a final summary table once everything finishes.

Reports TWO metrics per seed:
  - old_style: statistics.mean(accs[-3:]), run()'s own in-training sampling.
    Only has 4 possible values per seed (0, 1/3, 2/3, 1) -- confirmed too
    coarse to tell a real regression from sampling noise on its own, kept
    only for continuity against the existing REFERENCE table (itself
    measured with this same coarse metric, so it's an apples-to-apples,
    if noisy, comparison).
  - eval_acc: evaluate(model, 100, seed), a real post-training accuracy on
    100 fresh held-out sequences at full curriculum length. This is the
    metric that should actually be trusted -- REFERENCE has no evaluate()
    -based numbers yet (added after REFERENCE was last set), so THIS run's
    eval_acc numbers are establishing a fresh baseline, not being compared
    against a prior one. Future changes should diff against these.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/landmark_checklist.py [--seeds N] [--steps N]
"""
import argparse
import statistics
from scripts.l1_sparsity_probe import OriginalArchModel, run, evaluate

SEEDS = [1000, 1001, 1002, 1003, 1004]
N_STEPS = 15000
N_EVAL = 100
COEF = 0.05

# old_style reference (2026-08-12, post -ffast-math NaN fix). No eval_acc
# reference exists yet -- this run is establishing the first one.
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
    # rank-N re-test of the zero-init arms (sili__new block4 chain-rule
    # fix + rank-N generalization + headroom-starvation fix, this
    # session) -- baseline_zeroinit/zeroinit_energy above (rank=1
    # implicitly, the scale_rank default) previously "total failure,
    # all seeds" per REFERENCE, measured before ALL of: the chain-rule
    # fix (quant*=S not /=S, letting a dead weight escape 0 at all under
    # deterministic rounding), the rank-N generalization (avoids a
    # shared row-scale value_scale cancelling across columns with
    # opposite-signed demand), and the block4 headroom fix (a weight
    # that DID escape 0 was getting evicted right back out by
    # merge_row_workspace on the very next call, silently undoing every
    # step). rank1 here is a FRESH re-baseline under the fixed formula
    # (not comparable to REFERENCE's rank1 number, which predates the
    # chain-rule/headroom fixes too) -- rank2 is the actual new
    # mechanism being tested, added specifically to avoid the
    # cross-column cancellation the rank1 shared-scale is vulnerable to.
    ("zeroinit_rank1",         dict(use_energy=False, all_zero_init=True, scale_rank=1)),
    ("zeroinit_rank2",         dict(use_energy=False, all_zero_init=True, scale_rank=2)),
    ("zeroinit_energy_rank1",  dict(use_energy=True,  all_zero_init=True, scale_rank=1)),
    ("zeroinit_energy_rank2",  dict(use_energy=True,  all_zero_init=True, scale_rank=2)),
]

# No REFERENCE entry for the rank-N configs -- see their own comment
# above, these establish a fresh baseline, not a diff against history.
_NO_REF = {"zeroinit_rank1", "zeroinit_rank2", "zeroinit_energy_rank1", "zeroinit_energy_rank2"}


def run_config(name, kwargs, seeds, n_steps, n_eval):
    per_seed_old = []
    per_seed_eval = []
    tot_skips = tot_calls = 0
    step_times = []
    for seed in seeds:
        model = OriginalArchModel(
            seed, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
            l1_sparsity_coef=COEF, **kwargs,
        )
        print(f"[{name}] starting seed={seed} ({n_steps} steps)...", flush=True)
        accs, skips, total, avg_step_time = run(model, n_steps, seed, verbose=True)
        old_style = statistics.mean(accs[-3:])
        eval_acc = evaluate(model, n_eval, seed)
        per_seed_old.append(old_style)
        per_seed_eval.append(eval_acc)
        tot_skips += skips
        tot_calls += total
        step_times.append(avg_step_time)
        # Printed IMMEDIATELY per seed -- this is what makes partial
        # results pullable mid-run instead of only at the very end.
        print(f"[{name}] seed={seed} done: old_style={old_style:.4f} "
              f"eval_acc({n_eval})={eval_acc:.4f}", flush=True)

    mean_old = statistics.mean(per_seed_old)
    mean_eval = statistics.mean(per_seed_eval)
    skip_rate = tot_skips / tot_calls if tot_calls else 0.0
    avg_step_time_overall = statistics.mean(step_times)
    print(f"[{name}] CONFIG DONE: old_style_mean={mean_old:.4f} "
          f"eval_acc_mean={mean_eval:.4f} eval_per_seed={[round(v,4) for v in per_seed_eval]} "
          f"skip_rate={skip_rate:.3%} avg_step={avg_step_time_overall*1000:.1f}ms\n", flush=True)
    return mean_old, mean_eval, per_seed_old, per_seed_eval, skip_rate, avg_step_time_overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--eval", type=int, default=N_EVAL)
    args = ap.parse_args()
    seeds = SEEDS[: args.seeds]

    print(f"landmark_checklist: {len(seeds)} seeds x {args.steps} steps x {args.eval}-eval, coef={COEF}\n")
    rows = []
    for name, kwargs in CONFIGS:
        mean_old, mean_eval, per_seed_old, per_seed_eval, skip_rate, avg_step_time = run_config(
            name, kwargs, seeds, args.steps, args.eval)
        if name in _NO_REF:
            ref, delta_old, flag = float("nan"), float("nan"), "NEW (no reference)"
        else:
            ref = REFERENCE[name]["mean"]
            delta_old = mean_old - ref
            flag = "OK" if delta_old >= -0.05 else "REGRESSED (old_style, noisy)"
        rows.append((name, mean_old, mean_eval, ref, delta_old, flag, skip_rate, per_seed_eval, avg_step_time))

    print(f"{'config':<18} {'old_style':>10} {'eval_acc':>9} {'old_ref':>8} {'flag':<28} {'skip_rate':>10} {'avg_step':>10}")
    for name, mean_old, mean_eval, ref, delta_old, flag, skip_rate, per_seed_eval, avg_step_time in rows:
        print(f"{name:<18} {mean_old:>10.4f} {mean_eval:>9.4f} {ref:>8.4f} {flag:<28} {skip_rate:>9.3%} {avg_step_time*1000:>8.1f}ms")
        print(f"    eval_per_seed={[round(v, 4) for v in per_seed_eval]}")

    print("\nNote: eval_acc has no prior reference to diff against yet (added")
    print("after REFERENCE was last set) -- these numbers ARE the new baseline")
    print("for future comparisons. old_style/old_ref is the only apples-to-")
    print("apples check against history, and it's known-noisy (4 possible")
    print("values per seed).")


if __name__ == "__main__":
    main()
