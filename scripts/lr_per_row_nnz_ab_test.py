"""A/B test: lr_per_row_nnz=True vs False, both CORRECTLY implemented
(raw lr passed straight through to every layer's .forward(), no Python
-side pre-division) -- measures whether damping the per-synapse weight
-CODE update by each row's live-connection count actually helps or hurts
final task accuracy, now that both arms are known to be mechanistically
sound.

Superseded an earlier, broken version of this same comparison: that
version pre-divided lr in Python to approximate what lr_per_row_nnz=True
does internally, then called .forward(lr_per_row_nnz=False) -- this
silently double-divided value_scale's gradient (which is ALWAYS
normalized by nnz_row unconditionally, independent of the flag, inside
linear_disldo.hpp's scale_eff_lr), crippling value_scale's adaptation to
~1/32 of its correct rate under the "False" arm. Confirmed via a debug
print (nnz_row=32 exactly) and a 200-step bit-identical comparison that
the per-synapse CODE update itself was never the problem; confirmed via
step-bisection that the two arms diverged STRUCTURALLY (different live
-connection counts) a few thousand steps in, once value_scale's
under-adaptation let true (scale-multiplied) weight magnitudes drift far
enough apart to cross different FP4 rounding/promotion thresholds. Fixed
in l1_sparsity_probe.py's step() by removing the pre-division entirely;
this script now tests the two REAL, uncorrupted configurations.

Measured with the new evaluate() metric (n_eval=100 fresh sequences per
seed) rather than run()'s own in-training accuracy sampling, which only
has 4 possible per-seed values (0, 1/3, 2/3, 1) -- confirmed too coarse
to tell a real regression from sampling noise.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/lr_per_row_nnz_ab_test.py
"""

import statistics

from scripts.l1_sparsity_probe import OriginalArchModel, evaluate, run

SEEDS = [1000, 1001, 1002, 1003, 1004]
N_STEPS = 15000
N_EVAL = 100
COEF = 0.05

CONFIGS = [
    ("baseline", {"use_energy": False, "all_zero_init": False}),
    ("baseline_energy", {"use_energy": True, "all_zero_init": False}),
    ("baseline_zeroinit", {"use_energy": False, "all_zero_init": True}),
    ("zeroinit_energy", {"use_energy": True, "all_zero_init": True}),
]

LR_ARMS = [
    ("damped_true", {"lr_per_row_nnz": True}),
    ("undamped_false", {"lr_per_row_nnz": False}),
]


def run_arm(config_name, config_kwargs, arm_name, arm_kwargs, seeds, n_steps, n_eval):
    eval_per_seed = []
    old_style_per_seed = []
    tot_skips = tot_calls = 0
    for seed in seeds:
        model = OriginalArchModel(
            seed,
            dense=True,
            o_proj_coef=0.0,
            all_layer_coef=0.0,
            l1_sparsity_coef=COEF,
            **config_kwargs,
            **arm_kwargs,
        )
        print(f"[{config_name}/{arm_name}] starting seed={seed} ({n_steps} steps)...", flush=True)
        accs, skips, total, _avg_step_time = run(model, n_steps, seed, verbose=True)
        old_style_per_seed.append(statistics.mean(accs[-3:]))
        tot_skips += skips
        tot_calls += total
        eval_acc = evaluate(model, n_eval, seed)
        eval_per_seed.append(eval_acc)
        print(
            f"[{config_name}/{arm_name}] seed={seed} done: "
            f"old_style={old_style_per_seed[-1]:.4f} eval_acc({n_eval})={eval_acc:.4f}",
            flush=True,
        )
    skip_rate = tot_skips / tot_calls if tot_calls else 0.0
    return {
        "eval_mean": statistics.mean(eval_per_seed),
        "eval_per_seed": eval_per_seed,
        "old_style_mean": statistics.mean(old_style_per_seed),
        "old_style_per_seed": old_style_per_seed,
        "skip_rate": skip_rate,
    }


def main():
    print(
        f"lr_per_row_nnz A/B test (both arms correctly implemented): "
        f"{len(SEEDS)} seeds x {N_STEPS} steps x {N_EVAL}-eval, coef={COEF}\n"
    )
    results = {}
    for config_name, config_kwargs in CONFIGS:
        for arm_name, arm_kwargs in LR_ARMS:
            key = f"{config_name}/{arm_name}"
            results[key] = run_arm(config_name, config_kwargs, arm_name, arm_kwargs, SEEDS, N_STEPS, N_EVAL)

    print("\n" + "=" * 100)
    print(f"{'config/arm':<32} {'eval_mean':>10} {'old_style':>10} {'skip_rate':>10}")
    for config_name, _ in CONFIGS:
        for arm_name, _ in LR_ARMS:
            key = f"{config_name}/{arm_name}"
            r = results[key]
            print(f"{key:<32} {r['eval_mean']:>10.4f} {r['old_style_mean']:>10.4f} {r['skip_rate']:>9.3%}")
            print(f"    eval_per_seed={[round(v, 4) for v in r['eval_per_seed']]}")
        true_r = results[f"{config_name}/damped_true"]
        false_r = results[f"{config_name}/undamped_false"]
        delta = false_r["eval_mean"] - true_r["eval_mean"]
        print(f"    -> undamped_false vs damped_true eval_mean delta: {delta:+.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
