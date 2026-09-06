"""See docs/research/lr_calibration_probe.rst:module_overview."""

import statistics

from scripts.l1_sparsity_probe import OriginalArchModel, evaluate, run

SEEDS = [1000, 1001]
N_STEPS = 1500
N_EVAL = 50
COEF = 0.05
# See docs/research/lr_calibration_probe.rst:lr_multiplier_extension.
LR_MULTIPLIERS = [10.0, 20.0, 50.0, 100.0]

if __name__ == "__main__":
    print(f"baseline config, {N_STEPS} steps, seeds={SEEDS}, coef={COEF}\n", flush=True)
    for mult in LR_MULTIPLIERS:
        peak_lr = 0.002 * mult
        per_seed_old = []
        per_seed_eval = []
        for seed in SEEDS:
            model = OriginalArchModel(
                seed,
                dense=True,
                o_proj_coef=0.0,
                all_layer_coef=0.0,
                l1_sparsity_coef=COEF,
                use_energy=False,
                all_zero_init=False,
            )
            accs, skips, total, avg_step_time = run(model, N_STEPS, seed, verbose=False, peak_lr=peak_lr)
            old_style = statistics.mean(accs[-3:]) if accs else 0.0
            eval_acc = evaluate(model, N_EVAL, seed)
            per_seed_old.append(old_style)
            per_seed_eval.append(eval_acc)
            print(
                f"  [mult={mult:.3f} peak_lr={peak_lr:.5f}] seed={seed} "
                f"old_style={old_style:.4f} eval_acc({N_EVAL})={eval_acc:.4f} "
                f"skips={skips}/{total}",
                flush=True,
            )
        print(
            f"[mult={mult:.3f} peak_lr={peak_lr:.5f}] MEAN "
            f"old_style={statistics.mean(per_seed_old):.4f} "
            f"eval_acc={statistics.mean(per_seed_eval):.4f}\n",
            flush=True,
        )
