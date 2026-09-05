"""Fast (~5-10 min) check of the "is the effective lr smaller now" hypothesis
for the `baseline` config's landmark_checklist regression (0.8667 -> 0.3333
old_style_mean after sili__new's importance-signal contrib addition +
square-then-sum reversion, see conversation).

Rationale: `ci`'s denominator now includes `contrib^2` (or previously
`(g+contrib)^2`) on top of plain `g^2` -- a term that did not exist at all
when REFERENCE was measured. Since the weight step is
`-lr*g/sqrt(ci)`, a larger `ci` means a smaller step for essentially every
synapse, essentially all the time. If that's the whole story (not a
correctness bug), bumping peak_lr on a SHORT run should recover accuracy
back toward the un-regressed baseline without touching the C++ formula at
all. This is deliberately NOT the 15000-step x 5-seed landmark sweep --
short curriculum-saturated runs (seq_len maxes out at step 1000, see
STEPS_PER_STAGE/NUM_TILES) are enough to see the trend.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/lr_calibration_probe.py
"""

import statistics

from scripts.l1_sparsity_probe import OriginalArchModel, evaluate, run

SEEDS = [1000, 1001]
N_STEPS = 1500
N_EVAL = 50
COEF = 0.05
# 1x = current default (peak_lr=0.002). sqrt(2)x is the naive correction if
# g and contrib are typically comparable magnitude and square-then-sum's
# extra term roughly doubles ci on average (sqrt(2) undoes a doubled
# denominator). 2x is a coarser overcorrection to see if recovery
# continues past sqrt(2) (would suggest the effect is bigger than a simple
# doubling) or overshoots into instability.
LR_MULTIPLIERS = [10.0, 20.0, 50.0, 100.0]
# Extended past 1x-6x (eval_acc climbed 0.24->0.32->0.38->0.44->0.50->0.54,
# monotonic but flattening, no skips at any point) -- pushing much further
# to distinguish a real plateau from a log-shaped "diminishing but never
# actually stopping" curve, and to find where (if anywhere) instability
# actually kicks in.

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
