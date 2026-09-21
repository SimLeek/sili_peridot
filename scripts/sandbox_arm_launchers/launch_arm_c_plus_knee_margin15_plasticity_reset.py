"""Exact replica of launch_arm_c_plus_knee_margin15.py (Arm C backward
sine-gate + knee-adaptive x_r_target_auto, margin=0.15, the BEST of the
4 original knee variants): that run reached vocab=126/k=2 at step
34,660 then sat flat for the remaining 65,340 steps, never reaching
k=3. Supersedes launch_arm_c_plus_knee_margin15_loss_decay.py as the
primary test against this stall. Only intentional difference from the
original: plasticity_reset enabled at its default calibration (top-K
frozen pool gated by local gradient deviation + bottom-K dead pool,
both at 1%/1% -- see
docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design).
dy_time_gate_seed is left unset (matching the original exactly) --
deliberately NOT re-seeded even though a later seed-confound finding
(armc_gate_density_lr_seed_sweep) showed Arm C's own phase realization
matters; re-seeding here would add a SECOND uncontrolled difference from
the historical run being compared against, which is exactly what "do
science correctly" (enumerate every difference first) argues against.
Tests directly: does per-neuron plasticity reset let this combined
forward+backward sparsity mechanism break through k=2, unlock the k=3
it never reached."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2


def log_fn(
    step,
    vocab_size,
    k,
    phase,
    event,
    loss_ema,
    acc_ema,
    ranks=None,
    steps_per_sec=None,
    max_streak=None,
    dy_r_target=None,
    x_r_target=None,
    layer_timing=None,
    window_wall_s=None,
):
    loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
    acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
    tag = f"  [{event}]" if event else ""
    sps_s = f"  steps/sec={steps_per_sec:.1f}" if steps_per_sec is not None else ""
    streak_s = f"  max_streak={max_streak:>2}/10" if max_streak is not None else ""
    print(
        f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
        f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{streak_s}",
        flush=True,
    )


print(
    "# ARM C + knee-adaptive x_r_target (margin=0.15) + PLASTICITY_RESET "
    "(reset_fraction=0.01, dead_fraction=0.01), K_START=2, "
    "write_time_aux_targets=False precision=fp32 max_steps=100000 seed=1000 "
    "embed_width=36 k_first_target=3 NUM_CPUS=4 -- testing against the "
    "original's flat vocab=126/k=2 stall (65,340 steps flat, never reached k=3)",
    flush=True,
)
r = m.train_curriculum(
    "fp32",
    100000,
    1000,
    0.015,
    16,
    10,
    embed_width=36,
    wrong_streak_threshold=100000000,
    k_first_target=3,
    dy_time_gate_cutoff=0.3,
    x_r_target_auto=True,
    x_r_target_auto_margin=0.15,
    plasticity_reset_enable=True,
    log_every=250,
    log_fn=log_fn,
)
print(
    f"\nFINAL final_vocab={r['final_vocab']} final_k={r['final_k']} "
    f"final_phase={r['final_phase']} steps_per_sec={r['steps_per_sec']:.2f} "
    f"({r['elapsed_s']:.0f}s)",
    flush=True,
)
print(f"PEAK peak_vocab={r['peak_stage']['vocab']} peak_k={r['peak_stage']['k']}", flush=True)
print(f"STAGE_HISTORY {r['stage_history']}", flush=True)
print("DONE_ARM_C_PLUS_KNEE_MARGIN15_PLASTICITY_RESET", flush=True)
