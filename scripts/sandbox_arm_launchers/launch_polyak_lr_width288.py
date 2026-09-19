"""Validates per_layer_learning_rate_polyak against the current
leaderboard record: does polyak_lr=True (defaults, recalibrated
2026-09-19: f_star=0, c=0.0005, lr_max=0.05, denominator=Lbar not raw
E_t -- see apply_polyak_lr's own docstring for the recalibration
story) reach vocab=126/k=3 comparably to or better than the hand-found
peak_lr=0.01 (step 13,601, ~30min), without needing any peak_lr tuned
by hand at all. Otherwise identical to launch_dense_lr_scaled.py (full
dense, K_START=2, write_time_aux_targets=False, LEVEL_DOWN disabled)
-- peak_lr itself is irrelevant here (only used as polyak_lr's
bootstrap_lr for the very first step, before any layer has an Lbar
yet) since polyak_lr overrides it for all 5 wide layers every step
after that. First run (old c=0.5/lr_max=0.1/raw-E_t defaults) failed
to learn at all (vocab=16/k=3, flat for 99,287 steps) -- this is the
re-run under the fixed calibration. See
docs/research/toy_tile_recurrence_rmt.rst:per_layer_learning_rate_polyak."""

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
    "# DENSE + POLYAK_LR (f_star=0, c=0.5, lr_max=0.1), K_START=2, "
    "write_time_aux_targets=False precision=fp32 max_steps=100000 seed=1000 "
    "embed_width=36 k_first_target=3 NUM_CPUS=4",
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
    polyak_lr=True,
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
print("DONE_POLYAK_LR_WIDTH288", flush=True)
