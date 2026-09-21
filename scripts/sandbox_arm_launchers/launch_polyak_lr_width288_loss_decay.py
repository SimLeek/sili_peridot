"""Exact replica of launch_polyak_lr_width288.py (the RECALIBRATED
polyak_lr=True run -- f_star=0, c=0.0005, lr_max=0.05, denom=Lbar):
that run reached vocab=32/k=3 at step 6,149 then sat flat for the
remaining ~94,000 steps of a 100k-step budget, well short of the
hand-tuned peak_lr=0.01 record (vocab=126/k=3 at step 13,601). Only
intentional difference here: loss_adjusted_decay enabled at its default
calibration (importance-only, importance_half_life_touches=500, weight
arm off -- see
docs/research/toy_tile_recurrence_rmt.rst:loss_adjusted_decay_design).
Tests directly: does sustained-loss-triggered importance decay let
Polyak's own per-layer LR mechanism break through ITS stall, everything
else held identical (same seed=1000, same architecture, same
curriculum) -- a genuine post-bugfix stall, not a config that was
merely mis-tuned."""

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
    "# DENSE + POLYAK_LR + LOSS_ADJUSTED_DECAY (importance_half_life_touches=500, "
    "weight arm off), K_START=2, write_time_aux_targets=False precision=fp32 "
    "max_steps=100000 seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- "
    "testing against the original's flat vocab=32/k=3 stall (~94,000 steps flat)",
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
    loss_adjusted_decay_enable=True,
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
print("DONE_POLYAK_LR_WIDTH288_LOSS_DECAY", flush=True)
