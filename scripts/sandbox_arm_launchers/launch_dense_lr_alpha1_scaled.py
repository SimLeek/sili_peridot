"""Same as launch_width288_nolevel_down_control.py (full dense, all
three historical-success factors matched) but peak_lr=0.0067 -- the
alpha=1.0 (muP hidden-layer, lr ~ 1/fan_in) prediction of
train_curriculum.width_scaling_lr_fanin_hypothesis, as opposed to
launch_dense_lr_scaled.py's peak_lr=0.01 (the alpha=0.5 prediction,
already confirmed to resolve the degeneracy by itself). Third data
point on lr(N) ~= lr_base * (N_base/N)^alpha to help pin alpha instead
of only bracketing it. See docs/research/train_mqar_curriculum.rst:
train_curriculum.width_scaling_lr_fanin_hypothesis."""

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
    "# DENSE, peak_lr=0.0067 (alpha=1.0 muP prediction, 0.015*128/288), K_START=2, "
    "write_time_aux_targets=False precision=fp32 max_steps=100000 seed=1000 "
    "embed_width=36 k_first_target=3 NUM_CPUS=4",
    flush=True,
)
r = m.train_curriculum(
    "fp32",
    100000,
    1000,
    0.0067,
    16,
    10,
    embed_width=36,
    wrong_streak_threshold=100000000,
    k_first_target=3,
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
print("DONE_DENSE_LR_ALPHA1_SCALED", flush=True)
