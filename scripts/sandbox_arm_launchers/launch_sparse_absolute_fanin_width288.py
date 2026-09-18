"""Test 2 of train_curriculum.width_scaling_lr_fanin_hypothesis: an
ABSOLUTE (non-width-proportional) per-neuron fan-in cap, not dense's
fan-in=state_width. dense=False, wide_max_weights=64*state_width (avg
realized fan-in ~= 64 per neuron via synaptogenesis, same F as
launch_sparse_absolute_fanin_width128.py -- only state_width differs
between the two). peak_lr is left at the UNSCALED default (0.015) --
the whole point is testing whether a fixed absolute fan-in budget
removes the need for the LR cut dense needed at this width. Otherwise
identical to launch_width288_nolevel_down_control.py (K_START=2,
write_time_aux_targets=False, LEVEL_DOWN disabled). See
docs/research/train_mqar_curriculum.rst:
train_curriculum.width_scaling_lr_fanin_hypothesis."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

STATE_WIDTH = 288
FANIN_PER_NEURON = 64


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
    f"# SPARSE, dense=False wide_max_weights={FANIN_PER_NEURON}*{STATE_WIDTH}="
    f"{FANIN_PER_NEURON * STATE_WIDTH} (absolute fan-in~={FANIN_PER_NEURON}/neuron), "
    "peak_lr=0.015 (UNSCALED), K_START=2, write_time_aux_targets=False precision=fp32 "
    "max_steps=100000 seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=4",
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
    dense=False,
    wide_max_weights=FANIN_PER_NEURON * STATE_WIDTH,
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
print("DONE_SPARSE_ABSOLUTE_FANIN_WIDTH288", flush=True)
