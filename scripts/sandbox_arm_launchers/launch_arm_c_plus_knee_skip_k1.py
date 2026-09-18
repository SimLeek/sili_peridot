"""Same as launch_arm_c_plus_knee.py, but skips ONLY the k=1 stage --
every vocab tier starts at k=2, then cycles to k=3 as usual. Tests the
hypothesis that k=1's easy positional/relative shortcut solution
entrenches synapses in a way that actively blocks the harder k=3
associative-binding solution, rather than helping via an easier warm-up
(see this file's own K_START override, and the pre-existing codebase
WARNING on the same risk, docs/research/train_mqar_curriculum.rst:
train_curriculum.k_first_target_odometer_reordering). k=2 is kept, not
skipped too -- the model already passes it cleanly (step 291 in the
dense reference's own STAGE_HISTORY), so it's a legitimate warm-up, not
a suspected shortcut; only k=1 is implicated."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2  # skip k=1 only -- see module docstring


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
    "# ARM C + knee-adaptive x_r_target, K_START=2 (skip k=1 only) "
    "precision=fp32 max_steps=100000 seed=1000 embed_width=36 k_first_target=3 "
    "NUM_CPUS=4",
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
    x_r_target_auto_margin=0.05,
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
print("DONE_ARM_C_PLUS_KNEE_SKIP_K1", flush=True)
