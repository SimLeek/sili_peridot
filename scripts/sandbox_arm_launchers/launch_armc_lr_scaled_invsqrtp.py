"""Milder bracket for launch_armc_lr_scaled_invp.py's 1/p compensation
hypothesis: 1/sqrt(p) instead, same caution as
train_curriculum.width_scaling_lr_fanin_hypothesis's own alpha=1.0-vs-
0.5 finding (the "theoretically justified" stronger exponent turned out
WORSE than the milder one there -- don't assume 1/p is right just
because the reasoning sounds clean). Same cutoff=0.3 (p~=0.403) as the
1/p version -> peak_lr = 0.01/sqrt(0.403) ~= 0.0158. See
docs/research/train_mqar_curriculum.rst:armc_gate_density_lr_equation."""

import math
import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

GATE_CUTOFF = 0.3
GATE_DENSITY_P = 0.5 - math.asin(GATE_CUTOFF) / math.pi  # ~0.403
BASE_LR = 0.01  # current leaderboard record's peak_lr
SCALED_LR = BASE_LR / math.sqrt(GATE_DENSITY_P)  # ~0.0158


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
    f"# DENSE FORWARD + ARM C BACKWARD cutoff={GATE_CUTOFF} (p~={GATE_DENSITY_P:.4f}), "
    f"peak_lr={SCALED_LR:.5f} (=0.01/sqrt(p), 1/sqrt(p) compensation), K_START=2, "
    "write_time_aux_targets=False precision=fp32 max_steps=100000 seed=1000 "
    "embed_width=36 k_first_target=3 NUM_CPUS=4",
    flush=True,
)
r = m.train_curriculum(
    "fp32",
    100000,
    1000,
    SCALED_LR,
    16,
    10,
    embed_width=36,
    wrong_streak_threshold=100000000,
    k_first_target=3,
    dy_time_gate_cutoff=GATE_CUTOFF,
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
print("DONE_ARMC_LR_SCALED_INVSQRTP", flush=True)
