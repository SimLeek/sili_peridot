"""MQAR curriculum (k_first_target odometer mode) remote launcher with
per-layer wall-time breakdown + dy/x_r_target trajectory logging.
Refactored to import launcher_common instead of its own
near-byte-identical copy of log_fn/trajectory_log_fn (the same
closures now live in launcher_common.py, lifted from
train_mqar_curriculum.main() -- this file predates that extraction)."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m
from scripts.sandbox_arm_launchers.launcher_common import make_log_fn, make_trajectory_log_fn, print_final_summary

m.NUM_CPUS = 4

print(
    "# MQAR curriculum (k_first_target odometer mode) precision=fp32 max_steps=100000 seed=1000 "
    "peak_lr=0.015 num_tiles=16 k_max=10 embed_width=36 r_target_min=0.5 "
    "dy_r_target=0.9 x_r_target=0.9 target_steps_per_sec=95.69 k_first_target=3",
    flush=True,
)

r = m.train_curriculum(
    "fp32",
    100000,
    1000,
    0.015,
    16,
    10,
    additive_rank=1,
    dynamic_rank_control=True,
    rank_grace_period_steps=50,
    embed_width=36,
    wrong_streak_threshold=100000000,
    dy_r_target=0.9,
    target_steps_per_sec=95.69,
    x_r_target=0.9,
    r_target_min=0.5,
    k_first_target=3,
    trajectory_log_every=2000,
    log_fn=make_log_fn(),
    trajectory_log_fn=make_trajectory_log_fn(),
)
print_final_summary(r, label="kfirst_remote")
