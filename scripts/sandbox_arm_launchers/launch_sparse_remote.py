import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

sys.argv = [
    "train_mqar_curriculum.py",
    "fp32",
    "100000",
    "1000",
    "0.015",
    "16",
    "10",
    "1",
    "1",
    "50",
    "0",
    "0",
    "36",
    "-1",
    "-1",
    "-1",
    "0",
    "-1",
    "100000000",
    "0.9",
    "0",
    "-1",
    "95.69",
    "-1",  # dy_r_target=0.9 seed, target_steps_per_sec=95.69 (unreachable -> ratchets to floor)
    "0.9",
    "0",
    "2000",
    "0.5",  # x_r_target=0.9 seed, trajectory_log_every=2000, r_target_min=0.5 (matches task #417's confirmed-plateau config)
]
m.main()
