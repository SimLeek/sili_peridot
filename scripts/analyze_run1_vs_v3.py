"""Compare run1 (graduated, step 18366) against the original v3 run
(stuck at vocab=126/k=2 for ~88k steps), at matched step counts, using
the column-level stats both runs logged. One-off exploratory script
(see feedback_scripts_vs_tests_convention)."""

import glob
import os

import numpy as np

RUN1 = "logs/plasticity_column_snapshots/dense_lr_unscaled_v3_diagnostic_run1_graduated"
V3 = "logs/plasticity_column_snapshots/dense_lr_unscaled_v3_l2decay"

GRADUATION_STEP = 18366


def load_pool(base, pool):
    files = sorted(
        glob.glob(os.path.join(base, pool, "*.npz")),
        key=lambda f: int(f.split("step")[1].split(".")[0]),
    )
    steps, col_mean, l2sat, l2decay = [], [], [], []
    for f in files:
        d = np.load(f)
        steps.append(int(d["step"]))
        col_mean.append(float(d["col_importance"].mean()))
        l2sat.append(float(d["l2_sat_ratio"]))
        l2decay.append(float(d["l2_decay_strength"]))
    return {
        "steps": np.array(steps),
        "col_mean": np.array(col_mean),
        "l2sat": np.array(l2sat),
        "l2decay": np.array(l2decay),
    }


def nearest(data, target_step):
    i = np.argmin(np.abs(data["steps"] - target_step))
    return i


def main():
    pools = ["input_proj.block4", "q_proj.block4", "k_proj.block4", "v_proj.block4", "o_proj.block4", "lm_head.block4"]
    checkpoints = [2000, 5000, 8000, 11000, 14000, 17000, 18000]

    for pool in pools:
        run1 = load_pool(RUN1, pool)
        v3 = load_pool(V3, pool)
        print(f"\n=== {pool} ===")
        print(f"  run1(graduated) n_cycles={len(run1['steps'])} up to step {run1['steps'][-1]}")
        print(f"  v3(stuck)       n_cycles={len(v3['steps'])} up to step {v3['steps'][-1]} (full run 100k)")
        print("  step     | run1: col_mean l2sat l2decay  | v3: col_mean l2sat l2decay")
        for cp in checkpoints:
            if cp > run1["steps"][-1]:
                break
            i1 = nearest(run1, cp)
            i3 = nearest(v3, cp)
            print(
                f"  {cp:>7} | run1: {run1['col_mean'][i1]:>7.3f} {run1['l2sat'][i1]:>6.3f} {run1['l2decay'][i1]:>6.3f}"
                f"  | v3: {v3['col_mean'][i3]:>7.3f} {v3['l2sat'][i3]:>6.3f} {v3['l2decay'][i3]:>6.3f}"
                f"  (v3 step={v3['steps'][i3]}, run1 step={run1['steps'][i1]})"
            )


if __name__ == "__main__":
    main()
