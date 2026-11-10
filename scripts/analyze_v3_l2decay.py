"""Offline analysis of the v3 (l2decay) run's per-column snapshots.
One-off exploratory script (see feedback_scripts_vs_tests_convention)."""

import glob
import os

import numpy as np

BASE = "logs/plasticity_column_snapshots/dense_lr_unscaled_v3_l2decay"


def load_pool(pool):
    files = sorted(glob.glob(os.path.join(BASE, pool, "*.npz")))
    steps, loss, acc, importance, sat_ratio, decay_strength = ([] for _ in range(6))
    for f in files:
        d = np.load(f)
        steps.append(int(d["step"]))
        loss.append(float(d["loss_ema"]))
        acc.append(float(d["acc_ema"]))
        importance.append(d["col_importance"])
        sat_ratio.append(float(d["l2_sat_ratio"]))
        decay_strength.append(float(d["l2_decay_strength"]))
    return {
        "steps": np.array(steps),
        "loss": np.array(loss),
        "acc": np.array(acc),
        "importance": np.stack(importance),
        "sat_ratio": np.array(sat_ratio),
        "decay_strength": np.array(decay_strength),
    }


def main():
    pools = sorted(d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)))
    all_data = {p: load_pool(p) for p in pools}

    for pool, data in all_data.items():
        imp_mean = data["importance"].mean(axis=1)
        n = len(data["steps"])
        idxs = [int(n * f) for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)]
        print(f"\n=== {pool} === cycles={n}")
        print("  step      imp_mean   sat_ratio  decay_str  loss_ema  acc_ema")
        for i in idxs:
            print(
                f"  {data['steps'][i]:>7}   {imp_mean[i]:>8.4f}   {data['sat_ratio'][i]:>8.4f}"
                f"   {data['decay_strength'][i]:>8.4f}   {data['loss'][i]:>7.4f}  {data['acc'][i]:>7.4f}"
            )

    # global loss/acc trend using lm_head (never saturates, so its own
    # trajectory reflects overall training dynamics uncontaminated by
    # its own decay) plus a saturated pool for comparison.
    print("\n=== correlation check: decay onset vs global loss/acc ===")
    ref = all_data["lm_head.block4"]  # every pool logs the same global loss_ema/acc_ema
    v_proj = all_data["v_proj.block4"]
    onset_idx = np.argmax(v_proj["decay_strength"] > 0.5)
    onset_step = v_proj["steps"][onset_idx]
    print(f"v_proj decay_strength first exceeds 0.5 at step {onset_step}")
    pre = ref["steps"] < onset_step
    post = ref["steps"] >= onset_step
    print(
        f"  pre-onset  ({pre.sum()} pts): loss_ema mean={ref['loss'][pre].mean():.4f}  "
        f"acc_ema mean={ref['acc'][pre].mean():.4f}"
    )
    print(
        f"  post-onset ({post.sum()} pts): loss_ema mean={ref['loss'][post].mean():.4f}  "
        f"acc_ema mean={ref['acc'][post].mean():.4f}"
    )


if __name__ == "__main__":
    main()
