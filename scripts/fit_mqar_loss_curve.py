"""
scripts/fit_mqar_loss_curve.py
─────────────────────────────────
Fits loss(t) = loss_inf + (loss_0 - loss_inf) * exp(-t/tau) to a
train_mqar_rmt_ablation.py run's logged loss trajectory (its
TRAJECTORY_JSON line), and reports the estimated ASYMPTOTE loss_inf --
distinguishes "still converging toward ~0, just slower" from
"genuinely plateaued above 0" per direct instruction, since Adam isn't
tuned for a non-stationary/lifelong-learning-style objective with a
time-varying effective lr the way this project's own custom RMSprop-
with-importance policy was built for -- a run that LOOKS stuck at
15000 steps could still be asymptoting toward the right answer.

Run: python3 scripts/fit_mqar_loss_curve.py <log_file>
  (reads the TRAJECTORY_JSON line printed by train_mqar_rmt_ablation.py)
"""
import sys
import json

import numpy as np
from scipy.optimize import curve_fit


def exp_approach(t, loss_inf, amplitude, tau):
    return loss_inf + amplitude * np.exp(-t / tau)


def fit_and_report(trajectory, label=""):
    steps = np.array([t[0] for t in trajectory], dtype=np.float64)
    losses = np.array([t[1] for t in trajectory], dtype=np.float64)
    mask = np.isfinite(losses)
    steps, losses = steps[mask], losses[mask]
    if len(steps) < 4:
        print(f"{label}: not enough finite points to fit ({len(steps)})")
        return None

    loss0, loss_last = losses[0], losses[-1]
    p0 = [max(loss_last - 0.1, 0.0), max(loss0 - loss_last, 0.1), steps[-1] / 3]
    try:
        popt, _ = curve_fit(exp_approach, steps, losses, p0=p0,
                            bounds=([0.0, -20.0, 1.0], [20.0, 20.0, steps[-1] * 50]),
                            maxfev=20000)
        loss_inf, amplitude, tau = popt
        pred = exp_approach(steps, *popt)
        ss_res = np.sum((losses - pred) ** 2)
        ss_tot = np.sum((losses - losses.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"{label}: loss_inf={loss_inf:.4f}  tau={tau:.0f} steps  "
              f"R^2={r2:.4f}  (last observed loss={loss_last:.4f})")
        return {"loss_inf": loss_inf, "tau": tau, "r2": r2, "last_loss": loss_last}
    except Exception as e:
        print(f"{label}: fit failed - {type(e).__name__}: {e}")
        return None


def main():
    log_file = sys.argv[1]
    with open(log_file) as f:
        content = f.read()
    line = next((l for l in content.splitlines() if l.startswith("TRAJECTORY_JSON ")), None)
    if line is None:
        print("No TRAJECTORY_JSON line found in", log_file)
        return
    trajectory = json.loads(line[len("TRAJECTORY_JSON "):])
    fit_and_report(trajectory, label=log_file)


if __name__ == "__main__":
    main()
