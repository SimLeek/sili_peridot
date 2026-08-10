"""Generic learning-slope detector: parses any log with "step=N acc=X"
checkpoint lines (the format every train_tile_*.py script in this repo
already prints) and reports whether accuracy is trending up (learning),
flat (plateaued), or indistinguishable from chance-level noise (dead/
collapsed) -- not specific to any one model/task, works on any such log.

Method: robust linear regression (Theil-Sen slope, median of pairwise
slopes -- resistant to the kind of single-checkpoint noise spikes this
project's own eval convention is known to produce at small EVAL_SEQUENCES,
see the earlier VOCAB=40/n=60 chance-rate discussion) of acc vs step
over the last WINDOW checkpoints, plus a one-sided test of whether the
window's mean is statistically above the given chance rate.

Usage: python3 learning_slope.py <logfile> [chance_rate] [window]
  chance_rate defaults to 1/40 (this repo's usual VOCAB=40 toy task).
"""
from __future__ import annotations

import re
import sys

import numpy as np


def parse_checkpoints(path: str):
    steps, accs = [], []
    pat = re.compile(r"step=\s*(\d+).*?acc=([\d.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                accs.append(float(m.group(2)))
    return np.array(steps, dtype=np.float64), np.array(accs, dtype=np.float64)


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Median of all pairwise slopes -- robust to the single-checkpoint
    noise spikes a small eval sample size produces (a model that's
    actually dead can still show one checkpoint at 2x chance by pure
    variance, see the VOCAB=40/n=60 discussion this tool exists to
    replace eyeballing)."""
    n = len(x)
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0:
                slopes.append((y[j] - y[i]) / dx)
    return float(np.median(slopes)) if slopes else 0.0


def analyze(steps: np.ndarray, accs: np.ndarray, chance_rate: float, window: int):
    if len(steps) == 0:
        return {"status": "no_data"}

    w_steps = steps[-window:]
    w_accs = accs[-window:]
    n = len(w_accs)

    slope = theil_sen_slope(w_steps, w_accs)
    # slope in accuracy-per-step; scale to "per checkpoint" for readability
    step_gap = float(np.median(np.diff(steps))) if len(steps) > 1 else 1.0
    slope_per_ckpt = slope * step_gap

    mean_acc = float(np.mean(w_accs))
    std_acc = float(np.std(w_accs))
    sem = std_acc / np.sqrt(n) if n > 1 else float("inf")
    # one-sided z-test: is the window's mean accuracy above chance?
    z = (mean_acc - chance_rate) / sem if sem > 0 else (float("inf") if mean_acc > chance_rate else 0.0)

    # classification -- deliberately conservative thresholds, since this
    # project's own established lesson is that small-sample accuracy
    # noise is easy to mistake for a real signal (see JOURNAL.md's
    # repeated corrections on this exact point tonight)
    if mean_acc <= chance_rate * 1.5 and z < 2.0:
        status = "DEAD/CHANCE"          # not distinguishable from a collapsed model
    elif slope_per_ckpt > 0.01 and z > 1.5:
        status = "LEARNING"             # real upward trend, above-chance mean
    elif abs(slope_per_ckpt) <= 0.01:
        status = "PLATEAUED"            # flat, whether at chance or above it
    elif slope_per_ckpt < -0.01:
        status = "DEGRADING"            # real downward trend -- instability, not just noise
    else:
        status = "AMBIGUOUS"

    return {
        "status": status,
        "n_checkpoints_total": len(steps),
        "n_checkpoints_window": n,
        "window_step_range": (int(w_steps[0]), int(w_steps[-1])) if n else None,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "chance_rate": chance_rate,
        "z_vs_chance": z,
        "slope_per_step": slope,
        "slope_per_checkpoint": slope_per_ckpt,
        "latest_acc": float(accs[-1]),
        "max_acc_ever": float(np.max(accs)),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 learning_slope.py <logfile> [chance_rate] [window]")
        sys.exit(1)
    path = sys.argv[1]
    chance_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0 / 40.0
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    steps, accs = parse_checkpoints(path)
    result = analyze(steps, accs, chance_rate, window)

    print(f"# {path}")
    if result["status"] == "no_data":
        print("  no checkpoint lines found")
        return
    print(f"  status={result['status']}")
    print(f"  checkpoints: {result['n_checkpoints_total']} total, "
          f"window={result['n_checkpoints_window']} (steps {result['window_step_range']})")
    print(f"  mean_acc={result['mean_acc']:.4f}  std={result['std_acc']:.4f}  "
          f"chance={result['chance_rate']:.4f}  z_vs_chance={result['z_vs_chance']:.2f}")
    print(f"  slope: {result['slope_per_step']:.6f}/step "
          f"({result['slope_per_checkpoint']:+.4f}/checkpoint)")
    print(f"  latest_acc={result['latest_acc']:.4f}  max_acc_ever={result['max_acc_ever']:.4f}")


if __name__ == "__main__":
    main()
