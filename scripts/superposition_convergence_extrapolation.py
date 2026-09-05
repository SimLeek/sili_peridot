"""Fits best_weighted_loss(step) for each (hidden_width, arm) in the
long-run superposition width sweep's own log file, and estimates
whether/when each curve would cross under no_superposition_baseline --
i.e. does this arm ever beat the baseline, and if not yet, at roughly
what step count would it, if the observed trend continued.

Model: power-law-with-floor, L(step) = L_inf + A * (step+1)^(-p).
This is the standard "scaling law" functional form for loss-vs-training
-steps curves (Kaplan et al. and follow-ups), and matches this
project's own established convergence-extrapolation practice (see
sili__new's test_synapse_policy_long_horizon.cpp, which used a
geometric-ratio extrapolation on evenly-spaced checkpoints -- same
underlying idea, generalized here to a proper nonlinear fit instead of
a single constant ratio, since these curves are read off unevenly-
weighted early-vs-late data and a full curve_fit is more robust than a
single last-few-deltas ratio).

L_inf is the model's own estimate of the loss floor as step -> infinity.
If L_inf >= baseline, the fit says this arm's trend does NOT predict
ever beating the baseline (a real, falsifiable claim about the CURRENT
trend continuing -- not proof it's impossible, just what the observed
trajectory implies). If L_inf < baseline, solves for the step at which
L(step) == baseline.

Fits only the SECOND HALF of each arm's logged checkpoints by default
(skip_frac=0.5) -- the earliest steps show a much steeper initial drop
than the later "true" asymptotic tail, and including that transient
biases a single power-law fit across the whole range (confirmed by
comparing fit quality with/without the skip).

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/superposition_convergence_extrapolation.py [log_path]
Reads scripts/../superposition_width_sweep_long.log by default -- safe
to run against a log file that's still being appended to by an
in-progress run (reads whatever's on disk at invocation time; only
returns fits for (width, arm) pairs with at least MIN_POINTS logged
checkpoints).
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

LOG_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "superposition_width_sweep_long.log")
MIN_POINTS = 8
SKIP_FRAC = 0.5

LINE_RE = re.compile(r"hw=\s*(\d+)\s+(\S+)\s+step=\s*(\d+)/(\d+)\s+current=([\d.]+)\s+best=([\d.]+)")
BASELINE_RE = re.compile(r"hidden_width=(\d+) n_features=\d+ n_steps=\d+ decay=[\d.]+ baseline=([\d.]+)")


def parse_log(log_path: str):
    """Returns (series, baselines): series[(hidden_width, arm)] -> list of
    (step, best); baselines[hidden_width] -> baseline value."""
    series = defaultdict(list)
    baselines = {}
    with open(log_path) as f:
        for line in f:
            m = BASELINE_RE.search(line)
            if m:
                baselines[int(m.group(1))] = float(m.group(2))
                continue
            m = LINE_RE.search(line)
            if m:
                hw, arm, step, _total, _current, best = m.groups()
                series[(int(hw), arm)].append((int(step), float(best)))
    return series, baselines


def _power_law(step, l_inf, a, p):
    return l_inf + a * np.power(step + 1.0, -p)


# p bounded well away from 0 -- an unbounded-near-zero p lets A blow up arbitrarily while
# still tracking the visible points (p~0 means "no real decay", so L_inf/A become nearly
# unidentifiable and curve_fit can land on absurd A/L_inf combinations that fit the observed
# window fine but extrapolate to nonsense). Confirmed directly (see conversation): the
# unbounded version produced crossing-step estimates like 5.8e13 and A~1e13 on real data.
P_MIN, P_MAX = 0.05, 3.0
FLAT_R2_THRESHOLD = 0.02  # curve barely moved across the fit window at all


def fit_power_law(steps: np.ndarray, losses: np.ndarray):
    """Returns (l_inf, a, p, r_squared) or None if the fit didn't converge
    or the data is too flat/noisy to trust a fit at all."""
    spread = losses.max() - losses.min()
    if spread < FLAT_R2_THRESHOLD * max(losses.mean(), 1e-8):
        return None  # effectively flat over the fit window -- no trend to extrapolate
    l_inf_guess = max(losses.min() * 0.5, 1e-8)
    p0 = [l_inf_guess, max(losses.max() - l_inf_guess, 1e-6), 0.5]
    bounds = ([0.0, 0.0, P_MIN], [losses.max(), np.inf, P_MAX])
    try:
        popt, _ = curve_fit(_power_law, steps, losses, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        return None
    pred = _power_law(steps, *popt)
    ss_res = np.sum((losses - pred) ** 2)
    ss_tot = np.sum((losses - losses.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (*popt, r_squared)


def estimate_crossing_step(l_inf: float, a: float, p: float, baseline: float, max_observed_step: float):
    """Returns the step at which the fitted curve crosses baseline, or
    None if the fitted asymptote L_inf never gets there. Refuses to report
    a crossing step more than 1000x past the last observed step -- that far
    out, the power-law tail is pure extrapolation past anything the fit
    window actually constrains, not a real estimate (confirmed directly:
    unconstrained values like step=5.8e13 against a ~16000-step fit window
    are numerically "a fit" but not a meaningful prediction)."""
    if l_inf >= baseline or a <= 0:
        return None
    ratio = (baseline - l_inf) / a
    if ratio <= 0:
        return None
    crossing = ratio ** (-1.0 / p) - 1.0
    if crossing > 1000.0 * max_observed_step:
        return float("inf")  # signals "unbounded/unreliable extrapolation", not a real estimate
    return crossing


def main(log_path: str = LOG_PATH_DEFAULT):
    series, baselines = parse_log(log_path)
    widths = sorted({hw for hw, _ in series})
    print(
        f"{'hw':>4} {'arm':<12} {'n_pts':>6} {'L_inf':>9} {'A':>10} {'p':>6} {'R2':>6} "
        f"{'baseline':>9} {'crossing_step':>15}"
    )
    for hw in widths:
        baseline = baselines.get(hw)
        arms = sorted({arm for w, arm in series if w == hw})
        for arm in arms:
            pts = sorted(series[(hw, arm)])
            n = len(pts)
            if n < MIN_POINTS:
                print(f"{hw:>4} {arm:<12} {n:>6} -- too few points logged yet, skipping fit")
                continue
            baseline_str = f"{baseline:.4f}" if baseline is not None else "?"

            # Already-crossed is a fact readable directly off observed data --
            # no need to trust a power-law extrapolation for it, and the fit
            # (which only sees the tail after SKIP_FRAC) can otherwise produce
            # a nonsensical NEGATIVE "future" crossing step for an arm that
            # already crossed baseline earlier in training.
            already_crossed_step = None
            if baseline is not None:
                for s, b in pts:
                    if b < baseline:
                        already_crossed_step = s
                        break
            if already_crossed_step is not None:
                print(
                    f"{hw:>4} {arm:<12} {n:>6} {'':>9} {'':>10} {'':>6} {'':>6} "
                    f"{baseline_str:>9} {'already@' + str(already_crossed_step):>15}"
                )
                continue

            skip = int(n * SKIP_FRAC)
            steps = np.array([s for s, _ in pts[skip:]], dtype=np.float64)
            losses = np.array([b for _, b in pts[skip:]], dtype=np.float64)
            fit = fit_power_law(steps, losses)
            if fit is None:
                print(f"{hw:>4} {arm:<12} {n:>6} -- flat / no fittable trend over fit window")
                continue
            l_inf, a, p, r2 = fit
            crossing = estimate_crossing_step(l_inf, a, p, baseline, steps.max()) if baseline is not None else None
            if crossing is None:
                crossing_str = f"never (L_inf={l_inf:.4f}>baseline)" if baseline is not None else "n/a"
            elif crossing == float("inf"):
                crossing_str = "unreliable (>>1000x fit window)"
            else:
                crossing_str = f"{crossing:.0f}"
            print(
                f"{hw:>4} {arm:<12} {n:>6} {l_inf:>9.4f} {a:>10.4f} {p:>6.3f} {r2:>6.3f} "
                f"{baseline_str:>9} {crossing_str:>15}"
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else LOG_PATH_DEFAULT)
