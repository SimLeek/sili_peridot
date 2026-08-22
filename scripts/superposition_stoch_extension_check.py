"""One-time falsification check (see conversation): does ANY fp4-based
superposition arm actually cross under no_superposition_baseline in
REAL observed data, not just an extrapolated power-law prediction?
Across every width run so far (superposition_width_sweep_long.py,
superposition_multi_digit_probe.py), only fp32 has actually crossed --
fp4_det/multi_digit_det never showed a fittable downward trend at all,
and fp4_stoch/multi_digit_stoch only ever had FITTED crossing-step
predictions, never an observed one.

The two cheapest, most-plausible-to-actually-reach predictions were
both at hidden_width=5: single-digit fp4_stoch (fit predicted ~24596,
against its already-run 16000-step budget) and multi_digit_stoch (fit
predicted ~41964, against its already-run 16000-step budget). This
reruns BOTH from scratch with the SAME construction/training seeds used
in those original runs, extended well past each one's predicted
crossing step. NOTE (confirmed directly, see conversation): this does
NOT reproduce the earlier trajectories bit-for-bit -- FP4's stochastic
rounding draws from its own internal RNG that the `rng=` construction
parameter does not seed (only initial weights/connectivity are seeded
by it), matching this project's own already-documented gap (see memory
feedback_seed_stochastic_rng_for_comparisons.md). So this is an
independent stochastic replicate at the same config, not a literal
continuation -- still a valid, arguably BETTER test of "can fp4_stoch
actually cross baseline given enough steps" (an independent draw
reaching the same conclusion is stronger evidence than extending the
literal same trajectory would have been), just not what "extend the
run" normally implies. This is explicitly NOT meant to become a
standing test -- a one-off check of whether the curve-fit predictions
from superposition_convergence_extrapolation.py can be trusted, and
more importantly, whether ANY fp4 arm can be shown to genuinely achieve
superposition at all, not just a projected one.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/superposition_stoch_extension_check.py
"""
import datetime
import os
import time

import numpy as np

from model.eval_superposition import feature_importance, measure_superposition, no_superposition_baseline
from model.toy_precision_models import TrueMultiDigitLayer
from sili.sparse_rnn import DISLDOLayer

HIDDEN_WIDTH = 5
N_FEATURES = 20
DENSITY = 0.05
LR = 0.02
DECAY = 0.9  # hidden_width==BASE_HIDDEN_WIDTH in both source scripts -> decay is exactly 0.9, unscaled

FP4_STOCH_N_STEPS = 40000       # predicted crossing ~24596, extend with real margin past it
MULTI_DIGIT_STOCH_N_STEPS = 60000  # predicted crossing ~41964, extend with real margin past it

BASELINE = no_superposition_baseline(feature_importance(N_FEATURES, DECAY), HIDDEN_WIDTH, DENSITY)

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "superposition_stoch_extension_check.log")
EVAL_EVERY = 20
LOG_POINTS = 80


def _open_log():
    f = open(LOG_PATH, "a", buffering=1)
    f.write(f"\n=== superposition_stoch_extension_check.py run started {datetime.datetime.now().isoformat()} ===\n")
    f.write(f"HIDDEN_WIDTH={HIDDEN_WIDTH} baseline={BASELINE:.4f} "
            f"fp4_stoch_n_steps={FP4_STOCH_N_STEPS} multi_digit_stoch_n_steps={MULTI_DIGIT_STOCH_N_STEPS}\n")
    return f


def _make_log_fn(log_f, arm, n_steps):
    log_every_steps = max(EVAL_EVERY, ((n_steps // LOG_POINTS) // EVAL_EVERY) * EVAL_EVERY)
    t0 = time.time()
    crossed_at = {"step": None}

    def log_fn(step, total_steps, current, best):
        crossed = current < BASELINE
        if crossed and crossed_at["step"] is None:
            crossed_at["step"] = step
            log_f.write(f"  {arm:<18} *** CROSSED BASELINE at step={step} current={current:.4f} "
                        f"< baseline={BASELINE:.4f} ***\n")
        if step % log_every_steps != 0 and step != total_steps:
            return
        elapsed = time.time() - t0
        marker = " [BELOW BASELINE]" if crossed else ""
        log_f.write(f"  {arm:<18} step={step:>7}/{total_steps} current={current:.4f} best={best:.4f} "
                    f"elapsed={elapsed:.1f}s{marker}\n")

    return log_fn, crossed_at


def run_fp4_stoch(log_f):
    seed = 4000  # matches superposition_width_sweep_long.py's own seed for hidden_width=5
    max_weights = N_FEATURES * HIDDEN_WIDTH
    encoder = DISLDOLayer(N_FEATURES, HIDDEN_WIDTH, max_weights, num_cpus=1,
                          rng=np.random.default_rng(seed + 4), dense=True)
    decoder = DISLDOLayer(HIDDEN_WIDTH, N_FEATURES, max_weights, num_cpus=1,
                          rng=np.random.default_rng(seed + 5), dense=True)
    log_fn, crossed_at = _make_log_fn(log_f, "fp4_stoch", FP4_STOCH_N_STEPS)
    report = measure_superposition(encoder, decoder, N_FEATURES, HIDDEN_WIDTH, DENSITY,
                                    FP4_STOCH_N_STEPS, LR, seed=seed + 6, importance_decay=DECAY,
                                    log_fn=log_fn)
    return report, crossed_at["step"]


def run_multi_digit_stoch(log_f):
    seed = 6000  # matches superposition_multi_digit_probe.py's own seed for hidden_width=5
    max_weights = N_FEATURES * HIDDEN_WIDTH
    encoder = TrueMultiDigitLayer(N_FEATURES, HIDDEN_WIDTH, max_weights, num_cpus=1,
                                  digit_cls=DISLDOLayer, n_stages=2, base=12.0, lr_power=0.0,
                                  dense=True, rng=np.random.default_rng(seed + 1))
    decoder = TrueMultiDigitLayer(HIDDEN_WIDTH, N_FEATURES, max_weights, num_cpus=1,
                                  digit_cls=DISLDOLayer, n_stages=2, base=12.0, lr_power=0.0,
                                  dense=True, rng=np.random.default_rng(seed + 2))
    log_fn, crossed_at = _make_log_fn(log_f, "multi_digit_stoch", MULTI_DIGIT_STOCH_N_STEPS)
    report = measure_superposition(encoder, decoder, N_FEATURES, HIDDEN_WIDTH, DENSITY,
                                    MULTI_DIGIT_STOCH_N_STEPS, LR, seed=seed + 3, importance_decay=DECAY,
                                    log_fn=log_fn)
    return report, crossed_at["step"]


if __name__ == "__main__":
    log_f = _open_log()
    t0 = time.time()

    fp4_report, fp4_crossed = run_fp4_stoch(log_f)
    log_f.write(f"--- fp4_stoch DONE: best={fp4_report.best_weighted_loss:.4f} "
                f"baseline={BASELINE:.4f} crossed_at={fp4_crossed} ---\n")

    md_report, md_crossed = run_multi_digit_stoch(log_f)
    log_f.write(f"--- multi_digit_stoch DONE: best={md_report.best_weighted_loss:.4f} "
                f"baseline={BASELINE:.4f} crossed_at={md_crossed} ---\n")

    summary = (f"\n=== SUMMARY (hidden_width={HIDDEN_WIDTH}, baseline={BASELINE:.4f}) ===\n"
              f"fp4_stoch: best={fp4_report.best_weighted_loss:.4f} crossed_at={fp4_crossed} "
              f"(predicted ~24596)\n"
              f"multi_digit_stoch: best={md_report.best_weighted_loss:.4f} crossed_at={md_crossed} "
              f"(predicted ~41964)\n"
              f"total elapsed {time.time() - t0:.1f}s\n")
    log_f.write(summary)
    print(summary)
    log_f.close()
