"""Does TrueMultiDigitLayer (2 FP4 digit-layers at different scales, summed
-- base=12 by default, high_digit + low_digit/base) close the fp4-vs-fp32
superposition gap, relative to a single plain FP4 layer? Direct
motivation (see conversation): stochastic single-digit FP4 unexpectedly
BEAT the fp32 reference at rank-2 in the rank-floor test, and the user
suspects this might be related to the same effect TrueMultiDigitLayer is
built to exploit -- two low-precision representations combined can carry
more effective information than either alone. Checking whether that
holds for superposition packing specifically, at the two widths
(hidden_width=5, 20) already run in superposition_width_sweep_long.py so
the comparison is apples-to-apples (same n_steps, same importance-decay
scaling, same density/lr).

Logs periodic progress to superposition_multi_digit_probe.log (repo-root
-relative, matching superposition_width_sweep_long.py's own logging
convention) -- confirmed directly (see conversation): the first version
of this script had no log_fn wired in at all, so there was no trajectory
to inspect afterward, just final numbers. Fixed here rather than left
as a one-off gap, since this script is otherwise a near-duplicate of
the long-run sweep and should behave the same way.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/superposition_multi_digit_probe.py
"""

import datetime
import os
import time

import numpy as np
from sili.sparse_rnn import DISLDOLayer, DISLDOLayer32, DISLDOLayerDeterministic

from model.eval_superposition import (
    feature_importance,
    measure_superposition,
    no_superposition_baseline,
)
from model.toy_precision_models import TrueMultiDigitLayer

DENSITY = 0.05
LR = 0.02
COMPRESSION_RATIO = 4
BASE_HIDDEN_WIDTH = 5
# Matches superposition_width_sweep_long.py's own per-width config exactly,
# so results are directly comparable to the already-collected single-digit
# fp4_stoch/fp32/baseline numbers.
WIDTH_CONFIGS = {5: 16000, 20: 64000}

EVAL_EVERY = 20  # must match measure_superposition's own eval_every default -- see
# superposition_width_sweep_long.py's own EVAL_EVERY comment for why
# log_every_steps below has to be snapped to a multiple of this.
LOG_POINTS_PER_ARM = 60
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "superposition_multi_digit_probe.log")


def _open_log():
    f = open(LOG_PATH, "a", buffering=1)
    f.write(f"\n=== superposition_multi_digit_probe.py run started {datetime.datetime.now().isoformat()} ===\n")
    f.write(f"WIDTH_CONFIGS={WIDTH_CONFIGS} DENSITY={DENSITY} LR={LR}\n")
    return f


def _make_log_fn(log_f, hidden_width, arm, n_steps):
    raw_target = max(1, n_steps // LOG_POINTS_PER_ARM)
    log_every_steps = max(EVAL_EVERY, (raw_target // EVAL_EVERY) * EVAL_EVERY)
    t0 = time.time()

    def log_fn(step, total_steps, current, best):
        if step % log_every_steps != 0 and step != total_steps:
            return
        elapsed = time.time() - t0
        log_f.write(
            f"  hw={hidden_width:>3} {arm:<16} step={step:>7}/{total_steps} "
            f"current={current:.4f} best={best:.4f} elapsed={elapsed:.1f}s\n"
        )

    return log_fn


def _decay(hidden_width):
    return (0.9**BASE_HIDDEN_WIDTH) ** (1.0 / hidden_width)


def run_width(log_f, hidden_width, seed=6000):
    n_features = COMPRESSION_RATIO * hidden_width
    n_steps = WIDTH_CONFIGS[hidden_width]
    decay = _decay(hidden_width)
    baseline = no_superposition_baseline(feature_importance(n_features, decay), hidden_width, DENSITY)
    max_weights = n_features * hidden_width

    log_f.write(
        f"--- hidden_width={hidden_width} n_features={n_features} n_steps={n_steps} "
        f"decay={decay:.4f} baseline={baseline:.4f} ---\n"
    )
    width_t0 = time.time()

    md_stoch_encoder = TrueMultiDigitLayer(
        n_features,
        hidden_width,
        max_weights,
        num_cpus=1,
        digit_cls=DISLDOLayer,
        n_stages=2,
        base=12.0,
        lr_power=0.0,
        dense=True,
        rng=np.random.default_rng(seed + 1),
    )
    md_stoch_decoder = TrueMultiDigitLayer(
        hidden_width,
        n_features,
        max_weights,
        num_cpus=1,
        digit_cls=DISLDOLayer,
        n_stages=2,
        base=12.0,
        lr_power=0.0,
        dense=True,
        rng=np.random.default_rng(seed + 2),
    )
    md_stoch_report = measure_superposition(
        md_stoch_encoder,
        md_stoch_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed + 3,
        importance_decay=decay,
        log_fn=_make_log_fn(log_f, hidden_width, "multi_digit_stoch", n_steps),
    )

    md_det_encoder = TrueMultiDigitLayer(
        n_features,
        hidden_width,
        max_weights,
        num_cpus=1,
        digit_cls=DISLDOLayerDeterministic,
        n_stages=2,
        base=12.0,
        lr_power=0.0,
        dense=True,
        rng=np.random.default_rng(seed + 4),
    )
    md_det_decoder = TrueMultiDigitLayer(
        hidden_width,
        n_features,
        max_weights,
        num_cpus=1,
        digit_cls=DISLDOLayerDeterministic,
        n_stages=2,
        base=12.0,
        lr_power=0.0,
        dense=True,
        rng=np.random.default_rng(seed + 5),
    )
    md_det_report = measure_superposition(
        md_det_encoder,
        md_det_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed + 6,
        importance_decay=decay,
        log_fn=_make_log_fn(log_f, hidden_width, "multi_digit_det", n_steps),
    )

    fp32_encoder = DISLDOLayer32(
        n_features, hidden_width, n_features * hidden_width, num_cpus=1, rng=np.random.default_rng(seed + 7)
    )
    fp32_decoder = DISLDOLayer32(
        hidden_width, n_features, hidden_width * n_features, num_cpus=1, rng=np.random.default_rng(seed + 8)
    )
    fp32_report = measure_superposition(
        fp32_encoder,
        fp32_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed + 9,
        importance_decay=decay,
        log_fn=_make_log_fn(log_f, hidden_width, "fp32", n_steps),
    )

    result = {
        "hidden_width": hidden_width,
        "n_steps": n_steps,
        "baseline": baseline,
        "multi_digit_stoch": md_stoch_report.best_weighted_loss,
        "multi_digit_det": md_det_report.best_weighted_loss,
        "fp32": fp32_report.best_weighted_loss,
    }
    log_f.write(
        f"--- hidden_width={hidden_width} DONE in {time.time() - width_t0:.1f}s: "
        f"baseline={baseline:.4f} multi_digit_stoch={result['multi_digit_stoch']:.4f} "
        f"multi_digit_det={result['multi_digit_det']:.4f} fp32={result['fp32']:.4f} ---\n"
    )
    return result


if __name__ == "__main__":
    log_f = _open_log()
    header = (
        f"{'hw':>4} {'n_steps':>8} {'baseline':>9} {'multi_digit_stoch':>18} "
        f"{'multi_digit_det':>16} {'fp32':>9} {'md_stoch_beats_base':>20}\n"
    )
    print(header, end="")
    log_f.write(header)
    for hw in sorted(WIDTH_CONFIGS):
        r = run_width(log_f, hw)
        line = (
            f"{r['hidden_width']:>4} {r['n_steps']:>8} {r['baseline']:>9.4f} "
            f"{r['multi_digit_stoch']:>18.4f} {r['multi_digit_det']:>16.4f} {r['fp32']:>9.4f} "
            f"{r['multi_digit_stoch'] < r['baseline']!s:>20}\n"
        )
        print(line, end="")
        log_f.write(line)
    log_f.close()
