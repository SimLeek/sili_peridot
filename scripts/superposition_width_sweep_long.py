"""Long-horizon version of superposition_width_sweep.py.

Direct motivation (see conversation): the first width sweep used a
step budget that scaled with width but was still WAY too small --
superposition_step_budget_probe.py isolated width from steps entirely
(fixed hidden_width=10, only n_steps varying) and found the
DISLDOLayer-family arms (fp32/fp4_det/fp4_stoch) were STILL improving
at 32000 steps, not plateaued, while the plain float+Adam sanity arm
was bit-identical from 1600 through 32000 steps (already stuck in a
fixed-lr local optimum, unrelated to step budget). Direct instruction:
no lr_decay for Adam either -- same lifelong-learning incompatibility
already established for RMSprop's own lr-decay "fix" (a schedule that
anneals to zero can't support open-ended training), so Adam's own
plateau is left as-is rather than propped up.

This is explicitly flagged (see conversation) as the last blocker
before this project can claim it has verified every property wanted
of a recurrent-transformer FP4 model -- so it gets a genuinely long
budget, run in the background, with real progress logging built into
this file directly (not relying on shell redirection someone could
forget to set up) so partial results are inspectable at any point
before the whole sweep finishes.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/superposition_width_sweep_long.py
Log:   sili_peridot/superposition_width_sweep_long.log (repo-root-relative, appended each run)
"""

import datetime
import os
import time

import numpy as np
from sili.sparse_rnn import DISLDOLayer, DISLDOLayer32, DISLDOLayerDeterministic

from model.eval_rank_floor import FullRankDenseLayer
from model.eval_superposition import (
    feature_importance,
    measure_superposition,
    no_superposition_baseline,
)
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_

DENSITY = 0.05
LR = 0.02
COMPRESSION_RATIO = 4  # n_features = COMPRESSION_RATIO * hidden_width
BASE_HIDDEN_WIDTH = 5
BASE_N_STEPS = 16000  # hw=10 gets 32000 -- matches superposition_step_budget_probe.py's own
# already-collected hw=10 config exactly, so that run doubles as a
# same-seed cross-check on this sweep's hw=10 point.
MAX_N_STEPS = 300000
WIDTHS = [5, 10, 20, 40, 80]
LOG_POINTS_PER_ARM = 60  # ~60 log lines per arm regardless of that arm's own step budget

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "superposition_width_sweep_long.log")


def _open_log():
    f = open(LOG_PATH, "a", buffering=1)  # line-buffered -- flushes on every newline
    f.write(f"\n=== superposition_width_sweep_long.py run started {datetime.datetime.now().isoformat()} ===\n")
    f.write(f"WIDTHS={WIDTHS} BASE_N_STEPS={BASE_N_STEPS} MAX_N_STEPS={MAX_N_STEPS} DENSITY={DENSITY} LR={LR}\n")
    return f


def _n_steps_for_width(hidden_width: int) -> int:
    return min(MAX_N_STEPS, BASE_N_STEPS * max(1, hidden_width // BASE_HIDDEN_WIDTH))


def _importance_decay_for_width(hidden_width: int) -> float:
    return (0.9**BASE_HIDDEN_WIDTH) ** (1.0 / hidden_width)


EVAL_EVERY = 20  # must match measure_superposition's own eval_every default -- log_fn is only ever
# CALLED on eval_every-aligned steps, so log_every_steps below is snapped to a
# multiple of this. Confirmed directly (see conversation): the first version of
# this sweep didn't snap, so log_fn's own "step % log_every_steps == 0" gate only
# actually fired at lcm(eval_every, log_every_steps) -- e.g. eval_every=20 and
# log_every_steps=266 (16000//60, unsnapped) share no common multiple smaller than
# 2660, silently cutting the intended ~60 points/arm down to ~7. Every width's log
# was affected identically since none of the raw n_steps//60 values happened to
# land on a multiple of 20.


def _make_log_fn(log_f, hidden_width: int, arm: str, n_steps: int):
    raw_target = max(1, n_steps // LOG_POINTS_PER_ARM)
    log_every_steps = max(EVAL_EVERY, (raw_target // EVAL_EVERY) * EVAL_EVERY)
    t0 = time.time()

    def log_fn(step, total_steps, current, best):
        if step % log_every_steps != 0 and step != total_steps:
            return
        elapsed = time.time() - t0
        log_f.write(
            f"  hw={hidden_width:>3} {arm:<12} step={step:>7}/{total_steps} "
            f"current={current:.4f} best={best:.4f} elapsed={elapsed:.1f}s\n"
        )

    return log_fn


def run_width(log_f, hidden_width: int, seed: int = 4000):
    n_features = COMPRESSION_RATIO * hidden_width
    decay = _importance_decay_for_width(hidden_width)
    n_steps = _n_steps_for_width(hidden_width)
    baseline = no_superposition_baseline(feature_importance(n_features, decay), hidden_width, DENSITY)

    log_f.write(
        f"--- hidden_width={hidden_width} n_features={n_features} n_steps={n_steps} "
        f"decay={decay:.4f} baseline={baseline:.4f} ---\n"
    )
    width_t0 = time.time()

    rng = np.random.default_rng(seed)
    float_encoder = FullRankDenseLayer(n_features, hidden_width, rng)
    float_decoder = FullRankDenseLayer(hidden_width, n_features, rng)
    float_report = measure_superposition(
        float_encoder,
        float_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed,
        importance_decay=decay,
        opt=AdamOptimizer(),
        opt_step=lambda o, p, lr: o.step(p, lr=lr),
        clip_grad_norm=clip_grad_norm_,
        log_fn=_make_log_fn(log_f, hidden_width, "float_adam", n_steps),
    )

    fp4_det_encoder = DISLDOLayerDeterministic(
        n_features, hidden_width, n_features * hidden_width, num_cpus=1, rng=np.random.default_rng(seed + 1), dense=True
    )
    fp4_det_decoder = DISLDOLayerDeterministic(
        hidden_width, n_features, hidden_width * n_features, num_cpus=1, rng=np.random.default_rng(seed + 2), dense=True
    )
    fp4_det_report = measure_superposition(
        fp4_det_encoder,
        fp4_det_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed + 3,
        importance_decay=decay,
        log_fn=_make_log_fn(log_f, hidden_width, "fp4_det", n_steps),
    )

    fp4_stoch_encoder = DISLDOLayer(
        n_features, hidden_width, n_features * hidden_width, num_cpus=1, rng=np.random.default_rng(seed + 4), dense=True
    )
    fp4_stoch_decoder = DISLDOLayer(
        hidden_width, n_features, hidden_width * n_features, num_cpus=1, rng=np.random.default_rng(seed + 5), dense=True
    )
    fp4_stoch_report = measure_superposition(
        fp4_stoch_encoder,
        fp4_stoch_decoder,
        n_features,
        hidden_width,
        DENSITY,
        n_steps,
        LR,
        seed=seed + 6,
        importance_decay=decay,
        log_fn=_make_log_fn(log_f, hidden_width, "fp4_stoch", n_steps),
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
        "n_features": n_features,
        "n_steps": n_steps,
        "baseline": baseline,
        "float_adam": float_report.best_weighted_loss,
        "fp4_det": fp4_det_report.best_weighted_loss,
        "fp4_stoch": fp4_stoch_report.best_weighted_loss,
        "fp32": fp32_report.best_weighted_loss,
    }
    log_f.write(
        f"--- hidden_width={hidden_width} DONE in {time.time() - width_t0:.1f}s: "
        f"baseline={baseline:.4f} float_adam={result['float_adam']:.4f} "
        f"fp4_det={result['fp4_det']:.4f} fp4_stoch={result['fp4_stoch']:.4f} "
        f"fp32={result['fp32']:.4f} ---\n"
    )
    return result


if __name__ == "__main__":
    log_f = _open_log()
    results = []
    sweep_t0 = time.time()
    try:
        for hw in WIDTHS:
            results.append(run_width(log_f, hw))
    finally:
        header = (
            f"\n{'hidden_w':>9} {'n_steps':>8} {'baseline':>9} {'float_adam':>11} "
            f"{'fp4_det':>9} {'fp4_stoch':>10} {'fp32':>9} "
            f"{'det/fp32':>9} {'stoch/fp32':>11} "
            f"{'det_beats_base':>15} {'stoch_beats_base':>17}\n"
        )
        log_f.write(header)
        print(header, end="")
        for r in results:
            line = (
                f"{r['hidden_width']:>9} {r['n_steps']:>8} {r['baseline']:>9.4f} "
                f"{r['float_adam']:>11.4f} {r['fp4_det']:>9.4f} {r['fp4_stoch']:>10.4f} "
                f"{r['fp32']:>9.4f} {r['fp4_det'] / r['fp32']:>9.3f} {r['fp4_stoch'] / r['fp32']:>11.3f} "
                f"{r['fp4_det'] < r['baseline']!s:>15} "
                f"{r['fp4_stoch'] < r['baseline']!s:>17}\n"
            )
            log_f.write(line)
            print(line, end="")
        log_f.write(
            f"=== TOTAL elapsed {time.time() - sweep_t0:.1f}s ({(time.time() - sweep_t0) / 60.0:.1f} min) ===\n"
        )
        log_f.close()
