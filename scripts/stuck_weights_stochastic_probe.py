"""Companion to stuck_weights_scale_dependence_probe.py: tests whether
STOCHASTIC rounding (DISLDOLayer, not DISLDOLayerDeterministic) unsticks
FP4 weights at TOY scale, as a control against the scale-dependence
probe. If stochastic rounding helps even at toy scale, that's evidence
this really is a rounding-threshold issue (a sub-ULP but genuinely
nonzero signal, which stochastic rounding is specifically designed to
let accumulate in expectation). If it does NOT help at toy scale, that's
evidence toward the OTHER hypothesis from conversation -- the rank-N
value_scale/output_scale correction is absorbing essentially all the
real gradient signal before it ever reaches individual synapses, in
which case there's no real signal left for stochastic rounding to
round in the first place, and the fix has to be something else (force
more of the work onto individual synapses, or scale-dependence rescues
it as state_width grows toward realistic model sizes).

Same controlled setup as the scale-dependence probe (only digit_cls
differs -- same lr/steps/seed/width).

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/stuck_weights_stochastic_probe.py
"""

import functools

import numpy as np
from sili.sparse_rnn import DISLDOLayer, DISLDOLayerDeterministic

from model.eval_stuck_weights import check_stuck_weights, snapshot_multi_digit_state
from model.toy_precision_models import TrueMultiDigitLayer
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from scripts.train_tile_curriculum import _build_tile_window, generate_copy_sequence

VOCAB = 10
EMBED_WIDTH = 8
COLUMN_NEURONS = 4
NUM_TILES = 4
MAX_WEIGHTS = 128
N_STEPS = 800
LR = 0.01
SEED = 1000


def run_probe(digit_backend, label):
    digit_cls = functools.partial(
        TrueMultiDigitLayer, digit_cls=digit_backend, n_stages=3, base=12.0, lr_power=0.0, dense=True, scale_rank=1
    )
    rng = np.random.default_rng(SEED)
    model = ToyTileRecurrenceRealFP4(
        VOCAB,
        EMBED_WIDTH,
        COLUMN_NEURONS,
        mlp_hidden=0,
        num_tiles=NUM_TILES,
        max_weights=MAX_WEIGHTS,
        num_cpus=1,
        disldo_cls=digit_cls,
        rng=rng,
    )
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    task_rng = np.random.RandomState(SEED)
    embed_table = task_rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3
    opt = AdamOptimizer()

    before = snapshot_multi_digit_state(model.o_proj)

    for _step in range(N_STEPS):
        tokens, pairs = generate_copy_sequence(task_rng, VOCAB, NUM_TILES)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        total_loss = None
        for i in range(NUM_TILES):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, _aux = model.step(window, M, LR)
            if i in targets:
                tgt_loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                total_loss = tgt_loss if total_loss is None else total_loss + tgt_loss
        if total_loss is not None:
            total_loss.backward()
            clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
            opt.step(model.parameters_for_optimizer(), lr=LR)

    after = snapshot_multi_digit_state(model.o_proj)
    report = check_stuck_weights(before, after)
    print(
        f"[{label}] n_synapses={report.n_synapses} "
        f"n_new={report.n_new} n_died={report.n_died} "
        f"churn_fraction={report.churn_fraction:.4f} "
        f"n_high_importance={report.n_high_importance} "
        f"stuck_fraction={report.stuck_fraction:.4f} "
        f"excess_stuck_ratio={report.excess_stuck_ratio:.3f} "
        f"mean_delta_w_high_imp={report.mean_delta_w_for_high_importance:.6f} "
        f"mean_delta_w_overall={report.mean_delta_w_overall:.6f}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    run_probe(DISLDOLayerDeterministic, "toy, deterministic (baseline)")
    run_probe(DISLDOLayer, "toy, STOCHASTIC")
