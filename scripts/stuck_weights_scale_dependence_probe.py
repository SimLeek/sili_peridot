"""Tests the hypothesis from conversation: FP4 per-synapse weights get
stuck at their initial nonzero value because the rank-N value_scale/
output_scale correction is expressive enough to absorb essentially all
the real gradient signal at TOY scale -- and that this may be a
scale-dependent artifact that eases (or doesn't) as state_width grows
toward realistic model sizes. Directly relevant to the project's actual
goal of converting billion-parameter models to FP4/2xFP4: if this is a
toy-scale-only artifact, it may not need an architectural fix; if it
persists at larger width, it does.

Controlled comparison (only state_width differs -- same lr, same
digit_cls/scale_rank, same steps, same seed) between the CURRENT toy
scale (embed_width=8, column_neurons=4 -> state_width=32, matching
l1_sparsity_probe.py's OriginalArchModel) and an 8x larger one
(embed_width=32, column_neurons=8 -> state_width=256). Uses
ToyTileRecurrenceRealFP4 directly (not OriginalArchModel) since it
already accepts embed_width/column_neurons as real constructor
params -- no L1-sparsity mechanism here, deliberately, to isolate the
core value_scale-vs-FP4-weight mechanism from that confound.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/stuck_weights_scale_dependence_probe.py
"""

import functools

import numpy as np
from sili.sparse_rnn import DISLDOLayerDeterministic

from model.eval_stuck_weights import check_stuck_weights, snapshot_multi_digit_state
from model.toy_precision_models import TrueMultiDigitLayer
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from scripts.train_tile_curriculum import _build_tile_window, generate_copy_sequence

VOCAB = 10
NUM_TILES = 4
MAX_WEIGHTS = 128
N_STEPS = 800
LR = 0.01  # same for both arms deliberately -- only state_width differs
SEED = 1000


def run_probe(embed_width, column_neurons, label):
    digit_cls = functools.partial(
        TrueMultiDigitLayer,
        digit_cls=DISLDOLayerDeterministic,
        n_stages=3,
        base=12.0,
        lr_power=0.0,
        dense=True,
        scale_rank=1,
    )
    rng = np.random.default_rng(SEED)
    model = ToyTileRecurrenceRealFP4(
        VOCAB,
        embed_width,
        column_neurons,
        mlp_hidden=0,
        num_tiles=NUM_TILES,
        max_weights=MAX_WEIGHTS,
        num_cpus=1,
        disldo_cls=digit_cls,
        rng=rng,
    )
    state_width = embed_width * column_neurons
    task_rng = np.random.RandomState(SEED)
    embed_table = task_rng.randn(VOCAB, embed_width).astype(np.float32) * 0.3
    opt = AdamOptimizer()

    before = snapshot_multi_digit_state(model.o_proj)

    for _step in range(N_STEPS):
        tokens, pairs = generate_copy_sequence(task_rng, VOCAB, NUM_TILES)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        total_loss = None
        for i in range(NUM_TILES):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, column_neurons)
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
        f"[{label}] state_width={state_width} n_synapses={report.n_synapses} "
        f"n_high_importance={report.n_high_importance} "
        f"stuck_fraction={report.stuck_fraction:.4f} "
        f"excess_stuck_ratio={report.excess_stuck_ratio:.3f} "
        f"mean_delta_w_high_imp={report.mean_delta_w_for_high_importance:.6f} "
        f"mean_delta_w_overall={report.mean_delta_w_overall:.6f}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    run_probe(8, 4, "toy (current)")
    run_probe(32, 8, "8x wider")
