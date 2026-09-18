"""Characterizes the nnz drift under stochastic DISLDOLayer with
dense=True (found in conversation: 2660 -> 2686 over 800 steps, when
dense init should mean stable 100% density with no ongoing
synaptogenesis). Prints nnz at regular intervals to see whether it's
monotonic pruning, monotonic growth, oscillation, or something else --
and separately reports how many times weight+importance both round to
exactly (0,0) for some synapse in a single call (the specific mechanism
suspected: block4's maybe_compress prunes on (weight=0 AND importance=0),
and stochastic rounding gives every near-zero value a real per-step
chance of landing exactly on 0, unlike deterministic rounding which only
rounds to 0 when truly closer to 0 than to the next FP4 code).

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/stochastic_nnz_drift_probe.py
"""

import functools

import numpy as np
from sili.sparse_rnn import DISLDOLayer

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
PRINT_EVERY = 20


def nnz_of(layer):
    return [d._c.nnz for d in layer.digits]


if __name__ == "__main__":
    digit_cls = functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayer, n_stages=3, base=12.0, lr_power=0.0, dense=True, scale_rank=1
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

    print(f"initial nnz (o_proj digits): {nnz_of(model.o_proj)}", flush=True)
    zero_weight_zero_imp_seen = 0

    for step in range(1, N_STEPS + 1):
        tokens, pairs = generate_copy_sequence(task_rng, VOCAB, NUM_TILES)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        total_loss = None
        for i in range(NUM_TILES):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, LR)
            if i in targets:
                tgt_loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                total_loss = tgt_loss if total_loss is None else total_loss + tgt_loss
        if total_loss is not None:
            total_loss.backward()
            clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
            opt.step(model.parameters_for_optimizer(), lr=LR)

        # Count exact (weight==0 AND importance==0) synapses across
        # every digit -- the specific pattern maybe_compress prunes on.
        both_zero = 0
        for d in model.o_proj.digits:
            w = np.asarray(d._c.weights_vals)
            imp = np.asarray(d._c.importance)
            both_zero += int(np.sum((w == 0.0) & (imp == 0.0)))
        zero_weight_zero_imp_seen = max(zero_weight_zero_imp_seen, both_zero)

        if step % PRINT_EVERY == 0:
            print(f"step={step} nnz={nnz_of(model.o_proj)} synapses_at_(0,0)_now={both_zero}", flush=True)

    print(f"final nnz: {nnz_of(model.o_proj)}", flush=True)
    print(f"max simultaneous (weight=0,importance=0) synapses seen: {zero_weight_zero_imp_seen}", flush=True)
