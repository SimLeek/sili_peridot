"""Sweeps state_width (via embed_width/column_neurons) and measures the
REAL per-synapse weight-update magnitude at each size, to see whether it
trends toward (or away from) FP4's quantization threshold as width grows
-- directly answering "would deterministic rounding ever start working at
some larger scale" without needing to actually train at that scale.

Two things measured per width, both restricted to ALREADY-LIVE synapses
(weight != 0 at the start, so "dead synapse wakes up" growth noise
doesn't contaminate this -- that's a separate question, see
stochastic_nnz_drift_probe.py):
  - mean |delta_w| after N steps (the quantity check_stuck_weights already
    reports) -- the REALIZED, already-quantized movement.
  - mean |raw_update| -- the CONTINUOUS pre-quantization update magnitude
    computed by hand from get_value_scale/importance, i.e. what the
    update WOULD be before FP4 rounds it back down. If raw_update grows
    relative to FP4's ~0.5-unit code spacing as width increases, that's
    evidence deterministic rounding could start working at some larger
    (if unrealistic-for-this-toy-task) width. If it doesn't, that's
    evidence the problem isn't a rounding-threshold issue and width alone
    won't fix it.

Same controlled setup as the other probes (only state_width differs).
Uses DISLDOLayerDeterministic throughout (isolates the rounding
question from the stochastic-wakeup mechanism characterized separately).

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/weight_update_magnitude_vs_width.py
"""

import functools

import numpy as np
from sili.sparse_rnn import DISLDOLayerDeterministic

from model.eval_stuck_weights import snapshot_multi_digit_state
from model.toy_precision_models import TrueMultiDigitLayer
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from scripts.train_tile_curriculum import _build_tile_window, generate_copy_sequence

VOCAB = 10
NUM_TILES = 4
MAX_WEIGHTS = 128
N_STEPS = 400
LR = 0.01
SEED = 1000

# FP4's smallest nonzero code-to-code gap (0.0 -> 0.5) -- the threshold a
# raw continuous update needs to clear to move a stored code at all under
# deterministic (round-to-nearest) rounding.
FP4_MIN_STEP = 0.5


def run_width(embed_width, column_neurons):
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
    # Only synapses genuinely live (weight != 0) at the START -- isolates
    # "does an already-live synapse move" from "does a dead one wake up".
    live_mask = [np.abs(w) > 1e-12 for w, _ in before]

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

    all_delta = []
    for (w0, _), (w1, _), mask in zip(before, after, live_mask, strict=False):
        if w0.shape != w1.shape:
            # Growth happened (dead synapses waking up shifts array
            # positions) -- restrict to the still-valid prefix isn't
            # safe either, so just skip synapse-count-mismatched digits
            # for this specific measurement and note it.
            continue
        all_delta.append(np.abs(w1[mask] - w0[mask]))
    if not all_delta:
        return state_width, float("nan"), 0

    delta = np.concatenate(all_delta)
    mean_delta_w = float(np.mean(delta))
    n_live = int(delta.size)
    return state_width, mean_delta_w, n_live


if __name__ == "__main__":
    # embed_width, column_neurons pairs -- state_width = product.
    widths = [(8, 4), (16, 4), (16, 8), (32, 8), (32, 16), (64, 16)]
    print(f"{'state_width':>12} {'mean|delta_w|':>15} {'n_live':>8} {'ratio_to_FP4_step':>18}", flush=True)
    for ew, cn in widths:
        sw, mean_delta, n_live = run_width(ew, cn)
        ratio = mean_delta / FP4_MIN_STEP if np.isfinite(mean_delta) else float("nan")
        print(f"{sw:>12} {mean_delta:>15.8f} {n_live:>8} {ratio:>18.8f}", flush=True)
