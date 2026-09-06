"""Sweeps whether stochastic-vs-deterministic FP4 rounding's accuracy gap on
an out-of-context copy task shrinks as state_width or input sparsity grows.
See docs/research/stochastic_stability_vs_scale_sparsity.rst:
research_question_and_ooc_task_design, :stuck_weights_followup_motivation,
:input_density_not_connectivity_density, :two_axis_grid_and_gap_metric.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/stochastic_stability_vs_scale_sparsity.py
"""

import functools

import numpy as np
from sili.sparse_rnn import DISLDOLayer, DISLDOLayerDeterministic

from model.toy_precision_models import TrueMultiDigitLayer
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum, predicted_token
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from scripts.train_tile_curriculum import _build_tile_window, generate_copy_sequence

VOCAB = 10
NUM_TILES = 4
OOC_MARGIN = 2  # seq_len = NUM_TILES + OOC_MARGIN -- see docs/research/
# stochastic_stability_vs_scale_sparsity.rst:research_question_and_ooc_task_design.
SEQ_LEN = NUM_TILES + OOC_MARGIN
N_STEPS = 200
EVAL_SEQUENCES = 30
LR = 0.01
SEEDS = [1000, 1001, 1002]

# (embed_width, column_neurons) -> state_width = product.
WIDTHS = [(8, 4), (16, 8), (32, 16)]  # state_width 32, 128, 512
INPUT_DENSITIES = [1.0, 0.25, 0.0625]  # fraction of embedding dims left nonzero per token


def _sparse_embed_table(task_rng, seed, vocab, embed_width, input_density):
    """See docs/research/stochastic_stability_vs_scale_sparsity.rst:
    sparse_embed_table_fixed_mask_design."""
    embed_table = task_rng.randn(vocab, embed_width).astype(np.float32) * 0.3
    if input_density >= 1.0:
        return embed_table
    mask_rng = np.random.RandomState(seed + 5000)
    n_active = max(1, round(input_density * embed_width))
    mask = np.zeros((vocab, embed_width), dtype=np.float32)
    for v in range(vocab):
        active_dims = mask_rng.choice(embed_width, size=n_active, replace=False)
        mask[v, active_dims] = 1.0
    return embed_table * mask


def _train_and_eval(embed_width, column_neurons, input_density, digit_backend, seed):
    state_width = embed_width * column_neurons
    # Connectivity always fully dense.
    # See docs/research/stochastic_stability_vs_scale_sparsity.rst:input_density_not_connectivity_density.
    max_weights = state_width * state_width

    digit_cls = functools.partial(
        TrueMultiDigitLayer, digit_cls=digit_backend, n_stages=3, base=12.0, lr_power=0.0, dense=True, scale_rank=1
    )
    rng = np.random.default_rng(seed)
    model = ToyTileRecurrenceRealFP4(
        VOCAB,
        embed_width,
        column_neurons,
        mlp_hidden=0,
        num_tiles=NUM_TILES,
        max_weights=max_weights,
        num_cpus=1,
        disldo_cls=digit_cls,
        rng=rng,
    )
    task_rng = np.random.RandomState(seed)
    embed_table = _sparse_embed_table(task_rng, seed, VOCAB, embed_width, input_density)
    opt = AdamOptimizer()

    for _step in range(N_STEPS):
        tokens, pairs = generate_copy_sequence(task_rng, VOCAB, SEQ_LEN)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        total_loss = None
        for i in range(SEQ_LEN):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, column_neurons)
            M, logits, _aux = model.step(window, M, LR)
            if i in targets:
                tgt_loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                total_loss = tgt_loss if total_loss is None else total_loss + tgt_loss
        if total_loss is not None:
            total_loss.backward()
            clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
            opt.step(model.parameters_for_optimizer(), lr=LR)

    eval_rng = np.random.RandomState(seed + 777)
    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, pairs = generate_copy_sequence(eval_rng, VOCAB, SEQ_LEN)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        for i in range(SEQ_LEN):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, column_neurons)
            M, logits, _aux = model.step(window, M, 0.0)
            if i in targets:
                pred = predicted_token(logits, NUM_TILES - 1)
                correct += int(pred == targets[i])
                total += 1
    return correct / total if total else 0.0


def run_grid_point(embed_width, column_neurons, input_density):
    det_accs = [_train_and_eval(embed_width, column_neurons, input_density, DISLDOLayerDeterministic, s) for s in SEEDS]
    stoch_accs = [_train_and_eval(embed_width, column_neurons, input_density, DISLDOLayer, s) for s in SEEDS]
    det_mean = float(np.mean(det_accs))
    stoch_mean = float(np.mean(stoch_accs))
    stoch_std = float(np.std(stoch_accs))
    gap = det_mean - stoch_mean
    return det_mean, stoch_mean, stoch_std, gap, det_accs, stoch_accs


if __name__ == "__main__":
    print(
        f"{'state_width':>12} {'input_density':>13} {'det_mean':>9} {'stoch_mean':>10} {'stoch_std':>9} {'gap':>8}",
        flush=True,
    )
    for ew, cn in WIDTHS:
        state_width = ew * cn
        for input_density in INPUT_DENSITIES:
            det_mean, stoch_mean, stoch_std, gap, det_accs, stoch_accs = run_grid_point(ew, cn, input_density)
            print(
                f"{state_width:>12} {input_density:>13.4f} {det_mean:>9.3f} {stoch_mean:>10.3f} "
                f"{stoch_std:>9.3f} {gap:>8.3f}   det={det_accs} stoch={stoch_accs}",
                flush=True,
            )
