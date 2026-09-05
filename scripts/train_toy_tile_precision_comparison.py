"""
scripts/train_toy_tile_precision_comparison.py
──────────────────────────────────────────────────
Real MQAR run for `ToyTileRecurrenceRealFP4`
(model/toy_tile_precision_models.py) -- tile-recurrence ported onto
real sili (DISLDOLayer) per direct correction ("The tile recurrence is
meant to be set up on sili"). See the approved plan
(fuzzy-plotting-starlight.md) for the full design rationale.

A bounded (not full combinatorial) three-stage sweep:

1. **Optimizer variant** at the control scale (column_neurons=8,
   seq_len=16/kv=2): plain DISLDOLayer (importance only) vs
   AdamRowScaleDISLDOLayer vs AdamRank1DISLDOLayer
   (model/toy_precision_models.py). Picks a winner.
2. **Capacity scaling**, using the Stage-1 winner: column_neurons=16
   vs the column_neurons=8 control -- direct request ("recurrence 16
   instead of 8 as a test vs control").
3. **Tile-count/context scaling**, using the Stage-1 winner: seq_len=32
   /kv=4 (num_tiles=32, since num_tiles=seq_len) at both
   column_neurons values -- direct request ("more tiles").

Same MQAR task, same unified per-position target convention already
validated for ToyTileRecurrence (query positions -> recalled value,
key/value context region -> real next token, elsewhere skipped -- see
scripts/train_toy_recall_comparison.py's own docstring for the full
rationale). Reported against the fp32 ToyTileRecurrence baseline
(0.10/0.05, DenseTensorLinear+Adam) and the plain dense fp32
-transformer ceiling (0.57/0.26) already on record.

Run: python -m scripts.train_toy_tile_precision_comparison
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from sili.sparse_rnn import DISLDOLayer

from model.toy_precision_models import AdamRank1DISLDOLayer, AdamRowScaleDISLDOLayer
from model.toy_recall_models import AdamOptimizer, cross_entropy_sum, lr_schedule, predicted_token
from model.toy_recall_task import generate_mqar_sequence
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4

TRAIN_STEPS = 3000
WARMUP_STEPS = 100
PEAK_LR = 0.02
EVAL_SEQUENCES = 60
EMBED_WIDTH = 32
MAX_WEIGHTS_PER_LAYER = 4096  # a FIXED sparse connection budget, independent of
# column_neurons/state_width -- checked directly:
# the original max_weights=state_width*mlp_hidden
# gave per_row=512 at n_outputs=256 (state_width=8),
# i.e. 100% dense -- every q/k/v/o layer was fully
# dense while still paying DISLDOLayer's sparse
# bookkeeping overhead on top, explaining timings far
# worse than the ~3x-vs-dense DISLDOLayer overhead
# normally seen. This value keeps real sparsity
# (per_row well under n_outputs at every tested
# column_neurons) -- the entire reason to route
# tile-recurrence's wide state through sili at all.

VARIANTS = {
    "plain-DISLDOLayer": DISLDOLayer,
    "row-scale-Adam": AdamRowScaleDISLDOLayer,
    "rank1-Adam": AdamRank1DISLDOLayer,
}


def _build_targets(tokens: np.ndarray, mqar_pairs: list, num_kv_pairs: int) -> dict:
    context_size = num_kv_pairs * 2
    targets = dict(mqar_pairs)
    for i in range(context_size - 1):
        targets.setdefault(i, int(tokens[i + 1]))
    return targets


def _build_tile_window(
    embed_table: np.ndarray, tokens: np.ndarray, i: int, num_tiles: int, M_prev: np.ndarray, column_neurons: int
) -> np.ndarray:
    state_width = embed_table.shape[1] * column_neurons
    window = np.empty((num_tiles, state_width), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        window[j] = np.repeat(embed_table[tokens[src]], column_neurons) if src >= 0 else M_prev[j]
    return window


def train_and_eval_tile_real_fp4(
    seq_len, num_kv_pairs, vocab, column_neurons, disldo_cls, seed, train_steps=TRAIN_STEPS
):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    num_tiles = seq_len
    state_width = EMBED_WIDTH * column_neurons
    mlp_hidden = state_width * 2
    model = ToyTileRecurrenceRealFP4(
        vocab,
        EMBED_WIDTH,
        column_neurons,
        mlp_hidden,
        num_tiles,
        MAX_WEIGHTS_PER_LAYER,
        num_cpus=4,
        disldo_cls=disldo_cls,
    )
    opt = AdamOptimizer()
    embed_table = rng.randn(vocab, EMBED_WIDTH).astype(np.float32) * 0.3

    for step in range(train_steps):
        lr = lr_schedule(step, train_steps, PEAK_LR, WARMUP_STEPS)
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        targets = _build_targets(tokens, mqar_pairs, num_kv_pairs)
        M = np.zeros((num_tiles, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles, M, column_neurons)
            M, logits, _aux = model.step(window, M, lr)
            if i in targets:
                loss = cross_entropy_sum(logits, [(num_tiles - 1, targets[i])])
                loss.grad = np.array(1.0, dtype=np.float32)
                loss.backward()
                opt.step(model.parameters_for_optimizer(), lr=lr)

    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        mqar_by_pos = dict(mqar_pairs)
        M = np.zeros((num_tiles, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles, M, column_neurons)
            M, logits, _aux = model.step(window, M, 0.0)
            if i in mqar_by_pos:
                pred = predicted_token(logits, num_tiles - 1)
                correct += int(pred == mqar_by_pos[i])
                total += 1
    return correct / total


def main():
    print(
        f"train_steps={TRAIN_STEPS} warmup={WARMUP_STEPS} peak_lr={PEAK_LR} "
        f"eval_sequences={EVAL_SEQUENCES} embed_width={EMBED_WIDTH}\n"
    )

    print("Stage 1: optimizer variant sweep (column_neurons=8, seq_len=16/kv=2)")
    stage1 = {}
    for name, cls in VARIANTS.items():
        t0 = time.time()
        acc = train_and_eval_tile_real_fp4(16, 2, 20, column_neurons=8, disldo_cls=cls, seed=6000)
        print(f"  {name}: acc={acc:.2f}  ({time.time() - t0:.1f}s)")
        stage1[name] = acc
    winner_name = max(stage1, key=stage1.get)
    winner_cls = VARIANTS[winner_name]
    print(f"  winner: {winner_name} (acc={stage1[winner_name]:.2f})\n")

    print(f"Stage 2: capacity scaling using {winner_name} (column_neurons=16 vs 8 control)")
    stage2 = {}
    for cn in (8, 16):
        t0 = time.time()
        acc = train_and_eval_tile_real_fp4(16, 2, 20, column_neurons=cn, disldo_cls=winner_cls, seed=7000 + cn)
        print(f"  column_neurons={cn}: acc={acc:.2f}  ({time.time() - t0:.1f}s)")
        stage2[cn] = acc
    print()

    print(f"Stage 3: tile-count/context scaling using {winner_name} (seq_len=32/kv=4)")
    stage3 = {}
    for cn in (8, 16):
        t0 = time.time()
        acc = train_and_eval_tile_real_fp4(32, 4, 40, column_neurons=cn, disldo_cls=winner_cls, seed=8000 + cn)
        print(f"  column_neurons={cn}: acc={acc:.2f}  ({time.time() - t0:.1f}s)")
        stage3[cn] = acc

    print(
        "\nReference (already on record): fp32 ToyTileRecurrence "
        "(DenseTensorLinear+Adam) = 0.10 (seq16) / 0.05 (seq32); "
        "dense fp32 transformer ceiling = 0.57 (seq16) / 0.26 (seq32)."
    )
    return stage1, stage2, stage3


if __name__ == "__main__":
    main()
