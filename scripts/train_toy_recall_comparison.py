"""
scripts/train_toy_recall_comparison.py
────────────────────────────────────────
Real training experiment (not a pytest sanity check): does tile-
recurrence actually learn to retrieve information from BEYOND its own
direct attention window (num_tiles), by carrying it through the
recurrent additive state -- the core property this whole architecture
exists to have more of than the old single-carried-vector design?

For each LAG in a sweep, trains both ToySmallTransformer (given the
WHOLE sequence as visible context every step -- its own causal
attention can trivially reach any lag directly, a ceiling/sanity
reference, not really the object of study) and ToyTileRecurrence
(num_tiles deliberately kept SMALL and FIXED across the whole sweep,
so lags beyond num_tiles can only be solved via the recurrent M state
carrying information forward across ticks, not direct windowed
attention) on FRESH random sequences each step (tests generalization,
not memorization of one fixed example), then evaluates recall accuracy
on held-out fresh sequences at that same lag.

Run: python -m scripts.train_toy_recall_comparison
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from model.config import MiniCPM5Config  # noqa: F401  (unused, keeps import style consistent)
from model.toy_recall_task import generate_sequence, induction_correct
from model.toy_recall_models import (
    ToySmallTransformer, ToyTileRecurrence,
    cross_entropy_sum, predicted_token, apply_gradient_step,
)

VOCAB = 8               # smaller vocab (chance=12.5%) -- calibrated directly:
HIDDEN = 16             # vocab=16/lr=0.005/steps=400 left even the dense
MLP_HIDDEN = 24         # baseline stuck at chance (0.05) -- this is a genuinely
NUM_TILES = 4           # hard task from FRESH sequences every step (no
LAGS = [2, 4, 8, 16]    # memorization shortcut), matching real induction-head
TRAIN_STEPS = 2000      # literature (these take real training to emerge, not
EVAL_SEQUENCES = 40     # instant convergence) -- not a tile-recurrence-specific
LR = 0.02               # problem, confirmed by the dense baseline failing too.
SEQ_LEN_MARGIN = 6      # sequence length = lag + this margin


def train_and_eval_dense(lag: int, seed: int) -> float:
    rng = np.random.RandomState(seed)
    tf = ToySmallTransformer(VOCAB, HIDDEN, MLP_HIDDEN, n_layers=2,
                             max_weights=HIDDEN * MLP_HIDDEN, num_cpus=2)
    seq_len = lag + SEQ_LEN_MARGIN
    embed_table = rng.randn(VOCAB, HIDDEN).astype(np.float32) * 0.3

    for _step in range(TRAIN_STEPS):
        tokens, pos = generate_sequence(rng, VOCAB, seq_len, lag)
        embedded = embed_table[tokens]
        logits = tf.forward(embedded, learning_rate=LR)
        loss = cross_entropy_sum(logits, [(pos, int(tokens[pos + 1]))])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        apply_gradient_step(tf.parameters(), lr=LR)

    correct = 0
    for _i in range(EVAL_SEQUENCES):
        tokens, pos = generate_sequence(rng, VOCAB, seq_len, lag)
        embedded = embed_table[tokens]
        logits = tf.forward(embedded, learning_rate=0.0)
        pred = predicted_token(logits, pos)
        correct += int(induction_correct(pred, tokens, pos))
    return correct / EVAL_SEQUENCES


def _build_tile_window(embed_table, tokens, i, num_tiles, M_prev):
    hidden = embed_table.shape[1]
    window = np.empty((num_tiles, hidden), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        window[j] = embed_table[tokens[src]] if src >= 0 else M_prev[j]
    return window


def train_and_eval_tile(lag: int, seed: int) -> float:
    rng = np.random.RandomState(seed)
    model = ToyTileRecurrence(VOCAB, HIDDEN, MLP_HIDDEN, NUM_TILES,
                              max_weights=HIDDEN * MLP_HIDDEN, num_cpus=2)
    seq_len = lag + SEQ_LEN_MARGIN
    embed_table = rng.randn(VOCAB, HIDDEN).astype(np.float32) * 0.3

    for _step in range(TRAIN_STEPS):
        tokens, pos = generate_sequence(rng, VOCAB, seq_len, lag)
        M = np.zeros((NUM_TILES, HIDDEN), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, M)
            M, logits = model.step(window, M, learning_rate=LR)
            pairs = []
            for j in range(NUM_TILES):
                tgt_pos = i - (NUM_TILES - 1) + j + 1
                if 0 <= tgt_pos < seq_len:
                    pairs.append((j, int(tokens[tgt_pos])))
            if pairs:
                loss = cross_entropy_sum(logits, pairs)
                loss.grad = np.array(1.0, dtype=np.float32)
                loss.backward()
                apply_gradient_step(model.parameters(), lr=LR)

    correct = 0
    for _i in range(EVAL_SEQUENCES):
        tokens, pos = generate_sequence(rng, VOCAB, seq_len, lag)
        M = np.zeros((NUM_TILES, HIDDEN), dtype=np.float32)
        logits = None
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, M)
            M, logits = model.step(window, M, learning_rate=0.0)
            if i == pos:
                pred = predicted_token(logits, NUM_TILES - 1)
                correct += int(induction_correct(pred, tokens, pos))
    return correct / EVAL_SEQUENCES


def main():
    print(f"vocab={VOCAB} hidden={HIDDEN} mlp_hidden={MLP_HIDDEN} num_tiles={NUM_TILES} "
          f"train_steps={TRAIN_STEPS} eval_sequences={EVAL_SEQUENCES} lr={LR}\n")
    print(f"{'lag':>5}  {'dense_acc':>10}  {'tile_acc':>10}  {'within_window?':>15}")
    results = []
    for lag in LAGS:
        t0 = time.time()
        dense_acc = train_and_eval_dense(lag, seed=1000 + lag)
        tile_acc = train_and_eval_tile(lag, seed=2000 + lag)
        within_window = lag < NUM_TILES
        elapsed = time.time() - t0
        print(f"{lag:>5}  {dense_acc:>10.2f}  {tile_acc:>10.2f}  {str(within_window):>15}   ({elapsed:.1f}s)")
        results.append((lag, dense_acc, tile_acc, within_window))

    print("\nDense baseline sees the WHOLE sequence every step (its own causal "
          "attention can trivially reach any lag directly) -- expected near-1.0 "
          "throughout, a ceiling/sanity reference, not the object of study.")
    print(f"Tile-recurrence's own direct attention window is fixed at "
          f"num_tiles={NUM_TILES} -- lags >= {NUM_TILES} can ONLY be solved via "
          f"the recurrent M state carrying information forward across ticks, "
          f"not direct windowed attention. This is the real test.")
    return results


if __name__ == "__main__":
    main()
