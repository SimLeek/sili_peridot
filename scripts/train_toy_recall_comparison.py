"""
scripts/train_toy_recall_comparison.py
────────────────────────────────────────
Real training experiment (not a pytest sanity check): does tile-
recurrence actually learn to retrieve information from BEYOND its own
direct attention window (num_tiles), by carrying it through the
recurrent additive state -- the core property this whole architecture
exists to have more of than the old single-carried-vector design?

Per direct decision, uses the STANDARD Multi-Query Associative Recall
(MQAR) benchmark (Arora, Eyuboglu et al., "Zoology", 2023 --
model/toy_recall_task.py's own generate_mqar_sequence, ported directly
from HazyResearch/zoology's reference implementation) instead of a
hand-rolled task -- this is the established benchmark for exactly the
property under test here, used throughout the linear-attention/SSM/
efficient-architecture literature specifically to compare recurrent/
efficient models against full attention. Also uses gradient clipping
(backward_with_grad_clip) and a warmup+cosine LR schedule (lr_schedule,
nanoGPT's own convention, looked up rather than guessed) -- both per
direct decision. NOT batched -- single CPU, batching doesn't buy
parallelism here (also per direct decision).

ToySmallTransformer sees the WHOLE sequence every step (its own causal
attention can trivially reach any position directly -- a ceiling/
sanity reference, not really the object of study). ToyTileRecurrence's
num_tiles is kept small and FIXED, so recall gaps beyond it can only
be solved via the recurrent M state carrying information forward
across ticks, not direct windowed attention -- the real test.

Run: python -m scripts.train_toy_recall_comparison
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from model.toy_recall_task import generate_mqar_sequence
from model.toy_recall_models import (
    ToySmallTransformer, ToyTileRecurrence,
    cross_entropy_sum, predicted_token, apply_gradient_step,
    backward_with_grad_clip, lr_schedule,
)

NUM_TILES = 4              # deliberately small and FIXED across the whole sweep
TRAIN_STEPS = 1500
WARMUP_STEPS = 100
PEAK_LR = 0.02
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 40

# (seq_len, num_kv_pairs, vocab_size) -- MQAR requires vocab_size > seq_len
# and seq_len >= 4*num_kv_pairs. Two difficulty levels: config 1 mostly
# fits inside NUM_TILES's own direct window (power-law gaps bias small);
# config 2 is deliberately larger on both axes.
CONFIGS = [
    (16, 2, 20),
    (32, 4, 40),
]


def train_and_eval_dense(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, seed):
    rng = np.random.RandomState(seed)
    tf = ToySmallTransformer(vocab, hidden, mlp_hidden, n_layers=2,
                             max_weights=hidden * mlp_hidden, num_cpus=2)
    embed_table = rng.randn(vocab, hidden).astype(np.float32) * 0.3

    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        tokens, pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        embedded = embed_table[tokens]
        logits = tf.forward(embedded, learning_rate=lr)
        loss = cross_entropy_sum(logits, pairs)
        backward_with_grad_clip(loss, MAX_GRAD_NORM)
        apply_gradient_step(tf.parameters(), lr=lr)

    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        embedded = embed_table[tokens]
        logits = tf.forward(embedded, learning_rate=0.0)
        for pos, target in pairs:
            correct += int(predicted_token(logits, pos) == target)
            total += 1
    return correct / total


def _build_tile_window(embed_table, tokens, i, num_tiles, M_prev):
    hidden = embed_table.shape[1]
    window = np.empty((num_tiles, hidden), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        window[j] = embed_table[tokens[src]] if src >= 0 else M_prev[j]
    return window


def train_and_eval_tile(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, seed):
    rng = np.random.RandomState(seed)
    model = ToyTileRecurrence(vocab, hidden, mlp_hidden, NUM_TILES,
                              max_weights=hidden * mlp_hidden, num_cpus=2)
    embed_table = rng.randn(vocab, hidden).astype(np.float32) * 0.3

    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        mqar_by_pos = dict(mqar_pairs)
        M = np.zeros((NUM_TILES, hidden), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, M)
            M, logits = model.step(window, M, learning_rate=lr)
            # tiles 0..NUM_TILES-2: always-defined "column" target (what
            # the next tile currently holds) -- auxiliary local signal,
            # unrelated to whether i is an MQAR query position.
            pairs = []
            for j in range(NUM_TILES - 1):
                tgt_pos = i - (NUM_TILES - 1) + j + 1
                if 0 <= tgt_pos < seq_len:
                    pairs.append((j, int(tokens[tgt_pos])))
            # last tile: the TRUE MQAR recall target if i is a query
            # position, else skip entirely (matching MQAR's own -100
            # "ignore" convention for non-query positions -- tokens[i+1]
            # there is just random filler, not a meaningful target).
            if i in mqar_by_pos:
                pairs.append((NUM_TILES - 1, mqar_by_pos[i]))
            if pairs:
                loss = cross_entropy_sum(logits, pairs)
                backward_with_grad_clip(loss, MAX_GRAD_NORM)
                apply_gradient_step(model.parameters(), lr=lr)

    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        mqar_by_pos = dict(mqar_pairs)
        M = np.zeros((NUM_TILES, hidden), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, M)
            M, logits = model.step(window, M, learning_rate=0.0)
            if i in mqar_by_pos:
                pred = predicted_token(logits, NUM_TILES - 1)
                correct += int(pred == mqar_by_pos[i])
                total += 1
    return correct / total


def main():
    hidden, mlp_hidden = 16, 24
    print(f"num_tiles={NUM_TILES} hidden={hidden} mlp_hidden={mlp_hidden} "
          f"train_steps={TRAIN_STEPS} warmup={WARMUP_STEPS} peak_lr={PEAK_LR} "
          f"max_grad_norm={MAX_GRAD_NORM} eval_sequences={EVAL_SEQUENCES}\n")
    print(f"{'seq_len':>8}  {'kv_pairs':>9}  {'vocab':>6}  {'dense_acc':>10}  {'tile_acc':>10}")
    results = []
    for seq_len, num_kv_pairs, vocab in CONFIGS:
        t0 = time.time()
        dense_acc = train_and_eval_dense(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden,
                                         seed=1000 + seq_len)
        tile_acc = train_and_eval_tile(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden,
                                       seed=2000 + seq_len)
        elapsed = time.time() - t0
        print(f"{seq_len:>8}  {num_kv_pairs:>9}  {vocab:>6}  {dense_acc:>10.2f}  "
              f"{tile_acc:>10.2f}   ({elapsed:.1f}s)")
        results.append((seq_len, num_kv_pairs, vocab, dense_acc, tile_acc))

    print(f"\nDense baseline sees the WHOLE sequence every step (a ceiling/sanity "
          f"reference). Tile-recurrence's own direct attention window is fixed at "
          f"num_tiles={NUM_TILES} -- MQAR's power-law gap distribution means most "
          f"queries fall within that window, but some don't; only those beyond it "
          f"genuinely test the recurrent-memory property this architecture exists for.")
    return results


if __name__ == "__main__":
    main()
