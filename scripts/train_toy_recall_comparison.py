"""
scripts/train_toy_recall_comparison.py
────────────────────────────────────────
Real training experiment (not a pytest sanity check): does
`ToyTileRecurrence`'s column-averaged wide recurrent state actually
learn, at rough parity of context visibility with a dense causal
baseline? Second real run, after the first found two design mistakes
(fixed here) -- see
docs/research/train_toy_recall_comparison.rst:train_toy_recall_comparison.module_overview,
:train_toy_recall_comparison.second_run_fixes,
:train_toy_recall_comparison.unified_target_design.

Run: python -m scripts.train_toy_recall_comparison
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from model.toy_recall_models import (
    AdamOptimizer,
    ToySmallTransformer,
    ToyTileRecurrence,
    clip_grad_norm_,
    cross_entropy_sum,
    lr_schedule,
    predicted_token,
)
from model.toy_recall_task import generate_mqar_sequence

COLUMN_NEURONS = 8  # state_width = embed_width * COLUMN_NEURONS
TRAIN_STEPS = 3000  # matches scripts/torch_mqar_control.py's own step count
WARMUP_STEPS = 100
PEAK_LR = 0.02
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 60

# Each entry below is a seq_len/num_kv_pairs/vocab_size triple -- see
# docs/research/train_toy_recall_comparison.rst:train_toy_recall_comparison.config_and_dims_provenance.
CONFIGS = [
    (16, 2, 20),
    (32, 4, 40),
]


def _build_targets(tokens: np.ndarray, mqar_pairs: list, num_kv_pairs: int) -> dict:
    """Unified per-position target dict: query positions -> recalled
    value; context-laydown region -> real next token; everywhere else
    -> no entry (pure random filler).

    See docs/research/train_toy_recall_comparison.rst:train_toy_recall_comparison.build_targets_semantics.
    """
    context_size = num_kv_pairs * 2
    targets = dict(mqar_pairs)
    for i in range(context_size - 1):
        targets.setdefault(i, int(tokens[i + 1]))
    return targets


def train_and_eval_dense(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, seed):
    """The CONTROL -- unmodified from the first real comparison run,
    trains on the true MQAR pairs only, nothing else.

    See docs/research/train_toy_recall_comparison.rst:train_toy_recall_comparison.unified_target_design.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)  # DenseTensorLinear's own init uses the global RNG
    tf = ToySmallTransformer(vocab, hidden, mlp_hidden, n_layers=2, num_cpus=2)
    opt = AdamOptimizer()
    embed_table = rng.randn(vocab, hidden).astype(np.float32) * 0.3

    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        embedded = embed_table[tokens]
        logits = tf.forward(embedded)
        loss = cross_entropy_sum(logits, mqar_pairs)
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        clip_grad_norm_(tf.parameters(), MAX_GRAD_NORM)
        opt.step(tf.parameters(), lr=lr)

    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        embedded = embed_table[tokens]
        logits = tf.forward(embedded)
        for pos, target in mqar_pairs:
            correct += int(predicted_token(logits, pos) == target)
            total += 1
    return correct / total


def _build_tile_window(
    embed_table: np.ndarray, tokens: np.ndarray, i: int, num_tiles: int, M_prev: np.ndarray, column_neurons: int
) -> np.ndarray:
    """[num_tiles, state_width]. See
    docs/research/train_toy_recall_comparison.rst:train_toy_recall_comparison.tile_window_construction.
    """
    state_width = embed_table.shape[1] * column_neurons
    window = np.empty((num_tiles, state_width), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        window[j] = np.repeat(embed_table[tokens[src]], column_neurons) if src >= 0 else M_prev[j]
    return window


def train_and_eval_tile(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, seed):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    num_tiles = seq_len
    state_width = hidden * COLUMN_NEURONS
    model = ToyTileRecurrence(vocab, hidden, COLUMN_NEURONS, mlp_hidden, num_tiles, num_cpus=2)
    opt = AdamOptimizer()
    embed_table = rng.randn(vocab, hidden).astype(np.float32) * 0.3

    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        targets = _build_targets(tokens, mqar_pairs, num_kv_pairs)
        M = np.zeros((num_tiles, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles, M, COLUMN_NEURONS)
            M, logits = model.step(window, M)
            if i in targets:
                loss = cross_entropy_sum(logits, [(num_tiles - 1, targets[i])])
                loss.grad = np.array(1.0, dtype=np.float32)
                loss.backward()
                clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step(model.parameters(), lr=lr)

    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        mqar_by_pos = dict(mqar_pairs)
        M = np.zeros((num_tiles, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles, M, COLUMN_NEURONS)
            M, logits = model.step(window, M)
            if i in mqar_by_pos:
                pred = predicted_token(logits, num_tiles - 1)
                correct += int(pred == mqar_by_pos[i])
                total += 1
    return correct / total


def main():
    # hidden/mlp_hidden/tile_mlp_hidden provenance: see
    # docs/research/train_toy_recall_comparison.rst:train_toy_recall_comparison.config_and_dims_provenance.
    hidden, mlp_hidden = 32, 48
    tile_mlp_hidden = hidden * COLUMN_NEURONS * 2
    print(
        f"column_neurons={COLUMN_NEURONS} hidden={hidden} mlp_hidden={mlp_hidden} "
        f"tile_mlp_hidden={tile_mlp_hidden} train_steps={TRAIN_STEPS} warmup={WARMUP_STEPS} "
        f"peak_lr={PEAK_LR} max_grad_norm={MAX_GRAD_NORM} eval_sequences={EVAL_SEQUENCES} "
        f"optimizer=Adam+global-norm-clip\n"
    )
    print(f"{'seq_len':>8}  {'kv_pairs':>9}  {'vocab':>6}  {'dense_acc':>10}  {'tile_acc':>10}")
    results = []
    for seq_len, num_kv_pairs, vocab in CONFIGS:
        t0 = time.time()
        dense_acc = train_and_eval_dense(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, seed=1000 + seq_len)
        tile_acc = train_and_eval_tile(seq_len, num_kv_pairs, vocab, hidden, tile_mlp_hidden, seed=2000 + seq_len)
        elapsed = time.time() - t0
        print(f"{seq_len:>8}  {num_kv_pairs:>9}  {vocab:>6}  {dense_acc:>10.2f}  {tile_acc:>10.2f}   ({elapsed:.1f}s)")
        results.append((seq_len, num_kv_pairs, vocab, dense_acc, tile_acc))

    print(
        "\nDense is the unmodified control (true MQAR pairs only, unchanged from "
        "the first comparison run). ToyTileRecurrence gets the unified per-position "
        "target (query positions -> recalled value, context-laydown region -> real "
        "next token, elsewhere skipped) -- a fix specific to its own column-mean "
        "readout, not applied to dense. num_tiles=seq_len per config -- "
        "tile-recurrence can see the whole sequence directly this run "
        "(parity-of-visibility test), not yet testing recall genuinely beyond a "
        "narrow window."
    )
    return results


if __name__ == "__main__":
    main()
