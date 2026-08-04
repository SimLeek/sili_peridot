"""
scripts/train_toy_recall_comparison.py
────────────────────────────────────────
Real training experiment (not a pytest sanity check): does
`ToyTileRecurrence`'s column-averaged wide recurrent state actually
learn, at rough parity of context visibility with a dense causal
baseline?

Standard Multi-Query Associative Recall (MQAR) benchmark (Arora,
Eyuboglu et al., "Zoology", 2023 -- model/toy_recall_task.py's own
generate_mqar_sequence, a direct port of zoology's reference
implementation), AdamOptimizer + clip_grad_norm_ (real global-norm
clipping -- see clip_grad_norm_'s own docstring for why that's correct
now: both models are built from DenseTensorLinear, nothing self
-updates during backward() anymore), warmup+cosine LR schedule.

This is the SECOND real run of this comparison. The first (see
JOURNAL.md's "Real MQAR comparison run: tile-recurrence fails, root
cause found") found ToyTileRecurrence stuck at or below chance, and an
ablation traced it to two real design mistakes, both fixed here per
direct correction:

1. `num_tiles` was a fixed, tiny constant (4) -- widened here to
   `num_tiles = seq_len` per config, removing the window-narrowness
   confound. This run tests whether the mechanism can learn at all
   when it CAN see the whole sequence (same visibility as the dense
   baseline's own full causal attention) -- testing genuine cross-tick
   recall BEYOND a narrow window stays explicitly deferred to a
   follow-up once this passes.
2. The old per-tile "column" loss (next-tile classification, no state
   -width expansion at all) is gone. `ToyTileRecurrence` now has a
   genuinely WIDER internal recurrent state (`state_width =
   embed_width * column_neurons`) than its input/output, read out via
   a parameter-free column-MEAN pool (not sum, not a learned
   down-projection -- see model/toy_recall_models.py's own docstring
   for why: `lm_head` stands in for the real system's fixed-width
   pretrained output head). The unified per-position target below
   (used for BOTH models now, not just the tile one) replaces the old
   MQAR-only / column-classification split entirely.

**Unified per-position training target for ToyTileRecurrence only**
(NOT the dense control -- see train_and_eval_dense's own docstring for
why: this fix exists for the column-mean readout's specific
width-mismatch problem, which a standard dense transformer doesn't
have; changing the control's own training procedure at the same time
would confound the comparison). At a query position, the target is
the recalled value (unchanged, the real task). Within the key/value
CONTEXT-laydown region (positions `0` to `context_size-2`), the target
is the literal real next token -- true structure (each key is
genuinely followed by its value in the data), reinforcing exactly the
key->value adjacency the later query needs. Everywhere else is pure
random filler (checked directly against model/toy_recall_task.py:
`random_non_queries=True` fills non-query slots with noise, so
`tokens[i+1]` is NOT a meaningful target at a query position) --
skipped, not trained on.

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
    cross_entropy_sum, predicted_token, AdamOptimizer, clip_grad_norm_, lr_schedule,
)

COLUMN_NEURONS = 8         # state_width = embed_width * COLUMN_NEURONS
TRAIN_STEPS = 3000         # matches scripts/torch_mqar_control.py's own step count
WARMUP_STEPS = 100
PEAK_LR = 0.02
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 60

# (seq_len, num_kv_pairs, vocab_size) -- MQAR requires vocab_size > seq_len
# and seq_len >= 4*num_kv_pairs.
CONFIGS = [
    (16, 2, 20),
    (32, 4, 40),
]


def _build_targets(tokens: np.ndarray, mqar_pairs: list, num_kv_pairs: int) -> dict:
    """Unified per-position target dict -- see module docstring.
    query positions -> recalled value; context-laydown region -> real
    next token; everywhere else -> no entry (pure random filler)."""
    context_size = num_kv_pairs * 2
    targets = dict(mqar_pairs)
    for i in range(context_size - 1):
        targets.setdefault(i, int(tokens[i + 1]))
    return targets


def train_and_eval_dense(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, seed):
    """The CONTROL -- unmodified from the first real comparison run
    (JOURNAL.md's "Real MQAR comparison run"). Trains on the true MQAR
    pairs only, nothing else. Deliberately NOT given the unified
    context-region next-token target: that fix exists specifically for
    ToyTileRecurrence's column-mean readout (a mechanism for backprop
    -ing an output error into a state much WIDER than the output --
    see model/toy_recall_models.py's own docstring). A standard dense
    causal transformer has no such width mismatch (its hidden state
    already matches lm_head's own input width directly), so there's no
    equivalent problem for this loss to fix here -- changing the
    control's own training procedure at the same time as fixing tile
    would confound whatever the comparison is trying to isolate."""
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


def _build_tile_window(embed_table: np.ndarray, tokens: np.ndarray, i: int,
                       num_tiles: int, M_prev: np.ndarray, column_neurons: int) -> np.ndarray:
    """[num_tiles, state_width] -- real-token slots get their
    embed_width embedding broadcast up to state_width
    (np.repeat, parameter-free, matches the readout's own
    parameter-free column-mean pool); fallback slots (no real token
    yet) use M_prev[j] directly, already state_width since M itself
    lives at state_width now."""
    state_width = embed_table.shape[1] * column_neurons
    window = np.empty((num_tiles, state_width), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        window[j] = (np.repeat(embed_table[tokens[src]], column_neurons)
                     if src >= 0 else M_prev[j])
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
    hidden, mlp_hidden = 32, 48   # zoology's own smallest real attention-baseline
                                   # d_model for MQAR is 32 (models_repo.py's
                                   # add_attention sweeps d_model in [32, 64, 128],
                                   # n_layers=2) -- looked up, not guessed.
    tile_mlp_hidden = hidden * COLUMN_NEURONS * 2  # scaled with state_width, same ratio as dense
    print(f"column_neurons={COLUMN_NEURONS} hidden={hidden} mlp_hidden={mlp_hidden} "
          f"tile_mlp_hidden={tile_mlp_hidden} train_steps={TRAIN_STEPS} warmup={WARMUP_STEPS} "
          f"peak_lr={PEAK_LR} max_grad_norm={MAX_GRAD_NORM} eval_sequences={EVAL_SEQUENCES} "
          f"optimizer=Adam+global-norm-clip\n")
    print(f"{'seq_len':>8}  {'kv_pairs':>9}  {'vocab':>6}  {'dense_acc':>10}  {'tile_acc':>10}")
    results = []
    for seq_len, num_kv_pairs, vocab in CONFIGS:
        t0 = time.time()
        dense_acc = train_and_eval_dense(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden,
                                         seed=1000 + seq_len)
        tile_acc = train_and_eval_tile(seq_len, num_kv_pairs, vocab, hidden, tile_mlp_hidden,
                                       seed=2000 + seq_len)
        elapsed = time.time() - t0
        print(f"{seq_len:>8}  {num_kv_pairs:>9}  {vocab:>6}  {dense_acc:>10.2f}  "
              f"{tile_acc:>10.2f}   ({elapsed:.1f}s)")
        results.append((seq_len, num_kv_pairs, vocab, dense_acc, tile_acc))

    print(f"\nDense is the unmodified control (true MQAR pairs only, unchanged from "
          f"the first comparison run). ToyTileRecurrence gets the unified per-position "
          f"target (query positions -> recalled value, context-laydown region -> real "
          f"next token, elsewhere skipped) -- a fix specific to its own column-mean "
          f"readout, not applied to dense. num_tiles=seq_len per config -- "
          f"tile-recurrence can see the whole sequence directly this run "
          f"(parity-of-visibility test), not yet testing recall genuinely beyond a "
          f"narrow window.")
    return results


if __name__ == "__main__":
    main()
