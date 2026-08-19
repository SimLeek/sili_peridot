"""
scripts/train_mqar_k_sweep.py
──────────────────────────────
MQAR (Multi-Query Associative Recall, model/toy_recall_task.py:
generate_mqar_sequence) on ToyTileRecurrenceRealFP4, scaled by K =
num_kv_pairs (the number of simultaneous key->value associations the
model must hold in its recurrent state at once) -- a genuine rank/
capacity test: how many independent associations can a given
state_width actually retain before accuracy degrades, distinct from
the out-of-context-distance curriculum (train_tile_curriculum.py)
already covered elsewhere.

Uses the current best-known production configuration, assembled from
this project's own validated findings rather than guessed:
  - disldo_cls: TrueMultiDigitLayer(digit_cls=DISLDOLayer, n_stages=3,
    base=12.0, lr_power=0.0) -- STOCHASTIC rounding (beats deterministic
    for genuine superposition/rank-floor properties, see JOURNAL.md and
    project_hybrid_precision_plan memory), base=12 exact-tiling residual
    scale (the project default).
  - dense=True -- fully dense block4-loaded connectivity, confirmed to
    beat sparse-echo connectivity once instability is fixed (project_
    sili_block4_dense_loader memory).
  - l1_sparsity_coef=0.05 -- the LANDMARK dense-connectivity stability
    mechanism (JOURNAL.md 2026-08-13: mean=1.0000 across 5 seeds at
    coef=0.05/0.07, beating spectral_norm_target's own 0.8858), now
    ported into ToyTileRecurrenceRealFP4 itself (see its own
    l1_sparsity_coef docstring) -- used ALONE, replacing spectral_norm_
    target/magnitude_penalty_coef entirely (combining hurts, see the
    same docstring).
  - clip_range=6.0 (the model's own validated default, matching FP4's
    max representable magnitude).

Deliberately IN-CONTEXT (num_tiles=seq_len, matching train_toy_tile_
precision_comparison.py's own established convention) -- isolates "does
capacity degrade with K" from "does out-of-context recall work",
already a separate, previously-tested axis.

seq_len grows with K (generate_mqar_sequence requires seq_len >=
4*num_kv_pairs) -- runtime grows at least quadratically with K (longer
sequences AND a wider num_tiles attention), so this starts with a
modest default sweep/step count for a first correctness check before
scaling up.

Run: python3 scripts/train_mqar_k_sweep.py [train_steps] [seed] [k_values_csv]
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from sili.sparse_rnn import DISLDOLayer
from sili import _cpu
from model.toy_recall_task import generate_mqar_sequence
from model.toy_recall_models import cross_entropy_sum, predicted_token, AdamOptimizer, lr_schedule, clip_grad_norm_
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from model.toy_precision_models import TrueMultiDigitLayer
import functools

EMBED_WIDTH = 16
COLUMN_NEURONS = 8
MAX_WEIGHTS_PER_LAYER = 512  # dense=True loads full density into block4
                             # regardless (see block4_load_dense) --
                             # this budget is not the connectivity
                             # bottleneck for the dense path, kept
                             # modest just for headroom bookkeeping.
NUM_CPUS = 4
VOCAB = 128                 # must exceed seq_len (generate_mqar_sequence's
                            # own requirement) and needs a large enough
                            # key-vocab half (vocab//2) to draw K unique
                            # keys without replacement at the largest K
                            # tested below.
PEAK_LR = 0.01
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 60
L1_SPARSITY_COEF = 0.05
CLIP_RANGE = 6.0

DISLDO_CLS = functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                               n_stages=3, base=12.0, lr_power=0.0, dense=True)


def _build_targets(tokens: np.ndarray, mqar_pairs: list, num_kv_pairs: int) -> dict:
    """Context region (laying down the kv pairs themselves) also trains
    on ordinary next-token prediction; query positions get the real
    recall target -- matches train_toy_tile_precision_comparison.py's
    own established convention."""
    context_size = num_kv_pairs * 2
    targets = dict(mqar_pairs)
    for i in range(context_size - 1):
        targets.setdefault(i, int(tokens[i + 1]))
    return targets


def _build_tile_window(embed_table: np.ndarray, tokens: np.ndarray, i: int,
                       num_tiles: int, M_prev: np.ndarray, column_neurons: int) -> np.ndarray:
    state_width = embed_table.shape[1] * column_neurons
    window = np.empty((num_tiles, state_width), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        window[j] = (np.repeat(embed_table[tokens[src]], column_neurons)
                     if src >= 0 else M_prev[j])
    return window


def seq_len_for_k(num_kv_pairs: int) -> int:
    """generate_mqar_sequence requires seq_len >= 4*num_kv_pairs and
    even; a little headroom above the bare minimum gives the power-law
    gap sampling more room to actually vary gap sizes."""
    minimum = 4 * num_kv_pairs
    return minimum + (minimum % 2)


def train_and_eval(num_kv_pairs: int, seed: int, train_steps: int, log_fn=None,
                   pool_size: int = 1, refresh_every: int = 1, eval_every: int = None) -> dict:
    """pool_size/refresh_every: sample-efficiency fix, per direct diagnosis
    (see conversation) -- drawing a brand-new random MQAR sequence every
    single step gives each specific key/value/gap pattern exactly ONE
    gradient update before being discarded, confirmed too little signal
    to generalize within a few thousand steps at this model's scale
    (directly verified: the SAME architecture converges from loss=9.69 to
    0.006 on ONE repeated fixed example within 500 reps, so the
    architecture itself is not the problem). pool_size=1/refresh_every=1
    (the default) reproduces the original fresh-example-every-step
    behavior exactly. pool_size=N/refresh_every=M: maintains a pool of N
    generated sequences, cycled through round-robin, giving each ~M/N
    gradient updates before the whole pool is replaced with N fresh
    draws -- more repeated exposure per pattern without ever training on
    only one example forever (which wouldn't generalize to unseen
    key/value draws at all).

    eval_every: if set, runs a real (fresh-sequence) accuracy eval every
    this many steps and logs it via log_fn's optional 6th arg -- lets a
    comparison between pooled and non-pooled training see the ACTUAL
    accuracy trajectory over wall-clock time, not just training loss
    (which can look fine while still not generalizing)."""
    seq_len = seq_len_for_k(num_kv_pairs)
    num_tiles = seq_len
    state_width = EMBED_WIDTH * COLUMN_NEURONS

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    model_rng = np.random.default_rng(seed)

    model = ToyTileRecurrenceRealFP4(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, 0, num_tiles, MAX_WEIGHTS_PER_LAYER,
        num_cpus=NUM_CPUS, disldo_cls=DISLDO_CLS, rng=model_rng,
        clip_range=CLIP_RANGE, l1_sparsity_coef=L1_SPARSITY_COEF)
    opt = AdamOptimizer()
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

    def _quick_eval(n_sequences: int) -> float:
        correct, total = 0, 0
        for _ in range(n_sequences):
            eval_tokens, eval_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
            eval_by_pos = dict(eval_pairs)
            M_eval = np.zeros((num_tiles, state_width), dtype=np.float32)
            for i in range(seq_len):
                window = _build_tile_window(embed_table, eval_tokens, i, num_tiles, M_eval, COLUMN_NEURONS)
                M_eval, eval_logits, _ = model.step(window, M_eval, 0.0)
                if i in eval_by_pos:
                    pred = predicted_token(eval_logits, num_tiles - 1)
                    correct += int(pred == eval_by_pos[i])
                    total += 1
        return correct / total if total else 0.0

    t0 = time.time()
    recent_query_loss = []   # raw cross-entropy on QUERY positions only, no aux --
                             # isolates task-learning signal from the L1 penalty's
                             # own magnitude, which would otherwise mask whether the
                             # main task loss is moving at all.
    pool: list = []
    for step in range(1, train_steps + 1):
        lr = lr_schedule(step, train_steps, PEAK_LR, WARMUP_STEPS)
        if pool_size <= 1 or (step - 1) % refresh_every == 0 or not pool:
            pool = [generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs) for _ in range(pool_size)]
        tokens, mqar_pairs = pool[(step - 1) % pool_size]
        targets = _build_targets(tokens, mqar_pairs, num_kv_pairs)
        query_positions = set(pos for pos, _ in mqar_pairs)
        M = np.zeros((num_tiles, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles, M, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, lr)
            if i in targets:
                loss = cross_entropy_sum(logits, [(num_tiles - 1, targets[i])])
                if i in query_positions:
                    recent_query_loss.append(float(loss.data))
                if aux is not None:
                    loss = loss + aux
                loss.backward()
                clip_grad_norm_(model.parameters_for_optimizer(), MAX_GRAD_NORM)
                opt.step(model.parameters_for_optimizer(), lr=lr)

        if log_fn is not None and (step % max(train_steps // 10, 1) == 0 or step == train_steps):
            mean_q_loss = float(np.mean(recent_query_loss)) if recent_query_loss else float("nan")
            recent_query_loss = []
            quick_acc = _quick_eval(40) if eval_every and step % eval_every == 0 else None
            log_fn(num_kv_pairs, step, train_steps, time.time() - t0, mean_q_loss, quick_acc)

    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
        mqar_by_pos = dict(mqar_pairs)
        M = np.zeros((num_tiles, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles, M, COLUMN_NEURONS)
            M, logits, _aux = model.step(window, M, 0.0)
            if i in mqar_by_pos:
                pred = predicted_token(logits, num_tiles - 1)
                correct += int(pred == mqar_by_pos[i])
                total += 1

    return {"num_kv_pairs": num_kv_pairs, "seq_len": seq_len, "acc": correct / total if total else 0.0,
            "elapsed_s": time.time() - t0}


def main():
    train_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    k_values = ([int(k) for k in sys.argv[3].split(",")] if len(sys.argv) > 3
                else [1, 2, 4, 8])

    print(f"# train_steps={train_steps} seed={seed} k_values={k_values} "
          f"embed_width={EMBED_WIDTH} column_neurons={COLUMN_NEURONS} "
          f"state_width={EMBED_WIDTH*COLUMN_NEURONS} vocab={VOCAB} "
          f"l1_sparsity_coef={L1_SPARSITY_COEF} peak_lr={PEAK_LR}", flush=True)

    def log_fn(k, step, total_steps, elapsed, mean_q_loss, quick_acc=None):
        acc_str = f"  quick_acc={quick_acc:.4f}" if quick_acc is not None else ""
        print(f"  [K={k}] step={step:>6}/{total_steps}  mean_query_loss={mean_q_loss:.4f}{acc_str}  "
              f"({elapsed:.0f}s elapsed, {elapsed/step:.4f}s/step)", flush=True)

    results = []
    for k in k_values:
        print(f"\n=== K={k} (seq_len={seq_len_for_k(k)}) ===", flush=True)
        r = train_and_eval(k, seed, train_steps, log_fn=log_fn)
        results.append(r)
        print(f"K={k:>3}  seq_len={r['seq_len']:>4}  acc={r['acc']:.4f}  "
              f"({r['elapsed_s']:.0f}s)", flush=True)

    print("\n# SUMMARY")
    print(f"{'K':>4}  {'seq_len':>8}  {'acc':>8}")
    for r in results:
        print(f"{r['num_kv_pairs']:>4}  {r['seq_len']:>8}  {r['acc']:>8.4f}")


if __name__ == "__main__":
    main()
