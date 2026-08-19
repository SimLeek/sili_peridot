"""
scripts/train_mqar_rmt_reference.py
────────────────────────────────────
Runs ToyTileRecurrenceRMT (model/toy_tile_recurrence_rmt.py -- a faithful
sili__new-native reproduction of Recurrent Memory Transformer's own
mechanism) on the SAME K=1 MQAR task/harness as train_mqar_k_sweep.py's
ToyTileRecurrenceRealFP4 arms, as the control this project's own
investigation needs: same engine, same precision handling, only the
recurrence architecture differs -- see conversation and task #230 for
why plain torch is a deliberately deferred fallback, not the first
attempt.

Run: python3 scripts/train_mqar_rmt_reference.py [train_steps] [seed] [precision]
  precision: "fp4" (default, TrueMultiDigitLayer+dense, matches
  train_mqar_k_sweep.py's own production DISLDO_CLS exactly) or "fp32"
  (DISLDOLayer32, unlimited precision, matching this project's own
  established fp32-control convention).
"""
from __future__ import annotations

import sys
import time
import functools

import numpy as np

sys.path.insert(0, ".")

from sili.sparse_rnn import DISLDOLayer, DISLDOLayer32
from sili import _cpu
from model.toy_recall_task import generate_mqar_sequence
from model.toy_recall_models import cross_entropy_sum, predicted_token, AdamOptimizer, lr_schedule, clip_grad_norm_
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT
from model.toy_precision_models import TrueMultiDigitLayer
from scripts.train_tile_curriculum import _build_tile_window

EMBED_WIDTH = 16
COLUMN_NEURONS = 8
NUM_MEMORY_SLOTS = 2   # RMT's own experiments use a small handful; not yet tuned for this task
MAX_WEIGHTS_PER_LAYER = 512
NUM_CPUS = 4
VOCAB = 128
PEAK_LR = 0.01
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 60
L1_SPARSITY_COEF = 0.05
CLIP_RANGE = 6.0

DISLDO_CLS_BY_PRECISION = {
    "fp4": functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                             n_stages=3, base=12.0, lr_power=0.0, dense=True),
    "fp32": DISLDOLayer32,
}


def seq_len_for_k(num_kv_pairs: int) -> int:
    minimum = 4 * num_kv_pairs
    return minimum + (minimum % 2)


def _build_targets(tokens: np.ndarray, mqar_pairs: list, num_kv_pairs: int) -> dict:
    context_size = num_kv_pairs * 2
    targets = dict(mqar_pairs)
    for i in range(context_size - 1):
        targets.setdefault(i, int(tokens[i + 1]))
    return targets


def train_and_eval(num_kv_pairs: int, seed: int, train_steps: int, precision: str = "fp4",
                   log_fn=None, eval_every: int = None) -> dict:
    seq_len = seq_len_for_k(num_kv_pairs)
    num_tiles = seq_len
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    disldo_cls = DISLDO_CLS_BY_PRECISION[precision]
    dense = (precision == "fp4")

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    model_rng = np.random.default_rng(seed)

    model = ToyTileRecurrenceRMT(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, num_tiles, NUM_MEMORY_SLOTS,
        MAX_WEIGHTS_PER_LAYER, num_cpus=NUM_CPUS, disldo_cls=disldo_cls,
        dense=dense, clip_range=CLIP_RANGE, l1_sparsity_coef=L1_SPARSITY_COEF,
        rng=model_rng)
    opt = AdamOptimizer()
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

    def _quick_eval(n_sequences: int) -> float:
        correct, total = 0, 0
        for _ in range(n_sequences):
            eval_tokens, eval_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
            eval_by_pos = dict(eval_pairs)
            memory_eval = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
            for i in range(seq_len):
                window = _build_tile_window(embed_table, eval_tokens, i, num_tiles)
                memory_eval, eval_logits, _ = model.step(window, memory_eval, 0.0)
                if i in eval_by_pos:
                    pred = predicted_token(eval_logits, num_tiles - 1)
                    correct += int(pred == eval_by_pos[i])
                    total += 1
        return correct / total if total else 0.0

    t0 = time.time()
    recent_query_loss = []
    for step in range(1, train_steps + 1):
        lr = lr_schedule(step, train_steps, PEAK_LR, WARMUP_STEPS)
        tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
        targets = _build_targets(tokens, mqar_pairs, num_kv_pairs)
        query_positions = set(pos for pos, _ in mqar_pairs)
        memory = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles)
            memory, logits, aux = model.step(window, memory, lr)
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
        memory = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles)
            memory, logits, _aux = model.step(window, memory, 0.0)
            if i in mqar_by_pos:
                pred = predicted_token(logits, num_tiles - 1)
                correct += int(pred == mqar_by_pos[i])
                total += 1

    return {"num_kv_pairs": num_kv_pairs, "seq_len": seq_len, "precision": precision,
            "acc": correct / total if total else 0.0, "elapsed_s": time.time() - t0}


def main():
    train_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    precision = sys.argv[3] if len(sys.argv) > 3 else "fp4"

    print(f"# ToyTileRecurrenceRMT reference (K=1) train_steps={train_steps} seed={seed} "
          f"precision={precision} num_memory_slots={NUM_MEMORY_SLOTS} "
          f"embed_width={EMBED_WIDTH} column_neurons={COLUMN_NEURONS} "
          f"state_width={EMBED_WIDTH*COLUMN_NEURONS} vocab={VOCAB}", flush=True)

    def log_fn(k, step, total_steps, elapsed, mean_q_loss, quick_acc=None):
        acc_str = f"  quick_acc={quick_acc:.4f}" if quick_acc is not None else ""
        print(f"  step={step:>6}/{total_steps}  mean_query_loss={mean_q_loss:.4f}{acc_str}  "
              f"({elapsed:.0f}s elapsed)", flush=True)

    r = train_and_eval(1, seed, train_steps, precision=precision, log_fn=log_fn, eval_every=max(train_steps // 10, 1))
    print(f"\nFINAL acc={r['acc']:.4f} ({r['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
