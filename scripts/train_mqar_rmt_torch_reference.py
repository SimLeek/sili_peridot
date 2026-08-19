"""
scripts/train_mqar_rmt_torch_reference.py
───────────────────────────────────────────
Runs ToyTileRecurrenceRMTTorch (model/toy_tile_recurrence_rmt_torch.py --
an exact-as-possible torch port of the sili-based ToyTileRecurrenceRMT
control, see that module's own docstring for the full fidelity notes)
on the SAME K=1 MQAR task/harness as train_mqar_rmt_reference.py, for a
direct apples-to-apples comparison. Triggered per direct instruction
ONLY because the sili-based control (task #230) failed to learn K=1
MQAR at either precision (fp4 acc=0.200, fp32 acc=0.100) -- this
isolates "is sili__new the engine broken" from "is even a correctly
-implemented proven architecture failing here."

fp32-only (this torch port doesn't implement FP4/FP8 quantization
codecs -- see the model module's own docstring for what's deliberately
not reproduced), dense connectivity only (matching the sili fp4 arm's
dense=True, NOT the fp32 arm's dense=False -- see task #232's own
confound note).

Run: python3 scripts/train_mqar_rmt_torch_reference.py [train_steps] [seed]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from model.toy_recall_task import generate_mqar_sequence
from model.toy_tile_recurrence_rmt_torch import ToyTileRecurrenceRMTTorch, clip_grad_norm_
from scripts.train_tile_curriculum import _build_tile_window

EMBED_WIDTH = 16
COLUMN_NEURONS = 8
NUM_MEMORY_SLOTS = 2
VOCAB = 128
PEAK_LR = 0.01
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 60
L1_SPARSITY_COEF = 0.05
CLIP_RANGE = 6.0


def lr_schedule(step: int, total_steps: int, peak_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


def predicted_token(logits: torch.Tensor, row: int) -> int:
    return int(torch.argmax(logits[row]).item())


def seq_len_for_k(num_kv_pairs: int) -> int:
    minimum = 4 * num_kv_pairs
    return minimum + (minimum % 2)


def _build_targets(tokens: np.ndarray, mqar_pairs: list, num_kv_pairs: int) -> dict:
    context_size = num_kv_pairs * 2
    targets = dict(mqar_pairs)
    for i in range(context_size - 1):
        targets.setdefault(i, int(tokens[i + 1]))
    return targets


def train_and_eval(num_kv_pairs: int, seed: int, train_steps: int,
                   log_fn=None, eval_every: int = None) -> dict:
    seq_len = seq_len_for_k(num_kv_pairs)
    num_tiles = seq_len
    state_width = EMBED_WIDTH * COLUMN_NEURONS

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model_rng = np.random.default_rng(seed)

    model = ToyTileRecurrenceRMTTorch(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, num_tiles, NUM_MEMORY_SLOTS,
        clip_range=CLIP_RANGE, l1_sparsity_coef=L1_SPARSITY_COEF, rng=model_rng)
    opt = torch.optim.Adam(model.parameters_for_optimizer())
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

    def _quick_eval(n_sequences: int) -> float:
        correct, total = 0, 0
        for _ in range(n_sequences):
            eval_tokens, eval_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
            eval_by_pos = dict(eval_pairs)
            memory_eval = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
            for i in range(seq_len):
                window = _build_tile_window(embed_table, eval_tokens, i, num_tiles)
                with torch.no_grad():
                    _mp, eval_logits, _ = model.step(window, memory_eval, 0.0)
                memory_eval = model.extract_memory()
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
            _mp, logits, aux = model.step(window, memory, lr)
            if i in targets:
                target = torch.tensor([targets[i]], dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(logits[num_tiles - 1:num_tiles], target)
                if i in query_positions:
                    recent_query_loss.append(float(loss))
                total_loss = loss if aux is None else loss + aux
                model.zero_grad()
                total_loss.backward()
                memory = model.extract_memory()
                model.apply_updates()
                clip_grad_norm_(model.parameters_for_optimizer(), MAX_GRAD_NORM)
                opt.step()
            else:
                with torch.no_grad():
                    pass
                memory = model.extract_memory()

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
            with torch.no_grad():
                _mp, logits, _aux = model.step(window, memory, 0.0)
            memory = model.extract_memory()
            if i in mqar_by_pos:
                pred = predicted_token(logits, num_tiles - 1)
                correct += int(pred == mqar_by_pos[i])
                total += 1

    return {"num_kv_pairs": num_kv_pairs, "seq_len": seq_len,
            "acc": correct / total if total else 0.0, "elapsed_s": time.time() - t0}


def main():
    train_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"# ToyTileRecurrenceRMTTorch reference (K=1) train_steps={train_steps} seed={seed} "
          f"num_memory_slots={NUM_MEMORY_SLOTS} embed_width={EMBED_WIDTH} "
          f"column_neurons={COLUMN_NEURONS} state_width={EMBED_WIDTH*COLUMN_NEURONS} "
          f"vocab={VOCAB}", flush=True)

    def log_fn(k, step, total_steps, elapsed, mean_q_loss, quick_acc=None):
        acc_str = f"  quick_acc={quick_acc:.4f}" if quick_acc is not None else ""
        print(f"  step={step:>6}/{total_steps}  mean_query_loss={mean_q_loss:.4f}{acc_str}  "
              f"({elapsed:.0f}s elapsed)", flush=True)

    r = train_and_eval(1, seed, train_steps, log_fn=log_fn, eval_every=max(train_steps // 10, 1))
    print(f"\nFINAL acc={r['acc']:.4f} ({r['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
