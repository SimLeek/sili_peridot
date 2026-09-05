"""
scripts/train_mqar_rmt_standard.py
─────────────────────────────────────
Runs ToyTileRecurrenceRMTStandard (model/toy_tile_recurrence_rmt_standard.py
-- a genuinely standard/vanilla torch RMT implementation, NOT another
exact port of this project's own model) on the SAME K=1 MQAR task/
harness as train_mqar_rmt_reference.py / train_mqar_rmt_torch_reference.py,
for direct comparison. See task #233 for why this exists: task #232's
exact-fidelity torch port got a "middle of the road" result (acc=0.433)
that doesn't confirm either "the port works" or "the port is broken" --
this checks whether a genuinely standard implementation solves the task
cleanly before spending time bit-diffing the two existing ports against
each other.

Run: python3 scripts/train_mqar_rmt_standard.py [train_steps] [seed]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from model.toy_recall_task import generate_mqar_sequence
from model.toy_tile_recurrence_rmt_standard import ToyTileRecurrenceRMTStandard
from scripts.train_tile_curriculum import _build_tile_window

EMBED_WIDTH = 16
COLUMN_NEURONS = 8
NUM_MEMORY_SLOTS = 2
VOCAB = 128
PEAK_LR = 0.001
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 60


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


def train_and_eval(num_kv_pairs: int, seed: int, train_steps: int, log_fn=None, eval_every: int | None = None) -> dict:
    seq_len = seq_len_for_k(num_kv_pairs)
    num_tiles = seq_len
    state_width = EMBED_WIDTH * COLUMN_NEURONS

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    model = ToyTileRecurrenceRMTStandard(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, num_tiles, NUM_MEMORY_SLOTS)
    opt = torch.optim.Adam(model.parameters(), lr=PEAK_LR)
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

    def _quick_eval(n_sequences: int) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(n_sequences):
                eval_tokens, eval_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
                eval_by_pos = dict(eval_pairs)
                memory_eval = torch.zeros(NUM_MEMORY_SLOTS, state_width)
                for i in range(seq_len):
                    window = torch.tensor(_build_tile_window(embed_table, eval_tokens, i, num_tiles))
                    memory_eval, eval_logits = model(window, memory_eval)
                    if i in eval_by_pos:
                        pred = predicted_token(eval_logits, num_tiles - 1)
                        correct += int(pred == eval_by_pos[i])
                        total += 1
        model.train()
        return correct / total if total else 0.0

    t0 = time.time()
    recent_query_loss = []
    for step in range(1, train_steps + 1):
        lr = lr_schedule(step, train_steps, PEAK_LR, WARMUP_STEPS)
        for g in opt.param_groups:
            g["lr"] = lr
        tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
        targets = _build_targets(tokens, mqar_pairs, num_kv_pairs)
        query_positions = {pos for pos, _ in mqar_pairs}
        memory = torch.zeros(NUM_MEMORY_SLOTS, state_width)
        for i in range(seq_len):
            window = torch.tensor(_build_tile_window(embed_table, tokens, i, num_tiles))
            memory, logits = model(window, memory)
            if i in targets:
                target = torch.tensor([targets[i]], dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(logits[num_tiles - 1 : num_tiles], target)
                if i in query_positions:
                    recent_query_loss.append(float(loss))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()
                memory = memory.detach()
            else:
                memory = memory.detach()

        if log_fn is not None and (step % max(train_steps // 10, 1) == 0 or step == train_steps):
            mean_q_loss = float(np.mean(recent_query_loss)) if recent_query_loss else float("nan")
            recent_query_loss = []
            quick_acc = _quick_eval(40) if eval_every and step % eval_every == 0 else None
            log_fn(num_kv_pairs, step, train_steps, time.time() - t0, mean_q_loss, quick_acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(EVAL_SEQUENCES):
            tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
            mqar_by_pos = dict(mqar_pairs)
            memory = torch.zeros(NUM_MEMORY_SLOTS, state_width)
            for i in range(seq_len):
                window = torch.tensor(_build_tile_window(embed_table, tokens, i, num_tiles))
                memory, logits = model(window, memory)
                if i in mqar_by_pos:
                    pred = predicted_token(logits, num_tiles - 1)
                    correct += int(pred == mqar_by_pos[i])
                    total += 1

    return {
        "num_kv_pairs": num_kv_pairs,
        "seq_len": seq_len,
        "acc": correct / total if total else 0.0,
        "elapsed_s": time.time() - t0,
    }


def main():
    train_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(
        f"# ToyTileRecurrenceRMTStandard (K=1) train_steps={train_steps} seed={seed} "
        f"num_memory_slots={NUM_MEMORY_SLOTS} embed_width={EMBED_WIDTH} "
        f"column_neurons={COLUMN_NEURONS} state_width={EMBED_WIDTH * COLUMN_NEURONS} "
        f"vocab={VOCAB} peak_lr={PEAK_LR}",
        flush=True,
    )

    def log_fn(k, step, total_steps, elapsed, mean_q_loss, quick_acc=None):
        acc_str = f"  quick_acc={quick_acc:.4f}" if quick_acc is not None else ""
        print(
            f"  step={step:>6}/{total_steps}  mean_query_loss={mean_q_loss:.4f}{acc_str}  ({elapsed:.0f}s elapsed)",
            flush=True,
        )

    r = train_and_eval(1, seed, train_steps, log_fn=log_fn, eval_every=max(train_steps // 10, 1))
    print(f"\nFINAL acc={r['acc']:.4f} ({r['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
