"""
scripts/train_mqar_rmt_ablation.py
─────────────────────────────────────
Runs ToyTileRecurrenceRMTAblation (model/toy_tile_recurrence_rmt_ablation.py)
on the same K=1 MQAR task/harness as every other reference in this
investigation, for one named single-factor-swap config at a time. See
task #234 for the full rationale: task #232's exact-fidelity torch
port (acc=0.433) and task #233's standard torch reference (acc=1.000)
differ in five real ways (optimizer, clip, attention positional
mechanism, norm, L1 sparsity) -- this runs all five single-swaps
independently (baseline A = exact-fidelity config, minus exactly one
factor each) plus both endpoints as sanity checks that this new
unified implementation actually reproduces the two already-known
results before trusting anything in between.

Logs every 500 steps (denser than the other reference scripts in this
investigation) specifically so the loss trajectory has enough points
for a real curve fit afterward (scripts/fit_mqar_loss_curve.py) --
distinguishing "still converging, just slower" from "genuinely
plateaued" needs more than 10 checkpoints.

Run: python3 scripts/train_mqar_rmt_ablation.py <config_name> [train_steps] [seed]
  config_name: baseline_a | swap_optimizer | swap_clip | swap_attn_bias
             | swap_norm | swap_l1_sparsity | baseline_b
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

# Multiple instances of this script run in parallel (background ablation
# sweep) -- each torch process defaults to using every available core for
# its own ops, so N parallel processes on an M-core machine oversubscribe
# to N*M effective threads and thrash instead of parallelizing (confirmed
# directly: 7-way parallel launch on an 8-thread CPU ran ~35x slower than
# isolated single-process timing predicted). Cap each process to 1 thread
# so 7 processes actually divide the 8 real threads instead of fighting
# over all of them.
torch.set_num_threads(1)

sys.path.insert(0, ".")

from model.toy_recall_task import generate_mqar_sequence
from model.toy_tile_recurrence_rmt_ablation import ToyTileRecurrenceRMTAblation, clip_grad_norm_
from scripts.train_tile_curriculum import _build_tile_window

EMBED_WIDTH = 16
COLUMN_NEURONS = 8
NUM_MEMORY_SLOTS = 2
VOCAB = 128
PEAK_LR = 0.01
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 60

_BASE = {
    "use_custom_optimizer": True,
    "use_hard_clip": True,
    "use_gaussian_bias": True,
    "use_rmsnorm": True,
    "l1_sparsity_coef": 0.05,
}
_END = {
    "use_custom_optimizer": False,
    "use_hard_clip": False,
    "use_gaussian_bias": False,
    "use_rmsnorm": False,
    "l1_sparsity_coef": 0.0,
}


def _swap(**overrides):
    cfg = dict(_BASE)
    cfg.update(overrides)
    return cfg


CONFIGS = {
    "baseline_a": _BASE,
    "swap_optimizer": _swap(use_custom_optimizer=False),
    "swap_clip": _swap(use_hard_clip=False),
    "swap_attn_bias": _swap(use_gaussian_bias=False),
    "swap_norm": _swap(use_rmsnorm=False),
    "swap_l1_sparsity": _swap(l1_sparsity_coef=0.0),
    "swap_optimizer_and_clip": _swap(use_custom_optimizer=False, use_hard_clip=False),
    "baseline_b": _END,
}

# use_custom_optimizer=False configs need Adam's OWN properly-tuned lr
# (0.001, matching the validated standard reference -- model/
# toy_tile_recurrence_rmt_standard.py, task #233), NOT the custom
# DISLDO optimizer's lr=0.01 -- confirmed as a real, not cosmetic,
# confound directly: DISLDO's effective_lr = lr/row_degree (row_degree
# ~=128 here) means its nominal 0.01 is really ~0.000078 effective, so
# reusing "0.01" for plain Adam is actually ~128x too aggressive, not
# "the same lr" at all. First pass at swap_optimizer/baseline_b using
# a uniform lr=0.01 gave curve fits with R^2<0.6 (vs >0.99 for every
# custom-optimizer config) -- a clear signature of lr-driven
# instability, not a genuine "Adam is worse" result.
_ADAM_ONLY_PEAK_LR = 0.001
NEEDS_ADAM_LR = {"swap_optimizer", "swap_optimizer_and_clip", "baseline_b"}


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


def train_and_eval(
    config_name: str,
    num_kv_pairs: int,
    seed: int,
    train_steps: int,
    log_every: int = 500,
    log_fn=None,
    peak_lr: float | None = None,
) -> dict:
    cfg = CONFIGS[config_name]
    if peak_lr is None:
        peak_lr = _ADAM_ONLY_PEAK_LR if config_name in NEEDS_ADAM_LR else PEAK_LR
    seq_len = seq_len_for_k(num_kv_pairs)
    num_tiles = seq_len
    state_width = EMBED_WIDTH * COLUMN_NEURONS

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model_rng = np.random.default_rng(seed)

    model = ToyTileRecurrenceRMTAblation(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, num_tiles, NUM_MEMORY_SLOTS, rng=model_rng, **cfg
    )
    adam_params = model.parameters_for_optimizer()
    opt = torch.optim.Adam(adam_params) if adam_params else None
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

    def _quick_eval(n_sequences: int) -> float:
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(n_sequences):
                eval_tokens, eval_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
                eval_by_pos = dict(eval_pairs)
                memory_eval = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
                for i in range(seq_len):
                    window = _build_tile_window(embed_table, eval_tokens, i, num_tiles)
                    _mp, eval_logits, _ = model.step(window, memory_eval, 0.0)
                    memory_eval = model.extract_memory()
                    if i in eval_by_pos:
                        pred = predicted_token(eval_logits, num_tiles - 1)
                        correct += int(pred == eval_by_pos[i])
                        total += 1
        return correct / total if total else 0.0

    t0 = time.time()
    recent_query_loss = []
    trajectory = []  # (step, mean_query_loss, quick_acc)
    for step in range(1, train_steps + 1):
        lr = lr_schedule(step, train_steps, peak_lr, WARMUP_STEPS)
        if opt is not None:
            for g in opt.param_groups:
                g["lr"] = lr
        tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
        targets = _build_targets(tokens, mqar_pairs, num_kv_pairs)
        query_positions = {pos for pos, _ in mqar_pairs}
        memory = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles)
            _mp, logits, aux = model.step(window, memory, lr)
            if i in targets:
                target = torch.tensor([targets[i]], dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(logits[num_tiles - 1 : num_tiles], target)
                if i in query_positions:
                    recent_query_loss.append(float(loss))
                total_loss = loss if aux is None else loss + aux
                model.zero_grad()
                total_loss.backward()
                memory = model.extract_memory()
                model.apply_updates()
                if opt is not None:
                    clip_grad_norm_(adam_params, MAX_GRAD_NORM)
                    opt.step()
            else:
                memory = model.extract_memory()

        if step % log_every == 0 or step == train_steps:
            mean_q_loss = float(np.mean(recent_query_loss)) if recent_query_loss else float("nan")
            recent_query_loss = []
            quick_acc = _quick_eval(40)
            trajectory.append((step, mean_q_loss, quick_acc))
            if log_fn is not None:
                log_fn(step, train_steps, time.time() - t0, mean_q_loss, quick_acc)

    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(EVAL_SEQUENCES):
            tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
            mqar_by_pos = dict(mqar_pairs)
            memory = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
            for i in range(seq_len):
                window = _build_tile_window(embed_table, tokens, i, num_tiles)
                _mp, logits, _aux = model.step(window, memory, 0.0)
                memory = model.extract_memory()
                if i in mqar_by_pos:
                    pred = predicted_token(logits, num_tiles - 1)
                    correct += int(pred == mqar_by_pos[i])
                    total += 1

    return {
        "config": config_name,
        "acc": correct / total if total else 0.0,
        "elapsed_s": time.time() - t0,
        "trajectory": trajectory,
    }


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "baseline_a"
    train_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    peak_lr_override = float(sys.argv[4]) if len(sys.argv) > 4 else None

    resolved_lr = (
        peak_lr_override
        if peak_lr_override is not None
        else (_ADAM_ONLY_PEAK_LR if config_name in NEEDS_ADAM_LR else PEAK_LR)
    )
    print(
        f"# ToyTileRecurrenceRMTAblation config={config_name} train_steps={train_steps} "
        f"seed={seed} peak_lr={resolved_lr} cfg={CONFIGS[config_name]}",
        flush=True,
    )

    def log_fn(step, total_steps, elapsed, mean_q_loss, quick_acc):
        print(
            f"  step={step:>6}/{total_steps}  mean_query_loss={mean_q_loss:.4f}  "
            f"quick_acc={quick_acc:.4f}  ({elapsed:.0f}s elapsed)",
            flush=True,
        )

    r = train_and_eval(config_name, 1, seed, train_steps, log_fn=log_fn, peak_lr=peak_lr_override)
    print(f"\nFINAL config={config_name} acc={r['acc']:.4f} ({r['elapsed_s']:.0f}s)", flush=True)
    print("TRAJECTORY_JSON " + json.dumps(r["trajectory"]), flush=True)


if __name__ == "__main__":
    main()
