"""
scripts/train_mqar_rmt_synapse_ablation.py
─────────────────────────────────────────────
Replicates train_mqar_rmt_optimizer_ablation.py's clip-vs-no-clip finding
(task #237/#238, torch-only: dropping DISLDOTorchLinear's max_abs_delta
hard clip alone reached acc=1.0000 on K=1 MQAR, vs production's 0.700) --
but on the REAL sili__new engine (ToyTileRecurrenceRMT, model/
toy_tile_recurrence_rmt.py), now that two real prerequisite fixes landed
in sili__new (see conversation): (1) linear_disldo.hpp's contrib formula
bug (used the raw unscaled FP4/FP8 code instead of the fully-scaled true
weight at all 8 real call sites), (2) update_ci/update_cw had no NaN/Inf
guard at all, unlike RMSpropScalePolicy's own value_scale/output_scale
update. Also required making min_decay_frac/max_abs_delta/max_ci real
per-call Python kwargs (previously hardcoded C++ literals) -- this
script is the reason that plumbing was built.

The torch port was found to NOT faithfully reproduce the real contrib
formula or the real per-row sequential update semantics -- so its
"drop the clip -> acc=1.0" result, while directionally suggestive, was
never trustworthy as a prediction for the real engine. This script
tests the real thing directly.

Tests both precisions (matching train_mqar_rmt_reference.py's own
DISLDO_CLS_BY_PRECISION): "fp4" (TrueMultiDigitLayer+dense, the actual
production precision) and "fp32" (DISLDOLayer32, unlimited precision,
isolates the optimizer-rule question from quantization).

Run: python3 scripts/train_mqar_rmt_synapse_ablation.py <config_name> <precision> [train_steps] [seed] [peak_lr]
  config_name: production | clip_off
  precision: fp4 | fp32
"""
from __future__ import annotations

import sys
import time
import json
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
from scripts.train_mqar_rmt_reference import (
    DISLDO_CLS_BY_PRECISION, seq_len_for_k, _build_targets,
    EMBED_WIDTH, COLUMN_NEURONS, NUM_MEMORY_SLOTS, MAX_WEIGHTS_PER_LAYER,
    NUM_CPUS, VOCAB, WARMUP_STEPS, MAX_GRAD_NORM, EVAL_SEQUENCES,
    L1_SPARSITY_COEF, CLIP_RANGE,
)

# lr=0.03, not this project's own PEAK_LR=0.01 default -- task #235's
# LR sweep (torch ablation) found 0.01 was never well-tuned for the
# custom RMSprop-with-importance optimizer; 0.03 was the best single
# value found (acc=0.700 at production max_abs_delta=2.0, vs 0.417 at
# lr=0.01). Using the SAME already-validated-better lr here so this
# ablation isn't confounded by re-discovering the same lr issue.
DEFAULT_PEAK_LR = 0.03

CONFIGS = {
    "production": {},                    # C++'s own tuned defaults (max_abs_delta=2.0)
    "clip_off": {"max_abs_delta": 1e30},  # effectively no clip
}


def train_and_eval(config_name: str, precision: str, num_kv_pairs: int, seed: int,
                   train_steps: int, peak_lr: float = DEFAULT_PEAK_LR,
                   log_every: int = 500, log_fn=None) -> dict:
    synapse_kwargs = CONFIGS[config_name]
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
        synapse_kwargs=synapse_kwargs, rng=model_rng)
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
    trajectory = []
    for step in range(1, train_steps + 1):
        lr = lr_schedule(step, train_steps, peak_lr, WARMUP_STEPS)
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

        if step % log_every == 0 or step == train_steps:
            mean_q_loss = float(np.mean(recent_query_loss)) if recent_query_loss else float("nan")
            recent_query_loss = []
            quick_acc = _quick_eval(40)
            trajectory.append((step, mean_q_loss, quick_acc))
            if log_fn is not None:
                log_fn(step, train_steps, time.time() - t0, mean_q_loss, quick_acc)

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

    return {"config": config_name, "precision": precision,
            "acc": correct / total if total else 0.0,
            "elapsed_s": time.time() - t0, "trajectory": trajectory}


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "production"
    precision = sys.argv[2] if len(sys.argv) > 2 else "fp4"
    train_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    peak_lr = float(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_PEAK_LR

    print(f"# ToyTileRecurrenceRMT synapse ablation config={config_name} precision={precision} "
          f"train_steps={train_steps} seed={seed} peak_lr={peak_lr} "
          f"synapse_kwargs={CONFIGS[config_name]}", flush=True)

    def log_fn(step, total_steps, elapsed, mean_q_loss, quick_acc):
        print(f"  step={step:>6}/{total_steps}  mean_query_loss={mean_q_loss:.4f}  "
              f"quick_acc={quick_acc:.4f}  ({elapsed:.0f}s elapsed)", flush=True)

    r = train_and_eval(config_name, precision, 1, seed, train_steps, peak_lr=peak_lr, log_fn=log_fn)
    print(f"\nFINAL config={config_name} precision={precision} acc={r['acc']:.4f} ({r['elapsed_s']:.0f}s)",
          flush=True)
    print("TRAJECTORY_JSON " + json.dumps(r["trajectory"]), flush=True)


if __name__ == "__main__":
    main()
