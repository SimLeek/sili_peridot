"""
scripts/train_mqar_precision_sweep.py
──────────────────────────────────────
Real-engine MQAR test matrix (task #247): {fp4, fp4_dual, fp8} x
{rank1, rank2} x {magnitude_scale on/off}, plus a fp32 arm as the
unquantized control (outside the 3x2x2 matrix -- fp32's DISLDOLayerV
backend has no scale/magnitude concept at all, see
ToyTileRecurrenceRMT.magnitude_rescale_output's own guard).

Built directly on train_mqar_rmt_synapse_ablation.py's harness
(ToyTileRecurrenceRMT, K=1 MQAR, real disldo_cls training) -- clip
config is held FIXED at "nocaps" (task #237/#238's own validated
winner: BOTH max_abs_delta AND max_ci removed) so this sweep isolates
precision/rank/magnitude-scale specifically, not re-litigating clip
config (see feedback_do_science_correctly memory: one varied axis at
a time).

precision arms (NOT train_mqar_rmt_reference.py's own
DISLDO_CLS_BY_PRECISION -- that dict's "fp4" is the OLD 3-stage
TrueMultiDigitLayer; this sweep needs a genuinely PLAIN single-FP4
arm as its own comparison point, so it defines precisions locally):
  fp4      -- plain DISLDOLayer, single real stochastic FP4 (E2M1)
  fp4_dual -- TrueMultiDigitLayer(digit_cls=DISLDOLayer, n_stages=2,
              base=12.0) -- task #246's new arm, "fp4+fp4 dual"
  fp8      -- DISLDOLayer8, single real stochastic FP8 (E4M3)
  fp32     -- DISLDOLayer32, control (unlimited precision, no
              magnitude-scale applicable)

magnitude_scale on/off: when on, calls model.magnitude_rescale_output
(target=16.0, correction_rate=0.01, scale_invariant=False) after every
real optimizer step -- same target/correction_rate/cadence as the
torch-validated prototype (toy_tile_recurrence_rmt_torch.py's own
default use_magnitude_scale config), not a new untested value.

Run: python3 scripts/train_mqar_precision_sweep.py <precision> <scale_rank> <magnitude_scale:0|1> [train_steps] [seed] [peak_lr]
  precision: fp4 | fp4_dual | fp8 | fp32
  scale_rank: 1 | 2 (ignored for fp32 -- DISLDOLayer32 has no scale_rank concept)
  magnitude_scale: 0 | 1 (ignored for fp32)
"""
from __future__ import annotations

import sys
import time
import json
import functools

import numpy as np

sys.path.insert(0, ".")

from sili.sparse_rnn import DISLDOLayer, DISLDOLayer8, DISLDOLayer32
from sili import _cpu
from model.toy_recall_task import generate_mqar_sequence
from model.toy_recall_models import cross_entropy_sum, predicted_token, AdamOptimizer, lr_schedule, clip_grad_norm_
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT
from model.toy_precision_models import TrueMultiDigitLayer
from scripts.train_tile_curriculum import _build_tile_window
from scripts.train_mqar_rmt_reference import (
    seq_len_for_k, _build_targets,
    EMBED_WIDTH, COLUMN_NEURONS, NUM_MEMORY_SLOTS, MAX_WEIGHTS_PER_LAYER,
    NUM_CPUS, VOCAB, WARMUP_STEPS, MAX_GRAD_NORM, EVAL_SEQUENCES,
    L1_SPARSITY_COEF, CLIP_RANGE,
)

DEFAULT_PEAK_LR = 0.03  # see train_mqar_rmt_synapse_ablation.py's own rationale
NOCAPS_KWARGS = {"max_abs_delta": 1e30, "max_ci": 1e30}
MAGNITUDE_SCALE_TARGET = 16.0
MAGNITUDE_CORRECTION_RATE = 0.01

PRECISION_CLS = {
    "fp4": DISLDOLayer,
    "fp4_dual": functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                                  n_stages=2, base=12.0, lr_power=0.0),
    "fp8": DISLDOLayer8,
    "fp32": DISLDOLayer32,
}


def train_and_eval(precision: str, scale_rank: int, magnitude_scale: bool,
                   num_kv_pairs: int, seed: int, train_steps: int,
                   peak_lr: float = DEFAULT_PEAK_LR,
                   log_every: int = 500, log_fn=None) -> dict:
    seq_len = seq_len_for_k(num_kv_pairs)
    num_tiles = seq_len
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    disldo_cls = PRECISION_CLS[precision]
    dense = precision != "fp32"
    effective_rank = scale_rank if precision != "fp32" else 1
    use_magscale = magnitude_scale and precision != "fp32"

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    model_rng = np.random.default_rng(seed)

    model = ToyTileRecurrenceRMT(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, num_tiles, NUM_MEMORY_SLOTS,
        MAX_WEIGHTS_PER_LAYER, num_cpus=NUM_CPUS, disldo_cls=disldo_cls,
        dense=dense, clip_range=CLIP_RANGE, l1_sparsity_coef=L1_SPARSITY_COEF,
        synapse_kwargs=NOCAPS_KWARGS, scale_rank=effective_rank, rng=model_rng)
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
                if use_magscale:
                    model.magnitude_rescale_output(MAGNITUDE_SCALE_TARGET, MAGNITUDE_CORRECTION_RATE)

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

    return {"precision": precision, "scale_rank": effective_rank, "magnitude_scale": use_magscale,
            "acc": correct / total if total else 0.0,
            "elapsed_s": time.time() - t0, "trajectory": trajectory}


def main():
    precision = sys.argv[1] if len(sys.argv) > 1 else "fp4"
    scale_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    magnitude_scale = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    train_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    peak_lr = float(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_PEAK_LR

    print(f"# MQAR precision sweep precision={precision} scale_rank={scale_rank} "
          f"magnitude_scale={magnitude_scale} train_steps={train_steps} seed={seed} "
          f"peak_lr={peak_lr} config=nocaps", flush=True)

    def log_fn(step, total_steps, elapsed, mean_q_loss, quick_acc):
        print(f"  step={step:>6}/{total_steps}  mean_query_loss={mean_q_loss:.4f}  "
              f"quick_acc={quick_acc:.4f}  ({elapsed:.0f}s elapsed)", flush=True)

    r = train_and_eval(precision, scale_rank, magnitude_scale, 1, seed, train_steps,
                       peak_lr=peak_lr, log_fn=log_fn)
    print(f"\nFINAL precision={precision} scale_rank={r['scale_rank']} "
          f"magnitude_scale={r['magnitude_scale']} acc={r['acc']:.4f} ({r['elapsed_s']:.0f}s)",
          flush=True)
    print("TRAJECTORY_JSON " + json.dumps(r["trajectory"]), flush=True)


if __name__ == "__main__":
    main()
