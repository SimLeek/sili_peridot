"""
scripts/disldo_tanh_sparse_ablation.py
─────────────────────────────────────────
Third rung of the ablation ladder (see JOURNAL.md for the full trace):

1. disldo_no_bptt_ablation.py (full density, residual accumulate, no
   tanh): loss exploded to 1e22-1e26, accuracy near/below chance.
   Traced with FROZEN weights (lr=0): h_norm doubles every tick
   regardless of training -- a pure forward-pass linear-feedback
   instability (no squashing nonlinearity anywhere), nothing to do
   with DISLDO's training dynamics.

2. disldo_tanh_no_bptt_ablation.py (full density, h_new =
   tanh(cell(...)), matching nn.RNN's actual formula): forward-pass
   instability fixed (h_norm bounded ~1.2-1.4 indefinitely, loss stays
   sane 0.66-1.17) -- but accuracy STILL near chance. Traced weight
   updates directly: at full density (nnz_this_row=128, chosen to
   parameter-match nn.RNN), `lr_per_row_nnz=True` (hardcoded in
   sili.sparse_rnn.DISLDOLayer.forward) divides the already-modest
   peak_lr=0.05 by 128 for EVERY trainable quantity -- both the
   FP4-quantized stored weight (effective_lr ~4e-4 vs FP4's 0.5
   quantization floor -- gets stuck after at most one level jump,
   confirmed: only 0.6% of weights moved after 300 steps) AND the
   continuous (non-quantized) per-row value_scale (doesn't get stuck,
   but crawls at the same ~1/128 rate -- confirmed: <1% deviation
   from 1.0 after 300 steps, real gradient signal, just far too slow
   to matter within a couple thousand steps). Directly matches this
   project's own pre-existing `disldo_max_weights_sizing` warning
   (never size max_weights as in*out) -- now confirmed as a training
   -RATE problem, not just a speed one.

This script: same tanh/full-overwrite cell as rung 2, but SPARSE
density (PER_ROW_K=8, not full in*out) -- keeps the forward-stability
fix, removes the effective-lr collapse. No longer exactly
parameter-matched to nn.RNN's count (that parity is what caused the
problem in the first place) -- reports the real (smaller) parameter
count instead of pretending otherwise.

Run: python -m scripts.disldo_tanh_sparse_ablation
"""
from __future__ import annotations

import time

import numpy as np

from sili.sparse_rnn import DISLDOLayer
from sili import _cpu
from sili.tensor import Tensor

from model.toy_beyond_context_task import generate_deviation_sequence, VOCAB_SIZE
from model.toy_recall_models import cross_entropy_sum, predicted_token, lr_schedule

HIDDEN = 128
NUM_CPUS = 1
TRAIN_STEPS = 2000
PEAK_LR = 0.05
WARMUP_STEPS = 100
EVAL_SEQUENCES = 100
OUT_OF_CONTEXT_MAX = 6
EVAL_N_VALUES = [2, 3, 4, 6]

PER_ROW_K = 8  # sparse, not full density -- see module docstring
CELL_MAX_WEIGHTS = (HIDDEN * 2) * PER_ROW_K
HEAD_MAX_WEIGHTS = HIDDEN * VOCAB_SIZE


class DisldoTanhSparseControl:
    def __init__(self, seed: int):
        embed_rng = np.random.RandomState(seed)
        self.embed_matrix = (embed_rng.randn(VOCAB_SIZE, HIDDEN)
                              * (1.0 / np.sqrt(HIDDEN))).astype(np.float32)
        rng1 = np.random.default_rng(seed + 1)
        rng2 = np.random.default_rng(seed + 2)
        self.cell = DISLDOLayer(HIDDEN * 2, HIDDEN, CELL_MAX_WEIGHTS, NUM_CPUS, rng=rng1)
        self.head = DISLDOLayer(HIDDEN, VOCAB_SIZE, HEAD_MAX_WEIGHTS, NUM_CPUS, rng=rng2)

    def _embed(self, tok: int) -> np.ndarray:
        onehot = np.zeros(VOCAB_SIZE, dtype=np.float32)
        onehot[tok] = 1.0
        return onehot @ self.embed_matrix

    def step(self, tok: int, h_prev: np.ndarray, lr: float):
        x = np.concatenate([self._embed(tok), h_prev])[None, :]
        pre = self.cell.forward(x, lr)
        h_new = pre.tanh()
        logits = self.head.forward(h_new, lr)
        return h_new.data[0], logits

    def query_step(self, tok: int, h_prev: np.ndarray, lr: float, answer: int):
        x = np.concatenate([self._embed(tok), h_prev])[None, :]
        pre = self.cell.forward(x, lr)
        h_new = pre.tanh()
        logits = self.head.forward(h_new, lr)
        loss = cross_entropy_sum(logits, [(0, answer)])
        loss.backward()
        return h_new.data[0], logits, float(loss.data)


def train_and_eval(seed: int):
    _cpu.seed_fp4_stochastic_rng(seed)
    rng = np.random.RandomState(seed)
    model = DisldoTanhSparseControl(seed=seed + 10_000)

    losses = []
    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        n_bits = int(rng.randint(2, OUT_OF_CONTEXT_MAX + 1))
        tokens, pairs = generate_deviation_sequence(rng, n_bits)
        query_pos, answer = pairs[0]
        h = np.zeros(HIDDEN, dtype=np.float32)
        for i in range(query_pos):
            h, _ = model.step(int(tokens[i]), h, lr)
        h, _logits, loss_val = model.query_step(int(tokens[query_pos]), h, lr, answer)
        losses.append(loss_val)

    results = {}
    for n_bits in EVAL_N_VALUES:
        correct = 0
        for _ in range(EVAL_SEQUENCES):
            tokens, pairs = generate_deviation_sequence(rng, n_bits)
            query_pos, answer = pairs[0]
            h = np.zeros(HIDDEN, dtype=np.float32)
            for i in range(query_pos + 1):
                h, logits = model.step(int(tokens[i]), h, 0.0)
            pred = predicted_token(logits, 0)
            correct += int(pred == answer)
        results[n_bits] = correct / EVAL_SEQUENCES
    return results, float(np.mean(losses[-100:]))


def main():
    print(f"hidden={HIDDEN} per_row_k={PER_ROW_K} cell_params={CELL_MAX_WEIGHTS} "
          f"train_steps={TRAIN_STEPS} peak_lr={PEAK_LR} eval_sequences={EVAL_SEQUENCES} "
          f"(DISLDO, tanh full-overwrite, SPARSE, no BPTT, no curriculum)\n")
    N_SEEDS = 5
    agg = {n: [] for n in EVAL_N_VALUES}
    t0 = time.time()
    for s in range(N_SEEDS):
        results, final_loss = train_and_eval(seed=1000 + s)
        for n in EVAL_N_VALUES:
            agg[n].append(results[n])
        print(f"seed {s}: final_loss(last100)={final_loss:.4f}  "
              f"{ {n: round(results[n], 2) for n in EVAL_N_VALUES} }")
    print(f"\n({time.time()-t0:.1f}s total)")
    print(f"{'n_bits':>8}  {'mean':>6}  {'std':>6}")
    for n in EVAL_N_VALUES:
        arr = np.array(agg[n])
        print(f"{n:>8}  {arr.mean():>6.3f}  {arr.std():>6.3f}")
    print("\n(chance = 0.5 for a single binary answer bit)")


if __name__ == "__main__":
    main()
