"""
scripts/dense_tanh_no_bptt_control.py
────────────────────────────────────────
Direct follow-up to the disldo_tanh_no_bptt_ablation.py seed-2 trace:
that run converged to a confident but WRONG decision boundary (same
input always gives identical logits -- not per-step noise -- but
prediction distribution [7,13,0] is nearly the exact mirror of the
true answer distribution [13,7,0]). Per direct instruction: isolate
whether this is FP4/DISLDO-specific or something else in sili, by
swapping ONLY the cell+head layer type -- DISLDOLayer (FP4-quantized,
importance-damped inline C++ update) out, sili.tensor's own
DenseTensorLinear (plain float32 matmul) + AdamOptimizer (standard
Kingma & Ba 2014, already established/validated in this exact
codebase -- see model/toy_recall_models.py's AdamOptimizer docstring:
"full-precision + Adam control converged easily and fast on the
identical task" in an earlier, unrelated isolation) in -- everything
else IDENTICAL: same tanh/full-overwrite cell formula, same fixed
(non-trained) embedding lift, same task/curriculum-free n_bits
sampling, same seeds, same train_steps.

If this ALSO shows the same confident-wrong-boundary collapse, the
problem is in the architecture/task/no-BPTT regime itself, not FP4 or
DISLDO's own update rule. If this converges cleanly (matching the
torch no-BPTT control's near-100%), the problem is specific to
DISLDO -- narrowing down to either FP4's quantization noise
specifically, or the importance-damped update rule itself (a further,
separate question this script alone doesn't resolve).

No-BPTT convention preserved exactly: h is a fresh Tensor built from a
plain numpy leaf every tick (no graph connection to prior ticks,
matching M_prev's own established convention elsewhere in this
project) -- Adam's own per-parameter state persists across ticks (that
IS the optimizer's whole job), but the COMPUTATION GRAPH does not.

Run: python -m scripts.dense_tanh_no_bptt_control
"""
from __future__ import annotations

import time

import numpy as np

from sili.tensor import Tensor

from model.toy_beyond_context_task import generate_deviation_sequence, VOCAB_SIZE
from model.toy_recall_models import (
    DenseTensorLinear, AdamOptimizer, cross_entropy_sum, predicted_token,
)

HIDDEN = 128
TRAIN_STEPS = 2000
LR = 1e-3  # identical to the DISLDO ablation and the torch control
EVAL_SEQUENCES = 100
OUT_OF_CONTEXT_MAX = 6
EVAL_N_VALUES = [2, 3, 4, 6]


class DenseTanhRecurrentControl:
    """h_new = tanh(cell([x_embed, h_prev])) -- identical formula to
    DisldoTanhRecurrentControl, DenseTensorLinear instead of
    DISLDOLayer, trained via AdamOptimizer instead of DISLDO's own
    inline importance-damped update."""

    def __init__(self, seed: int):
        embed_rng = np.random.RandomState(seed)
        self.embed_matrix = (embed_rng.randn(VOCAB_SIZE, HIDDEN)
                              * (1.0 / np.sqrt(HIDDEN))).astype(np.float32)
        np.random.seed(seed + 1)
        self.cell = DenseTensorLinear(HIDDEN * 2, HIDDEN, scale=0.1)
        np.random.seed(seed + 2)
        self.head = DenseTensorLinear(HIDDEN, VOCAB_SIZE, scale=0.1)
        self.opt = AdamOptimizer()

    def _embed(self, tok: int) -> np.ndarray:
        onehot = np.zeros(VOCAB_SIZE, dtype=np.float32)
        onehot[tok] = 1.0
        return onehot @ self.embed_matrix

    def _params(self):
        return self.cell.parameters() + self.head.parameters()

    def step(self, tok: int, h_prev: np.ndarray, lr: float):
        # No weight update on non-query ticks -- matches DISLDO ablation's
        # own PlainCell/query_step convention (only the query tick trains).
        x = Tensor(np.concatenate([self._embed(tok), h_prev])[None, :].astype(np.float32))
        pre = self.cell.forward(x)
        h_new = pre.tanh()
        logits = self.head.forward(h_new)
        return h_new.data[0], logits

    def query_step(self, tok: int, h_prev: np.ndarray, lr: float, answer: int):
        x = Tensor(np.concatenate([self._embed(tok), h_prev])[None, :].astype(np.float32))
        pre = self.cell.forward(x)
        h_new = pre.tanh()
        logits = self.head.forward(h_new)
        loss = cross_entropy_sum(logits, [(0, answer)])
        loss.backward()
        if lr != 0.0:
            self.opt.step(self._params(), lr)
        return h_new.data[0], logits, float(loss.data)


def train_and_eval(seed: int):
    rng = np.random.RandomState(seed)
    model = DenseTanhRecurrentControl(seed=seed + 10_000)

    losses = []
    for step in range(TRAIN_STEPS):
        n_bits = int(rng.randint(2, OUT_OF_CONTEXT_MAX + 1))
        tokens, pairs = generate_deviation_sequence(rng, n_bits)
        query_pos, answer = pairs[0]
        h = np.zeros(HIDDEN, dtype=np.float32)
        for i in range(query_pos):
            h, _ = model.step(int(tokens[i]), h, LR)
        h, _logits, loss_val = model.query_step(int(tokens[query_pos]), h, LR, answer)
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
    print(f"hidden={HIDDEN} train_steps={TRAIN_STEPS} lr={LR} "
          f"eval_sequences={EVAL_SEQUENCES} (sili DENSE Tensor + Adam, tanh full-overwrite, "
          f"no BPTT, no curriculum -- no FP4/DISLDO anywhere)\n")
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
