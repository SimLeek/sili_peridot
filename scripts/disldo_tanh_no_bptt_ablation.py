"""See docs/research/disldo_tanh_no_bptt_ablation.rst:module_overview."""

from __future__ import annotations

import time

import numpy as np
from sili import _cpu
from sili.sparse_rnn import DISLDOLayer

from model.toy_beyond_context_task import VOCAB_SIZE, generate_deviation_sequence
from model.toy_recall_models import cross_entropy_sum, predicted_token

HIDDEN = 128
NUM_CPUS = 1
TRAIN_STEPS = 2000
LR = 1e-3  # literally torch's own Adam lr -- flat, no schedule
EVAL_SEQUENCES = 100
OUT_OF_CONTEXT_MAX = 6
EVAL_N_VALUES = [2, 3, 4, 6]

CELL_MAX_WEIGHTS = (HIDDEN * 2) * HIDDEN
HEAD_MAX_WEIGHTS = HIDDEN * VOCAB_SIZE


class DisldoTanhRecurrentControl:
    """See docs/research/disldo_tanh_no_bptt_ablation.rst:module_overview."""

    def __init__(self, seed: int):
        embed_rng = np.random.RandomState(seed)
        self.embed_matrix = (embed_rng.randn(VOCAB_SIZE, HIDDEN) * (1.0 / np.sqrt(HIDDEN))).astype(np.float32)
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
        pre = self.cell.forward(x, lr, lr_per_row_nnz=False)
        h_new = pre.tanh()
        logits = self.head.forward(h_new, lr, lr_per_row_nnz=False)
        return h_new.data[0], logits

    def query_step(self, tok: int, h_prev: np.ndarray, lr: float, answer: int):
        x = np.concatenate([self._embed(tok), h_prev])[None, :]
        pre = self.cell.forward(x, lr, lr_per_row_nnz=False)
        h_new = pre.tanh()
        logits = self.head.forward(h_new, lr, lr_per_row_nnz=False)
        loss = cross_entropy_sum(logits, [(0, answer)])
        loss.backward()
        return h_new.data[0], logits, float(loss.data)


def train_and_eval(seed: int):
    _cpu.seed_fp4_stochastic_rng(seed)
    rng = np.random.RandomState(seed)
    model = DisldoTanhRecurrentControl(seed=seed + 10_000)

    losses = []
    for _step in range(TRAIN_STEPS):
        lr = LR
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
    print(
        f"hidden={HIDDEN} train_steps={TRAIN_STEPS} lr={LR} lr_per_row_nnz=False "
        f"eval_sequences={EVAL_SEQUENCES} (DISLDO, tanh full-overwrite, no BPTT, no curriculum)\n"
    )
    N_SEEDS = 5
    agg = {n: [] for n in EVAL_N_VALUES}
    t0 = time.time()
    for s in range(N_SEEDS):
        results, final_loss = train_and_eval(seed=1000 + s)
        for n in EVAL_N_VALUES:
            agg[n].append(results[n])
        print(f"seed {s}: final_loss(last100)={final_loss:.4f}  { {n: round(results[n], 2) for n in EVAL_N_VALUES} }")
    print(f"\n({time.time() - t0:.1f}s total)")
    print(f"{'n_bits':>8}  {'mean':>6}  {'std':>6}")
    for n in EVAL_N_VALUES:
        arr = np.array(agg[n])
        print(f"{n:>8}  {arr.mean():>6.3f}  {arr.std():>6.3f}")
    print("\n(chance = 0.5 for a single binary answer bit)")


if __name__ == "__main__":
    main()
