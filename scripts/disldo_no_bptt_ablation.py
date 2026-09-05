"""
scripts/disldo_no_bptt_ablation.py
───────────────────────────────────
Ablation control, direct follow-up to scripts/torch_rnn_control.py's
no-BPTT result (both nn.RNN and nn.LSTM stayed near-100% even with the
hidden state detached every tick -- BPTT itself isn't what explains
the from-scratch system's out-of-context struggle). Per direct
instruction: copy that no-BPTT structure to a new file and swap ONLY
the recurrent cell -- PyTorch's nn.RNN out, sili's real DISLDOLayer in
-- same task (generate_deviation_sequence), same no-curriculum uniform
n_bits sampling, same train_steps/eval_sequences/n_bits range, same
tick-by-tick/detached-hidden-state training regime. If THIS still
holds up near-100%, DISLDO itself isn't the culprit either, narrowing
down what actually breaks the full from-scratch system (energy gating,
peak-synapse correction, curriculum dependence, or something else
entirely). If it collapses toward chance here, DISLDO's own
quantization/training dynamics are a real candidate, worth isolating
further.

Parameter-matched to nn.RNN's own recurrent weight count as closely as
reasonably possible without inventing new machinery: torch's control
uses nn.Embedding(vocab=3, hidden=128) -> nn.RNN(input_size=128,
hidden_size=128) (Whx 128x128 + Whh 128x128 = 32768 recurrent params)
-> nn.Linear(128, 3). Here: a FIXED (not trained) random one-hot lift
onehot(3) @ E -> hidden(128) stands in for the embedding table (E is
deterministic/seeded, not gradient-updated -- a change of basis, not a
learned representation; DISLDO has no established trainable-embedding
convention in this codebase to reuse, and wiring one up would need new
plumbing this ablation doesn't need to isolate the actual question:
does swapping in the RECURRENT CELL itself break things). The cell is
DISLDOLayer(HIDDEN*2, HIDDEN, max_weights=HIDDEN*2*HIDDEN) -- 32768
params at full density, exactly matching Whx+Whh's combined count
(ignoring nn.RNN's small bias terms). Head is
DISLDOLayer(HIDDEN, VOCAB_SIZE, max_weights=HIDDEN*VOCAB_SIZE) -- 384
params, close to nn.Linear(128,3)'s 387 (ignoring its 3-param bias).

Learning rate: NOT torch's Adam lr=1e-3 (DISLDO's own inline C++
per-row training isn't Adam, so blindly reusing that number would
confound "does DISLDO collapse" with "is DISLDO untuned at this lr").
Uses this project's own already-validated lr_schedule (peak_lr=0.05,
warmup) from prototype_peak_synapse_learning_comparison.py instead --
DISLDO evaluated under its own best-known hyperparameters, not an
arbitrary borrowed number.

No EnergyDynamics, no peak-synapse correction, no curriculum -- the
plainest possible DISLDO recurrent cell, matching "just disldo" per
direct instruction. `loss.backward()` is NOT preceded by a manual
`loss.grad = np.array(1.0, ...)` assignment (see JOURNAL.md for why
that line was removed from the sibling prototype script -- verified
functionally identical to Tensor.backward()'s own default root-grad
init for a scalar loss, but removed anyway per direct instruction to
match this codebase's actual convention going forward).

Run: python -m scripts.disldo_no_bptt_ablation
"""

from __future__ import annotations

import time

import numpy as np
from sili import _cpu
from sili.sparse_rnn import DISLDOLayer
from sili.tensor import Tensor

from model.toy_beyond_context_task import VOCAB_SIZE, generate_deviation_sequence
from model.toy_recall_models import cross_entropy_sum, lr_schedule, predicted_token

HIDDEN = 128
NUM_CPUS = 1  # required for FP4 stochastic-rounding reproducibility (see sibling script)
TRAIN_STEPS = 2000
PEAK_LR = 0.05
WARMUP_STEPS = 100
EVAL_SEQUENCES = 100
OUT_OF_CONTEXT_MAX = 6
EVAL_N_VALUES = [2, 3, 4, 6]

CELL_MAX_WEIGHTS = (HIDDEN * 2) * HIDDEN  # matches nn.RNN's Whx+Whh combined count
HEAD_MAX_WEIGHTS = HIDDEN * VOCAB_SIZE  # matches nn.Linear(128, vocab)'s weight count


class DisldoRecurrentControl:
    def __init__(self, seed: int):
        embed_rng = np.random.RandomState(seed)
        # Fixed (not trained) random lift -- see module docstring.
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
        """No-BPTT tick: h_prev is a plain numpy array (fresh Tensor
        leaf every call, no graph connection to prior ticks) -- exactly
        matching torch_rnn_control.py's detach-every-tick convention."""
        x = np.concatenate([self._embed(tok), h_prev])[None, :]
        delta = self.cell.forward(x, lr)
        h_new = Tensor(h_prev[None, :].astype(np.float32)) + delta
        logits = self.head.forward(h_new, lr)
        return h_new.data[0], logits

    def query_step(self, tok: int, h_prev: np.ndarray, lr: float, answer: int):
        x = np.concatenate([self._embed(tok), h_prev])[None, :]
        delta = self.cell.forward(x, lr)
        h_new = Tensor(h_prev[None, :].astype(np.float32)) + delta
        logits = self.head.forward(h_new, lr)
        loss = cross_entropy_sum(logits, [(0, answer)])
        loss.backward()
        return h_new.data[0], logits, float(loss.data)


def train_and_eval(seed: int):
    _cpu.seed_fp4_stochastic_rng(seed)
    rng = np.random.RandomState(seed)
    model = DisldoRecurrentControl(seed=seed + 10_000)

    losses = []
    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        n_bits = int(rng.randint(2, OUT_OF_CONTEXT_MAX + 1))  # uniform, no curriculum
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
        f"hidden={HIDDEN} train_steps={TRAIN_STEPS} peak_lr={PEAK_LR} "
        f"eval_sequences={EVAL_SEQUENCES} (DISLDO, no BPTT, no curriculum, no energy)\n"
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
