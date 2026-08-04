"""
scripts/torch_rnn_control.py
───────────────────────────────
Diagnostic CONTROL, not part of the real model -- a genuine "this
should just work" ceiling for the out-of-context deviation-detection
task (model/toy_beyond_context_task.py's generate_deviation_sequence),
using PyTorch's OWN built-in, real-BPTT recurrent modules (nn.RNN,
nn.LSTM) and a standard optimizer (Adam), on the EXACT SAME task
generator the from-scratch peak-eligibility experiment uses.

Task is functionally a detection/latch problem (did any deviation
occur anywhere in the sequence -- 1 bit of carried state, not a
k-symbol copy), closer to Hochreiter & Schmidhuber's original
long-lag/latch problems (1997, "Long Short-Term Memory", Neural
Computation 9(8)) than to the harder standard copy task (Le, Jaitly &
Hinton 2015, arXiv:1504.00941; reference PyTorch implementation with
LSTM/GRU baselines: Bai, Kolter & Koltun 2018, arXiv:1803.01271,
github.com/locuslab/TCN). At n_bits<=6 this is nowhere near the
several-dozen-step regime where vanilla RNNs' vanishing gradients
become a real barrier -- both nn.RNN AND nn.LSTM are run here
deliberately: if even the plain, no-gating vanilla RNN solves this
cleanly, that's the strongest possible sanity check that the TASK
itself is easy for anything with genuine BPTT, independent of gating
architecture. num_layers=1, hidden=128, Adam lr=1e-3 -- standard
literature defaults for a synthetic RNN memory task at this scale, not
tuned to make either arm look good.

Full, real BPTT: the whole sequence (body + query token) is fed to the
RNN module in one call (torch's fused RNN implementations backprop
through the entire unrolled sequence for free) -- no truncation, no
curriculum needed (unlike the from-scratch experiment, real BPTT
doesn't have a credit-assignment gap to work around).

Run: python -m scripts.torch_rnn_control
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from model.toy_beyond_context_task import generate_deviation_sequence, VOCAB_SIZE

HIDDEN = 128
TRAIN_STEPS = 2000
LR = 1e-3
EVAL_SEQUENCES = 100
OUT_OF_CONTEXT_MAX = 6
EVAL_N_VALUES = [2, 3, 4, 6]


class RecurrentControl(nn.Module):
    def __init__(self, rnn_cls, vocab: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.rnn = rnn_cls(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [T] (single sequence, body + query token, no batch dim)
        x = self.embed(tokens).unsqueeze(0)   # [1, T, hidden]
        out, _ = self.rnn(x)                  # full real BPTT through the whole sequence
        return self.head(out.squeeze(0))       # [T, vocab] -- logits at every position


def train_and_eval(rnn_cls, label, seed):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = RecurrentControl(rnn_cls, VOCAB_SIZE, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for step in range(TRAIN_STEPS):
        n_bits = int(rng.randint(2, OUT_OF_CONTEXT_MAX + 1))  # uniform, no curriculum needed
        tokens, pairs = generate_deviation_sequence(rng, n_bits)
        query_pos, answer = pairs[0]
        tokens_t = torch.from_numpy(tokens[:query_pos + 1]).long()
        logits = model(tokens_t)
        loss = F.cross_entropy(logits[-1:], torch.tensor([answer]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    results = {}
    model.eval()
    with torch.no_grad():
        for n_bits in EVAL_N_VALUES:
            correct = 0
            for _ in range(EVAL_SEQUENCES):
                tokens, pairs = generate_deviation_sequence(rng, n_bits)
                query_pos, answer = pairs[0]
                tokens_t = torch.from_numpy(tokens[:query_pos + 1]).long()
                logits = model(tokens_t)
                pred = int(logits[-1].argmax())
                correct += int(pred == answer)
            results[n_bits] = correct / EVAL_SEQUENCES
    return results, float(np.mean(losses[-100:]))


def main():
    print(f"hidden={HIDDEN} train_steps={TRAIN_STEPS} lr={LR} optimizer=Adam "
          f"eval_sequences={EVAL_SEQUENCES} (real BPTT, no curriculum)\n")
    for rnn_cls, label in [(nn.RNN, "nn.RNN (vanilla Elman)"), (nn.LSTM, "nn.LSTM")]:
        t0 = time.time()
        results, final_loss = train_and_eval(rnn_cls, label, seed=1000)
        print(f"{label}: final_loss(last100)={final_loss:.4f}  ({time.time()-t0:.1f}s)")
        print(f"  {' '.join(f'n={n}:{results[n]:.2f}' for n in EVAL_N_VALUES)}")
    print("\n(chance = 0.5 for a single binary answer bit)")


if __name__ == "__main__":
    main()
