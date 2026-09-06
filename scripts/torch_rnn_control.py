"""
scripts/torch_rnn_control.py
───────────────────────────────
Diagnostic CONTROL, not part of the real model -- a "this should just
work" ceiling for the out-of-context deviation-detection task, using
PyTorch's own real-BPTT recurrent modules (nn.RNN, nn.LSTM) on the
same task generator the from-scratch peak-eligibility experiment uses.
See docs/research/torch_rnn_control.rst:module_overview_task_characterization
for task characterization/citations and
docs/research/torch_rnn_control.rst:bptt_hypothesis_rejected_result for
the experimental result (BPTT-per-se is NOT the missing ingredient).

Run: python -m scripts.torch_rnn_control
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, ".")

from model.toy_beyond_context_task import VOCAB_SIZE, generate_deviation_sequence

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
        x = self.embed(tokens).unsqueeze(0)  # [1, T, hidden]
        out, _ = self.rnn(x)  # docs/research/torch_rnn_control.rst:full_sequence_bptt_design
        return self.head(out.squeeze(0))  # [T, vocab] -- logits at every position


def evaluate(model, rng):
    """See docs/research/torch_rnn_control.rst:evaluate_forward_only_shared."""
    results = {}
    model.eval()
    with torch.no_grad():
        for n_bits in EVAL_N_VALUES:
            correct = 0
            for _ in range(EVAL_SEQUENCES):
                tokens, pairs = generate_deviation_sequence(rng, n_bits)
                query_pos, answer = pairs[0]
                tokens_t = torch.from_numpy(tokens[: query_pos + 1]).long()
                logits = model(tokens_t)
                pred = int(logits[-1].argmax())
                correct += int(pred == answer)
            results[n_bits] = correct / EVAL_SEQUENCES
    return results


def train_and_eval(rnn_cls, label, seed):
    """See docs/research/torch_rnn_control.rst:full_sequence_bptt_design."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = RecurrentControl(rnn_cls, VOCAB_SIZE, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for _step in range(TRAIN_STEPS):
        n_bits = int(rng.randint(2, OUT_OF_CONTEXT_MAX + 1))  # uniform, no curriculum needed
        tokens, pairs = generate_deviation_sequence(rng, n_bits)
        query_pos, answer = pairs[0]
        tokens_t = torch.from_numpy(tokens[: query_pos + 1]).long()
        logits = model(tokens_t)
        loss = F.cross_entropy(logits[-1:], torch.tensor([answer]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    results = evaluate(model, rng)
    return results, float(np.mean(losses[-100:]))


def train_and_eval_no_bptt(rnn_cls, label, seed):
    """See docs/research/torch_rnn_control.rst:no_bptt_variant_isolation."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = RecurrentControl(rnn_cls, VOCAB_SIZE, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for _step in range(TRAIN_STEPS):
        n_bits = int(rng.randint(2, OUT_OF_CONTEXT_MAX + 1))
        tokens, pairs = generate_deviation_sequence(rng, n_bits)
        query_pos, answer = pairs[0]
        h = None
        out = None
        for t in range(query_pos + 1):
            tok = torch.tensor([[int(tokens[t])]]).long()  # [1,1]
            x = model.embed(tok)  # [1,1,hidden]
            out, h = model.rnn(x) if h is None else model.rnn(x, h)
            h = tuple(hi.detach() for hi in h) if isinstance(h, tuple) else h.detach()
        logits = model.head(out.squeeze(0))  # [1, vocab] -- the query tick's own output
        loss = F.cross_entropy(logits, torch.tensor([answer]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    results = evaluate(model, rng)
    return results, float(np.mean(losses[-100:]))


def main():
    print(f"hidden={HIDDEN} train_steps={TRAIN_STEPS} lr={LR} optimizer=Adam eval_sequences={EVAL_SEQUENCES}\n")

    print("=== full BPTT ===")
    for rnn_cls, label in [(nn.RNN, "nn.RNN (vanilla Elman)"), (nn.LSTM, "nn.LSTM")]:
        t0 = time.time()
        results, final_loss = train_and_eval(rnn_cls, label, seed=1000)
        print(f"{label}: final_loss(last100)={final_loss:.4f}  ({time.time() - t0:.1f}s)")
        print(f"  {' '.join(f'n={n}:{results[n]:.2f}' for n in EVAL_N_VALUES)}")

    print("\n=== NO BPTT (hidden state detached every tick, matching the from-scratch system) ===")
    for rnn_cls, label in [(nn.RNN, "nn.RNN (vanilla Elman)"), (nn.LSTM, "nn.LSTM")]:
        t0 = time.time()
        results, final_loss = train_and_eval_no_bptt(rnn_cls, label, seed=1000)
        print(f"{label}: final_loss(last100)={final_loss:.4f}  ({time.time() - t0:.1f}s)")
        print(f"  {' '.join(f'n={n}:{results[n]:.2f}' for n in EVAL_N_VALUES)}")

    print("\n(chance = 0.5 for a single binary answer bit)")


if __name__ == "__main__":
    main()
