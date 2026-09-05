"""
scripts/torch_mqar_control.py
───────────────────────────────
Diagnostic CONTROL, not part of the real model -- isolates whether the
toy tile-recurrence experiment's stuck-at-chance result is caused by
FP4 quantization / DISLDOLayer's own from-scratch training dynamics,
or by this session's own hand-rolled training loop (no established
optimizer, no momentum -- gradient clipping + plain SGD only) being
under-powered/confounded.

Same architecture shape as model.toy_recall_models.ToySmallTransformer
(RMSNorm -> single-head causal self-attn -> O -> residual -> RMSNorm ->
SwiGLU MLP -> residual, no positional encoding, kept identical on
purpose so precision/optimizer are the ONLY things that differ), full
fp32 precision (plain torch.nn.Linear, no FP4/DISLDOLayer), trained
with a real, established optimizer (Adam) via torch's own autograd --
on the EXACT SAME generate_mqar_sequence task (model/toy_recall_task.py,
itself already a faithful port of the standard benchmark, unaffected
by this control).

Run: python -m scripts.torch_mqar_control
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, ".")

from model.toy_recall_task import generate_mqar_sequence


class DenseLayer(nn.Module):
    def __init__(self, hidden: int, mlp_hidden: int):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(hidden, mlp_hidden, bias=False)
        self.up = nn.Linear(hidden, mlp_hidden, bias=False)
        self.down = nn.Linear(mlp_hidden, hidden, bias=False)
        self.ln1 = nn.RMSNorm(hidden)
        self.ln2 = nn.RMSNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, hidden = x.shape
        normed = self.ln1(x)
        q, k, v = self.q(normed), self.k(normed), self.v(normed)
        scores = (q @ k.T) / (hidden**0.5)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1) @ v
        x = x + self.o(attn)
        normed2 = self.ln2(x)
        mlp = self.down(F.silu(self.gate(normed2)) * self.up(normed2))
        return x + mlp


class DenseModel(nn.Module):
    def __init__(self, vocab: int, hidden: int, mlp_hidden: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([DenseLayer(hidden, mlp_hidden) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def train_and_eval(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, n_layers, train_steps, lr, eval_sequences, seed):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = DenseModel(vocab, hidden, mlp_hidden, n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(train_steps):
        tokens, pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        tokens_t = torch.from_numpy(tokens).long()
        logits = model(tokens_t)
        positions = torch.tensor([p for p, _t in pairs])
        targets = torch.tensor([t for _p, t in pairs])
        loss = F.cross_entropy(logits[positions], targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % max(1, train_steps // 10) == 0:
            recent = np.mean(losses[-50:])
            print(f"    step={step:5d}  loss(avg last 50)={recent:.3f}")

    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for _ in range(eval_sequences):
            tokens, pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
            tokens_t = torch.from_numpy(tokens).long()
            logits = model(tokens_t)
            for pos, target in pairs:
                pred = int(logits[pos].argmax())
                correct += int(pred == target)
                total += 1
    return correct / total


def main():
    hidden, mlp_hidden, n_layers = 32, 48, 2
    train_steps = 3000
    lr = 1e-3
    eval_sequences = 60
    configs = [(16, 2, 20), (32, 4, 40)]

    print(
        f"hidden={hidden} mlp_hidden={mlp_hidden} n_layers={n_layers} "
        f"train_steps={train_steps} lr={lr} optimizer=Adam precision=fp32\n"
    )
    for seq_len, num_kv_pairs, vocab in configs:
        print(f"seq_len={seq_len} num_kv_pairs={num_kv_pairs} vocab={vocab}")
        t0 = time.time()
        acc = train_and_eval(
            seq_len,
            num_kv_pairs,
            vocab,
            hidden,
            mlp_hidden,
            n_layers,
            train_steps,
            lr,
            eval_sequences,
            seed=1000 + seq_len,
        )
        print(f"  -> eval accuracy: {acc:.2f}  ({time.time() - t0:.1f}s)\n")


if __name__ == "__main__":
    main()
