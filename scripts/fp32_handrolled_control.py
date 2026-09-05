"""
scripts/fp32_handrolled_control.py
────────────────────────────────────
Second diagnostic CONTROL, not part of the real model. The first
control (scripts/torch_mqar_control.py) changed BOTH precision (fp32
vs FP4) and optimizer (Adam vs this session's own hand-rolled per-node
gradient clipping + plain SGD) at once, and converged easily -- proving
the ARCHITECTURE wasn't the bottleneck, but leaving FP4 vs the
hand-rolled optimizer unresolved as the actual cause.

This control changes ONLY precision: same architecture shape as
ToySmallTransformer (RMSNorm -> single-head causal self-attn -> O ->
residual -> RMSNorm -> SwiGLU MLP -> residual, no positional
encoding), but built from DenseTensorLinear (plain fp32
sili.tensor matmul, no DISLDOLayer/FP4) instead -- trained with the
EXACT SAME hand-rolled loop as the real toy models
(backward_with_grad_clip, apply_gradient_step, lr_schedule), on the
EXACT SAME generate_mqar_sequence task. If THIS converges, the
optimizer was never the problem -- FP4 quantization is. If it also
gets stuck, the hand-rolled optimizer itself is the bottleneck (or a
real part of it), independent of precision.

Run: python -m scripts.fp32_handrolled_control
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from sili.tensor import Tensor, banded_attention, silu

from model.toy_recall_models import (
    DenseTensorLinear,
    apply_gradient_step,
    backward_with_grad_clip,
    cross_entropy_sum,
    lr_schedule,
    predicted_token,
    rmsnorm_tensor,
)
from model.toy_recall_task import generate_mqar_sequence


class Fp32Layer:
    def __init__(self, hidden: int, mlp_hidden: int):
        self.q = DenseTensorLinear(hidden, hidden)
        self.k = DenseTensorLinear(hidden, hidden)
        self.v = DenseTensorLinear(hidden, hidden)
        self.o = DenseTensorLinear(hidden, hidden)
        self.gate = DenseTensorLinear(hidden, mlp_hidden)
        self.up = DenseTensorLinear(hidden, mlp_hidden)
        self.down = DenseTensorLinear(mlp_hidden, hidden)
        self.input_ln = Tensor(np.ones(hidden, dtype=np.float32))
        self.post_ln = Tensor(np.ones(hidden, dtype=np.float32))

    def parameters(self):
        params = [self.input_ln, self.post_ln]
        for layer in (self.q, self.k, self.v, self.o, self.gate, self.up, self.down):
            params += layer.parameters()
        return params


class Fp32Transformer:
    """Same architecture as ToySmallTransformer, DenseTensorLinear
    instead of DISLDOLayer -- see module docstring."""

    def __init__(self, vocab_size, hidden, mlp_hidden, n_layers, num_cpus=2, rms_eps=1e-6):
        self.hidden = hidden
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.layers = [Fp32Layer(hidden, mlp_hidden) for _ in range(n_layers)]
        self.lm_head = DenseTensorLinear(hidden, vocab_size)

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params + self.lm_head.parameters()

    def forward(self, embedded: np.ndarray) -> Tensor:
        T = embedded.shape[0]
        x = Tensor(embedded.astype(np.float32))
        for layer in self.layers:
            normed = rmsnorm_tensor(x, layer.input_ln, self.rms_eps)
            q = layer.q.forward(normed)
            k = layer.k.forward(normed)
            v = layer.v.forward(normed)
            attn = banded_attention(q, k, v, half_bandwidth=T, num_cpus=self.num_cpus, causal=True)
            attn = layer.o.forward(attn)
            x = x + attn
            normed2 = rmsnorm_tensor(x, layer.post_ln, self.rms_eps)
            gate = layer.gate.forward(normed2)
            up = layer.up.forward(normed2)
            mlp_out = layer.down.forward(silu(gate) * up)
            x = x + mlp_out
        return self.lm_head.forward(x)


def train_and_eval(
    seq_len,
    num_kv_pairs,
    vocab,
    hidden,
    mlp_hidden,
    n_layers,
    train_steps,
    warmup_steps,
    peak_lr,
    max_grad_norm,
    eval_sequences,
    seed,
):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)  # DenseTensorLinear's own init uses the global RNG
    model = Fp32Transformer(vocab, hidden, mlp_hidden, n_layers)
    embed_table = rng.randn(vocab, hidden).astype(np.float32) * 0.3

    losses = []
    for step in range(train_steps):
        lr = lr_schedule(step, train_steps, peak_lr, warmup_steps)
        tokens, pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        logits = model.forward(embed_table[tokens])
        loss = cross_entropy_sum(logits, pairs)
        backward_with_grad_clip(loss, max_grad_norm)
        apply_gradient_step(model.parameters(), lr=lr)
        losses.append(float(loss.data))
        if step % max(1, train_steps // 10) == 0:
            print(f"    step={step:5d}  lr={lr:.4f}  loss(avg last 50)={np.mean(losses[-50:]):.3f}")

    correct, total = 0, 0
    for _ in range(eval_sequences):
        tokens, pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        logits = model.forward(embed_table[tokens])
        for pos, target in pairs:
            correct += int(predicted_token(logits, pos) == target)
            total += 1
    return correct / total


def main():
    hidden, mlp_hidden, n_layers = 32, 48, 2
    train_steps = 3000
    warmup_steps = 100
    peak_lr = 0.02
    max_grad_norm = 1.0
    eval_sequences = 60
    configs = [(16, 2, 20), (32, 4, 40)]

    print(
        f"hidden={hidden} mlp_hidden={mlp_hidden} n_layers={n_layers} "
        f"train_steps={train_steps} warmup={warmup_steps} peak_lr={peak_lr} "
        f"max_grad_norm={max_grad_norm} precision=fp32 optimizer=hand-rolled-SGD+clip\n"
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
            warmup_steps,
            peak_lr,
            max_grad_norm,
            eval_sequences,
            seed=1000 + seq_len,
        )
        print(f"  -> eval accuracy: {acc:.2f}  ({time.time() - t0:.1f}s)\n")


if __name__ == "__main__":
    main()
