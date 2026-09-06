"""sili_peridot/model/toy_recall_models.py
Training-oriented toy models for the tile-recurrence architecture on the
synthetic induction-recall task (model/toy_recall_task.py).
See docs/research/toy_recall_models.rst:module_overview,
:optimizer_choice_isolation_controls."""

from __future__ import annotations

import numpy as np
from sili.tensor import (
    Tensor,
    _topo_sort,
    banded_attention,
    exp,
    gather,
    gaussian_attention,
    log,
    neg,
    power,
    reduce_sum,
    silu,
)


class DenseTensorLinear:
    """Plain fp32 Tensor-graph linear layer (matmul-based, no quantization),
    trains via AdamOptimizer.step().
    See docs/research/toy_recall_models.rst:optimizer_choice_isolation_controls."""

    def __init__(self, in_features: int, out_features: int, scale: float = 0.1):
        self.weight = Tensor((np.random.randn(in_features, out_features) * scale).astype(np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight

    def parameters(self) -> list[Tensor]:
        return [self.weight]


def rmsnorm_tensor(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """x: [T, hidden] Tensor. Same formula as sili_block.rmsnorm, built from
    Tensor ops so gradient can flow through it.
    See docs/research/toy_recall_models.rst:rmsnorm_tensor_gradient_flow."""
    hidden = x.data.shape[-1]
    mean_sq = reduce_sum(x * x, axis=-1) * (1.0 / hidden)  # [T]
    mean_sq = mean_sq.reshape((x.data.shape[0], 1))  # [T, 1]
    rrms = (mean_sq + eps) ** -0.5  # [T, 1]
    return (x * rrms) * weight


def sigmoid_tensor(x: Tensor) -> Tensor:
    """1/(1+e^-x), built from existing Tensor primitives -- NOT `bounded_gate`.
    See docs/research/toy_recall_models.rst:sigmoid_tensor_vs_bounded_gate."""
    return power(exp(neg(x)) + 1.0, -1.0)


def cross_entropy_sum(logits: Tensor, row_target_pairs: list[tuple[int, int]]) -> Tensor:
    """logits: [N, vocab_size] Tensor. row_target_pairs: [(row, target_
    token_id), ...] -- returns the SUM of softmax cross-entropy loss
    over each pair (caller divides by len(...) for a mean).
    See docs/research/toy_recall_models.rst:cross_entropy_sum_and_predicted_token."""
    vocab_size = logits.data.shape[-1]
    row_max = logits.data.max(axis=-1, keepdims=True).astype(np.float32)  # [N,1], detached
    shifted = logits + Tensor(-row_max)  # [N,vocab], stable
    log_sum_exp = log(reduce_sum(exp(shifted), axis=-1)) + Tensor(row_max.reshape(-1))  # [N]
    rows = [r for r, _t in row_target_pairs]
    log_sum_exp_rows = gather(log_sum_exp, rows)  # [len(pairs)]
    flat_target_idx = [r * vocab_size + t for r, t in row_target_pairs]
    target_logits = gather(logits, flat_target_idx)  # [len(pairs)]
    return reduce_sum(log_sum_exp_rows - target_logits)


def predicted_token(logits: Tensor, row: int) -> int:
    """Inference-time-only readout -- reads .data directly (no gradient
    possible through argmax).
    See docs/research/toy_recall_models.rst:cross_entropy_sum_and_predicted_token."""
    return int(np.argmax(logits.data[row]))


def apply_gradient_step(params: list[Tensor], lr: float) -> None:
    """Plain SGD step + zero_grad for ordinary Tensor leaves. Kept for
    tests/comparison, superseded by AdamOptimizer for real training.
    See docs/research/toy_recall_models.rst:optimizer_choice_isolation_controls."""
    for p in params:
        if p.grad is not None:
            p.data = p.data - lr * np.asarray(p.grad, dtype=np.float32)
            p.zero_grad()


class AdamOptimizer:
    """Standard Adam (Kingma & Ba, 2014) for plain Tensor leaves, keyed by
    `id(param)` (each distinct Tensor leaf gets its own independent moment
    state).
    See docs/research/toy_recall_models.rst:optimizer_choice_isolation_controls."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params: list[Tensor], lr: float) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for p in params:
            if p.grad is None:
                continue
            g = np.asarray(p.grad, dtype=np.float32)
            key = id(p)
            if key not in self.m:
                self.m[key] = np.zeros_like(p.data)
                self.v[key] = np.zeros_like(p.data)
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[key] / bc1
            v_hat = self.v[key] / bc2
            p.data = p.data - lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.zero_grad()


def backward_with_grad_clip(loss: Tensor, max_grad_norm: float) -> None:
    """Gradient-clipped replacement for `loss.backward()` -- clips the L2
    norm of EVERY node's incoming gradient (not just the final parameter
    gradients) to `max_grad_norm`, right before that node's own
    `_backward()` fires.
    See docs/research/toy_recall_models.rst:backward_with_grad_clip_per_node."""
    if loss.grad is None:
        loss.grad = np.ones_like(loss.data)
    for node in reversed(_topo_sort(loss)):
        if node.grad is not None:
            g = np.asarray(node.grad, dtype=np.float32)
            norm = float(np.sqrt(np.sum(g.astype(np.float64) ** 2)))
            if norm > max_grad_norm and norm > 0:
                node.grad = (g * (max_grad_norm / norm)).astype(np.float32)
        node._backward()


def lr_schedule(step: int, total_steps: int, peak_lr: float, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup + cosine decay, matching nanoGPT's own convention.
    See docs/research/toy_recall_models.rst:lr_schedule_nanogpt_convention."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine))


def clip_grad_norm_(params: list[Tensor], max_norm: float) -> float:
    """Textbook GLOBAL gradient-norm clipping -- the total L2 norm ACROSS
    ALL of `params`' gradients combined is capped to `max_norm` (matching
    torch.nn.utils.clip_grad_norm_). Call after plain `loss.backward()`
    (not `backward_with_grad_clip`) and before `optimizer.step()`.
    See docs/research/toy_recall_models.rst:clip_grad_norm_global_vs_per_node."""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float(np.sum(np.asarray(p.grad, dtype=np.float64) ** 2))
    total_norm = total_sq**0.5
    if not np.isfinite(total_norm):
        # Real bug, fixed: zero the gradient instead of clipping it when the
        # norm isn't finite -- see the RST anchor above for why an unguarded
        # NaN/Inf norm would otherwise silently skip the clip below.
        for p in params:
            if p.grad is not None:
                p.grad = np.zeros_like(np.asarray(p.grad, dtype=np.float32))
        return total_norm
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = (np.asarray(p.grad, dtype=np.float32) * scale).astype(np.float32)
    return total_norm


# ═══════════════════════════════════════════════════════════════════════════
#  ToySmallTransformer -- the "pre-converted transformer" baseline
# ═══════════════════════════════════════════════════════════════════════════


class _ToyTransformerLayer:
    def __init__(self, hidden: int, mlp_hidden: int):
        self.q_proj = DenseTensorLinear(hidden, hidden)
        self.k_proj = DenseTensorLinear(hidden, hidden)
        self.v_proj = DenseTensorLinear(hidden, hidden)
        self.o_proj = DenseTensorLinear(hidden, hidden)
        self.gate_proj = DenseTensorLinear(hidden, mlp_hidden)
        self.up_proj = DenseTensorLinear(hidden, mlp_hidden)
        self.down_proj = DenseTensorLinear(mlp_hidden, hidden)
        self.input_ln = Tensor(np.ones(hidden, dtype=np.float32))
        self.post_ln = Tensor(np.ones(hidden, dtype=np.float32))

    def parameters(self) -> list[Tensor]:
        params = [self.input_ln, self.post_ln]
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.gate_proj, self.up_proj, self.down_proj):
            params += layer.parameters()
        return params


class ToySmallTransformer:
    """Stacked causal dense transformer -- each layer has its OWN distinct
    weights. Single-head attention, no positional encoding.
    `half_bandwidth` defaults to unlimited; set to an int `W` for a
    genuinely bounded context window, structurally unable to see more than
    `W` positions back.
    See docs/research/toy_recall_models.rst:tosmalltransformer_half_bandwidth."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
        half_bandwidth: int | None = None,
    ):
        self.hidden = hidden
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.half_bandwidth = half_bandwidth
        self.layers = [_ToyTransformerLayer(hidden, mlp_hidden) for _ in range(n_layers)]
        self.lm_head = DenseTensorLinear(hidden, vocab_size)

    def parameters(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params + self.lm_head.parameters()

    def forward(self, embedded: np.ndarray) -> Tensor:
        """embedded: [T, hidden] numpy (fixed embedding lookups).
        Returns logits [T, vocab_size] Tensor."""
        T = embedded.shape[0]
        half_bandwidth = self.half_bandwidth if self.half_bandwidth is not None else T
        x = Tensor(embedded.astype(np.float32))
        for layer in self.layers:
            normed = rmsnorm_tensor(x, layer.input_ln, self.rms_eps)
            q = layer.q_proj.forward(normed)
            k = layer.k_proj.forward(normed)
            v = layer.v_proj.forward(normed)
            attn = banded_attention(q, k, v, half_bandwidth=half_bandwidth, num_cpus=self.num_cpus, causal=True)
            attn = layer.o_proj.forward(attn)
            x = x + attn
            normed2 = rmsnorm_tensor(x, layer.post_ln, self.rms_eps)
            gate = layer.gate_proj.forward(normed2)
            up = layer.up_proj.forward(normed2)
            mlp_out = layer.down_proj.forward(silu(gate) * up)
            x = x + mlp_out
        return self.lm_head.forward(x)


# ═══════════════════════════════════════════════════════════════════════════
#  ToyTileRecurrence
# ═══════════════════════════════════════════════════════════════════════════


class ToyTileRecurrence:
    """One shared tile network (DenseTensorLinear q/k/v/o/gate/up/down),
    gaussian_attention across tiles, additive energy-free gated residual.
    Single head, no positional encoding.
    `embed_width` (E) matches the real token-embedding width; the internal
    recurrent `state_width` = E * `column_neurons` (C) is deliberately
    WIDER, read out via column-mean pooling back to `embed_width`.
    See docs/research/toy_recall_models.rst:tile_recurrence_state_width_column_mean."""

    def __init__(
        self,
        vocab_size: int,
        embed_width: int,
        column_neurons: int,
        mlp_hidden: int,
        num_tiles: int,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
    ):
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        state_width = self.state_width
        self.q_proj = DenseTensorLinear(state_width, state_width)
        self.k_proj = DenseTensorLinear(state_width, state_width)
        self.v_proj = DenseTensorLinear(state_width, state_width)
        self.o_proj = DenseTensorLinear(state_width, state_width)
        self.gate_proj = DenseTensorLinear(state_width, mlp_hidden)
        self.up_proj = DenseTensorLinear(state_width, mlp_hidden)
        self.down_proj = DenseTensorLinear(mlp_hidden, state_width)
        self.input_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.post_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.lm_head = DenseTensorLinear(embed_width, vocab_size)
        self.centers = Tensor(np.array([i + 0.5 for i in range(num_tiles)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(num_tiles, dtype=np.float32))

    def parameters(self) -> list[Tensor]:
        params = [self.input_ln, self.post_ln, self.centers, self.log_sigmas]
        for layer in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
            self.lm_head,
        ):
            params += layer.parameters()
        return params

    def step(self, x_window: np.ndarray, M_prev: np.ndarray) -> tuple[np.ndarray, Tensor]:
        """One recurrence tick. x_window, M_prev: [num_tiles, state_width]
        numpy, DETACHED (no BPTT) -- both already widened to state_width by
        the caller. Returns (M_new numpy [num_tiles, state_width], logits
        Tensor [num_tiles, vocab_size] -- one row per tile's own
        column-mean-pooled next-token prediction; only the LAST row is
        ever actually trained by the caller).
        Q/K/V draw from x_window BLENDED with M_prev.
        See docs/research/toy_recall_models.rst:tile_recurrence_step_qkv_blend."""
        x_normed = rmsnorm_tensor(Tensor(x_window.astype(np.float32)), self.input_ln, self.rms_eps)
        m_normed = rmsnorm_tensor(Tensor(M_prev.astype(np.float32)), self.input_ln, self.rms_eps)
        qkv_source = x_normed + m_normed
        q = self.q_proj.forward(qkv_source)
        k = self.k_proj.forward(qkv_source)
        v = self.v_proj.forward(qkv_source)
        sigmas = exp(self.log_sigmas)
        attn = gaussian_attention(q, k, v, self.centers, sigmas, num_cpus=self.num_cpus, causal=False)
        attn = self.o_proj.forward(attn)

        M_new_t = Tensor(M_prev.astype(np.float32)) + attn
        normed2 = rmsnorm_tensor(M_new_t, self.post_ln, self.rms_eps)
        gate = self.gate_proj.forward(normed2)
        up = self.up_proj.forward(normed2)
        mlp_out = self.down_proj.forward(silu(gate) * up)
        M_new_t = M_new_t + mlp_out

        pooled = M_new_t.reshape((self.num_tiles, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)  # [num_tiles, embed_width]
        logits = self.lm_head.forward(pooled)  # [num_tiles, vocab_size]
        return M_new_t.data, logits
