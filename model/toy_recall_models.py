"""
sili_peridot/model/toy_recall_models.py
─────────────────────────────────────────
Small, TRAINING-oriented (not frozen-inference) models for validating
the tile-recurrence architecture on the synthetic induction-recall task
(model/toy_recall_task.py) -- see the approved plan
(fuzzy-plotting-starlight.md) for the full design rationale and the
toy-track-only simplifications (fixed embeddings, no positional
encoding, single head) this module deliberately makes.

Built from DenseTensorLinear (plain fp32 sili.tensor matmul) trained
via a real AdamOptimizer, NOT sili.sparse_rnn.DISLDOLayer -- per direct
decision, after two isolation controls (scripts/torch_mqar_control.py,
scripts/fp32_handrolled_control.py) confirmed the earlier stuck-at
-chance training result was caused by this session's own hand-rolled
optimizer (plain per-node-clipped SGD, no momentum) diverging, NOT by
FP4 quantization -- fp32 with that SAME hand-rolled optimizer diverged
identically. Fixing this for DISLDOLayer's own inline C++ weight
update would need real new work in sili__new (disldo_backward); fixing
it for plain Tensor leaves is pure Python (this file's own
AdamOptimizer) and directly answers this track's actual question (does
tile-recurrence learn genuine recall), so per direct decision that's
the path taken here. DISLDOLayer/FP4's own training dynamics remain a
distinct, not-revisited-here concern (already partially validated
elsewhere per direct feedback -- importance-driven training behaves
similarly to other optimizers).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from sili.tensor import (
    Tensor, banded_attention, gaussian_attention, exp, log, reduce_sum, silu, gather, _topo_sort,
)


class DenseTensorLinear:
    """Plain fp32 Tensor-graph linear layer (matmul-based, no
    quantization) -- trains via AdamOptimizer.step(), an ordinary
    Tensor leaf like RMSNorm weights/centers/log_sigmas, not an
    inline-self-updating primitive."""

    def __init__(self, in_features: int, out_features: int, scale: float = 0.1):
        self.weight = Tensor(
            (np.random.randn(in_features, out_features) * scale).astype(np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight

    def parameters(self) -> List[Tensor]:
        return [self.weight]


def rmsnorm_tensor(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """x: [T, hidden] Tensor. Same formula as sili_block.rmsnorm, but
    built from Tensor ops so gradient flows through it (needed here,
    unlike sili_block's frozen-inference plain-numpy version)."""
    hidden = x.data.shape[-1]
    mean_sq = reduce_sum(x * x, axis=-1) * (1.0 / hidden)   # [T]
    mean_sq = mean_sq.reshape((x.data.shape[0], 1))          # [T, 1]
    rrms = (mean_sq + eps) ** -0.5                           # [T, 1]
    return (x * rrms) * weight


def cross_entropy_sum(logits: Tensor, row_target_pairs: List[Tuple[int, int]]) -> Tensor:
    """logits: [N, vocab_size] Tensor. row_target_pairs: [(row, target_
    token_id), ...] -- returns the SUM of softmax cross-entropy loss
    over each pair (caller divides by len(...) for a mean).

    `Tensor` has no `__getitem__`/slicing, so per-row loss can't be
    computed by indexing a row out directly -- built instead from
    reduce_sum(axis=-1)/exp/log (whole-tensor ops) plus `gather`'s
    FLAT indexing (`out[i] = a.flat[indices[i]]`) for both the
    per-row log-sum-exp lookup and the (row, target) logit lookup --
    handles either a single (row, target) pair or many at once (e.g.
    every tile's own "column" prediction target in one call) with no
    row-slicing needed anywhere.

    Standard max-subtraction numerical stability trick IS needed here
    (an earlier version of this function skipped it, assuming toy-scale
    logits would stay small -- wrong, confirmed directly: raw exp()
    overflowed after a few dozen real training steps, since logits
    grow as the model gets more confident, toy scale or not). The
    per-row max is computed from `logits.data` directly (plain numpy,
    detached) rather than a Tensor op -- subtracting a constant shift
    from logits before the exp/sum/log chain doesn't change the loss's
    gradient w.r.t. the ORIGINAL logits at all, so nothing needs to
    backprop through the max itself; using a `reduce_max` Tensor op
    (which doesn't exist in sili.tensor) would only be necessary if
    gradient had to flow through the max, which it doesn't."""
    vocab_size = logits.data.shape[-1]
    row_max = logits.data.max(axis=-1, keepdims=True).astype(np.float32)  # [N,1], detached
    shifted = logits + Tensor(-row_max)                                  # [N,vocab], stable
    log_sum_exp = log(reduce_sum(exp(shifted), axis=-1)) + Tensor(row_max.reshape(-1))  # [N]
    rows = [r for r, _t in row_target_pairs]
    log_sum_exp_rows = gather(log_sum_exp, rows)                  # [len(pairs)]
    flat_target_idx = [r * vocab_size + t for r, t in row_target_pairs]
    target_logits = gather(logits, flat_target_idx)                # [len(pairs)]
    return reduce_sum(log_sum_exp_rows - target_logits)


def predicted_token(logits: Tensor, row: int) -> int:
    """Inference-time-only readout (no gradient needed/possible through
    argmax) -- reads .data directly, doesn't need Tensor slicing."""
    return int(np.argmax(logits.data[row]))


def apply_gradient_step(params: List[Tensor], lr: float) -> None:
    """Plain SGD step + zero_grad for ordinary Tensor leaves. Kept for
    tests/comparison -- superseded by AdamOptimizer for real training
    (see module docstring: plain per-node-clipped SGD, no momentum, was
    confirmed via two isolation controls to diverge on this task,
    independent of precision).

    Leaves whose .grad is still None (e.g. a tile whose column target
    didn't apply this specific tick) are skipped, not zeroed against a
    nonexistent gradient."""
    for p in params:
        if p.grad is not None:
            p.data = p.data - lr * np.asarray(p.grad, dtype=np.float32)
            p.zero_grad()


class AdamOptimizer:
    """Standard Adam (Kingma & Ba, 2014) for plain Tensor leaves --
    per-parameter first/second moment estimates with bias correction.
    Added per direct decision after two isolation controls
    (scripts/torch_mqar_control.py, scripts/fp32_handrolled_control.py)
    confirmed this session's earlier hand-rolled plain-SGD-with-clipping
    optimizer (apply_gradient_step) was the actual cause of every
    stuck-at-chance/diverging toy training result -- NOT FP4
    quantization, NOT the architecture. A full-precision + Adam control
    converged easily and fast on the identical task; full precision
    with the OLD plain-SGD optimizer diverged identically to the FP4
    version. Momentum/adaptive per-parameter scaling was the missing
    piece, not precision.

    Keyed by `id(param)` (Tensor doesn't define __hash__/__eq__, so
    default object-identity hashing is exactly what's wanted here --
    each distinct Tensor leaf gets its own independent moment state)."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params: List[Tensor], lr: float) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
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
    """Gradient-clipped replacement for `loss.backward()` -- clips the
    L2 norm of EVERY node's incoming gradient (not just the final
    parameter gradients) to `max_grad_norm`, right before that node's
    own `_backward()` fires. Still used alongside AdamOptimizer
    (clip+Adam together is standard practice, e.g. nanoGPT's own
    convention) -- clipping alone was never sufficient (see
    AdamOptimizer's own docstring), but it's still good practice
    combined with real momentum.

    `Tensor.backward()` (`sili/tensor.py`) is just `for node in
    reversed(_topo_sort(self)): node._backward()` -- replicated here
    with a clip inserted in the loop, so every node's `.grad` (already
    fully accumulated from all its consumers by the time its own turn
    comes, per topological order) is bounded before it propagates
    further."""
    if loss.grad is None:
        loss.grad = np.ones_like(loss.data)
    for node in reversed(_topo_sort(loss)):
        if node.grad is not None:
            g = np.asarray(node.grad, dtype=np.float32)
            norm = float(np.sqrt(np.sum(g.astype(np.float64) ** 2)))
            if norm > max_grad_norm and norm > 0:
                node.grad = (g * (max_grad_norm / norm)).astype(np.float32)
        node._backward()


def lr_schedule(step: int, total_steps: int, peak_lr: float,
                 warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup + cosine decay, matching nanoGPT's own convention
    (widely-used, well-tested defaults for small transformer training --
    looked up rather than guessed, per direct decision)."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine))


def clip_grad_norm_(params: List[Tensor], max_norm: float) -> float:
    """Textbook GLOBAL gradient-norm clipping -- the total L2 norm
    ACROSS ALL of `params`' gradients combined is capped to `max_norm`
    (matching torch.nn.utils.clip_grad_norm_ exactly, including the
    control script that used it: scripts/torch_mqar_control.py).

    `backward_with_grad_clip`'s per-NODE clipping exists specifically
    because `DISLDOLayer`'s own weights self-update INLINE during
    `backward()`, so a true global-norm measurement isn't available
    before an update already happened (see its own docstring). That
    constraint doesn't apply to this module's models anymore (per
    direct decision -- they're built from `DenseTensorLinear` now, see
    module docstring): NOTHING updates until `optimizer.step()` is
    called explicitly, so the real global norm can be measured first,
    same as any standard training loop. Confirmed this distinction
    actually matters, not just theoretically: per-node clipping still
    let AdamOptimizer diverge on the real MQAR task (every one of many
    parameter tensors independently allowed up to norm `max_norm` is a
    much LARGER aggregate step than one norm-`max_norm` budget shared
    across all of them) -- this is the correct, stronger clip to use
    for models built from DenseTensorLinear; call after `loss.backward()`
    (plain, ordinary -- not `backward_with_grad_clip`) and before
    `optimizer.step()`."""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float(np.sum(np.asarray(p.grad, dtype=np.float64) ** 2))
    total_norm = total_sq ** 0.5
    if not np.isfinite(total_norm):
        # `total_norm > max_norm` and `total_norm > 0` are BOTH False when
        # total_norm is NaN (IEEE 754) -- a NaN/Inf gradient would silently
        # skip the clip below and sail straight into opt.step(), permanently
        # poisoning Adam's m/v moving averages (every future step also NaN
        # after that). Confirmed as the real, direct cause of dense
        # connectivity's permanent NaN divergence via the C++-side analog
        # of this exact bug (sili__new's ScalePolicy::update, see its own
        # docstring; JOURNAL.md 2026-08-10). Zero the gradient instead --
        # skips this step's update rather than corrupting all future ones.
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

    def parameters(self) -> List[Tensor]:
        params = [self.input_ln, self.post_ln]
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj,
                      self.gate_proj, self.up_proj, self.down_proj):
            params += layer.parameters()
        return params


class ToySmallTransformer:
    """Stacked causal dense transformer -- each layer has its OWN
    distinct weights (real depth-stacking, unlike tile-recurrence's
    single shared tile network). Single-head attention (see module
    docstring's simplifications), no positional encoding.

    `half_bandwidth`: defaults to unlimited (full causal visibility --
    the default matches every existing call site's behavior). Set to
    an int `W` to give this model a GENUINELY bounded context window
    -- structurally unable to see more than `W` positions back,
    regardless of training. Used as the real "standard LLM" stand-in
    for the out-of-context benchmark suite (see
    scripts/train_toy_beyond_context_comparison.py) -- tile-recurrence's
    own `num_tiles` already plays this same role, no equivalent knob
    needed there."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int,
                 n_layers: int, num_cpus: int = 2, rms_eps: float = 1e-6,
                 half_bandwidth: Optional[int] = None):
        self.hidden = hidden
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.half_bandwidth = half_bandwidth
        self.layers = [_ToyTransformerLayer(hidden, mlp_hidden) for _ in range(n_layers)]
        self.lm_head = DenseTensorLinear(hidden, vocab_size)

    def parameters(self) -> List[Tensor]:
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
            attn = banded_attention(q, k, v, half_bandwidth=half_bandwidth,
                                    num_cpus=self.num_cpus, causal=True)
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
    gaussian_attention across tiles, additive energy-free gated residual
    (toy scale -- see module docstring; no EnergyDynamics here, plain
    residual add is enough to test the core retrieval mechanism without
    pulling in another moving part). Single head, no positional
    encoding (see module docstring's simplifications).

    `embed_width` (E) matches the real token-embedding width. The
    internal recurrent `state_width` = E * `column_neurons` (C) is
    deliberately WIDER -- solves "how do you backprop a prediction
    error into a state much wider than the output" (see the approved
    plan, fuzzy-plotting-starlight.md, for the full worked example:
    selecting a fixed subset starves the rest of gradient, summing a
    column forces the state's own values small to avoid blowup,
    AVERAGING a column works and stays in the same natural magnitude
    range). `lm_head` stays fixed at `embed_width` -- standing in for
    the real system's pretrained, fixed-width output head, which is
    exactly why the width-reduction has to be the parameter-free
    column-mean, not a new learned down-projection."""

    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int,
                 mlp_hidden: int, num_tiles: int, num_cpus: int = 2, rms_eps: float = 1e-6):
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

    def parameters(self) -> List[Tensor]:
        params = [self.input_ln, self.post_ln, self.centers, self.log_sigmas]
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj,
                      self.gate_proj, self.up_proj, self.down_proj, self.lm_head):
            params += layer.parameters()
        return params

    def step(self, x_window: np.ndarray, M_prev: np.ndarray) -> Tuple[np.ndarray, Tensor]:
        """One recurrence tick. x_window, M_prev: [num_tiles,
        state_width] numpy, DETACHED (no BPTT, matching
        tile_recurrence.py's own design) -- both already widened to
        state_width by the caller (see _build_tile_window in
        scripts/train_toy_recall_comparison.py), so nothing here needs
        to handle mixed widths. Returns (M_new numpy [num_tiles,
        state_width], logits Tensor [num_tiles, vocab_size] -- one row
        per tile's own column-mean-pooled next-token prediction; only
        the LAST row, the real current tick position, is ever actually
        trained by the caller).

        Q/K/V draw from x_window BLENDED with M_prev (per direct
        correction -- see [[feedback_attention_needs_combined_input_state]]:
        input-only attention forces anything relating fresh input to
        carried state through an artificial "write to state, wait a
        tick, then attend" detour; attention must be able to relate
        input and state directly, in one step)."""
        x_normed = rmsnorm_tensor(Tensor(x_window.astype(np.float32)), self.input_ln, self.rms_eps)
        m_normed = rmsnorm_tensor(Tensor(M_prev.astype(np.float32)), self.input_ln, self.rms_eps)
        qkv_source = x_normed + m_normed
        q = self.q_proj.forward(qkv_source)
        k = self.k_proj.forward(qkv_source)
        v = self.v_proj.forward(qkv_source)
        sigmas = exp(self.log_sigmas)
        attn = gaussian_attention(q, k, v, self.centers, sigmas,
                                  num_cpus=self.num_cpus, causal=False)
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
