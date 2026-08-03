"""
sili_peridot/model/toy_recall_models.py
─────────────────────────────────────────
Small, TRAINING-oriented (not frozen-inference) models for validating
the tile-recurrence architecture on the synthetic induction-recall task
(model/toy_recall_task.py) -- see the approved plan
(fuzzy-plotting-starlight.md) for the full design rationale and the
toy-track-only simplifications (fixed embeddings, no positional
encoding, single head) this module deliberately makes.

Unlike model/sili_block.py's apply_fold_step / model/tile_recurrence.py's
apply_tile_step (built around FROZEN pretrained weights -- `_forward`
calls SparseLinearLayer.forward_dense(x, learning_rate=0.0) directly on
raw numpy, discarding the Tensor graph), everything here routes through
sili.sparse_rnn.DISLDOLayer -- a Tensor-graph-integrated wrapper whose
forward() returns a real Tensor node and whose backward calls
backward_dense(dy, learning_rate), which both computes dx (keeps
backprop flowing) AND applies DISLDOLayer's own inline weight update
using the real downstream gradient. A single loss.backward() call at
the end of a forward pass drives every weight's own local update.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from sili.sparse_rnn import DISLDOLayer
from sili.tensor import (
    Tensor, banded_attention, gaussian_attention, exp, log, reduce_sum, silu, gather, _topo_sort,
)


class DenseTensorLinear:
    """Plain fp32 Tensor-graph linear layer (matmul-based, NO
    quantization, NO DISLDOLayer) -- trainable via apply_gradient_step,
    the SAME hand-rolled SGD+clip convention as every other leaf in
    this module (RMSNorm weights, centers, log_sigmas), NOT Adam.

    Exists specifically to isolate FP4-quantization-via-DISLDOLayer
    from this session's own hand-rolled optimizer as the cause of the
    toy training experiments' stuck-at-chance result: a full-precision
    + Adam control (scripts/torch_mqar_control.py) converged easily and
    fast on the identical architecture/task, proving the ARCHITECTURE
    wasn't the problem -- but that control changed BOTH precision and
    optimizer at once. This class changes ONLY precision (still uses
    this module's own optimizer), so a model built from it isolates
    the remaining variable directly."""

    def __init__(self, in_features: int, out_features: int, scale: float = 0.1):
        self.weight = Tensor(
            (np.random.randn(in_features, out_features) * scale).astype(np.float32))

    def forward(self, x: Tensor, learning_rate: float = 0.0) -> Tensor:
        """learning_rate accepted (unused) only so this drops into the
        SAME call signature DISLDOLayer.forward uses -- this class's
        own weight trains via apply_gradient_step(self.parameters(),
        lr), called by the caller's training loop like any other leaf,
        not via an inline per-call update."""
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
    """Plain SGD step + zero_grad for ordinary Tensor LEAVES (RMSNorm
    weights, centers, log_sigmas) -- NOT for DISLDOLayer's own internal
    weights, which already self-update inline during backward_dense
    (see module docstring). Call once per training step, after
    .backward(). Leaves whose .grad is still None (e.g. a tile whose
    column target didn't apply this specific tick) are skipped, not
    zeroed against a nonexistent gradient.

    Forgetting this (an actual bug hit while smoke-testing this module)
    is silent and severe, not just "doesn't learn": these leaves are
    reused across every training step, so _acc's plain accumulation
    keeps ADDING each step's gradient onto whatever was already there
    forever -- unbounded grad growth, then a divergent loss within a
    few dozen steps, not a slow-to-converge one."""
    for p in params:
        if p.grad is not None:
            p.data = p.data - lr * np.asarray(p.grad, dtype=np.float32)
            p.zero_grad()


def backward_with_grad_clip(loss: Tensor, max_grad_norm: float) -> None:
    """Gradient-clipped replacement for `loss.backward()` -- clips the
    L2 norm of EVERY node's incoming gradient (not just the final
    parameter gradients) to `max_grad_norm`, right before that node's
    own `_backward()` fires.

    Textbook gradient clipping computes the TOTAL norm across all
    parameter gradients FIRST, then rescales once -- not possible here
    in one pass: `DISLDOLayer`'s own weights self-update INLINE, during
    the SAME `_backward()` call that computes their gradient (see
    module docstring), so by the time a global norm could be measured,
    the (unclipped) update has already happened. A true two-pass
    version (a dry run at `learning_rate=0` to measure the norm, then a
    real pass with a correctly pre-scaled seed) would double the
    forward+backward cost of every single training step -- not worth
    it here.

    Per-NODE clipping is the single-pass-compatible alternative that
    still directly bounds what any individual weight update can see:
    `Tensor.backward()` (`sili/tensor.py`) is just `for node in
    reversed(_topo_sort(self)): node._backward()` -- replicated here
    with a clip inserted in the loop, so every node's `.grad` (already
    fully accumulated from all its consumers by the time its own turn
    comes, per topological order) is bounded before it can either
    propagate further OR drive a `DISLDOLayer`'s inline update. This is
    what actually fixed the training instability (overflow warnings,
    divergent loss) seen during this session's own unclipped runs."""
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


# ═══════════════════════════════════════════════════════════════════════════
#  ToySmallTransformer -- the "pre-converted transformer" baseline
# ═══════════════════════════════════════════════════════════════════════════

class _ToyTransformerLayer:
    def __init__(self, hidden: int, mlp_hidden: int, max_weights: int, num_cpus: int):
        self.q_proj = DISLDOLayer(hidden, hidden, max_weights, num_cpus)
        self.k_proj = DISLDOLayer(hidden, hidden, max_weights, num_cpus)
        self.v_proj = DISLDOLayer(hidden, hidden, max_weights, num_cpus)
        self.o_proj = DISLDOLayer(hidden, hidden, max_weights, num_cpus)
        self.gate_proj = DISLDOLayer(hidden, mlp_hidden, max_weights, num_cpus)
        self.up_proj = DISLDOLayer(hidden, mlp_hidden, max_weights, num_cpus)
        self.down_proj = DISLDOLayer(mlp_hidden, hidden, max_weights, num_cpus)
        self.input_ln = Tensor(np.ones(hidden, dtype=np.float32))
        self.post_ln = Tensor(np.ones(hidden, dtype=np.float32))


class ToySmallTransformer:
    """Stacked causal dense transformer -- each layer has its OWN
    distinct weights (real depth-stacking, unlike tile-recurrence's
    single shared tile network). Single-head attention (see module
    docstring's simplifications), no positional encoding."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int,
                 n_layers: int, max_weights: int, num_cpus: int = 2,
                 rms_eps: float = 1e-6):
        self.hidden = hidden
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.layers = [_ToyTransformerLayer(hidden, mlp_hidden, max_weights, num_cpus)
                        for _ in range(n_layers)]
        self.lm_head = DISLDOLayer(hidden, vocab_size, max_weights, num_cpus)

    def parameters(self) -> List[Tensor]:
        """Plain Tensor leaves needing apply_gradient_step -- every
        layer's own RMSNorm weight. DISLDOLayer weights aren't included
        (self-update inline, see module docstring)."""
        params = []
        for layer in self.layers:
            params += [layer.input_ln, layer.post_ln]
        return params

    def forward(self, embedded: np.ndarray, learning_rate: float) -> Tensor:
        """embedded: [T, hidden] numpy (fixed embedding lookups).
        Returns logits [T, vocab_size] Tensor."""
        T = embedded.shape[0]
        x = Tensor(embedded.astype(np.float32))
        for layer in self.layers:
            normed = rmsnorm_tensor(x, layer.input_ln, self.rms_eps)
            q = layer.q_proj.forward(normed, learning_rate)
            k = layer.k_proj.forward(normed, learning_rate)
            v = layer.v_proj.forward(normed, learning_rate)
            attn = banded_attention(q, k, v, half_bandwidth=T,
                                    num_cpus=self.num_cpus, causal=True)
            attn = layer.o_proj.forward(attn, learning_rate)
            x = x + attn
            normed2 = rmsnorm_tensor(x, layer.post_ln, self.rms_eps)
            gate = layer.gate_proj.forward(normed2, learning_rate)
            up = layer.up_proj.forward(normed2, learning_rate)
            mlp_out = layer.down_proj.forward(silu(gate) * up, learning_rate)
            x = x + mlp_out
        return self.lm_head.forward(x, learning_rate)


# ═══════════════════════════════════════════════════════════════════════════
#  ToyTileRecurrence
# ═══════════════════════════════════════════════════════════════════════════

class ToyTileRecurrence:
    """One shared tile network (DISLDOLayer q/k/v/o/gate/up/down),
    gaussian_attention across tiles, additive energy-free gated residual
    (toy scale -- see module docstring; no EnergyDynamics here, plain
    residual add is enough to test the core retrieval mechanism without
    pulling in another moving part). Single head, no positional
    encoding (see module docstring's simplifications).

    Keeps Kimi's staggered per-tile "column" prediction (the direct
    tile-shaped descendant of this project's own A3/A4 column-averaging
    work): tile j's own state, projected through the SAME shared
    lm_head, is trained to predict the token tile j+1 currently holds
    this same tick; the last tile predicts the genuinely novel next
    token."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int,
                 num_tiles: int, max_weights: int, num_cpus: int = 2,
                 rms_eps: float = 1e-6):
        self.hidden = hidden
        self.num_tiles = num_tiles
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.q_proj = DISLDOLayer(hidden, hidden, max_weights, num_cpus)
        self.k_proj = DISLDOLayer(hidden, hidden, max_weights, num_cpus)
        self.v_proj = DISLDOLayer(hidden, hidden, max_weights, num_cpus)
        self.o_proj = DISLDOLayer(hidden, hidden, max_weights, num_cpus)
        self.gate_proj = DISLDOLayer(hidden, mlp_hidden, max_weights, num_cpus)
        self.up_proj = DISLDOLayer(hidden, mlp_hidden, max_weights, num_cpus)
        self.down_proj = DISLDOLayer(mlp_hidden, hidden, max_weights, num_cpus)
        self.input_ln = Tensor(np.ones(hidden, dtype=np.float32))
        self.post_ln = Tensor(np.ones(hidden, dtype=np.float32))
        self.lm_head = DISLDOLayer(hidden, vocab_size, max_weights, num_cpus)
        self.centers = Tensor(np.array([i + 0.5 for i in range(num_tiles)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(num_tiles, dtype=np.float32))

    def parameters(self) -> List[Tensor]:
        """Plain Tensor leaves needing apply_gradient_step -- RMSNorm
        weights plus centers/log_sigmas. DISLDOLayer weights aren't
        included (self-update inline, see module docstring)."""
        return [self.input_ln, self.post_ln, self.centers, self.log_sigmas]

    def step(self, x_window: np.ndarray, M_prev: np.ndarray,
             learning_rate: float) -> Tuple[np.ndarray, Tensor]:
        """One recurrence tick. x_window: [num_tiles, hidden] numpy
        (see toy_recall_task/build_tile_window-style sliding-window
        injection -- built by the caller). M_prev: [num_tiles, hidden]
        numpy, DETACHED (no BPTT, matching tile_recurrence.py's own
        design). Returns (M_new numpy, logits Tensor [num_tiles,
        vocab_size] -- one row per tile's own "column" prediction,
        last row is the genuinely novel next-token prediction).

        Q/K/V all draw from normed(x_window) + normed(M_prev) (summed,
        matching apply_tile_step's own established convention) -- NOT
        x_window alone. Without M_prev's own contribution here,
        attention could only look across the current tick's fresh
        window, never genuinely retrieve older content carried in
        M_prev -- exactly the capability this whole architecture
        exists to test, so it can't be simplified away even in this
        toy version (unlike the interleaved-key-space / asymmetric-V
        detail from Phase 2.7b's production design, which IS
        simplified away here -- a single blended source for Q/K/V is
        enough to keep memory genuinely attend-able without that
        extra machinery)."""
        x_normed = rmsnorm_tensor(Tensor(x_window.astype(np.float32)), self.input_ln, self.rms_eps)
        m_normed = rmsnorm_tensor(Tensor(M_prev.astype(np.float32)), self.input_ln, self.rms_eps)
        qkv_source = x_normed + m_normed
        q = self.q_proj.forward(qkv_source, learning_rate)
        k = self.k_proj.forward(qkv_source, learning_rate)
        v = self.v_proj.forward(qkv_source, learning_rate)
        sigmas = exp(self.log_sigmas)
        attn = gaussian_attention(q, k, v, self.centers, sigmas,
                                  num_cpus=self.num_cpus, causal=False)
        attn = self.o_proj.forward(attn, learning_rate)

        M_new_t = Tensor(M_prev.astype(np.float32)) + attn
        normed2 = rmsnorm_tensor(M_new_t, self.post_ln, self.rms_eps)
        gate = self.gate_proj.forward(normed2, learning_rate)
        up = self.up_proj.forward(normed2, learning_rate)
        mlp_out = self.down_proj.forward(silu(gate) * up, learning_rate)
        M_new_t = M_new_t + mlp_out

        logits = self.lm_head.forward(M_new_t, learning_rate)  # [num_tiles, vocab_size]
        return M_new_t.data, logits
