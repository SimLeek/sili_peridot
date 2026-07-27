"""
sili_peridot/model/sili_block.py
─────────────────────────────────
B6: attention assembly. Runs MiniCPM5's 24-layer stack as a single folded
block over a 24-step fold-depth recurrence (state=0; for step: out=block(x
+state); state+=out -- see RNNFoldedBlock.forward's docstring in
sili__new's rnn_fold.py), computing every projection and the attention
itself through real sili ops -- no torch anywhere in this module.

Each fold step gets its OWN small real sili SparseLinearLayer per suffix
(q/k/v/o/gate/up/down), built by slicing that step's weights out of the
pre-quantization stacked CSR (FoldedBlockDescriptor.fold_weight_csr) and
FP4-quantizing them independently -- not by reusing B5/B5a's
stacked/rank-1-quantized FoldedLayer, which shares one scale scheme across
all 24 layers for storage efficiency. Per-step independent quantization
matches what running each original layer separately would actually see.

RMSNorm/RoPE weights are never part of the 7 folded suffixes (B3/B5 never
prune or quantize them), so they're read directly from sparse_state as
plain float32 vectors, not built into any SparseLinearLayer.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from sili import _cpu
from sili.tensor import Tensor, banded_attention

from .config import MiniCPM5Config
from .fold import SUFFIXES, fold_suffix

_FP4_MAX = 6.0
_ATTN_SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                  ".self_attn.v_proj.weight", ".self_attn.o_proj.weight")
_MLP_SUFFIXES  = (".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")


def _build_step_layer(csr_slice_out_in: torch.Tensor, num_cpus: int):
    """
    csr_slice_out_in: torch sparse CSR [out_dim, in_dim], one fold step's
    weight slice (FoldedBlockDescriptor.fold_weight_csr's own layout).

    Real FP4-quantized SparseLinearLayer, per-row (per-input-feature) value
    scale -- same scheme as FoldedLayer.from_descriptor's "per_row" mode
    (see sparse_rnn.py), just applied to one step's own [out_dim, in_dim]
    slice instead of the full stacked matrix.
    """
    csr_t = csr_slice_out_in.t().to_sparse_csr()   # [in_dim, out_dim]
    n_in, n_out = int(csr_t.shape[0]), int(csr_t.shape[1])
    nnz = int(csr_t.values().numel())
    layer = _cpu.SparseLinearLayer(n_in, n_out, int(nnz * 1.3) + 64, num_cpus)

    ptrs = csr_t.crow_indices().numpy().astype(np.int32)
    idx  = csr_t.col_indices().numpy().astype(np.int32)
    vals = csr_t.values().float().numpy().copy()

    row_scales = np.ones(n_in, dtype=np.float32)
    for r in range(n_in):
        start, end = int(ptrs[r]), int(ptrs[r + 1])
        if end > start:
            max_abs = float(np.abs(vals[start:end]).max())
            if max_abs > 0.0:
                row_scales[r] = max_abs / _FP4_MAX
                vals[start:end] /= row_scales[r]

    layer.load_weights(ptrs, idx, vals)
    for r in range(n_in):
        if row_scales[r] != 1.0:
            layer.set_value_scale_raw(r, row_scales[r])
    return layer


def build_step_layers(
    sparse_state: Dict[str, dict],
    cfg: MiniCPM5Config,
    prefix: str = "model.layers.",
    band_half_width_override=None,
    num_cpus: int = 4,
) -> Tuple[List[Dict[str, object]], List[np.ndarray], List[np.ndarray]]:
    """
    Build every fold step's real sili layers (one suffix-keyed dict per
    step) plus that step's own RMSNorm weight vectors, streaming one
    suffix at a time (mirrors fold.build_folded_layers_streaming's
    discipline) -- MUTATES sparse_state, popping each suffix's per-layer
    tensors immediately after slicing all 24 steps out of it, and popping
    each layer's two layernorm vectors directly (they aren't part of any
    suffix descriptor).

    Returns (step_layers, input_ln_weights, post_attn_ln_weights):
      step_layers[i]        -- {suffix: SparseLinearLayer} for fold step i
      input_ln_weights[i]   -- float32 [hidden_size], layer i's input_layernorm
      post_attn_ln_weights[i] -- float32 [hidden_size], layer i's post_attention_layernorm
    """
    n = cfg.num_hidden_layers
    step_layers: List[Dict[str, object]] = [dict() for _ in range(n)]

    for suffix in SUFFIXES:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        for i in range(n):
            del sparse_state[f"{prefix}{i}{suffix}"]
        for i in range(n):
            csr_slice = desc.fold_weight_csr(suffix, i)
            step_layers[i][suffix] = _build_step_layer(csr_slice, num_cpus)
        del desc

    input_ln = []
    post_ln  = []
    for i in range(n):
        input_ln.append(sparse_state.pop(f"{prefix}{i}.input_layernorm.weight")["raw"]
                        .float().numpy().copy())
        post_ln.append(sparse_state.pop(f"{prefix}{i}.post_attention_layernorm.weight")["raw"]
                       .float().numpy().copy())
    return step_layers, input_ln, post_ln


# ── Elementwise math (RMSNorm / RoPE / SiLU) ─────────────────────────────────
# No sparsity, no learned structure beyond a per-channel scale vector --
# sili doesn't claim these as ops, plain numpy is the right tool.

def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """x: [T, hidden]. Matches sili__new's model_reconstruct.py _LlamaRMSNorm."""
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(var + eps))).astype(np.float32) * weight


def rope_cos_sin(seq_len: int, head_dim: int, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)                        # [T, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)         # [T, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rotary(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: [T, head_dim], cos/sin: [T, head_dim]."""
    return x * cos + _rotate_half(x) * sin


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# ── One fold step: RMSNorm -> GQA causal attention (RoPE) -> RMSNorm -> SwiGLU MLP

def apply_fold_step(
    x: np.ndarray,               # [T, hidden] = original input + accumulated state
    layers: Dict[str, object],   # this step's {suffix: SparseLinearLayer}
    input_ln_weight: np.ndarray,
    post_attn_ln_weight: np.ndarray,
    cfg: MiniCPM5Config,
    cos: np.ndarray, sin: np.ndarray,   # from rope_cos_sin(T, head_dim, rope_theta)
    half_bandwidth: int,
    num_cpus: int = 4,
) -> np.ndarray:
    """Returns this step's own output [T, hidden] (the recurrence's caller
    accumulates it into state)."""
    T = x.shape[0]
    n_heads, n_kv_heads, head_dim = (cfg.num_attention_heads,
                                     cfg.num_key_value_heads, cfg.head_dim)
    groups = n_heads // n_kv_heads

    normed = rmsnorm(x, input_ln_weight, cfg.rms_norm_eps)

    q = layers[".self_attn.q_proj.weight"].forward_dense(normed, learning_rate=0.0)
    k = layers[".self_attn.k_proj.weight"].forward_dense(normed, learning_rate=0.0)
    v = layers[".self_attn.v_proj.weight"].forward_dense(normed, learning_rate=0.0)

    q = q.reshape(T, n_heads, head_dim)
    k = k.reshape(T, n_kv_heads, head_dim)
    v = v.reshape(T, n_kv_heads, head_dim)

    attn_out = np.empty((T, n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // groups
        qh = Tensor(apply_rotary(q[:, h, :], cos, sin))
        kh = Tensor(apply_rotary(k[:, kv_h, :], cos, sin))
        vh = Tensor(np.ascontiguousarray(v[:, kv_h, :]))
        out_h = banded_attention(qh, kh, vh, half_bandwidth=half_bandwidth,
                                 num_cpus=num_cpus, causal=True)
        attn_out[:, h, :] = out_h.data

    attn_out = attn_out.reshape(T, n_heads * head_dim)
    attn_out = layers[".self_attn.o_proj.weight"].forward_dense(attn_out, learning_rate=0.0)

    x = x + attn_out
    normed2 = rmsnorm(x, post_attn_ln_weight, cfg.rms_norm_eps)

    gate = layers[".mlp.gate_proj.weight"].forward_dense(normed2, learning_rate=0.0)
    up   = layers[".mlp.up_proj.weight"].forward_dense(normed2, learning_rate=0.0)
    mlp_out = layers[".mlp.down_proj.weight"].forward_dense(
        silu(gate) * up, learning_rate=0.0)

    return attn_out + mlp_out


def run_folded_recurrence(
    x: np.ndarray,                          # [T, hidden] embedded input
    step_layers: List[Dict[str, object]],
    input_ln_weights: List[np.ndarray],
    post_attn_ln_weights: List[np.ndarray],
    final_norm_weight: np.ndarray,
    cfg: MiniCPM5Config,
    half_bandwidth: int,
    num_cpus: int = 4,
) -> np.ndarray:
    """state=0; for step: out=block(x+state); state+=out -- see
    RNNFoldedBlock.forward's docstring in sili__new for why this recurrence
    (not a plain 24-layer sequential replay) and why averaging/summing
    per-step outputs is not done here (skip_connection_outputs=False:
    final accumulated state is returned, RMSNorm'd, ready for lm_head)."""
    T = x.shape[0]
    cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)

    state = np.zeros_like(x)
    for i in range(cfg.num_hidden_layers):
        out = apply_fold_step(
            x + state, step_layers[i], input_ln_weights[i], post_attn_ln_weights[i],
            cfg, cos, sin, half_bandwidth, num_cpus)
        state = state + out

    return rmsnorm(state, final_norm_weight, cfg.rms_norm_eps)
