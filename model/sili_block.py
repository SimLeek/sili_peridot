"""
sili_peridot/model/sili_block.py
Per-position fold-step layers (build_step_layers) plus the growable
window-scoped combined matrix (grow_window_layer) that column-averaging
training needs.
See docs/research/sili_block.rst:sili_block.module_overview.
"""

from __future__ import annotations

import numpy as np
from sili import _cpu
from sili.energy import EnergyDynamics
from sili.sparse_rnn import fit_rank1_scale_envelope
from sili.tensor import Tensor, banded_attention, exp, gaussian_attention

from .config import MiniCPM5Config
from .fold import SUFFIXES, fold_suffix

_FP4_MAX = 6.0
_ATTN_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
)
_MLP_SUFFIXES = (".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")


def _build_step_layer_from_arrays(
    n_in: int,
    n_out: int,
    ptrs: np.ndarray,
    idx: np.ndarray,
    vals: np.ndarray,
    num_cpus: int,
    value_scale_mode: str = "per_row",
    rank1_iters: int = 6,
):
    """ptrs/idx/vals: one fold step's own [n_in, n_out] CSR slice. Builds a
    real FP4-quantized SparseLinearLayer; value_scale_mode "per_row" or
    "rank1".
    See docs/research/sili_block.rst:sili_block.build_step_layer_from_arrays.value_scale_mode.
    """
    nnz = int(vals.shape[0])
    layer = _cpu.SparseLinearLayer(n_in, n_out, int(nnz * 1.3) + 64, num_cpus)

    vals = vals.copy()
    if value_scale_mode == "rank1":
        row_of_nnz = np.repeat(np.arange(n_in, dtype=np.int64), np.diff(ptrs))
        row_env, col_env = fit_rank1_scale_envelope(
            row_of_nnz, idx.astype(np.int64), np.abs(vals), n_in, n_out, n_iters=rank1_iters
        )
        row_scales = (row_env / _FP4_MAX).astype(np.float32)
        col_scales = col_env.astype(np.float32)
        combined = row_scales[row_of_nnz] * col_scales[idx]
        nonzero_combined = combined > 0
        vals[nonzero_combined] /= combined[nonzero_combined]
    else:
        row_scales = np.ones(n_in, dtype=np.float32)
        col_scales = np.ones(n_out, dtype=np.float32)
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
    for c in range(n_out):
        if col_scales[c] != 1.0:
            layer.set_output_scale_raw(c, col_scales[c])
    return layer


def build_step_layers(
    sparse_state: dict[str, dict],
    cfg: MiniCPM5Config,
    prefix: str = "model.layers.",
    band_half_width_override=None,
    num_cpus: int = 4,
    value_scale_mode: str = "per_row",
    rank1_iters: int = 6,
) -> tuple[list[dict[str, object]], list[np.ndarray], list[np.ndarray]]:
    """Builds every fold step's sili layers + RMSNorm weight vectors,
    streaming one suffix at a time; MUTATES sparse_state.

    Returns (step_layers, input_ln_weights, post_attn_ln_weights):
      step_layers[i]        -- {suffix: SparseLinearLayer} for fold step i
      input_ln_weights[i]   -- float32 [hidden_size], layer i's input_layernorm
      post_attn_ln_weights[i] -- float32 [hidden_size], layer i's post_attention_layernorm

    See docs/research/sili_block.rst:sili_block.build_step_layers.streaming_and_build_time.
    """
    n = cfg.num_hidden_layers
    step_layers: list[dict[str, object]] = [{} for _ in range(n)]

    for suffix in SUFFIXES:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        for i in range(n):
            del sparse_state[f"{prefix}{i}{suffix}"]
        for i in range(n):
            csr_slice = desc.fold_weight_csr(suffix, i)
            csr_t = csr_slice.t().to_sparse_csr()
            n_in, out_dim = int(csr_t.shape[0]), int(csr_t.shape[1])
            ptrs = csr_t.crow_indices().numpy().astype(np.int32)
            idx = csr_t.col_indices().numpy().astype(np.int32)
            vals = csr_t.values().float().numpy()
            step_layers[i][suffix] = _build_step_layer_from_arrays(
                n_in, out_dim, ptrs, idx, vals, num_cpus, value_scale_mode=value_scale_mode, rank1_iters=rank1_iters
            )
        del desc

    input_ln = []
    post_ln = []
    for i in range(n):
        input_ln.append(sparse_state.pop(f"{prefix}{i}.input_layernorm.weight")["raw"].float().numpy().copy())
        post_ln.append(sparse_state.pop(f"{prefix}{i}.post_attention_layernorm.weight")["raw"].float().numpy().copy())
    return step_layers, input_ln, post_ln


# ── Window-scoped combined matrix. See docs/research/sili_block.rst:sili_block.module_overview.


def _extract_true_csr(layer: _cpu.SparseLinearLayer):
    """Read a SparseLinearLayer's stored (ptrs, indices, values) back out
    in TRUE units (true_w = weights_vals * value_scale[row] * output_scale[col]).
    See docs/research/sili_block.rst:sili_block.csr_accessors.true_vs_raw_units."""
    n_in, n_out = layer.n_inputs, layer.n_outputs
    ptrs = np.asarray(layer.ptrs).astype(np.int32)
    idx = np.asarray(layer.indices).astype(np.int32)
    row = np.repeat(np.arange(n_in, dtype=np.int64), np.diff(ptrs.astype(np.int64)))
    row_scale = np.array([layer.get_value_scale(r) for r in range(n_in)], dtype=np.float32)
    col_scale = np.array([layer.get_output_scale(c) for c in range(n_out)], dtype=np.float32)
    vals = np.asarray(layer.weights_vals).astype(np.float32) * row_scale[row] * col_scale[idx]
    return ptrs, idx, vals


def _raw_stored_csr(layer: _cpu.SparseLinearLayer):
    """Read (ptrs, indices, weights_vals) EXACTLY as stored -- FP4-nominal
    units, not multiplied by value_scale/output_scale -- plus the per-row
    and per-column scale arrays. Used by grow_window_layer to reuse
    existing rows/columns verbatim without refitting scales.
    See docs/research/sili_block.rst:sili_block.csr_accessors.true_vs_raw_units."""
    n_in, n_out = layer.n_inputs, layer.n_outputs
    ptrs = np.asarray(layer.ptrs).astype(np.int64)
    idx = np.asarray(layer.indices).astype(np.int32)
    stored = np.asarray(layer.weights_vals).astype(np.float32)
    row_scale = np.array([layer.get_value_scale(r) for r in range(n_in)], dtype=np.float32)
    col_scale = np.array([layer.get_output_scale(c) for c in range(n_out)], dtype=np.float32)
    return ptrs, idx, stored, row_scale, col_scale


def _fixed_band_span(row: int, in_dim: int, out_dim: int, bw: int) -> tuple[int, int]:
    """This row's recurrent-band reach in absolute output-column units,
    using its own position's fixed in/out ratio -- invariant to later
    window growth.
    See docs/research/sili_block.rst:sili_block.fixed_band_span.invariant_center."""
    center = (row * out_dim) // in_dim
    return max(0, center - bw + 1), center + bw - 1


def grow_window_layer(
    new_position_layer: _cpu.SparseLinearLayer,
    in_dim: int,
    out_dim: int,
    num_cpus: int = 4,
    recurrent_bandwidth: int | None = None,
    existing_window_layer: _cpu.SparseLinearLayer | None = None,
    existing_window_size: int = 0,
) -> _cpu.SparseLinearLayer:
    """Add ONE position to the window's combined matrix. new_position_layer
    is that position's own already-built, already-quantized small layer
    (step_layers[i][suffix]). existing_window_layer=None means the window
    is growing from 0->1 positions (new diagonal block IS the whole
    matrix). Returns a NEW SparseLinearLayer (old one is not mutated).
    recurrent_bandwidth: None (default) picks max(1, min(in_dim, out_dim) // 8).
    See docs/research/sili_block.rst:sili_block.grow_window_layer.design_and_bandwidth_tradeoff.
    """
    assert (existing_window_layer is None) == (existing_window_size == 0), (
        "existing_window_layer and existing_window_size must agree: both "
        "absent (first position) or both present (growing further)"
    )
    total_in = (existing_window_size + 1) * in_dim
    total_out = (existing_window_size + 1) * out_dim
    off_in = existing_window_size * in_dim
    off_out = existing_window_size * out_dim
    bw = recurrent_bandwidth if recurrent_bandwidth is not None else max(1, min(in_dim, out_dim) // 8)

    new_ptrs, new_idx, new_stored, new_row_scale, new_col_scale = _raw_stored_csr(new_position_layer)
    assert new_ptrs.shape[0] - 1 == in_dim

    if existing_window_layer is not None:
        old_ptrs, old_idx, old_stored, old_row_scale, old_col_scale = _raw_stored_csr(existing_window_layer)
        assert old_ptrs.shape[0] - 1 == off_in

    ptrs = np.zeros(total_in + 1, dtype=np.int64)
    row_scale = np.empty(total_in, dtype=np.float32)
    col_scale = np.empty(total_out, dtype=np.float32)
    col_scale[off_out:total_out] = new_col_scale
    idx_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []

    for r in range(off_in):
        s, e = int(old_ptrs[r]), int(old_ptrs[r + 1])
        idx_parts = [old_idx[s:e]]
        val_parts = [old_stored[s:e]]
        lo, hi = _fixed_band_span(r, in_dim, out_dim, bw)
        lo, hi = max(lo, off_out), min(hi, total_out - 1)
        if lo <= hi:
            band_idx = np.arange(lo, hi + 1, dtype=np.int32)
            idx_parts.append(band_idx)
            val_parts.append(np.zeros(band_idx.shape[0], dtype=np.float32))
        row_idx = idx_parts[0] if len(idx_parts) == 1 else np.concatenate(idx_parts)
        row_val = val_parts[0] if len(val_parts) == 1 else np.concatenate(val_parts)
        idx_chunks.append(row_idx)
        val_chunks.append(row_val)
        ptrs[r + 1] = ptrs[r] + row_idx.shape[0]
        row_scale[r] = old_row_scale[r]

    if off_in > 0:
        col_scale[0:off_out] = old_col_scale

    for col in range(in_dim):
        r = off_in + col
        s, e = int(new_ptrs[col]), int(new_ptrs[col + 1])
        idx_parts = [new_idx[s:e].astype(np.int64) + off_out]
        val_parts = [new_stored[s:e]]
        lo, hi = _fixed_band_span(r, in_dim, out_dim, bw)
        lo, hi = max(lo, 0), min(hi, off_out - 1)
        if lo <= hi:
            band_idx = np.arange(lo, hi + 1, dtype=np.int64)
            idx_parts.append(band_idx)
            val_parts.append(np.zeros(band_idx.shape[0], dtype=np.float32))
        row_idx = np.concatenate(idx_parts)
        row_val = np.concatenate(val_parts)
        order = np.argsort(row_idx)
        idx_chunks.append(row_idx[order].astype(np.int32))
        val_chunks.append(row_val[order])
        ptrs[r + 1] = ptrs[r] + row_idx.shape[0]
        row_scale[r] = new_row_scale[col]

    u_idx = np.concatenate(idx_chunks)
    u_val = np.concatenate(val_chunks)
    u_ptrs = ptrs.astype(np.int32)

    nnz = int(u_val.shape[0])
    layer = _cpu.SparseLinearLayer(total_in, total_out, int(nnz * 1.3) + 64, num_cpus)
    layer.load_weights(u_ptrs, u_idx, u_val)
    for r in range(total_in):
        if row_scale[r] != 1.0:
            layer.set_value_scale_raw(r, row_scale[r])
    for c in range(total_out):
        if col_scale[c] != 1.0:
            layer.set_output_scale_raw(c, col_scale[c])
    return layer


# ── Elementwise math (RMSNorm / RoPE / SiLU) -- plain numpy, no sparsity.


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """x: [T, hidden]. Matches sili__new's model_reconstruct.py _LlamaRMSNorm."""
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(var + eps))).astype(np.float32) * weight


def rope_cos_sin(seq_len: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)  # [T, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [T, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rotary(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: [T, head_dim], cos/sin: [T, head_dim]."""
    return x * cos + _rotate_half(x) * sin


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _forward(layer, x: np.ndarray, activation_density: float | None) -> np.ndarray:
    """activation_density=None (default): dense forward_dense, unchanged.
    activation_density=d (0 < d <= 1): keep only the top round(d*n_features)
    entries by magnitude PER ROW of x, route through forward_sparse instead.
    See docs/research/sili_block.rst:sili_block.forward.activation_density_argpartition.
    """
    if activation_density is None:
        return layer.forward_dense(x)
    T, n_features = x.shape
    k = max(1, round(activation_density * n_features))
    abs_x = np.abs(x)
    top_idx = np.argpartition(abs_x, n_features - k, axis=1)[:, n_features - k :]
    top_idx = np.sort(top_idx, axis=1)
    top_vals = np.take_along_axis(x, top_idx, axis=1)
    idx = top_idx.ravel().astype(np.int32)
    vals = top_vals.ravel().astype(np.float32)
    ptrs = np.arange(0, (T + 1) * k, k, dtype=np.int32)
    return layer.forward_sparse(ptrs, idx, vals, T)


_ActivationDensity = float | dict[str, float | None] | None


def _density_for_suffix(step_density: _ActivationDensity, suffix: str) -> float | None:
    """step_density is either a single value (applies to every projection)
    or a dict {suffix: None/float} isolating individual projections.
    See docs/research/sili_block.rst:sili_block.forward.activation_density_argpartition."""
    if isinstance(step_density, dict):
        return step_density.get(suffix)
    return step_density


# ── One fold step: RMSNorm -> GQA causal attention (RoPE) -> RMSNorm -> SwiGLU MLP


def apply_fold_step(
    x: np.ndarray,  # [T, hidden] = original input + accumulated state
    layers: dict[str, object],  # this step's {suffix: SparseLinearLayer}
    input_ln_weight: np.ndarray,
    post_attn_ln_weight: np.ndarray,
    cfg: MiniCPM5Config,
    cos: np.ndarray,
    sin: np.ndarray,  # from rope_cos_sin(T, head_dim, rope_theta)
    half_bandwidth: int,
    num_cpus: int = 4,
    activation_density: _ActivationDensity = None,
) -> np.ndarray:
    """Returns this step's own output [T, hidden] (the recurrence's caller
    accumulates it into state). See _forward for activation_density; may
    also be a dict keyed by suffix to sparsify only some projections --
    see _density_for_suffix."""
    T = x.shape[0]
    n_heads, n_kv_heads, head_dim = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    groups = n_heads // n_kv_heads

    normed = rmsnorm(x, input_ln_weight, cfg.rms_norm_eps)

    q = _forward(
        layers[".self_attn.q_proj.weight"], normed, _density_for_suffix(activation_density, ".self_attn.q_proj.weight")
    )
    k = _forward(
        layers[".self_attn.k_proj.weight"], normed, _density_for_suffix(activation_density, ".self_attn.k_proj.weight")
    )
    v = _forward(
        layers[".self_attn.v_proj.weight"], normed, _density_for_suffix(activation_density, ".self_attn.v_proj.weight")
    )

    q = q.reshape(T, n_heads, head_dim)
    k = k.reshape(T, n_kv_heads, head_dim)
    v = v.reshape(T, n_kv_heads, head_dim)

    attn_out = np.empty((T, n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // groups
        qh = Tensor(apply_rotary(q[:, h, :], cos, sin))
        kh = Tensor(apply_rotary(k[:, kv_h, :], cos, sin))
        vh = Tensor(np.ascontiguousarray(v[:, kv_h, :]))
        out_h = banded_attention(qh, kh, vh, half_bandwidth=half_bandwidth, num_cpus=num_cpus, causal=True)
        attn_out[:, h, :] = out_h.data

    attn_out = attn_out.reshape(T, n_heads * head_dim)
    attn_out = _forward(
        layers[".self_attn.o_proj.weight"],
        attn_out,
        _density_for_suffix(activation_density, ".self_attn.o_proj.weight"),
    )

    x = x + attn_out
    normed2 = rmsnorm(x, post_attn_ln_weight, cfg.rms_norm_eps)

    gate = _forward(
        layers[".mlp.gate_proj.weight"], normed2, _density_for_suffix(activation_density, ".mlp.gate_proj.weight")
    )
    up = _forward(
        layers[".mlp.up_proj.weight"], normed2, _density_for_suffix(activation_density, ".mlp.up_proj.weight")
    )
    mlp_out = _forward(
        layers[".mlp.down_proj.weight"],
        silu(gate) * up,
        _density_for_suffix(activation_density, ".mlp.down_proj.weight"),
    )

    return attn_out + mlp_out


def apply_window_step(
    x_common_t: np.ndarray,  # [hidden] -- ONE token, SAME starting input for every window position
    carried_state: np.ndarray,  # [window_size, hidden] -- persisted from the PREVIOUS token step
    window_layers: dict[str, object],  # {suffix: combined layer spanning window_size positions}
    window_size: int,
    input_ln_weights: list[np.ndarray],  # window order (index 0 = last fold-step)
    post_attn_ln_weights: list[np.ndarray],  # window order, same indexing
    cfg: MiniCPM5Config,
    energy_dynamics: EnergyDynamics,
    centers: Tensor,  # [window_size] -- see WindowState.centers
    log_sigmas: Tensor,  # [window_size] -- see WindowState.log_sigmas
    num_cpus: int = 4,
    activation_density: _ActivationDensity = None,
) -> tuple[np.ndarray, np.ndarray, Tensor]:
    """The in-window counterpart to apply_fold_step: processes ONE token at
    a time, routing q/k/v/o/gate/up/down through ONE combined matrix per
    suffix spanning the whole window (grow_window_layer's output), and
    blending token+carried-state via an interleaved gaussian_attention.

    Returns (delta [window_size, hidden] -- this step's own output, NOT
    yet added to x_common_t; new_carried_state [window_size, hidden];
    aux_loss -- a Tensor scalar from EnergyDynamics, safely ignorable in
    forward-only use).
    See docs/research/sili_block.rst:sili_block.apply_window_step.major_pivot_design.
    """
    hidden = cfg.hidden_size
    n_heads, n_kv_heads, head_dim = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    groups = n_heads // n_kv_heads
    q_proj_out, _kv_proj_out = cfg.q_proj_out, cfg.kv_proj_out

    ln_stack = np.stack(input_ln_weights[:window_size])  # [window_size, hidden]
    x_common_stack = np.broadcast_to(x_common_t, (window_size, hidden))
    normed = rmsnorm(x_common_stack, ln_stack, cfg.rms_norm_eps)  # [window_size, hidden]
    carried_normed = rmsnorm(carried_state, ln_stack, cfg.rms_norm_eps)  # [window_size, hidden]

    normed_flat = normed.reshape(1, window_size * hidden)
    carried_flat = carried_normed.reshape(1, window_size * hidden)
    # Q draws from token AND carried state together -- see
    # docs/research/sili_block.rst:sili_block.apply_window_step.major_pivot_design.
    qk_source_flat = (normed + carried_normed).reshape(1, window_size * hidden)

    q = _forward(
        window_layers[".self_attn.q_proj.weight"],
        qk_source_flat,
        _density_for_suffix(activation_density, ".self_attn.q_proj.weight"),
    )[0]
    k_new = _forward(
        window_layers[".self_attn.k_proj.weight"],
        normed_flat,
        _density_for_suffix(activation_density, ".self_attn.k_proj.weight"),
    )[0]
    k_state = _forward(
        window_layers[".self_attn.k_proj.weight"],
        carried_flat,
        _density_for_suffix(activation_density, ".self_attn.k_proj.weight"),
    )[0]
    v_new = _forward(
        window_layers[".self_attn.v_proj.weight"],
        normed_flat,
        _density_for_suffix(activation_density, ".self_attn.v_proj.weight"),
    )[0]
    v_state = _forward(
        window_layers[".self_attn.v_proj.weight"],
        carried_flat,
        _density_for_suffix(activation_density, ".self_attn.v_proj.weight"),
    )[0]

    q = q.reshape(window_size, n_heads, head_dim)
    k_new = k_new.reshape(window_size, n_kv_heads, head_dim)
    k_state = k_state.reshape(window_size, n_kv_heads, head_dim)
    v_new = v_new.reshape(window_size, n_kv_heads, head_dim)
    v_state = v_state.reshape(window_size, n_kv_heads, head_dim)

    k_new_exp = np.repeat(k_new, groups, axis=1)  # [window_size, n_heads, head_dim]
    k_state_exp = np.repeat(k_state, groups, axis=1)
    v_new_exp = np.repeat(v_new, groups, axis=1)
    v_state_exp = np.repeat(v_state, groups, axis=1)

    # Interleave fresh-token/carried-state entries: index 2p = position
    # p's fresh-token (K,V), 2p+1 = its carried-state (K,V). See
    # docs/research/sili_block.rst:sili_block.apply_window_step.major_pivot_design.
    combined_k = np.empty((2 * window_size, n_heads, head_dim), dtype=np.float32)
    combined_v = np.empty((2 * window_size, n_heads, head_dim), dtype=np.float32)
    combined_k[0::2] = k_new_exp
    combined_k[1::2] = k_state_exp
    combined_v[0::2] = v_new_exp
    combined_v[1::2] = v_state_exp

    sigmas = exp(log_sigmas)
    blended = np.empty((window_size, n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        qh = Tensor(np.ascontiguousarray(q[:, h, :]))
        kh = Tensor(np.ascontiguousarray(combined_k[:, h, :]))
        vh = Tensor(np.ascontiguousarray(combined_v[:, h, :]))
        out_h = gaussian_attention(qh, kh, vh, centers, sigmas, num_cpus=num_cpus, causal=False)
        blended[:, h, :] = out_h.data

    blended_flat = blended.reshape(1, window_size * q_proj_out)

    attn_out = _forward(
        window_layers[".self_attn.o_proj.weight"],
        blended_flat,
        _density_for_suffix(activation_density, ".self_attn.o_proj.weight"),
    )
    attn_out = attn_out.reshape(window_size, hidden)

    pre_gate_state = (x_common_stack + attn_out).reshape(-1).astype(np.float32)  # [window_size*hidden]
    gated_tensor, aux_loss, _actual_p = energy_dynamics(Tensor(pre_gate_state))
    new_carried_state = gated_tensor.data.reshape(window_size, hidden)

    normed2 = rmsnorm(new_carried_state, np.stack(post_attn_ln_weights[:window_size]), cfg.rms_norm_eps)
    normed2_flat = normed2.reshape(1, window_size * hidden)

    gate_mlp = _forward(
        window_layers[".mlp.gate_proj.weight"],
        normed2_flat,
        _density_for_suffix(activation_density, ".mlp.gate_proj.weight"),
    )
    up_mlp = _forward(
        window_layers[".mlp.up_proj.weight"],
        normed2_flat,
        _density_for_suffix(activation_density, ".mlp.up_proj.weight"),
    )
    mlp_out = _forward(
        window_layers[".mlp.down_proj.weight"],
        silu(gate_mlp) * up_mlp,
        _density_for_suffix(activation_density, ".mlp.down_proj.weight"),
    )
    mlp_out = mlp_out.reshape(window_size, hidden)

    delta = (new_carried_state - x_common_stack) + mlp_out
    return delta, new_carried_state, aux_loss


def default_window_energy(percent_active: float = 0.25) -> EnergyDynamics:
    """Placeholder EnergyDynamics defaults; real tuning is Phase 5's job.
    See docs/research/sili_block.rst:sili_block.default_window_helpers.placeholder_and_init_conventions."""
    r = percent_active / 0.02
    density = min(0.9, percent_active)
    p = min(1.0, percent_active * 5.0)
    activation_cost = min(0.5, max(0.01, 0.08 * r))
    return EnergyDynamics(
        drive=0.08 * percent_active * r,
        activation_cost=activation_cost,
        density=density,
        exploration=0.001 * r,
        reactivity=0.01 * r,
        precision=0.04 * r,
        setpoint=1.0,
        activation_threshold=1e-4,
        p=p,
    )


def default_window_gaussian_params(window_size: int) -> tuple[Tensor, Tensor]:
    """Fresh, from-scratch centers/log_sigmas for a window_size-wide window.
    See docs/research/sili_block.rst:sili_block.default_window_helpers.placeholder_and_init_conventions."""
    centers = Tensor(np.array([2.0 * p + 0.5 for p in range(window_size)], dtype=np.float32))
    log_sigmas = Tensor(np.zeros(window_size, dtype=np.float32))
    return centers, log_sigmas


def run_folded_recurrence(
    x: np.ndarray,  # [T, hidden] embedded input
    step_layers: list[dict[str, object]],
    input_ln_weights: list[np.ndarray],
    post_attn_ln_weights: list[np.ndarray],
    final_norm_weight: np.ndarray,
    cfg: MiniCPM5Config,
    half_bandwidth: int,
    num_cpus: int = 4,
    activation_density: _ActivationDensity | list[_ActivationDensity] = None,
    window_state=None,  # curriculum.WindowState, or None -- see below
    window_activation_density: _ActivationDensity = None,
    window_energy: EnergyDynamics | None = None,
    window_carried_state: np.ndarray | None = None,
) -> np.ndarray:
    """state=0; for step: out=block(x+state); state+=out.

    window_state=None (default): the plain pre-window path, unchanged
    since B6 -- every position runs sequentially.

    window_state=<a curriculum.WindowState-shaped object> (duck-typed, not
    imported here to avoid a curriculum<->sili_block import cycle): positions
    before the window run the same plain sequential loop; positions inside
    the window (window_size >= 2) switch to one token at a time via
    apply_window_step, threading carried_state token to token.
    window_size==1 bypasses apply_window_step entirely (provably identical
    to the plain path). Returns column-averaged prediction, RMSNorm'd.

    See docs/research/sili_block.rst:sili_block.run_folded_recurrence.window_curriculum_design.
    """
    T = x.shape[0]
    cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)
    per_step = isinstance(activation_density, list)
    if per_step and len(activation_density) != cfg.num_hidden_layers:
        raise ValueError(
            f"activation_density list has {len(activation_density)} entries, "
            f"expected cfg.num_hidden_layers={cfg.num_hidden_layers}"
        )

    has_window = window_state is not None and window_state.window_size > 0
    pre_window_end = window_state.window_positions[-1] if has_window else cfg.num_hidden_layers

    state = np.zeros_like(x)
    for i in range(pre_window_end):
        step_density = activation_density[i] if per_step else activation_density
        out = apply_fold_step(
            x + state,
            step_layers[i],
            input_ln_weights[i],
            post_attn_ln_weights[i],
            cfg,
            cos,
            sin,
            half_bandwidth,
            num_cpus,
            step_density,
        )
        state = state + out

    if not has_window:
        return rmsnorm(state, final_norm_weight, cfg.rms_norm_eps)

    x_common = x + state
    window_size = window_state.window_size
    positions = window_state.window_positions

    if window_size == 1:
        pos = positions[0]
        out = apply_fold_step(
            x_common,
            step_layers[pos],
            input_ln_weights[pos],
            post_attn_ln_weights[pos],
            cfg,
            cos,
            sin,
            half_bandwidth,
            num_cpus,
            window_activation_density,
        )
        mean_column = state + out
    else:
        hidden = cfg.hidden_size
        energy = window_energy if window_energy is not None else default_window_energy()
        carried_state = (
            window_carried_state.copy()
            if window_carried_state is not None
            else np.zeros((window_size, hidden), dtype=np.float32)
        )
        window_ln = [input_ln_weights[p] for p in positions]
        window_post_ln = [post_attn_ln_weights[p] for p in positions]

        mean_column = np.empty((T, hidden), dtype=np.float32)
        for t in range(T):
            delta, carried_state, _aux_loss = apply_window_step(
                x_common[t],
                carried_state,
                window_state.suffix_windows,
                window_size,
                window_ln,
                window_post_ln,
                cfg,
                energy,
                window_state.centers,
                window_state.log_sigmas,
                num_cpus,
                window_activation_density,
            )
            columns_t = state[t][None, :] + delta  # [window_size, hidden]
            mean_column[t] = columns_t.mean(axis=0)

    return rmsnorm(mean_column, final_norm_weight, cfg.rms_norm_eps)
