"""
sili_peridot/model/sili_block.py
─────────────────────────────────
B6/B8-Phase2: attention assembly, PLUS the growable window-scoped
combined matrix that column-averaging training needs.

Every one of MiniCPM5's 24 fold-depth positions gets its OWN small,
independently-quantized SparseLinearLayer per suffix (build_step_layers,
168 matrices total) -- unchanged from B6. A position outside the current
B8a curriculum window passes exactly one token through the whole system
at a time, so it can never have a recurrent/cross-position connection;
running it through anything heavier than its own small layer would be
wasted compute. run_folded_recurrence therefore keeps every pre-window
position on this plain per-position path, exactly as it always has
(state=0; for step: out=block(x+state); state+=out).

Only the CURRENT window (the last few positions B8a's curriculum is
training column-averaging over) needs a combined matrix -- that's the
only place a cross-position (recurrent/skip) synapse has anywhere to
live. grow_window_layer() builds that combined matrix INCREMENTALLY:
each time the window widens by one position (a curriculum stage
transition), the newly-included position's own already-quantized
step_layers[i][suffix] is folded in as a new diagonal block, and the
existing (already-trained) window matrix's rows/columns are reused
VERBATIM, not rescanned or rebuilt from scratch -- see grow_window_layer
and _raw_stored_csr for why this is exact (not approximate) regardless
of whether per_row or rank1 quantization built the underlying layers.

RMSNorm/RoPE weights are never part of the 7 folded suffixes (B3/B5 never
prune or quantize them), so they're read directly from sparse_state as
plain float32 vectors, not built into any SparseLinearLayer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from sili import _cpu
from sili.energy import EnergyDynamics
from sili.sparse_rnn import fit_rank1_scale_envelope
from sili.tensor import Tensor, banded_attention, gaussian_attention, exp

from .config import MiniCPM5Config
from .fold import SUFFIXES, fold_suffix

_FP4_MAX = 6.0
_ATTN_SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                  ".self_attn.v_proj.weight", ".self_attn.o_proj.weight")
_MLP_SUFFIXES  = (".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")


def _build_step_layer_from_arrays(
    n_in: int, n_out: int, ptrs: np.ndarray, idx: np.ndarray, vals: np.ndarray, num_cpus: int,
    value_scale_mode: str = "per_row", rank1_iters: int = 6,
):
    """
    ptrs/idx/vals: one fold step's own [n_in, n_out] CSR, already sliced
    out of the suffix's full stacked-and-transposed matrix (see
    build_step_layers -- the transpose/conversion happens ONCE per suffix,
    not once per step, since torch's to_sparse_csr()/t() carry real
    per-call overhead that a 24x-per-suffix call count made the dominant
    cost of building the model at all).

    Real FP4-quantized SparseLinearLayer. value_scale_mode="per_row"
    (default): one value_scale per input row, matching
    FoldedLayer.from_descriptor's "per_row" mode. "rank1": also fits a
    per-output-column scale (fit_rank1_scale_envelope), same scheme as
    B5a's from_descriptor "rank1" mode but fit independently per fold
    step instead of shared across all 24 -- each step's own weight
    distribution gets its own row+col envelope, not a compromise shared
    across the whole stacked matrix.
    """
    nnz = int(vals.shape[0])
    layer = _cpu.SparseLinearLayer(n_in, n_out, int(nnz * 1.3) + 64, num_cpus)

    vals = vals.copy()
    if value_scale_mode == "rank1":
        row_of_nnz = np.repeat(np.arange(n_in, dtype=np.int64), np.diff(ptrs))
        row_env, col_env = fit_rank1_scale_envelope(
            row_of_nnz, idx.astype(np.int64), np.abs(vals), n_in, n_out, n_iters=rank1_iters)
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
    sparse_state: Dict[str, dict],
    cfg: MiniCPM5Config,
    prefix: str = "model.layers.",
    band_half_width_override=None,
    num_cpus: int = 4,
    value_scale_mode: str = "per_row",
    rank1_iters: int = 6,
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

    These per-position layers are the ONLY construction needed for
    positions outside the current B8a curriculum window (see module
    docstring) -- they also double as the source grow_window_layer()
    folds in when a position newly enters the window, so there is no
    separate/duplicate build path for that case.

    NOTE: two different attempts to reduce build time by cutting the
    number of torch .t().to_sparse_csr() calls (transposing the whole
    suffix once instead of once per step, and separately a pure-numpy
    stable-sort transpose avoiding torch's CSR machinery altogether) were
    both measured SLOWER on the real checkpoint than the current
    per-step fold_weight_csr approach below (real regressions: ~150-165s
    vs ~77-81s) -- reducing call count wasn't the actual lever, and the
    real bottleneck hasn't been isolated yet. See JOURNAL.md.
    """
    n = cfg.num_hidden_layers
    step_layers: List[Dict[str, object]] = [dict() for _ in range(n)]

    for suffix in SUFFIXES:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        for i in range(n):
            del sparse_state[f"{prefix}{i}{suffix}"]
        for i in range(n):
            csr_slice = desc.fold_weight_csr(suffix, i)
            csr_t = csr_slice.t().to_sparse_csr()
            n_in, out_dim = int(csr_t.shape[0]), int(csr_t.shape[1])
            ptrs = csr_t.crow_indices().numpy().astype(np.int32)
            idx  = csr_t.col_indices().numpy().astype(np.int32)
            vals = csr_t.values().float().numpy()
            step_layers[i][suffix] = _build_step_layer_from_arrays(
                n_in, out_dim, ptrs, idx, vals, num_cpus,
                value_scale_mode=value_scale_mode, rank1_iters=rank1_iters)
        del desc

    input_ln = []
    post_ln  = []
    for i in range(n):
        input_ln.append(sparse_state.pop(f"{prefix}{i}.input_layernorm.weight")["raw"]
                        .float().numpy().copy())
        post_ln.append(sparse_state.pop(f"{prefix}{i}.post_attention_layernorm.weight")["raw"]
                       .float().numpy().copy())
    return step_layers, input_ln, post_ln


# ── Window-scoped combined matrix: grown incrementally, one position at a
# time, as B8a's curriculum widens the window. Never built for positions
# outside the window -- see module docstring.

def _extract_true_csr(layer: "_cpu.SparseLinearLayer"):
    """Read a SparseLinearLayer's stored (ptrs, indices, values) back out
    in TRUE units (true_w = weights_vals * value_scale[row] *
    output_scale[col], see cpu_backend.cpp) -- same pattern
    quantize.py's build_quantized_dense_state_dict_streaming already
    uses. Not used by grow_window_layer itself (see _raw_stored_csr) --
    kept as a general true-unit accessor, used by tests/reporting that
    want to inspect a layer's real weight values."""
    n_in, n_out = layer.n_inputs, layer.n_outputs
    ptrs = np.asarray(layer.ptrs).astype(np.int32)
    idx  = np.asarray(layer.indices).astype(np.int32)
    row  = np.repeat(np.arange(n_in, dtype=np.int64), np.diff(ptrs.astype(np.int64)))
    row_scale = np.array([layer.get_value_scale(r) for r in range(n_in)], dtype=np.float32)
    col_scale = np.array([layer.get_output_scale(c) for c in range(n_out)], dtype=np.float32)
    vals = np.asarray(layer.weights_vals).astype(np.float32) * row_scale[row] * col_scale[idx]
    return ptrs, idx, vals


def _raw_stored_csr(layer: "_cpu.SparseLinearLayer"):
    """Read (ptrs, indices, weights_vals) EXACTLY as stored -- FP4-nominal
    units, NOT multiplied by value_scale/output_scale -- plus the
    per-row and per-column scale arrays, with no dequantization
    arithmetic at all. Used by grow_window_layer (see its docstring) to
    reuse existing rows/columns verbatim instead of round-tripping
    through true units and refitting scales that provably can't have
    changed -- true REGARDLESS of value_scale_mode (see grow_window_layer):
    a row's/column's scale only depends on the REAL (nonzero) entries
    touching it, and every entry grow_window_layer newly introduces
    outside a position's own diagonal block is zero-valued, which never
    binds a max-based fit (per_row's per-row max_abs, or rank1's
    alternating max-fit in fit_rank1_scale_envelope -- both are pure
    max operations, and 0 never raises a max above what real content
    already set)."""
    n_in, n_out = layer.n_inputs, layer.n_outputs
    ptrs = np.asarray(layer.ptrs).astype(np.int64)
    idx  = np.asarray(layer.indices).astype(np.int32)
    stored = np.asarray(layer.weights_vals).astype(np.float32)
    row_scale = np.array([layer.get_value_scale(r) for r in range(n_in)], dtype=np.float32)
    col_scale = np.array([layer.get_output_scale(c) for c in range(n_out)], dtype=np.float32)
    return ptrs, idx, stored, row_scale, col_scale


def _fixed_band_span(row: int, in_dim: int, out_dim: int, bw: int) -> Tuple[int, int]:
    """This row's recurrent-band reach in absolute output-column units,
    using its OWN position's fixed in/out ratio -- NOT recentered
    against however large total_in/total_out have grown to since. An
    absolute row = p*in_dim + l always maps to
    center = p*out_dim + l*out_dim/in_dim, i.e. always INSIDE row's own
    position's own output block (out_dim/in_dim is the same ratio for
    every position of this suffix, so p cancels out of where the block
    starts) -- provably invariant to how many further positions get
    added later. That's what lets grow_window_layer only ever touch an
    OLD row once per later stage (to extend it into whatever NEW column
    range just opened), never recomputing/rescanning its already-placed
    entries."""
    center = (row * out_dim) // in_dim
    return max(0, center - bw + 1), center + bw - 1


def grow_window_layer(
    new_position_layer: "_cpu.SparseLinearLayer",
    in_dim: int, out_dim: int, num_cpus: int = 4,
    recurrent_bandwidth: Optional[int] = None,
    existing_window_layer: Optional["_cpu.SparseLinearLayer"] = None,
    existing_window_size: int = 0,
) -> "_cpu.SparseLinearLayer":
    """
    Add ONE position to the window's combined matrix. `new_position_layer`
    is that position's own already-built, already-quantized small layer
    (step_layers[i][suffix] -- see build_step_layers; no separate build
    path, no `desc`/pretrained-tensor access needed here at all). Works
    identically regardless of what value_scale_mode built
    `new_position_layer`/`existing_window_layer` (per_row or rank1) --
    see _raw_stored_csr's docstring for the proof: growth only ever adds
    ZERO-valued cross-position entries, which never move a max-based
    scale fit of either kind, so every row's AND every column's scale
    (when set at all) is reused verbatim from wherever it already came
    from, never refit.

    existing_window_layer=None (window growing from 0->1 positions):
    the new diagonal block IS the whole matrix. Positions are appended
    in window-growth order (index 0 = first position added to the
    window, i.e. the LAST fold-step under B8a's backward-growing
    curriculum) -- existing blocks' row/col offsets never shift as the
    window grows.

    Returns a NEW SparseLinearLayer (old one is not mutated in place --
    caller replaces its reference). Structure (old rows extended into
    whatever new column range just opened; new rows carrying their own
    diagonal block plus a band reaching back into existing columns) is
    a plain disjoint concatenation, never a value-changing union or a
    rescan -- old-row content and new-position content live in strictly
    disjoint column ranges (old content < off_out <= new diagonal; new
    backward band < off_out <= new diagonal too), so no
    conflict-resolution is needed; only a new row's own two pieces
    (backward band + diagonal) need an explicit sort.

    recurrent_bandwidth: None (default) picks max(1, min(in_dim,
    out_dim) // 8) -- deliberately NOT scaled to 2*max(in_dim, out_dim)
    (an earlier version of this function did that, matching
    build_fold_skip_layer's own "one hop = out_dim" convention doubled).
    Measured directly at MiniCPM5's real dims (in_dim=1536,
    out_dim=4608): a bandwidth on that same order made the "band" cover
    the ENTIRE row width, i.e. fully DENSE, not sparse -- ~7M zero
    entries for a single position's block in one suffix alone, a real
    "ton of memory at large layers" problem. This default keeps
    nnz-per-row (and therefore total memory) a small, bounded fraction
    of the layer's own width regardless of how wide the real layer is.
    This is a genuine richness-vs-memory tradeoff, not a fully "solved"
    number -- a small bandwidth means only neurons near a position's own
    block boundary ever get a pre-seeded cross-position slot at all (see
    _fixed_band_span: the proportional center always falls inside a
    row's OWN position block, so reaching a neighbor at all requires
    bw comparable to the distance from that row to its block's edge).
    Phase 4's reporting is the place to check whether this is generous
    enough for synaptogenesis to find useful cross-position connections
    in practice; tune via this argument, not by editing the default.
    """
    assert (existing_window_layer is None) == (existing_window_size == 0), (
        "existing_window_layer and existing_window_size must agree: both "
        "absent (first position) or both present (growing further)")
    total_in  = (existing_window_size + 1) * in_dim
    total_out = (existing_window_size + 1) * out_dim
    off_in  = existing_window_size * in_dim
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
    idx_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []

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

    for l in range(in_dim):
        r = off_in + l
        s, e = int(new_ptrs[l]), int(new_ptrs[l + 1])
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
        row_scale[r] = new_row_scale[l]

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


def _forward(layer, x: np.ndarray, activation_density: Optional[float]) -> np.ndarray:
    """
    activation_density=None (default): dense DISLDO forward_dense, current
    behavior, unchanged.

    activation_density=d (0 < d <= 1): keep only the top round(d*n_features)
    entries by magnitude PER ROW (per token) of x, route through the
    existing SISLDO forward_sparse path instead.

    Per-row top-k is done via np.argpartition (not a Python loop calling
    _cpu.dense_to_top_k_csr once per row -- that was ~74x slower at
    T=30,F=1536, measured directly: 57ms/call vs 0.77ms/call, dominated
    by per-call pybind overhead x T rows x 7 projections x 24 layers).
    dense_to_top_k_csr's own k is a GLOBAL budget over the whole
    flattened [rows, cols] array (see sili__new's csr.hpp top_k_csr ->
    top_k_indices), not a per-row budget -- np.argpartition(..., axis=1)
    is naturally per-row, sidestepping that mismatch entirely.
    """
    if activation_density is None:
        return layer.forward_dense(x, learning_rate=0.0)
    T, n_features = x.shape
    k = max(1, round(activation_density * n_features))
    abs_x = np.abs(x)
    top_idx = np.argpartition(abs_x, n_features - k, axis=1)[:, n_features - k:]
    top_idx = np.sort(top_idx, axis=1)
    top_vals = np.take_along_axis(x, top_idx, axis=1)
    idx = top_idx.ravel().astype(np.int32)
    vals = top_vals.ravel().astype(np.float32)
    ptrs = np.arange(0, (T + 1) * k, k, dtype=np.int32)
    return layer.forward_sparse(ptrs, idx, vals, T, learning_rate=0.0)


_ActivationDensity = Union[None, float, Dict[str, Optional[float]]]


def _density_for_suffix(step_density: _ActivationDensity, suffix: str) -> Optional[float]:
    """step_density is either a single value (None/float, applies to every
    projection this step -- the original global behavior) or a dict
    {suffix: None/float} that isolates individual projections (q/k/v/o/
    gate/up/down) -- lets a diagnostic sparsify only e.g. the MLP suffixes
    while leaving attention dense, to localize which projections tolerate
    top-k activation sparsification and which collapse it."""
    if isinstance(step_density, dict):
        return step_density.get(suffix)
    return step_density


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
    activation_density: _ActivationDensity = None,
) -> np.ndarray:
    """Returns this step's own output [T, hidden] (the recurrence's caller
    accumulates it into state). See _forward for activation_density; may
    also be a dict keyed by suffix to sparsify only some projections --
    see _density_for_suffix."""
    T = x.shape[0]
    n_heads, n_kv_heads, head_dim = (cfg.num_attention_heads,
                                     cfg.num_key_value_heads, cfg.head_dim)
    groups = n_heads // n_kv_heads

    normed = rmsnorm(x, input_ln_weight, cfg.rms_norm_eps)

    q = _forward(layers[".self_attn.q_proj.weight"], normed,
                 _density_for_suffix(activation_density, ".self_attn.q_proj.weight"))
    k = _forward(layers[".self_attn.k_proj.weight"], normed,
                 _density_for_suffix(activation_density, ".self_attn.k_proj.weight"))
    v = _forward(layers[".self_attn.v_proj.weight"], normed,
                 _density_for_suffix(activation_density, ".self_attn.v_proj.weight"))

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
    attn_out = _forward(layers[".self_attn.o_proj.weight"], attn_out,
                        _density_for_suffix(activation_density, ".self_attn.o_proj.weight"))

    x = x + attn_out
    normed2 = rmsnorm(x, post_attn_ln_weight, cfg.rms_norm_eps)

    gate = _forward(layers[".mlp.gate_proj.weight"], normed2,
                    _density_for_suffix(activation_density, ".mlp.gate_proj.weight"))
    up   = _forward(layers[".mlp.up_proj.weight"], normed2,
                    _density_for_suffix(activation_density, ".mlp.up_proj.weight"))
    mlp_out = _forward(layers[".mlp.down_proj.weight"], silu(gate) * up,
                       _density_for_suffix(activation_density, ".mlp.down_proj.weight"))

    return attn_out + mlp_out


def apply_window_step(
    x_common_t: np.ndarray,        # [hidden] -- ONE token, SAME starting input for every window position
    carried_state: np.ndarray,     # [window_size, hidden] -- persisted from the PREVIOUS token step
    window_layers: Dict[str, object],   # {suffix: combined layer spanning window_size positions}
    window_size: int,
    input_ln_weights: List[np.ndarray],       # window order (index 0 = last fold-step)
    post_attn_ln_weights: List[np.ndarray],   # window order, same indexing
    cfg: MiniCPM5Config,
    energy_dynamics: EnergyDynamics,
    centers: Tensor,                # [window_size] -- see WindowState.centers
    log_sigmas: Tensor,              # [window_size] -- see WindowState.log_sigmas
    num_cpus: int = 4,
    activation_density: _ActivationDensity = None,
) -> Tuple[np.ndarray, np.ndarray, Tensor]:
    """The in-window counterpart to apply_fold_step -- per the MAJOR
    PIVOT (2026-08-02), scoped to the window ONLY and no longer a
    T-token-batched causal-attention block. Processes ONE token at a
    time: every window position runs its OWN RMSNorm (still strictly
    per-position, unchanged in kind from apply_fold_step), but the linear
    projections (q/k/v/o/gate/up/down) route through ONE combined matrix
    per suffix spanning the whole window (grow_window_layer's output)
    instead of separate small per-position layers -- that combined
    matrix is the only place a cross-position (recurrent/skip) synapse
    can live.

    Time-axis mechanism (replaces causal attention over T, which is
    meaningless once T=1, AND replaces Phase 2.5/2.6's two separate,
    only-one-of-them-trainable mechanisms -- see Phase 2.7): ONE real
    attention op, `sili.tensor.gaussian_attention`, per head, over a
    combined key/value space that INTERLEAVES each window position's
    fresh-token entry and carried-state entry: index `2p` = position
    p's fresh-token (K,V), index `2p+1` = position p's carried-state
    (K,V) -- length `2*window_size`. `q` is the existing per-position
    query, drawing from BOTH this token AND the carried state together
    (`q_proj` applied to `normed(token) + normed(state)` -- linear, so
    this is exactly `q_proj(normed(token)) + q_proj(normed(state))`,
    just cheaper to compute once), per direct correction: tying Q
    exclusively to the current token means it goes dead (content-blind,
    `q=0`) with no fresh input, which forecloses any "sleep"/
    consolidation-style internal dynamics driven by EnergyDynamics' own
    exploration noise. With no real input, `normed(token)` is simply
    zero and Q still carries the state contribution, so attention keeps
    running over memory alone -- genuine ongoing internal dynamics, not
    a dead mechanism. `k`/`v` stay asymmetric (token-only / state-only
    per interleaved slot) -- unlike Q, there is no reason to blur K/V's
    identity, since which slot attention lands on is exactly what
    distinguishes "trust this token" from "trust memory."

    `centers`/`log_sigmas` are per-window-position (length `window_size`)
    trainable `Tensor` leaves, owned by the caller's `WindowState` (see
    `curriculum.WindowState`/`advance_window`) -- NOT re-initialized
    here. `sigmas = exp(log_sigmas)` keeps sigma strictly positive via
    the ordinary autograd chain rule (see `sili.tensor.exp`/
    `gaussian_attention`). Per direct decision, training these relies
    entirely on plain backprop through the task loss plus
    `EnergyDynamics`' own `aux_loss` (returned below) once Phase 3 wires
    up `.backward()` -- no separate sparsity-pressure mechanism for
    attention spread, and no actor-critic hook here.

    `attn_out = o_proj(blended)` (pretrained, unchanged role). The
    residual `x_common_t + attn_out` is gated by `energy_dynamics`
    (caller-owned, persisted across calls the same way `carried_state`
    is -- see run_folded_recurrence) before being used both as this
    step's own residual (feeds the MLP below) and as `new_carried_state`
    for the NEXT token. This is the first place `EnergyDynamics` is
    wired into `sili_peridot` at all.

    Returns (delta [window_size, hidden] -- this step's own output, NOT
    yet added to x_common_t, same "caller accumulates it" contract
    apply_fold_step already uses; new_carried_state [window_size,
    hidden]; aux_loss -- a Tensor scalar from EnergyDynamics, for the
    caller to fold into a training loss once Phase 3 wires up backprop,
    safely ignorable in forward-only use).
    """
    hidden = cfg.hidden_size
    n_heads, n_kv_heads, head_dim = (cfg.num_attention_heads,
                                     cfg.num_key_value_heads, cfg.head_dim)
    groups = n_heads // n_kv_heads
    q_proj_out, kv_proj_out = cfg.q_proj_out, cfg.kv_proj_out

    ln_stack = np.stack(input_ln_weights[:window_size])                  # [window_size, hidden]
    x_common_stack = np.broadcast_to(x_common_t, (window_size, hidden))
    normed = rmsnorm(x_common_stack, ln_stack, cfg.rms_norm_eps)         # [window_size, hidden]
    carried_normed = rmsnorm(carried_state, ln_stack, cfg.rms_norm_eps)  # [window_size, hidden]

    normed_flat = normed.reshape(1, window_size * hidden)
    carried_flat = carried_normed.reshape(1, window_size * hidden)
    # Q draws from token AND carried state together -- per direct
    # correction: tying Q exclusively to the current token meant the
    # mechanism went dead (degenerate, content-blind) with no fresh
    # input, which breaks any "sleep"/consolidation-style internal
    # dynamics driven by EnergyDynamics' own exploration noise. q_proj
    # is linear, so proj(normed(token)) + proj(normed(state)) ==
    # proj(normed(token) + normed(state)) exactly -- summing once and
    # projecting once is equivalent and cheaper. With no real input,
    # normed(token) is just zero (RMSNorm of an all-zero vector is
    # zero, no special-casing needed) and Q still carries the state
    # contribution -- genuine internal dynamics, not a dead gate.
    qk_source_flat = (normed + carried_normed).reshape(1, window_size * hidden)

    q = _forward(window_layers[".self_attn.q_proj.weight"], qk_source_flat,
                 _density_for_suffix(activation_density, ".self_attn.q_proj.weight"))[0]
    k_new = _forward(window_layers[".self_attn.k_proj.weight"], normed_flat,
                     _density_for_suffix(activation_density, ".self_attn.k_proj.weight"))[0]
    k_state = _forward(window_layers[".self_attn.k_proj.weight"], carried_flat,
                       _density_for_suffix(activation_density, ".self_attn.k_proj.weight"))[0]
    v_new = _forward(window_layers[".self_attn.v_proj.weight"], normed_flat,
                     _density_for_suffix(activation_density, ".self_attn.v_proj.weight"))[0]
    v_state = _forward(window_layers[".self_attn.v_proj.weight"], carried_flat,
                       _density_for_suffix(activation_density, ".self_attn.v_proj.weight"))[0]

    q       = q.reshape(window_size, n_heads, head_dim)
    k_new   = k_new.reshape(window_size, n_kv_heads, head_dim)
    k_state = k_state.reshape(window_size, n_kv_heads, head_dim)
    v_new   = v_new.reshape(window_size, n_kv_heads, head_dim)
    v_state = v_state.reshape(window_size, n_kv_heads, head_dim)

    k_new_exp   = np.repeat(k_new,   groups, axis=1)  # [window_size, n_heads, head_dim]
    k_state_exp = np.repeat(k_state, groups, axis=1)
    v_new_exp   = np.repeat(v_new,   groups, axis=1)
    v_state_exp = np.repeat(v_state, groups, axis=1)

    # Interleave fresh-token/carried-state entries: index 2p = position
    # p's fresh-token (K,V), 2p+1 = its carried-state (K,V) -- see
    # docstring. centers[p] is defined in this same interleaved space
    # (own pair's midpoint at init is 2p+0.5, see advance_window).
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

    attn_out = _forward(window_layers[".self_attn.o_proj.weight"], blended_flat,
                        _density_for_suffix(activation_density, ".self_attn.o_proj.weight"))
    attn_out = attn_out.reshape(window_size, hidden)

    pre_gate_state = (x_common_stack + attn_out).reshape(-1).astype(np.float32)  # [window_size*hidden]
    gated_tensor, aux_loss, _actual_p = energy_dynamics(Tensor(pre_gate_state))
    new_carried_state = gated_tensor.data.reshape(window_size, hidden)

    normed2 = rmsnorm(new_carried_state, np.stack(post_attn_ln_weights[:window_size]), cfg.rms_norm_eps)
    normed2_flat = normed2.reshape(1, window_size * hidden)

    gate_mlp = _forward(window_layers[".mlp.gate_proj.weight"], normed2_flat,
                        _density_for_suffix(activation_density, ".mlp.gate_proj.weight"))
    up_mlp   = _forward(window_layers[".mlp.up_proj.weight"], normed2_flat,
                        _density_for_suffix(activation_density, ".mlp.up_proj.weight"))
    mlp_out = _forward(window_layers[".mlp.down_proj.weight"], silu(gate_mlp) * up_mlp,
                       _density_for_suffix(activation_density, ".mlp.down_proj.weight"))
    mlp_out = mlp_out.reshape(window_size, hidden)

    delta = (new_carried_state - x_common_stack) + mlp_out
    return delta, new_carried_state, aux_loss


def default_window_energy(percent_active: float = 0.25) -> EnergyDynamics:
    """Placeholder defaults for the window's EnergyDynamics gate, loosely
    matching SparseRNNCell's own percent_active-derived formula (see
    sili__new/sili/sparse_rnn.py's constructor) -- real tuning is Phase
    5's job (the eventual actor-critic controls energy drive); this just
    needs to be a safe, finite starting point."""
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


def default_window_gaussian_params(window_size: int) -> Tuple[Tensor, Tensor]:
    """Fresh, from-scratch `centers`/`log_sigmas` for a `window_size`-wide
    window: `center[p] = 2p + 0.5` (own fresh-token/carried-state pair's
    midpoint in apply_window_step's interleaved `2*window_size` key
    space), `log_sigma[p] = 0.0` (`sigma=1.0`) -- concentrates ~68% of
    attention mass on position p's own pair at init, while still
    reachable by neighbors. Matches `advance_window`'s own incremental
    per-position init exactly (see its docstring) -- this is for
    callers that want a whole window's worth at once (tests, or a
    from-scratch WindowState) rather than growing one position at a
    time."""
    centers = Tensor(np.array([2.0 * p + 0.5 for p in range(window_size)], dtype=np.float32))
    log_sigmas = Tensor(np.zeros(window_size, dtype=np.float32))
    return centers, log_sigmas


def run_folded_recurrence(
    x: np.ndarray,                          # [T, hidden] embedded input
    step_layers: List[Dict[str, object]],
    input_ln_weights: List[np.ndarray],
    post_attn_ln_weights: List[np.ndarray],
    final_norm_weight: np.ndarray,
    cfg: MiniCPM5Config,
    half_bandwidth: int,
    num_cpus: int = 4,
    activation_density: Union[_ActivationDensity, List[_ActivationDensity]] = None,
    window_state=None,               # curriculum.WindowState, or None -- see below
    window_activation_density: _ActivationDensity = None,
    window_energy: Optional[EnergyDynamics] = None,
    window_carried_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """state=0; for step: out=block(x+state); state+=out -- see
    RNNFoldedBlock.forward's docstring in sili__new for why this recurrence
    (not a plain 24-layer sequential replay) and why averaging/summing
    per-step outputs is not done here (skip_connection_outputs=False:
    final accumulated state is returned, RMSNorm'd, ready for lm_head).

    window_state=None (default): the PLAIN pre-window path, unchanged
    since B6 -- every position runs sequentially, exactly as it always
    has. Every existing caller/test keeps working unmodified.

    window_state=<a curriculum.WindowState-shaped object> (duck-typed,
    not imported here to avoid a curriculum<->sili_block import cycle --
    needs .window_size, .window_positions, .suffix_windows, .centers,
    .log_sigmas): positions BEFORE window_state.window_positions[-1]
    (the SMALLEST/earliest
    absolute index currently in the window -- window_positions[0] is
    instead the LARGEST/last-added, since curriculum.advance_window
    appends new positions in the order they enter the window: last
    fold-step first, then working backward) still run exactly this same
    plain sequential loop, over the WHOLE `[T, hidden]` batch at once,
    ending at a `state` -- this is what "positions outside the window
    compute sequentially, unchanged" means throughout the project plan,
    and per the MAJOR PIVOT (2026-08-02) this is untouched by the T=1
    window redesign -- `apply_fold_step` never sees a single token.

    Only ONCE INSIDE the window does processing switch to one token at a
    time (`window_size >= 2` -- `window_size == 1` bypasses this
    entirely, see below): `x_common = x + state` (the SAME starting
    input, per token, every window position sees) is walked token by
    token, threading each window position's own `carried_state` from one
    token to the next via `apply_window_step` -- see that function's
    docstring for the unified `gaussian_attention` mechanism (Phase 2.7)
    this now uses instead of causal attention over T (meaningless once a
    single call only ever sees one token). `centers`/`log_sigmas` come
    from `window_state` itself (persisted, growable trainable
    parameters -- see `curriculum.WindowState`/`advance_window`).
    `window_carried_state`/`window_energy` are caller-owned (matching
    how `window_state.suffix_windows` already is) so a caller CAN
    persist them across separate `run_folded_recurrence` calls if it
    wants continuity across sequences -- None (the default) starts
    fresh state/energy every call, the simplest and current choice
    pending Phase 3's real training loop clarifying whether cross-call
    persistence is actually needed.

    The return value generalizes from "final accumulated state (sum of
    every position's own delta, NOT including x -- see RNNFoldedBlock.
    forward's docstring in sili__new, x is only ever added back in to
    build the NEXT layer's input, never into the accumulator itself),
    RMSNorm'd" to "column-averaged prediction, RMSNorm'd", matching
    RNNFoldedBlock.forward's own skip_connection_outputs=True mode
    (return mean(outputs), each outputs[i] itself already excluding x):
    each window position's own column, PER TOKEN, = `state` (the
    PRE-WINDOW accumulated delta sum for that token, WITHOUT x) + that
    position's own delta output from `apply_window_step` at that token
    step -- NOT x_common (x_common includes x, and is only the correct
    thing to feed IN, never to add into the accumulated column).
    window_size==1 (B8a's stage 0) is provably identical to the plain
    path for this exact reason (only one column, `state + that
    position's delta` matches exactly what the plain path's own
    accumulator would hold after that same position) -- and is also
    handled by bypassing `apply_window_step`/the per-token loop entirely
    (window_state.window_size==1 uses step_layers[window_state.
    window_positions[0]] directly, over the whole T-batch, exactly like
    the pre-window loop), per curriculum.WindowState's own documented
    recommendation, since there's nothing for a combined matrix or a
    carried-state mechanism to usefully do with only one column.
    window_size>1 averages window_size columns instead of trusting only
    the last one, at every token.

    See _forward for activation_density; may also be a per-step list of
    length num_hidden_layers (each entry itself None/float/dict) to
    isolate which LAYERS tolerate sparsification, not just which
    projections -- applies to the PRE-WINDOW positions only when
    window_state is given (errors from top-k truncation compound through
    the fold-depth recurrence's accumulated state, so a layer near the
    start is not necessarily equivalent to the same layer near the end).
    window_activation_density is the window's own (single, not
    per-position) density, passed to apply_window_step."""
    T = x.shape[0]
    cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)
    per_step = isinstance(activation_density, list)
    if per_step and len(activation_density) != cfg.num_hidden_layers:
        raise ValueError(
            f"activation_density list has {len(activation_density)} entries, "
            f"expected cfg.num_hidden_layers={cfg.num_hidden_layers}")

    has_window = window_state is not None and window_state.window_size > 0
    pre_window_end = window_state.window_positions[-1] if has_window else cfg.num_hidden_layers

    state = np.zeros_like(x)
    for i in range(pre_window_end):
        step_density = activation_density[i] if per_step else activation_density
        out = apply_fold_step(
            x + state, step_layers[i], input_ln_weights[i], post_attn_ln_weights[i],
            cfg, cos, sin, half_bandwidth, num_cpus, step_density)
        state = state + out

    if not has_window:
        return rmsnorm(state, final_norm_weight, cfg.rms_norm_eps)

    x_common = x + state
    window_size = window_state.window_size
    positions = window_state.window_positions

    if window_size == 1:
        pos = positions[0]
        out = apply_fold_step(
            x_common, step_layers[pos], input_ln_weights[pos], post_attn_ln_weights[pos],
            cfg, cos, sin, half_bandwidth, num_cpus, window_activation_density)
        mean_column = state + out
    else:
        hidden = cfg.hidden_size
        energy = window_energy if window_energy is not None else default_window_energy()
        carried_state = (window_carried_state.copy() if window_carried_state is not None
                         else np.zeros((window_size, hidden), dtype=np.float32))
        window_ln = [input_ln_weights[p] for p in positions]
        window_post_ln = [post_attn_ln_weights[p] for p in positions]

        mean_column = np.empty((T, hidden), dtype=np.float32)
        for t in range(T):
            delta, carried_state, _aux_loss = apply_window_step(
                x_common[t], carried_state, window_state.suffix_windows, window_size,
                window_ln, window_post_ln, cfg, energy,
                window_state.centers, window_state.log_sigmas,
                num_cpus, window_activation_density)
            columns_t = state[t][None, :] + delta  # [window_size, hidden]
            mean_column[t] = columns_t.mean(axis=0)

    return rmsnorm(mean_column, final_norm_weight, cfg.rms_norm_eps)
