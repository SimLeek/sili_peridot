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
existing (already-trained) window matrix is preserved via csr_union --
not rebuilt from scratch. Stage 0 (window=1 position) needs no combined
matrix at all: it's just that position's own existing small layer.

RMSNorm/RoPE weights are never part of the 7 folded suffixes (B3/B5 never
prune or quantize them), so they're read directly from sparse_state as
plain float32 vectors, not built into any SparseLinearLayer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from sili import _cpu
from sili.sparse_rnn import fit_rank1_scale_envelope, _build_rectangular_banded_csr, csr_union
from sili.tensor import Tensor, banded_attention

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
    uses. csr_union expects true units, not FP4-stored ones. Only used
    by grow_window_layer's rank1 path now -- see _raw_stored_csr for the
    per_row fast path, which deliberately avoids this dequantization."""
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
    units, NOT multiplied by value_scale -- plus the per-row scale array,
    with no dequantization arithmetic at all. Used by grow_window_layer's
    per_row fast path (see its docstring) to reuse existing rows verbatim
    instead of round-tripping through true units and refitting a scale
    that, in per_row mode, provably can't have changed: a row's scale
    only depends on THAT row's own max_abs, and appending zero-valued
    band entries never changes a row's max_abs."""
    n_in = layer.n_inputs
    ptrs = np.asarray(layer.ptrs).astype(np.int64)
    idx  = np.asarray(layer.indices).astype(np.int32)
    stored = np.asarray(layer.weights_vals).astype(np.float32)
    row_scale = np.array([layer.get_value_scale(r) for r in range(n_in)], dtype=np.float32)
    return ptrs, idx, stored, row_scale


def _quantize_and_load(u_ptrs, u_idx, u_val, total_in, total_out, num_cpus,
                        value_scale_mode, rank1_iters):
    """Tail of grow_window_layer's rank1 fallback path only (per_row uses
    _grow_window_layer_per_row_fast instead, which never needs a full
    rescan -- see grow_window_layer's docstring): fit a fresh
    quantization scale over the WHOLE unioned matrix and load it into a
    new SparseLinearLayer. Same math build_step_layers/
    _build_step_layer_from_arrays uses, just applied to the combined
    window shape instead of one small matrix.

    KNOWN RISK, not resolved here, worth watching via Phase 4's
    reporting: a row's value_scale is fit from ITS OWN max_abs, which
    (since every row belongs to exactly one position's diagonal block)
    reflects that position's REAL pretrained weight magnitude -- likely
    much larger than a growing recurrent connection's early gradient
    steps. If so, a recurrent entry's small gradient could round back to
    zero under FP4's step size before ever becoming visible, the same
    class of "value_scale too coarse for a fresh connection to move"
    landmine build_fold_skip_layer hit once already (see JOURNAL.md).
    Not fixed speculatively -- Phase 4's "pre-seeded, still exactly
    zero" count is the real signal for whether this is worth a follow-up.
    """
    vals = u_val.copy()
    row_scales = np.ones(total_in, dtype=np.float32)
    col_scales = np.ones(total_out, dtype=np.float32)
    if value_scale_mode == "rank1":
        row_of_nnz = np.repeat(np.arange(total_in, dtype=np.int64), np.diff(u_ptrs))
        row_env, col_env = fit_rank1_scale_envelope(
            row_of_nnz, u_idx.astype(np.int64), np.abs(vals), total_in, total_out, n_iters=rank1_iters)
        row_scales = (row_env / _FP4_MAX).astype(np.float32)
        col_scales = col_env.astype(np.float32)
        combined = row_scales[row_of_nnz] * col_scales[u_idx]
        nz = combined > 0
        vals[nz] /= combined[nz]
    else:
        for r in range(total_in):
            start, end = int(u_ptrs[r]), int(u_ptrs[r + 1])
            if end > start:
                max_abs = float(np.abs(vals[start:end]).max())
                if max_abs > 0.0:
                    row_scales[r] = max_abs / _FP4_MAX
                    vals[start:end] /= row_scales[r]

    nnz = int(vals.shape[0])
    layer = _cpu.SparseLinearLayer(total_in, total_out, int(nnz * 1.3) + 64, num_cpus)
    layer.load_weights(u_ptrs, u_idx, vals)
    for r in range(total_in):
        if row_scales[r] != 1.0:
            layer.set_value_scale_raw(r, row_scales[r])
    for c in range(total_out):
        if col_scales[c] != 1.0:
            layer.set_output_scale_raw(c, col_scales[c])
    return layer


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


def _grow_window_layer_per_row_fast(
    new_position_layer, in_dim: int, out_dim: int, num_cpus: int, bw: int,
    existing_window_layer, existing_window_size: int,
    total_in: int, total_out: int, off_in: int, off_out: int,
) -> "_cpu.SparseLinearLayer":
    """value_scale_mode="per_row" fast path -- see grow_window_layer.
    Every row's scale is reused verbatim (old rows from
    existing_window_layer, new rows from new_position_layer's own
    already-fit scale); only the STRUCTURE (which columns are nonzero)
    changes, via plain disjoint concatenation, never a value-changing
    union or a max_abs rescan. Old-row content and new-position content
    live in strictly disjoint column ranges (old content < off_out <=
    new diagonal; new backward band < off_out <= new diagonal too), so
    no conflict-resolution is needed either -- only new rows' own two
    pieces (backward band + diagonal) need an explicit sort."""
    new_ptrs, new_idx, new_stored, new_row_scale = _raw_stored_csr(new_position_layer)
    assert new_ptrs.shape[0] - 1 == in_dim

    if existing_window_layer is not None:
        old_ptrs, old_idx, old_stored, old_row_scale = _raw_stored_csr(existing_window_layer)
        assert old_ptrs.shape[0] - 1 == off_in

    ptrs = np.zeros(total_in + 1, dtype=np.int64)
    row_scale = np.empty(total_in, dtype=np.float32)
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
    return layer


def grow_window_layer(
    new_position_layer: "_cpu.SparseLinearLayer",
    in_dim: int, out_dim: int, num_cpus: int = 4,
    recurrent_bandwidth: Optional[int] = None,
    value_scale_mode: str = "per_row", rank1_iters: int = 6,
    existing_window_layer: Optional["_cpu.SparseLinearLayer"] = None,
    existing_window_size: int = 0,
) -> "_cpu.SparseLinearLayer":
    """
    Add ONE position to the window's combined matrix. `new_position_layer`
    is that position's own already-built, already-quantized small layer
    (step_layers[i][suffix] -- see build_step_layers; no separate build
    path, no `desc`/pretrained-tensor access needed here at all).

    existing_window_layer=None (window growing from 0->1 positions):
    the new diagonal block IS the whole matrix. Positions are appended
    in window-growth order (index 0 = first position added to the
    window, i.e. the LAST fold-step under B8a's backward-growing
    curriculum) -- existing blocks' row/col offsets never shift as the
    window grows.

    Returns a NEW SparseLinearLayer (old one is not mutated in place --
    caller replaces its reference). Off-diagonal (recurrent/skip) band
    entries carry forward from `existing_window_layer` -- training
    progress on in-window recurrent connections is preserved across
    stage transitions, not discarded.

    recurrent_bandwidth: None (default) picks max(1, min(in_dim,
    out_dim) // 8) -- deliberately NOT scaled to 2*max(in_dim, out_dim)
    (an earlier version of this function did that, matching
    build_fold_skip_layer's own "one hop = out_dim" convention doubled).
    Measured directly at MiniCPM5's real dims (in_dim=1536,
    out_dim=4608): a bandwidth on that same order makes
    _build_rectangular_banded_csr's "band" cover the ENTIRE row width,
    i.e. fully DENSE, not sparse -- ~7M zero entries for a single
    position's block in one suffix alone, the "ton of memory at large
    layers" this was flagged for directly. This default keeps
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

    value_scale_mode="per_row" (default): fast path (see
    _grow_window_layer_per_row_fast) -- every row's scale only depends
    on ITS OWN max_abs, so growing the window never needs to
    dequantize+rescan already-placed rows, only construct the new
    position's own rows (reusing ITS already-fit scale verbatim too) and
    append zero band entries. "rank1": output_scale couples across the
    WHOLE matrix jointly (fit_rank1_scale_envelope fits row+col envelopes
    together), so a newly-opened column range genuinely changes the
    right scale for every existing column too -- falls back to the
    slower dequantize -> union -> refit path, since per-row's shortcut
    doesn't apply there.
    """
    assert (existing_window_layer is None) == (existing_window_size == 0), (
        "existing_window_layer and existing_window_size must agree: both "
        "absent (first position) or both present (growing further)")
    total_in  = (existing_window_size + 1) * in_dim
    total_out = (existing_window_size + 1) * out_dim
    off_in  = existing_window_size * in_dim
    off_out = existing_window_size * out_dim
    bw = recurrent_bandwidth if recurrent_bandwidth is not None else max(1, min(in_dim, out_dim) // 8)

    if value_scale_mode == "per_row":
        return _grow_window_layer_per_row_fast(
            new_position_layer, in_dim, out_dim, num_cpus, bw,
            existing_window_layer, existing_window_size,
            total_in, total_out, off_in, off_out)

    new_ptrs, new_idx, new_val = _extract_true_csr(new_position_layer)
    new_rows = new_ptrs.shape[0] - 1
    assert new_rows == in_dim, f"new position has {new_rows} rows, expected in_dim={in_dim}"

    diag_row_of_nnz = np.repeat(np.arange(in_dim, dtype=np.int64), np.diff(new_ptrs.astype(np.int64)))
    diag_rows = diag_row_of_nnz + off_in
    diag_cols = new_idx.astype(np.int64) + off_out
    diag_vals = new_val

    order = np.lexsort((diag_cols, diag_rows))
    diag_rows, diag_cols, diag_vals = diag_rows[order], diag_cols[order], diag_vals[order]
    diag_ptrs = np.searchsorted(diag_rows, np.arange(total_in + 1)).astype(np.int32)
    diag_idx  = diag_cols.astype(np.int32)
    diag_val  = diag_vals.astype(np.float32)

    rec_ptrs, rec_idx, rec_val = _build_rectangular_banded_csr(total_in, total_out, bw)

    u_ptrs, u_idx, u_val = csr_union(
        diag_ptrs, diag_idx, diag_val, rec_ptrs, rec_idx, rec_val,
        total_in, prefer="a", num_cpus=num_cpus)

    if existing_window_layer is not None:
        old_ptrs, old_idx, old_val = _extract_true_csr(existing_window_layer)
        old_total_in = existing_window_size * in_dim
        # Pad old ptrs to the new (larger) row count -- rows beyond the
        # old window contribute nothing from the old side, so every
        # appended ptr entry repeats the final (total) nnz count.
        pad = np.full(total_in - old_total_in, old_ptrs[-1], dtype=np.int32)
        old_ptrs_padded = np.concatenate([old_ptrs, pad])
        u_ptrs, u_idx, u_val = csr_union(
            old_ptrs_padded, old_idx, old_val, u_ptrs, u_idx, u_val,
            total_in, prefer="a", num_cpus=num_cpus)

    return _quantize_and_load(u_ptrs, u_idx, u_val, total_in, total_out, num_cpus,
                               value_scale_mode, rank1_iters)


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
) -> np.ndarray:
    """state=0; for step: out=block(x+state); state+=out -- see
    RNNFoldedBlock.forward's docstring in sili__new for why this recurrence
    (not a plain 24-layer sequential replay) and why averaging/summing
    per-step outputs is not done here (skip_connection_outputs=False:
    final accumulated state is returned, RMSNorm'd, ready for lm_head).

    This is the PLAIN, pre-window path -- unchanged since B6, and still
    exactly what every position outside the current B8a curriculum
    window uses (see module docstring). The window itself (retained
    per-position state, the combined per-suffix matrix from
    grow_window_layer, column-averaging over the window) is a distinct
    code path, not yet added here -- see Phase 3 in the project plan.

    See _forward for activation_density; may also be a per-step list of
    length num_hidden_layers (each entry itself None/float/dict) to
    isolate which LAYERS tolerate sparsification, not just which
    projections -- errors from top-k truncation compound through the
    fold-depth recurrence's accumulated state, so a layer near the start
    is not necessarily equivalent to the same layer near the end."""
    T = x.shape[0]
    cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)
    per_step = isinstance(activation_density, list)
    if per_step and len(activation_density) != cfg.num_hidden_layers:
        raise ValueError(
            f"activation_density list has {len(activation_density)} entries, "
            f"expected cfg.num_hidden_layers={cfg.num_hidden_layers}")

    state = np.zeros_like(x)
    for i in range(cfg.num_hidden_layers):
        step_density = activation_density[i] if per_step else activation_density
        out = apply_fold_step(
            x + state, step_layers[i], input_ln_weights[i], post_attn_ln_weights[i],
            cfg, cos, sin, half_bandwidth, num_cpus, step_density)
        state = state + out

    return rmsnorm(state, final_norm_weight, cfg.rms_norm_eps)
