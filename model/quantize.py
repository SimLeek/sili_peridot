"""
sili_peridot/model/quantize.py
─────────────────────────────
Simulate B5's FP4 quantization (sili__new's FoldedLayer.from_descriptor,
sili/lib/headers/fp4quant.hpp's 16-level lookup table) on MiniCPM5's
per-layer suffix weights, entirely in torch/numpy -- no sili runtime
involved. Same "isolate one variable, measure real next-token quality"
methodology as model/prune.py + model/eval_pruning.py (B3b), applied to
the quantization step instead of the pruning step -- see
model/eval_quantization.py for the next-token comparison this feeds.

TWO schemes are simulated here, matching from_descriptor's two
value_scale_mode options (sili__new PR #10):

  "per_row" (from_descriptor's ORIGINAL scheme, kept for regression
  documentation -- see PER_ROW functions below): one scale per INPUT
  FEATURE, shared across ALL 24 layers' folded output rows for that
  suffix. Found catastrophic on the real checkpoint: next-token
  accuracy 0.482 -> ~0.09-0.12 (~200x perplexity increase), with NO
  bug found in the simulation (unit-tested, and the shared scale isn't
  dominated by a rogue outlier layer -- see JOURNAL.md).

  "rank1" (the fix, RANK1 functions below, now from_descriptor's
  default): a genuine per-OUTPUT scale too, via a rank-1 (outer
  -product) envelope fit (sili__new's fit_rank1_scale_envelope) --
  found necessary because per-output magnitude within one folded layer
  varies by a min/max ratio as low as ~0.05-0.10, which a per-row-only
  scale can't represent at all. Recovers to ~0.297 accuracy -- real
  progress (perplexity ~36x lower than "per_row"), though still short
  of "small drop". See JOURNAL.md for the full investigation, including
  why a finer per-layer-only scale and percentile-based clipping were
  tried and rejected (percentile-clipping made it WORSE, not better).

Both share the same real 16-level FP4 table and the same "only NONZERO
elements are quantized" rule (zeros represent "no synapse" in the
sparse encoding, not a real weight to round) -- they differ only in
how the scale is computed.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .config import MiniCPM5Config
from .fold import SUFFIXES, fold_suffix

FP4_MAX = 6.0

# Mirrors fp4quant.hpp's FP4_TABLE exactly, minus the NaN slot (index 8
# in the C++ table) -- a real weight is never the nearest match to NaN.
FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def fp4_round(values: np.ndarray) -> np.ndarray:
    """Nearest-neighbour quantize to sili's 15 real (non-NaN) FP4
    levels -- matches fp4quant.hpp's fp4_quantize."""
    flat = values.reshape(-1, 1).astype(np.float32)
    idx = np.abs(flat - FP4_TABLE[None, :]).argmin(axis=1)
    return FP4_TABLE[idx].reshape(values.shape)


def compute_input_column_scale(stacked_csr: torch.Tensor, fp4_max: float = FP4_MAX) -> np.ndarray:
    """
    Replicates FoldedLayer.from_descriptor's exact per-row scale
    computation: transpose the stacked [n_folds*out_dim, in_dim] CSR to
    [in_dim, n_folds*out_dim], then one scale per input feature (row of
    the transposed matrix) = max(|values in that row|) / fp4_max. A row
    with no nonzero values keeps scale=1.0 (from_descriptor's own
    fallback -- never divides by zero).
    """
    dense_t = stacked_csr.to_dense().t().numpy()   # [in_dim, n_folds*out_dim]
    in_dim = dense_t.shape[0]
    scale = np.ones(in_dim, dtype=np.float32)
    max_abs = np.abs(dense_t).max(axis=1)
    nonzero = max_abs > 0.0
    scale[nonzero] = max_abs[nonzero] / fp4_max
    return scale


def simulate_fp4_quantize_layer(
    weight: torch.Tensor, input_scale: np.ndarray, fp4_max: float = FP4_MAX,
) -> torch.Tensor:
    """
    Quantize ONE layer's own [out_dim, in_dim] weight tensor using a
    shared per-input-column scale (from compute_input_column_scale) --
    input feature j uses column j's scale, matching from_descriptor's
    per-row treatment of the transposed stacked matrix. Only nonzero
    entries are quantized (zeros are "no synapse", not a value to
    round).
    """
    w = weight.detach().float().numpy()
    mask = w != 0
    scale = input_scale[None, :]           # broadcast over out_dim rows
    scaled = np.divide(w, scale, out=np.zeros_like(w), where=mask)
    quantized = fp4_round(scaled) * scale
    out = np.where(mask, quantized, w)
    return torch.from_numpy(out.astype(np.float32)).reshape(weight.shape)


def fit_rank1_scale_envelope(abs_mat: np.ndarray, n_iters: int = 6):
    """
    Pure-Python mirror of sili__new's sili.sparse_rnn.fit_rank1_scale_envelope
    (kept as a separate copy here so this module's simulation doesn't need
    to import sili -- see module docstring for why this exists: a
    per-row-only scale wastes most of FP4's resolution on any output well
    below that row's max). Alternating max-fit producing a rank-1
    (outer-product) envelope: row_scale[r] * col_scale[c] >= abs_mat[r,c]
    for every entry, using O(rows+cols) parameters instead of
    O(rows*cols). Returns (row_scale, col_scale), float32 -- deliberately
    NOT float64: row_scale/col_scale's dtype determines what dtype every
    `abs_mat / scale` broadcast upcasts to internally, so float64 here
    would silently double the transient memory of every iteration for
    the largest suffixes (materializing a float64-sized copy of abs_mat
    each time) even if abs_mat itself is stored as float32 -- exactly
    what OOM-killed this machine once with a real HF model also loaded
    (see JOURNAL.md). A max/divide envelope fit has no numerical
    stability need for float64 (unlike an accumulating sum).
    """
    n_rows, n_cols = abs_mat.shape
    col_scale = np.ones(n_cols, dtype=np.float32)
    row_scale = np.ones(n_rows, dtype=np.float32)
    for _ in range(n_iters):
        with np.errstate(divide='ignore', invalid='ignore'):
            row_scale = np.nanmax(np.where(col_scale > 0, abs_mat / col_scale[None, :], 0.0), axis=1)
        row_scale = np.maximum(row_scale, 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_scale = np.nanmax(np.where(row_scale[:, None] > 0, abs_mat / row_scale[:, None], 0.0), axis=0)
        col_scale = np.maximum(col_scale, 1e-12)
    return row_scale, col_scale


def simulate_fp4_quantize_layer_rank1(
    weight: torch.Tensor, input_scale: np.ndarray, output_scale_slice: np.ndarray,
    fp4_max: float = FP4_MAX,
) -> torch.Tensor:
    """
    Rank-1 counterpart to simulate_fp4_quantize_layer: quantize ONE
    layer's own [out_dim, in_dim] weight tensor using BOTH a shared
    per-input scale (row_scale/fp4_max, from fit_rank1_scale_envelope,
    shared across all 24 folded layers) AND that layer's own slice of
    the per-output scale (col_scale, one value per output neuron in
    THIS layer -- a slice of the full stacked envelope's column vector,
    since folding concatenates all 24 layers' outputs along that axis).
    """
    w = weight.detach().float().numpy()
    mask = w != 0
    effective_scale = np.outer(output_scale_slice, input_scale) / fp4_max
    scaled = np.divide(w, effective_scale, out=np.zeros_like(w), where=mask)
    quantized = fp4_round(scaled) * effective_scale
    out = np.where(mask, quantized, w)
    return torch.from_numpy(out.astype(np.float32)).reshape(weight.shape)


def compute_suffix_rank1_scales_streaming(
    sparse_state: Dict[str, dict],
    cfg: MiniCPM5Config,
    suffixes: List[str] = SUFFIXES,
    prefix: str = "model.layers.",
    band_half_width_override=None,
    rank1_iters: int = 6,
) -> Dict[str, tuple]:
    """
    Rank-1 counterpart to compute_suffix_scales_streaming: folds one
    suffix at a time (same memory discipline -- pops sparse_state as it
    goes, discards each descriptor before the next), fitting a rank-1
    envelope on each suffix's stacked matrix instead of a plain per-row
    max. Returns {suffix: (input_scale, output_scale)} -- input_scale is
    shared across all cfg.num_hidden_layers layers (length in_dim),
    output_scale spans the full stacked output axis (length
    n_folds*out_dim) and must be SLICED per layer (see
    apply_rank1_quantization).
    """
    from .fold import _EXPECTED_OUT_DIM_PROPERTY
    scales: Dict[str, tuple] = {}
    for suffix in suffixes:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        stacked = desc.stacked_weights[suffix]
        dense_t = stacked.to_dense().t().numpy().astype(np.float32)   # [in_dim, n_folds*out_dim]
        in_scale, out_scale = fit_rank1_scale_envelope(np.abs(dense_t), n_iters=rank1_iters)
        scales[suffix] = (in_scale, out_scale)
        for i in range(cfg.num_hidden_layers):
            del sparse_state[f"{prefix}{i}{suffix}"]
        del desc, dense_t
    return scales


def apply_rank1_quantization(
    dense_state_dict: Dict[str, torch.Tensor],
    rank1_scales: Dict[str, tuple],
    cfg: MiniCPM5Config,
    suffixes: List[str] = SUFFIXES,
    prefix: str = "model.layers.",
) -> Dict[str, torch.Tensor]:
    """
    Rank-1 counterpart to apply_quantization: returns a COPY of
    dense_state_dict with every one of the 7*24=168 per-layer suffix
    tensors replaced by its rank-1 FP4-quantization-simulated version.
    `rank1_scales` is compute_suffix_rank1_scales_streaming's output.
    """
    from .fold import _EXPECTED_OUT_DIM_PROPERTY
    out = dict(dense_state_dict)
    for suffix in suffixes:
        in_scale, out_scale = rank1_scales[suffix]
        out_dim = getattr(cfg, _EXPECTED_OUT_DIM_PROPERTY[suffix])
        for i in range(cfg.num_hidden_layers):
            name = f"{prefix}{i}{suffix}"
            layer_out_scale = out_scale[i * out_dim:(i + 1) * out_dim]
            out[name] = simulate_fp4_quantize_layer_rank1(out[name], in_scale, layer_out_scale)
    return out


def compute_suffix_scales(
    descriptors: Dict[str, "FoldedBlockDescriptor"],
    suffixes: List[str] = SUFFIXES,
) -> Dict[str, np.ndarray]:
    """
    Extract just the per-input-column scale vector for each suffix from
    its FoldedBlockDescriptor -- small (one float per input feature,
    e.g. 1536 or 4608 floats), unlike `descriptors` itself which holds
    each suffix's full stacked CSR (hundreds of MB to ~1.8GB for the
    largest suffixes). Call this, then discard `descriptors`, BEFORE
    building a full quantized dense state dict -- holding both the full
    descriptors AND a real HF model AND a dense state dict copy
    simultaneously is what caused a real OOM on this machine (see
    JOURNAL.md); apply_quantization only needs these small scale
    vectors, not the descriptors themselves.
    """
    return {
        suffix: compute_input_column_scale(descriptors[suffix].stacked_weights[suffix])
        for suffix in suffixes
    }


def compute_suffix_scales_streaming(
    sparse_state: Dict[str, dict],
    cfg: MiniCPM5Config,
    suffixes: List[str] = SUFFIXES,
    prefix: str = "model.layers.",
    band_half_width_override=None,
) -> Dict[str, np.ndarray]:
    """
    Same real result as compute_suffix_scales(fold_all_suffixes(...)),
    but folds ONE suffix at a time and discards that suffix's
    descriptor (and pops its raw tensors out of `sparse_state`, which
    this MUTATES in place) before moving to the next -- mirrors
    fold.build_folded_layers_streaming's own discipline.

    Needed on top of that fix: fold_all_suffixes alone (no real
    FoldedLayer construction at all, just building all 7 descriptors)
    already OOM-killed this machine when a real HF model was ALSO
    loaded at the same time for the next-token comparison in
    eval_quantization -- holding all 7 suffixes' full stacked CSRs
    (hundreds of MB to ~1.8GB each for the largest) simultaneously,
    on top of the HF model's own ~4.3GB float32 footprint and the
    already-pruned sparse_state, exceeded 15GB (see JOURNAL.md). This
    function never holds more than one suffix's descriptor at a time.
    """
    scales: Dict[str, np.ndarray] = {}
    for suffix in suffixes:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        scales[suffix] = compute_input_column_scale(desc.stacked_weights[suffix])
        for i in range(cfg.num_hidden_layers):
            del sparse_state[f"{prefix}{i}{suffix}"]
        del desc
    return scales


def apply_quantization(
    dense_state_dict: Dict[str, torch.Tensor],
    scales: Dict[str, np.ndarray],
    cfg: MiniCPM5Config,
    suffixes: List[str] = SUFFIXES,
    prefix: str = "model.layers.",
) -> Dict[str, torch.Tensor]:
    """
    Returns a COPY of dense_state_dict with every one of the 7*24=168
    per-layer suffix tensors replaced by its FP4-quantization-simulated
    version -- everything else (embed_tokens, lm_head, norms) passes
    through unchanged, since B5's fold only covers these 7 suffixes.
    `scales` is compute_suffix_scales's own output (one real
    per-input-column scale vector per suffix, derived from that
    suffix's actual folded/stacked weights) -- this function itself
    never needs the descriptors, only the small scale vectors, so
    callers can free the (much larger) descriptors before this step.
    """
    out = dict(dense_state_dict)
    for suffix in suffixes:
        scale = scales[suffix]
        for i in range(cfg.num_hidden_layers):
            name = f"{prefix}{i}{suffix}"
            out[name] = simulate_fp4_quantize_layer(out[name], scale)
    return out


def quantize_suffixes_in_state_dict(
    dense_state_dict: Dict[str, torch.Tensor],
    descriptors: Dict[str, "FoldedBlockDescriptor"],
    cfg: MiniCPM5Config,
    suffixes: List[str] = SUFFIXES,
    prefix: str = "model.layers.",
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper composing compute_suffix_scales + apply_quantization
    in one call -- fine for tests/small-scale use where holding
    `descriptors` a little longer doesn't matter. For the real checkpoint
    alongside a loaded HF model, prefer calling compute_suffix_scales
    then apply_quantization directly so `descriptors` can be freed
    in between (see their docstrings).
    """
    scales = compute_suffix_scales(descriptors, suffixes)
    return apply_quantization(dense_state_dict, scales, cfg, suffixes, prefix)
