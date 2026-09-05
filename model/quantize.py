"""
sili_peridot/model/quantize.py
─────────────────────────────
B5's real FP4 quantization of MiniCPM5's per-layer suffix weights,
using sili__new's actual FoldedLayer.from_descriptor (rank-1 or
per-row quantization scale, sili__new PR #10) -- not a Python/numpy
simulation of it. Builds one suffix's real FoldedLayer at a time,
reads back its true post-quantization weight values, and discards it
before moving to the next suffix.
"""

from __future__ import annotations

import numpy as np
import torch
from sili.sparse_rnn import FoldedLayer

from .config import MiniCPM5Config
from .fold import SUFFIXES, fold_suffix


def build_quantized_dense_state_dict_streaming(
    sparse_state: dict[str, dict],
    cfg: MiniCPM5Config,
    suffixes: list[str] = SUFFIXES,
    prefix: str = "model.layers.",
    value_scale_mode: str = "rank1",
    band_half_width_override: int | None = None,
    rank1_iters: int = 6,
    learning_rate: float = 0.01,
    num_cpus: int = 4,
) -> dict[str, torch.Tensor]:
    """
    Fold+quantize one suffix at a time (MUTATES `sparse_state`, popping
    each suffix's raw tensors out as soon as it's folded -- same
    discipline as fold.build_folded_layers_streaming). Returns
    {layer_name: dense tensor} for every suffix*layer, holding at most
    one suffix's real FoldedLayer and one [n_in, n_out] reconstructed
    array at a time.

    Reads the real post-quantization weight back from the built layer
    via its zero-copy ptrs/indices/weights_vals plus per-row
    get_value_scale/per-col get_output_scale (true_w = weights_vals *
    value_scale[row] * output_scale[col], see cpu_backend.cpp) -- never
    densifies the stacked [n_in, n_out] matrix more than once, and only
    for this suffix's own reconstruction, not for the scale fit itself
    (that's sili__new's own sparse-triplet fit_rank1_scale_envelope,
    already used internally by from_descriptor).
    """
    out: dict[str, torch.Tensor] = {}
    for suffix in suffixes:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        for i in range(cfg.num_hidden_layers):
            del sparse_state[f"{prefix}{i}{suffix}"]
        out_dim = desc.out_dims[suffix]

        layer = FoldedLayer.from_descriptor(
            desc,
            learning_rate=learning_rate,
            num_cpus=num_cpus,
            value_scale_mode=value_scale_mode,
            rank1_iters=rank1_iters,
        )
        del desc

        raw = layer._sili_layers[suffix]
        n_in, n_out = raw.n_inputs, raw.n_outputs
        ptrs = np.asarray(raw.ptrs)
        col = np.asarray(raw.indices)
        row = np.repeat(np.arange(n_in, dtype=np.int64), np.diff(ptrs))
        row_scale = np.array([raw.get_value_scale(r) for r in range(n_in)], dtype=np.float32)
        col_scale = np.array([raw.get_output_scale(c) for c in range(n_out)], dtype=np.float32)
        true_vals = np.asarray(raw.weights_vals) * row_scale[row] * col_scale[col]

        dense_t = np.zeros((n_in, n_out), dtype=np.float32)
        dense_t[row, col] = true_vals
        for i in range(cfg.num_hidden_layers):
            out[f"{prefix}{i}{suffix}"] = torch.from_numpy(dense_t[:, i * out_dim : (i + 1) * out_dim].T.copy())
        del layer, dense_t, true_vals
    return out
