"""See docs/research/quantize.rst:module_overview."""

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
    """See docs/research/quantize.rst:build_quantized_dense_state_dict_streaming_discipline."""
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
