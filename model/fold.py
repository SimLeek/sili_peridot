"""
sili_peridot/model/fold.py
─────────────────────────────
Folds MiniCPM5's 7 per-layer suffixes into per-suffix FoldedBlockDescriptors,
one suffix at a time, never bundled together.
See docs/research/fold.rst:fold.module_overview.
See docs/research/fold.rst:fold.band_half_width_rope_deferral.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from sili.conversion.rnn_fold import FoldedBlockDescriptor, fold_block_group
from sili.sparse_rnn import FoldedLayer

from .config import MiniCPM5Config

SUFFIXES: list[str] = [
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
]

# See docs/research/fold.rst:fold.expected_out_dim_shape_check.
_EXPECTED_OUT_DIM_PROPERTY: dict[str, str] = {
    ".self_attn.q_proj.weight": "q_proj_out",
    ".self_attn.k_proj.weight": "kv_proj_out",
    ".self_attn.v_proj.weight": "kv_proj_out",
    ".self_attn.o_proj.weight": "attn_out",
    ".mlp.gate_proj.weight": "mlp_hidden",
    ".mlp.up_proj.weight": "mlp_hidden",
    ".mlp.down_proj.weight": "mlp_out",
}


def fold_suffix(
    sparse_state: dict[str, dict],
    suffix: str,
    cfg: MiniCPM5Config,
    prefix: str = "model.layers.",
    band_half_width_override: int | None = None,
) -> FoldedBlockDescriptor:
    """Fold one suffix's per-layer tensors into a single FoldedBlockDescriptor.
    See docs/research/fold.rst:fold.module_overview."""
    names = [f"{prefix}{i}{suffix}" for i in range(cfg.num_hidden_layers)]
    missing = [n for n in names if n not in sparse_state]
    if missing:
        raise KeyError(f"missing {len(missing)} tensor(s) for suffix {suffix!r}, e.g. {missing[0]!r}")
    filtered = {n: sparse_state[n] for n in names}

    desc = fold_block_group(
        list(range(cfg.num_hidden_layers)),
        filtered,
        prefix,
        band_half_width_override=band_half_width_override,
    )

    expected = getattr(cfg, _EXPECTED_OUT_DIM_PROPERTY[suffix])
    actual = desc.out_dims.get(suffix)
    if actual != expected:
        raise ValueError(
            f"{suffix!r} folded to out_dim={actual}, expected {expected} "
            f"from MiniCPM5Config.{_EXPECTED_OUT_DIM_PROPERTY[suffix]} -- "
            f"shape mismatch somewhere upstream of folding"
        )
    return desc


def fold_all_suffixes(
    sparse_state: dict[str, dict],
    cfg: MiniCPM5Config,
    suffixes: list[str] = SUFFIXES,
    prefix: str = "model.layers.",
    band_half_width_override: int | None = None,
) -> dict[str, FoldedBlockDescriptor]:
    """Fold all suffixes independently: {suffix: descriptor}."""
    return {suffix: fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override) for suffix in suffixes}


def build_folded_layers(
    descriptors: dict[str, FoldedBlockDescriptor],
    learning_rate: float = 0.01,
    num_cpus: int = 4,
    value_scale_mode: str = "rank1",
) -> dict[str, FoldedLayer]:
    """B5: build one real FoldedLayer per suffix via
    FoldedLayer.from_descriptor. See
    docs/research/fold.rst:fold.build_folded_layers.rank1_value_scale for why
    value_scale_mode="rank1" is the default."""
    return {
        suffix: FoldedLayer.from_descriptor(
            desc, learning_rate=learning_rate, num_cpus=num_cpus, value_scale_mode=value_scale_mode
        )
        for suffix, desc in descriptors.items()
    }


def build_folded_layers_streaming(
    sparse_state: dict[str, dict],
    cfg: MiniCPM5Config,
    suffixes: list[str] = SUFFIXES,
    prefix: str = "model.layers.",
    learning_rate: float = 0.01,
    num_cpus: int = 4,
    band_half_width_override: int | None = None,
    value_scale_mode: str = "rank1",
) -> dict[str, FoldedLayer]:
    """Like build_folded_layers, but pops each suffix's raw tensors out of
    `sparse_state` (MUTATES it in place) immediately after folding, instead
    of holding all layers resident for the whole loop. Destructive by
    design -- use fold_all_suffixes if `sparse_state` must stay intact. See
    docs/research/fold.rst:fold.build_folded_layers_streaming.memory_discipline.
    """
    layers: dict[str, FoldedLayer] = {}
    for suffix in suffixes:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        for i in range(cfg.num_hidden_layers):
            del sparse_state[f"{prefix}{i}{suffix}"]
        layers[suffix] = FoldedLayer.from_descriptor(
            desc, learning_rate=learning_rate, num_cpus=num_cpus, value_scale_mode=value_scale_mode
        )
        del desc
    return layers


def _save_folded_layer_state_dict(suffix: str, layer: FoldedLayer, out_dir: str) -> str:
    """One .npz per suffix. See
    docs/research/fold.rst:fold.build_and_save_folded_layers.disk_offload_headroom.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{suffix.strip('.').replace('.', '_')}.npz")
    sd = layer.state_dict()[suffix]
    np.savez(path, **sd)
    return path


def build_and_save_folded_layers(
    sparse_state: dict[str, dict],
    cfg: MiniCPM5Config,
    out_dir: str,
    suffixes: list[str] = SUFFIXES,
    prefix: str = "model.layers.",
    learning_rate: float = 0.01,
    num_cpus: int = 4,
    band_half_width_override: int | None = None,
    value_scale_mode: str = "rank1",
) -> dict[str, str]:
    """Same one-suffix-at-a-time streaming discipline as
    build_folded_layers_streaming, but additionally serializes each
    FoldedLayer to disk and discards it immediately after, so peak memory
    never holds more than one suffix's built FoldedLayer at a time. See
    docs/research/fold.rst:fold.build_and_save_folded_layers.disk_offload_headroom.

    Returns {suffix: saved .npz path}.
    """
    paths: dict[str, str] = {}
    for suffix in suffixes:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        for i in range(cfg.num_hidden_layers):
            del sparse_state[f"{prefix}{i}{suffix}"]
        layer = FoldedLayer.from_descriptor(
            desc, learning_rate=learning_rate, num_cpus=num_cpus, value_scale_mode=value_scale_mode
        )
        del desc
        paths[suffix] = _save_folded_layer_state_dict(suffix, layer, out_dir)
        del layer
    return paths


def reference_fold_forward(
    descriptor: FoldedBlockDescriptor,
    suffix: str,
    x: np.ndarray,
) -> np.ndarray:
    """The unquantized analytic fold-sum for one suffix -- exactly
    FoldedLayer.forward's own math, with no FP4 rounding. See
    docs/research/fold.rst:fold.reference_fold_forward.quantization_isolation.
    """
    stacked = descriptor.stacked_weights[suffix]
    dense = stacked.to_dense().numpy().astype(np.float32)  # [n_folds*out_dim, in_dim]
    n_folds = descriptor.n_folds
    out_dim = descriptor.out_dims[suffix]
    batch = x.shape[0]
    raw = x.astype(np.float32) @ dense.T  # [batch, n_folds*out_dim]
    return raw.reshape(batch, n_folds, out_dim).sum(axis=1)


def _true_nnz(entry: dict) -> int:
    t = entry["csr"] if "csr" in entry else entry["raw"]
    if isinstance(t, torch.Tensor) and t.layout == torch.sparse_csr:
        return int(t.values().numel())
    return int((t != 0).sum())


@dataclass
class FoldReport:
    n_folds: int
    per_suffix_nnz_before: dict[str, int] = field(default_factory=dict)
    per_suffix_nnz_after: dict[str, int] = field(default_factory=dict)

    @property
    def lossless(self) -> bool:
        return self.per_suffix_nnz_before == self.per_suffix_nnz_after


def verify_lossless(
    sparse_state: dict[str, dict],
    descriptors: dict[str, FoldedBlockDescriptor],
    cfg: MiniCPM5Config,
    prefix: str = "model.layers.",
) -> FoldReport:
    """B4's own correctness check. See
    docs/research/fold.rst:fold.verify_lossless.nnz_invariant."""
    report = FoldReport(n_folds=cfg.num_hidden_layers)
    for suffix, desc in descriptors.items():
        before = sum(_true_nnz(sparse_state[f"{prefix}{i}{suffix}"]) for i in range(cfg.num_hidden_layers))
        after = int(desc.stacked_weights[suffix].values().numel())
        report.per_suffix_nnz_before[suffix] = before
        report.per_suffix_nnz_after[suffix] = after
    return report


def print_fold_report(report: FoldReport) -> None:
    CW = 30
    print(f"  {'Suffix':<{CW}} {'nnz before':>12} {'nnz after':>12} {'lossless':>9}")
    print("-" * (CW + 12 + 12 + 9 + 4))
    for suffix in report.per_suffix_nnz_before:
        before = report.per_suffix_nnz_before[suffix]
        after = report.per_suffix_nnz_after[suffix]
        print(f"  {suffix:<{CW}} {before:>12,} {after:>12,} {before == after!s:>9}")
    print(f"[summary] n_folds={report.n_folds}  lossless={report.lossless}")
