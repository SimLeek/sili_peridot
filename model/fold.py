"""
sili_peridot/model/fold.py
─────────────────────────────
Fold each of MiniCPM5's 7 per-layer 2-D suffixes (self_attn.q_proj,
k_proj, v_proj, o_proj, mlp.gate_proj, up_proj, down_proj) across all 24
layers INDEPENDENTLY -- each suffix gets its OWN FoldedBlockDescriptor,
never bundled together into one. sili__new's fold_block_group bundles
every suffix sharing a block-index range into ONE descriptor if given
the whole state dict at once -- FoldedLayer.forward sums every suffix in
its `layers` dict down to the FIRST suffix's out_dim, silently wrong
here since every suffix has a different out_dim (q=2048, k/v=256,
o=1536, gate/up=4608, down=1536). This module filters to one suffix at
a time before calling fold_block_group, specifically to avoid that.

band_half_width is NOT set correctly here on purpose: fold_block_group's
own auto-heuristic (infer_seq_len_from_attn_weight) assumes fixed
-position attention, which is wrong for MiniCPM5's RoPE. B6 is where the
real value gets decided (a practical context window for this
environment, not the full 131072 training context) -- this module
exposes band_half_width_override so B6 can supply it, defaulting to
None (the current, known-wrong-for-RoPE auto-heuristic) so folding
itself doesn't block on that decision.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sili.conversion.rnn_fold import fold_block_group, FoldedBlockDescriptor
from sili.sparse_rnn import FoldedLayer

from .config import MiniCPM5Config

SUFFIXES: List[str] = [
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
]

# Which MiniCPM5Config property each suffix's per-fold-step out_dim
# should match -- checked at fold time so a shape mismatch fails loudly
# here, not as a confusing error deep in FoldedLayer construction later.
_EXPECTED_OUT_DIM_PROPERTY: Dict[str, str] = {
    ".self_attn.q_proj.weight": "q_proj_out",
    ".self_attn.k_proj.weight": "kv_proj_out",
    ".self_attn.v_proj.weight": "kv_proj_out",
    ".self_attn.o_proj.weight": "attn_out",
    ".mlp.gate_proj.weight":    "mlp_hidden",
    ".mlp.up_proj.weight":      "mlp_hidden",
    ".mlp.down_proj.weight":    "mlp_out",
}


def fold_suffix(
    sparse_state: Dict[str, dict],
    suffix: str,
    cfg: MiniCPM5Config,
    prefix: str = "model.layers.",
    band_half_width_override: Optional[int] = None,
) -> FoldedBlockDescriptor:
    """
    Fold ONE suffix's per-layer tensors (e.g. every layer's own
    self_attn.q_proj.weight) into a single FoldedBlockDescriptor whose
    stacked_weights has exactly ONE key -- never combine multiple
    suffixes into one descriptor (see module docstring).
    """
    names = [f"{prefix}{i}{suffix}" for i in range(cfg.num_hidden_layers)]
    missing = [n for n in names if n not in sparse_state]
    if missing:
        raise KeyError(f"missing {len(missing)} tensor(s) for suffix {suffix!r}, "
                       f"e.g. {missing[0]!r}")
    filtered = {n: sparse_state[n] for n in names}

    desc = fold_block_group(
        list(range(cfg.num_hidden_layers)), filtered, prefix,
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
    sparse_state: Dict[str, dict],
    cfg: MiniCPM5Config,
    suffixes: List[str] = SUFFIXES,
    prefix: str = "model.layers.",
    band_half_width_override: Optional[int] = None,
) -> Dict[str, FoldedBlockDescriptor]:
    """Fold all 7 MiniCPM5 suffixes independently: {suffix: descriptor},
    each descriptor covering exactly one suffix's own weights stacked
    across all cfg.num_hidden_layers layers."""
    return {
        suffix: fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        for suffix in suffixes
    }


def build_folded_layers(
    descriptors: Dict[str, FoldedBlockDescriptor],
    learning_rate: float = 0.01,
    num_cpus: int = 4,
) -> Dict[str, FoldedLayer]:
    """
    B5: turn each suffix's FoldedBlockDescriptor into a real sili
    FoldedLayer (sili.sparse_rnn.FoldedLayer.from_descriptor -- already
    does the correct FP4 handling this needed: per-row pre-scale to
    max_abs/FP4_MAX, load_weights, set_value_scale_raw, plus
    set_importance_scale_raw(lr/FP4_MAX) so future importance updates
    are representable. No new library code required here -- B5's
    remaining work was calling it once per suffix and confirming it
    actually builds at MiniCPM5's real scale, not toy-Mistral scale.)
    One suffix at a time -- each descriptor's stacked_weights has
    exactly one key, so from_descriptor builds exactly one internal
    SparseLinearLayer per call, never bundling suffixes (see module
    docstring).
    """
    return {
        suffix: FoldedLayer.from_descriptor(desc, learning_rate=learning_rate, num_cpus=num_cpus)
        for suffix, desc in descriptors.items()
    }


def build_folded_layers_streaming(
    sparse_state: Dict[str, dict],
    cfg: MiniCPM5Config,
    suffixes: List[str] = SUFFIXES,
    prefix: str = "model.layers.",
    learning_rate: float = 0.01,
    num_cpus: int = 4,
    band_half_width_override: Optional[int] = None,
) -> Dict[str, FoldedLayer]:
    """
    Fold and build a real sili FoldedLayer for each suffix ONE AT A
    TIME, popping that suffix's 24 raw per-layer tensors out of
    `sparse_state` (MUTATES it in place) immediately after folding --
    so an already-processed suffix's memory is released before the next
    suffix's fold+construct begins, instead of holding all 219 tensors
    resident for the whole loop.

    Why this exists (measured, not assumed, on the real checkpoint --
    see JOURNAL.md): FoldedLayer.from_descriptor's own C++ construction
    adds ~0 marginal peak RSS beyond fold_suffix's own CSR-stacking step
    -- even for the largest suffix (mlp.gate_proj, budget~=170M,
    nnz~=136M). The real memory driver is holding raw per-layer dense
    tensors (already fully resident in RAM from B3's pruning, since most
    suffixes are still 80-93% dense at B3's validated per-role
    thresholds -- deeper sparsification is intentionally deferred to
    training-time synaptogenesis, not this conversion step) for suffixes
    that have ALREADY been folded and are no longer needed. A naive
    "fold each suffix in a loop, keep sparse_state untouched" version
    (fold_all_suffixes + build_folded_layers called on its output) lets
    that dead weight accumulate across the whole loop instead. Using
    this function is destructive to `sparse_state` by design -- pass a
    dict you don't need afterward, or use fold_all_suffixes if you need
    to keep sparse_state intact and can afford the extra memory.
    """
    layers: Dict[str, FoldedLayer] = {}
    for suffix in suffixes:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        for i in range(cfg.num_hidden_layers):
            del sparse_state[f"{prefix}{i}{suffix}"]
        layers[suffix] = FoldedLayer.from_descriptor(
            desc, learning_rate=learning_rate, num_cpus=num_cpus)
        del desc
    return layers


def _save_folded_layer_state_dict(suffix: str, layer: FoldedLayer, out_dir: str) -> str:
    """One .npz per suffix -- FoldedLayer.state_dict() is already a plain
    dict of numpy arrays (per-suffix weight/scale arrays plus
    n_folds/out_dim/lr), np.savez handles it directly without any
    torch/sili dependency to read back later."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{suffix.strip('.').replace('.', '_')}.npz")
    sd = layer.state_dict()[suffix]
    np.savez(path, **sd)
    return path


def build_and_save_folded_layers(
    sparse_state: Dict[str, dict],
    cfg: MiniCPM5Config,
    out_dir: str,
    suffixes: List[str] = SUFFIXES,
    prefix: str = "model.layers.",
    learning_rate: float = 0.01,
    num_cpus: int = 4,
    band_half_width_override: Optional[int] = None,
) -> Dict[str, str]:
    """
    Same one-suffix-at-a-time streaming discipline as
    build_folded_layers_streaming (MUTATES `sparse_state` in place,
    releasing each suffix's raw tensors as soon as it's folded), but
    additionally serializes each suffix's real FoldedLayer to disk and
    discards the live C++ object immediately after -- so peak memory
    never holds more than ONE suffix's built FoldedLayer at a time, on
    top of whatever's left of sparse_state. Confirmed on the real
    checkpoint (see JOURNAL.md): holding all 7 real FoldedLayer objects
    simultaneously (build_folded_layers_streaming's own return value)
    peaks at ~13.5GB on this 15GB machine -- workable but with barely
    any headroom left for anything else. This function exists to give
    that headroom back for the conversion step specifically; B7's real
    runtime will still need all 7 loaded together to run a forward
    pass, but building/verifying them doesn't.

    Returns {suffix: saved .npz path}.
    """
    paths: Dict[str, str] = {}
    for suffix in suffixes:
        desc = fold_suffix(sparse_state, suffix, cfg, prefix, band_half_width_override)
        for i in range(cfg.num_hidden_layers):
            del sparse_state[f"{prefix}{i}{suffix}"]
        layer = FoldedLayer.from_descriptor(desc, learning_rate=learning_rate, num_cpus=num_cpus)
        del desc
        paths[suffix] = _save_folded_layer_state_dict(suffix, layer, out_dir)
        del layer
    return paths


def reference_fold_forward(
    descriptor: FoldedBlockDescriptor, suffix: str, x: np.ndarray,
) -> np.ndarray:
    """
    The UNQUANTIZED analytic fold-sum for one suffix: x @ W^T per fold
    step, summed over the fold axis -- exactly FoldedLayer.forward's
    own math, computed directly from the exact float32 stacked CSR
    values with no FP4 rounding. Comparing this against the real
    (quantized) FoldedLayer.forward(x) on the same x isolates
    quantization's own effect, since both compute the identical fold
    -sum given the identical input -- the fold-approximation itself
    (same x fed to every virtual layer) contributes zero difference
    between the two, only quantization does.
    """
    stacked = descriptor.stacked_weights[suffix]
    dense = stacked.to_dense().numpy().astype(np.float32)   # [n_folds*out_dim, in_dim]
    n_folds = descriptor.n_folds
    out_dim = descriptor.out_dims[suffix]
    batch = x.shape[0]
    raw = x.astype(np.float32) @ dense.T                    # [batch, n_folds*out_dim]
    return raw.reshape(batch, n_folds, out_dim).sum(axis=1)


def _true_nnz(entry: dict) -> int:
    t = entry["csr"] if "csr" in entry else entry["raw"]
    if isinstance(t, torch.Tensor) and t.layout == torch.sparse_csr:
        return int(t.values().numel())
    return int((t != 0).sum())


@dataclass
class FoldReport:
    n_folds:                int
    per_suffix_nnz_before:  Dict[str, int] = field(default_factory=dict)
    per_suffix_nnz_after:   Dict[str, int] = field(default_factory=dict)

    @property
    def lossless(self) -> bool:
        return self.per_suffix_nnz_before == self.per_suffix_nnz_after


def verify_lossless(
    sparse_state: Dict[str, dict],
    descriptors: Dict[str, FoldedBlockDescriptor],
    cfg: MiniCPM5Config,
    prefix: str = "model.layers.",
) -> FoldReport:
    """B4's own correctness check: total nonzeros per suffix must be
    IDENTICAL before and after folding -- stacking must never lose or
    duplicate real weight values, only change their storage layout."""
    report = FoldReport(n_folds=cfg.num_hidden_layers)
    for suffix, desc in descriptors.items():
        before = sum(
            _true_nnz(sparse_state[f"{prefix}{i}{suffix}"])
            for i in range(cfg.num_hidden_layers)
        )
        after = int(desc.stacked_weights[suffix].values().numel())
        report.per_suffix_nnz_before[suffix] = before
        report.per_suffix_nnz_after[suffix]  = after
    return report


def print_fold_report(report: FoldReport) -> None:
    CW = 30
    print(f"  {'Suffix':<{CW}} {'nnz before':>12} {'nnz after':>12} {'lossless':>9}")
    print("-" * (CW + 12 + 12 + 9 + 4))
    for suffix in report.per_suffix_nnz_before:
        before = report.per_suffix_nnz_before[suffix]
        after  = report.per_suffix_nnz_after[suffix]
        print(f"  {suffix:<{CW}} {before:>12,} {after:>12,} {str(before == after):>9}")
    print(f"[summary] n_folds={report.n_folds}  lossless={report.lossless}")
