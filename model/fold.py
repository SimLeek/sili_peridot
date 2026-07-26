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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from sili.conversion.rnn_fold import fold_block_group, FoldedBlockDescriptor

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
