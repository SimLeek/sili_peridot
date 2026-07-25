"""
sili_peridot/model/prune.py
─────────────────────────────
Prune MiniCPM5's checkpoint to a mixed sparse(CSR)/dense payload.

Doesn't reuse sili__new's toy-Mistral pruning defaults uncritically (see
todolist.md Phase B3): empirically, target_sparsity=0.5 (sparse_prune's
own default) leaves almost the entire model dense here -- CSR only wins
over dense once a tensor's OWN sparsity clears ~70%, given
_keep_dense_reason's 12-bytes-per-nonzero (value + int64 column index)
vs. 4-bytes-per-element dense estimate; at 50% sparsity CSR costs MORE
than dense (6 bytes/original-element equivalent vs. 4). Verified on the
real checkpoint: target_sparsity=0.5 -> 2/170 eligible tensors go
sparse, ~1.00x compression; target_sparsity=0.8 -> 127/170 sparse,
~1.65x compression. DEFAULT_TARGET_SPARSITY below reflects that finding,
not sili__new's own default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from sili.conversion.sparse_prune import (
    calibrate_min_abs_param, _keep_dense_reason, csr_bytes,
)

DEFAULT_TARGET_SPARSITY = 0.8


@dataclass
class TensorPruneReport:
    name:          str
    shape:         tuple
    sparsity:      float
    stored_format: str   # "sparse" or "dense"
    dense_reason:  str = ""   # only set when stored_format == "dense"


@dataclass
class PruneReport:
    min_abs_param:      float
    tensors:             List[TensorPruneReport] = field(default_factory=list)
    total_elements:       int = 0
    total_pruned:         int = 0
    total_dense_bytes:    int = 0
    total_stored_bytes:   int = 0

    @property
    def overall_sparsity(self) -> float:
        return self.total_pruned / self.total_elements if self.total_elements else 0.0

    @property
    def compression_ratio(self) -> float:
        return (self.total_dense_bytes / self.total_stored_bytes
               if self.total_stored_bytes else float("inf"))

    def n_sparse(self) -> int:
        return sum(1 for t in self.tensors if t.stored_format == "sparse")

    def n_dense(self) -> int:
        return sum(1 for t in self.tensors if t.stored_format == "dense")


def prune_state_dict(
    state_dict: Dict[str, torch.Tensor],
    target_sparsity: float = DEFAULT_TARGET_SPARSITY,
    min_abs_param: Optional[float] = None,
    min_sparsity: float = 0.33,
    max_sparse_ratio: float = 0.9,
    max_sample: int = 5_000_000,
) -> "tuple[Dict[str, dict], PruneReport]":
    """
    Prune `state_dict` to a {name: {"csr": tensor, "shape": ...}} or
    {name: {"raw": tensor, "shape": ...}} payload (same entry shapes
    sili__new's rnn_fold.py / sparse_runtime.py already expect via the
    "raw"/"csr" keys), plus a PruneReport for B3a-style verification
    (actual density, not just "it ran").

    Vectors (1-D) and higher-rank tensors always stay dense (matches
    sili__new's _keep_dense_reason rules exactly) -- LayerNorm weights
    here, no biases/qk-norm tensors exist in this checkpoint at all (see
    checkpoint.verify_checkpoint_matches_config).

    min_abs_param: explicit threshold, bypassing calibration entirely if
    given (same priority convention as sili__new's sparsify_model --
    explicit always wins). Mainly for tests that need a deterministic,
    hand-picked threshold rather than whatever the calibrated percentile
    happens to be.
    """
    threshold = (min_abs_param if min_abs_param is not None else
                calibrate_min_abs_param(
                    state_dict, target_sparsity=target_sparsity, max_sample=max_sample))

    sparse_state: Dict[str, dict] = {}
    report = PruneReport(min_abs_param=threshold)

    for name, param in state_dict.items():
        n_elem = param.numel()
        dense_bytes = max(n_elem, 1) * 4

        if n_elem == 0 or param.ndim == 0 or param.ndim == 1:
            sparse_state[name] = {"raw": param.detach(), "shape": param.shape}
            report.tensors.append(TensorPruneReport(
                name, tuple(param.shape), 0.0, "dense", "scalar/vector"))
            report.total_elements  += n_elem
            report.total_dense_bytes  += dense_bytes
            report.total_stored_bytes += dense_bytes
            continue

        t_f32 = param.detach().float()
        if t_f32.ndim != 2:
            pruned = t_f32
        else:
            pruned = t_f32 * (t_f32.abs() >= threshold)

        n_nz     = int((pruned != 0).sum().item())
        n_pruned = n_elem - n_nz
        sparsity = n_pruned / n_elem

        report.total_elements += n_elem
        report.total_pruned   += n_pruned

        reason = _keep_dense_reason(pruned, min_sparsity, max_sparse_ratio)
        if reason is not None:
            sparse_state[name] = {"raw": pruned, "shape": param.shape}
            stored_bytes = dense_bytes
            report.tensors.append(TensorPruneReport(
                name, tuple(param.shape), sparsity, "dense", reason))
        else:
            csr = pruned.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
            sparse_state[name] = {"csr": csr, "shape": param.shape}
            stored_bytes = csr_bytes(csr)
            report.tensors.append(TensorPruneReport(
                name, tuple(param.shape), sparsity, "sparse"))

        report.total_dense_bytes  += dense_bytes
        report.total_stored_bytes += stored_bytes

    return sparse_state, report


def print_report(report: PruneReport) -> None:
    """Human-readable per-tensor + summary table, mirroring sili__new's
    own sparsify_model() console output."""
    CW, CS = 45, 18
    hdr = f"  {'Layer':<{CW}} {'Shape':<{CS}} {'Sparsity':>9} {'Fmt':>7}"
    print(hdr)
    print("-" * len(hdr))
    for t in report.tensors:
        note = f"  [{t.dense_reason}]" if t.dense_reason else ""
        print(f"  {t.name:<{CW}} {str(t.shape):<{CS}} {t.sparsity:>8.1%} "
              f"{t.stored_format:>7}{note}")
    print("-" * len(hdr))
    print(f"[summary] threshold={report.min_abs_param:.6e}  "
          f"overall_sparsity={report.overall_sparsity:.1%}  "
          f"sparse={report.n_sparse()}  dense={report.n_dense()}  "
          f"compression={report.compression_ratio:.2f}x  "
          f"stored_MB={report.total_stored_bytes/1e6:.1f}")
