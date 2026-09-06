"""
sili_peridot/model/prune.py
─────────────────────────────
Prune MiniCPM5's checkpoint to a mixed sparse(CSR)/dense payload.

See docs/research/prune.rst:prune.module_overview and
docs/research/prune.rst:prune.default_target_sparsity_by_role.iterative_search_result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sili.conversion.prune_sensitivity import group_tensor_names_by_role
from sili.conversion.sparse_prune import (
    _keep_dense_reason,
    calibrate_min_abs_param,
    csr_bytes,
)

DEFAULT_TARGET_SPARSITY = 0.8

# See docs/research/prune.rst:prune.default_target_sparsity_by_role.iterative_search_result.
DEFAULT_TARGET_SPARSITY_BY_ROLE: dict[str, float] = {
    "embed_tokens": 0.8,
    "q_proj": 0.4,
    "k_proj": 0.4,
    "lm_head": 0.3,
    "gate_proj": 0.2,
    "up_proj": 0.25,
    "down_proj": 0.1,
    "o_proj": 0.1,
    "v_proj": 0.05,
}


@dataclass
class TensorPruneReport:
    name: str
    shape: tuple
    sparsity: float
    stored_format: str  # "sparse" or "dense"
    dense_reason: str = ""  # only set when stored_format == "dense"


@dataclass
class PruneReport:
    min_abs_param: float | dict[str, float]  # per-role dict if prune_state_dict_by_role
    tensors: list[TensorPruneReport] = field(default_factory=list)
    total_elements: int = 0
    total_pruned: int = 0
    total_dense_bytes: int = 0
    total_stored_bytes: int = 0

    @property
    def overall_sparsity(self) -> float:
        return self.total_pruned / self.total_elements if self.total_elements else 0.0

    @property
    def compression_ratio(self) -> float:
        return self.total_dense_bytes / self.total_stored_bytes if self.total_stored_bytes else float("inf")

    def n_sparse(self) -> int:
        return sum(1 for t in self.tensors if t.stored_format == "sparse")

    def n_dense(self) -> int:
        return sum(1 for t in self.tensors if t.stored_format == "dense")


def _record_pruned_tensor(
    name: str,
    param: torch.Tensor,
    threshold: float,
    min_sparsity: float,
    max_sparse_ratio: float,
    sparse_state: dict[str, dict],
    report: PruneReport,
) -> None:
    """See docs/research/prune.rst:prune.record_pruned_tensor.shared_format_decision."""
    n_elem = param.numel()
    dense_bytes = max(n_elem, 1) * 4

    if n_elem == 0 or param.ndim in {0, 1}:
        sparse_state[name] = {"raw": param.detach(), "shape": param.shape}
        report.tensors.append(TensorPruneReport(name, tuple(param.shape), 0.0, "dense", "scalar/vector"))
        report.total_elements += n_elem
        report.total_dense_bytes += dense_bytes
        report.total_stored_bytes += dense_bytes
        return

    t_f32 = param.detach().float()
    pruned = t_f32 if t_f32.ndim != 2 else t_f32 * (t_f32.abs() >= threshold)

    n_nz = int((pruned != 0).sum().item())
    n_pruned = n_elem - n_nz
    sparsity = n_pruned / n_elem

    report.total_elements += n_elem
    report.total_pruned += n_pruned

    reason = _keep_dense_reason(pruned, min_sparsity, max_sparse_ratio)
    if reason is not None:
        sparse_state[name] = {"raw": pruned, "shape": param.shape}
        stored_bytes = dense_bytes
        report.tensors.append(TensorPruneReport(name, tuple(param.shape), sparsity, "dense", reason))
    else:
        csr = pruned.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        sparse_state[name] = {"csr": csr, "shape": param.shape}
        stored_bytes = csr_bytes(csr)
        report.tensors.append(TensorPruneReport(name, tuple(param.shape), sparsity, "sparse"))

    report.total_dense_bytes += dense_bytes
    report.total_stored_bytes += stored_bytes


def prune_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_sparsity: float = DEFAULT_TARGET_SPARSITY,
    min_abs_param: float | None = None,
    min_sparsity: float = 0.33,
    max_sparse_ratio: float = 0.9,
    max_sample: int = 5_000_000,
) -> tuple[dict[str, dict], PruneReport]:
    """See docs/research/prune.rst:prune.prune_state_dict.reference_only.

    Returns (sparse_state, PruneReport); min_abs_param overrides calibration
    when given.
    """
    threshold = (
        min_abs_param
        if min_abs_param is not None
        else calibrate_min_abs_param(state_dict, target_sparsity=target_sparsity, max_sample=max_sample)
    )

    sparse_state: dict[str, dict] = {}
    report = PruneReport(min_abs_param=threshold)
    for name, param in state_dict.items():
        _record_pruned_tensor(name, param, threshold, min_sparsity, max_sparse_ratio, sparse_state, report)
    return sparse_state, report


_ROLE_NAMES = ["embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _role_of(group_key: str) -> str:
    """See docs/research/prune.rst:prune.role_of.fail_loud_on_unknown_role."""
    for role in _ROLE_NAMES:
        if role in group_key:
            return role
    raise ValueError(
        f"group {group_key!r} doesn't match any known MiniCPM5 tensor role "
        f"{_ROLE_NAMES} -- add it to DEFAULT_TARGET_SPARSITY_BY_ROLE and "
        f"_ROLE_NAMES, or this group would silently never get pruned"
    )


def prune_state_dict_by_role(
    state_dict: dict[str, torch.Tensor],
    target_sparsity_by_role: dict[str, float] = DEFAULT_TARGET_SPARSITY_BY_ROLE,
    min_sparsity: float = 0.33,
    max_sparse_ratio: float = 0.9,
    max_sample: int = 5_000_000,
) -> tuple[dict[str, dict], PruneReport]:
    """See docs/research/prune.rst:prune.prune_state_dict_by_role.grouping_and_per_role_calibration.

    The recommended path. Returns (sparse_state, PruneReport) with
    report.min_abs_param as a {role: threshold} dict.
    """
    groups = group_tensor_names_by_role(state_dict)
    thresholds_by_group: dict[str, float] = {}
    for gname, names in groups.items():
        role = _role_of(gname)
        target = target_sparsity_by_role[role]
        sub_sd = {n: state_dict[n] for n in names}
        thresholds_by_group[gname] = (
            calibrate_min_abs_param(sub_sd, target_sparsity=target, max_sample=max_sample) if target > 0.0 else 0.0
        )

    name_to_group = {n: g for g, names in groups.items() for n in names}
    sparse_state: dict[str, dict] = {}
    report = PruneReport(min_abs_param={_role_of(g): t for g, t in thresholds_by_group.items()})
    for name, param in state_dict.items():
        gname = name_to_group.get(name)
        threshold = thresholds_by_group[gname] if gname is not None else 0.0
        _record_pruned_tensor(name, param, threshold, min_sparsity, max_sparse_ratio, sparse_state, report)
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
        print(f"  {t.name:<{CW}} {t.shape!s:<{CS}} {t.sparsity:>8.1%} {t.stored_format:>7}{note}")
    print("-" * len(hdr))
    threshold_str = (
        f"{report.min_abs_param:.6e}" if isinstance(report.min_abs_param, float) else str(report.min_abs_param)
    )
    print(
        f"[summary] threshold={threshold_str}  "
        f"overall_sparsity={report.overall_sparsity:.1%}  "
        f"sparse={report.n_sparse()}  dense={report.n_dense()}  "
        f"compression={report.compression_ratio:.2f}x  "
        f"stored_MB={report.total_stored_bytes / 1e6:.1f}"
    )


def sparse_state_to_dense_state_dict(sparse_state: dict[str, dict]) -> dict[str, torch.Tensor]:
    """See docs/research/prune.rst:prune.sparse_state_to_dense_state_dict.round_trip_for_eval."""
    out = {}
    for name, entry in sparse_state.items():
        shape = entry["shape"]
        dense = entry["csr"].to_dense() if "csr" in entry else entry["raw"]
        out[name] = dense.reshape(shape)
    return out
