"""
sili_peridot/model/prune.py
─────────────────────────────
Prune MiniCPM5's checkpoint to a mixed sparse(CSR)/dense payload.

A SINGLE global threshold (prune_state_dict, DEFAULT_TARGET_SPARSITY)
turned out to be the wrong tool for this model, discovered the hard way
by actually measuring next-token prediction quality (model/eval_pruning.py)
before this landed, not by CSR-shape/sparsity-percentage reasoning alone:

1. Doesn't reuse sili__new's toy-Mistral pruning defaults uncritically
   (see todolist.md Phase B3): target_sparsity=0.5 (sparse_prune's own
   default) leaves almost the entire model dense here -- CSR only wins
   over dense once a tensor's OWN sparsity clears ~70%, given
   _keep_dense_reason's 12-bytes-per-nonzero estimate vs. 4 bytes/element
   dense. DEFAULT_TARGET_SPARSITY=0.8 gets real compression (~1.65x,
   127/170 tensors sparse) at this global level.

2. But at target_sparsity=0.8, next-token accuracy collapses to 0.0 (vs.
   0.503 dense) with NO retraining involved -- a single global threshold
   destroys the model long before it reaches CSR-viable sparsity.
   Per-tensor-ROLE sensitivity varies enormously (embed_tokens tolerated
   90% fine; v_proj -- architecturally near-identical to k_proj, which
   tolerated 70% -- collapsed already past ~25%), and isolated per-role
   thresholds compound MUCH worse once combined (see
   sili.conversion.prune_sensitivity.stepwise_cumulative_eval). See
   sili_peridot/JOURNAL.md for the full search.

DEFAULT_TARGET_SPARSITY_BY_ROLE / prune_state_dict_by_role is the actual
result of that search (iterative_threshold_search, sili__new PR #8):
combined next-token accuracy 0.482 (search set) / 0.478 (independent
held-out set) vs. a 0.503 dense baseline -- real quality preserved, at
the cost of most groups being well below CSR-viable sparsity for now.
That's intentional at this stage (see JOURNAL.md): the actual memory win
this whole pipeline is chasing is Phase B4's folding (24 layers -> 1),
which doesn't need genuine sparsity, just CSR-shaped tensors -- deeper
sparsification is left to training-time synaptogenesis, not this
conversion-time step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from sili.conversion.sparse_prune import (
    calibrate_min_abs_param, _keep_dense_reason, csr_bytes,
)
from sili.conversion.prune_sensitivity import group_tensor_names_by_role

DEFAULT_TARGET_SPARSITY = 0.8

# The validated result of the group-sensitivity + iterative-search
# calibration (see module docstring and JOURNAL.md). Keys are ROLE names
# (as returned by _role_of below), not raw group keys from
# group_tensor_names_by_role -- those embed a regex-derived prefix/suffix
# that isn't a stable, human-writable identifier.
DEFAULT_TARGET_SPARSITY_BY_ROLE: Dict[str, float] = {
    "embed_tokens": 0.8,
    "q_proj":       0.4,
    "k_proj":       0.4,
    "lm_head":      0.3,
    "gate_proj":    0.2,
    "up_proj":      0.25,
    "down_proj":    0.1,
    "o_proj":       0.1,
    "v_proj":       0.05,
}


@dataclass
class TensorPruneReport:
    name:          str
    shape:         tuple
    sparsity:      float
    stored_format: str   # "sparse" or "dense"
    dense_reason:  str = ""   # only set when stored_format == "dense"


@dataclass
class PruneReport:
    min_abs_param:      "float | Dict[str, float]"   # per-role dict if prune_state_dict_by_role
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


def _record_pruned_tensor(
    name: str, param: torch.Tensor, threshold: float,
    min_sparsity: float, max_sparse_ratio: float,
    sparse_state: Dict[str, dict], report: PruneReport,
) -> None:
    """Shared per-tensor logic: zero below `threshold`, decide sparse vs.
    dense storage, and record both into `sparse_state`/`report` in place.
    Used by both prune_state_dict (one global threshold for every
    tensor) and prune_state_dict_by_role (a different threshold per
    tensor, looked up by role) so the format-decision/reporting logic
    isn't duplicated between them."""
    n_elem = param.numel()
    dense_bytes = max(n_elem, 1) * 4

    if n_elem == 0 or param.ndim == 0 or param.ndim == 1:
        sparse_state[name] = {"raw": param.detach(), "shape": param.shape}
        report.tensors.append(TensorPruneReport(
            name, tuple(param.shape), 0.0, "dense", "scalar/vector"))
        report.total_elements  += n_elem
        report.total_dense_bytes  += dense_bytes
        report.total_stored_bytes += dense_bytes
        return

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


def prune_state_dict(
    state_dict: Dict[str, torch.Tensor],
    target_sparsity: float = DEFAULT_TARGET_SPARSITY,
    min_abs_param: Optional[float] = None,
    min_sparsity: float = 0.33,
    max_sparse_ratio: float = 0.9,
    max_sample: int = 5_000_000,
) -> "tuple[Dict[str, dict], PruneReport]":
    """
    Prune `state_dict` with ONE global threshold for every eligible
    tensor, to a {name: {"csr": tensor, "shape": ...}} or {name: {"raw":
    tensor, "shape": ...}} payload (same entry shapes sili__new's
    rnn_fold.py / sparse_runtime.py already expect via the "raw"/"csr"
    keys), plus a PruneReport for B3a-style verification (actual
    density, not just "it ran").

    Kept for reference/comparison -- see module docstring for why
    prune_state_dict_by_role is the actual recommended path; a single
    global threshold cannot get real CSR compression without destroying
    next-token prediction quality on this model.

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
        _record_pruned_tensor(name, param, threshold, min_sparsity, max_sparse_ratio,
                              sparse_state, report)
    return sparse_state, report


_ROLE_NAMES = ["embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]


def _role_of(group_key: str) -> str:
    """Map a group_tensor_names_by_role() key (an internal, regex-derived
    identifier) to one of MiniCPM5's own known tensor roles. Raises if a
    group doesn't match any known role -- silently skipping an unknown
    tensor role would mean it never gets pruned at all without anyone
    noticing."""
    for role in _ROLE_NAMES:
        if role in group_key:
            return role
    raise ValueError(
        f"group {group_key!r} doesn't match any known MiniCPM5 tensor role "
        f"{_ROLE_NAMES} -- add it to DEFAULT_TARGET_SPARSITY_BY_ROLE and "
        f"_ROLE_NAMES, or this group would silently never get pruned"
    )


def prune_state_dict_by_role(
    state_dict: Dict[str, torch.Tensor],
    target_sparsity_by_role: Dict[str, float] = DEFAULT_TARGET_SPARSITY_BY_ROLE,
    min_sparsity: float = 0.33,
    max_sparse_ratio: float = 0.9,
    max_sample: int = 5_000_000,
) -> "tuple[Dict[str, dict], PruneReport]":
    """
    Prune `state_dict` with a SEPARATE calibrated threshold per tensor
    role (see module docstring for why: a single global threshold
    destroys this model's next-token prediction quality long before it
    reaches CSR-viable sparsity). This is the recommended path, not
    prune_state_dict.

    Groups tensors via sili__new's group_tensor_names_by_role (repeated
    per-layer tensors grouped by shared suffix, e.g. every layer's own
    self_attn.q_proj.weight; embed_tokens/lm_head/norm each their own
    singleton group), calibrates ONE threshold per group on that group's
    own tensors, then reuses the same per-tensor sparse/dense
    format-decision logic as prune_state_dict.

    Returns the same (sparse_state, PruneReport) shape as
    prune_state_dict -- report.min_abs_param is a {role: threshold} dict
    here instead of a single float.
    """
    groups = group_tensor_names_by_role(state_dict)
    thresholds_by_group: Dict[str, float] = {}
    for gname, names in groups.items():
        role = _role_of(gname)
        target = target_sparsity_by_role[role]
        sub_sd = {n: state_dict[n] for n in names}
        thresholds_by_group[gname] = (
            calibrate_min_abs_param(sub_sd, target_sparsity=target, max_sample=max_sample)
            if target > 0.0 else 0.0
        )

    name_to_group = {n: g for g, names in groups.items() for n in names}
    sparse_state: Dict[str, dict] = {}
    report = PruneReport(min_abs_param={_role_of(g): t for g, t in thresholds_by_group.items()})
    for name, param in state_dict.items():
        gname = name_to_group.get(name)
        threshold = thresholds_by_group[gname] if gname is not None else 0.0
        _record_pruned_tensor(name, param, threshold, min_sparsity, max_sparse_ratio,
                              sparse_state, report)
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
    threshold_str = (f"{report.min_abs_param:.6e}" if isinstance(report.min_abs_param, float)
                     else str(report.min_abs_param))
    print(f"[summary] threshold={threshold_str}  "
          f"overall_sparsity={report.overall_sparsity:.1%}  "
          f"sparse={report.n_sparse()}  dense={report.n_dense()}  "
          f"compression={report.compression_ratio:.2f}x  "
          f"stored_MB={report.total_stored_bytes/1e6:.1f}")


def sparse_state_to_dense_state_dict(sparse_state: Dict[str, dict]) -> Dict[str, torch.Tensor]:
    """
    Convert a prune_state_dict / prune_state_dict_by_role payload (each
    entry {"csr": tensor, "shape": ...} or {"raw": tensor, "shape": ...})
    back into a plain {name: tensor} state dict with each tensor's
    ORIGINAL shape restored -- directly loadable via
    model.load_state_dict, e.g. for eval_pruning.compare_dense_vs_pruned
    to check a pruning decision's actual effect on model quality.
    """
    out = {}
    for name, entry in sparse_state.items():
        shape = entry["shape"]
        if "csr" in entry:
            dense = entry["csr"].to_dense()
        else:
            dense = entry["raw"]
        out[name] = dense.reshape(shape)
    return out
