"""
sili_peridot/model/sili_model.py
Assembles the full model (embedding -> fold-depth recurrence -> RMSNorm
-> lm_head) and evaluates next-token prediction entirely through sili.
See docs/research/sili_model.rst:sili_model.module_overview.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from .config import MiniCPM5Config
from .eval_pruning import EvalResult
from .sili_block import _ActivationDensity, build_step_layers, run_folded_recurrence

_EmbedOrHead = np.ndarray | scipy.sparse.csr_matrix


def _to_dense_numpy(entry: dict) -> np.ndarray:
    """entry is a prune.py sparse_state value ({"csr": ...} or {"raw": ...})."""
    t = entry["csr"].to_dense() if "csr" in entry else entry["raw"]
    return t.float().numpy().copy()


def _to_sparse_or_dense(entry: dict) -> _EmbedOrHead:
    """entry is a prune.py sparse_state value. Kept as scipy CSR when B3
    pruned it (cheaper than dense at real density); {"raw": ...} entries
    fall back to dense, matching layernorm's raw convention.
    See docs/research/sili_model.rst:sili_model.embed_head_sparse_storage.
    """
    if "csr" not in entry:
        return entry["raw"].float().numpy().copy()
    t = entry["csr"]
    ptrs = t.crow_indices().numpy().astype(np.int32)
    idx = t.col_indices().numpy().astype(np.int32)
    vals = t.values().float().numpy()
    return scipy.sparse.csr_matrix((vals, idx, ptrs), shape=tuple(t.shape))


def build_sili_model(
    sparse_state: dict[str, dict],
    cfg: MiniCPM5Config,
    band_half_width_override=None,
    num_cpus: int = 4,
    value_scale_mode: str = "per_row",
    rank1_iters: int = 6,
) -> dict:
    """Pops embed_tokens/lm_head/final-norm and every fold step's layers
    out of sparse_state (mutates it, streaming discipline). Returns a
    dict bundling everything compute_logits_sili needs.
    See docs/research/sili_model.rst:sili_model.embed_head_sparse_storage.
    """
    embed_tokens = _to_sparse_or_dense(sparse_state.pop("model.embed_tokens.weight"))
    lm_head = _to_sparse_or_dense(sparse_state.pop("lm_head.weight"))
    final_norm = _to_dense_numpy(sparse_state.pop("model.norm.weight"))

    step_layers, input_ln, post_ln = build_step_layers(
        sparse_state,
        cfg,
        band_half_width_override=band_half_width_override,
        num_cpus=num_cpus,
        value_scale_mode=value_scale_mode,
        rank1_iters=rank1_iters,
    )

    return {
        "embed_tokens": embed_tokens,
        "lm_head": lm_head,
        "final_norm": final_norm,
        "step_layers": step_layers,
        "input_ln": input_ln,
        "post_ln": post_ln,
    }


def compute_logits_sili(
    token_ids: np.ndarray,  # [T] int
    sili_model: dict,
    cfg: MiniCPM5Config,
    half_bandwidth: int,
    num_cpus: int = 4,
    activation_density: _ActivationDensity | list[_ActivationDensity] = None,
) -> np.ndarray:
    """Returns [T, vocab_size] float32 logits.
    See docs/research/sili_model.rst:sili_model.activation_density_interface
    for activation_density semantics."""
    embed_tokens = sili_model["embed_tokens"]
    x = embed_tokens[token_ids]  # [T, hidden] -- cheap row gather either way
    if scipy.sparse.issparse(x):
        x = x.toarray()
    hidden = run_folded_recurrence(
        x,
        sili_model["step_layers"],
        sili_model["input_ln"],
        sili_model["post_ln"],
        sili_model["final_norm"],
        cfg,
        half_bandwidth,
        num_cpus,
        activation_density,
    )

    lm_head = sili_model["lm_head"]
    if scipy.sparse.issparse(lm_head):
        # See docs/research/sili_model.rst:sili_model.lm_head_sparse_matmul.
        return (lm_head @ hidden.T).T.astype(np.float32)
    return hidden @ lm_head.T


def _cross_entropy_and_accuracy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """logits: [N, vocab], targets: [N] int.
    See docs/research/sili_model.rst:sili_model.eval_parity."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss = -log_probs[np.arange(len(targets)), targets].mean()
    preds = logits.argmax(axis=-1)
    acc = float((preds == targets).mean())
    return float(loss), acc


def evaluate_next_token_prediction_sili(
    sili_model: dict,
    tokenizer,
    cfg: MiniCPM5Config,
    half_bandwidth: int,
    texts: list[str],
    num_cpus: int = 4,
    activation_density: _ActivationDensity | list[_ActivationDensity] = None,
) -> EvalResult:
    """sili-only counterpart to eval_pruning.evaluate_next_token_prediction.
    See docs/research/sili_model.rst:sili_model.eval_parity."""
    losses, accs = [], []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0].numpy()
        logits = compute_logits_sili(ids, sili_model, cfg, half_bandwidth, num_cpus, activation_density)
        loss, acc = _cross_entropy_and_accuracy(logits[:-1], ids[1:])
        losses.append(loss)
        accs.append(acc)
    return EvalResult(per_text_loss=losses, per_text_accuracy=accs)
