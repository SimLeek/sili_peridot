"""
sili_peridot/model/sili_model.py
──────────────────────────────────
B7: assemble the full model (embedding lookup -> B6's fold-depth
recurrence -> final RMSNorm -> lm_head) and evaluate next-token
prediction quality entirely through sili -- no torch forward pass
anywhere in this module, mirroring eval_pruning.evaluate_next_token_
prediction's exact loss/accuracy methodology so the two are directly
comparable.

embed_tokens/lm_head are never part of the 7 folded suffixes (B3 never
prunes them, B5 never quantizes them) -- read as plain float32 arrays
and applied via a numpy gather / matmul, same scope boundary as
sili_block.py's layernorm weights.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.sparse

from .config import MiniCPM5Config
from .eval_pruning import EvalResult
from .sili_block import _ActivationDensity, build_step_layers, run_folded_recurrence

_EmbedOrHead = Union[np.ndarray, scipy.sparse.csr_matrix]


def _to_dense_numpy(entry: dict) -> np.ndarray:
    """entry is a prune.py sparse_state value ({"csr": ...} or {"raw": ...})."""
    t = entry["csr"].to_dense() if "csr" in entry else entry["raw"]
    return t.float().numpy().copy()


def _to_sparse_or_dense(entry: dict) -> _EmbedOrHead:
    """
    B3's role-based thresholds DO prune embed_tokens/lm_head (both 2-D,
    ~20%/~70% density respectively on the real checkpoint) -- storing
    them fully dense regardless (the previous behavior) cost ~1.5GB for
    two tensors that compress to ~360MB combined at their real density.
    Keeps a real {"csr": ...} entry as a scipy CSR matrix (cheap row
    gather for embed_tokens, cheap sparse-dense matmul for lm_head) --
    scipy is already a transitive dependency (via sili__new), not a new
    one. {"raw": ...} entries (never pruned to CSR) fall back to dense,
    matching the 1-D layernorm vectors' own always-raw convention.
    """
    if "csr" not in entry:
        return entry["raw"].float().numpy().copy()
    t = entry["csr"]
    ptrs = t.crow_indices().numpy().astype(np.int32)
    idx  = t.col_indices().numpy().astype(np.int32)
    vals = t.values().float().numpy()
    return scipy.sparse.csr_matrix((vals, idx, ptrs), shape=tuple(t.shape))


def build_sili_model(
    sparse_state: Dict[str, dict],
    cfg: MiniCPM5Config,
    band_half_width_override=None,
    num_cpus: int = 4,
    value_scale_mode: str = "per_row",
    rank1_iters: int = 6,
) -> dict:
    """
    Pops embed_tokens/lm_head/final-norm and every fold step's real sili
    layers out of `sparse_state` (MUTATES it -- same streaming discipline
    as build_step_layers). Returns a dict bundling everything
    compute_logits_sili/evaluate_next_token_prediction_sili need.

    embed_tokens/lm_head are kept as scipy CSR matrices when B3 pruned
    them (the real checkpoint case), not densified -- see
    _to_sparse_or_dense.
    """
    embed_tokens = _to_sparse_or_dense(sparse_state.pop("model.embed_tokens.weight"))
    lm_head      = _to_sparse_or_dense(sparse_state.pop("lm_head.weight"))
    final_norm   = _to_dense_numpy(sparse_state.pop("model.norm.weight"))

    step_layers, input_ln, post_ln = build_step_layers(
        sparse_state, cfg, band_half_width_override=band_half_width_override,
        num_cpus=num_cpus, value_scale_mode=value_scale_mode, rank1_iters=rank1_iters)

    return dict(
        embed_tokens=embed_tokens, lm_head=lm_head, final_norm=final_norm,
        step_layers=step_layers, input_ln=input_ln, post_ln=post_ln,
    )


def compute_logits_sili(
    token_ids: np.ndarray,   # [T] int
    sili_model: dict,
    cfg: MiniCPM5Config,
    half_bandwidth: int,
    num_cpus: int = 4,
    activation_density: Union[_ActivationDensity, List[_ActivationDensity]] = None,
) -> np.ndarray:
    """Returns [T, vocab_size] float32 logits. activation_density: None
    (default) = dense forward throughout (current behavior); a float in
    (0, 1] sparsifies every projection's input activation to that
    per-token top-k density and routes through SISLDO's forward_sparse
    instead; a dict isolates specific suffixes; a list of length
    cfg.num_hidden_layers isolates specific fold steps -- see
    sili_block.apply_fold_step's _forward helper and
    run_folded_recurrence's per_step handling."""
    embed_tokens = sili_model["embed_tokens"]
    x = embed_tokens[token_ids]   # [T, hidden] -- cheap row gather either way
    if scipy.sparse.issparse(x):
        x = x.toarray()
    hidden = run_folded_recurrence(
        x, sili_model["step_layers"], sili_model["input_ln"], sili_model["post_ln"],
        sili_model["final_norm"], cfg, half_bandwidth, num_cpus, activation_density)

    lm_head = sili_model["lm_head"]
    if scipy.sparse.issparse(lm_head):
        # sparse [vocab, hidden] @ dense [hidden, T] -> dense [vocab, T],
        # never materializes lm_head densely -- .T to hidden @ lm_head.T's shape.
        return (lm_head @ hidden.T).T.astype(np.float32)
    return hidden @ lm_head.T


def _cross_entropy_and_accuracy(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """logits: [N, vocab], targets: [N] int. Standard numerically-stable
    softmax cross-entropy (mean over N) + top-1 accuracy -- same
    definition as HF's shifted labels=input_ids loss."""
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
    texts: List[str],
    num_cpus: int = 4,
    activation_density: Union[_ActivationDensity, List[_ActivationDensity]] = None,
) -> EvalResult:
    """sili-only counterpart to eval_pruning.evaluate_next_token_prediction
    -- same teacher-forced next-token loss/top-1-accuracy definition,
    same EvalResult, so perplexity/accuracy are directly comparable.
    See compute_logits_sili for activation_density."""
    losses, accs = [], []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0].numpy()
        logits = compute_logits_sili(ids, sili_model, cfg, half_bandwidth, num_cpus, activation_density)
        loss, acc = _cross_entropy_and_accuracy(logits[:-1], ids[1:])
        losses.append(loss)
        accs.append(acc)
    return EvalResult(per_text_loss=losses, per_text_accuracy=accs)
