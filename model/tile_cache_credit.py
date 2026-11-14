"""Mechanism A of the KV-cache synaptic hold/credit plan: budgeted exact
credit for content rows still sitting in ``tile_cache``, using the real
gradient ``gaussian_attention``'s backward already computes on them for
free (no ``requires_grad`` gate on ``Tensor`` -- see JOURNAL.md's
2026-11-10 entry for the full derivation, the dose-response validation,
and why selection here is blind round-robin, not magnitude-ranked
(project memory ``project_dense_vs_sparse_mqar_confusion_matrix``).
See docs/research/toy_tile_recurrence_rmt.rst:tile_cache_exact_credit."""

import numpy as np


def apply_tile_cache_exact_credit(model, lr: float, k_rows: int = 1) -> int:
    """Call once per step, right after loss.backward(). No-op (returns 0)
    unless step_cached(requires_grad=True) ran this step. Rotation state
    lives on `model._tile_cache_credit_cursor` (lazy, same convention as
    self._energy/self._sleep_energy)."""
    dbg = model.last_debug
    cache_k_tensor = dbg.get("cache_k_tensor")
    cache_v_tensor = dbg.get("cache_v_tensor")
    cache_x_rows = dbg.get("cache_x_rows")
    if cache_k_tensor is None or cache_x_rows is None or cache_x_rows.shape[0] == 0:
        return 0

    live_rows = [r for r in range(cache_x_rows.shape[0]) if np.any(cache_x_rows[r] != 0.0)]
    if not live_rows:
        return 0

    cursor = getattr(model, "_tile_cache_credit_cursor", 0)
    n_corrected = 0
    for i in range(min(k_rows, len(live_rows))):
        r = live_rows[(cursor + i) % len(live_rows)]
        x_row = cache_x_rows[r : r + 1]
        if cache_k_tensor.grad is not None:
            dy_k = cache_k_tensor.grad[r : r + 1]
            if np.any(dy_k != 0.0):
                model.k_proj._c.backward(x_row, dy_k, lr, lr_per_row_nnz=True)
                n_corrected += 1
        if cache_v_tensor is not None and cache_v_tensor.grad is not None:
            dy_v = cache_v_tensor.grad[r : r + 1]
            if np.any(dy_v != 0.0):
                model.v_proj._c.backward(x_row, dy_v, lr, lr_per_row_nnz=True)
    model._tile_cache_credit_cursor = cursor + k_rows
    return n_corrected
