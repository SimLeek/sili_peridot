"""
sili_peridot/model/eval_stuck_weights.py
───────────────────────────────────────────
Is the model's own importance signal (`ci`) an accurate predictor of
which synapses are stuck vs actually moving under training? See
docs/research/eval_stuck_weights.rst:eval_stuck_weights.module_overview
for the research question and findings, and
docs/research/eval_stuck_weights.rst:eval_stuck_weights.snapshot_row_col_keying
for why the snapshot-diff format below is keyed by (row, col) instead of
raw array position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (row, col) -> (weight, importance)
Snapshot = dict[tuple[int, int], tuple[float, float]]


def snapshot_layer_state(layer) -> Snapshot:
    """layer must be a raw DISLDOLayer-style wrapper exposing ._c (NOT a
    TrueMultiDigitLayer directly, see snapshot_multi_digit_state). See
    docs/research/eval_stuck_weights.rst:eval_stuck_weights.snapshot_row_col_keying."""
    c = getattr(layer, "_c", None)
    if c is None:
        raise ValueError(
            "layer has no ._c -- not a raw DISLDOLayer-style wrapper "
            "(pass a single digit, or use snapshot_multi_digit_state "
            "for a TrueMultiDigitLayer)"
        )
    ptrs = np.asarray(c.ptrs)
    indices = np.asarray(c.indices)
    weights = np.asarray(c.weights_vals, dtype=np.float64)
    importance = np.asarray(c.importance, dtype=np.float64)
    n_rows = len(ptrs) - 1
    snap: Snapshot = {}
    for r in range(n_rows):
        start, end = int(ptrs[r]), int(ptrs[r + 1])
        for k in range(start, end):
            col = int(indices[k])
            snap[(r, col)] = (float(weights[k]), float(importance[k]))
    return snap


def snapshot_multi_digit_state(layer) -> list[Snapshot]:
    """Snapshots ALL of a TrueMultiDigitLayer's digit stages, or falls
    back to a single snapshot for a plain ._c-bearing layer. See
    docs/research/eval_stuck_weights.rst:eval_stuck_weights.multi_digit_fallback."""
    digits = getattr(layer, "digits", None)
    if digits is not None:
        return [snapshot_layer_state(d) for d in digits]
    return [snapshot_layer_state(layer)]


@dataclass
class StuckWeightsReport:
    n_synapses: int  # size of the before/after KEY INTERSECTION -- see n_new/n_died
    n_new: int  # present in `after` but not `before` (woke up / grew in)
    n_died: int  # present in `before` but not `after` (pruned / went dead)
    n_high_importance: int
    n_stuck: int
    stuck_fraction: float  # n_stuck / n_high_importance (0.0 if none high-importance)
    expected_stuck_fraction_if_independent: float  # baseline if importance/movement were unrelated
    mean_delta_w_for_high_importance: float
    mean_delta_w_overall: float
    importance_threshold: float
    movement_threshold: float

    @property
    def excess_stuck_ratio(self) -> float:
        """See docs/research/eval_stuck_weights.rst:eval_stuck_weights.excess_stuck_ratio_signal."""
        if self.expected_stuck_fraction_if_independent <= 0:
            return float("nan")
        return self.stuck_fraction / self.expected_stuck_fraction_if_independent

    @property
    def churn_fraction(self) -> float:
        """See docs/research/eval_stuck_weights.rst:eval_stuck_weights.churn_fraction_signal."""
        if self.n_synapses <= 0:
            return float("nan")
        return (self.n_new + self.n_died) / self.n_synapses


def check_stuck_weights(
    before: list[Snapshot],
    after: list[Snapshot],
    *,
    importance_percentile: float = 75.0,
    movement_percentile: float = 25.0,
) -> StuckWeightsReport:
    """before/after: lists of {(row,col): (weight,importance)} snapshots
    (one per digit/layer, from snapshot_multi_digit_state), taken at two
    points separated by some real training interval. See
    docs/research/eval_stuck_weights.rst:eval_stuck_weights.check_stuck_weights_semantics
    for the STUCK definition and why appeared/disappeared synapses are
    excluded rather than counted."""
    w_before_list, imp_before_list, delta_list = [], [], []
    n_new = n_died = 0
    for b, a in zip(before, after, strict=False):
        common = b.keys() & a.keys()
        n_new += len(a.keys() - b.keys())
        n_died += len(b.keys() - a.keys())
        for key in common:
            w0, imp0 = b[key]
            w1, _ = a[key]
            w_before_list.append(w0)
            imp_before_list.append(imp0)
            delta_list.append(abs(w1 - w0))

    n = len(w_before_list)
    if n == 0:
        raise ValueError(
            "no synapses present in BOTH before and after snapshots -- "
            "nothing to compare (connectivity changed completely, or "
            "empty snapshots were passed in)"
        )

    imp_before = np.array(imp_before_list, dtype=np.float64)
    delta_w = np.array(delta_list, dtype=np.float64)

    imp_thresh = float(np.percentile(imp_before, importance_percentile))
    move_thresh = float(np.percentile(delta_w, movement_percentile))

    high_importance = imp_before >= imp_thresh
    low_movement = delta_w <= move_thresh
    stuck = high_importance & low_movement

    n_high = int(high_importance.sum())
    stuck_fraction = (float(stuck.sum()) / n_high) if n_high > 0 else 0.0

    return StuckWeightsReport(
        n_synapses=n,
        n_new=n_new,
        n_died=n_died,
        n_high_importance=n_high,
        n_stuck=int(stuck.sum()),
        stuck_fraction=stuck_fraction,
        expected_stuck_fraction_if_independent=movement_percentile / 100.0,
        mean_delta_w_for_high_importance=float(delta_w[high_importance].mean()) if n_high > 0 else 0.0,
        mean_delta_w_overall=float(delta_w.mean()),
        importance_threshold=imp_thresh,
        movement_threshold=move_thresh,
    )
