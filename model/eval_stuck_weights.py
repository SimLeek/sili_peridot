"""
sili_peridot/model/eval_stuck_weights.py
───────────────────────────────────────────
Are synapses the model's own importance signal (ci -- the combined
gradient+forward-contribution second moment DISLDO's RMSprop-style
update tracks, see sili__new's linear_disldo.hpp) has flagged as
important actually MOVING, or are they stuck? A synapse can end up
stuck for reasons genuinely worth distinguishing: its update rounds
away below FP4/FP8's quantization step (this project's own history has
multiple real bugs of exactly this shape -- the zero-escape/ULP-
rounding work earlier this session), `ci` itself has grown large enough
to over-damp the step (importance-as-optimizer doing its job TOO well),
or the lr is genuinely too small for that synapse's local gradient
scale (the same calibration issue eval_lr.py's find_optimal_lr exists
to catch at the whole-model level, here at single-synapse resolution).

Snapshot-diff based, not a live per-step hook: call
snapshot_multi_digit_state() before and after some real training
interval, then check_stuck_weights() on the two snapshots. Works
directly off each DISLDOLayer-family layer's own ._c.weights_vals/
.importance arrays -- no reconstruction needed (unlike eval_eigenvalues
.py's dense_weight_matrix, this doesn't need the layer treated as an
opaque linear map, it reads the sparse storage directly).

Relative/percentile-based thresholds, not fixed absolute tolerances --
useful movement magnitude depends heavily on lr/step-count/
architecture and there's no universal "small enough to count as stuck"
number (confirmed directly this session: a real weight update at an
under-calibrated lr can be 100x smaller than the same update at a
properly-calibrated one, see eval_lr.py's own docstring).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Snapshot = Tuple[np.ndarray, np.ndarray]  # (weights_vals, importance), parallel arrays


def snapshot_layer_state(layer) -> Snapshot:
    """layer must be a raw DISLDOLayer-style wrapper exposing ._c with
    .weights_vals/.importance (the sili__new C++ binding's own per-
    synapse parallel arrays) -- NOT a TrueMultiDigitLayer directly, see
    snapshot_multi_digit_state for that."""
    c = getattr(layer, "_c", None)
    if c is None:
        raise ValueError(
            "layer has no ._c -- not a raw DISLDOLayer-style wrapper "
            "(pass a single digit, or use snapshot_multi_digit_state "
            "for a TrueMultiDigitLayer)")
    return np.array(c.weights_vals, dtype=np.float64), np.array(c.importance, dtype=np.float64)


def snapshot_multi_digit_state(layer) -> List[Snapshot]:
    """TrueMultiDigitLayer holds n_stages separate DISLDOLayer digits,
    each with its own weights/importance -- snapshots ALL of them (not
    just stage 0), since a stuck synapse in any digit stage is a real
    stuck synapse. Falls back to a single snapshot for a layer that
    isn't a TrueMultiDigitLayer (has no .digits) but does have ._c
    directly."""
    digits = getattr(layer, "digits", None)
    if digits is not None:
        return [snapshot_layer_state(d) for d in digits]
    return [snapshot_layer_state(layer)]


@dataclass
class StuckWeightsReport:
    n_synapses: int
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
        """stuck_fraction relative to what pure chance would predict --
        1.0 means "no more stuck synapses than random overlap would
        produce", meaningfully above 1.0 (say 2x+) is the real signal
        that high-importance synapses are specifically failing to move,
        not just unlucky sampling."""
        if self.expected_stuck_fraction_if_independent <= 0:
            return float("nan")
        return self.stuck_fraction / self.expected_stuck_fraction_if_independent


def check_stuck_weights(before: List[Snapshot], after: List[Snapshot], *,
                         importance_percentile: float = 75.0,
                         movement_percentile: float = 25.0) -> StuckWeightsReport:
    """before/after: lists of (weights, importance) snapshots (one per
    digit/layer, from snapshot_multi_digit_state -- pass snapshots from
    MULTIPLE layers concatenated together for a whole-model check, or
    one layer's own list for a per-layer check), taken at two points
    separated by some real training interval.

    Flags a synapse STUCK if its importance at the BEFORE snapshot (the
    model's own belief about how much this synapse matters going INTO
    the interval) is in the top importance_percentile, but its realized
    |weight change| over the interval is in the bottom
    movement_percentile among ALL synapses (not just the high-
    importance ones). Requires stable connectivity between snapshots
    (same synapse count) -- raises if nnz changed (synaptogenesis ran
    in between), since there'd be no stable per-synapse correspondence
    to diff.
    """
    w_before = np.concatenate([b[0] for b in before])
    imp_before = np.concatenate([b[1] for b in before])
    w_after = np.concatenate([a[0] for a in after])
    if w_before.shape != w_after.shape:
        raise ValueError(
            f"before/after synapse count mismatch ({w_before.shape[0]} vs "
            f"{w_after.shape[0]}) -- synaptogenesis changed connectivity "
            f"between snapshots, this check needs stable connectivity "
            f"to diff per-synapse")

    delta_w = np.abs(w_after - w_before)
    imp_thresh = float(np.percentile(imp_before, importance_percentile))
    move_thresh = float(np.percentile(delta_w, movement_percentile))

    high_importance = imp_before >= imp_thresh
    low_movement = delta_w <= move_thresh
    stuck = high_importance & low_movement

    n = len(w_before)
    n_high = int(high_importance.sum())
    stuck_fraction = (float(stuck.sum()) / n_high) if n_high > 0 else 0.0

    return StuckWeightsReport(
        n_synapses=n,
        n_high_importance=n_high,
        n_stuck=int(stuck.sum()),
        stuck_fraction=stuck_fraction,
        expected_stuck_fraction_if_independent=movement_percentile / 100.0,
        mean_delta_w_for_high_importance=float(delta_w[high_importance].mean()) if n_high > 0 else 0.0,
        mean_delta_w_overall=float(delta_w.mean()),
        importance_threshold=imp_thresh,
        movement_threshold=move_thresh,
    )
