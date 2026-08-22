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

Confirmed directly this session (conversation): under deterministic
rounding at toy scale (and at every width tested up to 1024, an 8x-
plus sweep), essentially every already-live synapse is stuck at
mean|delta_w|=0.0. Under stochastic rounding, dead (weight=0,
importance=0) synapses DO wake up over time (nnz grows) -- but whether
stochastic rounding ALSO helps already-LIVE synapses move more (not
just wakes dead ones) needs a clean before/after diff that survives
connectivity CHANGING between snapshots, since stochastic rounding's
whole point is that connectivity isn't stable.

Snapshot-diff based, not a live per-step hook: call
snapshot_multi_digit_state() before and after some real training
interval, then check_stuck_weights() on the two snapshots. Snapshots
are keyed by (row, col) -- NOT raw array position -- specifically so a
snapshot pair survives nnz changing between them (new synapses waking
up, in sili__new terms, insert into the middle of a row's CSR data,
shifting every later array position; diffing by array index instead of
by stable key silently compared the WRONG pairs of synapses whenever
that happened, confirmed directly as the reason an earlier attempt at
this comparison failed with a shape-mismatch guard instead of a wrong
answer -- only checking BOTH shapes AND (implicitly) index-for-index
correspondence, via a hard reject, is what caught it). The comparison
here only uses the INTERSECTION of keys present at both snapshots (see
n_new/n_died on the report for how much churn that intersection is
throwing away) -- a genuinely fair "did an already-alive-at-both-points
synapse move" comparison, independent of how many synapses appeared or
disappeared in between.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# (row, col) -> (weight, importance)
Snapshot = Dict[Tuple[int, int], Tuple[float, float]]


def snapshot_layer_state(layer) -> Snapshot:
    """layer must be a raw DISLDOLayer-style wrapper exposing ._c with
    .ptrs/.indices/.weights_vals/.importance (the sili__new C++
    binding's own CSR-format per-synapse storage) -- NOT a
    TrueMultiDigitLayer directly, see snapshot_multi_digit_state for
    that. Keyed by (row, col), not raw array position -- see this
    module's own docstring for why that distinction is the whole point."""
    c = getattr(layer, "_c", None)
    if c is None:
        raise ValueError(
            "layer has no ._c -- not a raw DISLDOLayer-style wrapper "
            "(pass a single digit, or use snapshot_multi_digit_state "
            "for a TrueMultiDigitLayer)")
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
    n_synapses: int  # size of the before/after KEY INTERSECTION -- see n_new/n_died
    n_new: int        # present in `after` but not `before` (woke up / grew in)
    n_died: int        # present in `before` but not `after` (pruned / went dead)
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

    @property
    def churn_fraction(self) -> float:
        """(n_new + n_died) relative to the stable intersection size --
        a direct measure of how much connectivity moved around between
        snapshots (near 0 for deterministic rounding at dense init,
        meaningfully positive for stochastic rounding's dead-synapse
        wakeup churn -- see this module's own docstring)."""
        if self.n_synapses <= 0:
            return float("nan")
        return (self.n_new + self.n_died) / self.n_synapses


def check_stuck_weights(before: List[Snapshot], after: List[Snapshot], *,
                         importance_percentile: float = 75.0,
                         movement_percentile: float = 25.0) -> StuckWeightsReport:
    """before/after: lists of {(row,col): (weight,importance)} snapshots
    (one per digit/layer, from snapshot_multi_digit_state -- pass
    snapshots from MULTIPLE layers concatenated together for a whole-
    model check, or one layer's own list for a per-layer check), taken
    at two points separated by some real training interval.

    Flags a synapse STUCK if its importance at the BEFORE snapshot (the
    model's own belief about how much this synapse matters going INTO
    the interval) is in the top importance_percentile, but its realized
    |weight change| over the interval is in the bottom
    movement_percentile among ALL synapses (not just the high-
    importance ones). Only synapses present at BOTH snapshots (same
    (row,col) key, in the SAME digit) are compared -- synapses that
    appeared or disappeared in between (n_new/n_died on the report)
    have no well-defined "delta" and are excluded, not treated as
    either stuck or moving."""
    w_before_list, imp_before_list, delta_list = [], [], []
    n_new = n_died = 0
    for b, a in zip(before, after):
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
            "empty snapshots were passed in)")

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
