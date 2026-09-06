``eval_stuck_weights.py`` research notes
============================================

Companion doc to ``model/eval_stuck_weights.py``. Source comments point back
here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers, frozen code snippets on
real-bug/non-obvious-derivation sections only).

.. _eval_stuck_weights.module_overview:

Module purpose: is importance-flagged synapse movement real, or stuck
--------------------------------------------------------------------------

*ID:* ``eval_stuck_weights.module_overview``

Are synapses the model's own importance signal (``ci`` -- the combined
gradient+forward-contribution second moment DISLDO's RMSprop-style update
tracks, see sili__new's ``linear_disldo.hpp``) has flagged as important
actually MOVING, or are they stuck? A synapse can end up stuck for reasons
genuinely worth distinguishing: its update rounds away below FP4/FP8's
quantization step (this project's own history has multiple real bugs of
exactly this shape -- the zero-escape/ULP-rounding work earlier this
session), ``ci`` itself has grown large enough to over-damp the step
(importance-as-optimizer doing its job TOO well), or the lr is genuinely too
small for that synapse's local gradient scale (the same calibration issue
``eval_lr.py``'s ``find_optimal_lr`` exists to catch at the whole-model
level, here at single-synapse resolution).

Confirmed directly this session (conversation): under deterministic
rounding at toy scale (and at every width tested up to 1024, an 8x-plus
sweep), essentially every already-live synapse is stuck at
``mean|delta_w|=0.0``. Under stochastic rounding, dead (weight=0,
importance=0) synapses DO wake up over time (nnz grows) -- but whether
stochastic rounding ALSO helps already-LIVE synapses move more (not just
wakes dead ones) needs a clean before/after diff that survives connectivity
CHANGING between snapshots, since stochastic rounding's whole point is that
connectivity isn't stable.

.. _eval_stuck_weights.snapshot_row_col_keying:

Snapshot format: why keyed by (row, col), not raw array position
--------------------------------------------------------------------

*ID:* ``eval_stuck_weights.snapshot_row_col_keying``

Snapshot-diff based, not a live per-step hook: call
``snapshot_multi_digit_state()`` before and after some real training
interval, then ``check_stuck_weights()`` on the two snapshots.

Snapshots are keyed by ``(row, col)`` -- NOT raw array position --
specifically so a snapshot pair survives nnz changing between them (new
synapses waking up, in sili__new terms, insert into the middle of a row's
CSR data, shifting every later array position; diffing by array index
instead of by stable key silently compared the WRONG pairs of synapses
whenever that happened). This was confirmed directly as the reason an
earlier attempt at this comparison failed with a shape-mismatch guard
instead of a wrong answer -- only checking BOTH shapes AND (implicitly)
index-for-index correspondence, via a hard reject, is what caught it.

``snapshot_layer_state`` takes a raw DISLDOLayer-style wrapper exposing
``._c`` (the sili__new C++ binding's own CSR-format per-synapse storage:
``.ptrs``/``.indices``/``.weights_vals``/``.importance``) -- NOT a
``TrueMultiDigitLayer`` directly (see
``eval_stuck_weights.multi_digit_fallback`` for that case).

The comparison in ``check_stuck_weights`` only uses the INTERSECTION of
keys present at both snapshots (see ``n_new``/``n_died`` on the report for
how much churn that intersection is throwing away) -- a genuinely fair "did
an already-alive-at-both-points synapse move" comparison, independent of
how many synapses appeared or disappeared in between.

.. _eval_stuck_weights.multi_digit_fallback:

``snapshot_multi_digit_state``: all digit stages, not just stage 0
------------------------------------------------------------------------

*ID:* ``eval_stuck_weights.multi_digit_fallback``

``TrueMultiDigitLayer`` holds ``n_stages`` separate ``DISLDOLayer`` digits,
each with its own weights/importance -- this snapshots ALL of them (not
just stage 0), since a stuck synapse in any digit stage is a real stuck
synapse. Falls back to a single snapshot for a layer that isn't a
``TrueMultiDigitLayer`` (has no ``.digits``) but does have ``._c`` directly.

.. _eval_stuck_weights.excess_stuck_ratio_signal:

``excess_stuck_ratio``: normalizing against the chance baseline
------------------------------------------------------------------

*ID:* ``eval_stuck_weights.excess_stuck_ratio_signal``

``stuck_fraction`` relative to what pure chance would predict
(``expected_stuck_fraction_if_independent``, i.e. the ``movement_percentile``
itself -- what overlap you'd see if importance and movement were
unrelated). 1.0 means "no more stuck synapses than random overlap would
produce"; meaningfully above 1.0 (say 2x+) is the real signal that
high-importance synapses are specifically failing to move, not just
unlucky sampling.

.. _eval_stuck_weights.churn_fraction_signal:

``churn_fraction``: how much connectivity moved between snapshots
------------------------------------------------------------------------

*ID:* ``eval_stuck_weights.churn_fraction_signal``

``(n_new + n_died)`` relative to the stable intersection size -- a direct
measure of how much connectivity moved around between snapshots. Expected
near 0 for deterministic rounding at dense init, meaningfully positive for
stochastic rounding's dead-synapse wakeup churn (see
``eval_stuck_weights.module_overview``).

.. _eval_stuck_weights.check_stuck_weights_semantics:

``check_stuck_weights``: the stuck definition, and why appeared/disappeared synapses are excluded
--------------------------------------------------------------------------------------------------------

*ID:* ``eval_stuck_weights.check_stuck_weights_semantics``

Flags a synapse STUCK if its importance at the BEFORE snapshot (the
model's own belief about how much this synapse matters going INTO the
interval) is in the top ``importance_percentile``, but its realized
``|weight change|`` over the interval is in the bottom
``movement_percentile`` among ALL synapses (not just the high-importance
ones).

Only synapses present at BOTH snapshots (same ``(row, col)`` key, in the
SAME digit) are compared -- synapses that appeared or disappeared in
between (``n_new``/``n_died`` on the report) have no well-defined "delta"
and are excluded, not treated as either stuck or moving. ``before``/
``after`` are lists of snapshots (one per digit/layer, from
``snapshot_multi_digit_state``) -- pass snapshots from MULTIPLE layers
concatenated together for a whole-model check, or one layer's own list for
a per-layer check.
