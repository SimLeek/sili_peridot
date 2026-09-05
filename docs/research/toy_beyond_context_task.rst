``toy_beyond_context_task.py`` research notes
================================================

Companion doc to ``model/toy_beyond_context_task.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers).

.. _toy_beyond_context_task.module_overview:

Tier 1 of the out-of-context benchmark suite
-----------------------------------------------

*ID:* ``toy_beyond_context_task.module_overview``

See the approved plan (``fuzzy-plotting-starlight.md``) for the full
three-tier design and rationale: tile-recurrence is strictly more general
than a bounded-window/bounded-depth transformer (it can carry information in
persistent recurrent state across unbounded time); this task is
deliberately the simplest possible genuine sequential computation (a single
XOR accumulator) so failure past the window is clearly a
WINDOW-VISIBILITY limitation, not a capacity or task-difficulty confound.

``generate_parity_sequence``: ``n_bits`` random bits followed by a ``'?'``
query token followed by the correct running-parity answer bit -- the answer
is itself the real next token after ``'?'``, not a bolted-on label (ordinary
next-token prediction). ``VOCAB_SIZE=3`` (``0``, ``1``, ``'?'`` -> token ids
0, 1, 2) gives a genuine way to signal "answer now" vs. mid-sequence, unlike
an earlier 2-token design that had no way to distinguish those.

.. _toy_beyond_context_task.deviation_sequence_design:

``generate_deviation_sequence``: single-deviation isolation task
--------------------------------------------------------------------

*ID:* ``toy_beyond_context_task.deviation_sequence_design``

``DEVIATION_BASE_PATTERN`` is fixed and shared across EVERY sequence -- the
network learns this ONE repeating motif once, rather than re-deriving a
fresh per-sequence pattern.

The body is a dense repeating base pattern with at most ONE sparse deviation
(a single flipped bit) inserted at a random position. Answer = 1 if any
deviation occurred, else 0. ``deviation_prob`` is tuned to ~0.5 so a constant
"always predict no-deviation" strategy can't trivially win the way it could
if deviations were rare.

This is a deliberate contrast with parity: parity's XOR answer is already
balanced, but the SIGNAL isn't -- every single bit has to be tracked
correctly for XOR to come out right, diluting any one credit-assignment
improvement across all the OTHER bits that also need to be correct. Here the
answer depends on exactly ONE bit's worth of remembered state, a cleaner
isolation of "does the credited tick's influence survive until the query"
specifically.

``n_positions`` plays the same curriculum role as ``generate_parity_sequence``'s
``n_bits`` -- growing it past the visible window is what makes the single
deviation (when present) sometimes fall OUTSIDE context, forcing reliance on
carried state rather than direct visibility.
