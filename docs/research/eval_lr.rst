``eval_lr.py`` research notes
================================

Companion doc to ``model/eval_lr.py``. Source comments point back here by
anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers, frozen code snippets on
real-bug/non-obvious-derivation sections only).

.. _eval_lr.module_overview:

Module purpose: automating a manual lr sweep with golden-section search over log(lr)
----------------------------------------------------------------------------------------

*ID:* ``eval_lr.module_overview``

Fast, generic optimal-learning-rate search for sili_peridot's models.
Motivated directly by a real regression hunt (conversation): sili__new's
importance-signal contrib addition shrank baseline's effective step size,
and a manual geometric sweep (1x, 1.41x, 2x, ..., 100x the default lr) on a
short (1500-step) run found a clean single-peaked curve -- eval_acc rose
0.24 -> 0.56 by ~10x, then fell back toward chance by 100x. That sweep is
the algorithm this module automates: golden-section search over log(lr)
space, not linear -- lr's useful range spans orders of magnitude and it's
MULTIPLICATIVE steps that matter, confirmed directly by that sweep's own
shape (not, say, some fixed additive offset from the default).

Domain-agnostic core: ``find_optimal_lr()`` knows nothing about tile-
recurrence, ``OriginalArchModel``, or any specific training loop --
callers supply ``trial_fn(lr, seed) -> score`` (build a fresh model at
that lr, run a short train, return a scalar to MAXIMIZE). See
``tests/test_eval_lr.py`` for a worked adapter around
``scripts/l1_sparsity_probe.py``'s ``OriginalArchModel``/``run``/
``evaluate``, which is what this module replaces the manual version of.

.. _eval_lr.unimodal_assumption_and_multistart:

Unimodality assumption: finds A peak, not necessarily the global peak
---------------------------------------------------------------------------

*ID:* ``eval_lr.unimodal_assumption_and_multistart``

Assumes the score-vs-log(lr) curve is UNIMODAL (single global optimum) --
true for the sweep that motivated this module, not guaranteed for every
model/problem. If a real curve has multiple local optima, this finds A
peak (the one geometric bracketing first walks into from ``initial_lr``),
not necessarily the global best. Multi-start (varying ``initial_lr``) is
the straightforward extension if that ever turns out to matter -- not
built here since nothing so far has needed it.

.. _eval_lr.anytime_time_budgeted_design:

Anytime / time-budgeted execution, and the history field
-----------------------------------------------------------------

*ID:* ``eval_lr.anytime_time_budgeted_design``

Anytime / time-budgeted: runs trials against a wall-clock deadline,
tracking the best ``(lr, score)`` seen at every point -- if the deadline
hits mid-bracket or mid-search, returns whatever's currently best rather
than raising or needing to finish. "Runs continuously, returns the
optimal learning rate when stopped" (conversation), not a fixed trial
count.

``LRSearchResult.history`` holds ``(log(lr), mean_score)`` for every
DISTINCT lr actually evaluated, in the order tried -- lets a caller
plot/inspect the search trajectory, not just the final answer.

.. _eval_lr.golden_ratio_constant_reuse:

``_GOLD``: one constant reused for both bracketing expansion and interval-split
------------------------------------------------------------------------------------

*ID:* ``eval_lr.golden_ratio_constant_reuse``

Golden ratio conjugate (~0.618) -- both the expansion factor used while
bracketing and the interval-split ratio used while refining. Using the
SAME constant for both phases is standard (Numerical Recipes' mnbrak/
golden), not a coincidence: it's the ratio that makes golden-section
search reuse one already-evaluated interior point per iteration instead
of re-evaluating both ends every time -- the entire reason to prefer it
over plain bisection here, since every "evaluation" is a real (if short)
training run, not a cheap function call.

.. _eval_lr.find_optimal_lr_two_phase_algorithm:

``find_optimal_lr``: bracket then golden-section refine, sharing one deadline and cache
-----------------------------------------------------------------------------------------

*ID:* ``eval_lr.find_optimal_lr_two_phase_algorithm``

Finds the lr maximizing ``trial_fn``'s seed-averaged score.

Phase 1 (bracket): starting from ``initial_lr``, expand geometrically in
whichever direction improves the score until it stops improving -- gives
a triple ``(lo, mid, hi)`` in log(lr) space with mid's score >= both
neighbors', a valid bracket for phase 2.

Phase 2 (golden-section): repeatedly shrinks that bracket, one new
evaluation per iteration, until its width in log-space is under
``log_tol`` (default 0.05 ~= within a factor of ``e^0.05`` ~= 1.05x --
close enough for a "peak lr" answer that a follow-up full-scale run would
round anyway) or time runs out.

Both phases share one time-budget deadline and one cache (so the bracket
phase's last two points feed directly into golden-section without
re-evaluating either).

.. _eval_lr.bracket_reversal_direction_fix:

Bracket-reversal direction: a real sign bug, root-caused
-----------------------------------------------------------------

*ID:* ``eval_lr.bracket_reversal_direction_fix``

When the first geometric-growth step makes the score WORSE (``f_b < f_a``),
the bracket must reverse direction (shrink instead of grow). After
swapping so ``log_b`` holds the original (better) ``initial_lr`` point and
``log_a`` holds the worse (grown) one, continuing in the SAME direction we
were already heading (worse -> better) means stepping from ``log_b`` AWAY
from ``log_a``, i.e. ``step = log_b - log_a``, which is negative here
(shrinking lr further).

.. code-block:: python

   # WRONG (the bug this replaces): points back at the already-rejected
   # point, collapsing the whole bracket phase back to initial_lr
   # immediately instead of shrinking away from it.
   step = log_a - log_b

   # Fix: step away from the worse point, toward and past the better one.
   log_a, log_b = log_b, log_a
   f_a, f_b = f_b, f_a
   step = log_b - log_a  # negative-going from here
