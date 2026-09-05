``prototype_peak_synapse_learning_comparison.py`` research notes
====================================================================

Companion doc to ``scripts/prototype_peak_synapse_learning_comparison.py``.
Source comments point back here by anchor ID (``*ID:* `` marker under each
heading below). See ``sili__new/docs/research/sparse_rnn.rst`` for the
pattern this follows (semantic dotted anchor IDs, visible ID markers via the
``*ID:*`` line since plain ``.. _id:`` targets alone render invisible on
GitHub, frozen code snippets on real-bug/non-obvious-derivation sections
only).

.. _prototype_peak_synapse_learning_comparison.module_overview_and_latest_results:

Module overview and the latest run's results
-----------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.module_overview_and_latest_results``

Small, fast (seconds) end-to-end learning comparison: does the per-synapse
peak-correction mechanism (``backward_sparse`` + 1-hot substitution, verified
to fire correctly in ``prototype_synapse_peak_credit.py``) actually help a
tiny recurrent net LEARN the delayed-credit deviation-detection task, at
REALISTIC learning rates and real repeated training -- not just "does the
gradient move once when pushed hard". No attention, no MLP, no embedding
table, no tile system -- one raw recurrent ``DISLDOLayer`` cell (``state_t =
state_t-1 + cell([token_onehot, state_t-1])``) + one ``DISLDOLayer`` readout.
Per direct instruction: verify here, in seconds, before touching the full
tile system.

**Latest run** (50 seeds x 40 eval sequences, ``STATE_WIDTH=16``,
``TRAIN_STEPS=6000``, ``GENTLE_ENERGY_CONFIG``, eval-mode energy bug fixed --
see ``prototype_peak_synapse_learning_comparison.energy_train_only_eval_skip_convention``
below), with REAL hypothesis tests (paired -- ``plain[s]``/``peak[s]`` share
the same eval-sequence seed at each ``s``, so a paired test uses that shared
variance rather than discarding it; Wilcoxon signed-rank as a non-parametric
check against the t-test's normality assumption)::

    n_bits  plain mean  peak mean   diff  paired-t     p(t)   p(Wilcoxon)
         2      0.5262     0.5386  0.0124    0.318    0.752       0.746
         3      0.5132     0.5410  0.0278    0.953    0.345       0.653
         4      0.4930     0.5010  0.0080    0.200    0.842       0.766
         6      0.4092     0.4514  0.0422    1.220    0.229       0.356

NOT statistically significant at any point (best case, n=6, p~=0.23) -- both
tests agree, so this isn't a normality-assumption artifact. Honest verdict:
no statistically significant evidence that peak-synapse outperforms plain
DISLDO at this scale/task/sample size. peak-synapse IS nominally ahead at all
four points (a real, consistent direction, unlike earlier pre-fix runs where
the sign flipped between reruns) -- worth continued tracking with more
seeds, not something to lean on as evidence yet. Both arms remain at/below
chance at n=6 specifically -- the hardest, most out-of-context case is still
the weak point for both, independent of mechanism.

.. _prototype_peak_synapse_learning_comparison.bugs_found_before_trustworthy_run:

Multiple real bugs found and fixed to get a run worth trusting at all
--------------------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.bugs_found_before_trustworthy_run``

See JOURNAL.md for the full narrative. Four separate issues, all fixed:

- sili__new's ``_preseed_random_sparse`` ignored ``np.random.seed()``
  entirely (fixed upstream, rng injection).
- FP4's own stochastic weight rounding was ALSO unseeded (needs
  ``seed_fp4_stochastic_rng()`` per-thread, hence ``NUM_CPUS=1`` here -- see
  ``prototype_peak_synapse_learning_comparison.num_cpus_single_thread_reproducibility``).
- ``MAX_WEIGHTS`` was landing on a k=1 bare-floor connection count -- see
  ``prototype_peak_synapse_learning_comparison.max_weights_per_row_floor_fix``.
- ``EnergyDynamics`` -- calibrated at h sizes 20-64 -- was both (a) gating
  out ~69% of every state-write at this width by default and (b) being
  applied during EVAL calls too (no train/eval mode of its own), corrupting
  every accuracy number measured before these fixes -- see
  ``prototype_peak_synapse_learning_comparison.gentle_energy_config_calibration``
  and
  ``prototype_peak_synapse_learning_comparison.energy_train_only_eval_skip_convention``.

The isolated mechanism check (``prototype_synapse_peak_credit.py``) still
holds throughout -- a silent-at-query-tick row genuinely gets real credit
that plain DISLDO cannot give it.

.. _prototype_peak_synapse_learning_comparison.selection_criterion_bptt_derivation:

Selection criterion: derived from the true BPTT sum, not guessed
----------------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.selection_criterion_bptt_derivation``

Selection criterion since revised, derived directly from the true BPTT sum
rather than guessed (see JOURNAL.md for the full derivation):

.. code-block:: text

   dL/dW[r,c] = Sum_t [dL/dh_T . dh_T/dh_t] . x_r(t) . phi'(delta_c(t))

Because the state update is residual (``h_t = h_{t-1} + phi(delta_t)``),
``dh_T/dh_t`` is dominated by an identity term regardless of ``T-t``, so
CAPTURE (substituting a tagged ``x_r(t)`` into ``backward_sparse`` against
the real captured error at query time) is exactly the leading-order term of
that sum -- already validated, no kernel work needed, confirmed directly
against ``linear_disldo.hpp``'s ``disldo_forward``: the per-synapse ``contrib
= w * iv`` is already materialized before the scatter-add, so a
max-tracking addition there would be nearly free, but turns out to be
unnecessary -- SELECTION doesn't need the weight at all.

Comparing candidate ticks for the SAME synapse needs to rank by ``x_r(t) .
phi'(delta_c(t))``; comparing DIFFERENT rows competing for the SAME output
``c`` at a FIXED tick has ``phi'(delta_c(t))`` cancel out entirely (shared by
every row feeding ``c``), leaving ``|x_r(t)|`` alone as correct there -- so
weight never enters either comparison.

``phi'(delta_c(t))`` itself isn't cheaply available, but the residual
structure gives an exact stand-in: ``state_change(t) = phi(delta_c(t))`` IS
the tick's actual state delta (no need to compute a difference,
``delta_gated`` already IS ``dh`` for that tick) -- a reasonable proxy for
``phi'`` in the near-linear regime.

Net criterion: ``score_r(t) = |x_r(t)| * |state_change(t)|`` (aggregated
across output dims, since this prototype tracks one peak per INPUT ROW, not
per ``(row, col)`` synapse -- true per-synapse tracking is the bigger,
deferred core change). This is the score ``PeakSynapseCell._update_peak``
computes and ``PeakSynapseCell.query_step`` uses to select correction
candidates (see
``prototype_peak_synapse_learning_comparison.peak_cell_correction_criterion``
below).

.. _prototype_peak_synapse_learning_comparison.decay_from_horizon_derivation:

``decay_from_horizon``: derived decay rate, not a tuned constant
----------------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.decay_from_horizon_derivation``

Derived, not tuned: ``P(a peak set now survives n further ticks unbeaten)``
requires the decayed threshold to stay low enough for that long; solving
``decay**n = retain_fraction`` for ``decay`` gives the minimum decay rate
that keeps at least ``retain_fraction`` of the tag's original score alive out
to a target credit-assignment horizon of ``n`` ticks. E.g.
``decay_from_horizon(100, 0.1) ~= 0.977`` -- "I want tags to plausibly
survive out to 100 ticks, retaining at least 10% of their original score."

``PEAK_DECAY`` is derived from the actual horizon this script tests
(``OUT_OF_CONTEXT_MAX`` ticks, the longest gap a tag might need to survive)
and a 10% retain target -- replaces the earlier hardcoded ``peak_decay=0.9``,
which wasn't calibrated to any particular horizon at all.

.. _prototype_peak_synapse_learning_comparison.onehot_constant_magnitude_limitation:

Known limitation: one-hot token rows have constant magnitude
------------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.onehot_constant_magnitude_limitation``

Not yet worked around: one-hot token rows have CONSTANT magnitude every time
they fire, so any later occurrence of the same token always beats a decayed
peak of the same original magnitude regardless of decay rate -- for those
rows specifically, the mechanism can't reach back further than the token's
own most recent occurrence.

The STATE-portion rows (``M_prev``, continuously-valued, carrying an
additive trace of everything written so far thanks to the residual update)
don't have this problem the same way, though whether the recurrent dynamics
preserve or cancel an old write is an open, state-width-dependent question --
plausibly safer with a wide recurrent state (matching this project's own
established column-averaging design principle), not yet tested here.

.. _prototype_peak_synapse_learning_comparison.num_cpus_single_thread_reproducibility:

``NUM_CPUS=1``: ``seed_fp4_stochastic_rng`` only reseeds the calling thread
--------------------------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.num_cpus_single_thread_reproducibility``

``NUM_CPUS=1``, not 2: ``_cpu.seed_fp4_stochastic_rng`` only reseeds the
CALLING thread's RNG state (checked directly, not assumed -- its own
docstring: "does not control a real (OpenMP-parallel) training run's
outcome, only this one thread's RNG state"). With ``num_cpus>1``, worker
threads would keep independent, unseeded RNG state regardless of calling
this -- single-threaded is what actually makes a run reproducible,
appropriate anyway at this toy scale.

.. _prototype_peak_synapse_learning_comparison.max_weights_per_row_floor_fix:

``MAX_WEIGHTS``: removing the per-row floor-clamp cliff
-------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.max_weights_per_row_floor_fix``

DERIVED, not guessed: ``_preseed_random_sparse`` computes ``per_row =
max(2, max_weights // n_inputs)`` and then ``k = per_row // 2`` -- for the
cell layer (``n_inputs=IN_FEATURES``, ``n_outputs=STATE_WIDTH``), the worst
case, ``per_row`` was landing on the bare floor of 2 (``k=1``: literally ONE
random connection per input row, zero redundancy) at the previous
``MAX_WEIGHTS=32``. Setting ``max_weights`` so ``per_row`` can reach
``n_outputs`` (full column coverage for the widest layer) removes that
floor-clamp cliff.

This alone did NOT fix the random-collapse instability -- turned out to be
only ONE of two separate uncontrolled RNG sources (see the
``seed_fp4_stochastic_rng`` call in ``train()``, the second one: FP4's own
stochastic weight rounding, used on every backward call, was ALSO never
seeded -- see
``prototype_peak_synapse_learning_comparison.num_cpus_single_thread_reproducibility``).

.. _prototype_peak_synapse_learning_comparison.gentle_energy_config_calibration:

``GENTLE_ENERGY_CONFIG``: recalibrating EnergyDynamics for a small state width
------------------------------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.gentle_energy_config_calibration``

Verified directly (not guessed): the ORIGINAL calibration (``drive=0.1,
activation_cost=0.05, precision=0.01, density=0.05, p=0.3``), taken from a
config validated at h sizes 20-64, defaults to gating out ~69% of this
tick's state-write at our much smaller ``STATE_WIDTH`` (measured: 5/16
neurons pass at init, ``actual_p=0.3125`` -- the model NEVER gets past its
own hard ceiling ``p``, since ``density`` is a target the gate only grows
toward via training, not the starting point). That was catastrophic at this
scale: a fixed ``n_bits=2`` sanity check (the easiest possible version of
this task) went from loss 0.009/100% accuracy with energy off to loss
staying ABOVE chance-level and 22-45% final accuracy with the original
config on.

This config turns ``activation_cost``/``drive``/``precision``/``exploration``
down to their practical floors (``activation_cost=0.01`` is
``EnergyDynamics``' own hard minimum) and ``p``/``density`` up to fully
permissive (measured: 16/16 neurons pass at init, ``actual_p=1.0``) -- same
fixed-``n_bits=2`` check recovers to 0.750+ accuracy with this config once
the separate eval-mode bug (see
``prototype_peak_synapse_learning_comparison.energy_train_only_eval_skip_convention``)
is also fixed.

**Why drive=0.005, not higher**: a single still-collapsing seed's own
``energy.energy`` array, traced over its full training run at
``drive=0.005``, showed mean energy trending NEGATIVE (settles ~-1.2 to
-1.6, only 0-3/16 neurons near the firing threshold at any snapshot, down
from all 16 at init) -- ``activation_cost*|h|`` drains energy proportional
to a neuron's OWN real magnitude while drive accumulates it at a flat rate,
so a neuron with genuinely large, useful output gets pushed toward the
shutoff floor rather than toward firing. Raising drive to 0.05 DID flip that
ONE seed's final mean energy positive (-1.10 -> +1.05) as predicted -- but
made the real 8-seed stability check WORSE overall (means dropped from
0.41-0.60 to 0.34-0.35, MORE seeds hit exact 0.0 collapse, not fewer). A fix
validated on one seed's own diagnostic doesn't necessarily generalize --
reverted to 0.005, the config that actually performed better across the
full seed set, not the one that looked better on paper for a single case.

.. _prototype_peak_synapse_learning_comparison.energy_train_only_eval_skip_convention:

EnergyDynamics is training-only: skip the gate entirely at eval (``lr==0.0``)
------------------------------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.energy_train_only_eval_skip_convention``

``EnergyDynamics`` is a TRAINING-time mechanism only -- not designed for
eval/inference (direct correction: no train/eval mode of its own, so
applying it during eval runs the trained weights through a DIFFERENT,
still-noisy computational path than what any temperature-style inference use
would need real reparameterization for, not just reuse). ``lr==0.0`` is this
codebase's own existing convention for "this is an eval call" (matches every
other ``lr==0`` skip elsewhere in this file and in DISLDO's own backward) --
skip the gate entirely then, use the raw, ungated delta. Applied identically
in both ``PlainCell.step``/``query_step`` and
``PeakSynapseCell._cell_step``.

.. _prototype_peak_synapse_learning_comparison.plain_cell_energy_and_aux_loss:

``PlainCell``: energy's role, and discarding non-query aux_loss
------------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.plain_cell_energy_and_aux_loss``

Baseline: plain ``DISLDOLayer``, no correction. ``use_energy`` wraps the
cell's own contribution (``delta``, before the residual add) through
``EnergyDynamics`` every tick -- same role ``EnergyDynamics`` plays on the
attention output in the full tile-recurrence system, where it twice fixed
this exact "collapse at one specific point" pattern (JOURNAL.md: n_bits=2
collapse, later n_bits=24 collapse) by keeping more neurons active, raising
the odds a useful state-carrying pattern survives.

``aux_loss`` from non-query ticks is simply discarded (never reaches a
``.backward()`` call) -- same no-BPTT precedent as everywhere else in this
project; only the query tick's own ``aux_loss`` is added to the real loss.

.. _prototype_peak_synapse_learning_comparison.peak_cell_correction_criterion:

``PeakSynapseCell``: score design and the active-row correction gate
--------------------------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.peak_cell_correction_criterion``

Cell + readout, cell ALSO gets a per-synapse peak correction at the query
tick (see ``prototype_synapse_peak_credit.py`` for the mechanism itself).
Same energy wiring as ``PlainCell`` -- ``delta`` stays the RAW cell output
(what the peak-correction's ``delta.grad`` check needs); ``delta_gated``
(after ``EnergyDynamics``) is what actually goes into the residual add.
``delta.grad`` is still a real, correctly-backpropagated quantity when
energy sits downstream of it -- just incorporating energy's own local
derivative too.

Selection score = ``|x_r(t)| * |state_change(t)|`` -- DERIVED (see
``prototype_peak_synapse_learning_comparison.selection_criterion_bptt_derivation``),
not the earlier magnitude-only ``|x_r(t)|`` version. ``state_change`` (=
``delta_gated`` for this tick, since the residual update makes ``dh``
literally equal to the gated cell output, no separate subtraction needed)
stands in for the otherwise-unavailable ``phi'(delta_c(t))`` term in the
true per-tick gradient contribution. The weight is deliberately NOT part of
this score -- shown not to belong in either the same-row-across-time or
same-tick-across-row comparison.

**Correction gate in** ``query_step`` (extra per-synapse correction, using
the REAL error that just flowed into the cell's own output, ``delta.grad``,
populated by the residual-add's backward before it reached the cell):
DERIVED criterion, not a tuned threshold -- row r's normal gradient is ``g =
dy*iv`` (confirmed directly from ``linear_disldo.hpp``), exactly zero
whenever ``x_row[r]`` is zero, so only rows that are CURRENTLY zero
(numerically, not "small") are ones normal training structurally cannot
touch this tick; correcting anything else would duplicate/fight training
that already works. ``ZERO_EPS`` distinguishes exact-zero from nonzero in
float32, not a magnitude cutoff. A row is only corrected if it is currently
inactive AND has a real historical peak score to credit.

.. _prototype_peak_synapse_learning_comparison.paired_hypothesis_test_design:

``main()``: why the hypothesis tests are paired
------------------------------------------------------

*ID:* ``prototype_peak_synapse_learning_comparison.paired_hypothesis_test_design``

Paired tests: ``plain[s]`` and ``peak[s]`` share the same eval-sequence seed
(``5000+s``) at each ``s``, so a paired test uses that shared per-seed
variance rather than discarding it -- more sensitive than treating the two
arms as independent samples, and matches how the data was actually
generated. Wilcoxon signed-rank runs as a non-parametric cross-check against
the t-test's normality assumption (accuracy is itself a mean over
``EVAL_SEQUENCES`` binary trials, not obviously normal at N=50). A
``ValueError`` from ``scipy_stats.wilcoxon`` (all-zero differences) is a
degenerate case, not an error -- reported as ``nan`` rather than raising.
