``train_toy_beyond_context_peak_eligibility_only.py`` research notes
========================================================================

Companion doc to
``scripts/train_toy_beyond_context_peak_eligibility_only.py``. Source
comments point back here by anchor ID (``*ID:* `` marker under each heading
below). See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this
follows.

.. _train_toy_beyond_context_peak_eligibility_only.module_overview:

Replacing the structurally-flawed e-prop arm with peak-eligibility
--------------------------------------------------------------------

*ID:* ``train_toy_beyond_context_peak_eligibility_only.module_overview``

Runs ONLY the new peak-eligibility arm (``PeakEligibilityDISLDOLayer``) of
the Tier 1 beyond-context comparison, at a REDUCED step budget.

Replaces the earlier e-prop-only script -- e-prop (both the plain and Adam
variants) was found structurally flawed (see JOURNAL.md's postmortem: its
delta-trick gradient proxy is provably zero for a row silent at the query
tick, which is exactly the row this mechanism most needs to credit).
``PeakEligibilityDISLDOLayer`` replaces it: instead of any Python-side
gradient approximation, it substitutes a peak-held (signed) value directly
into ``SparseLinearLayer``'s own ``last_input`` buffer before backward
fires, so DISLDO's REAL C++ gradient math computes the correction.

Same reduced-budget rationale as before: the 120k-step (~1hr) scale-up was
compensating for the no-BPTT/no-gradient-pathway problem; peak-eligibility
exists to provide exactly that missing pathway, so it should not need the
same 40x scale-up to show a signal.

The dense/tile/tile+energy numbers are from the PRIOR 120,000-step run
(JOURNAL.md, "Tier 1, ~1hr budget (40x steps)") and are shown for CONTEXT
ONLY -- different step budget, not a strict equal-budget comparison.

Run: ``python -m scripts.train_toy_beyond_context_peak_eligibility_only``

.. _train_toy_beyond_context_peak_eligibility_only.reduced_budget_monkeypatch:

Reduced-budget monkeypatch onto the imported comparison module
--------------------------------------------------------------------

*ID:* ``train_toy_beyond_context_peak_eligibility_only.reduced_budget_monkeypatch``

Reduced budget: 30% of the 120k run, same curriculum ratios/shape as the
earlier e-prop probe (``WARMUP_STEPS``/``TRAIN_STEPS`` = 1/30,
``STEPS_PER_LEVEL``/``TRAIN_STEPS`` = 1/20). Monkeypatched onto the imported
module since ``_train_tile``/``_sample_n_bits`` read these as module
globals.

The ``RECORDED_*`` dicts are from JOURNAL.md, "Tier 1, ~1hr budget (40x
steps)" -- ``TRAIN_STEPS=120_000`` (3.3x this run's budget), same curriculum
shape, seeds 1/2/3 for dense/tile-no-energy/tile+energy. Shown for CONTEXT
ONLY.
