``debug_energy_log_sigmas_runaway.py`` research notes
=========================================================

Companion doc to ``scripts/debug_energy_log_sigmas_runaway.py``. Source
comments point back here by anchor ID (``*ID:* `` marker under each heading
below). See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this
follows (semantic dotted anchor IDs, visible ID markers via the ``*ID:*``
line since plain ``.. _id:`` targets alone render invisible on GitHub, frozen
code snippets on real-bug/non-obvious-derivation sections only).

.. _debug_energy_log_sigmas_runaway.module_overview_and_nonreproducibility:

Module overview: the baseline_energy skip-rate investigation, and a non-reproducibility correction
--------------------------------------------------------------------------------------------------------

*ID:* ``debug_energy_log_sigmas_runaway.module_overview_and_nonreproducibility``

Checkpoint/repro for the ``baseline_energy`` skip-rate investigation
(2026-08-12): ``OriginalArchModel(dense=True, use_energy=True,
l1_sparsity_coef=0.05)``, seed=1000, hits a non-finite gradient early
(observed at step=1959 in the original run; ``run()`` then goes on to skip
46.8% of steps over a full 15000-step run under this config).

CORRECTED, per direct user question -- this script is NOT reliably
reproducible step-for-step, despite using a fixed seed. Directly tested: two
back-to-back reruns of this exact file gave ``first_nonfinite`` at step=2202
and step=1611 respectively -- neither matches the original 1959, and they
don't match each other. This is the SAME residual run-to-run
nondeterminism already found at the full multi-seed landmark scale (post
block4 StochasticRounding fix, commit 900b318: seed=1001 flipped between
0.6667/1.0000 across two identical runs, baseline skip_rate varied
0.011%-0.509%) -- it is NOT confined to multi-seed loops; it shows up in a
single isolated model over a long enough (15000-step) run too. The
stochastic-rounding fix genuinely fixed short-timescale/few-step
determinism (confirmed via multiple independent checks) but did NOT
eliminate whatever causes long-horizon divergence; that source remains
unidentified.

Do not treat "step 1959" (or any specific step number from this script) as
a fixed reference -- treat this as "``log_sigmas`` drifts monotonically
negative and unbounded, with no sign of leveling off, and this reliably (if
not at a fixed step) leads to a non-finite gradient somewhere in the first
~1500-2500 steps" instead.

.. _debug_energy_log_sigmas_runaway.original_failure_state_snapshot:

State at the original first failure: kept for reference, not as a reproducible target
------------------------------------------------------------------------------------------

*ID:* ``debug_energy_log_sigmas_runaway.original_failure_state_snapshot``

Logged before touching ``log_sigmas`` clamping (see
``l1_sparsity_probe.clip_log_sigmas_runaway_investigation``), so the
underlying mechanism can still be investigated after a clamp fix lands and
makes the RAW unbounded-drift failure unreproducible::

    step=1959 i=3: gradnorm non-finite! aux_loss=0.7474427819252014
    log_sigmas=[-1.3583835, -1.3720931, -1.0657132, -0.66924304]
    input_ln_absmax=1.0218 (crossed back above 1.0 in the same window)

``log_sigmas`` had been drifting monotonically negative, unbounded, since
~step 20 with no sign of leveling off before the failure -- ``sigmas =
exp(log_sigmas)`` was NOT yet near-zero at the failure point (~0.26-0.51),
so "sigma collapses to exactly 0" is not the direct trigger at this step,
though the drift itself is still a real, unbounded problem worth fixing
regardless of the precise NaN mechanism. ``input_ln_absmax`` reversing from
a ~0.69 minimum (around step 1000-1100) back up through 1.0 in the same
step window as the failure is a notable but NOT yet confirmed-causal
coincidence.
