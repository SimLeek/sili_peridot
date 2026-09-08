``estimate_wide_scaling.py`` research notes
====================================================

Companion doc to ``scripts/estimate_wide_scaling.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below). See
``docs/research/toy_tile_recurrence_rmt.rst`` for the pattern this follows.

.. _estimate_wide_scaling.module_overview:

Purpose: pre-calculate speedup before committing compute
----------------------------------------------------------

*ID:* ``estimate_wide_scaling.module_overview``

Pre-calculate the expected sparsity speedup at a TARGET ``state_width``
before committing compute to actually training there. Requested 2026-09-08
after task #415's per-component timing breakdown showed wide-layer sparsity
savings and the sparsification mechanism's own CSR-construction overhead
scale very differently with width -- extrapolating from one measured width
lets us predict whether a given scale-up is even worth it before running it.

.. _estimate_wide_scaling.physical_argument:

The physical argument (verified against real layer shapes, not guessed)
--------------------------------------------------------------------------

*ID:* ``estimate_wide_scaling.physical_argument``

``model/toy_tile_recurrence_rmt.py``'s actual layer constructors:

- ``input_proj``: ``(embed_width, state_width)`` -- ``embed_width =
  state_width/COLUMN_NEURONS``
- ``q/k/v/o_proj``: ``(state_width, state_width)``
- ``lm_head/critic_head``: ``(embed_width, vocab_size)``
- ``attention`` (``gaussian_attention``): ``total_slots`` (``num_tiles +
  NUM_MEMORY_SLOTS``, FIXED) x ``state_width`` per query

A dense linear layer's compute is ``O(in_features * out_features)``. So as
``state_width`` (W) scales:

- ``input_proj``, ``q/k/v/o_proj``: ``O(W^2)`` (quadratic)
- ``attention``: ``O(W)`` (linear -- the other factor is a
  WIDTH-INDEPENDENT constant, ``total_slots``)
- ``lm_head``, ``critic_head``: ``O(embed_width * vocab(W)) = O(W *
  vocab(W))``. **Correction (2026-09-08, user-flagged)**: an earlier
  version of this script treated ``vocab_size`` as a fixed constant
  (matching the CURRENT curriculum's ``VOCAB=128``), giving
  ``lm_head``/``critic_head`` ``O(W)`` scaling and letting them vanish to a
  negligible fraction at large W. That's only valid if we scale width
  WITHOUT ALSO scaling task difficulty. But the reason to scale width at
  all is to unlock more capability -- and per
  ``project_sili_wide_model_mqar_baseline`` / task #318, vocab is ALREADY
  saturated (126-128, curriculum-capped) at the CURRENT width. A wider
  model built to actually use its extra capacity would need vocab to grow
  too, not stay pinned at 128. If vocab scales linearly with W,
  ``lm_head``/``critic_head`` become ``O(W^2)`` -- same order as the wide
  layers -- and do NOT vanish at scale. We don't know the real
  vocab-vs-width relationship for this task, so ``vocab_exponent`` is an
  explicit, swept parameter (0 = old fixed-vocab assumption, 1 = vocab
  scales linearly with W), not a hidden guess.
- ``"other"`` (token embed lookup, target-building, Python loop overhead,
  AND -- only when ``x_r_target``/``dy_r_target`` sparsity is active -- the
  nucleus top-k CSR construction itself, ``_to_sparse``'s
  ``_nucleus_top_k_csr`` call across the 5 wide layers): this is the one
  genuinely mixed bucket. Comparing a real dense calibration run
  (``x_r_target=None``, ``_to_sparse`` is a no-op) against a real sparse
  one (``x_r_target=0.5``) at the SAME width shows ``"other"`` is NOT
  simply a width-independent constant that becomes proportionally larger
  as everything else shrinks -- sparsity's CSR construction is a REAL
  ADDITIONAL per-step cost that doesn't exist at all when dense. Modeled
  here as: ``other_dense(W) = orchestration_const`` (``O(1)``,
  width-independent -- token gen/target-building/loop dispatch);
  ``other_sparse(W) = orchestration_const + csr_overhead(W)``, where
  ``csr_overhead`` scales ``O(W)`` (a top-k/partial-sort over a length-W
  vector, done once per wide layer per step).

.. _estimate_wide_scaling.csr_overhead_consequence:

The consequence: overhead vanishes as width grows
------------------------------------------------------

*ID:* ``estimate_wide_scaling.csr_overhead_consequence``

Wide-layer SAVINGS from sparsity scale ``O(W^2)`` (same order as the base
cost they're cutting -- a fixed fractional reduction of an ``O(W^2)``
quantity is still ``O(W^2)``), while the CSR-construction overhead that
BUYS those savings scales only ``O(W)``. So even though the overhead eats a
large fraction of the win at small W (measured: ~30-35% of the gross
wide-layer savings at W=128, r_target=0.5 -- see
``CALIBRATION_SPARSE_W128`` in source), it becomes a vanishingly small
fraction as W grows. This matches the user's own intuition (2026-09-08):
"IO scales... O(N^2)... top-k scales... pretty close to just N."

.. _estimate_wide_scaling.calibration_data:

Calibration data provenance
--------------------------------

*ID:* ``estimate_wide_scaling.calibration_data``

Real measurements, task #415/#414, W=128, embed_width=16. Per-step seconds,
derived from ``log_fn``'s ``t[...]`` breakdown windows (dense:
tests/../smoke run, ``log_every=200``, steps 401-600; sparse:
``r_target_min=0.5`` clean run, ``log_every=500``, steps 6501-7000, x_r/dy_r
both AT their 0.5 floor -- true steady state, not mid-descent). Both fp32,
``K_START=2``, LEVEL_DOWN-disabled, no-aux-loss base config
(``arm_nolevel_down``-style).

Small-magnitude components (attention, lm_head -- both well under 1s per
window) show noisy ratios between the two runs (measurement jitter
dominates real signal at that scale) -- don't over-trust their SPECIFIC
ratio, only their small absolute contribution to the total. This is why
``LM_HEAD_BASE_W128`` in source averages the two raw readings instead of
using them separately: lm_head/critic_head are NOT sparsified by design,
so their measured cost should be IDENTICAL between the dense and sparse
calibration runs, and the 5x raw discrepancy (0.0043 vs 0.00086) is pure
noise on a sub-millisecond quantity.

.. _estimate_wide_scaling.usage:

Usage
---------

*ID:* ``estimate_wide_scaling.usage``

.. code-block:: text

    python3 scripts/estimate_wide_scaling.py [target_state_width ...]

Prints projected per-step time, steps/sec, and speedup vs dense at each
target width, plus the projected "other" fraction (the thing to watch --
if a target width still shows other dominating, sparsity isn't paying off
there yet).
