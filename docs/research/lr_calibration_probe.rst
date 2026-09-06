``lr_calibration_probe.py`` research notes
=============================================

Companion doc to ``scripts/lr_calibration_probe.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows.

.. _lr_calibration_probe.module_overview:

Testing the "effective LR shrank" hypothesis for the landmark regression
----------------------------------------------------------------------------

*ID:* ``lr_calibration_probe.module_overview``

Fast (~5-10 min) check of the "is the effective lr smaller now" hypothesis
for the ``baseline`` config's ``landmark_checklist`` regression (0.8667 ->
0.3333 ``old_style_mean`` after sili__new's importance-signal contrib
addition + square-then-sum reversion).

Rationale: ``ci``'s denominator now includes ``contrib^2`` (or previously
``(g+contrib)^2``) on top of plain ``g^2`` -- a term that did not exist at
all when REFERENCE was measured. Since the weight step is
``-lr*g/sqrt(ci)``, a larger ``ci`` means a smaller step for essentially
every synapse, essentially all the time. If that's the whole story (not a
correctness bug), bumping ``peak_lr`` on a SHORT run should recover accuracy
back toward the un-regressed baseline without touching the C++ formula at
all. This is deliberately NOT the 15000-step x 5-seed landmark sweep --
short curriculum-saturated runs (seq_len maxes out at step 1000, see
``STEPS_PER_STAGE``/``NUM_TILES``) are enough to see the trend.

Usage: ``PYTHONPATH=<sili_peridot repo root> python scripts/lr_calibration_probe.py``

.. _lr_calibration_probe.lr_multiplier_extension:

Extending the multiplier sweep past the naive correction
--------------------------------------------------------------

*ID:* ``lr_calibration_probe.lr_multiplier_extension``

1x is the current default (``peak_lr=0.002``). sqrt(2)x is the naive
correction if ``g`` and ``contrib`` are typically comparable magnitude and
square-then-sum's extra term roughly doubles ``ci`` on average (sqrt(2)
undoes a doubled denominator). 2x is a coarser overcorrection to see if
recovery continues past sqrt(2) (would suggest the effect is bigger than a
simple doubling) or overshoots into instability.

Extended past 1x-6x (``eval_acc`` climbed
0.24->0.32->0.38->0.44->0.50->0.54, monotonic but flattening, no skips at
any point) -- pushing much further to distinguish a real plateau from a
log-shaped "diminishing but never actually stopping" curve, and to find
where (if anywhere) instability actually kicks in.
