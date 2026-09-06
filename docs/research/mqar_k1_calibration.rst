``mqar_k1_calibration.py`` research notes
============================================

Companion doc to ``scripts/mqar_k1_calibration.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows.

.. _mqar_k1_calibration.module_overview:

Diagnosing the K-sweep's near-zero accuracy as a step-budget problem
--------------------------------------------------------------------------

*ID:* ``mqar_k1_calibration.module_overview``

Single-K (K=1) long-horizon calibration run for MQAR-on-tile-recurrence,
built after the original K-sweep (``train_mqar_k_sweep.py``, 3000 steps/K)
came back near-zero everywhere. Direct diagnosis: the architecture genuinely
CAN learn this task -- a fixed single example converges from loss=9.69 to
0.006 in 500 repeated exposures -- so the original sweep's near-zero
accuracy was a step-BUDGET problem (a fresh random MQAR sequence every step
gives too little repeated exposure per pattern to generalize in only 3000
steps), not a structural bug.

A pooled/repeated-exposure training-loop redesign was tried as a cheaper
alternative and empirically DISPROVEN (2000-step head-to-head: baseline
final acc=0.0167 vs pooled acc=0.0000 -- pooling overfits a small example
pool rather than generalizing) -- so this run just scales ``train_steps`` up
substantially instead, tracking accuracy over the whole trajectory (not just
the final step) to find where K=1 actually saturates before committing a
full step budget to the harder K values.

Run: ``python3 scripts/mqar_k1_calibration.py [train_steps] [seed] [eval_every]``
