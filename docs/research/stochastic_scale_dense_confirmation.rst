``stochastic_scale_dense_confirmation.py`` research notes
=============================================================

Companion doc to ``scripts/stochastic_scale_dense_confirmation.py``. Source
comments point back here by anchor ID (``*ID:* `` marker under each heading
below). See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this
follows.

.. _stochastic_scale_dense_confirmation.module_overview:

Statistical power follow-up, not seed-pinning, for the deterministic-vs-stochastic gap
------------------------------------------------------------------------------------------

*ID:* ``stochastic_scale_dense_confirmation.module_overview``

Statistical follow-up, not a seed-pinning fix: both prior grid runs
(weight-sparsity version and input-sparsity version) agree the most
interesting cell is ``state_width=512``, full weight connectivity density,
full input density -- weight-sparsity grid saw stochastic BEAT deterministic
there (0.811 vs 0.767, gap=-0.044); input-sparsity grid's identical cell saw
them roughly TIE (0.756 vs 0.767, gap=+0.011). Both are n=3 seeds -- not
enough to tell a real effect from seed noise. The fix for "is this a real
effect" is MORE independent trials, not fixing/pinning the stochastic-
rounding RNG to make a single run reproducible (that would just answer
"what does seed X do", not "is the population-level gap actually near zero
or not").

Reuses ``_train_and_eval`` from ``stochastic_stability_vs_scale_sparsity.py``
unmodified (same task, same training procedure) at exactly that one grid
cell (``embed_width=32``, ``column_neurons=16`` -> ``state_width=512``,
``input_density=1.0``), just with many more seeds, so this is a real
statistical estimate of the deterministic-vs-stochastic gap at this specific
scale/density point, not a fresh comparison on different grounds.

Usage: ``PYTHONPATH=<sili_peridot repo root> python scripts/stochastic_scale_dense_confirmation.py``

``SEEDS`` uses a fresh seed block (2000-2011), no overlap with either prior
grid.

.. _stochastic_scale_dense_confirmation.paired_t_stat:

Paired-difference statistic, not an independent-samples test
------------------------------------------------------------------

*ID:* ``stochastic_scale_dense_confirmation.paired_t_stat``

Paired (same seed -> same task instances/init for both arms) t-test-style
stat -- gives a rough sense of whether the paired gap is distinguishable
from 0.
