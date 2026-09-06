``mqar_k1_precision_control.py`` research notes
==================================================

Companion doc to ``scripts/mqar_k1_precision_control.py``. Source comments
point back here by anchor ID (``*ID:* `` marker under each heading below).
See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets alone render invisible on GitHub, frozen code
snippets on real-bug sections only).

.. _mqar_k1_precision_control.module_overview:

Module overview: is K=1's plateau a precision bottleneck, not an architecture one?
------------------------------------------------------------------------------------

*ID:* ``mqar_k1_precision_control.module_overview``

Direct test of the user's hypothesis (see conversation): K=1's saturation at
~15-27% accuracy (``mqar_k1_calibration.py``, 200000 steps, dense FP4
multi-digit) might be a superposition-style precision/capacity bottleneck
(routing one of ~4032 possible key/value identities through several
FP4-quantized transforms), not an architecture, attention, or step-budget
problem.

Decisive test: same K=1 task, same step budget, FP32 in place of the FP4
multi-digit stack (``DISLDOLayer32`` -- this project's own established
"precision ceiling reference", isolates quantization coarseness from
architecture/training-dynamics limits, matching e.g.
``train_tile_curriculum.py``'s own "fp32" arm convention).

Run: ``python3 scripts/mqar_k1_precision_control.py [train_steps] [seed]
[eval_every]``.

.. _mqar_k1_precision_control.sparse_only_comparison_design:

Why both arms run sparse, not dense-fp4-vs-sparse-fp32
-----------------------------------------------------------

*ID:* ``mqar_k1_precision_control.sparse_only_comparison_design``

Found while building this, not guessed: ``DISLDOLayer32`` has no ``dense``
kwarg -- it's a pure diagnostic class, sparse-echo connectivity only, no
block4/dense support. So an fp32-vs-fp4 comparison can only be
apples-to-apples at matched connectivity if the FP4 side is ALSO run sparse
(not ``dense=True``, which is what ``mqar_k1_calibration.py`` used).

This script therefore runs BOTH arms sparse (sparse-fp32 and
sparse-fp4-multi-digit) so precision is the only isolated variable between
them -- the existing dense-fp4 result from ``mqar_k1_calibration.py``
remains a separate, already-available third reference point for the
connectivity axis, not directly compared against fp32 here (it's only
printed alongside the summary table for reference).

.. _mqar_k1_precision_control.state_width_crash_workaround:

Two sparse-connectivity crashes at the module's own default scale, and the workaround
------------------------------------------------------------------------------------------

*ID:* ``mqar_k1_precision_control.state_width_crash_workaround``

Also found while building this (not a fluke -- each reproduced directly).
Sparse (non-dense) connectivity needs a real per-row weight budget:
``train_mqar_k_sweep.py``'s own default ``MAX_WEIGHTS_PER_LAYER=512``
(sized for ``dense=True``, which ignores it) caused a heap-corruption crash
("free(): unaligned chunk detected in tcache") at ``state_width=128``
sparse. This script raises it to 4096, matching
``train_toy_tile_precision_comparison.py``'s own established budget at a
comparable scale.

Separately, ``state_width=128`` (the module's own ``EMBED_WIDTH=16``/
``COLUMN_NEURONS=8`` default) ALSO crashes sparse ``TrueMultiDigitLayer``
with a DIFFERENT heap-corruption error ("free(): too many chunks detected
in tcache"), even at ``max_weights=4096`` -- confirmed directly this does
NOT reproduce at ``state_width=32``. Out of scope to root-cause here, so
this control instead runs at ``EMBED_WIDTH=8``/``COLUMN_NEURONS=4``
(``state_width=32``) -- the scale used everywhere else in this project (the
copy-task curriculum, the model-level long-horizon test), sidestepping the
crash AND using the most validated configuration for ``gaussian_attention``
itself.

Consequence: this control's own numbers are NOT directly comparable to
``mqar_k1_calibration.py``'s 0.1667 (different ``state_width``) -- it
answers "is FP4 precision the bottleneck at the well-validated scale," not
"...at the specific scale that plateaued."

.. _mqar_k1_precision_control.num_cpus_race_workaround:

``NUM_CPUS=1``: sidestepping an already-known thread-safety race
------------------------------------------------------------------

*ID:* ``mqar_k1_precision_control.num_cpus_race_workaround``

``num_cpus>1`` hits an ALREADY-KNOWN pre-existing thread-safety race (same
class as ``test_stats_thread_safety.cpp``'s tracked segfault, and the
intermittent crash found earlier this session in
``test_toy_tile_precision_models.py`` at ``num_cpus=2``) -- confirmed
directly this is genuinely intermittent (1/3 runs crashed with the SAME
config, not deterministic), not something this control script introduced.
``num_cpus=1`` has no concurrency, sidesteps it entirely.
