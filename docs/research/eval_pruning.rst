``eval_pruning.py`` research notes
=====================================

Companion doc to ``model/eval_pruning.py``. Source comments point back here
by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows.

.. _eval_pruning.module_overview:

Isolating pruning's own quality cost from later conversion steps
----------------------------------------------------------------------

*ID:* ``eval_pruning.module_overview``

Does a given pruning decision actually degrade MiniCPM5-1B-Base's
next-token prediction quality by an unacceptable amount? Loads the real
HuggingFace model, evaluates it dense, then loads an already-pruned dense
state dict (built by ``model/prune.py`` -- no CSR conversion, no folding, no
column-averaging, no sili runtime involved at all here -- purely "does
zeroing these specific weights hurt", isolated from every later conversion
step) and compares next-token loss/perplexity and top-1 accuracy on a small
held-out text sample.

This module has NO pruning-construction logic of its own -- see
``model/prune.py`` for that (``prune_state_dict``/``prune_state_dict_by_role``,
plus ``prune.sparse_state_to_dense_state_dict`` to turn either's output back
into plain tensors loadable via ``model.load_state_dict``). Keeping the two
concerns separate avoids two parallel implementations of "which weights get
zeroed."

torch/transformers-only; not part of the sili runtime path, only a
validation step for the conversion pipeline.

.. _eval_pruning.eval_texts_design:

``EVAL_TEXTS``: sanity-scale, not a rigorous benchmark
---------------------------------------------------------

*ID:* ``eval_pruning.eval_texts_design``

Short, diverse, hand-written passages -- no external dataset dependency.
Plain declarative English is exactly what a BASE model (no instruction
tuning) should already predict well; this isn't meant to be a rigorous
benchmark, just a sanity-scale check that pruning hasn't broken the model in
an obvious way.

.. _eval_pruning.eval_texts_heldout_design:

``EVAL_TEXTS_HELDOUT``: overfitting check for chosen thresholds
---------------------------------------------------------------------

*ID:* ``eval_pruning.eval_texts_heldout_design``

A second, disjoint set (different topics/style) -- never used during
threshold search, only to check a chosen set of thresholds isn't overfit to
the specific ``EVAL_TEXTS`` snippets above. Confirmed on the final
``DEFAULT_TARGET_SPARSITY_BY_ROLE`` thresholds: pruned accuracy held steady
across both sets (0.482 vs. 0.478) even though the dense baseline itself
varies more between them (0.503 vs. 0.584) -- see ``JOURNAL.md``.

.. _eval_pruning.compare_dense_vs_pruned_restore:

``compare_dense_vs_pruned``: clean-start + restore-on-exit
-----------------------------------------------------------------

*ID:* ``eval_pruning.compare_dense_vs_pruned_restore``

Evaluate ``model`` dense, then with ``pruned_dense_state_dict`` loaded in,
then restore its original weights in a ``finally`` block -- so the caller's
model isn't left mutated regardless of whether evaluation succeeds.
``pruned_dense_state_dict`` is typically
``prune.sparse_state_to_dense_state_dict(prune.prune_state_dict_by_role(...)[0])``.

Returns a plain dict so callers/tests don't need ``EvalResult`` imported.
