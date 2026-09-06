``eval_quantization.py`` research notes
==========================================

Companion doc to ``model/eval_quantization.py``. Source comments point back
here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows.

.. _eval_quantization.module_overview:

Isolating FP4 quantization's own quality cost from pruning's
-------------------------------------------------------------

*ID:* ``eval_quantization.module_overview``

Does B5's real FP4 quantization (``model/quantize.py``'s real
``FoldedLayer.from_descriptor``, sili__new PR #10) degrade MiniCPM5-1B-Base's
next-token prediction quality by an unacceptable amount, ON TOP OF B3's
already-validated pruning?

Same methodology as ``model/eval_pruning.py``'s ``compare_dense_vs_pruned``
(B3b): load the real HF model, evaluate with B3's pruned weights (the
accepted baseline), then with those SAME weights additionally FP4-quantized,
and compare next-token loss/perplexity/accuracy -- isolating quantization's
own effect from pruning's (already measured separately).

.. _eval_quantization.compare_pruned_vs_quantized_partial_load:

``compare_pruned_vs_quantized``: partial state-dict load, restore-on-exit
----------------------------------------------------------------------------

*ID:* ``eval_quantization.compare_pruned_vs_quantized_partial_load``

Evaluate ``model`` with ``pruned_dense_state_dict`` loaded (B3's already-
validated baseline), then with ``quantized_dense_state_dict`` applied on top
(partial -- only the suffixes ``model.quantize`` actually touches, loaded
with ``strict=False``), then restore the model's original weights in a
``finally`` block so the caller isn't left with a mutated model regardless
of whether evaluation succeeds.
