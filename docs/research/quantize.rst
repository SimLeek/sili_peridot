``quantize.py`` research notes
=================================

Companion doc to ``model/quantize.py``. Source comments point back here by
anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows.

.. _quantize.module_overview:

Real FP4 quantization via FoldedLayer, not a numpy simulation
------------------------------------------------------------------

*ID:* ``quantize.module_overview``

B5's real FP4 quantization of MiniCPM5's per-layer suffix weights, using
sili__new's actual ``FoldedLayer.from_descriptor`` (rank-1 or per-row
quantization scale, sili__new PR #10) -- not a Python/numpy simulation of
it. Builds one suffix's real ``FoldedLayer`` at a time, reads back its true
post-quantization weight values, and discards it before moving to the next
suffix.

.. _quantize.build_quantized_dense_state_dict_streaming_discipline:

``build_quantized_dense_state_dict_streaming``: one-suffix-at-a-time memory discipline
-------------------------------------------------------------------------------------------

*ID:* ``quantize.build_quantized_dense_state_dict_streaming_discipline``

Fold+quantize one suffix at a time (MUTATES ``sparse_state``, popping each
suffix's raw tensors out as soon as it's folded -- same discipline as
``fold.build_folded_layers_streaming``). Returns ``{layer_name: dense
tensor}`` for every suffix*layer, holding at most one suffix's real
``FoldedLayer`` and one ``[n_in, n_out]`` reconstructed array at a time.

Reads the real post-quantization weight back from the built layer via its
zero-copy ``ptrs``/``indices``/``weights_vals`` plus per-row
``get_value_scale``/per-col ``get_output_scale`` (``true_w = weights_vals *
value_scale[row] * output_scale[col]``, see ``cpu_backend.cpp``) -- never
densifies the stacked ``[n_in, n_out]`` matrix more than once, and only for
this suffix's own reconstruction, not for the scale fit itself (that's
sili__new's own sparse-triplet ``fit_rank1_scale_envelope``, already used
internally by ``from_descriptor``).
