``toy_tile_recurrence_rmt_standard.py`` research notes
==============================================================

Companion doc to ``model/toy_tile_recurrence_rmt_standard.py``. Source
comments point back here by anchor ID (``*ID:* `` marker under each heading
below). See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this
follows (semantic dotted anchor IDs, visible ID markers via the ``*ID:*``
line since plain ``.. _id:`` targets alone render invisible on GitHub, frozen
code snippets on real-bug/non-obvious-derivation sections only). See
``docs/research/toy_tile_recurrence_rmt.rst`` for the sili-native sibling
this file is a deliberately non-sili control for -- that doc covers a
different model class entirely; this one only documents what's genuinely
different or notable about the STANDARD-torch variant.

.. _toy_tile_recurrence_rmt_standard.module_overview:

Why this file exists: a genuinely standard-torch RMT control, not another port
------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_standard.module_overview``

A genuinely STANDARD/textbook torch implementation of Recurrent Memory
Transformer's mechanism -- NOT another exact port of this project's own
model. Built per direct instruction after BOTH the sili-based control (task
#230: fp4 acc=0.200, fp32 acc=0.100) AND the exact-fidelity torch port (task
#232: acc=0.433) landed well short of the near-1.0 a working RMT should hit
trivially on K=1 MQAR, with torch's own result clearly "middle of the road"
rather than confirming either "the port is fine" or "the port is broken" --
direct guidance: check the standard-reference road FIRST (before bit-diffing
the two existing ports against each other), since it answers "is 0.43 even
in the right ballpark" before spending time tracing a divergence between two
implementations that might BOTH be wrong.

.. _toy_tile_recurrence_rmt_standard.standard_vs_custom_choices:

What's deliberately standard, not matched to this project's own choices
------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_standard.standard_vs_custom_choices``

- Standard multi-head-capable scaled dot-product attention
  (``softmax(QK^T/sqrt(d))V``), no Gaussian positional bias term -- that bias
  is this project's own invention, not part of RMT/transformers generally.
- A learned absolute positional embedding (``pos_embed``, added to every
  token before attention) instead of the Gaussian bias -- the standard way
  transformers give attention positional information at all.
- Standard LayerNorm (mean+variance normalization, learned affine), not this
  project's own RMSNorm.
- Standard residual connections, NO hard clip anywhere (LayerNorm's own
  normalization is the standard way scale is controlled).
- Standard end-to-end backprop: every parameter (including what this project
  calls "weights") trained by one shared ``torch.optim.Adam`` over the whole
  model -- no custom per-synapse RMSprop-with-importance rule, no
  L1-sparsity split-backward mechanism.

.. _toy_tile_recurrence_rmt_standard.controlled_variables:

What's kept the same as the rest of this investigation, deliberately
------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_standard.controlled_variables``

Kept identical on purpose, so this stays a fair "is the architecture/task
solvable at all" check rather than changing everything at once:

- Same task/harness (``model/toy_recall_task.py``'s ``generate_mqar_sequence``,
  same ``embed_table`` convention -- fixed random, never trained).
- Same ``num_memory_slots`` concat-into-the-window mechanism (memory tokens
  genuinely concatenated with content tokens, not additively merged).
- Same readout: mean-pool each ``column_neurons``-wide block down to
  ``embed_width`` before the final vocab projection (this project's own
  "column" convention) -- not swapped for e.g. a CLS-token readout, to
  isolate the OPTIMIZER/ATTENTION/NORM/CLIP question specifically, not the
  readout question too.
