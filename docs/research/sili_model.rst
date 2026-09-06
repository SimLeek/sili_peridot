``sili_model.py`` research notes
===================================

Companion doc to ``model/sili_model.py``. Source comments point back here by
anchor ID (``*ID:* `` marker under each heading below). See
``docs/research/sili_block.rst`` for the pattern this follows (semantic
dotted anchor IDs, visible ID markers, frozen code snippets on real-bug/
non-obvious-derivation sections only).

.. _sili_model.module_overview:

Module scope: B7 full-model assembly, and why embed/head stay outside the folded suffixes
----------------------------------------------------------------------------------------------

*ID:* ``sili_model.module_overview``

B7: assembles the full model (embedding lookup -> B6's fold-depth
recurrence -> final RMSNorm -> lm_head) and evaluates next-token
prediction quality entirely through sili -- no torch forward pass anywhere
in this module, mirroring ``eval_pruning.evaluate_next_token_prediction``'s
exact loss/accuracy methodology so the two are directly comparable (same
teacher-forced next-token loss/top-1-accuracy definition, same
``EvalResult``, so perplexity/accuracy numbers from the sili path and the
torch path can be placed side by side).

``embed_tokens``/``lm_head`` are never part of the 7 folded suffixes (B3
never prunes them via the fold machinery, B5 never quantizes them) -- read
as plain float32/CSR arrays and applied via a numpy gather / matmul,
matching the same scope boundary as ``sili_block.py``'s RMSNorm/RoPE
weights (see ``sili_block.module_overview``): both are "read directly from
``sparse_state``, never built into a ``SparseLinearLayer``" categories, just
at different tensor shapes (1-D layernorm vectors there, 2-D embed/head
matrices here).

.. _sili_model.embed_head_sparse_storage:

``_to_sparse_or_dense``: keeping embed_tokens/lm_head as real scipy CSR instead of densifying
----------------------------------------------------------------------------------------------------

*ID:* ``sili_model.embed_head_sparse_storage``

B3's role-based pruning thresholds DO prune ``embed_tokens``/``lm_head``
(both 2-D, ~20%/~70% density respectively on the real checkpoint) even
though they're outside the folded-suffix machinery. Storing them fully
dense regardless (the previous behavior) cost ~1.5GB for two tensors that
compress to ~360MB combined at their real density -- a real memory win, not
a hypothetical one.

``_to_sparse_or_dense`` keeps a real ``{"csr": ...}`` entry as a
``scipy.sparse.csr_matrix`` instead: cheap row gather for
``embed_tokens`` (indexing a CSR by row is fine), cheap sparse-dense matmul
for ``lm_head``. scipy is already a transitive dependency (via
``sili__new``), so this doesn't add a new one. ``{"raw": ...}`` entries
(never pruned to CSR by B3) fall back to plain dense float32, matching the
1-D layernorm vectors' own always-raw convention (see
``sili_model.module_overview``).

``build_sili_model`` pops ``embed_tokens``/``lm_head``/final-norm and every
fold step's real sili layers out of ``sparse_state`` (MUTATES it -- the
same streaming-pop discipline ``build_step_layers`` already uses, so the
whole checkpoint's memory footprint never doubles during construction) and
bundles everything ``compute_logits_sili``/``evaluate_next_token_prediction_sili``
need into one dict. ``embed_tokens``/``lm_head`` are kept as scipy CSR
matrices when B3 pruned them (the real checkpoint case), not densified --
see ``_to_sparse_or_dense`` above for why.

.. _sili_model.activation_density_interface:

``compute_logits_sili``: the activation_density parameter at the whole-model assembly level
--------------------------------------------------------------------------------------------------

*ID:* ``sili_model.activation_density_interface``

``activation_density``: ``None`` (default) = dense forward throughout
(current/original behavior, unchanged numerically from before sparsity was
added); a float in ``(0, 1]`` sparsifies EVERY projection's input
activation to that per-token top-k density and routes through SISLDO's
``forward_sparse`` path instead of dense ``forward_dense``; a dict isolates
specific suffixes (e.g. sparsify only ``.mlp.down_proj.weight``); a list of
length ``cfg.num_hidden_layers`` isolates specific fold steps, letting a
caller sparsify only some LAYERS while leaving others dense. See
``sili_block.forward.activation_density_argpartition`` for the per-row
``np.argpartition`` mechanism this bottoms out in, and
``sili_block.run_folded_recurrence.window_curriculum_design`` for why a
per-step list is not necessarily equivalent between an early and a late
fold-depth position (top-k truncation errors compound through the
recurrence's accumulated state).

.. _sili_model.lm_head_sparse_matmul:

Computing logits without ever densifying a sparse lm_head
-------------------------------------------------------------

*ID:* ``sili_model.lm_head_sparse_matmul``

When ``lm_head`` is a scipy CSR matrix (see
``sili_model.embed_head_sparse_storage``), ``compute_logits_sili`` computes
logits as ``(lm_head @ hidden.T).T`` rather than ``hidden @ lm_head.T``:
sparse ``[vocab, hidden]`` @ dense ``[hidden, T]`` -> dense ``[vocab, T]``,
transposed back to the usual ``[T, vocab]`` shape. This never materializes
``lm_head`` densely at any point -- the ``.T`` is there purely to match
``hidden @ lm_head.T``'s shape/orientation using an operand order scipy can
actually multiply against a dense array efficiently.

.. code-block:: python

   # lm_head: scipy.sparse.csr_matrix [vocab, hidden]; hidden: [T, hidden]
   # sparse @ dense (not dense @ sparse-transposed) avoids ever
   # densifying lm_head, at the cost of a transpose on the small side.
   return (lm_head @ hidden.T).T.astype(np.float32)

.. _sili_model.eval_parity:

Evaluation parity with eval_pruning's torch-based methodology
-------------------------------------------------------------------

*ID:* ``sili_model.eval_parity``

``_cross_entropy_and_accuracy`` uses the standard numerically-stable
softmax cross-entropy (mean over N) + top-1 accuracy -- deliberately the
SAME definition as HuggingFace's shifted ``labels=input_ids`` loss, not a
from-scratch metric, so a loss number computed here means the same thing as
a loss number computed by the torch reference path.

``evaluate_next_token_prediction_sili`` is the sili-only counterpart to
``eval_pruning.evaluate_next_token_prediction``: same teacher-forced
next-token loss/top-1-accuracy definition, same ``EvalResult`` return type,
so perplexity/accuracy from the two evaluation paths are directly
comparable rather than needing a conversion or caveat. This comparability
is the whole point of keeping this module torch-free (see
``sili_model.module_overview``) -- any numerical difference between the two
paths is then attributable to the sili engine itself, not to a
methodology mismatch.
