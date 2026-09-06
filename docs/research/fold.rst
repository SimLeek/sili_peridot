``fold.py`` research notes
=============================

Companion doc to ``model/fold.py``. Source comments point back here by
anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/rnn_fold.rst`` for the directly analogous
folding module this pattern follows (semantic dotted anchor IDs, visible
ID markers, frozen code snippets on real-bug/non-obvious-derivation
sections only).

.. _fold.module_overview:

Why each suffix gets its own FoldedBlockDescriptor, never bundled
---------------------------------------------------------------------

*ID:* ``fold.module_overview``

Fold each of MiniCPM5's 7 per-layer 2-D suffixes (``self_attn.q_proj``,
``k_proj``, ``v_proj``, ``o_proj``, ``mlp.gate_proj``, ``up_proj``,
``down_proj``) across all 24 layers INDEPENDENTLY -- each suffix gets its
OWN ``FoldedBlockDescriptor``, never bundled together into one.
``sili__new``'s ``fold_block_group`` bundles every suffix sharing a
block-index range into ONE descriptor if given the whole state dict at
once -- ``FoldedLayer.forward`` sums every suffix in its ``layers`` dict
down to the FIRST suffix's ``out_dim``, silently wrong here since every
suffix has a different ``out_dim`` (q=2048, k/v=256, o=1536,
gate/up=4608, down=1536). ``fold_suffix`` filters ``sparse_state`` down
to one suffix's own 24 per-layer tensors before calling
``fold_block_group``, specifically to avoid that -- so every
``FoldedBlockDescriptor`` produced here has exactly one key in
``stacked_weights``, and ``build_folded_layers``/the streaming variants
below build exactly one ``SparseLinearLayer`` per call, never bundling.

.. _fold.band_half_width_rope_deferral:

Why ``band_half_width`` is not auto-inferred here on purpose
-------------------------------------------------------------------

*ID:* ``fold.band_half_width_rope_deferral``

``band_half_width`` is NOT set correctly here on purpose:
``fold_block_group``'s own auto-heuristic
(``infer_seq_len_from_attn_weight``) assumes fixed-position attention,
which is wrong for MiniCPM5's RoPE. B6 is where the real value gets
decided (a practical context window for this environment, not the full
131072 training context) -- this module exposes
``band_half_width_override`` so B6 can supply it, defaulting to ``None``
(the current, known-wrong-for-RoPE auto-heuristic) so folding itself
doesn't block on that decision.

.. _fold.expected_out_dim_shape_check:

Why every suffix's folded ``out_dim`` is checked against config at fold time
---------------------------------------------------------------------------------

*ID:* ``fold.expected_out_dim_shape_check``

``_EXPECTED_OUT_DIM_PROPERTY`` maps each suffix to the
``MiniCPM5Config`` property its per-fold-step ``out_dim`` must match.
``fold_suffix`` checks this immediately after ``fold_block_group``
returns, so a shape mismatch fails loudly right here with the suffix
name and expected/actual dims, instead of surfacing later as a
confusing error deep inside ``FoldedLayer`` construction.

.. _fold.build_folded_layers.rank1_value_scale:

``build_folded_layers``: why ``value_scale_mode="rank1"`` is the default
------------------------------------------------------------------------------

*ID:* ``fold.build_folded_layers.rank1_value_scale``

B5: turns each suffix's ``FoldedBlockDescriptor`` into a real sili
``FoldedLayer`` (``FoldedLayer.from_descriptor``), one suffix at a time.

``value_scale_mode="rank1"`` (default, B5a): ``from_descriptor``'s
original ``"per_row"`` scheme (one ``value_scale`` per input row, shared
across all ``n_out`` output positions) collapsed real next-token
accuracy from 0.482 to ~0.09-0.12 on MiniCPM5 -- per-output magnitude
within one folded layer varies by a min/max ratio as low as ~0.05-0.10,
so a single per-row scale wastes most of FP4's resolution on most
outputs. ``"rank1"`` adds a genuine per-output scale too (``sili__new``'s
``fit_rank1_scale_envelope``), recovering to ~0.297 accuracy in
simulation before this was wired into the real C++ path. See
``JOURNAL.md`` for the full investigation and ``eval_quantization.py``
for the real (not simulated) confirmation.

.. _fold.build_folded_layers_streaming.memory_discipline:

``build_folded_layers_streaming``: why suffixes are popped one at a time
------------------------------------------------------------------------------

*ID:* ``fold.build_folded_layers_streaming.memory_discipline``

Folds and builds a real sili ``FoldedLayer`` for each suffix ONE AT A
TIME, popping that suffix's 24 raw per-layer tensors out of
``sparse_state`` (MUTATES it in place) immediately after folding -- so an
already-processed suffix's memory is released before the next suffix's
fold+construct begins, instead of holding all 219 tensors resident for
the whole loop.

Why this exists (measured, not assumed, on the real checkpoint -- see
``JOURNAL.md``): ``FoldedLayer.from_descriptor``'s own C++ construction
adds ~0 marginal peak RSS beyond ``fold_suffix``'s own CSR-stacking step
-- even for the largest suffix (``mlp.gate_proj``, budget~=170M,
nnz~=136M). The real memory driver is holding raw per-layer dense
tensors (already fully resident in RAM from B3's pruning, since most
suffixes are still 80-93% dense at B3's validated per-role thresholds --
deeper sparsification is intentionally deferred to training-time
synaptogenesis, not this conversion step) for suffixes that have ALREADY
been folded and are no longer needed. A naive "fold each suffix in a
loop, keep ``sparse_state`` untouched" version (``fold_all_suffixes`` +
``build_folded_layers`` called on its output) lets that dead weight
accumulate across the whole loop instead.

Using this function is destructive to ``sparse_state`` by design -- pass
a dict you don't need afterward, or use ``fold_all_suffixes`` if you
need to keep ``sparse_state`` intact and can afford the extra memory.

.. _fold.build_and_save_folded_layers.disk_offload_headroom:

``build_and_save_folded_layers``: why each ``FoldedLayer`` is discarded after saving
-------------------------------------------------------------------------------------------

*ID:* ``fold.build_and_save_folded_layers.disk_offload_headroom``

Same one-suffix-at-a-time streaming discipline as
``build_folded_layers_streaming`` (MUTATES ``sparse_state`` in place,
releasing each suffix's raw tensors as soon as it's folded), but
additionally serializes each suffix's real ``FoldedLayer`` to disk and
discards the live C++ object immediately after -- so peak memory never
holds more than ONE suffix's built ``FoldedLayer`` at a time, on top of
whatever's left of ``sparse_state``.

Confirmed on the real checkpoint (see ``JOURNAL.md``): holding all 7
real ``FoldedLayer`` objects simultaneously
(``build_folded_layers_streaming``'s own return value) peaks at ~13.5GB
on this 15GB machine -- workable but with barely any headroom left for
anything else. This function exists to give that headroom back for the
conversion step specifically; B7's real runtime will still need all 7
loaded together to run a forward pass, but building/verifying them
doesn't.

``_save_folded_layer_state_dict`` saves one ``.npz`` per suffix --
``FoldedLayer.state_dict()`` is already a plain dict of numpy arrays
(per-suffix weight/scale arrays plus ``n_folds``/``out_dim``/``lr``), so
``np.savez`` handles it directly with no torch/sili dependency needed to
read it back later.

.. _fold.reference_fold_forward.quantization_isolation:

``reference_fold_forward``: isolating quantization error from fold-approximation error
-------------------------------------------------------------------------------------------

*ID:* ``fold.reference_fold_forward.quantization_isolation``

The UNQUANTIZED analytic fold-sum for one suffix: ``x @ W^T`` per fold
step, summed over the fold axis -- exactly ``FoldedLayer.forward``'s own
math, computed directly from the exact float32 stacked CSR values with
no FP4 rounding. Comparing this against the real (quantized)
``FoldedLayer.forward(x)`` on the same ``x`` isolates quantization's own
effect, since both compute the identical fold-sum given the identical
input -- the fold-approximation itself (same ``x`` fed to every virtual
layer) contributes zero difference between the two, only quantization
does.

.. _fold.verify_lossless.nnz_invariant:

``verify_lossless``: B4's own correctness check
-----------------------------------------------------

*ID:* ``fold.verify_lossless.nnz_invariant``

Total nonzeros per suffix must be IDENTICAL before and after folding --
stacking must never lose or duplicate real weight values, only change
their storage layout. ``FoldReport.lossless`` compares the per-suffix
nnz-before/nnz-after dicts directly; a mismatch on any suffix means the
stacking step is not a pure layout change and needs investigating before
trusting anything downstream of it.
