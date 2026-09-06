``train_toy_recall_comparison.py`` research notes
=====================================================

Companion doc to ``scripts/train_toy_recall_comparison.py``. Source comments
point back here by anchor ID (``*ID:* `` marker under each heading below).
See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets alone render invisible on GitHub, frozen code
snippets on real-bug/non-obvious-derivation sections only).

.. _train_toy_recall_comparison.module_overview:

Module overview: second real MQAR comparison run, after root-cause fixes
------------------------------------------------------------------------------

*ID:* ``train_toy_recall_comparison.module_overview``

Real training experiment (not a pytest sanity check): does
``ToyTileRecurrence``'s column-averaged wide recurrent state actually learn,
at rough parity of context visibility with a dense causal baseline? Uses the
standard Multi-Query Associative Recall (MQAR) benchmark (Arora, Eyuboglu et
al., "Zoology", 2023 -- ``model/toy_recall_task.py``'s own
``generate_mqar_sequence``, a direct port of zoology's reference
implementation), ``AdamOptimizer`` + ``clip_grad_norm_`` (real global-norm
clipping -- see ``clip_grad_norm_``'s own docstring for why that's correct
now: both models are built from ``DenseTensorLinear``, nothing self-updates
during ``backward()`` anymore), warmup+cosine LR schedule.

This is the SECOND real run of this comparison. The first (see JOURNAL.md's
"Real MQAR comparison run: tile-recurrence fails, root cause found") found
``ToyTileRecurrence`` stuck at or below chance, and an ablation traced it to
two real design mistakes, both fixed here -- see
``train_toy_recall_comparison.second_run_fixes``.

.. _train_toy_recall_comparison.second_run_fixes:

Two fixes from the first run: full-width tiles, and dropping the per-tile column loss
------------------------------------------------------------------------------------------

*ID:* ``train_toy_recall_comparison.second_run_fixes``

1. ``num_tiles`` was a fixed, tiny constant (4) -- widened here to
   ``num_tiles = seq_len`` per config, removing the window-narrowness
   confound. This run tests whether the mechanism can learn at all when it
   CAN see the whole sequence (same visibility as the dense baseline's own
   full causal attention) -- testing genuine cross-tick recall BEYOND a
   narrow window stays explicitly deferred to a follow-up once this passes.
2. The old per-tile "column" loss (next-tile classification, no
   state-width expansion at all) is gone. ``ToyTileRecurrence`` now has a
   genuinely WIDER internal recurrent state (``state_width = embed_width *
   column_neurons``) than its input/output, read out via a parameter-free
   column-MEAN pool (not sum, not a learned down-projection -- see
   ``docs/research/toy_recall_models.rst:tile_recurrence_state_width_column_mean``
   for why: ``lm_head`` stands in for the real system's fixed-width
   pretrained output head). The unified per-position target (see
   ``train_toy_recall_comparison.unified_target_design``) replaces the old
   MQAR-only / column-classification split entirely, and now applies to
   BOTH models' training loops.

.. _train_toy_recall_comparison.unified_target_design:

Unified per-position training target, and why dense stays the unmodified control
----------------------------------------------------------------------------------

*ID:* ``train_toy_recall_comparison.unified_target_design``

The unified per-position target is used for ``ToyTileRecurrence`` ONLY (NOT
the dense control). At a query position, the target is the recalled value
(unchanged, the real task). Within the key/value CONTEXT-laydown region
(positions ``0`` to ``context_size-2``), the target is the literal real next
token -- true structure (each key is genuinely followed by its value in the
data), reinforcing exactly the key->value adjacency the later query needs.
Everywhere else is pure random filler (checked directly against
``model/toy_recall_task.py``: ``random_non_queries=True`` fills non-query
slots with noise, so ``tokens[i+1]`` is NOT a meaningful target at a query
position) -- skipped, not trained on.

``train_and_eval_dense`` is the CONTROL -- unmodified from the first real
comparison run (JOURNAL.md's "Real MQAR comparison run"). It trains on the
true MQAR pairs only, nothing else. It is deliberately NOT given the unified
context-region next-token target: that fix exists specifically for
``ToyTileRecurrence``'s column-mean readout (a mechanism for backprop-ing an
output error into a state much WIDER than the output -- see
``docs/research/toy_recall_models.rst:tile_recurrence_state_width_column_mean``).
A standard dense causal transformer has no such width mismatch (its hidden
state already matches ``lm_head``'s own input width directly), so there's no
equivalent problem for this loss to fix here -- changing the control's own
training procedure at the same time as fixing tile would confound whatever
the comparison is trying to isolate.

.. _train_toy_recall_comparison.build_targets_semantics:

``_build_targets``: query positions take priority over context-region filler
------------------------------------------------------------------------------

*ID:* ``train_toy_recall_comparison.build_targets_semantics``

Builds the unified per-position target dict (see
``train_toy_recall_comparison.unified_target_design``): starts from
``dict(mqar_pairs)`` (query positions -> recalled value), then
``setdefault``'s in the context-laydown region's real-next-token entries for
positions ``0`` to ``context_size-2`` -- ``setdefault`` rather than a plain
assignment so an actual MQAR query position occupying that same index (not
expected to happen given how ``generate_mqar_sequence`` lays out the
context/query regions, but not structurally impossible) would keep its real
recall target rather than being overwritten by the context filler rule.

.. _train_toy_recall_comparison.tile_window_construction:

``_build_tile_window``: broadcasting embeddings vs carrying forward ``M_prev``
---------------------------------------------------------------------------------

*ID:* ``train_toy_recall_comparison.tile_window_construction``

Builds the ``[num_tiles, state_width]`` window fed to ``model.step`` at
position ``i``. Real-token slots (source index ``src >= 0``) get their
``embed_width`` embedding broadcast up to ``state_width`` via ``np.repeat``
-- parameter-free, matching the readout's own parameter-free column-mean
pool (see ``train_toy_recall_comparison.second_run_fixes``). Fallback slots
(before the sequence starts, ``src < 0``) use ``M_prev[j]`` directly, which
is already ``state_width``-wide since ``M`` itself lives at ``state_width``
now, not ``embed_width``.

.. _train_toy_recall_comparison.config_and_dims_provenance:

``CONFIGS`` constraints, and where the ``hidden``/``mlp_hidden`` dimensions come from
-----------------------------------------------------------------------------------------

*ID:* ``train_toy_recall_comparison.config_and_dims_provenance``

Each ``CONFIGS`` entry is ``(seq_len, num_kv_pairs, vocab_size)``. MQAR
requires ``vocab_size > seq_len`` and ``seq_len >= 4*num_kv_pairs`` (see
``generate_mqar_sequence``'s own validation in
``model/toy_recall_task.py``).

``hidden=32, mlp_hidden=48`` in ``main()`` is zoology's own smallest real
attention baseline -- looked up, not guessed: zoology's ``models_repo.py``
own ``add_attention`` sweeps ``d_model`` in ``[32, 64, 128]`` with
``n_layers=2``, and ``d_model=32`` is the smallest of that sweep.
``tile_mlp_hidden = hidden * COLUMN_NEURONS * 2`` scales the tile model's
own MLP width with its wider ``state_width``, keeping the same
hidden:mlp_hidden ratio dense uses rather than leaving tile's MLP
under-sized relative to its much wider recurrent state.
