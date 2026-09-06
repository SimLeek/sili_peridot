``prune.py`` research notes
==============================

Companion doc to ``model/prune.py``. Source comments point back here by
anchor ID (``*ID:* `` marker under each heading below). See
``docs/research/fold.rst`` for the directly analogous checkpoint-conversion
module this pattern follows (semantic dotted anchor IDs, visible ID
markers, frozen code snippets on real-bug/non-obvious-derivation sections
only).

.. _prune.module_overview:

Why a single global threshold was tried first, and why it failed
---------------------------------------------------------------------

*ID:* ``prune.module_overview``

Prunes MiniCPM5's checkpoint to a mixed sparse(CSR)/dense payload. A
SINGLE global threshold (``prune_state_dict``, ``DEFAULT_TARGET_SPARSITY``)
turned out to be the wrong tool for this model, discovered the hard way by
actually measuring next-token prediction quality (``model/eval_pruning.py``)
before this landed, not by CSR-shape/sparsity-percentage reasoning alone.

Doesn't reuse ``sili__new``'s toy-Mistral pruning defaults uncritically
(see ``todolist.md`` Phase B3): ``target_sparsity=0.5`` (``sparse_prune``'s
own default) leaves almost the entire model dense here -- CSR only wins
over dense once a tensor's OWN sparsity clears ~70%, given
``_keep_dense_reason``'s 12-bytes-per-nonzero estimate vs. 4 bytes/element
dense. ``DEFAULT_TARGET_SPARSITY=0.8`` gets real compression (~1.65x,
127/170 tensors sparse) at this global level.

But at ``target_sparsity=0.8``, next-token accuracy collapses to 0.0 (vs.
0.503 dense) with NO retraining involved -- a single global threshold
destroys the model long before it reaches CSR-viable sparsity. Swept
``target_sparsity`` globally: quality holds through ~0.2, degrades
continuously (not one sharp cliff) from ~0.3, catastrophic by ~0.4-0.5 --
nowhere near the ~70%+ per-tensor sparsity real CSR compression needs.
There is no "conservative default" middle ground to just pick.

.. _prune.default_target_sparsity_by_role.iterative_search_result:

``DEFAULT_TARGET_SPARSITY_BY_ROLE``: the actual result of a per-role search
----------------------------------------------------------------------------

*ID:* ``prune.default_target_sparsity_by_role.iterative_search_result``

Per-tensor-ROLE sensitivity varies enormously: ``embed_tokens`` tolerated
90% sparsity fine; ``v_proj`` -- architecturally near-identical to
``k_proj``, which tolerated 70% -- collapsed already past ~25%. A first
coarse sweep (0.3/0.5/0.7/0.9 only) suggested ``v_proj``'s safe zone was
~0.05; a finer sweep (0.03 steps) found the real safe zone is actually
~0.2 -- the coarse grid was simply too coarse to see it, not evidence the
fine-grained safe zone didn't exist.

Isolated per-role thresholds compound MUCH worse once combined: the first
combined attempt (every role at its own isolated-safe number) gave
accuracy 0.111, not the ~0.4-0.5 isolated numbers implied. ``q_proj`` +
``k_proj`` together cost about 2x either alone (they interact directly in
QK^T); ``o_proj`` and ``v_proj`` each roughly quadrupled the cumulative
damage on their own turn in the stepwise trace. Reaching an acceptable
combined result took 3 rounds of "find whichever role caused the biggest
single jump in the cumulative trace, shrink just that one, recheck the
combined result" by hand -- then generalized into
``sili.conversion.prune_sensitivity`` (``group_tensor_names_by_role``,
``sweep_group_sensitivity``, ``apply_group_thresholds``,
``stepwise_cumulative_eval``, ``iterative_threshold_search``, sili__new
PR #8) as a reusable grouping/sweeping/combining-verification/
threshold-walkback loop instead of an ad hoc one-off script.
``iterative_threshold_search`` automates the manual "shrink the worst
offender, recheck" loop above -- greedy, not globally optimal (shrinking
one role changes how much a role added AFTER it costs, since costs
compound along the step order), but it's exactly the procedure that
worked here, made repeatable.

``DEFAULT_TARGET_SPARSITY_BY_ROLE`` below is that search's validated
result: combined next-token accuracy 0.482 (search set) / 0.478
(independent held-out set) vs. a 0.503 dense baseline -- real quality
preserved, at the cost of most groups being well below CSR-viable
sparsity for now. That's intentional at this stage: the actual memory
win this whole pipeline is chasing is Phase B4's folding (24 layers ->
1), which doesn't need genuine sparsity, just CSR-shaped tensors --
deeper sparsification is left to training-time synaptogenesis, not this
conversion-time step. See ``JOURNAL.md`` for the full search, including
the held-out validation confirming the thresholds aren't overfit (dense
baseline itself varied more between the two text samples, 0.503 / 0.584,
than pruned accuracy did, 0.482 / 0.478).

Keys in ``DEFAULT_TARGET_SPARSITY_BY_ROLE`` are ROLE names (as returned by
``_role_of``), not raw group keys from ``group_tensor_names_by_role`` --
those embed a regex-derived prefix/suffix that isn't a stable,
human-writable identifier.

.. code-block:: python

   # WRONG (measured on the real checkpoint): one global threshold gets
   # real CSR compression but destroys the model.
   DEFAULT_TARGET_SPARSITY = 0.8  # 127/170 tensors sparse, ~1.65x compression
   # next-token accuracy collapses to 0.0 (vs. 0.503 dense), no retraining

   # Fix: a separate threshold per tensor ROLE, found by iterative search
   # (sili.conversion.prune_sensitivity.iterative_threshold_search).
   DEFAULT_TARGET_SPARSITY_BY_ROLE = {
       "embed_tokens": 0.8, "q_proj": 0.4, "k_proj": 0.4, "lm_head": 0.3,
       "gate_proj": 0.2, "up_proj": 0.25, "down_proj": 0.1, "o_proj": 0.1,
       "v_proj": 0.05,
   }
   # combined accuracy 0.482 (search set) / 0.478 (held-out) vs 0.503 dense

.. _prune.record_pruned_tensor.shared_format_decision:

``_record_pruned_tensor``: shared per-tensor logic between both entry points
-------------------------------------------------------------------------------

*ID:* ``prune.record_pruned_tensor.shared_format_decision``

Zeroes below ``threshold``, decides sparse vs. dense storage, and records
both into ``sparse_state``/``report`` in place. Used by both
``prune_state_dict`` (one global threshold for every tensor) and
``prune_state_dict_by_role`` (a different threshold per tensor, looked up
by role) so the format-decision/reporting logic isn't duplicated between
them.

.. _prune.role_of.fail_loud_on_unknown_role:

``_role_of``: raising on an unrecognized role instead of silently skipping
------------------------------------------------------------------------------

*ID:* ``prune.role_of.fail_loud_on_unknown_role``

Maps a ``group_tensor_names_by_role()`` key (an internal, regex-derived
identifier) to one of MiniCPM5's own known tensor roles. Raises if a group
doesn't match any known role -- silently skipping an unknown tensor role
would mean it never gets pruned at all without anyone noticing.

.. _prune.prune_state_dict.reference_only:

``prune_state_dict``: kept for reference/comparison, not the recommended path
----------------------------------------------------------------------------------

*ID:* ``prune.prune_state_dict.reference_only``

Prunes ``state_dict`` with ONE global threshold for every eligible tensor,
to a ``{name: {"csr": tensor, "shape": ...}}`` or
``{name: {"raw": tensor, "shape": ...}}`` payload (same entry shapes
sili__new's ``rnn_fold.py`` / ``sparse_runtime.py`` already expect via the
``"raw"``/``"csr"`` keys), plus a ``PruneReport`` for B3a-style
verification (actual density, not just "it ran").

Kept for reference/comparison -- see
:ref:`prune.module_overview` for why ``prune_state_dict_by_role`` is the
actual recommended path; a single global threshold cannot get real CSR
compression without destroying next-token prediction quality on this
model.

``min_abs_param``: explicit threshold, bypassing calibration entirely if
given (same priority convention as sili__new's ``sparsify_model`` --
explicit always wins). Mainly for tests that need a deterministic,
hand-picked threshold rather than whatever the calibrated percentile
happens to be.

.. _prune.prune_state_dict_by_role.grouping_and_per_role_calibration:

``prune_state_dict_by_role``: the recommended path -- per-role grouping and calibration
--------------------------------------------------------------------------------------------

*ID:* ``prune.prune_state_dict_by_role.grouping_and_per_role_calibration``

Prunes ``state_dict`` with a SEPARATE calibrated threshold per tensor role
(see :ref:`prune.default_target_sparsity_by_role.iterative_search_result`
for why: a single global threshold destroys this model's next-token
prediction quality long before it reaches CSR-viable sparsity). This is
the recommended path, not ``prune_state_dict``.

Groups tensors via sili__new's ``group_tensor_names_by_role`` (repeated
per-layer tensors grouped by shared suffix, e.g. every layer's own
``self_attn.q_proj.weight``; ``embed_tokens``/``lm_head``/``norm`` each
their own singleton group), calibrates ONE threshold per group on that
group's own tensors, then reuses the same per-tensor sparse/dense
format-decision logic as ``prune_state_dict`` (see
:ref:`prune.record_pruned_tensor.shared_format_decision`).

Returns the same ``(sparse_state, PruneReport)`` shape as
``prune_state_dict`` -- ``report.min_abs_param`` is a
``{role: threshold}`` dict here instead of a single float.

.. _prune.sparse_state_to_dense_state_dict.round_trip_for_eval:

``sparse_state_to_dense_state_dict``: round-tripping back to a loadable state dict
----------------------------------------------------------------------------------------

*ID:* ``prune.sparse_state_to_dense_state_dict.round_trip_for_eval``

Converts a ``prune_state_dict`` / ``prune_state_dict_by_role`` payload
(each entry ``{"csr": tensor, "shape": ...}`` or
``{"raw": tensor, "shape": ...}``) back into a plain
``{name: tensor}`` state dict with each tensor's ORIGINAL shape restored
-- directly loadable via ``model.load_state_dict``, e.g. for
``eval_pruning.compare_dense_vs_pruned`` to check a pruning decision's
actual effect on model quality.
