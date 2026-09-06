``toy_recall_models.py`` research notes
==========================================

Companion doc to ``model/toy_recall_models.py``. Source comments point back
here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets alone render invisible on GitHub, frozen code
snippets on real-bug/non-obvious-derivation sections only).

.. _toy_recall_models.module_overview:

Module overview: training-oriented toy models and the toy-track simplifications
-----------------------------------------------------------------------------------

*ID:* ``toy_recall_models.module_overview``

Small, TRAINING-oriented (not frozen-inference) models for validating the
tile-recurrence architecture on the synthetic induction-recall task
(``model/toy_recall_task.py``) -- see the approved plan
(``fuzzy-plotting-starlight.md``) for the full design rationale and the
toy-track-only simplifications (fixed embeddings, no positional encoding,
single head) this module deliberately makes.

.. _toy_recall_models.optimizer_choice_isolation_controls:

``DenseTensorLinear`` + ``AdamOptimizer``: why plain Tensor leaves and Adam replaced ``DISLDOLayer``/hand-rolled SGD
--------------------------------------------------------------------------------------------------------------------------

*ID:* ``toy_recall_models.optimizer_choice_isolation_controls``

Built from ``DenseTensorLinear`` (plain fp32 ``sili.tensor`` matmul, an
ordinary Tensor leaf like RMSNorm weights/centers/log_sigmas, not an
inline-self-updating primitive) trained via a real ``AdamOptimizer``, NOT
``sili.sparse_rnn.DISLDOLayer`` -- per direct decision, after two isolation
controls (``scripts/torch_mqar_control.py``,
``scripts/fp32_handrolled_control.py``) confirmed the earlier stuck-at-chance
training result was caused by this session's own hand-rolled optimizer
(``apply_gradient_step``: plain per-node-clipped SGD, no momentum) diverging,
NOT by FP4 quantization -- fp32 with that SAME hand-rolled optimizer diverged
identically. A full-precision + Adam control converged easily and fast on the
identical task; full precision with the OLD plain-SGD optimizer diverged
identically to the FP4 version. Momentum/adaptive per-parameter scaling was
the missing piece, not precision.

Fixing this for ``DISLDOLayer``'s own inline C++ weight update would need
real new work in sili__new (``disldo_backward``); fixing it for plain Tensor
leaves is pure Python (this module's own ``AdamOptimizer``) and directly
answers this track's actual question (does tile-recurrence learn genuine
recall), so per direct decision that's the path taken here. ``DISLDOLayer``/
FP4's own training dynamics remain a distinct, not-revisited-here concern
(already partially validated elsewhere per direct feedback --
importance-driven training behaves similarly to other optimizers).

``AdamOptimizer`` is standard Adam (Kingma & Ba, 2014) -- per-parameter
first/second moment estimates with bias correction, keyed by ``id(param)``
(``Tensor`` doesn't define ``__hash__``/``__eq__``, so default object-identity
hashing is exactly what's wanted here -- each distinct Tensor leaf gets its
own independent moment state). ``apply_gradient_step`` (plain SGD + zero_grad)
is kept only for tests/comparison, superseded by ``AdamOptimizer`` for real
training; it also skips any leaf whose ``.grad`` is still ``None`` (e.g. a
tile whose column target didn't apply this specific tick) rather than zeroing
against a nonexistent gradient.

.. _toy_recall_models.rmsnorm_tensor_gradient_flow:

``rmsnorm_tensor``: rebuilt from Tensor ops so gradient can flow through it
---------------------------------------------------------------------------------

*ID:* ``toy_recall_models.rmsnorm_tensor_gradient_flow``

Same formula as ``sili_block.rmsnorm``, but built from ``Tensor`` ops instead
of plain numpy -- needed here, unlike ``sili_block``'s frozen-inference
version, because this module actually trains through it.

.. _toy_recall_models.sigmoid_tensor_vs_bounded_gate:

``sigmoid_tensor``: plain symmetric gate, not ``bounded_gate``
-------------------------------------------------------------------

*ID:* ``toy_recall_models.sigmoid_tensor_vs_bounded_gate``

``1/(1+e^-x)``, built from existing Tensor primitives (no new op needed) --
deliberately NOT ``bounded_gate`` (defined elsewhere in ``sili.tensor``),
which is a different shape (``f(0)=0``, domain ``[0, inf)``) meant for
energy-gated non-negative activations, not a symmetric LSTM-style gate
(``f(0)=0.5``, all reals) for mixing two signals.

.. _toy_recall_models.cross_entropy_sum_and_predicted_token:

``cross_entropy_sum`` / ``predicted_token``: gather-based row loss, and a real overflow bug fixed by max-subtraction
------------------------------------------------------------------------------------------------------------------------

*ID:* ``toy_recall_models.cross_entropy_sum_and_predicted_token``

``Tensor`` has no ``__getitem__``/slicing, so per-row loss can't be computed
by indexing a row out directly -- built instead from
``reduce_sum(axis=-1)``/``exp``/``log`` (whole-tensor ops) plus ``gather``'s
FLAT indexing (``out[i] = a.flat[indices[i]]``) for both the per-row
log-sum-exp lookup and the ``(row, target)`` logit lookup. This handles
either a single ``(row, target)`` pair or many at once (e.g. every tile's own
"column" prediction target in one call) with no row-slicing needed anywhere.
``predicted_token`` similarly reads ``.data`` directly for its argmax
readout, since it's inference-time-only and no gradient can flow through an
argmax anyway.

The standard max-subtraction numerical-stability trick IS needed here: an
earlier version of this function skipped it, assuming toy-scale logits would
stay small -- wrong, confirmed directly, since raw ``exp()`` overflowed after
a few dozen real training steps (logits grow as the model gets more
confident, toy scale or not). The per-row max is computed from ``logits.data``
directly (plain numpy, detached) rather than a Tensor op -- subtracting a
constant shift from logits before the exp/sum/log chain doesn't change the
loss's gradient w.r.t. the ORIGINAL logits at all, so nothing needs to
backprop through the max itself. A ``reduce_max`` Tensor op (which doesn't
exist in ``sili.tensor``) would only be necessary if gradient had to flow
through the max, which it doesn't.

.. _toy_recall_models.backward_with_grad_clip_per_node:

``backward_with_grad_clip``: per-node clipping replicated from ``Tensor.backward()``
------------------------------------------------------------------------------------------

*ID:* ``toy_recall_models.backward_with_grad_clip_per_node``

Gradient-clipped replacement for ``loss.backward()`` -- clips the L2 norm of
EVERY node's incoming gradient (not just the final parameter gradients) to
``max_grad_norm``, right before that node's own ``_backward()`` fires. Still
used alongside ``AdamOptimizer`` (clip+Adam together is standard practice,
e.g. nanoGPT's own convention) -- clipping alone was never sufficient (see
``toy_recall_models.optimizer_choice_isolation_controls``), but it's still
good practice combined with real momentum.

``Tensor.backward()`` (``sili/tensor.py``) is just ``for node in
reversed(_topo_sort(self)): node._backward()`` -- replicated here with a clip
inserted in the loop, so every node's ``.grad`` (already fully accumulated
from all its consumers by the time its own turn comes, per topological order)
is bounded before it propagates further.

.. _toy_recall_models.lr_schedule_nanogpt_convention:

``lr_schedule``: linear warmup + cosine decay, matching nanoGPT
----------------------------------------------------------------------

*ID:* ``toy_recall_models.lr_schedule_nanogpt_convention``

Matches nanoGPT's own convention (widely-used, well-tested defaults for
small transformer training -- looked up rather than guessed, per direct
decision).

.. _toy_recall_models.clip_grad_norm_global_vs_per_node:

``clip_grad_norm_``: real global-norm clipping, and a NaN-gradient bug it fixes
--------------------------------------------------------------------------------------

*ID:* ``toy_recall_models.clip_grad_norm_global_vs_per_node``

Textbook GLOBAL gradient-norm clipping -- the total L2 norm ACROSS ALL of
``params``' gradients combined is capped to ``max_norm`` (matching
``torch.nn.utils.clip_grad_norm_`` exactly, including the control script that
used it: ``scripts/torch_mqar_control.py``).

``backward_with_grad_clip``'s per-NODE clipping exists specifically because
``DISLDOLayer``'s own weights self-update INLINE during ``backward()``, so a
true global-norm measurement isn't available before an update already
happened. That constraint doesn't apply to this module's models (they're
built from ``DenseTensorLinear``, see
``toy_recall_models.optimizer_choice_isolation_controls``): NOTHING updates
until ``optimizer.step()`` is called explicitly, so the real global norm can
be measured first, same as any standard training loop. Confirmed this
distinction actually matters, not just theoretically: per-node clipping still
let ``AdamOptimizer`` diverge on the real MQAR task (every one of many
parameter tensors independently allowed up to norm ``max_norm`` is a much
LARGER aggregate step than one norm-``max_norm`` budget shared across all of
them) -- this is the correct, stronger clip to use for models built from
``DenseTensorLinear``; call after ``loss.backward()`` (plain, ordinary -- not
``backward_with_grad_clip``) and before ``optimizer.step()``.

**Real bug, fixed**: ``total_norm > max_norm`` and ``total_norm > 0`` are
BOTH False when ``total_norm`` is NaN (IEEE 754) -- a NaN/Inf gradient would
silently skip the clip and sail straight into ``opt.step()``, permanently
poisoning Adam's m/v moving averages (every future step also NaN after that).
Confirmed as the real, direct cause of dense connectivity's permanent NaN
divergence via the C++-side analog of this exact bug (sili__new's
``ScalePolicy::update``, see its own docstring; JOURNAL.md 2026-08-10). Fix:
zero the gradient instead of clipping it when the norm isn't finite -- skips
this step's update rather than corrupting all future ones.

.. code-block:: python

   if not np.isfinite(total_norm):
       # NaN fails BOTH `> max_norm` and `> 0` comparisons, so an unguarded
       # NaN/Inf gradient would silently skip the clip below and permanently
       # poison Adam's m/v moving averages. Zero it instead of clipping it.
       for p in params:
           if p.grad is not None:
               p.grad = np.zeros_like(np.asarray(p.grad, dtype=np.float32))
       return total_norm

.. _toy_recall_models.tosmalltransformer_half_bandwidth:

``ToySmallTransformer``: depth-stacked baseline, and the ``half_bandwidth`` context-window knob
-------------------------------------------------------------------------------------------------------

*ID:* ``toy_recall_models.tosmalltransformer_half_bandwidth``

Stacked causal dense transformer -- each layer has its OWN distinct weights
(real depth-stacking, unlike tile-recurrence's single shared tile network).
Single-head attention, no positional encoding (see
``toy_recall_models.module_overview``'s simplifications).

``half_bandwidth`` defaults to unlimited (full causal visibility -- matches
every existing call site's behavior). Set to an int ``W`` to give this model
a GENUINELY bounded context window -- structurally unable to see more than
``W`` positions back, regardless of training. Used as the real "standard LLM"
stand-in for the out-of-context benchmark suite (see
``scripts/train_toy_beyond_context_comparison.py``) -- tile-recurrence's own
``num_tiles`` already plays this same role, no equivalent knob needed there.

.. _toy_recall_models.tile_recurrence_state_width_column_mean:

``ToyTileRecurrence``: why the recurrent state is wider than the output, and why column-averaging (not summing or selecting)
-----------------------------------------------------------------------------------------------------------------------------------

*ID:* ``toy_recall_models.tile_recurrence_state_width_column_mean``

One shared tile network (``DenseTensorLinear`` q/k/v/o/gate/up/down),
``gaussian_attention`` across tiles, additive energy-free gated residual
(toy scale -- no ``EnergyDynamics`` here; plain residual add is enough to
test the core retrieval mechanism without pulling in another moving part).
Single head, no positional encoding (see
``toy_recall_models.module_overview``'s simplifications).

``embed_width`` (E) matches the real token-embedding width. The internal
recurrent ``state_width`` = ``E * column_neurons`` (C) is deliberately WIDER
-- solves "how do you backprop a prediction error into a state much wider
than the output" (see the approved plan, ``fuzzy-plotting-starlight.md``, for
the full worked example: selecting a fixed subset starves the rest of
gradient, summing a column forces the state's own values small to avoid
blowup, AVERAGING a column works and stays in the same natural magnitude
range). ``lm_head`` stays fixed at ``embed_width`` -- standing in for the
real system's pretrained, fixed-width output head, which is exactly why the
width-reduction has to be the parameter-free column-mean, not a new learned
down-projection.

.. _toy_recall_models.tile_recurrence_step_qkv_blend:

``ToyTileRecurrence.step``: detached recurrence, and why Q/K/V blend input with carried state
-------------------------------------------------------------------------------------------------

*ID:* ``toy_recall_models.tile_recurrence_step_qkv_blend``

One recurrence tick. ``x_window``/``M_prev`` are ``[num_tiles, state_width]``
numpy, DETACHED (no BPTT, matching ``tile_recurrence.py``'s own design) --
both already widened to ``state_width`` by the caller (see
``_build_tile_window`` in ``scripts/train_toy_recall_comparison.py``), so
nothing here needs to handle mixed widths. Returns ``(M_new, logits)`` --
``logits`` has one row per tile's own column-mean-pooled next-token
prediction, but only the LAST row (the real current tick position) is ever
actually trained by the caller.

Q/K/V draw from ``x_window`` BLENDED with ``M_prev`` (per direct correction
-- see the ``feedback_attention_needs_combined_input_state`` note:
input-only attention forces anything relating fresh input to carried state
through an artificial "write to state, wait a tick, then attend" detour;
attention must be able to relate input and state directly, in one step).
