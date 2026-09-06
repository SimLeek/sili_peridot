``toy_tile_recurrence_rmt_torch.py`` research notes
=======================================================

Companion doc to ``model/toy_tile_recurrence_rmt_torch.py``. Source comments
point back here by anchor ID (``*ID:* `` marker under each heading below).
See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets alone render invisible on GitHub, frozen code
snippets on real-bug/non-obvious-derivation sections only).

This is a sibling of ``model/toy_tile_recurrence_rmt.py`` (see
``docs/research/toy_tile_recurrence_rmt.rst``) and
``model/toy_tile_recurrence_rmt_standard.py``
(``docs/research/toy_tile_recurrence_rmt_standard.rst``) -- this doc only
covers what's genuinely different/notable about the torch reimplementation
itself (why it exists, what fidelity guarantees it makes, and the real bugs
found while building it), not the shared architecture already documented in
those files.

.. _toy_tile_recurrence_rmt_torch.module_overview:

Module purpose: an exact torch port, built as a control against a sili-engine bug
--------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.module_overview``

Plain-torch EXACT port of ``model/toy_tile_recurrence_rmt.py``
(``ToyTileRecurrenceRMT``), built per direct instruction ONLY after the
sili-based control failed to learn K=1 MQAR at either precision -- "move the
model to torch as exactly the same as is possible", NOT swap in
torch-idiomatic defaults (Adam instead of the real per-synapse update,
different clip/attention/norm conventions, etc.). The whole point is
isolating "is sili__new the engine broken" from "is even a
correctly-implemented proven architecture failing here" -- that requires
holding every other variable fixed.

Design: uses REAL torch autograd for gradient PROPAGATION through the whole
step (one connected graph, one ``loss.backward()`` call, exactly like sili's
own single topological backward pass) -- this is the safe way to get correct
chain-rule gradients through concat/RMSNorm/attention without hand-deriving
a backward pass. An earlier draft of this file tried a hand-derived backward
and was correctly rejected: a bug in a manually-derived backward would
itself become a torch-port-specific confound, defeating the entire purpose
of this control. The ONLY place this deviates from ordinary torch training
is the WEIGHT UPDATE itself: ``DISLDOTorchLinear`` does NOT use
``torch.optim`` on its weights -- after the single ``backward()`` call, each
layer reads its own ``true_weight.grad`` (exactly ``g=dL/d(true_weight)``,
matching ``disldo_backward``'s own g signal) and applies sili's real
per-synapse update rule by hand.

.. _toy_tile_recurrence_rmt_torch.disldo_reproduced_training_rule:

The production training rule this reproduces, extracted directly from the C++ source
------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.disldo_reproduced_training_rule``

``DISLDOTorchLinear`` reproduces sili__new's actual training rule, extracted
directly from the C++ source rather than guessed:

- Per-synapse ``ci`` (RMSprop-style second moment) and weight update:
  ``delta_csr_types.hpp``'s ``BoundedRMSpropSynapsePolicy`` (the production
  default) -- ``update_ci``/``update_cw``, ``beta2=0.999``, ``eps=1e-8``,
  ``max_abs_delta=2.0`` (raw-space, matches ``cpu_backend.cpp``'s real
  default), ``max_ci=100.0`` (``cpu_backend.cpp``'s chosen production
  default, not the function's own ``1e30`` no-op default),
  ``min_decay_frac=0.0`` (production default, a true no-op per its own
  docstring).
- ``g`` (per-synapse backward-sensitivity) = ``dL/d(true_weight)``, read
  directly from ``true_weight.grad`` after ``backward()`` -- exactly matches
  the C++ formula's ``g=dy*x`` (standard linear-layer weight gradient,
  aggregated/summed over the batch/token dimension already by construction
  of ordinary matmul backprop).
- ``contrib`` (per-synapse forward-contribution) = ``true_weight *
  x.sum(dim=0)`` (summed over the batch/token dimension, matching the C++
  formula's own row-aggregation).
- Row-level ``value_scale`` / column-level ``output_scale``:
  ``delta_csr_types.hpp``'s ``RMSpropScalePolicy`` (the production default)
  -- Adam-style bias-corrected RMSprop, ``g_agg``/``contrib_agg`` = the SAME
  per-synapse ``g``/``contrib`` further summed across the row (for
  ``value_scale``) or column (for ``output_scale``). ``scale_rank=1`` (the
  default this project actually uses -- no rank>1 machinery here).
- ``true_weight = w_stored * value_scale[row] * output_scale[col]``
  (``scale_rank=1``'s exact combination rule).
- ``lr_per_row_nnz`` degree normalization (``DISLDOLayer.forward``'s own
  default, ``True``): ``effective_lr = learning_rate / row_degree``. For a
  FULLY DENSE row (this reference always uses dense connectivity -- see
  below), ``row_degree = out_features`` for every row, so this is a flat
  ``1/out_features`` rescale, still reproduced exactly since it's part of
  the real formula ``ToyTileRecurrenceRMT``'s own calls actually hit (no
  ``lr_per_row_nnz=False`` override anywhere there).
- Hard clip (state/attn-output bounding): sili applies this via direct
  ``.data`` mutation, BYPASSING autograd entirely -- the gradient flows
  through AS IF the clip never happened (identity backward), NOT the
  zero-outside-bounds gradient plain ``torch.clamp`` gives by default.
  Reproduced here via an explicit straight-through clip
  (``x + (clamp(x)-x).detach()``, see
  ``toy_tile_recurrence_rmt_torch.straight_through_clip_design``), not bare
  ``torch.clamp``.

.. _toy_tile_recurrence_rmt_torch.dense_only_connectivity:

Connectivity: dense only, deliberately not the sparse echo-network path
------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.dense_only_connectivity``

sili's own ``DISLDOLayer``/``TrueMultiDigitLayer`` support sparse
(echo-network) OR dense (block4-loaded) connectivity; this torch port only
implements the DENSE case (a plain ``[in,out]`` matrix, no
synaptogenesis/pruning) -- matching what the fp4 arm of
``train_mqar_rmt_reference.py`` actually used (``dense=True``), and
deliberately NOT reproducing the fp32 arm's ``dense=False`` (sparse
echo-network) path, which task #232 flagged as a real, separate confound in
that earlier comparison, not something to carry forward here. This torch
reference is a dense-vs-dense comparison against the sili fp4 arm
specifically.

.. _toy_tile_recurrence_rmt_torch.not_reproduced_performance_details:

What's explicitly NOT reproduced: pure performance/memory-layout details
------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.not_reproduced_performance_details``

Explicitly NOT reproduced (pure C++ performance/memory-layout details with
no effect on the mathematical result for a fixed, dense, non-growing
topology): deferred-write batching, block4 tile storage, ``scale_rank>1``,
FP4/FP8 quantization codecs themselves (this port is fp32-only, matching
``DISLDOLayer32``'s own math exactly minus the quantize/dequantize
round-trip a real 4-bit/8-bit storage would add).

.. _toy_tile_recurrence_rmt_torch.straight_through_clip_design:

``straight_through_clip``: identity backward, not ``torch.clamp``'s zero-outside-bounds gradient
--------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.straight_through_clip_design``

Forward = ``clamp(x, lo, hi)``; backward = identity (gradient flows through
unchanged) -- matches sili's own ``.data``-mutation clip, which bypasses
autograd entirely rather than zeroing gradient outside the bounds the way
bare ``torch.clamp``'s default backward would.

.. code-block:: python

   def straight_through_clip(x, lo, hi):
       return x + (torch.clamp(x, lo, hi) - x).detach()

.. _toy_tile_recurrence_rmt_torch.disldo_torch_linear_autograd_leaf:

``DISLDOTorchLinear``: not an ``nn.Module``, ``true_weight`` is a fresh leaf every call
------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.disldo_torch_linear_autograd_leaf``

NOT an ``nn.Module`` (deliberately) -- no ``torch.optim`` ever touches
``w_stored``/``value_scale``/``output_scale``, only the hand-rolled update in
``apply_pending_updates`` does. ``true_weight`` (the actual matmul operand)
is a fresh leaf tensor with ``requires_grad=True`` created on every
``forward()`` call, so it participates correctly in the SAME big autograd
graph as everything else in one ``step()`` -- after the caller's single
``backward()`` call, ``true_weight.grad`` gives exactly the ``g`` signal
``disldo_backward``'s own formula needs, with zero hand-derived
backward-chain risk.

``apply_pending_updates`` is called AFTER the whole step's single
``loss.backward()``. Multiple ``forward()`` calls to the SAME layer within
one step (e.g. the L1 split's own second undamped call) each get their own
independent update, applied in call order -- matching sili's own per-call
inline-training semantics exactly (each ``layer.forward(x, lr)`` call is its
own training step there too).

.. _toy_tile_recurrence_rmt_torch.ablation_flag_defaults:

Constructor ablation flags: default to production behavior, exist for cheap single-factor sweeps
--------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.ablation_flag_defaults``

The five ``bias_correct_ci``/``use_momentum``/``momentum_beta1``/
``include_contrib_in_ci``/``clip_raw_delta`` args default to exactly the
production C++ ``BoundedRMSpropSynapsePolicy``'s own behavior (no bias
correction, no momentum, ``contrib^2`` included, raw delta hard-clipped) --
they exist to let single-factor optimizer-internals ablation run cheaply in
torch before touching ``delta_csr_types.hpp``, not to change the default.
Note the asymmetry this exposes: ``RMSpropScalePolicy``'s own
``_scale_update`` DOES bias-correct its EMA (see
``toy_tile_recurrence_rmt_torch.scale_update_log_space_step``) while this
per-synapse update never has -- ``bias_correct_ci=True`` makes the two
consistent.

.. _toy_tile_recurrence_rmt_torch.include_scale_chain_rule_bug:

``include_scale_chain_rule``: a real bug (missing S factor), kept default off for reproducibility
---------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.include_scale_chain_rule_bug``

Default False -- REPRODUCES A REAL BUG. ``true_weight = w_stored *
value_scale * output_scale``, so the correct chain-rule gradient is
``dL/d(w_stored) = g*S`` (``S = value_scale ⊗ output_scale``, matching
sili's real C++ non-deferred/quant formula, ``update_cw``'s own ``-g*S``
term, ``delta_csr_types.hpp``) and ``dL/d(value_scale[r]) =
sum_c(g[r,c]*w_stored[r,c]*output_scale[c])`` -- this class has always used
raw ``g`` directly for both instead (missing the ``S``/``w_stored*output_scale``
factor entirely), a genuinely different, simpler optimization dynamic than
what sili actually runs. Kept default False (i.e. reproduce the historical
bug) so every existing ablation result stays reproducible; True applies the
real chain rule, to test directly whether THIS specific difference (not the
clip, not batch-consistency -- both already investigated) is what makes
sili's real engine underperform this same architecture in torch.

When on, ``g_for_w = g*S`` feeds the weight update, while ``g_row_signal``/
``g_col_signal`` (and their ``contrib`` counterparts) get the matching
``w_stored*output_scale``/``w_stored*value_scale`` factors baked in for the
``value_scale``/``output_scale`` updates respectively; when off, all five
signals collapse back to raw ``g``/``contrib``, the historical behavior.

.. _toy_tile_recurrence_rmt_torch.use_magnitude_scale_rank1_limit:

``use_magnitude_scale``: a rank-1 structural limit found via direct measurement, and a gradient-free fix
--------------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.use_magnitude_scale_rank1_limit``

Default False -- direct measurement found ``value_scale``/``output_scale``'s
OWN RMSprop gradient signal is ~65-88% cancelled by cross-row/cross-column
sign disagreement among the per-synapse contributions that get summed into
it (worst in the 128-wide q/k/v/o_proj layers), so it barely moves from its
1.0 init even in a fully-converged run -- a genuine rank-1 structural limit
(project task #196), not a weak-signal problem. That leaves ``w_stored``'s
own raw magnitude to do ALL the work of representing small weights, which is
fine in fp32 but pushes the STORED CODE into a quantizer's coarse/subnormal
region once the codec is FP4/FP8.

This is a gradient-FREE fix instead of trying to un-cancel that signal: a
pure reparametrization that keeps ``true_weight =
w_stored*value_scale*output_scale`` algebraically IDENTICAL while moving
magnitude from ``output_scale`` into ``w_stored`` (or back) each step, so
``w_stored``'s own per-column RMS tracks toward ``magnitude_scale_target`` --
chosen once, not learned, so it can't be cancelled. No effect at all in
fp32 (same ``true_weight`` either way); the whole point is giving the STORED
representation a favorable magnitude before quantization ever happens. See
``toy_tile_recurrence_rmt_torch.magnitude_rescale_mechanism`` for the actual
rescale implementation.

.. _toy_tile_recurrence_rmt_torch.clip_pre_ci_ci_jump_bug:

``clip_pre_ci``: a real ci single-step jump to the ceiling, and the loss collapse it caused
------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.clip_pre_ci_ci_jump_bug``

Default False -- targets a DIFFERENT failure mode than ``clip_raw_delta``
(which clips the post-division update, AFTER ``ci`` already absorbed
whatever ``g``/``contrib`` produced it). Multi-seed ``scale_chain_rule``
sweeps found ``ci`` sometimes takes one single-step jump of +70-80 straight
to ``max_ci``'s ceiling (e.g. seed=1000: ~25.7->99.9 in ONE step), coincident
with the query loss permanently collapsing to ``ln(vocab)`` (uniform
guessing) a few hundred steps later -- ``clip_raw_delta`` doesn't prevent
this because the damage is in what THAT one step's raw ``g``/``contrib`` do
(to ``ci``'s EMA and, via ``g_for_w``, to ``w_stored`` itself), not in the
post-division magnitude. ``clip_pre_ci`` clips ``g`` and ``contrib`` to
``±pre_ci_clip_value`` (defaults to ``max_abs_delta``) BEFORE squaring into
``ci``'s EMA, capping how much any single anomalous step can move ``ci`` at
all.

.. _toy_tile_recurrence_rmt_torch.scale_invariant_chain_rule_quadratic_bug:

``scale_invariant_chain_rule``: the quadratic-in-S bug hidden until magnitude_scale exposed it
------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.scale_invariant_chain_rule_quadratic_bug``

Default False. Both branches of ``include_scale_chain_rule`` (on OR off)
share ``ci`` tracking RAW ``g^2`` -- unaffected by ``include_scale_chain_rule``
-- but when ``chain_rule=True``, ``numerator=g*S`` while ``ci`` is calibrated
to plain ``g^2``, so ``raw=numerator/sqrt(ci_hat)`` scales LINEARLY with
``S`` (not O(1)-normalized), making ``Δ(true_weight)=S*Δ(w_stored)`` scale
QUADRATICALLY with ``S`` -- i.e. the effective learning rate for
``true_weight`` silently collapses as ``S`` shrinks (or explodes as ``S``
grows). Never surfaced before because every prior config left ``S`` near its
1.0 init (``value_scale``/``output_scale`` barely move -- see
``toy_tile_recurrence_rmt_torch.use_magnitude_scale_rank1_limit``'s
cross-column-cancellation finding) -- ``magnitude_scale`` is the first
mechanism that deliberately drives ``S`` away from 1, exposing it.

Fix: compute ``raw`` from the RAW gradient ``g`` (properly self-normalized
by ``ci``, which already tracks ``g^2``), giving a step that's a fixed size
in TRUE_WEIGHT units regardless of parametrization -- then convert to
``w_stored``'s own units by dividing by ``S`` (the correct inverse chain
rule), so ``Δ(true_weight) = S * (eff_lr*raw/S) = eff_lr*raw``, independent
of ``S`` entirely. Per direct instruction: this (not a one-off rescale
patch) is the fix -- remove the "S stays near 1" assumption, don't just
compensate for it after the fact.

.. code-block:: python

   # numerator_for_ci_norm: raw g when the fix is active (self-normalized
   # by ci, which tracks g^2), vs g_for_w=g*S otherwise (the bug).
   numerator_for_ci_norm = (
       -g if (scale_invariant_chain_rule and include_scale_chain_rule) else -g_for_w
   )
   ...
   if scale_invariant_chain_rule and include_scale_chain_rule:
       delta_w = eff_lr * raw / S   # divide out S -- the correct inverse chain rule
   else:
       delta_w = eff_lr * raw

.. _toy_tile_recurrence_rmt_torch.immediate_requantize_no_float_shadow:

Immediate re-quantization after every update: no hidden float shadow
------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.immediate_requantize_no_float_shadow``

When ``fake_quantize_kind`` is set, ``w_stored`` is quantized IMMEDIATELY
after the RMSprop step, not once at the very end of the step -- the real
production system keeps NO float shadow of any weight (see
``TrueMultiDigitLayer``'s own docstring: "a real hardware implementation has
no room for a hidden full-precision shadow"); every update writes directly
to the quantized representation. Quantizing only once per step (as this code
used to) let ``w_stored`` silently accumulate float precision between the
RMSprop step and the magnitude-rescale step, which isn't how the real system
behaves. The same reasoning applies again inside ``_magnitude_rescale``
(see ``toy_tile_recurrence_rmt_torch.magnitude_rescale_mechanism``): the
rescaled ``w_stored`` is re-quantized immediately too, since that's what
actually gets "stored."

.. _toy_tile_recurrence_rmt_torch.magnitude_rescale_mechanism:

``_magnitude_rescale``: gradient-free reparametrization, ci rescaling, and quantization-noise feedback
--------------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.magnitude_rescale_mechanism``

Gradient-free reparametrization: ``true_weight = w_stored * value_scale *
output_scale`` is UNCHANGED by this (algebraically), only WHERE the
magnitude lives changes -- moves each column's share of magnitude between
``w_stored`` and ``output_scale`` so ``w_stored``'s own per-column RMS
drifts toward ``magnitude_scale_target`` (a fixed constant, so it can't be
cancelled the way a gradient-summed target could be). Applies a DAMPED
(``magnitude_correction_rate``) step toward the full correction each call
rather than jumping all the way there, so it doesn't fight the RMSprop
updates that just happened in the same step.

``ci`` (per-synapse RMSprop denominator) accumulates ``g^2+contrib^2`` from
the RAW chain-rule gradient ``g``, which is independent of how magnitude is
split between ``w_stored``/``output_scale`` -- but the actual update
numerator is ``g*S`` (``S=value_scale*output_scale``). Shrinking
``output_scale`` by ``k`` shrinks ``S`` by ``k``, shrinking the numerator,
while ``ci`` (calibrated to ``g^2``, not ``(g*S)^2``) doesn't track that --
silently damping ``w_stored``'s own effective step size every time this
fires. Found empirically: without this, fp32 accuracy dropped 1.0->0.18
despite ``true_weight`` being algebraically unchanged. Rescaling ``ci`` by
``k^2`` alongside keeps it calibrated to the new ``S`` regime -- ONLY needed
when ``scale_invariant_chain_rule`` is off: that flag already decouples
``ci`` from ``S`` entirely (``ci`` tracks raw ``g^2`` and the ``S``-dependence
is removed via explicit division at apply time instead, see
``toy_tile_recurrence_rmt_torch.scale_invariant_chain_rule_quadratic_bug``),
so rescaling ``ci`` here too would double-correct and be wrong.

``magnitude_rescale_ema_beta``: default None (use the instantaneous
``col_rms`` each call). With fake_quantize active, ``w_stored`` has ALREADY
been quantized to a coarse grid by the time this runs (immediately above, in
``apply_pending_updates``) -- measuring ``col_rms`` from that
coarsely-rounded ``w_stored`` means the rescale target itself carries
quantization noise, and since this fires EVERY step (not periodically), that
noise feeds straight back into the next quantize call, compounding. Found
empirically: fp32 (no quantization noise in the measurement) reaches a clean
1.0 with this mechanism, but fake-fp8 collapses to 0.0 despite the SAME
optimizer math. Set ``magnitude_rescale_ema_beta`` (e.g. 0.9) to track an EMA
of ``col_rms`` across steps instead, filtering out per-step quantization
jitter from the signal driving ``k``.

.. _toy_tile_recurrence_rmt_torch.magnitude_scale_both_axes:

``magnitude_scale_both_axes``: symmetric row-axis treatment, and what it tests for rank-2
------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.magnitude_scale_both_axes``

SAME treatment on the ROW axis (``value_scale``) -- per direct question:
this whole mechanism only ever touched ``output_scale`` (columns);
``value_scale`` (rows) was left to its own cancellation-limited RMSprop
signal and barely moves (see
``toy_tile_recurrence_rmt_torch.use_magnitude_scale_rank1_limit``'s
cross-row-cancellation finding). Nothing about the mechanism is
column-specific; applying it symmetrically gives the full rank-1
(``value_scale ⊗ output_scale``) envelope the same magnitude-matching
ability on both axes, not just one -- and is a useful data point for whether
rank-2 (task #196) is worth the extra cost: if a second SHARED axis doesn't
help even when it's allowed to move freely, more basis vectors on the same
axis structure are unlikely to help more.

.. _toy_tile_recurrence_rmt_torch.scale_update_log_space_step:

``_scale_update``: multiplicative log-space step under ``scale_invariant_chain_rule``
------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt_torch.scale_update_log_space_step``

Multiplicative (log-space) step instead of additive: an ADDITIVE step of
size ``~eff_lr`` is a huge RELATIVE change once ``scale`` has shrunk far
below 1 (which ``magnitude_scale`` deliberately does) and negligible once
it's grown large -- same "assumes scale stays near 1" bug as ``w_stored``'s
own update (see
``toy_tile_recurrence_rmt_torch.scale_invariant_chain_rule_quadratic_bug``),
just on ``scale`` itself. ``d(loss)/d(log(scale)) = d(loss)/d(scale)*scale =
g_agg*scale`` (chain rule through ``scale=exp(log_scale)``);
RMSprop-normalizing THAT keeps the step a fixed RELATIVE (percentage) size
regardless of ``scale``'s own magnitude. Bonus: ``scale`` can never cross
zero this way (``exp()>0``), unlike the additive step.
