``toy_tile_recurrence_rmt.py`` research notes
====================================================

Companion doc to ``model/toy_tile_recurrence_rmt.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below); this doc
links back to source by class/method name. See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets render invisible on GitHub, frozen code snippets on
real-bug/non-obvious-derivation sections only).

.. _toy_tile_recurrence_rmt.module_overview:

``ToyTileRecurrenceRMT``: a faithful RMT control, not a novel architecture
---------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.module_overview``

Faithful reference implementation of Recurrent Memory Transformer's actual
mechanism (Bulatov et al., "Recurrent Memory Transformer", NeurIPS 2022,
arXiv:2207.06881) -- built from the SAME sili__new primitives (disldo_cls
layers, Tensor autograd, ``gaussian_attention``, RMSNorm) as
``ToyTileRecurrenceRealFP4``, NOT plain torch. This IS the control for the
ongoing MQAR investigation: same engine, same precision handling (FP4 via
``disldo_cls``, or fp32 via a plain-float ``disldo_cls``), so a pass/fail
result on the exact same K=1 MQAR task/harness isolates the ARCHITECTURE
question specifically -- if this also fails to learn, the problem is the
task/embedding/harness, not ``ToyTileRecurrenceRealFP4``'s own recurrence
wiring. Plain torch is a deliberately DEFERRED fallback, only worth building
if THIS itself fails, to separate "the engine is broken" from "even a proven
architecture fails here somehow" -- building torch first would confound
engine differences with architecture differences and defeat the point of a
control (direct instruction).

Core mechanism, matching RMT exactly: ``num_memory_slots`` dedicated
memory-token positions are CONCATENATED into the same attention window as
the real content tile positions (never additively merged), so
``gaussian_attention`` processes memory and content together via ordinary
self-attention, exactly like RMT's own memory tokens being literally
re-inserted into the input/output sequence. The memory tokens' own output
positions after this step become next step's memory, carried purely by
being re-inserted into the window on the next call -- no separate
gate/combine mechanism at all (RMT itself has none; the memory update is
handled entirely by ordinary attention + residual, same as any other token
position).

.. _toy_tile_recurrence_rmt.critic_head_design:

``use_critic``: a real advantage-actor-critic value head
------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.critic_head_design``

Adds a ``critic_head`` (same shape as ``lm_head``, ``embed_width ->
vocab_size``) predicting the per-vocab-neuron squared error the actor's own
logits will incur -- a real advantage-actor-critic value head, not a
shortcut for the (exactly known) true loss itself. Default off, so every
existing caller is byte-identical (same first 6 layer-construction RNG
draws either way -- the critic's own seed is drawn separately, from ``rng``,
AFTER that fixed size-6 draw completes). Exposed via
``model.last_critic_pred`` (an attribute, not a 4th return value) since
``step()``'s 3-tuple return is unpacked by dozens of existing call sites
across both repos and a return-arity change would break every one of them.

.. _toy_tile_recurrence_rmt.magnitude_clip_and_min_sigma_design:

Two numerical-safety mechanisms, both from real overflow diagnostics
--------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.magnitude_clip_and_min_sigma_design``

**magnitude_clip_penalty_coef** (task #303/#304): a plain hard clip on
q/k/v/attn/combined_new gives ZERO backward gradient past the boundary
(``np.clip``'s own derivative is 0 there), so a layer whose output keeps
getting clipped never learns to stop producing that magnitude in the first
place. Confirmed via real diagnostics: ``v_proj``'s output ran unclipped
into the 1000s-2000s for hundreds of steps (masked downstream by the
existing hard clips on attn/combined_new) before an unscaled dot product
inside ``gaussian_attention`` finally overflowed to NaN. Adds a
differentiable hinge-squared penalty
(``coef*mean(relu(|x|-clip_range)**2)``) on q/k/v/attn/combined_new, so the
layers themselves get gradient pressure to shrink whenever they exceed
``clip_range``, on top of (not instead of) the existing hard clip on the
VALUES. Default off (0.0), matching ``l1_sparsity_coef``'s own opt-in
convention.

**min_sigma** (task #305): ``gaussian_attention``'s Gaussian bias term is
``1/(2*sigma**2)`` -- as sigma trains toward 0 (exactly what learning to
attend sharply to one position looks like), that term can hit Inf, and
``0*Inf=NaN`` if a key lands exactly on the query's center. Always-on floor
(matching ``rms_eps``'s own always-on convention, not ``l1_sparsity_coef``'s
opt-in-off one), applied to ``sigmas.data`` right after
``exp(log_sigmas)``. 1e-3 is deliberately generous: at that floor, one
integer position away from center already gives
``exp(-1/(2*1e-3**2)) = exp(-500000)`` -- functionally a one-hot -- so this
can't meaningfully constrain how sharply the model can attend, only prevent
the literal division-by-near-zero case.

.. _toy_tile_recurrence_rmt.recurrent_only_output_ablation:

``recurrent_only_output``: the RNN-validation ablation, and why the residual stays live
------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.recurrent_only_output_ablation``

Direct instruction, RNN recurrence validation ablation: when True,
content-row queries (this step's readout) are masked to attend ONLY
memory-row keys/values -- never their own or another content row's.
Memory-row queries stay unrestricted (read the full window, i.e.
in_proj->recurrent "write" stays allowed). Net effect: the
ATTENTION-derived portion of the output at step t can only carry
information that was already written into memory at some step <t.

Implemented as VALUE-masking (``_mem_only_value_mask``: 1.0 at memory
physical rows, 0.0 at content physical rows, applied elementwise and
differentiably to v_phys), not by slicing K/V down to memory-only rows --
slicing would collapse their array index back to a local ``0..n_mem-1``
range, silently reintroducing the exact clustered-position underflow fixed
in ``toy_tile_recurrence_rmt.interleaved_position_layout_bug`` below. Keys
stay the full, correctly-interleaved array (content's own key still
competes in the softmax); only content's VALUE contribution is zeroed.

Does NOT zero the direct ``x_wide`` residual skip into ``content_out``
(task #315 follow-up, direct instruction, post-validation): an earlier
version of this ablation ALSO zeroed that residual, to fully isolate
memory's contribution for a distance-sweep verification (confirmed genuine
cross-detach recurrent persistence -- accuracy well above chance at
distances requiring the value to survive ``step()`` boundaries no same-step
gradient path can reach across). That isolation already did its job; the
residual doesn't need to stay zeroed going forward. ``x_wide`` is the QUERY
token's own embedding, which never correlates with MQAR's correct recall
value (the task is information-theoretically undecidable from the query
token alone), so leaving the residual live can't reintroduce a "cheat" path
for this task -- it just avoids needlessly handicapping the model.
``recurrent_only_output`` now ONLY blocks content-content attention (the
actual "recurrent only" property); it no longer touches this residual.

Default off -- existing callers see zero behavior change (same 6+1
layer-seed draws either way; this doesn't touch construction at all, only
``step()``/``step_cached()``).

.. _toy_tile_recurrence_rmt.sparsity_phase6_design:

Sparsity plan Phase 6: ``input_sparsity_p``/``dy_sparsity_p``/``wide_max_weights``/``output_dy_sparsity_p``
------------------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.sparsity_phase6_design``

Task #335. Real values, not a bool toggle -- presence (non-None) IS the
toggle, matching this whole plan's own convention (see
``sili.sparse_rnn.DISLDOLayer.forward``'s identical ``dy_sparsity_p``). All
default to None, meaning every existing caller gets byte-identical behavior
(no CSR anywhere, ``max_weights`` unchanged for every layer).

Only ``input_proj``/``q_proj``/``k_proj``/``v_proj``/``o_proj`` are
affected -- ``lm_head``/``critic_head`` stay fully dense/unwidened the whole
time (their own budget stays ``max_weights``, never ``wide_max_weights``;
their ``forward()`` calls never see a CSR input). Widening those two isn't
part of this plan: they read ``pooled`` (already column-averaged down from
``state_width`` back to ``embed_width``), not one of the 5 layers whose
INPUT width doubles with ``embed_width``.

``input_sparsity_p``: density fraction for the 5 affected layers' forward
INPUT (reuses ``CSR.from_dense``'s own ``p`` convention). A layer whose
input width just doubled needs ``p~=0.5`` to keep total compute at ~2x
instead of 4x -- the whole point of pairing width-doubling with input
sparsification in this plan.

``dy_sparsity_p``: density fraction for those same 5 layers' backward
GRADIENT -- a genuinely separate axis from ``input_sparsity_p`` (forward-
input-sparsity and backward-gradient-sparsity are independent parameters on
the underlying ``sisldo_forward``/``disldo_backward_sparse_grad`` C++
functions). If left None while ``input_sparsity_p`` is set, defaults
internally to ``input_sparsity_p``; an explicit value overrides
independently.

``wide_max_weights``: per-layer synapse budget override for the 5 affected
layers only. None (default) means they share the same ``max_weights`` as
every other layer, today's exact behavior. Set to an int (e.g. 2048) to
give them a larger budget while ``lm_head``/``critic_head`` stay at the
original ``max_weights`` -- input+backprop sparsity means compute no longer
scales with the full stored budget every step, so the extra memory is
affordable.

``output_dy_sparsity_p`` (following the ``step_cached`` graded-schedule
speed work): density fraction for ``lm_head``/``critic_head``'s own
backward GRADIENT only -- genuinely separate axis, and NOT threaded through
``_to_sparse`` (``lm_head``/``critic_head`` read ``pooled``, a real dense
column-averaged readout with no structural sparsity -- unlike the 5
affected layers' inputs, there's no free-lunch argument for sparsifying
THIS input). The gradient side is different:
``g_logits[row] = (1+advantage) * (probs - onehot)`` where ``probs`` is a
softmax over ``vocab_size`` -- technically dense (softmax never hits exact
0) but concentrates hard onto a few classes as the model gets confident, so
most of ``probs-onehot`` is near-zero in practice. ``dy_sparsity_p``'s
top-k-by-magnitude selection is exactly suited to this shape. None
(default): byte-identical to today's exact dense backward for both heads.

.. _toy_tile_recurrence_rmt.dy_r_target_nucleus_design:

``dy_r_target``/``dy_k_min``/``dy_k_max``: nucleus grad sparsification, per-layer since task #372
-----------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.dy_r_target_nucleus_design``

Task #367, priority 1 (direct instruction -- "grad is the one that's
definitely required"). Nucleus/energy-threshold grad sparsification for the
SAME 5 layers ``dy_sparsity_p`` covers, TAKES PRIORITY over
``dy_sparsity_p`` when both are set (mirrors ``DISLDOLayer.forward``'s own
priority chain exactly). Unlike ``dy_sparsity_p``'s fixed fraction, k is a
CONSEQUENCE of ``dy_r_target`` and each step's actual gradient energy --
mutable after construction via ``apply_amortized_dy_r_target_control``, a
closed-loop controller against MEASURED steps/sec, not a guessed constant
(see ``toy_tile_recurrence_rmt.amortized_r_target_control_design`` below
for why the originally-sketched analytic formula didn't hold). ``dy_k_min``/
``dy_k_max`` are the hardware density floor/ceiling (see
``_nucleus_top_k_csr``'s own docstring, sili__new). None (default):
byte-identical to today's exact behavior, same as every other sparsity
kwarg here.

Task #372: this constructor arg is the INITIAL value applied uniformly to
all 5 wide layers -- internally stored as ``self.dy_r_target``, a per-layer
dict (name -> r_target), same pattern as ``self._l2_decay_factor``. A
model-level knob wastes each layer's own economics (different fwd:bwd cost
ratios per layer, task #368's own measurement), so per-layer state is the
natural representation even though this constructor arg still only offers
one shared starting point; task #374's per-layer surprise loop (see
``toy_tile_recurrence_rmt.dy_surprise_design``) makes each entry diverge
independently over training.

``_wide_extra_kwargs`` reads this per-layer dict live (not cached at
construction) since it's mutable post-construction; priority chain:
``dy_r_target`` (via ``_effective_dy_r_target``) > ``dy_sparsity_p`` >
``{}``, mirroring ``DISLDOLayer.forward``'s own chain exactly.

.. _toy_tile_recurrence_rmt.dy_surprise_design:

Task #374: per-layer surprise-modulated ``dy_r_target``, and why the one-step lag needs no bookkeeping
------------------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.dy_surprise_design``

See JOURNAL.md's "Grad-side k_t design, revised" + "Multi-actuator design
discussion" entries for the full derivation. Per-layer INNER loop, breathing
each layer's own EFFECTIVE r_target above/below its r_bar
(``self.dy_r_target[name]``, the OUTER sps-controlled anchor from
``apply_amortized_dy_r_target_control``) based on that layer's own recent
gradient energy:

.. code-block:: python

   E_t   = ||dy_t||^2                          # free byproduct of backward
   Lbar <- beta*Lbar + (1-beta)*E_t             # per-layer running EMA
   r_t   = clip(r_bar * (E_t/Lbar)^alpha, 0.05, 0.99)

A layer whose gradient just carried more energy than its own recent average
(surprising -- more to learn from this step) gets a HIGHER effective
capture ratio; a layer coasting near its own average gets pulled back down,
independent of every other layer's own r_bar/surprise state.

``E_t``/``Lbar`` are naturally LAGGED one step already, with no explicit
bookkeeping needed: a layer's ``_wide_extra_kwargs`` call happens at
``forward()``-time, strictly BEFORE that step's own backward (and therefore
that step's own ``dy``) exists -- so it always reads whatever ``E_t``/
``Lbar`` the PREVIOUS backward pass left behind (captured in
``_timed_layer_forward``'s timing wrapper, a free byproduct of the same
mechanism that measures per-layer timing -- see
``toy_tile_recurrence_rmt.layer_timing_design``), which is exactly the
one-step lag the design calls for (sidesteps the forward-before-dy-exists
chicken-and-egg without an autograd hook or C++ instrumentation).

``_update_layer_surprise`` initializes ``Lbar`` to ``E_t`` itself on a
layer's first-ever observation (not 0) -- starting ``Lbar`` at 0 would make
the very first ``E_t/Lbar`` ratio divide-by-zero; starting it AT ``E_t``
instead makes the first ratio exactly 1.0 (no modulation), a neutral cold
start that only diverges once real variation is observed. Called once per
REAL backward invocation (multiple times per ``step()`` for k/v/o_proj
under the write-then-read design -- each is a genuine, independent
observation of that layer's gradient energy at that moment, not an
artifact to dedupe).

``dy_surprise_alpha`` None (default): mechanism fully OFF, effective
r_target is exactly r_bar with zero modulation -- byte-identical to every
existing ``dy_r_target`` caller/test. Set (e.g. 0.5, an unvalidated
starting point) to turn it on. ``dy_surprise_beta`` (0.99, matching the
design note's own "beta ~ 0.99") only matters once alpha is set. r_min/r_max
for this INNER per-step clip are hardcoded at 0.05/0.99, matching
``apply_amortized_dy_r_target_control``'s own OUTER-loop defaults -- not
exposed as further constructor params, to avoid over-parameterizing an
unvalidated mechanism (direct-instruction pattern: don't build unproven
complexity).

.. _toy_tile_recurrence_rmt.x_r_target_design:

``x_r_target``/``x_k_min``/``x_k_max``: nucleus selection on the INPUT side
------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.x_r_target_design``

Task #365 -- see JOURNAL.md's "Multi-actuator design discussion" entry,
point 3. Nucleus/energy-threshold selection for the SAME 5 layers' forward
INPUT that ``input_sparsity_p`` currently covers with a fixed fraction --
TAKES PRIORITY over ``input_sparsity_p`` when both are set, mirroring
``dy_r_target``'s own priority chain over ``dy_sparsity_p`` exactly (task
#367). Per-layer dict (mirrors ``dy_r_target``'s own #372 pattern), NOT a
single model-level scalar -- same rationale: different layers/consumers
have different real economics.

Direct instruction, revised scope: input stays on the SIMPLE closed-loop
control ``dy_r_target``'s own OUTER loop already uses
(``apply_amortized_x_r_target_control``, structurally identical to
``apply_amortized_dy_r_target_control`` -- see
``toy_tile_recurrence_rmt.amortized_r_target_control_design``) -- measured
speed vs a target, adjusting r_bar up/down. ``r_target`` IS already "the
direct reproduction-percent target" in this design:
``R(v,k)=sum(v_topk^2)/sum(v^2)`` literally means "% of this row's own
squared magnitude reproduced by the kept entries," so no separate
reconstruction-error signal is needed for ``x_r_target`` to mean something
concrete. Unlike dy's surprise signal, there is no equivalent DERIVED
complexity signal for input in this non-autoencoder solver/curriculum task
(task #376, filed away as a future idea, not built: an action/prediction
output split would give input a genuine reconstruction-error signal to
drive a smarter allocation than measured-speed alone, but that's unproven
and explicitly deferred until after this epic).

None (default): byte-identical to today's exact ``input_sparsity_p``
behavior, same convention as every other opt-in kwarg here.

.. _toy_tile_recurrence_rmt.layer_timing_design:

``_timed_layer_forward``: real per-layer timing via a deferred-backward wrapper
--------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.layer_timing_design``

Task #373: real per-layer forward+backward timing, direct
``time.perf_counter()`` wraps, NOT a width-based cost proxy -- direct
correction: layers cost single-digit-to-tens of ms, timer overhead is
~100ns, real measurement is cheaper than a calibrated proxy would have
been. Feeds the cross-layer budget allocator (task #375, see
``toy_tile_recurrence_rmt.cross_layer_budget_allocator_design``) -- each
layer's own measured ``fwd_s``/``bwd_s`` is the real economics a shared
model-level knob (pre-#372) wasted.

Forward timing is straightforward (wraps the call itself). Backward is
deferred -- sili__new's autograd attaches a Python closure to
``out._backward`` that only actually runs later, inside the caller's own
``loss.backward()`` graph walk, outside ``step()``'s own call stack
entirely. Wrapping that closure here (instead of modifying sili__new's
engine) keeps this purely a sili_peridot-side concern: ``out._backward`` is
REPLACED with a thin timer wrapper around the original closure, so
whichever later ``loss.backward()`` call eventually triggers it, this
layer's own share of that walk gets attributed correctly. Accumulates into
``self._layer_timing[layer_name]`` (fwd_s/bwd_s/fwd_calls/bwd_calls) --
``reset_layer_timing()`` zeroes it before a measurement window (mirrors how
``steps_per_sec`` is computed over a window in ``train_mqar_curriculum.py``,
not since t=0).

Also the task #374 surprise-signal capture point: ``out.grad`` at the
moment ``out._backward`` actually runs IS ``dy`` for this layer (the real
gradient flowing into this forward call's output) -- a free byproduct of
the same wrapper, no separate instrumentation needed. Feeds
``self._update_layer_surprise`` (no-op unless ``dy_surprise_alpha`` is set
-- see ``toy_tile_recurrence_rmt.dy_surprise_design``).

.. _toy_tile_recurrence_rmt.interleaved_position_layout_bug:

Real bug: clustered memory positions + sigma=1 cold start caused a hard, gradient-dead underflow
------------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.interleaved_position_layout_bug``

Found via real diagnostics run during the RNN-validation ablation. The OLD
layout put all ``n_mem`` memory rows at physical positions ``[0, n_mem)``
and all content rows at ``[n_mem, total_slots)``, i.e. two separate
contiguous blocks -- so memory sat maximally far (~total_slots) from the
live input/output position (always the LAST content row,
``_build_tile_window``'s own convention). Combined with the OLD sigma=1
cold-start default, this is a real, confirmed bug:

.. code-block:: python

   # Gaussian bias at distance ~16.5 (memory clustered far from the
   # live output position), with the OLD sigma=1 cold start:
   exp(-diff**2 / (2*sigma**2))   # diff~16.5, sigma=1 -> ~1e-59

``1e-59`` is below float32's smallest representable value -- an EXACT zero
attention weight, and since ``d(score)/d(sigma)`` is itself scaled by that
same zero weight, ALSO exactly zero backward gradient. A genuine dead end
the model could never train its way out of, symmetric in both directions
(memory could never read fresh input, and the output position could never
read memory back) -- silently unexercised until now because every real
curriculum run so far stayed in the K<=4 in-context phase, solvable via
content-content attention alone (nearby array indices, no large-distance
underflow), never actually forcing reliance on memory.

Fix has two parts:

1. Spread the ``n_mem`` memory slots evenly across the position range
   instead of clustering them at one end, so no content tile is
   structurally privileged (nearest to memory) over any other.
2. Widen the cold-start sigma so distance alone can't underflow the
   attention weight to a hard, gradient-dead zero -- ``total_slots/4``
   keeps even the single farthest possible pair (distance ~= total_slots)
   at a representable, if weak, weight (``exp(-4**2/2)~=3e-4``), so real
   training signal can reach every position from the start and sharpen (or
   widen further) from there as the data actually wants.

Implementation: ``centers`` values (which position each LOGICAL row --
0..n_mem-1 memory, n_mem..total_slots-1 content -- is labeled as occupying)
are reassigned to the interleaved physical layout; the row DATA itself
stays in logical order (no need to move ``memory_normed``/``x_normed``'s
own concat order). What genuinely must move is K/V's ARRAY order when
they're fed into ``gaussian_attention``, since the C++ kernel has no
separate per-key position input -- it uses the key's raw array index j
directly as its position (``attention.hpp``'s ``gaussian_attention_forward``:
``diff = float(j) - c``). Q stays logical, K goes physical via
``_kv_phys_gather_idx`` (precomputed once at construction, reused every
``step()``/``step_cached()`` call). NOTE: this reordering assumes
``causal=False`` (matches this model's only usage) -- if causal attention
were ever added here, Q and K would need to share the SAME index space for
the ``j>t`` mask to mean anything, which this Q-stays-logical/K-goes-physical
split deliberately breaks.

.. _toy_tile_recurrence_rmt.amortized_r_target_control_design:

``apply_amortized_dy_r_target_control``/``apply_amortized_x_r_target_control``: closed-loop against measured speed, not an assumed cost ratio
--------------------------------------------------------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.amortized_r_target_control_design``

Task #368, revised design -- see JOURNAL.md's "Grad-side k_t design,
revised" entry for the full rationale. The original k_t sketch assumed an
analytic kbar derivable from a fixed backward:forward compute-cost ratio
(~10x, recalled). Direct measurement (``forward_dense`` vs
``backward_dense``, same layer, varying width) found the ratio is NOT
constant -- 2.8x at n=48, 6.8x at n=128, 12.0x at n=384 -- so any single
hardcoded ratio baked into a formula would be wrong at some scale. This
sidesteps that entirely: no assumed ratio anywhere, just react to what
steps/sec actually measures, same "closed-loop, measured-statistics, not
guessed constants" philosophy
``toy_tile_recurrence_rmt.amortized_l2_decay_design`` already uses
successfully.

Call periodically (e.g. every N steps) from the training loop with that
window's own measured steps/sec. ``measured_sps < target_sps``: r_target
shrinks (``down_factor``, capture less energy => fewer entries => cheaper).
``measured_sps > target_sps`` (no margin needed): r_target grows back
toward ``r_max`` (``up_factor``). Clipped to ``[r_min, r_max]`` --
``r_max<1.0`` by default since 1.0 would defeat the entire compute-savings
purpose ``dy_r_target``/``x_r_target`` exist for.

``layer_name`` (task #372): None (default) applies the SAME
``measured_sps``/``target_sps`` correction to every wide layer whose
r_target is currently set (mirrors the old model-level behavior exactly --
existing callers see the same net effect as before, since every layer
started at the same initial value and moves in lockstep). Pass a specific
name once real per-layer ``measured_sps`` is available (task #373's timing)
to adjust just that layer independently -- a no-op (KeyError) if that
layer's r_target entry was never enabled (None) to begin with, since this
controller only ADJUSTS an already-opted-in mechanism, it doesn't turn the
mechanism on for a fresh layer.

``apply_amortized_dy_r_target_control`` does NOT implement the per-step
E_t/Lbar energy-modulation half of the original design (r_t breathing
above/below r_bar based on THIS step's own gradient energy) -- that's task
#374 (see ``toy_tile_recurrence_rmt.dy_surprise_design``): ``dy_r_target``
is consumed at ``forward()``-CALL time, before that layer's own dy for this
step is known (chicken-and-egg; resolved there via a one-step lag, not
here). This method only implements the OUTER (r_bar-vs-measured-sps) loop.

``apply_amortized_x_r_target_control`` (task #365, INPUT side) is
structurally IDENTICAL to the dy_r_target version above (same formula,
same ``layer_name``/None convention, same clip defaults), just operating on
``x_r_target`` instead -- direct instruction: input stays on this SAME
simple control, no separate derived signal (see
``toy_tile_recurrence_rmt.x_r_target_design`` for why not).

.. _toy_tile_recurrence_rmt.cross_layer_budget_allocator_design:

``apply_cross_layer_budget_allocator``: why only the INPUT axis reacts to speed
--------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.cross_layer_budget_allocator_design``

Task #375: coordinates ``x_r_target`` (INPUT axis) against the remaining
compute budget using task #373's real per-layer timing as the "how
expensive is this layer actually" signal -- WITHOUT inventing an analytic
cost-vs-r_target formula (same "measured statistics, not guessed constants"
philosophy every other outer loop here already uses, since fwd:bwd cost
ratio varies 2.8x-12x across layers/widths).

``dy_r_target`` (GRAD axis) is DELIBERATELY NOT TOUCHED by this method at
all -- direct instruction: grad's r_target must be driven by its own need
(task #374's E_t/Lbar surprise signal), INDEPENDENT of the speed budget. The
problem this task exists to prevent is both axes independently reacting to
the SAME ``measured_sps`` and fighting/oscillating together -- the fix is
structural (only ONE axis reacts to speed in any one coordinated call), not
a smarter simultaneous-adjustment formula. A caller MAY still call
``apply_amortized_dy_r_target_control`` separately, rarely, to keep r_bar
from drifting arbitrarily over a very long run -- that's an orthogonal,
slower-timescale concern this method doesn't manage.

Per-layer WEIGHT: each layer's real measured (``fwd_s+bwd_s``) share of the
total across all 5 wide layers (from ``self._layer_timing``, task #373),
normalized against the uniform 1/5 baseline -- a layer currently eating an
above-average share of real time gets a correspondingly STRONGER
correction (it's the layer most responsible for the current speed), a
below-average layer gets a correspondingly WEAKER one. Weight is clipped to
``[0, 3]`` to keep the correction bounded and predictable -- an unclipped
weight could blow up arbitrarily for a layer that happens to have almost
zero competing cost this window. Falls back to weight=1.0 for every layer
when ``self._layer_timing`` has no data yet (cold start) -- caller should
call ``reset_layer_timing()`` at the start of each measurement window so
this weight reflects THAT window's real costs, not a stale accumulation.

.. _toy_tile_recurrence_rmt.amortized_l2_decay_design:

``apply_amortized_l2_decay``: real bug -- a fixed half-life badly overtuned a healthy layer
------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.amortized_l2_decay_design``

Applies the amortized decoupled L2 decay + rolling health-stats mechanism
to every real ``disldo_cls`` weight layer, INCLUDING the fp32 control
(unlike ``magnitude_rescale_output``, this is bound on all three backends --
ValueAccessor-generic, no ``hasattr`` guard needed). Meant to be called once
per real training step -- the "simple bound helps immediately, L2 helps
health" complement to ``NOCAPS_KWARGS``'s per-precision
``max_abs_delta``/``max_ci`` (``train_mqar_curriculum.py``).

CLOSED-LOOP, not a hand-picked half-life (direct instruction: "I thought we
could set the L2 hyperparameters based off the statistically measured
health of the neural network itself for an exact solution"). A first
attempt at a fixed half-life (2000 steps) DID verify the overflow fix but
was badly overtuned for the wide+sparse config: q/k/o_proj -- already
healthy with ZERO decay in the original buggy run (mean|w| ~0.04-0.13) --
got crushed to ~1e-5 by step 16000, meaning the guessed constant dominated
real learning instead of just providing a long-horizon health ceiling.
Picking a bigger constant (20000) would have been the same mistake again
with extra steps.

Real fix: no half-life at all. Each layer keeps its own ``decay_factor`` in
``self._l2_decay_factor`` (persistent across ``step()`` calls, lazily
initialized to 1.0 = no decay -- the ``max_abs_delta``/``max_ci`` hard bound
is the actual immediate safety net, so it's fine for L2 to do nothing until
it has real data). Every time a layer's rolling cursor completes a full
pass, its MEASURED rms is compared against the closed-form target
``1/sqrt(fan_in)`` (the same fan-in-normalized scale
``_preseed_dense_scattered`` already inits every dense fp32 layer to,
``sili/sparse_rnn.py``) and ``decay_factor`` is corrected multiplicatively
toward that target:

.. code-block:: python

   decay_factor *= clip((target / measured_rms) ** adaptation_rate, 0.5, 2.0)

clipped to ``(1e-6, 1.0]``. rms above target -> decay strengthens (factor
drops); rms below target -> decay relaxes (factor rises back toward 1.0,
never above -- L2 only ever shrinks). This converges toward the
fan-in-normalized target using the model's own measured statistics, the
same rms/mean_abs/max_abs this method already returns every cycle -- no
separate stats pass, no externally-guessed timescale. ``adaptation_rate``
is a control-loop GAIN (how fast the correction responds), not an
equilibrium-setting parameter like ``half_life_steps`` was -- a 2x-off gain
still converges to the same correct fixed point, just slower/faster,
unlike a 2x-off half-life which directly sets the wrong equilibrium
magnitude.

.. _toy_tile_recurrence_rmt.loss_adjusted_decay_design:

``apply_loss_adjusted_decay``: sustained-loss-triggered forgetting (critical-learning-periods hypothesis)
------------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.loss_adjusted_decay_design``

Direct instruction, following a rejected "variance in useful updates"
explanation for stalled/failed training runs ("no, there is no 'luck'.
This is pure math. If you're just getting lucky in math you're doing it
wrong."): the alternative hypothesis is that the network structurally
CAN'T UN-LEARN early bad adaptations -- matching the real
critical-learning-periods / loss-of-plasticity literature (Achille et al.
2019; Dohare et al., *Nature* 2024), which finds adaptive optimizers'
accumulated second-moment state specifically contributes to "frozen
units" that resist further updates, and that decaying that state (or, in
one study, full unit resets) is a documented mitigation. This fit the
observed crash-then-permanent-plateau shape of several stalled runs
(dense-unscaled, Polyak, Arm C+knee variants) far better than a gradual
variance story would.

Direct instruction: "implement the decays, implement them both while
making sure to note that we're actively testing them and may delete one
or more of the forms of decay if it doesn't work well in testing" --
``apply_loss_adjusted_decay`` implements BOTH arms as independently
toggleable half-lives (``importance_half_life_touches``,
``weight_half_life_touches``):

- **Importance decay** -- this project's per-synapse ``importance`` field
  plays the exact role of an Adam/RMSprop second-moment accumulator (see
  ``feedback_importance_is_already_the_optimizer``), so this is the
  direct, mechanism-specific fix: shrink the accumulator that's doing the
  freezing. Default-enabled (``importance_half_life_touches=500``) -- has
  its own dedicated C++ cursor (``_importance_decay_cursor``, see
  ``sili__new``'s ``docs/research/delta_csr_types.rst:
  amortized_decay.chunked_cursor``), no conflict with any other
  mechanism.
- **Weight/value decay** -- the more classic forgetting mechanism
  (biologically: synaptic decay, not optimizer-state decay). Default-OFF
  (``weight_half_life_touches=None``): it shares the SAME per-layer C++
  weight-decay cursor as the existing, already-in-production
  ``apply_amortized_l2_decay`` (RMS-closed-loop regularization, a
  different, unrelated mechanism -- see
  ``amortized_l2_decay_design`` above) -- there is only one
  ``_decay_cursor`` per layer in C++, so running both in the same
  training loop drives it out of sync with either mechanism's own intent.
  Enable this arm only in a loop that does NOT also call
  ``apply_amortized_l2_decay`` every step.

Direct instruction on granularity: "we need to limit the bits per synapse
to what they currently are" (no new per-synapse storage) and prefer
per-neuron over per-layer where possible. Loss is a single global scalar
with no natural per-neuron decomposition here, so this lands at
PER-LAYER granularity, uniformly applied via each real layer's own
``apply_amortized_importance_decay``/``apply_amortized_l2_decay`` call --
the same limitation ``per_layer_learning_rate_polyak`` already documents
for its own per-layer (not per-neuron) Polyak LR. Direct correction,
2026-09-20: a fully dense layer's uniform fan-in does NOT make per-layer
and per-neuron equivalent even there -- different neurons still receive
different gradient signal / contribute differently to the loss with
identical fan-in. Per-layer is accepted for now on granularity parity
with the source literature (Achille/Dohare test unit-/layer-level
interventions, not a loss-attributed per-neuron scheme either), not
because this method's dense case happens to erase the distinction. KNOWN
GAP, HIGH PRIORITY, tracked not fixed: a real per-neuron version needs
either a per-row-aware cursor or a per-row decay_factor scaled by that
row's own realized fan-in/gradient signal -- required before trusting
this on genuinely sparse layers with heterogeneous per-row nnz.

**Stall signal, purely LOSS-based, not instantaneous loss, never
accuracy/correctness**: a fast EMA of loss (``beta_fast=0.9``) is
compared against a slow-relaxing floor (``beta_floor=0.999``) tracking
the EMA's own best-ever value. Direct correction, 2026-09-20: the
``loss`` argument is whatever loss signal a caller has (prediction,
curiosity, external-input, ...) -- never correctness/accuracy/streak
(this project's curriculum advancement separately tracks its own
``acc_ema``/``streak`` for THEIR purposes, but neither feeds this
mechanism), since a deployed system may not have ground truth to check
predictions against. ``self._decay_stall_steps`` counts consecutive
steps since the EMA last beat the floor by a real margin
(``improve_tol``, relative) -- a TIE does NOT reset stall (real bug
caught by manual testing before ever launching a run: a plain ``<=``
comparison let the EMA's own filter-transient settling from its initial
value read as continuous "improvement" on a genuinely flat/oscillating
loss, permanently masking a real stall). Reset to 0 the instant a
genuine improvement occurs; the floor itself still relaxes slowly upward
on non-improving steps so a genuine curriculum-driven loss increase (a
harder level starting) doesn't read as a permanent stall forever.
``stall_frac = min(1, stall_steps/ramp_steps)`` is the actual 0..1 signal
driving decay severity -- deliberately insensitive to a single noisy
step (matching the "pure math, not luck" correction: a single loss spike
is noise, ``ramp_steps`` consecutive non-improving steps is a real
stall), saturating only after sustained elevation.

**Severity: half-life in touches, not an ad hoc strength/floor pair --
real miscalibration caught and fixed before ever launching a real run.**
The first version used ``decay_factor = max(min_decay_factor, 1 -
stall_frac*strength)`` with NO principled connection to how many times a
synapse actually gets touched over a real stall's duration -- and the
known historical stalls this hypothesis targets ran FLAT for
74,000-99,000+ training steps once stalled. At ``strength=0.3``, full
stall settled at ``decay_factor=0.7`` per touch (``min_decay_factor=0.5``
was never even reached -- ``max(0.5, 1-0.3)=0.7``); compounded over the
~100-call amortized cycle (see the touch_fraction paragraph below), an
empirical width=288 test crushed a real layer's weights to float-ZERO
within ~1400 calls -- a small fraction of one historical stall's real
length. That would have made the very thing under test ("does BOUNDED,
recoverable forgetting restore plasticity") into an accidental full
deletion well before the stall it's meant to help with even finishes.

Fixed by deriving ``decay_factor`` from a HALF-LIFE IN TOUCHES, the same
style this project's own ``apply_amortized_l2_decay`` docs already
establish for decay_factor derivation (``2^(-cycle_length/H)`` for a
chosen half-life ``H``): ``decay_factor_at_full_stall =
2**(-1/half_life_touches)``, interpolated toward 1.0 (no decay) as
``stall_frac`` falls: ``decay_factor = 1 - stall_frac*(1 -
decay_factor_at_full_stall)``. A synapse touched
``half_life_touches`` times in a row under FULLY saturated stall is
reduced to exactly half, never to numerical zero. Default 500 touches
(~50,000 calls at the default ``touch_fraction=0.01``, the SAME order of
magnitude as the historical stalls' own real length) to halve at worst.
Re-verified empirically at real width=288 scale over a 60,000-call
simulated full stall: ``lm_head``'s mean |weight| shrank from 0.137 to
0.0136 (~10x, the smallest/fastest-cycling layer), ``q_proj``'s from
0.047 to 0.0205 (~2.3x) -- bounded, gradual, real, and nowhere near
float-zero. ``None`` disables an arm entirely (an exact no-op, not
merely a large half-life).

**touch_fraction, not a raw chunk_size -- second real bug caught before
ever launching a run.** This project's layers span a ~150x nnz range at
width=288 (``lm_head`` early in the vocab curriculum: 576; q/k/v/o_proj
fully dense: 82944). One shared absolute chunk_size either barely
touches the big layers (a practical no-op) or, for the small layers,
exceeds their own nnz entirely -- the amortized cursor then wraps and
touches every synapse in that layer more than once per call, compounding
much faster than the half-life above assumes (each "touch" the half-life
counts is meant to be ~one per amortized cycle, not several per call).
``touch_fraction`` (default 0.01) sizes each layer's own chunk_size as
``clip(round(nnz*touch_fraction), min_chunk, max_chunk)``, making every
layer's own full amortized cycle take roughly ``1/touch_fraction`` calls
(~100 at the default) regardless of its absolute size.

Both arms also call each layer's block4 counterpart when present
(``apply_amortized_block4_importance_decay``/
``apply_amortized_block4_l2_decay``, ``sili__new`` task added
2026-09-20), since ``dense=True`` -- this investigation's actually-used
config throughout -- routes ``DISLDOLayerV`` weights into block4 storage,
not the scattered CSR path either arm's base call alone would cover; see
``feedback_block4_scattered_parity_required``. block4's own chunk_size
counts TILES (``_BLOCK4_TILE_SLOTS=16`` cells each), not individual
synapses -- reusing the scattered chunk_size directly there would
over-touch a block4-resident layer by roughly 16x relative to
touch_fraction's intent, so the block4 call gets its own chunk_size
divided by ``_BLOCK4_TILE_SLOTS`` (exact, not just an upper bound, for a
layer whose tiles are all still dense-format -- true for every
``dense=True`` config tested so far, since 16 live cells/tile at init
exceeds ``switch_point``'s 14-cell sparse threshold).

**Status: SUPERSEDED** by ``apply_plasticity_reset`` (below) as the
primary mechanism under test -- kept available, still fully tested, may
still be worth combining (Ablation D in the plasticity design) but is no
longer the first thing queued against the known-stalled configs. Built,
unit-tested at the engine level (``sili__new``'s
``test_block4_amortized_decay.cpp``/``test_amortized_decay_stats.cpp``)
and the model level (``TestLossAdjustedDecay`` in
``tests/test_toy_tile_recurrence_rmt.py``, 11 cases including the
tie/filter-transient regression, the ``min_stall_steps`` real-baseline
regression, and a direct half-life-halves-in-N-touches check), parameters
recalibrated against real width=288 nnz numbers.

.. _toy_tile_recurrence_rmt.plasticity_reset_design:

``apply_plasticity_reset``: per-neuron utility-based plasticity reset (Continual Backprop-style)
-------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.plasticity_reset_design``

Replaces ``apply_loss_adjusted_decay`` as the primary mechanism under
test for the critical-learning-periods/loss-of-plasticity hypothesis,
after web-verifying the actual mechanism the source paper (Dohare et
al. 2024, *Nature*, "Loss of plasticity in deep continual learning")
uses -- **Continual Backprop**: continuously resets the LOWEST-utility
MATURE hidden units at a tiny constant rate
(``u[i] = η·u[i] + (1-η)·|activation_i|·Σ|outgoing_weight_i|``), never
gated on loss at all. This is architecturally different from the
loss-stall-triggered design above in two ways that matter: (1) it
naturally never touches a genuinely-useful unit regardless of whether
global loss is flat -- an optimal, converged, flat-loss network is full
of high-utility units, protected by construction, no "is loss still
bad" threshold hack needed; (2) it's inherently per-NEURON, the
granularity gap the loss-based design could only flag as tracked debt.

**Direct refinements to the literal paper mechanism** (from
conversation, before implementation even started):

- Utility uses ``importance`` (this project's own per-synapse
  RMSprop-style accumulator, see ``feedback_importance_is_already_the_optimizer``)
  times ``|weight|``, not raw activation -- "in our current importance,
  an 'annoying' or noisy synapse will likely have high 'importance' but
  low value since it's mostly important to ignore, which means low
  actual utility." Sidesteps needing brand-new activation-tracking
  infrastructure.
- Amortized on TWO axes, not one -- not "reset one whole unit rarely"
  (the paper's literal shape) but "touch ~1% of a layer's own cell-grid
  every call" (reusing the SAME ``touch_fraction`` convention
  ``apply_loss_adjusted_decay`` already proved correct) AND make each
  touch a small BLEND toward the reset target, never an instant swap --
  "I don't like things that could add up and cause blips or stalls, but
  doing any compute that involves individual synapses will be expensive
  so we don't want to do all of it at once."

**A real, worked-through flaw, found and fixed before ever launching a
run**: direct question -- "calculate when, hopefully before the end of
the universe, a historically high utility column that was important for
5000 steps, that is now giving the wrong answer thanks to a new
curriculum, will take to become low enough utility again for this
algorithm to actually affect it." Worked the math: `col_util` (an EMA of
``importance*|weight|``) can only chase whatever that product CURRENTLY
is -- it can't fall below a value that isn't itself falling. The
literature's actual "frozen unit" failure mode is that a LARGE
accumulated ``importance`` suppresses the effective learning rate (the
update divides by ~√importance), so weight barely moves EVEN THOUGH the
column is now wrong and receiving real error signal -- and real error
signal means the gradient feeding ``importance`` stays large too, so
``importance`` stays high or keeps growing PRECISELY BECAUSE the column
is stuck. Self-sustaining bad equilibrium: **worst case, the answer is
never, not "a long time."** A mechanism built to fix loss-of-plasticity
that structurally cannot see the frozen-unit pathology wasn't testing
the hypothesis it claimed to test.

**Fixed design** (sili__new engine primitives:
``apply_amortized_plasticity_reset``/``apply_amortized_block4_plasticity_reset``,
``docs/research/delta_csr_types.rst:plasticity_reset.per_neuron_utility``
has the full engine-level derivation) -- two independent selection
pools sharing one amortized traversal:

- **FROZEN pool**: TOP-K by ``col_importance`` (high accumulated
  importance = frozen-risk, matching this project's own original
  hypothesis and the already-built ``apply_loss_adjusted_decay``
  importance arm, now per-neuron). Gated by a LOCAL per-column
  gradient-ACTIVITY deviation, not global loss -- "we're
  backpropagating through the whole model. 'Loss' is as local as we
  need it to be... a much, much, much better signal than 'global
  loss'." Two EMAs at different timescales
  (``col_grad_slow``/``col_grad_fast``) of a per-column
  gradient-activity signal; ``deviation = fast/slow``,
  ``plasticity_boost = max(0, deviation-(1+k))`` -- a column whose
  recent activity has spiked well above its own established baseline is
  frozen-and-now-demonstrably-wrong, attributed to THAT column, not a
  whole-network scalar. **Engine-safety correction, found DURING
  implementation**: the real backward kernel has 6+ SIMD-vectorized
  per-synapse update sites, too risky to hook correctly (this project's
  own documented fragility history around these exact kernels) -- so
  ``col_grad_slow``/``col_grad_fast`` are derived from
  ``col_importance``'s own per-CYCLE delta instead (``importance`` is
  already ``EMA(g²)``, only grows from real gradient activity, so its
  own trajectory is a safe, already-available proxy at this
  traversal's own cadence), never a real-kernel hook. Asymmetric EMA
  rate on ``col_grad_slow``: fast catchup on a genuine drop
  (improvement -- "the distribution can update its distribution"
  quickly), slow on a rise (so a real spike isn't instantly absorbed,
  which would defeat detecting it).
- **DEAD pool**: a SEPARATE, independent bottom-K by
  ``col_util_dead`` (``importance*|weight|``, the ORIGINAL formula --
  found, by the worst-case-timing math above, to be structurally unable
  to detect FROZEN columns, but honestly rescoped as exactly what it's
  good at: genuinely idle columns, a DIFFERENT pathology gradient-
  magnitude deviation alone can't tell apart from "optimal, low
  activity." Direct follow-up question confirmed this gap: "do those
  equations actually distinguish whether loss is above or below the
  usual in all cases" -- no, low gradient activity is ambiguous between
  "converged" and "dead/saturated," so this pool exists precisely to
  catch what the frozen-pool's gate structurally cannot). Full ``blend``
  rate, ungated -- lower risk of disturbing something still legitimately
  useful than the frozen pool, so the extra caution isn't needed.
  Mutually exclusive with the frozen pool by construction.

Both pools reset gradually (``blend``/``blend*plasticity_boost`` per
touch) toward a fan-in-scaled fresh sample
(``fp4_stochastic_normal01()``), never a sudden full swap -- and since
selection re-runs every cycle, a column that recovers (leaves the top/
bottom fraction, or whose deviation settles back down) simply stops
being touched. No fixed schedule fights against genuine continual
learning.

**Scale-invariance** (direct instruction: parameters "may be related to
... the average number of steps in [the model's] lifetime ... but if we
can get some non-infinite number as steps approaches infinite that could
give us an infinite time horizon setup too"): every quantity here is a
fixed-memory EMA, never a cumulative/growing running average -- an EMA's
responsiveness to new data is CONSTANT regardless of total elapsed
steps, so none of this mechanism's hyperparameters need retuning as a
deployment's total lifetime grows, unlike ``apply_loss_adjusted_decay``'s
own ``half_life_touches``, which WAS explicitly anchored to "the
historical stalls' observed 74k-99k-step length" (a finite-horizon
calibration this mechanism has no equivalent of).

**No ``loss`` argument anywhere** in ``apply_plasticity_reset`` -- purely
local per-column signals, cleanly distinct from
``apply_loss_adjusted_decay``, which is global-scalar-loss-gated.
Always structurally active when called (mirrors
``apply_amortized_l2_decay``'s own always-on shape).

Fp32-first, by direct instruction ("we're testing this for fp32 first
and then applying to everything if it works right") -- only wired on
``DISLDOLayerV``/``DISLDOLayer32`` (matching this whole investigation's
actually-used config, ``dense=True`` routing through block4) so far;
FP4/FP8 scattered classes NOT extended yet, a real, tracked follow-up
gap, not silently dropped -- ``TestPlasticityReset::test_non_fp32_backend_reports_nothing_yet``
pins down the current (intentional) behavior: an FP4-backed model
reports an empty dict from ``apply_plasticity_reset``, no error.

TDD throughout on the sili__new side (every primitive's test written
and confirmed red before its implementation existed) -- see
``docs/research/delta_csr_types.rst:plasticity_reset.per_neuron_utility``
for the full engine-level test inventory (21 C++ unit tests across 3
files, including a scattered-vs-block4 bit-equivalence check and a
dedicated worst-case-timing regression test). Model level:
``TestPlasticityReset`` in ``tests/test_toy_tile_recurrence_rmt.py`` (6
cases: touch_fraction sizing across this test file's real nnz range,
fp32-only backend dispatch, block4 substats present/absent as expected,
many-cycle smoke test with sane internally-consistent stats, no ``loss``
argument required, custom knobs threaded through correctly).

**Status: not yet empirically tested** against the stalled runs this
hypothesis was built to explain. Next: relaunch the same 3 known-stalled
configs already identified for ``apply_loss_adjusted_decay``
(``launch_width288_nolevel_down_control.py`` dense-unscaled,
``launch_polyak_lr_width288.py`` Polyak, and
``launch_arm_c_plus_knee_margin15.py`` Arm C+knee) with
``plasticity_reset_enable=True`` instead, compare against their original
stalled trajectory.

.. _toy_tile_recurrence_rmt.plasticity_column_state:

Per-column state logging for offline reset-equation fitting
-------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.plasticity_column_state``

Direct instruction, after reviewing the log files this investigation
already produces (importance, loss, accuracy): could a real
reset-selection equation be FIT from actual training data instead of
hand-derived heuristically? Scoped to per-COLUMN state, not
per-synapse -- the mechanism only ever selects at column granularity
(``col_importance``/deviation/age are per-column aggregates already),
so per-synapse snapshots (the original, much larger storage estimate)
would add cost without adding signal for this specific goal.

``apply_plasticity_reset(..., include_column_state=True)`` fetches a
copy of ``PlasticityState``'s own arrays (see
``docs/research/delta_csr_types.rst:plasticity_reset.plasticity_column_state``
for the engine-level accessor) for any pool whose cycle just
completed -- off by default (extra work most callers don't need).
``train_mqar_curriculum.py``'s ``plasticity_column_log_dir`` writes one
small ``.npz`` per completed cycle per layer/pool, including
``step``/``loss_ema``/``acc_ema`` alongside the column arrays -- the
~1% of columns reset each cycle serve as a natural "treatment" group
against the ~99% left alone, without needing any separate intervention.

**Real bug found via smoke-testing the actual wiring** (not caught in
planning): a dense-loaded real layer's content lives entirely in
block4 (``load_dense_values`` routes through ``block4_load_dense_fp32``
only) -- its scattered arm has 0 nnz for the whole run, so its own
``cycle_complete`` trivially fires on literally every call via the
engine's empty-nnz early-return path, with all-zero column state. A
naive 500-step smoke run wrote 3036 files, almost entirely that noise.
Fixed with ``scattered_nnz`` (sili__new), gating column-state
attachment to arms with real content -- re-verified: 36 files, all
holding genuine non-zero values.

Storage cost is small at this granularity: the same 500-step smoke run
produced 408KB total (block4 arms only). A full 100k-step run
extrapolates to roughly 80MB per arm -- negligible against the 180GB
free on this machine, and far cheaper than a full per-synapse weight
snapshot (which would have run into the tens-of-GB range for the same
horizon).

.. _toy_tile_recurrence_rmt.to_sparse_gradient_detach_bug:

``_to_sparse``: real bug -- ``CSR.as_tensor()`` silently detached the graph
--------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.to_sparse_gradient_detach_bug``

Sparsity plan Phase 6 (task #335) helper -- mirrors the existing
``sparse_rnn.py`` recurrent-cell precedent
(``if not isinstance(state.data, CSR): ... CSR.from_dense(...).as_tensor(...)``)
exactly, just as its own small reusable helper here. A no-op (returns x
unchanged) when neither ``x_r_target[layer_name]`` nor ``input_sparsity_p``
is set -- every existing caller sees a completely unmodified dense Tensor,
same object even.

Root-cause fix (found via a sigma-gradient debug probe): ``CSR.as_tensor()``
returns a bare leaf Tensor (no ``_children``/``_backward``), so the old
version of this method silently DETACHED x from the graph -- any layer
downstream of a ``_to_sparse`` call (o_proj, in particular) still updated
its OWN weights fine (its dx lands on the CSR tensor's own ``.grad``, which
``as_tensor``'s docstring already anticipated being dense), but that
gradient never propagated past the CSR tensor, since its ``_backward`` was
the default no-op. That orphaned everything upstream of any ``_to_sparse``
boundary -- confirmed directly: ``log_sigmas.grad`` was None on every
backward call at ``embed_width=32`` (``input_sparsity_p`` set) vs
real/nonzero every call at ``embed_width=16`` (``input_sparsity_p=None``,
this method a no-op), since centers/log_sigmas only reach the loss via
``gaussian_attention -> attn_pre_o_{mem,content} -> o_proj``, i.e.
exclusively through this exact boundary.

.. code-block:: python

   # WRONG (silently detaches): as_tensor() returns a bare leaf Tensor
   out = csr.as_tensor(x.backend)   # no _children, no _backward -- dead end
   return out

   # Fix: wire real _children/_backward so gradient flows back to x,
   # STRAIGHT-THROUGH (unmasked), not restricted to the k kept positions.
   out = Tensor(csr, _children=(x,), _op="to_sparse", backend=x.backend)

   def _bwd():
       if out.grad is None:
           return
       g = np.asarray(out.grad, dtype=np.float32)
       if x.grad is None:
           x.grad = x.backend.zeros_like(x.data)
       x.grad = x.backend.add(x.grad, g)

   out._backward = _bwd

Straight-through (the full dense downstream gradient is passed back to x
unmasked) rather than a hard mask-the-gradient-too version -- masking would
zero the learning signal for every non-selected position on every step,
making the selected/dropped split itself unable to shift over training,
which is its own kind of frozen state.

``layer_name`` (task #365): identifies WHICH of the 5 wide layers is about
to CONSUME this sparsified tensor as its forward input -- selects that
layer's own ``x_r_target`` entry. Several call sites feed the SAME
underlying dense tensor to multiple layers (e.g. ``combined_normed`` feeds
q_proj/k_proj/v_proj all three) -- each of those calls this method
separately with its own ``layer_name``, so a layer whose ``x_r_target``
differs from its sibling consumers gets its own independently-selected
top-k set, not a shared one. This means 3x the selection work for that
shared-input case when per-layer ``x_r_target`` values actually differ
(accepted cost of genuine per-layer control, same tradeoff task #372 already
made for ``dy_r_target``).

``x_r_target[layer_name]`` set (task #365): nucleus/energy-threshold
selection via ``_nucleus_top_k_csr`` -- TAKES PRIORITY over
``input_sparsity_p``, mirrors ``dy_r_target``'s own priority over
``dy_sparsity_p`` (task #367) exactly. Falls back to ``input_sparsity_p``'s
fixed-fraction ``CSR.from_dense`` when ``x_r_target[layer_name]`` is None
(today's exact pre-#365 behavior, unchanged).

.. _toy_tile_recurrence_rmt.input_selection_stats_design:

``_update_input_selection_stats``: a free-byproduct diagnostic, not an accumulator
------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.input_selection_stats_design``

Task #369: real per-layer INPUT-axis trajectory stats -- R (the actual
captured squared-magnitude ratio R(v,k) THIS call achieved, not just the
r_target it was asked for) and k (mean kept-entries-per-row), computed
directly from the CSR ``_to_sparse`` already just built (near-zero extra
cost -- unlike the grad axis, this method already has both the dense input
and the resulting CSR in hand, no new C++ instrumentation needed).
Meaningful for BOTH the nucleus (``x_r_target``) and fixed-fraction
(``input_sparsity_p``) paths -- R is a real, honest diagnostic either way.

Stored in ``self.last_input_selection[layer_name]``, OVERWRITTEN each call
(mirrors ``self.last_debug``'s own "most recent snapshot" convention), not
accumulated across steps. Direct instruction (bounded fine-grained logging,
opt-in, off by default): the actual "log every N steps" / "log this step
range" gating policy lives in the CALLER (e.g.
``train_mqar_curriculum.py``'s own ``trajectory_log_every``), which already
tracks its own step counter -- the model itself stays stateless about step
numbers, same existing design choice as every other per-step model
attribute here (``last_debug``, ``last_critic_pred``, etc.).

.. _toy_tile_recurrence_rmt.last_grad_selection_design:

``last_grad_selection``: the grad-axis counterpart, now instrumented (2026-09)
------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.last_grad_selection_design``

The "unlike the grad axis" caveat above is no longer true -- sili's own
``_nucleus_top_k_csr`` call sites for ``dy_r_target`` (``DISLDOLayer.forward``
and ``DISLDOLayer32.forward``, see sili__new's
``record_grad_selection_stats.design``) now capture the same R/k stats onto
each layer's own ``last_grad_selection``, right where the CSR is actually
produced (this model's own Python has no visibility into that call, so it
couldn't be computed here the way the x-axis stat is). This model's
``last_grad_selection`` PROPERTY (not a plain dict -- computed on read, not
maintained incrementally) pulls it back together across
``_named_real_layers()``, mirroring ``last_input_selection``'s shape so
``train_mqar_curriculum.py``'s trajectory printer can treat both axes the
same way. A layer with no entry (no ``dy_r_target`` backward call yet, or a
disldo_cls with no ``dy_r_target`` support at all, e.g. ``DISLDOLayer8``) is
simply omitted, not a missing-key error.

Real measurement (wide288 scale, ``r_target=0.9``, before the closed loop
ratchets down): the grad axis kept roughly 2x MORE of its tensor than the
x axis did at the same nominal setpoint (q/k/v: dy ``k~=128-133/288``
vs. x ``k~=58/288``) -- gradients are measurably less concentrated than
forward activations at this stage of training, so the two axes' densities
do not simply multiply together the way a naive combined-density estimate
would assume. Whether this gap narrows (or the grad axis becomes sparser
than x, as hypothesized) later in training is an open question this
instrumentation now lets a real run answer directly, rather than by
extrapolating from a single early-training snapshot.

.. _toy_tile_recurrence_rmt.l1_sparsity_split_design:

``_l1_sparsity_split``: a parallel probe branch, no longer order-sensitive
--------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.l1_sparsity_split_design``

Exact port of ``ToyTileRecurrenceRealFP4``'s own helper. This probe
``forward()``-calls the SAME layer instance a SECOND (or third, for
q/k/v/o_proj under the sequential write-then-read design) time within one
``step()`` -- a genuinely PARALLEL branch that only merges back into the
loss via simple addition, not a dependency of the main pass's own output.
sili__new's ``backward_dense``/``backward`` now take ``x`` as an explicit
argument (each Python closure holds its own input directly, same as every
other Tensor op), so this is no longer order-sensitive at all -- previously
needed ``use_explicit_token`` to avoid a real engine-side LIFO-cache bug,
now simply not a concern.

``layer_name`` (task #373): when given (one of the 5 wide layers), this
call's real forward+backward cost is folded into ``self._layer_timing``
too -- this probe is real compute the same as the main pass, so the budget
allocator (#375) needs to see it. None (default, used by ``lm_head``'s own
untimed call site) skips timing entirely.

.. _toy_tile_recurrence_rmt.step_design:

``step()``: write-then-read within one call, not BPTT across calls
------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.step_design``

``x_window``: ``[num_tiles, embed_width]`` -- same convention as
``ToyTileRecurrenceRealFP4`` (a real per-tile embedding, zeros for "nothing
here yet"). ``memory_prev``: ``[num_memory_slots, state_width]`` --
genuinely dedicated memory, DETACHED (no BPTT across steps, matching this
whole project's convention), unlike ``ToyTileRecurrenceRealFP4``'s
ambiguous per-window-slot rolling state. Returns ``(memory_new`` numpy
``[num_memory_slots, state_width]``, ``logits`` Tensor
``[num_tiles, vocab_size]``, ``aux_loss)``.

``requires_grad=False`` (direct instruction): most ``step()`` calls in a
real training loop are NOT query positions (no loss ever gets computed for
them, so ``backward()`` never runs) -- pass ``requires_grad=False`` for
those to skip building the backward graph entirely, matching
``train_mqar_curriculum.py``'s own ``i in targets`` check. The detached
``memory_new`` handoff to the NEXT ``step()`` call is completely unaffected
either way -- this only controls whether THIS step's own forward pass is
backprop-able, not what values it computes. (The sequential write-then-read
design calls k_proj/v_proj/o_proj TWICE per step -- once for the write,
once for the read -- but sili__new's ``backward_dense``/``backward`` now
take ``x`` explicitly, so there's no engine-side call-ordering concern to
manage either way.)

``content_dy_sparsity_schedule`` (query-step graded credit-assignment
design, see ``step_cached``'s own docstring and JOURNAL.md): list of length
``num_tiles``, index 0 = oldest content position ... index
``num_tiles-1`` = newest, giving each content row its OWN backward gradient
density instead of one uniform value -- e.g. full density for the newest
(just-computed) row, progressively less for older cached ones, cheaper
than uniform full density but richer than ``step_cached``'s
zero-credit-for-older-rows default. Memory rows always keep full density
(1.0) regardless -- they're the live recurrent state, not a graded-by-age
position. None (default): completely unchanged behavior, uses
``self._wide_extra_kwargs``'s own scalar ``dy_sparsity_p`` exactly as
before. Uses sili__new's ``dy_sparsity_schedule`` kwarg (real per-row
top-k, NOT the same as the scalar ``dy_sparsity_p``'s own surprising
global-across-the-batch top-k semantics).

.. _toy_tile_recurrence_rmt.write_then_read_pass_design:

The two-pass write-then-read structure inside ``step()``/``step_cached()``
--------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.write_then_read_pass_design``

**PASS 1 (WRITE)**: memory reads the (stale) full window, live unrestricted
regardless of ``recurrent_only_output`` (memory always reads everything;
"write" stays allowed per the ablation's own design), producing
``memory_new``.

Real bug fixed here (task #303/#304, direct instruction): q/k/v are clipped
BEFORE they enter ``gaussian_attention``, not after. Previously only
attn/combined_new were clipped, AFTER attention's own internal
dot-product/exp math had already run on unbounded q/k/v -- confirmed via
real diagnostics that ``v_proj``'s output alone reached 1000s-2000s
magnitude for hundreds of steps, invisible externally because the existing
downstream clips masked it, until an unscaled dot product inside
``gaussian_attention`` finally overflowed to NaN. Sigma is floored the same
way, before ``gaussian_attention`` uses it (see
``toy_tile_recurrence_rmt.magnitude_clip_and_min_sigma_design``). K/V are
also reordered into PHYSICAL (genuinely interleaved) position order before
acting as attention KEYS here -- see
``toy_tile_recurrence_rmt.interleaved_position_layout_bug``; Q stays in
LOGICAL order untouched, since each query row's bias center is looked up
from ``self.centers``, which already holds the correct physical-position
VALUE per logical row.

**PASS 2 (READ)**: content queries attend memory AS IT STANDS AFTER pass
1's write, not the stale ``memory_prev`` (direct instruction:
"input_proj->state_update->recurrent->state_update in one step, not BPTT"
-- two sequential layers run one after the other WITHIN this same
``step()`` call). Backprop from ``content_out``/``logits`` walks straight
through ``memory_new_t``, through ``attn_mem``/``attn_pre_o_mem``, through
``q_mem``/``k_phys``/``v_phys``, into ``x_wide``/``input_proj`` -- a real,
live, undetached gradient path entirely WITHIN this one call, NOT BPTT
(nothing here crosses a ``step()`` call boundary; only the numpy
``memory_new`` returned at the very end, after this whole graph is already
built, gets detached). Before this pass existed, ``input_proj``'s only live
signal was the thin "which memory slot does my query pick" channel
(confirmed via direct measurement: ``input_proj`` abs-grad-sum 0.037 vs
``v_proj``/``o_proj``'s 2.89/19.4, both of which get real credit only
because their SAME shared weights are also exercised, abundantly, by the
read side every step) -- this pass gives it a real path for "did this write
end up useful," without ever needing gradient to survive the hard-detach
between ``step()`` calls.

Debug instrumentation (task #303): ``self.last_debug`` is a cheap
reference-only capture (no copies) of every stage between the input
embedding and the readout, for bisecting exactly where a NaN/Inf first
appears in the forward chain -- ``np.clip`` does NOT sanitize NaN
(``clip(nan,...)==nan``), so the clip calls in ``step()`` are not
themselves proof any given stage is finite. ``self.last_debug["x_window_t"]``
holds the Tensor itself (not just ``.data``) as an embedding-learning hook
(direct instruction): when ``requires_grad=True`` and the caller runs
``loss.backward()`` after ``step()`` returns, this Tensor's ``.grad`` is
populated with ``dL/d(x_window)``, letting a caller scatter-update an
external embedding table (e.g. an SDR token embedding built outside this
model) without ``step()`` needing to know about tokens/vocab at all.

.. _toy_tile_recurrence_rmt.step_cached_design:

``step_cached()``: an incremental alternative, and the one approximation it accepts
------------------------------------------------------------------------------------------

*ID:* ``toy_tile_recurrence_rmt.step_cached_design``

Incremental alternative to ``step()``: takes ONE new token's raw embedding
``[embed_width]`` instead of a full ``[num_tiles, embed_width]`` sliding
window, plus an explicit ``tile_cache`` carrying the ``num_tiles-1`` older
content positions' ``(k_row, v_row)`` -- same explicit-state-in/out
convention as ``memory_prev``/``memory_new`` (no hidden instance-mutation
cache, matching this project's own established preference -- an earlier
engine-side hidden cache caused a real, hard-to-find correctness bug, see
project memory ``project_sili_dense_input_stack_simplification``).

Why this is correct, not just faster: ``input_proj``/``q_proj``/``k_proj``/
``v_proj`` are simple per-row (non-mixing) projections, so a content tile's
k/v depends ONLY on its own token embedding and the CURRENT weight values
-- never on other tiles, never on memory. ``step()``'s full-window rebuild
therefore recomputes up to ``num_tiles`` IDENTICAL values per token as the
window slides past it, every single call. Caching removes that redundancy.

Direct instruction on the one real approximation this introduces: weights
only change on ``requires_grad=True`` (query) steps, and even then only
~0.1% of individual synapses move per update (see project memory
``dy_sparsity_p_validated_speedup``'s backward-sparsity findings) -- so a
cached tile's k/v drifts by a tiny, bounded amount as it ages through the
window, rather than being invalidated wholesale after every weight update.
Treated as "mostly fine" per direct instruction, not chased to exact
invalidation.

Only the NEWEST content position's own logits/q ever get used downstream
(confirmed: ``train_mqar_curriculum.py``'s own loss/accuracy always reads
row ``num_tiles-1``, never any other content row) -- so q, x_wide,
attention, and the final ``lm_head``/``critic_head`` readout are all
computed for exactly ONE content row here, not ``num_tiles``. This also
means ``step_cached``'s own ``logits``/``aux_loss`` shapes are
``[1, vocab_size]`` (a single row), not ``[num_tiles, vocab_size]`` --
callers reading row 0 instead of row ``num_tiles-1`` is the one real
call-site change needed.

``tile_cache``: list of up to ``(num_tiles-1)`` ``(k_row, v_row)`` numpy
``[state_width]`` tuples, OLDEST FIRST. None or an empty/short list (fewer
than ``num_tiles-1`` entries) is padded with zero rows at the oldest end --
exactly reproducing ``_build_tile_window``'s own "zeros for nothing here
yet before sequence start" behavior (``input_proj``/``k_proj``/``v_proj``
have no bias term, so a zero raw embedding really does propagate to an
exact zero k/v row, not an approximation). Reset to None/[] at the start of
each new training sequence, same as ``memory_prev`` gets reset to zeros.

.. _per_layer_learning_rate_polyak:

``layer_lr_override``/``apply_polyak_lr``: per-layer dynamic LR
-------------------------------------------------------------------------------

*ID:* ``per_layer_learning_rate_polyak``

2026-09-18. Direct motivation: several full 100k-step runs were needed
just to bracket a good ``peak_lr`` by hand for the width-scaling
degeneracy investigation (see ``train_mqar_curriculum.rst:
train_curriculum.width_scaling_lr_fanin_hypothesis``). Researched
task-agnostic dynamic-LR methods (D-Adaptation/Prodigy need a distance-
to-solution estimate that's ungrounded for a real task we don't know
the answer to ahead of time; hypergradient descent needs consecutive-
step gradient VECTORS, which would mean exposing per-synapse detail --
too much; the Stochastic Polyak Step-size only needs the loss and a
gradient-energy scalar, both cheap and already close to hand) -- direct
instruction: Polyak, applied per-layer directly (not layered under a
separate per-layer mechanism like trust ratio, which needs a weight
norm -- "still kind of ridiculous" as a step-size signal, and is its
own extra cost).

``_timed_call`` (this file) already captures ``_out.grad`` -- the exact
``dy`` a layer's own backward consumes -- for every wide-layer call,
previously only used to update ``self._layer_surprise[name]`` (E_t/Lbar)
when ``dy_surprise_alpha`` was set. Now UNCONDITIONAL (cheap,
``sum(dy**2)`` over an already-realized array -- same "always track the
diagnostic" precedent as the knee-elbow tracker in ``_to_sparse``), so
``apply_polyak_lr`` has a real per-layer gradient-energy signal for
free, no new engine hook needed on this side (a parallel, independent
hook was ALSO added on ``sili__new``'s ``DISLDOLayer32`` --
``disldo_layer_forward.last_grad_norm_sq_polyak_hook`` -- for callers
that don't go through ``_timed_call``; this file's own training loop
uses the simpler already-existing ``_layer_surprise`` path instead).

``self.layer_lr_override: dict`` (empty by default) is checked FIRST in
``_timed_layer_forward``, ahead of whatever scalar ``learning_rate`` was
passed in -- only the 5 wide layers go through that method (lm_head/
critic_head never see a per-layer override, matches ``dy_r_target``'s
own scope exactly). ``apply_polyak_lr(loss, f_star, c, lr_max,
bootstrap_lr)`` computes ``lr_layer = min(lr_max, c * max(loss-f_star,
0) / E_t_layer)`` per wide layer and populates this dict -- one step
LAGGED (uses last step's loss and last backward's E_t, same convention
as ``_effective_dy_r_target``, unavoidable since a layer's own
``learning_rate`` has to be fixed before its forward+backward call
that WOULD produce this step's own E_t).

TODO, not yet built (direct instruction, deferred as real extra work):
a PER-NEURON version (one Polyak lr per row of a layer's weight
matrix, not one per whole layer) -- structural sparsity
(``max_weights``) and gradient sparsity (nucleus top-k,
``dy_gate_mask``) both act ROW-WISE, so a layer-wide scalar smears
together neurons with very different realized fan-in/update frequency
this step in a way per-neuron wouldn't. Revisit if per-layer doesn't
adapt well, or as its own later test -- per-layer is the first thing
to try.

**Recalibration, 2026-09-19: first validation run failed to learn at
all under the initial defaults.** ``launch_polyak_lr_width288.py``'s
first run reached only ``vocab=16, k=3`` at step 713, then flat/noisy
for the remaining 99,287 steps of its 100k budget -- far worse than
the hand-tuned ``peak_lr=0.01`` record (``vocab=126, k=3`` at step
13,601). Diagnosed directly (patched ``apply_polyak_lr`` to log its
own inputs/outputs, a first attempt with a shallow-copied snapshot
gave misleadingly frozen-looking values -- fixed with a proper deep
copy): the raw per-call ``E_t`` for q/k/v/o_proj swings across orders
of magnitude call to call (observed range ~0 to ~2000 for a single
layer), so ``c * residual / E_t`` constantly saturated against
``lr_max`` regardless of its value -- and ``lr_max=0.1`` (the original
default) is already well into the instability regime the LR range test
found (~0.1-0.32). Training 4 of 5 wide layers at ~0.1 nearly every
step for 100k steps is a very plausible explanation for the observed
flat/noisy failure.

Fixed: switched the denominator to ``Lbar`` (the EMA-smoothed running
gradient energy, already computed by the same ``_update_layer_surprise``
call, just unused before) instead of the noisy raw ``E_t`` -- and
recalibrated ``c`` from a literature-typical SPS damping factor (0.5)
down to ``0.0005``, ``lr_max`` from ``0.1`` down to ``0.05``, both
empirically, to land the computed values in the same ~0.01-0.06 range
the LR range test and grid search already found safe rather than
trusting an off-the-shelf constant that turned out not to transfer to
this setup's actual per-layer gradient-energy scale. A short (1500-
step) smoke test with the new defaults shows sane, decreasing lr
values (mostly 0.0006-0.05 for q/k/v/o_proj, falling as loss falls --
real annealing behavior) and genuine progress (reached k=3, loss
3.8->1.8) instead of the frozen pattern. One residual imperfection,
not yet addressed: ``input_proj``'s computed lr is driven near-zero
(its ``Lbar`` runs on a different scale than q/k/v/o_proj's, likely
from its different input/output dimensions -- ``embed_width x
state_width`` vs ``state_width x state_width``) -- a single shared
``c`` doesn't calibrate all 5 layers equally well. Worth watching if a
longer validation run underperforms.

.. _arm_c_single_state_angle_fix:

Arm C gate: corrected from a 3-state design to 1-state, 2026-09-19/20
-------------------------------------------------------------------------------

*ID:* ``arm_c_single_state_angle_fix``

Direct correction, caught in conversation, not by review of the running
code: the implementation built earlier this session
(``gate_j(t) = sin(2*pi*t/period_j + phase_j(t)) > cutoff``) used THREE
stateful pieces per neuron -- a shared global step counter ``t``, a
per-neuron period drawn once from ``Uniform(dy_time_gate_period_range)``
and held fixed, and a separately-accumulated additive phase random walk
``phase_j(t) = phase_j(t-1) + N(0, phase_step)`` -- none of which matches
the original spec ("small infinitesimals added to vars on each neuron
before sin is computed"). ``t`` was never supposed to be an absolute
training-step counter; ``period_j`` varying per neuron was unnecessary
complexity; the separate phase-noise term on top of a deterministic
``t/period`` term was two mechanisms doing one job.

**Fixed**: ``angle_j(t) = angle_j(t-1) + N(2*pi/period, phase_step)`` --
ONE accumulating per-neuron state (``self._dy_time_gate_angle``),
started uniform in ``[0, 2*pi)``, advanced each call by a single normal
draw whose MEAN is a fixed shared constant (``2*pi/dy_time_gate_period``,
same for every neuron -- no more per-neuron period draw) and whose
spread (``dy_time_gate_phase_step``, unchanged default 0.02) supplies
the "infinitesimal" wobble directly, with no separate phase term needed.
``gate_j(t) = sin(angle_j(t)) > cutoff``, same formula shape, same
density-vs-cutoff relationship (verified empirically:
``density(cutoff=0.3)`` measured 0.4030 over 20,000 steps/2,000 neurons
vs the theoretical ``0.5 - arcsin(0.3)/pi = 0.40301`` -- matches to 4
decimal places). No global step counter needed at all --
``self._dy_time_gate_t`` and its increment in ``step()`` were removed.

**Scope caveat**: every Arm C result recorded in this investigation so
far (the original confusion-matrix win, test 3's full grid, the
gate-density LR compensation tests, the 12-run seed sweep) used the OLD
three-state mechanism, not this corrected one. The qualitative
conclusions (Arm C beating signal-guided selection, the density-driven
reliability finding from the seed sweep, the LR-tolerance findings) are
not expected to flip from this fix -- the underlying behavior (each
neuron gets periodic, non-repeating trainable windows, density set by
cutoff) is the same in spirit -- but the EXACT numbers (which specific
neurons were on when, on any specific run) are not reproducible against
the old code anymore. Not rerun yet.
