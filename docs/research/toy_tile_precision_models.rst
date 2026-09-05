``toy_tile_precision_models.py`` research notes
====================================================

Companion doc to ``model/toy_tile_precision_models.py``. Source comments
point back here by anchor ID (``*ID:* `` marker under each heading below);
this doc links back to source by class/function/parameter name. See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line,
frozen code snippets on real-bug/non-obvious-derivation sections only).

.. _toy_tile_precision_models.module_overview:

``ToyTileRecurrenceRealFP4``: architecture overview, clip_range, SwiGLU/tanh removal
------------------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.module_overview``

``ToyTileRecurrence``'s exact architecture, built from ``DISLDOLayer``-family
layers (``disldo_cls=``) instead of ``DenseTensorLinear``. ``centers``/
``log_sigmas``/RMSNorm weights stay plain Tensor leaves, trained by a small
external ``AdamOptimizer`` via ``parameters_for_optimizer()``.

SwiGLU MLP and tanh have been removed in favor of a minimal attention-only
recurrence with ``[-clip_range, clip_range]`` state clipping. Default 6.0
(matching FP4/E2M1's own max representable magnitude) -- confirmed via
direct comparison against the original 2.0 default (mean_acc 0.98 vs 0.75,
3/3 seeds, lower variance, already converged vs still mid-learning at step
15000; see sili_peridot JOURNAL.md's clip-range test entry for the full
result).

``step()`` returns ``(M_new, logits, aux_loss)`` -- ``aux_loss`` is ``None``
unless ``use_energy=True``.

.. _toy_tile_precision_models.known_differences_from_proven_designs:

Known differences from proven segment/block-recurrent transformers
------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.known_differences_from_proven_designs``

Found by direct comparison against real, working designs (Recurrent Memory
Transformer/RMT, Block-Recurrent Transformer, Infini-attention, Perceiver
IO) after this model's own K=1 MQAR runs stalled around 0.12-0.25 despite an
fp32 hand-built witness reaching ~0.95-1.0 (see conversation) -- i.e. the
individual pieces (attention, RMSNorm, fp32 layers) provably work;
something about how they're WIRED TOGETHER here is the suspect. Listed so
each can be ablated independently, one at a time, toward whichever
known-working design it's checked against -- NOT to be "fixed all at once,"
since that would confound which change (if any) actually matters:

1. **State/input combination** (``qkv_source``) is a plain untrained
   elementwise sum of two independently-RMSNorm'd streams (``x_normed +
   m_normed``). Every cited design either keeps state as separate
   attention-visible tokens (RMT: memory tokens literally concatenated into
   the sequence) or merges via a LEARNED gate (Infini-attention: sigmoid
   beta). Partially ablated: ``gated_combine`` (see
   ``toy_tile_precision_models.gated_combine`` below) replaces the sum with
   a learned, per-channel, input-dependent sigmoid gate -- real trained
   comparison (K=1 MQAR, single seed, 20k steps) did not yet beat the
   plain-sum baseline (0.133 vs 0.250), though loss/trajectory were the
   smoothest of any arm tried -- inconclusive on n=1, multi-seed re-test
   pending.

2. **State UPDATE has no learned forget gate at all** -- STILL UNFIXED.
   ``M_new = M_prev + attn_o_proj_output`` is a plain residual add, then
   RMSNorm, then a hard (non-learned, autograd-bypassing) clip. This is
   architecturally the pre-LSTM "vanilla RNN" pattern gating was invented
   to fix. Block-Recurrent Transformer's actual LSTM-style gates control
   exactly THIS step (``forget_gate*M_prev + input_gate*new_content``), not
   the pre-attention combination -- ``gated_combine`` gates a DIFFERENT,
   earlier point in the pipeline and should not be assumed to substitute
   for this. ``gated_update`` (see
   ``toy_tile_precision_models.gated_update`` below) is the real ablation
   built for this, independent of #1.

3. **Readout** (``pooled``) is a uniform, content-blind mean over each
   block of ``column_neurons`` state channels -- every element in a column
   contributes EQUALLY regardless of whether it actually carries useful
   signal for that output channel. Perceiver IO (the closest proven
   precedent for a narrow<->wide bottleneck) bridges narrow<->wide via
   LEARNED cross-attention (the narrow side queries the wide side,
   content-based weights) in both directions, never fixed unweighted
   averaging. Not yet ablated.

4. **State occupies the SAME** ``num_tiles`` **window slots as fresh
   input**, pre-merged before attention ever runs -- attention itself never
   gets to choose "read from state" vs "read from input" as a separate
   degree of freedom, since that choice is collapsed before Q/K/V are even
   computed. RMT/Block-Recurrent Transformer instead give state its own
   separate token(s)/cross-attention target, kept genuinely distinct from
   input tokens throughout. Tracked as its own exploration (see project
   task list: "Explore RMT-style separate-token state"), not yet built -- a
   bigger structural change than 1-3 above, deliberately deferred until
   those are ruled in/out first.

Also planned (see project task list): a from-scratch reference
implementation of one of the proven designs above (RMT or Block-Recurrent
Transformer), run on the SAME K=1 MQAR task/harness as a sanity control --
if a known-good design also fails here, the problem is in the
task/embedding/harness, not this class's recurrence design specifically.

.. _toy_tile_precision_models.rng_per_layer_seeding_bug:

Per-layer RNG seeding: a real unseeded-RNG bug
----------------------------------------------------

*ID:* ``toy_tile_precision_models.rng_per_layer_seeding_bug``

Per-layer independent seeds are derived from ``rng``, matching this
project's own established convention (``scripts/disldo_*_ablation.py``:
``np.random.default_rng(seed+1)``/``(seed+2)`` per sublayer). NOT passing
``rng=`` down to ``disldo_cls`` at all was a real bug found directly (same
unseeded-RNG class as ``feedback_seed_stochastic_rng_for_comparisons``):
``_preseed_random_sparse`` defaults to ``np.random.default_rng()`` (fresh
OS entropy) whenever ``rng=None``, so every layer's initial connectivity
and initial weight values were NEVER controlled by the ``seed`` CLI arg --
confirmed directly, same command/seed gave 0.70 then 0.65 final-step
accuracy across two back-to-back runs. ``seed`` only ever controlled the
embed table, task generation, and (separately) FP4 stochastic rounding.

``gated_combine``'s and ``gated_update``'s extra layer seeds
(``gate_x_proj``/``gate_m_proj``, ``update_forget_proj``/
``update_input_proj``) are only reserved when actually used, so
``gated_combine=False``/``gated_update=False`` callers' RNG consumption
(and hence every existing test/script's reproducibility) is untouched --
same concern already flagged for ``spectral_norm_target``'s own
probe-vector RNG draws (see
``toy_tile_precision_models.spectral_norm_target`` below).

.. _toy_tile_precision_models.input_proj_column_averaging_misapplication:

``input_proj``: a real learned layer, not tiling -- a corrected misapplication of column-averaging
-----------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.input_proj_column_averaging_misapplication``

Input projection (``embed_width -> state_width``) is a REAL trained layer,
not the ``np.repeat`` tiling this class used to receive its window through.
Direct correction (see conversation): column-averaging's actual purpose is
letting a narrow OUTPUT's gradient reach the entire wide state on readout
(mean-pool down, so ``d(mean)/d(each element) = 1/column_neurons`` spreads
credit to every column) -- applying that same repeat/average pairing to the
INPUT side too was a misapplication of the same operator to a problem it
was never meant to solve, not an intentional design. The wide state's
actual content for a fresh token must come from a real learned mapping,
same as q/k/v/o_proj/lm_head.

.. _toy_tile_precision_models.use_attention_bypass_ablation:

``use_attention=False``: bypassing attention entirely, an ablation to isolate gaussian_attention
--------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.use_attention_bypass_ablation``

Bypasses ``q``/``k``/``v``/``gaussian_attention`` (and ``energy``, which only
ever gated the attention output) entirely -- collapses the recurrence into a
plain RNN cell, ``state = clip(rmsnorm(state + o_proj(rmsnorm(x)+
rmsnorm(state))))``. Ablation to isolate whether ``gaussian_attention``
itself is what's hard to learn, before assuming the whole architecture is
broken.

.. _toy_tile_precision_models.o_proj_depth_cascaded_quantization:

``o_proj_depth>1``: cascaded coarse layers as a substitute for width
----------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.o_proj_depth_cascaded_quantization``

Replaces the single ``o_proj`` with ``o_proj_depth`` ``disldo_cls``
sublayers applied in sequence (each ``state_width -> state_width``, no
nonlinearity between them), each given ``max_weights // o_proj_depth`` so
the total weight budget stays roughly comparable to depth=1 -- a
residual/cascaded-quantization-style test of whether N sequential coarse
(e.g. FP4) layers can compose into something closer to a single
higher-precision layer, rather than needing more WIDTH.

.. _toy_tile_precision_models.magnitude_penalty_coef:

``magnitude_penalty_coef``: a real gradient against unbounded recurrent magnitude
---------------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.magnitude_penalty_coef``

``magnitude_penalty_coef>0`` adds ``coef*mean(x**2)`` aux-loss terms on the
(post-clip) ``attn_o_proj``/state values -- a real gradient discouraging
large recurrent activation magnitude, independent of and in addition to the
hard clip. Motivated directly: the hard clip is a straight-through
``.data`` overwrite (bypasses autograd entirely), so nothing currently
tells the network NOT to keep driving activation magnitude up against it --
a pathological attractor under dense connectivity specifically (found via
JOURNAL.md 2026-08-10's dense-vs-sparse investigation: even with
NaN-safety fixed, dense connectivity still collapses to chance over a full
run, correlated with frequent non-finite-gradient skips concentrated right
at harder curriculum stages).

Computed from the ALREADY-CLIPPED value (not the pre-clip one)
deliberately: ``power``'s backward reads ``.data`` lazily at
backward-call time, so building it from the pre-clip Tensor before mutating
``.data`` in place would silently differentiate against the wrong
(already-overwritten) value by the time backward actually runs; using the
post-clip value instead is self-consistent (nothing mutates it again after)
AND gives a gradient magnitude that's itself bounded by ``clip_range``
(can't blow up for extreme pre-clip values), rather than reintroducing the
same kind of unbounded-magnitude risk this is meant to fix.

Deliberately independent of ``use_energy``/``EnergyDynamics`` (not
combined, not gated on it) -- direct instruction to keep this mechanism
isolated for testing rather than compounding it with energy_rl's own
extinguishing pressure, which "adds a lot right now" on its own and would
confound an isolated test of this.

.. _toy_tile_precision_models.spectral_norm_target:

``spectral_norm_target``: power-iteration rescale, the root-cause fix for dense instability
---------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.spectral_norm_target``

If set, rescales ``o_proj``'s real output by ``target/sigma_ema`` every
step, where ``sigma_ema`` is an EMA-smoothed power-iteration estimate of
``o_proj``'s dominant singular value -- the actual root-cause fix for dense
connectivity's instability (JOURNAL.md 2026-08-11: measured spectral
radius 1.19 at init, 1.50 after 400 steps for dense vs 0.85/0.83 flat for
sparse -- a spectral-radius-above-1 recurrent map structurally amplifies
signal every pass, independent of and NOT fixed by
``magnitude_penalty_coef``/``energy_rl``, which only constrain average
magnitude, not the weight matrix's dominant eigenvalue specifically).

Standard "Spectral Normalization" technique (Miyato et al. 2018): a
persistent probe vector ``u`` (NOT the actual recurrent state --
deliberately decoupled from the real data/RMSNorm/clip/residual path so
the estimate reflects ``o_proj`` alone) is updated via ONE power-iteration
step per real step (``layer.forward(u, 0.0)`` -- forward-only, same
zero-side-effect convention ``evaluate()`` already uses, no
backward/optimizer call needed at all, cheap: one forward pass plus a numpy
norm, nothing like the O(n^3) cost of an exact eigendecomposition).
``sigma_ema`` (not the raw per-step estimate) is what's actually used,
smoothed at ``spectral_norm_ema_decay`` -- important once synaptogenesis is
active: a structural change (new synapse) can make one step's raw estimate
jump before ``u`` re-converges to the new dominant eigenvector, and unlike
an init-time-only fix, this whole mechanism re-tracks automatically as the
weight matrix changes, whether from ordinary gradient updates or future
synaptogenesis. The rescale itself is an ordinary Tensor*float multiply
already supported by autograd (``out * (target/sigma_ema)``) -- no new
differentiable primitive needed anywhere. ``None``/off by default,
independent of ``magnitude_penalty_coef``/``use_energy`` (composable, not
mutually exclusive -- direct request to test combinations).

**Warm-start**: a single untrained random probe vector badly underestimates
the true dominant singular value (power iteration hasn't converged yet), so
the FIRST real-step rescale would overshoot the target -- confirmed
directly (measured effective spectral radius 1.38 at step 0 vs the 0.9
target, JOURNAL.md 2026-08-11). 20 extra iterations at construction time
(cheap, O(n^2) each, no backward/optimizer call) converge ``u``/
``sigma_ema`` BEFORE any real training step, matching ``EnergyDynamics``'
own "allow N steps for noise, don't wait forever" warm-start convention.
One persistent probe vector + EMA sigma PER ``o_proj`` sublayer
(``o_proj_depth>1`` chains several square ``state_width->state_width``
layers, each needs its own independent estimate). Only allocated/
RNG-consuming when actually used -- must NOT perturb ``rng``'s consumption
sequence for the default (every existing arm/test)
``spectral_norm_target=None`` path.

.. _toy_tile_precision_models.spectral_rescale_factor_radius_vs_norm_correction:

``_spectral_rescale_factor``: forward power iteration converges to spectral RADIUS, not NORM
------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.spectral_rescale_factor_radius_vs_norm_correction``

One power-iteration step against ``layer`` alone (NOT the real data -- a
persistent probe vector, decoupled from RMSNorm/clip/residual).
``forward(..., 0.0)``: zero-side-effect convention already used everywhere
(``evaluate()``, the earlier spectral-radius diagnostic) -- no
backward/optimizer call, so this costs one extra forward pass, nothing
like an eigendecomposition. EMA-smoothed sigma (not the raw per-step
estimate) is what's actually used -- see
``toy_tile_precision_models.spectral_norm_target`` for why
(synaptogenesis-readiness).

**Correction** (this method, including its own variable/parameter names
like ``spectral_norm_target``, was previously mislabeled "dominant singular
value" everywhere): this is forward-only iteration (``u_{k+1} =
layer(u_k)/||layer(u_k)||``), which only converges to the top SINGULAR
value when the underlying linear map is symmetric. For a real weight
matrix (generically NOT symmetric, and its dominant eigenvalue is
generically a complex pair, not real), this instead approximates something
close to the SPECTRAL RADIUS (max |eigenvalue|), not the spectral norm --
confirmed directly:

.. code-block:: python

   # Random 16x16 Gaussian matrix, comparing three quantities:
   #   this iteration's convergence point:      ~3.43
   #   true top singular value (np.linalg.svd): ~7.13
   #   true spectral radius (np.linalg.eigvals): ~3.49
   # Clearly tracking the latter (radius), not the former (norm).

Left as-is (this correction is comment-only, no behavior change) since
spectral radius is actually the theoretically correct quantity for
recurrent-dynamics stability anyway (spectral norm is a conservative upper
bound on it) -- but any existing ``spectral_norm_target`` tuning should be
understood as having tuned against radius-like behavior, not norm, this
whole time. See ``model/eval_eigenvalues.py`` for an EXACT (non-iterative,
SVD/eigval-based) alternative if a precise answer is ever needed instead of
this cheap per-step approximation.

.. _toy_tile_precision_models.l1_sparsity_coef_landmark:

``l1_sparsity_coef``: the landmark dense-connectivity stability mechanism
--------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.l1_sparsity_coef_landmark``

The LANDMARK dense-connectivity stability mechanism (see
``scripts/l1_sparsity_probe.py``'s own header docstring and sili_peridot
JOURNAL.md's 2026-08-13 entry) -- ported here from that probe script's
standalone ``OriginalArchModel`` reproduction, which is architecturally
identical to this class (single ``v_proj`` on the combined ``qkv_source``,
single ``o_proj``) and had NOT previously been merged into this shared
model.

Found to reach mean=1.0000 across 5 seeds at coef=0.05 AND 0.07 on the
15000-step out-of-context curriculum, ``dense_base12`` -- the single best
stability result of the entire investigation, beating
``spectral_norm_target``'s own 0.8858 (a hard rescale, since made
unavailable as a production mechanism) with NO hard rescale of any kind.
Do NOT combine with ``spectral_norm_target`` or ``magnitude_penalty_coef``
-- combining L1 with an L2-ratio-shaped mechanism was tested and HURTS
(0.7333 vs 1.0000), and spectral is off the table anyway; this is meant to
be used ALONE.

Applied via the same "split-backward" delivery as the probe: a SECOND,
independent ``forward(..., damp_by_importance=False)`` call per layer gives
the L1 term its own undamped gradient path, avoiding dilution by DISLDO's
own RMSprop-style per-synapse update (~1731x magnitude ratio otherwise, see
JOURNAL.md) -- this is what ``_l1_sparsity_split`` implements, an exact
port of the probe script's own helper of the same name.

Applied to ``input_proj`` (on the raw narrow ``x_window``), ``q_proj``/
``k_proj``/``v_proj`` (on ``qkv_source``), ``o_proj`` (on its own real
input -- the post-attention/post-energy tensor when ``use_attention=True``,
else ``qkv_source`` directly), and ``lm_head`` (on ``pooled``) -- all 6 real
weight layers (5 at the time this was first validated, before
``input_proj`` existed; treat its own L1 coverage as a direct, analogous
extension, not independently re-validated), matching the probe's own final
"lm_head previously had no L1 term" fix (task #176: under
``all_zero_init``, ``dL/d(pooled)`` is exactly 0 whenever ``W_lmhead=0``,
so ``lm_head`` needs this same direct escape route -- this is why the L1
term is applied directly at the ``logits = self.lm_head.forward(pooled,
...)`` call site in ``step()`` rather than relying on upstream layers'
terms to reach it).

``o_proj_depth>1`` is NOT covered by the validated result (the probe never
tested it) -- if set, L1 is applied per-sublayer using each sublayer's own
real input, a direct analogous extension, but treat that combination as
unvalidated until tested. ``gated_combine``/``gated_update``'s own new
layers (``gate_x_proj``/``gate_m_proj``, ``update_forget_proj``/
``update_input_proj``) get the same treatment for the same reason -- dense
connectivity is documented (JOURNAL.md 2026-08-13) to destabilize without
L1 on every real weight layer, and leaving new dense layers uncovered would
be a foreseeable regression of exactly that already-fixed failure mode.
Unvalidated as a combination until tested.

.. _toy_tile_precision_models.cosine_lm_head:

``cosine_lm_head``: fixing a magnitude/direction conflation in the readout, and a call-ordering bug
-------------------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.cosine_lm_head``

Makes ``lm_head``'s readout a "cosine classifier" (row-normalized matched
filter) instead of a raw dot product. Root cause found directly (see
conversation): with a fixed, untrained, random ``embed_table`` (e.g.
VOCAB=128, EMBED_WIDTH=16 in the MQAR K-sweep), a raw-dot-product readout
genuinely misclassifies ~2-5% of possible value tokens even with an OPTIMAL
hand-built ``lm_head``, because raw ``logit[v] = ||pooled|| * ||W_row[v]||
* cos_sim(pooled, W_row[v])`` conflates direction similarity with
``W_row[v]``'s own magnitude -- an unrelated token whose row happens to
have a larger norm can outscore the true match even at lower cosine
similarity. RMSNorm already fixes this on the INPUT side (``x_normed``/
``M_new`` have fixed magnitude regardless of which token arrived, so
``pooled``'s own scale never affects the argmax -- confirmed numerically,
normalizing ``pooled`` alone changes nothing), so the fix only needs to
touch ``lm_head``'s OUTPUT-row norms. This is NOT a fundamental
embedding-width/dimension-counting ceiling (an earlier hypothesis, WRONG,
retracted after direct numerical verification with a properly
row-normalized template reached exactly 64/64 at the SAME
``embed_width=16``) -- it's purely this one magnitude/direction
conflation, fixable without touching ``embed_width``, ``column_neurons``,
or the recurrent state at all.

Since ``lm_head`` is a real ``disldo_cls`` layer (its weights live in
FP4-quantized C++ storage, not a plain accessible matrix), row norms are
reconstructed the same way ``_spectral_rescale_factor`` already probes a
layer's weights elsewhere in this file: feed the ``embed_width`` standard
basis vectors through ``lm_head.forward(..., 0.0)`` (one batched,
zero-side-effect, no-backward call -- ``forward(e_i) = W_row[:, i]``, i.e.
column i of the effective weight matrix; stacking all E columns gives
every row's norm in one call) and divide the real logits by
``(row_norm + eps)`` before the loss ever sees them. Recomputed fresh
every step (no EMA smoothing, unlike ``spectral_norm_target``'s
``sigma_ema``) -- ``lm_head``'s weights update every step via its own
inline backward, so a stale norm would drift; the extra forward call is
cheap (E rows, not O(vocab)). Default False (opt-in, unvalidated against a
full multi-seed sweep yet -- confirmed only via the standalone numeric
witness in conversation).

**Call-ordering bug**: the probe forward MUST run before any other
``lm_head.forward()`` call in the same step (including the L1 split, and
before the real logits forward) -- ``disldo_cls`` layers cache their input
on the C++ side between ``forward()``/``backward()`` (a single
most-recent-call slot, not scoped per Tensor node), so a probe call
sandwiched AFTER the real forward would clobber that cached state with the
wrong (probe) batch shape before the real backward ever runs -- confirmed
directly (a ``(4,16) vs (16,16)`` broadcast crash in the real backward
pass) when this was first tried in the other order. Placing the probe
first guarantees the real ``pooled``-shaped forward call is always the
LAST ``lm_head.forward()`` before backward.

.. _toy_tile_precision_models.gated_combine:

``gated_combine``/``gate_floor``: a learned gate replacing the plain input+state sum
--------------------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.gated_combine``

Replaces the plain ``qkv_source = x_normed + m_normed`` sum with a LEARNED,
content-dependent gate. Root cause found directly (see conversation): every
real, working segment/block-recurrent transformer this project's design
was checked against (RMT, Block-Recurrent Transformer, Infini-attention)
combines fresh input and carried state either as separate attention-visible
tokens or through a LEARNED gate -- none of them use an untrained plain
elementwise sum. A plain sum forces two different signals into
superposition in the same channels before the network has any mechanism
(attention weights, a gate) to tell them apart, and the state update itself
(residual add + RMSNorm + hard clip) has no learned forget mechanism at all
-- architecturally the pre-LSTM "vanilla RNN" pattern gating was invented
to fix (see ``toy_tile_precision_models.known_differences_from_proven_designs``
item 2, and ``toy_tile_precision_models.gated_update`` below for the
matching fix on the update side).

Two new real ``disldo_cls`` layers, ``gate_x_proj``/``gate_m_proj``
(``state_width -> state_width`` each), computed from ``x_normed``/
``m_normed`` SEPARATELY and summed (``gate_x_proj(x_normed) +
gate_m_proj(m_normed)``) -- mathematically identical to a single
``state_width*2 -> state_width`` layer over the concatenation, but avoids
needing a Tensor concat op. Gate = ``sigmoid(that sum)``, matching
Infini-attention's learned gating scalar in spirit, but per-channel and
INPUT-DEPENDENT (a function of the actual ``x_normed``/``m_normed`` content
each step, not a single static learned scalar) -- closer to a real
LSTM/GRU-style gate, since the task fundamentally needs content-dependent
decisions ("keep old state when nothing new happened, overwrite when
something did"), which a static gate can't express.

``qkv_source = gate*x_normed + (1-gate)*m_normed``.

``gate_floor``: per direct instruction, the gate must NOT be able to reach
full-input-only (gate=1) or full-state-only (gate=0) -- both are real
failure modes (total forgetting every step, or the state going permanently
deaf to new input) -- so the raw sigmoid output is rescaled into
``[gate_floor, 1-gate_floor]`` rather than used directly on ``(0, 1)``.
Default 0.1: even at full saturation, each stream always keeps >=10%
weight. ``gated_update`` uses the same clamping convention.

Real trained comparison result: K=1 MQAR, single seed, 20k steps -- did
not yet beat the plain-sum baseline (0.133 vs 0.250), though loss/
trajectory were the smoothest of any arm tried -- inconclusive on n=1,
multi-seed re-test pending.

.. _toy_tile_precision_models.gated_update:

``gated_update``: a learned forget gate on the state update itself
----------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.gated_update``

Distinct from ``gated_combine`` -- see
``toy_tile_precision_models.known_differences_from_proven_designs`` item 2.
Replaces the plain residual ``M_new = M_prev + attn_o_proj_output`` with a
learned forget gate (``M_new = forget_gate*M_prev + (1-forget_gate)*
attn_o_proj_output``), the step Block-Recurrent Transformer's actual
LSTM-style gates control. Same ``gate_floor`` clamping, two-summed-
projections construction (``update_forget_proj``/``update_input_proj``),
and L1 coverage convention as ``gated_combine`` (see
``toy_tile_precision_models.gated_combine`` and
``toy_tile_precision_models.l1_sparsity_coef_landmark``). Composable with
``gated_combine`` (untested combination until both are validated
independently).

.. _toy_tile_precision_models.step_debug_stats:

``step(debug=True)``: per-stage value statistics for dense-vs-sparse divergence diagnosis
-------------------------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.step_debug_stats``

Records per-stage value statistics (mean/std/min/max, plus the fraction of
elements the hard clip actually touches) into
``self._last_step_debug_stats`` (a dict, stage name -> stats dict) -- does
NOT change the return signature (every existing caller unpacks a 3-tuple
positionally), so this is purely additive/inspectable after the call.
Default False (was previously an unused, dead ``debug: bool = True``
param -- this is the first real implementation) to avoid the extra numpy
reduction overhead in the normal training hot path. Built to diagnose why
fully-dense connectivity fails to train at all (JOURNAL.md 2026-08-10)
while the usual sparse echo-network succeeds -- compare
``_last_step_debug_stats`` across a dense and a sparse model at the same
seed/step to find exactly where their value distributions diverge.

.. _toy_tile_precision_models.forward_clip_necessity:

Forward clip on the residual update: gradient clipping alone was NOT enough
------------------------------------------------------------------------------

*ID:* ``toy_tile_precision_models.forward_clip_necessity``

The forward clip on the residual UPDATE itself (``attn``, before it's added
to state), not just the final state, was found NECESSARY, not just
belt-and-suspenders: gradient clipping alone (``clip_grad_norm_`` on the
plain-Tensor Adam params, ``train_tile_curriculum.py``) only bounds the
SIZE of each individual step, not the CUMULATIVE drift from many small
unclipped-in-effect steps compounding in the same direction over hundreds
of steps -- confirmed directly: with only gradient clipping, dense
connectivity's NaN divergence moved from step ~275 to step ~450 but still
happened. ``attn_o_proj`` was already measured reaching |9.47| BEFORE the
state's own post-residual-and-norm clip ever saw it. Same straight-through
bypass-autograd convention as the state clip (this is about bounding
FORWARD magnitude, not shaping the backward gradient -- that's
``clip_grad_norm_``'s job).
