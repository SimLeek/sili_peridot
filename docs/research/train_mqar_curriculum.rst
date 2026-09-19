``train_mqar_curriculum.py`` research notes
==================================================

Companion doc to ``scripts/train_mqar_curriculum.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below); this
doc links back to source by function/constant name. See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets alone render invisible on GitHub, frozen code
snippets on real-bug sections only).

.. _train_curriculum.student_paced_curriculum_design:

Module overview: student-paced difficulty, not a step-count schedule
-------------------------------------------------------------------------

*ID:* ``train_curriculum.student_paced_curriculum_design``

Adaptive MQAR curriculum (task #264): grows ``vocab_size`` first (K=1
fixed), then grows ``num_kv_pairs`` (K) once vocab reaches
``TASK_VOCAB_MAX``. A stage advances only after ``STREAK_THRESHOLD``
consecutive correct query predictions during live training, and REGRESSES
one stage after ``WRONG_STREAK_THRESHOLD`` consecutive wrong predictions --
"if a student isn't learning you go over the specific things they're not
doing good on" (direct instruction) -- not a step-count schedule, and not a
one-way ratchet either.

**Level-change signal** (direct instruction): two reserved tokens,
``LEVEL_UP_TOKEN``/``LEVEL_DOWN_TOKEN``, are fed as INPUT ONLY (never a
prediction target) at the start of the next sequence whenever a stage
transition happens, so the model can perceive that the rules just changed
instead of a stage transition being invisible in the token stream.
Implemented by prepending the token to that one sequence and shifting every
position by +1 -- structurally identical to how an ordinary filler token
flows through ``_build_tile_window``, just with a fixed, meaningful
identity instead of random noise, and no target attached to it.

**NUM_TILES** is a MODEL parameter (the local sliding-attention window),
deliberately DECOUPLED from the task's ``seq_len``/K (direct instruction --
the older ``train_mqar_precision_sweep.py`` sweep's ``num_tiles=seq_len``
coupling was flagged as wrong). Stays FIXED for the whole run. Default 16:
covers ``seq_len_for_k(k)`` for k=1..4 (max key-query distance 15) entirely
in-window, so the model gets a real base of in-context-solvable K stages
before K=5 (seq_len=20, max distance 19) forces the first genuine reliance
on the RMT memory tokens -- "K=4 would be a good amount of examples to
learn the pattern in context before requiring memory" (direct instruction).
The in-context -> memory-required transition is left to happen naturally
within one continuous curriculum run, not run as two separate experiments.

.. _train_curriculum.cli_gradient_sparsity_args:

CLI args: ``dy_r_target``/``dy_surprise_alpha``/``x_r_target``/``trajectory_log_every``
-----------------------------------------------------------------------------------------------

*ID:* ``train_curriculum.cli_gradient_sparsity_args``

``dy_r_target``: nucleus/energy-threshold captured-energy-ratio target
(0..1) for the wide layers' backward gradient (task #367) -- TAKES
PRIORITY over ``dy_sparsity_p`` when both are set. ``k`` is a CONSEQUENCE
of ``dy_r_target`` and each step's actual gradient energy, not a fixed
fraction. -1 (default, this script's CLI "unset" sentinel throughout) means
unset. See JOURNAL.md's "nucleus/energy-threshold top-k math" design note.
``dy_k_min``/``dy_k_max`` clamp the derived ``k`` afterward as a hardware
density floor/ceiling.

``target_steps_per_sec``: ARMS the closed-loop controller that adjusts
``dy_r_target`` every ``log_every`` steps against MEASURED steps/sec (task
#368) -- NOT an analytic formula from an assumed compute-cost ratio
(measured directly and found non-constant across widths, see JOURNAL.md's
"Grad-side k_t design, revised" entry). Unset: ``dy_r_target`` stays fixed
at its initial value for the whole run.

``dy_surprise_alpha`` (task #374): per-layer INNER loop -- breathes each
wide layer's own EFFECTIVE ``dy_r_target`` above/below its ``r_bar`` based
on that layer's own lagged gradient energy (``E_t=||dy||^2`` vs its running
EMA ``Lbar``, ``beta=0.99`` fixed, not CLI-exposed). Unset: mechanism off,
``r_bar`` used unmodified. See ``ToyTileRecurrenceRMT.__init__``'s own
``dy_surprise_alpha`` docstring for the full formula.

``x_r_target``/``x_k_min`` (task #365): INPUT-side mirror of
``dy_r_target``/``dy_k_min``. When ``target_steps_per_sec`` is ALSO set,
``x_r_target``'s periodic adjustment goes through
``apply_cross_layer_budget_allocator`` (task #375) instead of a second
independent speed-reactive loop -- grad stays need-driven via the surprise
signal above, independent of speed; only ``x_r_target`` reacts to the
speed budget, weighted by each layer's own real measured cost share
(``_layer_timing``, task #373) instead of a uniform correction across
layers.

``trajectory_log_every`` (task #369): fine-grained per-layer/both-axes
logging -- direct instruction: bounded/opt-in, OFF by default, separate
cadence from the coarse ``log_every`` ("I want to see a more fine-detail
log... but by default that's quite a lot of data"). When set, prints real
R/k_t (actual captured-energy ratio and mean kept-entries, not just the
r_target asked for) for the INPUT axis alongside ``dy_r_target``/E_t/Lbar
for the GRAD axis, every N steps. ``trajectory_log_steps`` (a step range)
is Python-kwarg-only, not CLI-exposed -- use ``train_curriculum()`` directly
for that.

.. _train_curriculum.fp8_max_abs_delta_scale_space_bug:

``NOCAPS_KWARGS_FP8``: why FP8 needs a real (non-infinite) ``max_abs_delta``
----------------------------------------------------------------------------------

*ID:* ``train_curriculum.fp8_max_abs_delta_scale_space_bug``

FP4's block4 backward computes ``cw`` in CODE-SPACE with ``S`` properly
threaded through ``SynapsePolicy::update_cw`` (sili__new's
``linear_disldo.hpp``), so its own quantization ceiling (raw code magnitude
<=6) gives it an IMPLICIT per-step safety margin even under an uncapped
``max_abs_delta``. FP8's block4 backward used to compute ``cw`` in
TRUE-WEIGHT space with ``S`` hardcoded to 1 -- a real, separate bug (fixed:
an S-independent RMSprop step divided by a small ``output_scale`` at write
time amplified every update by ``~1/S``, up to ~287x for a wide
fan-in-corrected layer). Once fixed to match FP4's convention, FP8's
REMAINING difference from FP4 is purely its much wider code range (E4M3 max
~448 vs FP4's ~6) -- an uncapped cold-start delta that FP4's narrow range
self-limits harmlessly can still drift FP8 noticeably before its own (much
later) natural ceiling kicks in.

Confirmed directly: FP8 is fully stable (40-step diagnostic, loss steady
~4.2-5.1) under the library's own tuned production default
(``kSynapsePolicyMaxAbsDelta=2.0``, ``sili/cpu_backend.cpp``), so reuse that
exact value here rather than guessing a new one -- gives FP8 the same kind
of per-step code-space ceiling FP4 gets for free, while FP4/FP32 stay on
the deliberately uncapped ``NOCAPS_KWARGS`` unchanged.

.. _train_curriculum.fp32_unbounded_weight_blowup:

``NOCAPS_KWARGS_FP32``: fp32 has no implicit ceiling, and a real blowup this fixes
-----------------------------------------------------------------------------------------

*ID:* ``train_curriculum.fp32_unbounded_weight_blowup``

fp32 does NOT get the same free pass as FP4/FP8: FP4/FP8's raw stored value
is a quantization CODE with a small fixed table ceiling (FP4 ~6, FP8 E4M3
~448), which gives an implicit per-step safety margin even under an
uncapped ``max_abs_delta``. fp32's raw stored value is a plain, genuinely
unbounded float -- there is no implicit ceiling at all.

Confirmed the hard way: a wide (3x embed_width), 10%-input-sparse/
10%-grad-sparse fp32 MQAR run under ``NOCAPS_KWARGS`` blew up
``input_proj``/``v_proj`` to ``~1e14-1e17`` within 16k steps
(overflow/invalid RuntimeWarnings, near-chance accuracy) while
``q``/``k``/``o_proj``/``lm_head`` -- same code, same NOCAPS -- stayed
healthy. ``NOCAPS_KWARGS_FP32`` (``max_abs_delta=2.0``, ``max_ci=100.0``,
reusing the library's own already-tuned production default rather than
guessing new numbers, matching what every other precision already runs
under) is the simple hard-bound half of the fix (immediate safety net); the
amortized L2 decay mechanism (``apply_amortized_l2_decay``, see
``train_curriculum.l2_decay_and_rank_control_ordering`` below) is the
complementary ongoing-health half.

.. _train_curriculum.min_queries_before_regress_thrashing:

``MIN_QUERIES_BEFORE_REGRESS``: a real thrashing bug, and its fix
------------------------------------------------------------------------

*ID:* ``train_curriculum.min_queries_before_regress_thrashing``

Grace period: a fresh stage gets ``MIN_QUERIES_BEFORE_REGRESS`` (30) query
attempts before regression can trigger at all. Without this, ordinary
first-contact difficulty on a harder stage (expected, not "isn't learning")
was hitting ``WRONG_STREAK_THRESHOLD`` almost immediately and thrashing
level_up/level_down every ~10-15 steps -- confirmed directly (a smoke test
oscillated vocab 16<->32 repeatedly).

.. _train_curriculum.advantage_actor_critic_design:

Advantage actor-critic backward: exact regression target, not TD/bootstrap
---------------------------------------------------------------------------------

*ID:* ``train_curriculum.advantage_actor_critic_design``

Task #272. The critic predicts the per-vocab-neuron squared error the
actor's own logits will incur; since the true target token is known the
same step, its regression target (``(softmax(logits) - onehot)**2``) is
EXACT, not estimated -- no TD/bootstrap/target-net needed, unlike
sili__new's mandelbrot RTAC, which needs those because its reward isn't
known until later. ``advantage = true - predicted`` then reweights the
actor's own cross-entropy gradient per vocab neuron (bigger correction
where the critic was most surprised).

``ADVANTAGE_CLIP`` (5.0) keeps the ``(1+advantage)`` multiplier bounded --
``true_loss_vec`` entries are bounded in [0,1], but a diverging critic
prediction early in training could otherwise blow this up unguarded.

``np.nan_to_num`` runs FIRST on the critic's raw prediction, before the
``np.clip``: ``np.clip`` does NOT sanitize NaN (``clip(nan, lo, hi) ==
nan``), so an unguarded critic divergence would flow straight through into
``g_logits`` and corrupt the whole model -- found via a real 45k-step fp8
run that NaN-collapsed by step 3400 (far earlier than any prior curriculum
run), matching this codebase's own established pattern (see
``_overflow_guard_array`` in sili__new's ``sparse_rnn.py``). Treating a
non-finite prediction as "no correction" (``advantage=0``) rather than
propagating it also protects the critic's OWN gradient, since ``g_critic``
reuses this same sanitized ``pred_row``.

.. code-block:: python

   pred_row = np.nan_to_num(
       np.asarray(critic_pred.data[row], dtype=np.float32),
       nan=0.0, posinf=ADVANTAGE_CLIP, neginf=-ADVANTAGE_CLIP,
   )
   advantage = np.clip(true_loss_vec - pred_row, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

.. _train_curriculum.reward_punish_asymmetry_removed:

Reward/punish asymmetry: REMOVED after a real 3-arm ablation
-------------------------------------------------------------------

*ID:* ``train_curriculum.reward_punish_asymmetry_removed``

Task #302/#306, direct instruction, after a real 3-arm ablation isolated
this as the cause of a genuine training regression, distinct from the NaN
investigation. Scaling every wrong-class neuron's gradient by a uniform
``punish_scale < 1`` introduces a systematic, one-directional bias into the
aggregate gradient flowing into the shared trunk -- the one-hot structure
means there's exactly ONE full-strength reward term per query but
``(vocab_size-1)`` DAMPENED punishment terms, so the two no longer cancel
in aggregate the way softmax's own math guarantees they do when both sides
are unscaled.

This isn't a per-step sign flip visible in the formula -- it's a slow,
compounding drift, confirmed directly: a 3-arm ablation (same seed) held
BOTH "neither mechanism" and "magnitude penalty only" stable (loss bounded
~3-30 through 3000 steps) while "reward/punish asymmetry only" diverged
smoothly from ~5 to ~4400 over the same window -- the gradual ramp is the
signature of a compounding bias, not a numerical edge case.

Tried twice (``1/((vocab_size-1)*STREAK_THRESHOLD)`` and
``1/STREAK_THRESHOLD``) and failed differently each time; not reintroduced
without a redesign that doesn't rely on a single uniform per-wrong-neuron
scale.

.. _train_curriculum.nan_bisection_diagnostics:

Diagnostics: ``DEBUG_FINITE_CHECK``'s per-step bisection, and the magnitude trace
------------------------------------------------------------------------------------

*ID:* ``train_curriculum.nan_bisection_diagnostics``

Task #303. When ``DEBUG_FINITE_CHECK`` is True, ``_check_finite_or_raise``
checks ``logits.data``/``critic_pred.data`` for non-finite values EVERY
step (not just every ``log_every``), right after ``model.step()`` returns
-- i.e. checks the raw forward output BEFORE any of
``_backward_with_critic``'s own processing touches it. It bisects through
the forward chain (``model.last_debug``: ``x_wide`` -> q/k/v -> attn
(pre/post o_proj) -> raw_combined/pre_norm_combined/combined_new -> pooled)
to find the EARLIEST stage that's already non-finite, not just that the
final output is. On first failure, dumps a full per-layer weight health
scan (raw ``value_scale``/``output_scale``/``additive_u``/``additive_v``
vectors, via ``_layer_health``) and raises immediately, so the exact step
and the exact layer/branch that went non-finite first are both known --
rather than only knowing "somewhere before the next periodic log line."

**Magnitude trace** (direct instruction -- "why isn't there some extremely
strong loss before outputs get NaN, or are the outputs not large but the
in-between kernel values large instead"): ``_record_magnitude_trace``
records ``max(|x|)`` over every ``last_debug`` array at EVERY position (not
just the one that eventually fails), capped at ``_MAGNITUDE_TRACE_MAXLEN``
(40) so memory doesn't grow unbounded on a long run. On failure, the tail
of this is included in the report -- direct evidence of whether the
blow-up is a gradual ramp visible from the OUTSIDE (finite activations
climbing over many steps) or a one-position cliff (everything looks
normal, then the very next position is already fully NaN with no warning).
``sigmas`` specifically also records the min (not just max|x|): the failure
mode being chased is UNDERFLOW toward 0 (``1/(2*sigma**2) -> Inf`` inside
gaussian_attention), which max|x| alone would hide.

.. _train_curriculum.ema_grad_scale_per_tensor:

``_ema_grad_scale``: per-tensor gradient EMA scaling before the global clip
------------------------------------------------------------------------------

*ID:* ``train_curriculum.ema_grad_scale_per_tensor``

Direct instruction: purely a function of each tensor's OWN recent
gradient-norm history, no step count or wall-clock anywhere in the
formula -- required for a lifelong-learning setting where "step N" can't
mean anything special.

Root problem this targets (found via the ``sigma_grad_debug_fn``/
``probe_sigma_trajectory`` investigation): ``input_ln``/``memory_ln``/
``state_ln``/``centers``/``log_sigmas`` were orphaned by the old
``_to_sparse`` autograd bug, so at ``embed_width=32`` they all take their
first-ever real gradient step simultaneously once that bug is fixed -- a
genuine cold start (their Adam moment estimates have never seen a real
value) landing on a group that's structurally prone to hitting q/k/v/attn's
hard clip (zero backward gradient past the boundary --
``magnitude_clip_penalty_coef`` targets the same failure mode).
``clip_grad_norm_``'s single GLOBAL norm across this whole param group
means one exploding tensor (``log_sigmas``) drowns every other tensor's
real update to a near-zero share of the shared budget -- this runs BEFORE
that global clip and scales each tensor independently, so an outlier
tensor doesn't cannibalize the others' learning signal.

Each tensor tracks its own ``_grad_norm_ema`` attribute (state living on
the Tensor object itself, updated every call -- not derived from step
index). First real gradient for a tensor: no EMA exists yet, so it passes
through unscaled and simply seeds the EMA. From the second real gradient
onward: if the current raw norm exceeds ``ratio_threshold`` times that
tensor's own EMA, the gradient is scaled down to exactly
``ratio_threshold * ema`` before anything else touches it; the EMA itself
is always updated with the (possibly just-scaled) value, so a suppressed
outlier doesn't inflate the reference either.

.. _train_curriculum.stage_key_kcycle_lexicographic:

``_stage_key``: total order across three curriculum phases
-------------------------------------------------------------

*ID:* ``train_curriculum.stage_key_kcycle_lexicographic``

Total order matching how stages are actually visited: every k-phase stage
is harder than every vocab-phase stage. "kcycle" (``k_first_target``
odometer mode, direct instruction -- "v16k1 v16k2 v16k3 v18k1 v18k2 v18k3
v20k1... like it's an n-ary number that's incrementing or decrementing"):
vocab is the more significant digit, k the less significant one, so
ordering is plain lexicographic ``(vocab, k)`` -- tags its own leading
element (2) so it never compares as equal/lesser against the other two
phases' tags in a mixed context.

.. _train_curriculum.new_vocab_forced_sampling_gate:

New-vocab forced sampling: a gate on the level-up query, not a per-query tax
--------------------------------------------------------------------------------

*ID:* ``train_curriculum.new_vocab_forced_sampling_gate``

Direct instruction. Keys are otherwise drawn uniformly from the whole
current key range, so a just-grown vocab's newest token(s) can go untested
for many queries by pure chance, letting a level-up streak complete on old
vocab alone. ``new_key_ids`` holds the key ids introduced at the most
recent vocab level-up that still need at least one real test.

Corrected design (direct instruction): ``streak_has_new_vocab`` gates the
LEVEL-UP itself, not a continuous per-query tax on the whole level. It
tracks whether the CURRENT in-progress streak (the one building toward
``streak_threshold``) has already included a real test of the newly-grown
vocab, and only forces new vocab onto the query that would otherwise
complete the streak without ever having tested it -- "if
``(streak_threshold-1)/streak_threshold`` of a level-up is done and new
vocab hasn't shown up in the correct sequence, force it on THIS query" --
so a level-up can never fire having never touched the new material, without
repeatedly taxing every re-entry into the level. Membership is checked
per-query-position (each query's own key), not a step-wide OR, so
``streak_has_new_vocab`` tracks whether THIS SPECIFIC query touched the new
vocab, not just any query somewhere in the same step's window. No forcing
is needed on regression -- that's re-practicing a lower level whose vocab
was already tested on the way up.

.. _train_curriculum.level_prefix_persistent_indicator:

Persistent level-indicator prefix: reusing task-vocab tokens, not new ones
---------------------------------------------------------------------------------

*ID:* ``train_curriculum.level_prefix_persistent_indicator``

Direct instruction. Every sequence gets an explicit signal of (current
vocab, current k) via a 2-token prefix ``[vocab indicator, k indicator]``,
not just the one-shot ``LEVEL_UP_TOKEN``/``LEVEL_DOWN_TOKEN`` on the single
sequence right after a transition -- otherwise the model has to re-detect
both from raw content (counting distinct key-value pairs, inferring vocab
range) on every other sequence with zero hint.

BOTH indicators reuse the real task-vocab token space directly (no new
dedicated tokens, no vocab_size extension) -- direct instruction, "you can
use the vocab token as the k token". vocab indicator = the newest/highest
currently-unlocked token (``vocab_size-1``); k indicator = token id ``k``
itself (``k_indicator_token``, trivial passthrough -- ``k`` is always well
within ``[0, TASK_VOCAB_MAX)``, so this needs no new tokens). Position
alone (1st vs 2nd prefix slot) disambiguates either from the same token ID
appearing later as a genuine in-sequence value.

.. _train_curriculum.use_tile_cache_query_step_fallback:

``use_tile_cache``: falls back to full-window recompute on query steps
-----------------------------------------------------------------------------

*ID:* ``train_curriculum.use_tile_cache_query_step_fallback``

``requires_grad=False`` for non-query steps (direct instruction): the
sequential write-then-read design makes k_proj/v_proj/o_proj
``forward()``-run TWICE per step, and most steps never get a
``loss.backward()`` at all (only ``i in targets`` does) -- building a
backward graph for those is both wasted work and, more importantly, would
leave the C++ engine's ``DenseInputStack`` accumulating never-popped
entries across every non-query step in the sequence until it hits its cap
and throws.

``use_tile_cache`` query-step handling (direct instruction):
``step_cached()`` alone only credits the NEWEST content position per
backward call (older cached positions are detached, zero gradient) -- a
real, measured stability concern (see ``project_tile_window_kv_cache``
memory). On query steps specifically (rare, this is where weights actually
move), fall back to ``step()``'s full-window recompute instead, but with a
GRADED per-position ``dy_sparsity_schedule`` (denser for the newest
position, sparser further back, via ``_default_graded_dy_schedule``) rather
than either full uniform density (expensive) or ``step_cached``'s
zero-credit default. Non-query steps (the majority) still use
``step_cached()``'s fast single-position path -- no backward ever runs on
those regardless, so there's no signal being lost there either way.

Cache refresh: ``model.last_debug["k"/"v"]`` are the FULL
``[total_slots, sw]`` pre-pass-2 arrays built at the top of ``step()`` --
drop the oldest content row (about to age out of the window on the next
call) and keep the rest, same append/trim convention ``step_cached()``
itself uses.

.. _train_curriculum.k_first_target_odometer_reordering:

``k_first_target``: K-then-vocab curriculum order, and the odometer wraparound
-------------------------------------------------------------------------------------

*ID:* ``train_curriculum.k_first_target_odometer_reordering``

Direct instruction, new curriculum ordering. Reverses the default
vocab-then-K phase order to K-then-vocab. Rationale: growing vocab first
for tens of thousands of steps means every synapse's "importance" (this
project's real optimizer state) gets reinforced for K=1-only patterns long
before the model is ever asked to do multi-key discrimination, by which
point those synapses resist the new K>1 feature the same way a CNN that
never saw horizontal lines during training would struggle to grow that
detector later with mostly-set weights. Unset (default): today's exact
vocab-first behavior, zero change for every existing caller.
``k_first_vocab`` defaults to ``seq_len_for_k(k_first_target) + 4`` (same
"+4 buffer over the minimum" margin ``VOCAB_START`` already uses over
``seq_len_for_k(1)``) if not given explicitly.

Odometer mode ("kcycle", direct instruction -- "v16k1 v16k2 v16k3 v18k1
v18k2 v18k3 v20k1... like it's an n-ary number that's incrementing or
decrementing"): k is the least-significant digit, cycling
``K_START..k_first_target`` at a fixed vocab; once it completes a cycle,
vocab (the more significant digit) advances one step and k resets to
``K_START``. Exception: once vocab is already clamped at
``TASK_VOCAB_MAX`` (``next_vocab`` is a no-op there), wrapping k back to
``K_START`` would spin forever at the top vocab tier without ever
terminating -- so instead k keeps climbing PAST ``k_first_target``,
degenerating into the same "just grow k forever" terminal behavior the
default (vocab-first) run already has once ITS vocab phase is exhausted.
Graduation for this mode mirrors the default run's own terminal condition
("vocab already maxed out AND k has grown past its own cap").

**Use this mode by default (2026-09-11).** Leaving ``k_first_target``
unset runs k=1 for the entire vocab ramp -- tens of thousands of steps
with only ONE query-key pair per sequence, which risks training a
positional/relative-position shortcut instead of genuine key-value
binding, and entrenches synapse importance against the k>1 feature
before the model ever needs it (see this section's own rationale above).
Only leave ``k_first_target`` unset when the test specifically calls for
isolating vocab growth from k growth -- not as a default choice.

.. _train_curriculum.embed_table_builder_sdr_hook:

``embed_table_builder``: caller-controlled SDR embedding structure
-------------------------------------------------------------------------

*ID:* ``train_curriculum.embed_table_builder_sdr_hook``

Direct instruction, wide-model SDR redesign. ``None`` (default) preserves
today's exact dense-random-projection embedding, zero change for every
existing caller. When set, the caller controls the embedding's own
structure -- returns ``(embed_table, active_mask)``, ``active_mask`` a
same-shape bool array marking each token's FIXED sparse-active positions
(e.g. a genuine sparse-distributed-representation table with a fixed
k-of-width active set per token), or ``None`` if the embedding has no
fixed sparsity structure to preserve under learning.

.. _train_curriculum.embed_learning_rate_scatter_gradient:

``embed_learning_rate``: scatter-add gradient into the embedding table
-------------------------------------------------------------------------

*ID:* ``train_curriculum.embed_learning_rate_scatter_gradient``

Direct instruction, "add learning too" on top of the fixed SDR structure.
``last_debug``'s ``x_window_t`` Tensor is a leaf of THIS step's backward
graph, so its ``.grad`` (populated by the ``backward()`` call just above)
is ``dL/d(x_window)`` -- a plain scatter-add embedding-lookup gradient,
exactly like a real embedding layer's backward. Only wired for the
non-tile-cache ``step()`` path (``use_tile_cache``'s ``step_cached()``
builds its window differently and isn't covered here). ``active_mask``
(when set) restricts the update to each token's fixed active positions, so
learning adjusts MAGNITUDES at the SDR's chosen indices without ever
growing new nonzero positions outside it.

.. _train_curriculum.l2_decay_and_rank_control_ordering:

Per-step order: amortized L2 decay, then orthogonality, then overflow guard
--------------------------------------------------------------------------------

*ID:* ``train_curriculum.l2_decay_and_rank_control_ordering``

Amortized decoupled L2 decay (direct instruction) is the "ongoing health"
complement to ``NOCAPS_KWARGS``'s per-precision ``max_abs_delta``/``max_ci``
hard bound (see ``train_curriculum.fp32_unbounded_weight_blowup``) --
independent of ``dynamic_rank_control`` since it decays raw synapse
weights, not AQRS scale/additive channels, so it applies identically
whether or not rank can mutate. ``None`` (default): zero overhead,
unchanged behavior.

Within ``dynamic_rank_control``, ``apply_channel_orthogonality_penalty()``
(task #295 follow-up, chosen over residual-targeted growth per direct
instruction -- stops rank channels converging to duplicate directions
during training, which nothing else here catches: neurogenesis's own
health check is magnitude-only, ``l1_sparsity_coef`` only sees the summed
output) is called BEFORE ``apply_scale_overflow_guard()``, not after.
Found via a real fp8 run: the orthogonality pass's own correction can in
principle still be large, so the overflow guard must always run LAST as
the actual numerical-safety net, never have something unguarded applied on
top of it. The overflow guard itself only matters once rank can genuinely
exceed the old hardcoded cap=4 (i.e. only under ``dynamic_rank_control``)
-- ``get_scale()``'s combined envelope has no clamp, and rank growing past
4 let it overflow in a real fp8 curriculum run (see JOURNAL.md).

.. _train_curriculum.additive_rank_validated_aqrs_config:

CLI default ``additive_rank=1``: the validated AQRS config
-----------------------------------------------------------------

*ID:* ``train_curriculum.additive_rank_validated_aqrs_config``

``additive_rank=1`` + ``dynamic_rank_control=True`` is the validated AQRS
config (sili__new PR #38 / sili_peridot PR #16, JOURNAL.md): fp8 reached
final/peak vocab=126,k=2 with the channel-orthogonality fix, vs
peak_vocab=64 for the plain (``additive_rank=0``) arm. The rank cap itself
is NOT set here -- ``DISLDOLayer``'s own constructor already defaults
``scale_rank_max``/``additive_rank_max`` to
``max(1, min(n_in,n_out)//4)`` (sili__new ``sparse_rnn.py``
``_default_rank_cap``), which for these ``state_width=128`` square layers
is exactly 32, matching the validated run without needing an explicit
override.

``rank_grace_period_steps`` (default 50): AQRS rank-mutation cooldown
(task #292 fix) -- interim "age-gate" refractory period, not yet tied to a
real resource/energy cost model (see sili__new ``delta_csr_types.hpp``'s
``apply_dynamic_rank_control_generic`` docstring). A real 60k-step run
showed the default still allows frequent churn since 12 independent
per-branch cooldowns (6 layers x 2 branches) all reset on their own
schedule; raise this for a calmer run.

.. _train_curriculum.wrong_streak_threshold_reward_hacking:

``wrong_streak_threshold``: a real reward-hacking plateau at 7/10
------------------------------------------------------------------------

*ID:* ``train_curriculum.wrong_streak_threshold_reward_hacking``

Direct instruction, reward-hacking concern. The default
``WRONG_STREAK_THRESHOLD=5`` vs ``STREAK_THRESHOLD=10`` asymmetry means a
model can reach the (easier, lower-loss) previous vocab stage via only 5
consecutive wrong answers, but needs 10 consecutive right answers to leave
it. A real diagnostic run (``embed_width=32``, see JOURNAL.md) showed
``max_streak`` plateaued hard at 7/10 for 2000+ steps, never crossing to
level up -- consistent with the model finding it cheaper (lower average
loss) to hover near the boundary and get bounced back to easy data than to
commit to genuinely harder representations.

.. _train_curriculum.dy_r_target_closed_loop_reset_ordering:

Closed-loop ``dy_r_target``/``x_r_target`` control: reset timing after each window
--------------------------------------------------------------------------------------

*ID:* ``train_curriculum.dy_r_target_closed_loop_reset_ordering``

Task #368/#375. ``apply_amortized_dy_r_target_control`` only fires when
``dy_r_target`` was actually enabled (no-op otherwise, matching
``apply_amortized_l2_decay``'s own "safe to always call" convention).
Reuses this loop's own cumulative ``steps_per_sec`` measurement rather than
a separate windowed timer. ``x_r_target`` goes through
``apply_cross_layer_budget_allocator`` instead (see
``train_curriculum.cli_gradient_sparsity_args`` above for why grad and
input must not both react to the same speed signal independently) -- also
a no-op when ``x_r_target`` was never enabled.

``model.reset_layer_timing()`` is called AFTER using this window's timing
data -- next window's per-layer weight should reflect only steps since
THIS checkpoint (``reset_layer_timing``'s own docstring convention), not a
slow cumulative drift across the whole run.

.. _train_curriculum.query_debug_fn_explainable_ai_hook:

``query_debug_fn``: a per-query probe hook for the explainable-AI investigation
---------------------------------------------------------------------------------------

*ID:* ``train_curriculum.query_debug_fn_explainable_ai_hook``

Direct instruction, explainable-AI investigation. Optional callback fired
at EVERY query step (not just periodic log points or LEVEL_UP/DOWN
events) with ``(step, correct, logit_row, model.last_debug,
logits.data[logit_row], targets[i])`` -- ``last_debug`` already exposes
``attn_mem``/``attn_content``/``sigmas`` (task #303's NaN-bisection
instrumentation), letting a caller correlate attention sharpness/pattern
with actual per-query correctness without needing a separate hand-rolled
probe script that risks diverging from this loop's own validated
task-generation/labeling logic. ``None`` (default): zero overhead, no
behavior change for existing callers.

``logits.data[logit_row]``/``targets[i]`` (direct instruction) are passed
alongside the binary correct/incorrect outcome so a caller can compute
real confidence/hedging signals (target-token probability, entropy)
itself instead of only seeing whether the prediction was right.

.. _train_curriculum.width_scaling_lr_fanin_hypothesis:

``DEFAULT_PEAK_LR``: width-scaling LR/fan-in hypothesis (UNCONFIRMED)
-------------------------------------------------------------------------------

*ID:* ``train_curriculum.width_scaling_lr_fanin_hypothesis``

2026-09-18, dense-vs-sparse-mqar-300k investigation (full writeup in
``JOURNAL.md`` under this date; memory ``project_width_scaling_lr_fanin_law``).
A dense ``state_width=288`` run stalled at vocab=64/k=3 for 86,257 of
100,000 steps; ``peak_lr=0.01`` broke straight past that wall (vocab=126/k=3
by step 13,601), while scaling ``max_grad_norm`` instead did nothing
comparable. Isolates the cause to the fixed learning rate, not gradient
clipping.

**Rough law, one data point, not yet confirmed**::

    lr(N) ~= lr_base * (N_base / N) ^ alpha,   0.5 <= alpha <= 1.0

for dense hidden-to-hidden matrices (``q``/``k``/``v``/``o_proj``, each
``N x N``, ``N = state_width``, realized fan-in = fan-out = ``N``).
``alpha=1.0`` is muP's known hidden-layer prescription (rank-1 weight
update, coherent effect across all ``N`` inputs, needs ``lr ~ 1/N`` to stay
O(1)); ``alpha=0.5`` is naive per-element/NTK scaling and is the value that
happened to match the one tested point (``0.01`` vs. predicted ``0.0067``
at alpha=1.0). Do not treat either exponent as settled from a single run.

**Correction, 2026-09-18 (same day): NOT a monotonic "lower is always
better" relationship.** A third data point, ``peak_lr=0.0067`` (the
alpha=1.0 value), came back WORSE than the unscaled ``peak_lr=0.015``
baseline -- ``vocab=32/k=3`` (stuck) vs. unscaled's ``vocab=64/k=3``
(stuck), where ``peak_lr=0.01`` reached ``vocab=126/k=3`` and held it.
``0.01`` is a real sweet spot with both neighbors doing worse, not an
endpoint on a one-directional curve -- the
``lr(N) ~= lr_base*(N_base/N)^alpha`` power-law framing may not even be
the right functional form (a single exponent can't produce a U-shape).
Needs more points near ``0.01`` (e.g. ``0.008``, ``0.012``) before
trusting any specific value as "the" width-288 LR.

**Correction to an earlier draft of this hypothesis**: sparsity does NOT
automatically cap this. ``x_r_target``/``dy_r_target`` (this file's own
selection knobs) are *proportional* -- a fraction of the layer's width --
so realized fan-in under them still scales with ``N`` exactly like dense
does; only an *absolute*, width-independent per-neuron synapse budget
(``max_weights``-style, sized as a fixed count rather than ``in*out``,
see ``sili__new``'s own ``disldo_max_weights_sizing`` guidance) would
decouple fan-in from width and, on this hypothesis, remove the need to
keep lowering ``peak_lr`` as ``N`` grows.

Separately, Arm C's backward sine-wave time-gate (this file's
``dy_time_gate_cutoff``, see ``train_curriculum.cli_gradient_sparsity_args``)
is itself a form of gradient sparsity that may already be doing something
like this by a different mechanism -- gating which neurons update *this
step* reduces the coherent per-step change a wide layer's rank-1 update
would otherwise cause, the same instability this LR law is about, without
touching the nominal ``peak_lr`` at all. If that reading is right, the
right effective-LR model is not ``lr(N)`` alone but ``lr(N, gate_density)``
jointly -- grad-sparsity density trading off against how much the nominal
LR needs to shrink.

**Correction, 2026-09-18 (same day)**: item 2 as originally written called
for testing this via a *structural* (``max_weights``-capped) fan-in cap.
That is a known-bad direction at this model scale, already tried and
reversed multiple times before this investigation -- see
``project_sili_wide_model_mqar_baseline`` memory (``dense=False`` + a
pre-chosen ``max_weights`` budget degenerated to ``per_row=2``, reversed
back to ``dense=True`` + activation/gradient-only sparsity) and
``project_sili_synaptogenesis_pruning_testing`` memory (structural
growth/pruning at this same toy scale produced chance-level accuracy).
Real biological fan-in (1000-10000 synapses/neuron) doesn't fit inside
``state_width<=288`` without forcing near-full density anyway. A launched
width=288 run under this design was killed mid-run once this was caught;
its launcher scripts were deleted, not left around to be reused by
accident. Don't re-attempt a pre-chosen structural budget at this scale --
see ``feedback_check_journal_before_new_mechanism`` (check standing
guidance before building a new mechanism for a fresh question, not just
for a fresh failure).

**Not yet confirmed -- pending before this is treated as established**:

1. A third ``(N, peak_lr)`` point (e.g. ``state_width=192`` or a second
   width=288 LR value) to actually pin ``alpha`` instead of bracketing it.
2. **Done, 2026-09-19 -- NEGATIVE at both widths.** ``dy_k_min=
   dy_k_max=64`` (forcing an exact fixed count, overriding ``dy_r_target``'s
   own natural energy-based selection entirely) stalled hard at both
   widths: ``launch_dy_fixed_count_width128.py`` reached only
   ``vocab=16/k=3`` (one level-up, step 925, then flat for 99,075
   steps); ``launch_dy_fixed_count_width288.py`` reached ``vocab=32/k=2``
   (step 25,688, then flat) -- got further before stalling but still far
   short of the dense/Arm C results at either width. Confirmed NOT a
   width-specific fluke. Reads as: forcing an EXACT count via this
   clamp, rather than letting ``dy_r_target`` pick its own natural k, is
   itself actively harmful -- not a safe dynamic stand-in for the
   abandoned structural fan-in cap after all. The underlying "does an
   absolute, non-width-proportional budget decouple LR from width"
   question is still open; a real answer needs a DIFFERENT dynamic
   mechanism than this one, not yet designed.
3. **Designed and queued (2026-09-18)**: sweep Arm C's
   ``dy_time_gate_cutoff`` (gate density) against required ``peak_lr`` at
   fixed width=288 -- does a denser gate (closer to ungated) need dense's
   lower LR, and does a sparser gate tolerate the current, un-scaled
   ``DEFAULT_PEAK_LR``. 2x2 grid: cutoff in {0.0 (~50% density), 0.6
   (~29.5% density)} x peak_lr in {0.015 unscaled, 0.01 scaled} --
   ``launch_armc_gatemid_lr_unscaled.py`` / ``launch_armc_gatemid_lr_scaled.py``
   / ``launch_armc_gatesparse_lr_unscaled.py`` /
   ``launch_armc_gatesparse_lr_scaled.py``. This is itself dynamic (not structural)
   sparsity, so no conflict with the correction above.

.. _train_curriculum.dense_was_hardcoded_true:

``dense``: promoted from a hardcoded ``True`` to a real parameter
-------------------------------------------------------------------------------

*ID:* ``train_curriculum.dense_was_hardcoded_true``

2026-09-18, discovered while designing an absolute (non-width-proportional)
fan-in-cap test for
``train_curriculum.width_scaling_lr_fanin_hypothesis``. `ToyTileRecurrenceRMT`'s
``dense`` constructor kwarg (forces every layer's ``max_weights`` up to
``in_features*out_features`` and preseeds fully connected -- see
``sili__new``'s ``DISLDOLayer32`` docstring) was unconditionally hardcoded to
``True`` at this file's model-construction call site -- not exposed as a
``train_curriculum`` parameter at all.

**Consequence, now fixed but worth recording**: every "sparse" arm run under
this curriculum to date (Arm A/B/C, the knee-adaptive `x_r_target`, all of
it) applied its sparsity ONLY as dynamic per-step top-k/nucleus SELECTION
(`x_r_target`/`dy_r_target`/`dy_gate_mask`) on top of a genuinely
fully-connected weight matrix underneath. None of them tested real
STRUCTURAL sparsity -- a bounded total synapse count
(`wide_max_weights`, grown/pruned via synaptogenesis,
`sili__new`'s `reserve_connections`/`synap_compute_cutoff`) -- because
there was no way to turn `dense` off. `dense: bool = True` is now a real
parameter (default unchanged, so every existing run/test is unaffected);
`dense=False` plus an explicit `wide_max_weights` is required to actually
exercise structural sparsity through this file.

**Do not actually set `dense=False` at this model scale** -- direct
correction, same day, after a launched width=288 structural-sparsity
test was caught and killed mid-run. `project_sili_wide_model_mqar_baseline`
and `project_sili_synaptogenesis_pruning_testing` memories already
document this exact failure mode repeatedly: a pre-chosen `max_weights`
budget at toy scale degenerates (`per_row=2`, or chance-level accuracy),
reversed back to `dense=True` + activation/gradient-only sparsity each
time it was tried. Real biological fan-in (1000-10000/neuron) doesn't
fit inside `state_width<=288` without forcing near-full density anyway.
The parameter exists for when model scale grows enough for a real
fan-in budget to make sense, not for use now.

.. _train_curriculum.lr_override_fn_range_test:

``lr_override_fn``: bypass hook for the LR range test
-------------------------------------------------------------------------------

*ID:* ``train_curriculum.lr_override_fn_range_test``

2026-09-18, built for ``scripts/lr_range_test.py`` in response to
``train_curriculum.width_scaling_lr_fanin_hypothesis`` needing several
full 100k-step runs just to bracket a good ``peak_lr`` by hand. A
callback ``lr_override_fn(step) -> float``, checked FIRST in the LR
block (ahead of warmup and the accuracy-decay schedule) -- when set,
this file's own peak_lr/warmup/accuracy-based LR schedule is skipped
entirely for the whole run, letting a caller impose an arbitrary
per-step LR (an exponential ramp, for the range test's purposes)
without needing to fork or reimplement any of this file's training
loop, curriculum, logging, or model-construction logic. ``None``
(default): zero behavior change for every existing caller.

.. _train_curriculum.polyak_lr_f_star_assumption:

``polyak_lr``/``polyak_f_star``: why ``f_star=0`` is reasonable here
-------------------------------------------------------------------------------

*ID:* ``train_curriculum.polyak_lr_f_star_assumption``

Per-layer Stochastic-Polyak-Step-size (SPS_max variant -- see
``model/toy_tile_recurrence_rmt.rst:per_layer_learning_rate_polyak`` for
the full mechanism). ``f_star`` is Polyak's assumed optimal/floor loss
value -- the classic SPS_max trick is to just assume ``f_star=0`` rather
than estimate it online. That's a reasonable assumption specifically
for this task: MQAR's per-query cross-entropy loss has a genuine near-0
floor (a solvable deterministic key->value mapping), and
``write_time_aux_targets=False`` (this file's own default, see that
parameter's own history) already removed the one source of structurally
irreducible loss at non-query write positions ("predict next token"
when the next token in MQAR's random layout isn't actually
predictable). A more involved online-estimated ``f_star`` (parameter-
free Polyak, Abdukhakimov et al. 2025) exists in the literature for
tasks where 0 isn't a reasonable floor -- not needed here, not built.

.. _armc_gate_density_lr_equation:

Arm C gate-density LR compensation: candidate equation
-------------------------------------------------------------------------------

*ID:* ``armc_gate_density_lr_equation``

2026-09-18, direct request to work out an equation relating Arm C's
gate density ``p`` (fraction of steps a given neuron is trainable) to
its needed ``peak_lr``, and test the predicted steps-to-milestone
against dense's own. Arm C doesn't change how STRONG an update is when
a neuron fires, only how OFTEN -- two effects both point toward
scaling ``lr`` up as ``p`` shrinks: (1) frequency dilution -- a neuron
sees ``~p*T`` update events over ``T`` steps instead of ``T``, so
matching dense's cumulative movement in the same budget needs each
event proportionally bigger; (2) stability headroom -- the width-
scaling law's instability comes from ALL neurons moving coherently on
the same step (train_curriculum.width_scaling_lr_fanin_hypothesis's
``lr ~ 1/N``); at density ``p`` only ``p*N`` move at once, so a larger
``lr`` should be tolerable before hitting the same instability.

**Naive candidate**: ``lr_armc(p) = lr_dense_optimal / p``. Bracketed
against the milder ``lr_dense_optimal / sqrt(p)``, same caution as the
width-scaling law's own alpha=1.0-vs-0.5 finding (the "theoretically
justified" stronger exponent came back WORSE than the milder one
there -- don't assume 1/p is right just because the derivation sounds
clean).

Two launchers at ``cutoff=0.3`` (``p~=0.403``, same cutoff as
``launch_dense_forward_arm_c_backward.py``):
``launch_armc_lr_scaled_invp.py`` (``peak_lr~=0.0248``),
``launch_armc_lr_scaled_invsqrtp.py`` (``peak_lr~=0.0158``). Direct
comparison point: dense's own record is ``vocab=126/k=3`` at step
13,601 (``peak_lr=0.01``, see ``MQAR_LEADERBOARD.md``) -- does either
compensation formula close the gap to a comparable step count, or is
Arm C's own dynamics (see ``project_dense_vs_sparse_mqar_confusion_
matrix`` memory -- won the confusion-matrix arm despite the LOWEST
gradient-energy retention of any row) enough on its own that
compensation isn't even the right frame.

The UNCOMPENSATED comparison (same nominal ``peak_lr`` as dense, not
scaled by ``p``) is already covered by test 3's grid
(``launch_armc_gatemid_lr_unscaled.py`` etc.) -- this is specifically
the compensated/equation-driven version, isolating whether the formula
itself closes the gap rather than just observing that SOME lr works
better than another.

**Update, 2026-09-19: test 3's grid complete, and it doesn't cleanly
support either direction.** cutoff=0.0 (~50%)/unscaled=0.015 and
cutoff=0.6 (~29.5%)/scaled=0.01 BOTH reached ``vocab=126, k=3``; both
cross-combinations (~50%/scaled, ~29.5%/unscaled) got stuck at
``k=2``. A genuine crossover with only 2 LR samples per density --
reads more like each density has its OWN narrow LR sweet spot (same
non-monotonic shape ``width_scaling_lr_fanin_hypothesis`` already
found for dense: 0.01 good, 0.015 too high, 0.0067 too low) than like
a clean monotonic ``lr(p)`` trend in either direction. The two
compensation tests here (both scaling LR UP as density drops) are one
more data point toward mapping that sweet-spot surface, not a
confirmation or refutation of ``1/p``/``1/sqrt(p)`` specifically --
don't over-read either result alone.

**Real confound found, 2026-09-19 (same day)**: ``dy_time_gate_seed``
defaulted to ``None`` (OS-entropy seeded) in all 4 of test 3's grid
launchers -- each cell got a different, uncontrolled Arm C phase/
period realization, so the crossover above could be partly or wholly
an artifact of that rather than a real (density, LR) interaction.
Direct correction on the fix: a single FIXED seed across cells doesn't
control anything either -- for a sine-wave gate whose period
(``dy_time_gate_period_range``, default 50-200 steps) is comparable to
the run length, one fixed seed just picks a different arbitrary phase
realization to compare against. Resolving this needs statistical power
(multiple independent seeds per cell), not seed-pinning -- see
``feedback_statistical_power_not_seeding`` memory, the same lesson.
``armc_gate_density_lr_seed_sweep.py``: same 2x2 grid x 3 seeds, short
budget (25000 steps, covers the vocab=64/k=3 milestone every original
cell reached by step 28,083) -- reports step-of-vocab=64/k=3 per cell
so within-cell seed spread can be compared against the between-cell
gap before trusting the crossover as real.

.. _armc_polyak_threshold_not_selection:

Polyak + Arm C: threshold-level control only, not selection -- design note, not built
-------------------------------------------------------------------------------------------

*ID:* ``armc_polyak_threshold_not_selection``

2026-09-18/19, resolving a real conflict found while exploring
combined dynamic-LR + dynamic-grad-sparsity: Polyak step size is
inherently SCALE-TRACKING (``lr`` must shrink as the residual/gradient
magnitude shrinks near convergence -- that's the whole point of its
annealing behavior). Magnitude-based selection (top-k, ``dy_r_target``)
is inherently SCALE-INVARIANT (always grabs "the biggest k available,"
whatever their absolute magnitude that step) -- so a Polyak ``lr``
computed from a magnitude-selected subset's own gradient norm can't
track true convergence progress (top-k keeps re-normalizing itself),
while a Polyak ``lr`` computed from the PRE-selection full gradient
describes a step that isn't the one actually applied. Neither resolves
cleanly -- Polyak and NAIVE magnitude-based (top-k-style) selection are
genuinely in tension, not just awkward to combine.

**Resolution (design note, not yet built)**: Arm C's gate is NOT
magnitude-based at all -- ``gate_j(t) = sin(...) > cutoff`` selects
WHICH neurons update this step from pure time/phase dynamics,
completely independent of gradient magnitude. That sidesteps the
conflict above by construction: a Polyak-style (or any residual-
tracking) controller could safely adjust Arm C's ``cutoff`` (the
DENSITY knob, a single scalar) as the residual shrinks, WITHOUT ever
touching the actual on/off selection identity -- no self-normalizing
top-k feedback loop to fight, since which neurons fire this step stays
governed by phase/time alone. Two cleanly separated roles: Polyak-like
control over HOW MUCH sparsity (the threshold/density), pure amortized
time-division over WHICH neurons.

**Broader principle, empirically grounded, not just theoretical**:
every SIGNAL-GUIDED grad-sparsity selection mechanism tried in this
investigation (``project_dense_vs_sparse_mqar_confusion_matrix``'s Arm
A/B, magnitude/energy-based ``dy_r_target`` itself) has underperformed
Arm C's pure time-division amortization on final outcome, despite Arm C
having the LOWEST gradient-energy retention of any row tested. Grad
sparsity's real value here looks like temporal amortization -- fair,
guaranteed turn-taking over time -- not a signal to be found and
optimized; repeated attempts at finding a good selection SIGNAL haven't
borne fruit. Worth remembering specifically if a future attempt at a
FULLY sparse model (both forward AND backward, not just Arm C's
backward-only-with-dense-forward setup tested so far) struggles to
reach the full vocab=126/k=3 milestone -- the fix is more likely to be
"amortize better" (time-division, coverage guarantees) than "select
better" (a smarter magnitude/energy signal).
