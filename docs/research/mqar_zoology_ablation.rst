.. _mqar_zoology_ablation:

MQAR ablation catalog: ``ToyTileRecurrenceRMT`` vs. real Zoology baselines
===========================================================================

*ID:* ``mqar_zoology_ablation``

Context: our MQAR curriculum (``scripts/train_mqar_curriculum.py``) plateaus
around k=3 (vocab-capped) or vocab=128/k=1 (k-capped), while HazyResearch's
own attention baseline solves k up to 64 at ``d_model=32``. This catalog
enumerates every concrete difference between our setup and the real
Zoology repo (cloned to verify against actual code, not just the paper --
per direct instruction, since a paper's stated design and its actual
shipped code are not guaranteed to match). Each item is tagged:

- **[CODE]**: verified directly against HazyResearch/zoology source (commit
  cloned 2026-09, see file paths given).
- **[LIT]**: a claim from a specific paper, recalled from training
  knowledge, not re-fetched live this session -- flagged separately from
  [CODE] items so confidence levels aren't conflated. Worth a live check
  before treating a [LIT] citation as load-bearing for a real decision.
- **[RULED OUT]**: checked directly and confirmed NOT a difference, or
  already isolated by a prior experiment in this project -- kept here so
  it isn't re-investigated from scratch later.

Zoology repo reference paths below are relative to a fresh
``git clone --depth 1 https://github.com/HazyResearch/zoology``.

Ruled out already
------------------

- **Position encoding.** [CODE] Zoology's real MQAR attention sweep sets
  ``max_position_embeddings=0`` (``zoology/experiments/models_repo.py:24``,
  ``add_attention``) -- no learned position embedding, same as ours. Not a
  difference.
- **State-mixer / FFN block.** [CODE] The MQAR sweep passes
  ``state_mixer=dict(name="torch.nn.Identity")`` via ``model_factory_kwargs``
  (``original_mqar_configs.py:43``) -- no FFN after attention for THIS
  comparison (Zoology's general default elsewhere is an MLP,
  ``hidden_mult=4``, but not here). Same as ours (no FFN at all). Not a
  difference for this specific comparison.
- **Multi-head attention.** [CODE] Zoology's own attention baseline uses
  ``num_heads=2`` (``models_repo.py:11``) and still solves k up to 64.
  Already independently re-derived this session from the task structure
  (MQAR's k queries land at different sequence positions, not
  simultaneously, and our model is recurrent -- one "turn" per query --
  so the standard multi-head "several simultaneous questions" argument
  doesn't transfer). Not a promising lever.
- **Raw width (d_model / state_width).** [CODE] Zoology's smallest arm,
  ``d_model=32``, already solves k=64. We've tested state_width well above
  the fp32/fp4 equivalent of that without breaking the k=3 ceiling. Not
  the bottleneck by itself.
- **gaussian_attention as a content-blind mechanism.** [CODE] Verified
  directly in ``sili/lib/headers/attention.hpp:710``:
  ``score[q,j] = (Q[q].K[j])*scale - (j-center[q])^2/(2*sigma[q]^2)``,
  full softmax, every key always reachable. At num_tiles~16 this is
  ordinary full content-based attention plus a learnable (and
  sigma-inflatable-to-negligible) positional prior, not a replacement for
  content addressing. Not a structural blocker.
- **Task generator ambiguity (repeated/overlapping vocab, gap
  distribution, random-fill).** [CODE] Directly compared
  ``model/toy_recall_task.py:generate_mqar_sequence`` line-by-line against
  ``zoology/zoology/data/multiquery_ar.py``: same disjoint key/value
  vocab halves, same no-replacement sampling, same power-law gap
  distribution (``power_a=0.01``) for query placement, same
  random-non-query-token filling. Byte-for-byte equivalent task
  definition. Not a difference.
- **Causal vs. non-causal attention, as a soundness question.** Zoology's
  attention MUST be causal because its training loop feeds the whole
  sequence through a single shifted-target (next-token) forward pass --
  non-causal attention there would let a position see its own label.
  Our model is a true step-by-step RNN: each ``step()`` call's window only
  ever contains tokens already received, so non-causal attention *within*
  that window cannot leak anything future. Different mechanism, but both
  sound; not a bug on our side.
Tier 1 -- high promise, direct mechanistic literature support
-----------------------------------------------------------------

0. **Window size (num_tiles=16 vs 8) genuinely gates reaching k=3 --
   REOPENED, no longer ruled out.** [CODE, real run] Originally filed
   under "ruled out" on the reasoning that
   ``seq_len_for_k(k) <= num_tiles`` at tested k means content stays live
   in-window regardless of the cross-step ``memory`` detach
   (``model/toy_tile_recurrence_rmt.py:908``,
   ``docs/research/toy_tile_recurrence_rmt.rst:step_design``). A completed
   40,000-step run at ``num_tiles=8, vocab=36`` (fp32/block4, kcycle
   curriculum) contradicts that: 447 total LEVEL_UP events over the full
   run, k never once reached 3 (``final_k=1 peak={vocab:36,k:2}``). The
   matched ``num_tiles=16, vocab=36`` arm ALSO now completed its own full
   40,000-step run: ``final_k=1 peak={vocab:36,k:3}`` -- reached k=3
   (once, step 14909), the narrow-window arm never did, over two
   complete, directly comparable full runs. This is now a hard,
   completed result, not an interim one. Same task, same vocab, same
   everything else -- the only difference is window size. Corroborating,
   now with a much larger sample (still in-progress, not yet run to
   completion, but no longer a small-n concern) at ``vocab=16``:
   ``num_tiles=16`` reached k=3 four times by step 11,664; ``num_tiles=8``
   had reached it ZERO times by step 15,500 -- more total steps, zero
   k=3 events, at the same vocab. Two independent vocab settings (16 and
   36) now both show this same asymmetry. Cross-step BPTT/credit-assignment
   absence (``step()``'s documented "no BPTT across calls" design) is
   back on the table as a real candidate, not ruled out -- next step is
   letting the ``num_tiles=8`` arms run to completion to make this the
   same kind of hard result the ``vocab=36`` pair already is.

Tier 1 (continued) -- direct mechanistic literature support

1. **Depth: 2 stacked layers (Zoology) vs. 1 (ours).**
   [CODE] ``add_attention(..., num_layers=2)`` -- every Zoology MQAR model
   has TWO independent attention blocks in sequence (each with its own
   Wqkv/out_proj). ``ToyTileRecurrenceRMT`` has exactly one q/k/v/o_proj
   set; the only "depth" is time (recurrent steps), not stacked
   transformation.
   [LIT] Olsson et al., "In-context Learning and Induction Heads"
   (Anthropic, 2022, arXiv:2209.11895) -- the mechanistic circuit
   transformers use to solve exactly this kind of copy/recall task is a
   TWO-layer minimum: a layer-1 "previous-token head" that writes
   "what came before me" into each position's residual stream, then a
   layer-2 "induction head" that does the actual key-match-and-copy using
   that enriched representation. A single attention layer structurally
   cannot do both hops in one shot for content it hasn't already
   locally tagged. This is the single most concrete, well-evidenced
   candidate explanation for a hard capability ceiling that doesn't move
   with width, curriculum, or optimizer tuning.
   **Test:** add a second write-then-read attention pass (own q/k/v/o_proj,
   or a cheap shared-then-refine variant) inside one ``step()`` call,
   feeding layer-1's output as layer-2's input; check if k=3+ becomes
   reachable at unchanged width.

2. **Short causal conv immediately before attention, every layer.**
   [CODE] Zoology's actual "attention" arm is a ``Hybrid`` of
   ``BaseConv(kernel_size=3, implicit_long_conv=True)`` THEN ``MHA``
   (``original_mqar_configs.py:47-63``, ``add_attention`` wraps both into
   one ``sequence_mixer``) -- not bare attention. We have no
   convolutional component anywhere.
   [LIT] The short conv is a cheap way to pre-mix each position with its
   immediate neighbors before attention runs -- functionally close to
   Olsson et al.'s "previous-token head" role above, done via a fixed/
   cheap op instead of spending a whole attention layer on it. Directly
   related to item 1, not independent evidence.
   **Test:** cheaper first cut than item 1 -- add a fixed or lightly
   learned 3-tap causal blend over the window before q/k/v projection,
   see if it substitutes for a full second layer.

3. **Auxiliary write-time supervision we have that Zoology does not.**
   [CODE] Zoology's labels (``multiquery_ar.py:130-131``) are -100
   (ignored) everywhere except the actual query positions --
   ``nn.CrossEntropyLoss()``'s default ``ignore_index=-100`` means ONLY
   real queries ever produce gradient. Our own
   ``scripts/train_mqar_rmt_reference.py:_build_targets`` does
   ``targets.setdefault(i, tokens[i+1])`` for every context position
   ``i < context_size-1`` -- meaning every key/value WRITE position also
   gets a "predict the immediate next token" (key->value copy) training
   signal, not just the queries. This is a real, confirmed difference,
   not a hypothesis.
   [LIT] No specific paper needed here -- this is a directly comparable
   training-signal difference, testable in isolation.
   **Test:** disable the write-time auxiliary targets (only supervise
   actual query positions, matching Zoology exactly) and see whether the
   k-ceiling moves in either direction. Could be helping (extra shaping
   signal) or could be a confound (different effective task difficulty
   than the literature's own scaling numbers assume).
   **Interim result (in-progress run, not yet complete), now a much
   stronger signal than first observed:** ``vocab=16, num_tiles=16``
   with the aux loss disabled reached k=3 34 times by step 11,648, vs.
   the matched arm with the aux loss left on reaching k=3 only 4 times
   by step 11,664 -- an 8.5x frequency difference at matched step
   counts (growing from the earlier-observed 6x), not just a
   directional lean. Checked for signs the aux loss was actually needed
   as a stability guard (the concern being raised, not confirmed): no
   NaN/Inf in the no-aux-loss run's log, and its loss_ema range (max
   3.68) is bounded and actually lower on average than the aux-loss-on
   arm's (max 4.07) -- no evidence of blow-up. Note the ACTUAL
   stability mechanisms in this project (L1 sparsity penalty,
   magnitude-clip penalty, the ``max_abs_delta``/``max_ci`` hard bounds
   in ``NOCAPS_KWARGS_FP32``, amortized L2 decay) are untouched by this
   ablation -- what's disabled here is only the extra write-time
   supervised label, not any regularizer. Strong evidence this
   write-time auxiliary signal is a real confound, not a helpful
   shaping signal -- dropping it should likely become the new default
   once the full run confirms this holds to completion. A combined
   ``K_START=2`` + no-aux-loss run (curriculum-only, no model changes,
   ``arm_combo_kstart2_noaux``) is now in progress at a 100,000-step
   budget specifically to check for SUSTAINED occupancy at k=3+ (does
   it eventually complete a full correct-streak at k=3 and advance past
   the ``vocab=16`` tier), not just peak-k-touched -- k=3 touch count
   climbed through four checkpoints (20/2455, 82/5722, 111/8000,
   124/10,382) with the growth rate itself accelerating then slowing --
   but as of step 12,500 it has now ALSO plateaued: still 124 events,
   unchanged for 2,118 steps -- since confirmed at full uncontended
   speed (the other 5 arms were stopped to free CPU capacity, 1.26sps
   -> 5.56sps) still flat at 124 all the way through step 30,500, a
   20,000+-step plateau, not an artifact of the earlier slow/contended
   sampling rate. Every arm tested (all four single ablations plus the
   combo) is now hard-plateaued: ``wide16_v16`` at 4 since step 11,664
   (15,000+ steps flat), ``wide16_kstart2`` at 2 since ~step 2686
   (12,800+ steps flat), ``wide16_noaux`` at 35 since step 17,119
   (11,300+ steps flat), ``combo_kstart2_noaux`` at 124 since step
   10,382 (20,000+ steps flat). This resolves the earlier open question:
   the combo run's touch-frequency advantage over every single ablation
   was real and large, but it still hit a hard ceiling rather than
   continuing toward a held streak -- weakens (does not eliminate) the
   "just needs more steps toward its own phase transition" reading and
   correspondingly strengthens the possibility of a genuine structural
   blocker (most likely candidate: Tier 1 item 1, the missing 2-layer/
   induction-head-circuit depth) that no curriculum-only lever tested so
   far is sufficient to overcome
   alone. Zero real vocab-tier advances across every arm.

   **Refined characterization (not just "plateaued" -- a likely
   regression):** since the step-10,382 LEVEL_DOWN, k has never left 2
   for 36,600+ steps (76/76 logged lines at k=2, zero further LEVEL_UP
   attempts, zero k=1 -- confirms the K_START=2 stack-floor mechanism
   held for the ENTIRE run). ``acc_ema`` sampled through that stretch:
   0.407 (right at the drop) -> 0.137, 0.259, 0.249, 0.160, 0.143,
   0.154, 0.152, 0.100, 0.189 (steps 12,500-47,500). Chance level for
   this task is ~0.125 (1/8 -- disjoint 8-token value half). Accuracy
   at k=2 has been hovering AROUND CHANCE for most of this stretch, not
   the 0.4-0.7 range k=2 showed earlier in this same run and in every
   other arm. This is a materially different finding than "stable but
   insufficient plateau" -- it looks like the model regressed into a
   degraded, barely-better-than-guessing state at k=2 after falling out
   of k=3, which directly explains why it never re-attempted LEVEL_UP
   (never strung together enough consecutive correct answers to trigger
   one). Open question for the eventual architectural test: is this
   regression specific to this run/seed, or a general failure mode
   worth checking for in future runs (e.g. via a min-accuracy floor
   diagnostic, not just k/vocab-tier tracking).

   Next step: a real architectural test (add the second attention
   "layer"/hop within one step()) rather than more curriculum-only
   sweeps of what's already plateaued -- deferred until this run
   completes its full 100,000-step budget, per direct instruction.

2. **LEVEL_DOWN-triggered repeated training disruption -- corroborated by
   an existing, independently-documented finding for a different axis.**
   [CODE, real run in progress] Direct hypothesis: repeated k=3-attempt-
   then-LEVEL_DOWN cycles (124 of them in the combo run before it
   stopped attempting k=3 at all) may be causing real training
   disruption each time -- weights converged for k=2 get pulled toward
   k=3 for a few dozen steps, then yanked back before adapting, and that
   partial adaptation doesn't necessarily undo cleanly. NOT a "learned
   avoidance" mechanism -- ordinary per-token supervised loss has no
   credit-assignment pathway connecting current accuracy to a FUTURE
   consequence (no BPTT/RL value function exists here, see Tier 1 item
   0's history), so the model cannot have learned to suppress accuracy
   to avoid a later disruption. The disruption itself, if real, is
   immediate-feedback (each excursion directly perturbs weights),
   which does have a mechanism. [LIT/CODE] Directly corroborated by an
   ALREADY-DOCUMENTED finding in this project for the vocab axis
   (``docs/research/train_mqar_curriculum.rst:
   train_curriculum.wrong_streak_threshold_reward_hacking``):
   ``WRONG_STREAK_THRESHOLD=5`` vs ``STREAK_THRESHOLD=10`` asymmetry
   previously caused ``max_streak`` to plateau hard at 7/10 for 2000+
   steps, "consistent with the model finding it cheaper... to hover near
   the boundary and get bounced back to easy data than to commit to
   genuinely harder representations." Same asymmetric-threshold
   mechanism, different axis (k instead of vocab) -- not a new
   phenomenon, a recurrence of an already-known one.
   [CODE] Confirmed Zoology's real curriculum design cannot produce this
   at all: ``zoology/zoology/data/utils.py:_SyntheticDataset`` docstring
   states data segments are "not to be mixed"; batch ordering iterates
   ``segment_idx`` (difficulty tier) as the OUTER loop, processing each
   tier's examples completely before moving to the next, with
   ``DataLoader(..., shuffle=False)`` explicit -- training difficulty is
   monotonic and one-way, no regression, no adaptive gating on accuracy
   at all.
   **Test:** ``wrong_streak_threshold=float("inf")`` (an existing,
   already-supported kwarg, ``scripts/train_mqar_curriculum.py:283``,
   gates the regression call at line 646) disables LEVEL_DOWN entirely
   -- true infinity, not a large finite approximation. Running
   ``arm_nolevel_down`` (same K_START=2 + no-aux-loss base as the combo
   run, plus this) as a direct comparison against
   ``arm_combo_kstart2_noaux``, which is the exact run exhibiting the
   repeated-disruption pattern this test targets.

   **RESULT: CONFIRMED.** ``arm_nolevel_down`` reached vocab=32 (step
   3,134, real ``LEVEL_UP``) then vocab=64 (step 8,672, real
   ``LEVEL_UP``) -- 4 total ``LEVEL_UP`` events by step 14,000 (2
   within-tier k=2->3 transitions, 2 real vocab-tier advances), each one
   a genuine held 10-in-a-row correct-streak, since that's structurally
   the only way ``LEVEL_UP`` can fire. At the exact same step count, the
   matched ``arm_combo_kstart2_noaux`` (identical config, LEVEL_DOWN
   still enabled) remains permanently stuck at vocab=16/k=2 with
   accuracy hovering near chance. This is the single most direct
   confirmation in this whole investigation: disabling LEVEL_DOWN (the
   one difference between these two runs) unblocked real, repeated k=3
   mastery and vocab advancement that the LEVEL_DOWN-enabled run never
   achieved even once in 70,000+ steps. Resolves the "structural blocker
   vs. curriculum artifact" question raised earlier in this item and in
   Tier 1 item 0/1 -- the actual blocker was the asymmetric-threshold
   LEVEL_DOWN mechanism, not missing architectural depth. The previously
   -planned architectural test (second attention layer/hop) is likely
   NOT needed to hit this session's original goal (k=3 AND vocab=128) --
   recommend continuing ``arm_nolevel_down`` toward that target instead
   of pursuing the architecture change.

   **Progress update**: a 5th ``LEVEL_UP`` at step 18,937 (``vocab=64,
   k=3``) -- took 10,265 steps, the longest phase yet, but succeeded
   (post-transition accuracy dipped low again as usual, then climbs
   back -- same pattern after every prior transition, not a new
   concern).

   **"Full vocab" milestone reached.** A 6th ``LEVEL_UP`` at step 43,286
   advanced straight to ``vocab=126`` (``TASK_VOCAB_MAX``, the odometer's
   ceiling), ``k`` reset to 2 (acc=0.51 at the transition). This phase
   (``vocab=64, k=3`` -> ``vocab=126``) took 24,349 steps -- more than
   double the previous longest phase, and for a stretch (steps ~27,500-
   41,500) loss was actually drifting up (3.8->4.8) with flat-low accuracy
   (0.03-0.13), which briefly looked like it might be a genuine
   model-capacity ceiling (state_width=128/embed_width=16 too narrow for
   vocab=64 at k=3) rather than a training-dynamics issue. It broke
   through anyway -- so at this scale it was "slow," not "stuck." Worth
   remembering as a false-alarm pattern: a long flat-or-worsening stretch
   under this curriculum does not reliably distinguish a real capacity
   ceiling from an unusually long LEVEL_DOWN-disabled convergence basin;
   only continued non-convergence all the way to ``max_steps`` would.

   **Final result: run completed at max_steps=100,000 without reaching
   "full k".** ``DONE wall_s=12999.8 total_steps=100000
   steps_per_sec(engine)=7.69 final_k=2 final_vocab=126
   vocab_tiers_seen=[16, 32, 64, 126] peak={'vocab': 126, 'k': 2}``. From
   the step-43,286 transition to the end of the run (56,714 steps -- more
   than double the previous longest phase, and the majority of the run's
   entire step budget), the model stayed at ``vocab=126, k=2`` and never
   held a 10-in-a-row streak long enough to ``LEVEL_UP`` to k=3. Unlike
   the earlier vocab=64/k=3 phase (which also ran unusually long, then
   broke through), this phase showed no recovery trend for its entire
   remaining duration -- accuracy stayed noisy and low the whole way
   (roughly 0.001-0.08, no upward drift) and loss, if anything, drifted
   slightly worse over time (4.4-4.8 in the back half vs. the 4.5-4.9
   range seen earlier in the same phase). So the "false alarm" caveat
   above does NOT apply here: this phase genuinely never converged within
   the tested budget, as opposed to the earlier phase which looked stuck
   but wasn't.

   **Honest summary of the goal from this session** (k=3 AND vocab=128,
   simultaneously): NOT reached. What WAS reached: full vocab (126) at
   k=2, and k=3 at a lower vocab (64). The LEVEL_DOWN-disabled fix
   clearly unblocked real progress that the LEVEL_DOWN-enabled run never
   achieved at all -- but "k=3 held at vocab=126" is evidently harder
   than either "k=3 at vocab=64" or "k=2 at vocab=126" individually, and
   56,714 steps (about 3 hours of wall time at this model's ~7.7 steps/
   sec) wasn't enough to cross that harder combination. Plausible next
   levers, not yet tested: (a) simply more steps -- this run was capped
   at 100,000 by an arbitrary probe budget, not evidence of a true
   ceiling; (b) more raw compute/throughput to make longer runs
   affordable (the user's "8x from a better CPU" plan, or eventually
   GPU); (c) the sparsity machinery (``input_sparsity_p``/
   ``dy_sparsity_p``, already built per Phase 0-8 above) combined with a
   wider ``embed_width``, which increases capacity per wall-clock step
   rather than just running longer at the same width. (b) and (c) are
   complementary, not alternatives -- discussed as the next planning
   topic with the user (2026-09-07).

Tier 2 -- plausible, well-established in general ML, less specific to this failure
----------------------------------------------------------------------------------

4. **Batch size / gradient-noise averaging.** [CODE] Zoology trains with
   ``batch_size=256`` (``original_mqar_configs.py:29``) -- every optimizer
   step averages gradient over 256 independent random examples. We train
   one sequence at a time (true online SGD, batch=1 at the sequence
   level). [LIT] Well-established (not citing a specific paper): larger
   batches reduce gradient-estimate variance, generally easing optimization
   on tasks with any per-example randomness (here: which keys/values/gaps
   got sampled). **Test:** accumulate gradients over N freshly-generated
   sequences before applying an update, holding total examples-seen
   constant, and compare.

5. **Optimizer + schedule: AdamW/cosine (Zoology) vs. custom
   RMSprop-like DISLDO/quality-based LR (ours).** [CODE] ``train.py:185-192``
   -- ``AdamW(lr, weight_decay=0.1)`` + ``CosineAnnealingLR(T_max=max_epochs)``.
   Ours has no momentum term by design (see
   [[feedback_importance_is_already_the_optimizer]]) and a validated
   quality-gated LR instead of a fixed decay schedule (task #262/#263).
   Different philosophy, already partially validated in-project; lower
   priority to re-litigate without a specific new symptom pointing here.
   **Test (if pursued):** momentum ablation in the existing torch-control
   reference model (task #237 already tracks this axis generally).

6. **Dataset scale / repetition structure.** [CODE] Zoology trains on
   ~100k-180k unique examples per sweep tier, repeated for 32 full epochs
   (~3-6M example-presentations total); we generate one fresh random
   example per online step, no repeats, for however many thousands of
   steps a curriculum run lasts. Different exposure budget in both
   directions (Zoology repeats a finite set many times; we see unlimited
   unique examples but likely far fewer total). **Test:** compare total
   query-example exposures actually reached in a real run against
   Zoology's ~3-6M figure -- if we're orders of magnitude short, that's
   independently sufficient to explain a lot, no architecture change
   needed first.

7. **Vocab size (8192 vs. our curriculum-grown max ~126).** [CODE]
   confirmed real difference in scale. [LIT] Arora et al., "Zoology:
   Measuring and Improving Recall in Efficient Language Models" (2023,
   arXiv:2312.04927) argues LARGE vocab is needed specifically to prevent
   architectures from cheating via shortcuts that don't generalize --
   i.e., a small vocab should, if anything, make the task EASIER for a
   given architecture, not harder. Since we already fail at small vocab,
   this cuts against vocab-size being the explanation for our ceiling
   (if anything it's evidence the ceiling is real, not a vocab-scale
   artifact). Kept for completeness, not expected to be the fix.

Tier 3 -- low priority / already substantially addressed
------------------------------------------------------------

8. **Attention dropout (0.1 in Zoology vs. 0 in ours).** [CODE] confirmed
   (``models_repo.py:10``). Regularization difference; unlikely to explain
   a hard capability ceiling rather than a generalization gap. Cheap to
   test if everything above is exhausted.
- **Embedding init details** (Zoology default ``std=0.02`` normal +
  GPT-2-style ``1/sqrt(2*n_layers)`` residual-branch scaling,
  ``zoology/model.py:143-180``). Standard init tricks, not expected to
  explain a hard k-ceiling.
- **Precision/quantization (fp4/fp8 vs. Zoology's plain float).** Already
  substantially isolated in this project: the fp32 arm (task #318, no
  quantization at all) hits essentially the same low-k ceiling as fp4/fp8,
  which rules OUT quantization noise as the primary cause of the ceiling
  itself (though it may still compound other factors).
- **Hardware (GPU-batched vs. CPU-SIMD single-sequence).** Affects
  reachable step count per wall-clock hour, not learnability in principle;
  relevant context for item 6, not its own lever.

Recommended order
------------------

Given the above, the highest-value next experiments, roughly in order:
(1) Tier 1 item 3 (disable write-time auxiliary loss -- cheapest, one-line
change, isolates a confirmed real difference), (2) the in-flight
window-boundary probe resolving the BPTT question definitively, (3) Tier 1
item 1 (second attention "layer"/hop within a step -- most direct
literature-backed candidate for a genuine architectural gap), (4) Tier 2
item 6 (measure actual example-exposure budget against Zoology's ~3-6M
before assuming anything else is broken).
