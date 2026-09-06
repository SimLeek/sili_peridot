``train_tile_curriculum.py`` research notes
==================================================

Companion doc to ``scripts/train_tile_curriculum.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets alone render invisible on GitHub, frozen code
snippets on real-bug sections only).

.. _train_tile_curriculum.module_overview:

Module overview: fast curriculum harness, built after an overnight no-learning result
------------------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.module_overview``

Fast (seconds-to-minutes) curriculum test harness for
``ToyTileRecurrenceRealFP4`` -- built to answer "does ANY version of this
architecture actually learn anything" before scaling up again, per direct
instruction after an overnight run showed no real learning for any arm
(confirmed via ``scripts/learning_slope.py``: PLATEAUED at a near-chance
biased collapse, not learning).

Curriculum: ``seq_len`` starts at ``SEQ_LEN_START`` and grows by 1 every
``STEPS_PER_STAGE`` steps up to ``SEQ_LEN_MAX`` (``<= NUM_TILES``, so this
stays a pure in-context test -- the tile window is always wide enough to hold
the whole sequence; out-of-context, ``seq_len > num_tiles``, is a later phase
once in-context is solid, per direct instruction).

``use_attention=False`` bypasses ``gaussian_attention`` entirely (see
``ToyTileRecurrenceRealFP4``'s own docstring) -- an ablation to isolate
whether attention itself is the hard-to-learn part, tested BEFORE assuming
the whole architecture is broken.

Usage: ``python3 train_tile_curriculum.py <arm> <use_energy 0|1>
<use_attention 0|1> <total_steps> [checkpoint_every] [seed]`` where ``arm``
is one of the keys in the ``ARMS`` registry (``rank1``/``rank2``/``fp8``/...).

.. _train_tile_curriculum.generate_copy_sequence_task_design:

``generate_copy_sequence``: simplest possible state-carrying task
-----------------------------------------------------------------------

*ID:* ``train_tile_curriculum.generate_copy_sequence_task_design``

``token[0]`` is the "key", everything else is random filler; the ONLY thing
to predict is ``token[0]`` again, queried at the FINAL tick. Works for any
``seq_len >= 2`` (``generate_mqar_sequence`` requires
``seq_len >= 4*num_kv_pairs``, which can't express ``seq_len=2/3`` -- this is
what "start at seq_len=2" per direct instruction actually needs). One
``(position, target)`` pair, always at the last position, matching this
architecture's own "only the last tile produces logits" convention.

.. _train_tile_curriculum.tuning_constants_and_grad_clip:

Tuning constants: tiny-scale sizing, and a real gradient-clip necessity
-------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.tuning_constants_and_grad_clip``

``NUM_TILES=4``: roughly matches num_cores per direct suggestion, and also
sets the curriculum's in-context ceiling (``seq_len <= this``). ``VOCAB=10``:
small vocab too, so chance is 0.1 (not 0.025) and a trivial task doesn't need
thousands of steps just to beat noise. ``MAX_WEIGHTS_PER_LAYER=128``:
``per_row=32`` at ``state_width=32`` -- generous at this tiny scale, not the
bottleneck being tested.

``MAX_GRAD_NORM=1.0`` is a global gradient-norm clip on the plain-``Tensor``
params (``input_ln``/``state_ln``/``centers``/``log_sigmas``, trained via the
external ``AdamOptimizer`` -- NOT ``DISLDOLayer``'s own weights, which update
inline during ``backward()`` and can't be clipped the same way, see
``clip_grad_norm_``'s own docstring). Found NECESSARY, not just good
practice: fully-dense connectivity's larger fan-in lets logits grow large
enough (``mean_acc`` collapse traced to real exponential blowup, JOURNAL.md
2026-08-10) that the resulting cross-entropy gradient reaching these params
via Adam produces a NaN parameter update -- confirmed via direct per-stage
diagnosis (``scripts/diagnose_dense_vs_sparse.py``), reproducibly at the same
step every time. 1.0 matches nanoGPT's own commonly-used default (already
this project's cited reference elsewhere, e.g. ``lr_schedule``'s docstring).

.. _train_tile_curriculum.arms_registry_overview:

``ARMS`` registry: a running precision/architecture sweep, one entry per hypothesis
-------------------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.arms_registry_overview``

Each key in ``ARMS`` is one arm of an ongoing series of A/B comparisons run
through this same curriculum harness -- ``rank1``/``rank2``/``fp8`` are the
base architecture options (plain ``DISLDOLayer``, a 4-bit rank-2 scale
scheme, and real FP8). ``fp32`` is the precision-ceiling reference: isolates
whether a gap is FP4 quantization coarseness or an architecture/
training-dynamics limit, since fp32 has no quantization noise at all.

Every other entry below groups into one of several investigations, each
covered in its own subsection: alternative 4-bit scale representations,
FP8 cold-start/scale-staleness fixes, the true-multi-digit residual-FP4
architecture and its lr_power/base/dense/connectivity axes, and a
stochastic-vs-deterministic-rounding root-cause hunt. Arms are added
incrementally as each hypothesis comes up rather than replacing prior
arms -- direct convention throughout this file: a new idea gets a NEW key
so old comparison points stay available, never a silent modification of an
existing arm's config.

.. _train_tile_curriculum.arms_4bit_scale_schemes:

4-bit scale-representation arms: same bit budget, different envelope shape
-------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.arms_4bit_scale_schemes``

``rank1_8bit`` (``bits=8, scheme=rank1``): the exact scheme that reached 1.0
at every out-of-context distance on the earlier tanh-cell task (JOURNAL.md)
-- direct retest here.

``row_4bit``/``rank1_4bit``/``rank4_4bit``/``multi_fp4`` are alternative
4-bit scale representations at the SAME bit budget as ``rank2``
(``bits=4, scheme=rankn, rank=2``), varying envelope shape (not bit-depth)
to see which one recovers out-of-context accuracy at 4-bit: ``row_4bit`` is
plain per-row max-abs; ``rank1_4bit`` is a row x col envelope; ``rank4_4bit``
is a higher-rank envelope; ``multi_fp4`` (``scheme=residual, n_stages=2``) is
true residual/cascaded quantization -- 2 stages of real 4-bit each, summed,
8 bits/weight total, a fair fight against ``rank1_8bit``'s single 8-bit
code, not the earlier ruled-out envelope-of-envelope idea.

``fixed_digit_2``/``_3``/``_4`` (``scheme=fixed_digit_residual``) use ZERO
trained/fitted scale anywhere (no row/col vector, no per-call
data-dependent recompute): 2/3/4 fixed FP4 "digit" stages, ``base=4.0``
derived directly from real FP4 (E2M1)'s own 1-mantissa-bit relative
precision, ``e_shared`` derived ONCE from init weights and frozen for the
whole run. ``fixed_digit_2`` is 8 bits/weight, same budget as
``rank1_8bit``/``multi_fp4`` -- a direct test of whether the whole
trained-scale mechanism (and its staleness bug, see
``train_tile_curriculum.arms_stale_scale_bug_isolation``) can be skipped
entirely, per direct design discussion. ``fixed_digit_3``/``_4`` (12/16
bits) check whether more digits closes any remaining gap to
``rank1_8bit``/``multi_fp4``'s near-1.0 result; ``fixed_digit_3`` was found
to need half the ``PEAK_LR`` to stabilize (finer quantization removes more
of the implicit noise-filtering coarse rounding was providing), so
``fixed_digit_4`` likely also needs a reduced ``--peak_lr`` to be fair.

.. _train_tile_curriculum.arms_fp8_scale_coldstart:

FP8 cold-start/scale-staleness arms
------------------------------------------

*ID:* ``train_tile_curriculum.arms_fp8_scale_coldstart``

``fp8_seeded``: real ``DISLDOLayer8`` (true C++ E4M3 + rank-1
``value_scale``/``output_scale``), scale seeded from a real closed-form fit
at init instead of left at 1.0 -- tests whether real fp8's out-of-context
collapse is a cold-start/undertrained-scale problem, not the representation
itself. ``fp8_reseeded`` (``reseed_every=250``): scale re-seeded from a
fresh closed-form fit every 250 training ``backward()`` calls -- tests
whether REPEATED correction (no change to the real weight-update math)
substitutes for the simulation's every-step refit.

``fp8_resync``: the REAL C++ fix (not a Python approximation) --
sili__new's ``disldo_backward`` now defers each touched entry's store until
``value_scale``/``output_scale`` are BOTH finalized for the call, instead of
storing under the stale pre-update scale. Seeded like ``fp8_seeded`` for a
fair comparison (isolates the deferred-write fix, not "was output_scale
active at all"). ``fp8_adamax``: same deferred-write fix, but
``value_scale``/``output_scale`` use an AdaMax-style decayed running-max
update instead of RMSprop -- see ``AdaMaxScalePolicy``'s docstring in
sili__new.

.. _train_tile_curriculum.arms_true_multi_digit_lr_power:

``TrueMultiDigitLayer`` and the ``lr_power`` sweep
---------------------------------------------------------

*ID:* ``train_tile_curriculum.arms_true_multi_digit_lr_power``

``true_multi_digit_lr0``/``lr1``/``lr2``: genuinely SEPARATE per-digit
training, REAL FP4 (``DISLDOLayer``) per digit -- no hidden fp32 shadow, no
extra Python-side quantization step (real ``disldo_backward`` already
stores FP4 natively). ``lr_power=0``: chain-rule-only scaling (the ordinary
``out_i*factor_i`` gradient reduction, nothing extra on top) -- this is the
natural baseline, NOT an unscaled naive one, see ``TrueMultiDigitLayer``'s
own corrected docstring. ``lr_power=1``: digit i's rate ALSO divided by
``base**i`` on top of the chain rule's own reduction -- extra damping beyond
natural. ``lr_power=2``: divided by ``base**(2i)`` on top -- more extra
damping.

``true_multi_digit_fp32_ref`` is a REFERENCE/ceiling control (per direct
request: disldo vs a non-quantized-storage comparison catches whether real
FP4 storage itself, not the residual-digit architecture, explains any gap)
-- same digit-residual structure and training dynamics, fp32-EXACT storage
with a fake-quantize simulation step matching
``fixed_digit_residual_quantize``'s own grid.

``true_multi_digit_dense`` is a DISLDO-vs-ordinary-Adam control (per direct
request): same digit-residual architecture, dense fp32 weights, trained via
a standard external ``AdamOptimizer`` instead of DISLDO's own inline
importance-based update -- catches whether something about DISLDO's OWN
mechanism (not the architecture) explains any gap vs a well-understood,
trusted baseline optimizer.

.. _train_tile_curriculum.arms_stale_scale_bug_isolation:

Stale ``value_scale`` bug isolation: root-cause follow-up to the lr0 collapse
-----------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.arms_stale_scale_bug_isolation``

Root-cause follow-up to ``true_multi_digit_lr0``'s chance-level collapse:
plain ``DISLDOLayer`` (FP4) was found to still call ``disldo_backward`` with
the DEFAULT ``ScalePolicy``/``DeferredScaleWrite=false`` args -- the exact
stale-``value_scale`` bug ``fp8_resync`` fixed for FP8, never applied to
FP4. ``row_4bit_resync``/``true_multi_digit_resync`` apply the
DeferredScaleWrite fix only. ``row_4bit_noscale``/``true_multi_digit_noscale``
force ``value_scale``/``output_scale`` to 1.0 forever -- a direct
real-hardware test of "zero trained scale" (per direct request: "Can we
just add an option to remove the scaling too?"), not just fixing staleness.
For the ``true_multi_digit_*`` pair, each digit is a real
``DISLDOLayerResync``/``DISLDOLayerNoScale`` instead of plain
``DISLDOLayer`` (same architecture as ``true_multi_digit_lr0`` otherwise);
for ``noscale``, the external per-digit ``factor_i`` composition is then the
ONLY scale in play, matching the zero-trained-scale design intent exactly.

Result: ``row_4bit_resync``/``noscale`` both STILL collapsed to chance --
value_scale staleness/presence isn't the explanation after all.

.. _train_tile_curriculum.arms_stochastic_vs_deterministic_rounding:

Stochastic vs. deterministic rounding: the remaining candidate explanation
-------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.arms_stochastic_vs_deterministic_rounding``

Remaining candidate after the scale-staleness dead end above: real FP4 uses
STOCHASTIC rounding on every store (``fp4_quantize_stochastic``, real
per-step noise); the things that DID succeed (``fixed_digit_2``,
``true_multi_digit_fp32_ref``'s ``simulate_quantize``) both use
DETERMINISTIC round-to-nearest. Direct single-variable isolation, single
digit first: ``row_4bit_deterministic`` (same RMSprop scale as plain
``DISLDOLayer``, rounding only changed),
``row_4bit_resync_deterministic``, ``row_4bit_noscale_deterministic`` (the
closest real-hardware match to ``fixed_digit_residual_quantize``'s own
design: zero trained scale AND deterministic rounding together).

``true_multi_digit_deterministic`` extends this to the full 3-stage
residual architecture at ``base=12.0`` -- the project's working default
(exact digit-range tiling, E2M1 math, see
``fixed_digit_residual_quantize``'s docstring). ``true_multi_digit_stochastic``
is otherwise identical (sparse, base=12, n_stages=3) but with STOCHASTIC
rounding -- per direct instruction, now the PREFERRED choice for real runs:
sili_peridot's rank-floor and superposition eval harnesses both found
deterministic rounding gets permanently stuck (never escapes its current
FP4 code once the residual is smaller than one quantization step), while
stochastic genuinely reaches properties deterministic never does (beats the
Eckart-Young rank floor AND the float32 reference at rank>2; the only FP4
arm that ever achieved genuine superposition, i.e. beat the
no-superposition baseline, in ``model/eval_superposition.py``). Kept as a
SEPARATE arm rather than repurposing the deterministic name, matching this
file's own established ``_base4``/``_base6``/``_dense`` convention (direct,
paired comparison points, not silent replacement).

``true_multi_digit_dual`` ("fp4+fp4 dual"): exactly TWO real stochastic-FP4
digits (``n_stages=2``) instead of three, per direct instruction to replace
the fixed-base 3-digit scheme with this as the new precision option used
going forward (real-engine MQAR sweep, task #247). ``base=12``'s
exact-tiling rationale is a PAIRWISE condition between adjacent digits, not
stage-count-dependent, so it carries over unchanged from the ``n_stages=3``
sweep -- not re-tuned here.

``true_multi_digit_noscale_deterministic``: zero trained scale AND
deterministic rounding, at ``base=12``, matching the closest
real-hardware-match convention above but for the full residual architecture.

.. _train_tile_curriculum.arms_base_sweep_and_unseeded_bug:

Base sweep (4/6/12/24) and a real unseeded-preseed bug found mid-sweep
----------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.arms_base_sweep_and_unseeded_bug``

``true_multi_digit_deterministic_base4``: ``base=4.0`` was the ORIGINAL
default, derived from real FP4 (E2M1)'s own worst-case relative rounding
error (~1/4) -- kept as an explicit comparison point now that ``base=12``
is the default. ``true_multi_digit_deterministic_base6``: halfway between
``base=4`` (overlapping digit ranges) and ``base=12`` (exact tiling) --
added per direct request to fill in the sparse 4/12/24 sweep.
``true_multi_digit_deterministic_base24``: the other end of the sweep.

**IMPORTANT bug found mid-sweep**: the single-seed sweep that first picked
``base=12`` (JOURNAL.md 2026-08-10, mean_acc 0.9375 vs base=4's 0.8771) was
run BEFORE a real bug was found and fixed -- ``ToyTileRecurrenceRealFP4``
never passed ``rng=`` down to ``disldo_cls``, so every layer's initial
connectivity/weight values were genuinely unseeded regardless of ``seed``
(confirmed directly: same seed, same command, gave final-step accuracies of
0.70 then 0.65 across two back-to-back runs). That single-seed comparison
only shows "there exists a draw where base=12 wins," not "usually wins" --
needs a proper multi-seed re-run now that construction is actually
reproducible (see ``tests/test_residual_base_sweep.py``) before this
default is fully trusted. Kept as the default in the meantime since the
theoretical argument (exact tiling) is independent of that bug.

``true_multi_digit_deterministic_lr1``/``lr2`` retest ``lr_power`` under
deterministic rounding (JOURNAL.md 2026-08-10 "Test 2"): the
stochastic-rounding-era ``lr0``/``lr1``/``lr2`` sweep found no real
difference between ``lr_power`` values, predicted to be because RMSprop's
own ``eff_lr*g/sqrt(importance)`` self-normalizes almost all of the extra
per-digit ``factor_i`` damping away on its own. Retesting under
deterministic rounding (base=12, matching the confirmed default) to confirm
that prediction still holds now that stochastic noise isn't swamping
everything else.

.. _train_tile_curriculum.arms_dense_connectivity_variants:

Dense-connectivity variants: testing the sparse "echo network" preseed itself
-------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.arms_dense_connectivity_variants``

The ``_dense`` variants (``true_multi_digit_deterministic_dense``,
``_base4_dense``, ``_base6_dense``, ``_base24_dense``,
``true_multi_digit_stochastic_dense``, ``true_multi_digit_dual_dense``) use
fully dense connectivity (every synapse present, loaded straight into
block4 via sili__new's ``load_dense_codes``) instead of the random SPARSE
"echo network" preseed every arm above uses -- per direct request, to test
whether that random-connectivity-draw is itself a significant source of the
seed-to-seed variance seen even at base=12 (std 0.043 across 5 seeds,
JOURNAL.md 2026-08-10), independent of base or bits. Same
``digit_cls``/``n_stages``/``lr_power``/``base`` as their non-dense
counterparts -- ``dense=True`` is the only varied axis, so each is a direct
paired comparison, not a new axis tangled with others.

``true_multi_digit_stochastic_dense`` combines the two winning axes found
separately this session (stochastic rounding beats deterministic for
genuine superposition/rank-floor properties; dense connectivity beats
sparse-echo once instability is fixed) for the first time. This is the
current best-known production combination when paired with
``l1_sparsity_coef=0.05-0.07`` at the training-script level (see
``main()``'s own CLI arg,
``train_tile_curriculum.cli_spectral_norm_vs_l1_sparsity`` below) -- L1
output-sparsity is what makes dense connectivity stable at all (JOURNAL.md
2026-08-13), replacing ``spectral_norm_target``, which is unavailable in
production.

.. _train_tile_curriculum.arms_shared_connectivity_hypothesis:

``true_multi_digit_shared_conn``: forcing digits onto identical connectivity
------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.arms_shared_connectivity_hypothesis``

Direct hypothesis check: each digit's independent preseed/synaptogenesis
means digits' connectivity is essentially disjoint (verified: 0/20, 0/20,
1/20 overlap on a fresh preseed) -- the residual-correction mechanism can
only fire where digits' connectivity actually coincides.
``share_connectivity=True`` forces every digit onto digit 0's exact
``(ptrs, indices)`` at construction.

.. _train_tile_curriculum.maybe_synaptogenesis_k4_design:

``_maybe_synaptogenesis``: real structural growth+pruning, k=4
----------------------------------------------------------------------

*ID:* ``train_tile_curriculum.maybe_synaptogenesis_k4_design``

Real structural growth+pruning (sili's actual ``synap_step``/
``build_probes``/``equalizer_step``, via ``_SparseLayerBase.synaptogenesis``
or ``TrueMultiDigitLayer``'s own per-digit delegate) on every disldo-family
sublayer of the model, ``k=4`` -- the established default elsewhere in the
codebase (``SparseRNNAgent``'s own ``synaptogenesis_k`` default; ``k=64``
was measured to saturate a 1000x1000 layer's connectivity in ONE call,
``k=4`` grows gradually instead). Each sublayer's own already-stored
``_max_row_weights`` cap keeps nnz roughly STABLE over time (grow-and-prune
balance against a fixed target, not unbounded growth) -- per direct
instruction, not a new/growing budget. Dense/Adam-controlled sublayers (no
``.synaptogenesis``, no sparse structure) are silently skipped.

.. _train_tile_curriculum.build_tile_window_not_tiled_correction:

``_build_tile_window``: a real embed vector per tile, not tiled into state_width
-------------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.build_tile_window_not_tiled_correction``

Returns ``[num_tiles, embed_width]`` -- a real ``embed_width`` vector per
tile position (zeros for "nothing here yet", before sequence start), NOT
tiled/repeated into ``state_width``. The model's own ``input_proj`` layer
maps this into the wide recurrent state (see ``ToyTileRecurrenceRealFP4``'s
own docstring for the full correction: this used to be done via
``np.repeat`` here, a misapplication of column-averaging's actual purpose
-- letting a narrow OUTPUT's gradient reach the whole wide state on
readout -- to the input side, which was never what it was for).
``column_neurons`` is kept as an unused, ignored parameter for
backward-compat with existing call sites that still pass it; new callers
should omit it.

.. _train_tile_curriculum.arm_value_bits_approximation:

``ARM_VALUE_BITS``/``estimate_value_bits``: an approximate, relative-only accounting
-------------------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.arm_value_bits_approximation``

Value-bits/weight for each arm's stored weight representation -- used only
for reporting an approximate memory footprint. Index/overhead bits are the
same across arms so they wash out of a *relative* comparison; this is an
approximation, not a byte-exact accounting.

.. _train_tile_curriculum.cli_seq_len_max_out_of_context:

CLI ``seq_len_max``: crossing it turns this into a real out-of-context test
-------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.cli_seq_len_max_out_of_context``

``seq_len_max > NUM_TILES`` is a real out-of-context test: once ``i``
exceeds ``NUM_TILES-1``, ``_build_tile_window``'s window no longer reaches
back to position 0, so recalling ``token[0]`` requires the info to have
survived in ``M_prev`` across ticks the window itself can no longer see.

.. _train_tile_curriculum.cli_peak_lr_per_row_nnz_fp4:

CLI ``peak_lr`` override: compensating for ``lr_per_row_nnz`` at fixed density
-------------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.cli_peak_lr_per_row_nnz_fp4``

``DISLDOLayer.forward``'s default ``lr_per_row_nnz=True`` divides the
effective rate by each row's connection count (``nnz_this_row``) -- real
and necessary when synaptogenesis makes degree vary, a silent crush at
fixed density. fp32 tolerates the crushed rate fine (continuous updates);
FP4 needs a large-enough step to move a value even one quantization level,
so this CLI override compensates directly instead of touching
``lr_per_row_nnz`` itself (which is buried inside ``disldo_cls``'s own
``forward()`` call, not exposed through ``ToyTileRecurrenceRealFP4``).

.. _train_tile_curriculum.cli_o_proj_depth_cascaded_idea:

CLI ``o_proj_depth``: cascaded sublayers vs one wider layer
------------------------------------------------------------------

*ID:* ``train_tile_curriculum.cli_o_proj_depth_cascaded_idea``

``o_proj_depth > 1``: N sequential FP4 sublayers instead of one wider layer
-- a cascaded/residual-quantization-style test of whether composing coarse
stages recovers precision that widening alone doesn't, per direct idea.

.. _train_tile_curriculum.cli_use_synaptogenesis_intent:

CLI ``use_synaptogenesis``: testing discovered vs. preseeded connectivity
------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.cli_use_synaptogenesis_intent``

Real dynamic growth+pruning (``build_probes``+``synap_step``+
``equalizer_step`` via ``_maybe_synaptogenesis``, k=4) every outer step,
instead of the static pre-seeded-only sparsity every arm has used so far
this session -- per direct request, testing whether the residual digits
want some OTHER connectivity pattern discovered via real importance-driven
growth/pruning, distinct from both fully-independent-random and
forced-identical (both already tested, see
``train_tile_curriculum.arms_shared_connectivity_hypothesis``). Default
off, backward-compatible with every existing invocation.

.. _train_tile_curriculum.cli_clip_range_finding:

CLI ``clip_range``: 6.0 beats the original unjustified 2.0
------------------------------------------------------------------

*ID:* ``train_tile_curriculum.cli_clip_range_finding``

The tile-recurrence state's hard clip bound
(``np.clip(M_new_t.data, -clip_range, clip_range)``) was originally picked
at 2.0 without much justification. Direct comparison confirmed 6.0
(matching FP4's own max representable magnitude) wins clearly (mean_acc
0.98 vs 0.75, 3/3 seeds) -- now the default, matching
``ToyTileRecurrenceRealFP4``'s own updated default.

.. _train_tile_curriculum.cli_magnitude_penalty_coef:

CLI ``magnitude_penalty_coef``: independent of ``use_energy`` by design
------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.cli_magnitude_penalty_coef``

Real gradient discouraging large recurrent activation magnitude (see
``ToyTileRecurrenceRealFP4.__init__``'s own docstring) -- default 0.0/off,
backward-compatible. Direct instruction to keep this independent of
``use_energy`` for isolated testing.

.. _train_tile_curriculum.cli_spectral_norm_vs_l1_sparsity:

CLI ``spectral_norm_target``/``l1_sparsity_coef``: the dense-connectivity stability fix
------------------------------------------------------------------------------------------------

*ID:* ``train_tile_curriculum.cli_spectral_norm_vs_l1_sparsity``

``spectral_norm_target`` rescales ``o_proj``'s real output by a persistent,
power-iteration-tracked estimate of its own dominant singular value (see
``ToyTileRecurrenceRealFP4.__init__``'s own docstring) -- the measured root
cause of dense connectivity's instability (spectral radius 1.2 at init,
growing to 1.5+ over training, vs sparse's flat 0.85). None/off by default,
backward-compatible, independent of ``magnitude_penalty_coef``/
``use_energy`` (composable per direct request).

``l1_sparsity_coef`` is the LANDMARK dense-connectivity stability mechanism
(see ``ToyTileRecurrenceRealFP4.__init__``'s own docstring for the full
rationale and JOURNAL.md 2026-08-13) -- reaches mean=1.0000 at coef=0.05 or
0.07, replacing ``spectral_norm_target`` entirely (do not set both).
None/0.0 off by default, backward-compatible.

.. _train_tile_curriculum.rng_seeding_stochastic_and_model_construction:

RNG seeding: two separate uncontrolled-noise bugs, both fixed
----------------------------------------------------------------------

*ID:* ``train_tile_curriculum.rng_seeding_stochastic_and_model_construction``

Real ``DISLDOLayer``-family (fp8/fp8_seeded/fp8_resync/fp8_adamax/rank1/
fp32) arms use stochastic rounding (``fp4quant.hpp``/``fp8quant.hpp``'s
``set_stochastic``) whose RNG is thread-local and, by design, seeded from
the thread id at process start -- NOT controlled by ``seed``, and NOT
reproducible run-to-run without an explicit call. Confirmed directly: the
SAME unchanged binary gave different single-step results across separate
process invocations (0.140625 vs 0.15625 for one stored weight) purely from
this. Without pinning it via ``_cpu.seed_fp4_stochastic_rng(seed)`` (guarded
by ``hasattr`` for older builds), comparisons between arms (or before/after
a C++ change) are confounded by an extra, uncontrolled noise source on top
of ``seed``.

.. code-block:: python

   if hasattr(_cpu, "seed_fp4_stochastic_rng"):
       _cpu.seed_fp4_stochastic_rng(seed)

Separately, ``model_rng`` (a fresh ``np.random.default_rng(seed)``, not the
legacy ``RandomState`` ``rng`` used for tokens/embed_table) is threaded down
by ``ToyTileRecurrenceRealFP4`` to each ``disldo_cls`` layer's initial
connectivity/weight values (see its own docstring for the bug this fixes:
``rng=`` was previously never passed down at all, so every layer's preseed
was genuinely unseeded regardless of ``seed`` -- the same root cause behind
``train_tile_curriculum.arms_base_sweep_and_unseeded_bug``'s single-seed
sweep needing a proper re-run).
