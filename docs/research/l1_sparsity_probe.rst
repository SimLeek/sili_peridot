``l1_sparsity_probe.py`` research notes
==========================================

Companion doc to ``scripts/l1_sparsity_probe.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets alone render invisible on GitHub, frozen code
snippets on real-bug/non-obvious-derivation sections only).

.. _l1_sparsity_probe.module_overview_landmark_result:

Module overview: the L1 output-sparsity landmark result (2026-08-13)
--------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.module_overview_landmark_result``

L1 output-sparsity penalty, applied to all 4 layers (q_proj/k_proj/v_proj/
o_proj) of the ORIGINAL architecture (single ``v_proj`` on the combined
input, matching ``ToyTileRecurrenceRealFP4`` -- NOT the ``v_in_proj``/
``v_state_proj`` split investigated earlier), reaches mean=1.0000 across all
5 seeds at coefficient 0.05 AND 0.07 on the 15000-step out-of-context
curriculum -- the best result of the entire dense-connectivity stability
investigation, matching/exceeding spectral normalization's own 0.8858, with
NO hard rescale of any kind. Found because spectral normalization (and any
hard rescale, including a mean-singular-value-targeting variant) is not
available as a production mechanism -- see ``sili_peridot/JOURNAL.md``'s
2026-08-12/13 entries for the complete investigation, all intermediate
results, and the full methodology (including how this exact result was
independently re-verified via the real training loop after an initial
"too good to be true" suspicion, given this project's repeated pattern of
promising short runs collapsing at full scale).

Full comparison table (all mean accuracy on the 15000-step out-of-context
copy-task curriculum, ``dense_base12``, 5 seeds)::

    mechanism                                          mean    notes
    L1-sparsity alone, orig arch, coef=0.05/0.07       1.0000  BEST
    spectral norm, orig arch, o_proj-only              0.8858  (hard rescale, unavailable in production)
    L1-sparsity + L2-ratio combined, orig arch         0.7333  combining L1 with L2-ratio HURTS
    sparse-echo (no dense connectivity at all)         0.7296
    L2-ratio (split-backward) alone, orig arch         0.3333-0.4667  best ~coef=10, non-monotonic
    L1-sparsity alone, NEW (v_in/v_state split) arch   0.2000-0.3333  87-89% skip rate, unstable
    L2-ratio alone, NEW (split) arch                   0.2667-0.4000
    any o_proj-only soft mechanism, either arch        0.0-0.27  total or near-total collapse

Also supports (not part of the landmark result, added for follow-up
testing): ``all_zero_init`` (only weight matrices zeroed, RMSNorm scales
stay at their 1.0 baseline; see
``l1_sparsity_probe.all_zero_init_importance_seeding``),
``use_energy``/``energy_kwargs`` (EnergyDynamics forced-firing/shutoff --
found the DEFAULT drive saturates to the firing ceiling almost immediately
under zero-init since ``E[|attn_raw|]`` starts at exactly 0, removing the
only force opposing ``drive`` in the continuous dynamics; a much smaller
drive, e.g. 1e-4 to 1e-5, is needed to keep firing rare rather than
continuous -- see JOURNAL.md), and ``scale_clip_max`` (O(w) value_scale/
output_scale clipping, task #165).

**Status**: NOT YET merged into the shared ``model/toy_tile_precision_models.py``
(``ToyTileRecurrenceRealFP4``) or wired into ``train_tile_curriculum.py``'s
CLI -- this file remains the reference/reproduction implementation for the
validated result above. Zero-init + energy_rl + L1-sparsity is a promising
but NOT yet full-scale-validated follow-up (short-run signal only).

.. _l1_sparsity_probe.split_backward_delivery_mechanism:

L1-sparsity mechanism: split-backward delivery avoids RMSprop dilution
--------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.split_backward_delivery_mechanism``

Mechanism: ``coef * mean(|layer_output|)`` via a "split-backward" delivery
-- a SECOND, independent ``layer.forward(..., damp_by_importance=False)``
call per layer gives the L1 term its own undamped gradient path, avoiding
dilution by DISLDO's own RMSprop-style per-synapse update (which shares
state with -- and gets swamped by -- the much larger main-task gradient if
delivered through the normal damped path; confirmed via direct
gradient-magnitude measurement, ~1731x ratio). See JOURNAL.md for the full
RMSprop-dominance derivation.

The same split-backward call also underlies ``_ratio_penalty_split``
(L2-ratio) and, on a linear layer, ``d(sum|Wx|)/dW = sign(y) outer x``,
which directly pushes weight magnitude down along active directions -- this
is why the output-activation L1 term acts as a workable proxy for genuine
weight-magnitude L1 (DISLDO stores weights as quantized codes internally,
not an exposed differentiable tensor, so direct weight-L1 isn't cleanly
buildable).

.. _l1_sparsity_probe.coefficient_sensitivity_goldilocks:

Coefficient sensitivity: a genuine Goldilocks zone, not a monotonic dial
--------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.coefficient_sensitivity_goldilocks``

Coefficient sensitivity is real and sharp::

    l1_coef=0.01/0.02  mean=0.13  skip_rate=27-30%  (too weak, unstable)
    l1_coef=0.03       mean=0.80
    l1_coef=0.05       mean=1.00  (PERFECT)
    l1_coef=0.07       mean=1.00  (PERFECT)
    l1_coef=0.10       mean=0.67  (too strong, degrades again)

.. _l1_sparsity_probe.empty_init_synaptogenesis_design:

``empty_init``: genuine zero-weight-init vs ``all_zero_init``'s dense hack
--------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.empty_init_synaptogenesis_design``

``empty_init`` is the GENUINE zero-weight-init design -- every layer starts
with literally zero connections (not ``all_zero_init``'s dense-grid-
preloaded-with-0 hack, a different synthetic arm -- see
``sili/sparse_rnn.py``'s ``_preseed_empty`` docstring). Real synapses are
created by ``synaptogenesis()`` (``build_probes``/``synap_step``/
``equalizer_step``, called every step in ``run()`` -- see its own "meant to
be called every online step" docstring), each starting with
``weight=0``/``importance=probe_score`` (a REAL, activity-derived nonzero
importance, not a hardcoded one) -- confirmed via isolated smoke test to
actually escape 0 given real training (unlike ``all_zero_init``'s dense
simultaneous-zero deadlock, since only a HANDFUL of synapses per row need to
escape, not the entire row/column at once). Mutually exclusive with
``dense``/``all_zero_init`` in practice (not enforced) -- doesn't make sense
combined with either.

``synaptogenesis_all()`` is a no-op unless ``empty_init`` (nothing to grow
into on a preseeded layer; ``TrueMultiDigitLayer.synaptogenesis`` is defined
regardless, but calling it on an already-full-capacity preseeded layer just
churns probes for no effect, so skip entirely rather than pay the cost for
nothing).

.. _l1_sparsity_probe.lr_per_row_nnz_double_damping_bug:

``lr_per_row_nnz``: a real double-damping bug from pre-dividing lr in Python
-------------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.lr_per_row_nnz_double_damping_bug``

``lr_per_row_nnz`` is passed straight through to every layer's ``.forward()``
call as RAW lr, with no Python-side pre-division. It controls ONLY whether
the per-synapse weight-CODE update is additionally damped by the row's
live-connection count -- ``value_scale``'s own gradient (a single scalar per
row, summed across every live synapse in it) is ALWAYS normalized by that
count regardless of this flag, unconditionally, inside the C++ layer
(``linear_disldo.hpp``'s ``scale_eff_lr``) -- that's mandatory averaging of a
summed scalar gradient, not a policy choice.

An EARLIER version of this code tried to replicate ``True``'s division from
the Python side by pre-dividing lr (``qkv_lr = lr/state_width``) before
calling ``.forward(..., lr_per_row_nnz=False)``. This was a real bug, not a
legitimate approximation: since ``scale_eff_lr`` unconditionally divides
whatever lr it's handed by ``nnz_row``, pre-dividing caused ``value_scale``'s
adaptation to be damped TWICE (once by the Python pre-division, once again
by the library's own always-on normalization), crippling ``value_scale`` to
~1/32 of its correct rate. Confirmed directly: ``effective_lr`` (the code
update) was bit-identical to ``True`` at correction=1.0 (verified via a
debug print showing ``nnz_row=32`` exactly, and a 200-step run with zero
diff across every weight/importance/index array) -- yet real training still
diverged structurally within a few thousand steps once ``value_scale``'s
under-adaptation let the two arms' true (scale-multiplied) weight magnitudes
drift apart far enough to cross different FP4 rounding/promotion
thresholds. No single scalar correction on the pre-divided lr could ever fix
this: ``effective_lr`` needed correction=1, ``scale_eff_lr`` needed
correction=32, same shared parameter.

.. code-block:: python

   # WRONG (earlier version): pre-divide lr in Python to approximate
   # per-row-nnz damping, then disable the library's own damping.
   qkv_lr = lr / state_width
   out = layer.forward(x, qkv_lr, lr_per_row_nnz=False)
   # scale_eff_lr STILL unconditionally divides by nnz_row internally,
   # so value_scale's adaptation gets damped TWICE -- crippled to ~1/32
   # of its correct rate, causing structural divergence within a few
   # thousand steps.

   # Fix: never touch lr in Python. Raw lr through to the library lets
   # its own correct, intentional split (code update: conditional on
   # lr_per_row_nnz; value_scale update: unconditional) do its job.
   out = layer.forward(x, lr, lr_per_row_nnz=True)

The fix is to not pre-divide at all -- raw lr through to the library lets
its own correct, intentional split (code update: conditional; value_scale
update: unconditional) do its job in both branches. This same convention is
reused verbatim in ``step()`` for ``qkv_lr``/``o_lr``/``lmhead_lr``.

.. _l1_sparsity_probe.stochastic_qkv_o_deterministic_rounding_floor:

``stochastic_qkv``/``stochastic_o``: deterministic rounding's sub-threshold floor
--------------------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.stochastic_qkv_o_deterministic_rounding_floor``

``stochastic_qkv``/``stochastic_o`` use stochastic-rounding storage
(``DISLDOLayer``) instead of deterministic (``DISLDOLayerDeterministic``)
for the selected layers -- ``lm_head`` always stays deterministic (its
``dy`` already comes for free from cross-entropy, the terminal loss, never
structurally zero).

Per direct request: under all-zero-init, deterministic rounding's
fresh-decode-then-requantize every call means any single update below half
the FP4 step (0.25, the code-0 -> code-0.5 gap) is discarded every time, so
sparse/small repeated pushes never accumulate -- confirmed directly
(``weights_vals`` stayed exactly 0 for hours of steps despite EnergyDynamics
genuinely firing). Stochastic rounding lets each sub-threshold delta round
up with real (if small) probability, so repetition compounds in expectation
instead of vanishing.

``o_proj`` was originally believed not to need this (it has no
``gaussian_attention``-style nonlinear escape hatch) -- CORRECTED: once
``o_proj`` gets its own direct energy wrap on ``raw`` (its own output,
giving it a legitimate nonzero ``dy``), it's in exactly the same "real
signal exists but gets rounded away" situation q/k/v were in, so it needs
the same fix. This project's own prior findings show stochastic rounding
can hurt final accuracy broadly, so it's kept opt-in per layer rather than
always-on.

.. _l1_sparsity_probe.uniform_energy_tap_design:

Energy taps: applied uniformly to every neuron-producing tensor, not curated
---------------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.uniform_energy_tap_design``

Energy is applied as a plain per-tap OP (``_apply_energy(state, tensor,
...)``, a bare function -- not a bespoke wrapper class per layer) to EVERY
neuron-producing tensor in the model -- input, every hidden layer, and the
output -- uniformly, not a hand-curated subset. Per direct correction:
reasoning case-by-case about which specific layer's gradient path "needs"
energy is exactly the kind of fragile, easy-to-miss analysis that caused an
earlier o_proj/lm_head cascade stall (see JOURNAL.md -- a curated 5-tap set
silently left o_proj and lm_head permanently stuck under ``empty_init``).
Since the number of neurons (layer widths) is cheap relative to the number
of parameters, wrapping every tap uniformly costs little and removes the
need to re-derive "does this specific layer need it" every time the
architecture changes.

Each tap (``_ENERGY_TAPS = ("input", "q", "k", "v", "attn", "raw",
"logits")``) gets its OWN independent ``EnergyDynamics`` instance (own
energy/steps_since_fired state). ``_apply_energy(None, tensor, ...)`` is
already a no-op pass-through, so ``step()`` can call the op unconditionally
at every tap regardless of ``use_energy``, rather than branching.

``EnergyDynamics``' own exploration-noise draw is unseeded global
``np.random`` unless an explicit ``rng=`` is given (fixed in sili__new,
confirmed a real run-to-run non-reproducibility bug otherwise) -- each
energy instance derives one from this model's own seed stream, same
convention as q/k/v/o_proj, unless the caller already supplied their own.

.. _l1_sparsity_probe.fire_wake_gradient_lr_rescaling:

``fire_wake_gradient``: disentangling push magnitude from the lr schedule
-------------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.fire_wake_gradient_lr_rescaling``

``fire_wake_gradient``'s EFFECT on the actual weight update is ``delta =
-effective_lr*g/(sqrt(ci)+eps)``, i.e. proportional to whatever ``lr``
happens to be THIS call (``lr_schedule``'s warmup/decay) -- a fixed
``fire_wake_gradient`` value therefore crosses the FP4 rounding threshold
differently depending on where training is in the schedule, entangling "how
big a push" with "what step number".

Disentangled here (not in ``EnergyDynamics`` itself -- this isn't part of
``forward()``'s own contract, it's purely a caller-side calibration
concern): pull the CONFIGURED ``fire_wake_gradient`` out of
``energy_kwargs`` before constructing the energies (so each starts with
``fire_wake_gradient=None``), then in ``step()`` rescale it by ``1/lr``
every call and assign it directly onto each energy instance right before
use -- the boost then represents a roughly LR-INDEPENDENT target push
magnitude.

.. _l1_sparsity_probe.all_zero_init_importance_seeding:

``all_zero_init``: why importance must be seeded nonzero, not zero
--------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.all_zero_init_importance_seeding``

Weight codes are set to 0 (value 0.0), importance codes to 1 (NOT 0):
``load_dense_codes(zeros, zeros)`` would get every synapse pruned to 0 live
entries by ``block4_load_dense``'s ``maybe_compress`` step
(``block4_count_live`` tests the packed ``weight|importance<<4`` byte
against exactly 0 -- both codes 0 means every packed byte is 0), and
``disldo_backward``'s block4 path then skips every row (``nnz_row==0``)
forever, independent of gradient magnitude. Not a library bug -- a synapse
that legitimately decays to weight=0 AND importance=0 during real training
is reasonable to prune -- but it means intentional zero-VALUE init must
seed importance nonzero to stay live.

``zero_init_importance_code`` default 1 is the bare minimum to survive the
initial ``maybe_compress`` call -- callers combining this with energy_rl
(whose gating can silence a neuron for stretches, decaying its synapses'
``g`` toward 0 and thus their RMSprop-style importance accumulator back
toward 0 too) may want a larger starting buffer; sweep via this param
rather than assuming 1 is always enough.

Neuron-level parameters (``input_ln``/``state_ln``, ``centers``/
``log_sigmas``) stay at their normal baseline -- zero-init is for SYNAPSES
(weight matrices) specifically, not neuron-level gain/threshold params.

.. _l1_sparsity_probe.clip_log_sigmas_runaway_investigation:

``clip_log_sigmas``: a real runaway found, but not the cause it was hoped to be
--------------------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.clip_log_sigmas_runaway_investigation``

Bounds the Gaussian-attention kernel width's log-scale parameter. Found via
direct investigation (2026-08-12,
``scripts/debug_energy_log_sigmas_runaway.py``): under ``use_energy=True``,
``log_sigmas`` drifts monotonically negative and UNBOUNDED from its
zero-init start, with no sign of leveling off. Symmetric bound
(``|log_sigma| <= max_val``), matching ``clip_scales``' single-magnitude
style: ``sigma=exp(log_sigma)`` stays in ``[exp(-max_val), exp(max_val)]``.

CORRECTED, tested directly with ``max_val=2.0`` on the same 5-seed sweep
used for ``baseline_energy``: this does NOT fix the ~47% gradient-skip rate
(46.76% unclamped vs 47.56% clamped, essentially unchanged; mean accuracy
0.1333 vs 0.2667, within this project's own established run-to-run noise
range). The unbounded ``log_sigmas`` drift is a real, independently-worth-
fixing problem, but it is NOT (or at least not the sole) cause of
``baseline_energy``'s instability -- the actual source is still
unidentified as of this writing.

.. _l1_sparsity_probe.lm_head_l1_deadlock_fix:

``lm_head``'s L1 term: closing a separate, un-instrumented deadlock
------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.lm_head_l1_deadlock_fix``

``lm_head`` previously had NO L1 term -- under ``all_zero_init`` this was a
separate, un-instrumented deadlock: ``dL/d(pooled) = W_lmhead^T @
grad_logits`` is exactly 0 whenever ``W_lmhead=0``, so main-task gradient
alone could never reach it until ``pooled`` was already nonzero from
upstream. This term gives ``lm_head`` the same direct, pooled-independent
escape route q/k/v/o_proj already have.

.. _l1_sparsity_probe.evaluate_methodology_and_resolution:

``evaluate()``: forward-only held-out check, and why it replaced ``last_accs``
--------------------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.evaluate_methodology_and_resolution``

Post-training capability check: ``n_eval`` FRESH, independently-drawn
sequences at full in-context length (``NUM_TILES``), forward-only -- never
calls ``.backward()``/``opt.step()``, so nothing trains (``forward_dense``
has no side-effect training path; only ``backward_dense``'s inline update
does, per its own docstring). Reports plain ``correct/n_eval``.

Replaces relying on ``run()``'s own in-training ``last_accs`` sampling for
regression comparisons -- that samples exactly ONE sequence's correctness
every 200 steps, and the run config only ever averaged the LAST 3 of those,
so every reported "mean" could only ever be one of ``{0, 1/3, 2/3, 1}`` per
seed. Confirmed directly this coarse resolution was large enough to make a
real regression indistinguishable from sampling noise (5 successes out of
15 total training-time samples swinging the reported mean by over 30
points). ``n_eval=100`` gives up to 101 distinct values instead of 4.

Uses the SAME ``embed_table`` the model was trained against (recomputed
deterministically from ``seed``, not threaded through ``run()`` -- the
embedding table isn't itself trainable, it's the fixed random
token->vector lookup the model's WEIGHTS were calibrated to use). Sequences
are drawn from a DIFFERENT RNG stream than training's own (seed offset), so
these are genuinely held-out draws, not a replay of the training
curriculum's own sequence order.

``verbose`` prints the last 5 (prediction, target) token-id pairs seen
(across the whole ``n_eval`` run, most-recent-last) -- per direct request:
a bare accuracy number can't distinguish "the model is degenerate/guessing
a narrow set of tokens" from "these particular held-out sequences happened
to skew toward one target token" (a real confound with a small ``n_eval``
and ``VOCAB`` tokens to choose from). Does NOT change the return value.

.. _l1_sparsity_probe.aux_accumulate_every_position:

``run()``: accumulating aux across every position, not just target positions
--------------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.aux_accumulate_every_position``

Accumulate aux (L1-sparsity + energy_aux_loss) across EVERY tile position,
not just target ones -- per direct correction, these were always meant to
backprop at every position (energy especially: ``fire_wake_gradient``'s
injected term only ever mattered if ``backward()`` actually ran on the
specific position that happened to fire, which is essentially never true
when only the LAST position is ever a target -- confirmed directly,
boosting ``fire_wake_gradient`` 25x made no difference at all, because the
aux computed at every non-target position was simply discarded before this
fix). One accumulate-then-backward per outer step (not one backward per
position) -- fewer, cheaper ``backward()`` calls than the alternative of
calling ``backward()`` at every position.

.. _l1_sparsity_probe.aux_averaged_by_seq_len:

``run()``: averaging total_aux by seq_len, confirmed necessary at scale
------------------------------------------------------------------------------

*ID:* ``l1_sparsity_probe.aux_averaged_by_seq_len``

``total_aux`` is averaged by ``seq_len`` (not summed raw) -- otherwise
``l1_sparsity_coef``'s effective per-step pressure grows with the
curriculum's own ``seq_len`` (2->4), silently over-regularizing as training
progresses. Averaging keeps the coefficient's meaning stable regardless of
how many positions happened to run this step, so it doesn't need re-tuning
every time ``seq_len`` changes (confirmed necessary directly: raw summing
regressed baseline 0.8667->0.6667 at real 5-seed scale). The target
cross-entropy term is NOT averaged -- it's always exactly one term
regardless of ``seq_len``, nothing to normalize there.
