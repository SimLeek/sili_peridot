``toy_precision_models.py`` research notes
===============================================

Companion doc to ``model/toy_precision_models.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below); this
doc links back to source by class/function name. See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers, frozen code snippets on
real-bug/rejected-alternative sections).

.. _toy_precision_models.module_overview:

Module purpose: a precision-MATCHED Adam-vs-importance comparison
-------------------------------------------------------------------------

*ID:* ``toy_precision_models.module_overview``

Answers a question deliberately deferred earlier: does this project's own
importance/row-scale training (``sili.sparse_rnn.DISLDOLayer``) beat a
standard, well-tuned optimizer (Adam), when BOTH sides represent weights at
the SAME FP4-level precision -- not fp32 vs FP4, which would just repeat
the precision/optimizer confound this project's JOURNAL.md already spent
real effort disentangling (its two isolation controls). See the approved
plan (``fuzzy-plotting-starlight.md``) for the full design rationale.

``use_energy`` is a SEPARATE, orthogonal toggle on BOTH model classes, not
baked into one arm only (per ``feedback_do_science_correctly`` memory): an
earlier version of this module gave ``EnergyDynamics`` to the real-FP4 arm
only, confounding "optimizer" with "energy's own added noise/aux-loss."
``EnergyDynamics`` wraps ACTIVATIONS (the attention output), not weight
training, so it attaches identically regardless of which linear-layer type
is used -- each layer builds its OWN fresh ``EnergyDynamics`` instance
(``_toy_scale_energy()``) when ``use_energy=True``, since its ``energy``
running state is per-instance and must not be shared across layers/models.

``fake_quantize_fp4``/``ArtificialFP4Linear`` reproduce the REAL FP4
representation (``sili/lib/headers/fp4quant.hpp``'s 16-level table,
``sili/sparse_rnn.py``'s ``per_row`` calibration formula
``scale=max(|row|)/6.0``) as a straight-through fake-quantization op on top
of an ordinary Adam-trainable fp32 leaf -- NOT a reimplementation of
DISLDOLayer's own (also gradient-trained) row-scale, which stays
real/untouched on the other arm. ``ToySmallTransformerRealFP4`` uses
DISLDOLayer directly (per ``feedback_importance_is_already_the_optimizer``
memory: its inline C++ update during ``backward()`` already IS an
optimizer -- no external ``AdamOptimizer.step()`` call on its own big
weight matrices).

``AdamRowScaleDISLDOLayer``/``ToySmallTransformerRealFP4RowScaleAdam``: per
direct correction, individual FP4 weight VALUES should keep importance as
their training signal (believed to be the right mechanism for
synaptogenesis/pruning), but the coarser per-row ``value_scale`` -- one
float per row, not per weight -- is cheap to give its own Adam-style
adaptive normalization on top of importance's own raw update, without
replacing importance's role anywhere. See
``adam_row_scale_disldo_layer.proxy_gradient_approximation`` below for the
exact mechanism and its one real approximation.

.. _toy_scale_energy.calibration:

``_toy_scale_energy``: why the toy scale needed its OWN EnergyDynamics tuning
-----------------------------------------------------------------------------------

*ID:* ``toy_scale_energy.calibration``

``sili_block.default_window_energy`` is explicitly a placeholder calibrated
for full-model-scale windows ("real tuning is Phase 5's job") -- checked
directly against this toy scale (``HIDDEN=12``, ``T*HIDDEN~=60``) rather
than assumed to transfer: its own ``aux_loss`` grew unbounded during
training (5.5 -> 17.9 over 300 steps) and total loss diverged. This config
instead matches sili__new's own small-scale ``EnergyDynamics`` test
convention (``tests/unit/python/test_sparse_rnn_cell.py``'s
``TestEnergyDynamicsKeptIndices``, h sizes 20-64 -- comparable to this toy
model's own ``T*hidden``), verified directly to behave far better here
(``aux_loss`` stays 0.005 -> ~0.3, loss reaches a real minimum instead of
diverging).

.. _fake_quantize_fp4.straight_through_qat:

``fake_quantize_fp4``: real 15-level FP4 table, straight-through backward
-------------------------------------------------------------------------------

*ID:* ``fake_quantize_fp4.straight_through_qat``

``FP4_TABLE`` is a verbatim copy of ``sili/lib/headers/fp4quant.hpp``'s
``FP4_TABLE`` (one of the 16 four-bit codes is repurposed as NaN there,
leaving 15 usable values here): ``{0, .5, 1, 1.5, 2, 3, 4, 6}`` and their
negatives (0 not duplicated).

``fake_quantize_fp4`` forward rounds each row of ``w`` to this real
15-level table using the same per-row ``scale=max(|row|)/6.0`` calibration
``sili/sparse_rnn.py``'s ``per_row`` mode uses; backward is identity
(standard QAT practice -- ``w`` itself, not the quantized output, is the
thing Adam actually trains). Same custom-``_bwd``-closure pattern as
``toy_recall_models.cross_entropy_sum``/``rmsnorm_tensor`` -- no
sili__new change needed.

.. _real_fp4_layer.inline_training:

``_RealFP4Layer``/``ToySmallTransformerRealFP4``: inline weight training and the aux_loss convention
---------------------------------------------------------------------------------------------------------

*ID:* ``real_fp4_layer.inline_training``

DISLDOLayer's own big weight matrices train inline during ``backward()``
(``learning_rate``-driven, no external optimizer -- see
``toy_precision_models.module_overview`` above); ``input_ln``/``post_ln``
are plain ``Tensor`` leaves, trained by a small separate
``AdamOptimizer`` step over ``ToySmallTransformerRealFP4.parameters_for_optimizer()``
-- these are the ONLY params that optimizer ever sees. ``disldo_cls`` lets
``ToySmallTransformerRealFP4RowScaleAdam`` reuse this exact layer shape
with ``AdamRowScaleDISLDOLayer`` in place of plain ``DISLDOLayer``.

``use_energy`` toggles ``EnergyDynamics`` on the attention output (matching
``model/tile_recurrence.py``'s own ``apply_tile_step`` pattern) --
independent of the optimizer/precision question this class otherwise
tests (an earlier version of this class always used energy, confounding
the two).

``forward()`` returns ``(logits, aux_loss)`` -- ``aux_loss`` is ``None``
when ``use_energy=False``, else must be added to the task loss before the
single shared ``.backward()`` call: that one call both trains
DISLDOLayer's weights (inline, via the gradient reaching each layer's own
output) AND accumulates gradient for the plain leaf parameters
(``parameters_for_optimizer``).

.. _adam_row_scale_disldo_layer.proxy_gradient_approximation:

``AdamRowScaleDISLDOLayer``: an observed-delta proxy gradient, not the true pre-damping one
---------------------------------------------------------------------------------------------------

*ID:* ``adam_row_scale_disldo_layer.proxy_gradient_approximation``

A real ``DISLDOLayer``, with its per-row ``value_scale`` re-normalized via
Adam AFTER each ``backward()`` -- individual FP4 weight VALUES are
completely untouched (importance keeps its existing role there entirely);
only the coarser row-scale (one float per row) gets an adaptive step on
top.

Mechanism: ``SparseLinearLayer.get_value_scale(row)``/
``set_value_scale_raw(row, scale)`` are real pybind accessors
(``cpu_backend.cpp:1082-1097``) -- checked directly, not assumed.
``forward()`` snapshots ``value_scale`` for every row BEFORE the caller's
``loss.backward()`` runs (which is when DISLDOLayer's own
``backward_dense`` applies its raw importance-damped update to
``value_scale``, inside the ``_bwd`` closure); once that raw update has
been applied, this treats the OBSERVED delta (``after - before``) as a
proxy gradient signal (``grad ~= -raw_delta / learning_rate``) and runs
one standard Adam moment-update step over that per-row vector, overwriting
``value_scale`` with the Adam-normalized step instead of the raw one.

This is an approximation, documented as such rather than silently assumed
correct: it re-normalizes an ALREADY-APPLIED delta rather than intercepting
the true pre-damping gradient before DISLDOLayer's own importance-damping
is applied to it -- a reasonable experimental probe for whether adaptive
normalization on just the row-scale helps, not a claim that this is
identical to running real Adam on the underlying gradient.

.. _adam_rank1_disldo_layer.output_scale_activation:

``AdamRank1DISLDOLayer``: why __init__ must call set_output_scale_raw once
-------------------------------------------------------------------------------

*ID:* ``adam_rank1_disldo_layer.output_scale_activation``

``AdamRowScaleDISLDOLayer``'s mechanism (see
``adam_row_scale_disldo_layer.proxy_gradient_approximation`` above),
extended from row-only to rank-1 (row ``value_scale`` AND column
``output_scale``, both Adam-normalized independently) -- individual FP4
weight VALUES stay completely untouched, same as
``AdamRowScaleDISLDOLayer``. Same approximation documented there:
re-normalizes an ALREADY-APPLIED delta, not the true pre-damping gradient.

``output_scale`` is real (``get_output_scale(col)``/
``set_output_scale_raw(col, scale)``, ``cpu_backend.cpp:1105-1125``) but --
checked directly, not assumed -- the pybind docstring states it only
becomes gradient-trainable in ``backward_dense()`` "after calling
``set_output_scale_raw`` at least once." ``__init__`` calls
``set_output_scale_raw(c, 1.0)`` for every column once (the documented
default value) specifically to activate that -- without it, ``output_scale``
would never move at all and there'd be nothing for the Adam step to
re-normalize.

.. _rankn_fake_quantize.magnitude_bucketed_columns:

``rankn_fake_quantize``: magnitude-bucketed columns, and why additive residual scale-fitting is a dead end
-------------------------------------------------------------------------------------------------------------------

*ID:* ``rankn_fake_quantize.magnitude_bucketed_columns``

Generalizes ``rank1_fake_quantize``'s single shared ``row_scale``/
``col_scale`` pair to ``rank`` independently-fit column-scale profiles, one
per row-magnitude bucket (rows bucketed by their own max ``|w|`` into
``rank`` equal-count quantile groups -- deterministic sort+split, no
iterative clustering).

An additive residual decomposition (fit rank-1, subtract, re-fit the
leftover, matching e.g. matching-pursuit/greedy-SVD) was the first thing
tried and does NOT work here, verified by hand before writing this:
``rank1_fake_quantize``'s envelope is a strict MAX-COVER (its
alternating-max-fit guarantees ``row_scale[r]*col_scale[c] >= |v|`` for
every synapse in the row/col, by construction of ``row_scale`` itself
being a row max) -- the residual after subtracting it is ``<= 0``
everywhere, so a second additive term has nothing left to refine. This
matters beyond being a dead end: an envelope is exactly what a real N-bit
fixed-point scale must be (never let a stored value exceed what ``levels``
codes can represent) -- an approach that doesn't preserve the cover
property would be simulating something real hardware couldn't actually do.

Bucketing rows by magnitude is the degree of freedom rank-1 alone can't
express: a single shared ``col_scale`` must cover the worst row sharing
that column even when most rows sharing it are far smaller -- every
small-magnitude row wastes precision matching a big outlier row it happens
to share a column with. Splitting rows into magnitude buckets lets each
bucket fit its own column envelope against only its own peers. Measured
effect is real but modest (checked on a synthetic bimodal-magnitude case:
rank-2 tightened the small-magnitude bucket's mean envelope/|value| ratio
by ~22%, the large-magnitude bucket unchanged, as expected) -- it is
bounded by how much true row/col scale correlation the data has, not a
free lunch, and higher rank than the number of genuinely distinct
magnitude regimes in the data won't keep helping.

Reduces EXACTLY to ``rank1_fake_quantize`` when ``rank=1`` (single bucket
containing every row = the identical 3-pass alternating fit).

.. _residual_fake_quantize.true_rvq_vs_rankn:

``residual_fake_quantize``: true residual VQ, refining the ROUNDING ERROR not the envelope
---------------------------------------------------------------------------------------------------

*ID:* ``residual_fake_quantize.true_rvq_vs_rankn``

True residual/cascaded quantization -- matching neural-audio-codec RVQ
(e.g. EnCodec/SoundStream): quantize ``vals`` via ``rank1_fake_quantize``
at ``bits_per_stage``, then quantize the ROUNDING RESIDUAL
(``vals - q1``) with a FRESH rank1 envelope fit to the residual's own
(much smaller) dynamic range, repeat ``n_stages`` times, sum every stage
to reconstruct.

NOT the same as ``rankn_fake_quantize``'s documented dead end (see
``rankn_fake_quantize.magnitude_bucketed_columns`` above): that attempt
tried to refine the rank1 SCALE ENVELOPE itself (``row_scale*col_scale``,
a max-cover bound -- provably nothing left to subtract a second time,
since the envelope already upper-bounds every ``|v|`` in its row/col by
construction). This instead refines the quantized VALUE's rounding error,
an unrelated, always-nonzero quantity bounded by half a quantization step
-- the actual mechanism real residual vector quantization exploits, and
genuinely untested before this.

Total cost: ``n_stages * bits_per_stage`` bits/weight (plus ``n_stages``
independent row/col scale pairs -- small, ``O(rows+cols)`` overhead per
stage) -- e.g. ``n_stages=2``, ``bits_per_stage=4`` is 8 bits/weight total,
a fair, apples-to-apples comparison against
``rank1_fake_quantize(bits=8)``'s single 8-bit code.

.. _fixed_digit_residual_quantize.base_and_e_shared_derivation:

``fixed_digit_residual_quantize``: zero-scaling-vector digits, and the base=12 tiling result
---------------------------------------------------------------------------------------------------

*ID:* ``fixed_digit_residual_quantize.base_and_e_shared_derivation``

Zero-scaling-VECTOR residual quantization, literal closed-form
digit-place-value construction per direct design discussion:
``fp(4n) ~= e_shared * sum_i digit_i * base**-i``. NO row/col fit anywhere
(contrast ``residual_fake_quantize``'s own per-stage
``rank1_fake_quantize`` calls, and ``rank1_8bit``'s trained
``value_scale``/``output_scale``) and NO per-call data-dependent
computation either (contrast even a fresh-every-step global max like
BitNet/XNOR-Net use) -- every stage's step size is a FIXED constant chosen
before any data is seen, matching the literal "maybe choose/learn B or
e_shared, but even then I'm not sure that's needed" framing directly.

``base``: ratio between consecutive residual stages' resolution. The
mantissa-derived value (real sili FP4/E2M1's own worst-case relative
rounding error ``~= 1/2**(mantissa_bits+1) = 1/4``, i.e. ``base=4``)
substantially OVERLAPS each stage's representable range with the one
before it. Default 12.0 instead: the value where digit i+1's ceiling
lands exactly on digit i's floor (exact tiling, zero overlap, zero gap),
confirmed the real winner on ``TrueMultiDigitLayer``'s out-of-context
curriculum:

.. code-block:: text

   base=12   mean_acc 0.9375  (plateaued, lowest variance)
   base=4    mean_acc 0.8771
   base=24   mean_acc 0.70    (still not converged at 15000 steps, inconclusive)

See ``JOURNAL.md`` 2026-08-10 and the ``project_hybrid_precision_plan``
memory for the full sweep.

``e_shared``: a single FIXED scalar (not a per-row/per-col vector, not
gradient-trained, never updated after being chosen) applied once to bring
the whole layer's values into the digit format's representable floor --
has nothing to go stale relative to, since it never changes after
construction. Real FP4 alone only covers ``~[0.5, 6]`` before hitting its
floor, and typical weight init (``~1/sqrt(fan_in)``) sits well below that
-- this is the ONE thing a pure residual stack genuinely cannot fix on its
own (each stage only refines PRECISION within the range the previous stage
already covers, it can never extend the floor downward), so this
parameter stays even in the otherwise fully zero-scale design. Default 1.0
(no rescaling) -- real usage should derive this once from initial weight
statistics at construction, never touch it again.

.. _true_multi_digit_layer.independent_digit_architecture:

``TrueMultiDigitLayer``: genuinely separate digits, digit_cls choice, and the lr_power correction
------------------------------------------------------------------------------------------------------

*ID:* ``true_multi_digit_layer.independent_digit_architecture``

Genuinely SEPARATE, independently-trained residual digit layers -- NO
hidden fp32 shadow across digits (contrast
``QuantizedDISLDOLayer32(scheme="fixed_digit_residual")``, which trains
ONE fp32 accumulator and only discretizes it for STORAGE after the fact --
a real hardware implementation has no room for a hidden full-precision
shadow of every weight). Each digit is its OWN real ``digit_cls`` instance,
contributing ``base**-i`` of the final combined output (``i=0`` is the
coarsest/first digit) -- the ``Tensor`` class's own ``*``/``+`` autograd
ops chain the gradient back into each digit's own real ``backward()``
automatically, no manual backward-wiring needed.

``digit_cls`` -- per direct correction, DISLDO's real FP4 codec should be
the PRIMARY test here (quantization is the whole point), not a fp32-backed
simulation: default ``DISLDOLayer`` (real 4-bit E2M1, matching sili's
actual production codec) needs NO extra Python-side quantization step at
all -- real ``disldo_backward`` already stores FP4 natively every update,
so ``simulate_quantize`` stays ``False`` for it. ``DISLDOLayer32`` (fp32
backend) is kept as an optional REFERENCE/ceiling arm -- same
digit-residual architecture and training dynamics, but exact storage,
isolating "does the residual-DIGIT architecture itself work" from "does it
survive real FP4 storage" (this project's own long-standing
precision-isolation-control convention) -- for that arm only, pass
``simulate_quantize=True`` to fake-quantize to a fixed grid after each step
(matching ``fixed_digit_residual_quantize``'s own grid, via
``_quantize_raw_digit_inplace``); passing it ``True`` for an
already-real-FP4 ``digit_cls`` would just double-round for no reason.

Direct test of whether LATER (finer) digits need their OWN,
separately-scaled-down effective learning rate to stay stable, per direct
discussion (found first empirically: ``fixed_digit_residual`` at 3+ stages
needed roughly HALF the overall peak LR to stop degrading -- more digits
remove the implicit noise-filtering coarse rounding was providing, letting
per-step gradient noise accumulate unchecked).

Corrects an earlier mislabeling: the combination ``out_i * factors[i]``
ALREADY multiplies the gradient reaching digit i by ``base**-i`` via the
ordinary chain rule (``Tensor.mul``'s own backward, verified directly) --
BEFORE ``eff_lr`` is even applied. So ``lr_power=0`` (uniform nominal
``learning_rate`` passed to every digit) is already naturally
chain-rule-scaled, not an unscaled "naive" baseline. ``lr_power`` sets an
EXTRA multiplicative reduction on top of that natural scaling:
``eff_lr = learning_rate / base**(lr_power*i)`` -- ``lr_power=1/2`` tests
damping LATER digits MORE than the chain rule alone already gives, not
"chain-rule-matched" as an earlier draft of this docstring incorrectly
claimed.

.. _true_multi_digit_layer.kwarg_forwarding_and_connectivity_sharing:

``TrueMultiDigitLayer.__init__``: conditional kwarg forwarding, and the connectivity-overlap experiment
----------------------------------------------------------------------------------------------------------------

*ID:* ``true_multi_digit_layer.kwarg_forwarding_and_connectivity_sharing``

``dense=True``/``scale_rank>1``/``empty_init=True`` are only forwarded to
``digit_cls`` when set (not unconditionally) -- only ``DISLDOLayer``/
``DISLDOLayerDeterministic`` (sili__new) accept these kwargs at all;
older/other ``digit_cls`` options (``DISLDOLayer32``,
``DISLDOLayerResync``, etc.) would ``TypeError`` on an unexpected kwarg
otherwise, breaking every existing caller that doesn't ask for it.

``share_connectivity``: per direct hypothesis check -- each digit's OWN
independent preseed/synaptogenesis (real ``DISLDOLayer``, no coordination
between digits) means digit i's active (row, col) synapses generally do
NOT coincide with digit 0's. Verified directly: fresh preseed of 3 digits
at ``max_weights=40``/``n=20`` showed 0/20, 0/20, 1/20 overlap --
essentially disjoint. The residual-correction mechanism this whole scheme
depends on (digit i correcting digit i-1's rounding error AT THE SAME
synapse) can only fire where digits' connectivity actually coincides --
with near-zero overlap it almost never does, unlike the SIMULATED
``fixed_digit_residual_quantize``, which decomposes one shared value at
one shared (row, col) by construction. When ``True``, force every digit
after the first onto digit 0's EXACT ``(ptrs, indices)`` -- same synapses,
each digit's own independently-drawn/trained weight values -- a direct
test of whether connectivity alignment, not more bits or a different LR,
is what the residual composition actually needed.

Tested directly and made things WORSE, not better -- see ``JOURNAL.md``
for the investigation. ``synaptogenesis()``/``magnitude_rescale_output()``
therefore delegate to each digit independently (no shared-connectivity
coordination at those call sites either), matching construction-time
behavior.

.. _true_multi_digit_dense_layer.architecture_isolation_control:

``TrueMultiDigitDenseLayer``: dense+Adam control, isolating architecture from DISLDO's update rule
-------------------------------------------------------------------------------------------------------

*ID:* ``true_multi_digit_dense_layer.architecture_isolation_control``

DENSE, ordinary-Adam-trained control for ``TrueMultiDigitLayer`` -- per
direct request: same digit-residual architecture (n separate plain
``DenseTensorLinear`` layers, each contributing ``base**-i``, combined via
the same ``Tensor`` ``*``/``+`` autograd chain), but trained via a
STANDARD external ``AdamOptimizer`` instead of DISLDO's own inline
importance-based update. If this dense+Adam version behaves very
differently from the real-FP4 ``TrueMultiDigitLayer`` at the same
``n_stages``/``base``/``lr_power``, that's evidence something about
DISLDO's OWN update mechanism specifically (not the residual-digit
ARCHITECTURE itself) explains the difference -- matching this project's
own long-standing precision/optimizer-isolation convention
(``ToySmallTransformerFP32Ref``, ``dense_tanh_no_bptt_control.py``). No
quantization anywhere (fp32 dense weights throughout) -- this isolates the
ARCHITECTURE + UPDATE-RULE question specifically, separate from "does it
survive low-bit storage" (already covered by ``TrueMultiDigitLayer``'s own
``digit_cls=DISLDOLayer`` vs ``DISLDOLayer32`` comparison).

``max_weights``/``num_cpus`` accepted but unused -- kept only so this
drops into the SAME ``disldo_cls(in_features, out_features, max_weights,
num_cpus)`` call convention ``ToyTileRecurrenceRealFP4`` already uses for
every other arm, no changes needed there.

.. _quantized_disldo_layer32.rank1_scale_envelope:

``QuantizedDISLDOLayer32``: the empirically-validated 8-bit+rank-1 winner
-------------------------------------------------------------------------------

*ID:* ``quantized_disldo_layer32.rank1_scale_envelope``

A real ``DISLDOLayer32`` (fp32 ``DeltaCSRBiValues`` backend, same
RMSprop-style importance formula as production ``DISLDOLayer``) whose
weight AND importance arrays get fake-quantized to ``bits`` right after
every ``backward()`` call that actually trains (``learning_rate != 0.0``).
Simulates "this layer's real storage is N-bit" while keeping the
forward/backward ARITHMETIC itself exact fp32 -- isolates "does training
survive N-bit storage" from "is the update-rule math itself precise",
matching real quantization-aware-training simulators.

Found empirically (see ``JOURNAL.md``, the original single-RNN-task sweep
in the quantization-exploration script): 8-bit + rank-1 scale (row*col
envelope), quantizing BOTH weight and importance, reaches near-FP32
convergence quality; plain per-row scale needs to leave importance in
FP32 to do nearly as well; 4-bit (either scale scheme) converges but at a
real quality cost even with rank-1 -- importance's dynamic range is shaped
by BOTH forward and backward signal, unlike a weight, which is why it
needs the extra rank-1 degree of freedom more. Defaults here (``bits=8``,
``scheme=rank1``, ``quantize_importance=True``) are that
empirically-validated winner, not an arbitrary default -- this class is
the vehicle for testing whether it generalizes across OTHER toy
models/tasks before being worth a real sili__new C++ variant.

``scheme="rankn"`` (with ``rank >= 2``) is a follow-up being tested to see
whether it helps 4-bit specifically -- see
``rankn_fake_quantize.magnitude_bucketed_columns`` above for the mechanism
and its honest, modest measured effect. ``rank`` is only consulted when
``scheme=="rankn"``.

``e_shared`` (only consulted when ``scheme=="fixed_digit_residual"``) is
computed ONCE at construction from the initial preseeded weights, then
frozen for the rest of training (matching
``fixed_digit_residual_quantize``'s own "chosen once, never touched again"
design). Same ``disldo_cls``-pluggable call convention as
``DISLDOLayer``/``DISLDOLayer32``/``AdamRowScaleDISLDOLayer`` -- drops
directly into ``ToySmallTransformerRealFP4``/``ToyTileRecurrenceRealFP4``
with no changes needed there.

.. _seed_rank1_scale.cold_start_diagnostic:

``_seed_rank1_scale``/``SeededRank1DISLDOLayer8``: is DISLDOLayer8's collapse cold-start or representational?
--------------------------------------------------------------------------------------------------------------------

*ID:* ``seed_rank1_scale.cold_start_diagnostic``

``_seed_rank1_scale`` seeds ``value_scale``/``output_scale`` ONCE, from a
real closed-form 3-pass alternating max-cover fit of the layer's CURRENT
(freshly-preseeded) weights -- same math as ``rank1_fake_quantize``'s own
envelope fit, applied here only to set the starting point, not to
quantize/round anything. Real ongoing training still uses ``DISLDOLayer8``'s
own gradient-based ``value_scale``/``output_scale`` update
(``linear_disldo.hpp``) after this.

``SeededRank1DISLDOLayer8`` (real ``DISLDOLayer8``, true C++ E4M3 storage,
true ``disldo_forward``/``backward`` kernels -- NOT a fake-quantize
simulation) is the direct diagnostic this seeding enables: isolates
whether real ``DISLDOLayer8``'s out-of-context collapse is a
cold-start/undertrained-scale problem rather than the 8-bit+rank-1
representation itself being insufficient. Found directly: real
``DISLDOLayer8`` collapsed out-of-context, ``mean_acc=0.19``, despite
nominally using the identical 8-bit+rank1 scheme the toy fake-quantize
simulation solved at ``mean_acc=0.97`` -- this class tests whether that
slow, noisy, query-tick-only-gradient learning process was simply
undertrained within a fixed step budget, vs the representation itself
being insufficient.

.. _seeded_disldo_layer8_resync.fair_comparison_rationale:

``SeededDISLDOLayer8Resync``/``SeededDISLDOLayer8AdaMax``: seeding for a FAIR comparison, not cold-start
-------------------------------------------------------------------------------------------------------------------

*ID:* ``seeded_disldo_layer8_resync.fair_comparison_rationale``

``SeededDISLDOLayer8Resync`` wraps ``DISLDOLayer8Resync`` (the
DeferredScaleWrite fix -- see sili__new's ``ScalePolicy``/
``disldo_backward`` docstrings) with ``value_scale``/``output_scale``
seeded once at construction, same as ``SeededRank1DISLDOLayer8``. Seeding
here isn't about cold-start (the fix under test is orthogonal to that) --
it's for a FAIR comparison against ``fp8_seeded``: plain ``DISLDOLayer8``'s
``output_scale`` never trains at all unless something calls
``set_output_scale_raw`` at least once (confirmed directly: no code in
``DISLDOLayer8``'s own construction path does), so without seeding here
too, any difference measured could just be "output_scale was active"
rather than "the deferred-write fix helped."

``SeededDISLDOLayer8AdaMax`` is the same idea, wrapping
``DISLDOLayer8AdaMax`` (AdaMax-style scale update -- see
``AdaMaxScalePolicy``'s docstring, sili__new's ``delta_csr_types.hpp`` --
instead of RMSprop).

.. _periodic_seed_rank1_disldo_layer8.repeated_correction_test:

``PeriodicSeedRank1DISLDOLayer8``: does repeated re-seeding substitute for the simulation's every-step refit?
--------------------------------------------------------------------------------------------------------------------

*ID:* ``periodic_seed_rank1_disldo_layer8.repeated_correction_test``

Like ``SeededRank1DISLDOLayer8``, but re-seeds ``value_scale``/
``output_scale`` from a fresh closed-form rank-1 fit every
``reseed_every`` training ``backward()`` calls, not just once at
construction -- direct test of whether REPEATEDLY correcting the envelope
(touching NOTHING about the real RMSprop weight-update math) can
substitute for the simulation's every-step refit, or whether real
``DISLDOLayer8``'s own separate, nested ``value_scale``/``output_scale``
optimizer (see ``linear_disldo.hpp``'s
``scale_eff_lr = learning_rate / nnz_row``, itself RMSprop-style via
``value_scale_importance``) genuinely can't hold a good fit between
corrections even when repeatedly given one.

.. _peak_eligibility_trace.signed_peak_hold_design:

``_PeakEligibilityTrace``: signed peak-hold, replacing a double-counting smooth-decay design
---------------------------------------------------------------------------------------------------

*ID:* ``peak_eligibility_trace.signed_peak_hold_design``

Leaky peak-hold tracker over the layer's FULL ``[batch, features]`` input
shape (matching ``SparseLinearLayer.last_input`` exactly, not reduced) --
remembers, per (tile, feature) cell, the SIGNED value from whichever
recent tick had the largest magnitude, decaying only until a new input
exceeds the decayed peak (then replaced), rather than blurring all history
into one running sum.

Direct replacement for an earlier smooth-decaying-SUM design
(``e = decay*e + activity``, no longer in this file) that was found, via
the actual C++ update formula (``linear_disldo.hpp``'s ``disldo_backward``:
``dL/d(value_scale[r])`` already bakes in the query tick's OWN input
magnitude via ``g = dy*iv``), to double-count activity magnitude when
multiplied against DISLDOLayer's own already-input-weighted row gradient
-- see ``JOURNAL.md``'s e-prop postmortem for the full diagnosis.

SIGNED (not magnitude-only): the substitution this feeds (see
``peak_eligibility_disldo_layer.last_input_substitution_mechanism`` below)
needs a real signed input value, since DISLDO's gradient math depends on
input sign for direction, not just magnitude -- this formulation was
worked out directly with the user.

.. _peak_eligibility_disldo_layer.last_input_substitution_mechanism:

``PeakEligibilityDISLDOLayer``: overwriting last_input so real C++ math computes the substituted gradient
--------------------------------------------------------------------------------------------------------------------

*ID:* ``peak_eligibility_disldo_layer.last_input_substitution_mechanism``

A real ``DISLDOLayer`` whose per-row ``value_scale`` credit-assignment
uses REAL C++ gradient math, applied to a peak-substituted input -- the
replacement for the earlier, found-broken ``EPropDISLDOLayer``/
``EPropAdamDISLDOLayer`` (removed; see ``JOURNAL.md``'s postmortem) AND
for an even-earlier broadcast-``out.grad`` version tried in between.

Mechanism, worked out directly with the user: ``SparseLinearLayer`` caches
its most recent forward input as ``_last_input`` and exposes it via the
``last_input`` property -- checked directly, not assumed
(``cpu_backend.cpp:1199-1214``): this is a ZERO-COPY, WRITABLE numpy view
straight onto the C++ buffer (verified: mutating the returned array from
Python propagates into the object the C++ backward reads from).
``backward_dense`` doesn't take ``x`` as an argument at all -- it reads
``_last_input`` directly. So instead of trying to hand-derive a
Python-side approximation of the gradient (two prior attempts, both found
flawed -- see ``JOURNAL.md``), this OVERWRITES ``last_input`` with the
peak-held (signed) value right after forward, BEFORE backward ever fires,
so DISLDO's OWN real ``backward_dense`` computes the row's (and,
internally, each synapse's) gradient AS IF the input had been whichever
recent tick was most salient for that cell -- zero Python-side gradient
approximation, reusing the actual C++ math end to end.

.. code-block:: python

   out = self._inner.forward(x, learning_rate)
   if learning_rate != 0.0:
       # Overwrite AFTER forward, BEFORE backward -- backward_dense reads
       # _last_input directly (it takes no x argument), so this makes the
       # real C++ gradient math operate on the peak-held value instead of
       # the raw current-tick input, no Python-side gradient math needed.
       self._inner._c.last_input[...] = peak_snapshot
   return out

Known, accepted side effect: ``dx`` (accumulated into this layer's input
Tensor's ``.grad``, e.g. into ``qkv_source.grad`` in
``ToyTileRecurrenceRealFP4.step()``) is ALSO computed from the substituted
input, not the true one -- this contaminates gradient reaching upstream
plain-Tensor leaves (``input_ln``/``post_ln``) a little, since
``backward_dense`` computes ``dx`` and the ``value_scale`` gradient from
the same ``_last_input`` in one pass with no way to split them apart
without a C++ change. Accepted as a small, documented tradeoff
(``input_ln``/``post_ln`` are secondary parameters, not the
credit-assignment mechanism itself) rather than deferred silently.

True per-SYNAPSE substitution (not just per-row) would need direct CSR
access -- the "expensive, large core change" the user already flagged as a
separate, later effort, not attempted here. If this layer's typical input
activation ends up genuinely ~1-sparse (few nonzero rows), ``SISLDOLayer``'s
CSR forward/backward path would let this same substitution touch only the
active indices instead of a full dense array -- a real future efficiency
angle at true MiniCPM5 scale, not needed at this toy width.
