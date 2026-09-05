``sili_block.py`` research notes
===================================

Companion doc to ``model/sili_block.py``. Source comments point back here by
anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers, frozen code snippets on
real-bug/non-obvious-derivation sections only).

.. _sili_block.module_overview:

Module architecture: per-position layers vs the window's combined matrix
--------------------------------------------------------------------------

*ID:* ``sili_block.module_overview``

B6/B8-Phase2: attention assembly, plus the growable window-scoped combined
matrix that column-averaging training needs.

Every one of MiniCPM5's 24 fold-depth positions gets its OWN small,
independently-quantized ``SparseLinearLayer`` per suffix
(``build_step_layers``, 168 matrices total) -- unchanged from B6. A position
outside the current B8a curriculum window passes exactly one token through
the whole system at a time, so it can never have a recurrent/cross-position
connection; running it through anything heavier than its own small layer
would be wasted compute. ``run_folded_recurrence`` therefore keeps every
pre-window position on this plain per-position path, exactly as it always
has (``state=0; for step: out=block(x+state); state+=out``).

Only the CURRENT window (the last few positions B8a's curriculum is training
column-averaging over) needs a combined matrix -- that's the only place a
cross-position (recurrent/skip) synapse has anywhere to live.
``grow_window_layer()`` builds that combined matrix INCREMENTALLY: each time
the window widens by one position (a curriculum stage transition), the
newly-included position's own already-quantized ``step_layers[i][suffix]``
is folded in as a new diagonal block, and the existing (already-trained)
window matrix's rows/columns are reused VERBATIM, not rescanned or rebuilt
from scratch -- see ``sili_block.grow_window_layer.design_and_bandwidth_tradeoff``
and ``sili_block.csr_accessors.true_vs_raw_units`` for why this is exact
(not approximate) regardless of whether ``per_row`` or ``rank1`` quantization
built the underlying layers.

RMSNorm/RoPE weights are never part of the 7 folded suffixes (B3/B5 never
prune or quantize them), so they're read directly from ``sparse_state`` as
plain float32 vectors, not built into any ``SparseLinearLayer``.

.. _sili_block.build_step_layer_from_arrays.value_scale_mode:

``_build_step_layer_from_arrays``: per-step CSR slicing, and ``per_row`` vs ``rank1``
------------------------------------------------------------------------------------------

*ID:* ``sili_block.build_step_layer_from_arrays.value_scale_mode``

``ptrs``/``idx``/``vals`` are one fold step's own ``[n_in, n_out]`` CSR,
already sliced out of the suffix's full stacked-and-transposed matrix (see
``build_step_layers``) -- the transpose/conversion happens ONCE per suffix,
not once per step, since torch's ``to_sparse_csr()``/``t()`` carry real
per-call overhead that a 24x-per-suffix call count made the dominant cost of
building the model at all.

Builds a real FP4-quantized ``SparseLinearLayer``. ``value_scale_mode="per_row"``
(default): one ``value_scale`` per input row, matching
``FoldedLayer.from_descriptor``'s ``"per_row"`` mode. ``"rank1"``: also fits a
per-output-column scale (``fit_rank1_scale_envelope``), same scheme as B5a's
``from_descriptor`` ``"rank1"`` mode but fit independently per fold step
instead of shared across all 24 -- each step's own weight distribution gets
its own row+col envelope, not a compromise shared across the whole stacked
matrix.

.. _sili_block.build_step_layers.streaming_and_build_time:

``build_step_layers``: streaming construction, and two build-time optimizations that made things worse
----------------------------------------------------------------------------------------------------------------

*ID:* ``sili_block.build_step_layers.streaming_and_build_time``

Builds every fold step's real sili layers (one suffix-keyed dict per step)
plus that step's own RMSNorm weight vectors, streaming one suffix at a time
(mirrors ``fold.build_folded_layers_streaming``'s discipline) -- MUTATES
``sparse_state``, popping each suffix's per-layer tensors immediately after
slicing all 24 steps out of it, and popping each layer's two layernorm
vectors directly (they aren't part of any suffix descriptor).

These per-position layers are the ONLY construction needed for positions
outside the current B8a curriculum window (see
``sili_block.module_overview``) -- they also double as the source
``grow_window_layer()`` folds in when a position newly enters the window, so
there is no separate/duplicate build path for that case.

Two different attempts to reduce build time by cutting the number of torch
``.t().to_sparse_csr()`` calls (transposing the whole suffix once instead of
once per step, and separately a pure-numpy stable-sort transpose avoiding
torch's CSR machinery altogether) were both measured SLOWER on the real
checkpoint than the current per-step ``fold_weight_csr`` approach: real
regressions of ~150-165s vs the current ~77-81s -- reducing call count
wasn't the actual lever, and the real bottleneck hasn't been isolated yet.
See JOURNAL.md.

.. _sili_block.csr_accessors.true_vs_raw_units:

``_extract_true_csr`` / ``_raw_stored_csr``: two CSR accessors, true units vs stored units
------------------------------------------------------------------------------------------------

*ID:* ``sili_block.csr_accessors.true_vs_raw_units``

``_extract_true_csr`` reads a ``SparseLinearLayer``'s stored
``(ptrs, indices, values)`` back out in TRUE units
(``true_w = weights_vals * value_scale[row] * output_scale[col]``, see
``cpu_backend.cpp``) -- same pattern ``quantize.py``'s
``build_quantized_dense_state_dict_streaming`` already uses. NOT used by
``grow_window_layer`` itself (see ``_raw_stored_csr`` below) -- kept as a
general true-unit accessor, used by tests/reporting that want to inspect a
layer's real weight values.

``_raw_stored_csr`` reads ``(ptrs, indices, weights_vals)`` EXACTLY as
stored -- FP4-nominal units, NOT multiplied by ``value_scale``/
``output_scale`` -- plus the per-row and per-column scale arrays, with no
dequantization arithmetic at all. Used by ``grow_window_layer`` to reuse
existing rows/columns verbatim instead of round-tripping through true units
and refitting scales that provably can't have changed. This holds true
REGARDLESS of ``value_scale_mode`` (``per_row`` or ``rank1``): a row's/
column's scale only depends on the REAL (nonzero) entries touching it, and
every entry ``grow_window_layer`` newly introduces outside a position's own
diagonal block is zero-valued, which never binds a max-based fit (``per_row``'s
per-row ``max_abs``, or ``rank1``'s alternating max-fit in
``fit_rank1_scale_envelope`` -- both are pure max operations, and 0 never
raises a max above what real content already set).

.. _sili_block.fixed_band_span.invariant_center:

``_fixed_band_span``: why the band center is invariant to later window growth
--------------------------------------------------------------------------------

*ID:* ``sili_block.fixed_band_span.invariant_center``

This row's recurrent-band reach in absolute output-column units, using its
OWN position's fixed in/out ratio -- NOT recentered against however large
``total_in``/``total_out`` have grown to since. An absolute
``row = p*in_dim + l`` always maps to
``center = p*out_dim + l*out_dim/in_dim``, i.e. always INSIDE row's own
position's own output block (``out_dim/in_dim`` is the same ratio for every
position of this suffix, so ``p`` cancels out of where the block starts) --
provably invariant to how many further positions get added later. That's
what lets ``grow_window_layer`` only ever touch an OLD row once per later
stage (to extend it into whatever NEW column range just opened), never
recomputing/rescanning its already-placed entries.

.. _sili_block.grow_window_layer.design_and_bandwidth_tradeoff:

``grow_window_layer``: incremental growth, disjoint concatenation, and the bandwidth-vs-memory tradeoff
----------------------------------------------------------------------------------------------------------------

*ID:* ``sili_block.grow_window_layer.design_and_bandwidth_tradeoff``

Adds ONE position to the window's combined matrix. ``new_position_layer`` is
that position's own already-built, already-quantized small layer
(``step_layers[i][suffix]`` -- see ``build_step_layers``; no separate build
path, no ``desc``/pretrained-tensor access needed here at all). Works
identically regardless of what ``value_scale_mode`` built
``new_position_layer``/``existing_window_layer`` (``per_row`` or ``rank1``) --
see ``sili_block.csr_accessors.true_vs_raw_units`` for the proof: growth
only ever adds ZERO-valued cross-position entries, which never move a
max-based scale fit of either kind, so every row's AND every column's scale
(when set at all) is reused verbatim from wherever it already came from,
never refit.

``existing_window_layer=None`` (window growing from 0->1 positions): the new
diagonal block IS the whole matrix. Positions are appended in window-growth
order (index 0 = first position added to the window, i.e. the LAST
fold-step under B8a's backward-growing curriculum) -- existing blocks'
row/col offsets never shift as the window grows.

Returns a NEW ``SparseLinearLayer`` (old one is not mutated in place --
caller replaces its reference). Structure (old rows extended into whatever
new column range just opened; new rows carrying their own diagonal block
plus a band reaching back into existing columns) is a plain disjoint
concatenation, never a value-changing union or a rescan -- old-row content
and new-position content live in strictly disjoint column ranges (old
content < ``off_out`` <= new diagonal; new backward band < ``off_out`` <=
new diagonal too), so no conflict-resolution is needed; only a new row's own
two pieces (backward band + diagonal) need an explicit sort.

**Bandwidth tradeoff, measured**: ``recurrent_bandwidth=None`` (default)
picks ``max(1, min(in_dim, out_dim) // 8)`` -- deliberately NOT scaled to
``2*max(in_dim, out_dim)`` (an earlier version of this function did that,
matching ``build_fold_skip_layer``'s own "one hop = out_dim" convention
doubled). Measured directly at MiniCPM5's real dims (``in_dim=1536``,
``out_dim=4608``): a bandwidth on that same order made the "band" cover the
ENTIRE row width, i.e. fully DENSE, not sparse -- ~7M zero entries for a
single position's block in one suffix alone, a real "ton of memory at large
layers" problem.

.. code-block:: python

   # WRONG (earlier version): bandwidth scaled like a full skip-hop
   bw = 2 * max(in_dim, out_dim)
   # At in_dim=1536, out_dim=4608 this covers the entire row width --
   # fully dense, not sparse: ~7M zero entries for one position's block.

   # Fix: bound bandwidth to a small fraction of the layer's own width,
   # independent of how wide the real layer is.
   bw = max(1, min(in_dim, out_dim) // 8)

This default keeps nnz-per-row (and therefore total memory) a small,
bounded fraction of the layer's own width regardless of how wide the real
layer is. This is a genuine richness-vs-memory tradeoff, not a fully
"solved" number -- a small bandwidth means only neurons near a position's
own block boundary ever get a pre-seeded cross-position slot at all (see
``sili_block.fixed_band_span.invariant_center``: the proportional center
always falls inside a row's OWN position block, so reaching a neighbor at
all requires ``bw`` comparable to the distance from that row to its block's
edge). Phase 4's reporting is the place to check whether this is generous
enough for synaptogenesis to find useful cross-position connections in
practice; tune via this argument, not by editing the default.

.. _sili_block.forward.activation_density_argpartition:

``_forward``: per-row top-k via ``np.argpartition``, and why the C++ per-row loop was ruled out
----------------------------------------------------------------------------------------------------

*ID:* ``sili_block.forward.activation_density_argpartition``

``activation_density=None`` (default): dense DISLDO ``forward_dense``,
unchanged behavior. ``activation_density=d`` (``0 < d <= 1``): keep only the
top ``round(d*n_features)`` entries by magnitude PER ROW (per token) of
``x``, route through the existing SISLDO ``forward_sparse`` path instead.

Per-row top-k is done via ``np.argpartition`` (not a Python loop calling
``_cpu.dense_to_top_k_csr`` once per row -- that was ~74x slower at
``T=30, F=1536``, measured directly: 57ms/call vs 0.77ms/call, dominated by
per-call pybind overhead x T rows x 7 projections x 24 layers).
``dense_to_top_k_csr``'s own ``k`` is a GLOBAL budget over the whole
flattened ``[rows, cols]`` array (see sili__new's ``csr.hpp``
``top_k_csr -> top_k_indices``), not a per-row budget --
``np.argpartition(..., axis=1)`` is naturally per-row, sidestepping that
mismatch entirely.

.. _sili_block.apply_window_step.major_pivot_design:

``apply_window_step``: the MAJOR PIVOT (2026-08-02) -- interleaved attention and why Q blends token+state
--------------------------------------------------------------------------------------------------------------------

*ID:* ``sili_block.apply_window_step.major_pivot_design``

The in-window counterpart to ``apply_fold_step`` -- per the MAJOR PIVOT
(2026-08-02), scoped to the window ONLY and no longer a T-token-batched
causal-attention block. Processes ONE token at a time: every window
position runs its OWN RMSNorm (still strictly per-position, unchanged in
kind from ``apply_fold_step``), but the linear projections
(q/k/v/o/gate/up/down) route through ONE combined matrix per suffix
spanning the whole window (``grow_window_layer``'s output) instead of
separate small per-position layers -- that combined matrix is the only
place a cross-position (recurrent/skip) synapse can live.

**Time-axis mechanism** (replaces causal attention over T, which is
meaningless once T=1, AND replaces Phase 2.5/2.6's two separate,
only-one-of-them-trainable mechanisms -- see Phase 2.7): ONE real attention
op, ``sili.tensor.gaussian_attention``, per head, over a combined key/value
space that INTERLEAVES each window position's fresh-token entry and
carried-state entry: index ``2p`` = position p's fresh-token (K,V), index
``2p+1`` = position p's carried-state (K,V) -- length ``2*window_size``.

``q`` is the existing per-position query, drawing from BOTH this token AND
the carried state together (``q_proj`` applied to
``normed(token) + normed(state)`` -- linear, so this is exactly
``q_proj(normed(token)) + q_proj(normed(state))``, just cheaper to compute
once), per direct correction: tying Q exclusively to the current token means
it goes dead (content-blind, ``q=0``) with no fresh input, which forecloses
any "sleep"/consolidation-style internal dynamics driven by
``EnergyDynamics``' own exploration noise. With no real input,
``normed(token)`` is simply zero and Q still carries the state contribution,
so attention keeps running over memory alone -- genuine ongoing internal
dynamics, not a dead mechanism. ``k``/``v`` stay asymmetric (token-only /
state-only per interleaved slot) -- unlike Q, there is no reason to blur
K/V's identity, since which slot attention lands on is exactly what
distinguishes "trust this token" from "trust memory."

``centers``/``log_sigmas`` are per-window-position (length ``window_size``)
trainable ``Tensor`` leaves, owned by the caller's ``WindowState`` (see
``curriculum.WindowState``/``advance_window``) -- NOT re-initialized here.
``sigmas = exp(log_sigmas)`` keeps sigma strictly positive via the ordinary
autograd chain rule. Per direct decision, training these relies entirely on
plain backprop through the task loss plus ``EnergyDynamics``' own
``aux_loss`` (returned) once Phase 3 wires up ``.backward()`` -- no separate
sparsity-pressure mechanism for attention spread, and no actor-critic hook
here.

``attn_out = o_proj(blended)`` (pretrained, unchanged role). The residual
``x_common_t + attn_out`` is gated by ``energy_dynamics`` (caller-owned,
persisted across calls the same way ``carried_state`` is -- see
``run_folded_recurrence``) before being used both as this step's own
residual (feeds the MLP below) and as ``new_carried_state`` for the NEXT
token. This is the first place ``EnergyDynamics`` is wired into
``sili_peridot`` at all.

.. _sili_block.run_folded_recurrence.window_curriculum_design:

``run_folded_recurrence``: pre-window vs window split, and the ``window_size==1`` equivalence proof
------------------------------------------------------------------------------------------------------------

*ID:* ``sili_block.run_folded_recurrence.window_curriculum_design``

Base recurrence: ``state=0; for step: out=block(x+state); state+=out`` --
see ``RNNFoldedBlock.forward``'s docstring in sili__new for why this
recurrence (not a plain 24-layer sequential replay) and why
averaging/summing per-step outputs is not done here
(``skip_connection_outputs=False``: final accumulated state is returned,
RMSNorm'd, ready for ``lm_head``).

``window_state=None`` (default): the PLAIN pre-window path, unchanged since
B6 -- every position runs sequentially, exactly as it always has. Every
existing caller/test keeps working unmodified.

``window_state=<a curriculum.WindowState-shaped object>`` (duck-typed, not
imported here to avoid a ``curriculum``<->``sili_block`` import cycle --
needs ``.window_size``, ``.window_positions``, ``.suffix_windows``,
``.centers``, ``.log_sigmas``): positions BEFORE
``window_state.window_positions[-1]`` (the SMALLEST/earliest absolute index
currently in the window -- ``window_positions[0]`` is instead the
LARGEST/last-added, since ``curriculum.advance_window`` appends new
positions in the order they enter the window: last fold-step first, then
working backward) still run exactly this same plain sequential loop, over
the WHOLE ``[T, hidden]`` batch at once, ending at a ``state`` -- this is
what "positions outside the window compute sequentially, unchanged" means
throughout the project plan, and per the MAJOR PIVOT (2026-08-02) this is
untouched by the T=1 window redesign -- ``apply_fold_step`` never sees a
single token.

Only ONCE INSIDE the window does processing switch to one token at a time
(``window_size >= 2`` -- ``window_size == 1`` bypasses this entirely, see
below): ``x_common = x + state`` (the SAME starting input, per token, every
window position sees) is walked token by token, threading each window
position's own ``carried_state`` from one token to the next via
``apply_window_step`` -- see ``sili_block.apply_window_step.major_pivot_design``
for the unified ``gaussian_attention`` mechanism (Phase 2.7) this now uses
instead of causal attention over T (meaningless once a single call only
ever sees one token). ``centers``/``log_sigmas`` come from ``window_state``
itself (persisted, growable trainable parameters -- see
``curriculum.WindowState``/``advance_window``). ``window_carried_state``/
``window_energy`` are caller-owned (matching how
``window_state.suffix_windows`` already is) so a caller CAN persist them
across separate ``run_folded_recurrence`` calls if it wants continuity
across sequences -- None (the default) starts fresh state/energy every
call, the simplest and current choice pending Phase 3's real training loop
clarifying whether cross-call persistence is actually needed.

**Return value / window_size==1 proof**: the return value generalizes from
"final accumulated state (sum of every position's own delta, NOT including
x -- see ``RNNFoldedBlock.forward``'s docstring in sili__new, x is only ever
added back in to build the NEXT layer's input, never into the accumulator
itself), RMSNorm'd" to "column-averaged prediction, RMSNorm'd", matching
``RNNFoldedBlock.forward``'s own ``skip_connection_outputs=True`` mode
(``return mean(outputs)``, each ``outputs[i]`` itself already excluding x):
each window position's own column, PER TOKEN, = ``state`` (the PRE-WINDOW
accumulated delta sum for that token, WITHOUT x) + that position's own delta
output from ``apply_window_step`` at that token step -- NOT ``x_common``
(``x_common`` includes x, and is only the correct thing to feed IN, never to
add into the accumulated column).

``window_size==1`` (B8a's stage 0) is provably identical to the plain path
for this exact reason (only one column, ``state + that position's delta``
matches exactly what the plain path's own accumulator would hold after that
same position) -- and is also handled by bypassing ``apply_window_step``/the
per-token loop entirely (``window_state.window_size==1`` uses
``step_layers[window_state.window_positions[0]]`` directly, over the whole
T-batch, exactly like the pre-window loop), per ``curriculum.WindowState``'s
own documented recommendation, since there's nothing for a combined matrix
or a carried-state mechanism to usefully do with only one column.
``window_size>1`` averages ``window_size`` columns instead of trusting only
the last one, at every token.

``activation_density`` (see ``sili_block.forward.activation_density_argpartition``)
may also be a per-step list of length ``num_hidden_layers`` (each entry
itself None/float/dict) to isolate which LAYERS tolerate sparsification,
not just which projections -- applies to the PRE-WINDOW positions only when
``window_state`` is given (errors from top-k truncation compound through
the fold-depth recurrence's accumulated state, so a layer near the start is
not necessarily equivalent to the same layer near the end).
``window_activation_density`` is the window's own (single, not
per-position) density, passed to ``apply_window_step``.
