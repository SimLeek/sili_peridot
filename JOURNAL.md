# Journal

Running log of discoveries, gotchas, and rejected approaches found while
building the MiniCPM5 -> sili conversion tooling. `todolist.md` is the plan
and current status; this is where the "why" behind non-obvious decisions
lives instead of in code docstrings/comments, so the code can stay terse.

## sili__new: FoldedColumnLayer / column-averaging (Phase A3/A4)

- **`FoldedLayer.forward` is only a first-order approximation of the real
  fold recurrence.** The reference semantics (`RNNFoldedBlock.forward` in
  `sili/conversion/rnn_fold.py`) are `state=0; for i: state +=
  block_i(x+state)` — each fold step sees everything accumulated by every
  prior step. `FoldedLayer`'s single stacked matmul computes every fold
  step's contribution from the same external input independently, then
  sums — cheap, but no fold step ever sees another's output. Any
  mechanism connecting "virtual layer i" to "virtual layer j" needs a
  separate layer over the *output* space, not something pre-seeded inside
  a suffix's own weight matrix (there's no row space representing "layer
  i's own output" to connect from).

- **`FoldedColumnLayer` converged on `SparseRNNCell`'s own
  `h = input_proj(obs) + recurrent(state)` split** — deliberate, not
  coincidental, once pointed out. Not merged/subclassed with
  `SparseRNNCell` yet: that class bundles `EnergyDynamics` +
  `BranchingRatioTracker` + CSR-caching directly into `forward()` and its
  `input_proj`/`recurrent` are the currently-broken `DISLDOLayer`/
  `SISLDOLayer` (see TODO.md), while `FoldedColumnLayer` needed
  `SparseLinearLayer`-based ones specifically to avoid that bug. Revisit
  extracting a shared base once `DISLDOLayer`/`SISLDOLayer` are rebuilt on
  `SparseLinearLayer` — two real working examples sharing the same
  primitive, not just resembling each other.

- **`FoldedLayer.state_dict()` silently dropped `recurrent`'s trained
  weights**, and never saved per-row `value_scale`/`importance_scale` —
  a round trip would have defaulted every row back to `scale=1.0`,
  corrupting every true weight value with no error. There was also no
  `load_state_dict()` at all. Fixed with shared save/restore helpers
  (`_sparse_linear_layer_state_dict`/`_load_state_dict`) covering weights
  + both per-row scales.

- **FP4 zero-stuck-weight gotcha (hit ~3 times across this project now):**
  a freshly zero-valued connection never moves under gradient updates
  unless `value_scale` is set relative to the learning rate that will
  actually train it — FP4's minimum nonzero magnitude is
  `0.5 * value_scale`, so an update of order `lr` rounds back to zero
  under the default scale (1.0). Fix used everywhere new zero-valued
  structure is seeded: `value_scale = expected_lr / FP4_MAX`.

- **Energy divergence under a literally-static repeated input turned out
  to be curiosity/novelty-seeking pressure working as intended, not a
  bug.** A neuron that fires but consistently loses the top-p competition
  never has its energy reset (`drive > 2*activation_cost` nets positive
  growth every step). Confirmed isolated from the column-averaging
  mechanism itself (both converge correctly on their own). Every real
  usage in this codebase (Mandelbrot, RL agents, MiniCPM5's actual token
  stream) varies input step to step, so this regime won't be hit in
  practice — not clamped away; noted in TODO.md as something to
  deliberately harness later (e.g. as a Phase E action-pathway signal).

- **`csr_union` is a construction/load-time weight merge, not a
  forward-pass merge.** First reading of "csr-csr merge" was wrong —
  combining `in_proj` and `recurrent` into one matmul call. Rejected:
  they must stay two separate forward calls so `recurrent`'s own activity
  is measurable *before* it's summed with `in_proj` (the A6
  branching-ratio-identifiability fix depends on this). The actual need:
  a dense LLM folded into `FoldedColumnLayer` has zero skip connections
  between fold-depth layers, and a fresh `recurrent` pre-seed has zero
  trained values — these need to be unioned at construction/load time
  only, never during training.

- **`in_proj` needed the same skip-preseed treatment `recurrent` got.**
  Caught in review after the csr_union work landed for `recurrent` alone.
  Without it, `in_proj`'s only connections are whatever fixed nonzero
  pattern the original dense LLM's own per-layer weights happened to
  have — no structural room for training to grow a new input->column
  path. A direct, pre-seeded, zero-valued input->column shortcut speeds
  up column-averaging convergence specifically because the target (track
  the recent input) often doesn't change step to step. General lesson:
  when a mechanism lands on one half of an `input_proj`/`recurrent`-style
  split, check whether the other half needs the same treatment before
  calling the task done.

- **`csr_union`'s per-row Python dict-merge loop doesn't scale to real
  model-sized weight matrices.** `load_weights` already goes through
  `delta_csr_from_absolute` in C++ once a merged CSR is built (confirmed
  by reading `cpu_backend.cpp`), so the only slow part was building the
  merge itself. Each row's merge is an independent two-pointer walk over
  two sorted column lists — no cross-row coordination needed (unlike
  `linear_sisldo.hpp`'s `synap_parallel_fill`, which also enforces a
  global importance budget across rows) — so it parallelizes directly
  over rows with a plain `#pragma omp parallel for`: one pass to count
  each row's union size, prefix-sum for ptrs, one pass to fill. Added
  `csr_union<>` to `sili/lib/headers/csr.hpp`, bound as `_cpu.csr_union`,
  with `sili.sparse_rnn.csr_union` now a thin wrapper. New
  `tests/unit/test_csr_union.cpp` (the legacy `test_csr.cpp` isn't even
  in the active CMake build — had to add a fresh file rather than extend
  it).

- **Column-averaging training does NOT reliably distinguish "predict the
  next input" from "echo the current input" without deliberately
  constructing a task where they diverge.** Both
  `TestColumnAveragingLossTraining` (trains the loss against a free `h`
  parameter, no layer at all) and `TestColumnAveragingEndToEnd` (target
  unrelated to the actual input sequence) were silent on this. Built a
  diagnostic: train on a constant sequence (no memory needed), a
  deterministic cycle (next symbol is a fixed function of the current
  one — solvable by `in_proj` alone), and an "ambiguous" cycle (the same
  symbol has different successors at different points in the cycle —
  provably *not* solvable without tracking position, only `recurrent`'s
  accumulated state can carry that). `recurrent`'s trained true-unit
  weight magnitude after training tracks this ordering cleanly (constant
  < deterministic < ambiguous) — but only:
  1. **With a heavily amplified `column_averaging_loss` weight (30x).**
     At the loss's normal scale, the gradient reaching `recurrent` is
     tiny (already documented in `TestColumnAveragingEndToEnd`'s own
     scope note) and doesn't move it perceptibly within a practical
     epoch budget for a unit test.
  2. **Without `EnergyDynamics` in the loop.** Wrapping the exact same
     three-regime comparison in energy gating (tried both a low-density/
     low-p and a high-density/high-p config) completely washed out the
     ordering — energy's stochastic top-p gating dominates which pathway
     gets gradient far more than the underlying task structure does, at
     this toy scale within a practical epoch budget. Matches
     `TestColumnAveragingEndToEnd`'s own finding that gradient reaching
     `recurrent` through the full energy-gated path is real but tiny and
     unpredictable — a second confirmation from a different angle, not a
     new problem.
  3. Raw weight-movement is more reliable than an ablation-loss metric
     (comparing full forward vs. `in_proj` alone with state forced to
     zero) here: on a constant/trivial sequence, gradient descent still
     happily routes some signal through `recurrent` even though it's not
     needed (no "minimal circuit" bias in plain SGD), so the ablation gap
     doesn't order cleanly by task complexity the way raw magnitude does.
  Landed as `tests/integration/test_column_averaging_predictive.py` — the
  no-energy ordering test is a real, useful diagnostic; the with-energy
  tests only assert training stays finite/stable (the ordering claim
  doesn't survive there at toy scale, and forcing it would just be
  re-tuning hyperparameters to fake a demo).

- **Follow-up on the energy-interaction finding above: it was too
  pessimistic.** The first EnergyDynamics attempt (`drive=
  activation_cost=0.08, exploration=0.002`) wasn't tuned tightly against
  this project's own constraints (`activation_cost >= 0.01`, `exploration
  < drive/2`). Sweeping `drive` in `[0.01, 0.04]` with `activation_cost=
  drive` and `exploration` properly scaled found a real result missed
  the first time: under aggressive/low-density gating, the coarse
  property (constant sequence uses `recurrent` least) DOES survive energy
  gating -- it just needed better-tuned drive, exactly as suspected in
  review. The finer distinction (deterministic-but-varying vs. ambiguous)
  still doesn't reliably separate under energy at this toy scale across
  the whole range tried, though -- a genuine open question, not
  something a slightly-different drive value fixed.
- **`EnergyDynamics`'s exploration noise uses the global, unseeded
  `np.random`**, discovered because the new energy test was flaky across
  reruns despite every other seed in the harness being fixed. Any test
  asserting something more specific than "stays finite" through
  `EnergyDynamics` needs `np.random.seed(...)` called explicitly first.
  Documented in `sili__new/TODO.md`.

## sili_peridot: B3 pruning -- global threshold destroys the model

- **A "prune to CSR" threshold chosen purely from sparsity/compression
  math (B3's own `DEFAULT_TARGET_SPARSITY=0.8`) turned out to
  completely destroy the model's next-token prediction, with no
  retraining involved.** Never checked until asked to specifically:
  "load both the sparse and the dense model... and check how well they
  do in next token prediction tasks." Built `model/eval_pruning.py`
  (loads the real HF model dense and with pruning applied, no
  folding/columns/sili runtime involved -- purely "does zeroing these
  weights hurt") and found accuracy 0.0 (from 0.503 dense) at 0.8 global
  sparsity. Swept target_sparsity globally: quality holds through ~0.2,
  degrades continuously (not one sharp cliff) from ~0.3, catastrophic by
  ~0.4-0.5 -- nowhere near the ~70%+ per-tensor sparsity real CSR
  compression needs. A single global threshold cannot get both real
  compression and preserved quality on this model; there is no
  "conservative default" middle ground to just pick.
- **Per-tensor-role sensitivity turned out to vary enormously** --
  `embed_tokens` tolerated 90% sparsity fine; `v_proj`, architecturally
  near-identical to `k_proj` (which tolerated 70%), collapsed already
  past ~25-30%. A first coarse sweep (0.3/0.5/0.7/0.9 only) suggested
  v_proj's safe zone was ~0.05; a finer sweep (0.03 steps) found the
  real safe zone is actually ~0.2 -- the coarse grid was simply too
  coarse to see it, not evidence the fine-grained safe zone didn't
  exist. No "improvement dip" from removing pure noise below the safe
  zone was found (the user's hypothesis going in) -- degradation was
  flat/negligible then smoothly increasing, no local minimum below
  baseline. Perplexity degrades earlier and more smoothly than accuracy
  as target_sparsity rises -- a more sensitive leading indicator worth
  watching even when accuracy still looks fine.
- **Combining each role's own isolated-safe threshold compounded MUCH
  worse than isolation predicted.** First combined attempt (every role
  at its own isolated-safe number) gave accuracy 0.111, not the ~0.4-0.5
  isolated numbers implied. `q_proj`+`k_proj` together cost about 2x
  either alone (they interact directly in QK^T -- makes sense in
  hindsight, wasn't obvious going in); `o_proj` and `v_proj` each
  roughly quadrupled the cumulative damage on their own turn in the
  stepwise trace. Reaching an acceptable combined result (accuracy
  0.482) took 3 rounds of "find whichever role caused the biggest single
  jump in the cumulative trace, shrink just that one, recheck the
  combined result" by hand.
- **Generalized the whole procedure into `sili__new` instead of leaving
  it as a one-off MiniCPM5 investigation** (per direct request: "this is
  a pretty important pattern/research to put in with the sili__new
  conversion tools too"). `sili.conversion.prune_sensitivity`
  (`group_tensor_names_by_role`, `sweep_group_sensitivity`,
  `apply_group_thresholds`, `stepwise_cumulative_eval`,
  `iterative_threshold_search`) makes the grouping/sweeping/combining
  -verification/threshold-walkback loop a reusable tool for the next
  model instead of ad hoc scripts -- the only model-specific piece
  callers provide is `eval_fn` (a scalar, higher-is-better quality
  metric over a candidate state dict). `iterative_threshold_search`
  specifically automates the "shrink the worst offender, recheck"
  manual loop above -- greedy, not globally optimal (shrinking one role
  changes how much a role added AFTER it costs, since costs compound
  along the step order), but it's exactly the procedure that worked
  here, made repeatable.
- **Held-out validation on a second, disjoint text sample (never used
  during the threshold search) confirmed the final thresholds aren't
  overfit**: pruned accuracy held steady across both sets (0.482 search
  / 0.478 held-out) even though the dense baseline itself varied more
  between them (0.503 / 0.584 -- natural variance in how predictable
  different short texts are for the dense model, not a pruning effect).
- **Background-process hygiene lesson, mid-investigation**: manually
  backgrounding heavy model-loading scripts via shell `&`/`wait` (rather
  than the harness's own background-task tracking) left orphaned
  processes running when a foreground `wait` timed out -- four Python
  processes ended up competing for the same 15GB of RAM simultaneously,
  making everything far slower than any single run should have been.
  Fixed by using the tool's own `run_in_background` exclusively going
  forward and checking `ps aux` for stray processes before starting a
  new heavy one -- worth remembering for any future multi-round
  iterative search that needs repeated real-model loads.

## sili_peridot: B5 -- FoldedLayer construction memory, and a real
FP4 quantization quality catastrophe

- **First alarm on `FoldedLayer.from_descriptor()` at real scale was a
  measurement artifact, not a real problem.** Testing v_proj -> k_proj
  -> o_proj sequentially in one long-running process showed a "4.4GB
  jump" landing on o_proj's construction; extrapolating that rate to
  `gate_proj`/`up_proj` (budget ~170M vs o_proj's ~75M) suggested ~10GB
  more, which wouldn't fit alongside the ~8GB already used loading+
  pruning the checkpoint. Re-tested each suffix in an ISOLATED fresh
  process instead: `from_descriptor`'s own C++ construction
  (SparseLinearLayer alloc, load_weights, scale-setting,
  equalize_to_capacity) added **zero** measurable peak RSS beyond what
  `fold_suffix` itself already used -- true even for `gate_proj` at its
  full ~170M budget (nnz~=136M, already ~80% of the dense ceiling). The
  "4.4GB" was cumulative fragmentation from testing 3 suffixes
  back-to-back in one process without releasing memory between them
  (Python/glibc doesn't reliably return freed large-tensor memory to the
  OS within a process's lifetime) -- misattributed entirely to whichever
  suffix happened to run last.
- **The REAL memory driver is `fold_block_group`'s own per-layer
  dense-to-CSR conversion loop, not `from_descriptor`.** Confirmed by
  isolated single-suffix tests: `fold_suffix(gate_proj)` alone jumped
  from ~8GB (post load+prune) to ~11.5GB. Root cause: B3's per-role
  thresholds deliberately leave most suffixes 80-93% dense by row
  (`gate_proj` mean row occupancy ~80% of its dense ceiling) -- "prune
  gently, let training-time synaptogenesis do the real sparsification
  later" was always the intentional trade (see B3b above), but it means
  folding, a pure reshape, has almost no headroom to work with:
  converting 24 already-mostly-dense per-layer tensors to CSR and
  stacking them costs real memory regardless of how the C++ layer itself
  stores things.
- **Fix: stream, don't accumulate.** `fold_all_suffixes` (build all 7
  descriptors, hold them all) let dead weight from already-folded
  suffixes pile up across the loop. `build_folded_layers_streaming`
  (`model/fold.py`) folds ONE suffix, pops its 24 raw per-layer tensors
  out of the caller's `sparse_state` dict immediately, THEN builds the
  real `FoldedLayer`, before moving to the next suffix. Confirmed on the
  real checkpoint, twice: all 7 real `FoldedLayer` objects buildable
  simultaneously, peak ~13.5-13.9GB on this 15GB machine, ~4-5 minutes.
  A further variant, `build_and_save_folded_layers`, additionally
  serializes each suffix's `state_dict()` to a `.npz` and discards the
  live C++ object right after -- didn't reduce the measured PEAK much
  further (the peak moment is dominated by whichever single suffix is
  largest at the time it's processed, not by how many are held
  afterward), but does mean the CONVERSION step itself doesn't need to
  hold all 7 simultaneously once written to disk. True per-row streaming
  (converting each suffix's dense-to-CSR-to-C++ pipeline in row batches,
  never materializing a whole suffix's dense form at once) would cut the
  real peak further but requires changes inside `fold_block_group`/
  `from_descriptor` in `sili__new` itself -- tracked as a follow-up
  (todolist B5), not done in this pass.
- **A real FP4 quantization catastrophe, found while building the
  "pre-quantized vs post-quantized" comparison the user asked for.**
  Simulated `from_descriptor`'s EXACT quantization scheme in
  numpy/torch (`model/quantize.py`: fp4quant.hpp's real 16-level table,
  the real per-input-column scale computed on the STACKED/folded matrix
  -- shared across all 24 layers, not computed independently per layer)
  and ran the same real end-to-end next-token comparison methodology
  B3b used for pruning (`model/eval_quantization.py`, reusing
  `eval_pruning.evaluate_next_token_prediction`). Result: accuracy
  collapses from 0.482 (B3b's pruned baseline) to ~0.09-0.12, perplexity
  jumps ~200x (16 -> ~3300) -- nowhere near the "small drop" expected
  going in. Ruled out a simulation bug before treating this as real:
  (a) unit-tested the scale computation against a hand-worked example
  (`test_quantize.py`), (b) checked whether the shared (across-24-layers)
  scale is dominated by a rogue outlier layer for `gate_proj` -- it
  isn't (per-layer/global max_abs ratio: mean=0.71, median=0.71, only
  1.1% of (layer,column) pairs below 0.3, 0% below 0.1), ruling out
  "one weird layer wrecks everyone's resolution" as the mechanism.
  Tried a finer, per-layer-only scale (each layer quantized against its
  own weights, not the shared/stacked one) as the obvious first fix:
  helps a lot (accuracy 0.094 -> 0.265) but still falls far short of
  "small drop" -- naive round-to-nearest 4-bit weight quantization with
  no calibration (no GPTQ-style Hessian-aware rounding, no smoothing,
  no outlier handling) is just genuinely destructive at this model
  scale, stacked on top of already-pruned weights. Locked in as a real
  regression test (`tests/test_eval_quantization.py`,
  `test_shared_scale_quantization_currently_destroys_quality`) rather
  than hidden -- same pattern as B3b's
  `test_global_threshold_pruning_destroys_the_model`: document the
  current bad number so a future fix has something concrete to beat.
  **Not yet resolved** -- needs the same kind of per-role/per-suffix
  sensitivity search B3b built for pruning
  (`sili.conversion.prune_sensitivity`), not yet attempted for
  quantization; plausibly some suffixes tolerate FP4 fine and others
  (like `v_proj` for pruning) don't, and per-suffix granularity /
  selective full-precision retention for sensitive suffixes is the
  natural next experiment.
- **A second, real OOM directly caused by NOT applying the same
  streaming discipline everywhere.** `fold_all_suffixes` alone (no real
  `FoldedLayer` construction, just building all 7 descriptors to extract
  their scale vectors) OOM-killed this machine (confirmed via
  `journalctl`/kernel oom-killer log, not inferred) once a real HF model
  was ALSO loaded at the same time for the next-token comparison --
  holding all 7 suffixes' full stacked CSRs (hundreds of MB to ~1.8GB
  each for the largest) simultaneously, on top of the HF model's own
  ~4.3GB float32 footprint and the still-resident pruned `sparse_state`,
  exceeded 15GB. Fixed the same way as the FoldedLayer construction
  case: `compute_suffix_scales_streaming` (`model/quantize.py`) folds
  one suffix, extracts its (small, ~1536-float) scale vector, discards
  the descriptor, before moving to the next. **Order matters**: build
  `pruned_dense` (needed for the whole rest of the comparison) BEFORE
  running the streaming/destructive scale computation, since the latter
  pops entries out of `sparse_state` -- doing it in the other order
  would silently produce an incomplete `pruned_dense`.

## sili_peridot/sili__new: B5a -- rank-1 quantization scale fixes most of
the catastrophe

- **User's diagnosis, confirmed directly before building anything**: "the
  layers were originally trained as separate dense layers, so merging
  them would harm things" -- i.e. `from_descriptor`'s per-row scale,
  SHARED across all 24 folded layers, forces every layer to share ONE
  input-side scale even though they were never trained with that
  constraint. Their proposed fix: per-input AND per-output scale (a rank
  -1/outer-product pair of vectors, not a full per-element matrix).
  Their diagnostic proposal -- check whether per-OUTPUT magnitude is
  roughly uniform WITHIN one folded layer (if so, per-layer scaling
  alone would already be "enough" and rank-1 wouldn't help) -- was run
  first, on the real checkpoint: within-layer coefficient of variation
  of per-output max|.| ~0.32-0.38, within-layer min/max ratio ~0.05-0.10
  (gate_proj/q_proj/down_proj all similar). Confirmed real structure a
  per-row-only OR per-layer-only scale can't capture.
- **Simulated in pure numpy before touching any C++** (per direct
  guidance -- validate the concept cheaply first): an alternating max
  -fit ("Sinkhorn-style, but max-based") producing a rank-1 envelope
  `row_scale[i] * col_scale[o] >= |W[i,o]|` for every entry, using
  O(n_in+n_out) parameters instead of O(n_in*n_out). Real end-to-end
  next-token result: accuracy 0.482 (baseline) -> 0.094 (old per-row
  scheme) -> **0.297 (rank-1)**, perplexity 3328 -> 91 (~36x lower).
  Real, substantial progress -- not full recovery.
- **Two further simulated attempts, both informative negatives**:
  (a) percentile-based envelope fitting (letting rare outlier weights
  clip/saturate instead of forcing the whole scale to cover them, a
  standard clipping-based quantization technique) tested WORSE than the
  plain max-envelope (accuracy 0.181 vs 0.297) -- rejected, not pursued
  further. (b) Confirmed the shared-scale catastrophe wasn't caused by
  one outlier layer dominating the scale (per-layer/global max_abs
  ratio: mean=0.71, only 1.1% of pairs below 0.3) before spending effort
  on a fix that wouldn't have addressed the real cause.
- **Built the real sili__new library support** once the concept was
  validated (sili__new PR #10): `output_scale` (per-column, mirroring
  `value_scale`'s existing per-row design exactly) added to
  `SparseLinearWeightsDelta`, wired through `disldo_forward`/
  `disldo_backward`'s dequantization (`true_w = stored_w * value_scale
  [row] * output_scale[col]`). `value_scale`'s OWN gradient (used when a
  layer is later trained, not at conversion time) had to be corrected to
  account for the new fixed `output_scale[col]` factor too -- easy to
  miss since `output_scale` itself isn't gradient-updated, but
  `value_scale`'s gradient formula still needs to know about it.
  `output_scale` deliberately NOT gradient-updated -- a fixed, write
  -once conversion-time constant, not a re-learned parameter (matches
  how `value_scale` is actually USED in `from_descriptor` today, even
  though the class also supports gradient-updating `value_scale`
  elsewhere, e.g. Mandelbrot RL). `FoldedLayer.from_descriptor` gets an
  opt-in `value_scale_mode="rank1"` (default stays `"per_row"` for exact
  backward compatibility) -- now `sili_peridot`'s own default once
  wired through `model/fold.py`'s builder functions.
- **Real (not simulated) C++ validation, on the actual checkpoint**:
  built a real `FoldedLayer` for `gate_proj` via both modes, compared
  `forward()` output against the exact unquantized analytic reference
  (`reference_fold_forward`) -- rank1's real reconstruction error was
  ~30% lower than per_row's (mean_abs_err 0.75 vs 1.08), confirming the
  wiring works correctly, not just the simulation.
- **A real bug found writing sili_peridot's OWN simulation copy of the
  fix** (`model/quantize.py`): the envelope fit needs `abs(W)`, not raw
  signed values -- a first draft dropped the `.abs()` call (present
  correctly in the original probe script and in sili__new's own
  `from_descriptor`), which silently made the simulated "rank1" result
  WORSE than per-row (error 180 vs 105 on synthetic data) rather than
  better. Caught because the new module-level test used i.i.d. random
  weights with no real per-output structure for rank1 to exploit in the
  first place -- fixed the test to build data WITH genuine per-output
  magnitude structure (matching what the real checkpoint actually has),
  which is what surfaced the missing-`.abs()` bug clearly.
- **A `float64` memory bug, found by a real, repeated OOM** (confirmed
  via kernel oom-killer log each time, not inferred): `fit_rank1_scale_
  envelope`'s `row_scale`/`col_scale` were `float64` by construction,
  which silently upcasts every `abs_mat / scale` broadcast to float64
  internally regardless of `abs_mat`'s own dtype -- doubling the
  transient memory of every envelope-fit iteration for the largest
  suffixes. Combined with a real HF model also loaded for the next
  -token comparison, this OOM-killed the machine three times before
  being found and fixed (switched to float32 throughout -- a max/divide
  fit has no numerical-stability need for float64). Fixed in both
  sili__new's copy and sili_peridot's own mirror.
- **Even after that fix, running BOTH real-checkpoint quantization test
  classes (per_row's and rank1's) together in one pytest invocation
  still risks OOM on this 15GB machine** -- each passes reliably in
  isolation. Documented directly in the test file rather than chased
  further: a real hardware constraint on this specific machine, not a
  code bug (the module-scoped fixtures are each individually correct
  and necessary).
- **Per direct instruction, this is where the search stops for now**:
  "0.297 isn't the best but it also isn't noise" -- real, substantial,
  validated progress (accuracy 3.2x higher, perplexity 36x lower than
  the original scheme), locked in as the new baseline via a real
  -checkpoint regression test, rather than continuing to chase full
  parity with the 0.482 dense-pruned baseline right now. B5a is closed
  as "addressed", not "solved" -- a finer-than-rank-1 group size or
  genuine calibration (Hessian/activation-aware rounding, e.g.
  GPTQ-style) remains a real, documented option if more quality is
  needed later, most likely resolved via B8's post-quantization
  training rather than further conversion-time work.

## sili_peridot/sili__new: retiring the torch quantization simulation,
## a real memory investigation, and B6 attention assembly

- **`model/quantize.py` was a pure-Python/numpy simulation of sili__new's
  FP4 quantization, duplicating `fit_rank1_scale_envelope` and doing its
  own `.to_dense()` calls** -- justified originally as "no sili runtime
  involved" so quantization quality could be measured without a full
  sili build, but once sili__new's real `FoldedLayer.from_descriptor`
  path existed and had the no-densify/rank-1 fixes (B5a), this was
  strictly worse duplicate code, per direct instruction ("we really
  don't need or want a pure python simulation anymore"). Replaced with
  `build_quantized_dense_state_dict_streaming`, which builds a real
  `FoldedLayer` per suffix and reads back the true post-quantization
  weights via its zero-copy `ptrs`/`indices`/`weights_vals` -- no
  simulation, one densify per suffix (unavoidable: HF's `nn.Linear`
  needs a real dense tensor) instead of the old path's two-plus.
- **The real-checkpoint `test_eval_quantization.py` tests OOM-killed at
  13.8GB even after that rewrite** (confirmed via kernel oom-killer
  log). Root cause was NOT the quantize.py rewrite -- it was two
  things: (1) `test_eval_quantization.py`'s two test classes used
  `scope="module"` fixtures, but since both classes live in one module,
  pytest never released the first class's ~4.3GB HF model before
  loading the second's (fixed: `scope="class"`); (2) freed
  Python/torch/sili objects were never actually leaked (confirmed via a
  standalone two-round load/prune/quantize/free script sampling RSS at
  each step: bounded, not growing, round to round) but glibc's
  allocator keeps freed arena pages for reuse by the SAME process
  instead of returning them to the OS, and that retained-but-free
  memory piles up across a whole pytest session. `gc.collect()` +
  `ctypes...malloc_trim(0)` dropped RSS by ~90% after each round in the
  diagnostic. Added a `conftest.py` `pytest_runtest_teardown` hook to
  trim after every test -- the full 3-file suite (33 tests, both
  real-checkpoint classes) now passes together in one invocation.
- **Direct follow-up question: fp4 weights + fp4 importances + scale
  vectors + ULEB128 delta indices should be ~2 bytes/param (~2GB for
  this checkpoint's ~1B params), so why does the real pipeline reach
  ~12GB?** Measured directly rather than guessed: built all 7 real
  `FoldedLayer`s for the pruned checkpoint (544.5M total connections)
  and sampled RSS after each suffix. The suspected culprit --
  `sparse_rnn.py`'s `budget = n_in * n_out` passed as `from_descriptor`'s
  reserve size (the theoretical fully-dense ceiling, e.g. ~170M for
  `mlp.gate_proj`) -- turned out NOT to be the driver:
  `std::vector::reserve()` on that size is virtual/lazy on Linux
  (mmap-backed, uncommitted pages until written), confirmed empirically
  (RSS actually DROPPED while building the two largest suffixes, since
  freeing the consumed pruned-torch-tensors outweighed the new sili
  storage). After all 7 layers + `malloc_trim`: 1951MB against a
  1039MB theoretical minimum -- only ~1.9x, not 6x. The real remaining
  lever was `equalize_to_capacity(mrw=n_out by default, ...)`, which
  DOES reserve real per-row growth headroom (20%+ per row, for
  synaptogenesis that isn't happening yet at conversion time) --
  calling `.compact()` after took it to 1118MB, within 8% of the
  theoretical minimum. Added `compact_after_build=True` (default) to
  `from_descriptor` in sili__new (PR #11) -- the old "~13.5GB for all 7
  layers" number documented in `fold.py` was measured before the
  `malloc_trim` fix existed, same root cause as the test-suite OOM
  above, not a uleb128/budget bug.
- **Asked directly whether sili is used for eval anywhere in the
  codebase: no, not anywhere, not even partially.** `sili` is imported
  only in `checkpoint.py`/`prune.py`/`quantize.py`/`fold.py` (all
  conversion-time); `eval_pruning.py`/`eval_quantization.py` always
  reload the converted weights back into the real HF
  `AutoModelForCausalLM` and run `model(**ids)` -- 100% torch forward,
  every time. `FoldedLayer.forward()` is only ever called from
  `reference_fold_forward`, a correctness-check helper, never in an
  actual quality-eval path. Direct instruction: get real evals running
  on sili, not torch, and build B6 (attention assembly) on real sili
  ops -- `sili/conversion/model_reconstruct.py` looked promising at
  first glance (RoPE, RMSNorm, causal LLaMA attention with GQA) but
  turned out to be pure-torch (an architecture-reconstruction/
  inspection tool, unrelated to sili compute) -- useful only as a
  verified reference spec to replicate against sili ops, not reusable
  code.
- **None of sili__new's three attention kernels
  (`banded_attention`/`sparse_banded_attention`/`sparse_attention`)
  masked the future** -- `banded_attention`'s band is centered
  symmetrically on a geometric-diagonal point, so a query attends to
  keys on both sides of its own position; `sparse_attention`'s global
  top-k selects queries and keys independently by L2 norm with no
  position awareness at all. Silently wrong for an autoregressive LM.
  Added `causal=false` to all three forward/backward pairs in sili__new
  (PR #12) -- banded variants clamp the band's upper bound to the
  query's own position (requires T==K), `sparse_attention` masks any
  selected (query,key) pair where the key's position is later. Also
  threaded `causal` through the `sili.tensor` autograd wrappers
  (`sparse_attention`/`banded_attention`/`sparse_banded_attention`),
  which existed (A2, already done) but didn't expose the new flag.
- **B6 (attention assembly) confirmed NOT yet built anywhere**: neither
  `RNNFoldedBlock` (torch reference skeleton, `_apply_block` raises
  `NotImplementedError` until subclassed) nor `SiliBlock.forward_sili`
  (sums every suffix's stacked-matrix output together with no actual
  attention computation at all -- a placeholder, not real attention)
  implement it. Built `model/sili_block.py`: the real fold-depth
  recurrence (`state=0; for step: out=block(x+state); state+=out`, per
  `RNNFoldedBlock.forward`'s own docstring) with a real per-step GQA
  causal-attention + SwiGLU-MLP block, every projection computed by a
  real per-step FP4-quantized `SparseLinearLayer` built from
  `FoldedBlockDescriptor.fold_weight_csr` (quantized independently per
  fold step, not reusing B5a's stacked/rank-1 scheme, which shares one
  scale across all 24 layers for storage efficiency -- not what running
  each original layer separately would see). RMSNorm/RoPE/SiLU are
  plain numpy (no sparsity, sili doesn't claim these as ops).
  Validated on synthetic dims only so far: RMSNorm/RoPE match a torch
  reference numerically, and causal integrity holds both for one fold
  step and end-to-end across the full recurrence (perturbing a token
  after position t never changes output at or before t). NOT yet
  validated at real MiniCPM5 scale, and B7 (full model assembly --
  embed_tokens/lm_head, a real next-token accuracy number entirely on
  sili, replacing eval_pruning.py/eval_quantization.py's torch-forward
  methodology) is not started. Scoped this way deliberately given the
  size of B6+B7 combined -- get the core mechanism right and tested
  first, real-checkpoint integration next.
- **B7 (full model assembly) + a real sili-vs-torch integration test,
  same session's direct follow-up**: `model/sili_model.py` wires
  embed_tokens (numpy gather) -> B6's fold-depth recurrence ->
  final RMSNorm -> lm_head (numpy matmul) and computes real next-token
  loss/accuracy, mirroring `eval_pruning.evaluate_next_token_prediction`
  exactly for a fair comparison. Found and fixed a real bug getting
  this working: `embed_tokens.weight`/`lm_head.weight` are 2-D, so B3's
  role-based pruning DOES apply to them and can store either as `{"csr":
  ...}` or `{"raw": ...}` depending on its own density decision --
  assumed `"raw"` always (matching the 1-D layernorm vectors, which
  really are always `"raw"`) and hit a `KeyError` on the real
  checkpoint immediately.
- **Real head-to-head run (`tests/test_sili_vs_torch_integration.py`,
  results in `sili_v_torch.md`)**: same B3-pruned weights, evaluated via
  torch (float32, pruned only) vs. sili end-to-end (B6/B7). sili:
  accuracy 0.265 (vs. torch's 0.483), perplexity 173 (vs. 16.1), ~161s
  wall-clock (vs. ~7.6s -- ~21x slower, expected given many small
  Python-level C++ calls per token per fold step vs. one fused torch
  forward), but LOWER peak RSS for the eval phase itself (6.8GB vs.
  11.8GB). Important correction caught before reporting this: the
  report's first draft claimed "no quantization on either side" --
  false. `SparseLinearLayer.load_weights` always FP4-quantizes, no
  opt-out, so sili's column combines pruning + quantization (per fold
  step independently, not B5a's stacked/rank-1 scheme) + the recurrence
  approximation, against torch's pruning-ONLY float32 baseline -- three
  things differ at once, not one. B5a's own isolated quantization
  measurement (~0.297 accuracy for the stacked/rank-1 scheme) is
  suggestively close to this run's 0.265, which could mean per-step
  independent quantization is similar to or slightly worse than sharing
  one scale across all 24 layers, OR that the recurrence approximation
  itself is the larger factor -- genuinely not isolated by this test,
  flagged as the real next step rather than guessed at.

## sili_peridot: chasing sili's slowness/memory on the real head-to-head

Direct follow-up to the sili-vs-torch numbers above (161s wall-clock,
accuracy well below torch): profile it, try num_cpus=4 vs 8, check
disldo_forward for redundant per-loop scale multiplication, try
prune-then-quantize-separately (rank-1 per fold step) to see if it
reaches B5a's 0.297, and break down peak RSS into model weights vs.
everything else.

- **cProfile is unreliable for this workload -- badly undercounts real
  wall-clock whenever the profiled call involves OpenMP worker threads
  or torch's own internal threading.** First attempt: wrapped
  `evaluate_next_token_prediction_sili` in cProfile -- it reported 0.535
  CPU-seconds total while the surrounding `time.perf_counter()` showed
  70s elapsed. Same story for `build_sili_model` (55.9s reported vs.
  79.4s real). Whatever cProfile's calling-thread-based timer is
  actually measuring, it isn't attributing genuine multi-threaded C++
  work correctly. Confirmed this wasn't "OpenMP thread-spawn overhead
  dominates, so serial is faster" either: forcing num_cpus=1 made eval
  slower (204s), not faster, than num_cpus=4 (70s) -- real parallelism
  is helping, not hurting. Switched to `py-spy record --native
  --format raw` (a sampling profiler, installed into the project venv
  since pip is externally-managed at the system level) for a trustworthy
  breakdown: 21358 samples, `disldo_forward` itself is ~27.5% of ALL
  samples across both phases (genuine sili compute, not overhead) --
  confirms the eval phase's cost is mostly real work.
- **num_cpus: 8 (this machine's real thread count) beats 4 beats 1 for
  the eval phase** (204s @ 1, ~70s @ 4, ~48-53s @ 8, all same real
  checkpoint) -- real parallelism benefit, not thread-spawn overhead to
  avoid. Switched sili_model.py's default to 8.
- **Checked disldo_forward directly for the "a*b+a*c=a*(b+c)" pattern
  (per direct instruction) -- already optimal, nothing to hoist.**
  `w = w_stored * val_scale * out_scale` (linear_disldo.hpp:109) is
  computed once per synapse, outside the batch loop, not recomputed per
  token -- the maximum possible hoisting given val_scale/out_scale
  genuinely differ per (row,col). py-spy's hottest single LINE (17.8%
  of all samples) is instead `mo[b*n_out+col] += contrib` (line 121) --
  the output accumulator write. Read the surrounding code closely (per
  direct follow-up: "this is DISLDO though, not SISLDO... we should be
  able to avoid cache-access issues if we're doing things right" --
  correct, and worth two concrete notes for a future kernel change, not
  made in this session since it's a shared-library change needing
  careful re-verification): (1) `t_out` is sized `num_cpus * batch *
  n_out`, one full PRIVATE [batch, n_out] buffer per thread, freshly
  allocated and zeroed on every single disldo_forward call then reduced
  after -- a fixed per-call cost that scales with num_cpus, cheap for
  few large calls but paid thousands of times for this session's
  many-small-calls B6/B7 usage pattern; (2) within one thread's private
  buffer, `mo[b*n_out+col] += contrib`'s batch loop (for a FIXED
  synapse/col) strides by `n_out` elements between writes, since `mo`
  is laid out batch-major ([batch, n_out]) -- since DISLDO's output is
  dense (unlike SISLDO), there's no sparsity forcing this layout;
  storing it output-major ([n_out, batch]) instead would make that same
  inner loop write contiguously.
- **Two different attempts to speed up build_sili_model by reducing the
  number of torch `.t().to_sparse_csr()` calls both made it SLOWER, a
  real regression, not the fix expected -- reverted.** py-spy also
  showed `to_sparse_csr`/`to_sparse` dominating the ~77-81s build phase
  (38+9.6 of ~56 cProfile-reported seconds, itself an undercount per
  above). Attempt 1: transpose each suffix's whole stacked matrix ONCE
  (7 calls total) instead of once per fold step (168 calls), then slice
  per-step from numpy arrays -- measured 151-163s, WORSE than the
  original 77-81s. Attempt 2: same call-count reduction but replacing
  torch's CSR transpose with a pure-numpy stable-sort/bucket transpose
  (no torch CSR machinery at all) -- still 163s, no better. Cutting call
  count wasn't the actual lever; CSR transpose's cost scales with data
  volume more than with fixed per-call overhead (which was the original
  hypothesis), and the real bottleneck in the reverted-to per-step
  approach hasn't been isolated. Reverted both attempts to the original
  `FoldedBlockDescriptor.fold_weight_csr`-per-step approach (confirmed
  fastest of the three, ~77-81s) rather than ship a regression.
- **Prune-then-quantize-separately, rank-1 per fold step: does NOT
  reach B5a's 0.297 -- it's worse than per-row, not better** (0.2426
  accuracy / 248.6 perplexity vs. per-row's 0.2652 / 173.4), the
  opposite of B5a's finding for the STACKED scheme (where rank-1 fixed
  a real catastrophe: 0.09-0.12 -> 0.297). Real, evidence-backed
  conclusion: each fold step's own weight matrix is already small and
  reasonably well-conditioned on its own (not an artificial 24-layer
  concatenation with wildly different per-layer scales the way B5a's
  stacked matrix was), so per-row-only scaling doesn't have the same
  catastrophic failure mode there was something for rank-1 to fix in
  the first place -- adding rank-1's column correction on top just adds
  its own fit noise.
  **CORRECTION, direct follow-up**: this was originally written up as
  "evidence the recurrence approximation, not quantization scheme, is
  the larger factor" -- wrong framing, caught by direct question ("the
  recurrence isn't an approximation, it should be exactly numerically
  equivalent -- are the values staying float32 throughout?"). Worked
  through the math: `state=0; for step: out=block(x+state);
  state+=out`, WITH each step using that fold-step's own distinct real
  weights and a real per-step attention+MLP computation (which is what
  B6 actually built, not the crude single-shot fold-sum
  `SiliBlock.forward_sili`/`FoldedLayer.forward` that the older
  "first-order approximation" note was actually about) -- by induction,
  `x+state` at step i exactly equals the true sequential model's h_i,
  so the recurrence IS exactly equivalent to true layer-by-layer
  composition, not an approximation of it, PROVIDED each step's own
  block computation is implemented correctly. torch runs float32
  throughout (confirmed: `dtype=torch.float32` explicit); sili's
  `SparseLinearLayer` stores every weight as FP4 (4 bits, 15
  representable levels -- far coarser than float16) via
  `load_weights`, with no opt-out. The rank-1-vs-per-row comparison
  only varied the FP4 scale-FITTING scheme -- both are still
  FP4-quantized, so it could never have isolated "recurrence" from
  "quantization" in the first place; that conclusion wasn't supported by
  the experiment run. FP4's own coarseness is the more likely dominant
  factor, not yet directly tested (would need either a genuine
  unquantized (float32) sili forward path, not currently possible since
  FP4 isn't optional in SparseLinearLayer, or a hand-rolled float32
  reference fed the SAME post-dequantization weight values to isolate
  "is apply_fold_step's own attention/RoPE/GQA/MLP math correct" from
  "does FP4 hurt".
- **Memory breakdown: model weights vs. everything else, computed from
  exact real numbers** (total checkpoint: 1,080,632,832 params exactly,
  4122MB dense float32; 7-suffix pruned nnz 544,477,322 confirmed
  matching the earlier B5/compact() measurement exactly; embed_tokens
  pruned nnz 40,427,529 of 200,540,160 dense = ~20% density; lm_head
  pruned nnz 140,628,074 of 200,540,160 = ~70% density).
  - **torch** (peak 11810MB): model-related memory is actually TWO full
    copies held simultaneously -- the HF model's own live parameters
    (4122MB) AND the separate `pruned_dense` source dict `load_state_dict`
    copied from, which stays referenced until freed after the peak
    measurement (another 4122MB) = ~8244MB "model", leaving ~3566MB
    (~30%) as interpreter/torch/transformers-library/tokenizer/
    activation-buffer "context". The double-copy is an artifact of this
    specific test's structure (freeing `pruned_dense` right after
    `load_state_dict` would roughly halve it), not fundamental to torch.
  - **sili** (peak 7070MB): the 7 real transformer-suffix weights cost
    ~1118MB in sili's actual compact FP4 format (measured earlier via
    `compact()`) -- but embed_tokens and lm_head are currently loaded as
    FULL DENSE numpy arrays (`sili_model.py`'s `_to_dense_numpy`, chosen
    as "out of B3/B5's scope" without checking whether B3 actually
    prunes them -- it does), costing 765MB each regardless of their real
    ~20%/~70% density. "Model" ≈ 1118+765+765 ≈ 2648MB (~37% of peak),
    "context" ≈ 4422MB (~63%) -- Python/still-imported-torch/
    per-suffix-SparseLinearLayer-object overhead/tokenizer/etc.
    **Concrete, not-yet-done follow-up this surfaces**: storing
    embed_tokens/lm_head compactly at the same ~2 bytes/nnz used
    elsewhere would cost ~81MB + ~281MB ≈ 362MB instead of the current
    1530MB dense -- a real ~1.2GB reduction available, left on the
    table because these two tensors were assumed out of scope rather
    than actually checked.

## sili_peridot: compact embed_tokens/lm_head, direct follow-up

Took the ~1.2GB estimate above as the next task. Real result differs
from the estimate in an instructive way.

- **`lm_head` stays dense on purpose, and that's correct, not a bug.**
  Storing it as scipy CSR (standard int32-index + float32-value format,
  8 bytes/nnz) at its real ~70% density would cost ~1125MB -- MORE than
  its current 765MB dense. B3's own pruning already made this call
  upstream (only stores `{"csr": ...}` when the resulting format is
  actually smaller; `lm_head` gets `{"raw": ...}` instead) -- the
  ~1.2GB estimate wrongly assumed BOTH tensors would compress to
  sili's own compact ~2-bytes/nnz FP4 scheme, but embed_tokens/lm_head
  were never routed through `SparseLinearLayer` at all (out of B3/B5's
  original suffix set), so they only ever get scipy's plainer format
  when B3 decides CSR is worth it in the first place.
  `model/sili_model.py`'s new `_to_sparse_or_dense` just respects
  whatever B3 already decided (`"csr"` key present -> scipy CSR,
  `"raw"` -> dense) rather than forcing a format.
- **`embed_tokens` (real ~20% density) is where the real savings are**:
  scipy CSR costs ~309MB (154.2MB values + 154.2MB int32 indices +
  0.5MB row pointers) vs. 765MB dense -- a real ~456MB reduction for
  that one tensor, plus avoiding the transient `.to_dense()` spike
  during `build_sili_model` that used to briefly materialize it in
  full.
- **Measured total peak RSS: ~6.8-7.1GB before -> ~4.5GB after** (real
  checkpoint, `build_sili_model` + one real `compute_logits_sili`
  call) -- a ~2.3-2.6GB reduction, larger than the single tensor's own
  ~456MB would suggest on its own, consistent with also no longer
  paying that transient densify-then-discard spike during the build
  step. `scipy.sparse` used for this (already a transitive dependency
  via sili__new, not new) rather than hand-rolling a sparse-dense
  matmul in numpy -- a naive vectorized `[T, nnz]` intermediate for
  lm_head's matmul would be ~11GB for T~20-30 and nnz~140M, so this
  genuinely needed either scipy or a much more careful chunked
  implementation; scipy was the lower-risk choice given the time
  already spent this session. Verified correct via a synthetic test
  building the same weights both ways (CSR vs. densified-by-hand
  reference) and checking `compute_logits_sili` gives numerically
  identical results either way.

## sili_peridot: activation sparsity -- a real sili__new bug, then real results

Added `_forward` (`model/sili_block.py`) so every projection in a fold
step can route through DISLDO (`forward_dense`, current default) or
SISLDO (`forward_sparse`, when `activation_density` is set) -- keeps
only the top `round(density*n_features)` entries by magnitude per
token before the sparse forward pass. `activation_density` also
accepts a dict (per-suffix: sparsify only some of q/k/v/o/gate/up/
down) or a list of length `num_hidden_layers` (per-fold-step) via
`_density_for_suffix`/`run_folded_recurrence`'s `per_step` branch --
manual stand-ins for true per-layer adaptive sparsity, which doesn't
exist yet (needs energy RL / branching factor).

- **First real-checkpoint run collapsed to ~0.0 accuracy at EVERY
  density tested, including 0.9** -- a uniform floor, not the smooth
  degradation genuine information loss would produce. Traced through
  three layers of investigation, in order:
  1. `_cpu.dense_to_top_k_csr`'s `k` is a GLOBAL budget over the whole
     flattened `[rows, cols]` batch, not per-row (verified directly:
     5 rows x 8 cols, k=4 gave nnz-per-row `[3,0,0,1,0]`, not
     `[4]*5`) -- calling it once on the full multi-token `x` starved
     most tokens of any active features at all. Fixed in
     `_forward` by looping per row (later replaced, see below).
  2. Row-budget fix alone didn't help -- even 90% density still
     collapsed (0.0071 accuracy). Isolated with fresh-layer,
     single-call synthetic tests (avoiding a real but harmless
     `forward_sparse`/`forward_dense` return-value ALIASING gotcha:
     the returned array is a live view into the layer's own
     `output_buf`, silently overwritten by the layer's next
     forward call -- copy immediately if holding onto more than one
     result from the same layer instance) down to a genuine bug in
     **sili__new**: `csr.hpp`'s `top_k_indices` sorted candidates by
     raw signed value (`a.second > b.second`), not magnitude. For
     zero-mean data (any post-RMSNorm activation) that keeps only the
     largest POSITIVE entries and discards every negative one
     regardless of magnitude -- even at k close to the full row
     width, since it's the most-negative (often highest-magnitude)
     entries that get dropped first. Fixed upstream: sili__new PR #15
     (`fix/top-k-indices-sort-by-magnitude`, both `partial_sort`
     comparators now compare `std::abs(...)`), 855 assertions passing,
     merge pending user review as of this writing.
  3. With both fixes, the real-checkpoint curve became sensible:
     dense=0.2652 acc/173.37 ppl; 0.9=0.2591-0.2663 acc/~173 ppl
     (matches dense); 0.5=0.15-0.17 acc/~870 ppl; 0.2=0.008-0.03
     acc/~170k-370k ppl; 0.1 and below=0.0 acc. **Real cliff is
     between 0.5 and 0.2 density, not near the low end** -- so the
     5%/2%/0.5% densities from the original request are all well past
     collapse; only ~90% (matches dense, no speed win) and marginally
     ~50% (meaningfully degraded) preserve any real accuracy.
- **Separately, the per-row Python loop itself (`dense_to_top_k_csr`
  called once per token) was ~74x slower than necessary** -- measured
  directly (T=30, F=1536, density=0.5): 57.06ms/call looped vs
  0.77ms/call for a fully vectorized `np.argpartition`-based
  per-row top-k (partition + per-row sort + `np.take_along_axis` to
  gather values, then build the CSR triplet directly -- no C++ call
  per row at all). Replaced the loop with this in `_forward`; `idx`
  sets per row confirmed identical between the two approaches before
  switching. `num_cpus` dropped from `_forward`'s signature (was only
  ever used by the removed per-row `_cpu` calls).
- **Even with vectorized top-k, global activation sparsity still has
  no density where speed and accuracy both hold**: real-checkpoint
  eval time was 215s at density=0.9 (vs ~65s dense -- 3.3x SLOWER),
  101s at 0.5, ~62s at 0.2 (roughly dense speed), 40s at 0.1 and below
  (faster, but already collapsed). `forward_sparse`'s own per-synapse
  delta-CSR walk has real fixed overhead per active connection that a
  batched dense matmul doesn't -- confirmed with a clean (no
  concurrent load) isolated single-layer microbenchmark across 8
  different layer shapes (512-4096 dims), weight densities
  (0.3-0.9), and batch sizes (5-50 tokens): the dense/sparse
  wall-clock crossover point landed consistently in the **0.15-0.2
  density range** across all 8 configs, not just the one shape first
  measured. Stable enough across shapes to be worth exposing in
  sili__new itself as a real auto-dispatch threshold (see
  `hoyer_sparsify.hpp`'s existing-but-unwired hoyer-score machinery
  and TODO.md's planned "auto-dispatching version" -- this is real
  data toward the "not obvious" threshold decision mentioned there),
  though not yet proposed/implemented -- flagging as a candidate,
  not treating one session's benchmark as final.
- **Local (per-projection / per-fold-step) sparsity at density=0.2,
  real checkpoint, dense baseline 0.2652 acc / 173.37 ppl**:

  | variant | accuracy | perplexity |
  |---|---|---|
  | attn-only (q/k/v/o, all 24 steps) | 0.0557 | 8311.48 |
  | mlp-only (gate/up/down, all 24 steps) | 0.0157 | 72971.38 |
  | early-8 fold steps (all 7 suffixes) | 0.0077 | 11962.08 |
  | **late-8 fold steps** | **0.2148** | **569.97** |
  | middle-8 fold steps | 0.1261 | 2223.70 |

  Late-8 is close to dense; early-8 is nearly as collapsed as
  sparsifying everything globally. Matches the fold-depth recurrence's
  own structure (`state=0; for step: out=block(x+state); state+=out`)
  -- corrupting an EARLY step's output poisons every subsequent step's
  input via the accumulated `state`, while corrupting only the last
  few steps limits the damage to whatever's added at the very end.
  MLP-only also hurts more than attn-only, consistent with B3's own
  role-based pruning thresholds already treating MLP and attention
  weights differently for a similar reason (different parts of the
  network carry different amounts of irreplaceable signal).
- **Follow-up depth sweep, density=0.2, trailing N fold steps sparsified
  (rest dense), same 5-text eval**:

  | trailing steps | accuracy | perplexity | eval time |
  |---|---|---|---|
  | 4 | 0.1974 | 292.76 | 65.07s |
  | 8 | 0.2148 | 569.97 | 57.86s |
  | 12 | 0.1512 | 1569.46 | 57.09s |
  | 16 | 0.0622 | 8658.95 | 54.58s |
  | 20 | 0.0080 | 64926.17 | 51.90s |
  | 24 (=global) | 0.0319 | 167598.48 | 50.80s |

  Clear boundary: 4-8 trailing steps stay close to dense accuracy,
  12 is visibly degrading but not collapsed, 16+ falls off a cliff
  toward the same collapse as sparsifying globally. (late-24 here
  is the same "global 0.2" case as the earlier curve/local-sparsity
  runs -- 0.0319 vs those runs' 0.03/0.0077 is normal run-to-run
  noise on a 5-text eval set, not a contradiction.)
  eval time drops MONOTONICALLY as more trailing steps are
  sparsified (65s @ N=4 down to 51s @ N=24) even though accuracy
  does not improve monotonically past N=8 -- consistent with
  `forward_sparse` genuinely being cheaper per call at this density
  once contention/measurement noise is controlled for (matches the
  clean crossover benchmark below), but the win from sparsifying
  only a handful of steps out of 24 is modest in absolute terms
  (a few seconds out of ~60s total), since most of the network stays
  dense regardless.
- **late-8 also degrades gracefully (not catastrophically) as density
  drops further**, unlike the sharp global cliff: 0.2->0.2148 acc,
  0.15->0.1415, 0.1->0.0953, 0.05->0.0876 acc (172, 1193, 2024, 10682
  ppl respectively) -- some usable signal survives even at 5% density
  when confined to the last 8 steps, where the same density applied
  globally gives exactly 0.0.
- **Checked whether trading depth for density beats late-8@0.2**
  (the best point found so far): late-12 at lower densities does
  NOT help -- 0.15->0.0795 acc, 0.1->0.0400, 0.05->0.0077 (4603,
  28203, 472728 ppl), all worse than late-8's own values at the same
  densities (0.1415/0.0953/0.0876 acc). late-10@0.2->0.1438 acc,
  sitting between late-8 (0.2148) and late-12 (0.1512) as expected.
  **late-8@density=0.2 is the best accuracy/speed point found in
  this whole sweep** -- pushing either axis (more steps, or the same
  step count at lower density) past that point loses more accuracy
  than it's worth for this checkpoint/eval set.
- **Net read**: a uniform global density has no usable
  accuracy/speed tradeoff point in this checkpoint. Confining
  sparsification to the LAST ~8 fold steps does have real accuracy
  headroom (and confirms the fold-recurrence's state-accumulation
  structure is the actual mechanism, not a fluke of one density
  value), but the speed win from sparsifying only 8 of 24 steps is
  modest, not dramatic -- getting a genuinely large speedup out of
  this direction would need a smarter (adaptive/learned, e.g. energy
  RL) mechanism for deciding per-step/per-token density rather than
  a fixed manual schedule; naively pushing the fixed schedule further
  (more steps, or lower density) has already been checked and both
  lose accuracy faster than they gain speed. Not pursued further this
  session; flagging as the natural next step for whoever picks this
  back up.

## ULEB128 SIMD decode: prototyped in isolation, real speedup much smaller than hoped

Per the earlier plan ("sparse activation tests first, then ULEB128 SIMD
optimizations"), prototyped the "batch-decode groups of 8, verify all
single-byte via SIMD, fall back to scalar on any multi-byte delta" idea
standalone (this machine has AVX2/BMI1/BMI2, no AVX-512, matching
sili__new's existing `-march=haswell` build flag) -- NOT wired into
sili__new, a throwaway benchmark
(`/home/simleek/.claude/jobs/4c378ed3/tmp/uleb128_simd_bench.cpp`, not
committed anywhere) using synthetic delta data matching the real
checkpoint's measured distribution (median delta=1, mostly single-byte).

- **First version** (widen 8 bytes via `_mm256_cvtepu8_epi32`, then a
  SCALAR loop for the running-sum/prefix-sum step): 1.34x speedup at
  0% multi-byte deltas, dropping to 0.80x (SLOWER than plain scalar)
  by 5% multi-byte.
- **Second version** (also vectorized the prefix-sum itself -- 8x
  uint32 prefix sum via 2 shift+add steps plus one cross-128-bit-lane
  carry broadcast, removing the scalar loop from the fast path
  entirely): only marginally better, 1.3-1.6x at 0% multi-byte
  (some run-to-run variance), still ~0.86x (slower) by 5%. Vectorizing
  the arithmetic did NOT meaningfully move the needle -- correctness
  verified against the scalar reference in both versions (bit-for-bit
  identical cumulative column indices).
- **Reading**: at N=2M synapses (roughly one suffix's real scale),
  this looks memory-bandwidth-bound rather than compute-bound --
  each byte is read once and each output written once either way, so
  batching the arithmetic only saves a fraction of the total time.
  The "8x" speculated earlier assumed the decode's sequential-
  dependency chain was the dominant cost; this prototype suggests the
  actual ceiling is much lower, and the "one bad delta forces the
  whole group of 8 through the scalar fallback" design pays for
  itself less and less as the real multi-byte rate rises above a
  couple percent (a smarter fallback that resyncs at just the bad
  byte instead of redoing the whole group might recover some of
  this, not attempted here).
- **This also isn't the only real bottleneck**: this session's
  earlier py-spy profiling of `disldo_forward` (see the b6/rank1
  investigation further up this file) found the single hottest LINE
  was `mo[b*n_out+col] += contrib` -- the scattered/strided
  accumulator WRITE, not the decode loop. Even a decode step with a
  true 8x speedup wouldn't directly fix that separate cost. Given
  this prototype's real (not assumed) numbers are much more modest
  than hoped, and the actual profiled hot line is elsewhere, this
  doesn't look like a good next investment as currently scoped --
  recommend re-profiling the CURRENT (post activation-sparsity-fix)
  code before sinking more time into ULEB128-specific SIMD work,
  rather than continuing on the original assumption.

**Re-profiled current code (py-spy, dense baseline, real checkpoint,
20500 samples) to check that recommendation before dropping it** --
confirms the prototype's implication directly: `uleb128_decode` is
only **0.33% of eval-phase samples**. Full eval-phase breakdown (build
phase separated out, since `fp4_quantize`/CSR-conversion-during-load
costs are a different question from per-token forward cost):
`disldo_forward`'s several hot lines together (121/117/119/120/106/122
-- the scattered accumulator write and its neighbors) are ~55% of
eval-phase samples, and `sgemm_` (BLAS) is ~34% -- traced to
`hidden @ lm_head.T` in `compute_logits_sili`, the full-vocabulary
logits matmul. That's a fundamentally unavoidable cost (any
implementation needs this exact operation to produce logits over the
whole vocab; `lm_head` stays dense on purpose per the earlier
embed_tokens/lm_head compaction work, since CSR would cost MORE at
its real ~70% density), not a sili-specific inefficiency to chase.
**Conclusion: ULEB128 SIMD decode is confirmed not worth pursuing
further in the current (dense-forward-dominant) pipeline** -- its
ceiling is under 1%. `disldo_forward`'s scattered-write pattern
remains the one real, identified, sili-specific hot spot, and it was
already flagged plus explicitly deprioritized by the user earlier
this session ("too much work for something that probably won't
result in much of a speedup at all" / "inference-only fast paths...
low-priority for now") -- not revisiting that call here, just noting
it's still the same answer with fresh data behind it.

## ULEB128 SIMD, round 2: six real design attempts, real numbers

User correctly pushed back on the first prototype's design (see above)
as fundamentally flawed, not just "tested and disproved" -- it had a
branchy all-or-nothing scalar fallback for any group containing a
multi-byte delta, and its "0% multi-byte" test case had a scalar
baseline whose branch predictor ran at ~100% accuracy on that specific
synthetic data, making the comparison unfair. Six substantively
different designs tried after that correction, all in
`/home/simleek/.claude/jobs/4c378ed3/tmp/uleb128_simd_v*.cpp`
(prototypes only, not committed -- correctness verified against a
scalar reference in every case before any timing was trusted):

1. **v1 (the flawed one above)**: branchy group fallback. 1.3x best
   case, negative once multi-byte rate exceeds ~2-3%.
2. **v2**: fixed the boundary math (value boundaries are fully
   determined by the continuation-bit pattern alone -- one SIMD
   `movemask` gives it for a whole 32-byte window, no need to decode
   payloads to find them) but still fell back to scalar decode for any
   window containing a multi-byte delta. Correct, bounded, but still
   has real scalar work in the (rare) escalation case.
3. **v3 -- group-varint re-encoding**: per user's explicit correction
   ("there shouldn't really be any slow or non-simd work... entire
   group is 8 bits per or it's 16 or more"), confirmed that TRUE
   zero-scalar-work decode requires the ENCODING itself to guarantee
   fixed width per group, not decoder cleverness on the existing raw
   ULEB128 stream (whose byte offsets are unknowable ahead of time
   without either a sequential scan or an encode-time guarantee --
   there is no way around this for the current format). New format:
   groups of 8 values, each group stored at a fixed 1/2/4 bytes/value
   (whichever its own max needs) behind a 1-byte tier descriptor.
   Decode dispatches once per group of 8 to one of three fully
   vectorized routines -- zero scalar work anywhere in the hot path.
   **Real per-row-scale result (N=100-8000, matching actual sili
   per-row nnz, not the earlier 2M-element stress test): ~1.6-2.5x
   speedup**, correctness verified exactly against scalar reference
   in every case. Costs ~12-60% more storage than raw ULEB128
   depending on multi-byte rate (descriptor tax + fixed-width padding
   for groups needing escalation), a real tradeoff for real hardware.
4. **v4 -- two-pass decoupled prefix sum**: diagnosed that even v3's
   zero-scalar design only got ~1.2-1.5x at large N, and that
   swapping working-set size (4K to 20M elements) didn't move the
   ratio much -- ruling out pure memory-bandwidth-bound. The actual
   limiter: the cumulative sum's cross-GROUP carry is inherently
   serial (each group's carry-in depends on the previous group's
   total), so even an internally-parallel per-group prefix sum still
   serializes group-to-group. Decoupled this into 3 passes -- (1)
   embarrassingly-parallel per-group LOCAL prefix sums + totals, (2)
   serial prefix-sum of the much-smaller (~N/8) totals array, (3)
   embarrassingly-parallel broadcast-add of each group's carry-in.
   Result: genuinely BETTER at cache-resident scale (2-3.7x at
   N=200K) but WORSE than scalar at large N (0.7-0.9x at 2M-20M,
   extra read/write pass over the output costs more than the latency
   it saves once bandwidth-bound) -- and at REAL per-row scale
   (100-8000), roughly comparable to or slightly worse than v3's
   simpler single-pass design (~1.5-1.9x vs v3's ~1.6-2.5x). Simpler
   v3 wins for the actual relevant scale.
5. **v5 -- wider lanes for the dominant tier**: per user's suggestion
   ("probably 32x instead if we're doing uint8 and clever"), kept the
   overwhelmingly-common tier-0 (single-byte) case in uint16 lanes
   (16-wide) instead of immediately widening to uint32 (8-wide) --
   safe from overflow since 16 single-byte deltas summed is at most
   4080, well inside uint16 range; only widens to uint32 for the
   final carry-in add. **~2.0-2.5x at real per-row scale** -- a real
   but modest improvement over v3's 8-wide version, roughly
   proportional to the wider lane count but not linearly so (matches
   v4's finding: cross-lane AVX2 operation latency, not raw lane
   count, is the binding constraint).
6. **v6 -- warp/SIMT-style decode across ROWS**: user's idea --
   sili already tracks known per-row starting byte offsets
   (`L.byte_start[row]`), so instead of fighting the intra-row
   unknown-boundary problem, run 8 independent scalar decoders in
   lockstep, one per SIMD lane, each walking a DIFFERENT row
   (matches GPU SIMT execution for divergent-length loops). Verified
   correct (both a scalar-lane-emulation version and a true-AVX2
   version matched the sequential reference exactly), but **5-10x
   SLOWER than plain sequential decode**, worse still with divergent
   row lengths (the exact scenario SIMT masking is supposed to help
   with). Root cause is CPU-specific, not a flaw in the idea itself:
   AVX2 has no efficient byte-granularity gather/scatter (had to
   gather/scatter via 8 individual scalar loads/stores per round),
   which (a) destroys the sequential memory-access pattern the
   hardware prefetcher relies on -- interleaving 8 independent
   streams touches 8 different cache lines every single byte instead
   of walking one region linearly -- and (b) for divergent lengths,
   pays a full vector-op round for every already-finished lane with
   no dedicated warp scheduler to hide that cost the way a real GPU
   does. A genuinely good idea that needs different hardware
   (AVX-512's better gather, or an actual GPU) to pay off here.

**Where these six left things**: converged on the same real ceiling --
~1.6-2.5x at real per-row scale, all still fundamentally limited by
SOME form of prefix-sum/cumulative-dependency latency, even the
zero-scalar-work designs (v3, v5). Superseded by attempt 7 below --
the actual fix was a different axis than any of v1-v6 tried.

## ULEB128 SIMD, attempt 7: the real fix -- eliminate the dependency via encoding

User's correction to the round-2 conclusion above: v1-v6 all still
encoded delta-from-PREVIOUS-element, which forces reconstructing
absolute values via a prefix sum no matter how cleverly the SIMD
itself is written (v4's two-pass design only removed the CROSS-group
portion of that dependency; v3/v5 still had an intra-group shift-add
tree). The actual fix is a different axis: encode
OFFSET-FROM-GROUP-START instead. Since deltas are all positive (column
indices are monotonic), every value in a group is independently
computable as `group_start + offset[i]` with NO dependency on any
other value in the group -- decode collapses to widening the
fixed-width offsets and ONE broadcast-add per group, no shift-add tree
at all. The group's own last (already-computed) output value doubles
as next group's `group_start` for free -- the cross-group dependency
that v4 needed a whole separate pass for here costs nothing extra.

Implemented as `ForCodec<G>` (Frame-of-Reference-style) in
`uleb128_simd_v7_for.cpp`, reusing v3's per-group width-tier
descriptor (1/2/4 bytes, chosen by each group's own max offset) but
applied to group-relative offsets instead of raw deltas. Tested
G=8/16/32/64, correctness verified exactly against scalar ULEB128
reference in every case (real checkpoint's data shape: median
delta=1, ~100% single-byte in the current per-delta encoding):

| G | speedup (0% multi-byte, real per-row N) | size overhead (0% mb) | size overhead (1% mb) |
|---|---|---|---|
| 8 | 1.7-2.6x | ~12-19% | ~17-20% |
| 16 | 2.8-3.9x | ~6-19% | ~17-22% |
| 32 | 4.0-5.3x | ~3-32% | ~27-53% |
| 64 | 4.3-12.0x (mostly 5-7x) | ~2-30% | ~52-53% |

**This clears the originally-hoped-for 8x at several points (G=64:
up to 11.98x at N=500)**, and sits solidly in the 4.5-7x range at
larger G for the density that actually matches the real checkpoint
(0% multi-byte column). Bigger groups amortize the 1-byte tier
descriptor over more values (lower overhead when tier stays low) but
are more exposed to a single large delta forcing the WHOLE group to a
wider tier (overhead grows faster with G as the multi-byte rate
rises) -- a real, visible tradeoff in the table above, not
hand-waved. Given the real checkpoint's actual distribution is
overwhelmingly single-byte (this session's earlier measurement:
~100% single-byte for the suffixes sampled), G=32 or G=64 both look
like strong, honest candidates -- G=64's higher peak speedup comes
with a noisier/higher worst-case size tax if the real multi-byte
rate turns out higher than the sampled suffixes suggested.

**This is the first attempt (of 7) that looks genuinely worth
integrating into sili__new**, not just documenting as a negative
result. It would need: a new encoder (replacing the current
delta-to-ULEB128 packer) and a new decoder (replacing
`DeltaCSRRowCursor`'s sequential `advance()`/`uleb128_decode`), plus
handling for the size/tier tradeoff above. Not yet started --
this is still prototype-only
(`/home/simleek/.claude/jobs/4c378ed3/tmp/uleb128_simd_v7_for.cpp`),
`sili__new` itself is untouched. Worth a real PR if the user wants to
proceed, given how much stronger this result is than attempts 1-6.

## B8 paused: tile-recurrence architecture (Kimi K3 collaboration)

Phase 2.7 (unifying B8's window mechanism onto one real trainable
`gaussian_attention` call, see the two entries above this) shipped
cleanly, but it didn't fix the real underlying problem: the window
caps at 24 fold-depth positions, each holding a SINGLE 1536-dim
carried-state vector -- nowhere near a real transformer's KV cache
(thousands of distinct token vectors). Confirmed by direct benchmark:
the attention math itself is cheap at that scale (<1.5% of
`apply_window_step`'s cost even at window_size=24); representational
*capacity*, not compute, was the actual bottleneck.

The user worked out an alternative architecture with Kimi K3 (an
external model, no knowledge of this repo) and pasted the full spec:
one persistent recurrent state `M[num_tiles, tile_dim]`, decoupled
entirely from fold depth -- `num_tiles` is a free axis. Per direct
decision: **B8's fold-depth-window curriculum is PAUSED** (not
abandoned -- `feature/b8-suffix-unification` stays as-is) in favor of
prototyping this. Backed by real prior art (reservoir computing/ESN,
RTRL, e-prop, Clockwork RNN) claiming an emergent internal "clock"
gives content-addressable memory without hand-coded NTM read/write
heads -- a real, unproven architectural bet, not something this
prototype validates (see below).

**Went through several rounds of correction against the generic
Kimi-generated spec** (full detail in the approved plan,
`feature/tile-recurrence-prototype` branch,
`model/tile_recurrence.py`'s own docstrings):
- Weight reuse: bootstrap the SHARED tile network from ONE already-
  folded fold position (`step_layers[i]`), not train from scratch --
  confirmed mechanically feasible (batched `forward_dense` over
  `[num_tiles, hidden]` works fine, shapes match exactly).
- Positional encoding: first drafted "drop RoPE, `gaussian_attention`'s
  center/sigma replaces it" -- wrong, center/sigma are fixed constants
  tied to tile INDEX, not a true relative-offset encoding. Then
  over-corrected to "no positional encoding at all." Final, correct
  answer (direct user correction): **just use RoPE** -- tiles have
  real, known sequence positions once injection is the full sliding
  window, so RoPE applies exactly like `apply_fold_step` already does.
  `gaussian_attention`'s center/sigma stays as an independent,
  complementary, learnable locality prior on top of RoPE, not a
  substitute for it.
- Input injection: first drafted "inject only into the last tile" --
  wrong, corrected to full sliding-window injection into EVERY tile
  each tick (`tile[j] = x[i-(num_tiles-1)+j]` when that index exists).
  Naturally reduces to "only the last tile gets real input" at the
  very first tick, not a special case.
- State update: additive/gated (`M_new = M_prev + energy_dynamics(
  attn_out)`), matching Kimi's own residual formula literally -- this
  is what makes "inject raw content into every tile every tick" still
  compatible with genuine memory persistence (the injection only feeds
  this tick's computation, it doesn't overwrite `M_prev`).
- No new training mechanism needed: `SparseLinearLayer`'s existing
  inline/local training (self-updates during `forward_dense`, no
  backprop from a downstream loss) already satisfies Kimi's own
  "no BPTT, `M_t-1` fully detached" requirement. Only `centers`/
  `log_sigmas` need real `Tensor` backprop -- a single non-recurrent
  op per tick, same as Phase 2.7 already validated.

**Sizing/speed, measured not assumed -- and a real error caught and
fixed along the way**: initial benchmarks (including the FLOP-parity
ceiling estimate) used 10% density, picked without checking. Real B3
-achieved density is 80-93% (this file, "B3 pruning" entry above,
`gate_proj` mean row occupancy ~80%) -- redid the comparison at 85%,
same sili/FP4 engine on both sides for a fair test:

  ORIGINAL, 24 real distinct layers, one real token, sili/FP4, 85%:
  **1460 ms/token**
  TILE, num_tiles=8,  same engine/density: 104.8 ms/tick -> **13.9x faster**
  TILE, num_tiles=32, same engine/density: 344.5 ms/tick -> **4.2x faster**

Notably, num_tiles=32 does slightly MORE raw FLOPs than the original
24-layer pass (32 tiles x one shared matrix ~= 1.54B FLOPs vs the
original's ~1.16B) yet is still ~4x faster in wall time -- consistent
with the cache-locality hypothesis raised mid-design: one small shared
weight matrix reused across many tile-rows stays cache-resident, vs
the original's 24 DISTINCT matrices that can't share residency at
all. Caveats: single-run timing, randomly-initialized fake CSR layers
(not real trained weights), no KV-cache-style reuse modeled on the
original's side -- real, but not a rigorous benchmark yet.

**Given these real speedups already, batch-parallelism (a genuinely
new C++ kernel task -- confirmed `sili__new` has no real batch-parallel
op today, passing a bigger array into `forward_dense` does NOT get
real batch parallelism, only whatever row/thread parallelism it
already does) was explicitly deferred per direct decision** -- noted
as future optimization work in `todolist.md`, not pursued now.

**Not tested at all, and there's no basis to expect it yet**: the
"emergent clock" hypothesis, or any other emergent property from the
cited literature. Phase T1 only built and sanity-tested the
architecture (shapes, gradient plumbing, injection-formula
correctness, statefulness under a FIXED untrained weight set) --
nothing has been trained, so nothing could have emerged. Real
validation needs an actual training loop (not built) plus new
instrumentation designed to detect it (not designed) -- both real
future work, not close to done.

See `/home/simleek/.claude/plans/fuzzy-plotting-starlight.md` for the
full approved plan and design rationale; `model/tile_recurrence.py`
and `tests/test_tile_recurrence.py` on `feature/tile-recurrence
-prototype` (11/11 tests passing) for the actual implementation.

## Tile-recurrence toy validation: real bugs found, learning signal inconclusive

Before wiring the real B1-B7 conversion pipeline onto tile-recurrence
(dropping the paused fold-depth-window design), built a toy-scale
validation harness to prove the architecture can actually LEARN, not
just run a forward pass without crashing (all Phase T1 verified).

**Two toy models** (`model/toy_recall_models.py`): `ToySmallTransformer`
(real stacked causal dense transformer, own weights per layer) and
`ToyTileRecurrence` (one shared tile network + `gaussian_attention`
across tiles, Q/K/V from `normed(x_window)+normed(M_prev)` so memory is
genuinely attend-able, not just added in afterward). Both route every
linear projection through `sili.sparse_rnn.DISLDOLayer` (Tensor-graph
-integrated, self-updates inline via `backward_dense`) rather than
`sili_block.py`'s frozen-inference convention. Kept Kimi's staggered
per-tile "column" prediction (tile j predicts what tile j+1 currently
holds) per direct decision -- the tile-shaped descendant of this
project's own A3/A4 column-averaging work.

**Task**: synthetic associative recall / "induction head"
(`model/toy_recall_task.py`) -- a cue-response bigram planted early,
repeated `lag` positions later, correct prediction at that point
requires genuine retrieval. Standard synthetic test for exactly the
property tile-recurrence exists to have more of.

**Building this surfaced THREE real upstream `sili__new` bugs** (all
fixed, PR #30):
1. `reduce_sum`'s backward broken for any axis other than 0 (needed
   for a batched RMSNorm).
2. `add()`/`mul()`'s backward didn't reduce broadcasted gradients
   (only matched when both operands already shared the same shape).
3. **The serious one**: `SparseLinearLayer.forward_dense`/
   `forward_sparse` and `DISLDOLayerV.forward` returned a numpy array
   ALIASED to the layer's own reused internal `output_buf` -- any
   caller holding a PREVIOUS call's result alive across a subsequent
   call to the same layer silently saw it change to the new call's
   answer. Not a crash -- wrong answers that looked like a design bug
   (a "tile-recurrence ignores M_prev" test failure turned out to be
   this, not the architecture). Root-caused via
   `np.shares_memory()` confirming two genuinely different inputs'
   returned arrays shared the same backing memory. This affects
   `DISLDOLayer` broadly, not just this toy work -- worth being aware
   of for any other code in this project using it.

**Real training result (`scripts/train_toy_recall_comparison.py`),
reported plainly, not dressed up**: inconclusive at the calibrated
settings. First pass (vocab=16, lr=0.005, 400 steps/lag) left even the
DENSE baseline -- which should be a near-ceiling reference, given full
attention over the whole sequence -- stuck exactly at chance (0.05).
Recalibrated (vocab=8, lr=0.02, 2000 steps/lag): dense only reliably
beat chance (12.5%) at lag=2 (0.45); lag=4/8/16 hovered at/barely above
chance (0.10/0.17/0.23), with visible numerical instability (overflow
warnings) during the longer run. Tile-recurrence's numbers
(0.10/0.12/0.15/0.05) show no clear signal relative to dense either
way. This is a genuinely hard task from FRESH sequences every step (no
memorization shortcut) -- matches real induction-head literature (these
take real training to emerge, not instant convergence) -- not evidence
the architecture can't learn it, but also not evidence it can, yet.

**Not resolved, real follow-up work**: getting a confident answer needs
actual training-loop engineering this session didn't have room for --
gradient clipping (the overflow warnings are a real signal, not
noise), a proper LR schedule, possibly batched (not pure per-step
online) training over multiple sequences at once, and/or substantially
more steps. Tracked in `todolist.md`.
