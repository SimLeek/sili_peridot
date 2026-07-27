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
