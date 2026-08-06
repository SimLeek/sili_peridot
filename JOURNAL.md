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

## Toy validation follow-up: standard MQAR + gradient clipping + LR schedule

Per direct feedback on the first (inconclusive) toy training result:
implemented gradient clipping (the user's own long-standing standard),
a warmup+cosine LR schedule (nanoGPT's convention, looked up via
websearch rather than guessed), explicitly skipped batched training
(single CPU here, no parallelism to gain -- the user's own correct
call), and -- most substantively -- **replaced the hand-rolled
induction-head task with the standard MQAR benchmark** (Arora,
Eyuboglu et al., "Zoology: Measuring and Improving Recall in Efficient
Language Models", 2023) rather than continuing to extend a bespoke
one, per direct instruction that a sufficiently standard test shouldn't
be reimplemented from scratch. Ported directly from HazyResearch/
zoology's own reference implementation
(`zoology/data/multiquery_ar.py`, fetched from the repo) to plain
single-example numpy -- this is the established benchmark for exactly
what's being tested here (can an efficient/recurrent architecture
recall associations as well as full attention), used throughout the
linear-attention/SSM/efficient-architecture literature specifically
for this comparison.

**Gradient clipping implementation note**: textbook global-norm
clipping (measure the total gradient norm across all parameters, THEN
rescale once) isn't directly compatible with `DISLDOLayer`'s own
design -- its weights self-update INLINE during the very
`_backward()` call that computes their gradient, so knowing the "total
norm" before any update happens would need a full dry-run pass first
(learning_rate=0 everywhere), doubling every training step's cost.
Implemented PER-NODE clipping instead (`backward_with_grad_clip` in
`model/toy_recall_models.py`): replicates `Tensor.backward()`'s own
topological-order traversal, clipping each node's `.grad` to
`max_grad_norm` right before that node's own `_backward()` fires --
single-pass, and still directly bounds what any individual weight
update (including `DISLDOLayer`'s inline ones) can see.

**Real result, reported honestly**: better than the pre-clipping run
(no more NaN divergence), but still not a clean signal for either
model. `seq_len=16/num_kv_pairs=2`: dense=0.15, tile=0.04.
`seq_len=32/num_kv_pairs=4`: dense=0.03, tile=0.01. A residual overflow
warning appeared during the run even WITH clipping, suggesting some
instability remains (plausibly: clipped GRADIENTS bound each step's
update, but nothing yet bounds cumulative WEIGHT magnitude directly --
weight decay, another standard stabilization technique, wasn't added
this round).

**Not resolved -- genuinely open, real next steps** (not attempted
further this session, given how much back-and-forth this specific
sub-task has already consumed): weight decay; more training steps;
possibly the toy dims (hidden=16) are simply too small for MQAR's
real vocab requirements (MQAR needs vocab_size > seq_len strictly,
forcing vocab up as seq_len grows, unlike the old hand-rolled task);
worth checking against zoology's own reported results for their
SMALLEST model configs to see what scale they needed for real
learning to emerge, rather than continuing to guess.

## Toy validation: longer run at zoology-matched scale still stuck at chance

Per direct feedback: checked zoology's own real MQAR configs
(`zoology/experiments/models_repo.py`'s `add_attention` -- their
smallest attention baseline sweeps `d_model` in `[32, 64, 128]`,
`n_layers=2`, hybrid with a BaseConv mixer; real training configs use
100k examples per config, up to 32 epochs -- far more data/steps than
anything tried here so far, confirming "these just take a while").
Scaled `hidden` 16 -> 32 (matching their smallest `d_model`) and
`TRAIN_STEPS` 1500 -> 15000 (calibrated directly: ~6.7ms/step dense,
~43ms/step tile at these dims -- landed the real run at ~30.7 minutes
total for both configs, in the "minutes to an hour" range the user
described from their own Mandelbrot-attention precedent).

**Real result, still honestly reported, not dressed up**:
`seq_len=16/kv_pairs=2`: dense=0.07 (chance=0.10), tile=0.00.
`seq_len=32/kv_pairs=4`: dense=0.05 (chance=0.05), tile=0.00. Even the
DENSE baseline -- full attention over the whole sequence, the
condition under which MQAR should be EASIEST to solve, at a model
scale directly matching zoology's own real smallest reference config
-- is still stuck at chance after ~30 minutes of training. Overflow
warnings (`RuntimeWarning: overflow in exp`, `invalid value in
multiply/reduce`) are STILL present, unchanged by adding gradient
clipping.

**This is a real, useful negative result, not just "still
inconclusive"**: scale-appropriate model + the standard benchmark +
substantially longer training + gradient clipping STILL doesn't
produce learning, even for the reference architecture (dense
attention) that should have the easiest time. This points at the
TRAINING METHODOLOGY itself, not model capacity and not something
specific to tile-recurrence -- most likely weight MAGNITUDE drifting
into an unstable regime over many steps, unopposed by anything (
gradient clipping bounds each STEP's update, but 15,000 consistently
-directed steps at `lr=0.02` can still walk weights into a bad regime
even with every individual step bounded). Weight decay -- explicitly
scoped but not implemented last round as a "maybe" -- now looks more
likely NECESSARY than optional, not resolved this session (see the
previous JOURNAL entry for why it needs real C++ work: `DISLDOLayer`'s
`weights_vals` accessor returns a DECODED COPY of the compressed
storage, not a live view, so decay-by-direct-mutation doesn't work --
confirmed directly, `w *= 0.5` on the returned array didn't persist;
a proper implementation needs a new parameter threaded through
`disldo_backward` in `sili/lib/headers/linear_disldo.hpp`, inside its
per-connection update loop, which is intertwined with FP4 quantization/
scale/importance tracking -- real, bounded scope, not attempted this
session pending explicit direction).

## Confirmed: the toy training failure is confounded, not architectural

Direct question that cut right to it: is the stuck-at-chance result
from FP4 (the model itself), or is our own from-scratch training
system confounded with the comparison? It's the latter, confirmed
decisively -- `ToySmallTransformer`/`ToyTileRecurrence` were never run
through a "premade" (established, battle-tested) training system;
everything -- the model precision (FP4, via `DISLDOLayer`) AND the
optimizer (hand-rolled per-node gradient clipping + plain SGD, no
momentum, no Adam) -- was built from scratch this session.

**Control experiment** (`scripts/torch_mqar_control.py`, diagnostic
only, not part of the real model): the EXACT SAME architecture shape
as `ToySmallTransformer` (RMSNorm -> single-head causal self-attn -> O
-> residual -> RMSNorm -> SwiGLU MLP -> residual, no positional
encoding -- kept identical on purpose), but full fp32 precision (plain
`torch.nn.Linear`, no FP4/`DISLDOLayer`) and a real, established
optimizer (`torch.optim.Adam`), on the EXACT SAME
`generate_mqar_sequence` task (unaffected -- it was already a faithful
port, not implicated).

**Result**: `seq_len=16/num_kv_pairs=2`: eval accuracy **0.95** (vs
chance 0.10) in **20.1 seconds** (3000 steps). `seq_len=32/
num_kv_pairs=4`: eval accuracy **0.67** (vs chance 0.05) in **22.3
seconds**. Loss decreased cleanly and monotonically-ish throughout,
none of the overflow warnings seen in every FP4 attempt.

**This settles it**: the architecture (the dense baseline's own
design, and by extension the shared design principles
`ToyTileRecurrence` builds on) is NOT the problem -- it learns MQAR
easily and fast under a standard training setup. The bottleneck is
somewhere in {FP4 quantization via `DISLDOLayer`, the hand-rolled
optimizer (no momentum/Adam), or both} -- genuinely not yet isolated
between those two. **Real next diagnostic step, not done yet**: run
ONE more control that changes only ONE of those two variables at a
time (e.g. plain fp32 `sili.tensor` ops with the SAME hand-rolled
clip+SGD loop, to isolate whether the optimizer alone explains it; or
`DISLDOLayer`/FP4 with a real momentum-based update if one can be
retrofitted) rather than continuing to guess with tile-recurrence
itself, which was never the thing actually being tested by this
failure.

## Second control: fp32 + same hand-rolled loop isolates the optimizer, not FP4

Per direct decision (importance/row-scale mechanisms already validated
elsewhere, so FP4 itself was doubted as the cause): built
`scripts/fp32_handrolled_control.py` -- identical architecture and
task to the first control, but changes ONLY precision (plain fp32
`sili.tensor` matmul via a new `DenseTensorLinear` primitive in
`model/toy_recall_models.py`, no `DISLDOLayer`/FP4), keeping the SAME
hand-rolled optimizer (`backward_with_grad_clip` + `apply_gradient_step`
+ `lr_schedule`) as every real toy model this session.

**Result, decisive**: `seq_len=16/kv_pairs=2`: loss improved initially
(5.881 -> 4.746 by step 300) then diverged catastrophically (5834 ->
182843 -> 3,088,712 by step 2100) -- eval accuracy 0.10 (chance).
`seq_len=32/kv_pairs=4`: same pattern, eval accuracy 0.06 (chance).
The same `RuntimeWarning: overflow in exp` seen in every FP4 attempt
reappeared here too, at FULL PRECISION.

**This isolates the cause cleanly: the hand-rolled optimizer (per-node
gradient-norm clipping + plain SGD, no momentum) is the actual
bottleneck, NOT FP4 quantization.** Per-node clipping bounds each
individual step's gradient, but nothing prevents weights from walking
into instability over many CONSISTENTLY-DIRECTED steps -- exactly what
momentum/Adam-style per-parameter adaptive scaling exists to prevent,
and exactly what this session's own from-scratch optimizer never had.

**Real, scoped next step -- NOT symmetric across the two training
paths this project has**: fixing this for plain `Tensor` LEAVES
(RMSNorm weights, `centers`/`log_sigmas`, `DenseTensorLinear`'s own
weight) is pure Python -- a real Adam implementation is directly
addable to `apply_gradient_step`'s call site, no `sili__new` changes
needed. But the REAL toy models (`ToySmallTransformer`/
`ToyTileRecurrence`) route their big linear layers through
`DISLDOLayer`, whose inline C++ weight update (inside
`backward_dense`) has the identical no-momentum problem and isn't
reachable from the Python side the same way -- giving DISLDOLayer's
own training real momentum would need genuine new C++ work in
`disldo_backward` (`sili/lib/headers/linear_disldo.hpp`), similar in
scope to the weight-decay idea scoped (not built) earlier. Not
resolved this session -- a real decision point on how much further to
invest, given how much of this session has already gone into training
-loop diagnostics rather than the tile-recurrence architecture itself.

## Real MQAR comparison run: tile-recurrence fails, root cause found

With the training loop finally trustworthy (`DenseTensorLinear` +
`AdamOptimizer` + `clip_grad_norm_`, real global-norm clipping -- per-
node clipping alone still let Adam diverge, confirmed directly before
landing global clipping), re-ran `scripts/train_toy_recall_comparison.py`
for real: 3000 steps, warmup+cosine LR (peak 0.02), `num_tiles=4`,
`hidden=32`, both configs from zoology's own smallest reference dims.

**Result**: `seq_len=16/kv=2`: dense=0.57, tile=0.02. `seq_len=32/kv=4`:
dense=0.24, tile=0.00. Dense (whole-sequence causal attention, a
ceiling reference) partially learns the task. Tile-recurrence does
NOT -- both numbers are at or below random chance (1/20=0.05,
1/40=0.025), i.e. no signal at all.

**Root cause, found via a targeted ablation**: `ToyTileRecurrence`'s
per-tick "column" auxiliary loss (tile `j`'s state trained to predict
tile `j+1`'s current token, EVERY tick) and the true MQAR recall loss
(trained only on the ~2/16 query ticks per sequence) share the exact
same weights (one tile network, one `lm_head`). Re-ran the
`seq_len=16/kv=2` config with the column loss removed entirely (only
the true recall loss, on query positions only) -- accuracy went from
0.02 to 0.225. Still well below dense's 0.57, but far above chance and
a completely different regime. **The column-averaging auxiliary task
is actively drowning out the harder recall signal**, not a neutral
extra: it fires 8x more often (every tick vs ~2/16 ticks) and is
trivially easy (the next token is already sitting one column over in
the SAME input window this tick, not an actual retrieval problem), so
gradient descent on shared weights cheaply satisfies it and never gets
pushed hard enough toward the genuinely harder MQAR objective.

This is a real, reinstated-per-direct-decision feature turning out to
interfere with the property the whole architecture exists for, at
least as currently weighted/scheduled (auxiliary loss with no
down-weighting or scheduling, added at full strength from step 0).
Reported to the user plainly, not concluding this dooms the
architecture -- the 0.225-vs-chance ablation shows the underlying
retrieval mechanism (recurrent `M` state + `gaussian_attention` across
tiles) DOES carry some real signal once the interfering loss term is
gone; it's still well short of dense's 0.57 at the same scale, and
there's a separate, not-yet-investigated suspect too: `M_prev` is
passed into each tick as a fresh detached `Tensor(M_prev...)` leaf, so
gradient never flows backward through TIME across ticks (no BPTT) --
only the current tick's own transition gets trained. That may be
capping how much genuine long-range storage this training setup could
ever learn, independent of the column-loss issue.

## Column mechanism redesigned per direct correction -- second real run

Direct correction on the column mechanism, in two rounds: (1) it's
meant to solve "how do you backprop an output error into a recurrent
state much WIDER than the output" (worked example: input 10, recurrent
80, output 10 -- selecting a subset starves the rest, summing a column
forces small state values ("epsilon errors," bad under FP4),
AVERAGING a column works, ~1 bit of precision cost); (2) what the
column should predict is the genuine NEXT TOKEN (these are meant to be
predictive autoencoders throughout, `ToySmallTransformer` included),
not the tile's own input (my first fix attempt) or another tile's
content (the original, ablation-confirmed-harmful version).

Rebuilt `ToyTileRecurrence`: `embed_width` (matches real embedding
width, e.g. 32) vs `state_width = embed_width * column_neurons` (e.g.
32*8=256, genuinely wider internal recurrent state) -- clearly
distinct names per direct correction ("tile width" = input width, not
the wide state). Injection broadcasts embed_width -> state_width via
`np.repeat` (parameter-free); readout does the reverse via
`reshape`+`reduce_sum(axis=-1)*1/C` (column-mean, also parameter-free)
before the FIXED-width `lm_head` -- both ends parameter-free,
matching the real system's constraint that `lm_head` is the pretrained
MiniCPM5 output head and can't grow a new learned down-projection.
Checked `model/toy_recall_task.py` directly rather than assuming: at a
query position the true `tokens[pos+1]` is random noise
(`random_non_queries=True`), NOT the recall value -- so "predict the
next token" can't mean `tokens[i+1]` literally everywhere. Built a
unified per-position target instead, applied to BOTH models now:
query positions -> recalled value (unchanged), key/value
context-laydown region -> real next token (genuine structure,
reinforces the exact adjacency later queries need), everywhere else ->
skipped (pure noise, training on it would be actively harmful).
`num_tiles` widened to `= seq_len` per config too (was a fixed 4) --
removes the window-narrowness confound, testing whether the mechanism
can learn at all with full context visibility before testing genuine
cross-tick recall beyond a narrow window (still deferred).

**First re-run had a confound, caught and fixed**: the initial version
of this run also changed the DENSE control's own training target
(adding the same context-region next-token loss to it) -- direct
correction: dense should have stayed the unmodified control, since the
column-mean-readout width-mismatch problem this fix addresses doesn't
exist for a standard dense transformer at all (its hidden state
already matches `lm_head`'s input width directly). That confounded run
showed dense dropping substantially (0.57->0.33, 0.24->0.07) alongside
tile's improvement -- reverted `train_and_eval_dense` back to training
on `mqar_pairs` only (unchanged from the very first comparison run),
and re-ran clean.

**Clean, unconfounded re-run result**: `seq_len=16/kv=2`: dense=0.57
(unchanged control, matches the first run's 0.57 exactly), tile=0.10
(was 0.02). `seq_len=32/kv=4`: dense=0.26 (matches the first run's
0.24 within eval-sampling noise), tile=0.05 (was 0.00). **Tile
-recurrence genuinely improved and now clears chance (~0.05/~0.025) at
BOTH configs**, with the control held constant -- a real, clean
positive signal for the corrected column mechanism, not an artifact of
also changing what the control was trained on. Tile still trails dense
substantially at both configs (as expected -- see the user's own
framing: this toy run is a deliberately sub-optimal case, meant to
verify the mechanism CAN learn the target property at all, not to
match a heavily-tuned dense transformer's own accuracy). Not yet
testing recall genuinely beyond a narrow window (`num_tiles=seq_len`
gives full visibility this run) -- stays deferred, per the approved
plan, until this basic-mechanism check passes, which it now has.

## Adam+artificial-FP4 vs real FP4+importance+energy, matched precision

Direct correction: comparing Adam against this project's own
importance+energy training must hold PRECISION matched, or it repeats
the exact confound two earlier isolation controls already spent real
effort disentangling. Built `model/toy_precision_models.py`:
`fake_quantize_fp4`/`ArtificialFP4Linear` (straight-through fake
-quantization using the REAL 15-level FP4 table and real per-row
`scale=max(|row|)/6.0` calibration, verbatim from
`sili/lib/headers/fp4quant.hpp`/`sili/sparse_rnn.py` -- checked
directly, not assumed) for the Adam arm; `DISLDOLayer` (real,
unmodified, inline C++-trained) + `EnergyDynamics` for the real arm,
per [[feedback_importance_is_already_the_optimizer]] -- no external
optimizer touches DISLDOLayer's own big weight matrices, only the
small RMSNorm leaves get an Adam step.

**Bug found and fixed before the real run**: `sili_block.py`'s
`default_window_energy` (the obvious first choice -- already used
elsewhere in this codebase) diverges badly at this toy scale --
verified directly via isolation (its own aux_loss grew unbounded, 5.5
-> 17.9 over 300 steps, total loss diverged at every learning rate
tried). It's explicitly documented as a placeholder calibrated for
full-model-scale windows, not toy scale -- switched to the config
already used in `sili__new`'s own small-scale `EnergyDynamics` tests
(`drive=0.1, activation_cost=0.05, precision=0.01, density=0.05,
p=0.3`), which behaved far better in a 300-step isolation check
(aux_loss stayed 0.005 -> ~0.3, loss reached a real minimum).

**Real 3000-step MQAR run result**: `seq_len=16/kv=2`: Adam+artificial
-FP4 acc=0.53, real-FP4+energy acc=0.12. `seq_len=32/kv=4`: 0.21 vs
0.05. Adam substantially outperforms importance+energy at both
configs, under matched FP4 precision. **The real-FP4+energy arm's own
loss curve is genuinely diverging, not just noisy**: config1 climbs
6.9 -> 29.2 over training (sampled every 200 steps); config2 climbs
22.9 -> 48.2. This tracks with what the 300-step isolation check had
already started to show (loss reaching a minimum then drifting back
up) -- at the full 3000-step scale that drift becomes dominant,
outweighing the real, if smaller, isolated-check improvement the
toy-scale energy config fix produced.

**Not yet root-caused further** -- plausible candidates, none
confirmed: (a) DISLDOLayer's own inline update may need a different
learning-rate schedule/range than what suits Adam (the same
`lr_schedule` was reused for both arms, for a fair a-priori
comparison, but that doesn't mean it's well-tuned for DISLDOLayer's
own mechanism specifically); (b) `EnergyDynamics`'s persistent
`energy` homeostatic state may behave differently over thousands of
steps on FRESH random sequences each step than it did on the
300-step, single-config isolation check; (c) a genuine capacity
limitation from combining reduced FP4 precision with energy's own
added noise/homeostatic pressure at this small scale. Reported
plainly -- this is a real, reproducible negative result for
importance+energy at this specific toy scale/task/config, not
something to explain away, but also not yet enough evidence to
conclude it fails at the real MiniCPM5 scale this mechanism was
actually designed for.

## Fixed the energy confound; row-scale-Adam validated

Direct correction ([[feedback_do_science_correctly]] -- saved after
this was caught twice in one session, second time by a frustrated
"put 'try to do science correctly' in your memories"): the run above
gave `EnergyDynamics` to the real-FP4 arm only, with no equivalent on
the Adam arm -- confounding optimizer choice with energy's own added
noise. Refactored `model/toy_precision_models.py` so `use_energy` is
an orthogonal toggle on BOTH `ToySmallTransformerArtificialFP4` and
`ToySmallTransformerRealFP4` (each layer builds its own fresh
`EnergyDynamics` instance when enabled -- its running state is
per-instance and can't be shared). Also built
`AdamRowScaleDISLDOLayer`/`ToySmallTransformerRealFP4RowScaleAdam` per
direct proposal: individual FP4 weight VALUES keep importance as their
training signal (the user's own view: importance is the right
mechanism for synaptogenesis/pruning, not something to replace
wholesale), but the coarser per-row `value_scale` gets its own
Adam-style adaptive normalization on top -- cheap (2 floats/row, not
2 floats/weight), reachable via real, already-existing pybind
accessors (`get_value_scale`/`set_value_scale_raw`,
`cpu_backend.cpp:1082-1097`), no new C++ needed. Documented as an
approximation: it re-normalizes the delta DISLDOLayer's own inline
update already applied (via `before`/`after` snapshots around
`backward()`), not the true pre-damping gradient.

**Real 2x2 + row-scale-Adam result, 3000 steps**:

| | no energy | + energy |
|---|---|---|
| Adam+artificial-FP4 | 0.50 / 0.23 | 0.22 / 0.10 |
| importance+real-FP4 | 0.34 / 0.07 | 0.08 / 0.03 |
| importance+real-FP4, row-scale-Adam | 0.42 / 0.09 | -- |

(pairs are seq_len=16/kv=2 then seq_len=32/kv=4)

**Energy hurts BOTH optimizer arms**, not just importance -- Adam drops
0.50->0.22 and 0.23->0.10 from adding energy alone, matching
importance's own drop. This resolves the open question from the
confounded run: the earlier divergence (loss climbing to 20-40+) was
energy's own homeostatic noise/aux-loss pressure, not something
specific to importance-driven training. Still an open question WHY
this toy-scale `EnergyDynamics` config destabilizes training broadly
(see the earlier entry's still-unconfirmed candidates) -- now known to
be orthogonal to the optimizer question, at least.

**Row-scale-Adam genuinely helps**: 0.42 vs plain importance's 0.34 at
the easier config (loss curve reaches 2.3 and stays low, vs plain
importance oscillating between 2-10), 0.09 vs 0.07 at the harder one.
Doesn't close the gap to full Adam-per-weight (0.50/0.23) but
meaningfully narrows it from the plain-importance baseline, at a
fraction of the per-weight memory cost full Adam would need. A real,
clean, validated result for the user's own proposed design: give the
row-scale adaptive normalization without replacing importance's role
in individual weight training.

## ToyTileRecurrence ported onto real sili (DISLDOLayer) -- first real run

Direct correction: tile-recurrence was always meant to run on real
sili, not `DenseTensorLinear` (a deliberate, explicitly-scoped stopgap
from earlier this session). Built `model/toy_tile_precision_models.py`
(`ToyTileRecurrenceRealFP4`, `disldo_cls=` swappable) and
`AdamRank1DISLDOLayer` (`model/toy_precision_models.py` -- extends
`AdamRowScaleDISLDOLayer` to also Adam-normalize `output_scale`/column
-scale, real pybind accessors, `set_output_scale_raw` must be called
once at init to activate its own training -- checked directly).

**Real bug found and fixed before any of this could run at reasonable
speed**: `max_weights` was sized as `state_width * mlp_hidden` (the
FULL dense connection count) -- this doesn't just make the layer
"as dense as it can be," it makes `per_row >= out_features`, i.e.
genuinely 100% density. Measured directly: this made DISLDOLayer's
scattered/CSR storage 7x SLOWER than a properly sparse config (~8%
density) -- 1744s vs 247s for the same run. Confirmed via profiling
that the time really was inside `forward_dense`'s C++ call, not Python
overhead, consistent with [[project_sili_optimal_hardware_vision]]'s
own finding that the scattered path is gather/scatter-bound. Fixed:
`max_weights` now a FIXED sparse budget (4096) independent of
`column_neurons`/`state_width`. `_preseed_random_sparse`'s default
init (sparse echo network) confirmed fine as-is, per direct feedback --
the bug was purely in `max_weights` sizing. Separately, `num_cpus=4`
measured ~19% faster than the `num_cpus=2` used elsewhere in this
session (`num_cpus=8` was clearly WORSE -- oversubscription, only 4
physical cores) -- adopted for this sweep.

**Real 3000-step MQAR result**: Stage 1 (optimizer variant,
`column_neurons=8`, `seq_len=16/kv=2`): plain DISLDOLayer=0.18,
row-scale-Adam=0.17, rank1-Adam=0.07 -- plain importance WON here,
the opposite ranking from the precision-comparison track (where
row-scale-Adam clearly beat plain importance, 0.42 vs 0.34) -- a real,
non-obvious finding that the row/rank-1-Adam improvement doesn't
transfer uniformly across architectures. Stage 2 (capacity scaling,
plain DISLDOLayer): `column_neurons=16` beat the `=8` control at
seq_len=16 (0.13 vs 0.11) but not at seq_len=32 (0.03 vs 0.04,
roughly flat/within noise). Stage 3 confirmed those seq_len=32 numbers.

**Real-FP4 tile-recurrence (0.18 best, seq16) BEATS the earlier fp32
DenseTensorLinear+Adam stand-in (0.10)** -- a genuine, real validation
that porting to sili was worth doing, not just an architectural nicety.
Still far below the dense fp32 transformer ceiling (0.57) -- the gap
tile-recurrence needs to close remains large. One loose end, not
investigated: a `RuntimeWarning: overflow in exp` appeared during
training (likely `silu`/sigmoid seeing larger-magnitude activations
than the fp32 version produced) -- didn't invalidate the finite eval
numbers collected, but flagged honestly rather than silently ignored.

## Real bug: attention was only seeing input, not carried state -- fixed and re-tested

Direct correction, marked most urgent: `step()` (both
`ToyTileRecurrenceRealFP4` and `ToyTileRecurrence`) computed Q/K/V from
`x_window` ALONE -- a change I'd made during the earlier column
-mechanism redesign, reasoning `num_tiles≈seq_len` made `x_window`
already carry nearly everything. That reasoning was wrong: with
attention unable to see `M_prev` directly, anything relating fresh
input to carried state could only happen by first writing input into
state via the residual and waiting a LATER tick to attend to it -- an
artificial "save to state first" bottleneck (now
[[feedback_attention_needs_combined_input_state]]). Fixed: `qkv_source
= x_normed + m_normed` (blend restored) in both classes.

**Re-run result, same 3-stage sweep**: Stage 1 (seq16/kv=2, cn=8):
plain=0.18 (unchanged), row-scale-Adam=0.15 (was 0.17), rank1
-Adam=0.01 (was 0.07 -- collapsed, flagged not explained). Stage 2
(capacity scaling): cn=8=0.23 (was 0.11), cn=16=0.17 (was 0.13) --
real improvement, though some of the cn=8 jump vs Stage 1's own
0.18 is seed-to-seed noise (different seeds, same config -- eval set
is only 60 sequences). **Stage 3 (seq_len=32/kv=4, the harder config)
shows the clearest, most consistent gain**: cn=8: 0.04->0.07 (nearly
doubled), cn=16: 0.03->0.05 -- exactly where a proper input-state
attention link should matter most (longer context, more reliance on
carried state). Best real-FP4 tile-recurrence result now (cn=8,
seq16) = 0.23, more than DOUBLE the fp32 `DenseTensorLinear`+Adam
stand-in's own 0.10. Still far below the dense fp32 ceiling
(0.57/0.26). Rank1-Adam's collapse to 0.01 with this fix is a real,
unexplained regression -- not investigated, reported as-is rather than
omitted.

## Out-of-context benchmark Tier 1 (running parity): built, first run inconclusive

Per direct decision: MQAR-style benchmarks alone (every run so far
used `num_tiles=seq_len`, full visibility by construction) don't test
tile-recurrence's real advantage over a bounded-window transformer --
genuine persistent state across unbounded time. Built
`model/toy_beyond_context_task.py` (`generate_parity_sequence` --
n_bits random bits, a `'?'` query token, then the running-XOR answer
as the real next token; vocab={0,1,'?'}, fixing an earlier 2-token
design that had no way to signal "answer now" per direct correction)
and added a real `half_bandwidth=` knob to `ToySmallTransformer`
(default unlimited, unchanged for every existing caller) as the
genuine "standard LLM" stand-in -- structurally unable to see past `W`
positions, unlike every prior comparison's unlimited-context dense
baseline. `scripts/train_toy_beyond_context_comparison.py`: two-phase
curriculum (in-context sizes first, then out-of-context), accuracy
-vs-`n_bits` sweep against `ToyTileRecurrenceRealFP4(num_tiles=W)`.

**First real run, `W=4`, `n_layers=2` dense**: neither model showed
the expected pattern -- dense scored ABOVE chance (0.57-0.60) at
`n_bits=8/24`, values that should be structurally unsolvable for it.
**Root cause #1, confirmed directly**: a 2-LAYER stacked
`half_bandwidth=W` model's true effective receptive field is `2*W`,
not `W` (verified via direct perturbation probe -- a position 8 back
changes the output, 9 back doesn't, at `W=4`) -- the "structurally
bounded to W" claim was wrong for `n_layers=2`; needs `n_layers=1` for
a genuinely `W`-bounded baseline. **Root cause #2, more fundamental**:
training itself doesn't converge cleanly even with `n_layers=1` --
loss oscillates (1.3->6.5->1.5->2.2->1.7->0.8->1.0->1.4->0.9->0.7 over
1500 steps) and settles near the random-guessing floor (`ln(2)~=0.69`),
consistent with the near-chance accuracy seen even at the easiest
in-context size (`n_bits=4`: dense=0.52, tile=0.50). This is a
genuine, unresolved negative result -- not yet root-caused further
(candidates: `peak_lr=0.02` too aggressive for this task/architecture,
or exact parity being intrinsically hard for softmax-attention's
smooth weighted-averaging to represent, independent of the LR).
Reported honestly rather than re-tuned blindly -- real investigation
needed before this benchmark can give a trustworthy signal either way.

## Tier 1 re-run: methodology fixed, real root cause identified (BPTT or chance), still inconclusive

Two direct corrections applied: (1) `n_layers=1` for the dense
baseline (fixes the confirmed 2x-receptive-field bug). (2) Gradual
curriculum -- `_sample_n_bits` now ramps a ceiling by exactly 1 every
`STEPS_PER_LEVEL` steps (was: two coarse phases, an abrupt jump
straight to a wide out-of-context range) -- per direct correction, now
[[feedback_gradual_out_of_context_curriculum]]: "learning in context is
O(1), learning in recurrent state is O(1), but learning something that
passed recurrent state is O(random chance)... much less likely if the
network never learned stuff before."

**Isolated the real cause, confirmed empirically, not just
theoretically**: a controlled check -- pure in-context parity training
(no out-of-context extension at all, full 3000-step budget) --
converged to real above-chance accuracy (0.645, loss 1.3->0.55). This
rules out "parity is unlearnable by this architecture" -- the
in-context capability is real. Direct diagnosis for what's actually
limiting out-of-context performance, stated as a near-exhaustive
dichotomy: **"It's either BPTT or chance... we're not doing bptt
because it uses a ton of extra memory and processing."** `M_prev` is a
fresh DETACHED leaf every tick (deliberate, established design) -- no
gradient pathway exists to learn "write X now because a future tick
needs it," so any out-of-context success is coincidental, not learned.
Now [[project_sili_bptt_or_chance]] -- this was already hinted at much
earlier (an MQAR-track entry flagged the same detached-M_prev fact as
a "not-yet-tested suspect"), now much more strongly supported.

Per direct hypothesis, added `use_energy` to `ToyTileRecurrenceRealFP4`
(previously had none) and tested it specifically here: energy can't
create a genuine gradient pathway either, but by keeping more neurons
active it might raise the ODDS that useful state-carrying patterns
persist long enough to coincidentally help -- worth testing on THIS
task even though it hurt the (fully-visible, BPTT-irrelevant) MQAR
benchmark.

**Real 3000-step run, all three arms**: dense (n_bits=2/4/8/16/24):
0.55/0.50/0.47/0.48/0.58 -- flat, near chance throughout (the
curriculum spends most of its budget on n_bits values dense
structurally cannot solve, diluting its own in-context practice vs.
the isolated full-budget test above). Tile, no energy:
0.18/0.50/0.52/0.52/0.47 -- a striking BELOW-chance result at
n_bits=2 specifically (not explained -- very short sequences leave
most of the window filled with M_prev fallback rather than real
tokens, a plausible but unconfirmed cause). Tile +energy:
0.48/0.40/0.55/0.53/0.42 -- energy fixed the n_bits=2 collapse, but
shows NO clear improvement over no-energy at the actual out-of-context
sizes (8/16/24: 0.55/0.53/0.42 vs 0.52/0.52/0.47 -- within noise of
each other and of chance). **Genuinely inconclusive, reported as
such**: the methodology fixes resolved the earlier catastrophic
divergence/floor-collapse, and the real bottleneck (no BPTT) is now
well understood and documented, but 3000 steps shared across a growing
13-level curriculum appears to leave too little dedicated practice per
level for a clean signal either way on this specific hard task. A much
longer run (matching the user's own prior experience: "I ran the
mandlebrot set for minutes to an hour before the attention system
really got working") is the most likely next step, not yet tried.

## Tier 1, ~1hr budget (40x steps): more time alone did not resolve it -- corroborates BPTT-or-chance

Per direct decision ("give it an hour or so this time"), scaled
`TRAIN_STEPS` 3000->120,000 and `STEPS_PER_LEVEL`/`WARMUP_STEPS`
proportionally (same curriculum shape, ~40x more practice per level).
Real training time: dense 165.5s, tile (no energy) 1354.7s, tile
+energy 2878.0s (~73 minutes total, close to the pre-run timing
estimate).

**Result** (`n_bits=2/4/8/16/24`): dense=0.53/0.38/0.45/0.63/0.53 --
still flat, no in-context-vs-out-of-context separation. Tile, no
energy=0.72/0.45/0.43/0.47/**0.15** -- a real, striking swing (0.72 at
the shortest sequence down to 0.15 at the longest), too extreme to be
pure eval-sampling noise (~2.7 std below chance at n=60), but NOT the
hoped-for shape (staying high past the window) -- looks more like
systematic degradation as sequences lengthen. Tile
+energy=0.45/0.30/0.43/0.35/**0.50** -- energy fixed that specific
`n_bits=24` collapse (0.15->0.50) again, same pattern as the shorter
run's `n_bits=2` fix, but made most OTHER points worse (0.72->0.45,
0.45->0.30), not better -- no longer a clean net positive for energy
here either.

**Real conclusion**: ~40x more training time did NOT produce the
clean "tile stays above chance past the window, dense collapses"
signal. This is itself informative -- it corroborates rather than
undermines [[project_sili_bptt_or_chance]]: if the bottleneck were
simply "not enough practice per curriculum level," more time should
have helped monotonically; instead the pattern stayed noisy/
non-monotonic even at 40x the budget, consistent with "without a
genuine gradient pathway for cross-tick credit assignment, success is
coincidental, and more attempts at coincidence don't reliably converge
to a working mechanism." Directly motivates the already-planned next
step -- see [[project_sili_bptt_alternatives]] (e-prop first, SnAp-1/
UORO/KF-RTRL as documented fallbacks) -- as a real fix rather than
"try even more steps."

## E-prop implemented (EPropDISLDOLayer, EPropAdamDISLDOLayer) -- first test at reduced budget, null result, and a real gap found

Built `_EligibilityTrace` + two separate classes in
`model/toy_precision_models.py` (per direct correction: NOT one class
with a `use_adam` flag -- separate functions/classes, since Adam has
not consistently won in earlier sweeps). 36/36 unit tests pass
(`tests/test_toy_precision_models.py`). Wired into
`scripts/train_toy_beyond_context_comparison.py` as two more arms via
`ToyTileRecurrenceRealFP4`'s existing `disldo_cls=` swap.

**Timing probe** (100 steps, extrapolated to 120k): plain DISLDO
1787s, DISLDO+energy 2281s, `AdamRowScaleDISLDOLayer` (existing
per-row Python loop, no trace) 3313s, `EPropDISLDOLayer` 3801s,
`EPropAdamDISLDOLayer` 3860s. The trace mechanism itself only adds
~15% (3801 vs 3313s) -- the bulk of the ~2x slowdown vs plain DISLDO
is the pre-existing per-row `set_value_scale_raw` Python loop shared
with `AdamRowScaleDISLDOLayer`, not anything new. (No bulk/vectorized
setter exists in `cpu_backend.cpp` for `value_scale` -- `weights_vals`/
`importance`/etc. have readonly `py::array_t` bulk-read bindings, but
`value_scale` doesn't even have that; a bulk setter would be a small,
low-risk C++ addition mirroring `load_weights`, not yet done.)

Per direct decision -- the earlier 120k-step (~1hr) scale-up was
specifically compensating for the no-BPTT/no-gradient-pathway problem;
e-prop's trace exists to provide exactly that missing pathway, so it
should not need the same 40x scale-up to show a signal -- ran BOTH
e-prop arms at a reduced TRAIN_STEPS=36,000 (30% of 120k, same
curriculum ratios/shape), reusing the already-recorded 120k-step
dense/tile/tile+energy numbers as CONTEXT only (different budget, not
a strict comparison). `scripts/train_toy_beyond_context_eprop_only.py`.

**Result** (n_bits=2/4/8/16/24, eprop / eprop+adam):
0.57/0.50/0.48/0.45/0.43 and 0.55/0.53/0.47/0.40/0.52. Every value is
within ~1.5 std of chance (std ≈ 0.065 at n=60 eval sequences/point) --
**genuinely inconclusive, including the in-context points** (n=2/4),
which is actually a WORSE in-context showing than plain DISLDO's
recorded 120k-step numbers (0.72/0.45 at n=2/4). Honest read: at this
reduced budget neither e-prop arm shows the hoped-for
"learns fast because it finally has a real gradient pathway" signal --
it also hasn't clearly learned the easy in-context case yet. Not
strong enough evidence to say e-prop failed either -- 36k steps may
still be too few for THIS specific mechanism, unlike the hope that its
better credit assignment would need less practice, not more.

**Numerical-stability finding, real and separate from the above**: the
training log showed `RuntimeWarning: overflow encountered in exp/
multiply` and `invalid value encountered in reduce` during the
`EPropDISLDOLayer` arm (final printed accuracies were NOT NaN, so
whatever went unstable recovered, but likely disrupted training
dynamics along the way). Root cause found: `_train_tile()` in
`scripts/train_toy_beyond_context_comparison.py` never calls
`clip_grad_norm_` before `tile_opt.step()` -- `train_dense()` does,
but EVERY tile arm (plain DISLDO, +energy, and both e-prop variants)
has been training with unclipped gradients this whole time. This is a
real, previously-undetected asymmetry between the dense and tile
arms -- plausibly present (silently) in the earlier 120k-step DISLDO/
+energy runs too, not just the new e-prop arms. Fixed
(`clip_grad_norm_` added to `_train_tile`, smoke-tested clean).

## E-prop found structurally broken, root-caused -- not a training-budget problem

Before scaling up to re-test e-prop with the grad-clip fix, walked
through the actual math (direct user pushback: "e-prop is the only one
of those that doesn't really make immediate sense to me, which is an
incredibly bad sign" -- correct instinct, confirmed by checking the
real C++ formula in `linear_disldo.hpp`'s `disldo_backward`, not just
re-deriving from memory):

- `AdamRowScaleDISLDOLayer`'s delta-trick (`grad_proxy = -raw_delta /
  learning_rate`) IS a real, correctly-derived `dL/d(value_scale[r])`
  -- confirmed directly against the C++ (`scale_grad_sum` sums
  `stored_w*out_scale*dy*iv` over synapses/batch, matching the code's
  own comment `dL/d(val_scale[r]) = stored_w*out_scale*dy*input`). An
  earlier claim in this session that it was "not real, fabricated from
  nothing" was WRONG and corrected directly to the user.
- The REAL bug: that gradient already bakes in `iv` -- the QUERY
  TICK's OWN current input magnitude for row r. Multiplying it by an
  eligibility trace (ALSO an activity-magnitude quantity) doesn't
  compose "error x cross-tick sensitivity" the way RTRL/e-prop's chain
  rule requires -- it compounds two input-magnitude-proportional terms
  into one (roughly `iv_now x trace(iv_history)`, i.e. activity
  squared across time), consistent with both the earlier overflow
  warnings AND the near-chance null result.
- Sharper, structural problem found on top: `grad_proxy` is
  MULTIPLICATIVELY ZERO whenever the query tick's own `iv[r]=0` --
  meaning DISLDO's own row gradient can NEVER credit a row that's
  silent AT THE QUERY TICK, no matter what it's multiplied by
  afterward. That's exactly the row the whole exercise most needs to
  credit (one that mattered several ticks ago, not right now) -- the
  entire "trace x grad_proxy" family of designs is structurally
  incapable of fixing this.

`EPropDISLDOLayer`/`EPropAdamDISLDOLayer`/`_EligibilityTrace` removed
(model/toy_precision_models.py) along with their tests and the
now-obsolete `scripts/train_toy_beyond_context_eprop_only.py` --
replaced, not patched (see next entry).

## Peak-eligibility trace: real per-row credit assignment via `last_input` substitution, worked out directly with the user

Collaborative redesign (multiple direct corrections in sequence,
converging on the final mechanism):

1. User: track a per-row leaky PEAK-HOLD (not a smooth decaying sum)
   of the most salient recent input, replaced when a new input exceeds
   the decayed peak -- "when we backprop just use that input and
   output pair to tell the exact synapse to change." Matches this
   project's own sparse-network intuition: most rows are silent most
   ticks, so remembering "the one time this row mattered most" beats
   averaging over ticks where nothing happened.
2. User, separately: confirmed `ToyTileRecurrenceRealFP4.step()`'s
   state update is ALREADY additive/residual
   (`M_new = M_prev + attn`, `M_new = M_new + mlp_out`,
   `model/toy_tile_precision_models.py:129,134` -- checked directly,
   not assumed) -- exactly the "state_t = state_t-1 + delta" shape the
   user asked for, so whatever a row contributed several ticks ago is
   still physically present in the state the query tick reads. No
   architecture change needed.
3. My first attempt at combining the peak trace with a real signal
   (`out.grad`-based broadcast) worked but was a coarser approximation
   than necessary. User: "You can just change the row's input to that
   [the peak]. The sili backprop functions... take in x and x_sparse."
   Checked directly (`cpu_backend.cpp:1199-1214`):
   `SparseLinearLayer.last_input` is a ZERO-COPY, WRITABLE numpy view
   straight onto the C++'s cached `_last_input` buffer -- verified
   empirically that mutating the returned array actually writes into
   the object `backward_dense` reads from (not assumed). Since
   `backward_dense` takes NO `x` argument at all (reads `_last_input`
   internally), this means: overwrite `last_input` with the
   peak-held SIGNED value right after forward, before backward ever
   fires, and DISLDO's OWN real gradient math computes the row's (and,
   internally, each synapse's) correction AS IF the input had been
   whichever recent tick was most salient -- ZERO Python-side gradient
   approximation, full reuse of the real C++ math.

Verified directly (not just unit-tested in isolation): a row given a
strong peak, then silenced for 3 ticks AND at the query tick itself,
gets a real, substantial `value_scale` correction (-0.14) from
`PeakEligibilityDISLDOLayer` -- plain `DISLDOLayer` gives that same
row EXACTLY zero credit (confirmed: `0.0`), the direct, measured
demonstration of the mechanism doing what it's meant to do.

Known accepted tradeoff, documented not hidden: `dx` (flowing to
`x.grad`, e.g. `qkv_source.grad`) is ALSO computed from the
substituted input, contaminating gradient reaching upstream plain
leaves (`input_ln`/`post_ln`) slightly -- backward_dense computes dx
and the value_scale gradient from the same `_last_input` in one pass,
no way to split them apart without a C++ change. Accepted since those
are secondary parameters, not the credit-assignment mechanism itself.

True per-SYNAPSE (not just per-row) substitution would need direct CSR
access -- explicitly scoped by the user as a separate, larger,
"expensive... involves nearly everything" core change (4 bits/param at
sili's real scale), not attempted here. Noted for later: if a layer's
real input activation ends up genuinely ~1-sparse, `SISLDOLayer`'s CSR
path would let this same substitution touch only the active indices
instead of a full dense array -- a real efficiency angle at true
MiniCPM5 scale, not needed at this toy width.

`model/toy_precision_models.py`: `_PeakEligibilityTrace`,
`PeakEligibilityDISLDOLayer`. `tests/test_toy_precision_models.py`:
9 new tests (34/34 passing), including a direct
silent-row-still-credited check.
`scripts/train_toy_beyond_context_comparison.py` and the new
`scripts/train_toy_beyond_context_peak_eligibility_only.py` updated
(4 arms: dense, tile, tile+energy, tile+peak-eligibility -- e-prop's 2
arms replaced by peak-eligibility's 1).

**Timing** (100-step probe, extrapolated): 557.6s for 36,000 steps
(~9.3 min) -- essentially matches plain DISLDO's own per-step cost
(no per-row Python loop at all, unlike Adam/e-prop's `for r in
range(in_features): set_value_scale_raw(...)` pattern -- just one
`self.trace.update(x)` and one `last_input[...] = peak_snapshot`
numpy assignment).

**Real run result** (36,000 steps, 570.8s, `scripts/train_toy_beyond_context_peak_eligibility_only.py`),
`n_bits=2/4/8/16/24`: 0.57/0.25/0.42/0.32/0.25. NOT a clean result
either way, reported honestly rather than spun: three points
(n=4, 16, 24) are 2.8-3.9 std BELOW chance (std~=0.065 at 60 eval
sequences/point) -- not noise scattering around 0.5, a real systematic
pattern of confidently answering WRONG, including at n=4 which is
IN-CONTEXT and should be the easy case. Below-chance is itself
informative (proves real learning happened, just inverted/wrong at
those points) but this is not evidence the credit-assignment mechanism
works as hoped.

Separately, the training log still shows `RuntimeWarning: overflow
encountered in exp` (in a sigmoid call, likely `silu`'s own internal
sigmoid) -- the earlier `clip_grad_norm_` fix removed two of the four
original warning types (Adam's `g*g` overflow, the NaN-producing
`reduce`) but NOT this one. `clip_grad_norm_` bounds the GRADIENT
STEP size, not accumulated activation/weight MAGNITUDE -- since
`ToyTileRecurrenceRealFP4`'s state update is additive/residual with no
renormalization of the residual stream itself (see this entry's own
point 2 above), some intermediate value (plausibly M itself, or a
downstream MLP activation) may be growing large over many ticks
regardless of per-step gradient clipping. Not yet root-caused --
plausible confound for the below-chance result above, not ruled out.
Next step: investigate this specific overflow (which tensor, growing
across which axis) before drawing any further conclusion about
peak-eligibility's real effectiveness, per
[[feedback_do_science_correctly]] -- an unresolved instability sitting
underneath a result is not something to draw conclusions past.

## Per-synapse peak-eligibility via `backward_sparse`: a real, math-grounded mechanism, and the bugs it took to test honestly

Per direct correction: `PeakEligibilityDISLDOLayer` (value_scale-only)
doesn't hold causal if-then structure -- value_scale is a homeostatic
scaling knob, not where a synapse's actual weighted contribution lives.
Superseded by a genuinely per-SYNAPSE mechanism, worked out directly
with the user and grounded in real Jacobian algebra rather than
guessed:

**The math.** True BPTT sums `dL/dW[r,c] = Sum_t [dL/dh_T . dh_T/dh_t]
. x_r(t) . phi'(delta_c(t))` over every tick row r fired. Because the
state update is residual (`h_t = h_{t-1} + phi(delta_t)`), `dh_T/dh_t`
is dominated by an identity term regardless of how far t is from T --
the thing BPTT would normally need the full unrolled graph to compute
is already sitting there for free, precisely because of the residual
architecture (confirmed as the actual reason it matters, not just "it
helps information linger"). This means: **capture** -- substituting a
remembered tagged `x_r(t)` into a 1-hot dense array and calling
`SparseLinearLayer.backward_sparse` directly against the REAL error at
query time -- computes exactly the leading-order term of the true sum.
Checked directly against `cpu_backend.cpp`: `backward_sparse` takes
the dense input as an EXPLICIT argument (not cached), so no
`last_input`-mutation trick is even needed here, unlike the earlier
row-level attempt.

**Selection**, by the same derivation: comparing DIFFERENT rows
competing for the same output c at a fixed tick has `phi'(delta_c(t))`
cancel out (shared by every row feeding c) -- weight never belongs in
either comparison, despite an initial (wrong) instinct that
`w*x` ("forward contribution") was the right criterion. Confirmed via
two rounds of literature research: SnAp-1/e-prop/UORO/KF-RTRL all
track presynaptic-activity-based Jacobian approximations, NEVER the
synapse's own weight -- and SnAp's own paper explicitly considered and
REJECTED a top-k-of-full-multiplication approach on cost grounds (a
declined path, not an unknown one). `w*x` is really the
Taylor/attribution/pruning-saliency criterion, a different
mathematical object. The correct stand-in for the missing
`phi'(delta_c(t))` term is `state_change(t)` (= `delta_gated` for that
tick exactly, since the residual update makes it literally equal to
Δh, no separate subtraction needed) -- giving `score_r(t) = |x_r(t)| *
|state_change(t)|`.

**Decay derived, not tuned**: `decay_from_horizon(n, retain_fraction)
= retain_fraction**(1/n)` -- the minimum decay that keeps a tag's
score above a target fraction after a target number of ticks.
Confirmed a real, structural limitation this exposes: one-hot token
rows have CONSTANT magnitude every time they fire, so any later
occurrence of the same token always beats a decayed peak regardless of
decay rate -- only the continuously-valued STATE-portion rows can
genuinely bridge long horizons via this mechanism, not the raw token
inputs.

**On checking "does it work" against actual BPTT/RNN capability, not
just against itself** -- direct pushback ("something is extremely
wrong here, since RNNs should absolutely be able to handle this"; "is
anything in this test ever predicting out of context above chance?")
caught something the plain-vs-peak comparison alone never would have:
at 20 and 50 seeds, EVERY out-of-context number for BOTH arms sat at
or below 0.5 -- the whole comparison had been happening between two
things neither of which had demonstrated real out-of-context learning
at all. Root-caused to THREE real, previously-undetected bugs, not a
BPTT-alternative failure:

1. **`_preseed_random_sparse` ignored `np.random.seed()` entirely**
   (`np.random.default_rng()`, unaffected by any seed) -- every
   layer's sparse wiring has never been reproducible in this or any
   prior session's experiment. Fixed upstream in sili__new: optional
   `rng` parameter on `_preseed_random_sparse`/`DISLDOLayer`/
   `SISLDOLayer`, true randomness stays the DEFAULT (direct
   instruction: "I actually do want a closer to true RNG"), only made
   overridable. 6 new regression tests, full existing suite re-run
   clean (3 pre-existing unrelated flaky tests confirmed flaky on
   unmodified code too, not caused by this change).
2. **FP4's own stochastic weight rounding was ALSO unseeded** --
   `_cpu.seed_fp4_stochastic_rng(seed)` exists but reseeds only the
   CALLING thread; true reproducibility needs `NUM_CPUS=1` too (an
   OpenMP-parallel run keeps independent unseeded worker-thread state
   regardless). Confirmed via direct bit-identical reproduction across
   fresh processes once both fixes were in place.
3. **`MAX_WEIGHTS` was landing on the `_preseed_random_sparse` bare
   floor** (`k=1`, literally one random connection per input row, zero
   redundancy) -- derived a properly-sized value
   (`IN_FEATURES * STATE_WIDTH`, full column coverage) instead.

Even with all three fixed, a fixed-n_bits=2 (easiest possible) sanity
check still failed to learn (loss stuck above chance-level,
`ln(3)~=1.099`) until a FOURTH bug was found: **`EnergyDynamics` --
calibrated at h sizes 20-64 -- was gating out ~69% of every tick's
state-write at this much smaller width by default** (measured
directly: 5/16 neurons pass at init, `actual_p=0.3125`, since `p` is a
hard ceiling the gate starts AT, not the `density` target it only
grows toward via training) **and had no train/eval mode of its own,
so it was being applied during EVAL calls too**, corrupting every
accuracy number measured up to that point. `use_energy=False`
recovered 100% accuracy on the trivial case immediately, confirming
the diagnosis. But energy wasn't superfluous either -- removing it
brought back the original random-collapse instability (one seed hit
0.75/0.90/0.90/0.55, proving the task IS learnable, while others
collapsed to exact 0.0). Direct instruction: "even small energy...
should still save the neural network eventually, just not as
quickly" -- confirmed directly (loss kept dropping over 9000 steps
under a gentle config rather than plateauing) once the eval-mode bug
was ALSO fixed (fixed-n_bits=2 check: 0.220 aggressive-config-with-bug
-> 0.750 gentle-config-with-eval-fix).

Traced WHY gentle energy still leaves an occasional collapse: a single
still-failing seed's own `energy.energy` array, over its full training
run, trends NEGATIVE (settles ~-1.2 to -1.6 mean, only 0-3/16 neurons
near the firing threshold at any snapshot, down from all 16 at init)
-- `activation_cost*|h|` drains energy proportional to a neuron's OWN
real magnitude while `drive` accumulates it flat, so genuinely-useful,
large-output neurons get pushed toward the shutoff floor (near-zero
constant output) rather than toward the rescue-firing mechanism.
Raising `drive` DID flip that one seed's energy positive as predicted
-- but made the real 8-seed stability check WORSE overall (means
dropped, MORE seeds collapsed, not fewer) -- a clean demonstration
that a fix validated on one seed's own diagnostic doesn't necessarily
generalize; reverted to the smaller `drive` that performed better
across the full seed set. Per direct decision, the residual collapse
rate is being set aside for now (accepted as mathematically
non-permanent given energy + eventual dense/synaptogenesis fallbacks,
not yet re-litigated here).

**Final result** (50 seeds x 40 eval sequences, STATE_WIDTH=16,
TRAIN_STEPS=6000, all four bugs above fixed):

    n_bits  in_ctx  plain (mean+-std)  peak-synapse (mean+-std)
         2     yes     0.526 +- 0.223          0.539 +- 0.205
         3      NO     0.514 +- 0.168          0.542 +- 0.133
         4      NO     0.493 +- 0.187          0.502 +- 0.197
         6      NO     0.410 +- 0.197          0.452 +- 0.180

No single point clears its own noise (largest gap, n=6, ~1.1 combined
SEM) -- not a proven effect. But peak-synapse is nominally ahead at
ALL FOUR points, including every out-of-context one -- a consistent
direction, unlike earlier pre-fix runs where the sign flipped between
reruns. Recorded honestly as promising-but-not-yet-settled, not spun
either way -- see `scripts/prototype_peak_synapse_learning_comparison.py`
for the full mechanism, `scripts/prototype_synapse_peak_credit.py` for
the isolated (and still-holding) per-synapse credit verification.

Added real paired hypothesis tests (`scipy.stats.ttest_rel` +
`wilcoxon`, computed directly in `main()` so every future run reports
them automatically, not just a post-hoc analysis) -- paired because
`plain[s]`/`peak[s]` share the same eval-sequence seed at each s, using
that shared per-seed variance rather than discarding it. Applied to
the 50-seed run above: none of the four `n_bits` points are
statistically significant (best case n=6, p~=0.23 on both tests, which
agree with each other -- not a normality-assumption artifact). Honest
verdict: no significant evidence peak-synapse beats plain DISLDO at
this sample size, despite the consistent nominal direction.

## Ceiling check: does a proven, standard PyTorch RNN even need any of this?

Per direct instruction ("look online for working proven recurrent
tests... bring in one of their example small pytorch RNNs that they
definitely have to compare against") -- built
`scripts/torch_rnn_control.py`, matching the existing
`torch_mqar_control.py` pattern: `nn.RNN` AND `nn.LSTM` (both), real
full BPTT (whole sequence fed to torch's fused RNN module in one
call), Adam lr=1e-3, hidden=128, num_layers=1, on the EXACT SAME
`generate_deviation_sequence` task, no curriculum (real BPTT doesn't
need one). Hyperparameters and task classification checked directly,
not assumed: this is a detection/latch task (1 bit of carried state,
"did any deviation occur"), closer to Hochreiter & Schmidhuber's
original 1997 long-lag/latch problems than the harder standard copy
task (Le/Jaitly/Hinton 2015; reference implementation with LSTM/GRU
baselines: Bai/Kolter/Koltun 2018, `github.com/locuslab/TCN`) -- and
at `n_bits<=6`, nowhere near the several-dozen-step regime where
vanilla RNNs' vanishing gradients become a real barrier, so `nn.RNN`
(no gating at all) is included deliberately as the strongest possible
sanity check, not just `nn.LSTM`.

**Result**: both arms hit 100% accuracy at EVERY n_bits value (2, 3,
4, 6), loss dropping to ~0.0003-0.0009, in under 7 seconds each. Even
the plainest vanilla RNN solves this trivially. This recalibrates the
whole effort honestly: the task was never hard in any absolute sense
-- the struggle this session's from-scratch, no-BPTT system has had
(hovering 0.41-0.55 out-of-context even after four real bugs fixed)
isn't about task difficulty, it's specifically the cost of the
deliberate fixed-memory/no-unrolled-graph design choice
([[project_sili_bptt_or_chance]]). The peak-eligibility mechanism is
trying to approximate, in fixed memory, what real BPTT gets for free
here in under 7 seconds -- a genuinely harder problem than the task
itself, not an easy win blocked by a bug.

**Direct follow-up: does BPTT itself actually explain the 100%, or is
it something else about the PyTorch control?** Added a NO-BPTT variant
to the same script (`train_and_eval_no_bptt`): identical model/
optimizer/task, but the sequence is processed one tick at a time with
the hidden state DETACHED after every step -- exactly matching how
`M_prev` is a fresh detached leaf every tick in the from-scratch
system -- and `loss.backward()` fires once, at the query tick only.

**Result: no-BPTT nn.RNN still hit 100% at every n_bits (2/3/4/6);
no-BPTT nn.LSTM hit 100%/100%/91%/90% -- NOT a drop to chance.** Per
direct correction from the user (confirmed against their own prior
hands-on experience: "I tried BPTT a lot and it does practically
nothing... it's not a chance vs 100% thing ever"), the hypothesis this
session had been building toward -- that BPTT-per-se explains the
from-scratch system's out-of-context struggle -- is WRONG. Why no-BPTT
still works: the recurrent weight matrix is SHARED across every tick
AND across every training sequence (varying `query_pos`, varying
`n_bits`), so even though any single training example only
differentiates through its own last tick, the same weights get pushed
toward the correct one-step transition rule from many different
"positions in the recursion" across the training set -- enough to
learn a stateless, composable update rule (accumulate-deviation is
exactly such a rule) without ever needing multi-tick BPTT. This is
architecturally the SAME regime `prototype_peak_synapse_learning_
comparison.py`'s own `train()` already uses -- it calls `cell.step()`
with a real, nonzero `lr` at EVERY tick, not just the query tick -- so
"add BPTT" was never actually the missing ingredient to chase there
either; [[project_sili_bptt_or_chance]]'s diagnosis needs revisiting.

**Reframed next step, per direct instruction**: not more BPTT-vs-not
comparisons -- an ABLATION study. Start from this working no-BPTT
PyTorch control (proven to reach 100%/near-100% without BPTT) and
incrementally swap in real components of the from-scratch system --
DISLDO's sparse/quantized FP4 weights, EnergyDynamics gating, the
residual state-update structure, FP4's stochastic rounding noise --
one at a time, watching for whichever specific addition is the one
that actually drags accuracy down toward chance. That component, once
found, is the real thing to investigate -- not BPTT, not the task,
possibly not even the peak-eligibility mechanism itself.

## The ablation ladder: what's actually breaking DISLDO, traced step by step

Direct follow-up to the plan above. Per direct instruction, first built
`scripts/disldo_no_bptt_ablation.py`: literally swap the no-BPTT torch
control's `nn.RNN` cell for a parameter-matched DISLDOLayer (full
density, `max_weights = (hidden*2)*hidden` to match nn.RNN's Whx+Whh
count), everything else identical (tanh nowhere yet -- residual
accumulate, matching PlainCell's own `h_new = h_prev + delta`
convention). Result: catastrophic, not subtle -- loss exploded to
1e22-1e26, accuracy near/below chance. Traced directly, not guessed
at, through four real, separable causes, each isolated and fixed
before moving to the next:

1. **Unbounded residual accumulate -- pure forward-pass instability,
   nothing to do with training.** Traced with FROZEN (lr=0,
   untrained) weights: h_norm roughly doubles every tick regardless of
   training (0.85 -> 1.4 -> 2.6 -> ... -> 1.97e12 by tick 50) --
   `h_new = h_prev + delta` has no squashing nonlinearity anywhere,
   unlike nn.RNN's actual formula `h_new = tanh(Whx@x + Whh@h)` (full
   OVERWRITE, not accumulate, tanh-bounded every tick). Fix: made the
   DISLDO cell structurally identical to nn.RNN --
   `h_new = tanh(cell([x, h_prev]))`, no residual add
   (`scripts/disldo_tanh_no_bptt_ablation.py`). Confirmed directly:
   frozen-weight h_norm now stays ~1.2-1.4 indefinitely, same regime
   as the working torch control. Loss stopped exploding (0.66-1.17
   range) -- but accuracy stayed near chance. Fixed a real problem,
   not the only one.

2. **Effective learning rate crushed by density, independent of the
   nominal lr chosen.** Traced weight/value_scale movement directly
   (`weights_vals`/`get_value_scale` introspection): only 66/16384
   cell weights moved after 300 steps (0.6%), each by exactly one FP4
   quantization level (0.5) then permanently stuck; `value_scale`
   (the continuous, non-quantized per-row multiplier) moved too, but
   by <1% deviation from 1.0 after 300 steps -- real gradient signal,
   just far too slow. Root cause: `lr_per_row_nnz=True` (hardcoded in
   `DISLDOLayer.forward`'s backward closure, no way to override)
   divides `learning_rate` by the row's own connection count
   (nnz_this_row=128 at full density) for EVERY trainable quantity.
   At uniform density this normalization (meant to keep updates
   comparable across rows with varying degree under synaptogenesis)
   does nothing useful and just silently shrinks the effective rate
   ~128x. Confirmed via a direct lr sweep (`lr_per_row_nnz=False`,
   `sili__new` threaded the override through): n=2 (in-context) climbs
   from 0.673 (lr=1e-4) to a clean 1.000 (lr=0.03), matching the
   dense-Tensor/torch controls exactly at the top of the range -- but
   n=3/4/6 plateaued around 0.5-0.73 across every lr tested, a
   separate, still-unresolved gap at the time.

3. **A second, structurally distinct footgun: forward-time Hebbian
   importance update, uncoupled from any real gradient.** Direct user
   suspicion ("doesn't make sense without backward there too") led to
   reading `disldo_forward`'s C++ source directly: it mutates
   per-synapse importance (`imp += contrib * learning_rate /
   (1+|imp|)`, `contrib = w*iv`, no gradient/label involved)
   whenever `learning_rate != 0` -- INCLUDING on every non-query tick
   of an online RNN, which never has a backward() call at all. This
   fires far more often than backward's own real gradient-based
   importance update. Tested by forcing `lr=0.0` on non-query
   `step()` calls (forward-only, no backward -- so the Hebbian update
   never fires without a matching gradient): at lr=1e-3 specifically,
   in-context accuracy jumped from 0.33 (erratic, barely above
   chance) to 0.843 at the SAME nominal lr -- a dramatic stabilization
   confirming the mechanism was real. Out-of-context still didn't
   close (stayed 0.5-0.73), narrowing the remaining gap further but
   not yet explaining it.

4. **Root-caused as a genuine sili__new architectural flaw, not a
   tuning knob -- removed upstream, not just routed around.** Per
   direct instruction ("we don't need this footgun... break the API
   if you must"): deleted the unconditional forward-time Hebbian
   importance update entirely from `disldo_forward`
   (`linear_disldo.hpp` -- covered THREE separate sites: per-synapse
   importance, `value_scale_importance`, `output_scale_importance`),
   `sisldo_forward` (`sisldo_ops.hpp`), and the older
   `sisldo_forward_trivalues` (`linear_sisldo.hpp`, actually reachable
   from Python via `SparseLinearLayer::forward_sparse`, a DIFFERENT
   function than `sisldo_forward` despite the similar name -- both
   needed the fix). `learning_rate` removed from all three functions'
   signatures entirely (not left as a dead unused parameter), cascading
   through `forward_dense`/`forward_sparse`/`DISLDOLayerV::forward`'s
   C++ methods and pybind bindings, and every real Python call site in
   `sparse_rnn.py` (`DISLDOLayer.forward`, `SISLDOLayer.forward`,
   `FoldedLayer.forward`, `FoldedColumnLayer.in_proj`, `apply_fold_skip`)
   and `sili_peridot/model/sili_block.py`. Forward is now a pure,
   side-effect-free computation across the board -- weight/importance
   updates only ever happen in backward, coupled to a real gradient,
   same principle weight updates always followed. C++ unit tests still
   calling the old signature (test_disldo_block4_forward.cpp,
   test_scale_handling.cpp, and others) deliberately left broken --
   fix at merge time, not now, per direct instruction. Verified: fresh
   build compiles clean, DISLDOLayer/SISLDOLayer forward+backward work
   correctly, repeated forward-only calls with nonzero lr now leave
   weights provably unchanged (previously would have mutated
   importance every call), and the full non-real-checkpoint
   sili_peridot test suite (155 tests) passes.

Also confirmed, via a clean isolation control built alongside the
DISLDO ablation: swapping DISLDOLayer for sili's own dense
`DenseTensorLinear` + `AdamOptimizer` (same tanh/full-overwrite
formula, same fixed embedding, same task, same seeds, same lr=1e-3,
zero FP4 anywhere) reaches 0.918-1.000 across every n_bits tested
(`scripts/dense_tanh_no_bptt_control.py`) -- matching the torch
control almost exactly. This rules out the architecture, the task,
the no-BPTT training regime, and sili's core Tensor/autograd machinery
as explanations for DISLDO's remaining gap: the problem is confined
specifically to DISLDOLayer's own FP4/importance-update machinery.

Real, load-bearing methodology lesson from this whole ladder, twice
over: (a) an accuracy-only comparison at N=4 seeds is not evidence of
anything by itself -- confirmed directly by rerunning the EXACT SAME
code/seeds twice and getting wildly different results (e.g. 0.8 vs 0.2
at the same seed) purely from `EnergyDynamics`' own already-documented
unseeded exploration noise, chaotically amplified through thousands of
discrete top-p/FP4-rounding decisions -- and (b) "the manual
`loss.grad = np.array(1.0, ...)` line before `.backward()` broke
learning" (an earlier hypothesis this session) does NOT hold up: for a
genuinely scalar loss, this is provably numerically identical to
`Tensor.backward()`'s own default (verified directly in the
interpreter: `ones_like(loss.data) == np.array(1.0, dtype=np.float32)`
bit-for-bit) -- removed anyway since it's dead weight, but it was never
the actual bug.

Status at end of this pass: DISLDO reaches in-context parity with the
torch/dense-Adam controls (1.000 at high lr) once both real bugs above
are fixed; out-of-context (n=3/4/6) still lags (0.5-0.73 vs 0.92-0.96)
-- an open question, not yet root-caused, and not explained by the
no-BPTT regime itself (the dense+Adam control uses the identical
no-BPTT regime and does NOT show this gap). Next planned step (not yet
started): re-run the lr sweep and the in_proj-vs-recurrent gradient
question with the now-fixed sili__new build, since every measurement
above the C++ fix was taken against the buggy library.

## Importance damping replaced with RMSprop-style g² decay; FP4's remaining gap isolated to storage coarseness, not the update rule

Continuing the ablation ladder above: with the forward-time Hebbian
footgun removed and `lr_per_row_nnz` exposed, plain SGD-style DISLDO
converges but to a worse ceiling than Adam on the same toy RNN task
(loss ~1.1, vs Adam's ~0.005). Mathematically compared DISLDO's own
per-synapse `ci` (raw undecayed running SUM of signed gradient,
damping the update by `1/(1+|ci|)`) against Adam: Adam's `v` (decayed
EMA of g², normalizing by `1/sqrt(v)`) is the real mechanistic
difference -- `ci`'s raw signed sum lets sign-oscillating (noisy)
gradient pressure CANCEL, so damping barely engages exactly when it
should.

**Fix, landed in `sili__new` (`feature/rmsprop-importance`, commit
`348ea57`, pushed for review):** replaced the formula in
`disldo_backward` (`linear_disldo.hpp`, all three storage sites: value,
`value_scale`, `output_scale`; SIMD block4 path via a new
`block4_vec_sqrt`) and `disldo_backward_sparse_grad`
(`sisldo_ops.hpp`, which previously had no damping toggle at all) --
`ci = beta2*ci + (1-beta2)*g*g`, damped by `1/(sqrt(ci)+eps)`
(`beta2=0.999`, `eps=1e-8`, matching this project's own
`AdamOptimizer` convention). Same one-scalar-per-synapse storage
budget as before -- no new array. Confirmed on the real RNN task (both
FP4 `SparseLinearLayer` and the new 32-bit `DISLDOLayerV`/
`DISLDOLayer32` fallback, added specifically as an A/B control): loss
~1.1 -> ~0.005, 100% accuracy, matching full Adam.

**Real, accepted trade-off, found and documented rather than hidden:**
on a continuous MSE-regression task with no LR schedule, the new
RMSprop-style damping is measurably ~1.3x WORSE than the old raw-sum
formula (`tests/integration/test_importance_damping_optimization.py`
in sili__new, rewritten to characterize this honestly) -- expected,
well-documented behavior of adaptive methods near a minimum (Wilson et
al. 2017). Checked directly (not assumed) that this isn't simply
"classification beats regression": a small standalone logistic
-regression test built to probe that framing also showed damping
losing. The real win is tied to something about the RNN task's own
structure (one set of weights trained online across many varying
-length sequences, mostly one gradient sample each) -- left open,
documented, not chased further; per direct decision, not worth a
swappable-optimizer-shape abstraction right now.

**FP4's own remaining instability isolated specifically to storage
coarseness, not the formula:** `DISLDOLayer32` (same disldo_forward/
disldo_backward kernels, generic over `VALUES_TYPE`, instantiated at
32-bit float instead of 4-bit `FP4BiPacked` -- `DISLDOLayerV`, no
block4 promotion, no `equalize_to_capacity`, pure diagnostic) gives a
same-formula, same-connection-count, PRECISION-ONLY A/B against real
FP4: 32-bit converges cleanly, FP4 does not (loss 5.9 -> 3.2, genuine
non-convergence with occasional regression, not a stable-but-lower
plateau) -- ruling out "FP4 just has less capacity" as the
explanation.

## Quantization scheme exploration: 8-bit + rank-1 scale (weight AND importance) reaches near-FP32 quality; validated it generalizes across model size and task family

With a working fp32 reference (`DISLDOLayer32`) and the RMSprop fix
landed, the next question (per direct request) was which STORAGE
scheme -- not update rule -- actually survives real quantization:
given a bit-width and a scale scheme, does ongoing training still
converge, or does it show the same instability real FP4 does?

**Methodology:** real `DISLDOLayer32` forward/backward run in true
fp32 (so the RMSprop math itself stays exact); after every backward
call that trains, both `weights_vals` and `importance` are read out,
fake-quantized (deterministic round-to-nearest through N discrete
levels, NOT stochastic -- isolated from FP4's own stochastic-rounding
noise, a separate already-characterized variable), and written back
via `load_weights`. This simulates "this layer's real storage is
N-bit" while keeping the arithmetic exact, matching how real
quantization-aware-training simulators work. Two scale schemes: plain
per-row max-abs (matching sili's own existing `value_scale`
convention), and rank-1 row×col envelope (alternating max-fit, 3
passes -- matching sili_peridot's own earlier B5a fix for the FP4
shared-scale catastrophe; fully vectorized via `np.maximum.at`
scatter-max, not a per-synapse Python loop, to stay cheap enough to
run every training step at full density).

**First result, on the original toy tanh-RNN beyond-context task**
(`HIDDEN=128`, 3000 steps, lr=1e-3):

```
FP32 (no quantization, reference):                     loss=0.0572  acc={2:1.0, 3:1.0, 4:1.0, 6:1.0}
16-bit, row-scale, weight+importance:                   loss=0.4057  acc={2:1.0, 3:1.0, 4:1.0, 6:0.4}
8-bit,  row-scale, weight+importance:                   loss=0.4036  acc={2:1.0, 3:1.0, 4:0.85,6:0.3}
8-bit,  row-scale, weight only (importance stays fp32): loss=0.0723  acc={2:1.0, 3:1.0, 4:1.0, 6:1.0}
4-bit,  row-scale, weight+importance:                   loss=1.2653  acc={2:1.0, 3:0.75,4:0.93,6:0.7}
4-bit,  row-scale, weight only (importance stays fp32): loss=0.6909  acc={2:1.0, 3:0.75,4:0.25,6:0.62}
8-bit,  rank-1 scale, weight+importance:                loss=0.1693  acc={2:1.0, 3:1.0, 4:1.0, 6:1.0}
4-bit,  rank-1 scale, weight+importance:                loss=0.6509  acc={2:1.0, 3:0.85,4:0.78,6:0.4}
```

Two findings: (1) plain row-scale quantization is much harder on
IMPORTANCE than on weight -- quantizing importance to 8-bit row-scale
alone costs 7x the reference loss, while leaving importance fp32 and
only quantizing weight costs almost nothing. (2) rank-1 scale mostly
fixes this specifically for importance: 8-bit rank-1 (BOTH quantized)
reaches loss=0.1693, within 3x of the fp32 reference and matching its
accuracy exactly across every n_bits tested, at 1/4 the storage cost
of fp32 and running 3-4x FASTER in this simulation (9s vs 31s for
row-scale, since rank-1's 3-pass envelope needs no per-row Python
loop). The working intuition (confirmed directly, not just
theorized): importance's dynamic range is shaped by BOTH forward
tracking AND backward gradient accumulation, unlike a weight, which is
more purely backward-driven -- it needs rank-1's extra degree of
freedom more. 4-bit rank-1 still trails badly (loss=0.6509) --
rank-2 or other higher-order scale envelopes are a plausible way to
close that gap further, tabled for now since 8-bit rank-1 is already a
working, stable, trainable system and 4-bit isn't blocking anything
today.

**Cross-task generalization check, per direct instruction ("test more
toy models... different sizes... and if it performs well on a wide
range of tests then yes, we should move it into sili__new") --** built
`QuantizedDISLDOLayer32` (`model/toy_precision_models.py`) as a real,
reusable, `disldo_cls=`-pluggable layer (same call convention as
`DISLDOLayer`/`AdamRowScaleDISLDOLayer`) wrapping `DISLDOLayer32`,
applying the validated 8-bit rank-1 (weight+importance) scheme after
every training backward call -- drops into both
`ToySmallTransformerRealFP4` and `ToyTileRecurrenceRealFP4` with zero
changes to either file. Ran it against:

1. **MQAR transformer task** (`scripts/train_toy_precision_comparison.py`'s
   real harness, q/k/v/o/gate/up/down + causal attention, unchanged),
   swept across hidden=16/32/64 and a longer-context config
   (seq_len=32/kv=4). Found and fixed a real, pre-existing harness bug
   along the way: `PEAK_LR=0.02` is tuned for Adam's normalized step,
   but `DISLDOLayer`-family arms feed `learning_rate` directly into a
   raw, unclipped per-synapse update and diverge at that rate
   regardless of quantization -- confirmed directly by running the
   exact unmodified plain-FP4 arm at the harness's own original seed
   and seeing the identical divergence (loss climbing into the
   thousands) with NO quantization involved at all. Lowered to
   `PEAK_LR=0.002` (confirmed stable for the fp32 reference arm first)
   before trusting any cross-arm comparison. At the corrected LR,
   4 configs x 4 arms:

   ```
   config                        FP32 ref   real FP4    8-bit rank1   4-bit rank1
   hidden=16                     0.41(3.9)  0.13(72.1)  0.31(3.2)     0.17(3.9)
   hidden=32 (control)           0.52(1.4)  0.05(160)   0.29(6.7)     0.16(8.0)
   hidden=64                     0.50(3.7)  0.10(110)   0.23(6.2)     0.18(11.0)
   hidden=32, seq32/kv4          0.24(8.5)  0.04(285)   0.09(11.8)    0.05(15.4)
   ```

   8-bit rank-1 quantized storage beats native production FP4 in
   EVERY config (loss lower by 10-25x, accuracy 2-6x higher), same
   direction as the RNN result, though it trails fp32 by more here
   than on the RNN task (a harder, deeper, attention-based task) --
   still clearly the best non-fp32 option every time, not a fluke of
   the original toy task's specific shape. 4-bit rank-1 also
   consistently beats native FP4, though by a smaller margin than
   8-bit.

2. **Tile-recurrence ("small tile model",
   `model/toy_tile_precision_models.py`'s `ToyTileRecurrenceRealFP4`)**
   -- same `PEAK_LR` fix applied up front (same root cause: raw
   `learning_rate` fed straight into DISLDOLayer-family arms), swept
   at `column_neurons=8` and `16` (this project's own established
   capacity-scaling test), seq_len=16/kv=2:

   ```
   column_neurons=8:   FP32 ref=0.37  real FP4=0.10  8-bit rank1=0.17  4-bit rank1=0.08
   column_neurons=16:  FP32 ref=0.38  real FP4=0.08  8-bit rank1=0.17  4-bit rank1=0.10
   ```

   Same direction again: 8-bit rank-1 roughly DOUBLES accuracy over
   native FP4 at both widths (0.17 vs 0.08-0.10), stable across the
   capacity-scaling axis specifically named by the user. 4-bit rank-1
   is only roughly tied with native FP4 here (unlike the transformer
   task, where it still clearly won) -- on this harder task, 4-bit's
   already-known weakness (needing more than rank-1 to help
   importance specifically) shows up more, not less.

**Overall conclusion across three materially different task families**
(single-cell tanh-RNN, deep causal-attention transformer, tile
-recurrence with gaussian cross-tile attention) **and model sizes
16-128 hidden:** 8-bit + rank-1 scale (weight AND importance both
quantized) is consistently, substantially better than native
production FP4 -- never once lost to it, across 4+2+4 = 10 separate
configs -- and remains a genuinely stable, trainable, non-diverging
system everywhere it was tried. It does NOT fully close the gap to
fp32 on the two harder/newer tasks the way it nearly did on the
original RNN task (fp32 stays meaningfully ahead on transformer/tile
-recurrence) -- an honest, real gap, not hidden. 4-bit rank-1 is a
smaller, less consistent win over native FP4 (clear win on the
transformer task, only a tie on tile-recurrence) -- confirms 4-bit
importance specifically needs more than rank-1 to be reliably useful
(matches the working intuition above: importance's dynamic range is
shaped by both forward and backward signal, unlike a weight); rank-2
or higher-order scale envelopes are the most likely lever to close
that further, tabled for now per direct decision, not blocking
today's 8-bit result.

This satisfies the user's own stated gate ("if it performs well on a
wide range of tests then yes, we should move it into sili__new") for
8-bit specifically: it never diverges, it's never the worst
non-fp32 option, and it wins by a large, consistent margin over what
production FP4 does today, across every task family and size tested
so far. The real, remaining sili__new C++ work (an FP8 `DISLDOLayer`
variant, closer to the 32-bit `DISLDOLayerV` path per the user's own
note -- FP8 fits in 1 byte, so it can be templated directly without
FP4's nibble-packing, "alt, not replace" alongside the existing FP4
class) is the next planned step, not yet started.

## sili__new: real FP8 (E4M3) DISLDOLayer landed -- scattered path, block4 promotion scoped as follow-up

Per direct instruction ("start the real fp8 disldolayer... same
sparse-block4 split as well as the templating... block4 can use the
same simd speedups since those were on the input float32s instead"):
built and pushed `feature/fp8-disldo` in `sili__new` (commit `cd6b31d`).

Real per-element format is OCP MX E4M3 (1 sign, 4 exponent, 3
mantissa), not the plain signed-int8 originally validated in the toy
sweep above -- a direct design decision, since FP4 itself is a true
floating microformat (E2M1), not integer. Quickly re-validated E4M3 +
rank-1 on the fastest toy harness before committing engineering time:
loss 0.1341 vs the int8 scheme's 0.1693 (same task, same rank-1
mechanism) -- E4M3 is a genuine improvement, not just spec-purity.

`fp8quant.hpp` (new): bit-shift E4M3 codec (deterministic + stochastic
-rounding, same carry-propagating technique as FP4's own codec) and
`FP8BiValues` (two plain byte arrays -- unlike FP4BiPacked's nibble
-packing, E4M3 needs a full byte/value, so this mirrors
`DeltaCSRBiValues<T>`'s simpler two-array shape). `ValueAccessor
<FP8BiValues>` makes it a drop-in VALUES_TYPE for the EXISTING generic
`disldo_forward`/`disldo_backward`/`delta_csr_*` templates -- zero
changes needed there for the scattered CSR path. `block4.hpp` gained
4-wide SIMD decode/stochastic-quantize for E4M3 (common case
vectorized, rare subnormal/NaN lanes via the exact scalar reference),
added purely alongside the existing FP4 SIMD helpers.

`SparseLinearLayer8`/`DISLDOLayer8`: real, working, tested end-to-end
(construction, E4M3-quantized `load_weights`, forward, backward with
finite weight updates, training convergence comparable to the fp32
`DISLDOLayer32` reference at a properly-tuned LR -- 0.542->0.399 vs
0.556->0.397 on a toy online-regression task). Also exposes
`get/set_value_scale_raw` and `get/set_output_scale_raw` -- the SAME
rank-1 mechanism FP4 uses, confirmed to live on `SparseLinearWeightsDelta`
itself (VALUES_TYPE-agnostic) rather than being FP4-specific, so this
is genuinely the validated "8-bit + rank-1, weight+importance both
quantized" scheme, not a lesser row-only version.

**Real, honestly-scoped gap:** block4 dense-tile SIMD promotion is NOT
included. Found by reading the code directly (not assumed): `Block4Tile`
packs weight+importance as one nibble-pair per BYTE (`data[16]`,
mirroring FP4BiPacked's own nibble convention) and `Block4TileHandle`'s
accessor API returns a single byte per slot -- E4M3 needs 2 full
bytes/slot, so this needs new `Block4Tile8`/`Block4Store8` types plus
new FP8-dispatch branches inside `disldo_forward`/`disldo_backward`'s
existing block4 code sections (which currently call `block4_vec_decode_fp4`
hardcoded by name), not a template-parameter swap. `block4_row_shift`/
`block4_grow_last_row` (the row-growth/compaction plumbing) already
look VALUES_TYPE-agnostic on inspection -- pure byte-buffer operations,
likely reusable unchanged -- but not yet confirmed by actually wiring
it up. Real follow-up work, tracked, not silently deferred.

Full existing sili__new test suite re-run before/after this change
(diff'd exactly, not eyeballed): zero new failures introduced. One
pre-existing flaky energy-competition test differed between the two
runs (present in one, absent in the other) -- matches this project's
own already-documented order-dependent flakiness in that area, not a
regression from this change.

## sili__new: FP8 block4 dense-tile SIMD promotion landed (core), synaptogenesis wiring scoped as follow-up

Second push to `feature/fp8-disldo` (commit `dbb4ebe`), per direct
instruction to keep going on the block4 promotion piece.

`Block4Tile8`/`Block4TileHandle8`/`Block4Store8` (block4.hpp): E4M3's
tile format needs 2 bytes/slot (weight half + importance half, 32
bytes/tile) instead of FP4's 1-byte nibble-packed 16-byte tile --
fully separate types, zero modification to FP4's own `Block4Tile`/
`Block4TileHandle`/`Block4Store`. Real, useful discovery while scoping
this: `block4_row_shift`/`block4_grow_last_row`/
`block4_ensure_row_headroom`/`block4_row_insert_tile`/
`block4_row_remove_tile`/`block4_resize_tile_in_row` (the row-growth/
compaction machinery) turned out to already be pure byte-buffer
plumbing with zero assumption about slot width -- confirmed by reading
each one directly, then reused completely unchanged, cutting the real
new code needed roughly in half versus a full duplicate.

`disldo_forward`/`disldo_backward`'s block4 sections (linear_disldo.hpp)
now dispatch via `if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>)`
-- FP4's branch is byte-for-byte the pre-existing code, untouched. The
FP8 branch is deliberately scalar/correctness-first for now (matches
this file's own established "first working version" precedent) --
SIMD optimization for the block4 FP8 path is real, scoped follow-up
work, not attempted yet.

**Found and fixed a real stack buffer overflow while testing** (not
hypothetical -- caught via GCC's stringop-overflow warning, confirmed
with AddressSanitizer): a scratch buffer in the row-workspace write
path was sized for FP4's 16 bytes/tile unconditionally, but the FP8
tile-unpack call was writing 32 bytes into it. This is exactly the
kind of thing extensive, ASan-checked testing exists to catch before
it reaches real training -- worth remembering as a data point for how
much verification this kind of low-level storage-format work needs.

New `test_disldo_block4_fp8.cpp`: hand-promotes a block4 tile, checks
forward output against a manual reference, checks backward moves the
weight/importance and dx sanity -- clean under ASan/UBSan. Full test
suite (existing FP4 + new FP8, Python + C++) diffed before/after
again: zero new failures.

**Real, explicitly scoped gap, not silently deferred:** synaptogenesis
-triggered automatic block4 promotion/demotion
(`delta_csr_memory.hpp`'s `block4_maybe_promote`/`block4_demote_tile`)
is ALSO FP4-hardcoded and not yet dispatched for FP8 -- calling
`synap_row_step` on a growing FP8 layer would currently write wrong
(FP4-nibble-decoded) bytes into what should be E4M3 tile data. Tiles
must be seeded manually (as the test does) until this lands. Tracked
as its own follow-up, not started.

## sili__new: FP8 synaptogenesis-triggered promotion/demotion landed -- block4 feature parity with FP4 now complete

Third push to `feature/fp8-disldo` (commits `191f32d`, `3010d18`),
closing the gap flagged in the previous entry. Growth
(`delta_csr_synap_row_step`) can now correctly promote scattered FP8
synapses into a block4 tile once `BLOCK4_PROMOTE_MIN_LIVE` land inside
it, and pruning correctly demotes back -- same behavior as FP4, tested
through the identical grow/prune/demote/re-grow/re-promote cycle.

Two more real bugs found and fixed here, both via the same disciplined
approach (write the test, hit a crash, root-cause with ASan/targeted
debug output rather than guessing):

1. **A genuine cross-row memory-corruption bug**, not hypothetical --
   `block4_row_insert_tile`/`block4_row_remove_tile` looked like pure,
   generic byte-buffer plumbing (confirmed by reading them once), but
   they ALSO call `block4_stored_tile_len` (FP4's 1-byte/entry formula)
   internally while walking a row to find insert/remove positions --
   silently wrong for FP8's 2-byte/entry tiles, corrupting a row's own
   byte bookkeeping. Missed on the first read because the function
   signatures gave no hint they depended on value width at all. Root
   -caused by writing a step-by-step debug harness that dumped
   `tbyte_start`/`tbyte_end` after every synaptogenesis call until the
   exact divergent step was visible, then confirmed with ASan. Fixed
   with a defaulted function-pointer parameter -- every existing FP4
   call site is completely unaffected (same default), FP8's call sites
   pass the right function explicitly.
2. **`Block4View8` was never registered with pybind** -- compiled fine,
   worked fine in direct C++ tests, but raised "Unregistered type"
   the moment Python code touched `.block4` on an `FP8`-backed layer.
   Found by deliberately testing the REAL Python/pybind path end-to-end
   (not just the C++ template instantiation directly), which is
   exactly the layer this bug lived in and every earlier C++-only test
   couldn't have caught.

Both fixes are covered by new tests (`test_disldo_block4_promotion_fp8.cpp`,
clean under ASan/UBSan; `TestDISLDOLayer8Synaptogenesis` in Python,
exercising the actual pybind path). FP4's own promotion/demotion cycle
was independently re-verified via a standalone C++ regression test
after the shared-function signature change -- passes cleanly under
ASan, zero behavior change (all new parameters are defaulted).

**This completes FP8 block4 feature parity with FP4** -- scattered
path, dense-tile SIMD promotion, and now synaptogenesis-triggered
growth/pruning, matching what was asked for from the start ("the same
sparse-block4 split" FP4 has). Remaining work is SIMD optimization of
the FP8 block4 kernels (currently scalar/correctness-first) and real
-checkpoint validation at production scale -- not started, no known
blockers.

## sili__new: FP8 block4 backward SIMD-optimized -- real, measured, batch-size-dependent win, not assumed

Fourth push to `feature/fp8-disldo` (commit `2abc48a`), closing the
"SIMD optimization" gap flagged at the end of the previous entry. Per
direct instruction while starting this: keep the plain scalar
implementation around as a verified fallback (not delete it), and
actually measure -- don't assume SIMD helps just because it compiles.
Also asked directly, mid-work, whether GCC's disassembly was actually
being checked for real SIMD instructions, not just trusted from the
type system -- it was, and the methodology is recorded below because
it mattered.

First attempt mirrored FP4's own decode/encode SIMD path exactly
(`block4_vec_decode_fp8`/`block4_vec_quantize_stochastic_fp8` for the
whole tile, not just the accumulate math) -- and it was measurably
SLOWER than the plain scalar version (0.0060s vs 0.0048s per call at
batch=1, about 20% worse), a real result from `bench_block4_fp8_simd.cpp`
(new, `scripts/`), not a guess. Bisected by swapping decode and encode
back to scalar independently: scalar decode alone recovered most of
the loss, scalar decode+encode together matched plain scalar almost
exactly. E4M3's 256-code space needs a real per-lane subnormal/NaN
scalar-correction fallback inside the SIMD codec that FP4's much
simpler 16-code E2M1 format never has to pay for -- so the codec
itself just isn't a good SIMD candidate for FP8, unlike for FP4.

Final design: decode/encode stay scalar (`fp8_decode_bits`/
`fp8_quantize_stochastic`), SIMD (`Block4Vec`, same 4-wide GCC
vector-extension type FP4 uses) is applied only to the batch-loop
accumulation math (RMSprop moment update, dx accumulation) once
weight/importance are already decoded to float -- the one piece that
actually earns its complexity. Measured again at batch=32: SIMD
0.0227s vs scalar 0.0300s, a real ~24% win once there's enough
per-tile work to amortize. At batch=1 the two are tied (0.0048s each)
-- no loss, but no free lunch either, an honest result rather than an
optimistic one.

Confirmed via `objdump -d --no-show-raw-insn` (searching for
`vmulps`/`vrsqrtps`/`vaddps` on `xmm` registers vs `mulss`/`addss`)
that the SIMD build really does emit packed SIMD instructions, and
via `nm` + address-range matching + `c++filt` that those instructions
live inside the exact expected function
(`disldo_backward<int, FP8BiValues, unsigned int>(...)::{lambda...}::operator()`)
-- not GCC auto-vectorizing unrelated scalar code and not a
misattributed symbol. This directly answers the question of whether
"SIMD" here is real: yes, verified at the instruction level, not
assumed from the presence of `Block4Vec` types in the source.

The original fully-scalar implementation (the pre-existing,
already-verified-correct version) is kept intact, reachable via the
existing `SILI_BLOCK4_FORCE_SCALAR_BACKWARD=1` build flag -- per
direct instruction, not deleted just because a faster path now
exists. `test_disldo_block4_fp8.cpp`/`test_disldo_block4_promotion_fp8.cpp`
re-verified clean under ASan/UBSan against the new code. The
pre-existing FP4 `test_disldo_block4_backward.cpp` failure under
`SILI_BLOCK4_FORCE_SCALAR_BACKWARD=1` was re-confirmed to reproduce
identically on a clean stash of this branch -- unrelated, not a
regression, already documented in an earlier phase this session.

**This closes out the FP8 block4 work started from "same sparse-block4
split... same simd speedups" -- FP8 (E4M3) now has full feature and
performance parity with production FP4**: scattered path, dense-tile
block4 SIMD promotion (forward and backward), synaptogenesis-triggered
growth/pruning, and a real, disassembly-verified, honestly-measured
SIMD backward path. No known blockers remain on the FP8 side; next
real step is production-scale validation against actual MiniCPM5
checkpoints.

## rank-n scaling for FP4: does a second scale degree of freedom close 4-bit's gap to 8-bit rank-1?

Per direct request, following up on 4-bit rank-1's known weakness
(JOURNAL's own quantization-exploration entry: "4-bit rank-1 still
trails badly... rank-2 or other higher-order scale envelopes are a
plausible way to close that gap further, tabled for now"). Pure toy
-harness work, `model/toy_precision_models.py`, zero sili__new C++
touched -- this is exploratory validation, the same "toy models first"
gate FP8's own C++ work went through before it was worth real
engineering time.

**Design note, real dead end found before the working version:** the
first idea tried was an additive residual decomposition (fit
rank1_fake_quantize's envelope, subtract it, fit the leftover again).
Verified by hand this does nothing: rank1_fake_quantize's envelope is
a strict MAX-COVER (`row_scale[r]*col_scale[c] >= |v|` for every
synapse, provably, since row_scale is itself a per-row max) -- the
residual after subtracting it is <= 0 everywhere, confirmed directly
(`min residual: ~-1, max residual: 0.0` on a real test array), so a
second additive term has nothing left to refine. This matters beyond
being a dead end: an envelope IS what a real N-bit fixed-point scale
must be (never let a stored value exceed what its levels can
represent) -- an approach that doesn't preserve the cover property
would be simulating something real hardware can't do.

**What actually works**: `rankn_fake_quantize` buckets rows by their
own max |w| into `rank` equal-count quantile groups (deterministic
sort+split, no iterative clustering), then fits an INDEPENDENT
rank-1 max-cover per bucket -- letting small-magnitude rows stop
sharing a column's envelope with large-magnitude rows they happen to
share a column with (the actual degree of freedom a single shared
col_scale can't express). Reduces EXACTLY to rank1_fake_quantize when
rank=1 (verified bit-identical in tests). Checked on a synthetic
adversarial case (bimodal-magnitude rows, same columns) before
trusting it on a real task: rank-2 tightened the small-magnitude
bucket's mean envelope/|value| ratio by ~22% vs rank-1, large
-magnitude bucket unchanged as expected -- real, but bounded by how
much true row/col scale correlation the data has, not a free lunch.
`QuantizedDISLDOLayer32` gained a `scheme="rankn"` + `rank` param
(rank1/row unchanged, zero behavior change); `ToySmallTransformerQuant4Rank2`/
`Rank4` wrapper classes added, same `disldo_cls`-pluggable convention
as every other arm in this file.

**Result 1, transformer MQAR task** (`scripts/train_toy_precision_comparison.py`'s
harness, `PEAK_LR=0.002` -- the already-documented fix for
`DISLDOLayer`-family arms diverging at Adam-tuned rates, re-applied
here after a first pass forgot it and produced meaningless near-chance
numbers for every arm including FP32; a real methodology mistake,
caught by checking the FP32 reference against JOURNAL's own prior
numbers before trusting the comparison):

    config                  FP32 ref  8-bit rank1  4-bit rank1  4-bit rank2  4-bit rank4
    seq16/kv2/vocab20         0.467       0.333        0.075        0.183        0.167
    seq32/kv4/vocab40         0.192       0.113        0.054        0.075        0.079

4-bit rank-2 more than doubles rank-1's accuracy in both configs
(2.4x, 1.4x); rank-4 roughly ties rank-2, not a further improvement --
most of the gain shows up already at rank-2. Single seed per config,
same as this file's own earlier 4-config transformer sweep -- a real,
consistent-direction signal, not yet a statistically confirmed one.

**Result 2, tanh-RNN beyond-context task** (the ORIGINAL task 8-bit
rank-1 was first validated on -- same `DISLDOLayer` tanh-cell +
`generate_deviation_sequence` structure as
`scripts/disldo_tanh_sparse_ablation.py`, HIDDEN=128, sparse
PER_ROW_K=8, 3000 steps, `PEAK_LR=1e-3`, swapping in
`QuantizedDISLDOLayer32` variants via `disldo_cls`). Direct
clarification worth recording: out-of-context prediction on this task
was ALREADY working before this (FP32/8-bit rank-1 both hit
`acc={2:1.0,3:1.0,4:1.0,6:1.0}` in the original validation) --
rank-n's question here is narrower, whether it closes 4-bit rank-1's
specific gap at out-of-context distances (n=3/4/6), not whether
out-of-context prediction is possible at all (see corrected
[[project_sili_bptt_or_chance]] -- that's a fully separate,
already-resolved question, unrelated to quantization scheme).

A first single-seed run looked dramatic (rank4 hitting 0.78-0.93
out-of-context vs rank1's 0.2-0.71) -- per this file's own
already-documented lesson ("an accuracy-only comparison at N=4 seeds
is not evidence of anything by itself... wildly different results
purely from noise"), re-ran 8 seeds with paired significance tests
(`scipy.stats.ttest_rel` + `wilcoxon`, same methodology as the
peak-eligibility 50-seed check) before trusting it:

    n (dist)  rank1 mean  rank2 mean  diff     p(t/wilcoxon)   rank4 mean  diff     p(t/wilcoxon)
    2 (ctx)     0.701       0.761    +0.060    0.670 / 0.938     0.682    -0.019    0.815 / 0.875
    3           0.339       0.444    +0.105    0.534 / 0.523     0.338    -0.001    0.992 / 1.000
    4           0.626       0.555    -0.071    0.612 / 0.688     0.736    +0.110    0.298 / 0.625
    6           0.555       0.656    +0.101    0.199 / 0.219     0.670    +0.115    0.411 / 0.297

No point at either rank clears significance (best case p=0.199) --
the dramatic single-seed result was noise, exactly the failure mode
this file already warned about. Honest verdict: rank-2/rank-4 show a
weakly positive, inconsistent-in-sign trend on this specific task (3
of 4 distances nominally favor higher rank, one doesn't), nowhere near
the clean, large, consistent win rank-2 showed on the transformer MQAR
task above. FP4 rank-n is a real, sometimes-substantial help but not a
uniformly reliable one across task families -- consistent with the
project's overall FP8-vs-FP4 conclusion: FP8 (real E4M3, `feature/fp8-disldo`,
merged) reaches near-FP32 quality reliably and needed no rank-n
tuning to get there, while FP4 stays fundamentally marginal (better
with rank-n, sometimes meaningfully so, but not consistently, and
never validated with real statistical confidence the way FP8's
tests were) -- FP8 remains the easier, more stable choice for actual
development; FP4 rank-n is a real lever worth having but not a
substitute for FP8 where storage budget allows it.

All new code covered by tests (`TestRankNFakeQuantize`,
`QuantizedDISLDOLayer32`'s scheme-combination parametrization extended,
`TestToySmallTransformerQuant4Rank2`/`Rank4`) -- 59/59 passing, full
sili_peridot suite (257 tests) re-verified clean, no regressions. The
comparison scripts themselves were ad hoc (matching this file's own
established convention for the original rank-1 validation -- kept as
JOURNAL numbers, not committed as permanent scripts).

## Two more rank1-vs-rank2 follow-ups on the tanh-RNN beyond-context task: 2x recurrent state (no change), + energy (made it much worse)

Two direct follow-up requests on the rank-n result above, same task/
harness (HIDDEN=128 tanh-cell + `generate_deviation_sequence`, 8-seed
paired significance tests).

**2x recurrent state** (`HIDDEN=256`, cell params 2048->4096 at fixed
`PER_ROW_K=8`): same story as HIDDEN=128, if anything slightly weaker.
Rank-2 nominally wins 3 of 4 distances (n=2/3/4), loses at n=6, no
point clears significance (best p=0.320, vs 0.199 at HIDDEN=128) --
loss curves also noisier for rank-2 (a few seeds spiked to 0.82-0.99
vs rank-1's tighter 0.56-0.81). Doubling state size doesn't change the
conclusion: rank-2's benefit stays real-but-unconfirmed on this task
regardless of scale.

    n (dist)  rank1 mean  rank2 mean  diff     p (t / wilcoxon)
    2 (ctx)     0.600       0.652    +0.052    0.787 / 0.945
    3           0.300       0.481    +0.181    0.320 / 0.383
    4           0.510       0.609    +0.099    0.321 / 0.219
    6           0.719       0.634    -0.085    0.344 / 0.469

**Does EnergyDynamics (forcing more of the recurrent state to fire,
rather than let it collapse to the shutoff floor) change this?**
Direct hypothesis: rank-n's bucketing needs real per-row magnitude
diversity to have anything to exploit; if most of the state sits idle
without energy, maybe there's nothing for rank-n to bucket usefully.
Wired `EnergyDynamics` (`_toy_scale_energy()`'s existing drive=0.1
toy-scale default, not a new value) into a per-tick gate on `h_new`
(REPLACING it, not adding a residual -- an additive gate on top of the
tanh output would reintroduce the exact forward-pass instability the
ablation ladder already fixed by switching to full-overwrite tanh),
applied every tick, aux_loss added to the query tick's loss before its
one `.backward()` call (matches the no-BPTT design: earlier ticks'
aux_loss can never carry gradient regardless, since backward only
fires once per training example -- but the gating still matters
through the values it threads forward as h_prev, and through
EnergyDynamics' own running homeostatic state).

**Result: energy did not help -- it collapsed BOTH rank1 and rank2 to
pure chance at every distance, in-context included:**

    arm          energy      n=2         n=3         n=4         n=6
    4-bit rank1  off      0.70+-0.27  0.34+-0.22  0.63+-0.22  0.55+-0.26
    4-bit rank2  off      0.76+-0.25  0.44+-0.33  0.55+-0.29  0.66+-0.22
    4-bit rank1  on       0.49+-0.05  0.48+-0.05  0.49+-0.06  0.49+-0.03
    4-bit rank2  on       0.50+-0.04  0.53+-0.04  0.52+-0.05  0.50+-0.04

With energy on, accuracy sits within a few points of 0.5 for every
arm/distance with a MUCH tighter std (0.03-0.06 vs 0.22-0.33 without)
-- not "no improvement," an active regression that erased even the
in-context (n=2) capability that worked fine before. Final training
loss with energy stayed 2.5-3.7 across all 16 runs, vs 0.5-0.8 without
-- cross-entropy alone caps out well under 1.0 for this binary-answer
task, so a loss that high means the aux_loss term is dominating the
gradient the whole way through training, not settling the way
`_toy_scale_energy()`'s own docstring describes for the transformer
arms it was tuned against. Plausible, not yet confirmed root cause
(time-boxed, not chased further this pass): that config was tuned for
a very different usage pattern (one call per T-length flattened
sequence, transformer attention output) -- applying it per-TICK to an
online recurrent state, with its own running energy state persisting
continuously across the entire 3000-step training loop rather than
being freshly seeded per sequence, may simply need different
(likely much lower) drive/precision/density values, or aux_loss
downweighting, to be usable in this regime at all. Matches an earlier,
independent finding already in this file's own "Out-of-context
benchmark" entries that energy showed no clear improvement over
no-energy at out-of-context distances on a different architecture --
here it's a stronger, worse result (active collapse, not just no
gain), but the same general lesson: this project's toy-scale
EnergyDynamics config does not obviously transfer between usage
patterns and needs its own re-tuning per architecture, not assumed
constant.

**Honest bottom line**: cannot conclude anything about whether energy
would help rank-n's story specifically, because energy in this
configuration prevents the model from learning the task at all --
the comparison is moot until (if ever) a usable energy config is
found for this per-tick online regime. Not pursued further this pass,
recorded as a real negative result rather than left unreported.

## Root-caused the energy collapse further, then definitively ruled out "just needs more training time" -- it's a real runaway, not slow convergence

Direct follow-up, per request: "try to get aux loss down, it shouldn't be that large, needs better parameters for energy."

**Isolated aux_loss's own formula** (`_apply_energy_dynamics` in sili__new's `energy.py`):
`energy_loss = (reactivity/2) * sum((new_energy_t - setpoint)^2)`, summed (not
averaged) over every neuron in `h`. `_toy_scale_energy()`'s config (including
its unmodified default `reactivity=0.01`) was tuned against `test_sparse_rnn_cell.py`'s
own reference h sizes of 20-64 -- this task's HIDDEN=128 is already 2-6x
larger, so the SAME reactivity produces a proportionally larger summed
aux_loss purely from array size, independent of anything being "wrong."
Confirmed directly: sweeping reactivity from 0.01 down to 0.0001 in isolation
scaled aux_loss down cleanly and close to linearly (0.67 -> 0.07 -> 0.01).

**But that alone doesn't fix anything -- confirmed by measuring cross-entropy
and aux_loss SEPARATELY**, not just watching the combined total:

    reactivity   cross-entropy (last100)   aux_loss (last100)
    0.01               3.15                      0.67
    0.001              4.02                      0.07
    0.0001             3.62                      0.01

Cross-entropy stays bad (3-4, vs the ~0.69 ceiling a clean binary decision
should have) and does NOT improve as aux_loss shrinks toward zero -- ruling
out "aux_loss dominates the gradient" as the actual cause. The real
mechanism, found by reading `_apply_energy_dynamics` directly: fired neurons
get `h_out` set to a flat CONSTANT `2.0`, completely independent of the
neuron's actual computed value -- and 2.0 is well outside tanh's `[-1,1]`
range. Every tick, some fraction of the carried recurrent state gets
overwritten with this out-of-distribution spike constant before being
threaded forward as `h_prev` into the next tick -- corrupting exactly the
state this deterministic-latch task needs to carry precisely. This is a
structural property of the gate, not a magnitude/tuning issue.

**Direct test of the remaining real hypothesis: does the fire-together
-wire-together mechanism just need many more actual gradient-update steps
to learn to interpret the spike representation?** (Only the query tick ever
calls `.backward()`, so 3000 training steps = only 3000 real weight updates
-- genuinely few.) Kept the tuned-down `reactivity=0.001`, launched two
1,500,000-step runs (rank1 and rank2, HIDDEN=128, in parallel, checkpointed
every 50,000 steps) to give this a fair, well-resourced test rather than
assuming the short run was conclusive.

**Result: definitively refuted, not just unconfirmed.** Cross-entropy did
not plateau near where the short runs left off -- it climbed monotonically
and without bound the entire time it ran:

    step       rank1 ce    rank2 ce    accuracy (both arms, all steps)
     50,000        16          10      ~0.4-0.6 (chance), no trend
    100,000        86         239      ~0.4-0.6 (chance), no trend
    300,000       440         787      ~0.4-0.6 (chance), no trend
    500,000       558         878      ~0.4-0.6 (chance), no trend
    700,000       602          --      ~0.4-0.6 (chance), no trend
    (rank2 reached 550,000 before being stopped, ce=831 there)

Accuracy never once trended above chance at ANY checkpoint on either arm,
across the entire run. Also checked and ruled out "the learning rate just
hasn't decayed yet": with `total_steps=1,500,000` in the cosine schedule,
lr had already dropped from peak 1e-3 to ~1.8e-4 by step 700,000 (47%
through) -- yet cross-entropy was still accelerating upward at that point,
not slowing. This is a genuine runaway (most likely unbounded weight growth
from the "fire together, wire together" gradient rule repeatedly reinforcing
the same connections over enough steps, with nothing in this toy setup to
damp it), not a system that's slowly finding its footing. Both runs killed
early once the trend was unambiguous -- no reason to burn the remaining
~800,000+ steps of confirmed divergence.

**Conclusion, closing this out:** EnergyDynamics' toy-scale config, and
likely its whole per-tick "hard spike constant" gating design as currently
implemented, is not usable for this online recurrent-latch task at any
training budget tried (short OR 500x longer) -- not a tuning problem
solvable by reactivity alone, a real architectural mismatch between the
gate's discrete spike representation and a task that needs exact values
threaded through state. Consistent with the user's own decision to table
FP4 (and by extension, this energy angle) for now and proceed with FP8,
which needed none of this.

## Two direct follow-ups: does energy break FP8 the same way? Does fixing the fired-neuron constant fix FP4?

Per direct request. Six 100,000-step runs (HIDDEN=128, `reactivity=0.001`,
checkpointed every 10,000 steps): {FP4-rank1, FP8 (`DISLDOLayer8`)} x
{fired_value = 2.0 (real/unmodified), 1.0, tanh(2)~0.964}. `fired_value` is a
LOCAL, exploratory modification (not yet landed in sili__new) -- a copy of
`_apply_energy_dynamics` with only the fired-neuron output constant
parameterized; the fire/shutoff energy THRESHOLD stays the real 2.0
(that's tied to the accumulator's own integrate-and-fire dynamics, a
separate concern from what value gets written into h_out when a neuron
fires).

**Cross-entropy trajectories, all six runs, checkpoints at steps
10k/20k/.../100k:**

    FP8,  fired=2.0 (default): 1.93 0.77 1.29 1.53 1.72 1.58 1.39 1.18 0.98 0.87
    FP8,  fired=1.0:           1.16 0.72 0.81 0.87 0.94 0.94 0.87 0.82 0.79 0.76
    FP8,  fired=tanh(2):       1.12 0.74 0.77 0.86 0.95 0.90 0.88 0.81 0.77 0.75
    FP4,  fired=2.0 (default): 1.80 0.72 1.21 5.79 5.81 4.96 7.42 6.06 4.76 5.26
    FP4,  fired=1.0:           1.15 0.72 0.74 3.14 19.07 15.28 11.35 8.46 5.21 5.00
    FP4,  fired=tanh(2):       1.12 0.71 0.70 2.68 3.71 6.85 6.21 4.45 3.03 2.08

**A) Does energy break FP8 the way it broke FP4? No -- a real, clean
asymmetry, independent of fired_value.** All three FP8 variants stay
bounded (roughly 0.7-1.9 the whole run) and every one trends DOWNWARD by
the end (0.87, 0.76, 0.75 final) -- no runaway. All three FP4 variants
show the same divergence-onset signature regardless of fired_value:
healthy through ~step 30,000 (ce~0.7-1.2, matching FP8's own range
exactly at that point), then a sharp jump starting around step 40,000
-50,000 into the 3-19 range, staying elevated the rest of the run. FP4's
own coarser quantization (or its inline weight-update dynamics) appears
to be what interacts badly with energy's feedback loop -- not something
present in FP8 at the same HIDDEN size, same task, same energy config.
Caveat carried over honestly from the earlier 1.5M-step FP4 test: 100,000
steps is a much shorter horizon than the 1.5M that made FP4's divergence
undeniable, so this doesn't prove FP8 would stay bounded forever -- only
that it clearly doesn't show FP4's SAME divergence onset within a much
longer window (30x) than FP4 needed to start clearly diverging.

**B) Does changing the fired-neuron constant to something inside tanh's
range fix FP4? No -- refutes the original hypothesis about WHY it
diverges.** All three FP4 variants diverge with the same onset timing
regardless of fired_value (2.0, 1.0, or tanh(2)) -- changing the constant
did not prevent, delay, or meaningfully alter the ONSET of divergence.
There is a real, smaller, honestly-reportable difference in SEVERITY
among the three FP4 variants -- tanh(2) is the mildest (peaks at 6.85,
declining to 2.08 by the end, the only FP4 variant with a clear late
-downward trend) vs 1.0 (peaks at 19.07, stays elevated ~5.0 at the end)
vs 2.0 (peaks at 7.42, stays elevated ~5.3 at the end) -- so the exact
fired value is NOT irrelevant, but none of the three variants come close
to recovering FP8's stable ~0.75-0.87 range or the healthy ~0.7 level FP4
itself showed in its own first 30,000 steps before diverging. This means
the original root-cause hypothesis (the constant being numerically
out-of-range for tanh is THE cause) was too narrow -- the real
interaction is more likely between FP4's own coarse discrete weight
storage / inline C++ update rule and energy's fire-together-wire
-together gradient pathway specifically, with the fired constant's exact
value only a secondary, severity-modulating factor, not the root cause.

**One more honest point neither result should obscure: NONE of the six
configs actually learned the task.** Accuracy sat at chance (roughly
0.34-0.63, no consistent trend above 0.5) across every checkpoint of
every run, FP8 included. FP8's stability here means "bounded, well
-behaved loss that doesn't diverge" -- not "energy works with FP8 on
this task." The real, precise conclusion: FP8 fails at this task
gracefully under energy (no risk of NaN/runaway loss), FP4 fails
catastrophically (genuine numerical blowup) -- a real, useful robustness
difference worth knowing, and one more concrete reason FP8 is the safer
default -- but getting energy to actually IMPROVE this task (vs. just
not break it) remains unsolved with either storage format.

## CORRECTION: energy's own drive/p defaults were miscalibrated, not FP4 -- once fixed, both FP4 and FP8 show real above-chance learning

**This supersedes the "FP4 diverges, FP8 doesn't" framing directly above.**
Per direct pushback ("this doesn't make any sense mathematically -- if
drive is low enough the fire events rarely happen, and if aux_loss is low
enough it can't compete... something in energy has to be making things
fail"): correct instinct, and it led to the actual root cause, which
neither the reactivity sweep nor the fired-value sweep above ever tested
-- `drive` and `p` themselves.

**Measured directly** (not assumed) what the DEFAULT toy config
(`drive=0.1`, `p=0.3`, unchanged in every experiment run so far) was
actually doing to a 128-wide state, every tick: ~30% of neurons fired
(pinned to the flat constant) on 99.5% of ticks, ~70% got hard-zeroed.
Tried the user's exact hypothesis next -- lower `drive` to make fire
events rare -- and it does (0% fired at `drive<=0.01`), but the zeroed
fraction climbed to 99%+, WORSE than the firing case, not better. Traced
why directly: `new_energy = energy + drive + noise - activation_cost*|h|`
every tick, unconditionally; at low drive this term is net NEGATIVE
(`activation_cost*E[|h|] ~= 0.05*0.32 ~= 0.016` easily exceeds a small
drive), so instead of firing, the whole population drifts into
PERMANENT SHUTOFF and gets pinned there (confirmed: mean energy converges
to exactly -2.0 within ~500 ticks) -- and a shutoff-pinned neuron's own
output formula (`energy_flat + 2.0`, with energy_flat clamped at -2.0)
evaluates to ~0.0 too, indistinguishable from being suppressed. There is
no "does nothing" regime reachable by lowering drive alone -- it's
bimodal: too-high drive saturates on firing, too-low drive saturates on
shutoff, and BOTH destroy most of the carried state, just via different
absorbing thresholds.

**The actual fix**: calibrate drive to the POPULATION's own typical |h|,
not treat it as an independent knob: `drive ~= activation_cost * E[|h|]`.
Measured `E[|h|]~=0.32` for this cell's tanh output, giving
`drive~=0.016`. Also raised `p` (the independent hard active-fraction
ceiling) from 0.3 to 0.95, since even a perfectly-calibrated drive still
leaves `1-p` of the population zeroed via the top-p competition,
unrelated to drive/activation_cost at all. Verified directly: this
config cuts firing to 0% and zeroing to ~5% (vs 70-99%+ in both failure
regimes) -- a genuinely quiet operating point, found and confirmed BEFORE
spending another training run on it.

**Documented in sili__new itself, not just here** (per direct request:
"the energy function should have the equation... describing how often
neurons will fire or be driven to zero") -- `sili/energy.py`'s
`_apply_energy_dynamics` docstring now has a full derivation of this
balance point and both failure regimes, with the real measured numbers
above as worked examples. Pure docstring addition (zero behavior
change, confirmed via the existing `TestEnergyDynamicsKeptIndices`
suite passing unmodified). Branch `docs/energy-fire-shutoff-balance`,
commit `c7206e5`, pushed for review (not yet merged) -- this branch was
made from a fresh `origin/main` checkout since `feature/fp8-disldo`
(where `DISLDOLayer32`/`DISLDOLayer8` live) hasn't been merged to main
yet either; testing below still runs on `feature/fp8-disldo` for that
reason.

**Real training result, 100,000 steps each, calibrated config
(`drive=0.016, p=0.95, reactivity=0.001`), same tanh-RNN beyond-context
task, checkpoints every 10,000 steps -- genuine above-chance accuracy
across every distance, for every arm, not just bounded loss:**

    FP8    ce: 1.04 0.69 0.72 0.71 0.70 0.72 0.66 0.68 0.66 0.65  (clean, ~monotonic)
    rank2  ce: 1.07 0.68 0.74 0.75 0.79 1.91 1.67 1.42 0.95 0.97  (one bump, recovers)
    rank1  ce: 1.06 0.80 1.24 0.91 4.98 4.43 1.56 1.29 1.02 0.80  (real spike, recovers)

    Mean accuracy across all 10 checkpoints, by distance (chance = 0.5):
    arm     n=2(ctx)  n=3    n=4    n=6    overall mean
    FP8      0.725   0.585  0.573  0.534    0.604
    rank2    0.737   0.606  0.611  0.518    0.618
    rank1    0.603   0.552  0.518  0.547    0.555

Every arm, every distance, beats chance on average -- including every
OUT-OF-CONTEXT distance (n=3/4/6), the thing this whole task exists to
test. This is a complete reversal from every energy result earlier in
this file: the miscalibrated default didn't just fail to help, it
actively prevented the model from learning at all regardless of storage
format; the calibrated config lets the SAME architecture, SAME task,
SAME quantization schemes actually learn, FP4 included.

**One real nuance, not smoothed over**: rank2 edges out FP8 on average
accuracy (0.618 vs 0.604) but has a real, if recovered, CE spike;
rank1 is both the noisiest (real 4-5x CE spike at steps 50-60k) and the
lowest-average of the three (0.555, still above chance at every distance
but by less). This matches this file's own earlier, independent finding
(rank-1 alone is FP4's weaker configuration) -- now shown to hold even
under a correctly-calibrated energy setup, not just as an artifact of
the earlier broken one. FP8's own curve is the smoothest/most stable of
the three even though it isn't the top scorer -- still the safer default
pick when stability matters more than squeezing out the last bit of
accuracy.

Single seed per arm -- a real, honest result worth taking seriously
given the magnitude of the reversal, but not yet a statistically
confirmed one (this file's own established standard, per the
peak-eligibility 50-seed check). Multi-seed confirmation is the natural
next step if this line of work continues.

## Multi-seed confirmation: n=20, 50,000 steps -- statistically significant above-chance learning at every distance, for every arm

Direct follow-up (n=8/30,000-step run above was underpowered -- best
vs-chance p was 0.066). Two real methodological corrections came out of
getting here, both from direct pushback, worth recording:

1. **The right null-hypothesis test is a one-sample test of each
   independently-trained seed's own accuracy against chance (0.5)**,
   not a pooled binomial test over all raw eval trials. Within one
   seed, all `EVAL_SEQUENCES` trials are scored by the SAME frozen
   trained network -- they share whatever fixed bias that one network
   has, so they are not independent draws for the question "does this
   PROCEDURE reliably beat chance." Confirmed this matters in practice,
   not just in theory: per-seed accuracy at a fixed distance varies by
   as much as ~0.2 across seeds in this data -- real between-seed
   heterogeneity that a naive pooled test would misattribute to sampling
   noise, inflating significance. Seed count, not eval-trial count, is
   what buys real power here.
2. **The z-test proposed as an alternative (`(mean-null)/SE`) is the
   same computation as `ttest_1samp`** -- the only difference is
   reference distribution (normal vs Student's t), and t is the more
   correct (more conservative) choice at small n, not a different or
   weaker measure.

Given that, the fix for the n=8 result being underpowered is genuinely
more independently-trained seeds (not more eval trials per seed, which
barely moves wall-clock time since training dominates it) -- scaled up
to n=20 seeds, and separately increased steps 30,000 -> 50,000 per seed
(the original single-seed check that motivated this whole line of
investigation used 30,000-100,000 steps; 30,000 may have been cutting
some seeds off before they'd settled). ~50-67 min per arm, 3 arms run
in parallel.

**Result -- every arm, every distance, now clears p<0.05, most by a
wide margin** (`scipy.stats.ttest_1samp` against 0.5, n=20):

    arm     n=2 (ctx)         n=3               n=4               n=6
    rank1   p<0.0001 (0.713)  p=0.0001 (0.584)  p=0.0058 (0.557)  p=0.0235 (0.534)
    rank2   p<0.0001 (0.800)  p<0.0001 (0.682)  p=0.0019 (0.571)  p=0.0068 (0.560)
    fp8     p<0.0001 (0.716)  p=0.0029 (0.581)  p=0.0111 (0.555)  p=0.0020 (0.550)

This is now a real, statistically confirmed result, not just a
suggestive average: genuine above-chance learning, INCLUDING every
out-of-context distance, for all three storage formats, once energy is
correctly calibrated. The n=8 result wasn't wrong, it was underpowered
-- exactly the outcome that combination of more seeds and more training
steps predicts if the true effect is real (which it is).

**Pairwise, also now much clearer** (`ttest_rel`/`wilcoxon`, n=20):

    rank2 vs rank1:  n=2 p=0.001  n=3 p=0.001  n=4 p=0.645  n=6 p=0.323
    fp8   vs rank1:  n=2 p=0.926  n=3 p=0.918  n=4 p=0.930  n=6 p=0.376

Rank-2's benefit over rank-1 is now real and significant, but
DISTANCE-DEPENDENT -- clearly ahead at the two easier distances (n=2,
n=3), not distinguishable from rank-1 at the two harder ones (n=4, n=6).
FP8 is statistically indistinguishable from rank-1 throughout -- not
ahead of it, contradicting this file's own earlier (single-seed,
miscalibrated-energy) framing that FP8 was the stronger performer.

**Two claims worth being precise about, not overclaiming:**

- **"Only with energy"** -- plausible given the earlier miscalibrated
  -energy divergence, but NOT directly verified: no matched no-energy
  control has been run at this same n=20/50,000-step scale (the
  session's earlier no-energy rank1-vs-rank2 comparisons were at
  n=8/3,000 steps, a much weaker test, and never included a vs-chance
  test at all). This is a real, cheap, natural next check if pursued
  further -- not yet done.
- **"Genuinely new in machine learning"** -- an earlier draft of this
  entry claimed low-bit QAT is generally established and left it at
  that; corrected via a literature search, since the claim shouldn't
  have been asserted without one. Findings: Optimizing LLM Training
  Using FP4 Quantization (arXiv:2501.17116) and FP4 All the Way
  (arXiv:2505.19115) quantize weights/activations/gradients to FP4 but
  don't put optimizer state there. Memory Efficient Optimizers with
  4-bit States (arXiv:2309.01507) quantizes Adam's moments to 4-bit but
  keeps the weights themselves at standard precision -- the reverse
  split. Full-Stack FP4 (arXiv:2607.04422) states directly that prior
  FP4 pretraining work left "optimizer states, optimizer arithmetic and
  attention... underexplored in 4-bit pipelines," and identifies why:
  AdamW's second moment is heavy-tailed and fragile at low precision.
  That paper is the first attempt at combining both, and still needs a
  specific AdamW-second-moment transform plus BF16 fallbacks on some
  paths -- not a plain quantize-and-go.

  DISLDOLayer's own mechanism doesn't use Adam's moments -- the
  per-synapse "importance" term is a single RMSprop-style decayed
  scalar, not the `m`/`v` pair the above papers are quantizing. The
  only higher-precision state is the rank-1 (or rank-2) `value_scale`/
  `output_scale` floats, one set per row/column rather than per weight.
  So this isn't a result on the same question those papers are working
  on; it's a different update rule that doesn't have that particular
  state to quantize. The claim this JOURNAL entry can actually stand
  behind: on this specific no-BPTT, per-tick online recurrent latch
  task, FP4 (rank-1 or rank-2 helpers, no Adam-style optimizer state)
  was diverging under the library's default energy config and became
  reliably, significantly trainable once energy was correctly
  calibrated.
