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
