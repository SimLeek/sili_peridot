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
