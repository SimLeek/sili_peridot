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
