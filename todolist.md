# sili_peridot todolist

Goal: convert `MiniCPM5-1B-Base` (standard `LlamaForCausalLM`, 24 layers,
hidden=1536, intermediate=4608, heads=16, kv_heads=2 GQA, head_dim=128,
vocab=130560, rope_theta=5e6, rms_norm_eps=1e-6, untied embeddings, no
biases, no qk-norm, no MoE/vision) into a model that runs and trains
**entirely on `sili__new`**, no PyTorch at inference. Repeated transformer
layers get folded into one sparse layer; that layer is then re-run as a
recurrence over "fold depth" (24 steps); we train a "column" per input
index — one neuron per fold-depth step for that index — whose *average
over the recurrence* is pushed toward the true input value at that index,
turning the folded stack into a sparse next-input predictor while the
energy system still manages sparsity/activity budget. Expect materially
lower quality than the dense original; the bar is "usable," not parity.

This file is the living plan. Update statuses as work lands. Keep
individual source files under ~1k lines where practical; tests/examples
are exempt from that budget.

## Environment (done)

- [x] Surveyed `sili__new`, `MiniCPM5-1B-Base`, `robonet_sili` (empty —
  not needed for this task, see note below), `sili_peridot` (git-init'd,
  empty, remote already set to `git@github.com:SimLeek/sili_peridot.git`).
- [x] `/home/simleek/claude_code/.venv` — `python3 -m venv
  --system-site-packages`, inherits global `torch==2.6.0`,
  `numpy==2.2.3`, `pybind11==2.13.6`. Added `safetensors`, `transformers`
  (transformers is a *reference-only* dependency, for loading the
  original HF model to diff against — never a runtime dependency of the
  converted sili model itself).
- [x] `pip install -e ./sili__new` — builds `sili._cpu` clean
  (`from sili import _cpu`, never bare `import _cpu` — pybind11
  double-registration hazard).
- [x] Confirmed `_cpu` already exposes `sparse_attention_backward`,
  `banded_attention_backward`, `sparse_banded_attention_backward` with
  real (non-stub) gradient math in `sili/lib/headers/attention.hpp` —
  bound in `cpu_backend.cpp` but **never called from any Python code and
  never tested**. This is a materially smaller gap than "write backward
  kernels" — see Phase A2.
- [ ] `robonet_sili`: empty repo, no code, not integrated with anything.
  Confirmed **not needed** for this task (text-only conversion, no
  remote-device/serving concerns). Leave untouched unless a later phase
  genuinely needs a serving interface.

## Required reading (for continuity across sessions)

Read once, keep in mind throughout:
- `sili__new/README.md` — repo layout, build/test instructions, the
  known `tests/unit/run_tests.sh` nonzero-exit-is-not-a-broken-build
  caveat.
- `sili__new/refactoring_todo.md` — active priority queue (batched
  `forward_dense`/`forward_sparse` test updates, adaptive rescale policy,
  getting `rnn_fold`/`gen_toy_mistral` working, final cleanup pass), and
  the deferred/backburner lists (MoE expert-merge, conv sparsification,
  RTAC critic-through-trunk/replay buffer, vision per-patch — **none of
  these block MiniCPM5**, which is dense/text-only/non-vision).
- `sili__new/energy-params.md`, `energy-personality.md`,
  `energy-proofs.md` — the homeostatic energy model's formulas and the
  personality-parameter mapping; needed before touching `energy.py`.
- `sili__new/docs/requirements_vlm_streaming_rtac.md` — the Mistral-24B
  VLM workstream this pipeline was built against; mainly useful as a
  worked example of hardening a real checkpoint's schema, not directly
  applicable (MiniCPM5 has no vision tower).
- `sili__new/sili/sparse_rnn.py` — `FoldedLayer`, `SparseRNNCell`,
  `EnergyDynamics` wiring, `SynaptogenesisSchedule`.
- `sili__new/sili/energy.py` — `EnergyDynamics`/`_apply_energy_dynamics`.
- `sili__new/sili/conversion/rnn_fold.py`,
  `sili__new/sili/conversion/model_reconstruct.py`,
  `sili__new/sili/conversion/sparse_prune.py`,
  `sili__new/sili/conversion/streaming_prune.py` — the conversion
  pipeline proper.
- `sili__new/tests/integration/test_toy_mistral.py` and
  `tests/unit/python/gen_toy_mistral.py` — the closest existing
  end-to-end worked example (KB-scale toy weights, not a real
  checkpoint); pattern to imitate for a real-MiniCPM5 version.
- `MiniCPM5-1B-Base/config.json`, `README.md` — architecture is exactly
  vanilla `LlamaForCausalLM` per the model card itself ("no custom
  kernels, no model-code fork"); no `scale_emb`/`scale_depth` MiniCPM
  quirks present in this config, unlike older MiniCPM releases.

## Phase A — `sili__new` library gaps (fix in sili__new, not sili_peridot)

These are genuine library gaps blocking correctness or the user's
explicit "training-through-attention" requirement — fix upstream in
`sili__new`, then depend on it from `sili_peridot`, per the standing
instruction to only modify the library repos when they're lacking
something as a library. **Push access**: `claude_sili.token` (fine-grained
GitHub token, HTTPS push, same pattern as `sili_peridot_access.token` —
`git push https://x-access-token:<token>@github.com/SimLeek/sili__new.git`,
never persisted into `.git/config`). Confirmed via fetch that nothing new
is on the remote beyond what's checked out locally (`main` @ `87daa2e`)
— the energy-dynamics redesign below (from a separate chat with another
Claude session) exists only as design notes, not committed code anywhere;
it needs to be implemented fresh against the current `sili/energy.py`
(that session called the file `energy_dynamics.py` — treat that as a
proposed rename to fold in, not evidence of a second file).

**Correctness requirement that cuts across all of Phase A: the converted
model must actually run sparse, not merely "compile against sparse
types."** `FoldedLayer.from_descriptor`'s CSR round-trip and
`forward_dense`/`backward_dense` can produce something that is
structurally CSR but behaviorally dense (near-100% activity density, or
a `from_dense`-style independent top-k re-derivation that ignores which
indices are actually energy-gated). Verify empirically post-conversion
and post-energy-gating (measured sparsity ratio, not just "it runs") at
every phase boundary — this is the actual point of using `sili` at all,
not a nice-to-have.

- [ ] **A1. Thread real hparams into the Llama reconstruction path
  instead of guessing.** `model_reconstruct.py:361` hardcodes
  `rms_eps=1e-5` (MiniCPM5 needs `1e-6`); `_LlamaRotaryEmbedding.__init__`
  (`model_reconstruct.py:195`) hardcodes RoPE `base=10000` (MiniCPM5
  needs `5000000`), and nothing in the module reads `config.json` at all
  — hparams are re-derived from tensor shapes only. Add an optional
  hparams-override path (explicit `rope_theta`, `rms_norm_eps` kwargs
  threaded through `_infer_llama_hparams`/`LlamaModel`, sourced from the
  checkpoint's real `config.json`) rather than continuing to guess values
  that aren't recoverable from weights alone. Also override
  `rnn_fold.infer_seq_len_from_attn_weight`'s `band_half_width` heuristic
  explicitly for RoPE models (it's tuned for fixed-position attention by
  default, per its own docstring). **Status: agreed, unchanged from
  original plan.**
- [ ] **A2. Wire attention into the Python `Tensor` autograd graph
  ("training-through-attention").** The C++ backward kernels already
  exist and are pybind-bound (`sparse_attention_backward`,
  `banded_attention_backward`, `sparse_banded_attention_backward`,
  confirmed real softmax-Jacobian gradient math in `attention.hpp`, not
  stubs) — the actual gap is a `Tensor`-graph wrapper (a Python
  op/Module, mirroring `FoldedLayer.forward`'s `_children`/manual
  `_backward` closure pattern in `sparse_rnn.py`) that calls the forward
  kernel, stashes what backward needs (Q/K/V), and calls the matching
  `*_backward` kernel from its `_backward` closure. Add finite-difference
  gradient-check tests (dQ/dK/dV vs numeric) for all three attention
  variants — none currently exist (`test_sili.py`'s `TestSparseAttention`
  /`TestBandedAttention`/`TestSparseBandedAttention` only check forward
  shapes/values). This is required before any transformer-attention layer
  in the converted model can be fine-tuned rather than frozen at
  conversion-time weights. **Status: agreed, unchanged from original
  plan.**
- [x] **A3. Build the column-averaging energy mechanism — revised
  parameterization.** Already scoped (not built) in
  `refactoring_todo.md`'s "Energy-as-input and column energy" note, and
  independently corroborated by the other session's "column/fiber
  averaging" note for `energy.py` — these are the *same* feature, unify
  them into one implementation. "Column" = one neuron per fold-depth
  step (24 neurons) per input index `i`; target = "average of the
  column over the fold-depth recurrence."
  - **`p` (hard top-p ceiling) is a hardware/telemetry compute-limit
    factor, not a learning-quality knob, and must sit clearly *above*
    the KL density target — not below or equal to it.** Corrected
    defaults: `p ≈ 0.05`, KL target density `≈ 0.01` (roughly 5x–10x
    apart, not the same order). Rationale (direct correction to the
    original plan, which had this backwards): the *actual* sparsified
    output should usually land below `p` on its own, driven down by KL/
    shutoff/forced-firing competing against the column-averaging
    pressure — `p` should only ever bind when genuinely resource
    -constrained (thermal/battery/compute-budget), not act as the
    default limiter of learned sparsity. Add
    `assert density <= p * 0.8` (or similar margin) so the two can't
    silently collide/invert.
  - **Do not exempt column neurons from KL/shutoff/forced-firing
    competition.** The original plan's instinct ("give column neurons a
    reserved slot, exclude them from shutoff") was wrong per direct
    correction: the whole point is that KL sparsity, shutoff, and forced
    firing should genuinely *fight* the column-averaging objective, so
    the network is forced to learn to manage energy **and** predict the
    next input well simultaneously, under real competitive pressure —
    not have column tracking made artificially easy by carving out
    protected neurons. Column neurons compete on equal footing;
    convergence is a training outcome, not a structural guarantee.
  - Add a target-aware loss term alongside `EnergyDynamics`'s existing
    `energy_loss` (in `_apply_energy_dynamics`, `sili/energy.py`) — needs
    both `h`/`h_out` *and* the original per-token input, which
    `EnergyDynamics.forward` doesn't currently receive at all.
  - Add unit tests: verify column mean moves toward the target input
    over training steps *despite* KL/shutoff/forced-firing pressure
    (not in spite of it being disabled), verify `actual_p` stays below
    `p` under normal (non-resource-constrained) operation, verify the
    assert catches a misconfigured `density > p` setup.
  - **Done as `sili.energy.column_averaging_loss`** — a plain
    differentiable Tensor expression (reshape+mean+MSE), not a
    hand-derived gradient, kept as its own function (not folded into
    `_apply_energy_dynamics`) and combined via the existing
    `combine_losses` pattern. Runs on `h_out`, per the "don't exempt
    column neurons" requirement above. Takes an optional `indices`
    parameter (a new general `Tensor.gather` op) so it can track a
    *subset* of a larger state rather than requiring the whole tensor to
    be exactly column-shaped — added after review pointed out state
    should be allowed to be bigger than what's tracked (see A4's
    redesign note below for the fuller story). 11 tests (gradient vs.
    finite differences, SGD convergence, `combine_losses` integration,
    weight scaling, `indices` subset-selection/gradient-isolation).
    Docstring trimmed after review (was several times longer than the
    function itself). PR [#5](https://github.com/SimLeek/sili__new/pull/5).
  - **Honest caveat found while validating end-to-end**: full
    convergence under REAL energy competition (this loss +
    `EnergyDynamics` gating + a real sparse layer, all three jointly
    optimizing) did not cleanly converge on a toy 96-neuron layer within
    a few thousand steps — it oscillates in a rough stalemate, even
    though each piece converges correctly in isolation. Treated as
    expected (a genuinely harder coupled dynamical system, not a bug —
    matches the design's own "the network will have to train a bit"
    expectation) rather than force-fit toy hyperparameters to fake a
    clean convergence demo; the committed end-to-end test asserts
    stability (bounded losses, hard `p` ceiling respected, no
    divergence) instead. **Full convergence validation needs real model
    scale — this is what Phase B8/B9 below actually has to demonstrate,
    not something a unit test can honestly claim first.**
- [x] **A4. Columns must track "next input average" at *every* fold
  step, not just after the full depth pass — this requires pre-seeded
  cross-depth synapses, not synaptogenesis alone.** Direct correction to
  the original plan (which had column convergence implicitly happening
  only "at the end," which is wrong): since folding an LLM into an RNN
  means one recurrence step only advances *one* virtual layer, the
  column average after step 1 should already approximate the next input
  — imperfectly at first, improving with training, but present from
  step 1, not only after 24 steps.
  - **Why synaptogenesis alone can't get there in practical training
    time (the math the user asked to work through):** `build_probes(k)`
    with the library's intended `k=4` evaluates a `k×k=16`-candidate
    cartesian window *per row, per synaptogenesis round* — sampled from
    a search space of size `n_in * n_out` per suffix (on the order of
    `1536 * 2048` ~ 3M for q_proj alone). The specific cross-fold-depth
    connections a working column needs are a tiny, structured subset of
    that space (roughly `input_size * n_folds` ≈ `1536 * 24` ≈ 37k
    target pairs out of millions of candidate positions). Random/
    activity-correlated `build_probes` sampling has vanishingly small
    per-round odds of proposing the *specific* pair a column needs, and
    each round already costs ~14x a plain backward pass (measured, see
    `benchmark_synaptogenesis.py`). Expecting synaptogenesis to
    *discover* the right cross-depth topology from scratch, at that cost
    per round, is not practical within any reasonable training budget.
  - **Fix: manually pre-seed zero-value, low-importance synapses
    connecting many virtual (fold-depth) layers to each other during the
    folding/conversion step itself**, so the CSR already has the
    *structural* connectivity a column needs before training starts —
    training only has to adjust weights (fast, ordinary backprop),
    not discover topology (slow, synaptogenesis-bound). These synapses
    start at value 0 / low importance so they don't disturb the
    pretrained-weight forward pass until the column-averaging objective
    actually needs and grows them.
  - **Weight the pre-seeding toward the "output" side, or duplicate
    those synapses, for columns at conversion time.** Because these
    zero-value synapses start with low importance, ordinary importance
    -based synaptogenesis pruning could kill them before the
    column-averaging loss has had a chance to make them useful — give
    the synapses feeding a column's *output*-facing connections either
    elevated initial importance or duplicate multiplicity (redundant
    parallel synapses) so the column-tracking signal reliably survives
    early pruning rounds while training ramps up its usefulness.
  - **Why this should work despite lag**: central/deeper virtual layers
    are closer to abstract concepts than to literal next-token identity,
    so a column average computed after only 1–2 virtual-layer steps will
    initially lag the true next input — but per direct guidance this is
    expected to be fixable with a modest amount of training, *given* the
    CSR is pre-seeded for it, precisely because deeper layers still
    carry usable (if less literal) predictive signal even before
    training converges.
  - Add a `FoldedColumnLayer`-style mode/subclass that (a) retains
    per-fold-step hidden state instead of summing immediately
    (`FoldedLayer.forward` currently reshapes
    `[batch, n_folds, out_dim]` and sums over the fold axis right away,
    `sparse_rnn.py:416` — the per-step values needed for columns aren't
    retained today), and (b) exposes the pre-seeded cross-depth synapse
    construction as an explicit conversion-time step, leaving
    `FoldedLayer`'s existing summed-output contract untouched for
    callers that don't need columns.
  - **Done as `sili.sparse_rnn.FoldedColumnLayer(FoldedLayer)`** — (a)
    `forward()` override skips the fold-sum, returns
    `[n_folds*out_dim]`; verified by manually summing it over the fold
    axis and confirming it exactly reproduces plain `FoldedLayer`'s
    output. `from_descriptor` is inherited unchanged — no override needed.
  - **(b) redesigned once already, per direct review, before landing.**
    First version pre-seeded *within* one suffix's existing stacked
    weight matrix (row `i` → column `(fold_step, i)` for every fold
    step). Review caught two real problems: (1) that construction creates
    **no actual connection between virtual layers** — row space stays the
    shared, non-fold-indexed input, so there's no pathway for layer `t`'s
    output to influence layer `t+1` at all, contradicting "skip
    connections from virtual layer i to virtual layer j." (2) the
    `out_dim == in_dim` requirement doesn't fit **any** of MiniCPM5's real
    suffixes — checked, none are square (`down_proj` 4608→1536, `o_proj`
    2048→1536, etc.) — so the whole mechanism as first built couldn't
    attach to the actual target model.
    Traced both back to `RNNFoldedBlock.forward` (`conversion/rnn_fold.py`,
    the real reference recurrence: `state=0; for i: state +=
    block_i(x+state)`, each layer seeing everything accumulated by every
    *prior* layer) — `FoldedLayer`'s single-matmul-then-sum trick is a
    cheap first-order approximation of that (every fold step computed
    independently from the same external input, no cross-step
    dependency). Neither problem is fixable by pre-seeding inside one
    suffix's existing matrix at all.
    **Current design**: `build_fold_skip_layer(n_folds, out_dim)` — a
    fresh, from-scratch sparse layer mapping a `FoldedColumnLayer`'s own
    `[n_folds*out_dim]` *output* space back to itself (always square,
    resolving problem 2 directly, regardless of the wrapped suffix's
    in/out shape), pre-seeded as a banded diagonal (bandwidth = `out_dim`
    by default — "manhattan distance less than the original layer size,"
    per direct guidance), all zero-valued since this connectivity has no
    pretrained equivalent in the original unfolded model. Genuine virtual
    -layer-to-virtual-layer skip connections (resolving problem 1).
    `apply_fold_skip(skip, raw, lr)` wires it into the Tensor autograd
    graph; typical use `refined = raw + apply_fold_skip(skip, raw, lr)`.
    24 tests total across A3/A4 (`build_fold_skip_layer` shape/banding/
    finiteness, composed with `FoldedColumnLayer` end-to-end incl.
    gradient reaching both layers and skip weights moving off zero).
    PR [#5](https://github.com/SimLeek/sili__new/pull/5).
  - **FP4 landmine hit and fixed while testing the composed pipeline**:
    `build_fold_skip_layer`'s all-zero initial weights were structurally
    stuck at zero — the same per-row `value_scale` FP4 gotcha this
    project's own notes already documented, which got missed on the first
    pass here too. Fixed via a new `expected_lr` parameter that sets
    `value_scale`/`importance_scale` relative to the learning rate the
    layer will actually be trained with — must match whatever `lr` is
    passed to `apply_fold_skip` later, or connections silently never move.
  - **The original "elevated per-connection importance" ask is still not
    achievable** with the currently-bound C++ API (`SparseLinearLayer.
    load_weights` has no importance array; `set_importance_scale_raw` is
    a per-ROW scale, not per-connection) — moot in the new design's exact
    original form (no pretrained/new-synapse mixing to protect against
    within a row anymore), but the underlying C++ gap is still real if a
    future need for true per-connection importance protection comes up.
  - **Second finding — reframed after review, not a bug**: while tuning
    the end-to-end validation, found that a fired-but-not-top-p-selected
    neuron's energy is never reset in `_apply_energy_dynamics` — under a
    literally repeated-identical input, `aux_loss` grew unboundedly
    (0.08 → 177+ over 270 steps) since `drive > 2*activation_cost` nets
    positive energy growth for a chronic top-p loser. Confirmed not a
    column-averaging bug (isolated, both pieces converge correctly).
    **Per direct feedback, this is curiosity/novelty-seeking pressure
    working as intended** — "nothing new is happening, pressure should
    build" is the correct response to a truly static input, not a defect;
    every other real usage in this codebase never exercises this regime
    because inputs vary step to step (as MiniCPM5's real token stream
    always will). Documented in `sili__new/TODO.md` as an emergent
    property worth deliberately harnessing later (e.g. as a Phase E
    action-pathway input), not something to clamp away.
- [ ] **A5. Sparse activation + sparse backprop conversion — this is
  core to why `sili` exists, not a deferred perf nice-to-have.** Direct
  correction to the original plan, which mis-scoped this as low
  priority: `FoldedLayer.forward`/`backward` are dense-only today even
  though the energy-gated state is mostly zero after the top-p gate.
  Scheduling correction — this is not "revisit if inference is
  impractically slow," it is **"do this once the model runs end-to-end,
  and directly check the performance difference"** — a required,
  scheduled step, not a maybe. Concretely this means: implement A1/A2/
  A6/A3/A4 now (Phase A proper), but **execute A5 after Phase B7**
  (first working dense-forward conversion of MiniCPM5 exists) and
  **before B8/B9** (column-averaging training) — not before Phase B
  starts, since "the model runs end-to-end" is a Phase B milestone, not
  something available yet inside Phase A itself:
  - Switch `FoldedLayer`'s hot path to the sparse fast path
    (`SparseRNNAgent.forward`'s `CSR.from_dense` pattern is the existing
    precedent) for both forward and backward.
  - This directly composes with the branching-ratio/CSR-construction
    fix below (A6 item 4): the sparse CSR fed into `FoldedLayer`'s
    sparse forward should be built from `EnergyDynamics`'s own
    kept-index set, not a redundant independent top-k pass.
  - Measure and report the actual sparsity ratio and the dense-vs-sparse
    wall-clock/throughput difference on the real converted MiniCPM5
    model, not a toy example — this is the concrete evidence that the
    conversion produced a genuinely sparse, not just nominally-CSR,
    model.

## Phase A6 — `SparseRNNCell`/`EnergyDynamics` measurement fix (from a
parallel design discussion; implement in strict order, each item unblocks
the next)

Context: a separate conversation identified a real identifiability
failure in how recurrent "healthiness" is measured. `SparseRNNCell.
forward` computes `h = input_proj(obs) + recurrent(state)` and only ever
measures activity on the *summed* `h` — so a `BranchingRatioTracker`
watching `h` cannot distinguish "the recurrent pathway genuinely
self-propagates" from "fresh input alone keeps activity in the target
band while the recurrent branching factor is silently 0." This matters
here specifically because column-averaging (A3/A4) *depends on* the
recurrent/fold-depth pathway actually carrying signal forward — if the
measurement can't tell recurrence-driven from input-driven activity, we
also can't tell whether training is actually building a working column
mechanism or just riding fresh input every step. Branching-process theory
adds a second reason this isn't cosmetic: a branching process with no
immigration term has an absorbing extinction state, and for branching
ratio `m <= 1` (the entire intended target band, `0.97`–`0.99`),
extinction is the eventual almost-sure outcome without something acting
like immigration (self-loop/leak/bias independent of fresh input) —
"keep `m` near 1" alone doesn't guarantee real persistence.

Order matters — items build on each other; do not skip ahead:

0. ~~[Blocking] repo access~~ — resolved, `claude_sili.token` grants
   push access to `sili__new`.
1. [x] **Call-site fix**: `SparseRNNCell.__init__`'s `EnergyDynamics(...,
   kl_eps=1e-4, ...)` call needs the `kl_eps` → `activation_threshold`
   rename applied at the call site too. Done, PR
   [#3](https://github.com/SimLeek/sili__new/pull/3).
2. [x] **`CSR` gets an `nnz` property** (`len(self.indices)`). Done.
3. [x] **`Tensor` gets `__getattr__` delegation** to `self.data` when it
   wraps a `CSR`, plus `is_csr`/`is_dense` properties, kept separate from
   the weight-side delta-CSR format flag. Done.
4. [x] **Unify the two sparsification passes.** `_apply_energy_dynamics`
   now returns `kept_indices` (the gate decision it already made);
   `CSR.from_kept_indices` builds a CSR from indices-from-the-gate +
   values-from-pre-gating-`h` (not `h_out`'s post-gating constants).
   `SparseRNNCell` caches `(_prev_kept_indices, _prev_h_dense)` and uses
   them for the next step's recurrent CSR, falling back to
   `CSR.from_dense`'s independent top-k only at true step-0 or after
   `reset()`/`load()` (cache invalidated there). Done.
5. [x] **Split the branching-ratio measurement.** `SparseRNNCell.forward`
   measures `recurrent(state)`'s own activity *before* summing with
   `input_proj(obs)`, feeding a new `BranchingRatioTracker` (single-lag
   OLS slope, a simplified Wilting & Priesemann-style estimator, plus
   `avalanche_sizes()` for the SOC power-law-tail check). Done. **Update
   per review**: added `EMABranchingRatioTracker` alongside it (same
   OLS-slope estimator, O(1) memory via exponentially-weighted running
   statistics instead of a hard window — `alpha` trades fast-response
   against long-term-smoothness continuously, where `window` only offered
   that indirectly/discretely; run two instances at different alphas for
   both at once). `SparseRNNCell(branching_tracker="window"|"ema", ...)`
   switches which backs `self.branching_recurrent`, default `"window"`
   (unchanged behavior). `avalanche_sizes()` stays windowed-only
   (inherently needs a retained sequence). PR
   [#3](https://github.com/SimLeek/sili__new/pull/3).
6. [~] **Open design question, documented not resolved** (deliberately —
   this needs real design thought, not a rushed answer): does the
   recurrent pathway need something acting like an internal immigration
   term (self-loop, leak, or a bias independent of `obs`) to have a fixed
   point other than extinction under `m <= 1`? Flagged in code comments;
   still needs an actual short design note in `energy-params.md` before
   anything depends on the answer.

**Also fixed alongside 1-5** (the actual bug, not just a docs gap): `p`
was inverted relative to `density` in both `EnergyDynamics`'s own default
(was `0.02`) and `SparseRNNCell`'s derivation (`density` could reach
`0.9*p`). New: `EnergyDynamics.p` defaults to `0.05`;
`SparseRNNCell` sets `density=percent_active`, `p=min(1.0,
percent_active*5)`; added `assert density <= p*0.8`. Also added
(opt-in, default off): `EnergyDynamics.forward(h,
density_override=...)` and `SparseRNNCell`'s
`dynamic_density_from_branching_ratio` flag — a first-cut proportional
nudge of the KL density target from the measured recurrent-only
branching ratio, explicitly labeled as a first cut, not a
first-principles derivation.

**Found and documented (`sili__new/TODO.md`), NOT fixed — separate task,
not on the MiniCPM5 critical path**: `SparseRNNCell`/`SparseRNNAgent`
cannot actually be constructed today. `DISLDOLayer`/`SISLDOLayer`
(Python) call C++ methods (`_cpu.DISLDOLayer`, `_cpu.SISLDOLayer`,
`optim_weights`, `decay_importance`, `optim_synaptogenesis`) that don't
exist on *any* currently-bound C++ class — three incompatible C++-layer
API generations exist in this codebase (whatever `DISLDOLayer`/
`SISLDOLayer` assume — never implemented; `DISLDOLayerV` — bound but a
different API; `SparseLinearLayer` — bound and what actually works,
proven by `FoldedLayer`/`test_toy_mistral` and Mandelbrot's
`SparseCore`/`MistralCore` via `make_grown_sparse_layer`). **Not blocking
MiniCPM5 conversion** (that goes through `FoldedLayer`/`SparseLinearLayer`
directly, never `SparseRNNCell`) — worth fixing eventually (rebuild
`DISLDOLayer`/`SISLDOLayer` on `SparseLinearLayer`'s real API) but not
urgent for this project. `tests/unit/python/test_sparse_rnn_cell.py` has
33 tests: 20 pass (everything above, verified without needing a working
`SparseRNNCell`), 13 marked `xfail(strict=True)` citing this bug, ready
to flip green once it's fixed.

### `sili/energy.py` changes (bundle with A3, same file)

- [ ] Rename `kl_eps` → `activation_threshold` (name described the
  mechanism, not the purpose) — update all call sites (`SparseRNNCell`,
  any others found via grep) in the same change. Done.
- [x] `p` default corrected upward (`0.05`) and documented explicitly as
  a hardware/telemetry-driven ceiling (thermal, battery, update-rate),
  never a tuning knob for learning quality. Done.
- [x] `assert density <= p * 0.8` — this was the actual bug in the
  original defaults, not just a documentation gap. Done, and confirmed
  every existing `EnergyDynamics(...)` call site in the tree already
  used `density=p/2`, so nothing else needed updating.
- [x] Docstrings for `precision` (`lambda_kl`) and `reactivity` (`alpha`)
  made explicit that these are two *different* control loops —
  population-level (KL, achieved density vs. target density) vs.
  per-neuron (energy_loss, `new_energy_t` vs. `setpoint`). Done.
- [x] **Biggest structural change**: compute the KL term's target
  dynamically from the *measured branching factor of the recurrent-only
  pathway*. Done as an explicit **opt-in** (`density_override` param on
  `EnergyDynamics.forward`, `dynamic_density_from_branching_ratio` flag
  on `SparseRNNCell`, default off) — a first-cut proportional nudge
  around the base density, NOT a first-principles derivation; labeled as
  such in code. Land the real formula later once avalanche/branching
  -ratio instrumentation (next item) has actually been run and observed,
  per the design discussion's own "needs real design thought" framing.
- [ ] Add column/fiber averaging as an optional mode: for
  `state_space >= ~2x input_space`, a subset of the state space is
  forced to track the per-input-neuron average — this is A3/A4's
  column-averaging mechanism; implement once, here, rather than as a
  separate parallel system. **Not done here** — belongs with A3/A4's
  fuller column-averaging design (fold-depth columns, pre-seeded
  cross-depth synapses), not bundled into the A6 measurement-fix commit.
- [x] Keep `exploration < drive/2` as-is, but add a doc line on *why*:
  symmetry-breaking between otherwise-identical neurons, not merely
  "avoid the hallucination/REM regime". Done.
- [~] **Instrumentation, not dynamics changes**: `BranchingRatioTracker`
  (per-region recurrent-only branching ratio) and `avalanche_sizes()`
  now exist and are wired into `SparseRNNCell.forward` — the
  *measurement* infrastructure is done and tested. **Not yet done**:
  actually logging/plotting these from a real training run (e.g. wiring
  into Mandelbrot's own metrics output) to check `m` tracks
  `1 - drive_normalized` as predicted, check for the SOC power-law tail,
  and watch `actual_p` vs. `p` for chronic pinning — that validation
  pass hasn't been run yet, only the plumbing to make it possible.
- [ ] **Noted dependency, not a change to this file**: a per-neuron
  critic (e.g. a transformer over energy/aux_loss context) needs to
  exist before any energy-management-via-world-modification action
  pathway is safe rather than wireheading-prone — relevant to the
  eventual Minecraft/robot capstone (Phase E), tracked as a blocker on
  *that* work, not on the MiniCPM5 conversion. Also note (no change
  needed): synaptogenesis already handles long-timescale resolution for
  neurons that persistently lose slot competition — this is load-bearing
  for "no permanently-starved rump population," worth documenting as a
  relied-upon existing property rather than something to (re)build.
- [x] **Citations** — added as `sili__new/CITATIONS.md` (linked from
  `energy.py`'s module docstring), grouped by topic:
  - *Homeostasis/cybernetics*: Ashby, W. R. — Homeostat (1948); *Design
    for a Brain*; Ashby's Law of Requisite Variety.
  - *Active inference/free energy*: Friston, K. — Free Energy Principle;
    predictive coding; dark-room problem resolution via interoceptive
    priors.
  - *Self-organized criticality/avalanches*: Bak, Tang, Wiesenfeld
    (1987) — sandpile model, SOC; Beggs & Plenz (2003) — neuronal
    avalanches; Wilting & Priesemann — multistep-regression (MR)
    estimator for branching ratio, separating external drive from
    internal propagation; Williams-García et al. — quasicriticality/
    "Widom line."
  - *Metabolic constraints/efficient coding*: Attwell & Laughlin (2001)
    — energy budget for cortical signaling; Levy & Baxter — efficient
    coding under energy constraint; Olshausen & Field (1996) — sparse
    coding.
  - *Reservoir computing*: echo state property/spectral radius tuning
    near 1 (Jaeger — echo state networks).
  - *Intrinsic motivation/curiosity*: Schmidhuber — "Driven by
    Compression Progress" (2009); Pathak et al. (2017) — Intrinsic
    Curiosity Module (ICM).
  - *Homeostatic RL (formal)*: Keramati & Gutkin — homeostatic
    reinforcement learning; Hull, C. — drive-reduction theory (1943).
  - *Sparse coding mechanisms*: Willmore & Tolhurst — lifetime vs.
    population sparsity (following Foldiak); Makhzani & Frey (2013) —
    k-sparse autoencoders; Rozell et al. (2008) — locally competitive
    algorithms (LCA).
  - *Signal detection/stochastic resonance*: signal detection theory
    (general); stochastic resonance literature (general).
  - *Multi-agent credit assignment* (relevant to the eventual
    action-pathway/critic work in Phase E): Wolpert & Tumer (1999) —
    COIN, factoredness/sensitivity, difference/aristocrat utility;
    Foerster et al. (2018) — COMA; Sunehag et al. (2017) — VDN; Rashid
    et al. (2018) — QMIX.
  - *Indirect encoding/decoupled updates*: Stanley, D'Ambrosio, Gauci
    (2009) — HyperNEAT/CPPN indirect encoding; Jaderberg et al. (2016) —
    synthetic gradients/Decoupled Neural Interfaces (DNI).

## Phase B — Convert MiniCPM5-1B-Base (in sili_peridot)

- [ ] **B1. Explicit hparams module** (`sili_peridot/model/config.py`):
  read `MiniCPM5-1B-Base/config.json` directly (don't rely on
  `model_reconstruct`'s shape-inference guesses) — hidden_size=1536,
  intermediate_size=4608, num_attention_heads=16, num_key_value_heads=2,
  head_dim=128 (note: 16*128=2048 ≠ hidden_size=1536 — q_proj/o_proj are
  NOT hidden_size-square, k_proj/v_proj out=256; treat every projection's
  in/out dims as explicit, never derived from hidden_size/num_heads),
  vocab_size=130560, rms_norm_eps=1e-6, rope_theta=5000000,
  tie_word_embeddings=False, num_hidden_layers=24, hidden_act=silu, no
  attention bias, no qk-norm.
- [ ] **B2. Load the checkpoint.** Single safetensors shard, ~2.16GB
  bf16 — fits fully in RAM (machine has 15GB, mostly free); the
  streaming/two-phase path (`streaming_prune.py`) is not needed for a
  model this size, use the plain in-RAM path
  (`sparse_prune.load_state_dict`/`model_reconstruct`'s loader).
- [ ] **B3. Prune to CSR.** Run `sparse_prune.sparsify_model` (or its
  primitives directly) with a threshold calibrated against MiniCPM5's own
  weight-magnitude distribution — don't reuse the toy-Mistral defaults
  uncritically. Keep 1-D tensors (norms) and embeddings' decision
  (dense vs sparse — both `embed_tokens` and `lm_head` are ~200M-param
  untied [130560, 1536] matrices, a meaningful chunk of the model; decide
  per `_keep_dense_reason()`'s existing rules, verify it makes a sane
  call for these shapes rather than assuming).
- [ ] **B3a. Verify actual sparsity, not just CSR-shaped output.** Per
  Phase A's cross-cutting correctness requirement: after pruning, check
  the real density (nnz / (n_in*n_out)) per tensor and log it — a
  threshold that leaves the model 60%+ dense defeats the point of this
  whole conversion, regardless of whether it technically round-trips
  through CSR types.
- [ ] **B4. Fold each of the 7 per-layer 2-D suffixes independently**
  (`self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`, `mlp.gate_proj`,
  `up_proj`, `down_proj`) across all 24 layers via
  `rnn_fold.detect_repeated_block_groups` + `stack_csr_vertical` +
  `fold_block_group` — block detection is generic name/shape matching,
  needs no MiniCPM-specific change. **Never combine suffixes into one
  `FoldedLayer`'s `layers` dict** — `FoldedLayer.forward` sums all
  suffixes down to the *first* suffix's `out_dim`, silently wrong for
  suffixes with different `out_dim` (q=2048, k/v=256, o=1536,
  gate/up=4608, down=1536, all different).
- [ ] **B5. `FoldedLayer.from_descriptor()` per suffix**, with correct
  FP4 handling: pre-scale rows to `max_abs/FP4_MAX`, `load_weights`,
  `set_value_scale_raw(r, row_scale)` (never `rescale_value_row()` after
  a pre-scaled load — re-encodes already-scaled values). Separately,
  before any post-conversion training/growth, also apply
  `set_value_scale_raw(r, lr/FP4_MAX)` reasoning from
  `test_mandelbrot_rl.py` so newly-grown synapses aren't structurally
  stuck at zero — these two value_scale requirements are in tension
  (faithful pretrained magnitude vs. trainable new-synapse step size);
  plan for a resync pass after initial conversion and before training
  starts.
- [ ] **B6. Attention assembly**: GQA (16 query heads : 2 KV heads,
  groups=8) + RoPE (`theta=5e6`) computed around the Q/K/V `FoldedLayer`
  outputs, using the new autograd-wrapped attention op from A2. Override
  `band_half_width` explicitly (RoPE, not fixed-position — the auto
  heuristic assumes the latter); pick a practical band/context width for
  this environment rather than the full 131072 training context.
- [ ] **B7. Assemble `MiniCPM5SparseModel`** (Python class in
  `sili_peridot/model/`): embedding lookup → 24-step fold-depth
  recurrence through the one folded transformer "layer" (mirrors
  `RNNFoldedBlock.forward`'s `state=0; for step: out=block(x+state);
  state+=out`, or `SparseRNNCell`'s persistent-state recurrence — pick
  whichever composes more cleanly with the column-retaining `FoldedLayer`
  from A4) → final RMSNorm → lm_head. Entirely `sili`/`sili._cpu` +
  numpy at inference — no PyTorch dependency in the runtime path (torch
  only used offline during conversion, and only by transformers for the
  comparison baseline).
- [ ] **B8. Column-averaging training loop**: for each input index `i`,
  the column of 24 fold-depth neurons should average toward `input[i]`
  **at every fold step, from step 1 onward** — not only after the full
  24-step pass (see A4's correction). This depends on B7's pre-seeded
  cross-fold-depth synapses (weighted/duplicated toward each column's
  output-facing connections, per A4) actually being in place before
  training starts — without them, expect the column mean to only ever
  reflect whatever a given step's *local* block naturally produces, with
  no path for gradient/importance signal to reach across depth steps.
  Use the Phase A3 column-averaging energy loss with corrected
  `p`/density parameterization (`p≈0.05`, density≈0.01, KL/shutoff/
  forced-firing left genuinely competing against the averaging
  objective — do not exempt column neurons), sparse activation
  throughout (Phase A5's sparse forward/backward, CSR built from
  `EnergyDynamics`'s kept-index set per A6 item 4 — not an independent
  top-k pass), sparse backprop (real per-row synaptogenesis:
  `build_probes(k=4)` once per layer, then `synap_step` once per row —
  not once total, and never scale `k` by row count, an already-reverted
  n²-blowup bug elsewhere in this codebase).
- [ ] **B9. Verify sparsity survives training, not just conversion.**
  Log density/energy stats (`actual_p` vs `p`, per-region recurrent-only
  branching ratio, avalanche size distribution) throughout training —
  these are the concrete falsifiable signals (per A6) that the model is
  actually operating in the intended sparse/critical regime, not one
  that collapsed to dense-and-thresholded or to input-only propagation
  with a silently-dead recurrent pathway.

## Phase C — Testing, evaluation, examples

- [ ] C1. Attention autograd correctness tests (finite-difference
  dQ/dK/dV) for all three C++ attention variants — lives in `sili__new`
  since it's testing the A2 library fix, mirror existing
  `TestSparseAttention` style.
- [ ] C2. Conversion correctness: convert MiniCPM5, run one forward pass,
  compare against the real HF `transformers` model (top-1/top-5 token
  overlap on a handful of prompts, rough perplexity ratio) — expect
  clearly worse than the dense original but not garbage; define
  "usable" concretely (e.g., "some, not-degenerate, moderately-coherent
  short completions" or a numeric perplexity bound) before calling this
  done.
- [ ] C3. Training test: short column-averaging training run, assert
  loss decreases, log density/energy stats, confirm column means track
  input better post-training than at conversion-time init.
- [ ] C4. End-to-end example (`examples/convert_and_run_minicpm5.py` or
  similar) analogous to `test_toy_mistral.py` but against the real
  checkpoint, fully torch-free at inference; document expected runtime
  and memory footprint on this machine (8 cores, 15GB RAM, no GPU
  offload assumed since this is a CPU-only sparse runtime).
- [ ] C5. Keep implementation files under ~1k lines each; split
  `sili_peridot/model/`, `sili_peridot/conversion/`, `sili_peridot/tests/`,
  `sili_peridot/examples/` rather than one monolith. Tests/examples are
  exempt from the line budget — favor coverage there.

## Phase D — Repo hygiene / publishing

- [x] `git init` already done, remote already set to
  `git@github.com:SimLeek/sili_peridot.git`.
- [x] Local `user.name`/`user.email` set to Claude's own identity (not
  the human user), per standing instruction — commits from this session
  attribute to "Claude <noreply@anthropic.com>", not co-authored.
- [ ] `pyproject.toml`/`setup.py` for `sili_peridot` itself (installable,
  editable, depends on `sili` from `sili__new` — path or git dependency,
  plus `numpy`; `transformers`/`safetensors`/`torch` stay dev/reference
  -only, never a runtime import of the converted model).
- [ ] README describing the conversion + column-averaging approach and
  how to run the example.
- [ ] Commit and push to `git@github.com:SimLeek/sili_peridot.git`. SSH
  key at `~/.ssh/id_ed25519` is present but not authorized for this repo
  (`Permission denied (publickey)` on test) — push over HTTPS using
  `sili_peridot_access.token` instead (`git push
  https://x-access-token:<token>@github.com/SimLeek/sili_peridot.git
  main`), without persisting the token into `.git/config`.

### Access tokens (`/home/simleek/claude_code/`, gitignored from any repo,
never printed to logs)

- `sili_peridot_access.token` — push access to `sili_peridot` (used
  above, working).
- `claude_sili.token` — fine-grained push access to `sili__new`, for
  Phase A/A6. Confirmed working (fetch succeeded); nothing new on the
  remote beyond local `main` @ `87daa2e` as of this writing.
- `claud_robot_pa_token.txt` — access for `robonet_sili`, for **later**
  (Phase E) — not needed while that repo stays empty/out of scope.

## Phase E — Future scope: embodiment (not started; do not begin before
Phase A–D are solid and validated)

Long-term destination for the converted model, laid out for continuity
so later sessions don't lose the plan, but explicitly **not** part of
the current conversion/training deliverable:

- **End goal**: an `ardu_blimp` robot with a tiny camera/mic/speaker —
  parked for now, since it costs money and needs in-person/IRL testing.
- **Interim testbed**: connect to a `robonet` endpoint running
  Minecraft. Capstone behavior: move around; when "very surprised"
  (large energy/prediction-error spike), route energy into a TTS-like
  output network that produces a sound on the speaker — the model
  "yelps" when startled, choosing whatever noise it wants (not a
  hardcoded scripted sound).
- **All I/O is neuralese (raw neuron activation), never token-based**:
  STT feeds directly into the tiny LLM's neuron space; a CNN over a
  heavily downscaled camera/vision input feeds directly into the LLM;
  TTS output comes directly from the LLM's neurons; plus a state-input/
  action-output network for movement. No tokenization anywhere in this
  loop.
- `robonet_sili`'s existing text-highlighting/reading-via-`xsel` feature
  doesn't apply to Minecraft (no text buffer to select from) — this
  capstone needs a genuinely different, vision/audio/action-based
  interface, not a rehash of the text-focused one.
- **Realistic expectations, stated directly by the user**: a 1B model
  probably isn't practically useful for much *in general*, but with
  vision + movement it's plausibly a good **surprise detector** that can
  navigate a routine path — especially if energy is placed on the
  vision/camera-lightness parameter inputs, so the model "desires"
  varying light/visual complexity for a while (an intrinsic-motivation
  pressure, same family as the curiosity citations above). Concrete
  target behaviors: "yelp" on spotting a threat (e.g. a Minecraft
  creeper), return to base when it needs food/recharge.
- **Explicit blocker carried over from Phase A6**: before any
  energy-management-via-world-modification action pathway ships, it
  needs a per-neuron critic (e.g. a transformer over energy/aux_loss
  context) anchoring action-selection to task outcome — otherwise the
  action pathway is wireheading-prone (it could learn to make itself
  feel less "surprised"/depleted without actually doing anything useful
  in the world). Do not wire an action-output network straight to the
  energy signal without this in place.
- Do not start any of this before Phase A–D produce a converted,
  trained, evaluated MiniCPM5 that demonstrably works as a sparse
  next-input predictor — this phase is here for continuity, not as a
  near-term task.

## Explicitly out of scope / backburner (mirrors sili__new's own
deferred list — do not chase these for v1)

- MoE expert-merge, conv-kernel sparsification, vision per-patch/spatial
  merge — MiniCPM5-1B-Base is dense, text-only, no vision tower.
- RTAC critic-through-trunk / replay buffer — unrelated RL workstream
  (though its critic need reappears, differently scoped, in Phase E).
- `robonet_sili` integration — empty repo today; Phase E depends on it
  eventually, but nothing to do there until Phase A–D land.
- GPU/Kompute device abstraction — no GPU compute path in scope here
  (though note: hardware notes mention the Vega iGPU could in principle
  take a Vulkan compute role later; not this task).
- Adaptive hoyer dense/sparse routing threshold, work-pointer-set
  redesign, `compact()`/`expand_headroom()` auto-handling — pre-existing
  `sili__new` backlog items, only touch if they concretely block this
  conversion.
