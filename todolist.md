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
something as a library.

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
  default, per its own docstring).
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
  conversion-time weights.
- [ ] **A3. Build the "column-averaging" energy mechanism.** Already
  scoped (not built) in `refactoring_todo.md`'s "Energy-as-input and
  column energy" note: partition hidden state into columns of `c`
  neurons mapped to input indices; loss ties `mean(column_i)` to
  `input_i`; gradient `(col_mean - input) * 2/(n*c)` broadcasts to column
  members. For our case, "column" = one neuron per fold-depth step (24
  neurons) per input index `i` in `input_size`, and the target is
  "average of the column over the fold-depth recurrence." Concretely:
  - Add a target-aware loss term alongside `EnergyDynamics`'s existing
    `energy_loss` (in `_apply_energy_dynamics`, `sili/energy.py`) or as a
    sibling module (`ColumnAveragingEnergy`) called from wherever the
    folded-layer recurrence lives (see A4) — needs both `h`/`h_out` *and*
    the original per-token input, which `EnergyDynamics.forward` doesn't
    currently receive at all.
  - **Must reconcile with three existing hard constraints that will
    otherwise fight an averaging objective** (see energy research
    findings): the hard top-p ceiling (`p`, globally competitive, no
    concept of "this neuron is reserved for column i") — give column
    neurons a reserved slot or a per-column top-p; the KL sparsity term
    (targets a global scalar density, neuron-identity-agnostic) — needs
    to not fight column-neuron selection; and the shutoff path (emits an
    energy-derived constant with no gradient to `h`) — must be excluded
    from the column mean or corrupt averages silently. Likely resolution:
    per-column `EnergyDynamics` parameters (already consistent with the
    docs' "parameters must be set per-region" design principle), not one
    global instance.
  - Add unit tests: verify column mean converges toward a target input
    over training steps, verify energy/sparsity budget is still
    respected (active fraction ≤ `p`), verify shutoff-path neurons are
    excluded from the column-mean computation.
- [ ] **A4. Retain per-fold-step hidden state instead of summing
  immediately.** `FoldedLayer.forward` currently reshapes
  `[batch, n_folds, out_dim]` and sums over the fold axis right away
  (`sparse_rnn.py:416`) — the per-fold-step values needed to form
  "columns" are not retained after the forward pass. Add a mode/flag (or
  a small subclass, e.g. `FoldedColumnLayer`) that keeps the pre-sum
  `[batch, n_folds, out_dim]` tensor available for the column-averaging
  loss in A3, while leaving the existing summed behavior as the default
  for callers that don't need it (avoid changing `FoldedLayer`'s current
  contract, since existing tests depend on it).
- [ ] A5 (explicitly lower priority, correctness > perf for v1):
  `FoldedLayer.forward`/`backward` are dense-only even though the
  energy-gated state is mostly zero after the top-p gate —
  `refactoring_todo.md` already flags the sparse fast path
  (`SparseRNNAgent.forward`'s `CSR.from_dense` pattern) as a 10x+ win.
  Not required for a correct v1 conversion; revisit once the model runs
  end-to-end if inference is impractically slow.

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
  over the recurrence (the "next-input predictor" — forming a
  representation that stays anchored to the input manifold rather than
  drifting into private internal dynamics, exactly the failure mode the
  NCD-based `ncd_view_hidden` metric in the Mandelbrot experiment was
  designed to catch). Use the Phase A3 column-averaging energy loss,
  sparse activation throughout (Hoyer-routed dense/sparse dispatch or
  explicit CSR conversion of the energy-gated state), sparse backprop
  (real per-row synaptogenesis: `build_probes(k=4)` once per layer, then
  `synap_step` once per row — not once total, and never scale `k` by row
  count, an already-reverted n²-blowup bug elsewhere in this codebase).

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

## Explicitly out of scope / backburner (mirrors sili__new's own
deferred list — do not chase these for v1)

- MoE expert-merge, conv-kernel sparsification, vision per-patch/spatial
  merge — MiniCPM5-1B-Base is dense, text-only, no vision tower.
- RTAC critic-through-trunk / replay buffer — unrelated RL workstream.
- `robonet_sili` integration — empty repo, no serving/networking need
  for this task; revisit only if/when a served-inference use case shows
  up.
- GPU/Kompute device abstraction — no GPU compute path in scope here
  (though note: hardware notes mention the Vega iGPU could in principle
  take a Vulkan compute role later; not this task).
- `FoldedLayer` sparse forward/backward dispatch (A5) — perf-only,
  correctness-first for v1.
- Adaptive hoyer dense/sparse routing threshold, work-pointer-set
  redesign, `compact()`/`expand_headroom()` auto-handling — pre-existing
  `sili__new` backlog items, only touch if they concretely block this
  conversion.
