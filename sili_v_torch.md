# sili vs torch: speed, and what's actually slow

Quality is covered by the overnight training-recovery run (JOURNAL.md,
sili's online training does recover meaningful accuracy under low-drive
energy gating). This document is about speed specifically: is sili
viable, and if it's slow, what concretely is the cost made of. Prior
versions of this file recorded a single build+eval number with neither
side's thread count written down, which made the number impossible to
reproduce or reason about -- this version fixes that, and goes further:
a real environment bug was found and fixed (below), and every real
bottleneck is now traced to a specific line of code or a specific
library choice, not left as an unexplained "sili is slow."

## Verdict

**Sili's per-inference-call cost, after fixing a real environment bug
below, is ~6.5x torch's on this eval corpus (41.75s vs 6.42s,
131 tokens) -- not the "10x, unusable" figure or the "a little
slower" figure floated earlier; both were measuring different, uncontrolled
things.** Of sili's remaining eval-phase cost, ~60% is one specific,
identified scatter-write pattern in `disldo_forward`/`disldo_backward`
(`linear_disldo.hpp`), confirmed (not guessed) to be structurally
unvectorizable by GCC across every real sparse kernel in this codebase
that touches the DeltaCSR/ULEB128 format, and additionally weaker than
usual on this specific CPU generation (AMD Zen+, no AVX-512, weak AVX2
gather). The other major finding, a numpy/scipy BLAS misconfiguration,
was a real, fixable bug, not a hardware or algorithm limit -- it alone
was worth a 35x difference on the exact GEMM shape sili's lm_head
matmul uses.

There is also a separate, one-time **build/quantization cost**
(~93-113s, roughly flat across thread counts) that a real deployment
pays once and amortizes across many inference calls -- previous
versions of this document lumped build+eval into one "wall-clock"
number, which overstates the recurring per-call cost for any usage
pattern that builds once and serves repeatedly.

## Primary head-to-head (current, post-BLAS-fix)

Same pruned MiniCPM5-1B-Base weights on both sides (B3's
`DEFAULT_TARGET_SPARSITY_BY_ROLE`); torch float32 pruned-only, sili
adds FP4 weight quantization + the fold-depth recurrence (exact, not
an approximation -- see sili_block.py's docstring). `EVAL_TEXTS` (5
short passages, 131 tokens total), teacher-forced next-token loss +
top-1 accuracy, identical methodology both sides. Neither side uses
autoregressive generation or KV-caching (see "KV-caching" below for
why that's a non-issue for *this* comparison specifically). Measured
on this machine (archengineeringpc1, AMD Ryzen 7 3750H, 4C/8T),
single run each, wall-clock via `time.perf_counter()`.

| | torch (HF forward, 4 threads = its own default) | sili (B6/B7 path, 8 threads) |
|---|---|---|
| Perplexity | 16.1493 | 173.3711 |
| Accuracy | 0.4825 | 0.2652 |
| Load / Build (one-time) | 4.76s | 92.86s |
| Eval (recurring, per call) | 6.42s (48.98 ms/token) | 41.75s (318.72 ms/token) |
| **Eval-only ratio** | 1x | **6.50x** |
| Total (load+eval / build+eval) | 11.18s | 134.61s (12.04x) |

Thread counts are recorded explicitly and matter -- see below. **This
run requires numpy/scipy resolving to their PyPI wheels (bundled
OpenBLAS), not this machine's system packages** -- see "BLAS" section.
Without that fix, sili's eval-only number was 75.56s (1.81x worse) and
total was 185.42s, at the same 8 threads. Quality numbers (perplexity/
accuracy) are unaffected by BLAS or thread count -- bit-for-bit
identical in every run.

Per-text loss (torch): [2.4744, 2.3127, 3.5558, 1.9847, 3.5818]
Per-text loss (sili):  [4.9213, 6.1931, 5.1339, 4.764, 4.765]
Per-text accuracy (torch): [0.4643, 0.6818, 0.36, 0.56, 0.3462]
Per-text accuracy (sili):  [0.2857, 0.0909, 0.32, 0.36, 0.2692]

sili is meaningfully worse on quality (accuracy well below torch's
pruned-only baseline). The recurrence itself is exact, not an
approximation (see sili_block.py's module docstring) -- the gap is
pruning (shared) + FP4 weight quantization (sili only, activations
stay float32 on both sides). A follow-up swapping this run's per-row
per-step quantization for a rank-1 (row+col) per-step scheme found
rank-1 accuracy *lower*, not higher (0.243 vs 0.265) -- the opposite
of B5a's finding for the stacked scheme; see JOURNAL.md for the full
investigation, including a real memory breakdown (model weights vs.
Python/library/per-call overhead).

## BLAS: a real, fixable environment bug, not a hardware limit

Investigating why `sgemm_` (the lm_head logits matmul, `hidden @
lm_head.T`) was ~32-38% of sili's eval-phase samples in py-spy (below)
-- and specifically why that alone seemed to cost more wall-clock than
torch's *entire* forward pass -- found the actual cause: this venv is
`--system-site-packages`, and `numpy`/`scipy` were resolving to Arch's
plain `blas`/`lapack` packages (Netlib reference BLAS 3.12.1 --
unoptimized, no SIMD tuning, no threading; OpenBLAS was not installed
on this machine at all). torch links Intel MKL, a tuned, threaded BLAS,
regardless of numpy's own configuration.

Direct, controlled proof -- the *exact* GEMM shape sili's lm_head
matmul uses (`[26,1536] @ [1536,130560]`, matching this eval corpus's
average text length and MiniCPM5's real hidden/vocab dims), timed in
isolation, same process, same machine:

| BLAS | ms/call |
|---|---|
| numpy + Arch's system reference BLAS | 5176.45 |
| numpy + PyPI OpenBLAS wheel (fix) | 147.33 |
| torch + Intel MKL, 4 threads | 359.71 |

**35.1x faster after the fix, and now ~2.4x faster than torch's own
MKL** for the identical operation -- this was never a hardware or
algorithm ceiling, just a system-vs-wheel BLAS selection bug that a
`--system-site-packages` venv can silently fall into.

**Fixed**: `pip install "numpy<2.5.0" scipy` (both PyPI wheels, not
system packages) inside the project venv -- scoped to the venv only,
no system/pacman changes, no sudo. numpy is capped `<2.5.0` because the
currently-installed scipy build (1.18.0, also now a PyPI OpenBLAS
wheel) requires it; verified zero ABI/compatibility warnings and a
clean full-stack import (`numpy`, `scipy`, `sili._cpu`,
`sili_peridot.model.sili_model`) after the fix. **Pinned as a project
requirement**: `sili_peridot/requirements.txt` (new file) -- this is
now a real, load-bearing dependency, not an implementation detail,
since silently falling back to system numpy inside a
`--system-site-packages` venv reproduces the 35x regression with no
error, only a slowdown.

Re-running the full sili build+eval after the fix (table above)
confirms the effect carries through: eval-phase time at num_cpus=8
dropped from 75.56s to 41.75s (1.81x). It didn't drop the full 35x
because BLAS is only one part of eval's cost (~32-38%, per py-spy
below) and doesn't touch build's cost at all (build time is
unchanged, ~93-113s regardless of BLAS -- build's cost is FP4
quantization/CSR construction, not a GEMM).

## Thread counts: recorded explicitly, and non-obvious

**Every historical number in this file's earlier versions had sili's
thread count pinned in code but never recorded in the document, and
torch's thread count was *never* pinned in code at all** (confirmed by
reading every commit that touched this file) -- it always ran at
whatever numpy/torch's ambient default was, unrecorded. That default,
measured directly on this machine: **torch defaults to 4 threads**
(`torch.get_num_threads()`, matches physical core count; this CPU is
4C/8T), separate from its own 8-thread interop pool.

**torch is slower with more threads on this eval corpus, not faster**
-- measured 3 times, consistently:

| torch threads | eval time (131 tokens) |
|---|---|
| 4 (its own default) | 5.81-6.42s |
| 8 | 8.39-10.18s |

Thread-pool synchronization overhead dominates at this small a
workload; "give torch 8 threads too, for fairness" would not actually
produce a fairer comparison, it would just make torch's own number
worse for reasons unrelated to sili at all.

**sili scales with threads, but sub-linearly past the physical core
count** -- consistent with this being a 4-core/8-thread chip (the
second 4 threads are SMT, not independent execution units):

| sili num_cpus | build (s) | eval (s), pre-BLAS-fix | eval ms/token |
|---|---|---|---|
| 1 | 112.95 | 227.70 | 1738.13 |
| 4 | 107.36 | 93.83 | 716.23 |
| 8 | 109.86 | 75.56 | 576.79 |

Eval: 1->4 threads gives 2.43x (of a possible 4x -- ~61% efficiency);
4->8 gives only a further 1.24x (of a possible 2x) -- exactly the
signature of hitting SMT rather than real physical cores. **Build time
is essentially flat across all three (~93-113s, no significant
trend)** -- the build/quantization phase isn't meaningfully
parallelized by the current `num_cpus` mechanism at all, a separate,
unaddressed cost from eval's. Not chased further here (build-phase
parallelization is a distinct investigation from the eval-phase work
below), but worth flagging as real headroom if build cost matters for
a given deployment (e.g. hot-reloading models).

## What's actually slow inside sili (py-spy + GCC vectorization report)

Fresh py-spy native sample (100Hz, 20s, 1893 samples, current code
including stochastic rounding / `damp_by_importance`, taken on the
live eval process) -- not cProfile: this codebase already established
cProfile badly undercounts wall-clock once OpenMP or torch's own
threading is involved (0.535 CPU-seconds reported for a call that took
70s real, an earlier investigation found), so every timing claim in
this file uses either raw `time.perf_counter()` wall-clock or a
sampling profiler, never cProfile.

| self-time | location |
|---|---|
| 38.1% | `linear_disldo.hpp:121` -- `mo[b*n_out+col] += contrib`, the scatter-write itself |
| 7.2% + 6.8% + 5.0% + 1.7% + 1.2% (~22%) | neighboring lines in the same loop (106-122) |
| 31.9% | `sgemm_` (BLAS -- lm_head matmul; this sample predates the BLAS fix above, so its *share* will be proportionally larger now that sgemm_'s absolute cost dropped ~35x while disldo_forward's didn't) |
| 0.2% | `uleb128_decode` |

This matches an earlier py-spy investigation in JOURNAL.md closely
(55%/34%/0.33% then vs. ~60%/32%/0.2% now) -- stable across the
stochastic-rounding change, not a fluke of one sample.

**GCC's own `-fopt-info-vec-missed`/`-optimized` compiler diagnostics**
(not inferred, actually compiled and read) confirm *why*, across every
real sparse per-synapse kernel that touches the DeltaCSR/ULEB128
format in this codebase -- not just `linear_disldo.hpp`:

- `delta_csr_types.hpp:232` (`uleb128_decode`'s byte-at-a-time decode
  loop, `do { byte = buf[pos++]; ... } while (byte & 0x80u)`) --
  **"not vectorized: number of iterations cannot be computed"**, at
  every one of its ~15 template instantiations in this build, zero
  successful vectorizations anywhere. Inherent to the format: each
  byte's meaning and the next byte's offset both depend on decoding
  the current byte first. Matches this session's own earlier ULEB128
  SIMD investigation (`sili__new/prototypes/for_delta_encoding/`,
  fuller notes on the still-open PR #18 branch
  `docs/for-encoding-and-disldo-simd-notes`): a genuinely different
  fixed-width group-varint re-encoding got 1.6-2.5x on the decode step
  alone, but decode is only ~0.2-0.33% of eval-phase time (confirmed
  above), so even a full fix there caps overall improvement under 1%
  -- correctly deprioritized already.
- `linear_disldo.hpp:104`/`117`/`118` (forward) and `:338`/`:352`/`:354`
  (backward) -- **"loop nest containing two or more consecutive inner
  loops cannot be vectorized"**, **"no vectype for stmt"** (the
  strided, per-batch-element input/gradient load), **"unsupported
  control flow in loop"** (the `if (iv == 0) continue` early-exit).
- `linear_disldo.hpp:121` (forward, `mo[...] += contrib`) and `:365`
  (backward, `mcol[col] += ...`) -- **"complicated access pattern"**,
  GCC's diagnosis for a scatter write to a data-dependent index.
- `delta_csr_ops.hpp:246`/`247` (`delta_csr_forward`, the
  sparse-*input* path used for activation-sparsity, structurally
  different work-balanced iteration but calling the same
  `cursor.advance()`/`advance_to()`) -- same **"loop nest..."** and
  **"could not determine main exit from loop with multiple exits"**
  failures. **This confirms the blocker is inherent to the DeltaCSR
  format's access pattern, not a one-kernel oversight in
  `linear_disldo.hpp` specifically.**

By contrast, **GCC successfully vectorizes this same file's dense,
unit-stride loops** (`linear_disldo.hpp:170`/`187`/`245`/`247`/`402`/
`416` -- the per-thread-buffer reduction, output-scale gradient loop,
dx accumulation) at 16-byte and 8-byte widths -- confirming the
compiler and `-O3 -march=native -ffast-math` flags aren't the limiter;
it's specifically the sparse per-synapse loop's data-dependent
decode + scatter that's blocked, everywhere it appears.

`linear_sisldo.hpp`/`SISLDOLayerV` (a plausible candidate for "the
other sparse version that works with SIMD") produces **zero lines in
either vectorization report** -- checked directly: it's defined in
`cpu_backend.cpp` but never bound via `py::class_`, unreachable from
Python, and (being a template) never instantiated/compiled in this
build at all. It isn't a working, faster alternative currently in use
anywhere -- it's dead code.

**How much worse is AMD specifically at this?** This machine is Zen+
(Ryzen 7 3750H): AVX2 only, no AVX-512. Two concrete architectural
facts matter, not just "GCC couldn't figure it out": (1) there is no
native SIMD scatter instruction at all pre-AVX-512 (VSCATTERDPS/DD),
so even a hand-vectorized version of the `mo[...] += contrib` write
would need to fall back to per-lane scalar stores for the write itself
-- only the gather/compute side could vectorize; (2) AMD's AVX2 gather
(VPGATHERDD/similar) on Zen/Zen+ is well-documented (Agner Fog's
instruction tables, widely reported elsewhere) as substantially
higher-latency than Intel's implementation of the same instructions,
often close to or worse than an equivalent scalar loop -- this is
architectural knowledge, not something re-benchmarked on this exact
chip in this session. Net effect: even setting aside GCC's own
vectorization failure, a genuinely different data layout built
specifically to enable hand-written SIMD gather/scatter here would
likely deliver less benefit on this CPU generation than on a newer or
Intel chip, reinforcing (with a concrete hardware reason, not just
"seems like a lot of work") the call already made earlier this session
to deprioritize this specific optimization.

## KV-caching

Resolved directly, not assumed either way: **for the benchmark
methodology used in this document, KV-caching is a non-issue on both
sides.** Both `evaluate_next_token_prediction` (torch,
`model(**ids, labels=ids)`) and `evaluate_next_token_prediction_sili`
(`compute_logits_sili`, `hidden @ lm_head.T`) compute ONE single-shot
teacher-forced forward pass over the whole known sequence at once --
neither does autoregressive, token-by-token generation, so there is no
second call for a KV-cache to be reused by. HF's `use_cache` config
only matters when a *subsequent* call consumes a prior call's
`past_key_values`; that never happens in this eval methodology on
either side.

Where it WOULD matter: (1) any future benchmark that measures
autoregressive `generate()`-style decoding (not currently measured at
all, on either side); (2) sili's own online-training loop
(`model/train_online.py`) once it moves past the current small,
repeating training corpus -- confirmed via `todolist.md`'s B10, not
started: sili has no genuine incremental KV-cache anywhere today. The
existing `_FrozenCache` memoization in `train_online.py` only helps
*because* training sentences currently repeat across the corpus (it
caches whole-text-keyed results, not per-token incremental state) --
it stops helping once training scales to a large, non-repeating
corpus, at which point a real incremental cache is the actual fix
(scoped, not built, per B10).

## Real online-training throughput (separate from the eval benchmark above)

The overnight training-recovery run (`scripts/overnight_run.log`,
`num_cpus=4`, single-token online training -- forward + backward +
inline FP4 stochastic-rounding update across the last fold step's 7
`SparseLinearLayer`s, no batching, per this project's online-learning
convention) gives a real, already-measured throughput number that was
sitting in a log file and never written into this document:

| variant | steps | wall time | s/step (token) |
|---|---|---|---|
| no_energy | 10256 | 14435s | 1.408 |
| with_energy | 9459 | 14434s | 1.526 |

This includes periodic full held-out-accuracy checkpoints (~every
900s) inside the average, not pure per-step cost in isolation -- and
predates the BLAS fix above. The trainable step's loss/gradient is
computed per-token via the lm_head matmul (same operation the BLAS fix
addressed), so this number likely improves with the fix in place too
-- **not yet re-measured, a real follow-up, not claimed here.**

## What this doesn't answer yet

- Build-phase (~93-113s, flat across `num_cpus`) hasn't been profiled
  the way eval-phase has -- unknown whether it has a similarly
  concrete, fixable hot spot or is inherently serial (Python-side CSR
  conversion, per-row FP4 quantization loop).
- Online-training throughput hasn't been re-measured with the BLAS fix
  in place.
- No autoregressive-generation benchmark exists on either side --
  KV-caching's real-world impact (as opposed to its irrelevance to
  *this* benchmark) is undemonstrated.
- The disldo_forward scatter-write (~60% of eval-phase time,
  structurally blocked, weaker on this CPU generation specifically) is
  the largest remaining, unaddressed cost -- no fix attempted here,
  consistent with the earlier session call that a bespoke SIMD rewrite
  here is a poor investment on this hardware.
