# Dual-matrix (disldo + dense-4x4-block) vs plain disldo: real MiniCPM5-1B-Base layers

Forward pass only (no backward/training in this prototype yet). `min_fill_frac=0.1`
(this machine's measured breakeven, see `sili__new/prototypes/sili_ell/BLOCK4_NOTES.md`
-- CPU- and data-distribution-specific, not a universal constant). `num_cpus=8`.
One representative tensor per role (layer 0), plus `embed_tokens`/`lm_head`, from the
real, already-pruned checkpoint (B3's `DEFAULT_TARGET_SPARSITY_BY_ROLE`).

Design: `y = disldo_forward(A, x) + block4_forward(B, x)`. A holds the scattered
remainder (existing production `SparseLinearLayer`, unchanged). B holds `sili__new`'s
new `dense_block4.hpp`: weights are partitioned into 4x4 tiles; a tile with local fill
>= `min_fill_frac` is stored fully dense (16 bytes, FP4 weight+importance, no
per-synapse index at all -- position IS the column); the rest stays in A. Both paths
use per-original-row FP4 value scaling (matching disldo's own default calibration,
`max_abs_in_row / 6.0`) -- see below, this was a real, necessary fix, not optional.

## Two real bugs found and fixed before these numbers were trustworthy

1. **Missing per-row scale (the big one)**. A first pass used no scaling at all, then
   one *global* scale for a whole tensor. Both were wrong on real weights: real
   `gate_proj` magnitudes span 0.0095-0.629 (66x) within one tensor. No scaling
   rounded 99.999% of real synapses to FP4 code 0 ("empty") -- 5.5M real nonzeros
   survived as 66. A global scale calibrated to the tensor max still left the smallest
   weights far below FP4's floor (0.5) even after scaling -- 43% still lost. Per-row
   scaling (this file's current state) recovers **99.9%** of real nonzeros (verified
   directly: 5,506,810 of 5,511,463 on `gate_proj`), matching disldo's own retention.
2. **An orientation mismatch** (`block4_forward`'s internal row=output/col=input
   convention vs. a caller passing disldo's row=input orientation) caused a real
   heap-buffer-overflow, confirmed via AddressSanitizer, not by inspection. Fixed by
   standardizing the dual-layer's Python-facing API on the natural PyTorch
   `[out_features, in_features]` orientation and transposing only for disldo's own
   leftover path internally.

Both are described in more detail in `sili__new`'s commit history for
`prototypes/sili_ell/` and `sili/lib/headers/dense_block4.hpp`.

## Results

| role | shape | density | nnz | disldo ms | dual ms | speedup | %nnz in block4 |
|---|---|---|---|---|---|---|---|
| model.layers.0.mlp.down_proj.weight | 1536x4608 | 0.8959 | 6340977 | 9.395 | 0.703 | 13.37x | 96.0% |
| model.layers.0.mlp.gate_proj.weight | 4608x1536 | 0.7787 | 5511463 | 7.228 | 0.720 | 10.04x | 99.9% |
| model.layers.0.mlp.up_proj.weight | 4608x1536 | 0.7455 | 5276871 | 10.029 | 1.336 | 7.51x | 100.0% |
| model.layers.0.self_attn.k_proj.weight | 256x1536 | 0.7820 | 307497 | 0.359 | 0.051 | 7.09x | 98.5% |
| model.layers.0.self_attn.o_proj.weight | 1536x2048 | 0.8035 | 2527665 | 3.063 | 0.314 | 9.75x | 99.9% |
| model.layers.0.self_attn.q_proj.weight | 2048x1536 | 0.6684 | 2102481 | 2.759 | 0.311 | 8.86x | 100.0% |
| model.layers.0.self_attn.v_proj.weight | 256x1536 | 0.7815 | 307306 | 0.365 | 0.050 | 7.30x | 100.0% |
| lm_head.weight | 130560x1536 | 0.7012 | 140628074 | 231.031 | 27.259 | 8.48x | 99.6% |
| model.embed_tokens.weight | 130560x1536 | 0.2016 | 40427529 | 61.780 | 29.700 | 2.08x | 96.5% |

Speedup scales with density, as expected from the synthetic breakeven work: the
sparsest real tensor here (`embed_tokens`, 20.2% density, closest to the ~10% local
breakeven) shows the smallest win (2.08x); the densest (`down_proj`, 89.6%) shows the
largest (13.37x). `%nnz in block4` (nearly all of it, for every role except the
sparsest) confirms this isn't a marginal effect on these real, already-pruned layers
-- B3's pruning targets are mild enough (5-30% sparsity on most roles, only
`embed_tokens` at 80%) that local block fill sits well above breakeven almost
everywhere.

## Quality: does the dual-matrix design cost anything beyond disldo's own loss?

Direct A/B, same real weights, same 5 random probe vectors each, relative L2 error
against the real dense reference (`W @ x`):

| tensor | disldo rel_err | dual rel_err | ratio |
|---|---|---|---|
| gate_proj | 0.1210 | 0.1104 | 0.912 |
| q_proj | 0.1138 | 0.1077 | 0.947 |
| down_proj | 0.1332 | 0.1320 | 0.991 |
| embed_tokens | 0.1035 | 0.1061 | 1.025 |

All four at parity (0.91-1.03x) -- **the dual-matrix design costs no meaningful
additional quality beyond disldo's own existing FP4 quantization loss** (~10-13%
relative error either way, consistent with this project's already-established
finding that naive round-to-nearest FP4 quantization is "genuinely destructive" at
this model scale, see `sili_v_torch.md`/JOURNAL.md's B5 investigation -- not a new
problem introduced by block4).

This required a third real bug fix: `embed_tokens` initially showed 2.07x WORSE
error in the dual layer than plain disldo. Traced to the leftover/A path inside
`build_dual_layer` never getting per-row value_scale calibration (unlike disldo's
own default) -- for a tensor where a meaningful fraction lands in leftover
(`embed_tokens`: 3.5%), those entries were quantized raw/unscaled, a strictly worse
disldo than disldo itself. Fixed by applying the same per-row `max_abs/6.0`
calibration to the leftover CSR before construction, matching
`model/sili_block.py`'s real convention exactly.

## What this does and doesn't establish

- **Real speed win, on real weights, forward pass only**: yes, established directly.
- **Quality: no meaningful cost vs. disldo's own baseline**: yes, established
  directly, on real weights, after fixing the leftover-calibration bug above.
- **Backward pass / training**: now built (`transpose_block4`, `block4_backward_dx`,
  `block4_weight_update` in `dense_block4.hpp`; exposed as `DualLayer.backward()` in
  the pybind binding). Verified correct (exact agreement with a dense reference once
  per-row FP4 rescaling noise is isolated out) and verified to actually learn
  end-to-end through the real Python API (a student layer fit to a fixed target's
  output dropped loss 68.6% over 1500 steps). **Known real cost, not yet addressed**:
  `backward()` rebuilds the block4 transpose from scratch every call (O(nnz)) --
  fine for batch/occasional training, but a genuinely online (single-token-at-a-
  time) loop, this project's own stated convention elsewhere, would pay this every
  step. An incremental transpose update is the real fix, not built here. This
  benchmark file itself only measured forward -- a real per-role speed number for
  the trained/backward path on the actual checkpoint is a natural next step, not yet
  run.
- **Synaptogenesis/pruning on the dual-matrix structure itself**: not built. The
  earlier synthetic benchmark (`prototypes/sili_ell/bench_synaptogenesis.cpp`)
  measured disldo/banked/packed growth cost in isolation, not this combined design's
  own promotion/demotion between A and B during online growth.
