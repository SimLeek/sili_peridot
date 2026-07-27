# sili vs torch: real MiniCPM5-1B-Base head-to-head

Same pruned weights (MiniCPM5-1B-Base, B3's
`DEFAULT_TARGET_SPARSITY_BY_ROLE`) on both sides, but NOT an isolated
recurrence-only comparison: torch evaluates them float32, pruned only,
no quantization (eval_pruning's existing methodology). sili's
`SparseLinearLayer.load_weights` always FP4-quantizes (no opt-out),
applied per fold step independently here (not B5a's stacked/rank-1
scheme -- see sili_block.py's module docstring for why), so the sili
column combines pruning + quantization + the fold-depth recurrence
approximation against torch's pruning-only baseline. `EVAL_TEXTS`
(5 short passages), teacher-forced next-token loss +
top-1 accuracy, identical methodology on both sides
(`evaluate_next_token_prediction`). sili's causal `banded_attention`
used with `half_bandwidth=29` (>= every eval text's own
length, i.e. full/unbanded causal attention -- a fair
apples-to-apples comparison, not sili's local-attention approximation).

Measured on this machine (archengineeringpc1), single run, RSS via
`/proc/self/status`, wall-clock via `time.perf_counter()`.

| | torch (HF forward) | sili (B6/B7 path) |
|---|---|---|
| Perplexity | 16.1493 | 173.3711 |
| Accuracy | 0.4825 | 0.2652 |
| Wall-clock (build+eval) | 6.51s | 109.32s (build 68.4s + eval 40.9s) |
| Peak RSS (this phase) | 11421 MB | 5749 MB |
| RSS before phase | 7217 MB | 6636 MB |

RSS checkpoints: start 262 MB, after load+prune
6452 MB, after freeing torch model+trim
6636 MB.

Per-text loss (torch): [2.4744, 2.3127, 3.5558, 1.9847, 3.5818]
Per-text loss (sili):  [4.9213, 6.1931, 5.1339, 4.764, 4.765]
Per-text accuracy (torch): [0.4643, 0.6818, 0.36, 0.56, 0.3462]
Per-text accuracy (sili):  [0.2857, 0.0909, 0.32, 0.36, 0.2692]

## Reading these numbers

sili is meaningfully worse on quality (accuracy well below torch's
pruned-only baseline) and slower, while using less peak RSS. The
recurrence itself is exact, not an approximation (see module
docstring) -- the gap here is pruning (shared) + FP4 weight
quantization (sili only, activations stay float32 on both sides). A
follow-up swapping this run's per-row per-step quantization for a
rank-1 (row+col) per-step scheme found rank-1 accuracy *lower*, not
higher (0.243 vs 0.265) -- the opposite of B5a's finding for the
stacked scheme, where rank-1 helped substantially; this only compares
two FP4 scale-fitting schemes against each other, it doesn't isolate
FP4's own effect from anything else (no working unquantized-weights
opt-out in the SparseLinearLayer path used here -- DISLDOLayerV/
SISLDOLayerV exist as float32-weight variants but their working state
is unverified). See JOURNAL.md for the full investigation, including a
real memory breakdown (model weights vs. Python/library/per-call
overhead) and why two attempts to speed up model-building by reducing
torch CSR call count both made it slower, not faster (reverted).
