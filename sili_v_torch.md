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
| Perplexity | 16.1493 | 173.3712 |
| Accuracy | 0.4825 | 0.2652 |
| Wall-clock (eval only) | 7.59s | 161.65s |
| Peak RSS (this phase) | 11822 MB | 6836 MB |
| RSS before phase | 7556 MB | 6848 MB |

RSS checkpoints: start 596 MB, after load+prune
6791 MB, after freeing torch model+trim
6848 MB.

Per-text loss (torch): [2.4744, 2.3127, 3.5558, 1.9847, 3.5818]
Per-text loss (sili):  [4.9213, 6.1931, 5.1339, 4.764, 4.765]
Per-text accuracy (torch): [0.4643, 0.6818, 0.36, 0.56, 0.3462]
Per-text accuracy (sili):  [0.2857, 0.0909, 0.32, 0.36, 0.2692]

## Reading these numbers

sili is meaningfully worse on quality (accuracy well below torch's
pruned-only baseline) and far slower (many small Python-level C++ calls
per token per fold step vs. one fused torch forward), while using less
peak RSS for the eval phase itself. Since three things differ from
torch at once (pruning -- shared -- plus FP4 quantization plus the
fold-depth recurrence's own approximation of true sequential layers),
this run alone can't attribute the accuracy gap to any one of them.
B5a already measured quantization's own effect in isolation (stacked/
rank-1 scheme: ~0.297 accuracy vs. this run's per-step-independent
scheme's 0.265 -- worth checking whether per-step independent
quantization is actually worse than sharing a scale, or whether the
recurrence approximation is the bigger factor). Not yet isolated in
this comparison; a real next step, not a conclusion drawn here.
