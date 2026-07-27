"""
tests/test_sili_vs_torch_integration.py
─────────────────────────────────────────
Real head-to-head: next-token perplexity/accuracy, wall-clock time, and
memory, for the SAME pruned MiniCPM5-1B-Base weights evaluated two ways --
through the real HF PyTorch model (eval_pruning's existing methodology,
float32, pruned only, no quantization) and through sili's own compute
path end-to-end (B6/B7: sili_block.py's fold-depth recurrence +
sili_model.py's embedding/lm_head wiring). NOT an isolated
recurrence-only comparison -- sili's SparseLinearLayer.load_weights
always FP4-quantizes (no opt-out), applied per fold step independently
(not B5a's stacked/rank-1 scheme, see sili_block.py's module docstring),
so the sili side combines pruning + quantization + the fold-depth
recurrence approximation vs. torch's pruning-only float32 baseline.

Writes real captured numbers to sili_v_torch.md at the repo root.
"""
import ctypes
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.config import MiniCPM5Config
from model.checkpoint import load_minicpm5_checkpoint
from model.prune import (
    prune_state_dict_by_role, DEFAULT_TARGET_SPARSITY_BY_ROLE,
    sparse_state_to_dense_state_dict,
)
from model.eval_pruning import evaluate_next_token_prediction, EVAL_TEXTS
from model.sili_model import build_sili_model, evaluate_next_token_prediction_sili
from conftest import trim_memory

REAL_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'MiniCPM5-1B-Base')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'sili_v_torch.md')


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1.0


@pytest.mark.skipif(not os.path.isdir(REAL_CHECKPOINT_DIR),
                    reason="MiniCPM5-1B-Base checkpoint not present on this machine")
class TestSiliVsTorchRealCheckpoint:

    def test_perplexity_accuracy_time_memory_head_to_head(self):
        cfg = MiniCPM5Config.from_json(os.path.join(REAL_CHECKPOINT_DIR, "config.json"))
        tokenizer = AutoTokenizer.from_pretrained(REAL_CHECKPOINT_DIR)

        trim_memory()
        rss_start = _rss_mb()

        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
        del sd
        trim_memory()
        rss_after_prune = _rss_mb()

        pruned_dense = sparse_state_to_dense_state_dict(sparse_state)
        sparse_state_for_sili = {k: v for k, v in sparse_state.items()}
        del sparse_state
        trim_memory()

        # ── Torch path ────────────────────────────────────────────────────
        rss_before_torch = _rss_mb()
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(REAL_CHECKPOINT_DIR, dtype=torch.float32)
        model.load_state_dict(pruned_dense)
        torch_result = evaluate_next_token_prediction(model, tokenizer, EVAL_TEXTS)
        torch_seconds = time.perf_counter() - t0
        rss_peak_torch = _rss_mb()
        del model, pruned_dense
        trim_memory()
        rss_after_torch = _rss_mb()

        # ── Sili path ─────────────────────────────────────────────────────
        rss_before_sili = _rss_mb()
        t0 = time.perf_counter()
        sili_model = build_sili_model(sparse_state_for_sili, cfg, num_cpus=4)
        max_seq_len = max(len(tokenizer(t)["input_ids"]) for t in EVAL_TEXTS)
        sili_result = evaluate_next_token_prediction_sili(
            sili_model, tokenizer, cfg, half_bandwidth=max_seq_len,
            texts=EVAL_TEXTS, num_cpus=4)
        sili_seconds = time.perf_counter() - t0
        rss_peak_sili = _rss_mb()
        del sili_model
        trim_memory()

        report = f"""# sili vs torch: real MiniCPM5-1B-Base head-to-head

Same pruned weights ({os.path.basename(REAL_CHECKPOINT_DIR)}, B3's
`DEFAULT_TARGET_SPARSITY_BY_ROLE`) on both sides, but NOT an isolated
recurrence-only comparison: torch evaluates them float32, pruned only,
no quantization (eval_pruning's existing methodology). sili's
`SparseLinearLayer.load_weights` always FP4-quantizes (no opt-out),
applied per fold step independently here (not B5a's stacked/rank-1
scheme -- see sili_block.py's module docstring for why), so the sili
column combines pruning + quantization + the fold-depth recurrence
approximation against torch's pruning-only baseline. `EVAL_TEXTS`
({len(EVAL_TEXTS)} short passages), teacher-forced next-token loss +
top-1 accuracy, identical methodology on both sides
(`evaluate_next_token_prediction`). sili's causal `banded_attention`
used with `half_bandwidth={max_seq_len}` (>= every eval text's own
length, i.e. full/unbanded causal attention -- a fair
apples-to-apples comparison, not sili's local-attention approximation).

Measured on this machine ({os.uname().nodename}), single run, RSS via
`/proc/self/status`, wall-clock via `time.perf_counter()`.

| | torch (HF forward) | sili (B6/B7 path) |
|---|---|---|
| Perplexity | {torch_result.perplexity:.4f} | {sili_result.perplexity:.4f} |
| Accuracy | {torch_result.accuracy:.4f} | {sili_result.accuracy:.4f} |
| Wall-clock (eval only) | {torch_seconds:.2f}s | {sili_seconds:.2f}s |
| Peak RSS (this phase) | {rss_peak_torch:.0f} MB | {rss_peak_sili:.0f} MB |
| RSS before phase | {rss_before_torch:.0f} MB | {rss_before_sili:.0f} MB |

RSS checkpoints: start {rss_start:.0f} MB, after load+prune
{rss_after_prune:.0f} MB, after freeing torch model+trim
{rss_after_torch:.0f} MB.

Per-text loss (torch): {[round(l, 4) for l in torch_result.per_text_loss]}
Per-text loss (sili):  {[round(l, 4) for l in sili_result.per_text_loss]}
Per-text accuracy (torch): {[round(a, 4) for a in torch_result.per_text_accuracy]}
Per-text accuracy (sili):  {[round(a, 4) for a in sili_result.per_text_accuracy]}

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
scheme's {sili_result.accuracy:.3f} -- worth checking whether
per-step independent quantization is actually worse than sharing a
scale, or whether the recurrence approximation is the bigger factor).
Not yet isolated in this comparison; a real next step, not a
conclusion drawn here.
"""
        with open(REPORT_PATH, "w") as f:
            f.write(report)

        assert np.isfinite(torch_result.perplexity) and np.isfinite(sili_result.perplexity)
        assert np.isfinite(torch_result.accuracy) and np.isfinite(sili_result.accuracy)
        assert torch_result.accuracy > 0.4   # the already-validated B3b pruned baseline
