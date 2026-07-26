import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.config import MiniCPM5Config
from model.checkpoint import load_minicpm5_checkpoint
from model.prune import (
    prune_state_dict_by_role, DEFAULT_TARGET_SPARSITY_BY_ROLE,
    sparse_state_to_dense_state_dict,
)
from model.quantize import compute_suffix_scales_streaming, apply_quantization
from model.eval_quantization import compare_pruned_vs_quantized
from model.eval_pruning import EVAL_TEXTS, EVAL_TEXTS_HELDOUT

REAL_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'MiniCPM5-1B-Base')


@pytest.mark.skipif(not os.path.isdir(REAL_CHECKPOINT_DIR),
                    reason="MiniCPM5-1B-Base checkpoint not present on this machine")
class TestRealCheckpointQuantizationQuality:
    """
    Does B5's real FP4 quantization scheme (one scale per input feature,
    SHARED across all 24 folded layers -- exactly what
    FoldedLayer.from_descriptor actually does) preserve next-token
    quality on top of B3's already-validated pruning?

    REAL FINDING (see JOURNAL.md for the full investigation): NO, not
    even close to "small drop" -- accuracy collapses from 0.482 (pruned
    baseline) to ~0.09-0.12, perplexity jumps ~200x (16 -> ~3300).
    Root-caused, not a bug in the simulation: verified the scale
    computation matches from_descriptor's real per-row formula exactly
    (unit-tested in test_quantize.py), and confirmed the shared scale
    isn't dominated by a rogue outlier layer (per-layer/global max_abs
    ratio: mean=0.71, only 1.1% of (layer,column) pairs below 0.3) --
    the damage is a genuine property of naive round-to-nearest 4-bit
    weight quantization at this granularity (one scale per input
    feature, shared across up to ~110K output positions for
    mlp.gate_proj/up_proj), stacked on top of already-pruned weights.
    Switching to a FINER per-layer-only scale (each layer quantized
    against its own weights, not the shared/stacked scale) helps a
    lot (accuracy 0.094 -> 0.265) but still falls far short of "small
    drop" -- this needs the same kind of per-role sensitivity search
    B3b built for pruning (sili.conversion.prune_sensitivity), not yet
    attempted for quantization. This test locks in the CURRENT
    (unresolved) finding as a real regression test, same pattern as
    test_eval_pruning.py's test_global_threshold_pruning_destroys_the_model
    -- documenting a known-bad result is still valuable so a future fix
    has something concrete to beat.
    """

    @pytest.fixture(scope="module")
    def cfg(self):
        return MiniCPM5Config.from_json(os.path.join(REAL_CHECKPOINT_DIR, "config.json"))

    @pytest.fixture(scope="module")
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(REAL_CHECKPOINT_DIR)

    @pytest.fixture(scope="module")
    def model(self):
        return AutoModelForCausalLM.from_pretrained(REAL_CHECKPOINT_DIR, dtype=torch.float32)

    @pytest.fixture(scope="module")
    def pruned_and_quantized(self, cfg):
        """Builds pruned_dense BEFORE the streaming scale computation on
        purpose (see model/quantize.py's compute_suffix_scales_streaming
        docstring) -- this ordering is itself part of what fixed a real
        OOM this investigation hit (see JOURNAL.md)."""
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
        del sd
        pruned_dense = sparse_state_to_dense_state_dict(sparse_state)
        scales = compute_suffix_scales_streaming(sparse_state, cfg, prefix="model.layers.")
        quantized_dense = apply_quantization(pruned_dense, scales, cfg)
        return pruned_dense, quantized_dense

    def test_shared_scale_quantization_currently_destroys_quality(
        self, model, tokenizer, pruned_and_quantized,
    ):
        pruned_dense, quantized_dense = pruned_and_quantized
        result = compare_pruned_vs_quantized(model, tokenizer, pruned_dense, quantized_dense, EVAL_TEXTS)
        assert result["pruned_accuracy"] > 0.4   # the already-validated B3b baseline
        # Documents the real, current finding -- not a target to defend,
        # a floor this must eventually rise above once quantization
        # granularity/calibration is fixed (see JOURNAL.md).
        assert result["quantized_accuracy"] < 0.2

    def test_shared_scale_quantization_holds_on_independent_text(
        self, model, tokenizer, pruned_and_quantized,
    ):
        pruned_dense, quantized_dense = pruned_and_quantized
        result = compare_pruned_vs_quantized(model, tokenizer, pruned_dense, quantized_dense, EVAL_TEXTS_HELDOUT)
        assert result["quantized_accuracy"] < 0.2   # same collapse, not overfit to EVAL_TEXTS

    def test_compare_pruned_vs_quantized_restores_original_weights(
        self, model, tokenizer, pruned_and_quantized,
    ):
        pruned_dense, quantized_dense = pruned_and_quantized
        original = {k: v.clone() for k, v in model.state_dict().items()}
        compare_pruned_vs_quantized(model, tokenizer, pruned_dense, quantized_dense, EVAL_TEXTS)
        after = model.state_dict()
        for k in original:
            assert torch.equal(original[k], after[k]), f"{k} not restored after comparison"
