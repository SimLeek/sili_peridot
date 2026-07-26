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
from model.quantize import (
    compute_suffix_scales_streaming, apply_quantization,
    compute_suffix_rank1_scales_streaming, apply_rank1_quantization,
)
from model.eval_quantization import compare_pruned_vs_quantized
from model.eval_pruning import EVAL_TEXTS, EVAL_TEXTS_HELDOUT

REAL_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'MiniCPM5-1B-Base')

# NOTE on running this file: the two real-checkpoint classes below each
# pass reliably in isolation (`pytest tests/test_eval_quantization.py::
# TestRealCheckpointQuantizationQuality` / `::TestRealCheckpointRank1Quant
# izationQuality`), but running BOTH together in one pytest invocation
# has OOM-killed this 15GB dev machine (confirmed via kernel oom-killer
# log, not inferred -- see JOURNAL.md) -- each class's own module-scoped
# HF model + pruned-dense fixtures don't fully release before the next
# class's fixtures build, on top of everything else already resident.
# This is a real hardware constraint on this machine, not a code bug (the
# module-scoped fixtures are each individually correct and necessary --
# see model/quantize.py's compute_suffix_scales_streaming docstring for
# why they're built in the specific order they are). On memory-
# constrained hardware, run the two classes as separate pytest
# invocations rather than one combined `pytest tests/test_eval_quantization.py`.


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

    HISTORICAL, kept for regression documentation: this is
    from_descriptor's ORIGINAL scheme ("per_row"). It is NOT the
    current default anymore -- see TestRealCheckpointRank1QuantizationQuality
    below for the fix (rank1 scale, sili__new PR #10) and the resulting
    real number this pipeline now ships with.
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


@pytest.mark.skipif(not os.path.isdir(REAL_CHECKPOINT_DIR),
                    reason="MiniCPM5-1B-Base checkpoint not present on this machine")
class TestRealCheckpointRank1QuantizationQuality:
    """
    The fix for TestRealCheckpointQuantizationQuality's catastrophe (see
    that class's docstring and JOURNAL.md): a rank-1 (per-input AND
    per-output) quantization scale instead of per-row-only, matching
    from_descriptor's new default value_scale_mode="rank1" (sili__new
    PR #10). Real, measured result: accuracy 0.482 (pruned baseline) ->
    ~0.297 (up from ~0.094 with the old per-row-only scheme -- perplexity
    ~36x lower). Real progress, not a full recovery -- per user
    direction ("0.297 isn't the best but it also isn't noise"), this is
    the number the pipeline proceeds with; B5a is closed as "addressed",
    not "solved to parity with dense".
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
    def pruned_and_quantized_rank1(self, cfg):
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
        del sd
        pruned_dense = sparse_state_to_dense_state_dict(sparse_state)
        rank1_scales = compute_suffix_rank1_scales_streaming(sparse_state, cfg, prefix="model.layers.")
        quantized_dense = apply_rank1_quantization(pruned_dense, rank1_scales, cfg)
        return pruned_dense, quantized_dense

    def test_rank1_quantization_recovers_most_of_the_loss(
        self, model, tokenizer, pruned_and_quantized_rank1,
    ):
        pruned_dense, quantized_dense = pruned_and_quantized_rank1
        result = compare_pruned_vs_quantized(model, tokenizer, pruned_dense, quantized_dense, EVAL_TEXTS)
        assert result["pruned_accuracy"] > 0.4
        # Real, validated floor -- meaningfully better than per_row's
        # ~0.094-0.12, still short of the 0.482 dense-pruned baseline.
        assert result["quantized_accuracy"] > 0.2

    def test_rank1_quantization_holds_on_independent_text(
        self, model, tokenizer, pruned_and_quantized_rank1,
    ):
        pruned_dense, quantized_dense = pruned_and_quantized_rank1
        result = compare_pruned_vs_quantized(model, tokenizer, pruned_dense, quantized_dense, EVAL_TEXTS_HELDOUT)
        assert result["quantized_accuracy"] > 0.2

    # A direct head-to-head (rank1 vs per_row on the SAME pruned weights,
    # in one test) was tried and dropped -- it duplicates a full
    # load+prune+dense-conversion cycle on top of everything else this
    # file already does, which OOM-killed this machine once (see
    # JOURNAL.md). The two test classes' own asserted bounds already
    # establish the same fact: TestRealCheckpointQuantizationQuality
    # (per_row) asserts quantized_accuracy < 0.2; this class asserts
    # > 0.2 on the identical pruning/eval pipeline, just a different
    # quantization scheme -- together that's the head-to-head, without
    # the redundant extra checkpoint load. test_quantize.py's
    # TestRank1QuantizationRecoversMoreThanPerRow already covers a
    # direct in-process comparison too, on a smaller synthetic case.
