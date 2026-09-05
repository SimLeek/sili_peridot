import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

torch = pytest.importorskip("torch")
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.checkpoint import load_minicpm5_checkpoint
from model.eval_pruning import (
    EVAL_TEXTS,
    EVAL_TEXTS_HELDOUT,
    EvalResult,
    compare_dense_vs_pruned,
    evaluate_next_token_prediction,
)
from model.prune import (
    DEFAULT_TARGET_SPARSITY_BY_ROLE,
    prune_state_dict,
    prune_state_dict_by_role,
    sparse_state_to_dense_state_dict,
)

REAL_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "MiniCPM5-1B-Base")


class TestEvalResult:
    def test_perplexity_is_exp_of_mean_loss(self):
        r = EvalResult(per_text_loss=[0.0, 0.0], per_text_accuracy=[1.0, 1.0])
        assert r.perplexity == pytest.approx(1.0)

    def test_accuracy_is_mean_of_per_text_accuracy(self):
        r = EvalResult(per_text_loss=[0.0], per_text_accuracy=[0.25, 0.75])
        assert r.accuracy == pytest.approx(0.5)


@pytest.mark.skipif(
    not os.path.isdir(REAL_CHECKPOINT_DIR), reason="MiniCPM5-1B-Base checkpoint not present on this machine"
)
class TestRealCheckpointEvaluation:
    """The whole point of this module: does a pruning decision actually
    preserve next-token quality, measured directly, not assumed from
    sparsity percentages. Module-scoped fixtures -- loading the real
    ~1B-parameter model is not free."""

    @pytest.fixture(scope="module")
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(REAL_CHECKPOINT_DIR)

    @pytest.fixture(scope="module")
    def model(self):
        return AutoModelForCausalLM.from_pretrained(REAL_CHECKPOINT_DIR, dtype=torch.float32)

    def test_dense_model_is_a_reasonable_base_model(self, model, tokenizer):
        # Sanity floor: a coherent base model should comfortably beat
        # random-guessing perplexity (vocab_size=130560) on plain
        # declarative English -- this is what "should be really good at
        # next-token prediction" (BASE model, no instruction tuning) means
        # in practice.
        r = evaluate_next_token_prediction(model, tokenizer, EVAL_TEXTS)
        assert r.perplexity < 50
        assert r.accuracy > 0.3

    def test_global_threshold_pruning_destroys_the_model(self, model, tokenizer):
        # Documents the actual finding (see model/prune.py's module
        # docstring) rather than just asserting it in prose: a SINGLE
        # global threshold at DEFAULT_TARGET_SPARSITY (0.8) -- needed for
        # real CSR compression -- collapses next-token accuracy to
        # essentially zero with no retraining involved.
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        sparse_state, _ = prune_state_dict(sd, target_sparsity=0.8)
        pruned_dense = sparse_state_to_dense_state_dict(sparse_state)
        result = compare_dense_vs_pruned(model, tokenizer, pruned_dense, EVAL_TEXTS)
        assert result["dense_accuracy"] > 0.3
        assert result["pruned_accuracy"] < 0.05

    def test_per_role_thresholds_preserve_quality(self, model, tokenizer):
        # The actual validated result of the group-sensitivity +
        # iterative-search calibration (sili__new PRs #7/#8, see
        # JOURNAL.md) -- this is the real regression test for those
        # thresholds, not just a sparsity/compression number.
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
        pruned_dense = sparse_state_to_dense_state_dict(sparse_state)
        result = compare_dense_vs_pruned(model, tokenizer, pruned_dense, EVAL_TEXTS)
        assert result["pruned_accuracy"] > 0.45
        # Relative degradation should also be modest, not just above an
        # absolute floor -- guards against the dense baseline itself
        # drifting on some future checkpoint update.
        assert result["pruned_accuracy"] > result["dense_accuracy"] * 0.85

    def test_per_role_thresholds_hold_up_on_independent_text(self, model, tokenizer):
        # Confirms the chosen thresholds aren't overfit to EVAL_TEXTS --
        # a disjoint set never used during the threshold search itself.
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
        pruned_dense = sparse_state_to_dense_state_dict(sparse_state)
        result = compare_dense_vs_pruned(model, tokenizer, pruned_dense, EVAL_TEXTS_HELDOUT)
        assert result["pruned_accuracy"] > 0.45

    def test_compare_dense_vs_pruned_restores_original_weights(self, model, tokenizer):
        original = {k: v.clone() for k, v in model.state_dict().items()}
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
        pruned_dense = sparse_state_to_dense_state_dict(sparse_state)
        compare_dense_vs_pruned(model, tokenizer, pruned_dense, EVAL_TEXTS)
        after = model.state_dict()
        for k, v in original.items():
            assert torch.equal(v, after[k]), f"{k} not restored after comparison"
