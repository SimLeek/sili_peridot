import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from model.checkpoint import load_minicpm5_checkpoint
from model.config import MiniCPM5Config
from model.prune import prune_state_dict, DEFAULT_TARGET_SPARSITY

REAL_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'MiniCPM5-1B-Base')


class TestPruneStateDictWithExplicitThreshold:
    """min_abs_param bypasses calibration entirely -- deterministic
    threshold lets these pin down exact dense/sparse decisions."""

    def _sd(self):
        return {
            "norm.weight": torch.full((8,), 100.0),   # 1-D -- always dense
            # All values 100.0: nothing gets pruned regardless of
            # threshold -- 0% sparsity, well under min_sparsity -> dense.
            "big_zero": torch.full((10, 10), 100.0),
            # 95 small (0.0001) + 5 large (100.0) values: a threshold
            # between them prunes exactly the 95 small ones -> 95%
            # sparsity, comfortably above min_sparsity and cheap enough
            # as CSR -> sparse.
            "mostly_pruned": torch.cat([
                torch.full((95,), 0.0001), torch.full((5,), 100.0)
            ]).reshape(10, 10),
        }

    def test_vector_always_dense(self):
        sd = self._sd()
        _, report = prune_state_dict(sd, min_abs_param=1.0)
        by_name = {t.name: t for t in report.tensors}
        assert by_name["norm.weight"].stored_format == "dense"
        assert by_name["norm.weight"].dense_reason == "scalar/vector"

    def test_low_sparsity_tensor_stays_dense(self):
        sd = self._sd()
        _, report = prune_state_dict(sd, min_abs_param=1.0)
        by_name = {t.name: t for t in report.tensors}
        t = by_name["big_zero"]
        assert t.sparsity == pytest.approx(0.0)
        assert t.stored_format == "dense"
        assert "sparsity" in t.dense_reason

    def test_high_sparsity_tensor_goes_sparse(self):
        sd = self._sd()
        sparse_state, report = prune_state_dict(sd, min_abs_param=1.0)
        by_name = {t.name: t for t in report.tensors}
        t = by_name["mostly_pruned"]
        assert t.sparsity == pytest.approx(0.95)
        assert t.stored_format == "sparse"
        assert "csr" in sparse_state["mostly_pruned"]

    def test_sparse_entry_preserves_original_shape(self):
        sd = self._sd()
        sparse_state, _ = prune_state_dict(sd, min_abs_param=1.0)
        assert sparse_state["mostly_pruned"]["shape"] == torch.Size([10, 10])

    def test_report_totals_and_compression(self):
        sd = self._sd()
        _, report = prune_state_dict(sd, min_abs_param=1.0)
        assert report.total_elements == 8 + 100 + 100
        assert report.n_sparse() == 1
        assert report.n_dense() == 2
        assert report.overall_sparsity > 0.0
        assert report.compression_ratio >= 1.0

    def test_min_abs_param_overrides_target_sparsity(self):
        sd = self._sd()
        # target_sparsity is nonsense (would calibrate to something
        # totally different) but min_abs_param must win regardless.
        _, report = prune_state_dict(sd, target_sparsity=0.01, min_abs_param=1.0)
        assert report.min_abs_param == 1.0


class TestPruneStateDictCalibrated:
    """No explicit threshold -- exercises the calibrate_min_abs_param path."""

    def test_calibration_hits_roughly_the_target_sparsity(self):
        torch.manual_seed(0)
        sd = {"w": torch.randn(200, 200)}
        _, report = prune_state_dict(sd, target_sparsity=0.6)
        assert report.overall_sparsity == pytest.approx(0.6, abs=0.02)


@pytest.mark.skipif(not os.path.isdir(REAL_CHECKPOINT_DIR),
                    reason="MiniCPM5-1B-Base checkpoint not present on this machine")
class TestRealCheckpointPruning:
    """B3a: verify ACTUAL density on the real checkpoint, not just that
    pruning ran without crashing. Also locks in the finding that drove
    DEFAULT_TARGET_SPARSITY away from sparse_prune's own 0.5 default --
    see model/prune.py's module docstring.

    Module-scoped fixtures: loading + pruning the real ~1B-param
    checkpoint isn't free (~10s each), and most of these tests want the
    SAME (state_dict, DEFAULT_TARGET_SPARSITY) report -- computing it
    once per test file instead of once per test cuts this file's runtime
    substantially.
    """

    @pytest.fixture(scope="module")
    def real_state_dict(self):
        return load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)

    @pytest.fixture(scope="module")
    def default_report(self, real_state_dict):
        _, report = prune_state_dict(real_state_dict, target_sparsity=DEFAULT_TARGET_SPARSITY)
        return report

    def test_default_target_sparsity_gets_real_compression(self, default_report):
        # Toy-Mistral's own default (0.5) gave ~1.00x here -- confirm the
        # MiniCPM5-calibrated default does meaningfully better.
        assert default_report.compression_ratio > 1.5
        assert default_report.n_sparse() > default_report.n_dense()

    def test_toy_default_target_sparsity_barely_compresses(self, real_state_dict):
        # Documents the actual finding, doesn't just assert it in a
        # docstring: sparse_prune's own default target_sparsity=0.5
        # leaves almost the whole model dense on this checkpoint. Needs
        # its own (uncached) run since the threshold differs.
        _, report = prune_state_dict(real_state_dict, target_sparsity=0.5)
        assert report.compression_ratio < 1.1

    def test_embed_tokens_and_lm_head_are_not_left_out(self, default_report):
        by_name = {t.name: t for t in default_report.tensors}
        for name in ("model.embed_tokens.weight", "lm_head.weight"):
            assert by_name[name].stored_format == "sparse", (
                f"{name} unexpectedly stayed dense at "
                f"target_sparsity={DEFAULT_TARGET_SPARSITY}"
            )

    def test_no_catastrophic_over_pruning(self, default_report):
        # Matches sparsify_model's own CATASTROPHE_THRESHOLD framing --
        # this is a sanity ceiling, not a claim about the "right" sparsity.
        assert default_report.overall_sparsity < 0.95
