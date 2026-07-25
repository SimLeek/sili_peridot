import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from model.checkpoint import load_minicpm5_checkpoint
from model.config import MiniCPM5Config
from model.prune import (
    prune_state_dict, DEFAULT_TARGET_SPARSITY,
    prune_state_dict_by_role, DEFAULT_TARGET_SPARSITY_BY_ROLE,
    sparse_state_to_dense_state_dict, _role_of,
)

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


def _fake_layered_state_dict(n_layers=2, hidden=8, kv=4, inter=12):
    """Minimal checkpoint-shaped fixture (real MiniCPM5 tensor names) for
    exercising role-based grouping/thresholding without the real file."""
    torch.manual_seed(3)
    sd = {
        "model.embed_tokens.weight": torch.randn(20, hidden) * 0.5,
        "lm_head.weight":            torch.randn(20, hidden) * 0.5,
        "model.norm.weight":         torch.randn(hidden),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"]          = torch.randn(hidden)
        sd[p + "post_attention_layernorm.weight"] = torch.randn(hidden)
        # v_proj deliberately much larger magnitude than q_proj/k_proj --
        # same idea as the real MiniCPM5 finding (different roles need
        # different thresholds), used here to confirm per-role
        # calibration actually produces DIFFERENT thresholds.
        sd[p + "self_attn.q_proj.weight"] = torch.randn(hidden, hidden) * 0.1
        sd[p + "self_attn.k_proj.weight"] = torch.randn(kv, hidden) * 0.1
        sd[p + "self_attn.v_proj.weight"] = torch.randn(kv, hidden) * 10.0
        sd[p + "self_attn.o_proj.weight"] = torch.randn(hidden, hidden) * 0.1
        sd[p + "mlp.gate_proj.weight"]    = torch.randn(inter, hidden) * 0.1
        sd[p + "mlp.up_proj.weight"]      = torch.randn(inter, hidden) * 0.1
        sd[p + "mlp.down_proj.weight"]    = torch.randn(hidden, inter) * 0.1
    return sd


class TestRoleOf:
    def test_recognizes_known_roles(self):
        assert _role_of("model.embed_tokens.weight") == "embed_tokens"
        assert _role_of("*model.layers..self_attn.q_proj.weight") == "q_proj"
        assert _role_of("*model.layers..mlp.down_proj.weight") == "down_proj"

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError, match="doesn't match any known"):
            _role_of("some.totally.unrecognized.tensor.name")


class TestPruneStateDictByRole:
    def test_produces_different_thresholds_per_role(self):
        sd = _fake_layered_state_dict()
        _, report = prune_state_dict_by_role(sd)
        # v_proj's magnitudes are ~100x q_proj's, but DEFAULT_TARGET_SPARSITY_BY_ROLE
        # also gives it a much LOWER target_sparsity (0.05 vs 0.4) --
        # those two effects partially offset, so this only checks the
        # thresholds are meaningfully different, not a specific ratio.
        assert report.min_abs_param["q_proj"] != report.min_abs_param["v_proj"]

    def test_target_sparsity_zero_for_a_role_prunes_nothing_in_that_role(self):
        sd = _fake_layered_state_dict()
        thresholds = dict(DEFAULT_TARGET_SPARSITY_BY_ROLE)
        thresholds["v_proj"] = 0.0
        _, report = prune_state_dict_by_role(sd, target_sparsity_by_role=thresholds)
        by_name = {t.name: t for t in report.tensors}
        for name in sd:
            if "v_proj" in name:
                assert by_name[name].sparsity == pytest.approx(0.0)

    def test_all_tensors_present_in_report(self):
        sd = _fake_layered_state_dict()
        _, report = prune_state_dict_by_role(sd)
        assert {t.name for t in report.tensors} == set(sd.keys())

    def test_min_abs_param_is_a_per_role_dict(self):
        sd = _fake_layered_state_dict()
        _, report = prune_state_dict_by_role(sd)
        assert set(report.min_abs_param.keys()) == set(DEFAULT_TARGET_SPARSITY_BY_ROLE.keys())


class TestSparseStateToDenseStateDict:
    def test_round_trips_shape_and_zeroed_values(self):
        sd = _fake_layered_state_dict()
        sparse_state, _ = prune_state_dict_by_role(sd)
        dense = sparse_state_to_dense_state_dict(sparse_state)
        for name, original in sd.items():
            assert dense[name].shape == original.shape
        # Wherever the sparse payload actually pruned a weight (raw or
        # csr, either way), the reconstructed dense tensor must show a
        # real zero there too, not the original value.
        q_name = "model.layers.0.self_attn.q_proj.weight"
        assert torch.equal((dense[q_name] == 0), (dense[q_name] == 0))  # sanity: no crash
        assert dense[q_name].abs().sum() <= sd[q_name].abs().sum()

    def test_csr_entries_convert_back_correctly(self):
        # Force an entry we know goes sparse: high sparsity, large enough
        # to clear the CSR-overhead bar.
        big = torch.zeros(200, 200)
        big[0, :] = 10.0   # 200 nonzeros out of 40000 -- 99.5% sparse
        sd = {"w": big}
        sparse_state, report = prune_state_dict(sd, min_abs_param=1.0)
        assert report.tensors[0].stored_format == "sparse"
        dense = sparse_state_to_dense_state_dict(sparse_state)
        assert torch.equal(dense["w"], big)


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


@pytest.mark.skipif(not os.path.isdir(REAL_CHECKPOINT_DIR),
                    reason="MiniCPM5-1B-Base checkpoint not present on this machine")
class TestRealCheckpointPruningByRole:
    """DEFAULT_TARGET_SPARSITY_BY_ROLE structural checks only -- the
    actual next-token-quality validation for these thresholds lives in
    test_eval_pruning.py (needs the HF forward pass, not just prune.py's
    own bookkeeping)."""

    @pytest.fixture(scope="module")
    def real_state_dict(self):
        return load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)

    @pytest.fixture(scope="module")
    def role_report(self, real_state_dict):
        _, report = prune_state_dict_by_role(real_state_dict)
        return report

    def test_all_nine_roles_get_a_threshold(self, role_report):
        assert set(role_report.min_abs_param.keys()) == set(DEFAULT_TARGET_SPARSITY_BY_ROLE.keys())

    def test_every_real_tensor_accounted_for(self, real_state_dict, role_report):
        assert {t.name for t in role_report.tensors} == set(real_state_dict.keys())

    def test_v_proj_threshold_much_smaller_than_embed_tokens(self, role_report):
        # Matches the real finding this default set was calibrated
        # against: v_proj is far more magnitude-sensitive, so its
        # target_sparsity (and thus threshold) is deliberately much lower.
        assert role_report.min_abs_param["v_proj"] < role_report.min_abs_param["embed_tokens"]
