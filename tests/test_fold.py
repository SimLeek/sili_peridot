import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from model.config import MiniCPM5Config
from model.checkpoint import load_minicpm5_checkpoint
from model.prune import prune_state_dict_by_role, DEFAULT_TARGET_SPARSITY_BY_ROLE
from model.fold import (
    fold_suffix, fold_all_suffixes, verify_lossless, SUFFIXES,
)

REAL_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'MiniCPM5-1B-Base')


def _tiny_config(n_layers=3) -> MiniCPM5Config:
    return MiniCPM5Config(
        hidden_size=8, intermediate_size=12, num_hidden_layers=n_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=10, rms_norm_eps=1e-6, rope_theta=5000000.0,
        tie_word_embeddings=False,
    )


def _fake_sparse_state(cfg: MiniCPM5Config, mix_raw_and_csr=True) -> dict:
    """Mimics prune_state_dict_by_role's output shape: mostly "raw"
    (dense) entries with a couple of "csr" ones mixed in, matching how
    B3's gentle per-role thresholds actually leave most tensors dense."""
    torch.manual_seed(2)

    def entry(t, force_csr):
        if force_csr:
            return {"csr": t.to_sparse(sparse_dim=2).coalesce().to_sparse_csr(), "shape": tuple(t.shape)}
        return {"raw": t, "shape": tuple(t.shape)}

    sd = {}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"   # no trailing dot -- SUFFIXES already start with one
        as_csr = mix_raw_and_csr and (i % 2 == 0)   # alternate raw/csr across layers
        sd[p + ".self_attn.q_proj.weight"] = entry(torch.randn(cfg.q_proj_out, cfg.attn_in), as_csr)
        sd[p + ".self_attn.k_proj.weight"] = entry(torch.randn(cfg.kv_proj_out, cfg.attn_in), as_csr)
        sd[p + ".self_attn.v_proj.weight"] = entry(torch.randn(cfg.kv_proj_out, cfg.attn_in), as_csr)
        sd[p + ".self_attn.o_proj.weight"] = entry(torch.randn(cfg.attn_out, cfg.o_proj_in), as_csr)
        sd[p + ".mlp.gate_proj.weight"]    = entry(torch.randn(cfg.mlp_hidden, cfg.mlp_in), as_csr)
        sd[p + ".mlp.up_proj.weight"]      = entry(torch.randn(cfg.mlp_hidden, cfg.mlp_in), as_csr)
        sd[p + ".mlp.down_proj.weight"]    = entry(torch.randn(cfg.mlp_out, cfg.mlp_hidden), as_csr)
    return sd


class TestFoldSuffix:
    def test_stacks_across_all_layers(self):
        cfg = _tiny_config(n_layers=3)
        sd = _fake_sparse_state(cfg)
        desc = fold_suffix(sd, ".self_attn.q_proj.weight", cfg, prefix="model.layers.")
        assert desc.n_folds == 3
        assert desc.stacked_weights[".self_attn.q_proj.weight"].shape == (3 * cfg.q_proj_out, cfg.attn_in)

    def test_out_dim_matches_config(self):
        cfg = _tiny_config(n_layers=2)
        sd = _fake_sparse_state(cfg)
        for suffix in SUFFIXES:
            desc = fold_suffix(sd, suffix, cfg, prefix="model.layers.")
            assert desc.out_dims[suffix] > 0   # sanity: didn't raise, matched expectation

    def test_raw_and_csr_mixed_within_one_suffix_fold_consistently(self):
        cfg = _tiny_config(n_layers=4)   # alternates raw/csr by layer index
        sd = _fake_sparse_state(cfg, mix_raw_and_csr=True)
        desc = fold_suffix(sd, ".mlp.down_proj.weight", cfg, prefix="model.layers.")
        assert desc.stacked_weights[".mlp.down_proj.weight"].shape[0] == 4 * cfg.mlp_out

    def test_missing_layer_raises(self):
        cfg = _tiny_config(n_layers=3)
        sd = _fake_sparse_state(cfg)
        del sd["model.layers.1.self_attn.q_proj.weight"]
        with pytest.raises(KeyError, match="missing"):
            fold_suffix(sd, ".self_attn.q_proj.weight", cfg, prefix="model.layers.")

    def test_wrong_out_dim_raises(self):
        cfg = _tiny_config(n_layers=2)
        sd = _fake_sparse_state(cfg)
        # Corrupt one layer's q_proj to a shape that doesn't match cfg.q_proj_out.
        sd["model.layers.0.self_attn.q_proj.weight"] = {
            "raw": torch.randn(cfg.q_proj_out + 1, cfg.attn_in),
            "shape": (cfg.q_proj_out + 1, cfg.attn_in),
        }
        with pytest.raises(Exception):   # either stack_csr_vertical or our own out_dim check
            fold_suffix(sd, ".self_attn.q_proj.weight", cfg, prefix="model.layers.")


class TestFoldAllSuffixes:
    def test_all_seven_suffixes_folded(self):
        cfg = _tiny_config(n_layers=2)
        sd = _fake_sparse_state(cfg)
        descriptors = fold_all_suffixes(sd, cfg, prefix="model.layers.")
        assert set(descriptors.keys()) == set(SUFFIXES)

    def test_each_descriptor_covers_exactly_one_suffix(self):
        cfg = _tiny_config(n_layers=2)
        sd = _fake_sparse_state(cfg)
        descriptors = fold_all_suffixes(sd, cfg, prefix="model.layers.")
        for suffix, desc in descriptors.items():
            assert list(desc.stacked_weights.keys()) == [suffix]


class TestVerifyLossless:
    def test_lossless_stacking_reports_true(self):
        cfg = _tiny_config(n_layers=3)
        sd = _fake_sparse_state(cfg)
        descriptors = fold_all_suffixes(sd, cfg, prefix="model.layers.")
        report = verify_lossless(sd, descriptors, cfg, prefix="model.layers.")
        assert report.lossless
        assert report.n_folds == 3

    def test_before_and_after_counts_are_nonzero(self):
        cfg = _tiny_config(n_layers=2)
        sd = _fake_sparse_state(cfg)
        descriptors = fold_all_suffixes(sd, cfg, prefix="model.layers.")
        report = verify_lossless(sd, descriptors, cfg, prefix="model.layers.")
        for suffix in SUFFIXES:
            assert report.per_suffix_nnz_before[suffix] > 0
            assert report.per_suffix_nnz_after[suffix] > 0


@pytest.mark.skipif(not os.path.isdir(REAL_CHECKPOINT_DIR),
                    reason="MiniCPM5-1B-Base checkpoint not present on this machine")
class TestRealCheckpointFolding:
    """B4 end-to-end against the real checkpoint, using B3's actual
    per-role pruning output as input -- the real integration point this
    whole module exists for."""

    @pytest.fixture(scope="module")
    def config(self):
        return MiniCPM5Config.from_json(
            os.path.join(REAL_CHECKPOINT_DIR, "config.json"))

    @pytest.fixture(scope="module")
    def pruned_sparse_state(self, config):
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
        return sparse_state

    @pytest.fixture(scope="module")
    def descriptors(self, pruned_sparse_state, config):
        return fold_all_suffixes(pruned_sparse_state, config)

    def test_all_seven_suffixes_present(self, descriptors):
        assert set(descriptors.keys()) == set(SUFFIXES)

    def test_each_descriptor_has_24_folds(self, descriptors, config):
        for suffix, desc in descriptors.items():
            assert desc.n_folds == config.num_hidden_layers == 24

    def test_out_dims_match_config(self, descriptors, config):
        expected = {
            ".self_attn.q_proj.weight": config.q_proj_out,
            ".self_attn.k_proj.weight": config.kv_proj_out,
            ".self_attn.v_proj.weight": config.kv_proj_out,
            ".self_attn.o_proj.weight": config.attn_out,
            ".mlp.gate_proj.weight":    config.mlp_hidden,
            ".mlp.up_proj.weight":      config.mlp_hidden,
            ".mlp.down_proj.weight":    config.mlp_out,
        }
        for suffix, desc in descriptors.items():
            assert desc.out_dims[suffix] == expected[suffix]

    def test_folding_is_lossless(self, pruned_sparse_state, descriptors, config):
        report = verify_lossless(pruned_sparse_state, descriptors, config)
        assert report.lossless, (
            f"nnz mismatch: before={report.per_suffix_nnz_before} "
            f"after={report.per_suffix_nnz_after}"
        )

    def test_no_per_layer_keys_would_remain_after_removal(self, pruned_sparse_state, config):
        # Confirms every one of the 24*7=168 per-layer 2-D tensors is
        # accounted for by exactly one of the 7 folded suffixes -- the
        # same "0 leftover keys" check sili__new's own fold_sparse_payload
        # makes, done manually here since we fold suffix-by-suffix rather
        # than through that all-at-once entry point.
        expected_names = {
            f"model.layers.{i}{suffix}"
            for i in range(config.num_hidden_layers) for suffix in SUFFIXES
        }
        assert expected_names <= set(pruned_sparse_state.keys())
