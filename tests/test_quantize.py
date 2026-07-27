import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest

from model.config import MiniCPM5Config
from model.fold import SUFFIXES
from model.quantize import build_quantized_dense_state_dict_streaming


def _tiny_config(n_layers=2) -> MiniCPM5Config:
    return MiniCPM5Config(
        hidden_size=8, intermediate_size=12, num_hidden_layers=n_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=10, rms_norm_eps=1e-6, rope_theta=5000000.0,
        tie_word_embeddings=False,
    )


def _fake_sparse_state(cfg: MiniCPM5Config, seed=3) -> dict:
    torch.manual_seed(seed)
    sd = {}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[p + ".self_attn.q_proj.weight"] = {"raw": torch.randn(cfg.q_proj_out, cfg.attn_in), "shape": (cfg.q_proj_out, cfg.attn_in)}
        sd[p + ".self_attn.k_proj.weight"] = {"raw": torch.randn(cfg.kv_proj_out, cfg.attn_in), "shape": (cfg.kv_proj_out, cfg.attn_in)}
        sd[p + ".self_attn.v_proj.weight"] = {"raw": torch.randn(cfg.kv_proj_out, cfg.attn_in), "shape": (cfg.kv_proj_out, cfg.attn_in)}
        sd[p + ".self_attn.o_proj.weight"] = {"raw": torch.randn(cfg.attn_out, cfg.o_proj_in), "shape": (cfg.attn_out, cfg.o_proj_in)}
        sd[p + ".mlp.gate_proj.weight"]    = {"raw": torch.randn(cfg.mlp_hidden, cfg.mlp_in), "shape": (cfg.mlp_hidden, cfg.mlp_in)}
        sd[p + ".mlp.up_proj.weight"]      = {"raw": torch.randn(cfg.mlp_hidden, cfg.mlp_in), "shape": (cfg.mlp_hidden, cfg.mlp_in)}
        sd[p + ".mlp.down_proj.weight"]    = {"raw": torch.randn(cfg.mlp_out, cfg.mlp_hidden), "shape": (cfg.mlp_out, cfg.mlp_hidden)}
    return sd


def _fake_sparse_state_with_output_structure(cfg: MiniCPM5Config) -> dict:
    """Each output row's magnitude scaled by a random per-output factor
    (~2 orders of magnitude) -- mimics the real per-output variance
    within one folded layer that rank1 (not per_row) can represent."""
    torch.manual_seed(3)
    sd = {}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        for suffix, out_dim, in_dim in [
            (".self_attn.q_proj.weight", cfg.q_proj_out, cfg.attn_in),
            (".self_attn.k_proj.weight", cfg.kv_proj_out, cfg.attn_in),
            (".self_attn.v_proj.weight", cfg.kv_proj_out, cfg.attn_in),
            (".self_attn.o_proj.weight", cfg.attn_out, cfg.o_proj_in),
            (".mlp.gate_proj.weight", cfg.mlp_hidden, cfg.mlp_in),
            (".mlp.up_proj.weight", cfg.mlp_hidden, cfg.mlp_in),
            (".mlp.down_proj.weight", cfg.mlp_out, cfg.mlp_hidden),
        ]:
            row_factor = torch.abs(torch.randn(out_dim, 1)) * torch.exp(torch.randn(out_dim, 1))
            w = torch.randn(out_dim, in_dim) * row_factor
            sd[p + suffix] = {"raw": w, "shape": (out_dim, in_dim)}
    return sd


class TestBuildQuantizedDenseStateDictStreaming:
    def test_returns_exactly_the_suffix_layer_tensors(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)
        expected_keys = {f"model.layers.{i}{suffix}" for suffix in SUFFIXES for i in range(2)}

        out = build_quantized_dense_state_dict_streaming(sparse_state, cfg)

        assert set(out.keys()) == expected_keys

    def test_mutates_sparse_state_in_place(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)

        build_quantized_dense_state_dict_streaming(sparse_state, cfg)

        assert sparse_state == {}

    def test_output_shapes_match_original(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)
        original_shapes = {name: entry["shape"] for name, entry in sparse_state.items()}

        out = build_quantized_dense_state_dict_streaming(sparse_state, cfg)

        for name, shape in original_shapes.items():
            assert tuple(out[name].shape) == tuple(shape)

    def test_zero_entries_stay_exactly_zero(self):
        cfg = _tiny_config(n_layers=1)
        sparse_state = _fake_sparse_state(cfg)
        q_name = "model.layers.0.self_attn.q_proj.weight"
        w = sparse_state[q_name]["raw"].clone()
        w[0, 0] = 0.0
        w[1, 1] = 0.0
        sparse_state[q_name]["raw"] = w

        out = build_quantized_dense_state_dict_streaming(sparse_state, cfg)

        assert out[q_name][0, 0] == 0.0
        assert out[q_name][1, 1] == 0.0

    def test_real_quantization_changes_at_least_some_values(self):
        # Random float32 weights essentially never land exactly on one of
        # FP4's 15 representable levels, so real FP4 quantization must
        # change at least some entries.
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)
        original = {name: entry["raw"].clone() for name, entry in sparse_state.items()}

        out = build_quantized_dense_state_dict_streaming(sparse_state, cfg)

        changed = any(
            not torch.equal(out[name], original[name])
            for name in original
        )
        assert changed

    def test_invalid_value_scale_mode_raises(self):
        cfg = _tiny_config(n_layers=1)
        sparse_state = _fake_sparse_state(cfg)
        with pytest.raises(ValueError, match="value_scale_mode"):
            build_quantized_dense_state_dict_streaming(
                sparse_state, cfg, value_scale_mode="bogus")


class TestRank1QuantizationRecoversMoreThanPerRow:
    def test_rank1_reconstruction_error_lower_than_per_row(self):
        cfg = _tiny_config(n_layers=3)
        sparse_state = _fake_sparse_state_with_output_structure(cfg)
        original = {name: entry["raw"].clone() for name, entry in sparse_state.items()}
        sparse_state_per_row = {name: {"raw": entry["raw"].clone(), "shape": entry["shape"]}
                                for name, entry in sparse_state.items()}

        quantized_rank1 = build_quantized_dense_state_dict_streaming(
            sparse_state, cfg, value_scale_mode="rank1")
        quantized_per_row = build_quantized_dense_state_dict_streaming(
            sparse_state_per_row, cfg, value_scale_mode="per_row")

        err_rank1 = sum(
            (quantized_rank1[name] - original[name]).abs().sum().item() for name in original
        )
        err_per_row = sum(
            (quantized_per_row[name] - original[name]).abs().sum().item() for name in original
        )
        assert err_rank1 < err_per_row
