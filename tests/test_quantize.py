import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest

from model.config import MiniCPM5Config
from model.fold import fold_suffix, SUFFIXES
from model.quantize import (
    FP4_TABLE, FP4_MAX, fp4_round, compute_input_column_scale,
    simulate_fp4_quantize_layer, quantize_suffixes_in_state_dict,
)


def _tiny_config(n_layers=2) -> MiniCPM5Config:
    return MiniCPM5Config(
        hidden_size=8, intermediate_size=12, num_hidden_layers=n_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=10, rms_norm_eps=1e-6, rope_theta=5000000.0,
        tie_word_embeddings=False,
    )


def _fake_sparse_state(cfg: MiniCPM5Config) -> dict:
    torch.manual_seed(3)
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


class TestFp4Round:
    def test_snaps_to_nearest_table_entry(self):
        values = np.array([0.24, 0.3, 5.9, -0.9, -3.4], dtype=np.float32)
        rounded = fp4_round(values)
        for v in rounded.flatten():
            assert float(v) in FP4_TABLE.tolist()

    def test_exact_table_values_are_stable(self):
        values = FP4_TABLE.copy()
        assert np.allclose(fp4_round(values), FP4_TABLE)

    def test_zero_stays_zero(self):
        assert fp4_round(np.array([0.0])) == 0.0


class TestComputeInputColumnScale:
    def test_scale_matches_max_abs_over_fp4_max(self):
        # 2 rows (input features), 3 cols (folded output positions).
        dense = torch.tensor([[1.0, 0.0, -3.0],
                              [0.0, 12.0, 0.0]])
        csr = dense.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        # compute_input_column_scale expects the STACKED [n_folds*out, in]
        # layout (rows = folded output, cols = in_dim) and transposes
        # internally -- transpose our hand-built example to match.
        stacked = csr.to_dense().t().to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        scale = compute_input_column_scale(stacked)
        # compute_input_column_scale transposes `stacked` back internally,
        # so this recovers `dense`'s own per-row max: row0={1,0,-3}
        # max_abs=3 -> scale=3/6=0.5; row1={0,12,0} max_abs=12 -> scale=2.0
        assert np.allclose(scale, [0.5, 2.0])

    def test_all_zero_row_falls_back_to_scale_one(self):
        dense = torch.zeros(2, 2)
        csr = dense.to_sparse(sparse_dim=2).coalesce().to_sparse_csr()
        scale = compute_input_column_scale(csr)
        assert np.allclose(scale, [1.0, 1.0])


class TestSimulateFp4QuantizeLayer:
    def test_zero_entries_stay_exactly_zero(self):
        w = torch.tensor([[1.5, 0.0], [0.0, -2.5]])
        scale = np.array([1.0, 1.0], dtype=np.float32)
        out = simulate_fp4_quantize_layer(w, scale)
        assert out[0, 1] == 0.0
        assert out[1, 0] == 0.0

    def test_nonzero_entries_quantized_to_table_times_scale(self):
        # column scale = 2.0 -> raw value 3.0 / 2.0 = 1.5 -> exact FP4 level
        w = torch.tensor([[3.0]])
        scale = np.array([2.0], dtype=np.float32)
        out = simulate_fp4_quantize_layer(w, scale)
        assert float(out[0, 0]) == pytest.approx(3.0)   # 1.5 * 2.0, exact FP4 level

    def test_output_shape_matches_input(self):
        w = torch.randn(5, 3)
        scale = np.ones(3, dtype=np.float32)
        out = simulate_fp4_quantize_layer(w, scale)
        assert out.shape == w.shape


class TestQuantizeSuffixesInStateDict:
    def test_only_suffix_tensors_change(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)
        dense_state = {name: (entry["csr"].to_dense() if "csr" in entry else entry["raw"])
                       for name, entry in sparse_state.items()}
        dense_state["model.norm.weight"] = torch.randn(cfg.hidden_size)

        descriptors = {
            suffix: fold_suffix(sparse_state, suffix, cfg, prefix="model.layers.")
            for suffix in SUFFIXES
        }
        quantized = quantize_suffixes_in_state_dict(dense_state, descriptors, cfg)

        assert torch.equal(quantized["model.norm.weight"], dense_state["model.norm.weight"])
        # at least one suffix tensor should differ after quantization
        # (random weights essentially never land exactly on FP4 levels)
        changed = any(
            not torch.equal(quantized[f"model.layers.{i}{suffix}"], dense_state[f"model.layers.{i}{suffix}"])
            for suffix in SUFFIXES for i in range(cfg.num_hidden_layers)
        )
        assert changed

    def test_returns_a_copy_not_mutating_input(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)
        dense_state = {name: (entry["csr"].to_dense() if "csr" in entry else entry["raw"])
                       for name, entry in sparse_state.items()}
        original = {k: v.clone() for k, v in dense_state.items()}

        descriptors = {
            suffix: fold_suffix(sparse_state, suffix, cfg, prefix="model.layers.")
            for suffix in SUFFIXES
        }
        quantize_suffixes_in_state_dict(dense_state, descriptors, cfg)

        for name in dense_state:
            assert torch.equal(dense_state[name], original[name])
