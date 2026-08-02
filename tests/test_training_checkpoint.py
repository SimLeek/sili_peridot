import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from model.config import MiniCPM5Config
from model.sili_block import build_step_layers, run_folded_recurrence, _extract_true_csr
from model.curriculum import WindowState, advance_window
from model.training_checkpoint import (
    save_training_checkpoint, load_training_checkpoint,
    sparse_linear_layer_state_dict, sparse_linear_layer_from_state_dict,
)

SUFFIXES = [
    ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
]


def _tiny_config(n_layers=4) -> MiniCPM5Config:
    return MiniCPM5Config(
        hidden_size=8, intermediate_size=12, num_hidden_layers=n_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=10, rms_norm_eps=1e-6, rope_theta=10000.0,
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
        sd[p + ".input_layernorm.weight"]         = {"raw": torch.ones(cfg.hidden_size), "shape": (cfg.hidden_size,)}
        sd[p + ".post_attention_layernorm.weight"] = {"raw": torch.ones(cfg.hidden_size), "shape": (cfg.hidden_size,)}
    return sd


class TestSparseLinearLayerStateDictRoundtrip:
    def test_roundtrip_preserves_weights_and_both_scales(self):
        cfg = _tiny_config(n_layers=1)
        sparse_state = _fake_sparse_state(cfg, seed=90)
        step_layers, _, _ = build_step_layers(sparse_state, cfg, value_scale_mode="rank1")
        layer = step_layers[0][".mlp.gate_proj.weight"]

        d = sparse_linear_layer_state_dict(layer)
        restored = sparse_linear_layer_from_state_dict(d, num_cpus=2)

        ptrs_a, idx_a, vals_a = _extract_true_csr(layer)
        ptrs_b, idx_b, vals_b = _extract_true_csr(restored)
        np.testing.assert_array_equal(ptrs_a, ptrs_b)
        np.testing.assert_array_equal(idx_a, idx_b)
        np.testing.assert_allclose(vals_a, vals_b, atol=1e-6)

        for c in range(layer.n_outputs):
            assert restored.get_output_scale(c) == layer.get_output_scale(c)


class TestTrainingCheckpointRoundtrip:
    def _built(self, n_layers=4, seed=91):
        cfg = _tiny_config(n_layers)
        sparse_state = _fake_sparse_state(cfg, seed=seed)
        step_layers, input_ln, post_ln = build_step_layers(sparse_state, cfg)
        final_norm = np.random.RandomState(seed + 1).randn(cfg.hidden_size).astype(np.float32)
        return cfg, step_layers, input_ln, post_ln, final_norm

    def test_roundtrip_without_window_state(self, tmp_path):
        cfg, step_layers, input_ln, post_ln, final_norm = self._built()
        ckpt_path = tmp_path / "ckpt.pkl"

        save_training_checkpoint(ckpt_path, step_layers, input_ln, post_ln, final_norm,
                                 stage_index=2, quality=0.31)

        assert ckpt_path.exists()
        assert not ckpt_path.with_name(ckpt_path.name + ".tmp").exists()

        (r_step_layers, r_input_ln, r_post_ln, r_final_norm,
         r_window_state, r_stage_index, r_quality) = load_training_checkpoint(ckpt_path)

        assert r_window_state is None
        assert r_stage_index == 2
        assert r_quality == 0.31
        assert len(r_step_layers) == cfg.num_hidden_layers
        for i in range(cfg.num_hidden_layers):
            np.testing.assert_allclose(r_input_ln[i], input_ln[i])
            np.testing.assert_allclose(r_post_ln[i], post_ln[i])
        np.testing.assert_allclose(r_final_norm, final_norm)

        for i in range(cfg.num_hidden_layers):
            for suffix in SUFFIXES:
                orig = step_layers[i][suffix]
                restored = r_step_layers[i][suffix]
                ptrs_a, idx_a, vals_a = _extract_true_csr(orig)
                ptrs_b, idx_b, vals_b = _extract_true_csr(restored)
                np.testing.assert_array_equal(ptrs_a, ptrs_b)
                np.testing.assert_array_equal(idx_a, idx_b)
                np.testing.assert_allclose(vals_a, vals_b, atol=1e-5)

    def test_roundtrip_with_window_state_reproduces_forward_output(self, tmp_path):
        cfg, step_layers, input_ln, post_ln, final_norm = self._built(n_layers=4, seed=92)
        state = WindowState()
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)

        ckpt_path = tmp_path / "ckpt_window.pkl"
        save_training_checkpoint(ckpt_path, step_layers, input_ln, post_ln, final_norm,
                                 window_state=state, stage_index=1, quality=0.4)

        (r_step_layers, r_input_ln, r_post_ln, r_final_norm,
         r_window_state, r_stage_index, r_quality) = load_training_checkpoint(ckpt_path)

        assert r_window_state is not None
        assert r_window_state.window_size == 2
        assert r_window_state.window_positions == [3, 2]

        T = 4
        x = np.random.RandomState(93).randn(T, cfg.hidden_size).astype(np.float32)
        original_out = run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm,
                                             cfg, half_bandwidth=T, window_state=state)
        restored_out = run_folded_recurrence(x, r_step_layers, r_input_ln, r_post_ln, r_final_norm,
                                             cfg, half_bandwidth=T, window_state=r_window_state)

        np.testing.assert_allclose(restored_out, original_out, rtol=1e-4, atol=1e-4)

    def test_save_is_atomic_no_leftover_tmp_file(self, tmp_path):
        cfg, step_layers, input_ln, post_ln, final_norm = self._built(n_layers=1, seed=94)
        ckpt_path = tmp_path / "nested" / "dir" / "ckpt.pkl"

        save_training_checkpoint(ckpt_path, step_layers, input_ln, post_ln, final_norm)

        assert ckpt_path.exists()
        assert list(ckpt_path.parent.glob("*.tmp")) == []
