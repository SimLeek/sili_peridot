import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest

from model.config import MiniCPM5Config
from model.sili_block import build_step_layers, _extract_true_csr
from model.curriculum import CurriculumStage, build_stage_list, WindowState, advance_window


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


SUFFIXES = [
    ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
]


class TestBuildStageList:
    def test_stage_sizes_grow_by_one_and_end_at_full_depth(self):
        stages = build_stage_list(4)
        assert [s.window_size for s in stages] == [1, 2, 3, 4]
        assert [s.index for s in stages] == [0, 1, 2, 3]

    def test_single_layer_model(self):
        stages = build_stage_list(1)
        assert stages == [CurriculumStage(index=0, window_size=1)]


class TestAdvanceWindow:
    def _step_layers(self, n_layers=4, seed=11):
        cfg = _tiny_config(n_layers)
        sparse_state = _fake_sparse_state(cfg, seed=seed)
        step_layers, _, _ = build_step_layers(sparse_state, cfg)
        return cfg, step_layers

    def test_first_advance_adds_the_last_fold_position(self):
        cfg, step_layers = self._step_layers()
        state = WindowState()
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)

        assert state.window_size == 1
        assert state.window_positions == [cfg.num_hidden_layers - 1]
        assert set(state.suffix_windows.keys()) == set(SUFFIXES)
        for suffix in SUFFIXES:
            assert state.suffix_windows[suffix].n_inputs == step_layers[-1][suffix].n_inputs
            assert state.suffix_windows[suffix].n_outputs == step_layers[-1][suffix].n_outputs

    def test_successive_advances_grow_backward_through_positions(self):
        cfg, step_layers = self._step_layers()
        state = WindowState()
        for _ in range(3):
            state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)

        assert state.window_size == 3
        assert state.window_positions == [3, 2, 1]  # last, then backward
        suffix = ".mlp.gate_proj.weight"
        in_dim = step_layers[0][suffix].n_inputs
        out_dim = step_layers[0][suffix].n_outputs
        assert state.suffix_windows[suffix].n_inputs == 3 * in_dim
        assert state.suffix_windows[suffix].n_outputs == 3 * out_dim

    def test_advancing_past_the_first_position_raises(self):
        cfg, step_layers = self._step_layers(n_layers=1)
        state = WindowState()
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        with pytest.raises(ValueError, match="no earlier position"):
            advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)

    def test_growing_further_preserves_earlier_diagonal_blocks(self):
        # advance_window must compose with grow_window_layer's own
        # "old rows reused verbatim" guarantee (see test_sili_block.py's
        # TestGrowWindowLayer) across a full curriculum walk, not just a
        # single grow_window_layer call in isolation.
        cfg, step_layers = self._step_layers()
        suffix = ".mlp.gate_proj.weight"
        in_dim = step_layers[0][suffix].n_inputs
        out_dim = step_layers[0][suffix].n_outputs

        state = WindowState()
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        layer_after_1 = state.suffix_windows[suffix]
        ptrs1, idx1, vals1 = _extract_true_csr(layer_after_1)

        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        layer_after_3 = state.suffix_windows[suffix]
        ptrs3, idx3, vals3 = _extract_true_csr(layer_after_3)

        def dense_block(ptrs, idx, vals, row_lo, row_hi, col_lo, col_hi):
            out = np.zeros((row_hi - row_lo, col_hi - col_lo), dtype=np.float32)
            for r in range(row_lo, row_hi):
                s, e = int(ptrs[r]), int(ptrs[r + 1])
                cols, cvals = idx[s:e], vals[s:e]
                mask = (cols >= col_lo) & (cols < col_hi)
                out[r - row_lo, cols[mask] - col_lo] = cvals[mask]
            return out

        np.testing.assert_allclose(
            dense_block(ptrs1, idx1, vals1, 0, in_dim, 0, out_dim),
            dense_block(ptrs3, idx3, vals3, 0, in_dim, 0, out_dim),
            atol=1e-6)
