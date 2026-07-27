import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest

from model.config import MiniCPM5Config
from model.sili_block import (
    build_step_layers, apply_fold_step, run_folded_recurrence,
    rmsnorm, rope_cos_sin, apply_rotary,
)


def _tiny_config(n_layers=3) -> MiniCPM5Config:
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


class TestRMSNormAndRoPEMatchTorchReference:
    def test_rmsnorm_matches_torch(self):
        rng = np.random.RandomState(0)
        x = rng.randn(5, 8).astype(np.float32)
        weight = rng.randn(8).astype(np.float32)
        eps = 1e-6

        out = rmsnorm(x, weight, eps)

        xt = torch.from_numpy(x)
        wt = torch.from_numpy(weight)
        ref = xt * (xt.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()) * wt

        np.testing.assert_allclose(out, ref.numpy(), rtol=1e-4, atol=1e-5)

    def test_rope_matches_torch_reference(self):
        T, head_dim, theta = 6, 4, 10000.0
        cos, sin = rope_cos_sin(T, head_dim, theta)

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(T).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        ref_cos, ref_sin = emb.cos().numpy(), emb.sin().numpy()

        np.testing.assert_allclose(cos, ref_cos, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(sin, ref_sin, rtol=1e-5, atol=1e-6)

    def test_apply_rotary_matches_torch_reference(self):
        rng = np.random.RandomState(1)
        T, head_dim = 5, 4
        x = rng.randn(T, head_dim).astype(np.float32)
        cos, sin = rope_cos_sin(T, head_dim, 10000.0)

        out = apply_rotary(x, cos, sin)

        def rotate_half_t(xt):
            h = xt.shape[-1] // 2
            return torch.cat([-xt[..., h:], xt[..., :h]], dim=-1)
        xt = torch.from_numpy(x)
        cost, sint = torch.from_numpy(cos), torch.from_numpy(sin)
        ref = xt * cost + rotate_half_t(xt) * sint

        np.testing.assert_allclose(out, ref.numpy(), rtol=1e-4, atol=1e-5)


class TestBuildStepLayers:
    def test_shapes_and_keys(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)

        step_layers, input_ln, post_ln = build_step_layers(sparse_state, cfg)

        assert len(step_layers) == 2
        assert len(input_ln) == 2 and len(post_ln) == 2
        for i in range(2):
            assert set(step_layers[i].keys()) == {
                ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
                ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
            }
            q = step_layers[i][".self_attn.q_proj.weight"]
            assert q.n_inputs == cfg.attn_in
            assert q.n_outputs == cfg.q_proj_out
            assert input_ln[i].shape == (cfg.hidden_size,)
            assert post_ln[i].shape == (cfg.hidden_size,)

    def test_mutates_sparse_state_to_empty(self):
        cfg = _tiny_config(n_layers=2)
        sparse_state = _fake_sparse_state(cfg)
        build_step_layers(sparse_state, cfg)
        assert sparse_state == {}


class TestApplyFoldStep:
    def _one_step_layers(self, cfg):
        sparse_state = _fake_sparse_state(cfg, seed=5)
        step_layers, input_ln, post_ln = build_step_layers(sparse_state, cfg)
        return step_layers[0], input_ln[0], post_ln[0]

    def test_output_shape(self):
        cfg = _tiny_config(n_layers=1)
        layers, ln1, ln2 = self._one_step_layers(cfg)
        T = 5
        x = np.random.RandomState(2).randn(T, cfg.hidden_size).astype(np.float32)
        cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)

        out = apply_fold_step(x, layers, ln1, ln2, cfg, cos, sin, half_bandwidth=T)

        assert out.shape == (T, cfg.hidden_size)
        assert np.all(np.isfinite(out))

    def test_causal_output_at_t_does_not_depend_on_future_tokens(self):
        cfg = _tiny_config(n_layers=1)
        layers, ln1, ln2 = self._one_step_layers(cfg)
        T = 6
        x = np.random.RandomState(4).randn(T, cfg.hidden_size).astype(np.float32)
        cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)

        out_before = apply_fold_step(x, layers, ln1, ln2, cfg, cos, sin, half_bandwidth=T)

        t = 2
        x2 = x.copy()
        x2[t + 1:] += 5.0
        out_after = apply_fold_step(x2, layers, ln1, ln2, cfg, cos, sin, half_bandwidth=T)

        np.testing.assert_allclose(out_before[:t + 1], out_after[:t + 1], rtol=1e-5, atol=1e-5)


class TestRunFoldedRecurrence:
    def test_output_shape(self):
        cfg = _tiny_config(n_layers=3)
        sparse_state = _fake_sparse_state(cfg)
        step_layers, input_ln, post_ln = build_step_layers(sparse_state, cfg)
        final_norm = np.ones(cfg.hidden_size, dtype=np.float32)

        T = 5
        x = np.random.RandomState(6).randn(T, cfg.hidden_size).astype(np.float32)
        out = run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm,
                                    cfg, half_bandwidth=T)

        assert out.shape == (T, cfg.hidden_size)
        assert np.all(np.isfinite(out))

    def test_single_fold_step_matches_direct_call(self):
        # state=0 -> apply_fold_step(x+0) -> state=out -> loop ends (n_folds=1)
        # -> rmsnorm(state) is the return. Confirms the recurrence's own
        # bookkeeping (not apply_fold_step itself, already tested above)
        # is exactly this for the single-step case.
        cfg = _tiny_config(n_layers=1)
        sparse_state = _fake_sparse_state(cfg, seed=7)
        step_layers, input_ln, post_ln = build_step_layers(sparse_state, cfg)
        final_norm = np.random.RandomState(8).randn(cfg.hidden_size).astype(np.float32)

        T = 4
        x = np.random.RandomState(9).randn(T, cfg.hidden_size).astype(np.float32)
        cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)

        direct = apply_fold_step(x, step_layers[0], input_ln[0], post_ln[0],
                                 cfg, cos, sin, half_bandwidth=T)
        expected = rmsnorm(direct, final_norm, cfg.rms_norm_eps)

        actual = run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm,
                                       cfg, half_bandwidth=T)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_causal_output_at_t_does_not_depend_on_future_tokens(self):
        # The end-to-end version of TestApplyFoldStep's leakage check --
        # across the full 24-step-style recurrence (3 steps here), not
        # just one step, since a per-step leak could in principle cancel
        # out over one step but not survive being fed through the next.
        cfg = _tiny_config(n_layers=3)
        sparse_state = _fake_sparse_state(cfg, seed=10)
        step_layers, input_ln, post_ln = build_step_layers(sparse_state, cfg)
        final_norm = np.ones(cfg.hidden_size, dtype=np.float32)

        T = 6
        x = np.random.RandomState(11).randn(T, cfg.hidden_size).astype(np.float32)

        out_before = run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm,
                                           cfg, half_bandwidth=T)

        t = 1
        x2 = x.copy()
        x2[t + 1:] += 5.0
        out_after = run_folded_recurrence(x2, step_layers, input_ln, post_ln, final_norm,
                                          cfg, half_bandwidth=T)

        np.testing.assert_allclose(out_before[:t + 1], out_after[:t + 1], rtol=1e-5, atol=1e-5)
