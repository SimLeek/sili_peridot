import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest

from model.config import MiniCPM5Config
from model.sili_block import (
    build_step_layers, apply_fold_step, run_folded_recurrence,
    rmsnorm, rope_cos_sin, apply_rotary, _forward, _density_for_suffix,
    grow_window_layer, _extract_true_csr,
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


class _RecordingLayer:
    """Stub standing in for a real SparseLinearLayer -- records the exact
    CSR handed to forward_sparse so _forward's per-row nnz contract can be
    checked directly. A real layer's own weight-sparsity structure can
    coincidentally make even a wrong (e.g. global-budget-not-per-row) CSR
    produce the same dense output as a correctly-masked reference -- this
    is what let the real bug below hide behind
    test_sparse_path_matches_dense_forward_on_masked_input for a tiny toy
    layer with very few real weight connections per output."""
    def __init__(self, n_outputs):
        self.n_outputs = n_outputs
        self.last_call = None

    def forward_sparse(self, ptrs, idx, vals, batch, learning_rate=0.0):
        self.last_call = dict(ptrs=np.asarray(ptrs), idx=np.asarray(idx), vals=np.asarray(vals))
        return np.zeros((batch, self.n_outputs), dtype=np.float32)


class TestForwardActivationDensityRowBudget:
    def test_every_row_gets_its_own_top_k_not_a_global_budget(self):
        # Regression: _cpu.dense_to_top_k_csr's k is a GLOBAL budget over
        # the whole flattened [rows, cols] array, not per row -- calling
        # it once on multi-row x silently starved most rows of any
        # nonzero entries at all (verified directly: 5x8 random input,
        # k=4 gave nnz-per-row [3,0,0,1,0], not [4]*5), which caused a
        # real-checkpoint accuracy collapse to 0.0 at every density from
        # 0.9 down to 0.005 -- a uniform floor regardless of k, not the
        # smooth degradation genuine information loss would produce.
        # _forward must loop per row so every token gets exactly its own
        # top-k. Check the CSR actually handed to forward_sparse directly
        # (via _RecordingLayer) rather than through a real layer's output,
        # since that can coincidentally hide this exact bug.
        rng = np.random.RandomState(30)
        T, n_features = 10, 16
        x = rng.randn(T, n_features).astype(np.float32)
        density = 0.5
        k = round(density * n_features)

        layer = _RecordingLayer(n_outputs=4)
        _forward(layer, x, activation_density=density)

        nnz_per_row = np.diff(layer.last_call["ptrs"])
        np.testing.assert_array_equal(nnz_per_row, np.full(T, k))


class _RoutingSpyLayer:
    """Stub used only to record whether forward_dense or forward_sparse
    was called for THIS suffix/step -- unlike _RecordingLayer (used
    standalone with _forward for one layer at a time), apply_fold_step/
    run_folded_recurrence's dict/list routing needs per-suffix-per-step
    visibility across all 7 projections (and all fold steps) at once.
    Returns all-zero output of the right shape -- fine here since these
    tests check ROUTING, not numeric correctness (already covered by
    TestForwardActivationDensity)."""
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.dense_calls = 0
        self.sparse_calls = 0

    def forward_dense(self, x, learning_rate=0.0):
        self.dense_calls += 1
        return np.zeros((x.shape[0], self.n_out), dtype=np.float32)

    def forward_sparse(self, ptrs, idx, vals, batch, learning_rate=0.0):
        self.sparse_calls += 1
        return np.zeros((batch, self.n_out), dtype=np.float32)


def _spy_layers(cfg):
    return {
        ".self_attn.q_proj.weight": _RoutingSpyLayer(cfg.attn_in, cfg.q_proj_out),
        ".self_attn.k_proj.weight": _RoutingSpyLayer(cfg.attn_in, cfg.kv_proj_out),
        ".self_attn.v_proj.weight": _RoutingSpyLayer(cfg.attn_in, cfg.kv_proj_out),
        ".self_attn.o_proj.weight": _RoutingSpyLayer(cfg.o_proj_in, cfg.attn_out),
        ".mlp.gate_proj.weight": _RoutingSpyLayer(cfg.mlp_in, cfg.mlp_hidden),
        ".mlp.up_proj.weight": _RoutingSpyLayer(cfg.mlp_in, cfg.mlp_hidden),
        ".mlp.down_proj.weight": _RoutingSpyLayer(cfg.mlp_hidden, cfg.mlp_out),
    }


class TestPerSuffixActivationDensity:
    """_density_for_suffix / apply_fold_step's dict-based activation_density
    (isolate which of q/k/v/o/gate/up/down get sparsified) -- wired in but
    not yet exercised by any test."""

    def test_dict_only_sparsifies_named_suffixes(self):
        cfg = _tiny_config(n_layers=1)
        layers = _spy_layers(cfg)
        T = 5
        x = np.random.RandomState(31).randn(T, cfg.hidden_size).astype(np.float32)
        ln1 = np.ones(cfg.hidden_size, dtype=np.float32)
        ln2 = np.ones(cfg.hidden_size, dtype=np.float32)
        cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)

        apply_fold_step(x, layers, ln1, ln2, cfg, cos, sin, half_bandwidth=T,
                        activation_density={".mlp.gate_proj.weight": 0.5})

        sparsified = layers.pop(".mlp.gate_proj.weight")
        assert sparsified.sparse_calls == 1 and sparsified.dense_calls == 0
        for suffix, layer in layers.items():
            assert layer.dense_calls == 1, f"{suffix} should have stayed dense"
            assert layer.sparse_calls == 0, f"{suffix} should not have been sparsified"

    def test_dict_missing_suffix_defaults_to_dense(self):
        assert _density_for_suffix({}, ".mlp.gate_proj.weight") is None
        assert _density_for_suffix(
            {".mlp.up_proj.weight": 0.1}, ".mlp.gate_proj.weight") is None


class TestPerLayerActivationDensity:
    """run_folded_recurrence's list-based activation_density (isolate
    which FOLD STEPS/layers get sparsified, since errors compound through
    the accumulated recurrence state) -- wired in but not yet exercised by
    any test."""

    def test_list_applies_different_density_per_fold_step(self):
        cfg = _tiny_config(n_layers=3)
        step_layers = [_spy_layers(cfg) for _ in range(3)]
        input_ln = [np.ones(cfg.hidden_size, dtype=np.float32) for _ in range(3)]
        post_ln = [np.ones(cfg.hidden_size, dtype=np.float32) for _ in range(3)]
        final_norm = np.ones(cfg.hidden_size, dtype=np.float32)
        T = 4
        x = np.random.RandomState(32).randn(T, cfg.hidden_size).astype(np.float32)

        activation_density = [None, {".mlp.gate_proj.weight": 0.5}, None]
        run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm, cfg,
                              half_bandwidth=T, activation_density=activation_density)

        for i, layers in enumerate(step_layers):
            for suffix, layer in layers.items():
                if i == 1 and suffix == ".mlp.gate_proj.weight":
                    assert layer.sparse_calls == 1 and layer.dense_calls == 0, \
                        f"step {i} {suffix} should have been sparsified"
                else:
                    assert layer.dense_calls == 1 and layer.sparse_calls == 0, \
                        f"step {i} {suffix} should have stayed dense"

    def test_list_length_mismatch_with_num_hidden_layers(self):
        cfg = _tiny_config(n_layers=3)
        sparse_state = _fake_sparse_state(cfg, seed=33)
        step_layers, input_ln, post_ln = build_step_layers(sparse_state, cfg)
        final_norm = np.ones(cfg.hidden_size, dtype=np.float32)
        T = 4
        x = np.random.RandomState(34).randn(T, cfg.hidden_size).astype(np.float32)

        with pytest.raises(ValueError, match="activation_density"):
            run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm, cfg,
                                  half_bandwidth=T, activation_density=[None, None])


class TestForwardActivationDensity:
    """_forward is the new dense/sparse dispatch used by every projection
    in apply_fold_step -- test it directly against the single layer it
    wraps, rather than only through the full fold step, so a routing bug
    can't hide behind attention/MLP noise."""

    def _one_layer(self, cfg):
        sparse_state = _fake_sparse_state(cfg, seed=5)
        step_layers, _, _ = build_step_layers(sparse_state, cfg)
        return step_layers[0][".self_attn.q_proj.weight"]

    def test_activation_density_none_matches_forward_dense(self):
        cfg = _tiny_config(n_layers=1)
        layer = self._one_layer(cfg)
        x = np.random.RandomState(20).randn(5, cfg.attn_in).astype(np.float32)

        out = _forward(layer, x, activation_density=None)
        expected = layer.forward_dense(x, learning_rate=0.0)

        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_sparse_path_matches_dense_forward_on_masked_input(self):
        # Routing through forward_sparse with only the per-row top-k
        # entries kept should be mathematically identical to zeroing
        # every other entry and running the ordinary dense forward --
        # both sum contributions over exactly the same nonzero synapses
        # against the same underlying weights. Confirms _forward's
        # sparse branch isn't silently dropping/duplicating/misrouting
        # entries, not just that it runs and returns finite numbers.
        cfg = _tiny_config(n_layers=1)
        layer = self._one_layer(cfg)
        rng = np.random.RandomState(21)
        x = rng.randn(6, cfg.attn_in).astype(np.float32)
        density = 0.5
        k = max(1, round(density * cfg.attn_in))

        out = _forward(layer, x, activation_density=density)

        masked = np.zeros_like(x)
        for row in range(x.shape[0]):
            top_idx = np.argsort(-np.abs(x[row]))[:k]
            masked[row, top_idx] = x[row, top_idx]
        expected = layer.forward_dense(masked, learning_rate=0.0)

        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

class TestApplyFoldStepActivationDensity:
    def test_runs_and_shapes_match_dense(self):
        cfg = _tiny_config(n_layers=1)
        layers, ln1, ln2 = TestApplyFoldStep()._one_step_layers(cfg)
        T = 5
        x = np.random.RandomState(23).randn(T, cfg.hidden_size).astype(np.float32)
        cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)

        out = apply_fold_step(x, layers, ln1, ln2, cfg, cos, sin, half_bandwidth=T,
                              activation_density=0.5)

        assert out.shape == (T, cfg.hidden_size)
        assert np.all(np.isfinite(out))


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


def _dense_block(layer, row_lo, row_hi, col_lo, col_hi):
    """Densify one [row_lo:row_hi, col_lo:col_hi] block of `layer`'s true
    weights (see _extract_true_csr) -- used to check a specific
    diagonal/off-diagonal region of a grown window matrix in isolation."""
    ptrs, idx, vals = _extract_true_csr(layer)
    out = np.zeros((row_hi - row_lo, col_hi - col_lo), dtype=np.float32)
    for r in range(row_lo, row_hi):
        s, e = int(ptrs[r]), int(ptrs[r + 1])
        cols, cvals = idx[s:e], vals[s:e]
        mask = (cols >= col_lo) & (cols < col_hi)
        out[r - row_lo, cols[mask] - col_lo] = cvals[mask]
    return out


class TestGrowWindowLayer:
    """grow_window_layer builds/grows the B8a curriculum window's combined
    per-suffix matrix, ONE position at a time, from that position's own
    already-built step_layers[i][suffix] (see module docstring's "don't
    fold before the window needs it") -- no separate desc-based
    construction path. These checks stand in for Phase 2's real-checkpoint
    regression test (Phase 4/plan verification): with the recurrent band
    still all-zero (nothing trained yet), the window's combined forward
    pass must reproduce running each position's own layer independently."""

    def _grown_two_position_window(self, suffix=".mlp.gate_proj.weight"):
        cfg = _tiny_config(n_layers=3)
        sparse_state = _fake_sparse_state(cfg, seed=40)
        step_layers, _, _ = build_step_layers(sparse_state, cfg)
        in_dim, out_dim = cfg.mlp_in, cfg.mlp_hidden
        L_last, L_second_last = step_layers[2][suffix], step_layers[1][suffix]

        w1 = grow_window_layer(L_last, in_dim, out_dim, num_cpus=2)
        w2 = grow_window_layer(L_second_last, in_dim, out_dim, num_cpus=2,
                                existing_window_layer=w1, existing_window_size=1)
        return cfg, in_dim, out_dim, L_last, L_second_last, w1, w2

    def test_first_position_shape_matches_source_layer(self):
        cfg, in_dim, out_dim, L_last, _, w1, _ = self._grown_two_position_window()
        assert w1.n_inputs == in_dim
        assert w1.n_outputs == out_dim
        np.testing.assert_allclose(
            _dense_block(w1, 0, in_dim, 0, out_dim),
            _dense_block(L_last, 0, in_dim, 0, out_dim),
            atol=1e-6)

    def test_growing_widens_shape_and_preserves_both_diagonal_blocks(self):
        cfg, in_dim, out_dim, L_last, L_second_last, _, w2 = self._grown_two_position_window()
        assert w2.n_inputs == 2 * in_dim
        assert w2.n_outputs == 2 * out_dim

        # Position added FIRST (window index 0) keeps its own offset --
        # growing the window must not shift already-placed blocks.
        np.testing.assert_allclose(
            _dense_block(w2, 0, in_dim, 0, out_dim),
            _dense_block(L_last, 0, in_dim, 0, out_dim),
            atol=1e-6)
        # Position added when the window grew (index 1) lands at the new offset.
        np.testing.assert_allclose(
            _dense_block(w2, in_dim, 2 * in_dim, out_dim, 2 * out_dim),
            _dense_block(L_second_last, 0, in_dim, 0, out_dim),
            atol=1e-6)

    def test_forward_matches_independent_positions_while_band_is_zero(self):
        # Off-diagonal (recurrent/skip) entries start zero-valued and
        # untrained -- so until synaptogenesis/training touch them, the
        # window's combined forward pass on concatenated per-position
        # inputs must equal each position's own independent forward_dense.
        cfg, in_dim, out_dim, L_last, L_second_last, _, w2 = self._grown_two_position_window()
        rng = np.random.RandomState(41)
        x0 = rng.randn(4, in_dim).astype(np.float32)
        x1 = rng.randn(4, in_dim).astype(np.float32)

        out_window = w2.forward_dense(np.concatenate([x0, x1], axis=1), learning_rate=0.0)
        expected = np.concatenate([
            L_last.forward_dense(x0, learning_rate=0.0),
            L_second_last.forward_dense(x1, learning_rate=0.0),
        ], axis=1)

        np.testing.assert_allclose(out_window, expected, rtol=1e-4, atol=1e-4)
