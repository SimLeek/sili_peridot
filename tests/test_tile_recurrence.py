import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sili import _cpu
from sili.tensor import Tensor, exp, gaussian_attention

from model.config import MiniCPM5Config
from model.sili_block import build_step_layers, default_window_energy
from model.tile_recurrence import (
    TileState,
    apply_tile_step,
    bootstrap_tile_layers,
    build_tile_window,
    default_tile_gaussian_params,
    run_tile_recurrence,
)


def _tiny_config(n_layers=4) -> MiniCPM5Config:
    return MiniCPM5Config(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=10,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


def _fake_sparse_state(cfg: MiniCPM5Config, seed=3) -> dict:
    torch.manual_seed(seed)
    sd = {}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[p + ".self_attn.q_proj.weight"] = {
            "raw": torch.randn(cfg.q_proj_out, cfg.attn_in),
            "shape": (cfg.q_proj_out, cfg.attn_in),
        }
        sd[p + ".self_attn.k_proj.weight"] = {
            "raw": torch.randn(cfg.kv_proj_out, cfg.attn_in),
            "shape": (cfg.kv_proj_out, cfg.attn_in),
        }
        sd[p + ".self_attn.v_proj.weight"] = {
            "raw": torch.randn(cfg.kv_proj_out, cfg.attn_in),
            "shape": (cfg.kv_proj_out, cfg.attn_in),
        }
        sd[p + ".self_attn.o_proj.weight"] = {
            "raw": torch.randn(cfg.attn_out, cfg.o_proj_in),
            "shape": (cfg.attn_out, cfg.o_proj_in),
        }
        sd[p + ".mlp.gate_proj.weight"] = {
            "raw": torch.randn(cfg.mlp_hidden, cfg.mlp_in),
            "shape": (cfg.mlp_hidden, cfg.mlp_in),
        }
        sd[p + ".mlp.up_proj.weight"] = {
            "raw": torch.randn(cfg.mlp_hidden, cfg.mlp_in),
            "shape": (cfg.mlp_hidden, cfg.mlp_in),
        }
        sd[p + ".mlp.down_proj.weight"] = {
            "raw": torch.randn(cfg.mlp_out, cfg.mlp_hidden),
            "shape": (cfg.mlp_out, cfg.mlp_hidden),
        }
        sd[p + ".input_layernorm.weight"] = {"raw": torch.ones(cfg.hidden_size), "shape": (cfg.hidden_size,)}
        sd[p + ".post_attention_layernorm.weight"] = {"raw": torch.ones(cfg.hidden_size), "shape": (cfg.hidden_size,)}
    return sd


def _built(n_layers=4, seed=10):
    cfg = _tiny_config(n_layers)
    sparse_state = _fake_sparse_state(cfg, seed=seed)
    step_layers, input_ln, post_ln = build_step_layers(sparse_state, cfg, num_cpus=2)
    return cfg, step_layers, input_ln, post_ln


class TestDefaultTileGaussianParams:
    def test_init_matches_own_index_midpoint(self):
        centers, log_sigmas = default_tile_gaussian_params(5)
        np.testing.assert_allclose(centers.data, [0.5, 1.5, 2.5, 3.5, 4.5])
        np.testing.assert_allclose(log_sigmas.data, np.zeros(5))


class TestBuildTileWindow:
    def test_first_tick_only_last_tile_gets_real_input(self):
        hidden = 6
        x = np.arange(4 * hidden, dtype=np.float32).reshape(4, hidden)
        M_prev = -np.ones((4, hidden), dtype=np.float32)
        window = build_tile_window(x, i=0, num_tiles=4, M_prev=M_prev)
        # tiles 0,1,2 have no real token yet (negative source index) -- keep M_prev
        np.testing.assert_array_equal(window[0], M_prev[0])
        np.testing.assert_array_equal(window[1], M_prev[1])
        np.testing.assert_array_equal(window[2], M_prev[2])
        # tile 3 (the last) gets the real first token
        np.testing.assert_array_equal(window[3], x[0])

    def test_full_window_once_past_the_ramp(self):
        hidden = 6
        x = np.arange(10 * hidden, dtype=np.float32).reshape(10, hidden)
        M_prev = -np.ones((4, hidden), dtype=np.float32)
        window = build_tile_window(x, i=5, num_tiles=4, M_prev=M_prev)
        # tile j should hold x[5 - 3 + j]
        for j in range(4):
            np.testing.assert_array_equal(window[j], x[5 - 3 + j])

    def test_partial_ramp(self):
        hidden = 3
        x = np.arange(10 * hidden, dtype=np.float32).reshape(10, hidden)
        M_prev = -np.ones((4, hidden), dtype=np.float32)
        window = build_tile_window(x, i=1, num_tiles=4, M_prev=M_prev)
        # i=1, num_tiles=4: src = 1-3+j = j-2 -> tiles 0,1 negative, 2,3 real
        np.testing.assert_array_equal(window[0], M_prev[0])
        np.testing.assert_array_equal(window[1], M_prev[1])
        np.testing.assert_array_equal(window[2], x[0])
        np.testing.assert_array_equal(window[3], x[1])


class TestApplyTileStepGradientFlow:
    def test_centers_and_log_sigmas_receive_gradients_from_a_toy_backward(self):
        # Same spirit as Phase 2.7b's own test (test_curriculum.py) --
        # apply_tile_step's linear/MLP layers aren't Tensor-graph nodes
        # (SparseLinearLayer trains inline, not via backprop -- see the
        # approved plan's Training methodology section), so gradient
        # flow to centers/log_sigmas is only meaningful to check at the
        # gaussian_attention call itself, using the SAME shapes
        # apply_tile_step builds (num_tiles queries, num_tiles keys --
        # no interleaving, unlike Phase 2.7b's window mechanism).
        num_tiles = 5
        centers, log_sigmas = default_tile_gaussian_params(num_tiles)
        sigmas = exp(log_sigmas)
        rng = np.random.RandomState(7)
        q = Tensor(rng.randn(num_tiles, 4).astype(np.float32))
        k = Tensor(rng.randn(num_tiles, 4).astype(np.float32))
        v = Tensor(rng.randn(num_tiles, 4).astype(np.float32))
        out = gaussian_attention(q, k, v, centers, sigmas, num_cpus=2, causal=False)
        out.grad = np.ones_like(out.data)
        out._backward()
        sigmas._backward()
        assert centers.grad is not None and np.all(np.isfinite(centers.grad))
        assert log_sigmas.grad is not None and np.all(np.isfinite(log_sigmas.grad))


class TestApplyTileStepBehavior:
    def _setup(self, n_layers=4, num_tiles=4, seed=20, position_index=2):
        cfg, step_layers, input_ln, post_ln = _built(n_layers=n_layers, seed=seed)
        tile_layers = bootstrap_tile_layers(step_layers, position_index)
        tile_state = TileState.zeros(num_tiles, cfg.hidden_size)
        energy = default_window_energy()
        return cfg, tile_layers, input_ln[position_index], post_ln[position_index], tile_state, energy

    def test_shapes_and_finite_over_several_ticks(self):
        num_tiles = 4
        cfg, tile_layers, in_ln, post_ln, tile_state, energy = self._setup(num_tiles=num_tiles)
        T = 6
        x = np.random.RandomState(21).randn(T, cfg.hidden_size).astype(np.float32)
        M = tile_state.M
        for i in range(T):
            x_window = build_tile_window(x, i, num_tiles, M)
            M, logits, _aux_loss = apply_tile_step(
                x_window,
                i,
                M,
                tile_layers,
                in_ln,
                post_ln,
                cfg,
                tile_state.centers,
                tile_state.log_sigmas,
                energy,
                num_cpus=2,
            )
            assert M.shape == (num_tiles, cfg.hidden_size)
            assert np.all(np.isfinite(M))
            assert logits is None  # no lm_head given

    def test_run_tile_recurrence_produces_logits_of_expected_shape(self):
        num_tiles = 4
        cfg, tile_layers, in_ln, post_ln, tile_state, energy = self._setup(num_tiles=num_tiles)
        T = 5
        x = np.random.RandomState(22).randn(T, cfg.hidden_size).astype(np.float32)
        lm_head = np.random.RandomState(23).randn(cfg.vocab_size, cfg.hidden_size).astype(np.float32)

        M_final, logits = run_tile_recurrence(
            x, num_tiles, tile_state, tile_layers, in_ln, post_ln, cfg, energy, lm_head=lm_head, num_cpus=2
        )

        assert M_final.shape == (num_tiles, cfg.hidden_size)
        assert logits.shape == (T, cfg.vocab_size)
        assert np.all(np.isfinite(M_final)) and np.all(np.isfinite(logits))

    def test_M_actually_changes_tick_to_tick(self):
        # A fixed repeating token still drives real updates each tick
        # (not silently frozen/dead) -- the tile network + gaussian
        # attention + gated residual genuinely does something each step.
        num_tiles = 4
        cfg, tile_layers, in_ln, post_ln, tile_state, energy = self._setup(num_tiles=num_tiles, seed=30)
        x_t = np.random.RandomState(31).randn(cfg.hidden_size).astype(np.float32)
        x = np.tile(x_t, (8, 1)).astype(np.float32)

        M = tile_state.M
        seen = [M.copy()]
        for i in range(x.shape[0]):
            x_window = build_tile_window(x, i, num_tiles, M)
            M, _logits, _aux = apply_tile_step(
                x_window,
                i,
                M,
                tile_layers,
                in_ln,
                post_ln,
                cfg,
                tile_state.centers,
                tile_state.log_sigmas,
                energy,
                num_cpus=2,
            )
            seen.append(M.copy())

        # consecutive states must differ at least once past the first tick
        diffs = [not np.allclose(seen[t], seen[t + 1]) for t in range(len(seen) - 1)]
        assert any(diffs), "M never changed across 8 ticks of real input -- looks dead"

    def test_resetting_M_prev_changes_logits(self):
        # Statefulness check, same spirit as B8's own
        # test_carried_state_actually_affects_output.
        num_tiles = 4
        cfg, tile_layers, in_ln, post_ln, tile_state, _energy = self._setup(num_tiles=num_tiles, seed=40)
        lm_head = np.random.RandomState(41).randn(cfg.vocab_size, cfg.hidden_size).astype(np.float32)
        x_t = np.random.RandomState(42).randn(cfg.hidden_size).astype(np.float32)
        x_window = np.tile(x_t, (num_tiles, 1)).astype(np.float32)

        M_real = np.random.RandomState(43).randn(num_tiles, cfg.hidden_size).astype(np.float32)
        M_zero = np.zeros((num_tiles, cfg.hidden_size), dtype=np.float32)

        np.random.seed(999)
        energy_a = default_window_energy()
        _M_a, logits_a, _ = apply_tile_step(
            x_window,
            num_tiles - 1,
            M_real,
            tile_layers,
            in_ln,
            post_ln,
            cfg,
            tile_state.centers,
            tile_state.log_sigmas,
            energy_a,
            lm_head=lm_head,
            num_cpus=2,
        )

        np.random.seed(999)
        energy_b = default_window_energy()
        _M_b, logits_b, _ = apply_tile_step(
            x_window,
            num_tiles - 1,
            M_zero,
            tile_layers,
            in_ln,
            post_ln,
            cfg,
            tile_state.centers,
            tile_state.log_sigmas,
            energy_b,
            lm_head=lm_head,
            num_cpus=2,
        )

        assert not np.allclose(logits_a, logits_b), (
            "different M_prev produced identical logits -- the mechanism isn't actually using carried state"
        )


def _fast_sparse_layer(n_in, n_out, density, rng, num_cpus=4):
    """Direct random-CSR SparseLinearLayer construction at a realistic
    post-pruning density -- bypasses the full torch/fold pipeline
    entirely (measured far too slow for fully-dense random weights at
    real dims earlier this session), matching the technique used for
    this session's own compute-growth benchmarks."""
    k = max(1, round(density * n_out))
    ptrs = np.arange(0, (n_in + 1) * k, k, dtype=np.int32)
    idx = np.empty(n_in * k, dtype=np.int32)
    for r in range(n_in):
        idx[r * k : (r + 1) * k] = rng.choice(n_out, size=k, replace=False).astype(np.int32)
        idx[r * k : (r + 1) * k].sort()
    vals = (rng.randn(n_in * k) * 0.02).astype(np.float32)
    layer = _cpu.SparseLinearLayer(n_in, n_out, int(vals.shape[0] * 1.3) + 64, num_cpus)
    layer.load_weights(ptrs, idx, vals)
    return layer


@pytest.mark.parametrize("num_tiles", [8, 32])
def test_real_dims_smoke(num_tiles):
    """Confirms the whole pipeline runs end to end at real MiniCPM5
    shapes (not just tiny synthetic ones), and reports actual per-tick
    wall time -- see the approved plan's Sizing section, which needs
    real data (not a guess) to inform num_tiles choices later."""
    cfg = MiniCPM5Config(
        hidden_size=1536,
        intermediate_size=4608,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=10,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    rng = np.random.RandomState(0)
    tile_layers = {
        ".self_attn.q_proj.weight": _fast_sparse_layer(cfg.attn_in, cfg.q_proj_out, 0.1, rng),
        ".self_attn.k_proj.weight": _fast_sparse_layer(cfg.attn_in, cfg.kv_proj_out, 0.1, rng),
        ".self_attn.v_proj.weight": _fast_sparse_layer(cfg.attn_in, cfg.kv_proj_out, 0.1, rng),
        ".self_attn.o_proj.weight": _fast_sparse_layer(cfg.o_proj_in, cfg.attn_out, 0.1, rng),
        ".mlp.gate_proj.weight": _fast_sparse_layer(cfg.mlp_in, cfg.mlp_hidden, 0.1, rng),
        ".mlp.up_proj.weight": _fast_sparse_layer(cfg.mlp_in, cfg.mlp_hidden, 0.1, rng),
        ".mlp.down_proj.weight": _fast_sparse_layer(cfg.mlp_hidden, cfg.mlp_out, 0.1, rng),
    }
    input_ln = np.ones(cfg.hidden_size, dtype=np.float32)
    post_ln = np.ones(cfg.hidden_size, dtype=np.float32)
    tile_state = TileState.zeros(num_tiles, cfg.hidden_size)
    energy = default_window_energy()

    T = 4
    x = np.random.RandomState(1).randn(T, cfg.hidden_size).astype(np.float32)
    M = tile_state.M
    t0 = time.time()
    for i in range(T):
        x_window = build_tile_window(x, i, num_tiles, M)
        M, _logits, _aux = apply_tile_step(
            x_window,
            i,
            M,
            tile_layers,
            input_ln,
            post_ln,
            cfg,
            tile_state.centers,
            tile_state.log_sigmas,
            energy,
            num_cpus=4,
        )
    per_tick_ms = (time.time() - t0) / T * 1000
    print(f"\n[real-dims smoke] num_tiles={num_tiles}: {per_tick_ms:.2f} ms/tick")

    assert M.shape == (num_tiles, cfg.hidden_size)
    assert np.all(np.isfinite(M))
