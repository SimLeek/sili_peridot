import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pytest

from sili.tensor import Tensor, gaussian_attention, exp

from model.config import MiniCPM5Config
from model.sili_block import (
    build_step_layers, _extract_true_csr, run_folded_recurrence,
    apply_fold_step, apply_window_step, default_window_energy,
    default_window_gaussian_params, rope_cos_sin, rmsnorm,
)
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


class TestRunFoldedRecurrenceWindowed:
    """run_folded_recurrence's window_state branch (see its own
    docstring). window_size==1 must stay bit-identical to the plain
    (window_state=None) path -- B8a's stage-0 sanity check depends on
    this, and window_size==1 bypasses apply_window_step entirely, so
    this is unaffected by the MAJOR PIVOT (2026-08-02) T=1/carried-state
    redesign. window_size>=2 now runs a genuinely different mechanism
    (gated blend between fresh input and carried memory, reusing
    pretrained Q/K/V/O -- see apply_window_step's docstring) -- there is
    no simple "matches independent positions" baseline for it anymore
    the way there was for the old T-batched causal-attention design, so
    these tests instead check the new mechanism is wired correctly
    (matches a manual per-token replication) and genuinely stateful
    (carried state actually affects output across tokens)."""

    def _built(self, n_layers=4, seed=60):
        cfg = _tiny_config(n_layers)
        sparse_state = _fake_sparse_state(cfg, seed=seed)
        step_layers, input_ln, post_ln = build_step_layers(sparse_state, cfg)
        final_norm = np.random.RandomState(seed + 1).randn(cfg.hidden_size).astype(np.float32)
        return cfg, step_layers, input_ln, post_ln, final_norm

    def test_window_size_one_matches_plain_path(self):
        cfg, step_layers, input_ln, post_ln, final_norm = self._built()
        T = 5
        x = np.random.RandomState(61).randn(T, cfg.hidden_size).astype(np.float32)

        plain = run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm,
                                      cfg, half_bandwidth=T)

        state = WindowState()
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        windowed = run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm,
                                         cfg, half_bandwidth=T, window_state=state)

        np.testing.assert_allclose(windowed, plain, rtol=1e-5, atol=1e-5)

    def test_window_size_two_matches_manual_per_token_replication(self):
        # Regression for the T=1/carried-state wiring itself: replicate
        # run_folded_recurrence's own per-token loop by hand (same fresh
        # zero carried_state, same fresh default_window_energy(), same
        # pre-window state, same window_state.centers/log_sigmas) and
        # confirm it produces the identical output. EnergyDynamics'
        # exploration noise draws from the global, unseeded np.random
        # (see sili__new JOURNAL.md), so both runs re-seed identically
        # right before their own (identical-length, identical-order)
        # sequence of energy-gate calls.
        cfg, step_layers, input_ln, post_ln, final_norm = self._built(n_layers=4, seed=70)
        T = 4
        x = np.random.RandomState(71).randn(T, cfg.hidden_size).astype(np.float32)

        state = WindowState()
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        assert state.window_size == 2
        assert state.window_positions == [3, 2]

        np.random.seed(12345)
        windowed = run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm,
                                         cfg, half_bandwidth=T, window_state=state)

        # Manual reference: pre-window positions 0,1 run exactly the
        # plain sequential loop (unaffected by the pivot); then the
        # window's own per-token loop, replicated by hand.
        cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)
        pre_state = np.zeros_like(x)
        for i in range(2):
            out = apply_fold_step(x + pre_state, step_layers[i], input_ln[i], post_ln[i],
                                  cfg, cos, sin, half_bandwidth=T)
            pre_state = pre_state + out
        x_common = x + pre_state

        window_ln = [input_ln[3], input_ln[2]]
        window_post_ln = [post_ln[3], post_ln[2]]
        hidden = cfg.hidden_size
        carried = np.zeros((2, hidden), dtype=np.float32)
        energy = default_window_energy()
        mean_column = np.empty((T, hidden), dtype=np.float32)

        np.random.seed(12345)
        for t in range(T):
            delta, carried, _aux = apply_window_step(
                x_common[t], carried, state.suffix_windows, 2,
                window_ln, window_post_ln, cfg, energy,
                state.centers, state.log_sigmas, num_cpus=4)
            columns_t = pre_state[t][None, :] + delta
            mean_column[t] = columns_t.mean(axis=0)
        expected = rmsnorm(mean_column, final_norm, cfg.rms_norm_eps)

        np.testing.assert_allclose(windowed, expected, rtol=1e-5, atol=1e-5)

    def test_carried_state_actually_affects_output(self):
        # A genuine statefulness check: feed the SAME token twice through
        # the window's per-token loop. If carried_state actually carries
        # information forward, the SECOND token's output must differ
        # from a version where carried_state is reset to zero right
        # before it (simulating "no memory") -- confirms the mechanism
        # isn't silently degenerating to a stateless per-token function.
        cfg, step_layers, input_ln, post_ln, final_norm = self._built(n_layers=3, seed=90)
        state = WindowState()
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        window_ln = [input_ln[p] for p in state.window_positions]
        window_post_ln = [post_ln[p] for p in state.window_positions]
        hidden = cfg.hidden_size
        x_t = np.random.RandomState(91).randn(hidden).astype(np.float32)

        np.random.seed(555)
        energy_a = default_window_energy()
        carried_a = np.zeros((2, hidden), dtype=np.float32)
        _delta1, carried_a, _ = apply_window_step(
            x_t, carried_a, state.suffix_windows, 2, window_ln, window_post_ln, cfg, energy_a,
            state.centers, state.log_sigmas, num_cpus=2)
        delta2_with_memory, _, _ = apply_window_step(
            x_t, carried_a, state.suffix_windows, 2, window_ln, window_post_ln, cfg, energy_a,
            state.centers, state.log_sigmas, num_cpus=2)

        np.random.seed(555)
        energy_b = default_window_energy()
        carried_b = np.zeros((2, hidden), dtype=np.float32)
        _delta1b, _carried_b, _ = apply_window_step(
            x_t, carried_b, state.suffix_windows, 2, window_ln, window_post_ln, cfg, energy_b,
            state.centers, state.log_sigmas, num_cpus=2)
        delta2_no_memory, _, _ = apply_window_step(
            x_t, np.zeros((2, hidden), dtype=np.float32), state.suffix_windows, 2,
            window_ln, window_post_ln, cfg, energy_b, state.centers, state.log_sigmas, num_cpus=2)

        assert not np.allclose(delta2_with_memory, delta2_no_memory), (
            "resetting carried_state to zero before the second token didn't "
            "change the output -- the mechanism isn't using carried state")

    def test_no_input_still_produces_nontrivial_state_driven_output(self):
        # "Sleep"/consolidation capability: with x_common_t all zero (no
        # real input at all), Q and K both draw from token+state (see
        # apply_window_step's docstring) so they degenerate to
        # self-attention over memory alone, NOT a dead/content-blind
        # gate. Confirms the mechanism can keep running purely off
        # carried_state -- a non-trivial (nonzero, non-degenerate)
        # output that actually depends on what's in memory.
        cfg, step_layers, input_ln, post_ln, final_norm = self._built(n_layers=3, seed=95)
        state = WindowState()
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        window_ln = [input_ln[p] for p in state.window_positions]
        window_post_ln = [post_ln[p] for p in state.window_positions]
        hidden = cfg.hidden_size
        x_zero = np.zeros(hidden, dtype=np.float32)

        rng = np.random.RandomState(96)
        carried_1 = rng.randn(2, hidden).astype(np.float32)
        carried_2 = rng.randn(2, hidden).astype(np.float32)

        np.random.seed(777)
        energy_1 = default_window_energy()
        delta_1, new_carried_1, _ = apply_window_step(
            x_zero, carried_1, state.suffix_windows, 2, window_ln, window_post_ln, cfg, energy_1,
            state.centers, state.log_sigmas, num_cpus=2)

        np.random.seed(777)
        energy_2 = default_window_energy()
        delta_2, new_carried_2, _ = apply_window_step(
            x_zero, carried_2, state.suffix_windows, 2, window_ln, window_post_ln, cfg, energy_2,
            state.centers, state.log_sigmas, num_cpus=2)

        assert np.all(np.isfinite(delta_1)) and np.all(np.isfinite(new_carried_1))
        assert not np.allclose(delta_1, 0.0), (
            "zero input produced a zero (dead) output -- the mechanism "
            "isn't running self-sustained dynamics off carried_state alone")
        assert not np.allclose(delta_1, delta_2), (
            "two DIFFERENT carried_state values gave the same output under "
            "zero input -- Q/K aren't actually depending on carried_state "
            "when there's no real input")

    def test_gaussian_attention_concentrates_on_own_pair_at_init(self):
        # Phase 2.7b's own init claim: center[p] = 2p+0.5, sigma[p] = 1.0
        # should concentrate most attention mass on position p's own
        # (fresh-token, carried-state) pair at indices (2p, 2p+1) of the
        # interleaved key space -- checked directly at the
        # gaussian_attention level (not through apply_window_step's
        # linear layers) by making each key's V a one-hot indicator, so
        # the output IS the attention distribution.
        window_size = 4
        centers, log_sigmas = default_window_gaussian_params(window_size)
        sigmas = exp(log_sigmas)
        K = 2 * window_size
        # Q/K share V's dimensionality (the kernel assumes one shared d
        # for all three) -- zero Q/K isolates the Gaussian bias (Q.K=0
        # for every pair), and V=eye(K) then makes the output BE the
        # attention distribution directly.
        q = Tensor(np.zeros((window_size, K), dtype=np.float32))
        k = Tensor(np.zeros((K, K), dtype=np.float32))
        v = Tensor(np.eye(K, dtype=np.float32))
        out = gaussian_attention(q, k, v, centers, sigmas, num_cpus=2, causal=False)
        weights = out.data  # [window_size, K] -- weights[p] is query p's full attention distribution
        for p in range(window_size):
            own_pair_mass = weights[p, 2 * p] + weights[p, 2 * p + 1]
            assert own_pair_mass > 0.5, (
                f"position {p}: own-pair attention mass {own_pair_mass:.3f} "
                f"did not dominate at init (expected >0.5, roughly 68%)")
            other_pair_mass = 1.0 - own_pair_mass
            assert own_pair_mass > other_pair_mass

    def test_other_positions_carried_state_affects_this_positions_output(self):
        # Unlike Phase 2.5/2.6's old cross_position_weight (an on/off
        # switch with a literal zero-init no-op default), Phase 2.7's
        # gaussian_attention always genuinely mixes across window
        # positions -- there is no "off" state to test for. Confirm
        # this directly: changing ONLY position 1's carried_state (not
        # position 0's own) still changes position 0's own delta.
        cfg, step_layers, input_ln, post_ln, final_norm = self._built(n_layers=3, seed=104)
        state = WindowState()
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        window_ln = [input_ln[p] for p in state.window_positions]
        window_post_ln = [post_ln[p] for p in state.window_positions]
        hidden = cfg.hidden_size
        x_t = np.random.RandomState(105).randn(hidden).astype(np.float32)
        carried_a = np.random.RandomState(106).randn(2, hidden).astype(np.float32)
        carried_b = carried_a.copy()
        carried_b[1] = np.random.RandomState(107).randn(hidden).astype(np.float32)

        np.random.seed(404)
        energy_a = default_window_energy()
        delta_a, _, _ = apply_window_step(
            x_t, carried_a, state.suffix_windows, 2, window_ln, window_post_ln, cfg, energy_a,
            state.centers, state.log_sigmas, num_cpus=2)

        np.random.seed(404)
        energy_b = default_window_energy()
        delta_b, _, _ = apply_window_step(
            x_t, carried_b, state.suffix_windows, 2, window_ln, window_post_ln, cfg, energy_b,
            state.centers, state.log_sigmas, num_cpus=2)

        assert np.all(np.isfinite(delta_a)) and np.all(np.isfinite(delta_b))
        assert not np.allclose(delta_a[0], delta_b[0]), (
            "changing only position 1's carried_state didn't change position "
            "0's own delta -- gaussian_attention isn't genuinely mixing "
            "across window positions")

    def test_centers_and_log_sigmas_receive_gradients_from_a_toy_backward(self):
        # Phase 2.7's whole point: centers/log_sigmas are ordinary
        # trainable Tensor leaves, reachable by plain backprop -- not
        # wired into a real task loss yet (Phase 3), but the gradient
        # path itself must work end to end (through the log_sigma->exp
        # ->sigma chain) using the SAME shapes apply_window_step builds
        # (window_size queries, 2*window_size interleaved keys).
        window_size = 3
        centers, log_sigmas = default_window_gaussian_params(window_size)
        sigmas = exp(log_sigmas)
        K = 2 * window_size
        rng = np.random.RandomState(42)
        q = Tensor(rng.randn(window_size, 8).astype(np.float32))
        k = Tensor(rng.randn(K, 8).astype(np.float32))
        v = Tensor(rng.randn(K, 8).astype(np.float32))
        out = gaussian_attention(q, k, v, centers, sigmas, num_cpus=2, causal=False)
        out.grad = np.ones_like(out.data)
        out._backward()
        sigmas._backward()  # propagate through exp() into log_sigmas.grad
        assert centers.grad is not None and np.all(np.isfinite(centers.grad))
        assert log_sigmas.grad is not None and np.all(np.isfinite(log_sigmas.grad))

    def test_window_output_shape_and_finite_for_larger_window(self):
        cfg, step_layers, input_ln, post_ln, final_norm = self._built(n_layers=4, seed=80)
        T = 3
        x = np.random.RandomState(81).randn(T, cfg.hidden_size).astype(np.float32)

        state = WindowState()
        for _ in range(4):
            state = advance_window(state, step_layers, SUFFIXES, cfg.num_hidden_layers, num_cpus=2)
        assert state.window_size == 4

        out = run_folded_recurrence(x, step_layers, input_ln, post_ln, final_norm,
                                    cfg, half_bandwidth=T, window_state=state)

        assert out.shape == (T, cfg.hidden_size)
        assert np.all(np.isfinite(out))
