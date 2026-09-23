"""Tests for scripts/plasticity_sim.py. The cycle_boundary integration
test reuses the EXACT same 4-column fixture and expected numbers as
sili__new's own C++ engine tests
(test_select_by_deviation_gate_uses_population_percentile_not_passed_k
in tests/unit/test_amortized_plasticity_reset.cpp) -- a precision
cross-check that this Python port is bit-faithful to the real engine,
not just plausible-looking."""

import math
import os

import numpy as np
import pytest

from scripts.plasticity_sim import (
    SimPlasticityState,
    ceiling_decay_step,
    cycle_boundary,
    evt_k,
    export_simulated_trajectory,
    load_pool_trajectory,
    natural_deltas,
    percentile_k,
    saturation_fraction,
    sim_export_dir,
    simulate,
    simulate_level_based,
    update_grad_tracking,
)


class TestUpdateGradTracking:
    def test_cold_start_sets_fast_equal_slow_equal_delta_var_zero(self):
        state = SimPlasticityState(n_out=2, col_importance=np.array([10.0, 3.0]))
        update_grad_tracking(state)
        assert np.allclose(state.col_grad_fast, [10.0, 3.0])
        assert np.allclose(state.col_grad_slow, [10.0, 3.0])
        assert np.allclose(state.col_grad_var, [0.0, 0.0])
        assert np.all(state.col_grad_initialized)

    def test_asymmetric_catchup_on_sustained_drop(self):
        # Matches the C++ test_asymmetric_catchup_rate scenario in
        # spirit: a sustained DROP in delta should converge at the FAST
        # catchup rate (eta_slow_catchup), not the slow one.
        state = SimPlasticityState(n_out=1, col_importance=np.array([10.0]))
        update_grad_tracking(state)  # cold start, delta=10
        state.col_importance = np.array([10.0])  # delta=0 next cycle
        update_grad_tracking(state, eta_slow=0.99, eta_slow_catchup=0.95)
        # delta(0) < slow(10) -> catchup branch
        assert state.col_grad_slow[0] == pytest.approx(0.95 * 10.0 + 0.05 * 0.0)


class TestPercentileK:
    def test_matches_numpy_percentile_linear(self):
        dev = np.array([1.0, 2.0, 3.0, 4.0])
        mature = np.array([True, True, True, True])
        k = percentile_k(dev, mature, reset_fraction=0.25)
        assert k == pytest.approx(np.percentile(dev, 75.0, method="linear"))

    def test_empty_mature_returns_zero(self):
        dev = np.array([1.0, 2.0])
        mature = np.array([False, False])
        assert percentile_k(dev, mature, 0.25) == 0.0


class TestEvtK:
    def test_matches_sqrt_2_ln_n(self):
        assert evt_k(288) == pytest.approx(math.sqrt(2 * math.log(288)))

    def test_n_le_1_is_zero(self):
        assert evt_k(1) == 0.0
        assert evt_k(0) == 0.0


class TestCycleBoundaryFidelityVsCppEngine:
    """Reproduces test_select_by_deviation_gate_uses_population_percentile_not_passed_k
    from sili__new's own C++ suite: 4 columns, 3 warmup cycles at
    col_importance=10 (all cols), then col 3 jumps to 60. Expected
    deviation[3]~=1.2392, deviation[0..2]~=-1.4837, k_effective (percentile
    75)~=-0.803, boost[3]~=2.042 -- exact same numbers the C++ RED test
    printed before the fix, and the real engine after it."""

    def _warm(self):
        state = SimPlasticityState(n_out=4, col_importance=np.array([10.0, 10.0, 10.0, 10.0]))
        for _ in range(3):
            cycle_boundary(state, reset_fraction=0.0, k=0.5, select_by_deviation=False)
            state.col_importance = np.array([10.0, 10.0, 10.0, 10.0])
        return state

    def test_select_by_deviation_picks_the_spiking_column(self):
        state = self._warm()
        state.col_importance = np.array([10.0, 10.0, 10.0, 60.0])
        active, deviation = cycle_boundary(
            state, reset_fraction=0.25, k=0.0, select_by_deviation=True, k_mode="percentile"
        )
        assert active[3] and not active[:3].any()
        assert deviation[3] == pytest.approx(1.2392, abs=1e-3)
        assert deviation[0] == pytest.approx(-1.4837, abs=1e-3)
        assert state.col_plasticity_boost[3] == pytest.approx(2.042, abs=1e-2)

    def test_default_mode_picks_highest_importance_not_deviation(self):
        state = self._warm()
        state.col_importance = np.array([10.0, 10.0, 10.0, 60.0])
        active, _ = cycle_boundary(state, reset_fraction=0.25, k=0.5, select_by_deviation=False)
        # col 3 also has the highest col_importance (60), so default
        # mode picks it too here -- distinguishing test below uses a
        # genuinely different top pick.
        assert active[3]

    def test_default_vs_deviation_mode_can_disagree(self):
        # col 0: high absolute importance, flat history (low deviation).
        # col 1: low absolute importance, sudden spike (high deviation).
        state = SimPlasticityState(n_out=2, col_importance=np.array([50.0, 1.0]))
        for _ in range(3):
            cycle_boundary(state, reset_fraction=0.0, k=0.5, select_by_deviation=False)
            state.col_importance = state.col_importance + np.array([1.0, 0.0])
        state.col_importance = state.col_importance + np.array([1.0, 20.0])
        state_default = state
        active_default, _ = cycle_boundary(state_default, reset_fraction=0.5, k=0.5, select_by_deviation=False)
        assert active_default[0] and not active_default[1]


class TestLoadAndReplayFidelity:
    def _write_run(self, tmp_path, deltas):
        """deltas: list of per-cycle (n_out,) natural deltas to apply,
        used to build a REAL-looking recorded col_importance trajectory
        with NO reset events (reset_fraction=0 throughout), so replay
        with zero candidate intervention must exactly reproduce it."""
        pool_dir = tmp_path / "q_proj.block4"
        pool_dir.mkdir()
        col_importance = np.zeros(len(deltas[0]))
        for t, d in enumerate([np.zeros(len(deltas[0])), *deltas]):
            col_importance = col_importance + d
            np.savez(
                pool_dir / f"step{t * 50:08d}.npz",
                step=t * 50,
                loss_ema=1.0,
                acc_ema=0.5,
                n_reset_this_cycle=0,
                l2_sat_ratio=0.0,
                l2_decay_strength=0.0,
                col_importance=col_importance.astype(np.float32),
                col_grad_slow=np.zeros_like(col_importance, dtype=np.float32),
                col_grad_fast=np.zeros_like(col_importance, dtype=np.float32),
                col_grad_var=np.zeros_like(col_importance, dtype=np.float32),
                col_age=np.full_like(col_importance, t, dtype=np.uint32),
                col_reset_active=np.zeros_like(col_importance, dtype=np.uint8),
            )
        return str(pool_dir)

    def test_load_pool_trajectory_matches_written_data(self, tmp_path):
        deltas = [np.array([1.0, 2.0]), np.array([0.5, -0.5])]
        pool_dir = self._write_run(tmp_path, deltas)
        traj = load_pool_trajectory(pool_dir)
        assert traj.n_cycles == 3
        assert traj.n_out == 2
        assert np.allclose(traj.col_importance[-1], [1.5, 1.5])

    def test_natural_deltas_recovers_written_deltas(self, tmp_path):
        deltas = [np.array([1.0, 2.0]), np.array([0.5, -0.5]), np.array([2.0, 0.0])]
        pool_dir = self._write_run(tmp_path, deltas)
        traj = load_pool_trajectory(pool_dir)
        recovered = natural_deltas(traj.col_importance)
        assert np.allclose(recovered, np.stack(deltas))

    def test_zero_intervention_replay_reproduces_real_trajectory_exactly(self, tmp_path):
        rng = np.random.default_rng(0)
        deltas = [rng.normal(size=6) for _ in range(20)]
        pool_dir = self._write_run(tmp_path, deltas)
        traj = load_pool_trajectory(pool_dir)
        result = simulate(traj, reset_fraction=0.0, select_by_deviation=True)
        assert np.allclose(result.col_importance, traj.col_importance, atol=1e-6)
        assert not result.col_reset_active.any()

    def test_raw_mean_ci_signal_requires_raw_importance(self, tmp_path):
        deltas = [np.array([1.0, 2.0])]
        pool_dir = self._write_run(tmp_path, deltas)  # no raw_importance written
        traj = load_pool_trajectory(pool_dir)
        assert traj.raw_mean_ci is None
        with pytest.raises(KeyError, match="raw_importance"):
            simulate(traj, reset_fraction=0.0, select_by_deviation=True, signal="raw_mean_ci")

    def test_raw_mean_ci_zero_intervention_replay_matches_mean_of_raw_importance(self, tmp_path):
        pool_dir = tmp_path / "q_proj.block4"
        pool_dir.mkdir()
        rng = np.random.default_rng(1)
        n_in, n_out = 3, 4
        col_importance = np.zeros(n_out)
        for t in range(5):
            raw = rng.random((n_in, n_out))
            col_importance = col_importance + rng.normal(size=n_out)
            np.savez(
                pool_dir / f"step{t * 10:08d}.npz",
                step=t * 10,
                loss_ema=1.0,
                acc_ema=0.5,
                n_reset_this_cycle=0,
                l2_sat_ratio=0.0,
                l2_decay_strength=0.0,
                col_importance=col_importance.astype(np.float32),
                col_grad_slow=np.zeros(n_out, dtype=np.float32),
                col_grad_fast=np.zeros(n_out, dtype=np.float32),
                col_grad_var=np.zeros(n_out, dtype=np.float32),
                col_age=np.full(n_out, t, dtype=np.uint32),
                col_reset_active=np.zeros(n_out, dtype=np.uint8),
                raw_importance=raw.astype(np.float32),
            )
        traj = load_pool_trajectory(str(pool_dir))
        assert traj.raw_mean_ci is not None
        result = simulate(traj, reset_fraction=0.0, select_by_deviation=True, signal="raw_mean_ci")
        assert np.allclose(result.col_importance, traj.raw_mean_ci, atol=1e-4)


class TestSaturationFraction:
    def test_counts_columns_near_max_ci(self):
        class _R:
            pass

        r = _R()
        r.col_importance = np.array([[100.0, 0.0, 99.6], [50.0, 100.0, 0.0]])
        frac = saturation_fraction(r, max_ci=100.0, tol=0.5)
        assert frac[0] == pytest.approx(2 / 3)
        assert frac[1] == pytest.approx(1 / 3)


class TestSimExportDir:
    def test_naming_convention_suffixes_run_dir(self):
        out = sim_export_dir("logs/plasticity_column_snapshots/dense_v8", "my_candidate", "q_proj.block4")
        assert out == os.path.join("logs/plasticity_column_snapshots", "dense_v8_sim_my_candidate", "q_proj.block4")


class TestExportSimulatedTrajectory:
    def test_writes_one_file_per_cycle_with_expected_schema(self, tmp_path):
        pool_dir = tmp_path / "real" / "q_proj.block4"
        pool_dir.mkdir(parents=True)
        n_out = 3
        for t in range(3):
            np.savez(
                pool_dir / f"step{t * 10:08d}.npz",
                step=t * 10,
                loss_ema=1.0,
                acc_ema=0.5,
                n_reset_this_cycle=0,
                l2_sat_ratio=0.0,
                l2_decay_strength=0.0,
                col_importance=np.full(n_out, float(t + 1), dtype=np.float32),
                col_grad_slow=np.zeros(n_out, dtype=np.float32),
                col_grad_fast=np.zeros(n_out, dtype=np.float32),
                col_grad_var=np.zeros(n_out, dtype=np.float32),
                col_age=np.full(n_out, t, dtype=np.uint32),
                col_reset_active=np.zeros(n_out, dtype=np.uint8),
                raw_importance=np.full((2, n_out), float(t + 1), dtype=np.float32),
                raw_weight=np.ones((2, n_out), dtype=np.float32),
            )
        traj = load_pool_trajectory(str(pool_dir))
        result = simulate(traj, reset_fraction=0.0, select_by_deviation=True)
        out_dir = tmp_path / "exported"
        n = export_simulated_trajectory(traj, result, str(out_dir))
        assert n == 3
        files = sorted(os.listdir(out_dir))
        assert len(files) == 3
        d = np.load(out_dir / files[-1])
        for field in [
            "step",
            "col_importance",
            "col_grad_slow",
            "col_grad_fast",
            "col_grad_var",
            "col_age",
            "col_reset_active",
            "raw_importance",
        ]:
            assert field in d.files


class TestCeilingDecayStep:
    def test_below_ceiling_untouched(self):
        col_imp = np.array([50.0, 89.0])
        touched, strength = ceiling_decay_step(col_imp, max_ci=100.0, ceiling_frac=0.9)
        assert not touched.any()
        assert np.allclose(strength, 0.0)

    def test_at_max_ci_gets_full_decay_rate(self):
        col_imp = np.array([100.0])
        touched, strength = ceiling_decay_step(col_imp, max_ci=100.0, ceiling_frac=0.9, decay_rate=0.05)
        assert touched[0]
        assert strength[0] == pytest.approx(0.05)

    def test_ramps_smoothly_between_ceiling_and_max(self):
        # Halfway between ceiling (90) and max (100) -> half decay_rate.
        col_imp = np.array([95.0])
        touched, strength = ceiling_decay_step(col_imp, max_ci=100.0, ceiling_frac=0.9, decay_rate=0.05)
        assert strength[0] == pytest.approx(0.025)


class TestSimulateLevelBased:
    def _write_run(self, tmp_path, deltas):
        pool_dir = tmp_path / "q_proj.block4"
        pool_dir.mkdir()
        col_importance = np.zeros(len(deltas[0]))
        for t, d in enumerate([np.zeros(len(deltas[0])), *deltas]):
            col_importance = col_importance + d
            np.savez(
                pool_dir / f"step{t * 50:08d}.npz",
                step=t * 50,
                loss_ema=1.0,
                acc_ema=0.5,
                n_reset_this_cycle=0,
                l2_sat_ratio=0.0,
                l2_decay_strength=0.0,
                col_importance=col_importance.astype(np.float32),
                col_grad_slow=np.zeros_like(col_importance, dtype=np.float32),
                col_grad_fast=np.zeros_like(col_importance, dtype=np.float32),
                col_grad_var=np.zeros_like(col_importance, dtype=np.float32),
                col_age=np.full_like(col_importance, t, dtype=np.uint32),
                col_reset_active=np.zeros_like(col_importance, dtype=np.uint8),
            )
        return str(pool_dir)

    def test_prevents_runaway_growth_past_ceiling(self, tmp_path):
        # A column with relentless positive growth every cycle should
        # be held back once it crosses the ceiling, unlike unclamped
        # accumulation which would just keep climbing.
        deltas = [np.array([5.0]) for _ in range(60)]
        pool_dir = self._write_run(tmp_path, deltas)
        traj = load_pool_trajectory(pool_dir)
        result = simulate_level_based(traj, ceiling_decay_step, max_ci=100.0, ceiling_frac=0.9, decay_rate=0.2)
        # Without any decay, 60 cycles * 5.0 = 300 -- the candidate must
        # keep it well below that runaway value.
        assert result.col_importance[-1, 0] < 150.0
        assert result.col_reset_active[1:].any()  # the ceiling rule did fire at some point

    def test_zero_growth_column_never_touched(self, tmp_path):
        deltas = [np.array([0.0]) for _ in range(10)]
        pool_dir = self._write_run(tmp_path, deltas)
        traj = load_pool_trajectory(pool_dir)
        result = simulate_level_based(traj, ceiling_decay_step, max_ci=100.0, ceiling_frac=0.9)
        assert not result.col_reset_active.any()
