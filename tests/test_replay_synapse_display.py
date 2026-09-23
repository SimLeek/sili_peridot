"""Tests for scripts/replay_synapse_display.py. discover_pools and
build_event_timeline are pure filesystem-reading logic, tested against
a real tmp_path directory tree. replay() is tested with a stubbed
display object (dependency-injected, no real GUI/displayarray call)
and fps set high enough that pacing sleeps are always skipped."""

import os

import numpy as np
import pytest

from scripts.replay_synapse_display import (
    build_event_timeline,
    deviation_from_col_state,
    discover_pools,
    replay,
)


def _write_snapshot(path, step, n_out=4, with_raw=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = {
        "step": step,
        "loss_ema": 1.0,
        "acc_ema": 0.5,
        "n_reset_this_cycle": 0,
        "l2_sat_ratio": 0.0,
        "l2_decay_strength": 0.0,
        "col_importance": np.zeros(n_out, dtype=np.float32),
        "col_grad_slow": np.zeros(n_out, dtype=np.float32),
        "col_grad_fast": np.ones(n_out, dtype=np.float32),
        "col_grad_var": np.ones(n_out, dtype=np.float32),
        "col_age": np.ones(n_out, dtype=np.uint32),
        "col_reset_active": np.zeros(n_out, dtype=np.uint8),
    }
    if with_raw:
        kwargs["raw_importance"] = np.random.rand(n_out, n_out).astype(np.float32)
        kwargs["raw_weight"] = np.random.rand(n_out, n_out).astype(np.float32)
    np.savez(path, **kwargs)


class TestDiscoverPools:
    def test_finds_only_directories(self, tmp_path):
        (tmp_path / "input_proj.block4").mkdir()
        (tmp_path / "q_proj.block4").mkdir()
        (tmp_path / "not_a_pool.txt").write_text("x")
        pools = discover_pools(str(tmp_path))
        assert pools == ["input_proj.block4", "q_proj.block4"]


class TestBuildEventTimeline:
    def test_merges_and_sorts_across_pools_by_step(self, tmp_path):
        _write_snapshot(str(tmp_path / "a" / "step00000100.npz"), 100)
        _write_snapshot(str(tmp_path / "b" / "step00000050.npz"), 50)
        _write_snapshot(str(tmp_path / "a" / "step00000200.npz"), 200)
        events = build_event_timeline(str(tmp_path), ["a", "b"])
        steps = [e[0] for e in events]
        assert steps == [50, 100, 200]
        assert events[0][1] == "b"
        assert events[1][1] == "a"

    def test_empty_dir_returns_empty_list(self, tmp_path):
        (tmp_path / "a").mkdir()
        assert build_event_timeline(str(tmp_path), ["a"]) == []


class TestDeviationFromColState:
    def test_matches_engine_formula(self):
        d = {
            "col_grad_fast": np.array([5.0], dtype=np.float32),
            "col_grad_slow": np.array([1.0], dtype=np.float32),
            "col_grad_var": np.array([4.0], dtype=np.float32),
        }
        dev = deviation_from_col_state(d)
        assert dev[0] == pytest.approx((5.0 - 1.0) / (2.0 + 1e-8), rel=1e-5)


class _FakeDisplay:
    def __init__(self):
        self.calls = []
        self._closed = False

    def closed(self):
        return self._closed

    def update_pool(self, pool_key, step, importance, weight, deviation, reset_active):
        self.calls.append((pool_key, step))


class TestReplay:
    def test_plays_every_recorded_frame_in_step_order(self, tmp_path):
        _write_snapshot(str(tmp_path / "a" / "step00000100.npz"), 100)
        _write_snapshot(str(tmp_path / "a" / "step00000050.npz"), 50)
        fake = _FakeDisplay()
        n = replay(str(tmp_path), fps=1000.0, pool_order=["a"], display=fake)
        assert n == 2
        assert [c[1] for c in fake.calls] == [50, 100]

    def test_stops_early_when_display_closed(self, tmp_path):
        _write_snapshot(str(tmp_path / "a" / "step00000050.npz"), 50)
        _write_snapshot(str(tmp_path / "a" / "step00000100.npz"), 100)
        fake = _FakeDisplay()
        fake._closed = True
        n = replay(str(tmp_path), fps=1000.0, pool_order=["a"], display=fake)
        assert n == 0

    def test_raises_when_raw_importance_missing(self, tmp_path):
        _write_snapshot(str(tmp_path / "a" / "step00000050.npz"), 50, with_raw=False)
        fake = _FakeDisplay()
        with pytest.raises(KeyError, match="raw_importance"):
            replay(str(tmp_path), fps=1000.0, pool_order=["a"], display=fake)

    def test_no_snapshots_raises_file_not_found(self, tmp_path):
        (tmp_path / "a").mkdir()
        with pytest.raises(FileNotFoundError):
            replay(str(tmp_path), pool_order=["a"], display=_FakeDisplay())
