"""
Regression/sanity tests for scripts/train_mqar_curriculum.py's Phase 7
wiring (sparsity plan, task #336): embed_width/input_sparsity_p/
wide_max_weights threaded through to ToyTileRecurrenceRMT, plus
steps_per_sec instrumentation. Calls train_curriculum() directly (not via
subprocess/CLI) for speed -- exercises the exact same code path main()
does, just skipping argv parsing.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from scripts.train_mqar_curriculum import train_curriculum


@pytest.mark.integration  # runs real training steps (10-30s), not <10s pre-commit territory
class TestTrainMqarCurriculumPhase7:
    def test_default_args_run_cleanly_and_report_steps_per_sec(self):
        r = train_curriculum("fp4", max_steps=30, seed=1, peak_lr=0.015, num_tiles=16, k_max=2, log_every=10)
        assert "steps_per_sec" in r
        assert r["steps_per_sec"] > 0
        assert r["total_steps"] >= 1

    def test_widened_sparse_arm_runs_cleanly(self):
        r = train_curriculum(
            "fp4",
            max_steps=30,
            seed=2,
            peak_lr=0.015,
            num_tiles=16,
            k_max=2,
            log_every=10,
            embed_width=32,
            input_sparsity_p=0.5,
            wide_max_weights=2048,
        )
        assert r["steps_per_sec"] > 0
        assert r["total_steps"] >= 1
        assert r["final_vocab"] >= 1

    def test_log_fn_receives_steps_per_sec_at_periodic_log_points(self):
        calls = []

        def log_fn(
            step,
            vocab_size,
            k,
            phase,
            event,
            loss_ema,
            acc_ema,
            ranks=None,
            steps_per_sec=None,
            max_streak=None,
            dy_r_target=None,
            x_r_target=None,
        ):
            calls.append((step, event, steps_per_sec))

        train_curriculum("fp4", max_steps=25, seed=3, peak_lr=0.015, num_tiles=16, k_max=2, log_every=10, log_fn=log_fn)
        periodic_calls = [c for c in calls if c[1] == ""]
        assert periodic_calls, "no periodic (non-event) log_fn calls recorded"
        assert all(c[2] is not None and c[2] > 0 for c in periodic_calls), (
            "periodic log_fn calls must always receive a real steps_per_sec"
        )
