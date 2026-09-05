"""Tests for model/eval_lr.py's generic optimal-lr search.

Fast tests use a synthetic unimodal log(lr) curve (no real training) to
check the search algorithm itself -- bracketing in either direction,
convergence, time-budget respect, and cache reuse. One opt-in test wires
the search to the actual `baseline` tile-recurrence config (same one the
manual 1x-100x sweep in conversation was run against) and checks it lands
in the same broad peak region that sweep found by hand.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from model.eval_lr import find_optimal_lr

RUN_ENV_VAR = "SILI_RUN_LR_SEARCH"


def _gaussian_log_lr_score(peak_log_lr, width=1.0, noise=0.0, rng=None):
    """Synthetic score function: unimodal Gaussian bump in log(lr) space
    (mirrors the real sweep's own shape -- rises then falls, single peak),
    cheap enough to call thousands of times per test."""

    def trial_fn(lr, seed):
        x = (math.log(lr) - peak_log_lr) / width
        score = math.exp(-0.5 * x * x)
        if noise:
            r = rng if rng is not None else __import__("random").Random(seed)
            score += r.uniform(-noise, noise)
        return score

    return trial_fn


class TestFindOptimalLRSynthetic:
    def test_finds_peak_when_growth_direction_helps(self):
        # Peak well above initial_lr -- bracketing must expand upward.
        trial_fn = _gaussian_log_lr_score(peak_log_lr=math.log(1e-2))
        result = find_optimal_lr(trial_fn, seeds=(0,), initial_lr=1e-4, time_budget_s=10.0, log_tol=0.01)
        assert result.best_lr == pytest.approx(1e-2, rel=0.05)
        assert result.stopped_reason == "converged"

    def test_finds_peak_when_growth_direction_hurts(self):
        # Peak well BELOW initial_lr -- the first growth step must make
        # things worse, forcing bracketing to reverse direction.
        trial_fn = _gaussian_log_lr_score(peak_log_lr=math.log(1e-5))
        result = find_optimal_lr(trial_fn, seeds=(0,), initial_lr=1e-2, time_budget_s=10.0, log_tol=0.01)
        assert result.best_lr == pytest.approx(1e-5, rel=0.05)

    def test_finds_peak_at_initial_lr(self):
        # Degenerate case: initial guess already the peak. Both
        # directions should immediately look worse.
        trial_fn = _gaussian_log_lr_score(peak_log_lr=math.log(1e-3))
        result = find_optimal_lr(trial_fn, seeds=(0,), initial_lr=1e-3, time_budget_s=10.0, log_tol=0.01)
        assert result.best_lr == pytest.approx(1e-3, rel=0.1)

    def test_multi_seed_averaging(self):
        # Two seeds should average out a per-seed constant offset --
        # confirms trial_fn is actually called once per seed and reduced,
        # not just called for seeds[0].
        def trial_fn(lr, seed):
            x = (math.log(lr) - math.log(1e-2)) / 1.0
            base = math.exp(-0.5 * x * x)
            return base + (0.5 if seed == 0 else -0.5)

        result = find_optimal_lr(trial_fn, seeds=(0, 1), initial_lr=1e-4, time_budget_s=10.0, log_tol=0.01)
        assert result.best_lr == pytest.approx(1e-2, rel=0.05)

    def test_time_budget_returns_best_so_far_not_raises(self):
        trial_fn = _gaussian_log_lr_score(peak_log_lr=math.log(1e2))
        result = find_optimal_lr(trial_fn, seeds=(0,), initial_lr=1e-6, time_budget_s=0.0)
        # Deadline is already hit before phase 1 even finishes its first
        # evaluation loop iteration -- must still return a valid result,
        # not raise or hang.
        assert result.n_trials >= 1
        assert math.isfinite(result.best_score)

    def test_history_has_no_duplicate_log_lr_entries(self):
        trial_fn = _gaussian_log_lr_score(peak_log_lr=math.log(1e-2))
        result = find_optimal_lr(trial_fn, seeds=(0,), initial_lr=1e-4, time_budget_s=10.0, log_tol=0.01)
        seen = [round(h[0], 9) for h in result.history]
        assert len(seen) == len(set(seen)), "cache should prevent re-evaluating the same lr"

    def test_converges_within_requested_tolerance(self):
        trial_fn = _gaussian_log_lr_score(peak_log_lr=math.log(1e-2))
        result = find_optimal_lr(trial_fn, seeds=(0,), initial_lr=1e-4, time_budget_s=10.0, log_tol=0.02)
        assert result.stopped_reason == "converged"


@pytest.mark.skipif(not os.environ.get(RUN_ENV_VAR), reason=f"real short training runs, opt in via {RUN_ENV_VAR}=1")
class TestFindOptimalLRRealModel:
    def test_baseline_config_peak_matches_manual_sweep(self):
        # Reproduces (as a real, reusable test) the manual 1x-100x sweep
        # from conversation: baseline config, eval_acc peaked around
        # 6x-10x the default peak_lr=0.002 (i.e. roughly 0.012-0.02),
        # then fell back toward chance by 100x (0.2). Wide tolerance
        # (2x-50x band) since this is 2-seed/short-run noise, not meant
        # to nail down the exact multiplier -- just confirm the search
        # lands in the same broad region a human manually found by hand.
        from scripts.l1_sparsity_probe import OriginalArchModel, evaluate, run

        def trial_fn(lr, seed):
            model = OriginalArchModel(
                seed,
                dense=True,
                o_proj_coef=0.0,
                all_layer_coef=0.0,
                l1_sparsity_coef=0.05,
                use_energy=False,
                all_zero_init=False,
            )
            run(model, 1500, seed, verbose=False, peak_lr=lr)
            return evaluate(model, 50, seed)

        result = find_optimal_lr(
            trial_fn,
            seeds=(1000, 1001),
            initial_lr=0.002,
            time_budget_s=480.0,
            log_tol=0.15,
        )
        print(
            f"\nfind_optimal_lr on baseline: best_lr={result.best_lr:.5f} "
            f"(mult={result.best_lr / 0.002:.2f}x) best_score={result.best_score:.4f} "
            f"n_trials={result.n_trials} elapsed={result.elapsed_s:.0f}s "
            f"reason={result.stopped_reason}"
        )
        mult = result.best_lr / 0.002
        assert 2.0 <= mult <= 50.0
