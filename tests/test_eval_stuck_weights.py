"""Tests for model/eval_stuck_weights.py.

Fast tests build synthetic (weights, importance) snapshot pairs with a
KNOWN stuck/non-stuck relationship (some synapses deliberately high-
importance-but-frozen, others moving normally) to check the percentile-
based detection logic. One opt-in test wires snapshot_multi_digit_state
to the actual `baseline` OriginalArchModel's q_proj before/after a real
(calibrated-lr) training interval.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from model.eval_stuck_weights import (
    check_stuck_weights, snapshot_layer_state, snapshot_multi_digit_state,
)

RUN_ENV_VAR = "SILI_RUN_LR_SEARCH"


def _snapshot(weights, importance):
    return [(np.asarray(weights, dtype=np.float64), np.asarray(importance, dtype=np.float64))]


class TestCheckStuckWeightsSynthetic:
    def test_detects_deliberately_frozen_high_importance_synapses(self):
        rng = np.random.default_rng(0)
        n = 200
        importance = rng.uniform(0, 1, n)
        w_before = rng.standard_normal(n)
        w_after = w_before.copy()

        # Bottom 3/4 of synapses (by importance) move normally.
        low_imp_mask = importance < np.percentile(importance, 75)
        w_after[low_imp_mask] += rng.standard_normal(low_imp_mask.sum()) * 0.5

        # Top quartile by importance: half move normally too (should NOT
        # be flagged), half are DELIBERATELY frozen (should be flagged).
        high_imp_idx = np.where(~low_imp_mask)[0]
        rng.shuffle(high_imp_idx)
        half = len(high_imp_idx) // 2
        frozen_idx = high_imp_idx[:half]
        moving_high_idx = high_imp_idx[half:]
        w_after[moving_high_idx] += rng.standard_normal(len(moving_high_idx)) * 0.5
        # frozen_idx: w_after left exactly equal to w_before -- truly stuck.

        before = _snapshot(w_before, importance)
        after = _snapshot(w_after, importance)
        report = check_stuck_weights(before, after)

        assert report.n_synapses == n
        assert report.n_stuck > 0
        # Real signal: stuck synapses concentrated among the high-
        # importance group far beyond chance.
        assert report.excess_stuck_ratio > 1.5

    def test_no_relationship_gives_excess_ratio_near_one(self):
        # Importance and movement completely UNRELATED (independent
        # random draws) -- excess_stuck_ratio should sit near 1.0 (no
        # more overlap than chance), not flag a false pathology.
        rng = np.random.default_rng(1)
        n = 5000
        importance = rng.uniform(0, 1, n)
        w_before = rng.standard_normal(n)
        w_after = w_before + rng.standard_normal(n) * 0.3  # movement independent of importance
        before = _snapshot(w_before, importance)
        after = _snapshot(w_after, importance)
        report = check_stuck_weights(before, after)
        assert report.excess_stuck_ratio == pytest.approx(1.0, abs=0.3)

    def test_raises_on_synapse_count_mismatch(self):
        before = _snapshot([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        after = _snapshot([1.0, 2.0], [0.1, 0.2])
        with pytest.raises(ValueError, match="mismatch"):
            check_stuck_weights(before, after)

    def test_pools_multiple_layers(self):
        rng = np.random.default_rng(2)
        layer_a_before = _snapshot(rng.standard_normal(50), rng.uniform(0, 1, 50))
        layer_b_before = _snapshot(rng.standard_normal(50), rng.uniform(0, 1, 50))
        before = layer_a_before + layer_b_before
        after = [(w, i) for w, i in before]  # no movement anywhere -- everyone "stuck"
        report = check_stuck_weights(before, after)
        assert report.n_synapses == 100

    def test_zero_movement_everywhere_flags_all_high_importance(self):
        rng = np.random.default_rng(3)
        n = 100
        importance = rng.uniform(0, 1, n)
        w = rng.standard_normal(n)
        before = _snapshot(w, importance)
        after = _snapshot(w.copy(), importance)  # bit-identical -- nothing moved
        report = check_stuck_weights(before, after)
        # Every high-importance synapse has delta_w=0 <= move_thresh
        # (which is also 0 here, since >=25% of ALL deltas are exactly
        # 0) -- essentially all of them flagged.
        assert report.stuck_fraction == pytest.approx(1.0)


@pytest.mark.skipif(not os.environ.get(RUN_ENV_VAR),
                    reason=f"real short training run, opt in via {RUN_ENV_VAR}=1")
class TestStuckWeightsRealModel:
    def test_baseline_config_before_after_training(self):
        from scripts.l1_sparsity_probe import OriginalArchModel, run

        model = OriginalArchModel(
            1000, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
            l1_sparsity_coef=0.05, use_energy=False, all_zero_init=False,
        )
        before = snapshot_multi_digit_state(model.q_proj)
        run(model, 1500, 1000, verbose=False, peak_lr=0.0483)
        after = snapshot_multi_digit_state(model.q_proj)

        report = check_stuck_weights(before, after)
        print(f"\nstuck-weights report (q_proj, 1500 steps @ calibrated lr): "
              f"n_synapses={report.n_synapses} n_high_importance={report.n_high_importance} "
              f"stuck_fraction={report.stuck_fraction:.3f} "
              f"excess_stuck_ratio={report.excess_stuck_ratio:.2f} "
              f"mean_delta_w_high_imp={report.mean_delta_w_for_high_importance:.5f} "
              f"mean_delta_w_overall={report.mean_delta_w_overall:.5f}")
        assert report.n_synapses > 0
        assert np.isfinite(report.stuck_fraction)
        assert np.isfinite(report.excess_stuck_ratio)
