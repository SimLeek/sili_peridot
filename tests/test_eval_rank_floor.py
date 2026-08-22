"""Tests for model/eval_rank_floor.py.

Fast tests check permutation_matrix/eckart_young_floor against known
closed-form properties, and confirm the training harness itself can
reach the predicted floor (LowRankDenseLayer) and predicted ~0 ceiling
(FullRankDenseLayer) before trusting any real-FP4-layer comparison
against them. One opt-in test runs the real 3-arm comparison
(LowRankDenseLayer vs DISLDOLayerDeterministic vs DISLDOLayer32) this
module was actually built for.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from model.eval_rank_floor import (
    FullRankDenseLayer, LowRankDenseLayer, eckart_young_floor,
    measure_rank_floor, permutation_matrix,
)
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_

RUN_ENV_VAR = "SILI_RUN_LR_SEARCH"


class TestPermutationMatrix:
    def test_is_a_permutation(self):
        P = permutation_matrix(6, shift=1)
        assert np.allclose(P @ P.T, np.eye(6))
        assert np.allclose(P.sum(axis=0), 1.0)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_shift_maps_correct_slot(self):
        P = permutation_matrix(5, shift=2)
        # row i should have its 1 at column (i+2)%5
        for i in range(5):
            assert P[i, (i + 2) % 5] == 1.0

    def test_rejects_invalid_shift(self):
        with pytest.raises(ValueError):
            permutation_matrix(4, shift=0)
        with pytest.raises(ValueError):
            permutation_matrix(4, shift=4)


class TestEckartYoungFloor:
    def test_full_rank_floor_is_zero(self):
        P = permutation_matrix(8, shift=1)
        assert eckart_young_floor(P, rank=8) == pytest.approx(0.0, abs=1e-8)

    def test_permutation_floor_matches_closed_form(self):
        # Orthogonal n x n permutation matrix: every singular value is
        # exactly 1, so the rank-k floor is exactly (n - k).
        n = 8
        P = permutation_matrix(n, shift=1)
        for k in range(1, n):
            assert eckart_young_floor(P, rank=k) == pytest.approx(n - k, abs=1e-6)

    def test_zero_rank_floor_equals_total_energy(self):
        n = 6
        P = permutation_matrix(n, shift=1)
        assert eckart_young_floor(P, rank=0) == pytest.approx(n, abs=1e-6)


class TestMeasureRankFloorHarnessSanity:
    """Before trusting any FP4-vs-floor comparison, confirm the harness
    itself can hit a known answer with a plain (non-quantized) layer."""

    def test_low_rank_dense_layer_reaches_its_own_floor(self):
        n = 6
        rank = 2
        target = permutation_matrix(n, shift=1)
        rng = np.random.default_rng(0)
        layer = LowRankDenseLayer(n, n, rank, rng)
        opt = AdamOptimizer()
        report = measure_rank_floor(
            layer, target, n_steps=300, lr=0.05, rank=rank,
            opt=opt, opt_step=lambda o, p, lr: o.step(p, lr=lr),
            clip_grad_norm=clip_grad_norm_,
        )
        assert report.ey_floor == pytest.approx(n - rank, abs=1e-6)
        # Should land close to (not below -- rank-limited) the floor.
        assert report.ratio_to_floor == pytest.approx(1.0, abs=0.15)
        assert not report.beat_floor

    def test_full_rank_dense_layer_reaches_near_zero(self):
        n = 6
        target = permutation_matrix(n, shift=1)
        rng = np.random.default_rng(1)
        layer = FullRankDenseLayer(n, n, rng)
        opt = AdamOptimizer()
        report = measure_rank_floor(
            layer, target, n_steps=300, lr=0.05, rank=n,
            opt=opt, opt_step=lambda o, p, lr: o.step(p, lr=lr),
            clip_grad_norm=clip_grad_norm_,
        )
        assert report.ey_floor == pytest.approx(0.0, abs=1e-6)
        assert report.final_sse < 0.1


@pytest.mark.skipif(not os.environ.get(RUN_ENV_VAR),
                    reason=f"real short training run, opt in via {RUN_ENV_VAR}=1")
class TestRankFloorRealModel:
    def test_disldo_vs_low_rank_vs_float32_at_scale_rank_2(self):
        from sili.sparse_rnn import DISLDOLayer, DISLDOLayer32, DISLDOLayerDeterministic

        n = 8
        rank = 2
        target = permutation_matrix(n, shift=1)
        n_steps = 400
        lr = 0.05

        rng = np.random.default_rng(2000)
        low_rank_layer = LowRankDenseLayer(n, n, rank, rng)
        low_rank_report = measure_rank_floor(
            low_rank_layer, target, n_steps, lr, rank,
            opt=AdamOptimizer(), opt_step=lambda o, p, l: o.step(p, lr=l),
            clip_grad_norm=clip_grad_norm_)

        fp4_det_layer = DISLDOLayerDeterministic(n, n, n * n, num_cpus=1,
                                                  rng=np.random.default_rng(2001),
                                                  dense=True, scale_rank=rank)
        fp4_det_report = measure_rank_floor(fp4_det_layer, target, n_steps, lr, rank)

        # Stochastic FP4 too -- per this same session's earlier finding
        # (deterministic rounding is exactly the variant that gets stuck on
        # already-live synapses), deterministic alone isn't the relevant
        # comparison for "can real FP4 weights reach a rank-limited target."
        fp4_stoch_layer = DISLDOLayer(n, n, n * n, num_cpus=1,
                                      rng=np.random.default_rng(2007),
                                      dense=True, scale_rank=rank)
        fp4_stoch_report = measure_rank_floor(fp4_stoch_layer, target, n_steps, lr, rank)

        # DISLDOLayer32 has no dense=/scale_rank= kwargs (no quantization
        # at all, so no scale envelope to constrain) -- max_weights=n*n
        # still reaches full density via _preseed_random_sparse's own
        # budget-saturated path, just not the dedicated dense=True code path.
        fp32_layer = DISLDOLayer32(n, n, n * n, num_cpus=1,
                                   rng=np.random.default_rng(2002))
        fp32_report = measure_rank_floor(fp32_layer, target, n_steps, lr, rank)

        print(f"\nrank-floor @ n={n} rank={rank} steps={n_steps}: "
              f"ey_floor={low_rank_report.ey_floor:.3f}\n"
              f"  low_rank_dense: final_sse={low_rank_report.final_sse:.3f} "
              f"ratio={low_rank_report.ratio_to_floor:.3f}\n"
              f"  fp4_deterministic: final_sse={fp4_det_report.final_sse:.3f} "
              f"ratio={fp4_det_report.ratio_to_floor:.3f} beat_floor={fp4_det_report.beat_floor}\n"
              f"  fp4_stochastic: final_sse={fp4_stoch_report.final_sse:.3f} "
              f"ratio={fp4_stoch_report.ratio_to_floor:.3f} beat_floor={fp4_stoch_report.beat_floor}\n"
              f"  fp32_reference: final_sse={fp32_report.final_sse:.3f} "
              f"ratio={fp32_report.ratio_to_floor:.3f} beat_floor={fp32_report.beat_floor}")

        assert np.isfinite(fp4_det_report.final_sse)
        assert np.isfinite(fp4_stoch_report.final_sse)
        assert np.isfinite(fp32_report.final_sse)
        # float32 reference should always be able to beat a rank-limited
        # floor (nothing there to get stuck on) -- this is the one hard
        # assertion; the FP4-vs-floor relationship is the open question
        # this test exists to observe, not assert a direction on.
        assert fp32_report.beat_floor
