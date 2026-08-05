import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from model.toy_precision_models import (
    FP4_TABLE, fake_quantize_fp4, ArtificialFP4Linear,
    ToySmallTransformerArtificialFP4, ToySmallTransformerRealFP4,
    AdamRowScaleDISLDOLayer, ToySmallTransformerRealFP4RowScaleAdam,
    AdamRank1DISLDOLayer, ToySmallTransformerRealFP4Rank1Adam,
    row_scale_fake_quantize, rank1_fake_quantize, QuantizedDISLDOLayer32,
    ToySmallTransformerFP32Ref, ToySmallTransformerQuant8Rank1,
    ToySmallTransformerQuant4Rank1,
    _PeakEligibilityTrace, PeakEligibilityDISLDOLayer,
)
from model.toy_recall_models import cross_entropy_sum, AdamOptimizer
from sili.tensor import Tensor


VOCAB, HIDDEN, MLP_HIDDEN = 10, 12, 16
MAX_WEIGHTS = HIDDEN * MLP_HIDDEN  # generous, fully-dense-capable at this toy scale


def _fp4_model(n_layers=2, num_cpus=2, use_energy=False):
    return ToySmallTransformerArtificialFP4(VOCAB, HIDDEN, MLP_HIDDEN, n_layers,
                                            use_energy=use_energy, num_cpus=num_cpus)


def _real_fp4_model(n_layers=2, num_cpus=2, use_energy=False):
    return ToySmallTransformerRealFP4(VOCAB, HIDDEN, MLP_HIDDEN, n_layers,
                                      MAX_WEIGHTS, use_energy=use_energy, num_cpus=num_cpus)


def _row_scale_adam_model(n_layers=2, num_cpus=2, use_energy=False):
    return ToySmallTransformerRealFP4RowScaleAdam(VOCAB, HIDDEN, MLP_HIDDEN, n_layers,
                                                   MAX_WEIGHTS, use_energy=use_energy,
                                                   num_cpus=num_cpus)


class TestFakeQuantizeFp4:
    def test_forward_values_are_from_the_real_table_times_row_scale(self):
        w = Tensor(np.array([[0.1, -0.2, 5.9, -6.1]], dtype=np.float32))
        out = fake_quantize_fp4(w)
        row_scale = np.max(np.abs(w.data), axis=-1, keepdims=True) / 6.0
        possible = FP4_TABLE * row_scale
        for v in out.data.flatten():
            assert np.any(np.isclose(v, possible)), f"{v} not a real FP4 level * row_scale"

    def test_zero_row_does_not_produce_nan(self):
        w = Tensor(np.zeros((2, 4), dtype=np.float32))
        out = fake_quantize_fp4(w)
        assert np.all(np.isfinite(out.data))
        np.testing.assert_array_equal(out.data, np.zeros((2, 4), dtype=np.float32))

    def test_backward_is_straight_through_identity(self):
        w = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        out = fake_quantize_fp4(w)
        out.grad = np.array([[0.5, -0.5, 2.0]], dtype=np.float32)
        out.backward()
        np.testing.assert_allclose(w.grad, [[0.5, -0.5, 2.0]])


class TestArtificialFP4Linear:
    def test_forward_shape_and_finite(self):
        layer = ArtificialFP4Linear(HIDDEN, VOCAB)
        x = Tensor(np.random.RandomState(0).randn(4, HIDDEN).astype(np.float32))
        out = layer.forward(x)
        assert out.data.shape == (4, VOCAB)
        assert np.all(np.isfinite(out.data))

    def test_gradient_reaches_the_fp32_master_weight(self):
        layer = ArtificialFP4Linear(HIDDEN, VOCAB)
        x = Tensor(np.random.RandomState(1).randn(4, HIDDEN).astype(np.float32))
        out = layer.forward(x)
        out.grad = np.ones_like(out.data)
        out.backward()
        assert layer.weight.grad is not None
        assert np.all(np.isfinite(layer.weight.grad))


class TestToySmallTransformerArtificialFP4:
    def test_shapes_and_finite_no_energy(self):
        model = _fp4_model(use_energy=False)
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits, aux_loss = model.forward(embedded)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None

    def test_shapes_and_finite_with_energy(self):
        model = _fp4_model(use_energy=True)
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits, aux_loss = model.forward(embedded)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is not None
        assert np.isfinite(float(aux_loss.data))

    def test_loss_decreases_on_a_single_repeated_example(self):
        model = _fp4_model(use_energy=False)
        opt = AdamOptimizer()
        T = 5
        embedded = np.random.RandomState(5).randn(T, HIDDEN).astype(np.float32) * 0.1
        pairs = list(enumerate([2, 4, 1, 6, 0]))
        lr = 0.02

        first_loss = None
        min_loss = None
        for step in range(150):
            logits, _aux = model.forward(embedded)
            loss = cross_entropy_sum(logits, pairs)
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            opt.step(model.parameters(), lr=lr)
            if step == 0:
                first_loss = float(loss.data)
            min_loss = float(loss.data) if min_loss is None else min(min_loss, float(loss.data))

        assert min_loss < first_loss * 0.3, (
            f"loss barely moved: {first_loss:.3f} -> best {min_loss:.3f}")


class TestToySmallTransformerRealFP4:
    def test_shapes_and_finite_no_energy(self):
        model = _real_fp4_model(use_energy=False)
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits, aux_loss = model.forward(embedded, learning_rate=0.01)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None

    def test_shapes_and_finite_with_energy(self):
        model = _real_fp4_model(use_energy=True)
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits, aux_loss = model.forward(embedded, learning_rate=0.01)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is not None
        assert np.isfinite(float(aux_loss.data))

    def test_big_weights_change_after_a_step_with_no_external_optimizer(self):
        # DISLDOLayer's own weights self-update inline during backward()
        # -- confirms that actually fires, with NOTHING but
        # loss.backward() called (no AdamOptimizer.step() on the big
        # weight matrices at all).
        model = _real_fp4_model(use_energy=False)
        probe = np.random.RandomState(3).randn(4, HIDDEN).astype(np.float32) * 0.1
        before = model.forward(probe, learning_rate=0.0)[0].data.copy()

        train_x = np.random.RandomState(4).randn(4, HIDDEN).astype(np.float32) * 0.1
        logits, _aux = model.forward(train_x, learning_rate=0.05)
        loss = cross_entropy_sum(logits, [(0, 1), (1, 2), (2, 3), (3, 4)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()

        after = model.forward(probe, learning_rate=0.0)[0].data
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after), (
            "output on a fixed probe never changed -- DISLDOLayer's inline "
            "update never fired")

    def test_leaf_parameters_are_only_rmsnorm_weights(self):
        model = _real_fp4_model(n_layers=2)
        params = model.parameters_for_optimizer()
        assert len(params) == 4  # input_ln + post_ln per layer, 2 layers
        for p in params:
            assert p.data.shape == (HIDDEN,)


class TestAdamRowScaleDISLDOLayer:
    def test_forward_shape_and_finite(self):
        layer = AdamRowScaleDISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        x = np.random.RandomState(0).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.01)
        assert out.data.shape == (4, VOCAB)
        assert np.all(np.isfinite(out.data))

    def test_value_scale_changes_after_a_training_step(self):
        layer = AdamRowScaleDISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        before = layer._row_scales().copy()
        x = np.random.RandomState(1).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.05)
        out.grad = np.ones_like(out.data)
        out.backward()
        after = layer._row_scales()
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after), (
            "value_scale never changed -- Adam row-scale re-normalization never fired")

    def test_eval_call_with_zero_learning_rate_does_not_change_value_scale(self):
        layer = AdamRowScaleDISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        before = layer._row_scales().copy()
        x = np.random.RandomState(2).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.0)
        out.grad = np.ones_like(out.data)
        out.backward()
        after = layer._row_scales()
        np.testing.assert_allclose(before, after)

    def test_no_external_optimizer_parameters(self):
        layer = AdamRowScaleDISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        assert layer.parameters() == []


class TestToySmallTransformerRealFP4RowScaleAdam:
    def test_shapes_and_finite(self):
        model = _row_scale_adam_model(use_energy=False)
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits, aux_loss = model.forward(embedded, learning_rate=0.01)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None

    def test_big_weights_change_after_a_step_with_no_external_optimizer(self):
        model = _row_scale_adam_model(use_energy=False)
        probe = np.random.RandomState(3).randn(4, HIDDEN).astype(np.float32) * 0.1
        before = model.forward(probe, learning_rate=0.0)[0].data.copy()

        train_x = np.random.RandomState(4).randn(4, HIDDEN).astype(np.float32) * 0.1
        logits, _aux = model.forward(train_x, learning_rate=0.05)
        loss = cross_entropy_sum(logits, [(0, 1), (1, 2), (2, 3), (3, 4)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()

        after = model.forward(probe, learning_rate=0.0)[0].data
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after)


class TestAdamRank1DISLDOLayer:
    def test_forward_shape_and_finite(self):
        layer = AdamRank1DISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        x = np.random.RandomState(0).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.01)
        assert out.data.shape == (4, VOCAB)
        assert np.all(np.isfinite(out.data))

    def test_row_and_column_scale_both_change_after_a_training_step(self):
        layer = AdamRank1DISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        before_row = layer._row_scales().copy()
        before_col = layer._col_scales().copy()
        x = np.random.RandomState(1).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.05)
        out.grad = np.ones_like(out.data)
        out.backward()
        after_row = layer._row_scales()
        after_col = layer._col_scales()
        assert np.all(np.isfinite(after_row)) and np.all(np.isfinite(after_col))
        assert not np.allclose(before_row, after_row), (
            "value_scale never changed -- Adam rank-1 row re-normalization never fired")
        assert not np.allclose(before_col, after_col), (
            "output_scale never changed -- Adam rank-1 column re-normalization never fired "
            "(check set_output_scale_raw was called at init to activate its training)")

    def test_column_scale_starts_at_one(self):
        layer = AdamRank1DISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        np.testing.assert_allclose(layer._col_scales(), np.ones(VOCAB, dtype=np.float32))

    def test_eval_call_with_zero_learning_rate_does_not_change_scales(self):
        layer = AdamRank1DISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        before_row = layer._row_scales().copy()
        before_col = layer._col_scales().copy()
        x = np.random.RandomState(2).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.0)
        out.grad = np.ones_like(out.data)
        out.backward()
        np.testing.assert_allclose(before_row, layer._row_scales())
        np.testing.assert_allclose(before_col, layer._col_scales())

    def test_no_external_optimizer_parameters(self):
        layer = AdamRank1DISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        assert layer.parameters() == []


class TestToySmallTransformerRealFP4Rank1Adam:
    def test_shapes_and_finite(self):
        model = ToySmallTransformerRealFP4Rank1Adam(VOCAB, HIDDEN, MLP_HIDDEN, 2,
                                                     MAX_WEIGHTS, use_energy=False, num_cpus=2)
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits, aux_loss = model.forward(embedded, learning_rate=0.01)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None

    def test_big_weights_change_after_a_step_with_no_external_optimizer(self):
        model = ToySmallTransformerRealFP4Rank1Adam(VOCAB, HIDDEN, MLP_HIDDEN, 2,
                                                     MAX_WEIGHTS, use_energy=False, num_cpus=2)
        probe = np.random.RandomState(3).randn(4, HIDDEN).astype(np.float32) * 0.1
        before = model.forward(probe, learning_rate=0.0)[0].data.copy()

        train_x = np.random.RandomState(4).randn(4, HIDDEN).astype(np.float32) * 0.1
        logits, _aux = model.forward(train_x, learning_rate=0.05)
        loss = cross_entropy_sum(logits, [(0, 1), (1, 2), (2, 3), (3, 4)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()

        after = model.forward(probe, learning_rate=0.0)[0].data
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after)


class TestRowScaleFakeQuantize:
    def test_reproduces_input_exactly_at_16_bits(self):
        # 16 signed levels' worth of headroom is plenty to round-trip
        # small random values without a visible quantization error.
        rng = np.random.RandomState(0)
        ptrs = np.array([0, 3, 3, 6], dtype=np.int32)
        vals = (rng.randn(6) * 0.1).astype(np.float32)
        out = row_scale_fake_quantize(vals, ptrs, bits=16)
        np.testing.assert_allclose(out, vals, atol=1e-3)

    def test_empty_row_is_a_no_op(self):
        ptrs = np.array([0, 0, 2], dtype=np.int32)
        vals = np.array([1.0, -1.0], dtype=np.float32)
        out = row_scale_fake_quantize(vals, ptrs, bits=8)
        assert np.all(np.isfinite(out))


class TestRank1FakeQuantize:
    def test_output_is_finite_and_bounded_by_input_envelope(self):
        rng = np.random.RandomState(0)
        n_in, n_out, k = 5, 6, 3
        ptrs = np.arange(0, (n_in + 1) * k, k).astype(np.int32)
        indices = np.concatenate([rng.choice(n_out, size=k, replace=False)
                                  for _ in range(n_in)]).astype(np.int32)
        vals = (rng.randn(n_in * k) * 0.5).astype(np.float32)
        out = rank1_fake_quantize(vals, ptrs, indices, n_out, bits=8)
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= np.max(np.abs(vals)) * 1.5


class TestQuantizedDISLDOLayer32:
    def test_forward_shape_and_finite(self):
        layer = QuantizedDISLDOLayer32(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        x = np.random.RandomState(0).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.01)
        assert out.data.shape == (4, VOCAB)
        assert np.all(np.isfinite(out.data))

    def test_weights_and_importance_stay_finite_after_training_and_quantization(self):
        layer = QuantizedDISLDOLayer32(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2,
                                       bits=8, scheme="rank1", quantize_importance=True)
        x = np.random.RandomState(1).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.05)
        out.grad = np.ones_like(out.data)
        out.backward()
        c = layer._inner._c
        assert np.all(np.isfinite(c.weights_vals))
        assert np.all(np.isfinite(c.importance))

    def test_eval_call_with_zero_learning_rate_does_not_quantize(self):
        layer = QuantizedDISLDOLayer32(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        before = np.array(layer._inner._c.weights_vals, copy=True)
        x = np.random.RandomState(2).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.0)
        out.grad = np.ones_like(out.data)
        out.backward()
        after = layer._inner._c.weights_vals
        np.testing.assert_allclose(before, after)

    def test_no_external_optimizer_parameters(self):
        layer = QuantizedDISLDOLayer32(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        assert layer.parameters() == []

    @pytest.mark.parametrize("bits,scheme", [(8, "row"), (4, "row"), (8, "rank1"), (4, "rank1")])
    def test_all_bit_width_and_scheme_combinations_stay_finite(self, bits, scheme):
        layer = QuantizedDISLDOLayer32(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2,
                                       bits=bits, scheme=scheme, quantize_importance=True)
        rng = np.random.RandomState(bits + (0 if scheme == "row" else 100))
        for _ in range(5):
            x = rng.randn(4, HIDDEN).astype(np.float32)
            out = layer.forward(x, learning_rate=0.05)
            out.grad = np.ones_like(out.data)
            out.backward()
        assert np.all(np.isfinite(layer._inner._c.weights_vals))
        assert np.all(np.isfinite(layer._inner._c.importance))


class TestToySmallTransformerQuant8Rank1:
    def test_shapes_and_finite(self):
        model = ToySmallTransformerQuant8Rank1(VOCAB, HIDDEN, MLP_HIDDEN, 2,
                                               MAX_WEIGHTS, use_energy=False, num_cpus=2)
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits, aux_loss = model.forward(embedded, learning_rate=0.01)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None

    def test_big_weights_change_after_a_step_with_no_external_optimizer(self):
        model = ToySmallTransformerQuant8Rank1(VOCAB, HIDDEN, MLP_HIDDEN, 2,
                                               MAX_WEIGHTS, use_energy=False, num_cpus=2)
        probe = np.random.RandomState(3).randn(4, HIDDEN).astype(np.float32) * 0.1
        before = model.forward(probe, learning_rate=0.0)[0].data.copy()

        train_x = np.random.RandomState(4).randn(4, HIDDEN).astype(np.float32) * 0.1
        logits, _aux = model.forward(train_x, learning_rate=0.01)
        loss = cross_entropy_sum(logits, [(0, 1), (1, 2), (2, 3), (3, 4)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()

        after = model.forward(probe, learning_rate=0.0)[0].data
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after)


class TestToySmallTransformerFP32Ref:
    def test_shapes_and_finite(self):
        model = ToySmallTransformerFP32Ref(VOCAB, HIDDEN, MLP_HIDDEN, 2,
                                           MAX_WEIGHTS, use_energy=False, num_cpus=2)
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits, aux_loss = model.forward(embedded, learning_rate=0.01)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None


class TestToySmallTransformerQuant4Rank1:
    def test_shapes_and_finite(self):
        model = ToySmallTransformerQuant4Rank1(VOCAB, HIDDEN, MLP_HIDDEN, 2,
                                               MAX_WEIGHTS, use_energy=False, num_cpus=2)
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits, aux_loss = model.forward(embedded, learning_rate=0.01)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None


class TestPeakEligibilityTrace:
    def test_replaces_when_new_input_exceeds_decayed_peak(self):
        trace = _PeakEligibilityTrace(shape=(1, 4), decay=0.9)
        x1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        x2 = np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        trace.update(x1)
        p2 = trace.update(x2)
        np.testing.assert_allclose(p2[0, 0], 5.0)  # replaced by the bigger new value

    def test_keeps_decayed_peak_when_new_input_is_smaller(self):
        trace = _PeakEligibilityTrace(shape=(1, 4), decay=0.9)
        x1 = np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        x2 = np.array([[0.1, 0.0, 0.0, 0.0]], dtype=np.float32)
        trace.update(x1)
        p2 = trace.update(x2)
        np.testing.assert_allclose(p2[0, 0], 0.9 * 5.0)  # decayed, not overwritten by the small new value

    def test_preserves_sign(self):
        trace = _PeakEligibilityTrace(shape=(1, 2), decay=0.9)
        p = trace.update(np.array([[-5.0, 3.0]], dtype=np.float32))
        assert p[0, 0] < 0
        assert p[0, 1] > 0

    def test_peak_eventually_fades_without_replacement(self):
        trace = _PeakEligibilityTrace(shape=(1, 1), decay=0.5)
        trace.update(np.array([[4.0]], dtype=np.float32))
        for _ in range(10):
            p = trace.update(np.array([[0.0]], dtype=np.float32))
        assert abs(p[0, 0]) < 1e-2  # decayed toward 0 after enough silent ticks


class TestPeakEligibilityDISLDOLayer:
    def test_forward_shape_and_finite(self):
        layer = PeakEligibilityDISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        x = np.random.RandomState(0).randn(4, HIDDEN).astype(np.float32)
        out = layer.forward(x, learning_rate=0.01)
        assert out.data.shape == (4, VOCAB)
        assert np.all(np.isfinite(out.data))

    def test_trace_updates_on_every_forward_even_without_backward(self):
        layer = PeakEligibilityDISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        x = np.random.RandomState(1).randn(4, HIDDEN).astype(np.float32)
        layer.forward(x, learning_rate=0.01)  # no .backward() called
        assert layer.trace is not None
        assert not np.allclose(layer.trace.peak, 0.0), (
            "peak trace never updated on a forward-only call")

    def test_value_scale_only_changes_when_backward_actually_fires(self):
        layer = PeakEligibilityDISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        before = np.array([layer._inner._c.get_value_scale(r) for r in range(HIDDEN)])
        x = np.random.RandomState(2).randn(4, HIDDEN).astype(np.float32)
        layer.forward(x, learning_rate=0.01)  # forward only, no backward
        no_backward = np.array([layer._inner._c.get_value_scale(r) for r in range(HIDDEN)])
        np.testing.assert_allclose(before, no_backward)

        out = layer.forward(x, learning_rate=0.05)
        out.grad = np.ones_like(out.data)
        out.backward()
        after_backward = np.array([layer._inner._c.get_value_scale(r) for r in range(HIDDEN)])
        assert np.all(np.isfinite(after_backward))
        assert not np.allclose(before, after_backward)

    def test_row_silent_at_query_tick_still_gets_credited_from_an_earlier_peak(self):
        # The actual point of this mechanism: a plain DISLDOLayer gives
        # EXACTLY zero credit to a row whose current input is zero
        # (g = dy*iv in the C++, structurally zero when iv=0) -- this
        # class must do better by crediting the row's remembered peak.
        layer = PeakEligibilityDISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        rng = np.random.RandomState(3)

        x_peak = np.zeros((1, HIDDEN), dtype=np.float32)
        x_peak[0, 0] = 5.0
        layer.forward(x_peak, learning_rate=0.0)  # row 0 peaks here
        for _ in range(3):
            x = (rng.randn(1, HIDDEN) * 0.05).astype(np.float32)
            x[0, 0] = 0.0  # row 0 silent every tick after its peak
            layer.forward(x, learning_rate=0.0)

        before = layer._inner._c.get_value_scale(0)
        x_query = (rng.randn(1, HIDDEN) * 0.05).astype(np.float32)
        x_query[0, 0] = 0.0  # silent at the query tick too
        out = layer.forward(x_query, learning_rate=0.05)
        out.grad = np.ones_like(out.data)
        out.backward()
        after = layer._inner._c.get_value_scale(0)

        assert after != before, (
            "row 0 was silent at the query tick but had a real earlier peak -- "
            "plain DISLDO would give it exactly zero credit; this must not")

    def test_no_external_optimizer_parameters(self):
        layer = PeakEligibilityDISLDOLayer(HIDDEN, VOCAB, MAX_WEIGHTS, num_cpus=2)
        assert layer.parameters() == []
