import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from model.toy_recall_models import (
    ToySmallTransformer, ToyTileRecurrence,
    cross_entropy_sum, predicted_token, apply_gradient_step,
    rmsnorm_tensor,
)
from model.toy_recall_task import generate_sequence
from sili.tensor import Tensor


VOCAB, HIDDEN, MLP_HIDDEN = 10, 12, 16


def _dense_model(n_layers=2, num_cpus=2):
    return ToySmallTransformer(VOCAB, HIDDEN, MLP_HIDDEN, n_layers,
                               max_weights=HIDDEN * MLP_HIDDEN, num_cpus=num_cpus)


def _tile_model(num_tiles=3, num_cpus=2):
    return ToyTileRecurrence(VOCAB, HIDDEN, MLP_HIDDEN, num_tiles,
                             max_weights=HIDDEN * MLP_HIDDEN, num_cpus=num_cpus)


class TestRMSNormTensor:
    def test_shape_preserved_and_finite(self):
        x = Tensor(np.random.RandomState(0).randn(5, HIDDEN).astype(np.float32))
        w = Tensor(np.ones(HIDDEN, dtype=np.float32))
        out = rmsnorm_tensor(x, w, eps=1e-6)
        assert out.data.shape == (5, HIDDEN)
        assert np.all(np.isfinite(out.data))

    def test_zero_input_does_not_produce_nan(self):
        x = Tensor(np.zeros((3, HIDDEN), dtype=np.float32))
        w = Tensor(np.ones(HIDDEN, dtype=np.float32))
        out = rmsnorm_tensor(x, w, eps=1e-6)
        assert np.all(np.isfinite(out.data))
        np.testing.assert_array_equal(out.data, np.zeros((3, HIDDEN), dtype=np.float32))


class TestToySmallTransformerForward:
    def test_shapes_and_finite(self):
        tf = _dense_model()
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits = tf.forward(embedded, learning_rate=0.0)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))

    def test_gradient_reaches_every_rmsnorm_leaf(self):
        tf = _dense_model()
        T = 4
        embedded = np.random.RandomState(2).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits = tf.forward(embedded, learning_rate=0.0)
        loss = cross_entropy_sum(logits, [(0, 1), (T - 1, 2)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        for p in tf.parameters():
            assert p.grad is not None
            assert np.all(np.isfinite(p.grad))

    def test_weights_actually_change_after_a_training_step(self):
        # DISLDOLayer's own weights aren't directly inspectable, but a
        # fixed probe input's output changing after one training step
        # is a direct behavioral proof the inline backward_dense update
        # actually happened, not just that .backward() ran without error.
        tf = _dense_model()
        probe = np.random.RandomState(3).randn(4, HIDDEN).astype(np.float32) * 0.1
        before = tf.forward(probe, learning_rate=0.0).data.copy()

        train_x = np.random.RandomState(4).randn(4, HIDDEN).astype(np.float32) * 0.1
        logits = tf.forward(train_x, learning_rate=0.05)
        loss = cross_entropy_sum(logits, [(0, 1), (1, 2), (2, 3), (3, 4)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        apply_gradient_step(tf.parameters(), lr=0.05)

        after = tf.forward(probe, learning_rate=0.0).data
        assert not np.allclose(before, after), "output on a fixed probe never changed after training"

    def test_loss_decreases_on_a_single_repeated_example(self):
        tf = _dense_model()
        T = 5
        embedded = np.random.RandomState(5).randn(T, HIDDEN).astype(np.float32) * 0.1
        targets = [2, 4, 1, 6, 0]
        pairs = list(enumerate(targets))
        lr = 0.01

        first_loss = None
        min_loss = None
        for step in range(150):
            logits = tf.forward(embedded, learning_rate=lr)
            loss = cross_entropy_sum(logits, pairs)
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            apply_gradient_step(tf.parameters(), lr=lr)
            if step == 0:
                first_loss = float(loss.data)
            min_loss = float(loss.data) if min_loss is None else min(min_loss, float(loss.data))

        # Track the MINIMUM loss seen, not the final step's value --
        # training is noisy near convergence (this is real gradient
        # descent on a recurrent-ish stack, not a toy convex problem),
        # so a strict final-step threshold is flaky by construction.
        assert min_loss < first_loss * 0.3, (
            f"loss barely moved: {first_loss:.3f} -> best {min_loss:.3f}")


class TestToyTileRecurrenceStep:
    def test_shapes_and_finite(self):
        model = _tile_model(num_tiles=3)
        x_window = np.random.RandomState(1).randn(3, HIDDEN).astype(np.float32) * 0.1
        M_prev = np.zeros((3, HIDDEN), dtype=np.float32)
        M_new, logits = model.step(x_window, M_prev, learning_rate=0.0)
        assert M_new.shape == (3, HIDDEN)
        assert logits.data.shape == (3, VOCAB)
        assert np.all(np.isfinite(M_new))
        assert np.all(np.isfinite(logits.data))

    def test_gradient_reaches_every_leaf_including_centers_and_log_sigmas(self):
        model = _tile_model(num_tiles=3)
        x_window = np.random.RandomState(2).randn(3, HIDDEN).astype(np.float32) * 0.1
        M_prev = np.zeros((3, HIDDEN), dtype=np.float32)
        _M_new, logits = model.step(x_window, M_prev, learning_rate=0.0)
        loss = cross_entropy_sum(logits, [(0, 1), (1, 2), (2, 3)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert np.all(np.isfinite(p.grad))

    def test_every_tile_column_target_contributes_gradient_not_just_the_last(self):
        # Confirms the reinstated staggered per-tile "column" loss is
        # actually wired in -- training on ONLY an early (non-last)
        # tile's own column target should still move the shared
        # weights (proof: a fixed probe's output changes), not just
        # the last tile's genuinely-novel-next-token target.
        model = _tile_model(num_tiles=3)
        probe_window = np.random.RandomState(3).randn(3, HIDDEN).astype(np.float32) * 0.1
        probe_M = np.zeros((3, HIDDEN), dtype=np.float32)
        before = model.step(probe_window, probe_M, learning_rate=0.0)[1].data.copy()

        train_window = np.random.RandomState(4).randn(3, HIDDEN).astype(np.float32) * 0.1
        train_M = np.zeros((3, HIDDEN), dtype=np.float32)
        _M_new, logits = model.step(train_window, train_M, learning_rate=0.05)
        loss = cross_entropy_sum(logits, [(0, 5)])  # ONLY tile 0's column target
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        apply_gradient_step(model.parameters(), lr=0.05)

        after = model.step(probe_window, probe_M, learning_rate=0.0)[1].data
        assert not np.allclose(before, after), (
            "training on only tile 0's column loss never changed the "
            "shared weights -- the per-tile loss isn't actually wired in")

    def test_loss_decreases_on_a_single_repeated_example(self):
        model = _tile_model(num_tiles=3)
        x_window = np.random.RandomState(5).randn(3, HIDDEN).astype(np.float32) * 0.1
        M_prev = np.zeros((3, HIDDEN), dtype=np.float32)
        targets = [1, 3, 5]
        pairs = list(enumerate(targets))
        lr = 0.01

        first_loss = None
        min_loss = None
        for step in range(200):
            _M_new, logits = model.step(x_window, M_prev, learning_rate=lr)
            loss = cross_entropy_sum(logits, pairs)
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            apply_gradient_step(model.parameters(), lr=lr)
            if step == 0:
                first_loss = float(loss.data)
            min_loss = float(loss.data) if min_loss is None else min(min_loss, float(loss.data))

        assert min_loss < first_loss * 0.3, (
            f"loss barely moved: {first_loss:.3f} -> best {min_loss:.3f}")

    def test_resetting_M_prev_changes_logits(self):
        # Statefulness check, same spirit as tile_recurrence.py's own
        # test_resetting_M_prev_changes_logits.
        model = _tile_model(num_tiles=3)
        x_window = np.random.RandomState(6).randn(3, HIDDEN).astype(np.float32) * 0.1
        M_real = np.random.RandomState(7).randn(3, HIDDEN).astype(np.float32)
        M_zero = np.zeros((3, HIDDEN), dtype=np.float32)

        _M_a, logits_a = model.step(x_window, M_real, learning_rate=0.0)
        _M_b, logits_b = model.step(x_window, M_zero, learning_rate=0.0)

        assert not np.allclose(logits_a.data, logits_b.data), (
            "different M_prev produced identical logits")


class TestApplyGradientStep:
    def test_updates_and_zeroes_only_leaves_with_grad(self):
        a = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        b = Tensor(np.array([5.0], dtype=np.float32))
        a.grad = np.array([0.1, -0.1], dtype=np.float32)
        # b.grad left as None -- should be skipped, not crash
        apply_gradient_step([a, b], lr=1.0)
        np.testing.assert_allclose(a.data, [0.9, 2.1])
        assert a.grad is None
        np.testing.assert_allclose(b.data, [5.0])
        assert b.grad is None
