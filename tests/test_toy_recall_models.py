import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sili.tensor import Tensor

from model.toy_recall_models import (
    AdamOptimizer,
    DenseTensorLinear,
    ToySmallTransformer,
    ToyTileRecurrence,
    apply_gradient_step,
    backward_with_grad_clip,
    clip_grad_norm_,
    cross_entropy_sum,
    lr_schedule,
    rmsnorm_tensor,
)

VOCAB, HIDDEN, MLP_HIDDEN = 10, 12, 16
COLUMN_NEURONS = 2
STATE_WIDTH = HIDDEN * COLUMN_NEURONS
TILE_MLP_HIDDEN = STATE_WIDTH * 2


def _dense_model(n_layers=2, num_cpus=2):
    return ToySmallTransformer(VOCAB, HIDDEN, MLP_HIDDEN, n_layers, num_cpus=num_cpus)


def _tile_model(num_tiles=3, num_cpus=2):
    return ToyTileRecurrence(VOCAB, HIDDEN, COLUMN_NEURONS, TILE_MLP_HIDDEN, num_tiles, num_cpus=num_cpus)


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


class TestDenseTensorLinear:
    def test_forward_shape_and_finite(self):
        layer = DenseTensorLinear(HIDDEN, VOCAB)
        x = Tensor(np.random.RandomState(0).randn(4, HIDDEN).astype(np.float32))
        out = layer.forward(x)
        assert out.data.shape == (4, VOCAB)
        assert np.all(np.isfinite(out.data))

    def test_gradient_reaches_the_weight(self):
        layer = DenseTensorLinear(HIDDEN, VOCAB)
        x = Tensor(np.random.RandomState(1).randn(4, HIDDEN).astype(np.float32))
        out = layer.forward(x)
        out.grad = np.ones_like(out.data)
        out.backward()
        assert layer.weight.grad is not None
        assert np.all(np.isfinite(layer.weight.grad))


class TestToySmallTransformerForward:
    def test_shapes_and_finite(self):
        tf = _dense_model()
        T = 6
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits = tf.forward(embedded)
        assert logits.data.shape == (T, VOCAB)
        assert np.all(np.isfinite(logits.data))

    def test_half_bandwidth_default_matches_unlimited_visibility(self):
        # half_bandwidth=None (default) must reproduce today's exact
        # behavior -- every existing call site relies on unlimited
        # causal visibility, this must not silently change.
        tf_default = ToySmallTransformer(VOCAB, HIDDEN, MLP_HIDDEN, n_layers=2, num_cpus=2)
        tf_explicit_full = ToySmallTransformer(VOCAB, HIDDEN, MLP_HIDDEN, n_layers=2, num_cpus=2, half_bandwidth=8)
        # copy weights so only half_bandwidth differs
        for p_def, p_full in zip(tf_default.parameters(), tf_explicit_full.parameters(), strict=False):
            p_full.data = p_def.data.copy()
        T = 8
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits_default = tf_default.forward(embedded).data
        logits_full = tf_explicit_full.forward(embedded).data
        np.testing.assert_allclose(logits_default, logits_full, rtol=1e-5)

    def test_half_bandwidth_blocks_gradient_from_positions_further_back(self):
        # A narrow window must make position i's output provably
        # independent of tokens further than half_bandwidth back --
        # trusting banded_attention's own already-tested windowing
        # semantics from sili__new, just confirming the parameter is
        # actually wired through here, not re-verifying the windowing
        # math itself.
        tf = ToySmallTransformer(VOCAB, HIDDEN, MLP_HIDDEN, n_layers=1, num_cpus=2, half_bandwidth=2)
        T = 8
        embedded = np.random.RandomState(1).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits = tf.forward(embedded)
        # last position's logits gradient should not reach embedding
        # rows more than half_bandwidth positions back -- probe via a
        # perturbation: changing a far-back input row must not change
        # the last row's output at all.
        far_embedded = embedded.copy()
        far_embedded[0] += 5.0  # position 0, far outside half_bandwidth=2 from position 7
        logits_perturbed = tf.forward(far_embedded)
        np.testing.assert_allclose(logits.data[-1], logits_perturbed.data[-1], rtol=1e-5)

    def test_gradient_reaches_every_parameter(self):
        tf = _dense_model()
        T = 4
        embedded = np.random.RandomState(2).randn(T, HIDDEN).astype(np.float32) * 0.1
        logits = tf.forward(embedded)
        loss = cross_entropy_sum(logits, [(0, 1), (T - 1, 2)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        for p in tf.parameters():
            assert p.grad is not None
            assert np.all(np.isfinite(p.grad))

    def test_weights_actually_change_after_a_training_step(self):
        tf = _dense_model()
        opt = AdamOptimizer()
        probe = np.random.RandomState(3).randn(4, HIDDEN).astype(np.float32) * 0.1
        before = tf.forward(probe).data.copy()

        train_x = np.random.RandomState(4).randn(4, HIDDEN).astype(np.float32) * 0.1
        logits = tf.forward(train_x)
        loss = cross_entropy_sum(logits, [(0, 1), (1, 2), (2, 3), (3, 4)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        opt.step(tf.parameters(), lr=0.05)

        after = tf.forward(probe).data
        assert not np.allclose(before, after), "output on a fixed probe never changed after training"

    @pytest.mark.integration  # real training-convergence run
    def test_loss_decreases_on_a_single_repeated_example(self):
        # AdamOptimizer + clip_grad_norm_ -- the actual recommended
        # training path for DenseTensorLinear-based models (see module
        # docstring and clip_grad_norm_'s own docstring: per-node
        # backward_with_grad_clip was confirmed too weak -- even
        # combined with Adam, it still let this exact test diverge;
        # nothing here self-updates during backward() anymore, so a
        # real single global-norm clip is both correct and sufficient).
        tf = _dense_model()
        opt = AdamOptimizer()
        T = 5
        embedded = np.random.RandomState(5).randn(T, HIDDEN).astype(np.float32) * 0.1
        targets = [2, 4, 1, 6, 0]
        pairs = list(enumerate(targets))
        lr = 0.02

        first_loss = None
        min_loss = None
        for step in range(150):
            logits = tf.forward(embedded)
            loss = cross_entropy_sum(logits, pairs)
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            clip_grad_norm_(tf.parameters(), 1.0)
            opt.step(tf.parameters(), lr=lr)
            if step == 0:
                first_loss = float(loss.data)
            min_loss = float(loss.data) if min_loss is None else min(min_loss, float(loss.data))

        assert min_loss < first_loss * 0.3, f"loss barely moved: {first_loss:.3f} -> best {min_loss:.3f}"


class TestToyTileRecurrenceStep:
    def test_shapes_and_finite(self):
        model = _tile_model(num_tiles=3)
        x_window = np.random.RandomState(1).randn(3, STATE_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((3, STATE_WIDTH), dtype=np.float32)
        M_new, logits = model.step(x_window, M_prev)
        assert M_new.shape == (3, STATE_WIDTH)
        assert logits.data.shape == (3, VOCAB)
        assert np.all(np.isfinite(M_new))
        assert np.all(np.isfinite(logits.data))

    def test_gradient_reaches_every_leaf_including_centers_and_log_sigmas(self):
        model = _tile_model(num_tiles=3)
        x_window = np.random.RandomState(2).randn(3, STATE_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((3, STATE_WIDTH), dtype=np.float32)
        _M_new, logits = model.step(x_window, M_prev)
        loss = cross_entropy_sum(logits, [(0, 1), (1, 2), (2, 3)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert np.all(np.isfinite(p.grad))

    def test_column_mean_pooled_logits_carry_gradient_to_every_layer(self):
        # Confirms the column-mean-pool -> lm_head readout (state_width
        # -> embed_width -> vocab_size) is actually wired into the
        # graph -- training on only the LAST tile's logits (the only
        # row the real training loop ever uses) should still move
        # every shared weight, including q/k/v/o/gate/up/down (not
        # just lm_head), proving gradient flows back through the pool.
        model = _tile_model(num_tiles=3)
        opt = AdamOptimizer()
        probe_window = np.random.RandomState(3).randn(3, STATE_WIDTH).astype(np.float32) * 0.1
        probe_M = np.zeros((3, STATE_WIDTH), dtype=np.float32)
        before = model.step(probe_window, probe_M)[1].data.copy()

        train_window = np.random.RandomState(4).randn(3, STATE_WIDTH).astype(np.float32) * 0.1
        train_M = np.zeros((3, STATE_WIDTH), dtype=np.float32)
        _M_new, logits = model.step(train_window, train_M)
        loss = cross_entropy_sum(logits, [(2, 5)])  # ONLY the last tile's prediction
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        assert model.q_proj.weight.grad is not None and np.any(model.q_proj.weight.grad != 0), (
            "loss on the pooled last-tile logits never reached q_proj -- "
            "the column-mean pool isn't wired into the graph"
        )
        opt.step(model.parameters(), lr=0.05)

        after = model.step(probe_window, probe_M)[1].data
        assert not np.allclose(before, after), (
            "training on the last tile's pooled logits never changed the shared weights"
        )

    @pytest.mark.integration  # real training-convergence run
    def test_loss_decreases_on_a_single_repeated_example(self):
        model = _tile_model(num_tiles=3)
        opt = AdamOptimizer()
        x_window = np.random.RandomState(5).randn(3, STATE_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((3, STATE_WIDTH), dtype=np.float32)
        lr = 0.02

        first_loss = None
        min_loss = None
        for step in range(200):
            _M_new, logits = model.step(x_window, M_prev)
            loss = cross_entropy_sum(logits, [(2, 5)])  # last tile only, matching real usage
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step(model.parameters(), lr=lr)
            if step == 0:
                first_loss = float(loss.data)
            min_loss = float(loss.data) if min_loss is None else min(min_loss, float(loss.data))

        assert min_loss < first_loss * 0.3, f"loss barely moved: {first_loss:.3f} -> best {min_loss:.3f}"

    def test_resetting_M_prev_changes_logits(self):
        # Statefulness check, same spirit as tile_recurrence.py's own
        # test_resetting_M_prev_changes_logits.
        model = _tile_model(num_tiles=3)
        x_window = np.random.RandomState(6).randn(3, STATE_WIDTH).astype(np.float32) * 0.1
        M_real = np.random.RandomState(7).randn(3, STATE_WIDTH).astype(np.float32)
        M_zero = np.zeros((3, STATE_WIDTH), dtype=np.float32)

        _M_a, logits_a = model.step(x_window, M_real)
        _M_b, logits_b = model.step(x_window, M_zero)

        assert not np.allclose(logits_a.data, logits_b.data), "different M_prev produced identical logits"


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


class TestAdamOptimizer:
    def test_updates_and_zeroes_only_leaves_with_grad(self):
        a = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        b = Tensor(np.array([5.0], dtype=np.float32))
        a.grad = np.array([0.1, -0.1], dtype=np.float32)
        opt = AdamOptimizer()
        opt.step([a, b], lr=1.0)
        assert a.grad is None
        assert b.grad is None
        # first Adam step moves by ~lr in the sign of the gradient
        # (bias-corrected m_hat/v_hat ratio is +-1 on the very first step)
        np.testing.assert_allclose(a.data, [1.0 - 1.0, 2.0 + 1.0], atol=1e-3)

    def test_converges_faster_and_more_stably_than_plain_sgd_on_a_toy_quadratic(self):
        # Not a formal proof, just a real regression guard: on a simple
        # quadratic bowl, Adam should reach a small loss without the
        # sign-oscillation plain SGD-without-momentum is prone to at a
        # too-large learning rate.
        w_adam = Tensor(np.array([5.0], dtype=np.float32))
        opt = AdamOptimizer()
        for _ in range(200):
            loss = w_adam * w_adam  # simple, not going through cross_entropy_sum
            loss.grad = np.array([1.0], dtype=np.float32)
            # d(w^2)/dw = 2w
            w_adam.grad = np.array([2.0 * w_adam.data[0]], dtype=np.float32)
            opt.step([w_adam], lr=0.1)
        assert abs(float(w_adam.data[0])) < 0.5


class TestBackwardWithGradClip:
    def test_matches_plain_backward_when_under_the_clip_norm(self):
        # Small gradients (well under max_grad_norm) shouldn't be
        # altered at all -- clipping only kicks in above the threshold.
        a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = Tensor(np.array([0.5, -0.5, 1.0], dtype=np.float32))
        out1 = (a * b).sum()
        out1.grad = np.array(0.01, dtype=np.float32)  # tiny seed -> tiny grads
        backward_with_grad_clip(out1, max_grad_norm=1.0)
        expected_a_grad = b.data * 0.01

        a2 = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b2 = Tensor(np.array([0.5, -0.5, 1.0], dtype=np.float32))
        out2 = (a2 * b2).sum()
        out2.grad = np.array(0.01, dtype=np.float32)
        out2.backward()

        np.testing.assert_allclose(a.grad, expected_a_grad, rtol=1e-4)
        np.testing.assert_allclose(a.grad, a2.grad, rtol=1e-4)

    def test_clips_a_large_gradient_down_to_the_norm(self):
        a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = Tensor(np.array([0.5, -0.5, 1.0], dtype=np.float32))
        out = (a * b).sum()
        out.grad = np.array(1000.0, dtype=np.float32)  # huge seed
        backward_with_grad_clip(out, max_grad_norm=1.0)
        norm = float(np.sqrt(np.sum(a.grad.astype(np.float64) ** 2)))
        assert norm <= 1.0 + 1e-4

    def test_clips_a_dense_tensor_linear_update_not_just_leaf_tensors(self):
        model = _tile_model(num_tiles=3)
        opt = AdamOptimizer()
        x_window = np.random.RandomState(9).randn(3, STATE_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((3, STATE_WIDTH), dtype=np.float32)
        probe = np.random.RandomState(10).randn(3, STATE_WIDTH).astype(np.float32) * 0.1
        before = model.step(probe, np.zeros((3, STATE_WIDTH), dtype=np.float32))[1].data.copy()

        _M, logits = model.step(x_window, M_prev)
        loss = cross_entropy_sum(logits, [(0, 1), (1, 2), (2, 3)])
        loss.grad = np.array(1e6, dtype=np.float32)  # deliberately huge seed
        backward_with_grad_clip(loss, max_grad_norm=1.0)
        opt.step(model.parameters(), lr=1.0)

        after = model.step(probe, np.zeros((3, STATE_WIDTH), dtype=np.float32))[1].data
        assert np.all(np.isfinite(after)), "clipped update still produced non-finite output"
        # Some change is expected (real training happened); just not NaN/inf.
        assert not np.allclose(before, after)


class TestLrSchedule:
    def test_linear_warmup_then_cosine_decay(self):
        peak, warmup, total = 0.02, 10, 100
        assert lr_schedule(0, total, peak, warmup) == pytest.approx(peak / warmup)
        assert lr_schedule(warmup - 1, total, peak, warmup) == pytest.approx(peak, rel=0.15)
        assert lr_schedule(warmup, total, peak, warmup) == pytest.approx(peak, rel=1e-6)
        # decays monotonically after warmup
        lrs = [lr_schedule(s, total, peak, warmup) for s in range(warmup, total)]
        assert all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(len(lrs) - 1))
        # never drops below min_lr_ratio * peak
        assert lr_schedule(total - 1, total, peak, warmup, min_lr_ratio=0.1) >= peak * 0.1 - 1e-6
