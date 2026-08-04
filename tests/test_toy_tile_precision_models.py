import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from model.toy_precision_models import AdamRowScaleDISLDOLayer, AdamRank1DISLDOLayer
from model.toy_recall_models import cross_entropy_sum, AdamOptimizer, clip_grad_norm_
from sili.sparse_rnn import DISLDOLayer


VOCAB, EMBED_WIDTH, COLUMN_NEURONS, MLP_HIDDEN, NUM_TILES = 10, 6, 2, 16, 3
STATE_WIDTH = EMBED_WIDTH * COLUMN_NEURONS
MAX_WEIGHTS = STATE_WIDTH * MLP_HIDDEN  # generous, fully-dense-capable at this toy scale


def _model(disldo_cls=DISLDOLayer, num_tiles=NUM_TILES, num_cpus=2):
    return ToyTileRecurrenceRealFP4(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, MLP_HIDDEN,
                                    num_tiles, MAX_WEIGHTS, num_cpus=num_cpus,
                                    disldo_cls=disldo_cls)


class TestToyTileRecurrenceRealFP4:
    def test_shapes_and_finite(self):
        model = _model()
        x_window = np.random.RandomState(1).randn(NUM_TILES, STATE_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        M_new, logits, aux_loss = model.step(x_window, M_prev, learning_rate=0.01)
        assert M_new.shape == (NUM_TILES, STATE_WIDTH)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(M_new))
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None

    def test_big_weights_change_after_a_step_with_no_external_optimizer(self):
        model = _model()
        probe_window = np.random.RandomState(3).randn(NUM_TILES, STATE_WIDTH).astype(np.float32) * 0.1
        probe_M = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        before = model.step(probe_window, probe_M, learning_rate=0.0)[1].data.copy()

        train_window = np.random.RandomState(4).randn(NUM_TILES, STATE_WIDTH).astype(np.float32) * 0.1
        train_M = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        _M_new, logits, _aux = model.step(train_window, train_M, learning_rate=0.05)
        loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 2)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()

        after = model.step(probe_window, probe_M, learning_rate=0.0)[1].data
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after), (
            "output on a fixed probe never changed -- DISLDOLayer's inline "
            "update never fired")

    def test_leaf_parameters_for_optimizer(self):
        model = _model()
        params = model.parameters_for_optimizer()
        assert len(params) == 4  # input_ln, post_ln, centers, log_sigmas
        assert params[0].data.shape == (STATE_WIDTH,)
        assert params[1].data.shape == (STATE_WIDTH,)
        assert params[2].data.shape == (NUM_TILES,)
        assert params[3].data.shape == (NUM_TILES,)

    def test_resetting_M_prev_changes_logits(self):
        model = _model()
        x_window = np.random.RandomState(6).randn(NUM_TILES, STATE_WIDTH).astype(np.float32) * 0.1
        M_real = np.random.RandomState(7).randn(NUM_TILES, STATE_WIDTH).astype(np.float32)
        M_zero = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)

        _M_a, logits_a, _ = model.step(x_window, M_real, learning_rate=0.0)
        _M_b, logits_b, _ = model.step(x_window, M_zero, learning_rate=0.0)

        assert not np.allclose(logits_a.data, logits_b.data), (
            "different M_prev produced identical logits")

    def test_loss_decreases_on_a_single_repeated_example(self):
        model = _model()
        opt = AdamOptimizer()
        x_window = np.random.RandomState(5).randn(NUM_TILES, STATE_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        lr = 0.02

        first_loss = None
        min_loss = None
        for step in range(200):
            _M_new, logits, _aux = model.step(x_window, M_prev, learning_rate=lr)
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 5)])
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            opt.step(model.parameters_for_optimizer(), lr=lr)
            if step == 0:
                first_loss = float(loss.data)
            min_loss = float(loss.data) if min_loss is None else min(min_loss, float(loss.data))

        assert min_loss < first_loss * 0.5, (
            f"loss never reached a real minimum: {first_loss:.3f} -> best {min_loss:.3f}")


class TestToyTileRecurrenceRealFP4WithEnergy:
    def test_shapes_and_finite_and_aux_loss_present(self):
        model = ToyTileRecurrenceRealFP4(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, MLP_HIDDEN,
                                         NUM_TILES, MAX_WEIGHTS, num_cpus=2,
                                         disldo_cls=DISLDOLayer, use_energy=True)
        x_window = np.random.RandomState(1).randn(NUM_TILES, STATE_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        M_new, logits, aux_loss = model.step(x_window, M_prev, learning_rate=0.01)
        assert M_new.shape == (NUM_TILES, STATE_WIDTH)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(M_new))
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is not None
        assert np.isfinite(float(aux_loss.data))


class TestToyTileRecurrenceRealFP4WithRowScaleAdam:
    def test_shapes_and_finite(self):
        model = _model(disldo_cls=AdamRowScaleDISLDOLayer)
        x_window = np.random.RandomState(1).randn(NUM_TILES, STATE_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        M_new, logits, aux_loss = model.step(x_window, M_prev, learning_rate=0.01)
        assert M_new.shape == (NUM_TILES, STATE_WIDTH)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(M_new))
        assert np.all(np.isfinite(logits.data))


class TestToyTileRecurrenceRealFP4WithRank1Adam:
    def test_shapes_and_finite(self):
        model = _model(disldo_cls=AdamRank1DISLDOLayer)
        x_window = np.random.RandomState(1).randn(NUM_TILES, STATE_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        M_new, logits, aux_loss = model.step(x_window, M_prev, learning_rate=0.01)
        assert M_new.shape == (NUM_TILES, STATE_WIDTH)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(M_new))
        assert np.all(np.isfinite(logits.data))
