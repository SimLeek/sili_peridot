import functools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from model.toy_precision_models import AdamRowScaleDISLDOLayer, AdamRank1DISLDOLayer, TrueMultiDigitLayer
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
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        M_new, logits, aux_loss = model.step(x_window, M_prev, learning_rate=0.01)
        assert M_new.shape == (NUM_TILES, STATE_WIDTH)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(M_new))
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None

    def test_big_weights_change_after_a_step_with_no_external_optimizer(self):
        model = _model()
        probe_window = np.random.RandomState(3).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        probe_M = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        before = model.step(probe_window, probe_M, learning_rate=0.0)[1].data.copy()

        train_window = np.random.RandomState(4).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
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
        assert len(params) == 4  # input_ln, state_ln, centers, log_sigmas
        assert params[0].data.shape == (STATE_WIDTH,)
        assert params[1].data.shape == (STATE_WIDTH,)
        assert params[2].data.shape == (NUM_TILES,)
        assert params[3].data.shape == (NUM_TILES,)

    def test_resetting_M_prev_changes_logits(self):
        model = _model()
        x_window = np.random.RandomState(6).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        M_real = np.random.RandomState(7).randn(NUM_TILES, STATE_WIDTH).astype(np.float32)
        M_zero = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)

        _M_a, logits_a, _ = model.step(x_window, M_real, learning_rate=0.0)
        _M_b, logits_b, _ = model.step(x_window, M_zero, learning_rate=0.0)

        assert not np.allclose(logits_a.data, logits_b.data), (
            "different M_prev produced identical logits")

    def test_loss_decreases_on_a_single_repeated_example(self):
        model = _model()
        opt = AdamOptimizer()
        x_window = np.random.RandomState(5).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
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
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
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
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        M_new, logits, aux_loss = model.step(x_window, M_prev, learning_rate=0.01)
        assert M_new.shape == (NUM_TILES, STATE_WIDTH)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(M_new))
        assert np.all(np.isfinite(logits.data))


class TestToyTileRecurrenceRealFP4WithRank1Adam:
    def test_shapes_and_finite(self):
        model = _model(disldo_cls=AdamRank1DISLDOLayer)
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        M_new, logits, aux_loss = model.step(x_window, M_prev, learning_rate=0.01)
        assert M_new.shape == (NUM_TILES, STATE_WIDTH)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(M_new))
        assert np.all(np.isfinite(logits.data))


class TestToyTileRecurrenceRealFP4LongHorizonStability:
    """MODEL-level (not isolated-layer) long-horizon check -- everything
    else in this file trains at most 200 steps, which would NOT have
    caught the original failure that motivated this whole session's
    RMSprop investigation: a real DISLDOLayer32 permutation-regression run
    (sili_peridot's eval_rank_floor.py) stayed rock-stable for 300+ steps,
    then diverged catastrophically past step ~400. That divergence was
    root-caused and fixed at the sili__new C++ layer level
    (BoundedRMSpropSynapsePolicy, now the real default there) and verified
    in isolation (sili__new's test_synapse_policy_long_horizon.cpp) -- but
    never against the ACTUAL model this project cares about, where 5
    DISLDOLayer-family projections (q/k/v/o_proj/lm_head) interact through
    real attention and recurrent state carry-over, not a single isolated
    layer. Uses the actual PRODUCTION arm config (see
    scripts/train_tile_curriculum.py's true_multi_digit_stochastic):
    TrueMultiDigitLayer(digit_cls=DISLDOLayer, n_stages=3, base=12.0),
    sparse (not dense) connectivity.

    N_STEPS=3000 matches sili__new's own long-horizon test convention
    (~7.5x the ~400-step point where the original divergence appeared).
    """

    N_STEPS = 3000

    def _production_model(self, num_tiles=NUM_TILES, num_cpus=1):
        disldo_cls = functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                                       n_stages=3, base=12.0, lr_power=0.0)
        return ToyTileRecurrenceRealFP4(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, MLP_HIDDEN,
                                        num_tiles, MAX_WEIGHTS, num_cpus=num_cpus,
                                        disldo_cls=disldo_cls)

    def test_no_late_onset_divergence_or_nan_over_long_horizon(self):
        model = self._production_model()
        x_window = np.random.RandomState(10).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        M_prev = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        lr = 0.02
        target_pos, target_tok = NUM_TILES - 1, 3

        losses = []
        for step in range(self.N_STEPS):
            _M_new, logits, _aux = model.step(x_window, M_prev, learning_rate=lr)
            loss = cross_entropy_sum(logits, [(target_pos, target_tok)])
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            loss_val = float(loss.data)
            assert np.isfinite(loss_val), f"loss went non-finite at step {step}"
            losses.append(loss_val)

        first_loss = losses[0]
        best_loss = min(losses)
        # Real learning happened at all (not just "never blew up").
        assert best_loss < first_loss * 0.5, (
            f"model never substantially reduced loss: first={first_loss:.3f} best={best_loss:.3f}")

        # Late-onset-divergence check, mirroring sili__new's own
        # test_synapse_policy_long_horizon.cpp convention: the ORIGINAL
        # failure this whole investigation started from was rock-stable
        # for 300+ steps before diverging -- so "it looked fine early"
        # must not be trusted; check the max loss AFTER an early warmup
        # window against what had already been achieved by then, not just
        # the final value.
        warmup = 200
        best_by_warmup = min(losses[:warmup])
        max_after_warmup = max(losses[warmup:])
        assert max_after_warmup < best_by_warmup * 20.0, (
            f"loss spiked well past its own early-training level later in "
            f"the run (best_by_step_{warmup}={best_by_warmup:.3f}, "
            f"max_after_step_{warmup}={max_after_warmup:.3f}) -- looks like "
            f"the same late-onset divergence pattern this test exists to catch")
