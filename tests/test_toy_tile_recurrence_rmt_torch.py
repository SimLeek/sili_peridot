import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from model.toy_tile_recurrence_rmt_torch import ToyTileRecurrenceRMTTorch


VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM = 10, 6, 2, 3, 2
STATE_WIDTH = EMBED_WIDTH * COLUMN_NEURONS


def _model(l1_sparsity_coef=0.0):
    return ToyTileRecurrenceRMTTorch(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
                                     l1_sparsity_coef=l1_sparsity_coef,
                                     rng=np.random.default_rng(0))


class TestToyTileRecurrenceRMTTorch:
    def test_shapes_and_finite(self):
        model = _model()
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _memory_placeholder, logits, aux_loss = model.step(x_window, memory_prev, learning_rate=0.01)
        assert logits.shape == (NUM_TILES, VOCAB)
        assert torch.isfinite(logits).all()
        assert aux_loss is None

    def test_l1_sparsity_coef_produces_finite_aux_loss(self):
        model = _model(l1_sparsity_coef=0.05)
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _mp, _logits, aux_loss = model.step(x_window, memory_prev, learning_rate=0.01)
        assert aux_loss is not None
        assert torch.isfinite(aux_loss).all()

    def test_big_weights_change_after_a_step(self):
        model = _model()
        probe_window = np.random.RandomState(3).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        probe_memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        before = model.step(probe_window, probe_memory, 0.0)[1].detach().clone()

        train_window = np.random.RandomState(4).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        train_memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _mp, logits, _aux = model.step(train_window, train_memory, 0.05)
        target = torch.zeros(NUM_TILES, dtype=torch.long)
        target[-1] = 2
        loss = torch.nn.functional.cross_entropy(logits[-1:], target[-1:])
        model.zero_grad()
        loss.backward()
        model.apply_updates()

        after = model.step(probe_window, probe_memory, 0.0)[1].detach()
        assert torch.isfinite(after).all()
        assert not torch.allclose(before, after), (
            "output on a fixed probe never changed -- inline weight update never fired")

    def test_extract_memory_matches_state_width_shape(self):
        model = _model()
        x_window = np.random.RandomState(5).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.random.RandomState(6).randn(NUM_MEM, STATE_WIDTH).astype(np.float32)
        model.step(x_window, memory_prev, learning_rate=0.0)
        memory_new = model.extract_memory()
        assert memory_new.shape == (NUM_MEM, STATE_WIDTH)
        assert np.all(np.isfinite(memory_new))

    def test_loss_decreases_on_a_single_repeated_example(self):
        model = _model()
        opt = torch.optim.Adam(model.parameters_for_optimizer(), lr=0.02)
        x_window = np.random.RandomState(7).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        target = torch.zeros(NUM_TILES, dtype=torch.long)
        target[-1] = 5

        first_loss = None
        min_loss = None
        for step in range(200):
            _mp, logits, _aux = model.step(x_window, memory_prev, learning_rate=0.02)
            loss = torch.nn.functional.cross_entropy(logits[-1:], target[-1:])
            model.zero_grad()
            loss.backward()
            model.apply_updates()
            opt.step()
            loss_val = float(loss)
            if step == 0:
                first_loss = loss_val
            min_loss = loss_val if min_loss is None else min(min_loss, loss_val)

        assert min_loss < first_loss * 0.5, (
            f"loss never reached a real minimum: {first_loss:.3f} -> best {min_loss:.3f}")
