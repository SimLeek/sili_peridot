import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model.toy_tile_recurrence_rmt_standard import ToyTileRecurrenceRMTStandard

VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM = 10, 6, 2, 3, 2
STATE_WIDTH = EMBED_WIDTH * COLUMN_NEURONS


def _model():
    torch.manual_seed(0)
    return ToyTileRecurrenceRMTStandard(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM)


class TestToyTileRecurrenceRMTStandard:
    def test_shapes_and_finite(self):
        model = _model()
        x_window = torch.tensor(np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1)
        memory_prev = torch.zeros(NUM_MEM, STATE_WIDTH)
        memory_new, logits = model(x_window, memory_prev)
        assert memory_new.shape == (NUM_MEM, STATE_WIDTH)
        assert logits.shape == (NUM_TILES, VOCAB)
        assert torch.isfinite(memory_new).all()
        assert torch.isfinite(logits).all()

    @pytest.mark.integration  # real training-convergence run
    def test_loss_decreases_on_a_single_repeated_example(self):
        model = _model()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x_window = torch.tensor(np.random.RandomState(5).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1)
        target = torch.zeros(NUM_TILES, dtype=torch.long)
        target[-1] = 5

        first_loss, min_loss = None, None
        for step in range(200):
            memory_prev = torch.zeros(NUM_MEM, STATE_WIDTH)
            _memory_new, logits = model(x_window, memory_prev)
            loss = torch.nn.functional.cross_entropy(logits[-1:], target[-1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss)
            if step == 0:
                first_loss = loss_val
            min_loss = loss_val if min_loss is None else min(min_loss, loss_val)

        assert min_loss < first_loss * 0.5, (
            f"loss never reached a real minimum: {first_loss:.3f} -> best {min_loss:.3f}"
        )
