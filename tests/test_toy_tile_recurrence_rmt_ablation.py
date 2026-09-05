import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model.toy_tile_recurrence_rmt_ablation import ToyTileRecurrenceRMTAblation, clip_grad_norm_

VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM = 10, 6, 2, 3, 2
STATE_WIDTH = EMBED_WIDTH * COLUMN_NEURONS

BASE = {
    "use_custom_optimizer": True,
    "use_hard_clip": True,
    "use_gaussian_bias": True,
    "use_rmsnorm": True,
    "l1_sparsity_coef": 0.05,
}


def _swap(**overrides):
    cfg = dict(BASE)
    cfg.update(overrides)
    return cfg


ALL_CONFIGS = {
    "baseline_a": BASE,
    "swap_optimizer": _swap(use_custom_optimizer=False),
    "swap_clip": _swap(use_hard_clip=False),
    "swap_attn_bias": _swap(use_gaussian_bias=False),
    "swap_norm": _swap(use_rmsnorm=False),
    "swap_l1_sparsity": _swap(l1_sparsity_coef=0.0),
    "baseline_b": {
        "use_custom_optimizer": False,
        "use_hard_clip": False,
        "use_gaussian_bias": False,
        "use_rmsnorm": False,
        "l1_sparsity_coef": 0.0,
    },
}


class TestToyTileRecurrenceRMTAblation:
    def test_every_config_runs_forward_backward_update_finite(self):
        for name, cfg in ALL_CONFIGS.items():
            torch.manual_seed(0)
            model = ToyTileRecurrenceRMTAblation(
                VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM, rng=np.random.default_rng(0), **cfg
            )
            adam_params = model.parameters_for_optimizer()
            opt = torch.optim.Adam(adam_params) if adam_params else None
            x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
            memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)

            for _ in range(3):
                _mp, logits, aux = model.step(x_window, memory, 0.02)
                target = torch.zeros(NUM_TILES, dtype=torch.long)
                target[-1] = 2
                loss = torch.nn.functional.cross_entropy(logits[-1:], target[-1:])
                total_loss = loss if aux is None else loss + aux
                model.zero_grad()
                total_loss.backward()
                memory = model.extract_memory()
                model.apply_updates()
                if opt is not None:
                    clip_grad_norm_(adam_params, 1.0)
                    opt.step()

            assert torch.isfinite(logits).all(), f"{name}: non-finite logits"
            assert np.all(np.isfinite(memory)), f"{name}: non-finite memory"

    def test_baseline_a_weights_change_after_a_step(self):
        model = ToyTileRecurrenceRMTAblation(
            VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM, rng=np.random.default_rng(0), **BASE
        )
        probe_window = np.random.RandomState(3).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        probe_memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        before = model.step(probe_window, probe_memory, 0.0)[1].detach().clone()

        train_window = np.random.RandomState(4).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        train_memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _mp, logits, aux = model.step(train_window, train_memory, 0.05)
        target = torch.zeros(NUM_TILES, dtype=torch.long)
        target[-1] = 2
        loss = torch.nn.functional.cross_entropy(logits[-1:], target[-1:]) + aux
        model.zero_grad()
        loss.backward()
        model.apply_updates()

        after = model.step(probe_window, probe_memory, 0.0)[1].detach()
        assert not torch.allclose(before, after), (
            "output on a fixed probe never changed -- inline weight update never fired"
        )
