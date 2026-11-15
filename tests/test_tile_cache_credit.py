"""Tests for model/tile_cache_credit.py's budgeted exact-credit mechanism
(Mechanism A of the KV-cache synaptic hold/credit plan) and the
runtime-tunable "dial" train_mqar_curriculum.py exposes around it
(tile_cache_credit_fraction / tile_cache_credit_fraction_fn). See
docs/research/toy_tile_recurrence_rmt.rst:tile_cache_exact_credit and
JOURNAL.md's dose-response sweep for the validated numbers behind the
0.5 default.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sili.sparse_rnn import DISLDOLayer32

import scripts.train_mqar_curriculum as m
from model.tile_cache_credit import apply_tile_cache_exact_credit
from model.toy_recall_models import cross_entropy_sum
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT

VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM = 10, 6, 2, 8, 2
STATE_WIDTH = EMBED_WIDTH * COLUMN_NEURONS
MAX_WEIGHTS = STATE_WIDTH * 16


def _dense_model(seed=0):
    return ToyTileRecurrenceRMT(
        VOCAB,
        EMBED_WIDTH,
        COLUMN_NEURONS,
        NUM_TILES,
        NUM_MEM,
        MAX_WEIGHTS,
        num_cpus=1,
        disldo_cls=DISLDOLayer32,
        dense=True,
        rng=np.random.default_rng(seed),
    )


def _fill_cache_and_query(model, n_fill=6, query_target=1, lr=0.01):
    """Runs n_fill cheap (requires_grad=False) step_cached calls to
    populate tile_cache, then one real requires_grad=True query step with
    a real loss.backward() -- the exact precondition
    apply_tile_cache_exact_credit needs (cache_k_tensor/cache_v_tensor in
    last_debug with real, non-None .grad)."""
    rng = np.random.RandomState(1)
    embed_table = (rng.randn(VOCAB, EMBED_WIDTH) * 0.1).astype(np.float32)
    memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
    tile_cache = None
    for i in range(n_fill):
        memory, _logits, _aux, tile_cache = model.step_cached(
            embed_table[i % VOCAB], memory, lr, tile_cache, requires_grad=False
        )
    memory, logits, aux, tile_cache = model.step_cached(
        embed_table[(n_fill) % VOCAB], memory, lr, tile_cache, requires_grad=True
    )
    loss = cross_entropy_sum(logits, [(0, query_target)])
    if aux is not None:
        loss = loss + aux
    loss.backward()
    return tile_cache


class TestApplyTileCacheExactCredit:
    def test_k_rows_bounds_number_of_corrections(self):
        model = _dense_model()
        _fill_cache_and_query(model, n_fill=6)
        n = apply_tile_cache_exact_credit(model, lr=0.01, k_rows=1)
        assert n == 1
        n = apply_tile_cache_exact_credit(model, lr=0.01, k_rows=3)
        assert n == 3
        # requesting more than the live cache holds caps at the cache size,
        # not a crash
        n = apply_tile_cache_exact_credit(model, lr=0.01, k_rows=1000)
        assert n == 6

    def test_rotation_cursor_advances(self):
        model = _dense_model()
        _fill_cache_and_query(model, n_fill=6)
        assert getattr(model, "_tile_cache_credit_cursor", 0) == 0
        apply_tile_cache_exact_credit(model, lr=0.01, k_rows=1)
        assert model._tile_cache_credit_cursor == 1
        apply_tile_cache_exact_credit(model, lr=0.01, k_rows=2)
        assert model._tile_cache_credit_cursor == 3

    def test_no_op_without_cached_gradient_tensors(self):
        # A plain step() (not step_cached()) never populates
        # cache_k_tensor/cache_v_tensor/cache_x_rows in last_debug at all.
        model = _dense_model()
        window = np.zeros((NUM_TILES, EMBED_WIDTH), dtype=np.float32)
        memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        model.step(window, memory, 0.01, requires_grad=False)
        assert apply_tile_cache_exact_credit(model, lr=0.01, k_rows=1) == 0

    def test_weights_actually_change(self):
        model = _dense_model()
        _fill_cache_and_query(model, n_fill=6)
        before = model.k_proj._c.weights_vals.copy()
        n = apply_tile_cache_exact_credit(model, lr=0.1, k_rows=3)
        assert n == 3
        after = model.k_proj._c.weights_vals
        assert not np.array_equal(before, after)


@pytest.mark.integration  # runs real training steps (seconds), not <10s pre-commit territory
class TestTileCacheCreditRuntimeDial:
    def _spy_calls(self, monkeypatch):
        calls = []
        real = m.apply_tile_cache_exact_credit

        def spy(model, lr, k_rows=1):
            calls.append(k_rows)
            return real(model, lr, k_rows=k_rows)

        monkeypatch.setattr(m, "apply_tile_cache_exact_credit", spy)
        return calls

    def _run(self, **kwargs):
        m.NUM_CPUS = 1
        m.K_START = 2
        m.train_curriculum(
            "fp32",
            60,
            1001,
            0.015,
            16,
            10,
            embed_width=12,
            wrong_streak_threshold=100000000,
            k_first_target=3,
            use_tile_cache=True,
            use_tile_cache_at_query=True,
            log_every=1000000,
            log_fn=lambda *_a, **_k: None,
            **kwargs,
        )

    def test_default_fraction_is_half_the_cache(self, monkeypatch):
        calls = self._spy_calls(monkeypatch)
        self._run()
        assert len(calls) > 0
        # num_tiles=16 -> cache size 15 -> round(0.5*15) == 8
        assert set(calls) == {8}

    def test_explicit_none_fraction_disables_credit(self, monkeypatch):
        calls = self._spy_calls(monkeypatch)
        self._run(tile_cache_credit_fraction=None)
        assert calls == []

    def test_fraction_fn_changes_k_rows_mid_run(self, monkeypatch):
        # The actual runtime-dial demonstration: the correction budget
        # changes DURING a single run, not just at construction time --
        # direct motivation: "sleep nightly instead of every minute" for
        # an online/embodied deployment should be a live-adjustable knob,
        # same pattern as this one.
        calls = self._spy_calls(monkeypatch)
        seen_steps = []

        def dial(step):
            seen_steps.append(step)
            return 0.1 if step < 30 else 1.0

        self._run(tile_cache_credit_fraction_fn=dial)
        assert len(seen_steps) > 0
        early_k_rows = set(calls[: len(calls) // 2])
        late_k_rows = set(calls[len(calls) // 2 :])
        # 0.1 * 15 -> round to 2; 1.0 * 15 -> 15 (num_tiles=16)
        assert 2 in calls
        assert 15 in calls
        assert early_k_rows != late_k_rows, "the dial never actually changed k_rows mid-run"
