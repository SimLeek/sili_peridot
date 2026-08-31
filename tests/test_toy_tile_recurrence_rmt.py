import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT
from model.toy_recall_models import cross_entropy_sum, AdamOptimizer
from sili.sparse_rnn import DISLDOLayer


VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM = 10, 6, 2, 3, 2
STATE_WIDTH = EMBED_WIDTH * COLUMN_NEURONS
MAX_WEIGHTS = STATE_WIDTH * 16


def _model(disldo_cls=DISLDOLayer, num_cpus=2, l1_sparsity_coef=0.0):
    return ToyTileRecurrenceRMT(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
                                MAX_WEIGHTS, num_cpus=num_cpus, disldo_cls=disldo_cls,
                                l1_sparsity_coef=l1_sparsity_coef)


class TestToyTileRecurrenceRMT:
    def test_shapes_and_finite(self):
        model = _model()
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        memory_new, logits, aux_loss = model.step(x_window, memory_prev, learning_rate=0.01)
        assert memory_new.shape == (NUM_MEM, STATE_WIDTH)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(memory_new))
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None

    def test_l1_sparsity_coef_produces_finite_aux_loss(self):
        model = _model(l1_sparsity_coef=0.05)
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _memory_new, _logits, aux_loss = model.step(x_window, memory_prev, learning_rate=0.01)
        assert aux_loss is not None
        assert np.isfinite(float(aux_loss.data))

    def test_big_weights_change_after_a_step_with_no_external_optimizer(self):
        model = _model()
        probe_window = np.random.RandomState(3).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        probe_memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        before = model.step(probe_window, probe_memory, learning_rate=0.0)[1].data.copy()

        train_window = np.random.RandomState(4).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        train_memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _memory_new, logits, _aux = model.step(train_window, train_memory, learning_rate=0.05)
        loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 2)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()

        after = model.step(probe_window, probe_memory, learning_rate=0.0)[1].data
        assert np.all(np.isfinite(after))
        assert not np.allclose(before, after), (
            "output on a fixed probe never changed -- inline weight update never fired")

    def test_resetting_memory_prev_changes_logits(self):
        model = _model()
        x_window = np.random.RandomState(6).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_real = np.random.RandomState(7).randn(NUM_MEM, STATE_WIDTH).astype(np.float32)
        memory_zero = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)

        _a, logits_a, _ = model.step(x_window, memory_real, learning_rate=0.0)
        _b, logits_b, _ = model.step(x_window, memory_zero, learning_rate=0.0)

        assert not np.allclose(logits_a.data, logits_b.data), (
            "different memory_prev produced identical logits")

    def test_loss_decreases_on_a_single_repeated_example(self):
        model = _model()
        opt = AdamOptimizer()
        x_window = np.random.RandomState(5).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        lr = 0.02

        first_loss = None
        min_loss = None
        for step in range(200):
            _memory_new, logits, _aux = model.step(x_window, memory_prev, learning_rate=lr)
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 5)])
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            opt.step(model.parameters_for_optimizer(), lr=lr)
            if step == 0:
                first_loss = float(loss.data)
            min_loss = float(loss.data) if min_loss is None else min(min_loss, float(loss.data))

        assert min_loss < first_loss * 0.5, (
            f"loss never reached a real minimum: {first_loss:.3f} -> best {min_loss:.3f}")


class TestToyTileRecurrenceRMTWidening:
    """Sparsity plan Phase 6 (task #335): input_sparsity_p/dy_sparsity_p/
    wide_max_weights. All three default to None -- must be a completely
    unchanged code path in that case; the widened arm must stay finite
    through a real forward+backward pass exercising every one of the 5
    affected layers' call sites (both write and read passes, plus the
    l1_sparsity_coef probe branch)."""

    def test_default_args_bit_identical_to_pre_phase6(self):
        # Two identically-seeded models, one built with the new kwargs
        # explicitly passed at their default (None) values, one without
        # them at all -- must produce identical output. Proves the new
        # kwargs' mere presence in the signature doesn't perturb anything.
        rng_seed = 42
        m_a = ToyTileRecurrenceRMT(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
                                   MAX_WEIGHTS, num_cpus=2, rng=np.random.default_rng(rng_seed))
        m_b = ToyTileRecurrenceRMT(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
                                   MAX_WEIGHTS, num_cpus=2, rng=np.random.default_rng(rng_seed),
                                   input_sparsity_p=None, dy_sparsity_p=None, wide_max_weights=None)
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _mem_a, logits_a, _ = m_a.step(x_window, memory_prev, learning_rate=0.05)
        _mem_b, logits_b, _ = m_b.step(x_window, memory_prev, learning_rate=0.05)
        np.testing.assert_array_equal(logits_a.data, logits_b.data)

    def test_widened_sparse_arm_stays_finite_through_full_backward(self):
        wide_embed = EMBED_WIDTH * 2
        wide_state = wide_embed * COLUMN_NEURONS
        model = ToyTileRecurrenceRMT(
            VOCAB, wide_embed, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
            MAX_WEIGHTS, num_cpus=2, rng=np.random.default_rng(8),
            input_sparsity_p=0.5, dy_sparsity_p=0.5, wide_max_weights=MAX_WEIGHTS * 4,
            l1_sparsity_coef=0.05)
        x_window = np.random.RandomState(9).randn(NUM_TILES, wide_embed).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, wide_state), dtype=np.float32)
        for step in range(5):
            memory_prev, logits, aux_loss = model.step(x_window, memory_prev, learning_rate=0.05)
            assert logits.data.shape == (NUM_TILES, VOCAB)
            assert memory_prev.shape == (NUM_MEM, wide_state)
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 2)])
            if aux_loss is not None:
                loss = loss + aux_loss
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            assert np.isfinite(float(loss.data)), f"step {step}: loss non-finite"
            assert np.all(np.isfinite(memory_prev)), f"step {step}: memory non-finite"

    def test_wide_max_weights_only_affects_the_5_layers_not_lm_head(self):
        # _max_row_weights is the Python-side per-row capacity
        # _preseed_random_sparse computed from the max_weights value it
        # was actually given -- the most direct observable proxy (no
        # dedicated max_weights accessor exists on the C++ layer itself).
        # Compare each of the 5 affected layers' own capacity WITH vs
        # WITHOUT wide_max_weights (not across differently-shaped layers,
        # which legitimately differ in row capacity at the same
        # max_weights since _max_row_weights depends on n_inputs too).
        base = ToyTileRecurrenceRMT(
            VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
            MAX_WEIGHTS, num_cpus=2, rng=np.random.default_rng(11))
        wide = ToyTileRecurrenceRMT(
            VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
            MAX_WEIGHTS, num_cpus=2, rng=np.random.default_rng(11),
            wide_max_weights=MAX_WEIGHTS * 4)
        for name in ("input_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
            assert getattr(wide, name)._max_row_weights > getattr(base, name)._max_row_weights, (
                f"{name}: wide_max_weights didn't increase its per-row capacity")
        assert wide.lm_head._max_row_weights == base.lm_head._max_row_weights, (
            "lm_head's capacity must stay unaffected by wide_max_weights")

    def test_dy_sparsity_p_defaults_from_input_sparsity_p(self):
        model = ToyTileRecurrenceRMT(
            VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
            MAX_WEIGHTS, num_cpus=2, rng=np.random.default_rng(12),
            input_sparsity_p=0.4)
        assert model.dy_sparsity_p == 0.4


def _build_window(embed_table, tokens, i, num_tiles):
    embed_width = embed_table.shape[1]
    window = np.zeros((num_tiles, embed_width), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        if src >= 0:
            window[j] = embed_table[tokens[src]]
    return window


class TestStepCached:
    """step_cached (direct instruction): incremental alternative to
    step() -- one new tile per call, K/V for the other num_tiles-1
    positions read from an explicit cache instead of recomputed. See
    step_cached's own docstring for the full rationale."""

    def test_matches_step_bit_exact_when_weights_never_change(self):
        # The load-bearing correctness claim: input_proj/q/k/v_proj are
        # per-row (non-mixing) projections, so a content tile's k/v
        # depends only on its own token and the CURRENT weights -- with
        # requires_grad=False (no backward, weights genuinely static),
        # caching introduces zero approximation, not a small one.
        model_a = ToyTileRecurrenceRMT(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
                                       MAX_WEIGHTS, num_cpus=2, rng=np.random.default_rng(42))
        model_b = ToyTileRecurrenceRMT(VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM,
                                       MAX_WEIGHTS, num_cpus=2, rng=np.random.default_rng(42))
        embed_table = np.random.RandomState(1).randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.1
        tokens = np.random.RandomState(2).randint(0, VOCAB, size=12)

        memory_a = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        memory_b = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        tile_cache = None
        for i in range(len(tokens)):
            window = _build_window(embed_table, tokens, i, NUM_TILES)
            memory_a, logits_a, _aux_a = model_a.step(window, memory_a, 0.0, requires_grad=False)
            memory_b, logits_b, _aux_b, tile_cache = model_b.step_cached(
                embed_table[tokens[i]], memory_b, 0.0, tile_cache, requires_grad=False)
            row_a = np.asarray(logits_a.data)[NUM_TILES - 1]
            row_b = np.asarray(logits_b.data)[0]
            assert np.array_equal(row_a, row_b), f"step {i}: logits diverged with static weights"
            assert np.array_equal(memory_a, memory_b), f"step {i}: memory_new diverged with static weights"

    def test_shapes_and_finite(self):
        model = _model()
        new_embed = np.random.RandomState(1).randn(EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        memory_new, logits, aux_loss, tile_cache = model.step_cached(
            new_embed, memory_prev, learning_rate=0.01, tile_cache=None)
        assert memory_new.shape == (NUM_MEM, STATE_WIDTH)
        assert logits.data.shape == (1, VOCAB)
        assert np.all(np.isfinite(memory_new))
        assert np.all(np.isfinite(logits.data))
        assert aux_loss is None
        assert len(tile_cache) == 1  # grows by one per call, caps at NUM_TILES-1
        for k_row, v_row in tile_cache:
            assert k_row.shape == (STATE_WIDTH,)
            assert v_row.shape == (STATE_WIDTH,)

    def test_cache_length_caps_at_num_tiles_minus_1(self):
        model = _model()
        memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        tile_cache = None
        rng = np.random.RandomState(5)
        for _ in range(NUM_TILES + 5):
            new_embed = (rng.randn(EMBED_WIDTH) * 0.1).astype(np.float32)
            memory, _logits, _aux, tile_cache = model.step_cached(
                new_embed, memory, learning_rate=0.0, tile_cache=tile_cache)
            assert len(tile_cache) <= NUM_TILES - 1

    def test_weights_actually_update_via_backward(self):
        model = _model()
        opt = AdamOptimizer()
        memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        new_embed = np.random.RandomState(6).randn(EMBED_WIDTH).astype(np.float32) * 0.1
        before = np.asarray(model.input_ln.data).copy()
        memory, logits, aux, tile_cache = model.step_cached(
            new_embed, memory, learning_rate=0.05, tile_cache=None, requires_grad=True)
        loss = cross_entropy_sum(logits, [(0, 1)])
        if aux is not None:
            loss = loss + aux
        loss.backward()
        opt.step(model.parameters_for_optimizer(), lr=0.05)
        assert not np.allclose(before, model.input_ln.data), "weights should move after a real backward+opt.step"
        assert np.all(np.isfinite(memory))
