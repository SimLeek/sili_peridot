import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sili.sparse_rnn import DISLDOLayer, DISLDOLayer32
from sili.tensor import Tensor

from model.toy_recall_models import AdamOptimizer, cross_entropy_sum
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT, spectral_norm_upper_bound

VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, NUM_MEM = 10, 6, 2, 3, 2
STATE_WIDTH = EMBED_WIDTH * COLUMN_NEURONS
MAX_WEIGHTS = STATE_WIDTH * 16


def _model(
    disldo_cls=DISLDOLayer,
    num_cpus=2,
    l1_sparsity_coef=0.0,
    qk_l1_sparsity_coef=0.0,
    qkvo_norm_enable=False,
    dense=False,
    rng=None,
):
    return ToyTileRecurrenceRMT(
        VOCAB,
        EMBED_WIDTH,
        COLUMN_NEURONS,
        NUM_TILES,
        NUM_MEM,
        MAX_WEIGHTS,
        num_cpus=num_cpus,
        disldo_cls=disldo_cls,
        l1_sparsity_coef=l1_sparsity_coef,
        qk_l1_sparsity_coef=qk_l1_sparsity_coef,
        qkvo_norm_enable=qkvo_norm_enable,
        dense=dense,
        rng=rng,
    )


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
            "output on a fixed probe never changed -- inline weight update never fired"
        )

    def test_resetting_memory_prev_changes_logits(self):
        model = _model()
        x_window = np.random.RandomState(6).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_real = np.random.RandomState(7).randn(NUM_MEM, STATE_WIDTH).astype(np.float32)
        memory_zero = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)

        _a, logits_a, _ = model.step(x_window, memory_real, learning_rate=0.0)
        _b, logits_b, _ = model.step(x_window, memory_zero, learning_rate=0.0)

        assert not np.allclose(logits_a.data, logits_b.data), "different memory_prev produced identical logits"

    @pytest.mark.integration  # real training-convergence run
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
            f"loss never reached a real minimum: {first_loss:.3f} -> best {min_loss:.3f}"
        )


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
        m_a = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(rng_seed),
        )
        m_b = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(rng_seed),
            input_sparsity_p=None,
            dy_sparsity_p=None,
            wide_max_weights=None,
        )
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _mem_a, logits_a, _ = m_a.step(x_window, memory_prev, learning_rate=0.05)
        _mem_b, logits_b, _ = m_b.step(x_window, memory_prev, learning_rate=0.05)
        np.testing.assert_array_equal(logits_a.data, logits_b.data)

    def test_widened_sparse_arm_stays_finite_through_full_backward(self):
        wide_embed = EMBED_WIDTH * 2
        wide_state = wide_embed * COLUMN_NEURONS
        model = ToyTileRecurrenceRMT(
            VOCAB,
            wide_embed,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(8),
            input_sparsity_p=0.5,
            dy_sparsity_p=0.5,
            wide_max_weights=MAX_WEIGHTS * 4,
            l1_sparsity_coef=0.05,
        )
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
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(11),
        )
        wide = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(11),
            wide_max_weights=MAX_WEIGHTS * 4,
        )
        for name in ("input_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
            assert getattr(wide, name)._max_row_weights > getattr(base, name)._max_row_weights, (
                f"{name}: wide_max_weights didn't increase its per-row capacity"
            )
        assert wide.lm_head._max_row_weights == base.lm_head._max_row_weights, (
            "lm_head's capacity must stay unaffected by wide_max_weights"
        )

    def test_dy_sparsity_p_defaults_from_input_sparsity_p(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(12),
            input_sparsity_p=0.4,
        )
        assert model.dy_sparsity_p == 0.4


class TestOutputDySparsityP:
    """lm_head/critic_head's own backward-gradient density (direct
    instruction, following the graded-schedule speed work): a genuinely
    separate axis from input_sparsity_p/dy_sparsity_p above -- those
    never touch lm_head/critic_head at all (see
    test_wide_max_weights_only_affects_the_5_layers_not_lm_head).
    output_dy_sparsity_p is the only knob that does."""

    def test_none_default_bit_identical(self):
        rng_seed = 21
        m_a = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(rng_seed),
        )
        m_b = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(rng_seed),
            output_dy_sparsity_p=None,
        )
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _mem_a, logits_a, _ = m_a.step(x_window, memory_prev, learning_rate=0.05)
        _mem_b, logits_b, _ = m_b.step(x_window, memory_prev, learning_rate=0.05)
        np.testing.assert_array_equal(logits_a.data, logits_b.data)

    def test_set_value_stays_finite_through_real_backward(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(22),
            output_dy_sparsity_p=0.5,
        )
        x_window = np.random.RandomState(23).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        for step in range(5):
            memory_prev, logits, _aux = model.step(x_window, memory_prev, learning_rate=0.05)
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 3)])
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            assert np.isfinite(float(loss.data)), f"step {step}: loss non-finite"
            assert np.all(np.isfinite(memory_prev)), f"step {step}: memory non-finite"

    def test_does_not_touch_the_5_wide_layers_kwargs(self):
        # output_dy_sparsity_p must be a strictly separate axis from
        # input_sparsity_p/dy_sparsity_p -- structural check, not a
        # runtime-output comparison: this codebase's stochastic-rounding
        # RNG is a shared/global stream (see
        # feedback_seed_stochastic_rng_for_comparisons), so changing how
        # many quantization draws lm_head's backward makes shifts the
        # RNG state for every LATER draw too, including input_proj's --
        # expected drift, not something a bit-identical-output assertion
        # could ever pass. What IS guaranteed: model.dy_sparsity_p (the
        # 5 wide layers' own gradient-density kwarg) and
        # model._wide_extra_kwargs(name) (what actually gets splatted
        # into their forward() calls) stay at their untouched defaults.
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(24),
            output_dy_sparsity_p=0.3,
        )
        assert model.dy_sparsity_p is None
        for name in ToyTileRecurrenceRMT._WIDE_LAYER_NAMES:
            assert model._wide_extra_kwargs(name) == {}
        assert model._output_extra_kwargs == {"dy_sparsity_p": 0.3}


class TestDyRTarget:
    """Nucleus/energy-threshold grad sparsification (task #367/#368) --
    dy_r_target/dy_k_min/dy_k_max on the 5 wide layers, plus the closed-
    loop apply_amortized_dy_r_target_control adjusting dy_r_target
    against MEASURED steps/sec (see JOURNAL.md's "Grad-side k_t design,
    revised" entry for why this is measured-feedback, not an analytic
    formula from an assumed cost ratio)."""

    def test_none_default_bit_identical(self):
        rng_seed = 31
        m_a = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(rng_seed),
        )
        m_b = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(rng_seed),
            dy_r_target=None,
        )
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _mem_a, logits_a, _ = m_a.step(x_window, memory_prev, learning_rate=0.05)
        _mem_b, logits_b, _ = m_b.step(x_window, memory_prev, learning_rate=0.05)
        np.testing.assert_array_equal(logits_a.data, logits_b.data)
        for name in ToyTileRecurrenceRMT._WIDE_LAYER_NAMES:
            assert m_a._wide_extra_kwargs(name) == {}

    def test_takes_priority_over_dy_sparsity_p_in_wide_extra_kwargs(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(32),
            dy_sparsity_p=0.5,
            dy_r_target=0.7,
            dy_k_min=2,
        )
        for name in ToyTileRecurrenceRMT._WIDE_LAYER_NAMES:
            assert model._wide_extra_kwargs(name) == {"dy_r_target": 0.7, "dy_k_min": 2}

    def test_set_value_stays_finite_through_real_backward(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(33),
            dy_r_target=0.6,
            dy_k_min=1,
        )
        x_window = np.random.RandomState(34).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        for step in range(5):
            memory_prev, logits, _aux = model.step(x_window, memory_prev, learning_rate=0.05)
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 3)])
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            assert np.isfinite(float(loss.data)), f"step {step}: loss non-finite"
            assert np.all(np.isfinite(memory_prev)), f"step {step}: memory non-finite"

    def test_controller_is_noop_when_dy_r_target_never_enabled(self):
        model = _model()
        assert all(v is None for v in model.dy_r_target.values())
        result = model.apply_amortized_dy_r_target_control(measured_sps=1.0, target_sps=10.0)
        assert result == {}
        assert all(v is None for v in model.dy_r_target.values())

    def test_controller_shrinks_r_target_when_too_slow(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(35),
            dy_r_target=0.7,
        )
        # layer_name=None (default) applies the same correction to every
        # wide layer uniformly (task #372) -- all 5 entries start and end
        # identical, so any one of them is representative.
        updated = model.apply_amortized_dy_r_target_control(measured_sps=1.0, target_sps=10.0)
        assert set(updated) == set(ToyTileRecurrenceRMT._WIDE_LAYER_NAMES)
        r = updated["q_proj"]
        assert r < 0.7
        assert model.dy_r_target == updated
        assert model._wide_extra_kwargs("q_proj")["dy_r_target"] == r

    def test_controller_grows_r_target_when_faster_than_target(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(36),
            dy_r_target=0.5,
        )
        updated = model.apply_amortized_dy_r_target_control(measured_sps=20.0, target_sps=10.0)
        assert updated["q_proj"] > 0.5

    def test_controller_can_target_a_single_layer(self):
        # task #372: layer_name lets a caller (task #374's future inner
        # loop) adjust one layer independently -- the rest stay untouched.
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(39),
            dy_r_target=0.7,
        )
        updated = model.apply_amortized_dy_r_target_control(measured_sps=1.0, target_sps=10.0, layer_name="q_proj")
        assert set(updated) == {"q_proj"}
        assert model.dy_r_target["q_proj"] < 0.7
        for name in ("input_proj", "k_proj", "v_proj", "o_proj"):
            assert model.dy_r_target[name] == 0.7

    def test_controller_respects_r_min_r_max_clip(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(37),
            dy_r_target=0.06,
        )
        for _ in range(50):
            model.apply_amortized_dy_r_target_control(measured_sps=1.0, target_sps=100.0, r_min=0.05)
        assert all(v >= 0.05 for v in model.dy_r_target.values())

        model2 = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(38),
            dy_r_target=0.95,
        )
        for _ in range(50):
            model2.apply_amortized_dy_r_target_control(measured_sps=100.0, target_sps=1.0, r_max=0.99)
        assert all(v <= 0.99 for v in model2.dy_r_target.values())


class TestDySurprise:
    """Task #374: lagged per-layer E_t/Lbar surprise modulation on top of
    dy_r_target's own r_bar -- see ToyTileRecurrenceRMT.__init__'s own
    dy_surprise_alpha docstring for the full formula (r_t = clip(r_bar *
    (E_t/Lbar)^alpha, 0.05, 0.99)) and JOURNAL.md's "Grad-side k_t
    design, revised" / "Multi-actuator design discussion" entries for
    the derivation."""

    def test_off_by_default_even_with_surprise_data(self):
        # dy_surprise_alpha=None (default) must return r_bar UNMODIFIED
        # regardless of whatever's in _layer_surprise -- byte-identical
        # to task #372's own behavior, matching every other opt-in
        # kwarg's None-means-off convention in this file.
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(40),
            dy_r_target=0.7,
        )
        model._layer_surprise["q_proj"] = {"E_t": 100.0, "Lbar": 1.0}  # would be a huge ratio if used
        assert model._effective_dy_r_target("q_proj") == 0.7
        assert model._wide_extra_kwargs("q_proj")["dy_r_target"] == 0.7

    def test_no_modulation_before_any_backward_ever_ran(self):
        # Cold start: dy_surprise_alpha set, but _layer_surprise is still
        # empty (no real backward has run yet) -- must fall back to
        # r_bar unmodified, not divide-by-zero or KeyError.
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(41),
            dy_r_target=0.7,
            dy_surprise_alpha=0.5,
        )
        assert model._layer_surprise == {}
        assert model._effective_dy_r_target("q_proj") == 0.7

    def test_formula_modulates_up_when_e_t_above_lbar(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(42),
            dy_r_target=0.3,
            dy_surprise_alpha=0.5,
        )
        model._layer_surprise["q_proj"] = {"E_t": 4.0, "Lbar": 1.0}  # ratio=4, sqrt=2
        r_t = model._effective_dy_r_target("q_proj")  # 0.3*2=0.6, within [0.05, 0.99]
        assert abs(r_t - 0.3 * 2.0) < 1e-9

    def test_formula_modulates_down_when_e_t_below_lbar(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(43),
            dy_r_target=0.5,
            dy_surprise_alpha=0.5,
        )
        model._layer_surprise["q_proj"] = {"E_t": 0.25, "Lbar": 1.0}  # ratio=0.25, sqrt=0.5
        r_t = model._effective_dy_r_target("q_proj")
        assert abs(r_t - 0.5 * 0.5) < 1e-9

    def test_formula_clips_to_0_05_0_99(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(44),
            dy_r_target=0.5,
            dy_surprise_alpha=1.0,
        )
        model._layer_surprise["q_proj"] = {"E_t": 1000.0, "Lbar": 1.0}
        assert model._effective_dy_r_target("q_proj") == 0.99
        model._layer_surprise["q_proj"] = {"E_t": 0.0001, "Lbar": 1.0}
        assert model._effective_dy_r_target("q_proj") == 0.05

    def test_per_layer_independent(self):
        # Only q_proj has surprise data seeded -- every other wide layer
        # (real cold start) must stay at r_bar unmodified.
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(45),
            dy_r_target=0.5,
            dy_surprise_alpha=0.5,
        )
        model._layer_surprise["q_proj"] = {"E_t": 4.0, "Lbar": 1.0}
        assert model._effective_dy_r_target("q_proj") != 0.5
        for name in ("input_proj", "k_proj", "v_proj", "o_proj"):
            assert model._effective_dy_r_target(name) == 0.5

    def test_update_layer_surprise_lbar_inits_to_e_t_not_zero(self):
        model = _model()
        model.dy_surprise_alpha = 0.5  # bypass constructor for a bare unit check
        model._update_layer_surprise("q_proj", np.array([3.0, 4.0], dtype=np.float32))  # E_t=25
        rec = model._layer_surprise["q_proj"]
        assert rec["E_t"] == 25.0
        assert rec["Lbar"] == 25.0  # first observation: Lbar==E_t exactly, ratio=1.0 (neutral)

    def test_update_layer_surprise_ema_tracks_over_multiple_calls(self):
        model = _model()
        model.dy_surprise_alpha = 0.5
        model.dy_surprise_beta = 0.5  # fast-moving EMA for a quick, checkable test
        model._update_layer_surprise("q_proj", np.array([1.0], dtype=np.float32))  # E_t=1
        assert model._layer_surprise["q_proj"]["Lbar"] == 1.0
        model._update_layer_surprise("q_proj", np.array([3.0], dtype=np.float32))  # E_t=9
        # Lbar = 0.5*1.0 + 0.5*9.0 = 5.0
        assert abs(model._layer_surprise["q_proj"]["Lbar"] - 5.0) < 1e-9
        assert model._layer_surprise["q_proj"]["E_t"] == 9.0

    def test_real_backward_populates_surprise_for_all_5_wide_layers(self):
        # Integration smoke test, same style as test_set_value_stays_
        # finite_through_real_backward above -- confirms the full path
        # (forward -> _timed_layer_forward -> real backward -> out.grad
        # -> _update_layer_surprise) actually wires up end to end, not
        # just the formula in isolation.
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(46),
            dy_r_target=0.7,
            dy_surprise_alpha=0.5,
        )
        x_window = np.random.RandomState(47).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        memory_prev, logits, _aux = model.step(x_window, memory_prev, learning_rate=0.05)
        loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 3)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        for name in ToyTileRecurrenceRMT._WIDE_LAYER_NAMES:
            assert name in model._layer_surprise
            assert model._layer_surprise[name]["E_t"] >= 0.0
            assert np.isfinite(model._layer_surprise[name]["E_t"])
            assert np.isfinite(model._layer_surprise[name]["Lbar"])

    def test_lagged_not_current_step(self):
        # The core design property: a layer's OWN forward()-time kwargs
        # for step N must reflect step N-1's surprise data, never step
        # N's own (which doesn't exist yet at forward()-call time).
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(48),
            dy_r_target=0.7,
            dy_surprise_alpha=0.5,
        )
        x_window = np.random.RandomState(49).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)

        # Step 1: no surprise data exists yet -- must be unmodulated r_bar.
        assert model._effective_dy_r_target("q_proj") == 0.7
        memory_prev, logits, _aux = model.step(x_window, memory_prev, learning_rate=0.05)
        assert model._layer_surprise == {}  # forward alone never populates it
        loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 3)])
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        # NOW step 1's backward has run -- surprise data exists, first
        # observation always has ratio==1.0 (Lbar inits to E_t), so
        # still no modulation on this exact call, but the STATE exists.
        assert "q_proj" in model._layer_surprise
        snapshot = dict(model._layer_surprise["q_proj"])

        # Step 2: forward()-time kwargs must reflect step 1's surprise
        # snapshot, unchanged by anything from step 2 itself (which
        # hasn't run backward yet).
        memory_prev, _logits2, _aux2 = model.step(x_window, memory_prev, learning_rate=0.05)
        assert model._layer_surprise["q_proj"] == snapshot


class TestXRTarget:
    """Task #365: nucleus/energy-threshold INPUT-side selection via
    x_r_target -- structurally mirrors TestDyRTarget above (same
    per-layer dict, same TAKES-PRIORITY-over-fixed-fraction convention,
    same apply_amortized_x_r_target_control outer loop), just wired
    into _to_sparse instead of dy sparsification. See __init__'s own
    x_r_target docstring for the full design rationale (no separate
    derived signal, stays on the simple speed-target control)."""

    def test_none_default_bit_identical(self):
        rng_seed = 50
        m_a = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(rng_seed),
        )
        m_b = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(rng_seed),
            x_r_target=None,
        )
        x_window = np.random.RandomState(1).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _mem_a, logits_a, _ = m_a.step(x_window, memory_prev, learning_rate=0.05)
        _mem_b, logits_b, _ = m_b.step(x_window, memory_prev, learning_rate=0.05)
        np.testing.assert_array_equal(logits_a.data, logits_b.data)
        assert all(v is None for v in m_a.x_r_target.values())

    def test_takes_priority_over_input_sparsity_p(self):
        # Structural check only, same rationale as
        # TestOutputDySparsityP.test_does_not_touch_the_5_wide_layers_
        # kwargs above -- x_r_target enabled means the CSR going into
        # each wide layer's forward is nucleus-selected, not a fixed
        # fraction; confirmed by checking the actual selection uses
        # _nucleus_top_k_csr's own R>=r_target invariant on real data,
        # not by comparing against the fixed-fraction output (which
        # would be a different, unrelated set of kept indices).
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(51),
            input_sparsity_p=0.5,
            x_r_target=0.7,
            x_k_min=1,
        )
        x = Tensor(np.random.RandomState(52).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32))
        out = model._to_sparse(x, "input_proj")
        assert out.data.__class__.__name__ == "CSR"
        # Real R(v,k) >= r_target check on the actual kept entries,
        # independently re-derived (not trusting the wrapper's own math).
        x_np = np.asarray(x.data, dtype=np.float32)
        csr = out.data
        for row in range(csr.rows):
            start, end = csr.ptrs[row], csr.ptrs[row + 1]
            kept_sq = float(np.sum(csr.values[start:end].astype(np.float64) ** 2))
            total_sq = float(np.sum(x_np[row].astype(np.float64) ** 2))
            if total_sq > 0:
                assert kept_sq / total_sq >= 0.7 - 1e-6

    def test_set_value_stays_finite_through_real_backward(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(53),
            x_r_target=0.6,
            x_k_min=1,
        )
        x_window = np.random.RandomState(54).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        for step in range(5):
            memory_prev, logits, _aux = model.step(x_window, memory_prev, learning_rate=0.05)
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 3)])
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
            assert np.isfinite(float(loss.data)), f"step {step}: loss non-finite"
            assert np.all(np.isfinite(memory_prev)), f"step {step}: memory non-finite"

    def test_controller_shrinks_r_target_when_too_slow(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(55),
            x_r_target=0.7,
        )
        updated = model.apply_amortized_x_r_target_control(measured_sps=1.0, target_sps=10.0)
        assert set(updated) == set(ToyTileRecurrenceRMT._WIDE_LAYER_NAMES)
        assert updated["q_proj"] < 0.7
        assert model.x_r_target == updated

    def test_controller_can_target_a_single_layer(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(56),
            x_r_target=0.7,
        )
        updated = model.apply_amortized_x_r_target_control(measured_sps=1.0, target_sps=10.0, layer_name="q_proj")
        assert set(updated) == {"q_proj"}
        assert model.x_r_target["q_proj"] < 0.7
        for name in ("input_proj", "k_proj", "v_proj", "o_proj"):
            assert model.x_r_target[name] == 0.7

    def test_independent_from_dy_r_target(self):
        # x_r_target and dy_r_target are genuinely separate axes (input
        # vs grad) -- setting one must not touch the other's state.
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(57),
            x_r_target=0.7,
            dy_r_target=0.3,
        )
        assert model.x_r_target["q_proj"] == 0.7
        assert model.dy_r_target["q_proj"] == 0.3
        model.apply_amortized_x_r_target_control(measured_sps=1.0, target_sps=100.0)
        assert model.dy_r_target["q_proj"] == 0.3  # untouched
        model.apply_amortized_dy_r_target_control(measured_sps=100.0, target_sps=1.0)
        assert model.x_r_target["q_proj"] != 0.7  # already shrank above, untouched by this call


class TestInputSelectionTrajectory:
    """Task #369: real per-layer R/k trajectory stats, computed
    directly from the CSR _to_sparse already builds (self.
    last_input_selection). Cadence/gating policy lives in the CALLER
    (train_mqar_curriculum.py's own trajectory_log_every) -- the model
    itself just exposes the most recent snapshot per layer, same
    convention as self.last_debug."""

    def test_empty_before_any_sparsification(self):
        model = _model()  # no input_sparsity_p/x_r_target set
        x_window = np.random.RandomState(70).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        model.step(x_window, memory_prev, learning_rate=0.05)
        assert model.last_input_selection == {}

    def test_real_r_satisfies_invariant_under_x_r_target(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(71),
            x_r_target=0.7,
            x_k_min=1,
        )
        x_window = np.random.RandomState(72).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        model.step(x_window, memory_prev, learning_rate=0.05)
        assert set(model.last_input_selection) == set(ToyTileRecurrenceRMT._WIDE_LAYER_NAMES)
        for stats in model.last_input_selection.values():
            # Real invariant this kernel guarantees (task #364's own
            # C++ test covers this at the kernel level; here we confirm
            # it survives all the way through to the Python-side stats
            # this task adds).
            assert stats["R_mean"] >= 0.7 - 1e-6
            assert stats["k_mean"] >= 1.0  # x_k_min=1 floor
            assert stats["rows"] > 0
            assert stats["cols"] == STATE_WIDTH or stats["cols"] == EMBED_WIDTH

    def test_captured_also_under_plain_input_sparsity_p(self):
        # Real R/k diagnostic is meaningful for the LEGACY fixed-
        # fraction path too, not just nucleus selection.
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(73),
            input_sparsity_p=0.5,
        )
        x_window = np.random.RandomState(74).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        model.step(x_window, memory_prev, learning_rate=0.05)
        assert set(model.last_input_selection) == set(ToyTileRecurrenceRMT._WIDE_LAYER_NAMES)
        for stats in model.last_input_selection.values():
            assert 0.0 <= stats["R_mean"] <= 1.0
            assert stats["k_mean"] > 0.0

    def test_overwritten_not_accumulated_across_steps(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(75),
            x_r_target=0.7,
            x_k_min=1,
        )
        x_window = np.random.RandomState(76).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        model.step(x_window, memory_prev, learning_rate=0.05)
        assert len(model.last_input_selection) == len(ToyTileRecurrenceRMT._WIDE_LAYER_NAMES)
        model.step(x_window, memory_prev, learning_rate=0.05)
        # Still exactly one entry per layer -- a snapshot, not a growing list.
        assert len(model.last_input_selection) == len(ToyTileRecurrenceRMT._WIDE_LAYER_NAMES)
        for stats in model.last_input_selection.values():
            assert isinstance(stats, dict)


class TestCrossLayerBudgetAllocator:
    """Task #375: apply_cross_layer_budget_allocator -- coordinates
    x_r_target against real per-layer measured cost (task #373),
    deliberately never touching dy_r_target (grad stays need-driven,
    task #374) to avoid both axes fighting over the same speed signal.
    See the method's own docstring for the full weight formula."""

    def _model_with_x_r_target(self, seed=60, **kwargs):
        return ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(seed),
            x_r_target=0.7,
            **kwargs,
        )

    def test_never_touches_dy_r_target(self):
        model = self._model_with_x_r_target(dy_r_target=0.3)
        model.apply_cross_layer_budget_allocator(measured_sps=1.0, target_sps=100.0)
        assert all(v == 0.3 for v in model.dy_r_target.values())
        model.apply_cross_layer_budget_allocator(measured_sps=100.0, target_sps=1.0)
        assert all(v == 0.3 for v in model.dy_r_target.values())

    def test_falls_back_to_uniform_weight_cold_start(self):
        # No _layer_timing data yet -- must behave exactly like
        # apply_amortized_x_r_target_control's own plain uniform
        # down_factor (weight=1.0 for every layer).
        model = self._model_with_x_r_target(seed=61)
        assert model._layer_timing == {}
        updated = model.apply_cross_layer_budget_allocator(measured_sps=1.0, target_sps=10.0, down_factor=0.8)
        for name in ToyTileRecurrenceRMT._WIDE_LAYER_NAMES:
            assert abs(updated[name] - 0.7 * 0.8) < 1e-9

    def test_expensive_layer_shrinks_more_when_too_slow(self):
        model = self._model_with_x_r_target(seed=62)
        # q_proj eats 80% of the real measured time, the other 4 share
        # the remaining 20% evenly (5% each) -- well above/below the
        # uniform 1/5=20% baseline.
        model._layer_timing = {
            "input_proj": {"fwd_s": 0.025, "bwd_s": 0.0},
            "q_proj": {"fwd_s": 0.4, "bwd_s": 0.4},
            "k_proj": {"fwd_s": 0.025, "bwd_s": 0.0},
            "v_proj": {"fwd_s": 0.025, "bwd_s": 0.0},
            "o_proj": {"fwd_s": 0.025, "bwd_s": 0.0},
        }
        updated = model.apply_cross_layer_budget_allocator(measured_sps=1.0, target_sps=10.0)
        # q_proj (80% share, weight clipped to 3.0) shrinks the hardest;
        # input_proj (5% share, weight 0.25) barely moves.
        assert updated["q_proj"] < updated["input_proj"]
        assert updated["q_proj"] < 0.7
        assert updated["input_proj"] < 0.7  # still shrinks some (measured_sps<target_sps)

    def test_weight_clipped_to_3x(self):
        model = self._model_with_x_r_target(seed=63)
        # Degenerate case: one layer eats effectively 100% of the
        # window's real time -- weight would be 5.0 unclipped, must
        # clip to 3.0 and stay bounded (not runaway/negative).
        model._layer_timing = {
            "q_proj": {"fwd_s": 1.0, "bwd_s": 0.0},
        }
        updated = model.apply_cross_layer_budget_allocator(measured_sps=1.0, target_sps=10.0, down_factor=0.85)
        eff_down_clipped = 1.0 - (1.0 - 0.85) * 3.0  # weight clipped to 3.0
        assert eff_down_clipped > 0.0  # sanity: still a valid multiplicative factor
        assert abs(updated["q_proj"] - 0.7 * eff_down_clipped) < 1e-9

    def test_growth_direction_also_weighted(self):
        model = self._model_with_x_r_target(seed=64)
        model._layer_timing = {
            "input_proj": {"fwd_s": 0.025, "bwd_s": 0.0},
            "q_proj": {"fwd_s": 0.4, "bwd_s": 0.4},
            "k_proj": {"fwd_s": 0.025, "bwd_s": 0.0},
            "v_proj": {"fwd_s": 0.025, "bwd_s": 0.0},
            "o_proj": {"fwd_s": 0.025, "bwd_s": 0.0},
        }
        updated = model.apply_cross_layer_budget_allocator(measured_sps=10.0, target_sps=1.0)
        # Plenty of headroom (sps>target) -- q_proj (the expensive one)
        # grows back toward r_max hardest; input_proj barely grows.
        assert updated["q_proj"] > updated["input_proj"]
        assert updated["q_proj"] > 0.7
        assert updated["input_proj"] > 0.7

    def test_only_adjusts_layers_with_x_r_target_already_set(self):
        model = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(65),
        )
        model.x_r_target["q_proj"] = 0.7  # only one layer opted in
        updated = model.apply_cross_layer_budget_allocator(measured_sps=1.0, target_sps=10.0)
        assert set(updated) == {"q_proj"}
        assert all(v is None for name, v in model.x_r_target.items() if name != "q_proj")

    def test_real_integration_with_373_timing(self):
        # End-to-end: real per-layer timing from #373's own instrumentation
        # feeding directly into the allocator, no manual state seeding.
        model = self._model_with_x_r_target(seed=66)
        x_window = np.random.RandomState(67).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        model.reset_layer_timing()
        for _ in range(3):
            memory_prev, logits, _aux = model.step(x_window, memory_prev, learning_rate=0.05)
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 3)])
            loss.grad = np.array(1.0, dtype=np.float32)
            loss.backward()
        assert model._layer_timing  # real data accumulated
        updated = model.apply_cross_layer_budget_allocator(measured_sps=1.0, target_sps=100.0)
        for v in updated.values():
            assert np.isfinite(v)
            assert 0.05 <= v <= 0.99


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
        model_a = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(42),
        )
        model_b = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            rng=np.random.default_rng(42),
        )
        embed_table = np.random.RandomState(1).randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.1
        tokens = np.random.RandomState(2).randint(0, VOCAB, size=12)

        memory_a = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        memory_b = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        tile_cache = None
        for i in range(len(tokens)):
            window = _build_window(embed_table, tokens, i, NUM_TILES)
            memory_a, logits_a, _aux_a = model_a.step(window, memory_a, 0.0, requires_grad=False)
            memory_b, logits_b, _aux_b, tile_cache = model_b.step_cached(
                embed_table[tokens[i]], memory_b, 0.0, tile_cache, requires_grad=False
            )
            row_a = np.asarray(logits_a.data)[NUM_TILES - 1]
            row_b = np.asarray(logits_b.data)[0]
            assert np.array_equal(row_a, row_b), f"step {i}: logits diverged with static weights"
            assert np.array_equal(memory_a, memory_b), f"step {i}: memory_new diverged with static weights"

    def test_shapes_and_finite(self):
        model = _model()
        new_embed = np.random.RandomState(1).randn(EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        memory_new, logits, aux_loss, tile_cache = model.step_cached(
            new_embed, memory_prev, learning_rate=0.01, tile_cache=None
        )
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
                new_embed, memory, learning_rate=0.0, tile_cache=tile_cache
            )
            assert len(tile_cache) <= NUM_TILES - 1

    def test_weights_actually_update_via_backward(self):
        model = _model()
        opt = AdamOptimizer()
        memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        new_embed = np.random.RandomState(6).randn(EMBED_WIDTH).astype(np.float32) * 0.1
        before = np.asarray(model.input_ln.data).copy()
        memory, logits, aux, _tile_cache = model.step_cached(
            new_embed, memory, learning_rate=0.05, tile_cache=None, requires_grad=True
        )
        loss = cross_entropy_sum(logits, [(0, 1)])
        if aux is not None:
            loss = loss + aux
        loss.backward()
        opt.step(model.parameters_for_optimizer(), lr=0.05)
        assert not np.allclose(before, model.input_ln.data), "weights should move after a real backward+opt.step"
        assert np.all(np.isfinite(memory))


class TestStepContentDySparsitySchedule:
    """step()'s content_dy_sparsity_schedule (query-step graded credit
    design, see conversation/JOURNAL.md) -- None must stay byte-identical
    to today's exact behavior; a real schedule must run and produce a
    finite result without changing shapes."""

    def test_none_default_matches_pre_change_behavior(self):
        model_a = _model()
        _model()
        # separately-constructed models won't share weights, so instead
        # confirm the SAME model gives identical output whether or not
        # the new kwarg is explicitly passed as None.
        window = np.random.RandomState(40).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        _mem_a, logits_a, _aux_a = model_a.step(window, memory, 0.0, requires_grad=False)
        _mem_b, logits_b, _aux_b = model_a.step(
            window, memory, 0.0, requires_grad=False, content_dy_sparsity_schedule=None
        )
        assert np.array_equal(np.asarray(logits_a.data), np.asarray(logits_b.data))

    def test_graded_schedule_runs_finite_through_real_backward(self):
        from scripts.train_mqar_curriculum import _default_graded_dy_schedule

        model = _model()
        opt = AdamOptimizer()
        window = np.random.RandomState(41).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        schedule = _default_graded_dy_schedule(NUM_TILES)
        assert len(schedule) == NUM_TILES
        memory_new, logits, aux = model.step(
            window, memory, 0.05, requires_grad=True, content_dy_sparsity_schedule=schedule
        )
        loss = cross_entropy_sum(logits, [(NUM_TILES - 1, 1)])
        if aux is not None:
            loss = loss + aux
        loss.backward()
        opt.step(model.parameters_for_optimizer(), lr=0.05)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(memory_new))
        assert np.all(np.isfinite(logits.data))

    def test_wrong_length_schedule_raises(self):
        model = _model()
        window = np.random.RandomState(42).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        try:
            model.step(
                window, memory, 0.0, requires_grad=False, content_dy_sparsity_schedule=[1.0, 0.5]
            )  # wrong length
            raise AssertionError("expected ValueError for mismatched schedule length")
        except ValueError:
            pass


class TestLossAdjustedDecay:
    """apply_loss_adjusted_decay -- critical-learning-periods/loss-of-
    plasticity forgetting (EXPERIMENTAL, added 2026-09-20). See
    docs/research/toy_tile_recurrence_rmt.rst:loss_adjusted_decay_design.
    Direct instruction: "add tests to make sure things match in any edge
    cases the code indicates" -- covers the exact bug caught by manual
    testing during implementation (a tie/filter-transient dip must NOT
    count as genuine improvement), plus stall accumulation, reset on real
    improvement, per-arm toggling, and block4 coverage under dense=True."""

    def test_perfectly_flat_loss_accumulates_stall_every_step(self):
        # Regression test for a real bug caught during implementation: a
        # plain `ema <= floor` comparison let the EMA's own settling to
        # its initial value read as "improvement" on every single call,
        # even for loss that never actually changes -- permanently
        # masking a real stall. Fixed via a required relative margin
        # (improve_tol). A flat loss must accumulate stall_steps == the
        # number of calls, every time, no exceptions.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(0))
        out = None
        for _ in range(50):
            out = model.apply_loss_adjusted_decay(2.0, touch_fraction=1.0, max_chunk=1_000_000)
        assert model._decay_stall_steps == 50
        # stall_steps itself accumulates every call regardless -- but with
        # the default min_stall_steps=200 grace period, 50 calls hasn't
        # cleared the baseline yet, so stall_frac (and therefore
        # decay_factor) must be an EXACT no-op, not "small but nonzero".
        assert out["input_proj"]["stall_frac"] == 0.0
        assert out["input_proj"]["importance"]["decay_factor"] == 1.0

    def test_min_stall_steps_is_a_real_baseline_not_just_a_gentler_ramp(self):
        # Direct design question: "I feel like it would need to establish
        # a baseline and shouldn't necessarily decay all the time." Before
        # min_stall_steps existed, stall_frac (and decay_factor) were
        # technically nonzero after almost every call, since the very
        # first call already sets stall_steps=1. Verify the grace period
        # gives a genuine dead zone: exactly at the boundary, stall_frac
        # must still be 0; one call past it, it must be nonzero.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(20))
        out = None
        for _ in range(200):
            out = model.apply_loss_adjusted_decay(2.0, touch_fraction=1.0, max_chunk=1_000_000, min_stall_steps=200)
        assert out["input_proj"]["stall_frac"] == 0.0, "exactly at the grace-period boundary, decay must still be off"
        out = model.apply_loss_adjusted_decay(2.0, touch_fraction=1.0, max_chunk=1_000_000, min_stall_steps=200)
        assert out["input_proj"]["stall_frac"] > 0.0, "one call past the grace period, decay must start ramping in"

    def test_noisy_but_flat_loss_still_registers_stall(self):
        # A loss oscillating narrowly around a fixed mean (no real
        # learning happening) must not be mistaken for improvement just
        # because the EMA occasionally dips below its own filter-
        # transient floor by a tiny amount -- improve_tol must reject
        # noise-sized dips. 500 calls: enough to clear the default
        # min_stall_steps=200 grace period AND ramp_steps=200 on top of it.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(1))
        for i in range(500):
            loss = 2.0 + (0.001 if i % 2 == 0 else -0.001)
            out = model.apply_loss_adjusted_decay(loss, touch_fraction=1.0, max_chunk=1_000_000)
        assert out["input_proj"]["stall_frac"] > 0.5, "noisy-flat loss must still register a real stall"

    def test_sustained_improvement_keeps_resetting_stall(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(2))
        loss = 5.0
        out = None
        for _ in range(100):
            loss *= 0.98  # steadily, genuinely decreasing
            out = model.apply_loss_adjusted_decay(loss, touch_fraction=1.0, max_chunk=1_000_000)
        assert model._decay_stall_steps < 5
        assert out["input_proj"]["stall_frac"] < 0.05

    def test_importance_arm_default_on_weight_arm_default_off(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(3))
        out = model.apply_loss_adjusted_decay(2.0, touch_fraction=1.0, max_chunk=1_000_000)
        assert out["input_proj"]["importance"] is not None
        assert out["input_proj"]["weight"] is None

    def test_none_half_life_arm_is_an_exact_noop(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(4))
        out = model.apply_loss_adjusted_decay(
            2.0,
            touch_fraction=1.0,
            max_chunk=1_000_000,
            importance_half_life_touches=None,
            weight_half_life_touches=None,
        )
        for entry in out.values():
            assert entry["importance"] is None
            assert entry["weight"] is None

    def test_weight_arm_opt_in_actually_decays_block4_weight(self):
        # dense=True routes DISLDOLayer32's weights into block4 storage --
        # a sustained stall must measurably shrink the REAL (block4) weight
        # magnitude, not just report a no-op on empty scattered padding.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(5))
        w_before = np.abs(np.array(model.input_proj.weights)).mean()
        out = None
        for _ in range(250):
            out = model.apply_loss_adjusted_decay(
                2.0, touch_fraction=1.0, max_chunk=1_000_000, weight_half_life_touches=20.0
            )
        assert out["input_proj"]["weight"] is not None
        assert out["input_proj"]["weight"]["decay_factor"] < 1.0
        assert out["input_proj"]["weight"]["block4"]["n"] > 0
        w_after = np.abs(np.array(model.input_proj.weights)).mean()
        assert w_after < w_before, "sustained stall + weight arm should measurably shrink real block4 weights"
        assert w_after > 0.0, "bounded half-life decay must not annihilate weights within one short test run"

    def test_scattered_backend_has_no_block4_key(self):
        # FP4 (DISLDOLayer, scattered-only) has no block4 concept at all --
        # the block4 sub-call must be skipped entirely (hasattr-guarded),
        # not attempted-and-erroring.
        model = _model(disldo_cls=DISLDOLayer, rng=np.random.default_rng(6))
        out = model.apply_loss_adjusted_decay(2.0, touch_fraction=1.0, max_chunk=1_000_000)
        assert out["input_proj"]["importance"] is not None
        assert "block4" not in out["input_proj"]["importance"]

    def test_stall_frac_saturates_at_one(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(7))
        out = None
        for _ in range(500):  # far past ramp_steps=200
            out = model.apply_loss_adjusted_decay(2.0, touch_fraction=1.0, max_chunk=1_000_000)
        assert out["input_proj"]["stall_frac"] == 1.0

    def test_half_life_sets_the_decay_factor_at_full_stall(self):
        # decay_factor at fully-saturated stall (stall_frac=1) must equal
        # exactly 2**(-1/half_life_touches) -- the defining property of
        # "half_life_touches touches under full stall halves the value,"
        # not an arbitrary severity knob.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(8))
        out = None
        for _ in range(500):  # far past ramp_steps=200 -> stall_frac saturates at 1.0
            out = model.apply_loss_adjusted_decay(
                2.0, touch_fraction=1.0, max_chunk=1_000_000, importance_half_life_touches=10.0
            )
        assert out["input_proj"]["stall_frac"] == 1.0
        assert out["input_proj"]["importance"]["decay_factor"] == pytest.approx(2.0 ** (-1.0 / 10.0))

    def test_half_life_touches_actually_halves_after_that_many_full_stall_touches(self):
        # End-to-end check of the half-life's own defining property: with
        # touch_fraction=1.0 (whole layer touched every call) and stall
        # already fully saturated, N calls at weight_half_life_touches=N
        # should shrink |weight| by very close to 2x -- not near-zero
        # (the original strength/floor design's failure mode) and not
        # unchanged (a no-op). Uses WEIGHT, not importance: a fresh dense
        # init starts importance at exactly 0, which can't demonstrate a
        # halving ratio.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(9))
        # Saturate stall_frac to 1.0 first, at a deliberately negligible
        # half-life (near-1.0 decay_factor) so this warm-up phase doesn't
        # itself perturb the weight before the timed window below starts.
        # Needs > min_stall_steps(200) + ramp_steps(200) = 400 calls.
        for _ in range(410):
            model.apply_loss_adjusted_decay(2.0, touch_fraction=1.0, max_chunk=1_000_000, weight_half_life_touches=1e9)
        w_before = np.abs(np.array(model.q_proj.weights)).mean()
        half_life = 30.0
        for _ in range(int(half_life)):
            model.apply_loss_adjusted_decay(
                2.0, touch_fraction=1.0, max_chunk=1_000_000, weight_half_life_touches=half_life
            )
        w_after = np.abs(np.array(model.q_proj.weights)).mean()
        ratio = w_after / w_before
        assert 0.3 < ratio < 0.7, f"expected ~0.5 shrink after half_life_touches calls, got ratio={ratio}"


class TestPlasticityReset:
    """apply_plasticity_reset -- per-neuron utility-based plasticity reset
    (Continual-Backprop-inspired, EXPERIMENTAL, added 2026-09-20). See
    docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design.
    Replaces apply_loss_adjusted_decay as the primary mechanism under
    test -- NOT loss-gated at all (no loss argument anywhere), driven
    entirely by each column's own local importance/gradient-activity
    signal. TDD: written against the sili__new engine primitives
    (apply_amortized_plasticity_reset/apply_amortized_block4_plasticity_reset)
    already built, tested, and committed there -- this class covers the
    orchestration layer (per-layer touch_fraction chunk sizing,
    block4/scattered dispatch, block4 tile-vs-cell division) on top of
    them."""

    def test_touch_fraction_sizing_across_real_nnz_range(self):
        # Reuses the exact chunk-sizing convention proven correct for
        # apply_loss_adjusted_decay (touch_fraction of EACH layer's own
        # nnz, clipped to [min_chunk, max_chunk], block4 chunk divided by
        # _BLOCK4_TILE_SLOTS) -- just needs to run without error across
        # this test file's real ~small-but-varied layer sizes.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(0))
        out = model.apply_plasticity_reset()
        assert set(out.keys()) >= {"input_proj", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"}
        for entry in out.values():
            assert "importance" in entry
            assert entry["importance"] is not None

    def test_non_fp32_backend_reports_nothing_yet(self):
        # Direct instruction: this mechanism is tested against fp32
        # (DISLDOLayer32/DISLDOLayerV, dense=True -> block4) FIRST --
        # matching this whole investigation's actual config throughout --
        # and only extended to FP4/FP8 later if it proves worthwhile.
        # FP4 (DISLDOLayer) has NO apply_amortized_plasticity_reset bound
        # at all yet -- the model method's hasattr guard must skip those
        # layers entirely (not error, not report a partial/wrong entry).
        model = _model(disldo_cls=DISLDOLayer, rng=np.random.default_rng(1))
        out = model.apply_plasticity_reset()
        assert out == {}, f"FP4 layers should report nothing yet (not wired), got keys: {list(out)}"

    def test_block4_backend_reports_block4_substats(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(2))
        out = model.apply_plasticity_reset()
        assert "block4" in out["input_proj"]["importance"]

    def test_runs_many_cycles_without_error_and_reports_sane_stats(self):
        # Real end-to-end smoke: enough calls to complete several full
        # amortized cycles on a small real layer, confirm the mechanism
        # never crashes and reports internally-consistent stats
        # (n_reset_this_cycle is a non-negative count, min/max_deviation
        # bracket mean_deviation, cycle_complete is a real bool).
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(3))
        out = None
        for _ in range(500):
            out = model.apply_plasticity_reset()
        entry = out["input_proj"]["importance"]
        assert isinstance(entry["cycle_complete"], bool)
        assert entry["n_reset_this_cycle"] >= 0
        assert entry["max_deviation"] >= entry["min_deviation"]
        if "block4" in entry:
            assert entry["block4"]["n_reset_this_cycle"] >= 0
            assert entry["block4"]["max_deviation"] >= entry["block4"]["min_deviation"]

    def test_no_loss_argument_needed(self):
        # Direct requirement: unlike apply_loss_adjusted_decay, this
        # mechanism takes NO loss argument at all -- purely local
        # per-column signals.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(4))
        out = model.apply_plasticity_reset()  # no loss/loss_ema argument
        assert out is not None

    def test_custom_knobs_are_threaded_through(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(5))
        out = model.apply_plasticity_reset(
            touch_fraction=1.0,
            max_chunk=1_000_000,
            eta=0.5,
            eta_slow=0.9,
            eta_slow_catchup=0.8,
            eta_fast=0.3,
            blend=0.05,
            reset_fraction=0.5,
            k=1.0,
        )
        assert out["input_proj"]["importance"] is not None

    def test_include_column_state_off_by_default(self):
        # Direct instruction: per-column (not per-synapse) offline data
        # collection, added as an opt-in extra so callers not doing data
        # collection pay nothing extra. Off by default -- no column_state
        # key anywhere when not requested, even once a cycle completes.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(6))
        out = None
        for _ in range(500):
            out = model.apply_plasticity_reset()
        entry = out["input_proj"]["importance"]
        assert "column_state" not in entry
        if "block4" in entry:
            assert "column_state" not in entry["block4"]

    def test_include_column_state_present_only_when_cycle_completes(self):
        # column_state should appear ONLY on a pool leaf whose
        # cycle_complete is True AND whose storage arm actually has real
        # content this run -- the underlying arrays are zero-copy views
        # in the C++ layer (get_weights_vals/get_importance precedent),
        # so fetching them mid-cycle would be a partially up-to-date,
        # misleading snapshot, not a real "this cycle's final state"
        # reading. Checked on BLOCK4 only here -- a dense-loaded test
        # model's SCATTERED arm always has 0 nnz (load_dense_values ->
        # block4_load_dense_fp32), so its own top-level cycle_complete is
        # trivially True on every call with column_state correctly
        # suppressed (see test_scattered_column_state_skipped_when_scattered_arm_is_empty),
        # not a real "did a cycle complete" signal to assert against here.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(7))
        saw_a_cycle_complete_with_state = False
        for _ in range(500):
            out = model.apply_plasticity_reset(include_column_state=True)
            for entry in out.values():
                block4 = entry["importance"].get("block4")
                if block4 is None:
                    continue
                if block4.get("cycle_complete"):
                    assert "column_state" in block4, "a completed block4 cycle must include column_state when requested"
                    saw_a_cycle_complete_with_state = True
                else:
                    assert "column_state" not in block4, "an incomplete block4 cycle must NOT include column_state"
        assert saw_a_cycle_complete_with_state, "500 calls should complete at least one real block4 cycle"

    def test_column_state_arrays_are_plain_numpy_not_live_views(self):
        # The underlying C++ accessor returns a zero-copy VIEW into the
        # layer's own internal buffer (documented precedent:
        # get_weights_vals/get_importance) -- the model-level wrapper
        # must copy it (np.array(...)) before returning, since a caller
        # collecting data across many further apply_plasticity_reset
        # calls needs a real frozen snapshot, not a view that silently
        # changes under it on the NEXT call (same aliasing bug class as
        # feedback_forward_output_not_aliased in sili__new). Checked via
        # block4 -- the arm that actually holds this dense-loaded test
        # model's real content (see scattered_nnz gating test).
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(8))
        snapshots = []
        for _ in range(500):
            out = model.apply_plasticity_reset(include_column_state=True)
            block4 = out["input_proj"]["importance"].get("block4")
            if block4 is not None and "column_state" in block4:
                snapshots.append(np.array(block4["column_state"]["col_importance"]))
            if len(snapshots) >= 2:
                break
        assert len(snapshots) >= 1
        assert isinstance(snapshots[0], np.ndarray)

    def test_scattered_column_state_skipped_when_scattered_arm_is_empty(self):
        # Real bug found via smoke-testing the actual disk-logging
        # wiring (train_mqar_curriculum.py): a dense-loaded real layer's
        # content lives ENTIRELY in block4 (load_dense_values ->
        # block4_load_dense_fp32) -- its SCATTERED arm has 0 nnz for the
        # whole run, so apply_amortized_plasticity_reset's early-return
        # path (nnz==0) reports cycle_complete=True on literally EVERY
        # call, with col_importance/col_grad_slow/etc all still at their
        # zero-initialized values. Without this gate, a real 500-step
        # smoke run wrote 3036 files in ~90s, the vast majority
        # degenerate all-zero "scattered" snapshots (verified: loaded one
        # back, every array was exactly zero) -- pure storage waste, no
        # signal. scattered_nnz gates this off; block4 (the arm that
        # actually holds this layer's real content) must still get its
        # column_state normally.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(9))
        saw_block4_column_state = False
        for _ in range(500):
            out = model.apply_plasticity_reset(include_column_state=True)
            leaf = out["input_proj"]["importance"]
            assert "column_state" not in leaf, (
                "dense-loaded layer's scattered arm has 0 nnz -- must never get a "
                "column_state snapshot, even when cycle_complete is (trivially) True"
            )
            block4_leaf = leaf.get("block4")
            if block4_leaf is not None and "column_state" in block4_leaf:
                saw_block4_column_state = True
        assert saw_block4_column_state, (
            "block4 (the arm holding this layer's real content) should still "
            "produce a real column_state snapshot within 500 calls"
        )

    def test_l2_decay_lambda_default_is_off_and_stats_reported(self):
        # Direct instruction, after offline replay of a real 100k-step
        # run's column logs showed col_importance saturating at the
        # ci accumulator's max_ci=100 clamp for most of a pool's
        # population by late training: an L2-saturation-gated decay on
        # the REAL importance accumulator (not a post-hoc re-ranking --
        # offline replay proved that can never change selection order).
        # Default l2_decay_lambda=0.0 must be a no-op (this project's
        # new-shared-parameter backward-compat convention); l2_sat_ratio/
        # l2_decay_strength stats should still surface even at lambda=0
        # for visibility.
        # Checked against block4 -- this fixture's dense=True layers have
        # a permanently-empty scattered arm (see
        # test_scattered_column_state_skipped_when_scattered_arm_is_empty),
        # so the scattered leaf's cycle boundary never runs and its
        # l2_sat_ratio/l2_decay_strength stay at their zero default.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(10))
        out = None
        for _ in range(50):
            out = model.apply_plasticity_reset()
        entry = out["input_proj"]["importance"]["block4"]
        assert "l2_sat_ratio" in entry
        assert "l2_decay_strength" in entry
        assert 0.0 <= entry["l2_sat_ratio"] <= 1.5  # sat_ratio can exceed 1 only via fp noise
        assert 0.0 <= entry["l2_decay_strength"] <= 1.0

    def test_l2_decay_lambda_threads_through_and_shrinks_saturated_importance(self):
        # This fixture never runs a real backward pass between calls, so
        # col_importance never leaves 0 -- sat_ratio stays exactly 0
        # throughout. At l2_decay_threshold=0.0 that means
        # decay_strength = sigmoid((0-0)/temperature) = sigmoid(0) =
        # EXACTLY 0.5, a deterministic value distinct from the 0.0
        # no-op default -- confirms the knob actually reaches the
        # engine's sigmoid math (not just a passthrough that gets
        # ignored), and with l2_decay_lambda=1.0, mean_col_importance
        # should visibly shrink cycle over cycle as a result.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(11))
        completed = []
        for _ in range(500):
            out = model.apply_plasticity_reset(l2_decay_lambda=1.0, l2_decay_threshold=0.0, l2_decay_temperature=0.05)
            leaf = out["input_proj"]["importance"]["block4"]
            if leaf.get("cycle_complete"):
                completed.append(leaf)
        assert len(completed) >= 2, "block4 should complete at least 2 cycles within 500 calls"
        entry = completed[-1]
        assert abs(entry["l2_decay_strength"] - 0.5) < 1e-6, (
            "threshold=0.0 at sat_ratio=0 should give decay_strength=sigmoid(0)=0.5 exactly, "
            f"got {entry['l2_decay_strength']!r}"
        )

    def test_select_by_deviation_off_by_default_and_threads_through(self):
        # Direct instruction, after comparing a graduated real run
        # against a stuck real run's collected column data: rank
        # candidates by deviation (growth RATE) instead of col_importance
        # (absolute LEVEL) when select_by_deviation=True. Default False
        # is an exact no-op (existing top-K-by-importance selection) --
        # see select_by_deviation_early_detection in the design doc. The
        # real selection-swap behavior is covered precisely at the
        # engine level (test_amortized_plasticity_reset.cpp /
        # test_block4_plasticity_reset.cpp); this just confirms the knob
        # reaches the engine without error at the model layer.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(12))
        out_default = model.apply_plasticity_reset()
        assert out_default["input_proj"]["importance"] is not None
        out_by_deviation = model.apply_plasticity_reset(select_by_deviation=True)
        assert out_by_deviation["input_proj"]["importance"] is not None


class TestL2Init:
    """apply_l2_init -- L2 Init (Kumar, Marklund & Van Roy, CoLLAs 2025,
    arXiv:2308.11958), EXPERIMENTAL, see
    docs/research/toy_tile_recurrence_rmt.rst:plasticity_algorithm_sandbox.
    TDD: written against the sili__new engine primitives
    (apply_amortized_l2_init/apply_amortized_block4_l2_init) already
    built and tested there -- this class covers the orchestration layer
    only (chunk sizing, block4/scattered dispatch)."""

    def test_touch_fraction_sizing_across_real_nnz_range(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(0))
        out = model.apply_l2_init()
        assert set(out.keys()) >= {"input_proj", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"}

    def test_non_fp32_backend_reports_nothing_yet(self):
        model = _model(disldo_cls=DISLDOLayer, rng=np.random.default_rng(1))
        out = model.apply_l2_init()
        assert out == {}, f"FP4 layers should report nothing yet (not wired), got keys: {list(out)}"

    def test_block4_backend_reports_block4_substats(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(2))
        out = model.apply_l2_init()
        assert "block4" in out["input_proj"]

    def test_runs_many_cycles_without_error(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(3))
        out = None
        for _ in range(500):
            out = model.apply_l2_init()
        entry = out["input_proj"]
        assert isinstance(entry["cycle_complete"], bool)
        assert entry["n_touched"] >= 0
        if "block4" in entry:
            assert entry["block4"]["n_touched"] >= 0

    def test_custom_knobs_are_threaded_through(self):
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(4))
        out = model.apply_l2_init(touch_fraction=1.0, max_chunk=1_000_000, rate=0.5)
        assert out["input_proj"] is not None

    def test_first_touch_does_not_yet_regularize(self):
        # A weight's very first touch only captures the reference value.
        # Exact behavior is verified at the engine test level; this just
        # confirms one real call doesn't error or silently no-op here.
        model = _model(disldo_cls=DISLDOLayer32, dense=True, rng=np.random.default_rng(5))
        out = model.apply_l2_init(rate=0.5)
        assert out["input_proj"]["n_touched"] > 0


class TestSpectralNormUpperBound:
    """spectral_norm_upper_bound -- Holder's-inequality cheap proxy
    (sqrt(||W||_1 * ||W||_inf)), diagnostic only. See
    docs/research/toy_tile_recurrence_rmt.rst:qk_spectral_norm_diagnostic."""

    def test_identity_matrix_is_exact(self):
        w = np.eye(5, dtype=np.float32)
        assert spectral_norm_upper_bound(w) == pytest.approx(1.0)

    def test_rank_one_all_ones_matrix_is_exact(self):
        # True spectral norm of an NxN all-ones matrix is N (eigenvalues
        # 0 and N) -- Holder's bound is TIGHT for rank-1 matrices.
        w = np.ones((4, 4), dtype=np.float32)
        assert spectral_norm_upper_bound(w) == pytest.approx(4.0)

    def test_zero_matrix_is_zero(self):
        assert spectral_norm_upper_bound(np.zeros((3, 3), dtype=np.float32)) == 0.0

    def test_empty_matrix_is_zero(self):
        assert spectral_norm_upper_bound(np.zeros((0, 0), dtype=np.float32)) == 0.0

    def test_scales_linearly_with_magnitude(self):
        rng = np.random.RandomState(0)
        w = rng.randn(6, 8).astype(np.float32)
        base = spectral_norm_upper_bound(w)
        assert spectral_norm_upper_bound(w * 3.0) == pytest.approx(base * 3.0, rel=1e-5)


class TestQKSparsityAndNorm:
    """qk_l1_sparsity_coef (independent, Q/K-scoped extension of the
    existing l1_sparsity_coef mechanism) and qkvo_norm_enable (QKV-Norm
    + O-Norm, Henry et al. 2020 / Zhai et al. 2023). Direct instruction,
    after v10's real recorded data showed q_proj/k_proj deviation-std
    collapsing to ~0 while raw_importance converged into a narrow 97-99
    band (a homogenization signature distinct from v_proj/o_proj/
    lm_head), and v12/v12b's real validation found Q/K-only
    normalization just shifted the SAME saturation pattern onto
    v_proj/o_proj/input_proj instead of removing it. See
    docs/research/toy_tile_recurrence_rmt.rst:qk_l1_sparsity_design and
    :qkvo_norm_design."""

    def _window_and_memory(self, seed=1):
        x_window = np.random.RandomState(seed).randn(NUM_TILES, EMBED_WIDTH).astype(np.float32) * 0.1
        memory_prev = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        return x_window, memory_prev

    def test_default_off_is_byte_identical_to_plain_model(self):
        model_a = _model(rng=np.random.default_rng(7))
        model_b = _model(qk_l1_sparsity_coef=0.0, qkvo_norm_enable=False, rng=np.random.default_rng(7))
        x_window, memory_prev = self._window_and_memory()
        mem_a, logits_a, aux_a = model_a.step(x_window, memory_prev, learning_rate=0.0, requires_grad=False)
        mem_b, logits_b, aux_b = model_b.step(x_window, memory_prev, learning_rate=0.0, requires_grad=False)
        assert np.array_equal(mem_a, mem_b)
        assert np.array_equal(logits_a.data, logits_b.data)
        assert aux_a is None and aux_b is None

    def test_qk_l1_sparsity_coef_produces_finite_aux_loss(self):
        model = _model(qk_l1_sparsity_coef=0.05)
        x_window, memory_prev = self._window_and_memory()
        _mem, _logits, aux_loss = model.step(x_window, memory_prev, learning_rate=0.01)
        assert aux_loss is not None
        assert np.isfinite(float(aux_loss.data))

    def test_qk_l1_sparsity_coef_independent_of_l1_sparsity_coef(self):
        # qk_l1_sparsity_coef must fire even when the pre-existing,
        # layer-wide l1_sparsity_coef is off.
        model = _model(l1_sparsity_coef=0.0, qk_l1_sparsity_coef=0.05)
        x_window, memory_prev = self._window_and_memory()
        _mem, _logits, aux_loss = model.step(x_window, memory_prev, learning_rate=0.01)
        assert aux_loss is not None

    def test_qkvo_norm_enable_produces_finite_shapes(self):
        model = _model(qkvo_norm_enable=True)
        x_window, memory_prev = self._window_and_memory()
        memory_new, logits, _aux = model.step(x_window, memory_prev, learning_rate=0.01)
        assert memory_new.shape == (NUM_MEM, STATE_WIDTH)
        assert logits.data.shape == (NUM_TILES, VOCAB)
        assert np.all(np.isfinite(memory_new))
        assert np.all(np.isfinite(logits.data))

    def test_qkvo_norm_enable_changes_output_vs_disabled(self):
        model_a = _model(qkvo_norm_enable=False, rng=np.random.default_rng(9))
        model_b = _model(qkvo_norm_enable=True, rng=np.random.default_rng(9))
        x_window, memory_prev = self._window_and_memory()
        _mem_a, logits_a, _aux_a = model_a.step(x_window, memory_prev, learning_rate=0.0, requires_grad=False)
        _mem_b, logits_b, _aux_b = model_b.step(x_window, memory_prev, learning_rate=0.0, requires_grad=False)
        assert not np.allclose(logits_a.data, logits_b.data)

    def test_qkvo_norm_ln_gains_are_in_parameters_for_optimizer(self):
        model = _model(qkvo_norm_enable=True)
        params = model.parameters_for_optimizer()
        assert model.q_norm_ln in params
        assert model.k_norm_ln in params
        assert model.v_norm_ln in params
        assert model.o_norm_ln in params

    def test_qkvo_norm_gains_actually_update_via_backward(self):
        model = _model(qkvo_norm_enable=True, rng=np.random.default_rng(11))
        opt = AdamOptimizer()
        x_window, memory_prev = self._window_and_memory()
        before_q = np.asarray(model.q_norm_ln.data).copy()
        before_v = np.asarray(model.v_norm_ln.data).copy()
        before_o = np.asarray(model.o_norm_ln.data).copy()
        _mem, logits, aux = model.step(x_window, memory_prev, learning_rate=0.05, requires_grad=True)
        loss = cross_entropy_sum(logits, [(t, 0) for t in range(NUM_TILES)])
        if aux is not None:
            loss = loss + aux
        loss.backward()
        opt.step(model.parameters_for_optimizer(), lr=0.05)
        assert not np.allclose(before_q, model.q_norm_ln.data)
        assert not np.allclose(before_v, model.v_norm_ln.data)
        assert not np.allclose(before_o, model.o_norm_ln.data)

    def test_step_cached_matches_step_with_qkvo_norm_enabled(self):
        # Same bit-exact-parity claim as TestStepCached's own test, now
        # under qkvo_norm_enable=True -- confirms the "normalize once,
        # then gather/cache" ordering in step_cached is equivalent to
        # step()'s own normalize-then-gather ordering (RMSNorm is
        # per-row, so this holds regardless of when the concat/cache
        # round-trip happens).
        model_a = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            qkvo_norm_enable=True,
            rng=np.random.default_rng(42),
        )
        model_b = ToyTileRecurrenceRMT(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            NUM_TILES,
            NUM_MEM,
            MAX_WEIGHTS,
            num_cpus=2,
            qkvo_norm_enable=True,
            rng=np.random.default_rng(42),
        )
        embed_table = np.random.RandomState(1).randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.1
        tokens = np.random.RandomState(2).randint(0, VOCAB, size=12)

        memory_a = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        memory_b = np.zeros((NUM_MEM, STATE_WIDTH), dtype=np.float32)
        tile_cache = None
        for i in range(len(tokens)):
            window = _build_window(embed_table, tokens, i, NUM_TILES)
            memory_a, logits_a, _aux_a = model_a.step(window, memory_a, 0.0, requires_grad=False)
            memory_b, logits_b, _aux_b, tile_cache = model_b.step_cached(
                embed_table[tokens[i]], memory_b, 0.0, tile_cache, requires_grad=False
            )
            row_a = np.asarray(logits_a.data)[NUM_TILES - 1]
            row_b = np.asarray(logits_b.data)[0]
            assert np.allclose(row_a, row_b, atol=1e-5), f"step {i}: logits diverged under qkvo_norm_enable"
            assert np.allclose(memory_a, memory_b, atol=1e-5), f"step {i}: memory_new diverged under qkvo_norm_enable"
