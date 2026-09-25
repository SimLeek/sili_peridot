"""
scripts/train_mqar_curriculum.py
──────────────────────────────────
Adaptive MQAR curriculum with student-paced difficulty (task #264).
See docs/research/train_mqar_curriculum.rst:train_curriculum.student_paced_curriculum_design.

Run: python3 scripts/train_mqar_curriculum.py <precision> [max_steps] [seed] [peak_lr] [num_tiles] [k_max]
  [additive_rank] [dynamic_rank_control] [rank_grace_period_steps] [use_critic]
  [recurrent_only_output] [embed_width] [input_sparsity_p] [wide_max_weights] [dy_sparsity_p]
  [use_tile_cache] [output_dy_sparsity_p] [wrong_streak_threshold] [dy_r_target] [dy_k_min]
  [dy_k_max] [target_steps_per_sec] [dy_surprise_alpha] [x_r_target] [x_k_min] [trajectory_log_every]
  [r_target_min]
  precision: fp4 | fp8 | fp32
See docs/research/train_mqar_curriculum.rst:train_curriculum.cli_gradient_sparsity_args for
dy_r_target/dy_k_min/dy_k_max/target_steps_per_sec/dy_surprise_alpha/x_r_target/x_k_min/
trajectory_log_every semantics. See each CLI arg's own comment in main() below for the rest.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from sili import _cpu
from sili.sparse_rnn import DIDLDOLayer32, DISLDOLayer, DISLDOLayer8, DISLDOLayer32
from sili.tensor import combine_losses

from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum, predicted_token
from model.toy_recall_task import generate_mqar_sequence
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT, spectral_norm_upper_bound
from scripts.train_mqar_rmt_reference import (
    CLIP_RANGE,
    COLUMN_NEURONS,
    EMBED_WIDTH,
    L1_SPARSITY_COEF,
    MAX_GRAD_NORM,
    MAX_WEIGHTS_PER_LAYER,
    NUM_CPUS,
    NUM_MEMORY_SLOTS,
    VOCAB,
    WARMUP_STEPS,
    _build_targets,
    seq_len_for_k,
)
from scripts.train_tile_curriculum import _build_tile_window

PRECISION_CLS = {
    "fp4": DISLDOLayer,
    "fp8": DISLDOLayer8,
    "fp32": DISLDOLayer32,
    # Group B (DIDLDO/SIDLDO, dense weight storage) -- the real dense
    # control arm for the dense-vs-sparse MQAR comparison, not
    # DISLDOLayer32(dense=True) (still Group A/sparse-capable storage,
    # just densely initialized). See sparse_rnn.rst:
    # sparse_rnn.engine_select_two_groups for the group distinction.
    "fp32_dense": DIDLDOLayer32,
}


def _default_graded_dy_schedule(num_tiles: int, floor: float = 0.02) -> list:
    """Linear decay from 1.0 (newest content position) down to `floor`
    (oldest) -- length num_tiles, matching content_dy_sparsity_schedule's
    own oldest-first convention. A real, tunable hyperparameter, not a
    derived constant -- `floor=0.02` is a starting guess only (see
    project_dy_sparsity_p_validated_speedup.md's correction: the earlier
    "~0.02 sweet spot" number came from dense_to_top_k_csr's surprising
    GLOBAL-not-per-row top-k semantics, so it doesn't directly transfer
    to this genuinely-per-row schedule -- needs its own real validation,
    not reused blind)."""
    if num_tiles <= 1:
        return [1.0] * num_tiles
    return [floor + (1.0 - floor) * (i / (num_tiles - 1)) for i in range(num_tiles)]


NOCAPS_KWARGS = {"max_abs_delta": 1e30, "max_ci": 1e30}
# See docs/research/train_mqar_curriculum.rst:train_curriculum.fp8_max_abs_delta_scale_space_bug.
NOCAPS_KWARGS_FP8 = {"max_abs_delta": 2.0, "max_ci": 1e30}
# See docs/research/train_mqar_curriculum.rst:train_curriculum.fp32_unbounded_weight_blowup.
# max_abs_grad=8.0: data-derived from v13's real recorded gradient-scale
# distribution (sqrt(raw_importance) across all 6 pools, p99.9~=6.34,
# p99.99~=9.85) -- see
# docs/research/toy_tile_recurrence_rmt.rst:col_importance_is_rmsprop_v_t
# and sili__new's docs/research/delta_csr_types.rst:synapse_policy.max_abs_grad_clip.
NOCAPS_KWARGS_FP32 = {"max_abs_delta": 2.0, "max_ci": 100.0, "max_abs_grad": 8.0}
PRECISION_SYNAPSE_KWARGS = {
    "fp4": NOCAPS_KWARGS,
    "fp8": NOCAPS_KWARGS_FP8,
    "fp32": NOCAPS_KWARGS_FP32,
    "fp32_dense": NOCAPS_KWARGS_FP32,
}

# See docs/research/train_mqar_curriculum.rst:train_curriculum.width_scaling_lr_fanin_hypothesis
# -- UNCONFIRMED: tuned near state_width=128, wider dense state_width may need this lowered.
DEFAULT_PEAK_LR = 0.015
DEFAULT_NUM_TILES = 16  # fixed local-attention window (model param, not a task param)
LEVEL_UP_TOKEN = VOCAB - 2  # 126 -- reserved, never chosen as an MQAR key/value
LEVEL_DOWN_TOKEN = VOCAB - 1  # 127 -- reserved, never chosen as an MQAR key/value
TASK_VOCAB_MAX = VOCAB - 2  # curriculum vocab_size grows up to (not including) this,
# so [0, TASK_VOCAB_MAX) never collides with the level tokens
VOCAB_START = 8  # min viable: must exceed seq_len_for_k(1)=4
VOCAB_GROWTH_FACTOR = 2.0  # doubles each promotion: 8->16->32->64->126(clamped)
K_START = 1
DEFAULT_K_MAX = 10
STREAK_THRESHOLD = 10  # consecutive correct queries to advance a stage
WRONG_STREAK_THRESHOLD = 5  # consecutive wrong queries to regress a stage
# See docs/research/train_mqar_curriculum.rst:train_curriculum.min_queries_before_regress_thrashing.
MIN_QUERIES_BEFORE_REGRESS = 30
MIN_LR_FRAC = 0.05
LOSS_EMA_DECAY = 0.98  # kept for logging only; LR itself is accuracy-driven
ACC_EMA_DECAY = 0.98

# See docs/research/train_mqar_curriculum.rst:train_curriculum.advantage_actor_critic_design.
ADVANTAGE_CLIP = 5.0

# Reward/punish asymmetry -- REMOVED.
# See docs/research/train_mqar_curriculum.rst:train_curriculum.reward_punish_asymmetry_removed.

# See docs/research/train_mqar_curriculum.rst:train_curriculum.nan_bisection_diagnostics.
DEBUG_FINITE_CHECK = False


def _layer_health(model) -> dict:
    """{layer_name: {array_name: (n_nonfinite, n_total)}} for every real
    disldo_cls weight layer with a C++ backend exposing the raw-vector
    accessors (fp32's DISLDOLayerV has no scale concept -- skipped, same
    guard convention as report_ranks/apply_scale_overflow_guard)."""
    health = {}
    for name, layer in model._named_real_layers():
        c = getattr(layer, "_c", None)
        if c is None or not hasattr(c, "get_value_scale_raw_vector"):
            continue
        arrays = {
            "value_scale": np.asarray(c.get_value_scale_raw_vector(), dtype=np.float32),
            "output_scale": np.asarray(c.get_output_scale_raw_vector(), dtype=np.float32),
            "additive_u": np.asarray(c.get_additive_u_raw_vector(), dtype=np.float32),
            "additive_v": np.asarray(c.get_additive_v_raw_vector(), dtype=np.float32),
        }
        health[name] = {k: (int((~np.isfinite(v)).sum()), int(v.size)) for k, v in arrays.items()}
    return health


def _describe(name: str, arr: np.ndarray) -> str:
    finite = arr[np.isfinite(arr)]
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    rng = f"[{finite.min():.4g}, {finite.max():.4g}]" if finite.size else "n/a"
    return f"{name}: size={arr.size} nan={n_nan} inf={n_inf} finite_range={rng}"


# See docs/research/train_mqar_curriculum.rst:train_curriculum.nan_bisection_diagnostics.
_MAGNITUDE_TRACE_MAXLEN = 40
_magnitude_trace = []


def _record_magnitude_trace(model, step: int, i: int, loss_ema) -> None:
    entry = {"step": step, "i": i, "loss_ema": loss_ema}
    for name, arr in getattr(model, "last_debug", {}).items():
        finite = arr[np.isfinite(arr)]
        entry[name] = float(np.abs(finite).max()) if finite.size else float("nan")
        # sigmas min too, see train_curriculum.nan_bisection_diagnostics in the RST.
        if name == "sigmas":
            entry["sigmas_min"] = float(finite.min()) if finite.size else float("nan")
    _magnitude_trace.append(entry)
    if len(_magnitude_trace) > _MAGNITUDE_TRACE_MAXLEN:
        _magnitude_trace.pop(0)


def _check_finite_or_raise(model, logits, step: int, i: int, loss_ema) -> None:
    _record_magnitude_trace(model, step, i, loss_ema)
    bad = []
    if not np.isfinite(logits.data).all():
        bad.append(_describe("logits.data", logits.data))
    cp = model.last_critic_pred
    if cp is not None and not np.isfinite(cp.data).all():
        bad.append(_describe("critic_pred.data", cp.data))
    # Bisects the forward chain; see train_curriculum.nan_bisection_diagnostics in the RST.
    for name, arr in getattr(model, "last_debug", {}).items():
        if not np.isfinite(arr).all():
            bad.append(_describe(f"model.last_debug[{name!r}]", arr))
    if not bad:
        return
    health = _layer_health(model)
    health_lines = []
    for lname, arrs in health.items():
        bad_arrs = {k: v for k, v in arrs.items() if v[0] > 0}
        if bad_arrs:
            health_lines.append(f"  {lname}: {bad_arrs}")
    trace_lines = [
        f"  step={e['step']} i={e['i']} loss_ema={e['loss_ema']} "
        + " ".join(f"{k}={v:.4g}" for k, v in e.items() if k not in ("step", "i", "loss_ema"))
        for e in _magnitude_trace
    ]
    report = (
        f"NON-FINITE at step={step} i={i}\n  "
        + "\n  ".join(bad)
        + "\nLayer weight health (nonfinite_count, total) for layers with issues:\n"
        + (
            "\n".join(health_lines)
            if health_lines
            else "  (none -- corruption is in activations only, not stored weights)"
        )
        + f"\nranks={model.report_ranks()}"
        + f"\nMagnitude trace, last {len(_magnitude_trace)} positions (max|finite value| per stage):\n"
        + "\n".join(trace_lines)
    )
    raise RuntimeError(report)


def k_indicator_token(k: int) -> int:
    """See docs/research/train_mqar_curriculum.rst:train_curriculum.level_prefix_persistent_indicator."""
    return k


def next_vocab(vocab_size: int, vocab_step: int | None = None) -> int:
    if vocab_step is not None:
        return min(TASK_VOCAB_MAX, vocab_size + vocab_step)
    return min(TASK_VOCAB_MAX, round(vocab_size * VOCAB_GROWTH_FACTOR))


def prev_vocab(vocab_size: int, vocab_step: int | None = None) -> int:
    if vocab_step is not None:
        return max(VOCAB_START, vocab_size - vocab_step)
    return max(VOCAB_START, round(vocab_size / VOCAB_GROWTH_FACTOR))


def _backward_with_critic(model, logits, target_token: int, row: int, aux) -> None:
    """See docs/research/train_mqar_curriculum.rst:train_curriculum.advantage_actor_critic_design."""
    vocab_size = logits.data.shape[-1]
    row_logits = logits.data[row]
    shifted = row_logits - row_logits.max()
    exp_l = np.exp(shifted)
    probs = exp_l / exp_l.sum()
    onehot = np.zeros(vocab_size, dtype=np.float32)
    onehot[target_token] = 1.0
    true_loss_vec = (probs - onehot) ** 2

    critic_pred = model.last_critic_pred
    # nan_to_num FIRST -- see train_curriculum.advantage_actor_critic_design in the RST.
    pred_row = np.nan_to_num(
        np.asarray(critic_pred.data[row], dtype=np.float32), nan=0.0, posinf=ADVANTAGE_CLIP, neginf=-ADVANTAGE_CLIP
    )
    advantage = np.clip(true_loss_vec - pred_row, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

    g_logits = np.zeros_like(logits.data)
    g_logits[row] = (1.0 + advantage) * (probs - onehot)

    g_critic = np.zeros_like(critic_pred.data)
    g_critic[row] = 2.0 * (pred_row - true_loss_vec)

    terms = [(logits, g_logits), (critic_pred, g_critic)]
    if aux is not None:
        terms.append(aux)
    combine_losses(*terms).backward()


def _stage_key(stage: dict) -> tuple:
    # See docs/research/train_mqar_curriculum.rst:train_curriculum.stage_key_kcycle_lexicographic.
    if stage["phase"] == "kcycle":
        return (2, stage["vocab"], stage["k"])
    return (1, stage["k"]) if stage["phase"] == "k" else (0, stage["vocab"])


def _ema_grad_scale(params, ratio_threshold: float, decay: float) -> None:
    """See docs/research/train_mqar_curriculum.rst:train_curriculum.ema_grad_scale_per_tensor."""
    for p in params:
        if p.grad is None:
            continue
        g = np.asarray(p.grad, dtype=np.float64)
        g_norm = float(np.linalg.norm(g))
        ema = getattr(p, "_grad_norm_ema", None)
        if ema is not None and ema > 0.0 and g_norm > ratio_threshold * ema:
            cap = ratio_threshold * ema
            p.grad = (g * (cap / g_norm)).astype(np.float32)
            g_norm = cap
        p._grad_norm_ema = g_norm if ema is None else (decay * ema + (1.0 - decay) * g_norm)


def train_curriculum(
    precision: str,
    max_steps: int,
    seed: int,
    peak_lr: float,
    num_tiles: int,
    k_max: int,
    log_every: int = 200,
    log_fn=None,
    additive_rank: int = 1,
    dynamic_rank_control: bool = True,
    rank_grace_period_steps: int = 50,
    rank_additive_grace_period_steps: int = 5000,
    use_critic: bool = False,
    magnitude_clip_penalty_coef: float = 0.0,
    recurrent_only_output: bool = False,
    embed_width: int = EMBED_WIDTH,
    input_sparsity_p: float | None = None,
    wide_max_weights: int | None = None,
    # See docs/research/train_mqar_curriculum.rst:train_curriculum.dense_was_hardcoded_true
    # -- was unconditionally True (not a real param) until 2026-09-18. Leave True at this
    # model scale: dense=False + a pre-chosen wide_max_weights has repeatedly degenerated
    # (project_sili_wide_model_mqar_baseline, project_sili_synaptogenesis_pruning_testing
    # memories) -- not validated as safe until state_width is large enough for a real
    # fan-in budget (biological target 1000-10000/neuron) to fit without forcing near-full
    # density anyway.
    dense: bool = True,
    dy_sparsity_p: float | None = None,
    use_tile_cache: bool = False,
    output_dy_sparsity_p: float | None = None,
    wrong_streak_threshold: int = WRONG_STREAK_THRESHOLD,
    streak_threshold: int = STREAK_THRESHOLD,
    vocab_step: int | None = None,
    require_new_vocab_before_levelup: bool = False,
    query_debug_fn=None,
    sigma_grad_debug_fn=None,
    clip_range: float = CLIP_RANGE,
    grad_ema_ratio_threshold: float | None = None,
    grad_ema_decay: float = 0.9,
    embed_table_builder=None,
    embed_learning_rate: float | None = None,
    k_first_target: int | None = None,
    k_first_vocab: int | None = None,
    l2_decay_chunk_size: int | None = None,
    l2_decay_adaptation_rate: float = 0.3,
    dy_r_target: float | None = None,
    dy_k_min: int = 0,
    dy_k_max: int | None = None,
    dy_surprise_alpha: float | None = None,
    dy_surprise_beta: float = 0.99,
    x_r_target: float | None = None,
    x_k_min: int = 0,
    x_k_max: int | None = None,
    x_balance_bias_step: float | None = None,
    x_balance_loss_coef: float = 0.0,
    x_r_target_auto: bool = False,
    x_r_target_auto_beta: float = 0.98,
    x_r_target_auto_margin: float = 0.0,
    dy_time_gate_cutoff: float | None = None,
    dy_time_gate_phase_step: float = 0.02,
    dy_time_gate_period: float = 125.0,
    dy_time_gate_seed: int | None = None,
    write_time_aux_targets: bool = False,
    max_grad_norm: float | None = None,
    target_steps_per_sec: float | None = None,
    trajectory_log_every: int | None = None,
    trajectory_log_steps: tuple[int, int] | None = None,
    trajectory_log_fn=None,
    r_target_min: float = 0.05,
    use_energy: bool = False,
    energy_kwargs: dict | None = None,
    # See docs/research/train_mqar_curriculum.rst:train_curriculum.lr_override_fn_range_test
    # -- bypasses the warmup/accuracy-decay schedule below entirely when set: lr = lr_override_fn(step).
    lr_override_fn=None,
    # See docs/research/train_mqar_curriculum.rst:polyak_lr_f_star_assumption -- per-layer
    # Stochastic-Polyak-Step-size, replacing lr for the 5 wide layers only (lm_head/embed
    # unaffected). False (default): byte-identical to today's exact behavior.
    polyak_lr: bool = False,
    polyak_f_star: float = 0.0,
    # c=0.5 (a typical literature SPS damping factor) was wildly miscalibrated
    # for this setup's actual per-layer Lbar scale -- see apply_polyak_lr's
    # own docstring for the recalibration story (first validation run failed
    # to learn at all under the old c=0.5/lr_max=0.1 defaults).
    polyak_c: float = 0.0005,
    polyak_lr_max: float = 0.05,
    # See docs/research/toy_tile_recurrence_rmt.rst:loss_adjusted_decay_design --
    # EXPERIMENTAL critical-learning-periods/loss-of-plasticity forgetting.
    # False (default): byte-identical to today's exact behavior, matching
    # l2_decay_chunk_size's own off-by-default convention.
    loss_adjusted_decay_enable: bool = False,
    # touch_fraction, not a raw chunk_size -- real bug caught before ever
    # launching a real run: see apply_loss_adjusted_decay's own
    # touch_fraction docstring section (a single shared absolute
    # chunk_size either barely touches big layers or, for small ones,
    # over-touches past their own nnz -- compounding toward an
    # accidental full-layer wipe rather than gradual, testable decay).
    loss_adjusted_decay_touch_fraction: float = 0.01,
    loss_adjusted_decay_min_chunk: int = 4,
    loss_adjusted_decay_max_chunk: int = 2048,
    # Half-life in TOUCHES (~1 per amortized cycle), not an abstract
    # severity knob -- real miscalibration caught before ever launching a
    # real run: see apply_loss_adjusted_decay's own SEVERITY docstring
    # section (an earlier strength/floor design crushed a real layer's
    # weights to float-zero within ~1400 calls under full stall, a small
    # fraction of one historical stall's real 74k-99k-step length).
    # 500 touches (~50,000 calls at the default touch_fraction, the same
    # order of magnitude as those historical stalls) to halve at worst.
    loss_adjusted_decay_importance_half_life_touches: float | None = 500.0,
    # weight arm default OFF -- see apply_loss_adjusted_decay's own
    # docstring: it shares l2_decay_chunk_size's C++ weight-decay cursor,
    # do not set this nonzero in the same run that also sets
    # l2_decay_chunk_size.
    loss_adjusted_decay_weight_half_life_touches: float | None = None,
    # Real baseline/grace period -- see apply_loss_adjusted_decay's own
    # min_stall_steps docstring section (direct question: "I feel like it
    # would need to establish a baseline and shouldn't necessarily decay
    # all the time"). Decay is an EXACT no-op for this many consecutive
    # non-improving calls before it starts ramping in at all.
    loss_adjusted_decay_min_stall_steps: int = 200,
    loss_adjusted_decay_ramp_steps: int = 200,
    loss_adjusted_decay_beta_fast: float = 0.9,
    loss_adjusted_decay_beta_floor: float = 0.999,
    loss_adjusted_decay_improve_tol: float = 1e-3,
    # See docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design --
    # EXPERIMENTAL per-neuron utility-based plasticity reset
    # (Continual-Backprop-inspired), SUPERSEDES loss_adjusted_decay above
    # as the primary mechanism under test. False (default): byte-identical
    # to today's exact behavior. NO loss argument needed -- purely local
    # per-column signals, unlike loss_adjusted_decay.
    plasticity_reset_enable: bool = False,
    plasticity_reset_touch_fraction: float = 0.01,
    plasticity_reset_min_chunk: int = 4,
    plasticity_reset_max_chunk: int = 2048,
    plasticity_reset_eta: float = 0.99,
    plasticity_reset_eta_slow: float = 0.99,
    plasticity_reset_eta_slow_catchup: float = 0.95,
    plasticity_reset_eta_fast: float = 0.5,
    plasticity_reset_blend: float = 0.10,
    plasticity_reset_reset_fraction: float = 0.01,
    plasticity_reset_k: float = 1.0,
    plasticity_reset_eta_var: float = 0.9,
    # L2-saturation-gated decay on col_importance (direct instruction,
    # after offline replay of a real 100k-step run's column logs showed
    # importance saturating at the ci accumulator's max_ci clamp for
    # most of a pool's population by late training). 0.0 (default):
    # byte-identical no-op. See
    # docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design.l2_saturation_decay.
    plasticity_reset_l2_decay_lambda: float = 0.0,
    plasticity_reset_l2_decay_threshold: float = 0.9,
    plasticity_reset_l2_decay_temperature: float = 0.05,
    plasticity_reset_max_ci: float = 100.0,
    # False (default): byte-identical top-K-by-importance selection.
    # True: rank candidates by deviation (growth RATE) instead of
    # col_importance (absolute LEVEL) -- direct instruction, after
    # comparing a graduated real run against a stuck real run's
    # collected data. See
    # docs/research/toy_tile_recurrence_rmt.rst:select_by_deviation_early_detection.
    plasticity_reset_select_by_deviation: bool = False,
    # Offline data collection toward fitting a reset-selection equation
    # from real training data (direct instruction: per-column, not
    # per-synapse -- the mechanism only ever selects at column
    # granularity). None (default): no extra work, byte-identical to
    # today. When set, one small .npz per completed cycle per
    # layer/pool is written under this directory -- see
    # docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design
    # plasticity_column_data_collection section.
    plasticity_column_log_dir: str | None = None,
    # Diagnostic (direct instruction, after finding col_importance is a
    # SECOND EMA on top of the real per-synapse ci accumulator, making it
    # impossible to see the real accumulator's own recovery dynamics
    # between plasticity touches): when True, each plasticity_column_log_dir
    # snapshot ALSO includes the raw, unsmoothed per-synapse importance
    # matrix (layer.importance reshaped to (in_features, out_features)),
    # not just the column aggregate. False (default): byte-identical,
    # no extra work. See
    # docs/research/toy_tile_recurrence_rmt.rst:raw_ci_landscape_capture.
    plasticity_raw_importance_log: bool = False,
    # Diagnostic: called once per REAL weight update (right after
    # opt.step()), as raw_ci_sample_fn(step, model) -- lets a caller
    # track individual synapse values at the TRUE update cadence (every
    # real step), unlike the once-per-cycle snapshots above. None
    # (default): no extra work.
    raw_ci_sample_fn=None,
    # Direct instruction ("let's ... get displayarray ... working and
    # display all of the networks side by side ... I'll just look at
    # all of the synapses for the next run"): when True, pushes a
    # live, tiled visualization (every pool's weight + importance
    # heatmaps plus the plasticity_reset algorithm's own live
    # selection/deviation state) to an on-screen window every time any
    # pool completes an amortized cycle -- see
    # scripts/live_synapse_display.py and
    # docs/research/toy_tile_recurrence_rmt.rst:live_synapse_display.
    # False (default): byte-identical, no extra work, no window opened.
    plasticity_live_display: bool = False,
    # Pixels per synapse cell in the live display. Default 1: literal
    # one pixel per synapse, no artificial zoom (direct correction --
    # upsampling by default was solving a problem that didn't exist).
    # Raise only if you deliberately want fewer, larger pools on
    # screen. Only used when plasticity_live_display.
    plasticity_live_display_cell_px: int = 1,
    # L2 Init (Kumar, Marklund & Van Roy, CoLLAs 2025, arXiv:2308.11958)
    # -- EXPERIMENTAL, see
    # docs/research/toy_tile_recurrence_rmt.rst:plasticity_algorithm_sandbox.
    # False (default): byte-identical, no extra work. Independent of
    # plasticity_reset_enable -- separate mechanism, separate cursor,
    # can run alongside it or alone.
    l2_init_enable: bool = False,
    l2_init_touch_fraction: float = 0.01,
    l2_init_min_chunk: int = 4,
    l2_init_max_chunk: int = 2048,
    l2_init_rate: float = 0.0001,
    # qk_l1_sparsity_coef: EXPERIMENTAL Q/K-specific candidate for the
    # attention-entropy-collapse pattern found in v10's real recorded
    # data (deviation std -> ~0 in q_proj/k_proj only). See
    # docs/research/toy_tile_recurrence_rmt.rst:qk_l1_sparsity_design.
    # Default off: byte-identical, no extra work.
    qk_l1_sparsity_coef: float = 0.0,
    # qkvo_norm_enable: supersedes the earlier qk_norm_enable (Q/K
    # only) -- v12/v12b's real validation found Q/K-only normalization
    # just shifted the same saturation pattern onto v_proj/o_proj/
    # input_proj instead of removing it. Now covers Q, K, V, and
    # o_proj's output. See
    # docs/research/toy_tile_recurrence_rmt.rst:qkvo_norm_design.
    # Default off: byte-identical, no extra work.
    qkvo_norm_enable: bool = False,
    # Diagnostic only (no training-time cost unless True): logs
    # spectral_norm_upper_bound(q_proj/k_proj weights) at the same
    # cadence as the main summary line, to directly verify whether
    # qk_l1_sparsity_coef/qkvo_norm_enable actually bound Q/K's
    # spectral norm growth, not just infer it from downstream MQAR
    # performance.
    # See docs/research/toy_tile_recurrence_rmt.rst:qk_spectral_norm_diagnostic.
    qk_spectral_norm_diag_log: bool = False,
    # AdaBelief-style row/column centering on the RMSprop-style ci
    # accumulator itself (sili__new's update_ci m parameter -- distinct
    # from every plasticity_reset_* mechanism above, which operates on
    # col_importance/weight, not on ci's own gradient-EMA formula).
    # Motivated by v14 (max_abs_grad clipping alone regressed curriculum
    # progress vs v13b): clipping defends a one-off spike, centering
    # defends a SUSTAINED large gradient that would otherwise keep ci
    # pinned near max_abs_grad^2 forever. False/False (default):
    # byte-identical, no extra work -- merged into synapse_kwargs (and
    # thus reaches every DISLDOLayer32 layer's forward() call via
    # self.synapse_kwargs) only when explicitly enabled. See sili__new's
    # docs/research/delta_csr_types.rst:synapse_policy.adabelief_centering.
    centering_row_enable: bool = False,
    centering_col_enable: bool = False,
    centering_beta1: float | None = None,
) -> dict:
    # query_debug_fn: see docs/research/train_mqar_curriculum.rst:
    # train_curriculum.query_debug_fn_explainable_ai_hook.
    # embed_width/input_sparsity_p/wide_max_weights/dy_sparsity_p: threaded
    # straight through to ToyTileRecurrenceRMT's identically-named args --
    # see its own constructor docstring. dy_sparsity_p profiling: see
    # JOURNAL.md's 2026-08-30 "backward's real cost profiled to per-synapse
    # update math, not snapshot/merge" entry.
    # use_tile_cache: see docs/research/train_mqar_curriculum.rst:
    # train_curriculum.use_tile_cache_query_step_fallback.
    # k_first_target/k_first_vocab: see docs/research/train_mqar_curriculum.rst:
    # train_curriculum.k_first_target_odometer_reordering.
    #
    # WARNING (2026-09-11): leaving k_first_target unset (the default) runs
    # the vocab-first curriculum -- k=1 for the ENTIRE vocab ramp, tens of
    # thousands of steps, before k ever grows. This risks training the model
    # to solve MQAR via a positional/relative shortcut (there's only ever
    # ONE query-key pair per sequence at k=1) rather than genuine key-value
    # binding, and per this file's own k_first_target_odometer_reordering
    # design note, entrenches synapse importance against the k>1 feature
    # before the model is ever asked to use it. Callers should set
    # k_first_target (the "kcycle" odometer -- v16k1,v16k2,v16k3,v18k1,...)
    # by default; only leave it unset when the test SPECIFICALLY calls for
    # isolating vocab growth from k growth (e.g. reproducing an older
    # vocab-first result, or a deliberate ablation of curriculum order
    # itself). Kept opt-in rather than flipping the default, per this
    # param's own backward-compatibility policy above -- existing callers
    # that already pass k_first_target explicitly are unaffected either way.
    if k_first_target is not None and k_first_vocab is None:
        k_first_vocab = seq_len_for_k(k_first_target) + 4

    # write_time_aux_targets: OFF by default, per direct instruction --
    # predict-next-token only makes sense as a training signal if the
    # next token is actually predictable, and in MQAR's random key/value
    # layout it structurally isn't (every write position's "next token"
    # is an unrelated random draw). _build_targets's own fallback
    # (targets.setdefault(i, tokens[i+1]) for every non-query write
    # position) mixes that unpredictable signal into every step
    # alongside the real associative-recall objective. The 2026-09-07
    # arm_nolevel_down investigation (JOURNAL.md) already found removing
    # it helped, not hurt (bounded, if anything lower loss) -- this was
    # previously only ever done via an ad-hoc module-level monkeypatch
    # (scripts/sandbox_arm_launchers/run_sandbox_dense_probe.py's `_q`),
    # never kept as a reusable, real parameter until now.
    build_targets_fn = _build_targets if write_time_aux_targets else (lambda _tok, pairs, _k: dict(pairs))

    # max_grad_norm: None -> MAX_GRAD_NORM (the module default, 1.0),
    # matching every existing caller's behavior unchanged. clip_grad_norm_
    # is a GLOBAL L2 norm across every parameter tensor combined, never
    # scaled by parameter count -- as width grows, q/k/v/o_proj's own
    # parameter count grows with state_width^2, so the natural (pre-clip)
    # norm grows roughly with sqrt(param count) even at unchanged
    # per-parameter gradient magnitudes. A fixed max_norm tuned at one
    # width clips increasingly aggressively at a wider one, silently
    # shrinking the effective step size below what was calibrated for the
    # narrower model. Exposed here (real parameter, not a monkeypatch --
    # see write_time_aux_targets's own history) to test that hypothesis
    # directly, per direct instruction.
    effective_max_grad_norm = MAX_GRAD_NORM if max_grad_norm is None else max_grad_norm

    disldo_cls = PRECISION_CLS[precision]
    # AQRS (additive_rank/dynamic_rank_control) exists to give LOW-BIT
    # storage (FP4/FP8) extra precision where it's sparse -- fp32 is
    # already full precision everywhere, so AQRS has nothing to correct
    # for. Force it off here rather than support it on DISLDOLayer32/
    # DIDLDOLayer32: the caller's own additive_rank=1/dynamic_rank_
    # control=True defaults would otherwise reach ToyTileRecurrenceRMT's
    # non-zero-value guard (rank_kwargs/additive_kwargs, model/
    # toy_tile_recurrence_rmt.py) and get forwarded into a constructor
    # that was never meant to have it.
    if precision in ("fp32", "fp32_dense"):
        additive_rank = 0
        dynamic_rank_control = False
    state_width = embed_width * COLUMN_NEURONS

    layer_synapse_kwargs = dict(PRECISION_SYNAPSE_KWARGS[precision])
    if centering_row_enable:
        layer_synapse_kwargs["centering_row_enable"] = True
    if centering_col_enable:
        layer_synapse_kwargs["centering_col_enable"] = True
    if centering_beta1 is not None:
        layer_synapse_kwargs["centering_beta1"] = centering_beta1

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    model_rng = np.random.default_rng(seed)

    # num_tiles fixed at construction; see train_curriculum.student_paced_curriculum_design.
    model = ToyTileRecurrenceRMT(
        VOCAB,
        embed_width,
        COLUMN_NEURONS,
        num_tiles,
        NUM_MEMORY_SLOTS,
        MAX_WEIGHTS_PER_LAYER,
        num_cpus=NUM_CPUS,
        disldo_cls=disldo_cls,
        dense=dense,
        clip_range=clip_range,
        l1_sparsity_coef=L1_SPARSITY_COEF,
        qk_l1_sparsity_coef=qk_l1_sparsity_coef,
        qkvo_norm_enable=qkvo_norm_enable,
        synapse_kwargs=layer_synapse_kwargs,
        scale_rank=1,
        additive_rank=additive_rank,
        dynamic_rank_control=dynamic_rank_control,
        use_critic=use_critic,
        magnitude_clip_penalty_coef=magnitude_clip_penalty_coef,
        recurrent_only_output=recurrent_only_output,
        input_sparsity_p=input_sparsity_p,
        wide_max_weights=wide_max_weights,
        dy_sparsity_p=dy_sparsity_p,
        output_dy_sparsity_p=output_dy_sparsity_p,
        dy_r_target=dy_r_target,
        dy_k_min=dy_k_min,
        dy_k_max=dy_k_max,
        dy_surprise_alpha=dy_surprise_alpha,
        dy_surprise_beta=dy_surprise_beta,
        x_r_target=x_r_target,
        x_k_min=x_k_min,
        x_k_max=x_k_max,
        x_balance_bias_step=x_balance_bias_step,
        x_balance_loss_coef=x_balance_loss_coef,
        x_r_target_auto=x_r_target_auto,
        x_r_target_auto_beta=x_r_target_auto_beta,
        x_r_target_auto_margin=x_r_target_auto_margin,
        dy_time_gate_cutoff=dy_time_gate_cutoff,
        dy_time_gate_phase_step=dy_time_gate_phase_step,
        dy_time_gate_period=dy_time_gate_period,
        dy_time_gate_seed=dy_time_gate_seed,
        r_target_min=r_target_min,
        use_energy=use_energy,
        energy_kwargs=energy_kwargs,
        rng=model_rng,
    )
    opt = AdamOptimizer()
    # See docs/research/train_mqar_curriculum.rst:train_curriculum.embed_table_builder_sdr_hook.
    active_mask = None
    if embed_table_builder is not None:
        embed_table, active_mask = embed_table_builder(rng, VOCAB, embed_width)
    else:
        embed_table = rng.randn(VOCAB, embed_width).astype(np.float32) * 0.3

    if k_first_target is not None:
        stage_stack = [{"vocab": k_first_vocab, "k": K_START, "phase": "kcycle"}]
    else:
        stage_stack = [{"vocab": VOCAB_START, "k": K_START, "phase": "vocab"}]
    streak = 0
    wrong_streak = 0
    # See docs/research/train_mqar_curriculum.rst:train_curriculum.new_vocab_forced_sampling_gate.
    new_key_ids: list = []
    streak_has_new_vocab = False
    max_streak_seen = 0  # best streak between log points; see wrong_streak_threshold_reward_hacking anchor
    # Per-layer/pool cumulative counts + latest deviation, so log_fn can
    # attribute a loss spike to plasticity_reset actually firing (vs some
    # other cause) instead of only seeing the downstream loss effect.
    plasticity_totals: dict = {}
    _live_display = None
    if plasticity_live_display:
        from scripts.live_synapse_display import LiveSynapseDisplay

        _live_display = LiveSynapseDisplay(
            pool_order=[f"{n}.{p}" for n, _ in model._named_real_layers() for p in ("scattered", "block4")],
            cell_px=plasticity_live_display_cell_px,
        )
    stage_step = 0
    queries_since_level_change = 0
    pending_level_token = None
    loss_ema = None
    acc_ema = None
    stage_history = []
    peak_key = _stage_key(stage_stack[0])
    peak_stage = dict(stage_stack[0])
    rank_mutation_count = 0
    rank_history = []

    def _current():
        s = stage_stack[-1]
        return s["vocab"], s["k"], s["phase"]

    def _log_ranks(step):
        # Per-layer rank snapshot at every log point + level transition (direct instruction).
        if not dynamic_rank_control:
            return None
        ranks = model.report_ranks()
        rank_history.append({"step": step, "ranks": ranks})
        return ranks

    def _advance_stage(step):
        nonlocal streak, wrong_streak, stage_step, queries_since_level_change
        nonlocal pending_level_token, peak_key, peak_stage
        nonlocal new_key_ids, streak_has_new_vocab
        cur = stage_stack[-1]
        if cur["phase"] == "kcycle":
            # Odometer; see docs/research/train_mqar_curriculum.rst:
            # train_curriculum.k_first_target_odometer_reordering.
            if cur["k"] < k_first_target or cur["vocab"] >= TASK_VOCAB_MAX:
                new_stage = {"vocab": cur["vocab"], "k": cur["k"] + 1, "phase": "kcycle"}
                new_key_ids = []
            else:
                nv = next_vocab(cur["vocab"], vocab_step)
                new_stage = {"vocab": nv, "k": K_START, "phase": "kcycle"}
                new_key_ids = list(range(cur["vocab"] // 2, nv // 2))
            streak_has_new_vocab = False
        elif cur["phase"] == "vocab":
            nv = next_vocab(cur["vocab"], vocab_step)
            new_phase = "k" if nv >= TASK_VOCAB_MAX else "vocab"
            new_stage = {"vocab": nv, "k": cur["k"], "phase": new_phase}
            new_key_ids = list(range(cur["vocab"] // 2, nv // 2))
            streak_has_new_vocab = False
        else:
            new_stage = {"vocab": cur["vocab"], "k": cur["k"] + 1, "phase": "k"}
        stage_history.append({"step": step, "event": "level_up", "from": cur, "to": new_stage})
        stage_stack.append(new_stage)
        if _stage_key(new_stage) > peak_key:
            peak_key = _stage_key(new_stage)
            peak_stage = dict(new_stage)
        streak = 0
        wrong_streak = 0
        stage_step = 0
        queries_since_level_change = 0
        pending_level_token = LEVEL_UP_TOKEN
        ranks = _log_ranks(step)
        if log_fn is not None:
            v, k, ph = _current()
            log_fn(step, v, k, ph, "LEVEL_UP", loss_ema, acc_ema, ranks=ranks)

    def _regress_stage(step):
        nonlocal streak, wrong_streak, stage_step, queries_since_level_change, pending_level_token
        nonlocal new_key_ids, streak_has_new_vocab
        # No forcing on regression; see new_vocab_forced_sampling_gate anchor.
        new_key_ids = []
        streak_has_new_vocab = False
        if len(stage_stack) <= 1:
            # already at the floor -- nothing to regress to, just reset
            streak = 0
            wrong_streak = 0
            stage_step = 0
            queries_since_level_change = 0
            return
        cur = stage_stack.pop()
        new_stage = stage_stack[-1]
        stage_history.append({"step": step, "event": "level_down", "from": cur, "to": new_stage})
        streak = 0
        wrong_streak = 0
        stage_step = 0
        queries_since_level_change = 0
        pending_level_token = LEVEL_DOWN_TOKEN
        ranks = _log_ranks(step)
        if log_fn is not None:
            v, k, ph = _current()
            log_fn(step, v, k, ph, "LEVEL_DOWN", loss_ema, acc_ema, ranks=ranks)

    t0 = time.time()
    window_t0 = t0
    model.reset_layer_timing()
    step = 0
    while step < max_steps:
        step += 1
        stage_step += 1
        if lr_override_fn is not None:
            lr = lr_override_fn(step)
        elif step <= WARMUP_STEPS:
            lr = peak_lr * step / WARMUP_STEPS
        elif acc_ema is None:
            lr = peak_lr
        else:
            frac = max(MIN_LR_FRAC, min(1.0, 1.0 - acc_ema))
            lr = peak_lr * frac

        if polyak_lr:
            # See docs/research/train_mqar_curriculum.rst:polyak_lr_f_star_assumption
            # -- lagged one step (uses last outer step's loss_ema + each wide layer's
            # E_t from its own last backward call), same granularity as lr itself.
            model.apply_polyak_lr(
                loss_ema if loss_ema is not None else 0.0,
                f_star=polyak_f_star,
                c=polyak_c,
                lr_max=polyak_lr_max,
                bootstrap_lr=lr,
            )

        vocab_size, k, phase = _current()
        seq_len = seq_len_for_k(k)
        # Gate, not tax; see docs/research/train_mqar_curriculum.rst:
        # train_curriculum.new_vocab_forced_sampling_gate.
        force_this_step = (
            require_new_vocab_before_levelup
            and phase == "vocab"
            and len(new_key_ids) > 0
            and streak == streak_threshold - 1
            and not streak_has_new_vocab
        )
        tokens, mqar_pairs = generate_mqar_sequence(
            rng, vocab_size, seq_len, k, forced_keys=(new_key_ids if force_this_step else None)
        )
        # Per-query-position, not a step-wide OR; see new_vocab_forced_sampling_gate anchor.
        new_vocab_query_positions = (
            {pos for pos, _ in mqar_pairs if int(tokens[pos]) in new_key_ids} if new_key_ids else set()
        )
        targets = build_targets_fn(tokens, mqar_pairs, k)
        query_positions = {pos for pos, _ in mqar_pairs}

        # [vocab indicator, k indicator]; see docs/research/train_mqar_curriculum.rst:
        # train_curriculum.level_prefix_persistent_indicator.
        level_prefix = [vocab_size - 1, k_indicator_token(k)]
        if pending_level_token is not None:
            level_prefix = [pending_level_token, *level_prefix]
            pending_level_token = None
        offset = len(level_prefix)
        combined_tokens = np.concatenate((level_prefix, tokens))
        targets = {pos + offset: tgt for pos, tgt in targets.items()}
        query_positions = {pos + offset for pos in query_positions}
        new_vocab_query_positions = {pos + offset for pos in new_vocab_query_positions}

        memory = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
        tile_cache = None  # reset every sequence, same as memory
        for i in range(seq_len + offset):
            # See docs/research/train_mqar_curriculum.rst:
            # train_curriculum.use_tile_cache_query_step_fallback.
            if use_tile_cache and (i in targets):
                window = _build_tile_window(embed_table, combined_tokens, i, num_tiles)
                memory, logits, aux = model.step(
                    window,
                    memory,
                    lr,
                    requires_grad=True,
                    content_dy_sparsity_schedule=_default_graded_dy_schedule(num_tiles),
                )
                logit_row = num_tiles - 1
                # Cache refresh; see use_tile_cache_query_step_fallback anchor.
                k_content = model.last_debug["k"][NUM_MEMORY_SLOTS:]
                v_content = model.last_debug["v"][NUM_MEMORY_SLOTS:]
                tile_cache = list(zip(k_content[1:].copy(), v_content[1:].copy(), strict=False))
            elif use_tile_cache:
                new_embed = embed_table[combined_tokens[i]]
                memory, logits, aux, tile_cache = model.step_cached(
                    new_embed, memory, lr, tile_cache, requires_grad=False
                )
                logit_row = 0  # step_cached returns only the newest position's row
            else:
                window = _build_tile_window(embed_table, combined_tokens, i, num_tiles)
                memory, logits, aux = model.step(window, memory, lr, requires_grad=(i in targets))
                logit_row = num_tiles - 1
            if DEBUG_FINITE_CHECK and use_critic:
                _check_finite_or_raise(model, logits, step, i, loss_ema)
            if i in targets:
                loss = cross_entropy_sum(logits, [(logit_row, targets[i])])
                loss_ema = (
                    float(loss.data)
                    if loss_ema is None
                    else (LOSS_EMA_DECAY * loss_ema + (1.0 - LOSS_EMA_DECAY) * float(loss.data))
                )
                if i in query_positions:
                    correct = predicted_token(logits, logit_row) == targets[i]
                    acc_ema = (
                        float(correct)
                        if acc_ema is None
                        else (ACC_EMA_DECAY * acc_ema + (1.0 - ACC_EMA_DECAY) * float(correct))
                    )
                    if query_debug_fn is not None:
                        # See train_curriculum.query_debug_fn_explainable_ai_hook.
                        query_debug_fn(step, correct, logit_row, model.last_debug, logits.data[logit_row], targets[i])
                    queries_since_level_change += 1
                    if i in new_vocab_query_positions:
                        streak_has_new_vocab = True
                    if correct:
                        streak += 1
                        wrong_streak = 0
                        max_streak_seen = max(max_streak_seen, streak)
                    else:
                        wrong_streak += 1
                        streak = 0
                        streak_has_new_vocab = False
                if use_critic:
                    _backward_with_critic(model, logits, targets[i], logit_row, aux)
                else:
                    if aux is not None:
                        loss = loss + aux
                    loss.backward()
                # See docs/research/train_mqar_curriculum.rst:
                # train_curriculum.embed_learning_rate_scatter_gradient.
                if embed_learning_rate is not None and not use_tile_cache:
                    x_grad = model.last_debug.get("x_window_t")
                    x_grad = x_grad.grad if x_grad is not None else None
                    if x_grad is not None:
                        for j in range(num_tiles):
                            src = i - (num_tiles - 1) + j
                            if src < 0:
                                continue
                            tok = combined_tokens[src]
                            g = x_grad[j]
                            if active_mask is not None:
                                g = g * active_mask[tok]
                            embed_table[tok] -= embed_learning_rate * g
                if grad_ema_ratio_threshold is not None:
                    _ema_grad_scale(model.parameters_for_optimizer(), grad_ema_ratio_threshold, grad_ema_decay)
                if sigma_grad_debug_fn is not None:
                    # Fired before clip_grad_norm_; see ema_grad_scale_per_tensor anchor.
                    sigma_grad_debug_fn(step, model.log_sigmas.grad, model.centers.grad)
                clip_grad_norm_(model.parameters_for_optimizer(), effective_max_grad_norm)
                opt.step(model.parameters_for_optimizer(), lr=lr)
                if raw_ci_sample_fn is not None:
                    raw_ci_sample_fn(step, model)
                # See docs/research/train_mqar_curriculum.rst:
                # train_curriculum.l2_decay_and_rank_control_ordering.
                if l2_decay_chunk_size is not None:
                    model.apply_amortized_l2_decay(l2_decay_chunk_size, l2_decay_adaptation_rate)
                if loss_adjusted_decay_enable:
                    # Uses loss_ema (not the raw per-token loss), same
                    # smoothing convention apply_polyak_lr already uses for
                    # its own residual -- see loss_adjusted_decay_design.
                    model.apply_loss_adjusted_decay(
                        loss_ema if loss_ema is not None else float(loss.data),
                        touch_fraction=loss_adjusted_decay_touch_fraction,
                        min_chunk=loss_adjusted_decay_min_chunk,
                        max_chunk=loss_adjusted_decay_max_chunk,
                        importance_half_life_touches=loss_adjusted_decay_importance_half_life_touches,
                        weight_half_life_touches=loss_adjusted_decay_weight_half_life_touches,
                        min_stall_steps=loss_adjusted_decay_min_stall_steps,
                        ramp_steps=loss_adjusted_decay_ramp_steps,
                        beta_fast=loss_adjusted_decay_beta_fast,
                        beta_floor=loss_adjusted_decay_beta_floor,
                        improve_tol=loss_adjusted_decay_improve_tol,
                    )
                if plasticity_reset_enable:
                    if plasticity_raw_importance_log or plasticity_live_display:
                        _layers_by_name = dict(model._named_real_layers())
                    # No loss argument -- purely local per-column signals.
                    _plasticity_stats = model.apply_plasticity_reset(
                        touch_fraction=plasticity_reset_touch_fraction,
                        min_chunk=plasticity_reset_min_chunk,
                        max_chunk=plasticity_reset_max_chunk,
                        eta=plasticity_reset_eta,
                        eta_slow=plasticity_reset_eta_slow,
                        eta_slow_catchup=plasticity_reset_eta_slow_catchup,
                        eta_fast=plasticity_reset_eta_fast,
                        blend=plasticity_reset_blend,
                        reset_fraction=plasticity_reset_reset_fraction,
                        k=plasticity_reset_k,
                        eta_var=plasticity_reset_eta_var,
                        l2_decay_lambda=plasticity_reset_l2_decay_lambda,
                        l2_decay_threshold=plasticity_reset_l2_decay_threshold,
                        l2_decay_temperature=plasticity_reset_l2_decay_temperature,
                        max_ci=plasticity_reset_max_ci,
                        select_by_deviation=plasticity_reset_select_by_deviation,
                        include_column_state=(plasticity_column_log_dir is not None or plasticity_live_display),
                    )
                    for _layer_name, _layer_stats in _plasticity_stats.items():
                        _scattered = _layer_stats.get("importance")
                        if _scattered is None:
                            continue
                        for _pool_name, _leaf in (
                            ("scattered", _scattered),
                            ("block4", _scattered.get("block4")),
                        ):
                            if _leaf is None or not _leaf.get("cycle_complete"):
                                continue
                            _key = f"{_layer_name}.{_pool_name}"
                            _tot = plasticity_totals.setdefault(
                                _key,
                                {
                                    "n_reset": 0,
                                    "last_deviation": 0.0,
                                    "last_min_deviation": 0.0,
                                    "last_max_deviation": 0.0,
                                    "last_importance": 0.0,
                                    "last_l2_sat_ratio": 0.0,
                                    "last_l2_decay_strength": 0.0,
                                },
                            )
                            _tot["n_reset"] += _leaf.get("n_reset_this_cycle", 0)
                            _tot["last_deviation"] = _leaf.get("mean_deviation", 0.0)
                            _tot["last_min_deviation"] = _leaf.get("min_deviation", 0.0)
                            _tot["last_max_deviation"] = _leaf.get("max_deviation", 0.0)
                            _tot["last_importance"] = _leaf.get("mean_col_importance", 0.0)
                            _tot["last_l2_sat_ratio"] = _leaf.get("l2_sat_ratio", 0.0)
                            _tot["last_l2_decay_strength"] = _leaf.get("l2_decay_strength", 0.0)
                            _col_state = _leaf.get("column_state")
                            if plasticity_column_log_dir is not None and _col_state is not None:
                                _snap_dir = os.path.join(plasticity_column_log_dir, f"{_layer_name}.{_pool_name}")
                                os.makedirs(_snap_dir, exist_ok=True)
                                _extra = {}
                                if plasticity_raw_importance_log:
                                    # Raw, unsmoothed per-synapse importance --
                                    # NOT col_importance (a second EMA on top of
                                    # this). Safe reshape: scattered_nnz==0 for
                                    # every dense-loaded real layer throughout
                                    # training (verified), so layer.importance
                                    # is exactly in_features*out_features long,
                                    # in row-major order. See
                                    # raw_ci_landscape_capture anchor. Also
                                    # captures the raw weight matrix (same
                                    # reshape, same safety guarantee) so a
                                    # later replay can show the actual synapse
                                    # values alongside importance, matching
                                    # live_synapse_display's own two panels.
                                    _layer = _layers_by_name[_layer_name]
                                    _extra["raw_importance"] = np.array(_layer.importance).reshape(
                                        _layer.in_features, _layer.out_features
                                    )
                                    _extra["raw_weight"] = np.array(_layer.weights).reshape(
                                        _layer.in_features, _layer.out_features
                                    )
                                np.savez(
                                    os.path.join(_snap_dir, f"step{step:08d}.npz"),
                                    step=step,
                                    loss_ema=(loss_ema if loss_ema is not None else float("nan")),
                                    acc_ema=(acc_ema if acc_ema is not None else float("nan")),
                                    n_reset_this_cycle=_leaf.get("n_reset_this_cycle", 0),
                                    l2_sat_ratio=_leaf.get("l2_sat_ratio", 0.0),
                                    l2_decay_strength=_leaf.get("l2_decay_strength", 0.0),
                                    **_col_state,
                                    **_extra,
                                )
                            if _live_display is not None and _col_state is not None:
                                _layer = _layers_by_name[_layer_name]
                                _imp = np.array(_layer.importance).reshape(_layer.in_features, _layer.out_features)
                                _w = np.array(_layer.weights).reshape(_layer.in_features, _layer.out_features)
                                _fast = _col_state["col_grad_fast"]
                                _slow = _col_state["col_grad_slow"]
                                _var = _col_state["col_grad_var"]
                                _dev = (_fast - _slow) / (np.sqrt(np.maximum(_var, 0.0)) + 1e-8)
                                _live_display.update_pool(_key, step, _imp, _w, _dev, _col_state["col_reset_active"])
                                if _live_display.closed():
                                    _live_display = None
                if l2_init_enable:
                    model.apply_l2_init(
                        touch_fraction=l2_init_touch_fraction,
                        min_chunk=l2_init_min_chunk,
                        max_chunk=l2_init_max_chunk,
                        rate=l2_init_rate,
                    )
                if dynamic_rank_control:
                    mutated = model.apply_dynamic_rank_control(
                        scale_grace_period_steps=rank_grace_period_steps,
                        additive_grace_period_steps=rank_additive_grace_period_steps,
                    )
                    rank_mutation_count += sum(1 for m in mutated.values() if m)
                    # Orthogonality BEFORE overflow guard -- see l2_decay_and_rank_control_ordering anchor.
                    model.apply_channel_orthogonality_penalty()
                    model.apply_scale_overflow_guard()

        if streak >= streak_threshold:
            _advance_stage(step)
            vocab_now, k_now, phase_now = _current()
            # Odometer graduation; see k_first_target_odometer_reordering anchor.
            graduated_now = (
                (phase_now == "kcycle" and vocab_now >= TASK_VOCAB_MAX and k_now > k_first_target)
                if k_first_target is not None
                else (phase_now == "k" and k_now > k_max)
            )
            if graduated_now:
                ranks = model.report_ranks() if dynamic_rank_control else None
                if log_fn is not None:
                    log_fn(step, *_current()[:2], phase_now, "GRADUATED", loss_ema, acc_ema, ranks=ranks)
                break
        elif wrong_streak >= wrong_streak_threshold and queries_since_level_change >= MIN_QUERIES_BEFORE_REGRESS:
            _regress_stage(step)

        # See docs/research/train_mqar_curriculum.rst:train_curriculum.cli_gradient_sparsity_args
        # (trajectory_log_every paragraph).
        if trajectory_log_fn is not None and (
            (trajectory_log_every is not None and step % trajectory_log_every == 0)
            or (trajectory_log_steps is not None and trajectory_log_steps[0] <= step <= trajectory_log_steps[1])
        ):
            trajectory_log_fn(step, model)

        if step % log_every == 0:
            ranks = _log_ranks(step)
            steps_per_sec = step / (time.time() - t0)
            window_wall_s = time.time() - window_t0
            # Component timing breakdown (task #415): the STANDARD metric
            # going forward, steps_per_sec kept as secondary info -- a
            # single aggregate rate hides which parts a change (e.g.
            # sparsity) actually sped up vs the Amdahl's-law fixed
            # remainder (lm_head/critic_head/attention/orchestration) that
            # doesn't shrink with it. Snapshot BEFORE the closed-loop calls
            # below consume + reset it, so it reflects exactly this window.
            layer_timing_snapshot = model.layer_timing_snapshot()
            # See docs/research/train_mqar_curriculum.rst:
            # train_curriculum.dy_r_target_closed_loop_reset_ordering.
            if target_steps_per_sec is not None:
                model.apply_amortized_dy_r_target_control(steps_per_sec, target_steps_per_sec)
                model.apply_cross_layer_budget_allocator(steps_per_sec, target_steps_per_sec)
            model.reset_layer_timing()
            # qk_spectral_norm_diag_log: cheap (O(elements), q_proj/k_proj
            # only), computed only at this same summary cadence -- see
            # docs/research/toy_tile_recurrence_rmt.rst:qk_spectral_norm_diagnostic.
            qk_spectral_norm = None
            if qk_spectral_norm_diag_log:
                _real_layers = dict(model._named_real_layers())
                qk_spectral_norm = {
                    _name: spectral_norm_upper_bound(
                        np.array(_layer.weights).reshape(_layer.in_features, _layer.out_features)
                    )
                    for _name, _layer in _real_layers.items()
                    if _name in ("q_proj", "k_proj")
                }
            window_t0 = time.time()
            if log_fn is not None:
                log_fn(
                    step,
                    vocab_size,
                    k,
                    phase,
                    "",
                    loss_ema,
                    acc_ema,
                    ranks=ranks,
                    steps_per_sec=steps_per_sec,
                    max_streak=max_streak_seen,
                    dy_r_target=model.dy_r_target,
                    x_r_target=model.x_r_target,
                    layer_timing=layer_timing_snapshot,
                    window_wall_s=window_wall_s,
                    plasticity_totals=(plasticity_totals if plasticity_reset_enable else None),
                    qk_spectral_norm=qk_spectral_norm,
                )
            max_streak_seen = 0

    elapsed_s = time.time() - t0
    final_vocab, final_k, final_phase = _current()
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "steps_per_sec": (step / elapsed_s) if elapsed_s > 0 else 0.0,
        "peak_rss_mb": peak_rss_mb,
        "precision": precision,
        "final_vocab": final_vocab,
        "final_k": final_k,
        "final_phase": final_phase,
        "peak_stage": peak_stage,
        "graduated": (
            (final_phase == "kcycle" and final_vocab >= TASK_VOCAB_MAX and final_k > k_first_target)
            if k_first_target is not None
            else (final_phase == "k" and final_k > k_max)
        ),
        "total_steps": step,
        "elapsed_s": elapsed_s,
        "stage_history": stage_history,
        "dynamic_rank_control": dynamic_rank_control,
        "rank_mutation_count": rank_mutation_count,
        "final_ranks": model.report_ranks() if hasattr(model, "report_ranks") else {},
        "rank_history": rank_history,
    }


def main():
    precision = sys.argv[1] if len(sys.argv) > 1 else "fp4"
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    peak_lr = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_PEAK_LR
    num_tiles = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_NUM_TILES
    k_max = int(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_K_MAX
    # additive_rank/dynamic_rank_control/rank_grace_period_steps: see
    # docs/research/train_mqar_curriculum.rst:train_curriculum.additive_rank_validated_aqrs_config.
    additive_rank = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    dynamic_rank_control = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
    rank_grace_period_steps = int(sys.argv[9]) if len(sys.argv) > 9 else 50
    # use_critic: see train_curriculum.advantage_actor_critic_design in the RST.
    use_critic = bool(int(sys.argv[10])) if len(sys.argv) > 10 else False
    # recurrent_only_output: RNN validation ablation -- see
    # docs/research/toy_tile_recurrence_rmt.rst:toy_tile_recurrence_rmt.recurrent_only_output_ablation.
    recurrent_only_output = bool(int(sys.argv[11])) if len(sys.argv) > 11 else False
    # embed_width/input_sparsity_p/wide_max_weights (Phase 7, task #336): threaded
    # through to ToyTileRecurrenceRMT unchanged; -1 is this script's argv "unset" sentinel.
    embed_width = int(sys.argv[12]) if len(sys.argv) > 12 else EMBED_WIDTH
    _input_sparsity_p_arg = float(sys.argv[13]) if len(sys.argv) > 13 else -1.0
    input_sparsity_p = _input_sparsity_p_arg if _input_sparsity_p_arg >= 0 else None
    _wide_max_weights_arg = int(sys.argv[14]) if len(sys.argv) > 14 else -1
    wide_max_weights = _wide_max_weights_arg if _wide_max_weights_arg >= 0 else None
    # dy_sparsity_p: see JOURNAL.md's 2026-08-30 "backward's real cost profiled
    # to per-synapse update math, not snapshot/merge" entry.
    _dy_sparsity_p_arg = float(sys.argv[15]) if len(sys.argv) > 15 else -1.0
    dy_sparsity_p = _dy_sparsity_p_arg if _dy_sparsity_p_arg >= 0 else None
    # use_tile_cache: see train_curriculum.use_tile_cache_query_step_fallback in the RST.
    use_tile_cache = bool(int(sys.argv[16])) if len(sys.argv) > 16 else False
    # output_dy_sparsity_p: same -1 "unset" sentinel convention as the sparsity args above.
    _output_dy_sparsity_p_arg = float(sys.argv[17]) if len(sys.argv) > 17 else -1.0
    output_dy_sparsity_p = _output_dy_sparsity_p_arg if _output_dy_sparsity_p_arg >= 0 else None
    # wrong_streak_threshold: see docs/research/train_mqar_curriculum.rst:
    # train_curriculum.wrong_streak_threshold_reward_hacking.
    _wrong_streak_threshold_arg = int(sys.argv[18]) if len(sys.argv) > 18 else -1
    wrong_streak_threshold = _wrong_streak_threshold_arg if _wrong_streak_threshold_arg >= 0 else WRONG_STREAK_THRESHOLD
    # dy_r_target/dy_k_min/dy_k_max/target_steps_per_sec/dy_surprise_alpha/x_r_target/
    # x_k_min/trajectory_log_every: see docs/research/train_mqar_curriculum.rst:
    # train_curriculum.cli_gradient_sparsity_args.
    _dy_r_target_arg = float(sys.argv[19]) if len(sys.argv) > 19 else -1.0
    dy_r_target = _dy_r_target_arg if _dy_r_target_arg >= 0 else None
    dy_k_min = int(sys.argv[20]) if len(sys.argv) > 20 else 0
    _dy_k_max_arg = int(sys.argv[21]) if len(sys.argv) > 21 else -1
    dy_k_max = _dy_k_max_arg if _dy_k_max_arg >= 0 else None
    _target_sps_arg = float(sys.argv[22]) if len(sys.argv) > 22 else -1.0
    target_steps_per_sec = _target_sps_arg if _target_sps_arg >= 0 else None
    _dy_surprise_alpha_arg = float(sys.argv[23]) if len(sys.argv) > 23 else -1.0
    dy_surprise_alpha = _dy_surprise_alpha_arg if _dy_surprise_alpha_arg >= 0 else None
    _x_r_target_arg = float(sys.argv[24]) if len(sys.argv) > 24 else -1.0
    x_r_target = _x_r_target_arg if _x_r_target_arg >= 0 else None
    x_k_min = int(sys.argv[25]) if len(sys.argv) > 25 else 0
    _trajectory_log_every_arg = int(sys.argv[26]) if len(sys.argv) > 26 else -1
    trajectory_log_every = _trajectory_log_every_arg if _trajectory_log_every_arg > 0 else None
    # r_target_min: floor the dy_r_target/x_r_target closed loop ratchets down
    # to -- see JOURNAL.md's 2026-09-07 adaptive_70k entry (quality collapsed
    # at the previously-hardcoded 0.05 floor well before reaching the
    # requested speedup). -1 sentinel keeps the 0.05 default, matching this
    # file's other sparsity-arg conventions.
    _r_target_min_arg = float(sys.argv[27]) if len(sys.argv) > 27 else -1.0
    r_target_min = _r_target_min_arg if _r_target_min_arg >= 0 else 0.05

    print(
        f"# MQAR curriculum precision={precision} max_steps={max_steps} seed={seed} "
        f"peak_lr={peak_lr} num_tiles={num_tiles} k_max={k_max} additive_rank={additive_rank} "
        f"dynamic_rank_control={dynamic_rank_control} rank_grace_period_steps={rank_grace_period_steps} "
        f"use_critic={use_critic} recurrent_only_output={recurrent_only_output} "
        f"embed_width={embed_width} input_sparsity_p={input_sparsity_p} wide_max_weights={wide_max_weights} "
        f"dy_sparsity_p={dy_sparsity_p} use_tile_cache={use_tile_cache} "
        f"output_dy_sparsity_p={output_dy_sparsity_p} "
        f"streak_threshold={STREAK_THRESHOLD} wrong_streak_threshold={wrong_streak_threshold} "
        f"dy_r_target={dy_r_target} dy_k_min={dy_k_min} dy_k_max={dy_k_max} "
        f"target_steps_per_sec={target_steps_per_sec} dy_surprise_alpha={dy_surprise_alpha} "
        f"x_r_target={x_r_target} x_k_min={x_k_min} trajectory_log_every={trajectory_log_every} "
        f"r_target_min={r_target_min}",
        flush=True,
    )

    _SHORT_NAME = {"input_proj": "in", "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "lm_head": "lm"}

    def _ranks_str(ranks):
        if not ranks:
            return ""
        parts = [f"{_SHORT_NAME.get(n, n)}={s}/{a}" for n, (s, a) in ranks.items()]
        return "  ranks[" + " ".join(parts) + "]"

    # Component timing breakdown (task #415) -- STANDARD per-window report
    # going forward. steps_per_sec (aggregate, whole-run-average) is kept
    # as secondary info alongside it, not replaced: a single rate can't
    # show which components a change actually sped up vs the Amdahl's-law
    # fixed remainder (lm_head/critic_head/attention/orchestration) that
    # doesn't shrink with sparsity. "other" is the residual --
    # window_wall_s minus every named component's fwd_s+bwd_s -- i.e.
    # token embedding, target-building, CSR construction, and all
    # Python-level loop orchestration this window, none of which is
    # sparsified by x_r_target/dy_r_target. See
    # docs/research/train_mqar_curriculum.rst:component_timing_breakdown_design.
    _TIMING_ORDER = ["input_proj", "q_proj", "k_proj", "v_proj", "o_proj", "attention", "lm_head", "critic_head"]

    def _layer_timing_str(layer_timing, window_wall_s):
        if not layer_timing or window_wall_s is None:
            return ""
        parts = []
        accounted_s = 0.0
        for name in _TIMING_ORDER:
            rec = layer_timing.get(name)
            if rec is None:
                continue
            comp_s = rec["fwd_s"] + rec["bwd_s"]
            accounted_s += comp_s
            pct = 100.0 * comp_s / window_wall_s if window_wall_s > 0 else 0.0
            parts.append(f"{_SHORT_NAME.get(name, name)}={comp_s:.2f}s({pct:.0f}%)")
        other_s = max(0.0, window_wall_s - accounted_s)
        other_pct = 100.0 * other_s / window_wall_s if window_wall_s > 0 else 0.0
        parts.append(f"other={other_s:.2f}s({other_pct:.0f}%)")
        return "  t[" + " ".join(parts) + f" / {window_wall_s:.2f}s]"

    def _plasticity_totals_str(totals):
        # Attributes a loss spike to plasticity_reset actually firing (and
        # on which layer/pool, at what deviation) instead of only showing
        # the downstream loss effect -- see feedback_present_before_keep_prune_decisions
        # (needed to untangle results DURING the run, not just at the end).
        if not totals:
            return ""
        n_reset = sum(t["n_reset"] for t in totals.values())
        worst_key, worst = max(totals.items(), key=lambda kv: kv[1]["last_deviation"])
        return (
            f"  plasticity[reset={n_reset} "
            f"worst={worst_key}(dev={worst['last_deviation']:.2f}"
            f"[{worst['last_min_deviation']:.2f},{worst['last_max_deviation']:.2f}]"
            f",imp={worst['last_importance']:.4f})]"
        )

    def _qk_spectral_norm_str(qk_spectral_norm):
        if not qk_spectral_norm:
            return ""
        parts = [f"{_SHORT_NAME.get(n, n)}={v:.2f}" for n, v in qk_spectral_norm.items()]
        return "  qk_specnorm[" + " ".join(parts) + "]"

    def log_fn(
        step,
        vocab_size,
        k,
        phase,
        event,
        loss_ema,
        acc_ema,
        ranks=None,
        steps_per_sec=None,
        max_streak=None,
        dy_r_target=None,
        x_r_target=None,
        layer_timing=None,
        window_wall_s=None,
        plasticity_totals=None,
        qk_spectral_norm=None,
    ):
        loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
        acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
        tag = f"  [{event}]" if event else ""
        # steps_per_sec: secondary info now, see _layer_timing_str's own
        # comment -- kept in the line, just no longer the primary signal.
        sps_s = f"  steps/sec={steps_per_sec:.1f}" if steps_per_sec is not None else ""
        streak_s = f"  max_streak={max_streak:>2}/{STREAK_THRESHOLD}" if max_streak is not None else ""
        timing_s = _layer_timing_str(layer_timing, window_wall_s)
        plasticity_s = _plasticity_totals_str(plasticity_totals)
        qk_specnorm_s = _qk_spectral_norm_str(qk_spectral_norm)

        def _r_target_str(label, d):
            # Shared dy_r_target/x_r_target formatter; see cli_gradient_sparsity_args anchor.
            if not d:
                return ""
            _set = {n: v for n, v in d.items() if v is not None}
            if not _set:
                return ""
            return "  " + label + "[" + " ".join(f"{_SHORT_NAME.get(n, n)}={v:.3f}" for n, v in _set.items()) + "]"

        dy_r_s = _r_target_str("dy_r_target", dy_r_target)
        x_r_s = _r_target_str("x_r_target", x_r_target)
        print(
            f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
            f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{timing_s}{sps_s}{streak_s}{dy_r_s}{x_r_s}"
            f"{_ranks_str(ranks)}{plasticity_s}{qk_specnorm_s}",
            flush=True,
        )

    def trajectory_log_fn_default(step, model) -> None:
        # Default fine-grained printer; see cli_gradient_sparsity_args anchor
        # (trajectory_log_every paragraph) in the RST.
        dy_parts = []
        for name, r_bar in model.dy_r_target.items():
            if r_bar is None:
                continue
            surprise = model._layer_surprise.get(name)
            sel = model.last_grad_selection.get(name)
            bits = [f"r{r_bar:.3f}"]
            if sel is not None:
                bits.append(f"R{sel['R_mean']:.3f}")
                bits.append(f"k{sel['k_mean']:.1f}")
            if surprise is not None:
                bits.append(f"E{surprise['E_t']:.2g}")
                bits.append(f"L{surprise['Lbar']:.2g}")
            dy_parts.append(f"{_SHORT_NAME.get(name, name)}=" + ",".join(bits))
        x_parts = []
        for name, x_target in model.x_r_target.items():
            if x_target is None:
                continue
            sel = model.last_input_selection.get(name)
            if sel is not None:
                x_parts.append(
                    f"{_SHORT_NAME.get(name, name)}=target{x_target:.3f},R{sel['R_mean']:.3f},k{sel['k_mean']:.1f}"
                )
            else:
                x_parts.append(f"{_SHORT_NAME.get(name, name)}=target{x_target:.3f}")
        if dy_parts:
            print(f"    [TRAJ] step={step:>7} axis=dy " + " ".join(dy_parts), flush=True)
        if x_parts:
            print(f"    [TRAJ] step={step:>7} axis=x  " + " ".join(x_parts), flush=True)

    r = train_curriculum(
        precision,
        max_steps,
        seed,
        peak_lr,
        num_tiles,
        k_max,
        log_fn=log_fn,
        trajectory_log_every=trajectory_log_every,
        trajectory_log_fn=(trajectory_log_fn_default if trajectory_log_every is not None else None),
        additive_rank=additive_rank,
        dynamic_rank_control=dynamic_rank_control,
        rank_grace_period_steps=rank_grace_period_steps,
        use_critic=use_critic,
        recurrent_only_output=recurrent_only_output,
        embed_width=embed_width,
        input_sparsity_p=input_sparsity_p,
        wide_max_weights=wide_max_weights,
        dy_sparsity_p=dy_sparsity_p,
        use_tile_cache=use_tile_cache,
        output_dy_sparsity_p=output_dy_sparsity_p,
        wrong_streak_threshold=wrong_streak_threshold,
        dy_r_target=dy_r_target,
        dy_k_min=dy_k_min,
        dy_k_max=dy_k_max,
        dy_surprise_alpha=dy_surprise_alpha,
        x_r_target=x_r_target,
        x_k_min=x_k_min,
        target_steps_per_sec=target_steps_per_sec,
        r_target_min=r_target_min,
    )
    print(
        f"\nFINAL precision={precision} final_vocab={r['final_vocab']} final_k={r['final_k']} "
        f"final_phase={r['final_phase']} graduated={r['graduated']} "
        f"total_steps={r['total_steps']} steps_per_sec={r['steps_per_sec']:.1f} "
        f"peak_rss_mb={r['peak_rss_mb']:.1f} "
        f"({r['elapsed_s']:.0f}s)",
        flush=True,
    )
    print(
        f"PEAK precision={precision} peak_vocab={r['peak_stage']['vocab']} "
        f"peak_k={r['peak_stage']['k']} peak_phase={r['peak_stage']['phase']}",
        flush=True,
    )
    if r["dynamic_rank_control"]:
        print(f"RANK_MUTATIONS precision={precision} count={r['rank_mutation_count']}", flush=True)
        for name, (scale_r, add_r) in r["final_ranks"].items():
            print(f"  {name:<12} scale_rank={scale_r}  additive_rank={add_r}", flush=True)
        # Per-log_every/per-transition rank trace; same convention as STAGE_HISTORY_JSON.
        print("RANK_HISTORY_JSON " + json.dumps(r["rank_history"]), flush=True)
    print("STAGE_HISTORY_JSON " + json.dumps(r["stage_history"]), flush=True)


if __name__ == "__main__":
    main()
