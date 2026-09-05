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
  precision: fp4 | fp8 | fp32
See docs/research/train_mqar_curriculum.rst:train_curriculum.cli_gradient_sparsity_args for
dy_r_target/dy_k_min/dy_k_max/target_steps_per_sec/dy_surprise_alpha/x_r_target/x_k_min/
trajectory_log_every semantics. See each CLI arg's own comment in main() below for the rest.
"""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from sili import _cpu
from sili.sparse_rnn import DISLDOLayer, DISLDOLayer8, DISLDOLayer32
from sili.tensor import combine_losses

from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum, predicted_token
from model.toy_recall_task import generate_mqar_sequence
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT
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

PRECISION_CLS = {"fp4": DISLDOLayer, "fp8": DISLDOLayer8, "fp32": DISLDOLayer32}


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
NOCAPS_KWARGS_FP32 = {"max_abs_delta": 2.0, "max_ci": 100.0}
PRECISION_SYNAPSE_KWARGS = {"fp4": NOCAPS_KWARGS, "fp8": NOCAPS_KWARGS_FP8, "fp32": NOCAPS_KWARGS_FP32}

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
    target_steps_per_sec: float | None = None,
    trajectory_log_every: int | None = None,
    trajectory_log_steps: tuple[int, int] | None = None,
    trajectory_log_fn=None,
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
    if k_first_target is not None and k_first_vocab is None:
        k_first_vocab = seq_len_for_k(k_first_target) + 4

    disldo_cls = PRECISION_CLS[precision]
    state_width = embed_width * COLUMN_NEURONS

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
        dense=True,
        clip_range=clip_range,
        l1_sparsity_coef=L1_SPARSITY_COEF,
        synapse_kwargs=dict(PRECISION_SYNAPSE_KWARGS[precision]),
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
    step = 0
    while step < max_steps:
        step += 1
        stage_step += 1
        if step <= WARMUP_STEPS:
            lr = peak_lr * step / WARMUP_STEPS
        elif acc_ema is None:
            lr = peak_lr
        else:
            frac = max(MIN_LR_FRAC, min(1.0, 1.0 - acc_ema))
            lr = peak_lr * frac

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
        targets = _build_targets(tokens, mqar_pairs, k)
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
                clip_grad_norm_(model.parameters_for_optimizer(), MAX_GRAD_NORM)
                opt.step(model.parameters_for_optimizer(), lr=lr)
                # See docs/research/train_mqar_curriculum.rst:
                # train_curriculum.l2_decay_and_rank_control_ordering.
                if l2_decay_chunk_size is not None:
                    model.apply_amortized_l2_decay(l2_decay_chunk_size, l2_decay_adaptation_rate)
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
            # See docs/research/train_mqar_curriculum.rst:
            # train_curriculum.dy_r_target_closed_loop_reset_ordering.
            if target_steps_per_sec is not None:
                model.apply_amortized_dy_r_target_control(steps_per_sec, target_steps_per_sec)
                model.apply_cross_layer_budget_allocator(steps_per_sec, target_steps_per_sec)
                model.reset_layer_timing()
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
        f"x_r_target={x_r_target} x_k_min={x_k_min} trajectory_log_every={trajectory_log_every}",
        flush=True,
    )

    _SHORT_NAME = {"input_proj": "in", "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "lm_head": "lm"}

    def _ranks_str(ranks):
        if not ranks:
            return ""
        parts = [f"{_SHORT_NAME.get(n, n)}={s}/{a}" for n, (s, a) in ranks.items()]
        return "  ranks[" + " ".join(parts) + "]"

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
    ):
        loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
        acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
        tag = f"  [{event}]" if event else ""
        sps_s = f"  steps/sec={steps_per_sec:.1f}" if steps_per_sec is not None else ""
        streak_s = f"  max_streak={max_streak:>2}/{STREAK_THRESHOLD}" if max_streak is not None else ""

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
            f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{streak_s}{dy_r_s}{x_r_s}{_ranks_str(ranks)}",
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
            if surprise is not None:
                dy_parts.append(
                    f"{_SHORT_NAME.get(name, name)}=r{r_bar:.3f},E{surprise['E_t']:.2g},L{surprise['Lbar']:.2g}"
                )
            else:
                dy_parts.append(f"{_SHORT_NAME.get(name, name)}=r{r_bar:.3f}")
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
