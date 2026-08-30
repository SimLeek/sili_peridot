"""
scripts/train_mqar_curriculum.py
──────────────────────────────────
Adaptive MQAR curriculum with student-paced difficulty (task #264):
grows vocab_size first (K=1 fixed), then grows num_kv_pairs (K) once
vocab reaches TASK_VOCAB_MAX. A stage advances only after a streak of
STREAK_THRESHOLD consecutive correct query predictions during live
training, and REGRESSES one stage after a streak of
WRONG_STREAK_THRESHOLD consecutive wrong predictions -- "if a student
isn't learning you go over the specific things they're not doing
good on" (direct instruction) -- not a step-count schedule, and not a
one-way ratchet either.

Level-change signal (direct instruction): two reserved tokens,
LEVEL_UP_TOKEN/LEVEL_DOWN_TOKEN, are fed as INPUT ONLY (never a
prediction target) at the start of the next sequence whenever a stage
transition happens, so the model can perceive that the rules just
changed instead of a stage transition being invisible in the token
stream. Implemented by prepending the token to that one sequence and
shifting every position by +1 -- structurally identical to how an
ordinary filler token flows through _build_tile_window, just with a
fixed, meaningful identity instead of random noise, and no target
attached to it.

NUM_TILES is a MODEL parameter (the local sliding-attention window,
see _build_tile_window's own docstring), deliberately DECOUPLED from
the task's seq_len/K (direct instruction -- the older
train_mqar_precision_sweep.py sweep's num_tiles=seq_len coupling was
flagged as wrong). Stays FIXED for the whole run. Default 16: covers
seq_len_for_k(k) for k=1..4 (max key-query distance 15) entirely
in-window, so the model gets a real base of in-context-solvable K
stages before K=5 (seq_len=20, max distance 19) forces the first
genuine reliance on the RMT memory tokens -- "K=4 would be a good
amount of examples to learn the pattern in context before requiring
memory" (direct instruction). The in-context -> memory-required
transition is left to happen naturally within one continuous
curriculum run, not run as two separate experiments.

Run: python3 scripts/train_mqar_curriculum.py <precision> [max_steps] [seed] [peak_lr] [num_tiles] [k_max]
  [additive_rank] [dynamic_rank_control] [rank_grace_period_steps] [use_critic]
  [recurrent_only_output] [embed_width] [input_sparsity_p] [wide_max_weights]
  precision: fp4 | fp8 | fp32
  embed_width: model width (sparsity plan Phase 6/7, task #335/#336) --
    default EMBED_WIDTH (16); pass 32 to widen input_proj/q/k/v/o_proj
  input_sparsity_p: density fraction (0..1) for those 5 layers' forward
    input; -1 (default) = unset/dense (today's exact behavior)
  wide_max_weights: per-layer synapse budget override for those same 5
    layers; -1 (default) = unset (shares max_weights with lm_head)
"""
from __future__ import annotations

import sys
import time
import json
import resource
from typing import Optional

import numpy as np

sys.path.insert(0, ".")

from sili.sparse_rnn import DISLDOLayer, DISLDOLayer8, DISLDOLayer32
from sili import _cpu
from model.toy_recall_task import generate_mqar_sequence
from model.toy_recall_models import cross_entropy_sum, predicted_token, AdamOptimizer, clip_grad_norm_
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT
from sili.tensor import combine_losses
from scripts.train_tile_curriculum import _build_tile_window
from scripts.train_mqar_rmt_reference import (
    seq_len_for_k, _build_targets,
    EMBED_WIDTH, COLUMN_NEURONS, NUM_MEMORY_SLOTS, MAX_WEIGHTS_PER_LAYER,
    NUM_CPUS, VOCAB, WARMUP_STEPS, MAX_GRAD_NORM, CLIP_RANGE, L1_SPARSITY_COEF,
)

PRECISION_CLS = {"fp4": DISLDOLayer, "fp8": DISLDOLayer8, "fp32": DISLDOLayer32}
NOCAPS_KWARGS = {"max_abs_delta": 1e30, "max_ci": 1e30}
# FP8 needs a real (non-infinite) max_abs_delta -- see conversation/
# sili__new's linear_disldo.hpp fix: FP4's block4 backward computed cw
# in CODE-SPACE with S properly threaded through SynapsePolicy::update_cw,
# so its own quantization ceiling (raw code magnitude <=6) gave it an
# IMPLICIT per-step safety margin even under an uncapped max_abs_delta.
# FP8's block4 backward used to compute cw in TRUE-WEIGHT space with
# S hardcoded to 1, which on its own was a real, separate bug (fixed:
# an S-independent RMSprop step divided by a small output_scale at
# write time amplified every update by ~1/S, up to ~287x for a wide
# fan-in-corrected layer). Once that's fixed to match FP4's convention,
# FP8's REMAINING difference from FP4 is purely its much wider code
# range (E4M3 max ~448 vs FP4's ~6) -- an uncapped cold-start delta
# that FP4's narrow range self-limits harmlessly can still drift FP8
# noticeably before its own (much later) natural ceiling kicks in.
# Confirmed directly: FP8 is fully stable (40-step diagnostic, loss
# steady ~4.2-5.1) under the library's own tuned production default
# (kSynapsePolicyMaxAbsDelta=2.0, sili/cpu_backend.cpp), so reuse that
# exact value here rather than guessing a new one -- gives FP8 the same
# kind of per-step code-space ceiling FP4 gets for free, while FP4/FP32
# stay on the deliberately uncapped NOCAPS_KWARGS unchanged.
NOCAPS_KWARGS_FP8 = {"max_abs_delta": 2.0, "max_ci": 1e30}
PRECISION_SYNAPSE_KWARGS = {"fp4": NOCAPS_KWARGS, "fp8": NOCAPS_KWARGS_FP8, "fp32": NOCAPS_KWARGS}

DEFAULT_PEAK_LR = 0.015
DEFAULT_NUM_TILES = 16         # fixed local-attention window (model param, not a task param)
LEVEL_UP_TOKEN = VOCAB - 2     # 126 -- reserved, never chosen as an MQAR key/value
LEVEL_DOWN_TOKEN = VOCAB - 1   # 127 -- reserved, never chosen as an MQAR key/value
TASK_VOCAB_MAX = VOCAB - 2     # curriculum vocab_size grows up to (not including) this,
                                # so [0, TASK_VOCAB_MAX) never collides with the level tokens
VOCAB_START = 8                # min viable: must exceed seq_len_for_k(1)=4
VOCAB_GROWTH_FACTOR = 2.0      # doubles each promotion: 8->16->32->64->126(clamped)
K_START = 1
DEFAULT_K_MAX = 10
STREAK_THRESHOLD = 10          # consecutive correct queries to advance a stage
WRONG_STREAK_THRESHOLD = 5     # consecutive wrong queries to regress a stage
MIN_QUERIES_BEFORE_REGRESS = 30  # grace period: a fresh stage gets this many query
                                  # attempts before regression can trigger at all --
                                  # without this, ordinary first-contact difficulty
                                  # on a harder stage (expected, not "isn't learning")
                                  # was hitting WRONG_STREAK_THRESHOLD almost
                                  # immediately and thrashing level_up/level_down
                                  # every ~10-15 steps -- confirmed directly (smoke
                                  # test oscillated vocab 16<->32 repeatedly).
MIN_LR_FRAC = 0.05
LOSS_EMA_DECAY = 0.98          # kept for logging only; LR itself is accuracy-driven
ACC_EMA_DECAY = 0.98

# Advantage actor-critic (task #272): the critic predicts the per-vocab-
# neuron squared error the actor's own logits will incur; since the true
# target token is known the same step, its regression target
# (softmax(logits) - onehot)**2 is exact, not estimated. advantage =
# true - predicted then reweights the actor's own cross-entropy gradient
# per vocab neuron (bigger correction where the critic was most
# surprised). Clipped to keep the (1+advantage) multiplier bounded --
# true_loss_vec entries are bounded in [0,1], but a diverging critic
# prediction early in training could otherwise blow this up unguarded.
ADVANTAGE_CLIP = 5.0

# Reward/punish asymmetry -- REMOVED (task #302/#306, direct instruction,
# after a real 3-arm ablation isolated it as the cause of a genuine
# training regression, distinct from the NaN investigation): scaling every
# wrong-class neuron's gradient by a uniform punish_scale < 1 introduces a
# systematic, one-directional bias into the aggregate gradient flowing
# into the shared trunk -- the one-hot structure means there's exactly ONE
# full-strength reward term per query but (vocab_size-1) DAMPENED
# punishment terms, so the two no longer cancel in aggregate the way
# softmax's own math guarantees they do when both sides are unscaled. This
# isn't a per-step sign flip visible in the formula -- it's a slow,
# compounding drift, confirmed directly: a 3-arm ablation (same seed) held
# BOTH "neither mechanism" and "magnitude penalty only" stable (loss
# bounded ~3-30 through 3000 steps) while "reward/punish asymmetry only"
# diverged smoothly from ~5 to ~4400 over the same window -- the gradual
# ramp is the signature of a compounding bias, not a numerical edge case.
# Tried twice (1/((vocab_size-1)*STREAK_THRESHOLD) and 1/STREAK_THRESHOLD)
# and failed differently each time; not reintroduced without a redesign
# that doesn't rely on a single uniform per-wrong-neuron scale.

# Diagnostics (task #303): when True, checks logits.data/critic_pred.data
# for non-finite values EVERY step (not just every log_every), right
# after model.step() returns -- i.e. checks the raw forward output BEFORE
# any of _backward_with_critic's own processing touches it. On first
# failure, dumps a full per-layer weight health scan (raw value_scale/
# output_scale/additive_u/additive_v vectors) and raises immediately, so
# the exact step and the exact layer/branch that went non-finite first
# are both known -- rather than only knowing "somewhere before the next
# periodic log line."
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


# Rolling per-position magnitude trace (task #303, direct instruction --
# "why isn't there some extremely strong loss before outputs get NaN, or
# are the outputs not large but the in-between kernel values large
# instead"): records max(|x|) over every last_debug array at EVERY
# position (not just the one that eventually fails), capped so memory
# doesn't grow unbounded on a long run. On failure, the tail of this is
# included in the report -- direct evidence of whether the blow-up is a
# gradual ramp visible from the OUTSIDE (finite activations climbing over
# many steps) or a one-position cliff (everything looks normal, then the
# very next position is already fully NaN with no warning).
_MAGNITUDE_TRACE_MAXLEN = 40
_magnitude_trace = []


def _record_magnitude_trace(model, step: int, i: int, loss_ema) -> None:
    entry = {"step": step, "i": i, "loss_ema": loss_ema}
    for name, arr in getattr(model, "last_debug", {}).items():
        finite = arr[np.isfinite(arr)]
        entry[name] = float(np.abs(finite).max()) if finite.size else float("nan")
        # sigmas specifically: the failure mode we're chasing is UNDERFLOW
        # toward 0 (1/(2*sigma**2) -> Inf inside gaussian_attention), so
        # max|x| alone would hide it -- record the min too, only for this
        # array, rather than doubling every entry's size for no reason.
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
    # Bisect through the forward chain (task #303) -- x_wide -> q/k/v ->
    # attn (pre/post o_proj) -> raw_combined/pre_norm_combined/
    # combined_new -> pooled -- to find the EARLIEST stage that's already
    # non-finite, not just that the final output is.
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
    trace_lines = [f"  step={e['step']} i={e['i']} loss_ema={e['loss_ema']} " +
                   " ".join(f"{k}={v:.4g}" for k, v in e.items()
                           if k not in ("step", "i", "loss_ema"))
                  for e in _magnitude_trace]
    report = (f"NON-FINITE at step={step} i={i}\n  " + "\n  ".join(bad) +
             "\nLayer weight health (nonfinite_count, total) for layers with issues:\n" +
             ("\n".join(health_lines) if health_lines else "  (none -- corruption is in activations only, not stored weights)") +
             f"\nranks={model.report_ranks()}" +
             f"\nMagnitude trace, last {len(_magnitude_trace)} positions (max|finite value| per stage):\n" +
             "\n".join(trace_lines))
    raise RuntimeError(report)


def next_vocab(vocab_size: int) -> int:
    return min(TASK_VOCAB_MAX, int(round(vocab_size * VOCAB_GROWTH_FACTOR)))


def prev_vocab(vocab_size: int) -> int:
    return max(VOCAB_START, int(round(vocab_size / VOCAB_GROWTH_FACTOR)))


def _backward_with_critic(model, logits, target_token: int, num_tiles: int, aux) -> None:
    """Advantage-actor-critic backward for one query position (task #272).
    logits: [num_tiles, vocab_size] Tensor, query row = num_tiles-1.
    Replaces the plain unweighted cross-entropy backward with a critic-
    reweighted one -- see ADVANTAGE_CLIP's own comment for the formula.
    The critic itself trains via plain one-step MSE regression against
    the EXACT per-vocab squared error (known immediately since the
    target token is known this same step -- no TD/bootstrap/target-net
    needed, unlike sili__new's mandelbrot RTAC, which needs those because
    its reward isn't known until later)."""
    row = num_tiles - 1
    vocab_size = logits.data.shape[-1]
    row_logits = logits.data[row]
    shifted = row_logits - row_logits.max()
    exp_l = np.exp(shifted)
    probs = exp_l / exp_l.sum()
    onehot = np.zeros(vocab_size, dtype=np.float32)
    onehot[target_token] = 1.0
    true_loss_vec = (probs - onehot) ** 2

    critic_pred = model.last_critic_pred
    # nan_to_num FIRST: np.clip does NOT sanitize NaN (clip(nan, lo, hi) ==
    # nan), so an unguarded critic divergence would flow straight through
    # into g_logits below and corrupt the whole model -- found via a real
    # 45k-step fp8 run that NaN-collapsed by step 3400 (far earlier than
    # any prior curriculum run), matching this codebase's own established
    # pattern (see _overflow_guard_array in sili__new's sparse_rnn.py).
    # Treating a non-finite prediction as "no correction" (advantage=0)
    # rather than propagating it also protects the critic's OWN gradient
    # below, since g_critic reuses this same sanitized pred_row.
    pred_row = np.nan_to_num(np.asarray(critic_pred.data[row], dtype=np.float32),
                             nan=0.0, posinf=ADVANTAGE_CLIP, neginf=-ADVANTAGE_CLIP)
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
    # total order matching how stages are actually visited: every
    # k-phase stage is harder than every vocab-phase stage.
    return (1, stage["k"]) if stage["phase"] == "k" else (0, stage["vocab"])


def train_curriculum(precision: str, max_steps: int, seed: int, peak_lr: float,
                     num_tiles: int, k_max: int, log_every: int = 200,
                     log_fn=None, additive_rank: int = 1,
                     dynamic_rank_control: bool = True,
                     rank_grace_period_steps: int = 50,
                     use_critic: bool = False,
                     magnitude_clip_penalty_coef: float = 0.0,
                     recurrent_only_output: bool = False,
                     embed_width: int = EMBED_WIDTH,
                     input_sparsity_p: Optional[float] = None,
                     wide_max_weights: Optional[int] = None) -> dict:
    # embed_width/input_sparsity_p/wide_max_weights (sparsity plan Phase
    # 7, task #336): real values threaded straight through to
    # ToyTileRecurrenceRMT's own identically-named constructor args (see
    # its own docstring for the full rationale) -- embed_width defaults
    # to the module-level EMBED_WIDTH constant (today's exact value,
    # unchanged), the other two default to None (today's exact
    # unwidened/dense behavior). COLUMN_NEURONS is NOT parameterized
    # here (stays the fixed numenta reference value, per direct
    # instruction) -- only embed_width varies.
    disldo_cls = PRECISION_CLS[precision]
    state_width = embed_width * COLUMN_NEURONS

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    model_rng = np.random.default_rng(seed)

    # num_tiles is fixed at construction and NEVER changes as the
    # curriculum's vocab/K stage grows -- this is the whole point of
    # decoupling the model's local window from the task.
    model = ToyTileRecurrenceRMT(
        VOCAB, embed_width, COLUMN_NEURONS, num_tiles, NUM_MEMORY_SLOTS,
        MAX_WEIGHTS_PER_LAYER, num_cpus=NUM_CPUS, disldo_cls=disldo_cls,
        dense=True, clip_range=CLIP_RANGE, l1_sparsity_coef=L1_SPARSITY_COEF,
        synapse_kwargs=dict(PRECISION_SYNAPSE_KWARGS[precision]), scale_rank=1,
        additive_rank=additive_rank, dynamic_rank_control=dynamic_rank_control,
        use_critic=use_critic,
        magnitude_clip_penalty_coef=magnitude_clip_penalty_coef,
        recurrent_only_output=recurrent_only_output,
        input_sparsity_p=input_sparsity_p, wide_max_weights=wide_max_weights,
        rng=model_rng)
    opt = AdamOptimizer()
    embed_table = rng.randn(VOCAB, embed_width).astype(np.float32) * 0.3

    stage_stack = [{"vocab": VOCAB_START, "k": K_START, "phase": "vocab"}]
    streak = 0
    wrong_streak = 0
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
        # Records a (scale_rank, additive_rank) snapshot per layer at
        # this step -- called at every periodic log point AND every
        # level transition, not just once at the end, so a run's rank
        # trajectory (growth/shrink timing relative to curriculum
        # progress) can be read back even if the run is killed early or
        # takes far longer than expected (direct instruction).
        if not dynamic_rank_control:
            return None
        ranks = model.report_ranks()
        rank_history.append({"step": step, "ranks": ranks})
        return ranks

    def _advance_stage(step):
        nonlocal streak, wrong_streak, stage_step, queries_since_level_change
        nonlocal pending_level_token, peak_key, peak_stage
        cur = stage_stack[-1]
        if cur["phase"] == "vocab":
            nv = next_vocab(cur["vocab"])
            new_stage = {"vocab": nv, "k": cur["k"],
                        "phase": "k" if nv >= TASK_VOCAB_MAX else "vocab"}
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
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab_size, seq_len, k)
        targets = _build_targets(tokens, mqar_pairs, k)
        query_positions = set(pos for pos, _ in mqar_pairs)

        offset = 1 if pending_level_token is not None else 0
        if offset:
            combined_tokens = np.concatenate(([pending_level_token], tokens))
            pending_level_token = None
        else:
            combined_tokens = tokens
        targets = {pos + offset: tgt for pos, tgt in targets.items()}
        query_positions = set(pos + offset for pos in query_positions)

        memory = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
        for i in range(seq_len + offset):
            window = _build_tile_window(embed_table, combined_tokens, i, num_tiles)
            # requires_grad=False for non-query steps (direct instruction):
            # the sequential write-then-read design (see step()'s own
            # docstring) makes k_proj/v_proj/o_proj forward()-run TWICE per
            # step, and most steps never get a loss.backward() at all (only
            # `i in targets` does) -- building a backward graph for those is
            # both wasted work and, more importantly, would leave the C++
            # engine's DenseInputStack accumulating never-popped entries
            # across every non-query step in the sequence until it hits its
            # cap and throws.
            memory, logits, aux = model.step(window, memory, lr, requires_grad=(i in targets))
            if DEBUG_FINITE_CHECK and use_critic:
                _check_finite_or_raise(model, logits, step, i, loss_ema)
            if i in targets:
                loss = cross_entropy_sum(logits, [(num_tiles - 1, targets[i])])
                loss_ema = float(loss.data) if loss_ema is None else (
                    LOSS_EMA_DECAY * loss_ema + (1.0 - LOSS_EMA_DECAY) * float(loss.data))
                if i in query_positions:
                    correct = predicted_token(logits, num_tiles - 1) == targets[i]
                    acc_ema = float(correct) if acc_ema is None else (
                        ACC_EMA_DECAY * acc_ema + (1.0 - ACC_EMA_DECAY) * float(correct))
                    queries_since_level_change += 1
                    if correct:
                        streak += 1
                        wrong_streak = 0
                    else:
                        wrong_streak += 1
                        streak = 0
                if use_critic:
                    _backward_with_critic(model, logits, targets[i], num_tiles, aux)
                else:
                    if aux is not None:
                        loss = loss + aux
                    loss.backward()
                clip_grad_norm_(model.parameters_for_optimizer(), MAX_GRAD_NORM)
                opt.step(model.parameters_for_optimizer(), lr=lr)
                if dynamic_rank_control:
                    mutated = model.apply_dynamic_rank_control(grace_period_steps=rank_grace_period_steps)
                    rank_mutation_count += sum(1 for m in mutated.values() if m)
                    # AQRS channel-diversity pass (task #295 follow-up,
                    # chosen over residual-targeted growth -- direct
                    # instruction): stops rank channels converging to
                    # duplicate directions during training, which
                    # nothing else here catches (neurogenesis's own
                    # health check is magnitude-only; l1_sparsity_coef
                    # only sees the summed output). Deliberately called
                    # BEFORE the overflow guard below, not after --
                    # found via a real fp8 run (see conversation): this
                    # pass's own correction can in principle still be
                    # large, so the overflow guard must always run LAST
                    # as the actual numerical-safety net, not have
                    # something unguarded applied on top of it.
                    model.apply_channel_orthogonality_penalty()
                    # AQRS scale/additive channel numerical-safety pass
                    # (task #295 follow-up): only relevant once rank can
                    # genuinely exceed the old hardcoded cap=4, i.e. only
                    # under dynamic_rank_control -- see conversation for
                    # the real fp8 NaN collapse this fixes (get_scale()'s
                    # combined envelope has no clamp, and rank growing
                    # past 4 let it overflow in a real curriculum run).
                    model.apply_scale_overflow_guard()

        if streak >= STREAK_THRESHOLD:
            _advance_stage(step)
            _, k_now, phase_now = _current()
            if phase_now == "k" and k_now > k_max:
                # _advance_stage already logged ranks for this exact step
                ranks = model.report_ranks() if dynamic_rank_control else None
                if log_fn is not None:
                    log_fn(step, *_current()[:2], phase_now, "GRADUATED", loss_ema, acc_ema, ranks=ranks)
                break
        elif (wrong_streak >= WRONG_STREAK_THRESHOLD
              and queries_since_level_change >= MIN_QUERIES_BEFORE_REGRESS):
            _regress_stage(step)

        if step % log_every == 0:
            ranks = _log_ranks(step)
            steps_per_sec = step / (time.time() - t0)
            if log_fn is not None:
                log_fn(step, vocab_size, k, phase, "", loss_ema, acc_ema, ranks=ranks,
                      steps_per_sec=steps_per_sec)

    elapsed_s = time.time() - t0
    final_vocab, final_k, final_phase = _current()
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "steps_per_sec": (step / elapsed_s) if elapsed_s > 0 else 0.0,
        "peak_rss_mb": peak_rss_mb,
        "precision": precision, "final_vocab": final_vocab, "final_k": final_k,
        "final_phase": final_phase, "peak_stage": peak_stage,
        "graduated": final_phase == "k" and final_k > k_max, "total_steps": step,
        "elapsed_s": elapsed_s, "stage_history": stage_history,
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
    # additive_rank=1 + dynamic_rank_control=True is the validated AQRS
    # config (sili__new PR #38 / sili_peridot PR #16, JOURNAL.md): fp8
    # reached final/peak vocab=126,k=2 with the channel-orthogonality fix,
    # vs peak_vocab=64 for the plain (additive_rank=0) arm. The rank cap
    # itself is NOT set here -- DISLDOLayer's own constructor already
    # defaults scale_rank_max/additive_rank_max to
    # max(1, min(n_in,n_out)//4) (sili__new sparse_rnn.py
    # _default_rank_cap), which for these state_width=128 square layers
    # is exactly 32, matching the validated run without needing an
    # explicit override.
    additive_rank = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    dynamic_rank_control = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
    # AQRS rank-mutation cooldown (task #292 fix): interim "age-gate"
    # refractory period, not yet tied to a real resource/energy cost
    # model -- see sili__new delta_csr_types.hpp's
    # apply_dynamic_rank_control_generic docstring. Kept as a real
    # tunable per direct instruction, not hardcoded -- a real 60k-step
    # run showed the default (50) still allows frequent churn since 12
    # independent per-branch cooldowns (6 layers x 2 branches) all reset
    # on their own schedule; raise this for a calmer run.
    rank_grace_period_steps = int(sys.argv[9]) if len(sys.argv) > 9 else 50
    # Advantage actor-critic (task #272): curriculum leveling itself is
    # UNCHANGED (still the streak heuristic above) -- this only swaps how
    # the model's own per-step gradient is computed, see
    # _backward_with_critic's own docstring.
    use_critic = bool(int(sys.argv[10])) if len(sys.argv) > 10 else False
    # RNN validation ablation (direct instruction): when set, the model's
    # content-row (this step's readout) queries can only attend memory-row
    # keys/values, and the direct x_wide->content_out residual is zeroed --
    # see ToyTileRecurrenceRMT.step()'s own docstring for the full
    # rationale. If MQAR still learns SOMETHING under this restriction,
    # the recurrent state itself is doing real work, not just riding along
    # while content-content attention silently solves the task within a
    # single window.
    recurrent_only_output = bool(int(sys.argv[11])) if len(sys.argv) > 11 else False
    # Sparsity plan Phase 7 (task #336): real values, no bool -- embed_width
    # defaults to the module-level EMBED_WIDTH constant (today's exact
    # value, pass 32 directly to widen). input_sparsity_p/wide_max_weights
    # use -1 as this script's own "unset" sentinel (its argv convention is
    # plain positional args, not None-able flags) -- translated to Python
    # None below before being passed through, matching
    # ToyTileRecurrenceRMT's own None-means-unwidened convention.
    embed_width = int(sys.argv[12]) if len(sys.argv) > 12 else EMBED_WIDTH
    _input_sparsity_p_arg = float(sys.argv[13]) if len(sys.argv) > 13 else -1.0
    input_sparsity_p = _input_sparsity_p_arg if _input_sparsity_p_arg >= 0 else None
    _wide_max_weights_arg = int(sys.argv[14]) if len(sys.argv) > 14 else -1
    wide_max_weights = _wide_max_weights_arg if _wide_max_weights_arg >= 0 else None

    print(f"# MQAR curriculum precision={precision} max_steps={max_steps} seed={seed} "
          f"peak_lr={peak_lr} num_tiles={num_tiles} k_max={k_max} additive_rank={additive_rank} "
          f"dynamic_rank_control={dynamic_rank_control} rank_grace_period_steps={rank_grace_period_steps} "
          f"use_critic={use_critic} recurrent_only_output={recurrent_only_output} "
          f"embed_width={embed_width} input_sparsity_p={input_sparsity_p} wide_max_weights={wide_max_weights} "
          f"streak_threshold={STREAK_THRESHOLD} wrong_streak_threshold={WRONG_STREAK_THRESHOLD}",
          flush=True)

    _SHORT_NAME = {"input_proj": "in", "q_proj": "q", "k_proj": "k",
                   "v_proj": "v", "o_proj": "o", "lm_head": "lm"}

    def _ranks_str(ranks):
        if not ranks:
            return ""
        parts = [f"{_SHORT_NAME.get(n, n)}={s}/{a}" for n, (s, a) in ranks.items()]
        return "  ranks[" + " ".join(parts) + "]"

    def log_fn(step, vocab_size, k, phase, event, loss_ema, acc_ema, ranks=None, steps_per_sec=None):
        loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
        acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
        tag = f"  [{event}]" if event else ""
        sps_s = f"  steps/sec={steps_per_sec:.1f}" if steps_per_sec is not None else ""
        print(f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
              f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{_ranks_str(ranks)}", flush=True)

    r = train_curriculum(precision, max_steps, seed, peak_lr, num_tiles, k_max, log_fn=log_fn,
                         additive_rank=additive_rank, dynamic_rank_control=dynamic_rank_control,
                         rank_grace_period_steps=rank_grace_period_steps, use_critic=use_critic,
                         recurrent_only_output=recurrent_only_output,
                         embed_width=embed_width, input_sparsity_p=input_sparsity_p,
                         wide_max_weights=wide_max_weights)
    print(f"\nFINAL precision={precision} final_vocab={r['final_vocab']} final_k={r['final_k']} "
          f"final_phase={r['final_phase']} graduated={r['graduated']} "
          f"total_steps={r['total_steps']} steps_per_sec={r['steps_per_sec']:.1f} "
          f"peak_rss_mb={r['peak_rss_mb']:.1f} "
          f"({r['elapsed_s']:.0f}s)", flush=True)
    print(f"PEAK precision={precision} peak_vocab={r['peak_stage']['vocab']} "
          f"peak_k={r['peak_stage']['k']} peak_phase={r['peak_stage']['phase']}", flush=True)
    if r["dynamic_rank_control"]:
        print(f"RANK_MUTATIONS precision={precision} count={r['rank_mutation_count']}", flush=True)
        for name, (scale_r, add_r) in r["final_ranks"].items():
            print(f"  {name:<12} scale_rank={scale_r}  additive_rank={add_r}", flush=True)
        # Full per-step (well, per-log_every/per-transition) rank trace as
        # JSON, same convention as STAGE_HISTORY_JSON -- lets the growth/
        # shrink timing be read back and cross-referenced against
        # stage_history even if a run is killed early or takes far
        # longer than expected (direct instruction: log this to a file
        # as we go, not just report a final snapshot).
        print("RANK_HISTORY_JSON " + json.dumps(r["rank_history"]), flush=True)
    print("STAGE_HISTORY_JSON " + json.dumps(r["stage_history"]), flush=True)


if __name__ == "__main__":
    main()
