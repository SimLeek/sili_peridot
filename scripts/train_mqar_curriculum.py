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
  [dy_sparsity_p]
  precision: fp4 | fp8 | fp32
  embed_width: model width (sparsity plan Phase 6/7, task #335/#336) --
    default EMBED_WIDTH (16); pass 32 to widen input_proj/q/k/v/o_proj
  input_sparsity_p: density fraction (0..1) for those 5 layers' forward
    input; -1 (default) = unset/dense (today's exact behavior)
  wide_max_weights: per-layer synapse budget override for those same 5
    layers; -1 (default) = unset (shares max_weights with lm_head)
  dy_sparsity_p: density fraction (0..1) for those same 5 layers'
    backward gradient; -1 (default) = unset, meaning it matches
    input_sparsity_p. Independent axis -- set lower than
    input_sparsity_p to cut backward's real per-call cost (profiled as
    the dominant cost driver, not snapshot/merge overhead) at whatever
    quality tradeoff that implies.
  [use_tile_cache] [output_dy_sparsity_p] [wrong_streak_threshold]
  [dy_r_target] [dy_k_min] [dy_k_max] [target_steps_per_sec] [dy_surprise_alpha]
  dy_r_target: nucleus/energy-threshold captured-energy-ratio target
    (0..1) for those same 5 layers' backward gradient (task #367) --
    TAKES PRIORITY over dy_sparsity_p when both are set. k is a
    CONSEQUENCE of dy_r_target and each step's actual gradient energy,
    not a fixed fraction. -1 (default) = unset. See JOURNAL.md's
    "nucleus/energy-threshold top-k math" design note.
  dy_k_min/dy_k_max: hardware density floor/ceiling clamping the
    dy_r_target-derived k afterward. dy_k_min default 0 (no floor);
    dy_k_max -1 (default) = no ceiling.
  target_steps_per_sec: ARMS the closed-loop controller that adjusts
    dy_r_target every log_every steps against MEASURED steps/sec (task
    #368) -- NOT an analytic formula from an assumed compute-cost ratio
    (measured directly and found non-constant across widths, see
    JOURNAL.md's "Grad-side k_t design, revised" entry). -1 (default) =
    unset, dy_r_target stays fixed at its initial value for the whole run.
  dy_surprise_alpha: task #374, per-layer INNER loop -- breathes each
    wide layer's own EFFECTIVE dy_r_target above/below its r_bar based
    on that layer's own lagged gradient energy (E_t=||dy||^2 vs its
    running EMA Lbar, beta=0.99 fixed, not CLI-exposed). -1 (default) =
    unset, mechanism off, r_bar used unmodified (same as before this
    task). See ToyTileRecurrenceRMT.__init__'s own dy_surprise_alpha
    docstring for the full formula.
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
# fp32 does NOT get the same free pass as FP4: FP4/FP8's raw stored value is
# a quantization CODE with a small fixed table ceiling (FP4 ~6, FP8 E4M3
# ~448), which gives it an implicit per-step safety margin even under an
# uncapped max_abs_delta (see NOCAPS_KWARGS_FP8's comment above). fp32's
# raw stored value is a plain, genuinely unbounded float -- there is no
# implicit ceiling at all. Confirmed the hard way: a wide (3x embed_width),
# 10%-input-sparse/10%-grad-sparse fp32 MQAR run under NOCAPS_KWARGS blew
# up input_proj/v_proj to ~1e14-1e17 within 16k steps (overflow/invalid
# RuntimeWarnings, near-chance accuracy) while q/k/o_proj/lm_head -- same
# code, same NOCAPS -- stayed healthy; this is the simple hard-bound half
# of the fix (immediate safety net), the amortized L2 decay mechanism
# (apply_amortized_l2_decay) is the complementary ongoing-health half.
# Reuse the library's own already-tuned production default
# (kSynapsePolicyMaxAbsDelta/MaxCi, sili/cpu_backend.cpp) rather than
# guessing new numbers -- these are what every OTHER precision already
# runs under whenever synapse_kwargs isn't explicitly overridden.
NOCAPS_KWARGS_FP32 = {"max_abs_delta": 2.0, "max_ci": 100.0}
PRECISION_SYNAPSE_KWARGS = {"fp4": NOCAPS_KWARGS, "fp8": NOCAPS_KWARGS_FP8, "fp32": NOCAPS_KWARGS_FP32}

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


def k_indicator_token(k: int) -> int:
    """Reuses the real task-vocab token space directly (no dedicated
    reservation, direct instruction: "you can use the vocab token as the
    k token") -- token id `k` itself signals k, position in the prefix
    (2nd slot vs the vocab indicator's 1st) disambiguates it from a
    genuine in-sequence key/value that happens to share the same id.
    k is always well within [0, TASK_VOCAB_MAX), so this needs no new
    tokens and no vocab_size extension."""
    return k


def next_vocab(vocab_size: int, vocab_step: Optional[int] = None) -> int:
    if vocab_step is not None:
        return min(TASK_VOCAB_MAX, vocab_size + vocab_step)
    return min(TASK_VOCAB_MAX, int(round(vocab_size * VOCAB_GROWTH_FACTOR)))


def prev_vocab(vocab_size: int, vocab_step: Optional[int] = None) -> int:
    if vocab_step is not None:
        return max(VOCAB_START, vocab_size - vocab_step)
    return max(VOCAB_START, int(round(vocab_size / VOCAB_GROWTH_FACTOR)))


def _backward_with_critic(model, logits, target_token: int, row: int, aux) -> None:
    """Advantage-actor-critic backward for one query position (task #272).
    logits: [N, vocab_size] Tensor, query row given explicitly by the
    caller -- num_tiles-1 under model.step()'s full-window logits, 0
    under model.step_cached()'s single-row logits (see use_tile_cache).
    Replaces the plain unweighted cross-entropy backward with a critic-
    reweighted one -- see ADVANTAGE_CLIP's own comment for the formula.
    The critic itself trains via plain one-step MSE regression against
    the EXACT per-vocab squared error (known immediately since the
    target token is known this same step -- no TD/bootstrap/target-net
    needed, unlike sili__new's mandelbrot RTAC, which needs those because
    its reward isn't known until later)."""
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
    # "kcycle" (k_first_target odometer mode, direct instruction --
    # "v16k1 v16k2 v16k3 v18k1 v18k2 v18k3 v20k1... like it's an n-ary
    # number that's incrementing or decrementing"): vocab is the more
    # significant digit, k the less significant one, so ordering is
    # plain lexicographic (vocab, k) -- tags its own leading element (2)
    # so it never compares as equal/lesser against the other two phases'
    # tags in a mixed context.
    if stage["phase"] == "kcycle":
        return (2, stage["vocab"], stage["k"])
    return (1, stage["k"]) if stage["phase"] == "k" else (0, stage["vocab"])


def _ema_grad_scale(params, ratio_threshold: float, decay: float) -> None:
    """Per-tensor gradient-norm EMA scaling (direct instruction): purely
    a function of each tensor's OWN recent gradient-norm history, no step
    count or wall-clock anywhere in the formula -- required for a
    lifelong-learning setting where "step N" can't mean anything special.

    Root problem this targets (found via the sigma_grad_debug_fn/
    probe_sigma_trajectory investigation): input_ln/memory_ln/state_ln/
    centers/log_sigmas were orphaned by the old _to_sparse autograd bug,
    so at embed_width=32 they all take their first-ever real gradient
    step simultaneously once that bug is fixed -- a genuine cold start
    (their Adam moment estimates have never seen a real value) landing on
    a group that's structurally prone to hitting q/k/v/attn's hard clip
    (zero backward gradient past the boundary -- see magnitude_clip_
    penalty_coef's own docstring for the same failure mode it targets).
    clip_grad_norm_'s single GLOBAL norm across this whole param group
    means one exploding tensor (log_sigmas) drowns every other tensor's
    real update to a near-zero share of the shared budget -- this runs
    BEFORE that global clip and scales each tensor independently, so an
    outlier tensor doesn't cannibalize the others' learning signal.

    Each tensor tracks its own `_grad_norm_ema` attribute (state living
    on the Tensor object itself, updated every call -- not derived from
    step index). First real gradient for a tensor: no EMA exists yet, so
    it passes through unscaled and simply seeds the EMA. From the second
    real gradient onward: if the current raw norm exceeds
    `ratio_threshold` times that tensor's own EMA, the gradient is scaled
    down to exactly `ratio_threshold * ema` before anything else touches
    it; the EMA itself is always updated with the (possibly just-scaled)
    value, so a suppressed outlier doesn't inflate the reference either.
    """
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


def train_curriculum(precision: str, max_steps: int, seed: int, peak_lr: float,
                     num_tiles: int, k_max: int, log_every: int = 200,
                     log_fn=None, additive_rank: int = 1,
                     dynamic_rank_control: bool = True,
                     rank_grace_period_steps: int = 50,
                     rank_additive_grace_period_steps: int = 5000,
                     use_critic: bool = False,
                     magnitude_clip_penalty_coef: float = 0.0,
                     recurrent_only_output: bool = False,
                     embed_width: int = EMBED_WIDTH,
                     input_sparsity_p: Optional[float] = None,
                     wide_max_weights: Optional[int] = None,
                     dy_sparsity_p: Optional[float] = None,
                     use_tile_cache: bool = False,
                     output_dy_sparsity_p: Optional[float] = None,
                     wrong_streak_threshold: int = WRONG_STREAK_THRESHOLD,
                     streak_threshold: int = STREAK_THRESHOLD,
                     vocab_step: Optional[int] = None,
                     require_new_vocab_before_levelup: bool = False,
                     query_debug_fn=None,
                     sigma_grad_debug_fn=None,
                     clip_range: float = CLIP_RANGE,
                     grad_ema_ratio_threshold: Optional[float] = None,
                     grad_ema_decay: float = 0.9,
                     embed_table_builder=None,
                     embed_learning_rate: Optional[float] = None,
                     k_first_target: Optional[int] = None,
                     k_first_vocab: Optional[int] = None,
                     l2_decay_chunk_size: Optional[int] = None,
                     l2_decay_adaptation_rate: float = 0.3,
                     dy_r_target: Optional[float] = None,
                     dy_k_min: int = 0,
                     dy_k_max: Optional[int] = None,
                     dy_surprise_alpha: Optional[float] = None,
                     dy_surprise_beta: float = 0.99,
                     target_steps_per_sec: Optional[float] = None) -> dict:
    # query_debug_fn (direct instruction, explainable-AI investigation):
    # optional callback fired at EVERY query step (not just periodic log
    # points or LEVEL_UP/DOWN events) with (step, correct, logit_row,
    # model.last_debug) -- last_debug already exposes attn_mem/
    # attn_content/sigmas (task #303's NaN-bisection instrumentation),
    # letting a caller correlate attention sharpness/pattern with actual
    # per-query correctness without needing a separate hand-rolled probe
    # script that risks diverging from this loop's own validated task-
    # generation/labeling logic. None (default): zero overhead, no
    # behavior change for existing callers.
    # embed_width/input_sparsity_p/wide_max_weights (sparsity plan Phase
    # 7, task #336): real values threaded straight through to
    # ToyTileRecurrenceRMT's own identically-named constructor args (see
    # its own docstring for the full rationale) -- embed_width defaults
    # to the module-level EMBED_WIDTH constant (today's exact value,
    # unchanged), the other two default to None (today's exact
    # unwidened/dense behavior). COLUMN_NEURONS is NOT parameterized
    # here (stays the fixed numenta reference value, per direct
    # instruction) -- only embed_width varies.
    # dy_sparsity_p: independent axis from input_sparsity_p (backward's
    # weight-update work scales with how many (row, active-dy-column)
    # pairs survive the gradient-side zero-skip -- profiled directly,
    # see conversation, this is the real per-call cost driver, not
    # snapshot/merge overhead). Defaults to None, meaning
    # ToyTileRecurrenceRMT's own internal default (dy_sparsity_p ==
    # input_sparsity_p) applies unchanged -- explicit values here let a
    # caller push dy sparser than the input for a pure speed lever, at
    # whatever quality cost that turns out to have.
    # use_tile_cache (direct instruction): switches the inner loop from
    # rebuilding the full [num_tiles, embed_width] sliding window every
    # step (model.step(), full recompute of every tile in view) to
    # model.step_cached() -- one new token per call, K/V for the other
    # num_tiles-1 positions read from an explicit cache instead of
    # recomputed. Correct (bit-exact vs step()) whenever weights are
    # static between calls -- verified directly. The one real effect
    # once training starts: each backward call now gets gradient
    # through only the NEWEST content position into input_proj/q/k/v_
    # proj, not all num_tiles -- a genuinely different (sparser)
    # training signal per call, not a numerical approximation error.
    # Default False (today's exact behavior, zero change for existing
    # callers) until a real curriculum accuracy comparison confirms
    # this reduced signal doesn't cost quality.
    # k_first_target/k_first_vocab (direct instruction, new curriculum
    # ordering): reverses the default vocab-then-K phase order to K-then-
    # vocab. Rationale -- growing vocab first for tens of thousands of
    # steps means every synapse's "importance" (this project's real
    # optimizer state, see feedback_importance_is_already_the_optimizer
    # memory) gets reinforced for K=1-only patterns long before the model
    # is ever asked to do multi-key discrimination, by which point those
    # synapses resist the new K>1 feature the same way a CNN that never
    # saw horizontal lines during training would struggle to grow that
    # detector later with mostly-set weights. None (default): today's
    # exact vocab-first behavior, zero change for every existing caller.
    # k_first_vocab defaults to seq_len_for_k(k_first_target) + 4 (same
    # "+4 buffer over the minimum" margin VOCAB_START already uses over
    # seq_len_for_k(1)) if not given explicitly.
    if k_first_target is not None and k_first_vocab is None:
        k_first_vocab = seq_len_for_k(k_first_target) + 4

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
        dense=True, clip_range=clip_range, l1_sparsity_coef=L1_SPARSITY_COEF,
        synapse_kwargs=dict(PRECISION_SYNAPSE_KWARGS[precision]), scale_rank=1,
        additive_rank=additive_rank, dynamic_rank_control=dynamic_rank_control,
        use_critic=use_critic,
        magnitude_clip_penalty_coef=magnitude_clip_penalty_coef,
        recurrent_only_output=recurrent_only_output,
        input_sparsity_p=input_sparsity_p, wide_max_weights=wide_max_weights,
        dy_sparsity_p=dy_sparsity_p, output_dy_sparsity_p=output_dy_sparsity_p,
        dy_r_target=dy_r_target, dy_k_min=dy_k_min, dy_k_max=dy_k_max,
        dy_surprise_alpha=dy_surprise_alpha, dy_surprise_beta=dy_surprise_beta,
        rng=model_rng)
    opt = AdamOptimizer()
    # embed_table_builder (direct instruction, wide-model SDR redesign):
    # None (default) preserves today's exact dense-random-projection
    # embedding, zero change for every existing caller. When set, the
    # caller controls the embedding's own structure -- returns
    # (embed_table, active_mask), active_mask a same-shape bool array
    # marking each token's FIXED sparse-active positions (e.g. a genuine
    # sparse-distributed-representation table with a fixed k-of-width
    # active set per token), or None if the embedding has no fixed
    # sparsity structure to preserve under learning.
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
    # new-vocab-forced-sampling (direct instruction): keys are otherwise
    # drawn uniformly from the whole current key range, so a just-grown
    # vocab's newest token(s) can go untested for many queries by pure
    # chance, letting a level-up streak complete on old vocab alone.
    # new_key_ids = key ids introduced at the most recent vocab level-up
    # that still need at least 1 real test within every
    # force_new_vocab_every-query window (see main loop below).
    new_key_ids: list = []
    # streak_has_new_vocab (corrected design, direct instruction): gates
    # the LEVEL-UP itself, not a continuous per-3-queries tax on the whole
    # level. Tracks whether the CURRENT in-progress streak (the one
    # building toward streak_threshold) has already included a real test
    # of the newly-grown vocab. Only forces new vocab onto the query that
    # would otherwise complete the streak without ever having tested it --
    # "if (streak_threshold-1)/streak_threshold of a level-up is done and
    # new vocab hasn't shown up in the correct sequence, force it on THIS
    # query" -- so a level-up can never fire having never touched the new
    # material, without repeatedly taxing every re-entry into the level.
    streak_has_new_vocab = False
    # Diagnostic (direct instruction): LEVEL_UP needs STREAK_THRESHOLD
    # CONSECUTIVE correct queries, not just a high acc_ema -- tracks the
    # best streak actually reached between periodic log points, so a run
    # that's "close" (streak often hits 7-9 then breaks) reads differently
    # from one that's structurally capped much lower, even at matching
    # acc_ema.
    max_streak_seen = 0
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
        nonlocal new_key_ids, streak_has_new_vocab
        cur = stage_stack[-1]
        if cur["phase"] == "kcycle":
            # k_first_target odometer (direct instruction): k is the
            # least-significant digit, cycling K_START..k_first_target at
            # a fixed vocab; once it completes a cycle, vocab (the more
            # significant digit) advances one step and k resets to
            # K_START -- "v16k1 v16k2 v16k3 v18k1 v18k2 v18k3 v20k1...".
            # Exception: once vocab is already clamped at TASK_VOCAB_MAX
            # (next_vocab is a no-op there), wrapping k back to K_START
            # would spin forever at the top vocab tier without ever
            # terminating -- so instead k keeps climbing PAST
            # k_first_target, degenerating into the same "just grow k
            # forever" terminal behavior the default (vocab-first) run
            # already has once ITS vocab phase is exhausted.
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
        # No forcing needed on regression -- we're re-practicing a lower
        # level whose vocab was already tested on the way up.
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
        # Gate (not tax): only force new vocab onto THIS query if the
        # in-progress streak is one query away from completing a level-up
        # and hasn't tested the new vocab yet -- so the level-up can never
        # fire on old vocab alone, without repeatedly forcing new vocab on
        # every re-entry into the level (see streak_has_new_vocab comment
        # above for the full rationale, direct instruction).
        force_this_step = (require_new_vocab_before_levelup and phase == "vocab"
                            and len(new_key_ids) > 0
                            and streak == streak_threshold - 1
                            and not streak_has_new_vocab)
        tokens, mqar_pairs = generate_mqar_sequence(
            rng, vocab_size, seq_len, k,
            forced_keys=(new_key_ids if force_this_step else None))
        # Per-query-position new-vocab membership (not a step-wide OR):
        # each query's own key, so streak_has_new_vocab tracks whether
        # THIS SPECIFIC query touched the new vocab, not just any query
        # somewhere in the same step's window.
        new_vocab_query_positions = (
            set(pos for pos, _ in mqar_pairs if int(tokens[pos]) in new_key_ids)
            if new_key_ids else set())
        targets = _build_targets(tokens, mqar_pairs, k)
        query_positions = set(pos for pos, _ in mqar_pairs)

        # Persistent level-indicator prefix (direct instruction): every
        # sequence gets an explicit signal of (current vocab, current k),
        # not just the one-shot LEVEL_UP_TOKEN/LEVEL_DOWN_TOKEN on the
        # single sequence right after a transition -- otherwise the model
        # has to re-detect both from raw content (counting distinct
        # key-value pairs, inferring vocab range) on every other sequence
        # with zero hint. Two tokens, no spaces:
        #   [vocab indicator, k indicator]
        # BOTH indicators reuse the real task-vocab token space directly
        # (no new dedicated tokens, no vocab_size extension) -- direct
        # instruction, "you can use the vocab token as the k token".
        # vocab indicator = the newest/highest currently-unlocked token
        # (vocab_size-1); k indicator = token id k itself (k_indicator_
        # token, trivial passthrough -- see its own docstring). Position
        # alone (1st vs 2nd prefix slot) disambiguates either from the
        # same token ID appearing later as a genuine in-sequence value.
        level_prefix = [vocab_size - 1, k_indicator_token(k)]
        if pending_level_token is not None:
            level_prefix = [pending_level_token] + level_prefix
            pending_level_token = None
        offset = len(level_prefix)
        combined_tokens = np.concatenate((level_prefix, tokens))
        targets = {pos + offset: tgt for pos, tgt in targets.items()}
        query_positions = set(pos + offset for pos in query_positions)
        new_vocab_query_positions = set(pos + offset for pos in new_vocab_query_positions)

        memory = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
        tile_cache = None  # reset every sequence, same as memory
        for i in range(seq_len + offset):
            # requires_grad=False for non-query steps (direct instruction):
            # the sequential write-then-read design (see step()'s own
            # docstring) makes k_proj/v_proj/o_proj forward()-run TWICE per
            # step, and most steps never get a loss.backward() at all (only
            # `i in targets` does) -- building a backward graph for those is
            # both wasted work and, more importantly, would leave the C++
            # engine's DenseInputStack accumulating never-popped entries
            # across every non-query step in the sequence until it hits its
            # cap and throws.
            #
            # use_tile_cache query-step handling (direct instruction):
            # step_cached() alone only credits the NEWEST content position
            # per backward call (older cached positions are detached, zero
            # gradient) -- a real, measured stability concern (see
            # project_tile_window_kv_cache.md). On query steps specifically
            # (rare, this is where weights actually move), fall back to
            # step()'s full-window recompute instead, but with a GRADED
            # per-position dy_sparsity_schedule (denser for the newest
            # position, sparser further back) rather than either full
            # uniform density (expensive) or step_cached's zero-credit
            # default. Non-query steps (the majority) still use
            # step_cached()'s fast single-position path -- no backward
            # ever runs on those regardless, so there's no signal being
            # lost there either way.
            if use_tile_cache and (i in targets):
                window = _build_tile_window(embed_table, combined_tokens, i, num_tiles)
                memory, logits, aux = model.step(
                    window, memory, lr, requires_grad=True,
                    content_dy_sparsity_schedule=_default_graded_dy_schedule(num_tiles))
                logit_row = num_tiles - 1
                # Refresh tile_cache from this step's own freshly-recomputed
                # k/v (model.last_debug["k"/"v"] are the FULL [total_slots, sw]
                # pre-pass-2 arrays built at the top of step()) -- drop the
                # oldest content row (about to age out of the window on the
                # next call) and keep the rest, same append/trim convention
                # step_cached() itself uses.
                k_content = model.last_debug["k"][NUM_MEMORY_SLOTS:]
                v_content = model.last_debug["v"][NUM_MEMORY_SLOTS:]
                tile_cache = list(zip(k_content[1:].copy(), v_content[1:].copy()))
            elif use_tile_cache:
                new_embed = embed_table[combined_tokens[i]]
                memory, logits, aux, tile_cache = model.step_cached(
                    new_embed, memory, lr, tile_cache, requires_grad=False)
                logit_row = 0  # step_cached returns only the newest position's row
            else:
                window = _build_tile_window(embed_table, combined_tokens, i, num_tiles)
                memory, logits, aux = model.step(window, memory, lr, requires_grad=(i in targets))
                logit_row = num_tiles - 1
            if DEBUG_FINITE_CHECK and use_critic:
                _check_finite_or_raise(model, logits, step, i, loss_ema)
            if i in targets:
                loss = cross_entropy_sum(logits, [(logit_row, targets[i])])
                loss_ema = float(loss.data) if loss_ema is None else (
                    LOSS_EMA_DECAY * loss_ema + (1.0 - LOSS_EMA_DECAY) * float(loss.data))
                if i in query_positions:
                    correct = predicted_token(logits, logit_row) == targets[i]
                    acc_ema = float(correct) if acc_ema is None else (
                        ACC_EMA_DECAY * acc_ema + (1.0 - ACC_EMA_DECAY) * float(correct))
                    if query_debug_fn is not None:
                        # logits.data[logit_row]/targets[i] added (direct
                        # instruction): lets a caller compute real
                        # confidence/hedging signals (target-token
                        # probability, entropy) itself instead of only
                        # seeing the binary correct/incorrect outcome.
                        query_debug_fn(step, correct, logit_row, model.last_debug,
                                       logits.data[logit_row], targets[i])
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
                # embed_learning_rate (direct instruction, "add learning
                # too" on top of the fixed SDR structure): last_debug's
                # x_window_t Tensor is a leaf of THIS step's backward
                # graph, so its .grad (populated by the backward() call
                # just above) is dL/d(x_window) -- a plain scatter-add
                # embedding-lookup gradient, exactly like a real embedding
                # layer's backward. Only wired for the non-tile-cache
                # step() path (use_tile_cache's step_cached() builds its
                # window differently -- see its own docstring -- and isn't
                # covered here). active_mask (when set) restricts the
                # update to each token's fixed active positions, so
                # learning adjusts MAGNITUDES at the SDR's chosen indices
                # without ever growing new nonzero positions outside it.
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
                    _ema_grad_scale(model.parameters_for_optimizer(),
                                     grad_ema_ratio_threshold, grad_ema_decay)
                if sigma_grad_debug_fn is not None:
                    # Fired BEFORE clip_grad_norm_ -- raw gradient (after
                    # any per-tensor EMA scaling above, before the shared
                    # global-norm clip), matching what "is this gradient
                    # genuinely zero" / "did the EMA scale actually act on
                    # it" needs to see.
                    sigma_grad_debug_fn(step, model.log_sigmas.grad, model.centers.grad)
                clip_grad_norm_(model.parameters_for_optimizer(), MAX_GRAD_NORM)
                opt.step(model.parameters_for_optimizer(), lr=lr)
                # Amortized decoupled L2 decay + health stats (direct
                # instruction): the "ongoing health" complement to
                # NOCAPS_KWARGS's per-precision max_abs_delta/max_ci hard
                # bound above -- independent of dynamic_rank_control (this
                # decays raw synapse weights, not AQRS scale/additive
                # channels, so it applies identically whether or not rank
                # can mutate). None (default): zero overhead, unchanged
                # behavior for every existing caller.
                if l2_decay_chunk_size is not None:
                    model.apply_amortized_l2_decay(l2_decay_chunk_size, l2_decay_adaptation_rate)
                if dynamic_rank_control:
                    mutated = model.apply_dynamic_rank_control(
                        scale_grace_period_steps=rank_grace_period_steps,
                        additive_grace_period_steps=rank_additive_grace_period_steps)
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

        if streak >= streak_threshold:
            _advance_stage(step)
            vocab_now, k_now, phase_now = _current()
            # k_first_target (odometer) run: graduation mirrors the
            # default run's own terminal condition ("vocab already
            # maxed out AND k has grown past its own cap") -- here that's
            # vocab clamped at TASK_VOCAB_MAX (_advance_stage's own wrap
            # guard degenerates into pure k-growth once that happens) and
            # k having grown past k_first_target on top of that.
            graduated_now = ((phase_now == "kcycle" and vocab_now >= TASK_VOCAB_MAX
                             and k_now > k_first_target)
                             if k_first_target is not None
                             else (phase_now == "k" and k_now > k_max))
            if graduated_now:
                # _advance_stage already logged ranks for this exact step
                ranks = model.report_ranks() if dynamic_rank_control else None
                if log_fn is not None:
                    log_fn(step, *_current()[:2], phase_now, "GRADUATED", loss_ema, acc_ema, ranks=ranks)
                break
        elif (wrong_streak >= wrong_streak_threshold
              and queries_since_level_change >= MIN_QUERIES_BEFORE_REGRESS):
            _regress_stage(step)

        if step % log_every == 0:
            ranks = _log_ranks(step)
            steps_per_sec = step / (time.time() - t0)
            # dy_r_target closed-loop control (task #368): only fires
            # when dy_r_target was actually enabled (apply_amortized_dy_
            # r_target_control is itself a no-op otherwise, matching
            # apply_amortized_l2_decay's own "safe to always call"
            # convention). Reuses this loop's own cumulative steps_per_sec
            # measurement rather than a separate windowed timer.
            if target_steps_per_sec is not None:
                model.apply_amortized_dy_r_target_control(steps_per_sec, target_steps_per_sec)
            if log_fn is not None:
                log_fn(step, vocab_size, k, phase, "", loss_ema, acc_ema, ranks=ranks,
                      steps_per_sec=steps_per_sec, max_streak=max_streak_seen,
                      dy_r_target=model.dy_r_target)
            max_streak_seen = 0

    elapsed_s = time.time() - t0
    final_vocab, final_k, final_phase = _current()
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "steps_per_sec": (step / elapsed_s) if elapsed_s > 0 else 0.0,
        "peak_rss_mb": peak_rss_mb,
        "precision": precision, "final_vocab": final_vocab, "final_k": final_k,
        "final_phase": final_phase, "peak_stage": peak_stage,
        "graduated": ((final_phase == "kcycle" and final_vocab >= TASK_VOCAB_MAX
                       and final_k > k_first_target)
                      if k_first_target is not None
                      else (final_phase == "k" and final_k > k_max)),
        "total_steps": step,
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
    # dy_sparsity_p: same -1 sentinel convention. Left unset (-1), defaults
    # to matching input_sparsity_p (ToyTileRecurrenceRMT's own internal
    # default) -- an explicit value here overrides independently, e.g.
    # pushing dy sparser than the input as a pure backward-speed lever
    # (see conversation: backward's real per-call cost scales with how
    # many (row, active-dy-column) pairs survive this gate, not with
    # snapshot/merge overhead).
    _dy_sparsity_p_arg = float(sys.argv[15]) if len(sys.argv) > 15 else -1.0
    dy_sparsity_p = _dy_sparsity_p_arg if _dy_sparsity_p_arg >= 0 else None
    # step_cached()+graded-schedule (project_tile_window_kv_cache): was only
    # reachable as a train_curriculum() kwarg, not from the CLI, blocking the
    # real long-run (2k-20k+ step) validation this design still needs before
    # merge. Off by default -- zero behavior change unless explicitly set.
    use_tile_cache = bool(int(sys.argv[16])) if len(sys.argv) > 16 else False
    # output_dy_sparsity_p (direct instruction): lm_head/critic_head's own
    # backward-gradient density -- same -1 "unset" sentinel convention as
    # input_sparsity_p/dy_sparsity_p above (this script's argv is plain
    # positional, not None-able flags).
    _output_dy_sparsity_p_arg = float(sys.argv[17]) if len(sys.argv) > 17 else -1.0
    output_dy_sparsity_p = _output_dy_sparsity_p_arg if _output_dy_sparsity_p_arg >= 0 else None
    # wrong_streak_threshold (direct instruction, reward-hacking concern):
    # the default WRONG_STREAK_THRESHOLD=5 vs STREAK_THRESHOLD=10 asymmetry
    # means a model can reach the (easier, lower-loss) previous vocab stage
    # via only 5 consecutive wrong answers, but needs 10 consecutive right
    # answers to leave it -- a real diagnostic run (embed_width=32, see
    # JOURNAL.md) showed max_streak plateaued hard at 7/10 for 2000+ steps,
    # never crossing to level up, which is consistent with the model
    # finding it cheaper (lower average loss) to hover near the boundary
    # and get bounced back to easy data than to commit to genuinely harder
    # representations. -1 sentinel keeps today's default (5) unchanged.
    _wrong_streak_threshold_arg = int(sys.argv[18]) if len(sys.argv) > 18 else -1
    wrong_streak_threshold = (_wrong_streak_threshold_arg if _wrong_streak_threshold_arg >= 0
                              else WRONG_STREAK_THRESHOLD)
    # dy_r_target/dy_k_min/dy_k_max/target_steps_per_sec (task #367/#368,
    # nucleus/energy-threshold grad sparsification -- see JOURNAL.md's
    # "nucleus/energy-threshold top-k math" + "Grad-side k_t design,
    # revised" entries): same -1/"unset" sentinel convention as every
    # other sparsity arg above. dy_r_target takes priority over
    # dy_sparsity_p at the model level when both are set. dy_k_min/
    # dy_k_max default to 0/unset (no clamp). target_steps_per_sec is
    # what actually ARMS the closed-loop controller each log_every
    # window -- dy_r_target alone just sets the (then-fixed) initial
    # value; leaving target_steps_per_sec at -1 keeps dy_r_target static
    # for the whole run instead of adapting.
    _dy_r_target_arg = float(sys.argv[19]) if len(sys.argv) > 19 else -1.0
    dy_r_target = _dy_r_target_arg if _dy_r_target_arg >= 0 else None
    dy_k_min = int(sys.argv[20]) if len(sys.argv) > 20 else 0
    _dy_k_max_arg = int(sys.argv[21]) if len(sys.argv) > 21 else -1
    dy_k_max = _dy_k_max_arg if _dy_k_max_arg >= 0 else None
    _target_sps_arg = float(sys.argv[22]) if len(sys.argv) > 22 else -1.0
    target_steps_per_sec = _target_sps_arg if _target_sps_arg >= 0 else None
    # dy_surprise_alpha (task #374, per-layer lagged E_t/Lbar inner
    # loop): same -1/"unset" sentinel convention. dy_surprise_beta
    # (EMA decay) stays at its 0.99 default -- not exposed via CLI, an
    # unvalidated mechanism doesn't need every sub-parameter tunable
    # from the command line yet (direct-instruction pattern from this
    # same session: don't over-parameterize unproven complexity).
    _dy_surprise_alpha_arg = float(sys.argv[23]) if len(sys.argv) > 23 else -1.0
    dy_surprise_alpha = _dy_surprise_alpha_arg if _dy_surprise_alpha_arg >= 0 else None

    print(f"# MQAR curriculum precision={precision} max_steps={max_steps} seed={seed} "
          f"peak_lr={peak_lr} num_tiles={num_tiles} k_max={k_max} additive_rank={additive_rank} "
          f"dynamic_rank_control={dynamic_rank_control} rank_grace_period_steps={rank_grace_period_steps} "
          f"use_critic={use_critic} recurrent_only_output={recurrent_only_output} "
          f"embed_width={embed_width} input_sparsity_p={input_sparsity_p} wide_max_weights={wide_max_weights} "
          f"dy_sparsity_p={dy_sparsity_p} use_tile_cache={use_tile_cache} "
          f"output_dy_sparsity_p={output_dy_sparsity_p} "
          f"streak_threshold={STREAK_THRESHOLD} wrong_streak_threshold={wrong_streak_threshold} "
          f"dy_r_target={dy_r_target} dy_k_min={dy_k_min} dy_k_max={dy_k_max} "
          f"target_steps_per_sec={target_steps_per_sec} dy_surprise_alpha={dy_surprise_alpha}",
          flush=True)

    _SHORT_NAME = {"input_proj": "in", "q_proj": "q", "k_proj": "k",
                   "v_proj": "v", "o_proj": "o", "lm_head": "lm"}

    def _ranks_str(ranks):
        if not ranks:
            return ""
        parts = [f"{_SHORT_NAME.get(n, n)}={s}/{a}" for n, (s, a) in ranks.items()]
        return "  ranks[" + " ".join(parts) + "]"

    def log_fn(step, vocab_size, k, phase, event, loss_ema, acc_ema, ranks=None, steps_per_sec=None,
               max_streak=None, dy_r_target=None):
        loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
        acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
        tag = f"  [{event}]" if event else ""
        sps_s = f"  steps/sec={steps_per_sec:.1f}" if steps_per_sec is not None else ""
        streak_s = f"  max_streak={max_streak:>2}/{STREAK_THRESHOLD}" if max_streak is not None else ""
        # dy_r_target (task #368, per-layer dict since task #372): only
        # printed once the mechanism is actually enabled on at least one
        # wide layer -- empty/all-None stays silent, same opt-in-visible
        # convention as sps_s/streak_s above. Compact per-layer format
        # since layers can diverge independently once task #374 lands
        # (today they still move in lockstep, but the log format doesn't
        # assume that).
        if dy_r_target:
            _set = {n: v for n, v in dy_r_target.items() if v is not None}
            dy_r_s = ("  dy_r_target[" +
                      " ".join(f"{_SHORT_NAME.get(n, n)}={v:.3f}" for n, v in _set.items()) +
                      "]") if _set else ""
        else:
            dy_r_s = ""
        print(f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
              f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{streak_s}{dy_r_s}{_ranks_str(ranks)}", flush=True)

    r = train_curriculum(precision, max_steps, seed, peak_lr, num_tiles, k_max, log_fn=log_fn,
                         additive_rank=additive_rank, dynamic_rank_control=dynamic_rank_control,
                         rank_grace_period_steps=rank_grace_period_steps, use_critic=use_critic,
                         recurrent_only_output=recurrent_only_output,
                         embed_width=embed_width, input_sparsity_p=input_sparsity_p,
                         wide_max_weights=wide_max_weights, dy_sparsity_p=dy_sparsity_p,
                         use_tile_cache=use_tile_cache,
                         output_dy_sparsity_p=output_dy_sparsity_p,
                         wrong_streak_threshold=wrong_streak_threshold,
                         dy_r_target=dy_r_target, dy_k_min=dy_k_min, dy_k_max=dy_k_max,
                         dy_surprise_alpha=dy_surprise_alpha,
                         target_steps_per_sec=target_steps_per_sec)
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
