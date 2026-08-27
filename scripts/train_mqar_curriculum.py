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
  precision: fp4 | fp8 | fp32
"""
from __future__ import annotations

import sys
import time
import json

import numpy as np

sys.path.insert(0, ".")

from sili.sparse_rnn import DISLDOLayer, DISLDOLayer8, DISLDOLayer32
from sili import _cpu
from model.toy_recall_task import generate_mqar_sequence
from model.toy_recall_models import cross_entropy_sum, predicted_token, AdamOptimizer, clip_grad_norm_
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT
from scripts.train_tile_curriculum import _build_tile_window
from scripts.train_mqar_rmt_reference import (
    seq_len_for_k, _build_targets,
    EMBED_WIDTH, COLUMN_NEURONS, NUM_MEMORY_SLOTS, MAX_WEIGHTS_PER_LAYER,
    NUM_CPUS, VOCAB, WARMUP_STEPS, MAX_GRAD_NORM, CLIP_RANGE, L1_SPARSITY_COEF,
)

PRECISION_CLS = {"fp4": DISLDOLayer, "fp8": DISLDOLayer8, "fp32": DISLDOLayer32}
NOCAPS_KWARGS = {"max_abs_delta": 1e30, "max_ci": 1e30}

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


def next_vocab(vocab_size: int) -> int:
    return min(TASK_VOCAB_MAX, int(round(vocab_size * VOCAB_GROWTH_FACTOR)))


def prev_vocab(vocab_size: int) -> int:
    return max(VOCAB_START, int(round(vocab_size / VOCAB_GROWTH_FACTOR)))


def _stage_key(stage: dict) -> tuple:
    # total order matching how stages are actually visited: every
    # k-phase stage is harder than every vocab-phase stage.
    return (1, stage["k"]) if stage["phase"] == "k" else (0, stage["vocab"])


def train_curriculum(precision: str, max_steps: int, seed: int, peak_lr: float,
                     num_tiles: int, k_max: int, log_every: int = 200,
                     log_fn=None, additive_rank: int = 0,
                     dynamic_rank_control: bool = False,
                     rank_grace_period_steps: int = 50) -> dict:
    disldo_cls = PRECISION_CLS[precision]
    state_width = EMBED_WIDTH * COLUMN_NEURONS

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    model_rng = np.random.default_rng(seed)

    # num_tiles is fixed at construction and NEVER changes as the
    # curriculum's vocab/K stage grows -- this is the whole point of
    # decoupling the model's local window from the task.
    model = ToyTileRecurrenceRMT(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, num_tiles, NUM_MEMORY_SLOTS,
        MAX_WEIGHTS_PER_LAYER, num_cpus=NUM_CPUS, disldo_cls=disldo_cls,
        dense=True, clip_range=CLIP_RANGE, l1_sparsity_coef=L1_SPARSITY_COEF,
        synapse_kwargs=dict(NOCAPS_KWARGS), scale_rank=1,
        additive_rank=additive_rank, dynamic_rank_control=dynamic_rank_control,
        rng=model_rng)
    opt = AdamOptimizer()
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

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
            memory, logits, aux = model.step(window, memory, lr)
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
                if aux is not None:
                    loss = loss + aux
                loss.backward()
                clip_grad_norm_(model.parameters_for_optimizer(), MAX_GRAD_NORM)
                opt.step(model.parameters_for_optimizer(), lr=lr)
                if dynamic_rank_control:
                    mutated = model.apply_dynamic_rank_control(grace_period_steps=rank_grace_period_steps)
                    rank_mutation_count += sum(1 for m in mutated.values() if m)
                    # AQRS scale/additive channel numerical-safety pass
                    # (task #295 follow-up): only relevant once rank can
                    # genuinely exceed the old hardcoded cap=4, i.e. only
                    # under dynamic_rank_control -- see conversation for
                    # the real fp8 NaN collapse this fixes (get_scale()'s
                    # combined envelope has no clamp, and rank growing
                    # past 4 let it overflow in a real curriculum run).
                    model.apply_scale_overflow_guard()
                    # AQRS channel-diversity pass (task #295 follow-up,
                    # chosen over residual-targeted growth -- direct
                    # instruction): stops rank channels converging to
                    # duplicate directions during training, which
                    # nothing else here catches (neurogenesis's own
                    # health check is magnitude-only; l1_sparsity_coef
                    # only sees the summed output). Same "only relevant
                    # once rank>1" gating as the overflow guard above.
                    model.apply_channel_orthogonality_penalty()

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
            if log_fn is not None:
                log_fn(step, vocab_size, k, phase, "", loss_ema, acc_ema, ranks=ranks)

    final_vocab, final_k, final_phase = _current()
    return {
        "precision": precision, "final_vocab": final_vocab, "final_k": final_k,
        "final_phase": final_phase, "peak_stage": peak_stage,
        "graduated": final_phase == "k" and final_k > k_max, "total_steps": step,
        "elapsed_s": time.time() - t0, "stage_history": stage_history,
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
    additive_rank = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    dynamic_rank_control = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False
    # AQRS rank-mutation cooldown (task #292 fix): interim "age-gate"
    # refractory period, not yet tied to a real resource/energy cost
    # model -- see sili__new delta_csr_types.hpp's
    # apply_dynamic_rank_control_generic docstring. Kept as a real
    # tunable per direct instruction, not hardcoded -- a real 60k-step
    # run showed the default (50) still allows frequent churn since 12
    # independent per-branch cooldowns (6 layers x 2 branches) all reset
    # on their own schedule; raise this for a calmer run.
    rank_grace_period_steps = int(sys.argv[9]) if len(sys.argv) > 9 else 50

    print(f"# MQAR curriculum precision={precision} max_steps={max_steps} seed={seed} "
          f"peak_lr={peak_lr} num_tiles={num_tiles} k_max={k_max} additive_rank={additive_rank} "
          f"dynamic_rank_control={dynamic_rank_control} rank_grace_period_steps={rank_grace_period_steps} "
          f"streak_threshold={STREAK_THRESHOLD} wrong_streak_threshold={WRONG_STREAK_THRESHOLD}",
          flush=True)

    _SHORT_NAME = {"input_proj": "in", "q_proj": "q", "k_proj": "k",
                   "v_proj": "v", "o_proj": "o", "lm_head": "lm"}

    def _ranks_str(ranks):
        if not ranks:
            return ""
        parts = [f"{_SHORT_NAME.get(n, n)}={s}/{a}" for n, (s, a) in ranks.items()]
        return "  ranks[" + " ".join(parts) + "]"

    def log_fn(step, vocab_size, k, phase, event, loss_ema, acc_ema, ranks=None):
        loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
        acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
        tag = f"  [{event}]" if event else ""
        print(f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
              f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{_ranks_str(ranks)}", flush=True)

    r = train_curriculum(precision, max_steps, seed, peak_lr, num_tiles, k_max, log_fn=log_fn,
                         additive_rank=additive_rank, dynamic_rank_control=dynamic_rank_control,
                         rank_grace_period_steps=rank_grace_period_steps)
    print(f"\nFINAL precision={precision} final_vocab={r['final_vocab']} final_k={r['final_k']} "
          f"final_phase={r['final_phase']} graduated={r['graduated']} "
          f"total_steps={r['total_steps']} ({r['elapsed_s']:.0f}s)", flush=True)
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
