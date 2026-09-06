"""
sili_peridot/model/eval_lr.py: golden-section search over log(lr) space
for a domain-agnostic trial_fn(lr, seed) -> score, replacing a manual
geometric lr sweep. See docs/research/eval_lr.rst:eval_lr.module_overview,
eval_lr.unimodal_assumption_and_multistart, and
eval_lr.anytime_time_budgeted_design.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# Golden ratio conjugate; reused for both bracketing and refinement. See
# docs/research/eval_lr.rst:eval_lr.golden_ratio_constant_reuse.
_GOLD = (math.sqrt(5) - 1) / 2  # ~0.618
_GOLD_EXPAND = 1 / _GOLD  # ~1.618, growth factor while bracketing


TrialFn = Callable[[float, int], float]  # (lr, seed) -> score, higher is better


@dataclass
class LRSearchResult:
    best_lr: float
    best_score: float
    n_trials: int
    elapsed_s: float
    stopped_reason: str  # "converged", "time_budget", "max_bracket_steps"
    # See docs/research/eval_lr.rst:eval_lr.anytime_time_budgeted_design.
    history: list[tuple[float, float]] = field(default_factory=list)


def find_optimal_lr(
    trial_fn: TrialFn,
    *,
    seeds: Sequence[int] = (1000, 1001),
    initial_lr: float = 1e-3,
    time_budget_s: float = 300.0,
    max_bracket_steps: int = 20,
    log_tol: float = 0.05,
) -> LRSearchResult:
    """Find the lr maximizing trial_fn's seed-averaged score via geometric
    bracketing then golden-section refinement in log(lr) space, sharing
    one time-budget deadline and one evaluation cache across both phases.
    See docs/research/eval_lr.rst:eval_lr.find_optimal_lr_two_phase_algorithm."""
    t_start = time.time()
    seeds = list(seeds)
    history: list[tuple[float, float]] = []
    cache: dict = {}  # rounded log(lr) -> score, avoids re-running an identical trial
    best_lr, best_score = initial_lr, float("-inf")

    def deadline_hit() -> bool:
        return (time.time() - t_start) >= time_budget_s

    def evaluate(log_lr: float) -> float:
        nonlocal best_lr, best_score
        key = round(log_lr, 9)
        if key in cache:
            return cache[key]
        lr = math.exp(log_lr)
        score = statistics.mean(trial_fn(lr, seed) for seed in seeds)
        cache[key] = score
        history.append((log_lr, score))
        if score > best_score:
            best_score, best_lr = score, lr
        return score

    # Phase 1: geometric bracketing (Numerical Recipes' mnbrak, adapted for maximization in log-space).
    log_a = math.log(initial_lr)
    f_a = evaluate(log_a)
    stopped_reason = "max_bracket_steps"
    if deadline_hit():
        return LRSearchResult(best_lr, best_score, len(history), time.time() - t_start, "time_budget", history)

    log_b = log_a + math.log(_GOLD_EXPAND)
    f_b = evaluate(log_b)
    if f_b < f_a:
        # Growing hurt immediately -- reverse direction. See
        # docs/research/eval_lr.rst:eval_lr.bracket_reversal_direction_fix.
        log_a, log_b = log_b, log_a
        f_a, f_b = f_b, f_a
        step = log_b - log_a  # negative-going from here
    else:
        step = log_b - log_a

    log_c, f_c = log_b, f_b
    for _ in range(max_bracket_steps):
        if deadline_hit():
            stopped_reason = "time_budget"
            break
        step *= _GOLD_EXPAND
        log_c = log_b + step
        f_c = evaluate(log_c)
        if f_c < f_b:
            stopped_reason = "converged"  # found a real bracket
            break
        log_a, f_a = log_b, f_b
        log_b, f_b = log_c, f_c
    # Normalize so lo < hi regardless of which direction we expanded in.
    lo, _mid, hi = sorted([(log_a, f_a), (log_b, f_b), (log_c, f_c)], key=lambda t: t[0])
    log_lo, log_hi = lo[0], hi[0]

    # Phase 2: golden-section refinement.
    if not deadline_hit() and stopped_reason != "time_budget":
        b1 = log_hi - _GOLD * (log_hi - log_lo)
        b2 = log_lo + _GOLD * (log_hi - log_lo)
        f1, f2 = evaluate(b1), evaluate(b2)
        while (log_hi - log_lo) > log_tol:
            if deadline_hit():
                stopped_reason = "time_budget"
                break
            if f1 > f2:
                log_hi = b2
                b2, f2 = b1, f1
                b1 = log_hi - _GOLD * (log_hi - log_lo)
                f1 = evaluate(b1)
            else:
                log_lo = b1
                b1, f1 = b2, f2
                b2 = log_lo + _GOLD * (log_hi - log_lo)
                f2 = evaluate(b2)
        else:
            stopped_reason = "converged"

    return LRSearchResult(best_lr, best_score, len(history), time.time() - t_start, stopped_reason, history)
