"""
sili_peridot/model/eval_lr.py
──────────────────────────────
Fast, generic optimal-learning-rate search for sili_peridot's models.
Motivated directly by a real regression hunt (conversation): sili__new's
importance-signal contrib addition shrank baseline's effective step size,
and a manual geometric sweep (1x, 1.41x, 2x, ..., 100x the default lr) on a
short (1500-step) run found a clean single-peaked curve -- eval_acc rose
0.24 -> 0.56 by ~10x, then fell back toward chance by 100x. That sweep is
the algorithm this module automates: golden-section search over log(lr)
space, not linear -- lr's useful range spans orders of magnitude and it's
MULTIPLICATIVE steps that matter, confirmed directly by that sweep's own
shape (not, say, some fixed additive offset from the default).

Assumes the score-vs-log(lr) curve is UNIMODAL (single global optimum) --
true for the sweep that motivated this module, not guaranteed for every
model/problem. If a real curve has multiple local optima, this finds A
peak (the one geometric bracketing first walks into from initial_lr), not
necessarily the global best. Multi-start (varying initial_lr) is the
straightforward extension if that ever turns out to matter -- not built
here since nothing so far has needed it.

Anytime / time-budgeted: runs trials against a wall-clock deadline,
tracking the best (lr, score) seen at every point -- if the deadline hits
mid-bracket or mid-search, returns whatever's currently best rather than
raising or needing to finish. "Runs continuously, returns the optimal
learning rate when stopped" (conversation), not a fixed trial count.

Domain-agnostic core: find_optimal_lr() knows nothing about tile-
recurrence, OriginalArchModel, or any specific training loop -- callers
supply trial_fn(lr, seed) -> score (build a fresh model at that lr, run a
short train, return a scalar to MAXIMIZE). See tests/test_eval_lr.py for
a worked adapter around scripts/l1_sparsity_probe.py's OriginalArchModel/
run/evaluate, which is what this module replaces the manual version of.
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

# Golden ratio conjugate (~0.618) -- both the expansion factor used while
# bracketing and the interval-split ratio used while refining. Using the
# SAME constant for both phases is standard (Numerical Recipes' mnbrak/
# golden), not a coincidence: it's the ratio that makes golden-section
# search reuse one already-evaluated interior point per iteration instead
# of re-evaluating both ends every time -- the entire reason to prefer it
# over plain bisection here, since every "evaluation" is a real (if short)
# training run, not a cheap function call.
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
    # (log(lr), mean_score) for every DISTINCT lr actually evaluated, in
    # the order tried -- lets a caller plot/inspect the search trajectory,
    # not just the final answer.
    history: List[Tuple[float, float]] = field(default_factory=list)


def find_optimal_lr(
    trial_fn: TrialFn,
    *,
    seeds: Sequence[int] = (1000, 1001),
    initial_lr: float = 1e-3,
    time_budget_s: float = 300.0,
    max_bracket_steps: int = 20,
    log_tol: float = 0.05,
) -> LRSearchResult:
    """Find the lr maximizing trial_fn's seed-averaged score.

    Phase 1 (bracket): starting from initial_lr, expand geometrically in
    whichever direction improves the score until it stops improving --
    gives a triple (lo, mid, hi) in log(lr) space with mid's score >=
    both neighbors', a valid bracket for phase 2.

    Phase 2 (golden-section): repeatedly shrinks that bracket, one new
    evaluation per iteration, until its width in log-space is under
    log_tol (default 0.05 ~= within a factor of e^0.05 ~= 1.05x -- close
    enough for a "peak lr" answer that a follow-up full-scale run would
    round anyway) or time runs out.

    Both phases share one time-budget deadline and one cache (so the
    bracket phase's last two points feed directly into golden-section
    without re-evaluating either).
    """
    t_start = time.time()
    seeds = list(seeds)
    history: List[Tuple[float, float]] = []
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

    # ── Phase 1: geometric bracketing (Numerical Recipes' mnbrak, adapted
    # for maximization in log-space) ────────────────────────────────────
    log_a = math.log(initial_lr)
    f_a = evaluate(log_a)
    stopped_reason = "max_bracket_steps"
    if deadline_hit():
        return LRSearchResult(best_lr, best_score, len(history),
                               time.time() - t_start, "time_budget", history)

    log_b = log_a + math.log(_GOLD_EXPAND)
    f_b = evaluate(log_b)
    if f_b < f_a:
        # Growing hurt immediately -- reverse direction (shrink instead).
        # After the swap, log_b holds the original (better) initial_lr
        # point and log_a holds the worse (grown) one -- continuing in
        # the SAME direction we were already heading (worse -> better)
        # means stepping from log_b AWAY from log_a, i.e. log_b - log_a,
        # which is negative here (shrinking lr further). Using log_a -
        # log_b instead (the bug this replaces) pointed back at the
        # already-rejected point and made the whole bracket phase
        # collapse back to initial_lr immediately.
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
    lo, mid, hi = sorted([(log_a, f_a), (log_b, f_b), (log_c, f_c)], key=lambda t: t[0])
    log_lo, log_hi = lo[0], hi[0]

    # ── Phase 2: golden-section refinement ──────────────────────────────
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

    return LRSearchResult(best_lr, best_score, len(history),
                           time.time() - t_start, stopped_reason, history)
