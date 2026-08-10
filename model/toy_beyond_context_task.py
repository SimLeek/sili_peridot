"""
sili_peridot/model/toy_beyond_context_task.py
─────────────────────────────────────────────────
Tier 1 of the out-of-context benchmark suite -- see the approved plan
(fuzzy-plotting-starlight.md) for the full three-tier design and
rationale: tile-recurrence is strictly more general than a
bounded-window/bounded-depth transformer (it can carry information in
persistent recurrent state across unbounded time); this task is
deliberately the simplest possible genuine sequential computation (a
single XOR accumulator) so failure past the window is clearly a
WINDOW-VISIBILITY limitation, not a capacity or task-difficulty
confound.

`generate_parity_sequence`: `n_bits` random bits followed by a `'?'`
query token followed by the correct running-parity answer bit -- the
answer is itself the real next token after `'?'`, not a bolted-on
label (ordinary next-token prediction). vocab_size=3 (`0`, `1`, `'?'`
-> token ids 0, 1, 2) gives a genuine way to signal "answer now" vs.
mid-sequence, unlike an earlier 2-token design that had no way to
distinguish those (direct correction).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

QUERY_TOKEN = 2
VOCAB_SIZE = 3


def generate_parity_sequence(rng: np.random.RandomState, n_bits: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Returns (tokens [n_bits+2], pairs) -- tokens = n_bits random
    bits, then QUERY_TOKEN, then the running-parity answer bit (XOR of
    all n_bits). pairs = [(query_position, answer_bit)], directly
    usable with cross_entropy_sum/predicted_token's own (row, target)
    convention."""
    if n_bits < 1:
        raise ValueError(f"n_bits must be >= 1, got {n_bits}")

    bits = rng.randint(0, 2, size=n_bits).astype(np.int64)
    answer = int(np.bitwise_xor.reduce(bits))
    query_pos = n_bits

    tokens = np.empty(n_bits + 2, dtype=np.int64)
    tokens[:n_bits] = bits
    tokens[query_pos] = QUERY_TOKEN
    tokens[query_pos + 1] = answer

    return tokens, [(query_pos, answer)]


# Fixed, shared across EVERY sequence -- the network learns this ONE
# repeating motif once, rather than re-deriving a fresh per-sequence
# pattern (see generate_deviation_sequence's own docstring for why).
DEVIATION_BASE_PATTERN = np.array([1, 0, 1], dtype=np.int64)


def generate_deviation_sequence(rng: np.random.RandomState, n_positions: int,
                                deviation_prob: float = 0.5) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Dense repeating base pattern (DEVIATION_BASE_PATTERN, fixed
    across all sequences) with at most ONE sparse deviation (a single
    flipped bit) inserted at a random position. Answer = 1 if any
    deviation occurred, else 0 -- deviation_prob tuned to ~0.5 so a
    constant "always predict no-deviation" strategy can't trivially
    win the way it could if deviations were rare (direct correction:
    parity's XOR answer is already balanced, but the SIGNAL isn't --
    every single bit has to be tracked correctly for XOR to come out
    right, diluting any one credit-assignment improvement across all
    the OTHER bits that also need to be correct; here the answer
    depends on exactly ONE bit's worth of remembered state, a cleaner
    isolation of "does the credited tick's influence survive until the
    query" specifically).

    `n_positions` plays the same curriculum role as generate_parity_sequence's
    `n_bits` -- growing it past the visible window is what makes the
    single deviation (when present) sometimes fall OUTSIDE context,
    forcing reliance on carried state rather than direct visibility."""
    base_len = len(DEVIATION_BASE_PATTERN)
    body = np.array([DEVIATION_BASE_PATTERN[i % base_len] for i in range(n_positions)], dtype=np.int64)
    has_deviation = rng.random() < deviation_prob
    if has_deviation:
        pos = rng.randint(0, n_positions)
        body[pos] = 1 - body[pos]
    answer = int(has_deviation)
    query_pos = n_positions

    tokens = np.empty(n_positions + 2, dtype=np.int64)
    tokens[:n_positions] = body
    tokens[query_pos] = QUERY_TOKEN
    tokens[query_pos + 1] = answer

    return tokens, [(query_pos, answer)]
