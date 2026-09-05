"""See docs/research/toy_beyond_context_task.rst:module_overview."""

from __future__ import annotations

import numpy as np

QUERY_TOKEN = 2
VOCAB_SIZE = 3


def generate_parity_sequence(rng: np.random.RandomState, n_bits: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Returns (tokens [n_bits+2], pairs); pairs directly usable with
    cross_entropy_sum/predicted_token's own (row, target) convention."""
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


# See docs/research/toy_beyond_context_task.rst:deviation_sequence_design.
DEVIATION_BASE_PATTERN = np.array([1, 0, 1], dtype=np.int64)


def generate_deviation_sequence(
    rng: np.random.RandomState, n_positions: int, deviation_prob: float = 0.5
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """See docs/research/toy_beyond_context_task.rst:deviation_sequence_design."""
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
