"""
sili_peridot/model/toy_recall_task.py
───────────────────────────────────────
Synthetic associative-recall ("induction head") task -- see the
approved plan (fuzzy-plotting-starlight.md) for why this specific task:
it directly tests genuine long-range, content-based retrieval, the
property tile-recurrence exists to have more of than the old
single-carried-vector window mechanism.

One cue-response bigram (A, B) is planted early in an otherwise random
token sequence; A is planted again `lag` positions later, and the
well-defined recall target at that second occurrence is B (also
planted as the following real token, so this is an ordinary "predict
the next real token" target, not a synthetic label bolted on
separately).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def generate_sequence(
    rng: np.random.RandomState,
    vocab_size: int,
    seq_len: int,
    lag: int,
) -> Tuple[np.ndarray, int]:
    """Returns (tokens [seq_len] int, induction_pos) -- the correct
    next-token prediction at `induction_pos` (i.e. tokens[induction_pos+1])
    is the well-defined recall target B, requiring the model to have
    retained the (A, B) association from `lag` positions earlier.

    lag must be >= 2 (room for A, B, ..., A, B without the two A/B pairs
    colliding) and seq_len >= lag + 3 (room for at least one valid cue
    position)."""
    if lag < 2:
        raise ValueError(f"lag must be >= 2, got {lag}")
    if seq_len < lag + 3:
        raise ValueError(f"seq_len={seq_len} too short for lag={lag} "
                          f"(need seq_len >= lag+3)")

    tokens = rng.randint(0, vocab_size, size=seq_len).astype(np.int64)
    cue_pos = int(rng.randint(0, seq_len - 2 - lag + 1))
    repeat_pos = cue_pos + lag

    a = int(rng.randint(0, vocab_size))
    b = int(rng.randint(0, vocab_size))
    tokens[cue_pos] = a
    tokens[cue_pos + 1] = b
    tokens[repeat_pos] = a
    tokens[repeat_pos + 1] = b

    return tokens, repeat_pos


def induction_correct(predicted_token: int, tokens: np.ndarray, induction_pos: int) -> bool:
    """True if predicted_token matches the well-defined recall target
    (tokens[induction_pos+1])."""
    return int(predicted_token) == int(tokens[induction_pos + 1])
