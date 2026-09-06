"""
sili_peridot/model/toy_recall_task.py
───────────────────────────────────────
Synthetic associative-recall ("induction head") task.
See docs/research/toy_recall_task.rst:toy_recall_task.module_overview.
"""

from __future__ import annotations

import numpy as np


def generate_sequence(
    rng: np.random.RandomState,
    vocab_size: int,
    seq_len: int,
    lag: int,
) -> tuple[np.ndarray, int]:
    """Returns (tokens [seq_len] int, induction_pos).

    See docs/research/toy_recall_task.rst:toy_recall_task.module_overview.
    """
    if lag < 2:
        raise ValueError(f"lag must be >= 2, got {lag}")
    if seq_len < lag + 3:
        raise ValueError(f"seq_len={seq_len} too short for lag={lag} (need seq_len >= lag+3)")

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


def generate_mqar_sequence(
    rng: np.random.RandomState,
    vocab_size: int,
    seq_len: int,
    num_kv_pairs: int,
    power_a: float = 0.01,
    random_non_queries: bool = True,
    forced_keys: np.ndarray = None,
) -> tuple[np.ndarray, list]:
    """Standard Multi-Query Associative Recall task (Arora, Eyuboglu et
    al., "Zoology: Measuring and Improving Recall in Efficient Language
    Models", 2023, arXiv:2312.04927), ported from HazyResearch/zoology's
    own reference implementation to plain single-example numpy.

    Returns (tokens [seq_len] int, pairs) where `pairs` is a list of
    (position, correct_next_token). `seq_len` must be even and
    `>= 4*num_kv_pairs`.

    See docs/research/toy_recall_task.rst:toy_recall_task.mqar_adoption_and_port.
    """
    if seq_len % 2 != 0:
        raise ValueError(f"seq_len must be even, got {seq_len}")
    if vocab_size <= seq_len:
        raise ValueError(f"vocab_size={vocab_size} must exceed seq_len={seq_len}")
    context_size = num_kv_pairs * 2
    if context_size * 2 > seq_len:
        raise ValueError(
            f"seq_len={seq_len} too short for num_kv_pairs={num_kv_pairs} (need seq_len >= 4*num_kv_pairs)"
        )

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)
    if forced_keys is not None and len(forced_keys) > 0:
        # See docs/research/toy_recall_task.rst:toy_recall_task.forced_keys_curriculum_guarantee.
        valid_forced = np.asarray(forced_keys, dtype=np.int64)
        valid_forced = valid_forced[(valid_forced >= 1) & (valid_forced < key_vocab_size)]
        n_forced = min(len(valid_forced), num_kv_pairs)
        if n_forced < len(valid_forced):
            valid_forced = rng.choice(valid_forced, size=n_forced, replace=False)
        remaining_choices = np.setdiff1d(key_choices, valid_forced)
        n_remaining = num_kv_pairs - n_forced
        rest = (
            rng.choice(remaining_choices, size=n_remaining, replace=False)
            if n_remaining > 0
            else np.array([], dtype=np.int64)
        )
        keys = np.concatenate([valid_forced, rest])
        rng.shuffle(keys)
    else:
        keys = rng.choice(key_choices, size=num_kv_pairs, replace=False)
    values = rng.choice(value_choices, size=num_kv_pairs, replace=False)

    kvs = np.zeros(context_size, dtype=np.int64)
    kvs[0::2] = keys
    kvs[1::2] = values

    space = (seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    gaps = rng.choice(space, size=num_kv_pairs, replace=False, p=p)

    query_region_len = seq_len - context_size
    queries = np.zeros(query_region_len, dtype=np.int64)
    queries[gaps * 2] = keys

    tokens = np.concatenate([kvs, queries])
    if random_non_queries:
        zero_mask = tokens == 0
        n_zero = int(zero_mask.sum())
        if n_zero > 0:
            tokens[zero_mask] = rng.randint(0, vocab_size, size=n_zero)

    query_positions = context_size + gaps * 2
    pairs = [(int(pos), int(val)) for pos, val in zip(query_positions, values, strict=False)]
    return tokens, pairs
