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

import numpy as np


def generate_sequence(
    rng: np.random.RandomState,
    vocab_size: int,
    seq_len: int,
    lag: int,
) -> tuple[np.ndarray, int]:
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
    Models", 2023, arXiv:2312.04927) -- per direct decision, adopted
    instead of continuing to extend the hand-rolled single-bigram
    `generate_sequence` above: this is the established benchmark for
    exactly the property under test here (does an efficient/recurrent
    architecture recall associations as well as full attention), used
    throughout the linear-attention/SSM/RWKV architecture literature.
    Ported from HazyResearch/zoology's own reference implementation
    (`zoology/data/multiquery_ar.py`, fetched directly from the repo)
    to plain single-example numpy -- no torch, no batching (this
    project trains online, one example at a time, and batching a
    single CPU doesn't buy anything here regardless).

    A context of `num_kv_pairs` unique (key, value) bigrams is laid
    down first (keys drawn from the lower half of the vocab, values
    from the upper half -- so a key can never be mistaken for a value
    or vice versa). Then, at power-law-distributed gap positions after
    the context (small gaps much likelier than large ones, matching
    real language's own repeated-bigram statistics -- see the original
    docstring's own explanation), each key is repeated as a QUERY; the
    correct prediction immediately after a query is that key's own
    value, requiring genuine retrieval from the earlier context (never
    duplicated directly in the input). Non-query filler positions get
    random noise tokens (`random_non_queries`, matching the reference
    default) rather than a fixed placeholder, so the model can't use
    "is this a special filler symbol" as a shortcut.

    Returns (tokens [seq_len] int, pairs) where `pairs` is a list of
    (position, correct_next_token) -- directly usable with
    `cross_entropy_sum`'s own (row, target) convention, one entry per
    query. `seq_len` must be even and `>= 4*num_kv_pairs` (context +
    queries both need `2*num_kv_pairs` slots)."""
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
        # forced_keys (curriculum "at least 1 new vocab per N queries"
        # guarantee): without this, keys are drawn uniformly from the
        # WHOLE current key range, so a just-grown vocab's newest token
        # can go untested for many steps by pure chance, letting a
        # level-up streak complete on old vocab alone.
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
