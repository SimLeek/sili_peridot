import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from model.toy_recall_task import generate_mqar_sequence, generate_sequence, induction_correct


class TestGenerateSequence:
    def test_shape_and_range(self):
        rng = np.random.RandomState(0)
        tokens, pos = generate_sequence(rng, vocab_size=10, seq_len=20, lag=5)
        assert tokens.shape == (20,)
        assert tokens.dtype == np.int64
        assert np.all((tokens >= 0) & (tokens < 10))
        assert 0 <= pos < 20

    def test_cue_and_response_bigram_appears_twice(self):
        rng = np.random.RandomState(1)
        tokens, pos = generate_sequence(rng, vocab_size=16, seq_len=40, lag=8)
        a, b = tokens[pos], tokens[pos + 1]
        assert tokens[pos - 8] == a
        assert tokens[pos - 8 + 1] == b

    def test_induction_correct_matches_the_real_next_token(self):
        rng = np.random.RandomState(2)
        tokens, pos = generate_sequence(rng, vocab_size=16, seq_len=40, lag=8)
        assert induction_correct(int(tokens[pos + 1]), tokens, pos)
        wrong = (int(tokens[pos + 1]) + 1) % 16
        assert not induction_correct(wrong, tokens, pos)

    def test_lag_too_small_raises(self):
        rng = np.random.RandomState(0)
        with pytest.raises(ValueError, match="lag must be"):
            generate_sequence(rng, vocab_size=10, seq_len=20, lag=1)

    def test_seq_len_too_short_for_lag_raises(self):
        rng = np.random.RandomState(0)
        with pytest.raises(ValueError, match="too short"):
            generate_sequence(rng, vocab_size=10, seq_len=5, lag=10)

    def test_repeat_position_is_exactly_cue_position_plus_lag(self):
        rng = np.random.RandomState(3)
        for lag in [2, 5, 15]:
            tokens, pos = generate_sequence(rng, vocab_size=10, seq_len=50, lag=lag)
            # pos IS the repeat position by construction/return contract
            cue_pos = pos - lag
            assert cue_pos >= 0
            assert tokens[cue_pos] == tokens[pos]
            assert tokens[cue_pos + 1] == tokens[pos + 1]

    def test_many_seeds_never_crash_at_the_tight_boundary(self):
        # seq_len exactly at the minimum valid size for this lag.
        rng = np.random.RandomState(4)
        lag = 6
        seq_len = lag + 3
        for _ in range(50):
            _tokens, pos = generate_sequence(rng, vocab_size=10, seq_len=seq_len, lag=lag)
            assert pos + 1 < seq_len
            assert pos - lag >= 0


class TestGenerateMqarSequence:
    """Standard MQAR (Arora, Eyuboglu et al., "Zoology", 2023) -- see
    generate_mqar_sequence's own docstring for the source/rationale."""

    def test_shape_and_pair_count(self):
        rng = np.random.RandomState(0)
        tokens, pairs = generate_mqar_sequence(rng, vocab_size=32, seq_len=16, num_kv_pairs=3)
        assert tokens.shape == (16,)
        assert len(pairs) == 3

    def test_every_query_target_is_the_correct_paired_value(self):
        # Direct correctness check: reconstruct the key->value map from
        # the context (first 2*num_kv_pairs tokens) and confirm every
        # returned (position, target) pair matches it exactly.
        rng = np.random.RandomState(1)
        num_kv_pairs = 4
        tokens, pairs = generate_mqar_sequence(rng, vocab_size=40, seq_len=32, num_kv_pairs=num_kv_pairs)
        context = tokens[: num_kv_pairs * 2]
        kv_map = {int(context[i]): int(context[i + 1]) for i in range(0, len(context), 2)}
        for pos, target in pairs:
            query_key = int(tokens[pos])
            assert query_key in kv_map, f"position {pos} doesn't hold a real key"
            assert kv_map[query_key] == target

    def test_keys_and_values_never_cross_the_vocab_half(self):
        # Keys drawn from the lower half, values from the upper half --
        # a key can never be mistaken for a value or vice versa.
        rng = np.random.RandomState(2)
        vocab_size = 40
        tokens, pairs = generate_mqar_sequence(rng, vocab_size=vocab_size, seq_len=16, num_kv_pairs=4)
        context = tokens[:8]
        keys, values = context[0::2], context[1::2]
        assert np.all(keys < vocab_size // 2)
        assert np.all(values >= vocab_size // 2)
        for _pos, target in pairs:
            assert target >= vocab_size // 2

    def test_no_duplicate_query_positions(self):
        rng = np.random.RandomState(3)
        _tokens, pairs = generate_mqar_sequence(rng, vocab_size=64, seq_len=32, num_kv_pairs=6)
        positions = [p for p, _t in pairs]
        assert len(positions) == len(set(positions))

    def test_seq_len_must_be_even(self):
        rng = np.random.RandomState(0)
        with pytest.raises(ValueError, match="even"):
            generate_mqar_sequence(rng, vocab_size=20, seq_len=15, num_kv_pairs=2)

    def test_vocab_must_exceed_seq_len(self):
        rng = np.random.RandomState(0)
        with pytest.raises(ValueError, match="exceed"):
            generate_mqar_sequence(rng, vocab_size=16, seq_len=16, num_kv_pairs=2)

    def test_seq_len_too_short_for_num_kv_pairs_raises(self):
        rng = np.random.RandomState(0)
        with pytest.raises(ValueError, match="too short"):
            generate_mqar_sequence(rng, vocab_size=100, seq_len=8, num_kv_pairs=4)

    def test_many_seeds_never_crash(self):
        rng = np.random.RandomState(4)
        for _ in range(50):
            tokens, pairs = generate_mqar_sequence(rng, vocab_size=40, seq_len=32, num_kv_pairs=4)
            assert tokens.shape == (32,)
            assert len(pairs) == 4
