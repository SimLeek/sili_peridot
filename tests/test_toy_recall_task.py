import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from model.toy_recall_task import generate_sequence, induction_correct


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
            tokens, pos = generate_sequence(rng, vocab_size=10, seq_len=seq_len, lag=lag)
            assert pos + 1 < seq_len
            assert pos - lag >= 0
