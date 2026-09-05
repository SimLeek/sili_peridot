import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from model.toy_beyond_context_task import QUERY_TOKEN, VOCAB_SIZE, generate_parity_sequence


class TestGenerateParitySequence:
    def test_shape_and_dtype(self):
        rng = np.random.RandomState(0)
        tokens, _pairs = generate_parity_sequence(rng, n_bits=5)
        assert tokens.shape == (7,)
        assert tokens.dtype == np.int64

    def test_query_token_placed_right_after_the_bits(self):
        rng = np.random.RandomState(1)
        tokens, pairs = generate_parity_sequence(rng, n_bits=6)
        assert tokens[6] == QUERY_TOKEN
        assert pairs == [(6, int(tokens[7]))]

    def test_answer_is_the_real_xor_of_the_bits(self):
        rng = np.random.RandomState(2)
        for _ in range(50):
            tokens, pairs = generate_parity_sequence(rng, n_bits=8)
            bits = tokens[:8]
            expected = int(np.bitwise_xor.reduce(bits))
            (_pos, answer) = pairs[0]
            assert answer == expected
            assert tokens[-1] == expected

    def test_bits_are_only_0_or_1(self):
        rng = np.random.RandomState(3)
        tokens, _pairs = generate_parity_sequence(rng, n_bits=20)
        assert set(tokens[:20].tolist()) <= {0, 1}

    def test_answer_is_binary_and_within_vocab(self):
        rng = np.random.RandomState(4)
        for _ in range(20):
            _tokens, pairs = generate_parity_sequence(rng, n_bits=3)
            _pos, answer = pairs[0]
            assert answer in (0, 1)
            assert answer < VOCAB_SIZE

    def test_deterministic_given_seeded_rng_state(self):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        tokens1, pairs1 = generate_parity_sequence(rng1, n_bits=10)
        tokens2, pairs2 = generate_parity_sequence(rng2, n_bits=10)
        np.testing.assert_array_equal(tokens1, tokens2)
        assert pairs1 == pairs2

    def test_rejects_zero_bits(self):
        rng = np.random.RandomState(5)
        with pytest.raises(ValueError):
            generate_parity_sequence(rng, n_bits=0)

    def test_single_bit_parity_equals_the_bit_itself(self):
        rng = np.random.RandomState(6)
        for _ in range(20):
            tokens, pairs = generate_parity_sequence(rng, n_bits=1)
            assert pairs[0][1] == int(tokens[0])
