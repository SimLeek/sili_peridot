"""Long-term-only MQAR: pairs, then a window of BLANKS, then the key as query.

Replaces the dual-path title task for testing whether the recurrent memory can be
usefully written. Better instrument for three reasons: the CUE IS THE WRITTEN
TOKEN so key and query align by construction (the title task needed a cross-token
alignment); attention PROVABLY cannot help, since the gap is >= the window and
blank, so `none` must score exactly chance; and blanks are maximally redundant, so
nearly free for the memory to carry (0.9997 vs 0.2247 for novel filler).

`short_term=True` sets gap=0, leaving the pair inside the window -- the control
that isolates a write-side change from a task change.

Rationale and results: JOURNAL.md 2026-11-13.

Run: python -m scripts.long_term_mqar_task      (self-test)
"""

import numpy as np

N_KEYS = 12
N_VALS = 12
KEY_LO = 0
VAL_LO = N_KEYS
BLANK = N_KEYS + N_VALS
VOCAB = BLANK + 1


def make_episode(rng, n_pairs: int, window: int, gap: int | None = None, short_term: bool = False):
    """[k1 v1 ... kn vn] + BLANK*gap + [query_key]; answer is that key's value.

    `short_term=True` sets gap=0, leaving the pair INSIDE the window so attention
    alone can answer. That is the control that isolates a write-side optimizer
    change from a task change: the same task family, the same vocab, the same
    format, with only the gap varying. A write-side rule that helps long-term but
    wrecks short-term is not an improvement, and the previous Adam write side did
    exactly that (in-context 0.2191 against a 0.8831 control).

    Returns (tokens, query_position, answer_token)."""
    if short_term:
        gap = 0
    else:
        gap = window if gap is None else gap
        if gap < window:
            raise ValueError(f"gap {gap} < window {window}: the pair would still be visible")
    keys = rng.choice(N_KEYS, size=n_pairs, replace=False) + KEY_LO
    vals = rng.choice(N_VALS, size=n_pairs, replace=False) + VAL_LO
    tokens = []
    for k, v in zip(keys, vals, strict=True):
        tokens.extend([int(k), int(v)])
    tokens.extend([BLANK] * gap)
    pick = int(rng.integers(n_pairs))
    tokens.append(int(keys[pick]))
    return np.array(tokens, dtype=np.int64), len(tokens) - 1, int(vals[pick])


def assert_window_has_the_pair(tokens, qpos, window):
    """Short-term control: the queried key MUST appear earlier in the window
    immediately followed by its value, or attention cannot answer either and the
    control is not a control."""
    lo = max(0, qpos - window + 1)
    win = tokens[lo : qpos + 1]
    key = int(tokens[qpos])
    if not any(int(win[j]) == key for j in range(len(win) - 1)):
        raise AssertionError(f"short-term: key {key} is not visible in the window")


def assert_window_is_information_free(tokens, qpos, window):
    """At the query position the window must contain ONLY blanks plus the query
    key -- no key/value pair, so attention has nothing to work with."""
    lo = max(0, qpos - window + 1)
    win = tokens[lo : qpos + 1]
    body = win[:-1]
    bad = [int(t) for t in body if int(t) != BLANK]
    if bad:
        raise AssertionError(
            f"window at query pos {qpos} contains non-blank tokens {bad}: attention could answer without memory"
        )


def chance() -> float:
    return 1.0 / N_VALS


def _self_test():
    rng = np.random.default_rng(0)
    W = 16
    print(f"vocab={VOCAB} keys={N_KEYS} vals={N_VALS} BLANK={BLANK} chance={chance():.4f}")
    for n_pairs in (1, 2, 4):
        lens = []
        for _ in range(300):
            toks, qpos, answer = make_episode(rng, n_pairs, W)
            assert_window_is_information_free(toks, qpos, W)
            assert VAL_LO <= answer < VAL_LO + N_VALS, "answer outside the value range"
            lens.append(len(toks))
        print(f"  n_pairs={n_pairs}: len={lens[0]}, window information-free at query: OK")
    # the assertion must FIRE if the gap is too small
    toks, qpos, _ = make_episode(rng, 2, W, gap=W)
    try:
        assert_window_is_information_free(toks, qpos, window=W + 8)
        print("  FAIL: assertion did not fire for an oversized window")
    except AssertionError:
        print("  assertion fires when the window is widened past the gap: OK")
    try:
        make_episode(rng, 2, W, gap=4)
        print("  FAIL: gap<window was accepted")
    except ValueError:
        print("  gap<window rejected at construction: OK")
    # short-term control must be attention-answerable
    for _ in range(200):
        toks, qpos, _a = make_episode(rng, 2, W, short_term=True)
        assert_window_has_the_pair(toks, qpos, W)
    print("  short_term=True keeps the pair inside the window: OK")


if __name__ == "__main__":
    _self_test()
