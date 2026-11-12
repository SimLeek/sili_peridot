"""The measuring instrument for "trains in-context AND recurrent" -- built
before any model change, because every wrong conclusion in this investigation
came from sweeping before the instrument existed.

THE SHAPE OF THE TARGET USE CASE. A model reads a page of a paper in context and
remembers the book title from far earlier; or watches a screen in context and
remembers earlier frames. Those are two different capabilities with two different
failure modes, and a single accuracy number cannot tell them apart -- which is
exactly how the earlier MQAR work got stuck. So this task emits TWO scores:

  IN-CONTEXT  questions answerable from the current attention window alone.
  RECURRENT   questions whose answer appeared ONCE, before the window, and is
              therefore unreachable except through persistent state.

The engineering criterion the user set earlier still governs: adding recurrence
must not HARM the in-context score. A recurrent win paid for by an in-context
loss is not a win, so both numbers are always reported together, plus the
per-capability chance level (they differ, so a single chance line would mislead).

STRUCTURE of one episode, with window W and title vocabulary T:
  tick 0            TITLE token t* (drawn from the title vocab)
  then, repeating:  PAGE of `page_len` content tokens, each page containing a
                    local (key, value) pair; a PAGE QUERY re-presents that key
  interleaved:      TITLE QUERY, a reserved token asking for t*

A title query is placed only after the title has left the window (position >
W), so it cannot be answered by attention -- verified by `assert_unreachable()`
rather than assumed, since "the gap was inside the window" is precisely the
mistake that made an earlier MQAR run look adversarial when it was mis-sized.

Vocabulary layout is disjoint so the two scores cannot contaminate each other:
titles, content/values, and the two query markers occupy separate id ranges, and
a title can never be produced as a page answer or vice versa.

Run: python -m scripts.dual_path_recall_task   (self-test of the task itself)
"""

import numpy as np

N_TITLES = 8
N_CONTENT = 24
PAGE_LEN = 6
TITLE_QUERY_EVERY = 3  # in pages
VOCAB = N_TITLES + N_CONTENT + 2

TITLE_LO = 0
CONTENT_LO = N_TITLES
Q_PAGE = N_TITLES + N_CONTENT
Q_TITLE = Q_PAGE + 1


class DualPathEpisode:
    """One episode's token stream plus, for each position, what (if anything)
    should be predicted there and which capability it tests."""

    def __init__(self, tokens, targets):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        # targets: list of (position, answer_token, kind) with kind in
        # {"page", "title"}; position is where the answer must be emitted.
        self.targets = targets

    def __len__(self):
        return len(self.tokens)


def make_episode(rng, n_pages: int, window: int) -> DualPathEpisode:
    title = int(rng.integers(N_TITLES)) + TITLE_LO
    tokens = [title]
    targets = []
    for page in range(n_pages):
        keys = rng.choice(N_CONTENT, size=2, replace=False) + CONTENT_LO
        vals = rng.choice(N_CONTENT, size=2, replace=False) + CONTENT_LO
        filler = rng.choice(N_CONTENT, size=max(0, PAGE_LEN - 4), replace=True) + CONTENT_LO
        # Shuffle CHUNKS, never tokens: each (key, value) must stay ADJACENT or
        # the binding is unrecoverable and the task is unsolvable in principle.
        # Shuffling the flat token list destroys exactly that, and a check for
        # "is the answer somewhere in the window" passes anyway -- which is how
        # a broken task survived its own self-test once already.
        chunks = [[int(keys[0]), int(vals[0])], [int(keys[1]), int(vals[1])]]
        chunks += [[int(t)] for t in filler.tolist()]
        rng.shuffle(chunks)
        tokens.extend(t for chunk in chunks for t in chunk)
        # page query: re-present one of this page's keys, answer is its value
        # Marker THEN key, with the target at the KEY's position: the answer is
        # only determined once the key is visible. Scoring at the marker instead
        # would ask the model to answer before knowing which key was queried,
        # capping page accuracy at 1/(keys per page) no matter how good it is.
        pick = int(rng.integers(2))
        tokens.append(Q_PAGE)
        tokens.append(int(keys[pick]))
        targets.append((len(tokens) - 1, int(vals[pick]), "page"))
        if (page + 1) % TITLE_QUERY_EVERY == 0 and len(tokens) > window:
            tokens.append(Q_TITLE)
            targets.append((len(tokens) - 1, title, "title"))
    return DualPathEpisode(tokens, targets)


def assert_unreachable(ep: DualPathEpisode, window: int) -> None:
    """Every title target must sit strictly outside the window containing the
    title token, or the 'recurrent' score is secretly an in-context score."""
    title_pos = 0
    for pos, _answer, kind in ep.targets:
        if kind != "title":
            continue
        if pos - title_pos < window:
            raise AssertionError(
                f"title query at {pos} is only {pos - title_pos} ticks after the "
                f"title but the window is {window}: attention can still see it, "
                f"so this would not test recurrence"
            )


def assert_page_recoverable(ep: DualPathEpisode, window: int) -> None:
    """Every page target must have its key appear earlier in the window
    IMMEDIATELY FOLLOWED BY the answer. Checking only that the answer token is
    somewhere in the window is far too weak -- that check passes even when the
    key/value binding has been shuffled apart and the task is unsolvable."""
    for pos, answer, kind in ep.targets:
        if kind != "page":
            continue
        lo = max(0, pos - window + 1)
        win = ep.tokens[lo : pos + 1]
        key = int(ep.tokens[pos])
        if not any(int(win[j]) == key and int(win[j + 1]) == answer for j in range(len(win) - 2)):
            raise AssertionError(
                f"page target at {pos}: key {key} is never followed by answer "
                f"{answer} inside the window, so the binding is unrecoverable"
            )


def chance_levels() -> dict:
    """Per-capability chance. They differ, so never quote one number for both."""
    return {"title": 1.0 / N_TITLES, "page": 1.0 / N_CONTENT}


def score(predictions: dict, ep: DualPathEpisode) -> dict:
    """predictions: {position: predicted_token}. Returns per-kind hit counts."""
    out = {"page": [0, 0], "title": [0, 0]}
    for pos, answer, kind in ep.targets:
        if pos not in predictions:
            continue
        out[kind][1] += 1
        out[kind][0] += int(predictions[pos] == answer)
    return out


def _self_test():
    rng = np.random.default_rng(0)
    window = 16
    print(f"vocab={VOCAB}  titles={N_TITLES} content={N_CONTENT} window={window}")
    print(f"chance: {chance_levels()}")
    tot = {"page": [0, 0], "title": [0, 0]}
    lens = []
    for _ in range(200):
        ep = make_episode(rng, n_pages=9, window=window)
        assert_unreachable(ep, window)
        assert_page_recoverable(ep, window)
        lens.append(len(ep))
        for _pos, _answer, kind in ep.targets:
            tot[kind][1] += 1
    print(f"episode length: mean {np.mean(lens):.1f} min {min(lens)} max {max(lens)}")
    print(f"targets per 200 episodes: page={tot['page'][1]} title={tot['title'][1]}")

    # a title answer must never be reachable as a page answer, and vice versa
    ep = make_episode(rng, n_pages=9, window=window)
    title_answers = {a for _p, a, k in ep.targets if k == "title"}
    page_answers = {a for _p, a, k in ep.targets if k == "page"}
    assert not (title_answers & page_answers), "vocab ranges overlap"
    print("disjoint vocab ranges: OK")

    # the unreachability assert must actually fire when the window is too wide
    try:
        assert_unreachable(ep, window=10_000)
        print("FAIL: unreachability assert did not fire")
    except AssertionError:
        print("unreachability assert fires when window is too wide: OK")

    # a title-blind oracle should score chance on titles and perfectly on pages,
    # which is the behaviour an attention-only model should reproduce
    preds = {}
    for pos, answer, kind in ep.targets:
        preds[pos] = answer if kind == "page" else int(rng.integers(N_TITLES))
    s = score(preds, ep)
    print(f"attention-only oracle: page {s['page'][0]}/{s['page'][1]}, title {s['title'][0]}/{s['title'][1]}")
    print("(page should be perfect, title near chance -- that is the baseline to beat)")


if __name__ == "__main__":
    _self_test()
