"""Verification run: does a Titans-MAC transformer learn the recurrent path
WITHOUT paying for it on the in-context path?

Three arms, reported side by side against per-capability chance:
  none    no memory. The control. Must do well in-context and sit at chance on
          recurrent questions; if it beats chance there, the task leaks and
          every other number is meaningless.
  frozen  memory on, write side random and frozen, read side trained. Strictly
          O(1) per step with no approximation anywhere.
  trace   write side additionally trained from an eligibility trace, still O(1).

The engineering criterion, set earlier and unchanged: recurrence must not HARM
the in-context score. So the headline is the PAIR (page, title), never one of
them, and `none` supplies the in-context reference the others must match.

Forward runs at every position because the memory has to accumulate; backward
runs only at the ~12 scored positions per episode. That asymmetry is the point
of the architecture -- cost per step is independent of how far back the answer
was written.

Run: python -m scripts.train_dual_path_titans [episodes] [seeds]
"""

import sys
import time

import numpy as np

from model.titans_transformer import TitansMAC
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum, predicted_token
from scripts.dual_path_recall_task import (
    VOCAB,
    assert_unreachable,
    chance_levels,
    make_episode,
)

HIDDEN, WINDOW, N_PAGES = 32, 16, 9
PEAK_LR = 0.01
EVAL_EPISODES = 60


def run_arm(mode: str, episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    model = TitansMAC(VOCAB, hidden=HIDDEN, window=WINDOW, mode=mode, rng=np.random.default_rng(seed + 7))
    opt = AdamOptimizer()
    t0 = time.time()
    n_backward = 0

    for _ in range(episodes):
        ep = make_episode(rng, n_pages=N_PAGES, window=WINDOW)
        by_pos = {p: (a, k) for p, a, k in ep.targets}
        model.reset()
        for i in range(len(ep)):
            logits = model.step(ep.tokens, i)
            if i not in by_pos:
                continue
            answer, _kind = by_pos[i]
            loss = cross_entropy_sum(logits, [(model.n_rows - 1, int(answer))])
            loss.backward()
            n_backward += 1
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step(model.parameters(), lr=PEAK_LR)
    train_s = time.time() - t0

    hits = {"page": [0, 0], "title": [0, 0]}
    for e in range(EVAL_EPISODES):
        ep = make_episode(np.random.default_rng(90_000 + e), n_pages=N_PAGES, window=WINDOW)
        assert_unreachable(ep, WINDOW)
        by_pos = {p: (a, k) for p, a, k in ep.targets}
        model.reset()
        for i in range(len(ep)):
            logits = model.step(ep.tokens, i)
            if i not in by_pos:
                continue
            answer, kind = by_pos[i]
            hits[kind][1] += 1
            hits[kind][0] += int(predicted_token(logits, model.n_rows - 1) == int(answer))
    return {
        "mode": mode,
        "seed": seed,
        "page": hits["page"][0] / max(1, hits["page"][1]),
        "title": hits["title"][0] / max(1, hits["title"][1]),
        "n_page": hits["page"][1],
        "n_title": hits["title"][1],
        "n_backward": n_backward,
        "train_s": train_s,
    }


def main() -> None:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ch = chance_levels()
    print(f"hidden={HIDDEN} window={WINDOW} pages={N_PAGES} vocab={VOCAB}")
    print(f"episodes={episodes} seeds={n_seeds}  chance: page={ch['page']:.4f} title={ch['title']:.4f}")
    print("\nin-context (page) must NOT drop when memory is added; recurrent")
    print("(title) must rise above chance. Both, or it does not count.\n")
    print(f"{'mode':>8} {'page':>16} {'title':>16} {'n_bwd':>7} {'train_s':>8}")
    for mode in ("none", "frozen", "trace"):
        rows = [run_arm(mode, episodes, 1000 + s) for s in range(n_seeds)]
        pg = np.array([r["page"] for r in rows])
        ti = np.array([r["title"] for r in rows])
        print(
            f"{mode:>8} {pg.mean():>9.4f}+-{pg.std(ddof=1) if len(pg) > 1 else 0:.3f} "
            f"{ti.mean():>9.4f}+-{ti.std(ddof=1) if len(ti) > 1 else 0:.3f} "
            f"{rows[0]['n_backward']:>7} {np.mean([r['train_s'] for r in rows]):>8.1f}",
            flush=True,
        )
    print(f"\neval targets per arm: page={rows[0]['n_page']} title={rows[0]['n_title']}")


if __name__ == "__main__":
    main()
