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
PEAK_LR = 0.003
SIGMA_INIT = 4.0
# EVAL SET MUST OUT-RESOLVE THE EFFECT. At 60 episodes the eval sets were 540
# page / 180 title targets, whose BINOMIAL sampling sd (0.0128 page) EXCEEDED the
# between-seed sd (0.0080) -- so "seed variance" was mostly eval noise and any
# MDE derived from it was meaningless. Sized so eval sd is ~0.004 page /
# ~0.009 title, well under the effects worth detecting. Eval is forward-only, so
# this costs ~90s per arm-seed.
EVAL_EPISODES = 400
# 8000 training episodes: induction forms as a PHASE TRANSITION around ep
# 3000-4000, so shorter runs measure "has the circuit formed yet" (bimodal,
# sd 0.43) rather than arm quality.
TRAIN_EPISODES = 8000


def run_arm(mode: str, episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    model = TitansMAC(
        VOCAB,
        hidden=HIDDEN,
        window=WINDOW,
        mode=mode,
        n_layers=2,
        sigma_init=SIGMA_INIT,
        rng=np.random.default_rng(seed + 7),
    )
    opt = AdamOptimizer()
    w_k0 = model.W_K_mem.copy()
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
            # Absence of THIS call made `trace` identical to `frozen` for two
            # whole comparison runs; see JOURNAL.md 2026-11-13.
            grad = model.read_grad()
            if grad is not None:
                model.apply_trace_credit(grad, PEAK_LR)
    train_s = time.time() - t0
    w_moved = float(np.abs(model.W_K_mem - w_k0).max())

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
        "w_moved": w_moved,
    }


def main() -> None:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_EPISODES
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ch = chance_levels()
    print(f"hidden={HIDDEN} window={WINDOW} pages={N_PAGES} vocab={VOCAB}")
    print(f"episodes={episodes} seeds={n_seeds}  chance: page={ch['page']:.4f} title={ch['title']:.4f}")
    print("\nin-context (page) must NOT drop when memory is added; recurrent")
    print("(title) must rise above chance. Both, or it does not count.\n")
    print(f"{'mode':>8} {'page  sd/evalsd':>24} {'title  sd/evalsd':>24} {'train':>7}")
    print("sd = between-seed; ev = binomial eval sd. If ev >= sd, the seed")
    print("number is measuring eval noise and no MDE can be read from it.\n")
    for mode in ("none", "frozen", "trace"):
        rows = [run_arm(mode, episodes, 1000 + s) for s in range(n_seeds)]
        pg = np.array([r["page"] for r in rows])
        ti = np.array([r["title"] for r in rows])
        n_pg, n_ti = rows[0]["n_page"], rows[0]["n_title"]
        # eval (binomial) sd at the observed rate, for comparison with seed sd:
        # if eval sd is the larger of the two, the seed sd is measuring noise.
        ev_pg = float(np.sqrt(max(pg.mean() * (1 - pg.mean()), 1e-9) / n_pg))
        ev_ti = float(np.sqrt(max(ti.mean() * (1 - ti.mean()), 1e-9) / n_ti))
        s_pg = pg.std(ddof=1) if len(pg) > 1 else 0.0
        s_ti = ti.std(ddof=1) if len(ti) > 1 else 0.0
        print(
            f"{mode:>8} {pg.mean():>7.4f} sd{s_pg:.4f}/ev{ev_pg:.4f} "
            f"{ti.mean():>7.4f} sd{s_ti:.4f}/ev{ev_ti:.4f} "
            f"{np.mean([r['train_s'] for r in rows]) / 60:>6.1f}m"
            f" wK{rows[0]['w_moved']:.2e}",
            flush=True,
        )
        # Arms must be DISTINGUISHABLE: the write side is frozen in none/frozen
        # and must actually move in trace. Without this, trace silently equals
        # frozen -- which already happened.
        if mode == "trace" and rows[0]["w_moved"] == 0.0:
            raise AssertionError("trace arm: W_K never moved -- credit not wired")
        if mode in ("none", "frozen") and rows[0]["w_moved"] != 0.0:
            raise AssertionError(f"{mode} arm: W_K moved -- arms are not isolated")
    print(f"\neval targets per arm: page={rows[0]['n_page']} title={rows[0]['n_title']}")


if __name__ == "__main__":
    main()
