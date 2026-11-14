"""Is the per-step backward O(1) in the horizon, or do I only believe it?

Two arms on the same model and the same work:
  per-step   backward at every position. Measured flat (1.00-1.19x across
             positions 0-512), so O(1) holds.
  deferred   one backward at episode end. A LEAK DETECTOR, not a BPTT
             baseline -- nothing spans steps, so it cannot grow; if it
             did, a graph would be crossing step boundaries. See
             JOURNAL.md 2026-11-14 for why the BPTT framing was wrong.

Falsified by: per-step time rising with position. Reported per bucket so a
trend stays visible. The shape matters, not the absolute microseconds.

Run: python -m scripts.measure_constant_time_backward
"""

import time

import numpy as np

from model.titans_transformer import TitansMAC
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum
from scripts.dual_path_recall_task import VOCAB

HIDDEN, WINDOW = 32, 16
LENGTHS = (32, 64, 128, 256, 512)


def _tokens(rng, n):
    return rng.integers(0, VOCAB, size=n).astype(np.int64)


def per_step_timing(length: int, seed: int = 0) -> tuple[list[float], float]:
    """Backward at every position. Returns (per-step seconds, total)."""
    rng = np.random.default_rng(seed)
    model = TitansMAC(VOCAB, hidden=HIDDEN, window=WINDOW, mode="frozen", rng=np.random.default_rng(seed + 1))
    opt = AdamOptimizer()
    tokens = _tokens(rng, length + 1)
    model.reset()
    times = []
    for i in range(length):
        t0 = time.perf_counter()
        logits = model.step(tokens, i)
        loss = cross_entropy_sum(logits, [(model.n_rows - 1, int(tokens[i + 1]))])
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step(model.parameters(), lr=0.003)
        times.append(time.perf_counter() - t0)
    return times, float(np.sum(times))


def deferred_timing(length: int, seed: int = 0) -> tuple[float, float]:
    """One backward over the whole episode's graph. Returns (forward_s, backward_s)."""
    rng = np.random.default_rng(seed)
    model = TitansMAC(VOCAB, hidden=HIDDEN, window=WINDOW, mode="frozen", rng=np.random.default_rng(seed + 1))
    tokens = _tokens(rng, length + 1)
    model.reset()
    t0 = time.perf_counter()
    total = None
    for i in range(length):
        logits = model.step(tokens, i)
        loss = cross_entropy_sum(logits, [(model.n_rows - 1, int(tokens[i + 1]))])
        total = loss if total is None else total + loss
    fwd = time.perf_counter() - t0
    t1 = time.perf_counter()
    total.backward()
    return fwd, time.perf_counter() - t1


def main() -> None:
    print(f"hidden={HIDDEN} window={WINDOW} mode=frozen")
    print("per-step: backward every position (memory detached across steps).")
    print("deferred: one backward over the episode graph -- a LEAK DETECTOR,")
    print("not a BPTT baseline (nothing spans steps, so it cannot grow).\n")

    print("PER-STEP COST BY POSITION -- the O(1) claim. Flat means O(1);")
    print("rising would mean the graph is silently spanning steps.")
    times, _tot = per_step_timing(512)
    buckets = [(0, 64), (64, 128), (128, 256), (256, 384), (384, 512)]
    print(f"{'positions':>12} {'ms/step':>9} {'vs first':>9}")
    first = None
    for lo, hi in buckets:
        ms = 1000.0 * float(np.median(times[lo:hi]))
        first = ms if first is None else first
        print(f"{f'{lo}-{hi}':>12} {ms:>9.3f} {ms / first:>8.2f}x")

    print("\nTOTAL COST VS EPISODE LENGTH -- per-step total should scale LINEARLY")
    print("(i.e. constant PER STEP). deferred is a leak detector, not a BPTT")
    print("baseline: flat means no graph spans step boundaries.")
    print(f"{'length':>7} {'per-step tot':>13} {'ms/step':>9} {'deferred bwd':>13} {'bwd ms/step':>12}")
    for n in LENGTHS:
        ts, tot = per_step_timing(n)
        _fwd, bwd = deferred_timing(n)
        print(
            f"{n:>7} {tot:>12.3f}s {1000 * tot / n:>9.3f} {bwd:>12.3f}s {1000 * bwd / n:>12.3f}",
            flush=True,
        )
    print("\nRead the per-step ms/step column: if it stays flat as length grows,")
    print("cost is independent of horizon, which is what real-time use needs.")


if __name__ == "__main__":
    main()
