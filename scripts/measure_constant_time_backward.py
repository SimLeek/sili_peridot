"""Is the per-step backward actually O(1) in the horizon, or do I only believe it?

This is the claim that matters for running the model in real time, and it is a
structural claim, so it should be measured rather than asserted. Two arms on the
SAME model and the same work:

  per-step   backward called at every position. The memory crossing a step
             boundary is detached numpy, so no graph spans steps and the cost of
             step t should not depend on t. MEASURED: 1.00x -> 1.19x across
             positions 0-512, and 3.48/3.40/2.88/3.15/3.62 ms/step at lengths
             32/64/128/256/512 -- no trend, so O(1) holds.
  deferred   losses summed across the episode and ONE backward at the end.

CORRECTION TO AN EARLIER FRAMING: `deferred` was described as a BPTT baseline
that should grow super-linearly. It does not, and the premise was wrong -- the
memory crossing a step boundary is detached numpy, so NO graph spans steps and
deferring merely batches independent graphs (measured 1.47-1.88 ms/step, flat).
So this arm is not a BPTT cost model; it is a LEAK DETECTOR. If any graph were
silently spanning steps, its backward would grow with length, and it doesn't.
A real BPTT contrast would need a variant holding M as a Tensor across steps,
which this file deliberately does not build.

What would falsify the claim: per-step time rising with position. That would mean
something is silently retaining the graph across steps -- which is exactly the
bug class that detaching is supposed to prevent, and worth catching here rather
than discovering on a long stream.

Reported per position bucket, so a trend is visible rather than averaged away.
Wall-clock on one machine is noisy; the SHAPE (flat vs rising) is the result, not
the absolute microseconds.

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
