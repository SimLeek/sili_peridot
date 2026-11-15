"""Why one r_target_min cannot work, and what to measure instead.

`r_target_min` is hand-tuned, and the wide-288 runs found a sharp CLIFF between
0.5 (zero progress in 100k steps) and 0.6-0.9 (level up in ~1-2k) -- the
signature of something breaking structurally, not of a quality knob.

Nucleus selection makes k a CONSEQUENCE of r_target and the row's own spectrum,
so an energy FRACTION has no natural scale: at fixed width the effectively-active
fraction spans 0.001 to 0.39 across gaussian/laplace/power-law/lognormal rows
while r_target is one number. The scale-free quantity is the participation ratio
PR(v) = (sum v^2)^2 / sum v^4, exact on the only unambiguous case (m equal
entries -> PR = m, capturing R = 1.0, where any r_target < 1 deletes live
directions). R(v, PR(v)) is NOT a universal constant (0.59-1.00), so the answer
is no r_target at all: k = PR(v) per row, measured.

Scope: neuron_grad_accum is gradient ACCUMULATED over steps, not the per-step dy
the kernel selects on, and this is a short probe, not the 100k-step wide-288 run
where the cliff was measured. This explains and predicts the cliff; it does not
show k = PR fixes it. Full numbers and the owed end-to-end test: JOURNAL.md
2026-11-12.

Run: python -m scripts.probe_sparsity_floor_participation_ratio
"""

import numpy as np

from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT

VOCAB, EMBED_WIDTH, NUM_TILES, MAX_WEIGHTS = 32, 12, 16, 4000
PEAK_LR, SEQ_LEN, SLOTS = 0.015, 48, 2
STEPS, POS_PER_STEP = 120, 6
LAYERS = ("input_proj", "q_proj", "k_proj", "v_proj", "o_proj")


def participation_ratio(v) -> float:
    """(sum v^2)^2 / sum v^4 -- the effective number of active components."""
    v = np.asarray(v, dtype=np.float64).ravel()
    s2 = float(np.sum(v**2))
    s4 = float(np.sum(v**4))
    return (s2 * s2 / s4) if s4 > 0 else 0.0


def captured_at_k(v, k) -> float:
    """R(v, k): the energy fraction the top-k components capture."""
    a = np.sort(np.asarray(v, dtype=np.float64).ravel() ** 2)[::-1]
    tot = a.sum()
    return float(a[: max(1, int(round(k)))].sum() / tot) if tot > 0 else 0.0


def k_for_r_target(v, r_target) -> int:
    """The k a fixed r_target would choose on this same row, for comparison."""
    a = np.sort(np.asarray(v, dtype=np.float64).ravel() ** 2)[::-1]
    tot = a.sum()
    if tot <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(a) / tot, r_target) + 1)


def probe(column_neurons: int, seed: int = 1001) -> list[tuple]:
    state_width = EMBED_WIDTH * column_neurons
    rng = np.random.default_rng(seed)
    model = ToyTileRecurrenceRMT(
        VOCAB,
        EMBED_WIDTH,
        column_neurons,
        NUM_TILES,
        SLOTS,
        MAX_WEIGHTS * max(1, column_neurons // 8),
        num_cpus=2,
        qkvo_norm_enable=True,
        rng=rng,
    )
    embed = (rng.standard_normal((VOCAB, EMBED_WIDTH)) * 0.3).astype(np.float32)
    opt = AdamOptimizer()

    def window(tokens, i):
        return np.stack([embed[tokens[max(0, i - (NUM_TILES - 1) + j)]] for j in range(NUM_TILES)], axis=0)

    for _ in range(STEPS):
        tokens = rng.integers(0, VOCAB, size=SEQ_LEN)
        memory = np.zeros((SLOTS, state_width), dtype=np.float32)
        for i in range(min(POS_PER_STEP, SEQ_LEN - 1)):
            memory, logits, aux = model.step(window(tokens, i), memory, PEAK_LR)
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, int(tokens[i + 1]))])
            if aux is not None:
                loss = loss + aux
            loss.backward()
            clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
            opt.step(model.parameters_for_optimizer(), lr=PEAK_LR)

    rows = []
    for name in LAYERS:
        g = np.asarray(getattr(model, name).neuron_grad_accum, dtype=np.float64)
        if g.size == 0 or not np.any(g):
            continue
        pr = participation_ratio(g)
        rows.append((name, g.size, pr, pr / g.size, captured_at_k(g, pr), k_for_r_target(g, 0.5) / g.size))
    return rows


def main() -> None:
    print("Per-layer participation ratio of real accumulated gradients.")
    print("PR/n is the fraction of output neurons that are effectively active;")
    print("r_tgt=0.5 k/n is what a fixed r_target would keep on the same row.\n")
    for cn in (8, 16, 24):
        rows = probe(cn)
        print(f"=== column_neurons={cn} (state_width={EMBED_WIDTH * cn}) ===")
        print(f"{'layer':>12} {'n_out':>6} {'PR':>8} {'PR/n':>7} {'R(v,PR)':>8} {'r_tgt=0.5 k/n':>14}")
        for name, n, pr, prn, rpr, k50 in rows:
            print(f"{name:>12} {n:>6} {pr:>8.2f} {prn:>7.4f} {rpr:>8.4f} {k50:>14.4f}")
        if rows:
            lo = min(r[3] for r in rows)
            hi = max(r[3] for r in rows)
            print(f"  PR/n spread across layers: {lo:.4f} .. {hi:.4f} = {hi / max(lo, 1e-9):.1f}x\n")
    print("Read it as: no single r_target_min can match every layer's PR/n, and")
    print("PR/n itself moves with width -- so the floor has to be measured.")


if __name__ == "__main__":
    main()
