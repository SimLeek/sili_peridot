"""Titans' two gates, each measured in the regime it actually targets.

This supersedes the gate numbers in `prototype_titans_adaptive_gates.py`, which
were confounded twice over. Recording both confounds here because each one
individually reversed the conclusion, and both are the same mistake -- changing
two things and crediting one (cf. feedback_do_science_correctly):

CONFOUND 1, hidden learning rate. eta does not only set the surprise trace, it
multiplies the EFFECTIVE step by 1/(1-eta): at eta=0.9 the memory takes a 10x
larger step than at eta=0. The high-eta cells were not learning, they were
diverging -- ||M|| reached 11093 for the matrix memory and end-of-stream surprise
19.2 vs 1.16 at eta=0. Fix: theta_scale = 1-eta, so eta changes only the trace.
With that, ||M|| stays ~5-8 and momentum's sign flips from negative to positive.

CONFOUND 2, write position. With theta ~ (1-eta), a write at time i contributes
~ grad_i * (1 - eta^(T-i)): ~1 for EARLY writes, ~(T-i)(1-eta) for recent ones.
So high eta is a low-pass filter privileging early content -- and the original
task wrote the probed pairs FIRST, handing momentum free credit. The opposite
regime confirmed it: with probes written LAST, eta=0.95 dropped to 0.818 from
0.980. Fix: scatter the probed pairs UNIFORMLY (order="random"), so no temporal
position is systematically rewarded, and keep first/last as explicit controls.

The three orders are NOT three samples of one quantity. A pair at position p has
(T - p) subsequent interfering writes: ~64 under "first", ~2 under "last", ~34
under "random". They measure different interference loads, which is why "random"
can exceed both controls rather than falling between them.

WHAT THE REGIMES SHOW (matrix memory, 400 eps x 3 seeds, chance 0.0156):
  * Momentum helps where probes are uniform: 0.665 -> 0.871 as eta 0 -> 0.95,
    with no positional bias available to it. Mechanism is visible in the
    controls: at eta=0 the memory is sharply recency-biased (first 0.23, last
    1.00), and momentum BROADENS that profile by re-injecting each write over
    later ticks. A broader profile beats a sharply recent one when probes are
    spread out.
  * Forgetting helps where probes are RECENT: down the alpha rows at eta=0.9 the
    "last" column runs 0.387 -> 0.451 -> 0.503 -> 0.749 -> 0.879, more than
    doubling, while the uniform column falls 0.793 -> 0.420. Decay only pays
    when old content is genuinely obsolete, and a uniform regime has nothing
    obsolete in it by construction.

So the gates trade off along the TEMPORAL axis -- momentum broadens retention,
weight decay sharpens it toward recent -- and that explains the paper's
published ordering (weight decay > momentum, Table 5): language modelling is
heavily recency-weighted, the regime where decay dominates. Neither gate is
universally good, which is why a single-regime sweep will always call one of
them harmful.

Run: python -m scripts.prototype_titans_gate_regimes
"""

import time

import numpy as np

from model.titans_memory import TitansDeepMemory, TitansMemory

D = 48
N_PAIRS = 4
N_FILLER = 64
N_VALUES = 64
EPISODES = 400
SEEDS = (1001, 1002, 1003)
ORDERS = ("random", "first", "last")


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def run(make, seed, *, order="random", episodes=EPISODES):
    """Recall@1 over the value pool. `order` places the probed pairs."""
    rng = np.random.default_rng(seed)
    values = _unit(rng, N_VALUES, D)
    mem = make(seed + 7)
    total_writes = N_FILLER + N_PAIRS
    correct = total = 0
    for _ in range(episodes):
        mem.reset()
        keys = _unit(rng, N_PAIRS, D)
        vidx = rng.choice(N_VALUES, size=N_PAIRS, replace=False)
        fk = _unit(rng, N_FILLER, D)
        fv = values[rng.choice(N_VALUES, size=N_FILLER)]
        if order == "random":
            slots = np.sort(rng.choice(total_writes, size=N_PAIRS, replace=False))
        elif order == "first":
            slots = np.arange(N_PAIRS)
        else:
            slots = np.arange(N_FILLER, total_writes)

        pi = fi = 0
        for pos in range(total_writes):
            if pi < N_PAIRS and pos == slots[pi]:
                mem.write_kv(keys[pi], values[vidx[pi]])
                pi += 1
            else:
                mem.write_kv(fk[fi % N_FILLER], fv[fi % N_FILLER])
                fi += 1
        for i in range(N_PAIRS):
            y = mem.read_q(keys[i])
            correct += int(int(np.argmax(values @ y)) == int(vidx[i]))
            total += 1
    return correct / total


def _matrix(seed, alpha, eta):
    return TitansMemory(
        D,
        D,
        const_alpha=alpha,
        const_eta=eta,
        theta_scale=max(1e-3, 1.0 - eta),
        rng=np.random.default_rng(seed),
    )


def _deep(seed, alpha, eta):
    return TitansDeepMemory(
        D,
        n_layers=2,
        const_alpha=alpha,
        const_eta=eta,
        theta_scale=max(1e-3, 1.0 - eta),
        rng=np.random.default_rng(seed),
    )


CELLS = (
    (0.0, 0.0),
    (0.0, 0.5),
    (0.0, 0.9),
    (0.0, 0.95),
    (0.005, 0.9),
    (0.01, 0.9),
    (0.05, 0.9),
    (0.1, 0.9),
    (0.01, 0.0),
    (0.05, 0.0),
)


def main():
    t0 = time.time()
    print(f"D={D} pairs={N_PAIRS} filler={N_FILLER} pool={N_VALUES} chance={1.0 / N_VALUES:.4f}")
    print(f"{EPISODES} episodes x {len(SEEDS)} seeds; theta_scale = 1-eta throughout\n")
    for label, mk in (("matrix (L=1)", _matrix), ("deep (L=2, MLP)", _deep)):
        print(f"=== {label} ===")
        print(f"{'alpha':>6} {'eta':>5} " + " ".join(f"{o:>9}" for o in ORDERS))
        for alpha, eta in CELLS:
            # bind mk/alpha/eta as defaults: a bare closure over the loop
            # variables would read whatever they hold when it is finally called.
            def build(seed, mk=mk, alpha=alpha, eta=eta):
                return mk(seed, alpha, eta)

            row = [float(np.mean([run(build, sd, order=o) for sd in SEEDS])) for o in ORDERS]
            print(f"{alpha:>6} {eta:>5} " + " ".join(f"{v:>9.4f}" for v in row), flush=True)
        print()
    print(f"total {time.time() - t0:.0f}s")
    print("random is the verdict column; first/last expose the temporal bias each")
    print("gate induces, and are not comparable to it (different interference load).")


if __name__ == "__main__":
    main()
