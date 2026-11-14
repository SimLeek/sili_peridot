"""Titans' two gates, each measured in the regime it actually targets.

Supersedes the gate numbers in `prototype_titans_adaptive_gates.py`, which
were confounded twice: eta also scaled the effective step by 1/(1-eta)
(fixed with theta_scale = 1-eta), and the task wrote probed pairs FIRST,
which high eta privileges. Both confounds, the derivation, and the regime
results are in JOURNAL.md 2026-11-14.

Results (matrix memory, 400 eps x 3 seeds, chance 0.0156): momentum helps
when probes are uniform (0.665 -> 0.871 as eta 0 -> 0.95); forgetting
helps when they are recent (the "last" column runs 0.387 -> 0.879 down the
alpha rows at eta=0.9, while uniform falls 0.793 -> 0.420). The gates
trade off along the temporal axis, so neither is universally good.

order=first/last/random are different INTERFERENCE LOADS, not three
samples of one quantity -- so "random" can exceed both controls.

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
