"""Does INTRINSIC gate adaptation beat the best constant gate in Titans' LMM?

Why this exists: the constant-gate sweep (JOURNAL.md 2026-11-12) failed to
reproduce Titans' published ablation ordering -- forgetting degraded recall
monotonically (alpha 0.0 -> 0.2 took 0.360 -> 0.019 at eta=0) and momentum cost
more than half the recall at alpha=0. The best configuration was BOTH MECHANISMS
OFF, which is the opposite of the paper.

That is not evidence against the mechanisms, because a constant gate is not a
weaker learned gate, it is a different mechanism. This task mixes a few must-keep
pairs with many junk writes, so ONE constant alpha decays both at the same rate
and no value of it can be right. The paper says as much: "a data-dependent eta
can control if memory needs to: (1) ignore the last surprise ... or (2) fully
incorporate" it. The dependence is the mechanism; its existence is not.

So this measures whether the dependence can be recovered WITHOUT the paper's
outer loop and without an oracle, from statistics the memory already computes:

  alpha_t = alpha_max * sigmoid(gamma * (||M||_F/sqrt(d) / u_target - 1))
  eta_t   = eta_max * max(0, cos(S_{t-1}, -grad_t))
  retrieval-gated alpha: decay column j in inverse proportion to query traffic

Honest limit, stated before the numbers: against IID RANDOM junk no intrinsic
signal can separate "novel junk" from "novel important" -- both are unrelatable
noise, so a ceiling applies to every arm here. What IS intrinsically separable is
recurrence and retrieval traffic, which is why the redundant-filler column and
the retrieval-gated arm exist. Expect the gates to earn their keep there, not in
the iid column.

Two filler regimes, because the capacity result depends entirely on which one:
  * NOVEL filler (iid random keys) -- consumes capacity, each write claims fresh
    directions. This is the worst case, and an untrained encoder's case.
  * REPEATED filler (a small reused set) -- nearly free, writes collide and
    overwrite. This is what a TRAINED encoder converts novel input into: "novel"
    is a property of the representation, not of the input, so these numbers
    bracket the real behaviour rather than the novel column alone.

Run: python -m scripts.prototype_titans_adaptive_gates
"""

import time

import numpy as np

from model.titans_memory import TitansMemory

D = 48
N_PAIRS = 4
N_FILLER = 64
N_VALUES = 64  # candidate pool for the retrieval decision => chance 1/64
N_REPEAT_SET = 4  # distinct filler pairs in the redundant regime
EPISODES = 500
SEEDS = (1001, 1002, 1003)


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def run_arm(kwargs, *, seed, repeated_filler, episodes=EPISODES, read_every=0):
    """Recall@1 over the value pool, averaged across episodes and probes.

    `read_every` interleaves probe queries INTO the filler phase. Needed because
    with reads only at the end, read_usage is identically zero for the whole
    write phase, so retrieval-gated forgetting degenerates to uniform decay and
    its hypothesis goes untested. Interleaving is also the realistic stream: a
    sequence model re-queries what it is actually using, so retrieval traffic is
    environment-supplied, not an oracle label on which keys matter."""
    rng = np.random.default_rng(seed)
    values = _unit(rng, N_VALUES, D)
    mem = TitansMemory(D, D, rng=np.random.default_rng(seed + 7), **kwargs)

    correct = total = 0
    alphas, etas = [], []
    for _ in range(episodes):
        mem.reset()
        keys = _unit(rng, N_PAIRS, D)
        vidx = rng.choice(N_VALUES, size=N_PAIRS, replace=False)
        for k, vi in zip(keys, values[vidx], strict=True):
            mem.write_kv(k, vi)

        if repeated_filler:
            fk = _unit(rng, N_REPEAT_SET, D)
            fv = values[rng.choice(N_VALUES, size=N_REPEAT_SET)]
        else:
            fk = _unit(rng, N_FILLER, D)
            fv = values[rng.choice(N_VALUES, size=N_FILLER)]
        n_set = N_REPEAT_SET if repeated_filler else N_FILLER
        for t in range(N_FILLER):
            mem.write_kv(fk[t % n_set], fv[t % n_set])
            alphas.append(mem.last["alpha"])
            etas.append(mem.last["eta"])
            if read_every and t % read_every == read_every - 1:
                mem.read_q(keys[(t // read_every) % N_PAIRS])

        for k, want in zip(keys, vidx, strict=True):
            y = mem.read_q(k)
            n = float(np.linalg.norm(y))
            got = int(np.argmax(values @ y)) if n > 1e-8 else -1
            correct += int(got == int(want))
            total += 1
    return correct / total, float(np.mean(alphas)), float(np.mean(etas))


def run_obsolescence(kwargs, *, seed, episodes=EPISODES, n_written=16, n_probed=4):
    """Same memory, but OLD CONTENT GOES STALE: `n_written` pairs are written
    across the stream and only the last `n_probed` are ever queried.

    This exists because the main task above cannot possibly reward forgetting:
    every pair there is needed at the final probe, so eviction can only destroy
    something useful and no policy -- adaptive, retrieval-gated, or oracle --
    beats keeping everything. Weight decay can only pay when old content stops
    being wanted, which is the regime it occupies in language modelling. If
    forgetting does not win HERE, the mechanism has no regime in this harness."""
    rng = np.random.default_rng(seed)
    values = _unit(rng, N_VALUES, D)
    mem = TitansMemory(D, D, rng=np.random.default_rng(seed + 7), **kwargs)
    gap = N_FILLER // n_written

    correct = total = 0
    for _ in range(episodes):
        mem.reset()
        keys = _unit(rng, n_written, D)
        vidx = rng.choice(N_VALUES, size=n_written, replace=False)
        fk = _unit(rng, N_FILLER, D)
        fv = values[rng.choice(N_VALUES, size=N_FILLER)]
        for t in range(n_written):
            mem.write_kv(keys[t], values[vidx[t]])
            for j in range(gap):
                mem.write_kv(fk[(t * gap + j) % N_FILLER], fv[(t * gap + j) % N_FILLER])
        for t in range(n_written - n_probed, n_written):
            y = mem.read_q(keys[t])
            got = int(np.argmax(values @ y)) if float(np.linalg.norm(y)) > 1e-8 else -1
            correct += int(got == int(vidx[t]))
            total += 1
    return correct / total


ARMS = {
    # controls: the two reference points from the constant-gate sweep
    "const best (a=0,e=0)": {"const_alpha": 0.0, "const_eta": 0.0},
    "const paper-ish (a=.01,e=.9)": {"const_alpha": 0.01, "const_eta": 0.9},
    # the question: intrinsic data-dependence, no outer loop, no oracle
    "adaptive gates": {"adaptive_gates": True, "alpha_max": 0.1},
    "adaptive, alpha_max=0.3": {"adaptive_gates": True, "alpha_max": 0.3},
    "adaptive + retrieval-gated": {
        "adaptive_gates": True,
        "alpha_max": 0.3,
        "retrieval_gated_forget": True,
    },
    # isolates the retrieval rule from the saturation rule
    "retrieval-gated only (a=.05)": {
        "const_alpha": 0.05,
        "const_eta": 0.0,
        "retrieval_gated_forget": True,
    },
}


def main():
    t0 = time.time()
    chance = 1.0 / N_VALUES
    print(f"D={D} pairs={N_PAIRS} filler={N_FILLER} pool={N_VALUES} => chance={chance:.4f}")
    print(f"{EPISODES} episodes x {len(SEEDS)} seeds per cell\n")
    hdr = f"{'arm':>30} {'novel':>14} {'repeated':>9} {'novel+rd':>9} {'stale':>7} {'<alpha>':>8} {'<eta>':>7}"
    print(hdr)
    for name, kw in ARMS.items():
        novel = [run_arm(kw, seed=s, repeated_filler=False) for s in SEEDS]
        rep = [run_arm(kw, seed=s, repeated_filler=True) for s in SEEDS]
        rd = [run_arm(kw, seed=s, repeated_filler=False, read_every=8) for s in SEEDS]
        stale = [run_obsolescence(kw, seed=s) for s in SEEDS]
        a_novel = float(np.mean([r[0] for r in novel]))
        a_rep = float(np.mean([r[0] for r in rep]))
        a_rd = float(np.mean([r[0] for r in rd]))
        a_st = float(np.mean(stale))
        g_a = float(np.mean([r[1] for r in novel]))
        g_e = float(np.mean([r[2] for r in novel]))
        sd = float(np.std([r[0] for r in novel], ddof=1))
        print(
            f"{name:>30} {a_novel:>9.4f}+-{sd:.3f} {a_rep:>9.4f} {a_rd:>9.4f} {a_st:>7.4f} {g_a:>8.4f} {g_e:>7.4f}",
            flush=True,
        )
    print(f"\ntotal {time.time() - t0:.0f}s")
    print("novel+rd interleaves a probe read every 8 filler writes, the only")
    print("column where retrieval-gated forgetting has signal to act on.")
    print("stale = 16 pairs written, only the last 4 probed: the ONLY column")
    print("where forgetting can pay, since the rest stay wanted to the end.")


if __name__ == "__main__":
    main()
