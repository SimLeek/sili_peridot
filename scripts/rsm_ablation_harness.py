"""Standing benchmark + harness for the RSM -> Titans ablation ladder.

Exists because the previous investigation's two failure modes were an
under-powered/confounded harness and sweeping before measuring (JOURNAL.md
2026-11-12). Every flat sweep there was flat for one of those reasons, not
because the mechanisms were neutral. So this harness enforces three things the
earlier ad-hoc scripts did not:

1. POWER IS MEASURED, NOT ASSUMED. `calibrate()` runs the base arm across many
   seeds, reports the per-seed standard deviation, and derives the minimum
   detectable effect (MDE) for a given seed count. Any comparison smaller than
   the MDE is reported as indistinguishable rather than as a result. The earlier
   long-gap probe had an MDE wider than any effect worth finding, which is why
   it was uninformative rather than negative.

2. SCHEDULE DIFFERENCES ARE MADE VISIBLE, NOT ELIMINATED. The confound that
   invalidated most of the earlier sweeps was that enabling a mechanism silently
   changed how often weights were updated (every token position instead of only
   at targets), so the accuracy change was attributed to the mechanism rather
   than to the schedule. Some Titans features (test-time memory updates) DO
   legitimately change the schedule, so the fix is not to forbid it but to count
   `n_backward`/`n_optstep` per arm and surface them next to the accuracy. A
   comparison across arms with differing counts is flagged.

3. A MECHANISM MEASUREMENT ACCOMPANIES EVERY ACCURACY NUMBER. Both real findings
   in the earlier study came from measuring the mechanism directly (per-tick
   state change exposed the rmsnorm bug; a no-op-mask control exposed the
   schedule confound), never from accuracy. So each arm also reports gate firing
   rate and the fraction of ticks whose memory update was bit-exactly zero.

Benchmark task: next-token prediction on a FIXED, memorizable generating process
with run-length redundancy -- the regime RSM/Titans claim (compressible stream,
most ticks uninformative) and the regime the Bend proof favours
(`all_inactive_is_exact` needs inactive ticks). Calibrated so the base arm lands
mid-range (~0.57 at 400 steps, chance 1/VOCAB = 0.031): neither floored nor
saturated, so it can resolve movement in either direction.

Deliberately NOT MQAR: recall capacity there grows with state size (Zoology,
arXiv:2312.04927) while these mechanisms trade state away, and our own sweep
reproduced that scaling law. See the model's rsm_* docstring.

Usage:
    python -m scripts.rsm_ablation_harness calibrate       # measure sd, get MDE
    python -m scripts.rsm_ablation_harness arm base 1001   # one arm, one seed
"""

import json
import sys
import time

import numpy as np

from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum, predicted_token
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT

VOCAB = 32
EMBED_WIDTH = 12
COLUMN_NEURONS = 8
NUM_TILES = 16
MAX_WEIGHTS = 4000
NUM_CPUS = 2
PEAK_LR = 0.015
SEQ_LEN = 64
RUN = 8  # redundancy (RUN-1)/RUN = 87.5%
PERIOD = 12
TRAIN_STEPS = 400
EVAL_SEQS = 60  # eval noise kept well under per-seed noise; see calibrate()
STATE_WIDTH = EMBED_WIDTH * COLUMN_NEURONS

BASE_PATTERN = [(7 * i + 3) % VOCAB for i in range(PERIOD)]

# Ablation ladder. Each rung changes exactly ONE thing from `base`; `titans_full`
# is the combination arm, present because features can interact -- this study
# already found one destructive interaction (decoder x mask: each neutral alone,
# 0.151 together), so a synergistic one is equally plausible and a ladder of
# nulls would hide it. Keys map to ToyTileRecurrenceRMT kwargs plus harness-only
# flags; unimplemented Titans rungs are declared here and raise until built, so
# the ladder is self-documenting rather than silently partial.
ARMS = {
    "base": {},
    "sparse_only": {"rsm_memory_write_enable": True, "k_frac": 0.25},
    "novelty_gate": {"rsm_memory_write_enable": True, "k_frac": 0.25, "rsm_novelty_threshold": 0.5},
    # --- Titans rungs (not yet implemented; see JOURNAL.md's ladder) ---
    "grad_surprise": {"_todo": "rung 1: reuse _update_layer_surprise (||dy||^2 EMA) to gate the write"},
    "surprise_momentum": {"_todo": "rung 2: two-timescale momentary + past surprise (an eligibility trace)"},
    "adaptive_forget": {"_todo": "rung 3: decay memory instead of pure additive accumulation"},
    "persistent_mem": {"_todo": "rung 5: data-independent learnable memory tokens"},
    "titans_full": {"_todo": "combination arm: rungs 1+2+3+5 together"},
}


def make_sequence(offset: int) -> np.ndarray:
    return np.array(
        [BASE_PATTERN[((i // RUN) + offset) % PERIOD] for i in range(SEQ_LEN)],
        dtype=np.int64,
    )


def _window(embed_table, tokens, i):
    return np.stack([embed_table[tokens[max(0, i - (NUM_TILES - 1) + j)]] for j in range(NUM_TILES)], axis=0)


def run_arm(arm: str, seed: int, num_memory_slots: int = 2) -> dict:
    """One arm, one seed. Returns accuracy plus the schedule counts and mechanism
    diagnostics needed to interpret it."""
    spec = dict(ARMS[arm])
    if "_todo" in spec:
        raise NotImplementedError(f"arm {arm!r} not built yet: {spec['_todo']}")
    k_frac = spec.pop("k_frac", None)

    rng = np.random.default_rng(seed)
    kwargs = dict(spec)
    if k_frac is not None:
        kwargs["rsm_k_active"] = max(1, round(k_frac * num_memory_slots * STATE_WIDTH))
    model = ToyTileRecurrenceRMT(
        VOCAB,
        EMBED_WIDTH,
        COLUMN_NEURONS,
        NUM_TILES,
        num_memory_slots,
        MAX_WEIGHTS,
        num_cpus=NUM_CPUS,
        qkvo_norm_enable=True,
        rng=rng,
        **kwargs,
    )
    rsm_on = bool(kwargs.get("rsm_memory_write_enable"))
    opt = AdamOptimizer()
    embed_table = (rng.standard_normal((VOCAB, EMBED_WIDTH)) * 0.3).astype(np.float32)

    n_backward = n_optstep = 0
    t0 = time.time()
    for step in range(TRAIN_STEPS):
        tokens = make_sequence(step % PERIOD)
        memory = np.zeros((num_memory_slots, STATE_WIDTH), dtype=np.float32)
        if rsm_on:
            model.rsm_reset_sequence()
        for i in range(SEQ_LEN - 1):
            memory, logits, aux = model.step(_window(embed_table, tokens, i), memory, PEAK_LR)
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, int(tokens[i + 1]))])
            if aux is not None:
                loss = loss + aux
            loss.backward()
            n_backward += 1
            clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
            opt.step(model.parameters_for_optimizer(), lr=PEAK_LR)
            n_optstep += 1

    correct = total = gate_fired = inactive = 0
    for e in range(EVAL_SEQS):
        tokens = make_sequence(e % PERIOD)
        memory = np.zeros((num_memory_slots, STATE_WIDTH), dtype=np.float32)
        if rsm_on:
            model.rsm_reset_sequence()
        for i in range(SEQ_LEN - 1):
            prev = memory
            memory, logits, _aux = model.step(_window(embed_table, tokens, i), memory, 0.0)
            correct += int(predicted_token(logits, NUM_TILES - 1) == int(tokens[i + 1]))
            total += 1
            if rsm_on:
                gate_fired += int(getattr(model, "last_rsm_gate", 1.0) != 0.0)
                inactive += int(np.abs(memory - prev).max() == 0.0)
    return {
        "arm": arm,
        "seed": seed,
        "slots": num_memory_slots,
        "acc": correct / total,
        "chance": 1.0 / VOCAB,
        "n_backward": n_backward,
        "n_optstep": n_optstep,
        "gate_fire_rate": (gate_fired / total) if rsm_on else None,
        "inactive_tick_frac": (inactive / total) if rsm_on else None,
        "elapsed_s": time.time() - t0,
    }


def mde(sd: float, n_seeds: int, k: float = 2.8) -> float:
    """Minimum detectable effect between two arms of `n_seeds` each, at roughly
    95% confidence (k=2.8 ~= 2*sqrt(2) for a two-sample comparison of means).
    Report any gap below this as indistinguishable, not as a result."""
    return k * sd / np.sqrt(n_seeds)


def calibrate(n_seeds: int = 8) -> None:
    """Measure the base arm's per-seed spread and print the MDE table, so seed
    count is chosen from data rather than guessed."""
    rows = [run_arm("base", 1000 + s) for s in range(n_seeds)]
    accs = np.array([r["acc"] for r in rows])
    sd = float(accs.std(ddof=1))
    print(f"base arm: n={n_seeds} mean={accs.mean():.4f} sd={sd:.4f} chance={rows[0]['chance']:.4f}")
    print(f"per-seed: {', '.join(f'{a:.3f}' for a in accs)}")
    print(f"schedule: n_backward={rows[0]['n_backward']} n_optstep={rows[0]['n_optstep']} (must match across arms)")
    print("\nminimum detectable effect vs seed count:")
    for n in (2, 4, 8, 16, 32):
        print(f"  n={n:>2} seeds -> MDE {mde(sd, n):.4f}")
    print("\nThe gate's observed edge in the pilot was ~0.05; size seeds so MDE < that.")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        calibrate(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif len(sys.argv) > 3 and sys.argv[1] == "arm":
        r = run_arm(sys.argv[2], int(sys.argv[3]))
        print(json.dumps(r, indent=2))
    else:
        print(__doc__)
        print("arms:", ", ".join(ARMS))


if __name__ == "__main__":
    main()
