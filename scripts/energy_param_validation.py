"""Standalone full validation of an ALTERNATE EnergyDynamics config on
the 4 dense arms x 5 seeds x 15000 steps -- built as a standalone
script (constructs models directly, custom energy_kwargs) rather than
adding CLI-exposed energy params to train_tile_curriculum.py, since a
separate full sweep against that file's CURRENT (unmodified) state was
already running in the background when this was written and editing
it would have contaminated that run. Reuses train_tile_curriculum.py's
own helpers (ARMS, current_seq_len, evaluate, etc.) via plain import
-- read-only, safe to run in a fully separate process alongside
anything else using the same file.

Candidate chosen from a quick 3-seed/1500-step skip-rate probe (not
committed -- see JOURNAL.md): activation_cost=0.02 (4x default),
precision=0.01 (10x default) gave the lowest observed skip rate
(0.33%) among several tested variants, on top of the project's
existing already-tuned-low ENERGY_KWARGS baseline.

Usage: python3 scripts/energy_param_validation.py [n_steps] [n_seeds]
"""

from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, ".")

import numpy as np
from sili import _cpu

from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from scripts.train_tile_curriculum import (
    ARMS,
    COLUMN_NEURONS,
    EMBED_WIDTH,
    MAX_GRAD_NORM,
    MAX_WEIGHTS_PER_LAYER,
    NUM_TILES,
    PEAK_LR,
    STEPS_PER_STAGE_DEFAULT,
    VOCAB,
    WARMUP_STEPS,
    AdamOptimizer,
    _build_tile_window,
    clip_grad_norm_,
    cross_entropy_sum,
    current_seq_len,
    evaluate,
    generate_copy_sequence,
    lr_schedule,
)

DENSE_ARMS = {
    "base4_dense": "true_multi_digit_deterministic_base4_dense",
    "base6_dense": "true_multi_digit_deterministic_base6_dense",
    "base12_dense": "true_multi_digit_deterministic_dense",
    "base24_dense": "true_multi_digit_deterministic_base24_dense",
}
SEEDS = [1000, 1001, 1002, 1003, 1004]

ENERGY_KWARGS_CANDIDATE = {
    "drive": 0.00535,
    "activation_cost": 0.02,
    "precision": 0.01,
    "density": 0.005,
    "p": 0.995,
    "reactivity": 0.0001,
}

DENSE_NOFIX_REFERENCE = {
    "base4_dense": 0.0938,
    "base6_dense": 0.1171,
    "base12_dense": 0.1050,
    "base24_dense": 0.0979,
}
SPARSE_REFERENCE = {
    "base4_dense": 0.6417,
    "base6_dense": 0.6929,
    "base12_dense": 0.7296,
    "base24_dense": 0.6775,
}


def run_one(arm_key: str, seed: int, n_steps: int, checkpoint_every: int) -> float:
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    model_rng = np.random.default_rng(seed)
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    mlp_hidden = state_width * 2
    model = ToyTileRecurrenceRealFP4(
        VOCAB,
        EMBED_WIDTH,
        COLUMN_NEURONS,
        mlp_hidden,
        NUM_TILES,
        MAX_WEIGHTS_PER_LAYER,
        num_cpus=2,
        disldo_cls=ARMS[DENSE_ARMS[arm_key]],
        use_energy=True,
        energy_kwargs=ENERGY_KWARGS_CANDIDATE,
        use_attention=True,
        o_proj_depth=1,
        rng=model_rng,
        clip_range=6.0,
    )
    opt = AdamOptimizer()
    rng = np.random.RandomState(seed)
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

    last_acc = 0.0
    for step in range(1, n_steps + 1):
        seq_len = current_seq_len(step, STEPS_PER_STAGE_DEFAULT)
        lr = lr_schedule(step, n_steps, PEAK_LR, WARMUP_STEPS)
        tokens, pairs = generate_copy_sequence(rng, VOCAB, seq_len)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, lr)
            if i in targets:
                loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                if aux is not None:
                    loss = loss + aux
                loss.backward()
                clip_grad_norm_(model.parameters_for_optimizer(), MAX_GRAD_NORM)
                opt.step(model.parameters_for_optimizer(), lr=lr)
        if step % checkpoint_every == 0:
            last_acc = evaluate(model, rng, embed_table, seq_len)
    return last_acc


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else len(SEEDS)
    checkpoint_every = max(n_steps // 20, 50)
    seeds = SEEDS[:n_seeds]

    results = {arm: {} for arm in DENSE_ARMS}
    t0 = time.time()
    for arm_key in DENSE_ARMS:
        for seed in seeds:
            acc = run_one(arm_key, seed, n_steps, checkpoint_every)
            results[arm_key][seed] = acc
            print(f"{arm_key:<12} seed={seed}  final_acc={acc:.4f}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    print("\narm            mean    std     per-seed  (activation_cost=0.02, precision=0.01)")
    for arm_key in DENSE_ARMS:
        per_seed = [results[arm_key][s] for s in seeds]
        print(
            f"{arm_key:<14} {statistics.mean(per_seed):.4f}  "
            f"{statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0:.4f}  "
            f"{[round(v, 4) for v in per_seed]}"
        )

    print("\ncandidate-energy vs no-fix vs sparse-echo:")
    print(f"{'arm':<14} {'candidate mean':<16} {'nofix':<10} {'sparse':<10}")
    for arm_key in DENSE_ARMS:
        per_seed = [results[arm_key][s] for s in seeds]
        print(
            f"{arm_key:<14} {statistics.mean(per_seed):<16.4f} "
            f"{DENSE_NOFIX_REFERENCE[arm_key]:<10.4f} {SPARSE_REFERENCE[arm_key]:<10.4f}"
        )


if __name__ == "__main__":
    main()
