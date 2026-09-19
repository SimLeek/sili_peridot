"""Statistical check on test 3's grid crossover, not a single-seed
"controlled" re-run -- direct correction: for a sine-wave gate whose
period is comparable to the run length, a single fixed seed just picks
a different arbitrary phase realization, it doesn't control anything.
Whether the crossover (denser+higher-LR and sparser+lower-LR both
reaching vocab=126/k=3, both cross-combinations stuck at k=2) is real
or noise needs MULTIPLE independent seeds per cell, not one -- see
feedback_statistical_power_not_seeding memory.

Same 2x2 grid (cutoff in {0.0, 0.6} x peak_lr in {0.015, 0.01}) x 3
seeds = 12 runs, SHORT budget (25000 steps -- comfortably covers the
vocab=64/k=3 milestone every original grid cell reached between steps
12,972-28,083, much cheaper than the full 100k used for the original
single-seed grid) since full mastery (vocab=126/k=3) took up to 43,905
steps for the slowest original cell -- too expensive to require at
this stage. Reports step-of-vocab=64/k=3 (and further progress if any)
per (cutoff, lr, seed) so the spread WITHIN each cell can be compared
against the gap BETWEEN cells."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

STEPS = 25000
SEEDS = (1000, 2000, 3000)
GRID = (
    ("gatemid_unscaled", 0.0, 0.015),
    ("gatemid_scaled", 0.0, 0.01),
    ("gatesparse_unscaled", 0.6, 0.015),
    ("gatesparse_scaled", 0.6, 0.01),
)


def run_one(name: str, cutoff: float, lr: float, seed: int) -> None:
    r = m.train_curriculum(
        "fp32",
        STEPS,
        seed,
        lr,
        16,
        10,
        embed_width=36,
        wrong_streak_threshold=100000000,
        k_first_target=3,
        dy_time_gate_cutoff=cutoff,
        log_every=STEPS,
    )
    print(f"\n########## {name} seed={seed} cutoff={cutoff} lr={lr} ##########", flush=True)
    print(
        f"  final: vocab={r['final_vocab']} k={r['final_k']} phase={r['final_phase']} "
        f"steps/sec={r['steps_per_sec']:.2f} peak={r['peak_stage']}",
        flush=True,
    )
    print(f"  stage_history: {r['stage_history']}", flush=True)


if __name__ == "__main__":
    for name, cutoff, lr in GRID:
        for seed in SEEDS:
            run_one(name, cutoff, lr, seed)
    print("\nDONE_ARMC_GATE_DENSITY_LR_SEED_SWEEP", flush=True)
