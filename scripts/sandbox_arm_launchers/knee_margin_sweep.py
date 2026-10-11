"""Calibration sweep, not a verdict: the knee-elbow detector tells us
WHERE the natural energy cutoff is, but not how far above it a real
training run needs to sit to preserve enough signal to actually learn.
x_r_target_auto_margin=0.05 (the one value tried so far, in the
standard-curriculum and skip-k1 full runs) is a single untested guess,
not a swept-and-chosen setting -- its poor early progress says nothing
about whether knee-adaptive x_r_target works, only that this one
margin might be wrong. Sweep several margins, short budget each, same
Arm C backward mechanism throughout, k_first_target=3 curriculum
(includes k=1 -- the full realistic setting, not the isolated k=2
probe) so results are comparable to the real full runs already
launched.
"""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

STEPS = 4000
SEED = 1000
EMBED_WIDTH = 36
K_FIRST_TARGET = 3
MARGIN_SWEEP = (0.0, 0.03, 0.05, 0.08, 0.12, 0.2)


def run_margin(margin: float) -> None:
    r = m.train_curriculum(
        "fp32",
        STEPS,
        SEED,
        0.015,
        16,
        10,
        embed_width=EMBED_WIDTH,
        wrong_streak_threshold=100000000,
        k_first_target=K_FIRST_TARGET,
        dy_time_gate_cutoff=0.3,
        x_r_target_auto=True,
        x_r_target_auto_margin=margin,
        log_every=STEPS,
    )
    print(f"\n########## margin={margin} ##########")
    print(
        f"  final: vocab={r['final_vocab']} k={r['final_k']} phase={r['final_phase']} "
        f"steps/sec={r['steps_per_sec']:.2f} peak={r['peak_stage']}"
    )
    print(f"  stage_history: {r['stage_history']}")


if __name__ == "__main__":
    for margin in MARGIN_SWEEP:
        run_margin(margin)
