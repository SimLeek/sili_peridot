"""Dense sandbox probe -- merges the former run_sandbox_dense_probe.py
and run_sandbox_dense_probe_cpu8.py (identical except NUM_CPUS=8 vs the
true default) into one file via --num-cpus. Monkeypatches
train_mqar_curriculum._build_targets to return raw mqar_pairs
unchanged (bypasses the usual next-token auxiliary targets), K_START=2.

Usage: python3 scripts/sandbox_arm_launchers/run_sandbox_dense_probe.py [--num-cpus N]
"""

import argparse
import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as tmc
from scripts.sandbox_arm_launchers.launcher_common import make_log_fn

tmc.K_START = 2


def _q(tokens, mqar_pairs, num_kv_pairs):
    return dict(mqar_pairs)


tmc._build_targets = _q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-cpus", type=int, default=None, dest="num_cpus")
    args = parser.parse_args()
    if args.num_cpus is not None:
        tmc.NUM_CPUS = args.num_cpus

    r = tmc.train_curriculum(
        precision="fp32",
        max_steps=3000,
        seed=0,
        peak_lr=0.015,
        num_tiles=8,
        k_max=8,
        log_every=500,
        log_fn=make_log_fn(),
        embed_width=36,
        k_first_target=3,
        wrong_streak_threshold=float("inf"),
    )
    print(f"\nFINAL steps_per_sec={r['steps_per_sec']:.2f} elapsed_s={r['elapsed_s']:.0f}", flush=True)


if __name__ == "__main__":
    main()
