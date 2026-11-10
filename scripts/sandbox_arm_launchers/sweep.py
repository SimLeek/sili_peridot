"""Generic sweep runner, replacing knee_margin_sweep.py,
armc_gate_density_lr_seed_sweep.py (and usable for similar future
sweeps) -- a base preset plus a list of override dicts, run
sequentially, printing a final comparison table. Sweep-axis
definitions live in launcher_configs.SWEEPS.

width_scaling_knee_probe.py/width_scaling_knee_probe_matched.py are
NOT folded in here -- they vary a module-level constant
(m.COLUMN_NEURONS), not a train_curriculum kwarg, and capture bespoke
per-layer model internals (model.last_input_selection/
model._x_knee_r_target) that don't fit this generic shape cleanly;
they stay as their own small scripts.

Usage: python3 scripts/sandbox_arm_launchers/sweep.py <sweep_name>
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")
from scripts.sandbox_arm_launchers.launch import run_preset
from scripts.sandbox_arm_launchers.launcher_configs import SWEEPS


def run_sweep(name: str) -> list[dict]:
    if name not in SWEEPS:
        available = ", ".join(sorted(SWEEPS))
        raise KeyError(f"Unknown sweep {name!r}. Available sweeps:\n{available}")
    base_preset, overrides_list = SWEEPS[name]
    results = []
    for i, overrides in enumerate(overrides_list):
        print(f"\n########## sweep={name} run {i + 1}/{len(overrides_list)} overrides={overrides} ##########")
        r = run_preset(base_preset, overrides=overrides)
        results.append({"overrides": overrides, "result": r})

    print(f"\n=== sweep={name} summary ===")
    for entry in results:
        r = entry["result"]
        print(f"  {entry['overrides']}  ->  final_vocab={r['final_vocab']} final_k={r['final_k']}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", nargs="?", help="sweep name from launcher_configs.SWEEPS")
    args = parser.parse_args()
    if args.sweep is None:
        print("Available sweeps:")
        for name in sorted(SWEEPS):
            print(f"  {name}")
        return
    run_sweep(args.sweep)


if __name__ == "__main__":
    main()
