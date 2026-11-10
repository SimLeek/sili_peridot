"""Generic single-run launcher, replacing ~68 near-duplicate launch_*.py
files that each hardcoded one train_curriculum(...) call differing from
a sibling by only a few kwargs (e.g. "v19b"/"v19c" were v19 with only
`seed` changed). The actual historical configs live in
launcher_configs.PRESETS -- this file is the one place that knows HOW
to run any of them; each preset is just data.

Usage:
    python3 scripts/sandbox_arm_launchers/launch.py <preset_name>
        [--seed N] [--max-steps N] [--num-cpus N] [--k-start N]

Run with no preset name to list all available presets.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m
from scripts.sandbox_arm_launchers.launcher_common import make_log_fn, make_trajectory_log_fn, print_final_summary
from scripts.sandbox_arm_launchers.launcher_configs import PRESETS


def run_preset(name: str, overrides: dict | None = None) -> dict:
    """Runs one named preset, applying any CLI-style overrides on top.
    Returns train_curriculum's own result dict. `overrides` may include
    the special keys "num_cpus"/"k_start" (module-level knobs, not
    train_curriculum kwargs) alongside any real train_curriculum kwarg."""
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset {name!r}. Available presets:\n{available}")
    cfg = dict(PRESETS[name])
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    # Defaults here MUST match train_mqar_curriculum.py's own true
    # module-level defaults (NUM_CPUS=16 via train_mqar_rmt_reference.py,
    # K_START=1) -- NOT a commonly-seen override value. Most presets
    # explicitly override both (almost always num_cpus=4), but a preset
    # that doesn't touch one (e.g. launch_sparse_energy_remote.py never
    # overrode either; launch_arm_c_plus_knee.py never touched K_START)
    # must still reproduce the real original default, not a guessed one.
    m.NUM_CPUS = cfg.pop("num_cpus", 16)
    m.K_START = cfg.pop("k_start", 1)

    trajectory_log_every = cfg.get("trajectory_log_every")
    log_fn = cfg.pop("log_fn", None) or make_log_fn()
    trajectory_log_fn = cfg.pop("trajectory_log_fn", None)
    if trajectory_log_fn is None and trajectory_log_every is not None:
        trajectory_log_fn = make_trajectory_log_fn()

    print(f"# launch.py --preset {name}  cfg={cfg}", flush=True)
    r = m.train_curriculum(log_fn=log_fn, trajectory_log_fn=trajectory_log_fn, **cfg)
    print_final_summary(r, label=name)
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", help="preset name from launcher_configs.PRESETS")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    parser.add_argument("--num-cpus", type=int, default=None, dest="num_cpus")
    parser.add_argument("--k-start", type=int, default=None, dest="k_start")
    args = parser.parse_args()

    if args.preset is None:
        print("Available presets:")
        for name in sorted(PRESETS):
            print(f"  {name}")
        return

    run_preset(
        args.preset,
        overrides={
            "seed": args.seed,
            "max_steps": args.max_steps,
            "num_cpus": args.num_cpus,
            "k_start": args.k_start,
        },
    )


if __name__ == "__main__":
    main()
