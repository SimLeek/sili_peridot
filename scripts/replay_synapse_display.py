"""Replay a completed (or in-progress) training run's recorded synapse
snapshots at a smooth, fixed frame rate. Direct instruction, after
noting that a LIVE-attached view is bound to the real training cadence
(one frame per completed amortized cycle, roughly once every several
seconds per pool -- nowhere near smooth): "Tbh I'd prefer to see the
replay unless it's running from 30-60fps."

Reuses the exact same per-pool panel composition as
live_synapse_display.py -- no duplicated rendering logic, just a
different data source (recorded .npz snapshots instead of a live
training loop) and a different pacing model (a fixed frame rate,
decoupled from the real training cadence, instead of "as fast as
training produces new cycles"). Requires the run to have been launched
with plasticity_raw_importance_log=True (captures raw_importance AND
raw_weight per snapshot, not just the column-level aggregate stats) --
see docs/research/toy_tile_recurrence_rmt.rst:live_synapse_display."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, ".")

from scripts.live_synapse_display import LiveSynapseDisplay


def discover_pools(column_log_dir: str) -> list[str]:
    """Every pool subdirectory under column_log_dir, sorted."""
    return sorted(name for name in os.listdir(column_log_dir) if os.path.isdir(os.path.join(column_log_dir, name)))


def build_event_timeline(column_log_dir: str, pools: list[str]) -> list[tuple[int, str, str]]:
    """One (step, pool_key, filepath) tuple per recorded snapshot,
    across ALL pools, sorted by step -- preserves the real chronological
    order snapshots were recorded in (different pools complete their
    own amortized cycle at different step offsets), even though
    playback itself runs at a fixed frame rate decoupled from real
    wall-clock training time."""
    events: list[tuple[int, str, str]] = []
    for pool in pools:
        for f in glob.glob(os.path.join(column_log_dir, pool, "step*.npz")):
            step = int(os.path.basename(f).replace("step", "").replace(".npz", ""))
            events.append((step, pool, f))
    events.sort(key=lambda e: e[0])
    return events


def deviation_from_col_state(d) -> np.ndarray:
    """Same formula the C++ engine itself uses for deviation_by_col."""
    fast = d["col_grad_fast"]
    slow = d["col_grad_slow"]
    var = d["col_grad_var"]
    return (fast - slow) / (np.sqrt(np.maximum(var, 0.0)) + 1e-8)


def replay(
    column_log_dir: str,
    fps: float = 30.0,
    cell_px: int = 1,
    pool_order: list[str] | None = None,
    display: LiveSynapseDisplay | None = None,
    window_name: str | None = None,
) -> int:
    """Play back every recorded snapshot under column_log_dir, oldest
    to newest, at a fixed frame rate. Returns the number of frames
    actually played (fewer than the event count if the window was
    closed early). window_name defaults to column_log_dir's own
    basename, so multiple replay windows opened at once (e.g. to
    compare candidates side by side) are distinguishable -- displayarray's
    own default title is otherwise just "synapses" for every window."""
    pools = pool_order or discover_pools(column_log_dir)
    events = build_event_timeline(column_log_dir, pools)
    if not events:
        raise FileNotFoundError(f"No .npz snapshots found under {column_log_dir}")
    if display is None:
        name = window_name or os.path.basename(os.path.normpath(column_log_dir))
        display = LiveSynapseDisplay(pool_order=pools, cell_px=cell_px, window_name=name)
    frame_interval = 1.0 / fps
    n_played = 0
    for step, pool, path in events:
        if display.closed():
            break
        d = np.load(path)
        if "raw_importance" not in d.files:
            raise KeyError(
                f"{path} has no raw_importance -- re-run with plasticity_raw_importance_log=True "
                "to capture per-synapse matrices, not just the column-level aggregate stats"
            )
        importance = d["raw_importance"]
        weight = d["raw_weight"] if "raw_weight" in d.files else None
        deviation = deviation_from_col_state(d)
        reset_active = d["col_reset_active"]
        t0 = time.monotonic()
        display.update_pool(pool, step, importance, weight, deviation, reset_active)
        n_played += 1
        remaining = frame_interval - (time.monotonic() - t0)
        if remaining > 0:
            time.sleep(remaining)
    return n_played


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("column_log_dir", help="e.g. logs/plasticity_column_snapshots/<run>")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cell-px", type=int, default=1)
    parser.add_argument("--window-name", default=None, help="defaults to column_log_dir's basename")
    args = parser.parse_args()
    n = replay(args.column_log_dir, fps=args.fps, cell_px=args.cell_px, window_name=args.window_name)
    print(f"Replayed {n} frames.")


if __name__ == "__main__":
    main()
