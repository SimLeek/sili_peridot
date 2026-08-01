"""
sili_peridot/model/curriculum.py
─────────────────────────────────
B8a's backward-growing column-averaging curriculum: an explicit stage
list (stage i = predict from the average of the last i+1 fold
positions) plus WindowState, which drives sili_block.grow_window_layer
to build up each suffix's combined window matrix ONE position at a
time as the curriculum advances -- never for positions outside the
current window (see sili_block's module docstring for why).

Stage 0 (window_size=1) is the sanity-check stage: a single position
has nothing to recur with, so there is nothing for a combined matrix to
usefully hold yet. advance_window still builds one (grow_window_layer's
very first call, from existing_window_layer=None, is unavoidable
plumbing -- window_size=2 needs SOME 1-position base to grow from, and
that base is bit-identical to the position's own plain layer, cheap to
build), but forward-pass callers should special-case window_size==1 and
use step_layers[window_positions[0]][suffix] directly instead of
routing through window_state.suffix_windows -- simpler, and avoids
paying for the window machinery's extra indirection when recurrence is
structurally impossible anyway.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .sili_block import grow_window_layer


@dataclass(frozen=True)
class CurriculumStage:
    index: int
    window_size: int


def build_stage_list(n_folds: int) -> List[CurriculumStage]:
    """Stage i's window is the last (i+1) fold positions -- stage
    n_folds-1 is the final stage, the full column average."""
    return [CurriculumStage(index=i, window_size=i + 1) for i in range(n_folds)]


@dataclass
class WindowState:
    """suffix_windows[suffix] holds a combined SparseLinearLayer for
    every window_size>=1 (see module docstring for why window_size==1's
    is usually better bypassed by forward-pass callers, not why it's
    absent). window_positions holds the ABSOLUTE fold-step indices
    currently in the window, in window-growth order (index 0 = first
    position added = the LAST fold-step, matching grow_window_layer's
    own convention) -- e.g. for a 24-layer model, window_size=3 means
    window_positions == [23, 22, 21]."""
    suffix_windows: Dict[str, object] = field(default_factory=dict)
    window_size: int = 0
    window_positions: List[int] = field(default_factory=list)


def advance_window(
    window_state: WindowState,
    step_layers: List[Dict[str, object]],
    suffixes: List[str],
    n_folds: int,
    num_cpus: int = 4,
    recurrent_bandwidth: Optional[int] = None,
) -> WindowState:
    """Grow the window by exactly one position -- the position
    immediately before window_state's current earliest one (or the
    LAST fold-step, n_folds-1, if the window is empty/stage 0). Calls
    grow_window_layer once per suffix, each reusing that suffix's own
    previous combined matrix (or None, for the very first position --
    see grow_window_layer's own None-existing_window_layer case) so
    already-trained in-window recurrent connections carry forward.
    grow_window_layer works the same regardless of what value_scale_mode
    built step_layers (per_row or rank1) -- no mode argument needed here.

    Does NOT mutate window_state -- returns a new one, matching the
    functional style grow_window_layer itself already uses (old
    layers untouched, caller replaces its reference)."""
    next_position = (window_state.window_positions[-1] - 1
                      if window_state.window_positions else n_folds - 1)
    if next_position < 0:
        raise ValueError("window already covers every fold position -- no earlier "
                          "position left to add")

    new_windows: Dict[str, object] = {}
    for suffix in suffixes:
        new_layer = step_layers[next_position][suffix]
        in_dim, out_dim = new_layer.n_inputs, new_layer.n_outputs
        new_windows[suffix] = grow_window_layer(
            new_layer, in_dim, out_dim, num_cpus=num_cpus,
            recurrent_bandwidth=recurrent_bandwidth,
            existing_window_layer=window_state.suffix_windows.get(suffix),
            existing_window_size=window_state.window_size)

    return WindowState(
        suffix_windows=new_windows,
        window_size=window_state.window_size + 1,
        window_positions=window_state.window_positions + [next_position])
