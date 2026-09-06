"""
sili_peridot/model/curriculum.py
─────────────────────────────────
See docs/research/curriculum.rst:curriculum.module_overview.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sili.tensor import Tensor

from .sili_block import grow_window_layer


@dataclass(frozen=True)
class CurriculumStage:
    index: int
    window_size: int


def build_stage_list(n_folds: int) -> list[CurriculumStage]:
    """Stage i's window is the last (i+1) fold positions -- stage
    n_folds-1 is the final stage, the full column average."""
    return [CurriculumStage(index=i, window_size=i + 1) for i in range(n_folds)]


@dataclass
class WindowState:
    """See docs/research/curriculum.rst:curriculum.window_state_ordering."""

    suffix_windows: dict[str, object] = field(default_factory=dict)
    window_size: int = 0
    window_positions: list[int] = field(default_factory=list)
    centers: Tensor | None = None
    log_sigmas: Tensor | None = None


def advance_window(
    window_state: WindowState,
    step_layers: list[dict[str, object]],
    suffixes: list[str],
    n_folds: int,
    num_cpus: int = 4,
    recurrent_bandwidth: int | None = None,
) -> WindowState:
    """See docs/research/curriculum.rst:curriculum.advance_window_growth."""
    next_position = window_state.window_positions[-1] - 1 if window_state.window_positions else n_folds - 1
    if next_position < 0:
        raise ValueError("window already covers every fold position -- no earlier position left to add")

    new_windows: dict[str, object] = {}
    for suffix in suffixes:
        new_layer = step_layers[next_position][suffix]
        in_dim, out_dim = new_layer.n_inputs, new_layer.n_outputs
        new_windows[suffix] = grow_window_layer(
            new_layer,
            in_dim,
            out_dim,
            num_cpus=num_cpus,
            recurrent_bandwidth=recurrent_bandwidth,
            existing_window_layer=window_state.suffix_windows.get(suffix),
            existing_window_size=window_state.window_size,
        )

    p = window_state.window_size
    new_center = np.float32(2.0 * p + 0.5)
    new_log_sigma = np.float32(0.0)
    if window_state.centers is None:
        centers = Tensor(np.array([new_center], dtype=np.float32))
        log_sigmas = Tensor(np.array([new_log_sigma], dtype=np.float32))
    else:
        centers = Tensor(np.concatenate([window_state.centers.data, [new_center]]).astype(np.float32))
        log_sigmas = Tensor(np.concatenate([window_state.log_sigmas.data, [new_log_sigma]]).astype(np.float32))

    return WindowState(
        suffix_windows=new_windows,
        window_size=window_state.window_size + 1,
        window_positions=[*window_state.window_positions, next_position],
        centers=centers,
        log_sigmas=log_sigmas,
    )
