"""Live, side-by-side visualization of every layer's synapses (weight
and importance matrices) plus the plasticity_reset algorithm's own
live state (which columns are currently selected/boosted, and by how
much), overlaid immediately adjacent to the synapses it's acting on.
Direct instruction: "let's ... get displayarray (my lib) or opencv
working and display all of the networks side by side with labels next
to each other in lock step and I'll just look at all of the synapses
for the next run. The algorithms acting on them can be included as
long as they're near the synapses they're affecting."

Built on displayarray (the user's own library, PyPI); falls back to
raw OpenCV windows if displayarray isn't installed. Not a one-off
analysis script (see feedback_scripts_vs_tests_convention) -- a
reusable live-training tool, wired into train_mqar_curriculum.py via
the plasticity_live_display flag. See
docs/research/toy_tile_recurrence_rmt.rst:live_synapse_display for the
design writeup."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from displayarray.window.subscriber_windows import SubscriberWindows

    _HAVE_DISPLAYARRAY = True
except ImportError:  # pragma: no cover -- exercised only when displayarray isn't installed
    SubscriberWindows = None
    _HAVE_DISPLAYARRAY = False


def normalize_to_uint8(mat: np.ndarray) -> np.ndarray:
    """Percentile-clipped (1st/99th) min-max normalize to 0-255 uint8.

    A plain min-max normalize gets washed out by a few extreme outlier
    synapses (e.g. importance pinned at max_ci while the rest of the
    population sits far below it) -- clipping to the 1st/99th
    percentile keeps the bulk of the real variation visible."""
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return np.zeros(mat.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.full(mat.shape, 128, dtype=np.uint8)
    clipped = np.clip(mat, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def heatmap(mat: np.ndarray, cell_px: int = 1) -> np.ndarray:
    """Colormap image, BGR uint8, shape (rows*cell_px, cols*cell_px, 3).
    cell_px=1 (default): literal one pixel per synapse, no artificial
    zoom. cell_px>1 upsamples (nearest-neighbor, so synapses stay crisp
    rectangles instead of blurring) -- only useful when you deliberately
    want a smaller number of pools to take up more screen space."""
    u8 = normalize_to_uint8(mat)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
    if cell_px == 1:
        return color
    h, w = color.shape[:2]
    return cv2.resize(color, (w * cell_px, h * cell_px), interpolation=cv2.INTER_NEAREST)


def deviation_strip(
    deviation_by_col: np.ndarray,
    reset_active: np.ndarray,
    width_px: int,
    height_px: int = 18,
) -> np.ndarray:
    """One colored cell per OUTPUT column, directly adjacent to the
    heatmap(s) it describes: red intensity = this column's own
    deviation z-score (how close it is to standing out), green flag =
    currently selected/reset THIS cycle by the algorithm. This is the
    "algorithm acting on the synapses" view, placed right at the
    boundary between the two matrices it can affect."""
    n_out = deviation_by_col.shape[0]
    dev_u8 = normalize_to_uint8(np.clip(deviation_by_col, 0, None))
    strip = np.zeros((1, n_out, 3), dtype=np.uint8)
    strip[0, :, 2] = dev_u8
    strip[0, reset_active.astype(bool), 1] = 255
    return cv2.resize(strip, (width_px, height_px), interpolation=cv2.INTER_NEAREST)


def label_panel(text: str, width_px: int, height_px: int = 22) -> np.ndarray:
    panel = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    cv2.putText(panel, text, (4, height_px - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def compose_pool_panel(
    pool_key: str,
    step: int,
    importance: np.ndarray,
    weight: np.ndarray | None,
    deviation_by_col: np.ndarray | None,
    reset_active: np.ndarray | None,
    cell_px: int = 1,
) -> np.ndarray:
    """Stack one pool's own label, weight heatmap ("the synapse"),
    the deviation/reset strip (the algorithm's live decision, sitting
    right between the two matrices it can touch), and the importance
    heatmap (the accumulator the decision is drawn from) into one
    panel. weight/deviation_by_col/reset_active are each optional --
    left out entirely when not supplied, no placeholder gap."""
    imp_map = heatmap(importance, cell_px)
    width_px = imp_map.shape[1]
    parts = [label_panel(f"{pool_key}  step={step}", width_px)]
    if weight is not None:
        parts.append(heatmap(weight, cell_px))
    if deviation_by_col is not None and reset_active is not None:
        parts.append(deviation_strip(deviation_by_col, reset_active, width_px))
    parts.append(imp_map)
    return np.concatenate(parts, axis=0)


def compose_frame(panels: dict[str, np.ndarray], pool_order: list[str] | None = None) -> np.ndarray | None:
    """Tile every pool's panel side by side, left to right, padding
    shorter panels to the tallest one's height so labels/heatmaps line
    up in a single row -- 'side by side ... in lock step': every panel
    always shows its own most-recently-seen state in the SAME frame,
    not that every pool updates on the exact same step (they don't --
    each pool completes its own amortized cycle on a different step
    offset)."""
    if not panels:
        return None
    order = pool_order or sorted(panels.keys())
    ordered = [panels[k] for k in order if k in panels]
    if not ordered:
        return None
    max_h = max(p.shape[0] for p in ordered)
    padded = []
    for p in ordered:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            padded.append(np.concatenate([p, pad], axis=0))
        else:
            padded.append(p)
    gap = np.zeros((max_h, 6, 3), dtype=np.uint8)
    frame_parts: list[np.ndarray] = []
    for i, p in enumerate(padded):
        if i > 0:
            frame_parts.append(gap)
        frame_parts.append(p)
    return np.concatenate(frame_parts, axis=1)


class LiveSynapseDisplay:
    """Owns the live window and the per-pool panel cache. Call
    `update_pool` once per pool per completed amortized cycle (the
    same call site that currently writes column-log npz snapshots);
    it recomposes and pushes the WHOLE tiled frame every time, so the
    window always reflects every pool's latest known state together."""

    def __init__(
        self,
        pool_order: list[str] | None = None,
        cell_px: int = 1,
        use_displayarray: bool = True,
        window_name: str = "synapses",
    ):
        self.pool_order = pool_order
        self.cell_px = cell_px
        self.window_name = window_name
        self._panels: dict[str, np.ndarray] = {}
        self._window = None
        self._use_displayarray = use_displayarray and _HAVE_DISPLAYARRAY
        self._cv_window_name = f"{window_name} (press ESC to quit)"

    def update_pool(
        self,
        pool_key: str,
        step: int,
        importance: np.ndarray,
        weight: np.ndarray | None = None,
        deviation_by_col: np.ndarray | None = None,
        reset_active: np.ndarray | None = None,
    ) -> np.ndarray | None:
        self._panels[pool_key] = compose_pool_panel(
            pool_key, step, importance, weight, deviation_by_col, reset_active, self.cell_px
        )
        frame = compose_frame(self._panels, self.pool_order)
        if frame is not None:
            self.show(frame)
        return frame

    def show(self, frame: np.ndarray) -> None:
        if self._use_displayarray:
            if self._window is None:
                self._window = SubscriberWindows(window_names=(), video_sources=(), silent=False)
            self._window.update(frame, self.window_name)
        else:
            cv2.imshow(self._cv_window_name, frame)
            cv2.waitKey(1)

    def closed(self) -> bool:
        if self._use_displayarray and self._window is not None:
            return bool(self._window.exited)
        return False
