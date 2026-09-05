"""
sili_peridot/model/training_checkpoint.py
──────────────────────────────────────────
Save/load B8's training state (step_layers, curriculum WindowState,
layernorms) to disk -- Phase 3 needs "save the top-quality checkpoint
seen so far during each stage" (not just the final one), and this
project has no checkpoint saving at all yet. Distinct from
checkpoint.py, which loads MiniCPM5's ORIGINAL pretrained torch
checkpoint -- this module round-trips the CONVERTED model's own
training state instead.

Does not save embed_tokens/lm_head -- B8's curriculum only trains the
folded suffixes (step_layers) and the window's recurrent connections
(window_state.suffix_windows); embed_tokens/lm_head are untouched by
anything built so far, and re-derivable from the original checkpoint
if that ever changes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sili import _cpu
from sili.tensor import Tensor

from .curriculum import WindowState


def sparse_linear_layer_state_dict(layer: _cpu.SparseLinearLayer) -> dict:
    """Full round-trippable state for a raw _cpu.SparseLinearLayer,
    INCLUDING output_scale -- sili__new's own
    _sparse_linear_layer_state_dict (sili/sparse_rnn.py) only saves
    value_scale/importance_scale, silently dropping output_scale, which
    grow_window_layer's rank1-mode-built layers rely on (see
    sili_block.py's _raw_stored_csr). importance is NOT restorable
    either way (load_weights has no path to restore it -- same
    limitation sili__new's own helper documents) -- a reload starts
    importance fresh, which only affects future synaptogenesis/pruning
    decisions, not the layer's actual computed values."""
    n_in, n_out = layer.n_inputs, layer.n_outputs
    return {
        "n_inputs": n_in,
        "n_outputs": n_out,
        "ptrs": np.asarray(layer.ptrs).copy(),
        "indices": np.asarray(layer.indices).copy(),
        "weights": np.asarray(layer.weights_vals).copy(),
        "value_scale": np.array([layer.get_value_scale(r) for r in range(n_in)], dtype=np.float32),
        "output_scale": np.array([layer.get_output_scale(c) for c in range(n_out)], dtype=np.float32),
    }


def sparse_linear_layer_from_state_dict(d: dict, num_cpus: int = 4) -> _cpu.SparseLinearLayer:
    """Inverse of sparse_linear_layer_state_dict -- builds a fresh
    SparseLinearLayer, sized to fit exactly the saved nnz (same
    int(nnz*1.3)+64 headroom convention used throughout sili_block.py)."""
    n_in, n_out = int(d["n_inputs"]), int(d["n_outputs"])
    nnz = int(d["weights"].shape[0])
    layer = _cpu.SparseLinearLayer(n_in, n_out, int(nnz * 1.3) + 64, num_cpus)
    layer.load_weights(
        np.asarray(d["ptrs"], dtype=np.int32),
        np.asarray(d["indices"], dtype=np.int32),
        np.asarray(d["weights"], dtype=np.float32),
    )
    for r in range(n_in):
        if d["value_scale"][r] != 1.0:
            layer.set_value_scale_raw(r, float(d["value_scale"][r]))
    for c in range(n_out):
        if d["output_scale"][c] != 1.0:
            layer.set_output_scale_raw(c, float(d["output_scale"][c]))
    return layer


def save_training_checkpoint(
    path: str | Path,
    step_layers: list[dict[str, object]],
    input_ln_weights: list[np.ndarray],
    post_attn_ln_weights: list[np.ndarray],
    final_norm_weight: np.ndarray,
    window_state: WindowState | None = None,
    stage_index: int = 0,
    quality: float | None = None,
    num_cpus: int = 4,
) -> None:
    """Write a single checkpoint file (pickle -- numpy arrays and the
    plain-dict layer state above round-trip natively, no need for a
    bespoke binary format here). Written to a .tmp path first, then
    atomically renamed into place (POSIX rename is atomic) -- a crash or
    kill mid-write can never leave a corrupted file at `path` itself,
    which matters since Phase 3's re-targeting logic trusts whatever the
    last successfully-saved checkpoint says was the best quality seen."""
    payload = {
        "num_cpus": num_cpus,
        "step_layers": [
            {suffix: sparse_linear_layer_state_dict(layer) for suffix, layer in step.items()} for step in step_layers
        ],
        "input_ln_weights": [np.asarray(w).copy() for w in input_ln_weights],
        "post_attn_ln_weights": [np.asarray(w).copy() for w in post_attn_ln_weights],
        "final_norm_weight": np.asarray(final_norm_weight).copy(),
        "stage_index": stage_index,
        "quality": quality,
        "window_state": None,
    }
    if window_state is not None and window_state.window_size > 0:
        payload["window_state"] = {
            "window_size": window_state.window_size,
            "window_positions": list(window_state.window_positions),
            "suffix_windows": {
                suffix: sparse_linear_layer_state_dict(layer) for suffix, layer in window_state.suffix_windows.items()
            },
            "centers": np.asarray(window_state.centers.data).copy(),
            "log_sigmas": np.asarray(window_state.log_sigmas.data).copy(),
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_training_checkpoint(
    path: str | Path,
) -> tuple[
    list[dict[str, object]], list[np.ndarray], list[np.ndarray], np.ndarray, WindowState | None, int, float | None
]:
    """Returns (step_layers, input_ln_weights, post_attn_ln_weights,
    final_norm_weight, window_state, stage_index, quality) -- window_state
    is None if none was saved (e.g. a checkpoint from before B8a's
    curriculum window existed at all, or one saved at pre-stage-0)."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    num_cpus = payload["num_cpus"]
    step_layers = [
        {suffix: sparse_linear_layer_from_state_dict(d, num_cpus) for suffix, d in step.items()}
        for step in payload["step_layers"]
    ]
    window_state = None
    if payload["window_state"] is not None:
        wd = payload["window_state"]
        window_state = WindowState(
            suffix_windows={
                suffix: sparse_linear_layer_from_state_dict(d, num_cpus) for suffix, d in wd["suffix_windows"].items()
            },
            window_size=wd["window_size"],
            window_positions=list(wd["window_positions"]),
            centers=Tensor(np.asarray(wd["centers"], dtype=np.float32)),
            log_sigmas=Tensor(np.asarray(wd["log_sigmas"], dtype=np.float32)),
        )
    return (
        step_layers,
        payload["input_ln_weights"],
        payload["post_attn_ln_weights"],
        payload["final_norm_weight"],
        window_state,
        payload["stage_index"],
        payload["quality"],
    )
