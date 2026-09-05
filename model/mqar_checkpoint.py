"""
sili_peridot/model/mqar_checkpoint.py
──────────────────────────────────────
Save/load a ToyTileRecurrenceRMT model's full trained state (all real
disldo_cls weight layers, norm weights, attention centers/log_sigmas,
plus the caller's own embed_table) to a single pickle file. Distinct
from training_checkpoint.py, which round-trips B8's own step_layers/
WindowState -- this is scoped to the MQAR curriculum's own model shape.

Generic across precisions: works for both DISLDOLayerV (fp32, no scale/
additive-branch concept) and SparseLinearLayer/SparseLinearLayer8 (fp4/
fp8, AQRS-capable) by checking hasattr before reading rank-N scale/
additive-branch state -- built with fp4-AQRS conversion testing in mind
(save an fp32-trained model, quantize+load its exact weights into a
fresh AQRS layer, see how much accuracy survives with zero further
training) even though today's only caller saves a plain fp32 model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def layer_state_dict(c) -> dict:
    """Round-trippable state for one raw C++ layer object (DISLDOLayerV
    or SparseLinearLayer/SparseLinearLayer8) -- ptrs/indices/weights_vals
    are always present (true float values, not raw quantization codes --
    load_weights already handles quantization internally); value_scale/
    output_scale/additive_u/additive_v/ranks are only included if the
    layer actually has that concept (fp32's DISLDOLayerV does not)."""
    d = {
        "n_inputs": c.n_inputs,
        "n_outputs": c.n_outputs,
        "ptrs": np.asarray(c.ptrs).copy(),
        "indices": np.asarray(c.indices).copy(),
        "weights_vals": np.asarray(c.weights_vals).copy(),
    }
    if hasattr(c, "get_scale_rank"):
        scale_rank = c.get_scale_rank()
        additive_rank = c.get_additive_rank()
        d["scale_rank"] = scale_rank
        d["scale_rank_max"] = c.get_scale_rank_max()
        d["additive_rank"] = additive_rank
        d["additive_rank_max"] = c.get_additive_rank_max()
        d["value_scale_k"] = [np.asarray(c.get_value_scale_raw_vector(k)).copy() for k in range(scale_rank)]
        d["output_scale_k"] = [np.asarray(c.get_output_scale_raw_vector(k)).copy() for k in range(scale_rank)]
        if additive_rank > 0:
            d["additive_u_k"] = [np.asarray(c.get_additive_u_raw_vector(k)).copy() for k in range(additive_rank)]
            d["additive_v_k"] = [np.asarray(c.get_additive_v_raw_vector(k)).copy() for k in range(additive_rank)]
    return d


def layer_from_state_dict(d: dict, c_class, num_cpus: int = 4):
    """Inverse of layer_state_dict -- builds a fresh C++ layer of the
    given class (e.g. _cpu.SparseLinearLayer, _cpu.DISLDOLayerV), sized
    to fit the saved nnz, then restores ranks/scale/additive state
    BEFORE load_weights (matches DISLDOLayer.__init__'s own
    set_scale_rank_max-before-seed convention -- rank caps must be raised
    before the corresponding rank is set)."""
    n_in, n_out = int(d["n_inputs"]), int(d["n_outputs"])
    nnz = int(d["weights_vals"].shape[0])
    c = c_class(n_in, n_out, int(nnz * 1.3) + 64, num_cpus)
    if "scale_rank" in d:
        c.set_scale_rank_max(int(d["scale_rank_max"]))
        c.set_additive_rank_max(int(d["additive_rank_max"]))
        c.set_scale_rank(int(d["scale_rank"]))
        c.set_additive_rank(int(d["additive_rank"]))
    c.load_weights(
        np.asarray(d["ptrs"], dtype=np.int32),
        np.asarray(d["indices"], dtype=np.int32),
        np.asarray(d["weights_vals"], dtype=np.float32),
    )
    if "scale_rank" in d:
        for k, vec in enumerate(d["value_scale_k"]):
            c.set_value_scale_raw_vector(k, np.asarray(vec, dtype=np.float32))
        for k, vec in enumerate(d["output_scale_k"]):
            c.set_output_scale_raw_vector(k, np.asarray(vec, dtype=np.float32))
        for k, vec in enumerate(d.get("additive_u_k", [])):
            c.set_additive_u_raw_vector(k, np.asarray(vec, dtype=np.float32))
        for k, vec in enumerate(d.get("additive_v_k", [])):
            c.set_additive_v_raw_vector(k, np.asarray(vec, dtype=np.float32))
    return c


def save_mqar_model(
    path: str | Path,
    model,
    embed_table: np.ndarray,
    step: int,
    vocab: int,
    k: int,
    num_cpus: int = 4,
    extra: dict | None = None,
) -> None:
    """Save a ToyTileRecurrenceRMT model's full state -- every real
    weight layer (model._named_real_layers()), norm/attention params
    (model.parameters_for_optimizer()), and the caller's own embed_table
    (owned by the training script, not the model). Atomic write (tmp
    path + rename), matching training_checkpoint.py's own convention."""
    payload = {
        "num_cpus": num_cpus,
        "step": step,
        "vocab": vocab,
        "k": k,
        "layers": {name: layer_state_dict(layer._c) for name, layer in model._named_real_layers()},
        "input_ln": np.asarray(model.input_ln.data).copy(),
        "memory_ln": np.asarray(model.memory_ln.data).copy(),
        "state_ln": np.asarray(model.state_ln.data).copy(),
        "centers": np.asarray(model.centers.data).copy(),
        "log_sigmas": np.asarray(model.log_sigmas.data).copy(),
        "embed_table": np.asarray(embed_table).copy(),
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_mqar_model_payload(path: str | Path) -> dict:
    """Returns the raw payload dict (layers as state-dicts, not yet
    reconstructed into C++ objects -- reconstruction needs the caller's
    target class per layer, e.g. fp32 source -> fp4 AQRS target, so it's
    not done here)."""
    with open(path, "rb") as f:
        return pickle.load(f)
