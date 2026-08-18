"""
sili_peridot/model/eval_eigenvalues.py
────────────────────────────────────────
Standalone eigenvalue/spectral-norm ("RNN health") diagnostics for
sili_peridot's tile-recurrence layers -- read-only, works on any layer
regardless of whether the model was built with spectral-norm regulation
active. The power-iteration probe below started as an extraction of
ToyTileRecurrenceRealFP4._spectral_rescale_factor (model/
toy_tile_precision_models.py), which only ever measures a layer when
spectral_norm_target is set -- so a config like `baseline` (no
spectral-norm mechanism at all) previously got zero eigenvalue
visibility. Keeping each piece of code doing its own job (per
conversation): the training-time file keeps its rescaling POLICY, this
module holds the pure measurement, usable on any layer independent of
whether anything downstream regulates it.

Two DIFFERENT quantities, both provided, DO NOT conflate them:

- SpectralProbe / track_spectral_health: cheap, iterative, ONE extra
  forward pass per measurement -- suitable for tracking every N steps
  during real training. Despite being modeled on "power iteration for
  the dominant singular value," this is FORWARD-ONLY iteration
  (u_{k+1} = layer(u_k)/||layer(u_k)||, no transpose step), which only
  converges to the true top singular value when the underlying map is
  symmetric. For a generic (non-symmetric) weight matrix -- the normal
  case here -- the dominant eigenvalue is generically a COMPLEX pair,
  and this instead approximates something close to the SPECTRAL RADIUS
  (max |eigenvalue|), not the spectral norm. Confirmed directly: for a
  random 16x16 Gaussian matrix this iteration converges to ~3.43 while
  the true top singular value is ~7.13 and the true spectral radius is
  ~3.49 -- tracking the latter, not the former (same correction now
  applied to _spectral_rescale_factor's own docstring, which had this
  mislabeled the same way). Kept because spectral radius actually IS
  the theoretically correct quantity for recurrent-dynamics stability
  (spectral norm is a conservative, often much larger, upper bound on
  it), and because it's cheap enough to run every training step.

- exact_spectral_norm / exact_spectral_radius: EXACT (not iterative),
  via a real np.linalg.svd/eigvals call on the layer's reconstructed
  dense weight matrix. Costs `in_features` forward passes to rebuild
  the matrix (each layer here is a purely linear map at learning_rate=
  0.0 -- no activation function inside a single DISLDOLayer/
  TrueMultiDigitLayer forward call -- so probing with each standard
  basis vector and collecting the outputs as columns reconstructs W
  exactly, then numpy does exact linear algebra on it), so this is
  fine for periodic diagnostic snapshots (every few hundred steps) but
  too expensive to call every single training step the way
  SpectralProbe is designed for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from sili.tensor import Tensor


class SpectralProbe:
    """One persistent probe vector + EMA state for a single SQUARE layer
    (in_features == out_features -- a state-to-state recurrent map, e.g.
    o_proj). Call .measure(layer) once per snapshot -- reuses the SAME
    vector across calls (not a fresh random one each time), which is
    what makes this power iteration rather than a single noisy one-shot
    estimate. Approximates a spectral-RADIUS-like quantity (max
    |eigenvalue|), NOT the spectral norm/top singular value -- see this
    module's own docstring for the full explanation and the
    exact_spectral_norm/exact_spectral_radius alternative if a precise
    number matters more than per-step cost.

    ONLY works for square layers: each step feeds the layer's OUTPUT
    back in as the NEXT step's input, which requires in_features ==
    out_features -- confirmed directly (a rectangular layer crashes on
    the second .measure() call with a shape mismatch, not just measures
    something different). This is exactly why the training-time
    mechanism this was extracted from only ever applied it to o_proj.
    For a genuinely rectangular layer (e.g. lm_head), use
    exact_spectral_norm instead -- real SVD has no such constraint."""

    def __init__(self, dim: int, seed: int = 0, ema_decay: float = 0.9):
        rng = np.random.default_rng(seed)
        u = rng.standard_normal(dim).astype(np.float32)
        self.u = u / (np.linalg.norm(u) + 1e-8)
        self.ema_decay = ema_decay
        self.sigma_ema: Optional[float] = None
        self.sigma_raw: Optional[float] = None  # last unsmoothed estimate, for convergence checks

    def measure(self, layer) -> float:
        """layer must expose .forward(x: Tensor, learning_rate: float) ->
        Tensor (the DISLDOLayer-family convention used throughout this
        repo) -- forward(..., 0.0) is the zero-side-effect convention
        already used elsewhere (evaluate(), the original training-time
        probe): no backward/optimizer call, no weight mutation, so this
        is safe to call at any point during or after training without
        disturbing it."""
        eps = 1e-8
        probe = Tensor(self.u.reshape(1, -1).astype(np.float32))
        raw = np.asarray(layer.forward(probe, 0.0).data).reshape(-1)
        sigma = float(np.linalg.norm(raw))
        self.u = raw / (sigma + eps)
        self.sigma_raw = sigma
        self.sigma_ema = sigma if self.sigma_ema is None else (
            self.ema_decay * self.sigma_ema + (1.0 - self.ema_decay) * sigma)
        return self.sigma_ema


@dataclass
class SpectralSnapshot:
    step: int
    sigma_ema: Dict[str, float]
    sigma_raw: Dict[str, float]


@dataclass
class SpectralTrajectory:
    snapshots: List[SpectralSnapshot] = field(default_factory=list)

    def layer_names(self) -> List[str]:
        return list(self.snapshots[0].sigma_ema.keys()) if self.snapshots else []

    def series(self, layer_name: str, *, raw: bool = False) -> List[float]:
        key = "sigma_raw" if raw else "sigma_ema"
        return [getattr(s, key)[layer_name] for s in self.snapshots]

    def max_ever(self, layer_name: str) -> float:
        return max(self.series(layer_name))

    def final(self, layer_name: str) -> float:
        return self.series(layer_name)[-1]


def probe_layers(layers: Mapping[str, object], *, seed: int = 0,
                  ema_decay: float = 0.9) -> Dict[str, SpectralProbe]:
    """Build one SpectralProbe per named layer, sized to each layer's own
    input width (reads .in_features/.out_features, matching every
    DISLDOLayer-family layer's own attributes). Requires SQUARE layers
    (in_features == out_features) -- see SpectralProbe's own docstring
    for why; a rectangular layer here would crash on its second measure()
    call, so this rejects it up front with a clear error instead."""
    probes = {}
    for i, (name, layer) in enumerate(layers.items()):
        in_dim = getattr(layer, "in_features", None)
        out_dim = getattr(layer, "out_features", None)
        if in_dim is None or out_dim is None:
            raise ValueError(
                f"layer '{name}' has no .in_features/.out_features -- "
                f"use a layer type this module doesn't yet support, or "
                f"exact_spectral_norm (no such requirement) instead")
        if in_dim != out_dim:
            raise ValueError(
                f"layer '{name}' is rectangular ({in_dim}x{out_dim}) -- "
                f"SpectralProbe only works on square (state-to-state) "
                f"layers, since it feeds its own output back in as the "
                f"next step's input. Use exact_spectral_norm instead "
                f"for a rectangular layer.")
        probes[name] = SpectralProbe(in_dim, seed=seed + i, ema_decay=ema_decay)
    return probes


def measure_snapshot(probes: Mapping[str, SpectralProbe],
                      layers: Mapping[str, object], step: int) -> SpectralSnapshot:
    """One measurement pass across every probed layer -- call this
    periodically from inside a training loop (same cadence pattern as
    run()'s own periodic_eval) to build up a SpectralTrajectory."""
    sigma_ema, sigma_raw = {}, {}
    for name, layer in layers.items():
        sigma_ema[name] = probes[name].measure(layer)
        sigma_raw[name] = probes[name].sigma_raw
    return SpectralSnapshot(step=step, sigma_ema=sigma_ema, sigma_raw=sigma_raw)


def track_spectral_health(
    model_step_fn: Callable[[], None],
    layers_fn: Callable[[], Mapping[str, object]],
    n_steps: int,
    *,
    probe_every: int = 200,
    seed: int = 0,
    ema_decay: float = 0.9,
) -> SpectralTrajectory:
    """Generic training-loop wrapper: calls model_step_fn() once per
    step (the caller's own single training step -- forward+backward+
    optimizer.step, whatever that model needs), and every probe_every
    steps takes a spectral-norm snapshot of layers_fn()'s current
    layers. layers_fn is called fresh each snapshot (not once up front)
    so this works even for models that replace/grow layers over time
    (e.g. synaptogenesis) -- probes themselves are keyed by name and
    persist across snapshots regardless.

    Domain-agnostic like find_optimal_lr's trial_fn -- this module knows
    nothing about OriginalArchModel or any specific architecture. See
    tests/test_eval_eigenvalues.py for a worked adapter.
    """
    probes: Dict[str, SpectralProbe] = {}
    trajectory = SpectralTrajectory()
    for step in range(1, n_steps + 1):
        model_step_fn()
        if step % probe_every == 0 or step == n_steps:
            layers = layers_fn()
            if not probes:
                probes = probe_layers(layers, seed=seed, ema_decay=ema_decay)
            trajectory.snapshots.append(measure_snapshot(probes, layers, step))
    return trajectory


def dense_weight_matrix(layer) -> np.ndarray:
    """Exact dense reconstruction of `layer`'s linear map at
    learning_rate=0.0, via forwarding each standard basis vector and
    collecting the outputs as columns: W[:, i] = layer.forward(e_i, 0.0).
    Only valid for layers that are genuinely LINEAR at lr=0 -- true for
    every DISLDOLayer-family layer in this repo (no activation function
    inside a single forward call; TrueMultiDigitLayer's residual sum of
    linear digit layers is still linear overall). Costs in_features
    forward passes; fine for periodic diagnostic snapshots, NOT
    something to call every training step (that's what SpectralProbe is
    for)."""
    in_f = getattr(layer, "in_features", None)
    if in_f is None:
        raise ValueError("layer has no .in_features -- can't reconstruct its dense matrix")
    cols = []
    for i in range(in_f):
        e = np.zeros(in_f, dtype=np.float32)
        e[i] = 1.0
        probe = Tensor(e.reshape(1, -1))
        out = np.asarray(layer.forward(probe, 0.0).data).reshape(-1)
        cols.append(out)
    return np.stack(cols, axis=1)  # (out_features, in_features)


def exact_spectral_norm(layer) -> float:
    """Exact top singular value (np.linalg.svd on the reconstructed
    dense matrix) -- the quantity SpectralProbe's own docstring
    clarifies it does NOT actually measure."""
    W = dense_weight_matrix(layer)
    return float(np.linalg.svd(W, compute_uv=False)[0])


def exact_spectral_radius(layer) -> float:
    """Exact max |eigenvalue| (np.linalg.eigvals on the reconstructed
    dense matrix) -- needs a SQUARE weight matrix (in_features ==
    out_features), which every state-to-state recurrent layer in this
    codebase's tile-recurrence architecture is (q/k/v/o_proj all map
    state_width -> state_width). Raises for a genuinely rectangular
    layer (e.g. lm_head, embed_width -> vocab) since eigenvalues aren't
    defined for a non-square matrix -- use exact_spectral_norm there
    instead."""
    W = dense_weight_matrix(layer)
    if W.shape[0] != W.shape[1]:
        raise ValueError(
            f"exact_spectral_radius needs a square weight matrix, got "
            f"shape {W.shape} -- eigenvalues aren't defined for a "
            f"rectangular layer (use exact_spectral_norm instead)")
    eigvals = np.linalg.eigvals(W)
    return float(np.max(np.abs(eigvals)))


def exact_spectral_snapshot(layers: Mapping[str, object]) -> Dict[str, Dict[str, Optional[float]]]:
    """One-shot EXACT measurement across every named layer -- returns
    {name: {"norm": exact top singular value, "radius": exact spectral
    radius, or None if that layer's matrix isn't square}}. Companion to
    measure_snapshot's cheap/approximate per-step version -- use this
    one when a precise answer matters more than call cost (e.g. a final
    post-training health check, not every-N-steps tracking)."""
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for name, layer in layers.items():
        W = dense_weight_matrix(layer)
        norm = float(np.linalg.svd(W, compute_uv=False)[0])
        radius: Optional[float] = None
        if W.shape[0] == W.shape[1]:
            radius = float(np.max(np.abs(np.linalg.eigvals(W))))
        result[name] = {"norm": norm, "radius": radius}
    return result
