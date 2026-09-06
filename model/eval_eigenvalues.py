"""
sili_peridot/model/eval_eigenvalues.py

Read-only eigenvalue/spectral-norm ("RNN health") diagnostics for
sili_peridot's tile-recurrence layers, usable regardless of whether the
model was built with spectral-norm regulation active.

Two DIFFERENT quantities are provided -- do not conflate them:

- SpectralProbe / track_spectral_health: cheap, iterative, approximates
  the spectral RADIUS (not the spectral norm).
- exact_spectral_norm / exact_spectral_radius: exact via SVD/eigvals on
  the reconstructed dense matrix, too expensive for every training step.

See docs/research/eval_eigenvalues.rst:eval_eigenvalues.module_overview.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from sili.tensor import Tensor


class SpectralProbe:
    """One persistent probe vector + EMA state for a single SQUARE layer
    (in_features == out_features, e.g. o_proj). .measure(layer) reuses
    the SAME vector across calls (power iteration, not a fresh one-shot
    estimate); approximates spectral RADIUS, NOT spectral norm/top
    singular value. Rectangular layers crash on the second .measure()
    call -- use exact_spectral_norm there instead. See
    docs/research/eval_eigenvalues.rst:
    eval_eigenvalues.spectral_probe_forward_only_iteration."""

    def __init__(self, dim: int, seed: int = 0, ema_decay: float = 0.9):
        rng = np.random.default_rng(seed)
        u = rng.standard_normal(dim).astype(np.float32)
        self.u = u / (np.linalg.norm(u) + 1e-8)
        self.ema_decay = ema_decay
        self.sigma_ema: float | None = None
        self.sigma_raw: float | None = None  # last unsmoothed estimate, for convergence checks

    def measure(self, layer) -> float:
        """layer must expose .forward(x: Tensor, learning_rate: float) ->
        Tensor (the DISLDOLayer-family convention). forward(..., 0.0) is
        the zero-side-effect convention -- no backward/optimizer call,
        no weight mutation, safe to call any time during or after
        training."""
        eps = 1e-8
        probe = Tensor(self.u.reshape(1, -1).astype(np.float32))
        raw = np.asarray(layer.forward(probe, 0.0).data).reshape(-1)
        sigma = float(np.linalg.norm(raw))
        self.u = raw / (sigma + eps)
        self.sigma_raw = sigma
        self.sigma_ema = (
            sigma if self.sigma_ema is None else (self.ema_decay * self.sigma_ema + (1.0 - self.ema_decay) * sigma)
        )
        return self.sigma_ema


@dataclass
class SpectralSnapshot:
    step: int
    sigma_ema: dict[str, float]
    sigma_raw: dict[str, float]


@dataclass
class SpectralTrajectory:
    snapshots: list[SpectralSnapshot] = field(default_factory=list)

    def layer_names(self) -> list[str]:
        return list(self.snapshots[0].sigma_ema.keys()) if self.snapshots else []

    def series(self, layer_name: str, *, raw: bool = False) -> list[float]:
        key = "sigma_raw" if raw else "sigma_ema"
        return [getattr(s, key)[layer_name] for s in self.snapshots]

    def max_ever(self, layer_name: str) -> float:
        return max(self.series(layer_name))

    def final(self, layer_name: str) -> float:
        return self.series(layer_name)[-1]


def probe_layers(layers: Mapping[str, object], *, seed: int = 0, ema_decay: float = 0.9) -> dict[str, SpectralProbe]:
    """Build one SpectralProbe per named layer, sized to each layer's own
    input width (.in_features/.out_features). Requires SQUARE layers --
    rejects a rectangular one up front with a clear error. See
    docs/research/eval_eigenvalues.rst:eval_eigenvalues.probe_layers_square_requirement."""
    probes = {}
    for i, (name, layer) in enumerate(layers.items()):
        in_dim = getattr(layer, "in_features", None)
        out_dim = getattr(layer, "out_features", None)
        if in_dim is None or out_dim is None:
            raise ValueError(
                f"layer '{name}' has no .in_features/.out_features -- "
                f"use a layer type this module doesn't yet support, or "
                f"exact_spectral_norm (no such requirement) instead"
            )
        if in_dim != out_dim:
            raise ValueError(
                f"layer '{name}' is rectangular ({in_dim}x{out_dim}) -- "
                f"SpectralProbe only works on square (state-to-state) "
                f"layers, since it feeds its own output back in as the "
                f"next step's input. Use exact_spectral_norm instead "
                f"for a rectangular layer."
            )
        probes[name] = SpectralProbe(in_dim, seed=seed + i, ema_decay=ema_decay)
    return probes


def measure_snapshot(probes: Mapping[str, SpectralProbe], layers: Mapping[str, object], step: int) -> SpectralSnapshot:
    """One measurement pass across every probed layer -- call
    periodically from a training loop (same cadence as run()'s own
    periodic_eval) to build up a SpectralTrajectory."""
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
    step, and every probe_every steps takes a spectral-norm snapshot of
    layers_fn()'s current layers. layers_fn is called fresh each
    snapshot (not once up front) so this works for models that
    replace/grow layers over time. Domain-agnostic like find_optimal_lr's
    trial_fn. See tests/test_eval_eigenvalues.py for a worked adapter,
    and docs/research/eval_eigenvalues.rst:
    eval_eigenvalues.track_spectral_health_layers_fn_recomputed.
    """
    probes: dict[str, SpectralProbe] = {}
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
    learning_rate=0.0: W[:, i] = layer.forward(e_i, 0.0) for each
    standard basis vector e_i. Only valid for layers genuinely LINEAR at
    lr=0 (true for every DISLDOLayer-family layer here). Costs
    in_features forward passes -- fine for periodic snapshots, not every
    training step (that's SpectralProbe). See
    docs/research/eval_eigenvalues.rst:
    eval_eigenvalues.dense_weight_matrix_reconstruction."""
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
    dense matrix) -- the quantity SpectralProbe does NOT actually
    measure. See docs/research/eval_eigenvalues.rst:
    eval_eigenvalues.exact_spectral_norm_radius_square_requirement."""
    W = dense_weight_matrix(layer)
    return float(np.linalg.svd(W, compute_uv=False)[0])


def exact_spectral_radius(layer) -> float:
    """Exact max |eigenvalue| (np.linalg.eigvals on the reconstructed
    dense matrix) -- needs a SQUARE weight matrix; raises for a
    rectangular layer (use exact_spectral_norm there instead). See
    docs/research/eval_eigenvalues.rst:
    eval_eigenvalues.exact_spectral_norm_radius_square_requirement."""
    W = dense_weight_matrix(layer)
    if W.shape[0] != W.shape[1]:
        raise ValueError(
            f"exact_spectral_radius needs a square weight matrix, got "
            f"shape {W.shape} -- eigenvalues aren't defined for a "
            f"rectangular layer (use exact_spectral_norm instead)"
        )
    eigvals = np.linalg.eigvals(W)
    return float(np.max(np.abs(eigvals)))


def exact_spectral_snapshot(layers: Mapping[str, object]) -> dict[str, dict[str, float | None]]:
    """One-shot EXACT measurement across every named layer -- returns
    {name: {"norm": ..., "radius": ... or None if not square}}.
    Companion to measure_snapshot's cheap/approximate per-step version.
    See docs/research/eval_eigenvalues.rst:
    eval_eigenvalues.exact_spectral_snapshot_companion."""
    result: dict[str, dict[str, float | None]] = {}
    for name, layer in layers.items():
        W = dense_weight_matrix(layer)
        norm = float(np.linalg.svd(W, compute_uv=False)[0])
        radius: float | None = None
        if W.shape[0] == W.shape[1]:
            radius = float(np.max(np.abs(np.linalg.eigvals(W))))
        result[name] = {"norm": norm, "radius": radius}
    return result
