"""
Fake-quantization for the torch reference (DISLDOTorchLinear) -- lets
the fast torch harness test whether a mechanism (e.g. magnitude-scale
reparametrization) actually helps REAL quantized precision, without a
30+ minute round trip through the real C++ engine for every hypothesis.

Encode reuses the REAL C++ codec (_cpu.fp4_quantize_array/
fp8_quantize_array) so rounding decisions are bit-exact to what the
real engine does. Decode is reimplemented here directly (both codecs
are small, deterministic table/formula lookups -- fp4quant.hpp's
FP4_TABLE and fp8quant.hpp's fp8_decode_bits, ported 1:1) since no
decode is exposed at the Python/pybind level.

Straight-through estimator: forward value is quantize-then-decode(x),
backward gradient is identity (dL/dx unchanged) -- standard QAT
convention, matches this project's own TrueMultiDigitLayer(simulate_
quantize=True) intent for a plain-torch context that has no direct
C++ layer to fake-quantize through.
"""
from __future__ import annotations

import numpy as np
import torch

FP4_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    np.nan, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def fp4_decode_np(codes: np.ndarray) -> np.ndarray:
    return FP4_TABLE[codes.astype(np.int64)]


def fp8_decode_np(codes: np.ndarray) -> np.ndarray:
    """Ports fp8quant.hpp's fp8_decode_bits exactly (E4M3, bias 7,
    subnormal slot value=m*2^-9, reserved NaN code e4=15,m=7)."""
    codes = codes.astype(np.uint32)
    s = (codes >> 7) & 1
    e4 = (codes >> 3) & 0xF
    m = codes & 0x7

    out = np.empty(codes.shape, dtype=np.float32)

    subnormal = e4 == 0
    mag = m[subnormal].astype(np.float32) * (1.0 / 512.0)
    out[subnormal] = np.where(s[subnormal] == 1, -mag, mag)

    nan_mask = (e4 == 15) & (m == 7)
    out[nan_mask] = np.nan

    normal = ~subnormal & ~nan_mask
    exp_field = e4[normal] + 120
    bits = (s[normal].astype(np.uint32) << 31) | (exp_field << 23) | (m[normal] << 20)
    out[normal] = bits.view(np.float32)

    return out


_GRID_CACHE: dict = {}


def _sorted_grid(kind: str) -> np.ndarray:
    """Full sorted list of exactly-representable (non-NaN) values for
    kind, used by the dithered stochastic rounder below."""
    if kind not in _GRID_CACHE:
        if kind == "fp4":
            vals = FP4_TABLE
        elif kind == "fp8":
            vals = fp8_decode_np(np.arange(256, dtype=np.uint8))
        else:
            raise ValueError(f"unknown fake-quantize kind {kind!r}")
        vals = np.unique(vals[np.isfinite(vals)]).astype(np.float32)
        _GRID_CACHE[kind] = vals
    return _GRID_CACHE[kind]


def _fake_quantize_values(x_np: np.ndarray, kind: str) -> np.ndarray:
    """Deterministic (nearest-neighbour) quantize -- reuses the REAL
    C++ codec for the encode decision, bit-exact to what the real
    engine's deterministic path does."""
    from sili import _cpu
    flat = x_np.flatten().astype(np.float32)
    if kind == "fp4":
        codes = _cpu.fp4_quantize_array(flat)
        decoded = fp4_decode_np(codes)
    elif kind == "fp8":
        codes = _cpu.fp8_quantize_array(flat)
        decoded = fp8_decode_np(codes)
    else:
        raise ValueError(f"unknown fake-quantize kind {kind!r}")
    return decoded.reshape(x_np.shape)


def _fake_quantize_stochastic_values(x_np: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Dithered stochastic rounding to the EXACT representable grid --
    matches production's own switch to stochastic rounding (task #219)
    that the deterministic path above does NOT reflect. Not bound to
    the real C++ RNG's own bit-level implementation (no bulk stochastic
    array quantizer is exposed at the Python level for either codec),
    but is a mathematically valid unbiased dithered rounder: for value
    v bracketed by grid points [lo, hi], rounds to hi with probability
    (v-lo)/(hi-lo), giving E[result] == v (clamped) exactly, same
    "unbiased" property fp8quant.hpp's own fp8_quantize_stochastic
    docstring calls out."""
    grid = _sorted_grid(kind)
    flat = x_np.flatten().astype(np.float32)
    clamped = np.clip(flat, grid[0], grid[-1])
    hi_idx = np.searchsorted(grid, clamped, side="left")
    hi_idx = np.clip(hi_idx, 1, len(grid) - 1)
    lo_idx = hi_idx - 1
    lo, hi = grid[lo_idx], grid[hi_idx]
    span = hi - lo
    frac = np.where(span > 0, (clamped - lo) / np.where(span > 0, span, 1.0), 0.0)
    draw = rng.random(size=flat.shape).astype(np.float32)
    out = np.where(draw < frac, hi, lo)
    return out.reshape(x_np.shape)


_STOCHASTIC_RNG = np.random.default_rng()


def fake_quantize(x: torch.Tensor, kind: str, stochastic: bool = False) -> torch.Tensor:
    """Straight-through fake-quantize: forward = decode(quantize(x)),
    backward gradient = identity."""
    with torch.no_grad():
        if stochastic:
            decoded_np = _fake_quantize_stochastic_values(x.detach().numpy(), kind, _STOCHASTIC_RNG)
        else:
            decoded_np = _fake_quantize_values(x.detach().numpy(), kind)
        decoded = torch.from_numpy(decoded_np)
    return x + (decoded - x).detach()
