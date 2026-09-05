"""
sili_peridot/model/tile_recurrence.py
──────────────────────────────────────
Tile-recurrence prototype (B8's fold-depth-window curriculum PAUSED in
favor of this -- see the approved plan, fuzzy-plotting-starlight.md,
for the full design/rationale; this module intentionally stays terse).

One persistent state `M[num_tiles, hidden]`, updated every real
sequence tick `i`:
  1. inject: tile `j`'s content becomes `x[i-(num_tiles-1)+j]` (the
     real token at that sequence position) whenever that index is
     `>= 0`, else the tile keeps its own prior `M[j]` untouched.
  2. ONE shared (bootstrapped from a single already-folded position,
     see `bootstrap_tile_layers`) q/k/v/o/gate/up/down network,
     applied identically to every tile via a batched forward call.
  3. RoPE'd (each tile keyed to its real sequence position) inter-TILE
     `gaussian_attention` -- every tile can attend every other tile,
     `centers`/`log_sigmas` giving a learnable, movable locality prior
     on top of RoPE's real relative-position-aware dot product.
  4. `M_new = M_prev + energy_dynamics(attn_out)` -- an ADDITIVE,
     energy-gated residual update (`EnergyDynamics.forward` already
     plays the role of "gate * f(attn_out)" as one call), then an MLP
     on top. `M_prev` is a plain numpy array (detached by
     construction -- see the approved plan's Training methodology
     section: no BPTT is needed here, this project's existing
     inline/local training convention already satisfies that).

Only the LAST tile (the one that always holds the most recent real
token) produces logits -- see the approved plan's Prediction target
note for why the fuller staggered per-tile scheme is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sili.energy import EnergyDynamics
from sili.tensor import Tensor, exp, gaussian_attention

from .config import MiniCPM5Config
from .sili_block import (
    _ActivationDensity,
    _density_for_suffix,
    _forward,
    apply_rotary,
    rmsnorm,
    rope_cos_sin,
    silu,
)


def default_tile_gaussian_params(num_tiles: int) -> tuple[Tensor, Tensor]:
    """center[i] = i+0.5, sigma=1.0 -- same soft-locality-at-init
    convention as sili_block.default_window_gaussian_params, adapted
    to tiles: each tile has exactly ONE key-space entry (not an
    interleaved 2*window_size space -- unlike Phase 2.7b's window
    mechanism, a tile has only one relevant content per tick, see
    module docstring), so `num_tiles` keys directly, no factor of 2."""
    centers = Tensor(np.array([i + 0.5 for i in range(num_tiles)], dtype=np.float32))
    log_sigmas = Tensor(np.zeros(num_tiles, dtype=np.float32))
    return centers, log_sigmas


def bootstrap_tile_layers(step_layers: list[dict[str, object]], position_index: int) -> dict[str, object]:
    """Pulls ONE already-built fold position's 7-suffix layer dict
    straight out of build_step_layers's output -- no new weight-
    building code, this dict is the SHARED tile network every tile
    routes through (a batched forward call, not a per-tile loop)."""
    return step_layers[position_index]


@dataclass
class TileState:
    """M: persistent [num_tiles, hidden] numpy state (detached by
    construction -- nothing here is a Tensor graph node). centers/
    log_sigmas: trainable Tensor leaves, one pair per tile, see
    default_tile_gaussian_params."""

    M: np.ndarray
    centers: Tensor
    log_sigmas: Tensor

    @staticmethod
    def zeros(num_tiles: int, hidden: int) -> TileState:
        M = np.zeros((num_tiles, hidden), dtype=np.float32)
        centers, log_sigmas = default_tile_gaussian_params(num_tiles)
        return TileState(M=M, centers=centers, log_sigmas=log_sigmas)


def build_tile_window(x: np.ndarray, i: int, num_tiles: int, M_prev: np.ndarray) -> np.ndarray:
    """x: [T, hidden] full token sequence, already resident (no true
    byte-level streaming front end exists yet, and none is needed for
    this -- see the approved plan's Input injection note). Tile j's
    input at real sequence position i is x[i-(num_tiles-1)+j] when
    that index is >= 0, else M_prev[j] untouched (no real token exists
    there yet -- naturally reduces to "only the last tile gets real
    input" at i=0, not a special case)."""
    hidden = x.shape[1]
    window = np.empty((num_tiles, hidden), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        window[j] = x[src] if src >= 0 else M_prev[j]
    return window


def apply_tile_step(
    x_window: np.ndarray,  # [num_tiles, hidden] -- see build_tile_window
    tick_index: int,  # real sequence position of the LAST tile this tick
    M_prev: np.ndarray,  # [num_tiles, hidden]
    tile_layers: dict[str, object],  # bootstrap_tile_layers's output -- SHARED across every tile
    input_ln_weight: np.ndarray,
    post_attn_ln_weight: np.ndarray,
    cfg: MiniCPM5Config,
    centers: Tensor,
    log_sigmas: Tensor,
    energy_dynamics: EnergyDynamics,
    lm_head: np.ndarray | None = None,
    num_cpus: int = 4,
    activation_density: _ActivationDensity = None,
) -> tuple[np.ndarray, np.ndarray | None, Tensor]:
    """One recurrence tick. Returns (M_new, logits, aux_loss) -- logits
    from the LAST tile only (see module docstring), or None if
    lm_head isn't given (e.g. architecture-only sanity tests that
    don't need a real vocab projection)."""
    num_tiles, hidden = x_window.shape
    n_heads, n_kv_heads, head_dim = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    groups = n_heads // n_kv_heads

    normed = rmsnorm(x_window, input_ln_weight, cfg.rms_norm_eps)  # [num_tiles, hidden]

    q = _forward(
        tile_layers[".self_attn.q_proj.weight"],
        normed,
        _density_for_suffix(activation_density, ".self_attn.q_proj.weight"),
    )
    k = _forward(
        tile_layers[".self_attn.k_proj.weight"],
        normed,
        _density_for_suffix(activation_density, ".self_attn.k_proj.weight"),
    )
    v = _forward(
        tile_layers[".self_attn.v_proj.weight"],
        normed,
        _density_for_suffix(activation_density, ".self_attn.v_proj.weight"),
    )

    q = q.reshape(num_tiles, n_heads, head_dim)
    k = k.reshape(num_tiles, n_kv_heads, head_dim)
    v = v.reshape(num_tiles, n_kv_heads, head_dim)

    # Tile j's real sequence position: tick_index - (num_tiles-1-j).
    # Clipped to >=0 purely for RoPE indexing safety -- tiles whose
    # real position doesn't exist yet hold M_prev content (see
    # build_tile_window), not meaningful "position" content, so a
    # correct RoPE angle for them isn't needed, just a valid one.
    positions = np.clip(tick_index - (num_tiles - 1) + np.arange(num_tiles), 0, None)
    cos, sin = rope_cos_sin(int(positions.max()) + 1, head_dim, cfg.rope_theta)
    cos_t, sin_t = cos[positions], sin[positions]  # [num_tiles, head_dim]

    attn_out = np.empty((num_tiles, n_heads, head_dim), dtype=np.float32)
    sigmas = exp(log_sigmas)
    for h in range(n_heads):
        kv_h = h // groups
        qh = Tensor(apply_rotary(q[:, h, :], cos_t, sin_t))
        kh = Tensor(apply_rotary(k[:, kv_h, :], cos_t, sin_t))
        vh = Tensor(np.ascontiguousarray(v[:, kv_h, :]))
        out_h = gaussian_attention(qh, kh, vh, centers, sigmas, num_cpus=num_cpus, causal=False)
        attn_out[:, h, :] = out_h.data

    attn_out = attn_out.reshape(num_tiles, n_heads * head_dim)
    attn_out = _forward(
        tile_layers[".self_attn.o_proj.weight"],
        attn_out,
        _density_for_suffix(activation_density, ".self_attn.o_proj.weight"),
    )

    gated_update, aux_loss, _actual_p = energy_dynamics(Tensor(attn_out.reshape(-1)))
    M_new = M_prev + gated_update.data.reshape(num_tiles, hidden)

    normed2 = rmsnorm(M_new, post_attn_ln_weight, cfg.rms_norm_eps)
    gate_mlp = _forward(
        tile_layers[".mlp.gate_proj.weight"], normed2, _density_for_suffix(activation_density, ".mlp.gate_proj.weight")
    )
    up_mlp = _forward(
        tile_layers[".mlp.up_proj.weight"], normed2, _density_for_suffix(activation_density, ".mlp.up_proj.weight")
    )
    mlp_out = _forward(
        tile_layers[".mlp.down_proj.weight"],
        silu(gate_mlp) * up_mlp,
        _density_for_suffix(activation_density, ".mlp.down_proj.weight"),
    )
    M_new = M_new + mlp_out

    logits = (M_new[-1:] @ lm_head.T)[0] if lm_head is not None else None
    return M_new, logits, aux_loss


def run_tile_recurrence(
    x: np.ndarray,  # [T, hidden] embedded input sequence
    num_tiles: int,
    tile_state: TileState,
    tile_layers: dict[str, object],
    input_ln_weight: np.ndarray,
    post_attn_ln_weight: np.ndarray,
    cfg: MiniCPM5Config,
    energy_dynamics: EnergyDynamics,
    lm_head: np.ndarray | None = None,
    num_cpus: int = 4,
    activation_density: _ActivationDensity = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Walk the whole sequence one tick at a time (see apply_tile_step).
    Returns (M_final, logits_per_tick) -- logits_per_tick is
    [T, vocab_size] (or None if lm_head isn't given), one row per
    tick's last-tile prediction."""
    T = x.shape[0]
    M = tile_state.M
    logits_per_tick = [] if lm_head is not None else None
    for i in range(T):
        x_window = build_tile_window(x, i, num_tiles, M)
        M, logits, _aux_loss = apply_tile_step(
            x_window,
            i,
            M,
            tile_layers,
            input_ln_weight,
            post_attn_ln_weight,
            cfg,
            tile_state.centers,
            tile_state.log_sigmas,
            energy_dynamics,
            lm_head,
            num_cpus,
            activation_density,
        )
        if lm_head is not None:
            logits_per_tick.append(logits)
    stacked_logits = np.stack(logits_per_tick) if lm_head is not None else None
    return M, stacked_logits
