"""
sili_peridot/model/toy_tile_precision_models.py
──────────────────────────────────────────────────
`ToyTileRecurrence` (model/toy_recall_models.py) ported onto real sili
(`DISLDOLayer`) instead of `DenseTensorLinear` -- per direct
correction: tile-recurrence was always meant to run on sili, not plain
dense fp32 matmul. The column-averaging mechanism (a recurrent state
much WIDER than the input, see
model/toy_recall_models.py's own ToyTileRecurrence docstring and
[[project_sili_column_averaging_mechanism]]) specifically depends on
sili's support for large SPARSE layers to be practical: a truly wide
state as a DENSE matrix costs O(width^2) memory/compute regardless of
how few connections matter, while a sparse layer's cost is governed by
`max_weights` (a fixed connection budget) independent of nominal width
-- exactly what makes pushing `column_neurons` up cheap instead of
prohibitive. `DenseTensorLinear` was a deliberate, explicitly-scoped
stopgap for the earlier toy-validation optimizer question, now
resolved and improved on by the precision-comparison track
(model/toy_precision_models.py's `AdamRowScaleDISLDOLayer`/
`AdamRank1DISLDOLayer`).

Same architecture as `ToyTileRecurrence` in every respect except the
linear-layer type: state-width expansion, column-mean-pool readout,
`gaussian_attention` across tiles (all already layer-type-agnostic --
plain `Tensor` ops or wrapping whatever `disldo_cls` is passed in).
`disldo_cls=` swap matches `ToySmallTransformerRealFP4`'s own pattern
(model/toy_precision_models.py) so plain `DISLDOLayer`,
`AdamRowScaleDISLDOLayer`, and `AdamRank1DISLDOLayer` are all directly
comparable under an identical architecture.

`use_energy` (per direct decision, added after the fact -- NOT tested
in the original MQAR sweep, where energy was found to hurt broadly):
without BPTT (see [[project_sili_bptt_or_chance]] -- `M_prev` is a
fresh DETACHED leaf every tick, so there is no gradient pathway to
learn "write X now because a future tick needs it"), any
out-of-context success is necessarily coincidental, not learned
-- EnergyDynamics can't create a genuine gradient pathway either, but
by keeping more neurons active (its own homeostatic mechanism) it may
raise the ODDS that useful state-carrying patterns persist long enough
to coincidentally help, worth testing specifically on out-of-context
tasks even though it hurt the (fully-visible, BPTT-irrelevant) MQAR
benchmark."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sili.tensor import Tensor, gaussian_attention, exp, reduce_sum, silu
from sili.sparse_rnn import DISLDOLayer
from sili.energy import EnergyDynamics

from .toy_recall_models import rmsnorm_tensor
from .toy_precision_models import _toy_scale_energy, _apply_energy


class ToyTileRecurrenceRealFP4:
    """ToyTileRecurrence's exact architecture, built from DISLDOLayer
    -family layers (`disldo_cls=`) instead of DenseTensorLinear.
    `centers`/`log_sigmas`/RMSNorm weights stay plain Tensor leaves,
    trained by a small external AdamOptimizer via
    `parameters_for_optimizer()` -- DISLDOLayer-family layers' own big
    weights never get an external optimizer call (they train inline
    during backward(), driven by `learning_rate`).

    `step()` returns (M_new, logits, aux_loss) -- aux_loss is None
    unless `use_energy=True` (see module docstring), in which case it
    must be added to the task loss before `.backward()`, same
    convention as `ToySmallTransformerRealFP4`."""

    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int,
                 mlp_hidden: int, num_tiles: int, max_weights: int,
                 num_cpus: int = 2, rms_eps: float = 1e-6, disldo_cls=DISLDOLayer,
                 use_energy: bool = False, energy_kwargs: Optional[dict] = None):
        """energy_kwargs: overrides _toy_scale_energy()'s defaults when
        given (passed straight to EnergyDynamics(**energy_kwargs)) --
        needed because the drive/activation_cost balance point depends
        on this call site's own attn-output magnitude (gaussian_attention's
        raw output here, pre-o_proj -- a different distribution than
        the tanh-cell recurrence _toy_scale_energy()'s defaults were
        ever checked against), not a fixed constant that transfers.
        See sili__new's energy.py docstring ("Choosing drive relative
        to activation_cost") for the derivation."""
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        if not use_energy:
            self.energy = None
        elif energy_kwargs is not None:
            self.energy = EnergyDynamics(**energy_kwargs)
        else:
            self.energy = _toy_scale_energy()
        state_width = self.state_width
        self.q_proj = disldo_cls(state_width, state_width, max_weights, num_cpus)
        self.k_proj = disldo_cls(state_width, state_width, max_weights, num_cpus)
        self.v_proj = disldo_cls(state_width, state_width, max_weights, num_cpus)
        self.o_proj = disldo_cls(state_width, state_width, max_weights, num_cpus)
        self.gate_proj = disldo_cls(state_width, mlp_hidden, max_weights, num_cpus)
        self.up_proj = disldo_cls(state_width, mlp_hidden, max_weights, num_cpus)
        self.down_proj = disldo_cls(mlp_hidden, state_width, max_weights, num_cpus)
        self.lm_head = disldo_cls(embed_width, vocab_size, max_weights, num_cpus)
        self.input_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.post_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.centers = Tensor(np.array([i + 0.5 for i in range(num_tiles)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(num_tiles, dtype=np.float32))

    def parameters_for_optimizer(self) -> List[Tensor]:
        """ONLY the plain leaf params (RMSNorm weights, gaussian
        centers/log_sigmas) -- DISLDOLayer-family layers' own big
        weight matrices train inline during backward(), never via an
        external optimizer step."""
        return [self.input_ln, self.post_ln, self.centers, self.log_sigmas]

    def step(self, x_window: np.ndarray, M_prev: np.ndarray,
            learning_rate: float) -> Tuple[np.ndarray, Tensor, Optional[Tensor]]:
        """One recurrence tick. x_window, M_prev: [num_tiles,
        state_width] numpy, DETACHED (no BPTT, matching
        ToyTileRecurrence's own design) -- both already widened to
        state_width by the caller. Returns (M_new numpy [num_tiles,
        state_width], logits Tensor [num_tiles, vocab_size], aux_loss
        (always None, see class docstring)).

        Q/K/V draw from x_window BLENDED with M_prev (per direct
        correction -- see [[feedback_attention_needs_combined_input_state]]:
        input-only attention forces anything relating fresh input to
        carried state through an artificial "write to state, wait a
        tick, then attend" detour; attention must be able to relate
        input and state directly, in one step)."""
        x_normed = rmsnorm_tensor(Tensor(x_window.astype(np.float32)), self.input_ln, self.rms_eps)
        m_normed = rmsnorm_tensor(Tensor(M_prev.astype(np.float32)), self.input_ln, self.rms_eps)
        qkv_source = x_normed + m_normed
        q = self.q_proj.forward(qkv_source, learning_rate)
        k = self.k_proj.forward(qkv_source, learning_rate)
        v = self.v_proj.forward(qkv_source, learning_rate)
        sigmas = exp(self.log_sigmas)
        attn = gaussian_attention(q, k, v, self.centers, sigmas,
                                  num_cpus=self.num_cpus, causal=False)
        attn, aux_loss = _apply_energy(self.energy, attn, self.num_tiles, self.state_width)
        attn = self.o_proj.forward(attn, learning_rate)

        M_new_t = Tensor(M_prev.astype(np.float32)) + attn
        normed2 = rmsnorm_tensor(M_new_t, self.post_ln, self.rms_eps)
        gate = self.gate_proj.forward(normed2, learning_rate)
        up = self.up_proj.forward(normed2, learning_rate)
        mlp_out = self.down_proj.forward(silu(gate) * up, learning_rate)
        M_new_t = M_new_t + mlp_out

        pooled = M_new_t.reshape((self.num_tiles, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)  # [num_tiles, embed_width]
        logits = self.lm_head.forward(pooled, learning_rate)  # [num_tiles, vocab_size]
        return M_new_t.data, logits, aux_loss
