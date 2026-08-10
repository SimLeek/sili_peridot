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
    `parameters_for_optimizer()`.

    SwiGLU MLP and tanh have been removed in favor of a minimal 
    attention-only recurrence with [-2.0, 2.0] state clipping.

    `step()` returns (M_new, logits, aux_loss) -- aux_loss is None
    unless `use_energy=True`."""

    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int,
                 mlp_hidden: int, num_tiles: int, max_weights: int,
                 num_cpus: int = 2, rms_eps: float = 1e-6, disldo_cls=DISLDOLayer,
                 use_energy: bool = False, energy_kwargs: Optional[dict] = None,
                 use_attention: bool = True, o_proj_depth: int = 1,
                 rng: Optional[np.random.Generator] = None):
        """mlp_hidden is retained in the signature for API compatibility
        but is no longer used since the MLP block was removed.

        use_attention=False: bypasses q/k/v/gaussian_attention (and
        energy, which only ever gated the attention output) entirely --
        collapses this into a plain RNN cell,
        state = clip(rmsnorm(state + o_proj(rmsnorm(x)+rmsnorm(state)))).
        Ablation to isolate whether gaussian_attention itself is what's
        hard to learn, before assuming the whole architecture is broken.

        o_proj_depth>1: replaces the single o_proj with `o_proj_depth`
        disldo_cls sublayers applied in sequence (each state_width ->
        state_width, no nonlinearity between them), each given
        max_weights // o_proj_depth so the total weight budget stays
        roughly comparable to depth=1 -- a residual/cascaded
        -quantization-style test of whether N sequential coarse (e.g.
        FP4) layers can compose into something closer to a single
        higher-precision layer, rather than needing more WIDTH."""
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.use_attention = use_attention
        self.o_proj_depth = o_proj_depth

        if not use_attention:
            self.energy = None
        elif not use_energy:
            self.energy = None
        elif energy_kwargs is not None:
            self.energy = EnergyDynamics(**energy_kwargs)
        else:
            self.energy = _toy_scale_energy()

        state_width = self.state_width

        # Per-layer independent seeds derived from `rng`, matching this
        # project's own established convention (scripts/disldo_*_ablation.py:
        # np.random.default_rng(seed+1)/(seed+2) per sublayer) -- NOT passing
        # `rng=` down to disldo_cls at all was a real bug found directly (same
        # unseeded-RNG class as feedback_seed_stochastic_rng_for_comparisons):
        # _preseed_random_sparse defaults to np.random.default_rng() (fresh OS
        # entropy) whenever rng=None, so every layer's initial connectivity
        # and initial weight values were NEVER controlled by the `seed` CLI
        # arg -- confirmed directly, same command/seed gave 0.70 then 0.65
        # final-step accuracy across two back-to-back runs. `seed` only ever
        # controlled the embed table, task generation, and (separately) FP4
        # stochastic rounding.
        if rng is None:
            rng = np.random.default_rng()
        n_layer_seeds = 4 + max(o_proj_depth, 1)  # q, k, v, lm_head + o_proj sublayer(s)
        layer_seeds = iter(int(s) for s in rng.integers(0, 2**31 - 1, size=n_layer_seeds))

        # 1. Core Attention & Output Projections
        if use_attention:
            self.q_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)))
            self.k_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)))
            self.v_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)))
        if o_proj_depth > 1:
            per_layer_weights = max(max_weights // o_proj_depth, state_width)
            self.o_proj = [disldo_cls(state_width, state_width, per_layer_weights, num_cpus,
                                      rng=np.random.default_rng(next(layer_seeds)))
                          for _ in range(o_proj_depth)]
        else:
            self.o_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)))
        self.lm_head = disldo_cls(embed_width, vocab_size, max_weights, num_cpus,
                                  rng=np.random.default_rng(next(layer_seeds)))
        
        # 2. Norms & Gaussian Attention Params
        self.input_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.state_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.centers = Tensor(np.array([i + 0.5 for i in range(num_tiles)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(num_tiles, dtype=np.float32))

    def parameters_for_optimizer(self) -> List[Tensor]:
        """ONLY the plain leaf params (RMSNorm weights, gaussian
        centers/log_sigmas) -- DISLDOLayer-family layers' own big
        weight matrices train inline during backward(), never via an
        external optimizer step."""
        return [self.input_ln, self.state_ln, self.centers, self.log_sigmas]

    def debug_learning_state(self):
        """Helper to verify gradients are flowing and params are updating."""
        print("\n=== Model Learning Diagnostics ===")
        params = self.parameters_for_optimizer()
        param_names = ["input_ln", "state_ln", "centers", "log_sigmas"]
        
        for name, p in zip(param_names, params):
            # Check if parameter data has variance (isn't completely zeroed out)
            data_active = np.any(p.data != 0)
            data_mean = np.mean(p.data)
            
            if p.grad is not None:
                # Use np.any to ensure gradients aren't getting squashed entirely to 0
                grad_active = np.any(p.grad != 0)
                grad_mean = np.mean(np.abs(p.grad))
                has_nans = np.any(np.isnan(p.grad))
                print(f"{name:12s} | Data Active: {data_active} (mean: {data_mean:.4f}) | "
                      f"Grad Active: {grad_active} (mean |grad|: {grad_mean:.6e}) | NaN Grad: {has_nans}")
            else:
                print(f"{name:12s} | Data Active: {data_active} (mean: {data_mean:.4f}) | Grad: NONE (Not computed yet)")
        print("==================================\n")

    def _apply_o_proj(self, x: Tensor, learning_rate: float) -> Tensor:
        if self.o_proj_depth > 1:
            for layer in self.o_proj:
                x = layer.forward(x, learning_rate)
            return x
        return self.o_proj.forward(x, learning_rate)

    def step(self, x_window: np.ndarray, M_prev: np.ndarray,
             learning_rate: float, debug: bool = True) -> Tuple[np.ndarray, Tensor, Optional[Tensor]]:
        """One recurrence tick. x_window, M_prev: [num_tiles,
        state_width] numpy, DETACHED (no BPTT, matching
        ToyTileRecurrence's own design). Returns (M_new numpy [num_tiles,
        state_width], logits Tensor [num_tiles, vocab_size], aux_loss)."""
        
        # 1. Combine Input and State
        x_normed = rmsnorm_tensor(Tensor(x_window.astype(np.float32)), self.input_ln, self.rms_eps)
        m_normed = rmsnorm_tensor(Tensor(M_prev.astype(np.float32)), self.input_ln, self.rms_eps)
        qkv_source = x_normed + m_normed
        
        # 2. Gaussian Attention (or bypass -- see use_attention docstring)
        if self.use_attention:
            q = self.q_proj.forward(qkv_source, learning_rate)
            k = self.k_proj.forward(qkv_source, learning_rate)
            v = self.v_proj.forward(qkv_source, learning_rate)
            sigmas = exp(self.log_sigmas)

            attn = gaussian_attention(q, k, v, self.centers, sigmas,
                                      num_cpus=self.num_cpus, causal=False)
            attn, aux_loss = _apply_energy(self.energy, attn, self.num_tiles, self.state_width)
            attn = self._apply_o_proj(attn, learning_rate)
        else:
            attn = self._apply_o_proj(qkv_source, learning_rate)
            aux_loss = None

        # 3. Residual & Hard Bounding
        M_new_t = Tensor(M_prev.astype(np.float32)) + attn
        M_new_t = rmsnorm_tensor(M_new_t, self.state_ln, self.rms_eps)
        
        # Hard clip bounds the state to avoid exploding activations over time.
        # Direct .data modification bypasses any autograd tracking for the clip.
        M_new_t.data = np.clip(M_new_t.data, -2.0, 2.0)

        # 4. Generate Logits
        pooled = M_new_t.reshape((self.num_tiles, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)  # [num_tiles, embed_width]
        logits = self.lm_head.forward(pooled, learning_rate)  # [num_tiles, vocab_size]
        
        return M_new_t.data, logits, aux_loss
