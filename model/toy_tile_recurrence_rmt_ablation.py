"""
model/toy_tile_recurrence_rmt_ablation.py
────────────────────────────────────────────
Single-factor ablation ladder between the exact-fidelity torch RMT
port (model/toy_tile_recurrence_rmt_torch.py, task #232, acc=0.433)
and the genuinely standard torch RMT reference (model/
toy_tile_recurrence_rmt_standard.py, task #233, acc=1.000 -- confirms
both of this project's own RMT builds share a real, fixable problem,
since a standard implementation solves the identical task/harness
essentially perfectly).

One flexible model, four independent boolean toggles -- each swaps
ONE exact-fidelity-port component for its standard-reference
counterpart, all four run independently (not stopping early on any
single result) to tell whether the effective cause is one factor or
several interacting ones, per direct instruction:
  - use_custom_optimizer: DISLDOTorchLinear's real per-synapse
    RMSprop-with-importance update (True, the exact-fidelity port's
    own choice) vs plain torch.optim.Adam over every parameter
    (False, the standard reference's own choice).
  - use_hard_clip: straight-through hard clip on attn/state (True) vs
    no clip at all (False).
  - use_gaussian_bias: Gaussian positional-bias attention (True) vs a
    learned absolute positional embedding + standard softmax attention
    (False).
  - use_rmsnorm: RMSNorm (True) vs standard LayerNorm (False).

All torch-only, deliberately -- no sili engine involved anywhere in
this file, keeping the engine variable fixed so it can't reintroduce
the same confound this whole investigation branch exists to avoid.

l1_sparsity_coef stays available but should be set to 0 for a clean
4-way comparison against the standard reference (which has no L1 term
at all) -- a 5th real difference between the exact-fidelity port and
the standard reference, not one of the four named here; disabling it
uniformly across all 4 ablation runs keeps it out of this specific
test rather than silently confounding it.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .toy_tile_recurrence_rmt_torch import (
    DISLDOTorchLinear, straight_through_clip, rmsnorm_torch, gaussian_attention_torch,
)


class PlainAdamLinear:
    """Adapter matching DISLDOTorchLinear's forward(x, lr, damp_by_
    importance)/apply_pending_updates() call interface, but backed by
    a plain nn.Linear (no bias, matching DISLDOTorchLinear's own
    bias-free convention for a fair single-axis swap) trained entirely
    by a SHARED external torch.optim.Adam -- no custom update logic at
    all. apply_pending_updates() is a genuine no-op here (the shared
    optimizer's own opt.step() handles everything)."""

    def __init__(self, in_features: int, out_features: int):
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, learning_rate: float,
                damp_by_importance: bool = True) -> torch.Tensor:
        return self.linear(x)

    def apply_pending_updates(self) -> None:
        pass

    def parameters(self):
        return self.linear.parameters()


class ToyTileRecurrenceRMTAblation:
    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int,
                 num_tiles: int, num_memory_slots: int, rms_eps: float = 1e-6,
                 clip_range: float = 6.0, l1_sparsity_coef: float = 0.0,
                 use_custom_optimizer: bool = True, use_hard_clip: bool = True,
                 use_gaussian_bias: bool = True, use_rmsnorm: bool = True,
                 rng: Optional[np.random.Generator] = None,
                 optimizer_kwargs: Optional[dict] = None):
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.num_memory_slots = num_memory_slots
        self.total_slots = num_tiles + num_memory_slots
        self.rms_eps = rms_eps
        self.clip_range = clip_range
        self.l1_sparsity_coef = l1_sparsity_coef
        self.use_custom_optimizer = use_custom_optimizer
        self.use_hard_clip = use_hard_clip
        self.use_gaussian_bias = use_gaussian_bias
        self.use_rmsnorm = use_rmsnorm

        if rng is None:
            rng = np.random.default_rng()
        sw = self.state_width
        self.optimizer_kwargs = optimizer_kwargs or {}

        def make_linear(in_f, out_f):
            return DISLDOTorchLinear(in_f, out_f, rng=rng, **self.optimizer_kwargs) \
                if use_custom_optimizer else PlainAdamLinear(in_f, out_f)

        self.input_proj = make_linear(embed_width, sw)
        self.q_proj = make_linear(sw, sw)
        self.k_proj = make_linear(sw, sw)
        self.v_proj = make_linear(sw, sw)
        self.o_proj = make_linear(sw, sw)
        self.lm_head = make_linear(embed_width, vocab_size)
        self._real_layers = [self.input_proj, self.q_proj, self.k_proj,
                             self.v_proj, self.o_proj, self.lm_head]

        self._adam_params: List[torch.Tensor] = []

        if use_rmsnorm:
            self.input_ln = torch.ones(sw, requires_grad=True)
            self.memory_ln = torch.ones(sw, requires_grad=True)
            self.state_ln = torch.ones(sw, requires_grad=True)
            self._adam_params += [self.input_ln, self.memory_ln, self.state_ln]
        else:
            self.input_ln_mod = nn.LayerNorm(sw)
            self.memory_ln_mod = nn.LayerNorm(sw)
            self.state_ln_mod = nn.LayerNorm(sw)
            self._adam_params += list(self.input_ln_mod.parameters())
            self._adam_params += list(self.memory_ln_mod.parameters())
            self._adam_params += list(self.state_ln_mod.parameters())

        if use_gaussian_bias:
            self.centers = torch.tensor([i + 0.5 for i in range(self.total_slots)],
                                        dtype=torch.float32, requires_grad=True)
            self.log_sigmas = torch.zeros(self.total_slots, requires_grad=True)
            self._adam_params += [self.centers, self.log_sigmas]
        else:
            self.pos_embed = nn.Parameter(torch.zeros(self.total_slots, sw))
            nn.init.normal_(self.pos_embed, std=0.02)
            self._adam_params += [self.pos_embed]

        if not use_custom_optimizer:
            for layer in self._real_layers:
                self._adam_params += list(layer.parameters())

    def parameters_for_optimizer(self) -> List[torch.Tensor]:
        return self._adam_params

    def zero_grad(self) -> None:
        for p in self._adam_params:
            p.grad = None

    def _norm(self, x: torch.Tensor, which: str) -> torch.Tensor:
        if self.use_rmsnorm:
            return rmsnorm_torch(x, getattr(self, which), self.rms_eps)
        return getattr(self, f"{which}_mod")(x)

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return straight_through_clip(x, -self.clip_range, self.clip_range) if self.use_hard_clip else x

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.use_gaussian_bias:
            sigmas = torch.exp(self.log_sigmas)
            return gaussian_attention_torch(q, k, v, self.centers, sigmas)
        d = q.shape[-1]
        scores = (q @ k.transpose(-1, -2)) / (d ** 0.5)
        return torch.softmax(scores, dim=-1) @ v

    def _l1_sparsity_split(self, layer, input_t: torch.Tensor, lr: float, coef: float) -> torch.Tensor:
        out_aux = layer.forward(input_t, lr, damp_by_importance=False)
        n = float(out_aux.numel())
        return out_aux.abs().sum() * (coef / n)

    def step(self, x_window: np.ndarray, memory_prev: np.ndarray,
             learning_rate: float) -> Tuple[np.ndarray, torch.Tensor, Optional[torch.Tensor]]:
        n_mem, n_content = self.num_memory_slots, self.num_tiles

        x_window_t = torch.tensor(x_window, dtype=torch.float32)
        x_wide = self.input_proj.forward(x_window_t, learning_rate)
        x_normed = self._norm(x_wide, "input_ln")

        memory_prev_t = torch.tensor(memory_prev, dtype=torch.float32)
        memory_normed = self._norm(memory_prev_t, "memory_ln")

        combined_normed = torch.cat([memory_normed, x_normed], dim=0)
        if not self.use_gaussian_bias:
            combined_normed = combined_normed + self.pos_embed

        q = self.q_proj.forward(combined_normed, learning_rate)
        k = self.k_proj.forward(combined_normed, learning_rate)
        v = self.v_proj.forward(combined_normed, learning_rate)
        attn = self._attend(q, k, v)
        attn = self.o_proj.forward(attn, learning_rate)
        attn_clipped = self._clip(attn)

        aux_loss = None
        if self.l1_sparsity_coef > 0.0:
            l1_terms = [
                self._l1_sparsity_split(self.input_proj, x_window_t, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.q_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.k_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.v_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.o_proj, self._attend(q, k, v), learning_rate, self.l1_sparsity_coef),
            ]
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term

        raw_combined = torch.cat([memory_prev_t, x_wide], dim=0)
        combined_new = raw_combined + attn_clipped
        combined_new = self._norm(combined_new, "state_ln")
        combined_new = self._clip(combined_new)

        content_out = combined_new[n_mem:]
        pooled = content_out.reshape(n_content, self.embed_width, self.column_neurons).mean(dim=-1)
        if self.l1_sparsity_coef > 0.0:
            lm_l1 = self._l1_sparsity_split(self.lm_head, pooled, learning_rate, self.l1_sparsity_coef)
            aux_loss = lm_l1 if aux_loss is None else aux_loss + lm_l1
        logits = self.lm_head.forward(pooled, learning_rate)

        self._combined_new_for_memory = combined_new
        return None, logits, aux_loss

    def extract_memory(self) -> np.ndarray:
        n_mem = self.num_memory_slots
        return self._combined_new_for_memory.detach()[:n_mem].numpy().copy()

    def apply_updates(self) -> None:
        for layer in self._real_layers:
            layer.apply_pending_updates()


def clip_grad_norm_(params: List[torch.Tensor], max_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], max_norm)
