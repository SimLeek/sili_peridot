"""
model/toy_tile_recurrence_rmt_torch.py
────────────────────────────────────────
Plain-torch EXACT port of model/toy_tile_recurrence_rmt.py
(ToyTileRecurrenceRMT), built per direct instruction ONLY after the
sili-based control failed to learn K=1 MQAR at either precision (see
conversation) -- "move the model to torch as exactly the same as is
possible", NOT swap in torch-idiomatic defaults (Adam instead of the
real per-synapse update, different clip/attention/norm conventions,
etc.). The whole point is isolating "is sili__new the engine broken"
from "is even a correctly-implemented proven architecture failing
here" -- that requires holding every other variable fixed.

Design: uses REAL torch autograd for gradient PROPAGATION through the
whole step (one connected graph, one loss.backward() call, exactly
like sili's own single topological backward pass) -- this is the safe
way to get correct chain-rule gradients through concat/RMSNorm/
attention without hand-deriving a backward pass (an earlier draft of
this file tried that and was correctly rejected: a bug in a manually-
derived backward would itself become a torch-port-specific confound,
defeating the entire purpose of this control). The ONLY place this
deviates from ordinary torch training is the WEIGHT UPDATE itself:
DISLDOTorchLinear does NOT use torch.optim on its weights -- after the
single backward() call, each layer reads its own true_weight.grad
(exactly g=dL/d(true_weight), matching disldo_backward's own g
signal) and applies sili's real per-synapse update rule by hand.

DISLDOTorchLinear reproduces sili__new's actual training rule,
extracted directly from the C++ source rather than guessed:
  - Per-synapse `ci` (RMSprop-style second moment) and weight update:
    delta_csr_types.hpp's BoundedRMSpropSynapsePolicy (the production
    default) -- update_ci/update_cw, beta2=0.999, eps=1e-8,
    max_abs_delta=2.0 (raw-space, matches cpu_backend.cpp's real
    default), max_ci=100.0 (cpu_backend.cpp's chosen production
    default, not the function's own 1e30 no-op default),
    min_decay_frac=0.0 (production default, a true no-op per its own
    docstring).
  - g (per-synapse backward-sensitivity) = dL/d(true_weight), read
    directly from true_weight.grad after backward() -- exactly matches
    the C++ formula's g=dy*x (standard linear-layer weight gradient,
    aggregated/summed over the batch/token dimension already by
    construction of ordinary matmul backprop).
  - contrib (per-synapse forward-contribution) = true_weight *
    x.sum(dim=0) (summed over the batch/token dimension, matching the
    C++ formula's own row-aggregation).
  - Row-level value_scale / column-level output_scale: delta_csr_
    types.hpp's RMSpropScalePolicy (the production default) --
    Adam-style bias-corrected RMSprop, g_agg/contrib_agg = the SAME
    per-synapse g/contrib further summed across the row (for
    value_scale) or column (for output_scale). scale_rank=1 (the
    default this project actually uses -- no rank>1 machinery here).
  - true_weight = w_stored * value_scale[row] * output_scale[col]
    (scale_rank=1's exact combination rule).
  - `lr_per_row_nnz` degree normalization (DISLDOLayer.forward's own
    default, True): effective_lr = learning_rate / row_degree. For a
    FULLY DENSE row (this reference always uses dense connectivity --
    see note below), row_degree = out_features for every row, so this
    is a flat 1/out_features rescale, still reproduced exactly since
    it's part of the real formula ToyTileRecurrenceRMT's own calls
    actually hit (no lr_per_row_nnz=False override anywhere there).
  - Hard clip (state/attn-output bounding): sili applies this via
    direct `.data` mutation, BYPASSING autograd entirely -- the
    gradient flows through AS IF the clip never happened (identity
    backward), NOT the zero-outside-bounds gradient plain
    `torch.clamp` gives by default. Reproduced here via an explicit
    straight-through clip (`x + (clamp(x)-x).detach()`), not bare
    `torch.clamp`.

NOTE on connectivity: sili's own DISLDOLayer/TrueMultiDigitLayer
support sparse (echo-network) OR dense (block4-loaded) connectivity;
this torch port only implements the DENSE case (a plain [in,out]
matrix, no synaptogenesis/pruning) -- matching what the fp4 arm of
train_mqar_rmt_reference.py actually used (dense=True), and
deliberately NOT reproducing the fp32 arm's dense=False (sparse
echo-network) path, which task #232 flagged as a real, separate
confound in that earlier comparison, not something to carry forward
here. This torch reference is a dense-vs-dense comparison against the
sili fp4 arm specifically.

Explicitly NOT reproduced (pure C++ performance/memory-layout details
with no effect on the mathematical result for a fixed, dense,
non-growing topology): deferred-write batching, block4 tile storage,
scale_rank>1, FP4/FP8 quantization codecs themselves (this port is
fp32-only, matching DISLDOLayer32's own math exactly minus the
quantize/dequantize round-trip a real 4-bit/8-bit storage would add).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


def straight_through_clip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Forward = clamp(x, lo, hi); backward = identity (gradient flows
    through unchanged) -- matches sili's own `.data`-mutation clip,
    which bypasses autograd entirely rather than zeroing gradient
    outside the bounds the way bare torch.clamp's default backward
    would."""
    return x + (torch.clamp(x, lo, hi) - x).detach()


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rrms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).pow(-0.5)
    return x * rrms * weight


def gaussian_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             centers: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Exact port of attention.hpp's gaussian_attention_forward: dot-
    product score * 1/sqrt(d), plus a per-query Gaussian positional
    bias -(j-center)^2/(2*sigma^2), softmax over keys, weighted V sum."""
    d = q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    scores = (q @ k.transpose(-1, -2)) * scale                       # [T, K]
    K = scores.shape[-1]
    positions = torch.arange(K, dtype=torch.float32)
    diff = positions.unsqueeze(0) - centers.unsqueeze(1)             # [T, K]
    scores = scores - diff.pow(2) / (2.0 * sigmas.unsqueeze(1).pow(2))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


class DISLDOTorchLinear:
    """NOT an nn.Module (deliberately) -- no torch.optim ever touches
    `w_stored`/`value_scale`/`output_scale`, only the custom update
    below does. `true_weight` (the actual matmul operand) is a fresh
    leaf tensor with requires_grad=True created on every forward()
    call, so it participates correctly in the SAME big autograd graph
    as everything else in one step() -- after the caller's single
    backward() call, `true_weight.grad` gives exactly the g signal
    disldo_backward's own formula needs, with zero hand-derived
    backward-chain risk."""

    def __init__(self, in_features: int, out_features: int,
                 beta2: float = 0.999, eps: float = 1e-8,
                 max_abs_delta: float = 2.0, max_ci: float = 100.0,
                 min_decay_frac: float = 0.0, lr_per_row_nnz: bool = True,
                 rng: Optional[np.random.Generator] = None,
                 bias_correct_ci: bool = False, use_momentum: bool = False,
                 momentum_beta1: float = 0.9, include_contrib_in_ci: bool = True,
                 clip_raw_delta: bool = True):
        """The five bias_correct_ci/use_momentum/momentum_beta1/
        include_contrib_in_ci/clip_raw_delta args default to exactly
        the production C++ BoundedRMSpropSynapsePolicy's own behavior
        (no bias correction, no momentum, contrib^2 included, raw
        delta hard-clipped) -- they exist to let single-factor
        optimizer-internals ablation run cheaply in torch before
        touching delta_csr_types.hpp, not to change the default. Note
        the asymmetry this exposes: RMSpropScalePolicy's own
        _scale_update DOES bias-correct its EMA (see below) while this
        per-synapse update never has -- bias_correct_ci=True makes the
        two consistent."""
        self.in_features = in_features
        self.out_features = out_features
        self.beta2 = beta2
        self.eps = eps
        self.max_abs_delta = max_abs_delta
        self.max_ci = max_ci
        self.min_decay_frac = min_decay_frac
        self.lr_per_row_nnz = lr_per_row_nnz
        self.bias_correct_ci = bias_correct_ci
        self.use_momentum = use_momentum
        self.momentum_beta1 = momentum_beta1
        self.include_contrib_in_ci = include_contrib_in_ci
        self.clip_raw_delta = clip_raw_delta

        if rng is None:
            rng = np.random.default_rng()
        # Matches sili's own _preseed_random_sparse/_preseed_dense init
        # convention: scale = 1/sqrt(n_outputs), full fan-out for a
        # dense row.
        init_scale = 1.0 / np.sqrt(out_features)
        w0 = (rng.standard_normal((in_features, out_features)) * init_scale).astype(np.float32)
        self.w_stored = torch.tensor(w0, dtype=torch.float32)
        self.value_scale = torch.ones(in_features, dtype=torch.float32)
        self.output_scale = torch.ones(out_features, dtype=torch.float32)
        self.ci = torch.zeros((in_features, out_features), dtype=torch.float32)
        self.m = torch.zeros((in_features, out_features), dtype=torch.float32)
        self.ci_step = 0
        self.value_scale_state = torch.zeros(in_features, dtype=torch.float32)
        self.output_scale_state = torch.zeros(out_features, dtype=torch.float32)
        self.value_scale_step = 0
        self.output_scale_step = 0
        self._pending: List[Tuple[torch.Tensor, torch.Tensor, float, bool]] = []

    def forward(self, x: torch.Tensor, learning_rate: float,
                damp_by_importance: bool = True) -> torch.Tensor:
        """x: [n_rows, in_features] (part of the SAME step's autograd
        graph -- must NOT be detached by the caller). Returns
        [n_rows, out_features], still part of that graph."""
        true_weight = (self.w_stored * self.value_scale.unsqueeze(1)
                       * self.output_scale.unsqueeze(0)).clone().requires_grad_(True)
        true_weight.retain_grad()
        y = x @ true_weight
        self._pending.append((x, true_weight, learning_rate, damp_by_importance))
        return y

    def apply_pending_updates(self) -> None:
        """Call AFTER the whole step's single loss.backward() -- reads
        each forward() call's true_weight.grad (real autograd, exactly
        g=dL/d(true_weight)) and performs the real inline update.
        Multiple forward() calls to the SAME layer within one step
        (e.g. the L1 split's own second undamped call) each get their
        own independent update, applied in call order -- matching
        sili's own per-call inline-training semantics exactly (each
        `layer.forward(x, lr)` call is its own training step there
        too)."""
        pending, self._pending = self._pending, []
        for x, true_weight, lr, damp in pending:
            g = true_weight.grad
            if g is None or lr == 0.0:
                continue
            x_det = x.detach()
            w_det = true_weight.detach()

            row_degree = self.out_features  # fully dense: every row's true degree
            eff_lr = (lr / row_degree) if self.lr_per_row_nnz else lr

            x_sum = x_det.sum(dim=0)                            # [in]
            contrib = w_det * x_sum.unsqueeze(1)                # [in, out]

            sq_term = g * g
            if self.include_contrib_in_ci:
                sq_term = sq_term + contrib * contrib
            ema = self.beta2 * self.ci + (1.0 - self.beta2) * sq_term
            floor = self.min_decay_frac * self.ci
            self.ci = torch.clamp(torch.maximum(ema, floor), max=self.max_ci)
            self.ci_step += 1

            if self.bias_correct_ci:
                bc2 = 1.0 - self.beta2 ** self.ci_step
                ci_hat = self.ci / bc2 if bc2 > 0 else self.ci
            else:
                ci_hat = self.ci

            if self.use_momentum:
                self.m = self.momentum_beta1 * self.m + (1.0 - self.momentum_beta1) * g
                if self.bias_correct_ci:
                    bc1 = 1.0 - self.momentum_beta1 ** self.ci_step
                    numerator = -(self.m / bc1 if bc1 > 0 else self.m)
                else:
                    numerator = -self.m
            else:
                numerator = -g

            raw = numerator / (torch.sqrt(ci_hat) + self.eps) if damp else numerator
            if self.clip_raw_delta:
                raw = torch.clamp(raw, -self.max_abs_delta, self.max_abs_delta)
            self.w_stored = self.w_stored + eff_lr * raw

            g_row = g.sum(dim=1)
            contrib_row = contrib.sum(dim=1)
            self.value_scale_step += 1
            self._scale_update("value_scale", g_row, contrib_row, eff_lr, self.value_scale_step)

            g_col = g.sum(dim=0)
            contrib_col = contrib.sum(dim=0)
            self.output_scale_step += 1
            self._scale_update("output_scale", g_col, contrib_col, eff_lr, self.output_scale_step)

    def _scale_update(self, name: str, g_agg: torch.Tensor, contrib_agg: torch.Tensor,
                       eff_lr: float, step: int) -> None:
        scale = getattr(self, name)
        state = getattr(self, f"{name}_state")
        new_state = self.beta2 * state + (1.0 - self.beta2) * (g_agg * g_agg + contrib_agg * contrib_agg)
        bias_correction = 1.0 - self.beta2 ** step
        state_hat = new_state / bias_correction if bias_correction > 0 else new_state
        new_scale = scale - eff_lr * g_agg / (torch.sqrt(state_hat) + self.eps)
        if torch.isfinite(new_state).all() and torch.isfinite(new_scale).all():
            setattr(self, f"{name}_state", new_state)
            setattr(self, name, new_scale)


class ToyTileRecurrenceRMTTorch:
    """Exact-as-possible torch port of ToyTileRecurrenceRMT -- see this
    module's own docstring for the full fidelity notes. Only the
    implementation language/engine differs from
    model/toy_tile_recurrence_rmt.py; every architectural/optimizer
    choice is intentionally identical."""

    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int,
                 num_tiles: int, num_memory_slots: int, rms_eps: float = 1e-6,
                 clip_range: float = 6.0, l1_sparsity_coef: float = 0.0,
                 rng: Optional[np.random.Generator] = None):
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.num_memory_slots = num_memory_slots
        self.total_slots = num_tiles + num_memory_slots
        self.rms_eps = rms_eps
        self.clip_range = clip_range
        self.l1_sparsity_coef = l1_sparsity_coef

        if rng is None:
            rng = np.random.default_rng()
        sw = self.state_width

        self.input_proj = DISLDOTorchLinear(embed_width, sw, rng=rng)
        self.q_proj = DISLDOTorchLinear(sw, sw, rng=rng)
        self.k_proj = DISLDOTorchLinear(sw, sw, rng=rng)
        self.v_proj = DISLDOTorchLinear(sw, sw, rng=rng)
        self.o_proj = DISLDOTorchLinear(sw, sw, rng=rng)
        self.lm_head = DISLDOTorchLinear(embed_width, vocab_size, rng=rng)
        self._real_layers = [self.input_proj, self.q_proj, self.k_proj,
                             self.v_proj, self.o_proj, self.lm_head]

        self.input_ln = torch.ones(sw, requires_grad=True)
        self.memory_ln = torch.ones(sw, requires_grad=True)
        self.state_ln = torch.ones(sw, requires_grad=True)
        self.centers = torch.tensor([i + 0.5 for i in range(self.total_slots)],
                                    dtype=torch.float32, requires_grad=True)
        self.log_sigmas = torch.zeros(self.total_slots, requires_grad=True)

    def parameters_for_optimizer(self) -> List[torch.Tensor]:
        return [self.input_ln, self.memory_ln, self.state_ln, self.centers, self.log_sigmas]

    def zero_grad(self) -> None:
        for p in self.parameters_for_optimizer():
            p.grad = None

    def _l1_sparsity_split(self, layer: DISLDOTorchLinear, input_t: torch.Tensor,
                           lr: float, coef: float) -> torch.Tensor:
        out_aux = layer.forward(input_t, lr, damp_by_importance=False)
        n = float(out_aux.numel())
        return out_aux.abs().sum() * (coef / n)

    def step(self, x_window: np.ndarray, memory_prev: np.ndarray,
             learning_rate: float) -> Tuple[np.ndarray, torch.Tensor, Optional[torch.Tensor]]:
        n_mem, n_content = self.num_memory_slots, self.num_tiles

        x_window_t = torch.tensor(x_window, dtype=torch.float32)
        x_wide = self.input_proj.forward(x_window_t, learning_rate)
        x_normed = rmsnorm_torch(x_wide, self.input_ln, self.rms_eps)

        memory_prev_t = torch.tensor(memory_prev, dtype=torch.float32)
        memory_normed = rmsnorm_torch(memory_prev_t, self.memory_ln, self.rms_eps)

        combined_normed = torch.cat([memory_normed, x_normed], dim=0)

        q = self.q_proj.forward(combined_normed, learning_rate)
        k = self.k_proj.forward(combined_normed, learning_rate)
        v = self.v_proj.forward(combined_normed, learning_rate)
        sigmas = torch.exp(self.log_sigmas)
        attn = gaussian_attention_torch(q, k, v, self.centers, sigmas)
        attn = self.o_proj.forward(attn, learning_rate)
        attn_clipped = straight_through_clip(attn, -self.clip_range, self.clip_range)

        aux_loss = None
        if self.l1_sparsity_coef > 0.0:
            l1_terms = [
                self._l1_sparsity_split(self.input_proj, x_window_t, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.q_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.k_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.v_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.o_proj, gaussian_attention_torch(q, k, v, self.centers, sigmas),
                                         learning_rate, self.l1_sparsity_coef),
            ]
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term

        raw_combined = torch.cat([memory_prev_t, x_wide], dim=0)
        combined_new = raw_combined + attn_clipped
        combined_new = rmsnorm_torch(combined_new, self.state_ln, self.rms_eps)
        combined_new = straight_through_clip(combined_new, -self.clip_range, self.clip_range)

        content_out = combined_new[n_mem:]
        pooled = content_out.reshape(n_content, self.embed_width, self.column_neurons).mean(dim=-1)
        if self.l1_sparsity_coef > 0.0:
            lm_l1 = self._l1_sparsity_split(self.lm_head, pooled, learning_rate, self.l1_sparsity_coef)
            aux_loss = lm_l1 if aux_loss is None else aux_loss + lm_l1
        logits = self.lm_head.forward(pooled, learning_rate)

        self._combined_new_for_memory = combined_new

        return None, logits, aux_loss

    def extract_memory(self) -> np.ndarray:
        """Call AFTER backward() (the graph must survive until then) --
        returns the plain-numpy memory carry for the next step()."""
        n_mem = self.num_memory_slots
        return self._combined_new_for_memory.detach()[:n_mem].numpy().copy()

    def apply_updates(self) -> None:
        """Call AFTER backward() -- runs every real weight layer's
        pending inline update in one pass."""
        for layer in self._real_layers:
            layer.apply_pending_updates()


def clip_grad_norm_(params: List[torch.Tensor], max_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], max_norm)
