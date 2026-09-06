"""
model/toy_tile_recurrence_rmt_torch.py
Plain-torch exact port of model/toy_tile_recurrence_rmt.py
(ToyTileRecurrenceRMT), an engine-vs-architecture control.
See docs/research/toy_tile_recurrence_rmt_torch.rst:
toy_tile_recurrence_rmt_torch.module_overview,
toy_tile_recurrence_rmt_torch.disldo_reproduced_training_rule,
toy_tile_recurrence_rmt_torch.dense_only_connectivity, and
toy_tile_recurrence_rmt_torch.not_reproduced_performance_details.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .fake_quantize_torch import fake_quantize


def straight_through_clip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """See docs/research/toy_tile_recurrence_rmt_torch.rst:
    toy_tile_recurrence_rmt_torch.straight_through_clip_design."""
    return x + (torch.clamp(x, lo, hi) - x).detach()


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rrms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).pow(-0.5)
    return x * rrms * weight


def gaussian_attention_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, centers: torch.Tensor, sigmas: torch.Tensor
) -> torch.Tensor:
    """Exact port of attention.hpp's gaussian_attention_forward.
    See docs/research/toy_tile_recurrence_rmt.rst:toy_tile_recurrence_rmt.module_overview."""
    d = q.shape[-1]
    scale = 1.0 / (d**0.5)
    scores = (q @ k.transpose(-1, -2)) * scale  # [T, K]
    K = scores.shape[-1]
    positions = torch.arange(K, dtype=torch.float32)
    diff = positions.unsqueeze(0) - centers.unsqueeze(1)  # [T, K]
    scores = scores - diff.pow(2) / (2.0 * sigmas.unsqueeze(1).pow(2))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


class DISLDOTorchLinear:
    """See docs/research/toy_tile_recurrence_rmt_torch.rst:
    toy_tile_recurrence_rmt_torch.disldo_torch_linear_autograd_leaf."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_abs_delta: float = 2.0,
        max_ci: float = 100.0,
        min_decay_frac: float = 0.0,
        lr_per_row_nnz: bool = True,
        rng: np.random.Generator | None = None,
        bias_correct_ci: bool = False,
        use_momentum: bool = False,
        momentum_beta1: float = 0.9,
        include_contrib_in_ci: bool = True,
        clip_raw_delta: bool = True,
        include_scale_chain_rule: bool = False,
        clip_pre_ci: bool = False,
        pre_ci_clip_value: float | None = None,
        scale_grad_diag_fn: Callable | None = None,
        use_magnitude_scale: bool = False,
        magnitude_scale_target: float = 16.0,
        magnitude_correction_rate: float = 0.01,
        fake_quantize_kind: str | None = None,
        scale_invariant_chain_rule: bool = False,
        fake_quantize_stochastic: bool = False,
        magnitude_rescale_ema_beta: float | None = None,
        magnitude_scale_both_axes: bool = False,
    ):
        """Ablation-flag defaults; see docs/research/toy_tile_recurrence_rmt_torch.rst:
        toy_tile_recurrence_rmt_torch.ablation_flag_defaults (bias_correct_ci/
        use_momentum/momentum_beta1/include_contrib_in_ci/clip_raw_delta),
        toy_tile_recurrence_rmt_torch.include_scale_chain_rule_bug
        (include_scale_chain_rule), toy_tile_recurrence_rmt_torch.use_magnitude_scale_rank1_limit
        (use_magnitude_scale), and toy_tile_recurrence_rmt_torch.clip_pre_ci_ci_jump_bug
        (clip_pre_ci)."""
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
        self.include_scale_chain_rule = include_scale_chain_rule
        self.momentum_beta1 = momentum_beta1
        self.include_contrib_in_ci = include_contrib_in_ci
        self.clip_raw_delta = clip_raw_delta
        self.clip_pre_ci = clip_pre_ci
        self.pre_ci_clip_value = pre_ci_clip_value if pre_ci_clip_value is not None else max_abs_delta
        self.scale_grad_diag_fn = scale_grad_diag_fn
        self.use_magnitude_scale = use_magnitude_scale
        self.magnitude_scale_target = magnitude_scale_target
        self.magnitude_correction_rate = magnitude_correction_rate
        self.fake_quantize_kind = fake_quantize_kind
        self.scale_invariant_chain_rule = scale_invariant_chain_rule
        self.fake_quantize_stochastic = fake_quantize_stochastic
        self.magnitude_rescale_ema_beta = magnitude_rescale_ema_beta
        self.col_rms_ema: torch.Tensor | None = None
        self.magnitude_scale_both_axes = magnitude_scale_both_axes
        self.row_rms_ema: torch.Tensor | None = None

        if rng is None:
            rng = np.random.default_rng()
        # Matches sili's own dense-layer init convention: scale = 1/sqrt(out_features).
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
        self._pending: list[tuple[torch.Tensor, torch.Tensor, float, bool]] = []

    def forward(self, x: torch.Tensor, learning_rate: float, damp_by_importance: bool = True) -> torch.Tensor:
        """x: [n_rows, in_features] (part of the SAME step's autograd
        graph -- must NOT be detached by the caller). Returns
        [n_rows, out_features], still part of that graph."""
        true_weight = (
            (self.w_stored * self.value_scale.unsqueeze(1) * self.output_scale.unsqueeze(0))
            .clone()
            .requires_grad_(True)
        )
        true_weight.retain_grad()
        y = x @ true_weight
        self._pending.append((x, true_weight, learning_rate, damp_by_importance))
        return y

    def apply_pending_updates(self) -> None:
        """See docs/research/toy_tile_recurrence_rmt_torch.rst:
        toy_tile_recurrence_rmt_torch.disldo_torch_linear_autograd_leaf."""
        pending, self._pending = self._pending, []
        for x, true_weight, lr, damp in pending:
            g = true_weight.grad
            if g is None or lr == 0.0:
                continue
            x_det = x.detach()
            w_det = true_weight.detach()

            row_degree = self.out_features  # fully dense: every row's true degree
            eff_lr = (lr / row_degree) if self.lr_per_row_nnz else lr

            x_sum = x_det.sum(dim=0)  # [in]
            contrib = w_det * x_sum.unsqueeze(1)  # [in, out]

            # See docs/research/toy_tile_recurrence_rmt_torch.rst:
            # toy_tile_recurrence_rmt_torch.include_scale_chain_rule_bug.
            if self.include_scale_chain_rule:
                S = self.value_scale.unsqueeze(1) * self.output_scale.unsqueeze(0)  # [in, out]
                g_for_w = g * S
                g_row_signal = self.w_stored * self.output_scale.unsqueeze(0) * g
                contrib_row_signal = self.w_stored * self.output_scale.unsqueeze(0) * contrib
                g_col_signal = self.w_stored * self.value_scale.unsqueeze(1) * g
                contrib_col_signal = self.w_stored * self.value_scale.unsqueeze(1) * contrib
            else:
                g_for_w = g
                g_row_signal = g
                contrib_row_signal = contrib
                g_col_signal = g
                contrib_col_signal = contrib

            g_for_ci, contrib_for_ci = g, contrib
            if self.clip_pre_ci:
                g_for_ci = torch.clamp(g, -self.pre_ci_clip_value, self.pre_ci_clip_value)
                contrib_for_ci = torch.clamp(contrib, -self.pre_ci_clip_value, self.pre_ci_clip_value)
            sq_term = g_for_ci * g_for_ci
            if self.include_contrib_in_ci:
                sq_term = sq_term + contrib_for_ci * contrib_for_ci
            ema = self.beta2 * self.ci + (1.0 - self.beta2) * sq_term
            floor = self.min_decay_frac * self.ci
            new_ci = torch.clamp(torch.maximum(ema, floor), max=self.max_ci)
            if torch.isfinite(new_ci).all():
                self.ci = new_ci
            self.ci_step += 1

            if self.bias_correct_ci:
                bc2 = 1.0 - self.beta2**self.ci_step
                ci_hat = self.ci / bc2 if bc2 > 0 else self.ci
            else:
                ci_hat = self.ci

            # See docs/research/toy_tile_recurrence_rmt_torch.rst:
            # toy_tile_recurrence_rmt_torch.scale_invariant_chain_rule_quadratic_bug.
            numerator_for_ci_norm = (
                -g if (self.scale_invariant_chain_rule and self.include_scale_chain_rule) else -g_for_w
            )

            if self.use_momentum:
                self.m = self.momentum_beta1 * self.m + (1.0 - self.momentum_beta1) * numerator_for_ci_norm
                if self.bias_correct_ci:
                    bc1 = 1.0 - self.momentum_beta1**self.ci_step
                    numerator = self.m / bc1 if bc1 > 0 else self.m
                else:
                    numerator = self.m
            else:
                numerator = numerator_for_ci_norm

            raw = numerator / (torch.sqrt(ci_hat) + self.eps) if damp else numerator
            if self.clip_raw_delta:
                raw = torch.clamp(raw, -self.max_abs_delta, self.max_abs_delta)
            if self.scale_invariant_chain_rule and self.include_scale_chain_rule:
                delta_w = eff_lr * raw / S
            else:
                delta_w = eff_lr * raw
            new_w = self.w_stored + delta_w
            if torch.isfinite(new_w).all():
                self.w_stored = new_w
            if self.fake_quantize_kind is not None:
                # See docs/research/toy_tile_recurrence_rmt_torch.rst:
                # toy_tile_recurrence_rmt_torch.immediate_requantize_no_float_shadow.
                self.w_stored = fake_quantize(
                    self.w_stored, self.fake_quantize_kind, stochastic=self.fake_quantize_stochastic
                )

            g_row = g_row_signal.sum(dim=1)
            contrib_row = contrib_row_signal.sum(dim=1)
            if self.scale_grad_diag_fn is not None:
                self.scale_grad_diag_fn("value_scale", g_row_signal, g_row)
            self.value_scale_step += 1
            self._scale_update("value_scale", g_row, contrib_row, eff_lr, self.value_scale_step)

            g_col = g_col_signal.sum(dim=0)
            contrib_col = contrib_col_signal.sum(dim=0)
            if self.scale_grad_diag_fn is not None:
                self.scale_grad_diag_fn("output_scale", g_col_signal, g_col)
            self.output_scale_step += 1
            self._scale_update("output_scale", g_col, contrib_col, eff_lr, self.output_scale_step)

            if self.use_magnitude_scale:
                self._magnitude_rescale()  # re-quantizes internally, see its own docstring

    def _magnitude_rescale(self) -> None:
        """See docs/research/toy_tile_recurrence_rmt_torch.rst:
        toy_tile_recurrence_rmt_torch.magnitude_rescale_mechanism."""
        with torch.no_grad():
            instant_col_rms = torch.sqrt((self.w_stored**2).mean(dim=0) + self.eps)
            if self.magnitude_rescale_ema_beta is not None:
                if self.col_rms_ema is None:
                    self.col_rms_ema = instant_col_rms.clone()
                else:
                    beta = self.magnitude_rescale_ema_beta
                    self.col_rms_ema = beta * self.col_rms_ema + (1.0 - beta) * instant_col_rms
                col_rms = self.col_rms_ema
            else:
                col_rms = instant_col_rms
            k = (self.magnitude_scale_target / col_rms).clamp(min=1e-6) ** self.magnitude_correction_rate
            new_w = self.w_stored * k.unsqueeze(0)
            new_output_scale = self.output_scale / k
            new_ci = self.ci if self.scale_invariant_chain_rule else self.ci / k.unsqueeze(0) ** 2
            if torch.isfinite(new_w).all() and torch.isfinite(new_output_scale).all() and torch.isfinite(new_ci).all():
                self.w_stored = new_w
                self.output_scale = new_output_scale
                self.ci = new_ci
                if self.fake_quantize_kind is not None:
                    # See toy_tile_recurrence_rmt_torch.immediate_requantize_no_float_shadow.
                    self.w_stored = fake_quantize(
                        self.w_stored, self.fake_quantize_kind, stochastic=self.fake_quantize_stochastic
                    )

            if self.magnitude_scale_both_axes:
                # See docs/research/toy_tile_recurrence_rmt_torch.rst:
                # toy_tile_recurrence_rmt_torch.magnitude_scale_both_axes.
                row_rms_instant = torch.sqrt((self.w_stored**2).mean(dim=1) + self.eps)
                if self.magnitude_rescale_ema_beta is not None:
                    if self.row_rms_ema is None:
                        self.row_rms_ema = row_rms_instant.clone()
                    else:
                        beta = self.magnitude_rescale_ema_beta
                        self.row_rms_ema = beta * self.row_rms_ema + (1.0 - beta) * row_rms_instant
                    row_rms = self.row_rms_ema
                else:
                    row_rms = row_rms_instant
                k_row = (self.magnitude_scale_target / row_rms).clamp(min=1e-6) ** self.magnitude_correction_rate
                new_w_row = self.w_stored * k_row.unsqueeze(1)
                new_value_scale = self.value_scale / k_row
                new_ci_row = self.ci if self.scale_invariant_chain_rule else self.ci / k_row.unsqueeze(1) ** 2
                if (
                    torch.isfinite(new_w_row).all()
                    and torch.isfinite(new_value_scale).all()
                    and torch.isfinite(new_ci_row).all()
                ):
                    self.w_stored = new_w_row
                    self.value_scale = new_value_scale
                    self.ci = new_ci_row
                    if self.fake_quantize_kind is not None:
                        self.w_stored = fake_quantize(
                            self.w_stored, self.fake_quantize_kind, stochastic=self.fake_quantize_stochastic
                        )

    def _scale_update(
        self, name: str, g_agg: torch.Tensor, contrib_agg: torch.Tensor, eff_lr: float, step: int
    ) -> None:
        scale = getattr(self, name)
        state = getattr(self, f"{name}_state")
        if self.scale_invariant_chain_rule:
            # See docs/research/toy_tile_recurrence_rmt_torch.rst:
            # toy_tile_recurrence_rmt_torch.scale_update_log_space_step.
            log_grad = g_agg * scale
            log_contrib = contrib_agg * scale
            new_state = self.beta2 * state + (1.0 - self.beta2) * (log_grad * log_grad + log_contrib * log_contrib)
            bias_correction = 1.0 - self.beta2**step
            state_hat = new_state / bias_correction if bias_correction > 0 else new_state
            log_step = eff_lr * log_grad / (torch.sqrt(state_hat) + self.eps)
            new_scale = scale * torch.exp(-log_step)
        else:
            new_state = self.beta2 * state + (1.0 - self.beta2) * (g_agg * g_agg + contrib_agg * contrib_agg)
            bias_correction = 1.0 - self.beta2**step
            state_hat = new_state / bias_correction if bias_correction > 0 else new_state
            new_scale = scale - eff_lr * g_agg / (torch.sqrt(state_hat) + self.eps)
        if torch.isfinite(new_state).all() and torch.isfinite(new_scale).all():
            setattr(self, f"{name}_state", new_state)
            setattr(self, name, new_scale)


class ToyTileRecurrenceRMTTorch:
    """See docs/research/toy_tile_recurrence_rmt_torch.rst:
    toy_tile_recurrence_rmt_torch.module_overview."""

    def __init__(
        self,
        vocab_size: int,
        embed_width: int,
        column_neurons: int,
        num_tiles: int,
        num_memory_slots: int,
        rms_eps: float = 1e-6,
        clip_range: float = 6.0,
        l1_sparsity_coef: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
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
        self._real_layers = [self.input_proj, self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.lm_head]

        self.input_ln = torch.ones(sw, requires_grad=True)
        self.memory_ln = torch.ones(sw, requires_grad=True)
        self.state_ln = torch.ones(sw, requires_grad=True)
        self.centers = torch.tensor([i + 0.5 for i in range(self.total_slots)], dtype=torch.float32, requires_grad=True)
        self.log_sigmas = torch.zeros(self.total_slots, requires_grad=True)

    def parameters_for_optimizer(self) -> list[torch.Tensor]:
        return [self.input_ln, self.memory_ln, self.state_ln, self.centers, self.log_sigmas]

    def zero_grad(self) -> None:
        for p in self.parameters_for_optimizer():
            p.grad = None

    def _l1_sparsity_split(
        self, layer: DISLDOTorchLinear, input_t: torch.Tensor, lr: float, coef: float
    ) -> torch.Tensor:
        out_aux = layer.forward(input_t, lr, damp_by_importance=False)
        n = float(out_aux.numel())
        return out_aux.abs().sum() * (coef / n)

    def step(
        self, x_window: np.ndarray, memory_prev: np.ndarray, learning_rate: float
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor | None]:
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
                self._l1_sparsity_split(
                    self.o_proj,
                    gaussian_attention_torch(q, k, v, self.centers, sigmas),
                    learning_rate,
                    self.l1_sparsity_coef,
                ),
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


def clip_grad_norm_(params: list[torch.Tensor], max_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], max_norm)
