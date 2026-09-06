from __future__ import annotations

import numpy as np
from sili.energy import EnergyDynamics
from sili.sparse_rnn import DISLDOLayer
from sili.tensor import Tensor, exp, gaussian_attention, power, reduce_sum, tensor_abs

from .toy_precision_models import _apply_energy, _toy_scale_energy
from .toy_recall_models import rmsnorm_tensor, sigmoid_tensor


class ToyTileRecurrenceRealFP4:
    """ToyTileRecurrence's exact architecture, built from DISLDOLayer-family
    layers. See docs/research/toy_tile_precision_models.rst:module_overview.

    See docs/research/toy_tile_precision_models.rst:known_differences_from_proven_designs
    for known architectural gaps vs proven segment/block-recurrent designs
    (RMT, Block-Recurrent Transformer, Infini-attention, Perceiver IO)."""

    def __init__(
        self,
        vocab_size: int,
        embed_width: int,
        column_neurons: int,
        mlp_hidden: int,
        num_tiles: int,
        max_weights: int,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
        disldo_cls=DISLDOLayer,
        use_energy: bool = False,
        energy_kwargs: dict | None = None,
        use_attention: bool = True,
        o_proj_depth: int = 1,
        dense: bool = False,
        clip_range: float = 6.0,
        magnitude_penalty_coef: float = 0.0,
        spectral_norm_target: float | None = None,
        spectral_norm_ema_decay: float = 0.9,
        l1_sparsity_coef: float = 0.0,
        cosine_lm_head: bool = False,
        gated_combine: bool = False,
        gated_update: bool = False,
        gate_floor: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        """mlp_hidden: unused, kept for API compatibility (MLP block removed).

        use_attention=False: see
        docs/research/toy_tile_precision_models.rst:use_attention_bypass_ablation.

        o_proj_depth>1: see
        docs/research/toy_tile_precision_models.rst:o_proj_depth_cascaded_quantization.

        magnitude_penalty_coef: see
        docs/research/toy_tile_precision_models.rst:magnitude_penalty_coef.

        spectral_norm_target: see
        docs/research/toy_tile_precision_models.rst:spectral_norm_target.

        l1_sparsity_coef: see
        docs/research/toy_tile_precision_models.rst:l1_sparsity_coef_landmark.

        cosine_lm_head: see
        docs/research/toy_tile_precision_models.rst:cosine_lm_head.

        gated_combine/gate_floor: see
        docs/research/toy_tile_precision_models.rst:gated_combine.

        gated_update: see
        docs/research/toy_tile_precision_models.rst:gated_update."""
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.rms_eps = rms_eps
        self.clip_range = clip_range
        self.cosine_lm_head = cosine_lm_head
        self.gated_combine = gated_combine
        self.gated_update = gated_update
        self.gate_floor = gate_floor
        self.magnitude_penalty_coef = magnitude_penalty_coef
        self.spectral_norm_target = spectral_norm_target
        self.spectral_norm_ema_decay = spectral_norm_ema_decay
        self.l1_sparsity_coef = l1_sparsity_coef
        self.num_cpus = num_cpus
        self.use_attention = use_attention
        self.o_proj_depth = o_proj_depth

        if not use_attention or not use_energy:
            self.energy = None
        elif energy_kwargs is not None:
            self.energy = EnergyDynamics(**energy_kwargs)
        else:
            self.energy = _toy_scale_energy()

        state_width = self.state_width

        # Per-layer independent seeds derived from `rng`. See
        # docs/research/toy_tile_precision_models.rst:rng_per_layer_seeding_bug.
        if rng is None:
            rng = np.random.default_rng()
        n_layer_seeds = 5 + max(o_proj_depth, 1)  # input_proj, q, k, v, lm_head + o_proj sublayer(s)
        if gated_combine:
            n_layer_seeds += 2  # gate_x_proj, gate_m_proj
        if gated_update:
            n_layer_seeds += 2  # update_forget_proj, update_input_proj
        layer_seeds = iter(int(s) for s in rng.integers(0, 2**31 - 1, size=n_layer_seeds))

        # dense=True forwarded only when set -- not every disldo_cls option
        # accepts that kwarg, so unconditionally passing it would TypeError.
        dense_kwargs = {"dense": True} if dense else {}

        # 0. Input projection: a real trained layer. See
        # docs/research/toy_tile_precision_models.rst:input_proj_column_averaging_misapplication.
        self.input_proj = disldo_cls(
            embed_width,
            state_width,
            max_weights,
            num_cpus,
            rng=np.random.default_rng(next(layer_seeds)),
            **dense_kwargs,
        )

        # 0.5. Gated combine/update (opt-in). See
        # docs/research/toy_tile_precision_models.rst:gated_combine and :gated_update.
        if gated_combine:
            self.gate_x_proj = disldo_cls(
                state_width,
                state_width,
                max_weights,
                num_cpus,
                rng=np.random.default_rng(next(layer_seeds)),
                **dense_kwargs,
            )
            self.gate_m_proj = disldo_cls(
                state_width,
                state_width,
                max_weights,
                num_cpus,
                rng=np.random.default_rng(next(layer_seeds)),
                **dense_kwargs,
            )
        if gated_update:
            self.update_forget_proj = disldo_cls(
                state_width,
                state_width,
                max_weights,
                num_cpus,
                rng=np.random.default_rng(next(layer_seeds)),
                **dense_kwargs,
            )
            self.update_input_proj = disldo_cls(
                state_width,
                state_width,
                max_weights,
                num_cpus,
                rng=np.random.default_rng(next(layer_seeds)),
                **dense_kwargs,
            )

        # 1. Core Attention & Output Projections
        if use_attention:
            self.q_proj = disldo_cls(
                state_width,
                state_width,
                max_weights,
                num_cpus,
                rng=np.random.default_rng(next(layer_seeds)),
                **dense_kwargs,
            )
            self.k_proj = disldo_cls(
                state_width,
                state_width,
                max_weights,
                num_cpus,
                rng=np.random.default_rng(next(layer_seeds)),
                **dense_kwargs,
            )
            self.v_proj = disldo_cls(
                state_width,
                state_width,
                max_weights,
                num_cpus,
                rng=np.random.default_rng(next(layer_seeds)),
                **dense_kwargs,
            )
        if o_proj_depth > 1:
            per_layer_weights = max(max_weights // o_proj_depth, state_width)
            self.o_proj = [
                disldo_cls(
                    state_width,
                    state_width,
                    per_layer_weights,
                    num_cpus,
                    rng=np.random.default_rng(next(layer_seeds)),
                    **dense_kwargs,
                )
                for _ in range(o_proj_depth)
            ]
        else:
            self.o_proj = disldo_cls(
                state_width,
                state_width,
                max_weights,
                num_cpus,
                rng=np.random.default_rng(next(layer_seeds)),
                **dense_kwargs,
            )
        self.lm_head = disldo_cls(
            embed_width, vocab_size, max_weights, num_cpus, rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs
        )

        # 2. Norms & Gaussian Attention Params
        self.input_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.state_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.centers = Tensor(np.array([i + 0.5 for i in range(num_tiles)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(num_tiles, dtype=np.float32))

        # 3. Spectral-norm probe state (only used if spectral_norm_target is
        # set). See docs/research/toy_tile_precision_models.rst:spectral_norm_target.
        if spectral_norm_target is not None:
            n_o_layers = o_proj_depth if o_proj_depth > 1 else 1
            self._spectral_u = [rng.standard_normal(state_width).astype(np.float32) for _ in range(n_o_layers)]
            self._spectral_u = [u / (np.linalg.norm(u) + 1e-8) for u in self._spectral_u]
            self._spectral_sigma_ema = [None] * n_o_layers
            # Warm-start iterations -- see
            # docs/research/toy_tile_precision_models.rst:spectral_norm_target.
            o_layers = self.o_proj if o_proj_depth > 1 else [self.o_proj]
            for _ in range(20):
                for idx, layer in enumerate(o_layers):
                    self._spectral_rescale_factor(layer, idx)

    def parameters_for_optimizer(self) -> list[Tensor]:
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

        for name, p in zip(param_names, params, strict=False):
            data_active = np.any(p.data != 0)
            data_mean = np.mean(p.data)

            if p.grad is not None:
                grad_active = np.any(p.grad != 0)
                grad_mean = np.mean(np.abs(p.grad))
                has_nans = np.any(np.isnan(p.grad))
                print(
                    f"{name:12s} | Data Active: {data_active} (mean: {data_mean:.4f}) | "
                    f"Grad Active: {grad_active} (mean |grad|: {grad_mean:.6e}) | NaN Grad: {has_nans}"
                )
            else:
                print(
                    f"{name:12s} | Data Active: {data_active} (mean: {data_mean:.4f}) | Grad: NONE (Not computed yet)"
                )
        print("==================================\n")

    def _spectral_rescale_factor(self, layer, idx: int) -> float:
        """One power-iteration step against `layer` alone. See
        docs/research/toy_tile_precision_models.rst:spectral_rescale_factor_radius_vs_norm_correction."""
        eps = 1e-8
        u = self._spectral_u[idx]
        probe = Tensor(u.reshape(1, -1).astype(np.float32))
        raw = np.asarray(layer.forward(probe, 0.0).data).reshape(-1)
        sigma = float(np.linalg.norm(raw))
        self._spectral_u[idx] = raw / (sigma + eps)
        prev = self._spectral_sigma_ema[idx]
        ema = (
            sigma
            if prev is None
            else (self.spectral_norm_ema_decay * prev + (1.0 - self.spectral_norm_ema_decay) * sigma)
        )
        self._spectral_sigma_ema[idx] = ema
        return self.spectral_norm_target / max(ema, eps)

    def _l1_sparsity_split(self, layer, input_t: Tensor, lr: float, coef: float) -> Tensor:
        """Exact port of l1_sparsity_probe.py's `_l1_sparsity_split`. See
        docs/research/toy_tile_precision_models.rst:l1_sparsity_coef_landmark."""
        out_aux = layer.forward(input_t, lr, damp_by_importance=False)
        n = float(np.asarray(out_aux.data).size)
        return reduce_sum(tensor_abs(out_aux)) * (coef / n)

    def _apply_o_proj(self, x: Tensor, learning_rate: float) -> Tensor:
        if self.o_proj_depth > 1:
            for idx, layer in enumerate(self.o_proj):
                x = layer.forward(x, learning_rate)
                if self.spectral_norm_target is not None:
                    x = x * self._spectral_rescale_factor(layer, idx)
            return x
        x = self.o_proj.forward(x, learning_rate)
        if self.spectral_norm_target is not None:
            x = x * self._spectral_rescale_factor(self.o_proj, 0)
        return x

    def step(
        self, x_window: np.ndarray, M_prev: np.ndarray, learning_rate: float, debug: bool = False
    ) -> tuple[np.ndarray, Tensor, Tensor | None]:
        """One recurrence tick. x_window: [num_tiles, embed_width] numpy
        (mapped into the wide state by input_proj -- see
        docs/research/toy_tile_precision_models.rst:input_proj_column_averaging_misapplication).
        M_prev: [num_tiles, state_width] numpy, DETACHED (no BPTT). Returns
        (M_new numpy [num_tiles, state_width], logits Tensor [num_tiles,
        vocab_size], aux_loss).

        debug=True: see docs/research/toy_tile_precision_models.rst:step_debug_stats."""

        def _stats(name, arr):
            a = np.asarray(arr)
            return {
                "mean": float(a.mean()),
                "std": float(a.std()),
                "min": float(a.min()),
                "max": float(a.max()),
                "abs_max": float(np.abs(a).max()),
            }

        dbg = {} if debug else None

        # 0. Project the narrow per-tile input into the wide state.
        x_window_t = Tensor(x_window.astype(np.float32))
        x_wide = self.input_proj.forward(x_window_t, learning_rate)
        if debug:
            dbg["x_wide"] = _stats("x_wide", x_wide.data)

        # 1. Combine Input and State
        x_normed = rmsnorm_tensor(x_wide, self.input_ln, self.rms_eps)
        m_normed = rmsnorm_tensor(Tensor(M_prev.astype(np.float32)), self.input_ln, self.rms_eps)
        if self.gated_combine:
            gate_logit = self.gate_x_proj.forward(x_normed, learning_rate) + self.gate_m_proj.forward(
                m_normed, learning_rate
            )
            gate_raw = sigmoid_tensor(gate_logit)
            gate = self.gate_floor + (1.0 - 2.0 * self.gate_floor) * gate_raw
            qkv_source = gate * x_normed + (1.0 - gate) * m_normed
            if debug:
                dbg["gate"] = _stats("gate", gate.data)
        else:
            qkv_source = x_normed + m_normed
        if debug:
            dbg["qkv_source"] = _stats("qkv_source", qkv_source.data)

        # 2. Gaussian Attention (or bypass). See
        # docs/research/toy_tile_precision_models.rst:use_attention_bypass_ablation.
        if self.use_attention:
            q = self.q_proj.forward(qkv_source, learning_rate)
            k = self.k_proj.forward(qkv_source, learning_rate)
            v = self.v_proj.forward(qkv_source, learning_rate)
            sigmas = exp(self.log_sigmas)
            if debug:
                dbg["q"] = _stats("q", q.data)
                dbg["k"] = _stats("k", k.data)
                dbg["v"] = _stats("v", v.data)

            attn = gaussian_attention(q, k, v, self.centers, sigmas, num_cpus=self.num_cpus, causal=False)
            if debug:
                dbg["attn_raw"] = _stats("attn_raw", attn.data)
            attn, aux_loss = _apply_energy(self.energy, attn, self.num_tiles, self.state_width)
            o_proj_input = attn
            attn = self._apply_o_proj(attn, learning_rate)
        else:
            o_proj_input = qkv_source
            attn = self._apply_o_proj(qkv_source, learning_rate)
            aux_loss = None

        if self.l1_sparsity_coef > 0.0:
            l1_terms = [self._l1_sparsity_split(self.input_proj, x_window_t, learning_rate, self.l1_sparsity_coef)]
            if self.gated_combine:
                l1_terms.append(
                    self._l1_sparsity_split(self.gate_x_proj, x_normed, learning_rate, self.l1_sparsity_coef)
                )
                l1_terms.append(
                    self._l1_sparsity_split(self.gate_m_proj, m_normed, learning_rate, self.l1_sparsity_coef)
                )
            if self.use_attention:
                l1_terms.append(self._l1_sparsity_split(self.q_proj, qkv_source, learning_rate, self.l1_sparsity_coef))
                l1_terms.append(self._l1_sparsity_split(self.k_proj, qkv_source, learning_rate, self.l1_sparsity_coef))
                l1_terms.append(self._l1_sparsity_split(self.v_proj, qkv_source, learning_rate, self.l1_sparsity_coef))
            if self.o_proj_depth > 1:
                cur = o_proj_input
                for layer in self.o_proj:
                    l1_terms.append(self._l1_sparsity_split(layer, cur, learning_rate, self.l1_sparsity_coef))
                    cur = layer.forward(cur, 0.0)
            else:
                l1_terms.append(
                    self._l1_sparsity_split(self.o_proj, o_proj_input, learning_rate, self.l1_sparsity_coef)
                )
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term
        # Forward clip on the residual update itself, not just the final
        # state -- see docs/research/toy_tile_precision_models.rst:forward_clip_necessity.
        attn.data = np.clip(attn.data, -self.clip_range, self.clip_range)
        if debug:
            dbg["attn_o_proj"] = _stats("attn_o_proj", attn.data)
        if self.magnitude_penalty_coef > 0:
            mag_penalty = reduce_sum(power(attn, 2)) * (self.magnitude_penalty_coef / attn.data.size)
            aux_loss = mag_penalty if aux_loss is None else aux_loss + mag_penalty

        # 3. Residual & Hard Bounding
        if self.gated_update:
            # Learned forget gate on the state update itself. See
            # docs/research/toy_tile_precision_models.rst:gated_update.
            M_prev_t = Tensor(M_prev.astype(np.float32))
            forget_logit = self.update_forget_proj.forward(M_prev_t, learning_rate) + self.update_input_proj.forward(
                attn, learning_rate
            )
            forget_raw = sigmoid_tensor(forget_logit)
            forget_gate = self.gate_floor + (1.0 - 2.0 * self.gate_floor) * forget_raw
            if self.l1_sparsity_coef > 0.0:
                update_l1 = self._l1_sparsity_split(
                    self.update_forget_proj, M_prev_t, learning_rate, self.l1_sparsity_coef
                ) + self._l1_sparsity_split(self.update_input_proj, attn, learning_rate, self.l1_sparsity_coef)
                aux_loss = update_l1 if aux_loss is None else aux_loss + update_l1
            M_new_t = forget_gate * M_prev_t + (1.0 - forget_gate) * attn
            if debug:
                dbg["forget_gate"] = _stats("forget_gate", forget_gate.data)
        else:
            M_new_t = Tensor(M_prev.astype(np.float32)) + attn
        M_new_t = rmsnorm_tensor(M_new_t, self.state_ln, self.rms_eps)
        if debug:
            dbg["pre_clip"] = _stats("pre_clip", M_new_t.data)
            dbg["clip_fraction"] = float(np.mean(np.abs(M_new_t.data) >= self.clip_range))

        # Hard clip bounds the state; direct .data write bypasses autograd.
        M_new_t.data = np.clip(M_new_t.data, -self.clip_range, self.clip_range)
        if debug:
            dbg["post_clip"] = _stats("post_clip", M_new_t.data)
        if self.magnitude_penalty_coef > 0:
            mag_penalty = reduce_sum(power(M_new_t, 2)) * (self.magnitude_penalty_coef / M_new_t.data.size)
            aux_loss = mag_penalty if aux_loss is None else aux_loss + mag_penalty

        # 4. Generate Logits
        pooled = M_new_t.reshape((self.num_tiles, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)  # [num_tiles, embed_width]
        if self.cosine_lm_head:
            # Must run before any other lm_head.forward() this step (call-
            # ordering bug) -- see docs/research/toy_tile_precision_models.rst:cosine_lm_head.
            probes = Tensor(np.eye(self.embed_width, dtype=np.float32))
            probe_out = self.lm_head.forward(probes, 0.0).data  # [embed_width, vocab_size]
            row_norms = np.sqrt((probe_out**2).sum(axis=0)) + self.rms_eps  # [vocab_size]
        if self.l1_sparsity_coef > 0.0:
            # lm_head's own L1 escape route -- see
            # docs/research/toy_tile_precision_models.rst:l1_sparsity_coef_landmark.
            lm_l1 = self._l1_sparsity_split(self.lm_head, pooled, learning_rate, self.l1_sparsity_coef)
            aux_loss = lm_l1 if aux_loss is None else aux_loss + lm_l1
        logits = self.lm_head.forward(pooled, learning_rate)  # [num_tiles, vocab_size]
        if self.cosine_lm_head:
            logits = logits / Tensor(row_norms.astype(np.float32))
            if debug:
                dbg["lm_head_row_norms"] = _stats("lm_head_row_norms", row_norms)
        if debug:
            dbg["logits"] = _stats("logits", logits.data)
            self._last_step_debug_stats = dbg

        return M_new_t.data, logits, aux_loss
