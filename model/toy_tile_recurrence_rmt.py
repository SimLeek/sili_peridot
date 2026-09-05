from __future__ import annotations

import math
import time

import numpy as np
from sili.sparse_rnn import CSR, DISLDOLayer, _nucleus_top_k_csr
from sili.tensor import Tensor, concat, exp, gather, gaussian_attention, power, reduce_sum, relu, tensor_abs

from .toy_recall_models import rmsnorm_tensor


class ToyTileRecurrenceRMT:
    """Faithful RMT reference control for the MQAR investigation.
    See docs/research/toy_tile_recurrence_rmt.rst:module_overview."""

    # Per-layer dict keys shared by dy_r_target/x_r_target/etc.
    # See docs/research/toy_tile_recurrence_rmt.rst:dy_r_target_nucleus_design.
    _WIDE_LAYER_NAMES = ("input_proj", "q_proj", "k_proj", "v_proj", "o_proj")

    def __init__(
        self,
        vocab_size: int,
        embed_width: int,
        column_neurons: int,
        num_tiles: int,
        num_memory_slots: int,
        max_weights: int,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
        disldo_cls=DISLDOLayer,
        dense: bool = False,
        clip_range: float = 6.0,
        l1_sparsity_coef: float = 0.0,
        magnitude_clip_penalty_coef: float = 0.0,
        min_sigma: float = 1e-3,
        synapse_kwargs: dict | None = None,
        scale_rank: int = 1,
        additive_rank: int = 0,
        dynamic_rank_control: bool = False,
        use_critic: bool = False,
        recurrent_only_output: bool = False,
        input_sparsity_p: float | None = None,
        dy_sparsity_p: float | None = None,
        wide_max_weights: int | None = None,
        output_dy_sparsity_p: float | None = None,
        dy_r_target: float | None = None,
        dy_k_min: int = 0,
        dy_k_max: int | None = None,
        dy_surprise_alpha: float | None = None,
        dy_surprise_beta: float = 0.99,
        x_r_target: float | None = None,
        x_k_min: int = 0,
        x_k_max: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        """See docs/research/toy_tile_recurrence_rmt.rst for full param rationale.

        num_memory_slots: small handful of memory tokens (matches RMT's
        own paper), not tuned against this task yet. Everything else
        mirrors ToyTileRecurrenceRealFP4's own conventions.

        use_critic: See docs/research/toy_tile_recurrence_rmt.rst:critic_head_design.
        magnitude_clip_penalty_coef, min_sigma:
            See docs/research/toy_tile_recurrence_rmt.rst:magnitude_clip_and_min_sigma_design.
        recurrent_only_output:
            See docs/research/toy_tile_recurrence_rmt.rst:recurrent_only_output_ablation.
        input_sparsity_p, dy_sparsity_p, wide_max_weights, output_dy_sparsity_p:
            See docs/research/toy_tile_recurrence_rmt.rst:sparsity_phase6_design.
        dy_r_target, dy_k_min, dy_k_max:
            See docs/research/toy_tile_recurrence_rmt.rst:dy_r_target_nucleus_design.
        dy_surprise_alpha, dy_surprise_beta:
            See docs/research/toy_tile_recurrence_rmt.rst:dy_surprise_design.
        x_r_target, x_k_min, x_k_max:
            See docs/research/toy_tile_recurrence_rmt.rst:x_r_target_design."""
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.num_memory_slots = num_memory_slots
        self.total_slots = num_tiles + num_memory_slots
        self.rms_eps = rms_eps
        self.clip_range = clip_range
        self.l1_sparsity_coef = l1_sparsity_coef
        self.magnitude_clip_penalty_coef = magnitude_clip_penalty_coef
        self.min_sigma = min_sigma
        self.recurrent_only_output = recurrent_only_output
        self.num_cpus = num_cpus
        # Passthrough, see sili.sparse_rnn.DISLDOLayer.forward docstring.
        self.synapse_kwargs = synapse_kwargs or {}

        # See docs/research/toy_tile_recurrence_rmt.rst:sparsity_phase6_design.
        self.input_sparsity_p = input_sparsity_p
        self.dy_sparsity_p = dy_sparsity_p if dy_sparsity_p is not None else input_sparsity_p
        # See docs/research/toy_tile_recurrence_rmt.rst:x_r_target_design.
        self.x_r_target: dict = dict.fromkeys(self._WIDE_LAYER_NAMES, x_r_target)
        self.x_k_min = x_k_min
        self.x_k_max = x_k_max
        # See docs/research/toy_tile_recurrence_rmt.rst:input_selection_stats_design.
        self.last_input_selection: dict = {}
        # See docs/research/toy_tile_recurrence_rmt.rst:dy_r_target_nucleus_design.
        self.dy_r_target: dict = dict.fromkeys(self._WIDE_LAYER_NAMES, dy_r_target)
        self.dy_k_min = dy_k_min
        self.dy_k_max = dy_k_max
        # See docs/research/toy_tile_recurrence_rmt.rst:dy_surprise_design.
        self.dy_surprise_alpha = dy_surprise_alpha
        self.dy_surprise_beta = dy_surprise_beta
        self._layer_surprise: dict = {}
        # See docs/research/toy_tile_recurrence_rmt.rst:sparsity_phase6_design.
        self.output_dy_sparsity_p = output_dy_sparsity_p
        self._output_extra_kwargs = {"dy_sparsity_p": output_dy_sparsity_p} if output_dy_sparsity_p is not None else {}

        state_width = self.state_width
        if rng is None:
            rng = np.random.default_rng()
        n_layer_seeds = 6  # input_proj, q, k, v, o_proj, lm_head
        layer_seeds = iter(int(s) for s in rng.integers(0, 2**31 - 1, size=n_layer_seeds))
        dense_kwargs = {"dense": True} if dense else {}
        # Conditionally forwarded (only some disldo_cls backends accept these
        # kwargs; splatting unconditionally would TypeError e.g. DISLDOLayer32).
        # additive_rank: see AQRS_DESIGN.md Theorem 3/4 (task #280).
        rank_kwargs = {"scale_rank": scale_rank} if scale_rank != 1 else {}
        additive_kwargs = {"additive_rank": additive_rank} if additive_rank != 0 else {}
        dynamic_kwargs = {"dynamic_rank_control": True} if dynamic_rank_control else {}
        layer_kwargs = {**dense_kwargs, **rank_kwargs, **additive_kwargs, **dynamic_kwargs}
        # See docs/research/toy_tile_recurrence_rmt.rst:sparsity_phase6_design.
        wide_max_weights_ = wide_max_weights if wide_max_weights is not None else max_weights

        self.input_proj = disldo_cls(
            embed_width,
            state_width,
            wide_max_weights_,
            num_cpus,
            rng=np.random.default_rng(next(layer_seeds)),
            **layer_kwargs,
        )
        self.q_proj = disldo_cls(
            state_width,
            state_width,
            wide_max_weights_,
            num_cpus,
            rng=np.random.default_rng(next(layer_seeds)),
            **layer_kwargs,
        )
        self.k_proj = disldo_cls(
            state_width,
            state_width,
            wide_max_weights_,
            num_cpus,
            rng=np.random.default_rng(next(layer_seeds)),
            **layer_kwargs,
        )
        self.v_proj = disldo_cls(
            state_width,
            state_width,
            wide_max_weights_,
            num_cpus,
            rng=np.random.default_rng(next(layer_seeds)),
            **layer_kwargs,
        )
        self.o_proj = disldo_cls(
            state_width,
            state_width,
            wide_max_weights_,
            num_cpus,
            rng=np.random.default_rng(next(layer_seeds)),
            **layer_kwargs,
        )
        self.lm_head = disldo_cls(
            embed_width, vocab_size, max_weights, num_cpus, rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs
        )

        # See docs/research/toy_tile_recurrence_rmt.rst:critic_head_design.
        self.use_critic = use_critic
        self.critic_head = None
        # See docs/research/toy_tile_recurrence_rmt.rst:amortized_l2_decay_design.
        self._l2_decay_factor: dict = {}
        # See docs/research/toy_tile_recurrence_rmt.rst:layer_timing_design.
        self._layer_timing: dict = {}
        if use_critic:
            critic_seed = int(rng.integers(0, 2**31 - 1))
            self.critic_head = disldo_cls(
                embed_width, vocab_size, max_weights, num_cpus, rng=np.random.default_rng(critic_seed), **layer_kwargs
            )

        # Separate RMSNorm gains for memory vs content tokens (never
        # summed together, unlike ToyTileRecurrenceRealFP4's additive
        # combine) -- matches RMT's own memory tokens getting their own
        # learned embeddings.
        self.input_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.memory_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.state_ln = Tensor(np.ones(state_width, dtype=np.float32))

        # Interleaved position layout + widened cold-start sigma: real bug
        # fix. See docs/research/toy_tile_recurrence_rmt.rst:interleaved_position_layout_bug.
        n_mem, n_content = self.num_memory_slots, self.num_tiles
        if n_mem > 0:
            raw_positions = [round((m + 0.5) * self.total_slots / n_mem) for m in range(n_mem)]
            used = set()
            mem_phys = []
            for p in raw_positions:
                pos = p
                while pos in used:
                    pos = (pos + 1) % self.total_slots
                used.add(pos)
                mem_phys.append(pos)
            mem_phys = sorted(mem_phys)
        else:
            mem_phys = []
        mem_phys_set = set(mem_phys)
        content_phys = [p for p in range(self.total_slots) if p not in mem_phys_set]
        self._mem_phys = mem_phys
        self._content_phys = content_phys

        # LOGICAL->PHYSICAL gather indices, precomputed once for reuse.
        # See docs/research/toy_tile_recurrence_rmt.rst:interleaved_position_layout_bug.
        phys_to_log = [0] * self.total_slots
        for i, p in enumerate(mem_phys):
            phys_to_log[p] = i
        for t, p in enumerate(content_phys):
            phys_to_log[p] = n_mem + t
        self._kv_phys_gather_idx = [
            phys_to_log[p] * state_width + c for p in range(self.total_slots) for c in range(state_width)
        ]

        # Value-mask for recurrent_only_output ablation.
        # See docs/research/toy_tile_recurrence_rmt.rst:recurrent_only_output_ablation.
        mem_only_mask = np.zeros((self.total_slots, state_width), dtype=np.float32)
        for p in mem_phys:
            mem_only_mask[p, :] = 1.0
        self._mem_only_value_mask = Tensor(mem_only_mask)

        self.centers = Tensor(
            np.array(
                [mem_phys[i] + 0.5 for i in range(n_mem)] + [content_phys[t] + 0.5 for t in range(n_content)],
                dtype=np.float32,
            )
        )
        sigma_init = max(self.total_slots / 4.0, 1.0)
        self.log_sigmas = Tensor(np.full(self.total_slots, np.log(sigma_init), dtype=np.float32))

        # step_cached's own precomputed constant index lists (flat,
        # matching gather()'s [row*sw+col] convention), built once.
        self._mem_idx_step = [m * state_width + c for m in range(n_mem) for c in range(state_width)]
        self._new_content_idx_step = [n_mem * state_width + c for c in range(state_width)]
        self._mem_center_idx = list(range(n_mem))
        self._newest_content_idx = [n_mem + n_content - 1]

    def _named_real_layers(self):
        """(name, layer) for every real disldo_cls weight layer, including
        critic_head when use_critic is set -- single source of truth for
        every per-layer maintenance pass below."""
        layers = [
            ("input_proj", self.input_proj),
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("o_proj", self.o_proj),
            ("lm_head", self.lm_head),
        ]
        if self.use_critic:
            layers.append(("critic_head", self.critic_head))
        return layers

    def _real_layers(self):
        return [layer for _name, layer in self._named_real_layers()]

    def _timed_layer_forward(self, layer, layer_name: str, *args, **kwargs) -> Tensor:
        """Real per-layer forward+backward timing, and the task #374
        surprise-signal capture point. See
        docs/research/toy_tile_recurrence_rmt.rst:layer_timing_design."""
        t0 = time.perf_counter()
        out = layer.forward(*args, **kwargs)
        rec = self._layer_timing.setdefault(layer_name, {"fwd_s": 0.0, "bwd_s": 0.0, "fwd_calls": 0, "bwd_calls": 0})
        rec["fwd_s"] += time.perf_counter() - t0
        rec["fwd_calls"] += 1
        orig_backward = out._backward
        if orig_backward is not None:

            def _timed_backward(_orig=orig_backward, _rec=rec, _out=out, _name=layer_name):
                tb0 = time.perf_counter()
                _orig()
                _rec["bwd_s"] += time.perf_counter() - tb0
                _rec["bwd_calls"] += 1
                if self.dy_surprise_alpha is not None and _out.grad is not None:
                    self._update_layer_surprise(_name, _out.grad)

            out._backward = _timed_backward
        return out

    def reset_layer_timing(self) -> None:
        """Zero self._layer_timing -- call at the start of a measurement
        window. See docs/research/toy_tile_recurrence_rmt.rst:layer_timing_design."""
        self._layer_timing = {}

    def _update_layer_surprise(self, layer_name: str, dy) -> None:
        """E_t/Lbar per-layer surprise EMA update.
        See docs/research/toy_tile_recurrence_rmt.rst:dy_surprise_design."""
        E_t = float(np.sum(np.asarray(dy, dtype=np.float64) ** 2))
        rec = self._layer_surprise.get(layer_name)
        if rec is None:
            self._layer_surprise[layer_name] = {"E_t": E_t, "Lbar": max(E_t, 1e-12)}
            return
        rec["E_t"] = E_t
        rec["Lbar"] = self.dy_surprise_beta * rec["Lbar"] + (1.0 - self.dy_surprise_beta) * E_t

    def _effective_dy_r_target(self, layer_name: str) -> float | None:
        """r_t = clip(r_bar * (E_t/Lbar)^alpha, 0.05, 0.99), lagged one step.
        See docs/research/toy_tile_recurrence_rmt.rst:dy_surprise_design."""
        r_bar = self.dy_r_target.get(layer_name)
        if r_bar is None or self.dy_surprise_alpha is None:
            return r_bar
        surprise = self._layer_surprise.get(layer_name)
        if surprise is None or surprise["Lbar"] <= 0.0:
            return r_bar
        ratio = surprise["E_t"] / surprise["Lbar"]
        r_t = r_bar * (ratio**self.dy_surprise_alpha)
        return min(max(r_t, 0.05), 0.99)

    def _wide_extra_kwargs(self, layer_name: str) -> dict:
        """Extra kwargs for ONE of the 5 affected layers' forward() calls;
        layer_name must be one of _WIDE_LAYER_NAMES. Live (not cached) since
        self.dy_r_target[name] is mutable post-construction. See
        docs/research/toy_tile_recurrence_rmt.rst:dy_r_target_nucleus_design."""
        r_target = self._effective_dy_r_target(layer_name)
        if r_target is not None:
            kw = {"dy_r_target": r_target}
            if self.dy_k_min:
                kw["dy_k_min"] = self.dy_k_min
            if self.dy_k_max is not None:
                kw["dy_k_max"] = self.dy_k_max
            return kw
        if self.dy_sparsity_p is not None:
            return {"dy_sparsity_p": self.dy_sparsity_p}
        return {}

    def apply_amortized_dy_r_target_control(
        self,
        measured_sps: float,
        target_sps: float,
        layer_name: str | None = None,
        down_factor: float = 0.85,
        up_factor: float = 1.05,
        r_min: float = 0.05,
        r_max: float = 0.99,
    ) -> dict:
        """Closed-loop controller adjusting self.dy_r_target against
        MEASURED steps/sec. See
        docs/research/toy_tile_recurrence_rmt.rst:amortized_r_target_control_design."""
        names = [layer_name] if layer_name is not None else list(self._WIDE_LAYER_NAMES)
        updated = {}
        for name in names:
            current = self.dy_r_target.get(name)
            if current is None:
                continue
            if measured_sps < target_sps:
                current = max(r_min, current * down_factor)
            elif measured_sps > target_sps:
                current = min(r_max, current * up_factor)
            self.dy_r_target[name] = current
            updated[name] = current
        return updated

    def apply_amortized_x_r_target_control(
        self,
        measured_sps: float,
        target_sps: float,
        layer_name: str | None = None,
        down_factor: float = 0.85,
        up_factor: float = 1.05,
        r_min: float = 0.05,
        r_max: float = 0.99,
    ) -> dict:
        """Same as apply_amortized_dy_r_target_control, operating on
        x_r_target (INPUT axis) instead. See
        docs/research/toy_tile_recurrence_rmt.rst:amortized_r_target_control_design."""
        names = [layer_name] if layer_name is not None else list(self._WIDE_LAYER_NAMES)
        updated = {}
        for name in names:
            current = self.x_r_target.get(name)
            if current is None:
                continue
            if measured_sps < target_sps:
                current = max(r_min, current * down_factor)
            elif measured_sps > target_sps:
                current = min(r_max, current * up_factor)
            self.x_r_target[name] = current
            updated[name] = current
        return updated

    def apply_cross_layer_budget_allocator(
        self,
        measured_sps: float,
        target_sps: float,
        down_factor: float = 0.85,
        up_factor: float = 1.05,
        r_min: float = 0.05,
        r_max: float = 0.99,
    ) -> dict:
        """Coordinates x_r_target (INPUT axis) against the remaining compute
        budget using per-layer timing; dy_r_target (GRAD axis) untouched.
        See docs/research/toy_tile_recurrence_rmt.rst:cross_layer_budget_allocator_design."""
        names = [n for n in self._WIDE_LAYER_NAMES if self.x_r_target.get(n) is not None]
        if not names:
            return {}
        total_t = sum(rec["fwd_s"] + rec["bwd_s"] for rec in self._layer_timing.values())
        uniform_share = 1.0 / len(self._WIDE_LAYER_NAMES)
        updated = {}
        for name in names:
            if total_t > 0:
                layer_t = self._layer_timing.get(name, {"fwd_s": 0.0, "bwd_s": 0.0})
                share = (layer_t["fwd_s"] + layer_t["bwd_s"]) / total_t
                weight = min(max(share / uniform_share, 0.0), 3.0)
            else:
                weight = 1.0
            current = self.x_r_target[name]
            if measured_sps < target_sps:
                eff_down = 1.0 - (1.0 - down_factor) * weight
                current = max(r_min, current * eff_down)
            elif measured_sps > target_sps:
                eff_up = 1.0 + (up_factor - 1.0) * weight
                current = min(r_max, current * eff_up)
            self.x_r_target[name] = current
            updated[name] = current
        return updated

    def parameters_for_optimizer(self) -> list[Tensor]:
        return [self.input_ln, self.memory_ln, self.state_ln, self.centers, self.log_sigmas]

    def magnitude_rescale_output(self, target: float, correction_rate: float, scale_invariant: bool = False) -> None:
        """Apply to every real weight layer, skipping backends with no
        scale concept (e.g. fp32 DISLDOLayerV). Call periodically, not
        every step -- see delta_csr_types.hpp's own docstring."""
        for layer in self._real_layers():
            if (hasattr(layer, "_c") and hasattr(layer._c, "magnitude_rescale_output")) or (
                hasattr(layer, "magnitude_rescale_output") and hasattr(layer, "digits")
            ):
                layer.magnitude_rescale_output(target, correction_rate, scale_invariant)

    def apply_amortized_l2_decay(self, chunk_size: int, adaptation_rate: float = 0.3) -> dict:
        """Amortized decoupled L2 decay + rolling health-stats, closed-loop
        against each layer's own measured rms (not a hand-picked half-life).
        See docs/research/toy_tile_recurrence_rmt.rst:amortized_l2_decay_design."""
        results = {}
        for name, layer in self._named_real_layers():
            nnz = layer.nnz
            if nnz <= 0:
                continue
            decay_factor = self._l2_decay_factor.get(name, 1.0)
            stats = layer.apply_amortized_l2_decay(chunk_size, decay_factor)
            if stats.get("cycle_complete") and stats.get("n", 0) > 0:
                target = 1.0 / math.sqrt(layer.in_features)
                measured_rms = stats.get("rms", 0.0)
                if measured_rms > 0.0:
                    correction = (target / measured_rms) ** adaptation_rate
                    correction = min(max(correction, 0.5), 2.0)
                    decay_factor = min(max(decay_factor * correction, 1e-6), 1.0)
                    self._l2_decay_factor[name] = decay_factor
                stats = dict(stats, target=target, decay_factor=decay_factor)
                results[name] = stats
        return results

    def apply_dynamic_rank_control(
        self,
        tau_death: float = 0.05,
        tau_active: float = 0.3,
        theta: float = 1e-4,
        seed_scale: float = 0.05,
        scale_grace_period_steps: int = 50,
        additive_grace_period_steps: int = 5000,
    ) -> dict:
        """Runs AQRS Theorem 10 dynamic rank control (task #292) on every
        real weight layer -- see sili__new's own
        DISLDOLayer.apply_dynamic_rank_control docstring for the full
        mechanism and the scale/additive grace-period biology citations
        (Holtmaat 2005, Grutzendler 2002, Turrigiano, Zenke & Gerstner 2017)
        behind the ~100x cross-branch cooldown asymmetry. Call once per
        training step, after backward. Returns {layer_name: mutated_bool}."""
        results = {}
        for name, layer in self._named_real_layers():
            if hasattr(layer, "apply_dynamic_rank_control"):
                results[name] = layer.apply_dynamic_rank_control(
                    tau_death, tau_active, theta, seed_scale, scale_grace_period_steps, additive_grace_period_steps
                )
        return results

    def apply_scale_overflow_guard(self, clip: float = 200.0, near: float = 20.0, coef: float = 0.1) -> None:
        """AQRS scale/additive channel numerical-safety pass (task #295
        follow-up): fixes raised scale/additive rank caps letting a real
        fp8 MQAR run's per-channel scales grow unbounded and NaN-collapse.
        Not a plain hard clip -- see sili__new's own
        DISLDOLayer.apply_scale_overflow_guard / _overflow_guard_array
        docstrings for the auto-correcting-shrink derivation. Call once
        per training step, any time after backward."""
        for layer in self._real_layers():
            if hasattr(layer, "apply_scale_overflow_guard"):
                layer.apply_scale_overflow_guard(clip, near, coef)

    def apply_channel_orthogonality_penalty(self, coef: float = 0.01) -> None:
        """AQRS channel-diversity pass, stopping rank channels from
        converging to duplicate directions -- chosen over residual-targeted
        growth since it's an ongoing per-step force, not init-time only.
        See sili__new's DISLDOLayer.apply_channel_orthogonality_penalty /
        _orthogonality_penalty_array docstrings. Call once per training
        step, independent of the other maintenance passes here."""
        for layer in self._real_layers():
            if hasattr(layer, "apply_channel_orthogonality_penalty"):
                layer.apply_channel_orthogonality_penalty(coef)

    def report_ranks(self) -> dict:
        """{layer_name: (scale_rank, additive_rank)} for every real layer
        with a C++ backend -- the answer to "what best rank numbers does
        dynamic control end up with" (task #292)."""
        results = {}
        for name, layer in self._named_real_layers():
            c = getattr(layer, "_c", None)
            if c is not None and hasattr(c, "get_scale_rank"):
                results[name] = (c.get_scale_rank(), c.get_additive_rank())
        return results

    def _to_sparse(self, x: Tensor, layer_name: str) -> Tensor:
        """Sparsity plan Phase 6 helper; no-op unless x_r_target[layer_name]
        or input_sparsity_p is set. Real bug fix: wires _children/_backward
        so gradient flows straight-through to x (CSR.as_tensor() alone
        silently detaches the graph). See
        docs/research/toy_tile_recurrence_rmt.rst:to_sparse_gradient_detach_bug
        and :x_r_target_design."""
        x_r_target = self.x_r_target.get(layer_name)
        if x_r_target is None and self.input_sparsity_p is None:
            return x
        x_np = np.asarray(x.data, dtype=np.float32)
        x2d = x_np[np.newaxis, :] if x_np.ndim == 1 else x_np
        if x_r_target is not None:
            ptrs, indices, values = _nucleus_top_k_csr(
                x2d, x_r_target, self.num_cpus, k_min=self.x_k_min, k_max=self.x_k_max
            )
            csr = CSR(ptrs, indices, values, rows=x2d.shape[0], cols=x2d.shape[1])
        else:
            csr = CSR.from_dense(x2d, self.input_sparsity_p, self.num_cpus)
        self._update_input_selection_stats(layer_name, x2d, csr)
        out = Tensor(csr, _children=(x,), _op="to_sparse", backend=x.backend)

        def _bwd():
            if out.grad is None:
                return
            g = np.asarray(out.grad, dtype=np.float32)
            if x.grad is None:
                x.grad = x.backend.zeros_like(x.data)
            x.grad = x.backend.add(x.grad, g)

        out._backward = _bwd
        return out

    def _update_input_selection_stats(self, layer_name: str, x2d: np.ndarray, csr: CSR) -> None:
        """Real per-layer INPUT-axis R/k trajectory stats, overwritten
        each call (not accumulated). See
        docs/research/toy_tile_recurrence_rmt.rst:input_selection_stats_design."""
        row_sq_total = np.sum(x2d.astype(np.float64) ** 2, axis=1)
        kept_sq = np.zeros(csr.rows, dtype=np.float64)
        k_per_row = np.zeros(csr.rows, dtype=np.int64)
        for row in range(csr.rows):
            start, end = int(csr.ptrs[row]), int(csr.ptrs[row + 1])
            kept_sq[row] = np.sum(csr.values[start:end].astype(np.float64) ** 2)
            k_per_row[row] = end - start
        with np.errstate(divide="ignore", invalid="ignore"):
            r_per_row = np.where(row_sq_total > 0, kept_sq / row_sq_total, 1.0)
        self.last_input_selection[layer_name] = {
            "R_mean": float(np.mean(r_per_row)),
            "k_mean": float(np.mean(k_per_row)),
            "rows": int(csr.rows),
            "cols": int(csr.cols),
        }

    def _l1_sparsity_split(
        self,
        layer,
        input_t: Tensor,
        lr: float,
        coef: float,
        requires_grad: bool = True,
        layer_name: str | None = None,
        **extra_kwargs,
    ) -> Tensor:
        """Exact port of ToyTileRecurrenceRealFP4's own helper -- see its
        l1_sparsity_coef docstring for the full split-backward rationale.
        See docs/research/toy_tile_recurrence_rmt.rst:l1_sparsity_split_design."""
        if layer_name is not None:
            out_aux = self._timed_layer_forward(
                layer,
                layer_name,
                input_t,
                lr,
                requires_grad=requires_grad,
                damp_by_importance=False,
                **self.synapse_kwargs,
                **extra_kwargs,
            )
        else:
            out_aux = layer.forward(
                input_t,
                lr,
                requires_grad=requires_grad,
                damp_by_importance=False,
                **self.synapse_kwargs,
                **extra_kwargs,
            )
        n = float(np.asarray(out_aux.data).size)
        return reduce_sum(tensor_abs(out_aux)) * (coef / n)

    def _magnitude_clip_penalty(self, out_tensor: Tensor) -> Tensor:
        """Hinge-squared penalty (not plain L2). See
        docs/research/toy_tile_recurrence_rmt.rst:magnitude_clip_and_min_sigma_design."""
        excess = relu(tensor_abs(out_tensor) - self.clip_range)
        n = float(np.asarray(out_tensor.data).size)
        return reduce_sum(power(excess, 2)) * (self.magnitude_clip_penalty_coef / n)

    def step(
        self,
        x_window: np.ndarray,
        memory_prev: np.ndarray,
        learning_rate: float,
        requires_grad: bool = True,
        content_dy_sparsity_schedule: list[float] | None = None,
    ) -> tuple[np.ndarray, Tensor, Tensor | None]:
        """Write-then-read within one call, not BPTT across calls.
        See docs/research/toy_tile_recurrence_rmt.rst:step_design."""
        sw = self.state_width
        n_mem, n_content = self.num_memory_slots, self.num_tiles

        if content_dy_sparsity_schedule is not None:
            if len(content_dy_sparsity_schedule) != n_content:
                raise ValueError(
                    f"content_dy_sparsity_schedule has {len(content_dy_sparsity_schedule)} "
                    f"entries, expected num_tiles={n_content}"
                )
            mem_schedule = [1.0] * n_mem
            content_schedule = list(content_dy_sparsity_schedule)
            full_schedule = mem_schedule + content_schedule
            _schedule_by_group = {"mem": mem_schedule, "content": content_schedule, "full": full_schedule}

            def _kw(layer_name: str, group: str) -> dict:
                return {"dy_sparsity_schedule": _schedule_by_group[group]}
        else:
            # group ignored: dy_r_target/dy_sparsity_p apply uniformly
            # across a layer's own rows, unlike the schedule branch above.
            def _kw(layer_name: str, group: str) -> dict:
                return self._wide_extra_kwargs(layer_name)

        x_window_t = Tensor(x_window.astype(np.float32))
        x_window_sparse = self._to_sparse(x_window_t, "input_proj")
        x_wide = self._timed_layer_forward(
            self.input_proj,
            "input_proj",
            x_window_sparse,
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **_kw("input_proj", "content"),
        )  # [n_content, sw]
        x_normed = rmsnorm_tensor(x_wide, self.input_ln, self.rms_eps)

        memory_prev_t = Tensor(memory_prev.astype(np.float32))
        memory_normed = rmsnorm_tensor(memory_prev_t, self.memory_ln, self.rms_eps)

        combined_normed = concat([memory_normed, x_normed], axis=0)  # [total_slots, sw]

        # Each of q/k/v_proj gets its OWN _to_sparse call on combined_normed.
        # See docs/research/toy_tile_recurrence_rmt.rst:to_sparse_gradient_detach_bug.
        q = self._timed_layer_forward(
            self.q_proj,
            "q_proj",
            self._to_sparse(combined_normed, "q_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **_kw("q_proj", "full"),
        )
        k = self._timed_layer_forward(
            self.k_proj,
            "k_proj",
            self._to_sparse(combined_normed, "k_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **_kw("k_proj", "full"),
        )
        v = self._timed_layer_forward(
            self.v_proj,
            "v_proj",
            self._to_sparse(combined_normed, "v_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **_kw("v_proj", "full"),
        )

        aux_loss = None

        def _accumulate_penalty(t: Tensor) -> None:
            nonlocal aux_loss
            if self.magnitude_clip_penalty_coef > 0.0:
                term = self._magnitude_clip_penalty(t)
                aux_loss = term if aux_loss is None else aux_loss + term

        # Penalty MUST be built from the PRE-clip value (each intermediate
        # tensor snapshots .data at construction) -- building it after the
        # clip below would read the already-clipped value and never fire.
        _accumulate_penalty(q)
        _accumulate_penalty(k)
        _accumulate_penalty(v)

        # Clip q/k/v BEFORE gaussian_attention -- real bug fix, see
        # docs/research/toy_tile_recurrence_rmt.rst:write_then_read_pass_design.
        q.data = np.clip(q.data, -self.clip_range, self.clip_range)
        k.data = np.clip(k.data, -self.clip_range, self.clip_range)
        v.data = np.clip(v.data, -self.clip_range, self.clip_range)
        sigmas = exp(self.log_sigmas)
        # Floor BEFORE gaussian_attention uses it.
        # See docs/research/toy_tile_recurrence_rmt.rst:magnitude_clip_and_min_sigma_design.
        sigmas.data = np.maximum(sigmas.data, self.min_sigma)

        # Reorder k/v into PHYSICAL position order for attention KEYS;
        # q stays LOGICAL (self.centers already holds the physical value).
        # See docs/research/toy_tile_recurrence_rmt.rst:interleaved_position_layout_bug.
        k_phys = gather(k, self._kv_phys_gather_idx).reshape((self.total_slots, sw))
        v_phys = gather(v, self._kv_phys_gather_idx).reshape((self.total_slots, sw))

        mem_idx = [m * sw + c for m in range(n_mem) for c in range(sw)]
        content_idx = [(n_mem + t) * sw + c for t in range(n_content) for c in range(sw)]

        # --- PASS 1: WRITE.
        # See docs/research/toy_tile_recurrence_rmt.rst:write_then_read_pass_design.
        q_mem = gather(q, mem_idx).reshape((n_mem, sw))
        centers_mem = gather(self.centers, list(range(n_mem)))
        sigmas_mem = gather(sigmas, list(range(n_mem)))
        attn_pre_o_mem = gaussian_attention(
            q_mem, k_phys, v_phys, centers_mem, sigmas_mem, num_cpus=self.num_cpus, causal=False
        )
        attn_mem = self._timed_layer_forward(
            self.o_proj,
            "o_proj",
            self._to_sparse(attn_pre_o_mem, "o_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **_kw("o_proj", "mem"),
        )
        _accumulate_penalty(attn_mem)
        attn_mem.data = np.clip(attn_mem.data, -self.clip_range, self.clip_range)
        memory_new_t = rmsnorm_tensor(memory_prev_t + attn_mem, self.state_ln, self.rms_eps)
        memory_new_t.data = np.clip(memory_new_t.data, -self.clip_range, self.clip_range)

        # --- PASS 2: READ, against the FRESH memory_new_t, not stale
        # memory_prev. Real live undetached gradient path within this
        # one call, NOT BPTT. See
        # docs/research/toy_tile_recurrence_rmt.rst:write_then_read_pass_design.
        memory_new_normed = rmsnorm_tensor(memory_new_t, self.memory_ln, self.rms_eps)
        k_mem_fresh = self._timed_layer_forward(
            self.k_proj,
            "k_proj",
            self._to_sparse(memory_new_normed, "k_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **_kw("k_proj", "mem"),
        )
        v_mem_fresh = self._timed_layer_forward(
            self.v_proj,
            "v_proj",
            self._to_sparse(memory_new_normed, "v_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **_kw("v_proj", "mem"),
        )
        _accumulate_penalty(k_mem_fresh)
        _accumulate_penalty(v_mem_fresh)
        k_mem_fresh.data = np.clip(k_mem_fresh.data, -self.clip_range, self.clip_range)
        v_mem_fresh.data = np.clip(v_mem_fresh.data, -self.clip_range, self.clip_range)

        # Content's own k/v (from pass 1) stay unchanged; only memory is
        # refreshed. Rebuilt via the SAME interleaved-physical gather.
        # See docs/research/toy_tile_recurrence_rmt.rst:interleaved_position_layout_bug.
        k_content_only = gather(k, content_idx).reshape((n_content, sw))
        v_content_only = gather(v, content_idx).reshape((n_content, sw))
        k2 = concat([k_mem_fresh, k_content_only], axis=0)
        v2 = concat([v_mem_fresh, v_content_only], axis=0)
        k2_phys = gather(k2, self._kv_phys_gather_idx).reshape((self.total_slots, sw))
        v2_phys = gather(v2, self._kv_phys_gather_idx).reshape((self.total_slots, sw))

        q_content = gather(q, content_idx).reshape((n_content, sw))
        centers_content = gather(self.centers, list(range(n_mem, n_mem + n_content)))
        sigmas_content = gather(sigmas, list(range(n_mem, n_mem + n_content)))
        if self.recurrent_only_output:
            # See docs/research/toy_tile_recurrence_rmt.rst:recurrent_only_output_ablation.
            v2_phys_mem_only = v2_phys * self._mem_only_value_mask
            attn_pre_o_content = gaussian_attention(
                q_content,
                k2_phys,
                v2_phys_mem_only,
                centers_content,
                sigmas_content,
                num_cpus=self.num_cpus,
                causal=False,
            )
        else:
            attn_pre_o_content = gaussian_attention(
                q_content, k2_phys, v2_phys, centers_content, sigmas_content, num_cpus=self.num_cpus, causal=False
            )
        attn_content = self._timed_layer_forward(
            self.o_proj,
            "o_proj",
            self._to_sparse(attn_pre_o_content, "o_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **_kw("o_proj", "content"),
        )
        _accumulate_penalty(attn_content)
        attn_content.data = np.clip(attn_content.data, -self.clip_range, self.clip_range)

        # Debug instrumentation: cheap reference-only capture for bisecting
        # NaN/Inf. See docs/research/toy_tile_recurrence_rmt.rst:write_then_read_pass_design.
        self.last_debug = {
            "x_wide": x_wide.data,
            "q": q.data,
            "k": k.data,
            "v": v.data,
            "attn_pre_o_mem": attn_pre_o_mem.data,
            "attn_pre_o_content": attn_pre_o_content.data,
            "attn_mem": attn_mem.data,
            "attn_content": attn_content.data,
            "sigmas": sigmas.data,
            "log_sigmas": self.log_sigmas.data,
            # x_window_t: embedding-learning hook (Tensor, not just .data).
            # See docs/research/toy_tile_recurrence_rmt.rst:write_then_read_pass_design.
            "x_window_t": x_window_t,
        }

        if self.l1_sparsity_coef > 0.0:
            l1_terms = [
                self._l1_sparsity_split(
                    self.input_proj,
                    x_window_sparse,
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="input_proj",
                    **self._wide_extra_kwargs("input_proj"),
                ),
                self._l1_sparsity_split(
                    self.q_proj,
                    self._to_sparse(combined_normed, "q_proj"),
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="q_proj",
                    **self._wide_extra_kwargs("q_proj"),
                ),
                self._l1_sparsity_split(
                    self.k_proj,
                    self._to_sparse(combined_normed, "k_proj"),
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="k_proj",
                    **self._wide_extra_kwargs("k_proj"),
                ),
                self._l1_sparsity_split(
                    self.v_proj,
                    self._to_sparse(combined_normed, "v_proj"),
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="v_proj",
                    **self._wide_extra_kwargs("v_proj"),
                ),
                self._l1_sparsity_split(
                    self.o_proj,
                    self._to_sparse(
                        gaussian_attention(
                            q, k_phys, v_phys, self.centers, sigmas, num_cpus=self.num_cpus, causal=False
                        ),
                        "o_proj",
                    ),
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="o_proj",
                    **self._wide_extra_kwargs("o_proj"),
                ),
            ]
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term

        # Residual against RAW content value, kept live regardless of
        # recurrent_only_output. See
        # docs/research/toy_tile_recurrence_rmt.rst:recurrent_only_output_ablation.
        pre_norm_content = x_wide + attn_content
        content_out = rmsnorm_tensor(pre_norm_content, self.state_ln, self.rms_eps)
        _accumulate_penalty(content_out)  # pre-clip, same reasoning as above
        content_out.data = np.clip(content_out.data, -self.clip_range, self.clip_range)
        self.last_debug["pre_norm_content"] = pre_norm_content.data
        self.last_debug["content_out"] = content_out.data

        # Detach only here, at the step() boundary -- see
        # docs/research/toy_tile_recurrence_rmt.rst:step_design.
        memory_new = memory_new_t.data.copy()

        pooled = content_out.reshape((n_content, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)
        self.last_debug["pooled"] = pooled.data
        if self.l1_sparsity_coef > 0.0:
            lm_l1 = self._l1_sparsity_split(
                self.lm_head, pooled, learning_rate, self.l1_sparsity_coef, requires_grad=requires_grad
            )
            aux_loss = lm_l1 if aux_loss is None else aux_loss + lm_l1
        logits = self.lm_head.forward(
            pooled, learning_rate, requires_grad=requires_grad, **self.synapse_kwargs, **self._output_extra_kwargs
        )

        # Advantage-actor-critic value head, exposed via attribute not
        # return value. See docs/research/toy_tile_recurrence_rmt.rst:critic_head_design.
        self.last_critic_pred = (
            self.critic_head.forward(
                pooled, learning_rate, requires_grad=requires_grad, **self.synapse_kwargs, **self._output_extra_kwargs
            )
            if self.use_critic
            else None
        )

        return memory_new, logits, aux_loss

    def step_cached(
        self,
        new_token_embed: np.ndarray,
        memory_prev: np.ndarray,
        learning_rate: float,
        tile_cache: list[tuple[np.ndarray, np.ndarray]] | None,
        requires_grad: bool = True,
    ) -> tuple[np.ndarray, Tensor, Tensor | None, list[tuple[np.ndarray, np.ndarray]]]:
        """Incremental alternative to step(), and the one approximation
        it accepts. See docs/research/toy_tile_recurrence_rmt.rst:step_cached_design."""
        sw = self.state_width
        n_mem, n_content = self.num_memory_slots, self.num_tiles

        new_embed_2d = np.asarray(new_token_embed, dtype=np.float32).reshape((1, self.embed_width))
        new_embed_t = Tensor(new_embed_2d)
        new_embed_sparse = self._to_sparse(new_embed_t, "input_proj")
        x_wide_new = self._timed_layer_forward(
            self.input_proj,
            "input_proj",
            new_embed_sparse,
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._wide_extra_kwargs("input_proj"),
        )  # [1, sw]
        x_normed_new = rmsnorm_tensor(x_wide_new, self.input_ln, self.rms_eps)

        memory_prev_t = Tensor(memory_prev.astype(np.float32))
        memory_normed = rmsnorm_tensor(memory_prev_t, self.memory_ln, self.rms_eps)

        combined_normed_step = concat([memory_normed, x_normed_new], axis=0)  # [n_mem+1, sw]

        # Separate _to_sparse call per consumer, see step()'s own comment.
        q_step = self._timed_layer_forward(
            self.q_proj,
            "q_proj",
            self._to_sparse(combined_normed_step, "q_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._wide_extra_kwargs("q_proj"),
        )
        k_step = self._timed_layer_forward(
            self.k_proj,
            "k_proj",
            self._to_sparse(combined_normed_step, "k_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._wide_extra_kwargs("k_proj"),
        )
        v_step = self._timed_layer_forward(
            self.v_proj,
            "v_proj",
            self._to_sparse(combined_normed_step, "v_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._wide_extra_kwargs("v_proj"),
        )

        aux_loss = None

        def _accumulate_penalty(t: Tensor) -> None:
            nonlocal aux_loss
            if self.magnitude_clip_penalty_coef > 0.0:
                term = self._magnitude_clip_penalty(t)
                aux_loss = term if aux_loss is None else aux_loss + term

        _accumulate_penalty(q_step)
        _accumulate_penalty(k_step)
        _accumulate_penalty(v_step)

        q_step.data = np.clip(q_step.data, -self.clip_range, self.clip_range)
        k_step.data = np.clip(k_step.data, -self.clip_range, self.clip_range)
        v_step.data = np.clip(v_step.data, -self.clip_range, self.clip_range)
        sigmas = exp(self.log_sigmas)
        sigmas.data = np.maximum(sigmas.data, self.min_sigma)

        q_mem = gather(q_step, self._mem_idx_step).reshape((n_mem, sw))
        k_mem_from_prev = gather(k_step, self._mem_idx_step).reshape((n_mem, sw))
        v_mem_from_prev = gather(v_step, self._mem_idx_step).reshape((n_mem, sw))
        q_new = gather(q_step, self._new_content_idx_step).reshape((1, sw))
        k_new = gather(k_step, self._new_content_idx_step).reshape((1, sw))
        v_new = gather(v_step, self._new_content_idx_step).reshape((1, sw))

        # Reassemble FULL [n_content, sw] content k/v: cache (oldest
        # first, zero-padded) plus this step's fresh row at the end.
        cache = list(tile_cache) if tile_cache else []
        pad = max(0, (n_content - 1) - len(cache))
        zero_row = np.zeros(sw, dtype=np.float32)
        cache_k_rows = [zero_row] * pad + [row[0] for row in cache[-(n_content - 1) :]] if n_content > 1 else []
        cache_v_rows = [zero_row] * pad + [row[1] for row in cache[-(n_content - 1) :]] if n_content > 1 else []
        cache_k_arr = np.stack(cache_k_rows, axis=0) if cache_k_rows else np.zeros((0, sw), dtype=np.float32)
        cache_v_arr = np.stack(cache_v_rows, axis=0) if cache_v_rows else np.zeros((0, sw), dtype=np.float32)
        k_content_full = concat([Tensor(cache_k_arr), k_new], axis=0)  # [n_content, sw]
        v_content_full = concat([Tensor(cache_v_arr), v_new], axis=0)

        k_full = concat([k_mem_from_prev, k_content_full], axis=0)  # [total_slots, sw]
        v_full = concat([v_mem_from_prev, v_content_full], axis=0)
        k_phys = gather(k_full, self._kv_phys_gather_idx).reshape((self.total_slots, sw))
        v_phys = gather(v_full, self._kv_phys_gather_idx).reshape((self.total_slots, sw))

        # --- PASS 1: WRITE (identical structure to step()'s own PASS 1) ---
        centers_mem = gather(self.centers, self._mem_center_idx)
        sigmas_mem = gather(sigmas, self._mem_center_idx)
        attn_pre_o_mem = gaussian_attention(
            q_mem, k_phys, v_phys, centers_mem, sigmas_mem, num_cpus=self.num_cpus, causal=False
        )
        attn_mem = self._timed_layer_forward(
            self.o_proj,
            "o_proj",
            self._to_sparse(attn_pre_o_mem, "o_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._wide_extra_kwargs("o_proj"),
        )
        _accumulate_penalty(attn_mem)
        attn_mem.data = np.clip(attn_mem.data, -self.clip_range, self.clip_range)
        memory_new_t = rmsnorm_tensor(memory_prev_t + attn_mem, self.state_ln, self.rms_eps)
        memory_new_t.data = np.clip(memory_new_t.data, -self.clip_range, self.clip_range)

        # --- PASS 2: READ, only the newest content row's own query ---
        memory_new_normed = rmsnorm_tensor(memory_new_t, self.memory_ln, self.rms_eps)
        k_mem_fresh = self._timed_layer_forward(
            self.k_proj,
            "k_proj",
            self._to_sparse(memory_new_normed, "k_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._wide_extra_kwargs("k_proj"),
        )
        v_mem_fresh = self._timed_layer_forward(
            self.v_proj,
            "v_proj",
            self._to_sparse(memory_new_normed, "v_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._wide_extra_kwargs("v_proj"),
        )
        _accumulate_penalty(k_mem_fresh)
        _accumulate_penalty(v_mem_fresh)
        k_mem_fresh.data = np.clip(k_mem_fresh.data, -self.clip_range, self.clip_range)
        v_mem_fresh.data = np.clip(v_mem_fresh.data, -self.clip_range, self.clip_range)

        k2_full = concat([k_mem_fresh, k_content_full], axis=0)
        v2_full = concat([v_mem_fresh, v_content_full], axis=0)
        k2_phys = gather(k2_full, self._kv_phys_gather_idx).reshape((self.total_slots, sw))
        v2_phys = gather(v2_full, self._kv_phys_gather_idx).reshape((self.total_slots, sw))

        centers_content = gather(self.centers, self._newest_content_idx)
        sigmas_content = gather(sigmas, self._newest_content_idx)
        if self.recurrent_only_output:
            v2_phys_mem_only = v2_phys * self._mem_only_value_mask
            attn_pre_o_content = gaussian_attention(
                q_new, k2_phys, v2_phys_mem_only, centers_content, sigmas_content, num_cpus=self.num_cpus, causal=False
            )
        else:
            attn_pre_o_content = gaussian_attention(
                q_new, k2_phys, v2_phys, centers_content, sigmas_content, num_cpus=self.num_cpus, causal=False
            )
        attn_content = self._timed_layer_forward(
            self.o_proj,
            "o_proj",
            self._to_sparse(attn_pre_o_content, "o_proj"),
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._wide_extra_kwargs("o_proj"),
        )
        _accumulate_penalty(attn_content)
        attn_content.data = np.clip(attn_content.data, -self.clip_range, self.clip_range)

        pre_norm_content = x_wide_new + attn_content
        content_out = rmsnorm_tensor(pre_norm_content, self.state_ln, self.rms_eps)
        _accumulate_penalty(content_out)
        content_out.data = np.clip(content_out.data, -self.clip_range, self.clip_range)

        self.last_debug = {
            "x_wide": x_wide_new.data,
            "q": q_new.data,
            "k": k_new.data,
            "v": v_new.data,
            "attn_pre_o_mem": attn_pre_o_mem.data,
            "attn_pre_o_content": attn_pre_o_content.data,
            "attn_mem": attn_mem.data,
            "attn_content": attn_content.data,
            "sigmas": sigmas.data,
            "log_sigmas": self.log_sigmas.data,
        }

        memory_new = memory_new_t.data.copy()

        pooled = content_out.reshape((1, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)
        self.last_debug["pooled"] = pooled.data

        if self.l1_sparsity_coef > 0.0:
            l1_terms = [
                self._l1_sparsity_split(
                    self.input_proj,
                    new_embed_sparse,
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="input_proj",
                    **self._wide_extra_kwargs("input_proj"),
                ),
                self._l1_sparsity_split(
                    self.q_proj,
                    self._to_sparse(combined_normed_step, "q_proj"),
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="q_proj",
                    **self._wide_extra_kwargs("q_proj"),
                ),
                self._l1_sparsity_split(
                    self.k_proj,
                    self._to_sparse(combined_normed_step, "k_proj"),
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="k_proj",
                    **self._wide_extra_kwargs("k_proj"),
                ),
                self._l1_sparsity_split(
                    self.v_proj,
                    self._to_sparse(combined_normed_step, "v_proj"),
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="v_proj",
                    **self._wide_extra_kwargs("v_proj"),
                ),
                self._l1_sparsity_split(
                    self.o_proj,
                    self._to_sparse(concat([attn_pre_o_mem, attn_pre_o_content], axis=0), "o_proj"),
                    learning_rate,
                    self.l1_sparsity_coef,
                    requires_grad=requires_grad,
                    layer_name="o_proj",
                    **self._wide_extra_kwargs("o_proj"),
                ),
            ]
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term

        if self.l1_sparsity_coef > 0.0:
            lm_l1 = self._l1_sparsity_split(
                self.lm_head, pooled, learning_rate, self.l1_sparsity_coef, requires_grad=requires_grad
            )
            aux_loss = lm_l1 if aux_loss is None else aux_loss + lm_l1
        logits = self.lm_head.forward(
            pooled, learning_rate, requires_grad=requires_grad, **self.synapse_kwargs, **self._output_extra_kwargs
        )

        self.last_critic_pred = (
            self.critic_head.forward(
                pooled, learning_rate, requires_grad=requires_grad, **self.synapse_kwargs, **self._output_extra_kwargs
            )
            if self.use_critic
            else None
        )

        new_cache = (list(tile_cache) if tile_cache else []) + [
            (k_new.data.copy().reshape(sw), v_new.data.copy().reshape(sw))
        ]
        if len(new_cache) > (n_content - 1):
            new_cache = new_cache[-(n_content - 1) :]

        return memory_new, logits, aux_loss, new_cache
