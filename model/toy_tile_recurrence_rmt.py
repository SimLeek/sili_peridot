from __future__ import annotations

import math
import time

import numpy as np
from sili.energy import EnergyDynamics
from sili.sparse_rnn import CSR, DISLDOLayer, _nucleus_top_k_csr
from sili.tensor import Tensor, concat, exp, gather, gaussian_attention, power, reduce_sum, relu, tensor_abs

from .toy_recall_models import rmsnorm_tensor


def _knee_elbow_r(x2d: np.ndarray) -> float:
    """Kneedle-style elbow detection on the batch-averaged, per-row-
    sorted cumulative squared-magnitude ("energy") curve: the point
    maximizing vertical distance from the diagonal y=x on the curve
    normalized to [0,1]x[0,1] -- where additional active neurons stop
    paying for themselves in captured signal energy. Answers "how much
    of the signal is actually needed" from the data itself instead of
    assuming a fixed threshold (e.g. 99.95%) by hand. Returns R (energy
    fraction retained at the elbow), not k directly, so it's a drop-in
    replacement value for x_r_target/dy_r_target.

    Each row is sorted independently (descending) before averaging --
    summing raw values across rows first would mix each row's OWN
    largest-magnitude dims with others' smallest, destroying the
    per-row energy-concentration shape this is meant to measure."""
    rows, cols = x2d.shape
    v2 = x2d.astype(np.float64) ** 2
    sorted_v2 = -np.sort(-v2, axis=1)
    mean_sorted = sorted_v2.mean(axis=0)
    total = float(mean_sorted.sum())
    if total <= 0:
        return 1.0
    cum = np.cumsum(mean_sorted) / total
    k_norm = np.arange(1, cols + 1) / cols
    elbow_idx = int(np.argmax(cum - k_norm))
    return float(cum[elbow_idx])


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
        x_balance_bias_step: float | None = None,
        x_balance_loss_coef: float = 0.0,
        x_balance_freq_beta: float = 0.99,
        x_r_target_auto: bool = False,
        x_r_target_auto_beta: float = 0.98,
        x_r_target_auto_margin: float = 0.0,
        dy_time_gate_cutoff: float | None = None,
        dy_time_gate_phase_step: float = 0.02,
        dy_time_gate_period: float = 125.0,
        dy_time_gate_seed: int | None = None,
        r_target_min: float = 0.05,
        use_energy: bool = False,
        energy_kwargs: dict | None = None,
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
            See docs/research/toy_tile_recurrence_rmt.rst:x_r_target_design.
        x_balance_bias_step, x_balance_loss_coef, x_balance_freq_beta:
            two independent, mutually-exclusive-in-practice load-balancing
            probes for the x_r_target nucleus selection, prototype-only.
            x_balance_bias_step != None: auxiliary-loss-free balancing
            (DeepSeek-V3-style, arXiv:2408.15664) -- a per-dim bias added
            to the ranking score BEFORE top-k, nudged +/-step each call
            by whether that dim was selected; never touches loss/gradient.
            x_balance_loss_coef > 0: classic MoE-style auxiliary
            load-balancing loss (f_j * mean(x_j^2) summed, f_j = EMA'd
            empirical selection frequency, detached) added to the total
            training loss; selection itself stays plain magnitude top-k.
            Both no-op by default. See project memory
            project_sparsity_floor_generalization_scoping.md's 2026-09-11
            entries for the full design rationale and MoE/biology
            precedent research.
        x_r_target_auto, x_r_target_auto_beta, x_r_target_auto_margin:
            whenever forward-axis nucleus selection runs (x_r_target set
            OR x_r_target_auto True), the batch's elbow R (_knee_elbow_r
            -- see its own docstring) is ALWAYS measured and EMA'd per
            layer into self._x_knee_r_target, purely as a diagnostic --
            "how much signal does this layer's own data actually need,"
            not a fixed hand-picked threshold. x_r_target_auto=True goes
            further: uses that live, self-discovered value (+ the fixed
            margin) AS x_r_target for the real selection, replacing the
            constructor's fixed x_r_target for this layer entirely.
            beta controls how fast the tracked elbow adapts; margin adds
            a fixed safety buffer on top of the raw measured elbow.
            Untested in combination with x_balance_bias_step/
            x_balance_loss_coef -- should compose (auto sets the target
            the balancing mechanisms then rank against) but not verified.
        dy_time_gate_cutoff, dy_time_gate_phase_step, dy_time_gate_period,
        dy_time_gate_seed: Arm C -- pure time-division fairness on the
            BACKWARD/dy axis, independent of any measured signal.
            gate_j(t) = sin(angle_j(t)) > cutoff decides whether neuron
            j's incoming weights are updated THIS step. angle_j(t) =
            angle_j(t-1) + N(2*pi/period, phase_step) -- ONE accumulating
            per-neuron state, started uniform in [0, 2*pi), advanced each
            call by a small NORMAL-distributed increment whose MEAN
            (2*pi/period) is the same fixed constant for every neuron (so
            every neuron cycles at the same average rate -- no per-neuron
            period draw) and whose spread (phase_step) supplies the
            wobble directly, no separate additive phase-noise term needed
            (task correction 2026-09-19/20 -- the earlier t/period_j/
            phase_j(t) three-state version used a global step counter, a
            separate per-neuron uniform-random period, AND a separate
            additive phase random walk; this replaces all three with one
            state variable and one random draw per step, same intended
            effect -- "infinitesimal" per-step drift, never an exactly
            repeating pattern -- with far fewer ops). cutoff is the
            single tunable density knob (higher = fewer neurons trainable
            per step) -- the time-axis analog of dy_r_target's density
            setpoint. No-op unless dy_time_gate_cutoff is set; mutually
            exclusive with dy_r_target/dy_sparsity_p when active (takes
            priority, see sili's DISLDOLayer32.forward dy_gate_mask).
            Requires disldo_cls to support dy_gate_mask (DISLDOLayer32
            only, currently).
        r_target_min: floor the amortized closed-loop controllers
            (apply_amortized_dy_r_target_control/apply_amortized_
            x_r_target_control/apply_cross_layer_budget_allocator) ratchet
            dy_r_target/x_r_target down to when measured throughput is
            below target. Was hardcoded 0.05 in all three call sites --
            see JOURNAL.md's 2026-09-07 adaptive_70k entry: at that floor,
            a 70k-param (embed_width=16) model's quality collapsed
            (curriculum never leveled past its starting vocab tier in
            100k steps) well before reaching the requested speedup
            target. Exposed here so the floor itself can be swept to find
            the largest speedup a given model size actually tolerates.
        use_energy, energy_kwargs: task #269 -- homeostatic energy gating
            (sili.energy.EnergyDynamics) applied to the two most nucleus-
            sparsity-starved activation regions -- "state" (combined_normed,
            feeding q/k/v_proj) and "embed_input" (x_window_t, feeding
            input_proj) -- one EnergyDynamics instance per region, not
            per-layer, not global. energy_kwargs is keyed BY REGION NAME,
            e.g. {"state": {...}, "embed_input": {...}} (mirrors x_r_target/
            dy_r_target's own per-layer-name dict convention) -- required
            per-region since the two regions' E[|h|] measure ~4x apart
            (state~=0.23, embed_input~=0.06 at width=288) and
            EnergyDynamics.drive must be calibrated against the region it
            actually gates, not shared. No-op unless use_energy=True."""
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
        # See __init__'s own x_balance_bias_step/x_balance_loss_coef docstring.
        self.x_balance_bias_step = x_balance_bias_step
        self.x_balance_loss_coef = x_balance_loss_coef
        self.x_balance_freq_beta = x_balance_freq_beta
        self._x_balance_bias: dict = {}  # layer_name -> np.float32[n_in], Arm B
        self._x_balance_freq: dict = {}  # layer_name -> np.float32[n_in], Arm A
        self._balance_aux_loss: Tensor | None = None
        # See __init__'s own x_r_target_auto docstring.
        self.x_r_target_auto = x_r_target_auto
        self.x_r_target_auto_beta = x_r_target_auto_beta
        self.x_r_target_auto_margin = x_r_target_auto_margin
        self._x_knee_r_target: dict = {}  # layer_name -> float, EMA'd elbow R
        # See __init__'s own dy_time_gate_* docstring (Arm C).
        self.dy_time_gate_cutoff = dy_time_gate_cutoff
        self.dy_time_gate_phase_step = dy_time_gate_phase_step
        self.dy_time_gate_period = dy_time_gate_period
        self._dy_time_gate_rng = np.random.default_rng(dy_time_gate_seed)
        self._dy_time_gate_angle: dict = {}  # layer_name -> np.float64[n_out]
        # See docs/research/toy_tile_recurrence_rmt.rst:input_selection_stats_design.
        self.last_input_selection: dict = {}
        # See docs/research/toy_tile_recurrence_rmt.rst:dy_r_target_nucleus_design.
        self.dy_r_target: dict = dict.fromkeys(self._WIDE_LAYER_NAMES, dy_r_target)
        self.dy_k_min = dy_k_min
        self.dy_k_max = dy_k_max
        # See __init__'s own r_target_min docstring above.
        self.r_target_min = r_target_min
        # See docs/research/toy_tile_recurrence_rmt.rst:dy_surprise_design.
        self.dy_surprise_alpha = dy_surprise_alpha
        self.dy_surprise_beta = dy_surprise_beta
        self._layer_surprise: dict = {}
        # Per-layer LR override for apply_polyak_lr, below. Empty (default):
        # byte-identical to today's exact behavior -- see
        # docs/research/toy_tile_recurrence_rmt.rst:per_layer_learning_rate_polyak.
        self.layer_lr_override: dict = {}
        # See docs/research/toy_tile_recurrence_rmt.rst:sparsity_phase6_design.
        self.output_dy_sparsity_p = output_dy_sparsity_p
        self._output_extra_kwargs = {"dy_sparsity_p": output_dy_sparsity_p} if output_dy_sparsity_p is not None else {}
        # See __init__'s own use_energy/energy_kwargs docstring above.
        self.use_energy = use_energy
        self.energy_kwargs = energy_kwargs or {}
        self._energy: dict = {}  # region name -> EnergyDynamics, lazily built
        self._energy_aux_loss: Tensor | None = None

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
        # EXPERIMENTAL, added 2026-09-20 -- critical-learning-periods/
        # loss-of-plasticity forgetting mechanism, sustained-loss stall
        # tracking for apply_loss_adjusted_decay. See
        # docs/research/toy_tile_recurrence_rmt.rst:loss_adjusted_decay_design.
        self._decay_loss_ema: float | None = None
        self._decay_loss_floor: float | None = None
        self._decay_stall_steps: int = 0
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

    @property
    def last_grad_selection(self) -> dict:
        """Real per-layer GRADIENT-axis (dy_r_target) R/k stats, pulled
        from each disldo_cls layer's own last_grad_selection (set inside
        sili's dy_r_target nucleus branch, sili/sparse_rnn.py) -- mirrors
        last_input_selection's x-axis equivalent, which previously had no
        counterpart: the dy axis only ever exposed its setpoint (r_bar),
        never a measured achieved density. A layer with no entry yet
        (never had a dy_r_target backward call, or its disldo_cls doesn't
        support dy_r_target at all, e.g. DISLDOLayer8/fp8) is simply
        omitted."""
        out = {}
        for name, layer in self._named_real_layers():
            sel = getattr(layer, "last_grad_selection", None)
            if sel is not None:
                out[name] = sel
        return out

    def _timed_call(self, name: str, fn, *args, **kwargs) -> Tensor:
        """Generic per-component forward+backward timing, and the task #374
        surprise-signal capture point -- same accounting as
        _timed_layer_forward but for any Tensor-returning callable (e.g.
        gaussian_attention, lm_head.forward), not just a disldo_cls layer's
        own .forward. See
        docs/research/toy_tile_recurrence_rmt.rst:layer_timing_design and
        :component_timing_breakdown_design (lm_head/critic_head/attention
        added as their own named buckets, task #415 -- previously only the
        5 wide layers were timed, so everything else silently fell into an
        unmeasured "the rest" bucket that neither speedup estimates nor
        Amdahl's-law reasoning could actually check against real numbers)."""
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        rec = self._layer_timing.setdefault(name, {"fwd_s": 0.0, "bwd_s": 0.0, "fwd_calls": 0, "bwd_calls": 0})
        rec["fwd_s"] += time.perf_counter() - t0
        rec["fwd_calls"] += 1
        orig_backward = out._backward
        if orig_backward is not None:

            def _timed_backward(_orig=orig_backward, _rec=rec, _out=out, _name=name):
                tb0 = time.perf_counter()
                _orig()
                _rec["bwd_s"] += time.perf_counter() - tb0
                _rec["bwd_calls"] += 1
                # Always tracked (not gated on dy_surprise_alpha) -- also the
                # per-layer gradient-energy signal for Polyak-style dynamic
                # LR (docs/research/toy_tile_recurrence_rmt.rst:
                # per_layer_learning_rate_polyak), same "cheap, always on"
                # precedent as the knee-elbow diagnostic in _to_sparse.
                if _out.grad is not None:
                    self._update_layer_surprise(_name, _out.grad)

            out._backward = _timed_backward
        return out

    def _timed_layer_forward(self, layer, layer_name: str, x, learning_rate, *args, **kwargs) -> Tensor:
        """Real per-layer forward+backward timing, and the task #374
        surprise-signal capture point. self.layer_lr_override (task #? --
        see docs/research/toy_tile_recurrence_rmt.rst:
        per_layer_learning_rate_polyak), when it has an entry for
        layer_name, OVERRIDES the passed-in learning_rate for THIS layer
        only -- empty dict (default): byte-identical to today's exact
        behavior, same as every other conditional kwarg here. lm_head/
        critic_head calls don't go through this method, so per-layer
        overrides never reach them (mirrors self.dy_r_target's own
        wide-layers-only scope). See
        docs/research/toy_tile_recurrence_rmt.rst:layer_timing_design."""
        lr = self.layer_lr_override.get(layer_name, learning_rate) if self.layer_lr_override else learning_rate
        return self._timed_call(layer_name, layer.forward, x, lr, *args, **kwargs)

    def reset_layer_timing(self) -> None:
        """Zero self._layer_timing -- call at the start of a measurement
        window. See docs/research/toy_tile_recurrence_rmt.rst:layer_timing_design."""
        self._layer_timing = {}

    def layer_timing_snapshot(self) -> dict:
        """Deep-enough copy of self._layer_timing (per-component fwd_s/
        bwd_s/fwd_calls/bwd_calls) for callers that need the CURRENT
        window's breakdown before it gets consumed/reset -- e.g. task #415's
        standard per-component timing report. See
        docs/research/toy_tile_recurrence_rmt.rst:layer_timing_design."""
        return {name: dict(rec) for name, rec in self._layer_timing.items()}

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
        """r_t = clip(r_bar * (E_t/Lbar)^alpha, self.r_target_min, 0.99),
        lagged one step. See docs/research/toy_tile_recurrence_rmt.rst:
        dy_surprise_design."""
        r_bar = self.dy_r_target.get(layer_name)
        if r_bar is None or self.dy_surprise_alpha is None:
            return r_bar
        surprise = self._layer_surprise.get(layer_name)
        if surprise is None or surprise["Lbar"] <= 0.0:
            return r_bar
        ratio = surprise["E_t"] / surprise["Lbar"]
        r_t = r_bar * (ratio**self.dy_surprise_alpha)
        return min(max(r_t, self.r_target_min), 0.99)

    # See docs/research/toy_tile_recurrence_rmt.rst:arm_c_single_state_angle_fix
    # -- corrected 2026-09-19/20 from an earlier t/period_j/phase_j(t) design
    # that didn't match original intent and used more state/ops than needed.
    def _dy_time_gate(self, layer_name: str) -> np.ndarray | None:
        """Arm C: gate_j(t) = sin(angle_j(t)) > cutoff. angle_j(t) =
        angle_j(t-1) + N(2*pi/period, phase_step) -- ONE accumulating
        per-neuron state (no global step counter, no separate per-neuron
        period draw, no separate additive phase term), started uniform in
        [0, 2*pi). The increment's MEAN (2*pi/period) is the same fixed
        constant for every neuron, so every neuron cycles at the same
        average rate; its spread (phase_step) supplies the "infinitesimal"
        wobble directly -- the co-active neuron set never exactly repeats,
        same intent as before with far fewer ops. No-op (returns None)
        unless dy_time_gate_cutoff is set. See __init__'s own docstring."""
        if self.dy_time_gate_cutoff is None:
            return None
        n_out = getattr(self, layer_name).out_features
        angle = self._dy_time_gate_angle.get(layer_name)
        if angle is None:
            angle = self._dy_time_gate_rng.uniform(0.0, 2 * np.pi, size=n_out)
        mean_step = 2 * np.pi / self.dy_time_gate_period
        angle = angle + self._dy_time_gate_rng.normal(mean_step, self.dy_time_gate_phase_step, size=n_out)
        self._dy_time_gate_angle[layer_name] = angle
        return np.sin(angle) > self.dy_time_gate_cutoff

    def _wide_extra_kwargs(self, layer_name: str) -> dict:
        """Extra kwargs for ONE of the 5 affected layers' forward() calls;
        layer_name must be one of _WIDE_LAYER_NAMES. Live (not cached) since
        self.dy_r_target[name] is mutable post-construction. See
        docs/research/toy_tile_recurrence_rmt.rst:dy_r_target_nucleus_design."""
        gate = self._dy_time_gate(layer_name)
        if gate is not None:
            return {"dy_gate_mask": gate}
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
        r_min: float | None = None,
        r_max: float = 0.99,
    ) -> dict:
        """Closed-loop controller adjusting self.dy_r_target against
        MEASURED steps/sec. See
        docs/research/toy_tile_recurrence_rmt.rst:amortized_r_target_control_design."""
        if r_min is None:
            r_min = self.r_target_min
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
        r_min: float | None = None,
        r_max: float = 0.99,
    ) -> dict:
        """Same as apply_amortized_dy_r_target_control, operating on
        x_r_target (INPUT axis) instead. See
        docs/research/toy_tile_recurrence_rmt.rst:amortized_r_target_control_design."""
        if r_min is None:
            r_min = self.r_target_min
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

    def apply_polyak_lr(
        self,
        loss: float,
        f_star: float = 0.0,
        c: float = 0.0005,
        lr_max: float = 0.05,
        bootstrap_lr: float = 0.01,
    ) -> dict:
        """Per-layer Stochastic-Polyak-Step-size (SPS_max variant):
        lr_layer = min(lr_max, c * max(loss - f_star, 0) / Lbar_layer),
        using each wide layer's OWN EMA-smoothed gradient energy
        (self._layer_surprise[name]["Lbar"] -- see _timed_call) -- same
        global loss, per-layer denominator, one step lagged (this
        step's Lbar reflects last step's backward, same lag convention
        as _effective_dy_r_target). f_star=0 assumed (SPS_max -- see
        docs/research/train_mqar_curriculum.rst:
        polyak_lr_f_star_assumption). bootstrap_lr: used for any layer
        with no Lbar yet.

        Uses Lbar (EMA-smoothed), NOT the raw per-call E_t -- direct
        finding, 2026-09-19: E_t swings by orders of magnitude call to
        call (observed range ~0 to ~2000 for a single wide layer),
        constantly saturating the formula against lr_max regardless of
        its value. c=0.5 (a typical SPS damping factor from the
        literature) was also wildly miscalibrated for THIS setup's
        actual Lbar scale (~0.02-0.15 typically) -- empirically
        recalibrated to c=0.0005, lr_max=0.05 so computed values land
        in the same range the LR range test/grid search already found
        safe (~0.01-0.06), not a literature default that turned out not
        to transfer. See
        docs/research/toy_tile_recurrence_rmt.rst:
        per_layer_learning_rate_polyak's update for the full story
        (first validation run, uncalibrated, failed to learn at all).

        TODO, not yet built: PER-NEURON Polyak (one lr per row of a
        layer, not one per layer) -- structural/grad sparsity act
        row-wise already, so this per-layer version can't adapt to a
        layer whose neurons have very different realized fan-in/update
        frequency this step. Real extra work (needs the per-row E_t
        version of this same hook, see sili__new's
        disldo_layer_forward.last_grad_norm_sq_polyak_hook TODO) -- try
        per-layer first, revisit if it doesn't adapt well.

        Sets self.layer_lr_override (consumed by _timed_layer_forward
        for the 5 wide layers only, never lm_head/critic_head) and
        returns it. See
        docs/research/toy_tile_recurrence_rmt.rst:
        per_layer_learning_rate_polyak."""
        residual = max(loss - f_star, 0.0)
        updated = {}
        for name in self._WIDE_LAYER_NAMES:
            lbar = self._layer_surprise.get(name, {}).get("Lbar")
            lr = bootstrap_lr if not lbar else min(lr_max, c * residual / lbar)
            self.layer_lr_override[name] = lr
            updated[name] = lr
        return updated

    def apply_cross_layer_budget_allocator(
        self,
        measured_sps: float,
        target_sps: float,
        down_factor: float = 0.85,
        up_factor: float = 1.05,
        r_min: float | None = None,
        r_max: float = 0.99,
    ) -> dict:
        """Coordinates x_r_target (INPUT axis) against the remaining compute
        budget using per-layer timing; dy_r_target (GRAD axis) untouched.
        See docs/research/toy_tile_recurrence_rmt.rst:cross_layer_budget_allocator_design."""
        if r_min is None:
            r_min = self.r_target_min
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

    # block4 tiles are BLOCK4_TILE x BLOCK4_TILE (4x4=16) cells -- a block4
    # chunk_size counts TILES, not individual synapses, unlike every other
    # storage type's decay call. See apply_loss_adjusted_decay's own
    # touch_fraction docstring section for why this matters.
    _BLOCK4_TILE_SLOTS = 16

    def apply_loss_adjusted_decay(
        self,
        loss: float,
        touch_fraction: float = 0.01,
        min_chunk: int = 4,
        max_chunk: int = 2048,
        importance_half_life_touches: float | None = 500.0,
        weight_half_life_touches: float | None = None,
        min_stall_steps: int = 200,
        ramp_steps: int = 200,
        beta_fast: float = 0.9,
        beta_floor: float = 0.999,
        improve_tol: float = 1e-3,
    ) -> dict:
        """EXPERIMENTAL, added 2026-09-20 -- critical-learning-periods/
        loss-of-plasticity forgetting (Achille et al. 2019 "Critical
        Learning Periods in Deep Neural Networks"; Dohare et al., *Nature*
        2024 "Loss of plasticity in deep continual learning"). Direct
        hypothesis this implements: a training stall isn't "variance" --
        the network structurally can't un-learn early bad adaptations
        (matching real critical-learning-periods findings), so SUSTAINED
        high loss (not one noisy step) should trigger MORE forgetting to
        restore plasticity. See
        docs/research/toy_tile_recurrence_rmt.rst:loss_adjusted_decay_design.

        Two INDEPENDENT, separately-toggleable arms -- direct instruction:
        implement both, note both are actively being tested, either may
        be removed if it doesn't show benefit:

        1. ``importance_half_life_touches`` (default on, 500 touches):
           decays each layer's per-synapse IMPORTANCE -- this project's
           RMSprop-style second-moment accumulator (see
           feedback_importance_is_already_the_optimizer) -- the
           mechanism-specific fix for adaptive optimizers' own
           documented "freezing" of heavily-used units.
        2. ``weight_half_life_touches`` (default OFF, ``None``): decays
           each layer's actual WEIGHT -- the more classic forgetting
           mechanism. Shares the SAME per-layer C++ weight-decay cursor
           as ``apply_amortized_l2_decay`` (there is only one
           ``_decay_cursor`` per layer) -- do not enable this arm in a
           training loop that also calls ``apply_amortized_l2_decay``
           every step; both would drive the same cursor out of sync with
           each other's own decay_factor intent. Off by default for
           exactly this reason.

        SEVERITY, real miscalibration caught before ever launching a real
        run: the first version used an abstract 0..1 "strength" combined
        with a per-touch floor (``decay_factor = max(min_decay_factor, 1 -
        stall_frac*strength)``) with NO principled connection to how many
        TIMES a synapse actually gets touched over a real stall's
        duration -- and the known historical stalls this hypothesis is
        meant to address ran FLAT for 74,000-99,000+ training steps once
        stalled. At ``strength=0.3``, full stall settled at
        ``decay_factor=0.7`` per touch (the ``min_decay_factor=0.5`` floor
        was never even reached -- ``max(0.5, 1-1*0.3)=0.7``); compounded
        over the ~100-call amortized cycle this design already uses,
        that's ~0.7 per ~100 calls per synapse, and empirically crushed a
        real width=288 layer's weights to float-zero within ~1400 calls
        -- a small fraction of one historical stall's real length. That
        would have made the very thing under test ("does BOUNDED,
        recoverable forgetting restore plasticity") into an accidental
        full deletion instead, well before the stall it's meant to help
        with even finishes.

        Fixed by deriving ``decay_factor`` from a HALF-LIFE IN TOUCHES
        instead, the same style ``apply_amortized_l2_decay``'s own
        RST docs already establish for THIS project's decay_factor
        derivations (``2^(-cycle_length/H)`` for a chosen half-life
        ``H``): ``decay_factor_at_full_stall = 2**(-1/half_life_touches)``
        -- a synapse touched ``half_life_touches`` times in a row under
        FULLY saturated stall (``stall_frac=1``) is reduced to exactly
        half, not to numerical zero. Interpolated toward 1.0 (no decay) as
        ``stall_frac`` falls: ``decay_factor = 1 - stall_frac*(1 -
        decay_factor_at_full_stall)``. Default ``500`` touches, each
        touch spaced ~1 amortized cycle (~100 calls at the default
        ``touch_fraction``) apart -- ~50,000 calls, the same ORDER OF
        MAGNITUDE as the historical stalls' own real length, to halve at
        worst -- bounded, testable, and deliberately conservative for a
        first empirical run rather than a guessed severity. ``None``
        disables an arm entirely (an exact no-op, not merely a large
        half-life).

        Stall signal: a fast EMA of loss (``beta_fast``) vs a
        slow-relaxing floor (``beta_floor``) -- ``self._decay_stall_steps``
        counts consecutive steps since the EMA last beat the floor by at
        least ``improve_tol`` (relative) -- a TIE or a smaller dip does
        NOT reset stall (real bug caught by testing: a plain ``<=``
        comparison let the EMA's own filter-transient settling from its
        initial value read as continuous "improvement" on a genuinely
        flat/oscillating loss, permanently masking a real stall -- fixed
        by requiring a real margin, not just a decrease). Reset to 0 the
        instant a genuine improvement occurs -- purely from LOSS (the
        ``loss`` argument, whatever a caller's own loss signal is --
        prediction/curiosity/external-input, never accuracy/correctness:
        this project's curriculum advancement separately tracks
        ``acc_ema``/``streak`` for its OWN purposes, but neither is an
        input here, since a deployed system may not have ground truth to
        check against). The floor still relaxes slowly upward on
        non-improving steps so a genuine curriculum-driven loss increase
        (harder level) doesn't read as a permanent stall forever.
        ``min_stall_steps``: a REAL BASELINE/GRACE PERIOD, not merely a
        gentle ramp -- real gap caught by direct question ("I feel like
        it would need to establish a baseline and shouldn't necessarily
        decay all the time"): without this, ``stall_frac`` (and therefore
        ``decay_factor``) is technically nonzero after almost every call,
        since ``stall_steps`` starts incrementing from the very FIRST call
        (the initial ``ema==floor`` tie never counts as improvement) and
        keeps incrementing on any call that doesn't beat the floor by a
        full ``improve_tol`` -- ordinary noisy-but-productive training
        doesn't clear a fresh margin every single call either, so decay
        was ALWAYS slightly active, just negligibly so most of the time --
        never a clean, exact "off" state. Fixed: ``stall_frac`` is now
        EXACTLY 0 (``decay_factor`` exactly 1.0, a true no-op) for the
        first ``min_stall_steps`` consecutive non-improving calls, only
        ramping in afterward:
        ``stall_frac = min(1, max(0, stall_steps - min_stall_steps) /
        ramp_steps)``. Default 200 (same order as ``ramp_steps``, so full
        severity is reached only after ~400 consecutive non-improving
        calls total) -- small relative to the historical stalls' own
        74,000-99,000+ step length, but large enough that ordinary
        training noise/short plateaus never trigger any decay at all.

        The DECAY_FACTOR (how hard each touched synapse decays) is applied
        UNIFORMLY across every real layer -- loss is a single global
        signal, no finer-than-per-layer signal is available without new
        per-neuron instrumentation (same per-layer, not per-neuron,
        limitation ``apply_polyak_lr`` already documents for itself).

        KNOWN GAP, high priority, not yet fixed (direct instruction:
        prefer per-neuron over per-layer wherever possible, since
        synapses-per-neuron differ between neurons in any genuinely
        sparse layer): the decay_factor itself is per-layer, and the
        CHUNK/touch selection below is not neuron-aware either -- it
        walks the flat underlying array/cursor, not per-row (per-neuron).
        NOTE, corrected 2026-09-20: a fully dense layer having uniform
        fan-in does NOT make per-layer and per-neuron equivalent, even
        there -- different neurons still receive different gradient
        signal / contribute differently to the loss even with identical
        fan-in, so a genuinely per-neuron mechanism would still decay
        them differently. The reason per-layer is acceptable FOR NOW is
        granularity parity with the source literature (Achille et al.
        2019; Dohare et al. 2024 test unit-/layer-level interventions,
        not a loss-attributed per-neuron scheme either) -- not that this
        method's fan-in-uniform case happens to make the two identical.
        A genuinely sparse layer additionally has heterogeneous per-row
        nnz on top of that (a high-fan-in neuron and a low-fan-in
        neuron's few synapses could be touched at very different
        effective rates purely from the flat cursor's walk order) --
        must be fixed (e.g. a per-row-aware cursor, or scaling each row's
        own decay_factor by its own realized fan-in or gradient signal)
        before trusting this on sparse configs, and ideally before
        trusting the PER-NEURON claim at all even on dense ones. Tracked,
        not silently dropped.

        ``touch_fraction``: FRACTION OF EACH LAYER'S OWN nnz to touch per
        call (chunk_size = clip(round(nnz*touch_fraction), min_chunk,
        max_chunk)), NOT one shared absolute chunk_size -- real bug
        caught before ever launching a real run: this project's layers
        span a ~150x nnz range at width=288 (lm_head early in the vocab
        curriculum: 576; q/k/v/o_proj fully dense: 82944). One shared
        absolute chunk_size either barely touches the big layers (a
        practical no-op) or, for the SMALL layers, exceeds their own nnz
        entirely -- the amortized cursor then wraps around and touches
        (decays) EVERY synapse in that layer more than once per call,
        which compounds much faster than the half-life above assumes
        (each "touch" the half-life counts is meant to be ~one per
        amortized cycle, not several per call). touch_fraction makes
        every
        layer's own full amortized cycle take roughly ``1/touch_fraction``
        calls regardless of its absolute size, sidestepping both
        failure modes.

        Calls the block4 counterpart
        (``apply_amortized_block4_l2_decay``/
        ``apply_amortized_block4_importance_decay``) too when the layer
        has one, since ``dense=True`` routes weights into block4 storage
        -- see feedback_block4_scattered_parity_required. block4's own
        chunk_size counts TILES (``_BLOCK4_TILE_SLOTS=16`` cells each),
        not individual synapses -- reusing the scattered chunk_size
        directly there would over-touch a block4-resident layer by
        roughly 16x relative to touch_fraction's intent, so the block4
        call gets its own chunk_size divided by ``_BLOCK4_TILE_SLOTS``.
        Exact (not just an upper bound) for a layer whose tiles are all
        still dense-format -- true for every ``dense=True`` config tested
        so far (16 live cells/tile at init exceeds ``switch_point``'s
        14-cell sparse threshold, forcing dense format from the start);
        becomes a slight over-touch approximation if synaptogenesis later
        demotes some tiles to sparse-packed (fewer than 16 live cells
        each) -- not yet a concern for any config this has been run
        against.

        Returns ``{layer_name: {"stall_frac": ..., "importance": {...} |
        None, "weight": {...} | None}}``."""
        if self._decay_loss_ema is None:
            self._decay_loss_ema = loss
            self._decay_loss_floor = loss
        else:
            self._decay_loss_ema = beta_fast * self._decay_loss_ema + (1.0 - beta_fast) * loss
        if self._decay_loss_ema < self._decay_loss_floor * (1.0 - improve_tol):
            self._decay_loss_floor = self._decay_loss_ema
            self._decay_stall_steps = 0
        else:
            self._decay_stall_steps += 1
            self._decay_loss_floor = beta_floor * self._decay_loss_floor + (1.0 - beta_floor) * self._decay_loss_ema
        stall_frac = min(1.0, max(0, self._decay_stall_steps - min_stall_steps) / max(ramp_steps, 1))

        results = {}
        for name, layer in self._named_real_layers():
            nnz = layer.nnz
            if nnz <= 0:
                continue
            # Per-LAYER (not per-neuron -- see the docstring's KNOWN GAP
            # section) touch amount, sized relative to THIS layer's own
            # nnz so a full amortized cycle takes ~1/touch_fraction calls
            # regardless of absolute layer size -- see touch_fraction's
            # own docstring section for the no-op/deletion-op failure
            # modes this avoids.
            chunk_size = int(min(max_chunk, max(min_chunk, round(nnz * touch_fraction))))
            # block4 chunk_size counts TILES, not synapses -- see
            # _BLOCK4_TILE_SLOTS's own docstring section.
            block4_chunk_size = int(max(1, round(chunk_size / self._BLOCK4_TILE_SLOTS)))
            entry = {"stall_frac": stall_frac, "importance": None, "weight": None}
            if importance_half_life_touches is not None and hasattr(layer, "apply_amortized_importance_decay"):
                floor = 2.0 ** (-1.0 / importance_half_life_touches)
                imp_decay_factor = 1.0 - stall_frac * (1.0 - floor)
                stats = layer.apply_amortized_importance_decay(chunk_size, imp_decay_factor)
                if hasattr(layer, "apply_amortized_block4_importance_decay"):
                    stats = dict(
                        stats,
                        block4=layer.apply_amortized_block4_importance_decay(block4_chunk_size, imp_decay_factor),
                    )
                entry["importance"] = dict(stats, decay_factor=imp_decay_factor, chunk_size=chunk_size)
            if weight_half_life_touches is not None and hasattr(layer, "apply_amortized_l2_decay"):
                floor = 2.0 ** (-1.0 / weight_half_life_touches)
                w_decay_factor = 1.0 - stall_frac * (1.0 - floor)
                stats = layer.apply_amortized_l2_decay(chunk_size, w_decay_factor)
                if hasattr(layer, "apply_amortized_block4_l2_decay"):
                    stats = dict(stats, block4=layer.apply_amortized_block4_l2_decay(block4_chunk_size, w_decay_factor))
                entry["weight"] = dict(stats, decay_factor=w_decay_factor, chunk_size=chunk_size)
            results[name] = entry
        return results

    def apply_plasticity_reset(
        self,
        touch_fraction: float = 0.01,
        min_chunk: int = 4,
        max_chunk: int = 2048,
        eta: float = 0.99,
        eta_slow: float = 0.99,
        eta_slow_catchup: float = 0.95,
        eta_fast: float = 0.5,
        blend: float = 0.10,
        reset_fraction: float = 0.01,
        k: float = 1.0,
        eta_var: float = 0.9,
        include_column_state: bool = False,
    ) -> dict:
        """EXPERIMENTAL -- per-neuron utility-based plasticity reset
        (Continual-Backprop-inspired). See
        docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design
        for the full derivation (single top-K FROZEN pool, gated by a
        local gradient-activity z-score; the dead pool was pruned after
        a real relaunch showed a self-reinforcing spiral; k=1.0
        recalibrated from real data). No ``loss`` argument. Same
        ``touch_fraction``/``_BLOCK4_TILE_SLOTS`` chunk sizing as
        ``apply_loss_adjusted_decay``. ``include_column_state=True``
        additionally fetches a per-column snapshot (plain numpy copies)
        for any pool whose cycle just completed -- see
        plasticity_column_state in the design doc. Returns
        ``{layer_name: {"importance": {...}}}``, nested ``"block4"`` key
        when that layer has block4 storage."""
        results = {}
        for name, layer in self._named_real_layers():
            nnz = layer.nnz
            if nnz <= 0:
                continue
            chunk_size = int(min(max_chunk, max(min_chunk, round(nnz * touch_fraction))))
            block4_chunk_size = int(max(1, round(chunk_size / self._BLOCK4_TILE_SLOTS)))
            if not hasattr(layer, "apply_amortized_plasticity_reset"):
                continue
            stats = layer.apply_amortized_plasticity_reset(
                chunk_size,
                eta,
                eta_slow,
                eta_slow_catchup,
                eta_fast,
                blend,
                reset_fraction,
                k,
                eta_var,
            )
            # scattered_nnz gate: a dense-loaded real layer's content lives
            # ENTIRELY in block4 (load_dense_values -> block4_load_dense_fp32),
            # so the scattered arm's apply call trivially reports
            # cycle_complete=True on every single call (early-return path
            # for nnz==0) with degenerate all-zero column state -- skip
            # attaching that noise rather than logging a stream of
            # meaningless snapshots for an arm that will never have real
            # content for this run's config.
            if (
                include_column_state
                and stats.get("cycle_complete")
                and hasattr(layer, "plasticity_column_state")
                and getattr(layer, "scattered_nnz", 1) > 0
            ):
                stats = dict(
                    stats, column_state={key: np.array(v) for key, v in layer.plasticity_column_state().items()}
                )
            if hasattr(layer, "apply_amortized_block4_plasticity_reset"):
                block4_stats = layer.apply_amortized_block4_plasticity_reset(
                    block4_chunk_size,
                    eta,
                    eta_slow,
                    eta_slow_catchup,
                    eta_fast,
                    blend,
                    reset_fraction,
                    k,
                    eta_var,
                )
                if (
                    include_column_state
                    and block4_stats.get("cycle_complete")
                    and hasattr(layer, "plasticity_column_state_block4")
                ):
                    block4_stats = dict(
                        block4_stats,
                        column_state={key: np.array(v) for key, v in layer.plasticity_column_state_block4().items()},
                    )
                stats = dict(stats, block4=block4_stats)
            results[name] = {"importance": stats}
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
        dynamic control end up with" (task #292). additive_rank reads 0
        for backends without AQRS's additive branch (e.g. DISLDOLayerV/
        fp32) rather than crashing -- get_scale_rank alone doesn't imply
        get_additive_rank exists."""
        results = {}
        for name, layer in self._named_real_layers():
            c = getattr(layer, "_c", None)
            if c is not None and hasattr(c, "get_scale_rank"):
                additive_rank = c.get_additive_rank() if hasattr(c, "get_additive_rank") else 0
                results[name] = (c.get_scale_rank(), additive_rank)
        return results

    def _apply_energy(self, x: Tensor, region: str) -> Tensor:
        """Homeostatic energy gating (task #269), applied to a region's
        FULL continuous tensor BEFORE its x_r_target nucleus selection --
        see __init__'s own use_energy/energy_kwargs docstring for why
        (breaking nucleus-selection's dead-neuron lock-in is the whole
        point, and that only works if energy sees the pre-selection
        values). No-op unless use_energy=True. Lazily builds one
        EnergyDynamics per region name on first call (region, not layer
        name -- q_proj/k_proj/v_proj all call this with region="state" on
        the SAME combined_normed tensor, so they share one instance, not
        three). energy_kwargs is keyed BY REGION NAME (mirrors x_r_target/
        dy_r_target's own per-layer-name dict convention) -- required,
        since "state" and "embed_input" measure ~4x apart in E[|h|]
        (state~=0.23, embed_input~=0.06 at width=288, measured directly,
        see JOURNAL.md) and EnergyDynamics.drive must be calibrated per
        the region it actually gates, not shared. aux_loss accumulates
        into self._energy_aux_loss for the caller to fold into step()'s
        own aux_loss before returning -- step() resets
        self._energy_aux_loss to None at its own start."""
        if not self.use_energy:
            return x
        ed = self._energy.get(region)
        if ed is None:
            ed = EnergyDynamics(**self.energy_kwargs[region])
            self._energy[region] = ed
        h_out, aux, _actual_p = ed(x)
        self._energy_aux_loss = aux if self._energy_aux_loss is None else self._energy_aux_loss + aux
        return h_out

    def _to_sparse(self, x: Tensor, layer_name: str) -> Tensor:
        """Sparsity plan Phase 6 helper; no-op unless x_r_target[layer_name]
        or input_sparsity_p is set. Real bug fix: wires _children/_backward
        so gradient flows straight-through to x (CSR.as_tensor() alone
        silently detaches the graph). See
        docs/research/toy_tile_recurrence_rmt.rst:to_sparse_gradient_detach_bug
        and :x_r_target_design."""
        x_r_target = self.x_r_target.get(layer_name)
        if x_r_target is None and self.input_sparsity_p is None and not self.x_r_target_auto:
            return x
        x_np = np.asarray(x.data, dtype=np.float32)
        x2d = x_np[np.newaxis, :] if x_np.ndim == 1 else x_np
        if x_r_target is not None or self.x_r_target_auto:
            knee_r = self._update_knee_r_target(layer_name, x2d)
            if self.x_r_target_auto:
                x_r_target = knee_r
            if self.x_balance_bias_step is not None:
                ptrs, indices, values = self._nucleus_top_k_balanced(x2d, layer_name, x_r_target)
            else:
                ptrs, indices, values = _nucleus_top_k_csr(
                    x2d, x_r_target, self.num_cpus, k_min=self.x_k_min, k_max=self.x_k_max
                )
            csr = CSR(ptrs, indices, values, rows=x2d.shape[0], cols=x2d.shape[1])
        else:
            csr = CSR.from_dense(x2d, self.input_sparsity_p, self.num_cpus)
        self._update_input_selection_stats(layer_name, x2d, csr)
        if x_r_target is not None:
            freq = self._update_x_balance_freq(layer_name, csr)
            if self.x_balance_loss_coef > 0.0:
                self._accumulate_balance_loss(x, layer_name, freq)
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

    def _nucleus_top_k_balanced(
        self, x2d: np.ndarray, layer_name: str, r_target
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arm B -- auxiliary-loss-free load balancing (DeepSeek-V3-style,
        arXiv:2408.15664). Same nucleus/energy-threshold contract as
        _nucleus_top_k_csr, but ranking uses vⱼ²+biasⱼ instead of raw vⱼ²
        -- biasⱼ is a persistent per-dim state nudged +/-x_balance_bias_step
        each call by whether that dim was selected THIS call, never touched
        by gradient. R(v,k) energy accounting stays in TRUE vⱼ² units
        (bias only changes ranking, never what gets stored). See __init__'s
        x_balance_bias_step docstring."""
        rows, cols = x2d.shape
        bias = self._x_balance_bias.get(layer_name)
        if bias is None:
            bias = np.zeros(cols, dtype=np.float64)
        ptrs = np.zeros(rows + 1, dtype=np.int64)
        all_indices: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        selected_mask = np.zeros(cols, dtype=bool)
        for row in range(rows):
            v_row = x2d[row]
            v_sq = v_row.astype(np.float64) ** 2
            total = float(np.sum(v_sq))
            score = v_sq + bias
            order = np.argsort(-score, kind="stable")
            if total <= 0:
                k = 0
            else:
                cum = np.cumsum(v_sq[order])
                k = int(np.searchsorted(cum, r_target * total) + 1)
                k = min(k, cols)
            if self.x_k_min:
                k = max(k, self.x_k_min)
            if self.x_k_max is not None:
                k = min(k, self.x_k_max)
            sel = np.sort(order[:k])
            all_indices.append(sel.astype(np.int32))
            all_values.append(v_row[sel])
            ptrs[row + 1] = ptrs[row] + k
            selected_mask[sel] = True
        step = self.x_balance_bias_step
        bias = bias + step * (~selected_mask).astype(np.float64) - step * selected_mask.astype(np.float64)
        self._x_balance_bias[layer_name] = bias
        indices = np.concatenate(all_indices) if all_indices else np.zeros(0, dtype=np.int32)
        values = np.concatenate(all_values) if all_values else np.zeros(0, dtype=np.float32)
        return ptrs, indices, values

    def _update_x_balance_freq(self, layer_name: str, csr: CSR) -> np.ndarray:
        """EMA'd per-dim forward-axis selection frequency -- the mode-1
        (input starvation) diagnostic proxy, per the dense-vs-sparse
        confusion matrix plan. Split out from _accumulate_balance_loss so
        it can run as a pure measurement regardless of which mechanism
        (if any) is active, not just under Arm A. Returns the updated
        frequency array (also stored in self._x_balance_freq)."""
        cols = csr.cols
        selected_mask = np.zeros(cols, dtype=np.float32)
        for row in range(csr.rows):
            start, end = int(csr.ptrs[row]), int(csr.ptrs[row + 1])
            selected_mask[csr.indices[start:end]] = 1.0
        freq = self._x_balance_freq.get(layer_name)
        beta = self.x_balance_freq_beta
        freq = selected_mask if freq is None else beta * freq + (1.0 - beta) * selected_mask
        self._x_balance_freq[layer_name] = freq
        return freq

    def _update_knee_r_target(self, layer_name: str, x2d: np.ndarray) -> float:
        """EMA's _knee_elbow_r's raw per-call measurement into
        self._x_knee_r_target (always tracked as a diagnostic whenever
        this is called) and returns the smoothed value + the configured
        safety margin, clipped to 0.999 -- the value x_r_target_auto
        uses AS x_r_target when enabled. See __init__'s own
        x_r_target_auto docstring."""
        raw = _knee_elbow_r(x2d)
        prev = self._x_knee_r_target.get(layer_name)
        beta = self.x_r_target_auto_beta
        smoothed = raw if prev is None else beta * prev + (1.0 - beta) * raw
        self._x_knee_r_target[layer_name] = smoothed
        return min(0.999, smoothed + self.x_r_target_auto_margin)

    def _accumulate_balance_loss(self, x: Tensor, layer_name: str, freq: np.ndarray) -> None:
        """Arm A -- classic MoE-style auxiliary load-balancing loss
        (differentiable, added to the total training loss; selection
        itself stays plain magnitude top-k, unaffected). fⱼ (EMA'd
        empirical selection frequency, detached bookkeeping, from
        _update_x_balance_freq) times differentiable per-dim energy xⱼ²,
        summed across dims -- pushes the model to shrink activation
        energy on frequently-selected dims relative to others, the same
        f_i*P_i shape as standard MoE router balancing losses.
        Accumulates into self._balance_aux_loss, reset each step() call
        and folded into the final aux_loss there (mirrors
        self._energy_aux_loss's own accumulate-then-combine pattern).
        See __init__'s x_balance_loss_coef docstring."""
        freq_t = Tensor(freq.astype(np.float32))
        term = reduce_sum(power(x, 2) * freq_t, axis=None) * self.x_balance_loss_coef
        self._balance_aux_loss = term if self._balance_aux_loss is None else self._balance_aux_loss + term

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
        self._energy_aux_loss = None  # see _apply_energy's own docstring
        self._balance_aux_loss = None  # see _accumulate_balance_loss's own docstring

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
        # Energy gates a SEPARATE variable, not x_window_t itself --
        # last_debug["x_window_t"] below is the embedding-learning hook's
        # dL/d(raw embedding) Tensor; gradient still flows through the
        # energy gate back to this same object (h*gate_np keeps h's
        # identity), so keeping x_window_t un-reassigned preserves that
        # hook's contract exactly, energy on or off.
        x_window_energized = self._apply_energy(x_window_t, "embed_input")
        x_window_sparse = self._to_sparse(x_window_energized, "input_proj")
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
        combined_normed = self._apply_energy(combined_normed, "state")

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
        attn_pre_o_mem = self._timed_call(
            "attention",
            gaussian_attention,
            q_mem,
            k_phys,
            v_phys,
            centers_mem,
            sigmas_mem,
            num_cpus=self.num_cpus,
            causal=False,
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
            attn_pre_o_content = self._timed_call(
                "attention",
                gaussian_attention,
                q_content,
                k2_phys,
                v2_phys_mem_only,
                centers_content,
                sigmas_content,
                num_cpus=self.num_cpus,
                causal=False,
            )
        else:
            attn_pre_o_content = self._timed_call(
                "attention",
                gaussian_attention,
                q_content,
                k2_phys,
                v2_phys,
                centers_content,
                sigmas_content,
                num_cpus=self.num_cpus,
                causal=False,
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
                        self._timed_call(
                            "attention",
                            gaussian_attention,
                            q,
                            k_phys,
                            v_phys,
                            self.centers,
                            sigmas,
                            num_cpus=self.num_cpus,
                            causal=False,
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
        logits = self._timed_call(
            "lm_head",
            self.lm_head.forward,
            pooled,
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._output_extra_kwargs,
        )

        # Advantage-actor-critic value head, exposed via attribute not
        # return value. See docs/research/toy_tile_recurrence_rmt.rst:critic_head_design.
        self.last_critic_pred = (
            self._timed_call(
                "critic_head",
                self.critic_head.forward,
                pooled,
                learning_rate,
                requires_grad=requires_grad,
                **self.synapse_kwargs,
                **self._output_extra_kwargs,
            )
            if self.use_critic
            else None
        )

        if self._energy_aux_loss is not None:
            aux_loss = self._energy_aux_loss if aux_loss is None else aux_loss + self._energy_aux_loss
        if self._balance_aux_loss is not None:
            aux_loss = self._balance_aux_loss if aux_loss is None else aux_loss + self._balance_aux_loss

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
        attn_pre_o_mem = self._timed_call(
            "attention",
            gaussian_attention,
            q_mem,
            k_phys,
            v_phys,
            centers_mem,
            sigmas_mem,
            num_cpus=self.num_cpus,
            causal=False,
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
            attn_pre_o_content = self._timed_call(
                "attention",
                gaussian_attention,
                q_new,
                k2_phys,
                v2_phys_mem_only,
                centers_content,
                sigmas_content,
                num_cpus=self.num_cpus,
                causal=False,
            )
        else:
            attn_pre_o_content = self._timed_call(
                "attention",
                gaussian_attention,
                q_new,
                k2_phys,
                v2_phys,
                centers_content,
                sigmas_content,
                num_cpus=self.num_cpus,
                causal=False,
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
        logits = self._timed_call(
            "lm_head",
            self.lm_head.forward,
            pooled,
            learning_rate,
            requires_grad=requires_grad,
            **self.synapse_kwargs,
            **self._output_extra_kwargs,
        )

        self.last_critic_pred = (
            self._timed_call(
                "critic_head",
                self.critic_head.forward,
                pooled,
                learning_rate,
                requires_grad=requires_grad,
                **self.synapse_kwargs,
                **self._output_extra_kwargs,
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
