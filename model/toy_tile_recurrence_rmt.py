from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sili.tensor import Tensor, gaussian_attention, exp, reduce_sum, tensor_abs, gather, concat, relu, power
from sili.sparse_rnn import DISLDOLayer

from .toy_recall_models import rmsnorm_tensor


class ToyTileRecurrenceRMT:
    """Faithful reference implementation of Recurrent Memory Transformer's
    actual mechanism (Bulatov et al., "Recurrent Memory Transformer",
    NeurIPS 2022, arXiv:2207.06881) -- built from the SAME sili__new
    primitives (disldo_cls layers, Tensor autograd, gaussian_attention,
    RMSNorm) as ToyTileRecurrenceRealFP4, NOT plain torch. This IS the
    control for the ongoing MQAR investigation (see conversation): same
    engine, same precision handling (FP4 via disldo_cls, or fp32 via a
    plain-float disldo_cls), so a pass/fail result on the exact same
    K=1 MQAR task/harness isolates the ARCHITECTURE question specifically
    -- if this also fails to learn, the problem is the task/embedding/
    harness, not ToyTileRecurrenceRealFP4's own recurrence wiring.
    Plain torch is a deliberately DEFERRED fallback, only worth building
    if THIS itself fails, to separate "the engine is broken" from "even
    a proven architecture fails here somehow" -- building torch first
    would confound engine differences with architecture differences and
    defeat the point of a control (direct instruction).

    Core mechanism, matching RMT exactly: `num_memory_slots` dedicated
    memory-token positions are CONCATENATED into the same attention
    window as the real content tile positions (never additively merged
    -- see ToyTileRecurrenceRealFP4's own "Known differences" docstring,
    items 1 and 4), so gaussian_attention processes memory and content
    together via ordinary self-attention, exactly like RMT's own memory
    tokens being literally re-inserted into the input/output sequence.
    The memory tokens' own output positions after this step become next
    step's memory, carried purely by being re-inserted into the window
    on the next call -- no separate gate/combine mechanism at all (RMT
    itself has none; the memory update is handled entirely by ordinary
    attention + residual, same as any other token position)."""

    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int,
                 num_tiles: int, num_memory_slots: int, max_weights: int,
                 num_cpus: int = 2, rms_eps: float = 1e-6, disldo_cls=DISLDOLayer,
                 dense: bool = False, clip_range: float = 6.0,
                 l1_sparsity_coef: float = 0.0,
                 magnitude_clip_penalty_coef: float = 0.0,
                 min_sigma: float = 1e-3,
                 synapse_kwargs: Optional[dict] = None,
                 scale_rank: int = 1,
                 additive_rank: int = 0,
                 dynamic_rank_control: bool = False,
                 use_critic: bool = False,
                 rng: Optional[np.random.Generator] = None):
        """num_memory_slots: RMT's own paper uses a small handful of
        memory tokens (their experiments: as few as 1-16 depending on
        task) -- default kept small here to match, not tuned against
        this specific task yet.

        Everything else mirrors ToyTileRecurrenceRealFP4's own
        conventions exactly (embed_width/column_neurons/state_width,
        disldo_cls/dense/l1_sparsity_coef, per-layer independent rng
        seeding) so a comparison between the two isolates the
        architecture question, not incidental convention differences.

        use_critic: adds a critic_head (same shape as lm_head, embed_width
        -> vocab_size) predicting the per-vocab-neuron squared error the
        actor's own logits will incur -- a real advantage-actor-critic
        value head, not a shortcut for the (exactly known) true loss
        itself. Default off, so every existing caller is byte-identical
        (same first 6 layer-construction RNG draws either way -- the
        critic's own seed is drawn separately, after).

        magnitude_clip_penalty_coef (task #303/#304): a plain hard clip
        on q/k/v/attn/combined_new gives ZERO backward gradient past the
        boundary (np.clip's own derivative is 0 there), so a layer whose
        output keeps getting clipped never learns to stop producing that
        magnitude in the first place -- direct instruction, confirmed via
        real diagnostics: v_proj's output ran unclipped into the
        1000s-2000s for hundreds of steps (masked downstream by the
        existing hard clips on attn/combined_new) before an unscaled dot
        product inside gaussian_attention finally overflowed to NaN. This
        adds a differentiable hinge-squared penalty
        (coef*mean(relu(|x|-clip_range)**2)) on q/k/v/attn/combined_new,
        so the layers themselves get gradient pressure to shrink whenever
        they exceed clip_range, on top of (not instead of) the existing
        hard clip on the VALUES. Default off (0.0), matching
        l1_sparsity_coef's own opt-in convention.

        min_sigma (task #305): gaussian_attention's Gaussian bias term is
        1/(2*sigma**2) -- as sigma trains toward 0 (exactly what learning
        to attend sharply to one position looks like), that term can hit
        Inf, and 0*Inf=NaN if a key lands exactly on the query's center.
        Always-on floor (matching rms_eps's own always-on convention, not
        l1_sparsity_coef's opt-in-off one), applied to sigmas.data right
        after exp(log_sigmas). 1e-3 is deliberately generous: at that
        floor, one integer position away from center already gives
        exp(-1/(2*1e-3**2)) = exp(-500000) -- functionally a one-hot --
        so this can't meaningfully constrain how sharply the model can
        attend, only prevent the literal division-by-near-zero case."""
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
        self.num_cpus = num_cpus
        # min_decay_frac/max_abs_delta/max_ci passthrough (see
        # sili.sparse_rnn.DISLDOLayer.forward's own docstring) -- lets a
        # caller toggle e.g. max_abs_delta=1e30 (effectively off) for a
        # real ablation on this actual model without a C++ rebuild or
        # touching every one of this step()'s own 6 disldo_cls.forward()
        # call sites by hand.
        self.synapse_kwargs = synapse_kwargs or {}

        state_width = self.state_width
        if rng is None:
            rng = np.random.default_rng()
        n_layer_seeds = 6  # input_proj, q, k, v, o_proj, lm_head
        layer_seeds = iter(int(s) for s in rng.integers(0, 2**31 - 1, size=n_layer_seeds))
        dense_kwargs = {"dense": True} if dense else {}
        # Conditionally forwarded, matching dense_kwargs' own pattern --
        # only DISLDOLayer/DISLDOLayerDeterministic/DISLDOLayer8-family
        # (and TrueMultiDigitLayer, which forwards it into each digit)
        # accept scale_rank at all; unconditionally splatting it would
        # TypeError any disldo_cls that doesn't (e.g. DISLDOLayer32).
        rank_kwargs = {"scale_rank": scale_rank} if scale_rank != 1 else {}
        # AQRS additive branch (task #280 -- re-validates the fp8 MQAR
        # "input-independent collapse" case, see AQRS_DESIGN.md Theorem
        # 3/4): same conditional-forwarding convention as rank_kwargs
        # above, only DISLDOLayer/DISLDOLayerDeterministic/DISLDOLayer8
        # accept additive_rank at all.
        additive_kwargs = {"additive_rank": additive_rank} if additive_rank != 0 else {}
        # Same conditional-forwarding convention: only
        # DISLDOLayer/DISLDOLayerDeterministic/DISLDOLayer8 accept this
        # kwarg. Requires additive_rank>=1 to have any effect -- gamma
        # tracking's own neurogenesis trigger can't fire from rank 0
        # (see sili__new sparse_rnn.py _activate_gamma_tracking).
        dynamic_kwargs = {"dynamic_rank_control": True} if dynamic_rank_control else {}
        layer_kwargs = {**dense_kwargs, **rank_kwargs, **additive_kwargs, **dynamic_kwargs}

        self.input_proj = disldo_cls(embed_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.q_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.k_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.v_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.o_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.lm_head = disldo_cls(embed_width, vocab_size, max_weights, num_cpus,
                                  rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)

        # Drawn from `rng` AFTER the fixed size=6 draw above completes, so
        # the first 6 layers' seeds are byte-identical whether or not
        # use_critic is set -- existing callers (use_critic defaults False)
        # see zero behavior change.
        self.use_critic = use_critic
        self.critic_head = None
        if use_critic:
            critic_seed = int(rng.integers(0, 2**31 - 1))
            self.critic_head = disldo_cls(embed_width, vocab_size, max_weights, num_cpus,
                                          rng=np.random.default_rng(critic_seed), **layer_kwargs)

        # Separate RMSNorm gains for memory vs content tokens -- unlike
        # ToyTileRecurrenceRealFP4 (which reuses one input_ln for both
        # sides of an additive combine), memory and content are now
        # genuinely different KINDS of token (never summed together),
        # so giving them their own learned scale is the more faithful
        # choice, matching how RMT's own memory tokens get their own
        # learned embeddings.
        self.input_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.memory_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.state_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.centers = Tensor(np.array([i + 0.5 for i in range(self.total_slots)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(self.total_slots, dtype=np.float32))

    def _named_real_layers(self):
        """(name, layer) for every real disldo_cls weight layer, INCLUDING
        critic_head when use_critic is set -- the single source of truth
        every per-layer maintenance pass (rank control, overflow guard,
        orthogonality penalty, rank reporting, magnitude rescale) iterates,
        so critic_head automatically gets the same treatment as every
        other layer rather than being special-cased at each call site."""
        layers = [("input_proj", self.input_proj), ("q_proj", self.q_proj),
                  ("k_proj", self.k_proj), ("v_proj", self.v_proj),
                  ("o_proj", self.o_proj), ("lm_head", self.lm_head)]
        if self.use_critic:
            layers.append(("critic_head", self.critic_head))
        return layers

    def _real_layers(self):
        return [layer for _name, layer in self._named_real_layers()]

    def parameters_for_optimizer(self) -> List[Tensor]:
        return [self.input_ln, self.memory_ln, self.state_ln, self.centers, self.log_sigmas]

    def magnitude_rescale_output(self, target: float, correction_rate: float,
                                 scale_invariant: bool = False) -> None:
        """Apply magnitude_rescale_output to every real disldo_cls weight
        layer (input_proj/q/k/v/o_proj/lm_head) -- skips a layer whose
        backend has no scale concept at all (e.g. the fp32 DISLDOLayerV
        control), matching _SparseLayerBase.magnitude_rescale_output's
        own guard convention. Meant to be called periodically from the
        training loop, not every step (see delta_csr_types.hpp's own
        magnitude_rescale_output docstring for its intended cadence)."""
        for layer in self._real_layers():
            if hasattr(layer, "_c") and hasattr(layer._c, "magnitude_rescale_output"):
                layer.magnitude_rescale_output(target, correction_rate, scale_invariant)
            elif hasattr(layer, "magnitude_rescale_output") and hasattr(layer, "digits"):
                layer.magnitude_rescale_output(target, correction_rate, scale_invariant)

    def apply_dynamic_rank_control(self, tau_death: float = 0.05, tau_active: float = 0.3,
                                   theta: float = 1e-4, seed_scale: float = 0.05,
                                   grace_period_steps: int = 50) -> dict:
        """Runs AQRS Theorem 10 dynamic rank control (task #292, see
        sili__new's delta_csr_types.hpp/DISLDOLayer.apply_dynamic_rank_
        control) on every real disldo_cls weight layer independently --
        same iteration pattern as magnitude_rescale_output above, same
        "skip a layer whose backend has no scale concept" guard (fp32
        DISLDOLayerV has no scale_rank/additive_rank at all). Meant to be
        called once per training step, after backward -- the EMA state
        driving the triggers is updated automatically inside each layer's
        own backward call, so calling this less often just means the
        triggers get evaluated on stale-but-still-accumulating EMA state,
        not that anything breaks; calling it MORE often than once/step
        has no effect since nothing new has been computed between calls.

        Returns {layer_name: mutated_bool} for every real layer -- lets a
        caller log/count real rank-mutation events per layer without
        needing to know the same layer-name tuple again itself.
        """
        results = {}
        for name, layer in self._named_real_layers():
            if hasattr(layer, "apply_dynamic_rank_control"):
                results[name] = layer.apply_dynamic_rank_control(
                    tau_death, tau_active, theta, seed_scale, grace_period_steps)
        return results

    def apply_scale_overflow_guard(self, clip: float = 200.0, near: float = 20.0,
                                   coef: float = 0.1) -> None:
        """AQRS scale/additive channel numerical-safety pass (task #295
        follow-up, see sili__new's sparse_rnn.py DISLDOLayer.apply_
        scale_overflow_guard) on every real disldo_cls weight layer --
        same iteration pattern/guard convention as apply_dynamic_rank_
        control above. Root cause this fixes: raising scale_rank_max/
        additive_rank_max past the old hardcoded 4 let a real fp8 MQAR
        curriculum run's per-channel value_scale_k/output_scale_k grow
        unbounded, overflowing the combined scale envelope S(row,col)
        in the forward pass and NaN-collapsing the whole run (see
        conversation) -- clip is deliberately NOT a plain hard clip
        (would give zero/wrong backward signal once a channel is
        pinned at the boundary); see _overflow_guard_array's own
        docstring for the full auto-correcting-shrink derivation. Meant
        to be called once per training step, any time after backward
        (independent of apply_dynamic_rank_control -- that mutates
        RANK, this corrects VALUES)."""
        for layer in self._real_layers():
            if hasattr(layer, "apply_scale_overflow_guard"):
                layer.apply_scale_overflow_guard(clip, near, coef)

    def apply_channel_orthogonality_penalty(self, coef: float = 0.01) -> None:
        """AQRS channel-diversity pass (see sili__new's sparse_rnn.py
        DISLDOLayer.apply_channel_orthogonality_penalty) on every real
        disldo_cls weight layer -- same iteration pattern as
        apply_scale_overflow_guard above. Real problem this addresses:
        nothing else in the AQRS design stops two rank channels from
        converging to duplicate directions during training -- the
        neurogenesis health check is purely magnitude-based (a
        redundant channel still shows real gradient/magnitude), and
        l1_sparsity_coef only sees the SUMMED output after every
        channel's already combined. Chosen over residual-targeted
        growth (direct instruction): that only fixes it at init time,
        this is an ongoing per-step force that keeps channels diverse
        throughout training, and needed no sili__new kernel changes at
        all (see _orthogonality_penalty_array's own docstring). Meant
        to be called once per training step, independent of
        apply_scale_overflow_guard/apply_dynamic_rank_control --
        diversity, numerical safety, and rank mutation are three
        separate concerns."""
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

    def _l1_sparsity_split(self, layer, input_t: Tensor, lr: float, coef: float) -> Tensor:
        """Exact port of ToyTileRecurrenceRealFP4's own helper -- see its
        l1_sparsity_coef docstring for the full split-backward rationale."""
        out_aux = layer.forward(input_t, lr, damp_by_importance=False, **self.synapse_kwargs)
        n = float(np.asarray(out_aux.data).size)
        return reduce_sum(tensor_abs(out_aux)) * (coef / n)

    def _magnitude_clip_penalty(self, out_tensor: Tensor) -> Tensor:
        """See magnitude_clip_penalty_coef's own __init__ docstring.
        Hinge-squared (not a plain L2 penalty like toy_tile_precision_
        models.py's own magnitude_penalty_coef) -- only pushes back once
        |x| actually exceeds clip_range, so it doesn't fight ordinary
        in-range activity the way a uniform L2 term would."""
        excess = relu(tensor_abs(out_tensor) - self.clip_range)
        n = float(np.asarray(out_tensor.data).size)
        return reduce_sum(power(excess, 2)) * (self.magnitude_clip_penalty_coef / n)

    def step(self, x_window: np.ndarray, memory_prev: np.ndarray,
             learning_rate: float) -> Tuple[np.ndarray, Tensor, Optional[Tensor]]:
        """x_window: [num_tiles, embed_width] -- same convention as
        ToyTileRecurrenceRealFP4 (a real per-tile embedding, zeros for
        "nothing here yet"). memory_prev: [num_memory_slots, state_width]
        -- genuinely dedicated memory, DETACHED (no BPTT across steps,
        matching this whole project's convention), unlike
        ToyTileRecurrenceRealFP4's ambiguous per-window-slot rolling
        state (see its own docstring item 4). Returns (memory_new numpy
        [num_memory_slots, state_width], logits Tensor [num_tiles,
        vocab_size], aux_loss)."""
        sw = self.state_width
        n_mem, n_content = self.num_memory_slots, self.num_tiles

        x_window_t = Tensor(x_window.astype(np.float32))
        x_wide = self.input_proj.forward(x_window_t, learning_rate, **self.synapse_kwargs)  # [n_content, sw]
        x_normed = rmsnorm_tensor(x_wide, self.input_ln, self.rms_eps)

        memory_prev_t = Tensor(memory_prev.astype(np.float32))
        memory_normed = rmsnorm_tensor(memory_prev_t, self.memory_ln, self.rms_eps)

        combined_normed = concat([memory_normed, x_normed], axis=0)          # [total_slots, sw]

        q = self.q_proj.forward(combined_normed, learning_rate, **self.synapse_kwargs)
        k = self.k_proj.forward(combined_normed, learning_rate, **self.synapse_kwargs)
        v = self.v_proj.forward(combined_normed, learning_rate, **self.synapse_kwargs)

        aux_loss = None

        def _accumulate_penalty(t: Tensor) -> None:
            nonlocal aux_loss
            if self.magnitude_clip_penalty_coef > 0.0:
                term = self._magnitude_clip_penalty(t)
                aux_loss = term if aux_loss is None else aux_loss + term

        # Penalty MUST be built from the PRE-clip value -- _magnitude_
        # clip_penalty's own intermediate tensors (tensor_abs(t)-
        # clip_range, relu(...), power(...)) each snapshot their own
        # .data at construction time, so building the penalty graph here
        # (before the in-place clip below overwrites t.data) captures the
        # real excess. Building it AFTER the clip would read the already-
        # clipped value, where |x|-clip_range is never positive -- the
        # penalty would silently never fire.
        _accumulate_penalty(q)
        _accumulate_penalty(k)
        _accumulate_penalty(v)

        # Clip q/k/v BEFORE they enter gaussian_attention (task #303/#304,
        # direct instruction): previously only attn/combined_new were
        # clipped, AFTER attention's own internal dot-product/exp math had
        # already run on unbounded q/k/v -- confirmed via real diagnostics
        # that v_proj's output alone reached 1000s-2000s magnitude for
        # hundreds of steps, invisible externally because the existing
        # downstream clips masked it, until an unscaled dot product
        # inside gaussian_attention finally overflowed to NaN.
        q.data = np.clip(q.data, -self.clip_range, self.clip_range)
        k.data = np.clip(k.data, -self.clip_range, self.clip_range)
        v.data = np.clip(v.data, -self.clip_range, self.clip_range)
        sigmas = exp(self.log_sigmas)
        # Floor BEFORE gaussian_attention uses it -- see min_sigma's own
        # __init__ docstring for the 1/(2*sigma**2)->Inf mechanism this
        # closes. Same in-place-mutation convention as q/k/v/attn/
        # combined_new's own clips above (forward and any later backward
        # read both see the same, already-floored value).
        sigmas.data = np.maximum(sigmas.data, self.min_sigma)
        attn_pre_o = gaussian_attention(q, k, v, self.centers, sigmas,
                                        num_cpus=self.num_cpus, causal=False)
        attn = self.o_proj.forward(attn_pre_o, learning_rate, **self.synapse_kwargs)
        _accumulate_penalty(attn)  # pre-clip, same reasoning as q/k/v above
        attn.data = np.clip(attn.data, -self.clip_range, self.clip_range)

        # Debug instrumentation (task #303): cheap reference-only capture
        # (no copies) of every stage between the input embedding and the
        # readout, for bisecting exactly where a NaN/Inf first appears in
        # the forward chain -- np.clip does NOT sanitize NaN (clip(nan)==
        # nan), so the clip calls in this function are not themselves
        # proof any given stage is finite.
        self.last_debug = {
            "x_wide": x_wide.data, "q": q.data, "k": k.data, "v": v.data,
            "attn_pre_o": attn_pre_o.data, "attn_post_o": attn.data,
            "sigmas": sigmas.data, "log_sigmas": self.log_sigmas.data,
        }

        if self.l1_sparsity_coef > 0.0:
            l1_terms = [
                self._l1_sparsity_split(self.input_proj, x_window_t, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.q_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.k_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.v_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.o_proj, gaussian_attention(
                    q, k, v, self.centers, sigmas, num_cpus=self.num_cpus, causal=False),
                    learning_rate, self.l1_sparsity_coef),
            ]
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term

        # Residual against the RAW (pre-RMSNorm) memory/content values,
        # matching ToyTileRecurrenceRealFP4's own convention (residual
        # from raw M_prev, not the normed qkv_source).
        raw_combined = concat([memory_prev_t, x_wide], axis=0)
        pre_norm_combined = raw_combined + attn
        combined_new = rmsnorm_tensor(pre_norm_combined, self.state_ln, self.rms_eps)
        _accumulate_penalty(combined_new)  # pre-clip, same reasoning as above
        combined_new.data = np.clip(combined_new.data, -self.clip_range, self.clip_range)
        self.last_debug["raw_combined"] = raw_combined.data
        self.last_debug["pre_norm_combined"] = pre_norm_combined.data
        self.last_debug["combined_new"] = combined_new.data

        # Split back into memory (next step's carry, plain numpy -- no
        # BPTT across steps, matching this project's convention) and
        # content (stays differentiable, feeds the readout below) via
        # gather+reshape (no slicing op exists on Tensor -- see
        # cross_entropy_sum's own docstring for why gather is this
        # codebase's established workaround).
        memory_new = combined_new.data[:n_mem].copy()
        content_flat_idx = [(n_mem + t) * sw + c for t in range(n_content) for c in range(sw)]
        content_out = gather(combined_new, content_flat_idx).reshape((n_content, sw))

        pooled = content_out.reshape((n_content, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)
        self.last_debug["pooled"] = pooled.data
        if self.l1_sparsity_coef > 0.0:
            lm_l1 = self._l1_sparsity_split(self.lm_head, pooled, learning_rate, self.l1_sparsity_coef)
            aux_loss = lm_l1 if aux_loss is None else aux_loss + lm_l1
        logits = self.lm_head.forward(pooled, learning_rate, **self.synapse_kwargs)

        # Advantage-actor-critic value head (opt-in, see __init__'s
        # use_critic docstring): exposed via an attribute rather than a
        # 4th return value, since step()'s 3-tuple return is unpacked by
        # dozens of existing call sites across both repos and a return-
        # arity change would break every one of them. Caller (e.g.
        # scripts/train_mqar_curriculum.py) reads model.last_critic_pred
        # right after this call.
        self.last_critic_pred = (
            self.critic_head.forward(pooled, learning_rate, **self.synapse_kwargs)
            if self.use_critic else None)

        return memory_new, logits, aux_loss
