from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from sili.tensor import Tensor, gaussian_attention, exp, reduce_sum, tensor_abs, gather, concat, relu, power
from sili.sparse_rnn import DISLDOLayer, CSR

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

    # The 5 layers dy_r_target/dy_sparsity_p/input_sparsity_p/
    # wide_max_weights all affect (never lm_head/critic_head) -- single
    # source of truth for the per-layer dict keys used below (task #372).
    _WIDE_LAYER_NAMES = ("input_proj", "q_proj", "k_proj", "v_proj", "o_proj")

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
                 recurrent_only_output: bool = False,
                 input_sparsity_p: Optional[float] = None,
                 dy_sparsity_p: Optional[float] = None,
                 wide_max_weights: Optional[int] = None,
                 output_dy_sparsity_p: Optional[float] = None,
                 dy_r_target: Optional[float] = None,
                 dy_k_min: int = 0,
                 dy_k_max: Optional[int] = None,
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
        attend, only prevent the literal division-by-near-zero case.

        recurrent_only_output (RNN validation ablation, direct
        instruction): when True, content-row queries (this step's
        readout) are masked to attend ONLY memory-row keys/values --
        never their own or another content row's. Memory-row queries
        stay unrestricted (read the full window, i.e. in_proj->recurrent
        "write" stays allowed). The net effect: the ATTENTION-derived
        portion of the output at step t can only carry information that
        was already written into memory at some step <t.

        Does NOT zero the direct x_wide residual skip into content_out
        (task #315 follow-up, direct instruction, post-validation): an
        earlier version of this ablation ALSO zeroed that residual, to
        fully isolate memory's contribution for a distance-sweep
        verification (confirmed genuine cross-detach recurrent
        persistence -- accuracy well above chance at distances requiring
        the value to survive step() boundaries no same-step gradient
        path can reach across). That isolation already did its job; the
        residual doesn't need to stay zeroed going forward. x_wide is
        the QUERY token's own embedding, which never correlates with
        MQAR's correct recall value (the task is information-
        theoretically undecidable from the query token alone), so
        leaving the residual live can't reintroduce a "cheat" path for
        this task -- it just avoids needlessly handicapping the model
        with a residual connection removed.

        Default off -- existing callers see zero behavior change (same
        6+1 layer-seed draws either way, this doesn't touch construction
        at all, only step()).

        input_sparsity_p/dy_sparsity_p/wide_max_weights (sparsity plan
        Phase 6, task #335): real values, not a bool toggle -- presence
        (non-None) IS the toggle, matching this whole plan's own
        convention (see sili.sparse_rnn.DISLDOLayer.forward's identical
        dy_sparsity_p). All three default to None, meaning every
        existing caller gets byte-identical behavior (no CSR anywhere,
        max_weights unchanged for every layer).

        Only input_proj/q_proj/k_proj/v_proj/o_proj are affected --
        lm_head/critic_head stay fully dense/unwidened the whole time
        (their own budget stays `max_weights`, never wide_max_weights;
        their forward() calls never see a CSR input). Widening those
        two isn't part of this plan: they read `pooled` (already
        column-averaged down from state_width back to embed_width), not
        one of the 5 layers whose INPUT width doubles with embed_width.

        input_sparsity_p: density fraction for the 5 affected layers'
        forward INPUT (reuses CSR.from_dense's own `p` convention, see
        _to_sparse below). A layer whose input width just doubled needs
        p~=0.5 to keep total compute at ~2x instead of 4x -- the whole
        point of pairing width-doubling with input sparsification in
        this plan.

        dy_sparsity_p: density fraction for those same 5 layers'
        backward GRADIENT -- a genuinely separate axis from
        input_sparsity_p (see DISLDOLayer.forward's own docstring: forward-
        input-sparsity and backward-gradient-sparsity are independent
        parameters on the underlying sisldo_forward/disldo_backward_
        sparse_grad C++ functions). If left None while input_sparsity_p
        is set, defaults internally to input_sparsity_p (matches the
        original requirement that dy gets the same treatment as the
        input by default); an explicit value overrides independently.

        wide_max_weights: per-layer synapse budget override for the 5
        affected layers only. None (default) means they share the same
        `max_weights` as every other layer, today's exact behavior. Set
        to an int (e.g. 2048, the quadrupled default this plan's own
        curriculum script uses) to give them a larger budget while
        lm_head/critic_head stay at the original `max_weights` --
        input+backprop sparsity means compute no longer scales with the
        full stored budget every step, so the extra memory is affordable
        (direct instruction).

        output_dy_sparsity_p (direct instruction, following the
        step_cached graded-schedule speed work): density fraction for
        lm_head/critic_head's own backward GRADIENT only -- genuinely
        separate axis from input_sparsity_p/dy_sparsity_p above, and
        NOT threaded through _to_sparse (lm_head/critic_head read
        `pooled`, a real dense column-averaged readout with no
        structural sparsity -- unlike the 5 affected layers' inputs,
        there's no free-lunch argument for sparsifying THIS input).
        The gradient side is different: `_backward_with_critic`
        computes `g_logits[row] = (1+advantage) * (probs - onehot)`
        where `probs` is a softmax over vocab_size -- technically dense
        (softmax never hits exact 0) but concentrates hard onto a few
        classes as the model gets confident, so most of `probs-onehot`
        is near-zero in practice. `dy_sparsity_p`'s top-k-by-magnitude
        selection is exactly suited to this shape. None (default):
        byte-identical to today's exact dense backward for both heads.

        dy_r_target/dy_k_min/dy_k_max (task #367, priority 1 -- see
        JOURNAL.md's "nucleus/energy-threshold top-k math" design note):
        nucleus/energy-threshold grad sparsification for the SAME 5
        layers dy_sparsity_p covers, TAKES PRIORITY over dy_sparsity_p
        when both are set (mirrors DISLDOLayer.forward's own priority
        chain exactly). Unlike dy_sparsity_p's fixed fraction, k is a
        CONSEQUENCE of dy_r_target and each step's actual gradient
        energy -- dy_r_target is the initial/current captured-energy
        ratio target (mutable after construction via
        apply_amortized_dy_r_target_control below, a closed-loop
        controller against MEASURED steps/sec, not a guessed constant --
        the originally-sketched analytic kbar-from-cost-ratio formula
        was found NOT to hold: forward:backward cost ratio measured
        2.8x-12x depending on width, not the recalled ~10x, so no
        single hardcoded ratio would be right at every scale). dy_k_min/
        dy_k_max are the hardware density floor/ceiling (see
        _nucleus_top_k_csr's own docstring, sili__new). None (default):
        byte-identical to today's exact behavior, same as every other
        sparsity kwarg here.

        Task #372: this scalar is the INITIAL value applied uniformly to
        all 5 wide layers -- internally stored as self.dy_r_target, a
        per-layer dict (name -> r_target), same pattern as
        self._l2_decay_factor. A model-level knob wastes each layer's
        own economics (different fwd:bwd cost ratios per layer, task
        #368's own measurement), so per-layer state is the natural
        representation even though this constructor arg still only
        offers one shared starting point; task #374's per-layer surprise
        loop will make each entry diverge independently over training."""
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
        # min_decay_frac/max_abs_delta/max_ci passthrough (see
        # sili.sparse_rnn.DISLDOLayer.forward's own docstring) -- lets a
        # caller toggle e.g. max_abs_delta=1e30 (effectively off) for a
        # real ablation on this actual model without a C++ rebuild or
        # touching every one of this step()'s own 6 disldo_cls.forward()
        # call sites by hand.
        self.synapse_kwargs = synapse_kwargs or {}

        # Phase 6 (task #335): see __init__'s own input_sparsity_p/
        # dy_sparsity_p/wide_max_weights docstring above.
        self.input_sparsity_p = input_sparsity_p
        self.dy_sparsity_p = (dy_sparsity_p if dy_sparsity_p is not None
                              else input_sparsity_p)
        # dy_r_target (task #367, per-layer dict since task #372): mutable
        # after construction via apply_amortized_dy_r_target_control below
        # -- see __init__'s own dy_r_target docstring. dy_k_min/dy_k_max
        # stay plain scalars (static hardware floor/ceiling, same value
        # for every layer -- no closed-loop control needed for those).
        self.dy_r_target: dict = {name: dy_r_target for name in self._WIDE_LAYER_NAMES}
        self.dy_k_min = dy_k_min
        self.dy_k_max = dy_k_max
        # _wide_extra_kwargs (below) is a live METHOD, keyed by layer name,
        # now that dy_r_target is a per-layer dict mutable post-construction
        # via apply_amortized_dy_r_target_control -- every one of step()'s
        # many self._wide_extra_kwargs(name) call sites reads it fresh each
        # time, so updating self.dy_r_target between steps takes effect
        # immediately with no separate cache-invalidation step needed.
        # Separate axis, lm_head/critic_head only -- see __init__'s own
        # output_dy_sparsity_p docstring above.
        self.output_dy_sparsity_p = output_dy_sparsity_p
        self._output_extra_kwargs = ({"dy_sparsity_p": output_dy_sparsity_p}
                                     if output_dy_sparsity_p is not None else {})

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
        # Phase 6 (task #335): per-layer budget override for ONLY the 5
        # affected layers (input_proj/q/k/v/o_proj) -- lm_head/critic_head
        # below deliberately keep using the plain `max_weights` positional
        # arg, never wide_max_weights. None (default) means
        # wide_max_weights_ == max_weights, i.e. today's exact behavior.
        wide_max_weights_ = wide_max_weights if wide_max_weights is not None else max_weights

        self.input_proj = disldo_cls(embed_width, state_width, wide_max_weights_, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.q_proj = disldo_cls(state_width, state_width, wide_max_weights_, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.k_proj = disldo_cls(state_width, state_width, wide_max_weights_, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.v_proj = disldo_cls(state_width, state_width, wide_max_weights_, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.o_proj = disldo_cls(state_width, state_width, wide_max_weights_, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)
        self.lm_head = disldo_cls(embed_width, vocab_size, max_weights, num_cpus,
                                  rng=np.random.default_rng(next(layer_seeds)), **layer_kwargs)

        # Drawn from `rng` AFTER the fixed size=6 draw above completes, so
        # the first 6 layers' seeds are byte-identical whether or not
        # use_critic is set -- existing callers (use_critic defaults False)
        # see zero behavior change.
        self.use_critic = use_critic
        self.critic_head = None
        # Persistent per-layer decay_factor state for apply_amortized_l2_decay
        # (closed-loop, corrected toward each layer's fan-in-normalized target
        # from its own measured rms -- see that method's own docstring).
        # Lazily populated (name -> factor), default 1.0 (no decay) until a
        # layer's first full cycle completes and supplies real data.
        self._l2_decay_factor: dict = {}
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

        # Interleaved position layout + widened cold-start sigma (direct
        # instruction, from real diagnostics run during the RNN-validation
        # ablation -- see conversation): the OLD layout put all n_mem
        # memory rows at physical positions [0, n_mem) and all content
        # rows at [n_mem, total_slots), i.e. two separate contiguous
        # blocks -- so memory sat maximally far (~total_slots) from the
        # live input/output position (always the LAST content row,
        # _build_tile_window's own convention). Combined with the OLD
        # sigma=1 cold-start default, this is a real, confirmed bug: the
        # Gaussian bias exp(-diff**2/(2*sigma**2)) at diff~16.5 is
        # ~1e-59, below float32's smallest representable value -- an
        # EXACT zero attention weight, and since d(score)/d(sigma) is
        # itself scaled by that same zero weight, ALSO exactly zero
        # backward gradient. A genuine dead end the model could never
        # train its way out of, symmetric in both directions (memory
        # could never read fresh input, and the output position could
        # never read memory back) -- silently unexercised until now
        # because every real curriculum run so far stayed in the K<=4
        # in-context phase, solvable via content-content attention alone
        # (nearby array indices, no large-distance underflow), never
        # actually forcing reliance on memory.
        #
        # Fix has two parts:
        #  1) Spread the n_mem memory slots evenly across the position
        #     range instead of clustering them at one end, so no content
        #     tile is structurally privileged (nearest to memory) over
        #     any other -- every tile should have a real shot at
        #     supplying/receiving recurrent detail, not just whichever
        #     one happens to sit next to memory's block.
        #  2) Widen the cold-start sigma so distance alone can't
        #     underflow the attention weight to a hard, gradient-dead
        #     zero -- total_slots/4 keeps even the single farthest
        #     possible pair (distance ~= total_slots) at a representable,
        #     if weak, weight (exp(-4**2/2)~=3e-4), so real training
        #     signal can reach every position from the start and sharpen
        #     (or widen further) from there as the data actually wants.
        #
        # Implementation: `centers` values (which position each LOGICAL
        # row -- 0..n_mem-1 memory, n_mem..total_slots-1 content -- is
        # labeled as occupying) are reassigned to the interleaved
        # physical layout; the row DATA itself stays in logical order
        # (no need to move memory_normed/x_normed's own concat order).
        # What genuinely must move is K/V's ARRAY order when they're fed
        # into gaussian_attention, since the C++ kernel has no separate
        # per-key position input -- it uses the key's raw array index j
        # directly as its position (see attention.hpp's
        # gaussian_attention_forward: `diff = float(j) - c`). See
        # step()'s own comment at the k_phys/v_phys gather for the other
        # half of this. NOTE: this reordering assumes causal=False
        # (matches this model's only usage) -- if causal attention were
        # ever added here, Q and K would need to share the SAME index
        # space for the j>t mask to mean anything, which this
        # Q-stays-logical/K-goes-physical split deliberately breaks.
        n_mem, n_content = self.num_memory_slots, self.num_tiles
        if n_mem > 0:
            raw_positions = [int(round((m + 0.5) * self.total_slots / n_mem)) for m in range(n_mem)]
            used = set()
            mem_phys = []
            for p in raw_positions:
                while p in used:
                    p = (p + 1) % self.total_slots
                used.add(p)
                mem_phys.append(p)
            mem_phys = sorted(mem_phys)
        else:
            mem_phys = []
        mem_phys_set = set(mem_phys)
        content_phys = [p for p in range(self.total_slots) if p not in mem_phys_set]
        self._mem_phys = mem_phys
        self._content_phys = content_phys

        # Gather indices to reorder a LOGICAL-order [total_slots, sw]
        # tensor (k or v) into PHYSICAL-order (array index == true
        # interleaved position) -- precomputed once since state_width is
        # fixed for the life of the model, reused every step() call.
        phys_to_log = [0] * self.total_slots
        for i, p in enumerate(mem_phys):
            phys_to_log[p] = i
        for t, p in enumerate(content_phys):
            phys_to_log[p] = n_mem + t
        self._kv_phys_gather_idx = [
            phys_to_log[p] * state_width + c
            for p in range(self.total_slots) for c in range(state_width)]

        # Value-mask for the recurrent_only_output ablation: 1.0 at
        # memory physical rows, 0.0 at content physical rows -- applied
        # to v_phys (elementwise, differentiable) so a content query's
        # attention WEIGHTS are still computed from the correct,
        # genuinely-interleaved distances (keys stay the real, full
        # array), but only memory rows can ever contribute actual VALUE
        # to the output. Deliberately NOT implemented by slicing K/V down
        # to memory-only rows -- that would collapse their array index
        # back to a local 0..n_mem-1 range, silently reintroducing the
        # exact clustered-position underflow this fix just closed.
        mem_only_mask = np.zeros((self.total_slots, state_width), dtype=np.float32)
        for p in mem_phys:
            mem_only_mask[p, :] = 1.0
        self._mem_only_value_mask = Tensor(mem_only_mask)

        self.centers = Tensor(np.array(
            [mem_phys[i] + 0.5 for i in range(n_mem)] +
            [content_phys[t] + 0.5 for t in range(n_content)], dtype=np.float32))
        sigma_init = max(self.total_slots / 4.0, 1.0)
        self.log_sigmas = Tensor(np.full(self.total_slots, np.log(sigma_init), dtype=np.float32))

        # step_cached's own precomputed constant index lists (flat,
        # matching gather()'s existing [row*sw+col] convention used
        # throughout step() above) -- fixed for the model's lifetime,
        # built once here rather than reconstructed every call.
        self._mem_idx_step = [m * state_width + c for m in range(n_mem) for c in range(state_width)]
        self._new_content_idx_step = [n_mem * state_width + c for c in range(state_width)]
        self._mem_center_idx = list(range(n_mem))
        self._newest_content_idx = [n_mem + n_content - 1]

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

    def _wide_extra_kwargs(self, layer_name: str) -> dict:
        """Extra kwargs threaded into ONE of the 5 affected layers'
        forward() calls (input_proj/q/k/v/o_proj), never lm_head/
        critic_head's -- layer_name must be one of _WIDE_LAYER_NAMES.
        Live (not cached at construction) because self.dy_r_target[name]
        is mutable post-construction via
        apply_amortized_dy_r_target_control below.

        Per-layer since task #372 (was a single shared @property before
        -- a model-level knob wastes each layer's own economics, task
        #368's own measurement found fwd:bwd cost ratio varies 2.8x-12x
        across layers/widths). dy_r_target takes PRIORITY over
        dy_sparsity_p when both end up set -- mirrors DISLDOLayer.
        forward's own priority chain exactly (task #367). Neither set:
        empty dict, byte-identical to today's exact dense behavior,
        matching every other conditional kwarg here."""
        r_target = self.dy_r_target.get(layer_name)
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

    def apply_amortized_dy_r_target_control(self, measured_sps: float, target_sps: float,
                                             layer_name: Optional[str] = None,
                                             down_factor: float = 0.85, up_factor: float = 1.05,
                                             r_min: float = 0.05, r_max: float = 0.99) -> dict:
        """Closed-loop controller adjusting self.dy_r_target against
        MEASURED steps/sec (task #368, revised design -- see JOURNAL.md's
        "Grad-side k_t design, revised" entry for the full rationale).

        The original k_t sketch assumed an analytic kbar derivable from a
        fixed backward:forward compute-cost ratio (~10x, recalled). Direct
        measurement (forward_dense vs backward_dense, same layer, varying
        width) found the ratio is NOT constant -- 2.8x at n=48, 6.8x at
        n=128, 12.0x at n=384 -- so any single hardcoded ratio baked into
        a formula would be wrong at some scale. This sidesteps that
        entirely: no assumed ratio anywhere, just react to what steps/sec
        actually measures, same "closed-loop, measured-statistics, not
        guessed constants" philosophy apply_amortized_l2_decay already
        uses successfully.

        Call periodically (e.g. every N steps) from the training loop with
        that window's own measured steps/sec. measured_sps < target_sps:
        r_target shrinks (down_factor, capture less energy => fewer
        entries => cheaper). measured_sps > target_sps (no margin needed,
        matching l2 decay's own asymmetric-but-simple correction
        convention): r_target grows back toward r_max (up_factor).
        Clipped to [r_min, r_max] -- r_max<1.0 by default since 1.0 would
        defeat the entire compute-savings purpose dy_r_target exists for.

        layer_name (task #372): None (default) applies the SAME
        measured_sps/target_sps correction to every wide layer whose
        dy_r_target is currently set (mirrors the old model-level
        behavior exactly -- existing callers like the speed-target sweep
        script pass a single model-wide measured_sps and see the same
        net effect as before, since every layer started at the same
        initial value and moves in lockstep). Pass a specific name once
        real per-layer measured_sps is available (task #373's timing)
        to adjust just that layer independently -- a no-op (KeyError) if
        that layer's dy_r_target entry was never enabled (None) to begin
        with, since this controller only ADJUSTS an already-opted-in
        mechanism, it doesn't turn the mechanism on for a fresh layer.

        Does NOT implement the per-step E_t/Lbar energy-modulation half
        of the original design (r_t breathing above/below r_bar based on
        THIS step's own gradient energy) -- deferred to task #374:
        dy_r_target is consumed at forward()-CALL time, before that
        layer's own dy for this step is known (chicken-and-egg; resolved
        there via a one-step lag, not built yet here). This method only
        implements the OUTER (r_bar-vs-measured-sps) loop for now.

        Returns the updated {layer_name: r_target} dict for whichever
        layers were actually touched (empty if none had dy_r_target set,
        matching the old None-return no-op case)."""
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

    def apply_amortized_l2_decay(self, chunk_size: int, adaptation_rate: float = 0.3) -> dict:
        """Apply the amortized decoupled L2 decay + rolling health-stats
        mechanism (direct instruction, see conversation) to every real
        disldo_cls weight layer, INCLUDING the fp32 control (unlike
        magnitude_rescale_output, apply_amortized_l2_decay is bound on
        all three backends -- ValueAccessor-generic, no hasattr guard
        needed). Meant to be called once per real training step -- this
        is the "simple bound helps immediately, L2 helps health"
        complement to NOCAPS_KWARGS's per-precision max_abs_delta/max_ci
        (train_mqar_curriculum.py).

        CLOSED-LOOP, not a hand-picked half-life (corrects an earlier
        version of this method that took a fixed `half_life_steps` --
        direct instruction: "I thought we could set the L2
        hyperparameters based off the statistically measured health of
        the neural network itself for an exact solution"). A first
        attempt at a fixed half-life (2000 steps) DID verify the
        overflow fix but was badly overtuned for the wide+sparse config:
        q/k/o_proj -- already healthy with ZERO decay in the original
        buggy run (mean|w| ~0.04-0.13) -- got crushed to ~1e-5 by step
        16000, meaning the guessed constant dominated real learning
        instead of just providing a long-horizon health ceiling. Picking
        a bigger constant (20000) would have been the same mistake again
        with extra steps.

        Real fix: no half-life at all. Each layer keeps its own
        decay_factor in self._l2_decay_factor (persistent across step()
        calls, lazily initialized to 1.0 = no decay -- the max_abs_delta/
        max_ci hard bound is the actual immediate safety net per the
        two-part design, so it's fine for L2 to do nothing until it has
        real data). Every time a layer's rolling cursor completes a full
        pass, its MEASURED rms is compared against the closed-form
        target 1/sqrt(fan_in) (the same fan-in-normalized scale
        _preseed_dense_scattered already inits every dense fp32 layer
        to, sili/sparse_rnn.py) and decay_factor is corrected
        multiplicatively toward that target:
            decay_factor *= clip((target / measured_rms) ** adaptation_rate, 0.5, 2.0)
        clipped to (1e-6, 1.0]. rms above target -> decay strengthens
        (factor drops); rms below target -> decay relaxes (factor rises
        back toward 1.0, never above -- L2 only ever shrinks). This
        converges toward the fan-in-normalized target using the model's
        own measured statistics, the same rms/mean_abs/max_abs
        apply_amortized_l2_decay already returns every cycle -- no
        separate stats pass, no externally-guessed timescale.
        adaptation_rate is a control-loop GAIN (how fast the correction
        responds), not an equilibrium-setting parameter like half_life_
        steps was -- a 2x-off gain still converges to the same correct
        fixed point, just slower/faster, unlike a 2x-off half-life which
        directly sets the wrong equilibrium magnitude.

        chunk_size: synapses touched PER LAYER PER call (same raw budget
        for every layer regardless of width).

        Returns {layer_name: stats_dict} for every layer that JUST
        completed a full pass this call (stats_dict has mean_abs/rms/
        max_abs/n plus target/decay_factor for diagnostics); layers
        still mid-cycle are omitted since their reported stats would be
        the stale previous cycle's numbers."""
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

    def apply_dynamic_rank_control(self, tau_death: float = 0.05, tau_active: float = 0.3,
                                   theta: float = 1e-4, seed_scale: float = 0.05,
                                   scale_grace_period_steps: int = 50,
                                   additive_grace_period_steps: int = 5000) -> dict:
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

        scale_grace_period_steps/additive_grace_period_steps: PER-BRANCH
        cooldowns (direct instruction, replacing an earlier within-branch
        grow-vs-shrink asymmetry after a biology literature check -- see
        sili__new's DISLDOLayer.apply_dynamic_rank_control docstring for
        the full citations: real dendritic spine formation/elimination
        rates are roughly comparable within a mechanism, Holtmaat et al.
        Neuron 2005 / Grutzendler et al. Nature 2002, so each branch now
        uses ONE symmetric value internally). The real asymmetry is
        cross-branch: the scale (multiplicative, per-synapse) branch
        defaults to 50, the additive (whole-layer, homeostatic-like)
        branch defaults to 5000 (~100x), reflecting the real gap between
        Hebbian/STDP synaptic timescales (seconds-minutes) and
        homeostatic synaptic scaling (~24-48h to manifest; Turrigiano
        and colleagues' activity-blockade experiments; Zenke & Gerstner,
        Phil. Trans. R. Soc. B, 2017).

        Returns {layer_name: mutated_bool} for every real layer -- lets a
        caller log/count real rank-mutation events per layer without
        needing to know the same layer-name tuple again itself.
        """
        results = {}
        for name, layer in self._named_real_layers():
            if hasattr(layer, "apply_dynamic_rank_control"):
                results[name] = layer.apply_dynamic_rank_control(
                    tau_death, tau_active, theta, seed_scale,
                    scale_grace_period_steps, additive_grace_period_steps)
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

    def _to_sparse(self, x: Tensor) -> Tensor:
        """Sparsity plan Phase 6 (task #335) -- mirrors the existing
        sparse_rnn.py recurrent-cell precedent (`if not isinstance(
        state.data, CSR): ... CSR.from_dense(...).as_tensor(...)`)
        exactly, just as its own small reusable helper here. A no-op
        (returns x unchanged) when input_sparsity_p is None -- every
        existing caller of the 5 affected layers (input_proj/q/k/v/
        o_proj) sees a completely unmodified dense Tensor, same object
        even, matching the plan's own "None = today's exact behavior"
        requirement.

        Root-cause fix (found via sigma_grad_debug_fn probe): CSR.
        as_tensor() returns a bare leaf Tensor (no _children/_backward),
        so the old version of this method silently DETACHED x from the
        graph -- any layer downstream of a _to_sparse call (o_proj, in
        particular) still updated its OWN weights fine (its dx lands on
        the CSR tensor's own .grad, which as_tensor's docstring already
        anticipated being dense), but that gradient never propagated past
        the CSR tensor, since its _backward was the default no-op. That
        orphaned everything upstream of any _to_sparse boundary --
        confirmed directly: log_sigmas.grad was None on every backward
        call at embed_width=32 (input_sparsity_p set) vs real/nonzero
        every call at embed_width=16 (input_sparsity_p=None, this method
        a no-op) -- since centers/log_sigmas only reach the loss via
        gaussian_attention -> attn_pre_o_{mem,content} -> o_proj, i.e.
        exclusively through this exact boundary.

        Fix: wire real _children/_backward so gradient flows back to x.
        Straight-through (the full dense downstream gradient is passed
        back to x unmasked, not restricted to the k positions this
        particular top-k call happened to keep) rather than a hard
        mask-the-gradient-too version -- masking would zero the learning
        signal for every non-selected position on every step, making the
        selected/dropped split itself unable to shift over training,
        which is its own kind of frozen state."""
        if self.input_sparsity_p is None:
            return x
        csr = CSR.from_dense(np.asarray(x.data, dtype=np.float32),
                             self.input_sparsity_p, self.num_cpus)
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

    def _l1_sparsity_split(self, layer, input_t: Tensor, lr: float, coef: float,
                           requires_grad: bool = True, **extra_kwargs) -> Tensor:
        """Exact port of ToyTileRecurrenceRealFP4's own helper -- see its
        l1_sparsity_coef docstring for the full split-backward rationale.

        This probe forward()-calls the SAME layer instance a SECOND (or
        third, for q/k/v/o_proj under the sequential write-then-read
        design) time within one step() -- a genuinely PARALLEL branch
        that only merges back into the loss via simple addition, not a
        dependency of the main pass's own output. sili__new's
        backward_dense/backward now take `x` as an explicit argument
        (each Python closure holds its own input directly, same as
        every other Tensor op), so this is no longer order-sensitive at
        all -- previously needed use_explicit_token to avoid a real
        engine-side LIFO-cache bug, now simply not a concern."""
        out_aux = layer.forward(input_t, lr, requires_grad=requires_grad,
                                damp_by_importance=False, **self.synapse_kwargs, **extra_kwargs)
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
             learning_rate: float, requires_grad: bool = True,
             content_dy_sparsity_schedule: Optional[List[float]] = None
             ) -> Tuple[np.ndarray, Tensor, Optional[Tensor]]:
        """x_window: [num_tiles, embed_width] -- same convention as
        ToyTileRecurrenceRealFP4 (a real per-tile embedding, zeros for
        "nothing here yet"). memory_prev: [num_memory_slots, state_width]
        -- genuinely dedicated memory, DETACHED (no BPTT across steps,
        matching this whole project's convention), unlike
        ToyTileRecurrenceRealFP4's ambiguous per-window-slot rolling
        state (see its own docstring item 4). Returns (memory_new numpy
        [num_memory_slots, state_width], logits Tensor [num_tiles,
        vocab_size], aux_loss).

        requires_grad=False (direct instruction): most step() calls in a
        real training loop are NOT query positions (no loss ever gets
        computed for them, so backward() never runs) -- pass
        requires_grad=False for those to skip building the backward
        graph entirely, matching train_mqar_curriculum.py's own
        `i in targets` check. The detached memory_new handoff to the
        NEXT step() call is completely unaffected either way -- this
        only controls whether THIS step's own forward pass is
        backprop-able, not what values it computes. (The sequential
        write-then-read design below calls k_proj/v_proj/o_proj TWICE
        per step -- once for the write, once for the read -- but
        sili__new's backward_dense/backward now take `x` explicitly, so
        there's no engine-side call-ordering concern to manage either
        way.)

        content_dy_sparsity_schedule (query-step graded credit-assignment
        design, see step_cached's own docstring and conversation/
        JOURNAL.md): list of length num_tiles, index 0 = oldest content
        position ... index num_tiles-1 = newest, giving each content
        row its OWN backward gradient density instead of one uniform
        value -- e.g. full density for the newest (just-computed) row,
        progressively less for older cached ones, cheaper than uniform
        full density but richer than step_cached's zero-credit-for-
        older-rows default. Memory rows always keep full density
        (1.0) regardless -- they're the live recurrent state, not a
        graded-by-age position. None (default): completely unchanged
        behavior, every existing caller sees zero difference -- uses
        `self._wide_extra_kwargs`'s own scalar `dy_sparsity_p` exactly
        as before. Uses sili__new's `dy_sparsity_schedule` kwarg (real
        per-row top-k, NOT the same as the scalar `dy_sparsity_p`'s own
        surprising global-across-the-batch top-k semantics -- see
        DISLDOLayer.forward's own docstring correction)."""
        sw = self.state_width
        n_mem, n_content = self.num_memory_slots, self.num_tiles

        if content_dy_sparsity_schedule is not None:
            if len(content_dy_sparsity_schedule) != n_content:
                raise ValueError(
                    f"content_dy_sparsity_schedule has {len(content_dy_sparsity_schedule)} "
                    f"entries, expected num_tiles={n_content}")
            mem_schedule = [1.0] * n_mem
            content_schedule = list(content_dy_sparsity_schedule)
            full_schedule = mem_schedule + content_schedule
            _schedule_by_group = {"mem": mem_schedule, "content": content_schedule, "full": full_schedule}

            def _kw(layer_name: str, group: str) -> dict:
                return {"dy_sparsity_schedule": _schedule_by_group[group]}
        else:
            # Task #372: _wide_extra_kwargs is now per-layer (dy_r_target
            # is a dict), so the row-GROUP selection (mem/content/full)
            # and the layer-NAME selection are genuinely independent axes
            # here -- layer_name picks which entry of self.dy_r_target to
            # read, group is ignored entirely in this branch (dy_r_target/
            # dy_sparsity_p apply uniformly across a layer's own rows,
            # unlike the schedule branch above).
            def _kw(layer_name: str, group: str) -> dict:
                return self._wide_extra_kwargs(layer_name)

        x_window_t = Tensor(x_window.astype(np.float32))
        # Phase 6 (task #335): _to_sparse is a no-op unless input_sparsity_p
        # is set -- x_window_sparse IS x_window_t in that (default) case.
        x_window_sparse = self._to_sparse(x_window_t)
        x_wide = self.input_proj.forward(x_window_sparse, learning_rate, requires_grad=requires_grad,
                                         **self.synapse_kwargs, **_kw("input_proj", "content"))  # [n_content, sw]
        x_normed = rmsnorm_tensor(x_wide, self.input_ln, self.rms_eps)

        memory_prev_t = Tensor(memory_prev.astype(np.float32))
        memory_normed = rmsnorm_tensor(memory_prev_t, self.memory_ln, self.rms_eps)

        combined_normed = concat([memory_normed, x_normed], axis=0)          # [total_slots, sw]
        combined_normed_sparse = self._to_sparse(combined_normed)

        q = self.q_proj.forward(combined_normed_sparse, learning_rate, requires_grad=requires_grad,
                                **self.synapse_kwargs, **_kw("q_proj", "full"))
        k = self.k_proj.forward(combined_normed_sparse, learning_rate, requires_grad=requires_grad,
                                **self.synapse_kwargs, **_kw("k_proj", "full"))
        v = self.v_proj.forward(combined_normed_sparse, learning_rate, requires_grad=requires_grad,
                                **self.synapse_kwargs, **_kw("v_proj", "full"))

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

        # Reorder k/v into PHYSICAL (genuinely interleaved) position
        # order before they act as attention KEYS -- see __init__'s own
        # centers/_kv_phys_gather_idx docstring for the full rationale.
        # q stays in LOGICAL order untouched: each query row's bias
        # center is looked up from self.centers, which already holds the
        # correct physical-position VALUE per logical row, so only K/V's
        # ARRAY order (which gaussian_attention implicitly reads as
        # position via raw index j) needs to move.
        k_phys = gather(k, self._kv_phys_gather_idx).reshape((self.total_slots, sw))
        v_phys = gather(v, self._kv_phys_gather_idx).reshape((self.total_slots, sw))

        mem_idx = [m * sw + c for m in range(n_mem) for c in range(sw)]
        content_idx = [(n_mem + t) * sw + c for t in range(n_content) for c in range(sw)]

        # --- PASS 1: WRITE. Memory reads the (stale) full window, live --
        # unrestricted regardless of recurrent_only_output (memory always
        # reads everything; "write" stays allowed per the ablation's own
        # design), producing memory_new.
        q_mem = gather(q, mem_idx).reshape((n_mem, sw))
        centers_mem = gather(self.centers, list(range(n_mem)))
        sigmas_mem = gather(sigmas, list(range(n_mem)))
        attn_pre_o_mem = gaussian_attention(q_mem, k_phys, v_phys, centers_mem, sigmas_mem,
                                            num_cpus=self.num_cpus, causal=False)
        attn_mem = self.o_proj.forward(self._to_sparse(attn_pre_o_mem), learning_rate, requires_grad=requires_grad,
                                       **self.synapse_kwargs, **_kw("o_proj", "mem"))
        _accumulate_penalty(attn_mem)
        attn_mem.data = np.clip(attn_mem.data, -self.clip_range, self.clip_range)
        memory_new_t = rmsnorm_tensor(memory_prev_t + attn_mem, self.state_ln, self.rms_eps)
        memory_new_t.data = np.clip(memory_new_t.data, -self.clip_range, self.clip_range)

        # --- PASS 2: READ. Content queries attend memory AS IT STANDS
        # AFTER pass 1's write, not the stale memory_prev (direct
        # instruction: "input_proj->state_update->recurrent->state_update
        # in one step, not BPTT" -- two sequential layers run one after
        # the other WITHIN this same step() call). Backprop from
        # content_out/logits walks straight through memory_new_t, through
        # attn_mem/attn_pre_o_mem, through q_mem/k_phys/v_phys, into
        # x_wide/input_proj -- a real, live, undetached gradient path
        # entirely WITHIN this one call, NOT BPTT (nothing here crosses a
        # step() call boundary; only the numpy memory_new returned at the
        # very end, after this whole graph is already built, gets
        # detached, matching every other step()'s own convention). Before
        # this, input_proj's only live signal was the thin "which memory
        # slot does my query pick" channel (confirmed via direct
        # measurement: input_proj abs-grad-sum 0.037 vs v_proj/o_proj's
        # 2.89/19.4, both of which get real credit only because their
        # SAME shared weights are also exercised, abundantly, by the
        # read side every step) -- this pass gives it a real path for
        # "did this write end up useful," without ever needing gradient
        # to survive the hard-detach between step() calls.
        memory_new_normed = rmsnorm_tensor(memory_new_t, self.memory_ln, self.rms_eps)
        memory_new_normed_sparse = self._to_sparse(memory_new_normed)
        k_mem_fresh = self.k_proj.forward(memory_new_normed_sparse, learning_rate, requires_grad=requires_grad,
                                          **self.synapse_kwargs, **_kw("k_proj", "mem"))
        v_mem_fresh = self.v_proj.forward(memory_new_normed_sparse, learning_rate, requires_grad=requires_grad,
                                          **self.synapse_kwargs, **_kw("v_proj", "mem"))
        _accumulate_penalty(k_mem_fresh)
        _accumulate_penalty(v_mem_fresh)
        k_mem_fresh.data = np.clip(k_mem_fresh.data, -self.clip_range, self.clip_range)
        v_mem_fresh.data = np.clip(v_mem_fresh.data, -self.clip_range, self.clip_range)

        # Content's OWN k/v (from pass 1, i.e. this step's raw input) stay
        # unchanged -- only the memory portion is refreshed. Rebuilt
        # through the SAME interleaved-physical gather as k_phys/v_phys
        # (see __init__'s own centers docstring) so the Gaussian bias math
        # still sees the correct, genuinely-spread positions -- collapsing
        # back to a bare [n_mem, sw] slice here would silently reintroduce
        # the exact clustered-position underflow the interleave fix
        # closed.
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
            # Same value-masking approach as before (see __init__'s
            # _mem_only_value_mask docstring): keys stay the full,
            # correctly-interleaved array (content's own key still
            # competes in the softmax), only content's VALUE contribution
            # is zeroed -- now against the FRESH (pass-1-updated) memory
            # values rather than the stale ones.
            v2_phys_mem_only = v2_phys * self._mem_only_value_mask
            attn_pre_o_content = gaussian_attention(q_content, k2_phys, v2_phys_mem_only,
                                                     centers_content, sigmas_content,
                                                     num_cpus=self.num_cpus, causal=False)
        else:
            attn_pre_o_content = gaussian_attention(q_content, k2_phys, v2_phys,
                                                     centers_content, sigmas_content,
                                                     num_cpus=self.num_cpus, causal=False)
        attn_content = self.o_proj.forward(self._to_sparse(attn_pre_o_content), learning_rate, requires_grad=requires_grad,
                                           **self.synapse_kwargs, **_kw("o_proj", "content"))
        _accumulate_penalty(attn_content)
        attn_content.data = np.clip(attn_content.data, -self.clip_range, self.clip_range)

        # Debug instrumentation (task #303): cheap reference-only capture
        # (no copies) of every stage between the input embedding and the
        # readout, for bisecting exactly where a NaN/Inf first appears in
        # the forward chain -- np.clip does NOT sanitize NaN (clip(nan)==
        # nan), so the clip calls in this function are not themselves
        # proof any given stage is finite.
        self.last_debug = {
            "x_wide": x_wide.data, "q": q.data, "k": k.data, "v": v.data,
            "attn_pre_o_mem": attn_pre_o_mem.data, "attn_pre_o_content": attn_pre_o_content.data,
            "attn_mem": attn_mem.data, "attn_content": attn_content.data,
            "sigmas": sigmas.data, "log_sigmas": self.log_sigmas.data,
            # x_window_t (direct instruction, embedding-learning hook): the
            # Tensor itself (not just .data) -- when requires_grad=True and
            # the caller runs loss.backward() after step() returns, this
            # Tensor's .grad is populated with dL/d(x_window), letting a
            # caller scatter-update an external embedding table (e.g. an
            # SDR token embedding built outside this model) without step()
            # needing to know about tokens/vocab at all.
            "x_window_t": x_window_t,
        }

        if self.l1_sparsity_coef > 0.0:
            l1_terms = [
                self._l1_sparsity_split(self.input_proj, x_window_sparse, learning_rate, self.l1_sparsity_coef,
                                        requires_grad=requires_grad, **self._wide_extra_kwargs("input_proj")),
                self._l1_sparsity_split(self.q_proj, combined_normed_sparse, learning_rate, self.l1_sparsity_coef,
                                        requires_grad=requires_grad, **self._wide_extra_kwargs("q_proj")),
                self._l1_sparsity_split(self.k_proj, combined_normed_sparse, learning_rate, self.l1_sparsity_coef,
                                        requires_grad=requires_grad, **self._wide_extra_kwargs("k_proj")),
                self._l1_sparsity_split(self.v_proj, combined_normed_sparse, learning_rate, self.l1_sparsity_coef,
                                        requires_grad=requires_grad, **self._wide_extra_kwargs("v_proj")),
                self._l1_sparsity_split(self.o_proj, self._to_sparse(gaussian_attention(
                    q, k_phys, v_phys, self.centers, sigmas, num_cpus=self.num_cpus, causal=False)),
                    learning_rate, self.l1_sparsity_coef, requires_grad=requires_grad, **self._wide_extra_kwargs("o_proj")),
            ]
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term

        # Residual against the RAW (pre-RMSNorm) content value, matching
        # ToyTileRecurrenceRealFP4's own convention -- kept live
        # regardless of recurrent_only_output (direct instruction,
        # post-validation task #315): the strict ablation used to ALSO
        # zero this residual, on top of blocking content-content
        # attention, to fully isolate memory's contribution for the
        # distance-sweep verification. That isolation already did its
        # job -- confirmed genuine cross-detach recurrent persistence
        # (accuracy well above chance at distances requiring the value
        # to survive step() boundaries the sequential write-then-read
        # design's own gradient can't reach across). Zeroing this
        # residual isn't needed for correctness going forward: x_wide is
        # the QUERY token's own embedding, which never correlates with
        # MQAR's correct recall value (the task is information-
        # theoretically undecidable from the query token alone), so
        # leaving it live can't reintroduce a "cheat" path for this
        # task -- it just restores an ordinary residual connection
        # instead of needlessly handicapping the model. recurrent_only_
        # output now ONLY blocks content-content attention (the actual
        # "recurrent only" property); it no longer touches this residual.
        pre_norm_content = x_wide + attn_content
        content_out = rmsnorm_tensor(pre_norm_content, self.state_ln, self.rms_eps)
        _accumulate_penalty(content_out)  # pre-clip, same reasoning as above
        content_out.data = np.clip(content_out.data, -self.clip_range, self.clip_range)
        self.last_debug["pre_norm_content"] = pre_norm_content.data
        self.last_debug["content_out"] = content_out.data

        # memory_new is the plain numpy carry to the NEXT step() call --
        # no BPTT across steps, matching this project's convention (see
        # this function's own docstring). Everything above it is still a
        # live Tensor at this point (memory_new_t), which is exactly what
        # lets pass 2's gradient reach back through pass 1 into
        # input_proj within THIS call; the detach happens only here, at
        # the very last moment before crossing the step() boundary.
        memory_new = memory_new_t.data.copy()

        pooled = content_out.reshape((n_content, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)
        self.last_debug["pooled"] = pooled.data
        if self.l1_sparsity_coef > 0.0:
            lm_l1 = self._l1_sparsity_split(self.lm_head, pooled, learning_rate, self.l1_sparsity_coef,
                                            requires_grad=requires_grad)
            aux_loss = lm_l1 if aux_loss is None else aux_loss + lm_l1
        logits = self.lm_head.forward(pooled, learning_rate, requires_grad=requires_grad,
                                      **self.synapse_kwargs, **self._output_extra_kwargs)

        # Advantage-actor-critic value head (opt-in, see __init__'s
        # use_critic docstring): exposed via an attribute rather than a
        # 4th return value, since step()'s 3-tuple return is unpacked by
        # dozens of existing call sites across both repos and a return-
        # arity change would break every one of them. Caller (e.g.
        # scripts/train_mqar_curriculum.py) reads model.last_critic_pred
        # right after this call.
        self.last_critic_pred = (
            self.critic_head.forward(pooled, learning_rate, requires_grad=requires_grad,
                                     **self.synapse_kwargs, **self._output_extra_kwargs)
            if self.use_critic else None)

        return memory_new, logits, aux_loss

    def step_cached(self, new_token_embed: np.ndarray, memory_prev: np.ndarray,
                    learning_rate: float, tile_cache: Optional[List[Tuple[np.ndarray, np.ndarray]]],
                    requires_grad: bool = True
                    ) -> Tuple[np.ndarray, Tensor, Optional[Tensor], List[Tuple[np.ndarray, np.ndarray]]]:
        """Incremental alternative to step(): takes ONE new token's raw
        embedding [embed_width] instead of a full [num_tiles, embed_width]
        sliding window, plus an explicit `tile_cache` carrying the
        num_tiles-1 older content positions' (k_row, v_row) -- same
        explicit-state-in/out convention as memory_prev/memory_new (no
        hidden instance-mutation cache, matching this project's own
        established preference, see [[project_sili_dense_input_stack_simplification]]
        in memory: an earlier engine-side hidden cache caused a real,
        hard-to-find correctness bug).

        Why this is correct, not just faster: input_proj/q_proj/k_proj/
        v_proj are simple per-row (non-mixing) projections, so a content
        tile's k/v depends ONLY on its own token embedding and the
        CURRENT weight values -- never on other tiles, never on memory.
        step()'s full-window rebuild therefore recomputes up to
        num_tiles IDENTICAL values per token as the window slides past
        it, every single call. Caching removes that redundancy.

        Direct instruction on the one real approximation this
        introduces: weights only change on requires_grad=True (query)
        steps, and even then only ~0.1% of individual synapses move per
        update (see [[project_dy_sparsity_p_validated_speedup]]'s
        backward-sparsity findings) -- so a cached tile's k/v drifts by
        a tiny, bounded amount as it ages through the window, rather
        than being invalidated wholesale after every weight update.
        Treated as "mostly fine" per direct instruction, not chased to
        exact invalidation.

        Only the NEWEST content position's own logits/q ever get used
        downstream (confirmed: train_mqar_curriculum.py's own loss/
        accuracy always reads row num_tiles-1, never any other content
        row) -- so q, x_wide, attention, and the final lm_head/
        critic_head readout are all computed for exactly ONE content
        row here, not num_tiles. This also means step_cached's own
        `logits`/`aux_loss` shapes are [1, vocab_size] (a single row),
        not [num_tiles, vocab_size] -- callers reading row 0 instead of
        row num_tiles-1 is the one real call-site change needed.

        tile_cache: list of up to (num_tiles-1) (k_row, v_row) numpy
        [state_width] tuples, OLDEST FIRST. None or an empty/short list
        (fewer than num_tiles-1 entries) is padded with zero rows at the
        oldest end -- exactly reproducing _build_tile_window's own
        "zeros for nothing here yet before sequence start" behavior
        (input_proj/k_proj/v_proj have no bias term, so a zero raw
        embedding really does propagate to an exact zero k/v row, not
        an approximation). Reset to None/[] at the start of each new
        training sequence, same as memory_prev gets reset to zeros.

        Returns (memory_new, logits [1, vocab_size], aux_loss,
        new_tile_cache) -- new_tile_cache is ready to pass back in on
        the very next call."""
        sw = self.state_width
        n_mem, n_content = self.num_memory_slots, self.num_tiles

        new_embed_2d = np.asarray(new_token_embed, dtype=np.float32).reshape((1, self.embed_width))
        new_embed_t = Tensor(new_embed_2d)
        new_embed_sparse = self._to_sparse(new_embed_t)
        x_wide_new = self.input_proj.forward(new_embed_sparse, learning_rate, requires_grad=requires_grad,
                                             **self.synapse_kwargs, **self._wide_extra_kwargs("input_proj"))  # [1, sw]
        x_normed_new = rmsnorm_tensor(x_wide_new, self.input_ln, self.rms_eps)

        memory_prev_t = Tensor(memory_prev.astype(np.float32))
        memory_normed = rmsnorm_tensor(memory_prev_t, self.memory_ln, self.rms_eps)

        combined_normed_step = concat([memory_normed, x_normed_new], axis=0)  # [n_mem+1, sw]
        combined_normed_step_sparse = self._to_sparse(combined_normed_step)

        q_step = self.q_proj.forward(combined_normed_step_sparse, learning_rate, requires_grad=requires_grad,
                                     **self.synapse_kwargs, **self._wide_extra_kwargs("q_proj"))
        k_step = self.k_proj.forward(combined_normed_step_sparse, learning_rate, requires_grad=requires_grad,
                                     **self.synapse_kwargs, **self._wide_extra_kwargs("k_proj"))
        v_step = self.v_proj.forward(combined_normed_step_sparse, learning_rate, requires_grad=requires_grad,
                                     **self.synapse_kwargs, **self._wide_extra_kwargs("v_proj"))

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

        # Reassemble the FULL [n_content, sw] content k/v from the cache
        # (oldest first, zero-padded at the oldest end if the sequence
        # just started) plus this step's one fresh row at the end --
        # same logical ordering _build_tile_window's own window array
        # always used (position 0 = oldest, n_content-1 = newest).
        cache = list(tile_cache) if tile_cache else []
        pad = max(0, (n_content - 1) - len(cache))
        zero_row = np.zeros(sw, dtype=np.float32)
        cache_k_rows = [zero_row] * pad + [row[0] for row in cache[-(n_content - 1):]] if n_content > 1 else []
        cache_v_rows = [zero_row] * pad + [row[1] for row in cache[-(n_content - 1):]] if n_content > 1 else []
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
        attn_pre_o_mem = gaussian_attention(q_mem, k_phys, v_phys, centers_mem, sigmas_mem,
                                            num_cpus=self.num_cpus, causal=False)
        attn_mem = self.o_proj.forward(self._to_sparse(attn_pre_o_mem), learning_rate, requires_grad=requires_grad,
                                       **self.synapse_kwargs, **self._wide_extra_kwargs("o_proj"))
        _accumulate_penalty(attn_mem)
        attn_mem.data = np.clip(attn_mem.data, -self.clip_range, self.clip_range)
        memory_new_t = rmsnorm_tensor(memory_prev_t + attn_mem, self.state_ln, self.rms_eps)
        memory_new_t.data = np.clip(memory_new_t.data, -self.clip_range, self.clip_range)

        # --- PASS 2: READ, only the newest content row's own query ---
        memory_new_normed = rmsnorm_tensor(memory_new_t, self.memory_ln, self.rms_eps)
        memory_new_normed_sparse = self._to_sparse(memory_new_normed)
        k_mem_fresh = self.k_proj.forward(memory_new_normed_sparse, learning_rate, requires_grad=requires_grad,
                                          **self.synapse_kwargs, **self._wide_extra_kwargs("k_proj"))
        v_mem_fresh = self.v_proj.forward(memory_new_normed_sparse, learning_rate, requires_grad=requires_grad,
                                          **self.synapse_kwargs, **self._wide_extra_kwargs("v_proj"))
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
            attn_pre_o_content = gaussian_attention(q_new, k2_phys, v2_phys_mem_only,
                                                     centers_content, sigmas_content,
                                                     num_cpus=self.num_cpus, causal=False)
        else:
            attn_pre_o_content = gaussian_attention(q_new, k2_phys, v2_phys,
                                                     centers_content, sigmas_content,
                                                     num_cpus=self.num_cpus, causal=False)
        attn_content = self.o_proj.forward(self._to_sparse(attn_pre_o_content), learning_rate, requires_grad=requires_grad,
                                           **self.synapse_kwargs, **self._wide_extra_kwargs("o_proj"))
        _accumulate_penalty(attn_content)
        attn_content.data = np.clip(attn_content.data, -self.clip_range, self.clip_range)

        pre_norm_content = x_wide_new + attn_content
        content_out = rmsnorm_tensor(pre_norm_content, self.state_ln, self.rms_eps)
        _accumulate_penalty(content_out)
        content_out.data = np.clip(content_out.data, -self.clip_range, self.clip_range)

        self.last_debug = {
            "x_wide": x_wide_new.data, "q": q_new.data, "k": k_new.data, "v": v_new.data,
            "attn_pre_o_mem": attn_pre_o_mem.data, "attn_pre_o_content": attn_pre_o_content.data,
            "attn_mem": attn_mem.data, "attn_content": attn_content.data,
            "sigmas": sigmas.data, "log_sigmas": self.log_sigmas.data,
        }

        memory_new = memory_new_t.data.copy()

        pooled = content_out.reshape((1, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)
        self.last_debug["pooled"] = pooled.data

        if self.l1_sparsity_coef > 0.0:
            l1_terms = [
                self._l1_sparsity_split(self.input_proj, new_embed_sparse, learning_rate, self.l1_sparsity_coef,
                                        requires_grad=requires_grad, **self._wide_extra_kwargs("input_proj")),
                self._l1_sparsity_split(self.q_proj, combined_normed_step_sparse, learning_rate, self.l1_sparsity_coef,
                                        requires_grad=requires_grad, **self._wide_extra_kwargs("q_proj")),
                self._l1_sparsity_split(self.k_proj, combined_normed_step_sparse, learning_rate, self.l1_sparsity_coef,
                                        requires_grad=requires_grad, **self._wide_extra_kwargs("k_proj")),
                self._l1_sparsity_split(self.v_proj, combined_normed_step_sparse, learning_rate, self.l1_sparsity_coef,
                                        requires_grad=requires_grad, **self._wide_extra_kwargs("v_proj")),
                self._l1_sparsity_split(self.o_proj, self._to_sparse(
                    concat([attn_pre_o_mem, attn_pre_o_content], axis=0)),
                    learning_rate, self.l1_sparsity_coef, requires_grad=requires_grad, **self._wide_extra_kwargs("o_proj")),
            ]
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term

        if self.l1_sparsity_coef > 0.0:
            lm_l1 = self._l1_sparsity_split(self.lm_head, pooled, learning_rate, self.l1_sparsity_coef,
                                            requires_grad=requires_grad)
            aux_loss = lm_l1 if aux_loss is None else aux_loss + lm_l1
        logits = self.lm_head.forward(pooled, learning_rate, requires_grad=requires_grad,
                                      **self.synapse_kwargs, **self._output_extra_kwargs)

        self.last_critic_pred = (
            self.critic_head.forward(pooled, learning_rate, requires_grad=requires_grad,
                                     **self.synapse_kwargs, **self._output_extra_kwargs)
            if self.use_critic else None)

        new_cache = (list(tile_cache) if tile_cache else []) + [
            (k_new.data.copy().reshape(sw), v_new.data.copy().reshape(sw))]
        if len(new_cache) > (n_content - 1):
            new_cache = new_cache[-(n_content - 1):]

        return memory_new, logits, aux_loss, new_cache
