from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sili.tensor import Tensor, gaussian_attention, exp, reduce_sum, silu, power, tensor_abs
from sili.sparse_rnn import DISLDOLayer
from sili.energy import EnergyDynamics

from .toy_recall_models import rmsnorm_tensor, sigmoid_tensor
from .toy_precision_models import _toy_scale_energy, _apply_energy


class ToyTileRecurrenceRealFP4:
    """ToyTileRecurrence's exact architecture, built from DISLDOLayer
    -family layers (`disldo_cls=`) instead of DenseTensorLinear.
    `centers`/`log_sigmas`/RMSNorm weights stay plain Tensor leaves,
    trained by a small external AdamOptimizer via
    `parameters_for_optimizer()`.

    SwiGLU MLP and tanh have been removed in favor of a minimal
    attention-only recurrence with `[-clip_range, clip_range]` state
    clipping. Default 6.0 (matching FP4/E2M1's own max representable
    magnitude) -- confirmed via direct comparison against the original
    2.0 default (mean_acc 0.98 vs 0.75, 3/3 seeds, lower variance,
    already converged vs still mid-learning at step 15000; see
    sili_peridot JOURNAL.md's clip-range test entry for the full
    result).

    `step()` returns (M_new, logits, aux_loss) -- aux_loss is None
    unless `use_energy=True`."""

    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int,
                 mlp_hidden: int, num_tiles: int, max_weights: int,
                 num_cpus: int = 2, rms_eps: float = 1e-6, disldo_cls=DISLDOLayer,
                 use_energy: bool = False, energy_kwargs: Optional[dict] = None,
                 use_attention: bool = True, o_proj_depth: int = 1,
                 dense: bool = False, clip_range: float = 6.0,
                 magnitude_penalty_coef: float = 0.0,
                 spectral_norm_target: Optional[float] = None,
                 spectral_norm_ema_decay: float = 0.9,
                 l1_sparsity_coef: float = 0.0,
                 cosine_lm_head: bool = False,
                 gated_combine: bool = False,
                 gate_floor: float = 0.1,
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
        higher-precision layer, rather than needing more WIDTH.

        magnitude_penalty_coef>0: adds coef*mean(x**2) aux-loss terms
        on the (post-clip) attn_o_proj/state values -- a real gradient
        discouraging large recurrent activation magnitude, independent
        of and in addition to the hard clip below. Motivated directly:
        the hard clip is a straight-through `.data` overwrite (bypasses
        autograd entirely), so nothing currently tells the network NOT
        to keep driving activation magnitude up against it -- a
        pathological attractor under dense connectivity specifically
        (found via JOURNAL.md 2026-08-10's dense-vs-sparse investigation:
        even with NaN-safety fixed, dense connectivity still collapses
        to chance over a full run, correlated with frequent non-finite
        -gradient skips concentrated right at harder curriculum stages).
        Computed from the ALREADY-CLIPPED value (not the pre-clip one)
        deliberately: `power`'s backward reads `.data` lazily at
        backward-call time, so building it from the pre-clip Tensor
        before mutating `.data` in place would silently differentiate
        against the wrong (already-overwritten) value by the time
        backward actually runs; using the post-clip value instead is
        self-consistent (nothing mutates it again after) AND gives a
        gradient magnitude that's itself bounded by clip_range (can't
        blow up for extreme pre-clip values), rather than reintroducing
        the same kind of unbounded-magnitude risk this is meant to fix.
        Deliberately independent of use_energy/EnergyDynamics (not
        combined, not gated on it) -- direct instruction to keep this
        mechanism isolated for testing rather than compounding it with
        energy_rl's own extinguishing pressure, which "adds a lot right
        now" on its own and would confound an isolated test of this.

        spectral_norm_target: if set, rescales o_proj's real output by
        `target/sigma_ema` every step, where `sigma_ema` is an
        EMA-smoothed power-iteration estimate of o_proj's dominant
        singular value -- the actual root-cause fix for dense
        connectivity's instability (JOURNAL.md 2026-08-11: measured
        spectral radius 1.19 at init, 1.50 after 400 steps for dense
        vs 0.85/0.83 flat for sparse -- a spectral-radius-above-1
        recurrent map structurally amplifies signal every pass,
        independent of and NOT fixed by magnitude_penalty_coef/energy_rl,
        which only constrain average magnitude, not the weight matrix's
        dominant eigenvalue specifically). Standard "Spectral
        Normalization" technique (Miyato et al. 2018): a persistent
        probe vector `u` (NOT the actual recurrent state -- deliberately
        decoupled from the real data/RMSNorm/clip/residual path so the
        estimate reflects o_proj alone) is updated via ONE power
        -iteration step per real step (`layer.forward(u, 0.0)` --
        forward-only, same zero-side-effect convention `evaluate()`
        already uses, no backward/optimizer call needed at all, cheap:
        one forward pass plus a numpy norm, nothing like the O(n^3) cost
        of an exact eigendecomposition). `sigma_ema` (not the raw
        per-step estimate) is what's actually used, smoothed at
        `spectral_norm_ema_decay` -- important once synaptogenesis is
        active: a structural change (new synapse) can make one step's
        raw estimate jump before `u` re-converges to the new dominant
        eigenvector, and unlike an init-time-only fix, this whole
        mechanism re-tracks automatically as the weight matrix changes,
        whether from ordinary gradient updates or future synaptogenesis.
        The rescale itself is an ordinary Tensor*float multiply already
        supported by autograd (out * (target/sigma_ema)) -- no new
        differentiable primitive needed anywhere. None/off by default,
        independent of magnitude_penalty_coef/use_energy (composable,
        not mutually exclusive -- direct request to test combinations).

        l1_sparsity_coef: the LANDMARK dense-connectivity stability
        mechanism (see scripts/l1_sparsity_probe.py's own header
        docstring and sili_peridot JOURNAL.md's 2026-08-13 entry) --
        ported here from that probe script's standalone
        `OriginalArchModel` reproduction, which is architecturally
        identical to this class (single v_proj on the combined
        qkv_source, single o_proj) and had NOT previously been merged
        into this shared model. Found to reach mean=1.0000 across 5
        seeds at coef=0.05 AND 0.07 on the 15000-step out-of-context
        curriculum, dense_base12 -- the single best stability result of
        the entire investigation, beating spectral_norm_target's own
        0.8858 (a hard rescale, since made unavailable as a production
        mechanism) with NO hard rescale of any kind. Do NOT combine with
        spectral_norm_target or magnitude_penalty_coef -- combining L1
        with an L2-ratio-shaped mechanism was tested and HURTS (0.7333
        vs 1.0000), and spectral is off the table anyway; this is meant
        to be used ALONE. Applied via the same "split-backward" delivery
        as the probe: a SECOND, independent forward(..., damp_by_
        importance=False) call per layer gives the L1 term its own
        undamped gradient path, avoiding dilution by DISLDO's own
        RMSprop-style per-synapse update (~1731x magnitude ratio
        otherwise, see JOURNAL.md). Applied to input_proj (on the raw
        narrow x_window), q_proj/k_proj/v_proj (on qkv_source), o_proj
        (on its own real input -- the post-attention/post-energy tensor
        when use_attention=True, else qkv_source directly), and lm_head
        (on pooled) -- all 6 real weight layers (5 at the time this was
        first validated, before input_proj existed -- see conversation
        for why input_proj was added; treat its own L1 coverage as a
        direct, analogous extension, not independently re-validated),
        matching the probe's own final "lm_head previously had no L1
        term" fix (task #176: under all_zero_init, dL/d(pooled) is
        exactly 0 whenever W_lmhead=0, so lm_head needs this same direct
        escape route). o_proj_depth>1 is NOT covered by the validated
        result (the probe never tested it) -- if set, L1 is applied
        per-sublayer using each sublayer's own real input, a direct
        analogous extension, but treat that combination as unvalidated
        until tested.

        cosine_lm_head: makes lm_head's readout a "cosine classifier"
        (row-normalized matched filter) instead of a raw dot product.
        Root cause found directly (see conversation): with a fixed,
        untrained, random `embed_table` (e.g. VOCAB=128,
        EMBED_WIDTH=16 in the MQAR K-sweep), a raw-dot-product readout
        genuinely misclassifies ~2-5% of possible value tokens even
        with an OPTIMAL hand-built lm_head, because raw logit[v] =
        ||pooled||*||W_row[v]||*cos_sim(pooled,W_row[v]) conflates
        direction similarity with W_row[v]'s own magnitude -- an
        unrelated token whose row happens to have a larger norm can
        outscore the true match even at lower cosine similarity.
        RMSNorm already fixes this on the INPUT side (x_normed/M_new
        have fixed magnitude regardless of which token arrived, so
        pooled's own scale never affects the argmax -- confirmed
        numerically, normalizing pooled alone changes nothing), so the
        fix only needs to touch lm_head's OUTPUT-row norms. This is
        NOT a fundamental embedding-width/dimension-counting ceiling
        (an earlier hypothesis, WRONG, retracted after direct
        numerical verification with a properly row-normalized template
        reached exactly 64/64 at the SAME embed_width=16) -- it's
        purely this one magnitude/direction conflation, fixable
        without touching embed_width, column_neurons, or the recurrent
        state at all.

        Since lm_head is a real disldo_cls layer (its weights live in
        FP4-quantized C++ storage, not a plain accessible matrix), row
        norms are reconstructed the same way `_spectral_rescale_factor`
        already probes a layer's weights elsewhere in this file: feed
        the `embed_width` standard basis vectors through
        `lm_head.forward(..., 0.0)` (one batched, zero-side-effect,
        no-backward call -- forward(e_i) = W_row[:, i], i.e. column i
        of the effective weight matrix; stacking all E columns gives
        every row's norm in one call) and divide the real logits by
        (row_norm + eps) before the loss ever sees them. Recomputed
        fresh every step (no EMA smoothing, unlike spectral_norm_
        target's sigma_ema) -- lm_head's weights update every step via
        its own inline backward, so a stale norm would drift; the
        extra forward call is cheap (E rows, not O(vocab)). Default
        False (opt-in, unvalidated against a full multi-seed sweep
        yet -- confirmed only via the standalone numeric witness in
        conversation).

        gated_combine: replaces the plain `qkv_source = x_normed +
        m_normed` sum with a LEARNED, content-dependent gate. Root
        cause found directly (see conversation): every real, working
        segment/block-recurrent transformer this project's design was
        checked against (Recurrent Memory Transformer, Block-Recurrent
        Transformer, Infini-attention) combines fresh input and
        carried state either as separate attention-visible tokens or
        through a LEARNED gate -- none of them use an untrained plain
        elementwise sum. A plain sum forces two different signals into
        superposition in the same channels before the network has any
        mechanism (attention weights, a gate) to tell them apart, and
        the state update itself (residual add + RMSNorm + hard clip)
        has no learned forget mechanism at all -- architecturally the
        pre-LSTM "vanilla RNN" pattern gating was invented to fix.

        Two new real disldo_cls layers, `gate_x_proj`/`gate_m_proj`
        (state_width -> state_width each), computed from x_normed/
        m_normed SEPARATELY and summed (`gate_x_proj(x_normed) +
        gate_m_proj(m_normed)`) -- mathematically identical to a
        single state_width*2 -> state_width layer over the
        concatenation, but avoids needing a Tensor concat op. Gate =
        sigmoid(that sum), matching Infini-attention's learned gating
        scalar in spirit, but per-channel and INPUT-DEPENDENT (a
        function of the actual x_normed/m_normed content each step,
        not a single static learned scalar) -- closer to a real LSTM/
        GRU-style gate, since the task fundamentally needs
        content-dependent decisions ("keep old state when nothing new
        happened, overwrite when something did"), which a static gate
        can't express.

        `qkv_source = gate*x_normed + (1-gate)*m_normed`.

        gate_floor: per direct instruction, the gate must NOT be able
        to reach full-input-only (gate=1) or full-state-only (gate=0)
        -- both are real failure modes (total forgetting every step,
        or the state going permanently deaf to new input) -- so the
        raw sigmoid output is rescaled into `[gate_floor,
        1-gate_floor]` rather than used directly on `(0, 1)`. Default
        0.1: even at full saturation, each stream always keeps >=10%
        weight.

        When `l1_sparsity_coef > 0`, `gate_x_proj`/`gate_m_proj` get
        the same split-backward L1 term as every other real weight
        layer (their own real inputs: x_normed/m_normed respectively)
        -- added for the same reason input_proj's L1 coverage was:
        dense connectivity is documented (JOURNAL.md 2026-08-13) to
        destabilize without L1 on every real weight layer, and leaving
        two NEW dense layers uncovered would be a foreseeable regression
        of exactly that already-fixed failure mode, not a hypothetical
        one. Unvalidated as a combination (same caveat as input_proj's
        own L1 term) until tested."""
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.rms_eps = rms_eps
        self.clip_range = clip_range
        self.cosine_lm_head = cosine_lm_head
        self.gated_combine = gated_combine
        self.gate_floor = gate_floor
        self.magnitude_penalty_coef = magnitude_penalty_coef
        self.spectral_norm_target = spectral_norm_target
        self.spectral_norm_ema_decay = spectral_norm_ema_decay
        self.l1_sparsity_coef = l1_sparsity_coef
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
        n_layer_seeds = 5 + max(o_proj_depth, 1)  # input_proj, q, k, v, lm_head + o_proj sublayer(s)
        if gated_combine:
            n_layer_seeds += 2  # gate_x_proj, gate_m_proj -- only reserved when
            # actually used, so gated_combine=False callers' RNG consumption
            # (and hence every existing test/script's reproducibility) is
            # untouched -- same concern already flagged for spectral_norm_
            # target's own probe-vector RNG draws above.
        layer_seeds = iter(int(s) for s in rng.integers(0, 2**31 - 1, size=n_layer_seeds))

        # dense=True only forwarded when set (not unconditionally) -- only
        # disldo_cls options that were actually updated for it (DISLDOLayer/
        # DISLDOLayerDeterministic in sili__new, TrueMultiDigitLayer here)
        # accept a `dense` kwarg at all; every other existing disldo_cls
        # option would TypeError on an unexpected kwarg otherwise, breaking
        # every caller that doesn't ask for it.
        dense_kwargs = {"dense": True} if dense else {}

        # 0. Input projection: embed_width -> state_width, a REAL trained
        # layer, not the np.repeat tiling this class used to receive its
        # window through. Direct correction (see conversation): column-
        # averaging's actual purpose is letting a narrow OUTPUT's gradient
        # reach the entire wide state on readout (mean-pool down, so
        # d(mean)/d(each element)=1/column_neurons spreads credit to every
        # column) -- applying that same repeat/average pairing to the
        # INPUT side too was a misapplication of the same operator to a
        # problem it was never meant to solve, not an intentional design.
        # The wide state's actual content for a fresh token must come from
        # a real learned mapping, same as q/k/v/o_proj/lm_head.
        self.input_proj = disldo_cls(embed_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)

        # 0.5. Gated combine (opt-in) -- see gated_combine's own docstring
        # above for the full rationale.
        if gated_combine:
            self.gate_x_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                          rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
            self.gate_m_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                          rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)

        # 1. Core Attention & Output Projections
        if use_attention:
            self.q_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
            self.k_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
            self.v_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
        if o_proj_depth > 1:
            per_layer_weights = max(max_weights // o_proj_depth, state_width)
            self.o_proj = [disldo_cls(state_width, state_width, per_layer_weights, num_cpus,
                                      rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
                          for _ in range(o_proj_depth)]
        else:
            self.o_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
        self.lm_head = disldo_cls(embed_width, vocab_size, max_weights, num_cpus,
                                  rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
        
        # 2. Norms & Gaussian Attention Params
        self.input_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.state_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.centers = Tensor(np.array([i + 0.5 for i in range(num_tiles)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(num_tiles, dtype=np.float32))

        # 3. Spectral-norm probe state (only used if spectral_norm_target is
        # set) -- one persistent probe vector + EMA sigma PER o_proj sublayer
        # (o_proj_depth>1 chains several square state_width->state_width
        # layers, each needs its own independent estimate). Seeded from the
        # same `rng` as everything else above for reproducibility.
        # Only allocated/RNG-consuming when actually used -- must NOT
        # perturb rng's consumption sequence for the (default, every
        # existing arm/test) spectral_norm_target=None path, or every
        # downstream draw from this same `rng` shifts silently, breaking
        # reproducibility for code that never asked for this feature.
        if spectral_norm_target is not None:
            n_o_layers = o_proj_depth if o_proj_depth > 1 else 1
            self._spectral_u = [rng.standard_normal(state_width).astype(np.float32)
                                for _ in range(n_o_layers)]
            self._spectral_u = [u / (np.linalg.norm(u) + 1e-8) for u in self._spectral_u]
            self._spectral_sigma_ema = [None] * n_o_layers
            # Warm-start: a single untrained random probe vector badly
            # underestimates the true dominant singular value (power
            # iteration hasn't converged yet), so the FIRST real-step
            # rescale would overshoot the target -- confirmed directly
            # (measured effective spectral radius 1.38 at step 0 vs the
            # 0.9 target, JOURNAL.md 2026-08-11). 20 extra iterations here
            # (cheap, O(n^2) each, no backward/optimizer call) converges
            # u/sigma_ema BEFORE any real training step, matching
            # EnergyDynamics' own "allow N steps for noise, don't wait
            # forever" warm-start convention.
            o_layers = self.o_proj if o_proj_depth > 1 else [self.o_proj]
            for _ in range(20):
                for idx, layer in enumerate(o_layers):
                    self._spectral_rescale_factor(layer, idx)

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

    def _spectral_rescale_factor(self, layer, idx: int) -> float:
        """One power-iteration step against `layer` alone (NOT the real
        data -- a persistent probe vector, decoupled from RMSNorm/clip/
        residual). forward(..., 0.0): zero-side-effect convention
        already used everywhere (evaluate(), the earlier spectral-
        radius diagnostic) -- no backward/optimizer call, so this costs
        one extra forward pass, nothing like an eigendecomposition.
        EMA-smoothed sigma (not the raw per-step estimate) is what's
        actually used -- see __init__'s own docstring for why
        (synaptogenesis-readiness).

        CORRECTION (was previously mislabeled "dominant singular
        value" everywhere in this method, including its own variable/
        parameter names like spectral_norm_target): this is forward-
        only iteration (u_{k+1} = layer(u_k)/||layer(u_k)||), which only
        converges to the top SINGULAR value when the underlying linear
        map is symmetric. For a real weight matrix (generically NOT
        symmetric, and its dominant eigenvalue is generically a complex
        pair, not real), this instead approximates something close to
        the SPECTRAL RADIUS (max |eigenvalue|), not the spectral norm --
        confirmed directly: for a random 16x16 Gaussian matrix, this
        iteration converges to ~3.43 while the true top singular value
        (np.linalg.svd) is ~7.13 and the true spectral radius
        (np.linalg.eigvals) is ~3.49 -- clearly tracking the latter, not
        the former. Left AS-IS here (this correction is comment-only,
        no behavior change) since spectral radius is actually the
        theoretically correct quantity for recurrent-dynamics stability
        anyway (spectral norm is a conservative upper bound on it) --
        but any existing spectral_norm_target tuning should be
        understood as having tuned against radius-like behavior, not
        norm, this whole time. See model/eval_eigenvalues.py for an
        EXACT (non-iterative, SVD/eigval-based) alternative if a
        precise answer is ever needed instead of this cheap per-step
        approximation."""
        eps = 1e-8
        u = self._spectral_u[idx]
        probe = Tensor(u.reshape(1, -1).astype(np.float32))
        raw = np.asarray(layer.forward(probe, 0.0).data).reshape(-1)
        sigma = float(np.linalg.norm(raw))
        self._spectral_u[idx] = raw / (sigma + eps)
        prev = self._spectral_sigma_ema[idx]
        ema = sigma if prev is None else (
            self.spectral_norm_ema_decay * prev + (1.0 - self.spectral_norm_ema_decay) * sigma)
        self._spectral_sigma_ema[idx] = ema
        return self.spectral_norm_target / max(ema, eps)

    def _l1_sparsity_split(self, layer, input_t: Tensor, lr: float, coef: float) -> Tensor:
        """Exact port of l1_sparsity_probe.py's own `_l1_sparsity_split`
        -- see l1_sparsity_coef's docstring in __init__ for the full
        rationale. A second, undamped forward call gives this term its
        own gradient path into `layer`'s weights, independent of (and
        not diluted by) the main-task damped forward call already made
        elsewhere in step()."""
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

    def step(self, x_window: np.ndarray, M_prev: np.ndarray,
             learning_rate: float, debug: bool = False) -> Tuple[np.ndarray, Tensor, Optional[Tensor]]:
        """One recurrence tick. x_window: [num_tiles, embed_width] numpy
        (a real, narrow per-tile input -- e.g. a token embedding, or
        zeros for "nothing here yet" -- mapped into the wide state by
        input_proj, a real trained layer, NOT tiled/repeated by the
        caller; see input_proj's own docstring in __init__ for why).
        M_prev: [num_tiles, state_width] numpy, DETACHED (no BPTT,
        matching ToyTileRecurrence's own design). Returns (M_new numpy [num_tiles,
        state_width], logits Tensor [num_tiles, vocab_size], aux_loss).

        `debug=True`: records per-stage value statistics (mean/std/min/
        max, plus the fraction of elements the hard clip actually
        touches) into `self._last_step_debug_stats` (a dict, stage name
        -> stats dict) -- does NOT change the return signature (every
        existing caller unpacks a 3-tuple positionally), so this is
        purely additive/inspectable after the call. Default False (was
        previously an unused, dead `debug: bool = True` param -- this
        is the first real implementation) to avoid the extra numpy
        reduction overhead in the normal training hot path. Built to
        diagnose why fully-dense connectivity fails to train at all
        (JOURNAL.md 2026-08-10) while the usual sparse echo-network
        succeeds -- compare `_last_step_debug_stats` across a dense and
        a sparse model at the same seed/step to find exactly where
        their value distributions diverge."""
        def _stats(name, arr):
            a = np.asarray(arr)
            return {"mean": float(a.mean()), "std": float(a.std()),
                   "min": float(a.min()), "max": float(a.max()),
                   "abs_max": float(np.abs(a).max())}

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
            gate_logit = self.gate_x_proj.forward(x_normed, learning_rate) \
                + self.gate_m_proj.forward(m_normed, learning_rate)
            gate_raw = sigmoid_tensor(gate_logit)
            gate = self.gate_floor + (1.0 - 2.0 * self.gate_floor) * gate_raw
            qkv_source = gate * x_normed + (1.0 - gate) * m_normed
            if debug:
                dbg["gate"] = _stats("gate", gate.data)
        else:
            qkv_source = x_normed + m_normed
        if debug:
            dbg["qkv_source"] = _stats("qkv_source", qkv_source.data)

        # 2. Gaussian Attention (or bypass -- see use_attention docstring)
        if self.use_attention:
            q = self.q_proj.forward(qkv_source, learning_rate)
            k = self.k_proj.forward(qkv_source, learning_rate)
            v = self.v_proj.forward(qkv_source, learning_rate)
            sigmas = exp(self.log_sigmas)
            if debug:
                dbg["q"] = _stats("q", q.data)
                dbg["k"] = _stats("k", k.data)
                dbg["v"] = _stats("v", v.data)

            attn = gaussian_attention(q, k, v, self.centers, sigmas,
                                      num_cpus=self.num_cpus, causal=False)
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
                l1_terms.append(self._l1_sparsity_split(self.gate_x_proj, x_normed, learning_rate, self.l1_sparsity_coef))
                l1_terms.append(self._l1_sparsity_split(self.gate_m_proj, m_normed, learning_rate, self.l1_sparsity_coef))
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
                l1_terms.append(self._l1_sparsity_split(self.o_proj, o_proj_input, learning_rate, self.l1_sparsity_coef))
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term
        # Forward clip on the residual UPDATE itself, not just the final
        # state (see clip below) -- found NECESSARY, not just belt-and
        # -suspenders: gradient clipping alone (clip_grad_norm_ on the
        # plain-Tensor Adam params, train_tile_curriculum.py) only bounds
        # the SIZE of each individual step, not the CUMULATIVE drift from
        # many small unclipped-in-effect steps compounding in the same
        # direction over hundreds of steps -- confirmed directly: with
        # only gradient clipping, dense connectivity's NaN divergence
        # moved from step ~275 to step ~450 but still happened.
        # attn_o_proj was already measured reaching |9.47| BEFORE the
        # state's own post-residual-and-norm clip ever saw it. Same
        # straight-through bypass-autograd convention as the state clip
        # below (this is about bounding FORWARD magnitude, not shaping
        # the backward gradient -- that's clip_grad_norm_'s job).
        attn.data = np.clip(attn.data, -self.clip_range, self.clip_range)
        if debug:
            dbg["attn_o_proj"] = _stats("attn_o_proj", attn.data)
        if self.magnitude_penalty_coef > 0:
            mag_penalty = reduce_sum(power(attn, 2)) * (self.magnitude_penalty_coef / attn.data.size)
            aux_loss = mag_penalty if aux_loss is None else aux_loss + mag_penalty

        # 3. Residual & Hard Bounding
        M_new_t = Tensor(M_prev.astype(np.float32)) + attn
        M_new_t = rmsnorm_tensor(M_new_t, self.state_ln, self.rms_eps)
        if debug:
            dbg["pre_clip"] = _stats("pre_clip", M_new_t.data)
            dbg["clip_fraction"] = float(np.mean(np.abs(M_new_t.data) >= self.clip_range))

        # Hard clip bounds the state to avoid exploding activations over time.
        # Direct .data modification bypasses any autograd tracking for the clip.
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
            # MUST run before any other lm_head.forward() call this step
            # (including the L1 split below) and before the real logits
            # forward -- disldo_cls layers cache their input on the C++
            # side between forward()/backward() (a single most-recent
            # -call slot, not scoped per Tensor node), so a probe call
            # sandwiched AFTER the real forward would clobber that cached
            # state with the wrong (probe) batch shape before the real
            # backward ever runs -- confirmed directly (a `(4,16) vs
            # (16,16)` broadcast crash in the real backward pass) when
            # this was first tried in the other order. Placing it first
            # guarantees the real `pooled`-shaped forward call below is
            # always the LAST lm_head.forward() before backward.
            probes = Tensor(np.eye(self.embed_width, dtype=np.float32))
            probe_out = self.lm_head.forward(probes, 0.0).data  # [embed_width, vocab_size]
            row_norms = np.sqrt((probe_out ** 2).sum(axis=0)) + self.rms_eps  # [vocab_size]
        if self.l1_sparsity_coef > 0.0:
            # lm_head's own direct escape route -- see l1_sparsity_coef's
            # docstring for why this can't just rely on q/k/v/o_proj's
            # terms (dL/d(pooled) is exactly 0 whenever W_lmhead=0).
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
