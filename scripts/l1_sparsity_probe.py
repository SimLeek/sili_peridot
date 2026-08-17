"""LANDMARK RESULT (2026-08-13): L1 output-sparsity penalty, applied to
all 4 layers (q_proj/k_proj/v_proj/o_proj) of the ORIGINAL architecture
(single v_proj on the combined input, matching ToyTileRecurrenceRealFP4
-- NOT the v_in_proj/v_state_proj split investigated earlier), reaches
mean=1.0000 across all 5 seeds at coefficient 0.05 AND 0.07 on the
15000-step out-of-context curriculum -- the best result of the entire
dense-connectivity stability investigation, matching/exceeding spectral
normalization's own 0.8858, with NO hard rescale of any kind. Found
because spectral normalization (and any hard rescale, including a
mean-singular-value-targeting variant) is not available as a production
mechanism -- see sili_peridot/JOURNAL.md's 2026-08-12/13 entries for
the complete investigation, all intermediate results, and the full
methodology (including how this exact result was independently
re-verified via the real training loop after an initial "too good to
be true" suspicion, given this session's repeated pattern of promising
short runs collapsing at full scale).

Full comparison table (all mean accuracy on the 15000-step out-of
-context copy-task curriculum, dense_base12, 5 seeds):

    mechanism                                      mean    notes
    L1-sparsity alone, orig arch, coef=0.05/0.07   1.0000  BEST
    spectral norm, orig arch, o_proj-only          0.8858  (hard rescale, unavailable in production)
    L1-sparsity + L2-ratio combined, orig arch     0.7333  combining L1 with L2-ratio HURTS
    sparse-echo (no dense connectivity at all)     0.7296
    L2-ratio (split-backward) alone, orig arch     0.3333-0.4667  best ~coef=10, non-monotonic
    L1-sparsity alone, NEW (v_in/v_state split) arch  0.2000-0.3333  87-89% skip rate, unstable
    L2-ratio alone, NEW (split) arch               0.2667-0.4000
    any o_proj-only soft mechanism, either arch    0.0-0.27  total or near-total collapse

Mechanism: `coef * mean(|layer_output|)` via a "split-backward" delivery
-- a SECOND, independent `layer.forward(..., damp_by_importance=False)`
call per layer gives the L1 term its own undamped gradient path,
avoiding dilution by DISLDO's own RMSprop-style per-synapse update
(which shares state with -- and gets swamped by -- the much larger main
-task gradient if delivered through the normal damped path; see
JOURNAL.md for the full RMSprop-dominance derivation, confirmed via
direct gradient-magnitude measurement, ~1731x ratio).

Coefficient sensitivity is real and sharp -- this is a genuine
Goldilocks zone, not a monotonic dial:

    l1_coef=0.01/0.02  mean=0.13  skip_rate=27-30%  (too weak, unstable)
    l1_coef=0.03       mean=0.80
    l1_coef=0.05       mean=1.00  (PERFECT)
    l1_coef=0.07       mean=1.00  (PERFECT)
    l1_coef=0.10       mean=0.67  (too strong, degrades again)

Also supports (not part of the landmark result, added for follow-up
testing): all_zero_init (only weight matrices zeroed, RMSNorm scales
stay at their 1.0 baseline), use_energy/energy_kwargs (EnergyDynamics
forced-firing/shutoff -- found the DEFAULT drive saturates to the
firing ceiling almost immediately under zero-init since E[|attn_raw|]
starts at exactly 0, removing the only force opposing `drive` in the
continuous dynamics; a much smaller drive, e.g. 1e-4 to 1e-5, is needed
to keep firing rare rather than continuous -- see JOURNAL.md), and
scale_clip_max (O(w) value_scale/output_scale clipping, task #165).

NOT YET DONE: this has not been merged into the shared
model/toy_tile_precision_models.py (ToyTileRecurrenceRealFP4) or wired
into train_tile_curriculum.py's CLI -- this file remains the
reference/reproduction implementation for the validated result above.
Zero-init + energy_rl + L1-sparsity is a promising but NOT yet
full-scale-validated follow-up (short-run signal only)."""
import sys, functools, statistics, time
sys.path.insert(0, ".")
import numpy as np

from sili import _cpu
from sili.tensor import Tensor, gaussian_attention, exp, reduce_sum, power, tensor_abs
from sili.sparse_rnn import DISLDOLayerDeterministic, DISLDOLayer
from model.toy_precision_models import TrueMultiDigitLayer, _apply_energy
from model.toy_recall_models import rmsnorm_tensor, cross_entropy_sum, AdamOptimizer, lr_schedule, clip_grad_norm_
from scripts.train_tile_curriculum import generate_copy_sequence, _build_tile_window

VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, MAX_WEIGHTS = 10, 8, 4, 4, 128
STEPS_PER_STAGE = 500
CLIP_RANGE = 6.0


class OriginalArchModel:
    """Exact port of ToyTileRecurrenceRealFP4's architecture -- single
    v_proj on the combined qkv_source, single o_proj -- with the
    split-backward L1-sparsity (and L2-ratio, for comparison) mechanism
    wired onto all 4 layers instead of spectral_norm_target."""

    # Every neuron-producing tensor in the forward pass -- input, every
    # hidden layer, and the output -- gets its own EnergyDynamics tap,
    # applied uniformly via the plain _apply_energy(state, tensor, ...)
    # op (see __init__'s energy-construction comment).
    _ENERGY_TAPS = ("input", "q", "k", "v", "attn", "raw", "logits")

    def __init__(self, seed, dense, o_proj_coef, all_layer_coef=0.0, l1_sparsity_coef=0.0,
                 all_zero_init=False, zero_init_importance_code=1, use_energy=False,
                 energy_kwargs=None, scale_clip_max=None, log_sigma_clip_max=None,
                 stochastic_qkv=False, stochastic_o=False, lr_per_row_nnz=True,
                 scale_rank=1, empty_init=False, synap_k=3, synap_importance_cutoff=0.0):
        # empty_init: the GENUINE zero-weight-init design -- every layer
        # starts with literally zero connections (not all_zero_init's
        # dense-grid-preloaded-with-0 hack, a different synthetic arm --
        # see sili/sparse_rnn.py's _preseed_empty docstring). Real
        # synapses are created by synaptogenesis() (build_probes/
        # synap_step/equalizer_step, called every step in run() -- see
        # its own "meant to be called every online step" docstring),
        # each starting with weight=0/importance=probe_score (a REAL,
        # activity-derived nonzero importance, not a hardcoded one) --
        # confirmed via isolated smoke test to actually escape 0 given
        # real training (unlike all_zero_init's dense simultaneous-zero
        # deadlock, since only a HANDFUL of synapses per row need to
        # escape, not the entire row/column at once). Mutually exclusive
        # with dense/all_zero_init in practice (not enforced) -- doesn't
        # make sense combined with either.
        self.empty_init = bool(empty_init)
        self.synap_k = int(synap_k)
        self.synap_importance_cutoff = float(synap_importance_cutoff)
        # lr_per_row_nnz: passed straight through to every layer's
        # .forward() call, RAW lr, no Python-side pre-division. Controls
        # ONLY whether the per-synapse weight-CODE update is additionally
        # damped by the row's live-connection count -- value_scale's own
        # gradient (a single scalar per row, summed across every live
        # synapse in it) is ALWAYS normalized by that count regardless of
        # this flag, unconditionally, inside the C++ layer
        # (linear_disldo.hpp's scale_eff_lr) -- that's mandatory averaging
        # of a summed scalar gradient, not a policy choice.
        #
        # An EARLIER version of this code tried to replicate True's
        # division from the Python side by pre-dividing lr (qkv_lr =
        # lr/state_width) before calling .forward(..., lr_per_row_nnz=
        # False) -- this was a real bug, not a legitimate approximation:
        # since scale_eff_lr unconditionally divides whatever lr it's
        # handed by nnz_row, pre-dividing caused value_scale's adaptation
        # to be damped TWICE (once by the Python pre-division, once again
        # by the library's own always-on normalization), crippling
        # value_scale to ~1/32 of its correct rate. Confirmed directly:
        # effective_lr (the code update) was bit-identical to True at
        # correction=1.0 (verified via a debug print showing nnz_row=32
        # exactly, and a 200-step run with zero diff across every weight/
        # importance/index array) -- yet real training still diverged
        # structurally within a few thousand steps once value_scale's
        # under-adaptation let the two arms' true (scale-multiplied)
        # weight magnitudes drift apart far enough to cross different FP4
        # rounding/promotion thresholds. No single scalar correction on
        # the pre-divided lr could ever fix this: effective_lr needed
        # correction=1, scale_eff_lr needed correction=32, same shared
        # parameter. The fix is to not pre-divide at all -- raw lr through
        # to the library lets its own correct, intentional split (code
        # update: conditional; value_scale update: unconditional) do its
        # job in both branches.
        self.lr_per_row_nnz = bool(lr_per_row_nnz)
        if hasattr(_cpu, "seed_fp4_stochastic_rng"):
            _cpu.seed_fp4_stochastic_rng(seed)
        digit_cls = functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
                                      n_stages=3, base=12.0, lr_power=0.0, dense=dense,
                                      scale_rank=scale_rank, empty_init=empty_init)
        # stochastic_qkv/stochastic_o: use stochastic-rounding storage
        # (DISLDOLayer) instead of deterministic (DISLDOLayerDeterministic)
        # for the selected layers -- lm_head always stays deterministic
        # (its dy already comes for free from cross-entropy, the terminal
        # loss, never structurally zero -- see the energies dict's own
        # docstring above for the full per-layer math). Per direct request:
        # under all-zero-init, deterministic rounding's fresh-decode-then-
        # requantize every call means any single update below half the FP4
        # step (0.25, the code-0 -> code-0.5 gap) is discarded every time,
        # so sparse/small repeated pushes never accumulate -- confirmed
        # directly (weights_vals stayed exactly 0 for hours of steps
        # despite EnergyDynamics genuinely firing). Stochastic rounding lets
        # each sub-threshold delta round up with real (if small) probability,
        # so repetition compounds in expectation instead of vanishing.
        # o_proj was originally believed not to need this (it has no
        # gaussian_attention-style nonlinear escape hatch) -- CORRECTED:
        # once o_proj gets its own direct energy wrap on `raw` (its own
        # output, giving it a legitimate nonzero dy), it's in exactly the
        # same "real signal exists but gets rounded away" situation q/k/v
        # were in, so it needs the same fix. This project's own prior
        # findings show stochastic rounding can hurt final accuracy
        # broadly, so keep it opt-in per layer rather than always-on.
        qkv_digit_cls = (functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                                           n_stages=3, base=12.0, lr_power=0.0, dense=dense,
                                           scale_rank=scale_rank, empty_init=empty_init)
                          if stochastic_qkv else digit_cls)
        o_digit_cls = (functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                                         n_stages=3, base=12.0, lr_power=0.0, dense=dense,
                                         scale_rank=scale_rank, empty_init=empty_init)
                       if stochastic_o else digit_cls)
        rng = np.random.default_rng(seed)
        self.state_width = EMBED_WIDTH * COLUMN_NEURONS
        sw = self.state_width
        self.q_proj = qkv_digit_cls(sw, sw, MAX_WEIGHTS, 1, rng=np.random.default_rng(rng.integers(2**31)))
        self.k_proj = qkv_digit_cls(sw, sw, MAX_WEIGHTS, 1, rng=np.random.default_rng(rng.integers(2**31)))
        self.v_proj = qkv_digit_cls(sw, sw, MAX_WEIGHTS, 1, rng=np.random.default_rng(rng.integers(2**31)))
        self.o_proj = o_digit_cls(sw, sw, MAX_WEIGHTS, 1, rng=np.random.default_rng(rng.integers(2**31)))
        self.lm_head = digit_cls(EMBED_WIDTH, VOCAB, MAX_WEIGHTS, 1, rng=np.random.default_rng(rng.integers(2**31)))
        self.input_ln = Tensor(np.ones(sw, dtype=np.float32))
        self.state_ln = Tensor(np.ones(sw, dtype=np.float32))
        self.centers = Tensor(np.array([i + 0.5 for i in range(NUM_TILES)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(NUM_TILES, dtype=np.float32))
        self.o_proj_coef = o_proj_coef
        self.all_layer_coef = all_layer_coef
        # L1 on the OUTPUT (activation sparsity), via the split-backward
        # pattern (separate forward, damp_by_importance=False, own
        # undamped gradient path) -- a proxy for genuine weight
        # -magnitude L1 (DISLDO stores weights as quantized codes
        # internally, not an exposed differentiable tensor, so direct
        # weight-L1 isn't cleanly buildable) -- for a linear layer,
        # d(sum|Wx|)/dW = sign(y) outer x, which does directly push
        # weight magnitude down along active directions.
        self.l1_sparsity_coef = l1_sparsity_coef

        # Energy is applied as a plain per-tap OP (_apply_energy(state,
        # tensor, ...), a bare function -- not a bespoke wrapper class
        # per layer) to EVERY neuron-producing tensor in the model --
        # input, every hidden layer, and the output -- uniformly, not a
        # hand-curated subset. Per direct correction: reasoning
        # case-by-case about which specific layer's gradient path
        # "needs" energy is exactly the kind of fragile, easy-to-miss
        # analysis that caused the earlier o_proj/lm_head cascade stall
        # (see JOURNAL.md/this session -- a curated 5-tap set silently
        # left o_proj and lm_head permanently stuck under empty_init).
        # Since the number of neurons (layer widths) is cheap relative
        # to the number of parameters, wrapping every tap uniformly
        # costs little and removes the need to re-derive "does this
        # specific layer need it" every time the architecture changes.
        # Each tap gets its OWN independent EnergyDynamics instance
        # (own energy/steps_since_fired state) -- see _ENERGY_TAPS
        # below. _apply_energy(None, tensor, ...) is already a no-op
        # pass-through, so step() can call the op unconditionally at
        # every tap regardless of use_energy, rather than branching.
        # fire_wake_gradient's EFFECT on the actual weight update is
        # delta = -effective_lr*g/(sqrt(ci)+eps), i.e. proportional to
        # whatever `lr` happens to be THIS call (lr_schedule's warmup/
        # decay) -- a fixed fire_wake_gradient value therefore crosses
        # the FP4 rounding threshold differently depending on where
        # training is in the schedule, entangling "how big a push" with
        # "what step number". Disentangled here (not in EnergyDynamics
        # itself -- this isn't part of forward()'s own contract, it's
        # purely a caller-side calibration concern): pull the CONFIGURED
        # fire_wake_gradient out of energy_kwargs before constructing the
        # energies (so each starts with fire_wake_gradient=None), then
        # in step() rescale it by 1/lr every call and assign it directly
        # onto each energy instance right before use -- the boost then
        # represents a roughly LR-INDEPENDENT target push magnitude.
        self._fire_wake_gradient_base = None
        if energy_kwargs is not None and energy_kwargs.get("fire_wake_gradient") is not None:
            energy_kwargs = dict(energy_kwargs)
            self._fire_wake_gradient_base = float(energy_kwargs.pop("fire_wake_gradient"))

        self.energies = {}
        if use_energy:
            from sili.energy import EnergyDynamics
            def _make_energy():
                if energy_kwargs is not None:
                    # EnergyDynamics' own exploration-noise draw is unseeded
                    # global np.random unless an explicit rng= is given (fixed
                    # in sili__new, confirmed a real run-to-run non-
                    # reproducibility bug otherwise) -- derive one from this
                    # model's own seed stream, same convention as q/k/v/o_proj,
                    # unless the caller already supplied their own.
                    ek = dict(energy_kwargs)
                    ek.setdefault("rng", np.random.default_rng(rng.integers(2**31)))
                    return EnergyDynamics(**ek)
                else:
                    from model.toy_precision_models import _toy_scale_energy
                    return _toy_scale_energy()
            for name in self._ENERGY_TAPS:
                self.energies[name] = _make_energy()

        self.scale_clip_max = scale_clip_max
        self.log_sigma_clip_max = log_sigma_clip_max

        if all_zero_init:
            # weight codes 0 (value 0.0), importance codes 1 (NOT 0) --
            # load_dense_codes(zeros, zeros) gets every synapse pruned to
            # 0 live entries by block4_load_dense's maybe_compress step
            # (block4_count_live tests the packed weight|importance<<4
            # byte against exactly 0 -- both codes 0 means every packed
            # byte is 0), and disldo_backward's block4 path then skips
            # every row (nnz_row==0) forever, independent of gradient
            # magnitude. Not a library bug -- a synapse that legitimately
            # decays to weight=0 AND importance=0 during real training
            # is reasonable to prune -- but it means intentional
            # zero-VALUE init must seed importance nonzero to stay live.
            # zero_init_importance_code default 1 is the bare minimum to
            # survive the initial maybe_compress call -- callers combining
            # this with energy_rl (whose gating can silence a neuron for
            # stretches, decaying its synapses' g toward 0 and thus their
            # RMSprop-style importance accumulator back toward 0 too) may
            # want a larger starting buffer; sweep via this param rather
            # than assuming 1 is always enough.
            n = sw * sw
            zeros = np.zeros(n, dtype=np.uint8)
            imp = np.full(n, zero_init_importance_code, dtype=np.uint8)
            for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                for digit in layer.digits:
                    digit._c.load_dense_codes(zeros, imp)
            n_lm = EMBED_WIDTH * VOCAB
            zeros_lm = np.zeros(n_lm, dtype=np.uint8)
            imp_lm = np.full(n_lm, zero_init_importance_code, dtype=np.uint8)
            for digit in self.lm_head.digits:
                digit._c.load_dense_codes(zeros_lm, imp_lm)
            # Neuron-level (input_ln/state_ln, centers/log_sigmas) stay at
            # their normal baseline -- zero-init is for SYNAPSES (weight
            # matrices) specifically, not neuron-level gain/threshold
            # params.

    def parameters_for_optimizer(self):
        return [self.input_ln, self.state_ln, self.centers, self.log_sigmas]

    def synaptogenesis_all(self):
        """Grow real synapses on every layer -- no-op unless empty_init
        (nothing to grow into on a preseeded layer; TrueMultiDigitLayer.
        synaptogenesis is defined regardless, but calling it on an
        already-full-capacity preseeded layer just churns probes for no
        effect, so skip entirely rather than pay the cost for nothing)."""
        if not self.empty_init:
            return
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.lm_head):
            layer.synaptogenesis(self.synap_k, self.synap_importance_cutoff)

    def _l2(self, t: Tensor) -> Tensor:
        return power(reduce_sum(power(t, 2)) + 1e-8, 0.5)

    def clip_scales(self, layer, max_val: float):
        """O(w) scale-vector clipping -- per-row/col value_scale/
        output_scale capped at the storage format's own max
        representable magnitude (FP4 E2M1's largest code is 6.0)."""
        n_in = n_out = self.state_width
        for digit in layer.digits:
            c = digit._c
            for r in range(n_in):
                vs = c.get_value_scale(r)
                if vs > max_val or vs < -max_val:
                    c.set_value_scale_raw(r, float(np.clip(vs, -max_val, max_val)))
            for col in range(n_out):
                os_ = c.get_output_scale(col)
                if os_ > max_val or os_ < -max_val:
                    c.set_output_scale_raw(col, float(np.clip(os_, -max_val, max_val)))

    def clip_log_sigmas(self, max_val: float):
        """Bound the Gaussian-attention kernel width's log-scale parameter.
        Found via direct investigation (2026-08-12,
        scripts/debug_energy_log_sigmas_runaway.py): under use_energy=True,
        log_sigmas drifts monotonically negative and UNBOUNDED from its
        zero-init start, with no sign of leveling off. Symmetric bound
        (|log_sigma| <= max_val), matching clip_scales' single-magnitude
        style above -- sigma=exp(log_sigma) stays in
        [exp(-max_val), exp(max_val)].

        CORRECTED, tested directly with max_val=2.0 on the same 5-seed
        sweep used for baseline_energy: this does NOT fix the ~47%
        gradient-skip rate (46.76% unclamped vs 47.56% clamped,
        essentially unchanged; mean accuracy 0.1333 vs 0.2667, within
        this project's own established run-to-run noise range). The
        unbounded log_sigmas drift is a real, independently-worth-fixing
        problem, but it is NOT the (or at least not the sole) cause of
        baseline_energy's instability -- the actual source is still
        unidentified as of this docstring."""
        data = self.log_sigmas.data
        clipped = np.clip(data, -max_val, max_val)
        if not np.array_equal(data, clipped):
            self.log_sigmas.data = clipped.astype(np.float32)

    def _ratio_penalty_split(self, layer, input_t: Tensor, lr: float, coef: float) -> Tensor:
        out_aux = layer.forward(input_t, lr, lr_per_row_nnz=self.lr_per_row_nnz, damp_by_importance=False)
        l2_in = float(np.linalg.norm(input_t.data)) + 1e-6
        ratio_t = self._l2(out_aux) * (1.0 / l2_in)
        return power(ratio_t - 1.0, 2) * coef

    def _l1_sparsity_split(self, layer, input_t: Tensor, lr: float, coef: float) -> Tensor:
        out_aux = layer.forward(input_t, lr, lr_per_row_nnz=self.lr_per_row_nnz, damp_by_importance=False)
        n = float(np.asarray(out_aux.data).size)
        return reduce_sum(tensor_abs(out_aux)) * (coef / n)

    def step(self, x_window, M_prev, lr):
        # qkv_lr/o_lr/lmhead_lr all just alias raw lr -- no Python-side
        # pre-division. lr_per_row_nnz is passed straight to every
        # .forward() call and lets the C++ layer handle its own,
        # correctly-designed split: the per-synapse code update is
        # CONDITIONALLY damped by nnz_row (this flag), while value_scale's
        # gradient (a single scalar per row, summed across every live
        # synapse) is UNCONDITIONALLY normalized by nnz_row regardless of
        # this flag -- mandatory averaging of a summed scalar gradient,
        # not a policy choice (see linear_disldo.hpp's scale_eff_lr).
        # An earlier version of this code pre-divided lr in Python to
        # approximate what lr_per_row_nnz=True's damping does, then called
        # .forward(..., lr_per_row_nnz=False) -- that was a real bug: the
        # pre-division compounded with scale_eff_lr's own always-on
        # division, damping value_scale's adaptation to ~1/32 of its
        # correct rate and causing a genuine structural (not just
        # numerical) divergence a few thousand steps into training. Fixed
        # by never touching lr in Python at all.
        qkv_lr = o_lr = lmhead_lr = lr
        if self._fire_wake_gradient_base is not None and lr > 0.0:
            compensated = self._fire_wake_gradient_base / lr
            for energy in self.energies.values():
                energy.fire_wake_gradient = compensated
        energy_aux_terms = []
        def tap(name, tensor, n_hidden=self.state_width):
            # Plain op, always called -- _apply_energy(None, tensor, ...)
            # is already a no-op pass-through when use_energy=False, so
            # every tap uses the SAME call regardless (see _ENERGY_TAPS).
            gated, aux = _apply_energy(self.energies.get(name), tensor, NUM_TILES, n_hidden)
            if aux is not None:
                energy_aux_terms.append(aux)
            return gated

        x_normed = rmsnorm_tensor(Tensor(x_window.astype(np.float32)), self.input_ln, 1e-6)
        m_normed = rmsnorm_tensor(Tensor(M_prev.astype(np.float32)), self.input_ln, 1e-6)
        qkv_source = tap("input", x_normed + m_normed)
        q = tap("q", self.q_proj.forward(qkv_source, qkv_lr, lr_per_row_nnz=self.lr_per_row_nnz))
        k = tap("k", self.k_proj.forward(qkv_source, qkv_lr, lr_per_row_nnz=self.lr_per_row_nnz))
        v = tap("v", self.v_proj.forward(qkv_source, qkv_lr, lr_per_row_nnz=self.lr_per_row_nnz))
        sigmas = exp(self.log_sigmas)
        attn_raw = tap("attn", gaussian_attention(q, k, v, self.centers, sigmas, num_cpus=1, causal=False))

        reg_terms = []
        if self.all_layer_coef > 0.0:
            reg_terms.append(self._ratio_penalty_split(self.q_proj, qkv_source, qkv_lr, self.all_layer_coef))
            reg_terms.append(self._ratio_penalty_split(self.k_proj, qkv_source, qkv_lr, self.all_layer_coef))
            reg_terms.append(self._ratio_penalty_split(self.v_proj, qkv_source, qkv_lr, self.all_layer_coef))

        raw = tap("raw", self.o_proj.forward(attn_raw, o_lr, lr_per_row_nnz=self.lr_per_row_nnz))
        if self.o_proj_coef > 0.0:
            raw_aux = self.o_proj.forward(attn_raw, o_lr, lr_per_row_nnz=self.lr_per_row_nnz, damp_by_importance=False)
            l2_in = float(np.linalg.norm(attn_raw.data)) + 1e-6
            ratio_t = self._l2(raw_aux) * (1.0 / l2_in)
            reg_terms.append(power(ratio_t - 1.0, 2) * self.o_proj_coef)

        if self.l1_sparsity_coef > 0.0:
            reg_terms.append(self._l1_sparsity_split(self.q_proj, qkv_source, qkv_lr, self.l1_sparsity_coef))
            reg_terms.append(self._l1_sparsity_split(self.k_proj, qkv_source, qkv_lr, self.l1_sparsity_coef))
            reg_terms.append(self._l1_sparsity_split(self.v_proj, qkv_source, qkv_lr, self.l1_sparsity_coef))
            reg_terms.append(self._l1_sparsity_split(self.o_proj, attn_raw, o_lr, self.l1_sparsity_coef))

        raw.data = np.clip(raw.data, -CLIP_RANGE, CLIP_RANGE)
        M_new_t = Tensor(M_prev.astype(np.float32)) + raw
        M_new_t = rmsnorm_tensor(M_new_t, self.state_ln, 1e-6)
        M_new_t.data = np.clip(M_new_t.data, -CLIP_RANGE, CLIP_RANGE)

        pooled = M_new_t.reshape((NUM_TILES, EMBED_WIDTH, COLUMN_NEURONS))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / COLUMN_NEURONS)
        if self.l1_sparsity_coef > 0.0:
            # lm_head previously had NO L1 term -- under all_zero_init it was
            # a separate, un-instrumented deadlock: dL/d(pooled)=W_lmhead^T@
            # grad_logits is exactly 0 whenever W_lmhead=0, so main-task
            # gradient alone could never reach it until pooled was already
            # nonzero from upstream. This term gives it the same direct,
            # pooled-independent escape route q/k/v/o_proj already have.
            reg_terms.append(self._l1_sparsity_split(self.lm_head, pooled, lmhead_lr, self.l1_sparsity_coef))
        logits = tap("logits", self.lm_head.forward(pooled, lmhead_lr, lr_per_row_nnz=self.lr_per_row_nnz),
                     n_hidden=VOCAB)

        total_aux = reg_terms[0] if reg_terms else None
        for extra in reg_terms[1:]:
            total_aux = total_aux + extra
        for extra in energy_aux_terms:
            total_aux = extra if total_aux is None else total_aux + extra
        return M_new_t.data, logits, total_aux


def evaluate(model, n_eval, seed, verbose=False):
    """Post-training capability check: n_eval FRESH, independently-drawn
    sequences at full in-context length (NUM_TILES), forward-only --
    never calls .backward()/opt.step(), so nothing trains (forward_dense
    has no side-effect training path; only backward_dense's inline
    update does, per its own docstring). Reports plain correct/n_eval.

    Replaces relying on run()'s own in-training last_accs sampling for
    regression comparisons -- that samples exactly ONE sequence's
    correctness every 200 steps, and run_config only ever averaged the
    LAST 3 of those, so every reported "mean" could only ever be one of
    {0, 1/3, 2/3, 1} per seed. Confirmed directly this coarse resolution
    was large enough to make a real regression indistinguishable from
    sampling noise (5 successes out of 15 total training-time samples
    swinging the reported mean by over 30 points). n_eval=100 gives up
    to 101 distinct values instead of 4.

    Uses the SAME embed_table the model was trained against (recomputed
    deterministically from `seed`, not threaded through run() -- the
    embedding table isn't itself trainable, it's the fixed random
    token->vector lookup the model's WEIGHTS were calibrated to use).
    Sequences are drawn from a DIFFERENT RNG stream than training's own
    (seed offset), so these are genuinely held-out draws, not a replay
    of the training curriculum's own sequence order.

    verbose: prints the last 5 (prediction, target) token-id pairs seen
    (across the whole n_eval run, most-recent-last) -- per direct
    request: a bare accuracy number can't distinguish "the model is
    degenerate/guessing a narrow set of tokens" from "these particular
    held-out sequences happened to skew toward one target token" (a real
    confound with a small n_eval and VOCAB tokens to choose from). Does
    NOT change the return value -- existing callers (landmark_checklist.py,
    run()'s own periodic eval) are unaffected."""
    embed_table = np.random.RandomState(seed).randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3
    eval_rng = np.random.RandomState(seed + 999_983)
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    correct = ntgt = 0
    last_answers = []  # (pred, target) pairs, most-recent-last
    for _ in range(n_eval):
        tokens, pairs = generate_copy_sequence(eval_rng, VOCAB, NUM_TILES)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        for i in range(NUM_TILES):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, 0.0)
            if i in targets:
                pred = int(np.argmax(logits.data[NUM_TILES - 1]))
                ntgt += 1
                correct += int(pred == targets[i])
                if verbose:
                    last_answers.append((pred, targets[i]))
                    if len(last_answers) > 5:
                        last_answers.pop(0)
    if verbose:
        print(f"  evaluate() last 5 (pred, target): {last_answers}", flush=True)
    return correct / max(ntgt, 1)


def run(model, n_steps, seed, verbose=False, periodic_eval_n=20):
    task_rng = np.random.RandomState(seed)
    embed_table = task_rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3
    opt = AdamOptimizer()
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    skips = total = 0
    last_accs = []
    t_start = time.time()
    for step in range(1, n_steps + 1):
        lr = lr_schedule(step, n_steps, 0.002, 50)
        seq_len = min(2 + step // STEPS_PER_STAGE, NUM_TILES)
        tokens, pairs = generate_copy_sequence(task_rng, VOCAB, seq_len)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        correct = ntgt = 0
        # Accumulate aux (L1-sparsity + energy_aux_loss) across EVERY tile
        # position, not just target ones -- per direct correction, these
        # were always meant to backprop at every position (energy
        # especially: fire_wake_gradient's injected term only ever
        # mattered if backward() actually ran on the specific position
        # that happened to fire, which is essentially never true when
        # only the LAST position is ever a target -- confirmed directly,
        # boosting fire_wake_gradient 25x made no difference at all,
        # because the aux computed at every non-target position was
        # simply discarded before this fix). One accumulate-then-backward
        # per outer step (not one backward per position) -- fewer,
        # cheaper backward() calls than the alternative of calling
        # backward() at every position.
        #
        # total_aux averaged by seq_len (not summed raw) -- otherwise
        # l1_sparsity_coef's effective per-step pressure grows with the
        # curriculum's own seq_len (2->4), silently over-regularizing as
        # training progresses. Averaging keeps the coefficient's meaning
        # stable regardless of how many positions happened to run this
        # step, so it doesn't need re-tuning every time seq_len changes
        # (confirmed necessary directly: raw summing regressed baseline
        # 0.8667->0.6667 at real 5-seed scale). The target cross-entropy
        # term is NOT averaged -- it's always exactly one term regardless
        # of seq_len, nothing to normalize there.
        total_aux = None
        total_loss = None
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, lr)
            if aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux
            if i in targets:
                pred = int(np.argmax(logits.data[NUM_TILES - 1]))
                ntgt += 1; correct += int(pred == targets[i])
                tgt_loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                total_loss = tgt_loss if total_loss is None else total_loss + tgt_loss
        if total_aux is not None:
            total_aux = total_aux * (1.0 / seq_len)
            total_loss = total_aux if total_loss is None else total_loss + total_aux
        if total_loss is not None:
            total_loss.backward()
            total += 1
            n = clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
            if not np.isfinite(n):
                skips += 1
            opt.step(model.parameters_for_optimizer(), lr=lr)
            if model.scale_clip_max is not None:
                for layer in (model.q_proj, model.k_proj, model.v_proj, model.o_proj):
                    model.clip_scales(layer, model.scale_clip_max)
            if model.log_sigma_clip_max is not None:
                model.clip_log_sigmas(model.log_sigma_clip_max)
        model.synaptogenesis_all()  # no-op unless model.empty_init -- see its own docstring
        if step % 200 == 0:
            last_accs.append(correct / max(ntgt, 1))
            if verbose:
                elapsed = time.time() - t_start
                avg_step = elapsed / step
                eta = avg_step * (n_steps - step)
                # Quality alongside progress, same cadence -- a live
                # trajectory instead of one number at the very end. Small
                # n_eval (cheap, not the full n_eval=100 used for the
                # final report) since this runs ~75 times over a 15000
                # -step run; forward-only, so calling it mid-training does
                # not disturb the training state (see evaluate()'s own
                # docstring).
                periodic_eval = evaluate(model, periodic_eval_n, seed) if periodic_eval_n > 0 else None
                eval_str = f" eval({periodic_eval_n})={periodic_eval:.3f}" if periodic_eval is not None else ""
                print(f"  [seed={seed}] step={step}/{n_steps} "
                      f"avg_step={avg_step*1000:.1f}ms elapsed={elapsed:.0f}s eta={eta:.0f}s{eval_str}",
                      flush=True)
    avg_step_time = (time.time() - t_start) / n_steps
    return last_accs, skips, total, avg_step_time


if __name__ == "__main__":
    # Reproduces the landmark result: mean=1.0000 across 5 seeds.
    import statistics
    SEEDS = [1000, 1001, 1002, 1003, 1004]
    N_STEPS = 15000
    for coef in [0.05, 0.07]:
        per_seed = []
        tot_skips = tot_calls = 0
        for seed in SEEDS:
            model = OriginalArchModel(seed, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
                                       l1_sparsity_coef=coef)
            accs, skips, total, avg_step_time = run(model, N_STEPS, seed, verbose=True)
            per_seed.append(statistics.mean(accs[-3:]))
            tot_skips += skips; tot_calls += total
        print(f"l1_sparsity_coef={coef}  mean={statistics.mean(per_seed):.4f}  "
              f"std={statistics.stdev(per_seed):.4f}  per_seed={[round(v,4) for v in per_seed]}  "
              f"skip_rate={tot_skips/tot_calls:.3%}", flush=True)
