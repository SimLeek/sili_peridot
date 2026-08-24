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

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from .fake_quantize_torch import fake_quantize


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
                 clip_raw_delta: bool = True, include_scale_chain_rule: bool = False,
                 clip_pre_ci: bool = False, pre_ci_clip_value: Optional[float] = None,
                 scale_grad_diag_fn: Optional[Callable] = None,
                 use_magnitude_scale: bool = False, magnitude_scale_target: float = 16.0,
                 magnitude_correction_rate: float = 0.01,
                 fake_quantize_kind: Optional[str] = None,
                 scale_invariant_chain_rule: bool = False,
                 fake_quantize_stochastic: bool = False,
                 magnitude_rescale_ema_beta: Optional[float] = None,
                 magnitude_scale_both_axes: bool = False):
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
        two consistent.

        include_scale_chain_rule: default False -- REPRODUCES A REAL BUG
        found this session (see conversation): true_weight = w_stored *
        value_scale * output_scale, so the correct chain-rule gradient
        is dL/d(w_stored) = g*S (S = value_scale (x) output_scale,
        matching sili's real C++ non-deferred/quant formula, update_cw's
        own `-g*S` term, delta_csr_types.hpp) and dL/d(value_scale[r]) =
        sum_c(g[r,c]*w_stored[r,c]*output_scale[c]) -- this class has
        always used raw g directly for both instead (missing the S/
        w_stored*output_scale factor entirely), a genuinely different,
        simpler optimization dynamic than what sili actually runs.
        Kept default False (i.e. reproduce the historical bug) so every
        existing ablation result stays reproducible; True applies the
        real chain rule, to test directly whether THIS specific
        difference (not the clip, not batch-consistency -- both already
        investigated) is what makes sili's real engine underperform
        this same architecture in torch.

        use_magnitude_scale: default False -- direct measurement (this
        conversation) found value_scale/output_scale's OWN RMSprop
        gradient signal is ~65-88% cancelled by cross-row/cross-column
        sign disagreement among the per-synapse contributions that get
        summed into it (worst in the 128-wide q/k/v/o_proj layers), so
        it barely moves from its 1.0 init even in a fully-converged
        run -- a genuine rank-1 structural limit (project task #196),
        not a weak-signal problem. That leaves w_stored's own raw
        magnitude to do ALL the work of representing small weights,
        which is fine in fp32 but pushes the STORED CODE into a
        quantizer's coarse/subnormal region once the codec is FP4/FP8.
        This is a gradient-FREE fix instead of trying to un-cancel that
        signal: a pure reparametrization that keeps true_weight =
        w_stored*value_scale*output_scale algebraically IDENTICAL while
        moving magnitude from output_scale into w_stored (or back) each
        step, so w_stored's own per-column RMS tracks toward
        magnitude_scale_target -- chosen once, not learned, so it can't
        be cancelled. No effect at all in fp32 (same true_weight either
        way); the whole point is giving the STORED representation a
        favorable magnitude before quantization ever happens.

        clip_pre_ci: default False -- targets a DIFFERENT failure mode
        than clip_raw_delta (which clips the post-division update, AFTER
        ci already absorbed whatever g/contrib produced it). Multi-seed
        scale_chain_rule sweeps (this conversation) found ci sometimes
        takes one single-step jump of +70-80 straight to max_ci's
        ceiling (e.g. seed=1000: ~25.7->99.9 in ONE step), coincident
        with the query loss permanently collapsing to ln(vocab) (uniform
        guessing) a few hundred steps later -- clip_raw_delta doesn't
        prevent this because the damage is in what THAT one step's raw
        g/contrib do (to ci's EMA and, via g_for_w, to w_stored itself),
        not in the post-division magnitude. clip_pre_ci clips g and
        contrib to +-pre_ci_clip_value (defaults to max_abs_delta) BEFORE
        squaring into ci's EMA, capping how much any single anomalous
        step can move ci at all."""
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
        self.col_rms_ema: Optional[torch.Tensor] = None
        self.magnitude_scale_both_axes = magnitude_scale_both_axes
        self.row_rms_ema: Optional[torch.Tensor] = None

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

            # S: dL/d(w_stored) = g*S, dL/d(value_scale[r]) needs
            # w_stored*output_scale, dL/d(output_scale[c]) needs
            # w_stored*value_scale -- see include_scale_chain_rule's own
            # docstring above for the full derivation. g_for_w/g_row/g_col
            # (and their contrib counterparts) default to the historical
            # (buggy, raw-g) behavior when the flag is off, so every
            # existing ablation result stays bit-reproducible.
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
                bc2 = 1.0 - self.beta2 ** self.ci_step
                ci_hat = self.ci / bc2 if bc2 > 0 else self.ci
            else:
                ci_hat = self.ci

            # scale_invariant_chain_rule: default False. Both branches
            # above (chain_rule on OR off) share ci tracking RAW g^2 (see
            # g_for_ci above, unaffected by include_scale_chain_rule) --
            # but when chain_rule=True, numerator=g*S while ci is
            # calibrated to plain g^2, so raw=numerator/sqrt(ci_hat)
            # scales LINEARLY with S (not O(1)-normalized), making
            # Delta(true_weight)=S*Delta(w_stored) scale QUADRATICALLY
            # with S -- i.e. the effective learning rate for true_weight
            # silently collapses as S shrinks (or explodes as S grows).
            # Never surfaced before because every prior config left S
            # near its 1.0 init (value_scale/output_scale barely move --
            # see the cross-column-cancellation finding, this
            # conversation) -- magnitude_scale is the first mechanism
            # that deliberately drives S away from 1, exposing it.
            # Fix: compute raw from the RAW gradient g (properly self-
            # normalized by ci, which already tracks g^2), giving a
            # step that's a fixed size in TRUE_WEIGHT units regardless
            # of parametrization -- then convert to w_stored's own
            # units by dividing by S (the correct inverse chain rule),
            # so Delta(true_weight) = S * (eff_lr*raw/S) = eff_lr*raw,
            # independent of S entirely. Per direct instruction: this
            # (not a one-off rescale patch) is the fix -- remove the
            # "S stays near 1" assumption, don't just compensate for it
            # after the fact.
            numerator_for_ci_norm = -g if (self.scale_invariant_chain_rule and self.include_scale_chain_rule) else -g_for_w

            if self.use_momentum:
                self.m = self.momentum_beta1 * self.m + (1.0 - self.momentum_beta1) * numerator_for_ci_norm
                if self.bias_correct_ci:
                    bc1 = 1.0 - self.momentum_beta1 ** self.ci_step
                    numerator = (self.m / bc1 if bc1 > 0 else self.m)
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
                # Quantize IMMEDIATELY, not once at the very end of the
                # step -- the real production system keeps NO float
                # shadow of any weight (see TrueMultiDigitLayer's own
                # docstring: "a real hardware implementation has no
                # room for a hidden full-precision shadow"); every
                # update writes directly to the quantized representation.
                # Quantizing only once per step (as this code used to)
                # let w_stored silently accumulate float precision
                # between the RMSprop step and the magnitude-rescale
                # step below, which isn't how the real system behaves.
                self.w_stored = fake_quantize(self.w_stored, self.fake_quantize_kind,
                                              stochastic=self.fake_quantize_stochastic)

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
        """Gradient-free reparametrization: true_weight = w_stored *
        value_scale * output_scale is UNCHANGED by this (algebraically),
        only WHERE the magnitude lives changes -- moves each column's
        share of magnitude between w_stored and output_scale so
        w_stored's own per-column RMS drifts toward magnitude_scale_target
        (a fixed constant, so it can't be cancelled the way a gradient-
        summed target could be). Applies a DAMPED (magnitude_correction_
        rate) step toward the full correction each call rather than
        jumping all the way there, so it doesn't fight the RMSprop
        updates that just happened in the same step.

        ci (per-synapse RMSprop denominator) accumulates g^2+contrib^2
        from the RAW chain-rule gradient g, which is independent of how
        magnitude is split between w_stored/output_scale -- but the
        actual update numerator is g*S (S=value_scale*output_scale).
        Shrinking output_scale by k shrinks S by k, shrinking the
        numerator, while ci (calibrated to g^2, not (g*S)^2) doesn't
        track that -- silently damping w_stored's own effective step
        size every time this fires. Found empirically: without this,
        fp32 accuracy dropped 1.0->0.18 despite true_weight being
        algebraically unchanged. Rescaling ci by k^2 alongside keeps it
        calibrated to the new S regime -- ONLY needed when
        scale_invariant_chain_rule is off: that flag already decouples
        ci from S entirely (ci tracks raw g^2 and the S-dependence is
        removed via explicit division at apply time instead), so
        rescaling ci here too would double-correct and be wrong.

        magnitude_rescale_ema_beta: default None (use the instantaneous
        col_rms each call). With fake_quantize active, w_stored has
        ALREADY been quantized to a coarse grid by the time this runs
        (immediately above, in apply_pending_updates) -- measuring
        col_rms from that coarsely-rounded w_stored means the rescale
        target itself carries quantization noise, and since this fires
        EVERY step (not periodically), that noise feeds straight back
        into the next quantize call, compounding. Found empirically:
        fp32 (no quantization noise in the measurement) reaches a clean
        1.0 with this mechanism, but fake-fp8 collapses to 0.0 despite
        the SAME optimizer math. Set magnitude_rescale_ema_beta (e.g.
        0.9) to track an EMA of col_rms across steps instead, filtering
        out per-step quantization jitter from the signal driving k."""
        with torch.no_grad():
            instant_col_rms = torch.sqrt((self.w_stored ** 2).mean(dim=0) + self.eps)
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
            if self.scale_invariant_chain_rule:
                new_ci = self.ci
            else:
                new_ci = self.ci / (k.unsqueeze(0) ** 2)
            if (torch.isfinite(new_w).all() and torch.isfinite(new_output_scale).all()
                    and torch.isfinite(new_ci).all()):
                self.w_stored = new_w
                self.output_scale = new_output_scale
                self.ci = new_ci
                if self.fake_quantize_kind is not None:
                    # Same "no float shadow" reasoning as the RMSprop
                    # update site -- re-quantize immediately, this
                    # rescaled w_stored is what actually gets "stored".
                    self.w_stored = fake_quantize(self.w_stored, self.fake_quantize_kind,
                                                  stochastic=self.fake_quantize_stochastic)

            if self.magnitude_scale_both_axes:
                # SAME treatment on the ROW axis (value_scale) -- per
                # direct question: this whole mechanism only ever
                # touched output_scale (columns), value_scale (rows)
                # was left to its own cancellation-limited RMSprop
                # signal and barely moves (see cross-row-cancellation
                # finding, this conversation). Nothing about the
                # mechanism is column-specific; applying it symmetrically
                # gives the full rank-1 (value_scale (x) output_scale)
                # envelope the same magnitude-matching ability on both
                # axes, not just one -- and is a useful data point for
                # whether rank-2 (task #196) is worth the extra cost:
                # if a second SHARED axis doesn't help even when it's
                # allowed to move freely, more basis vectors on the same
                # axis structure are unlikely to help more.
                row_rms_instant = torch.sqrt((self.w_stored ** 2).mean(dim=1) + self.eps)
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
                if self.scale_invariant_chain_rule:
                    new_ci_row = self.ci
                else:
                    new_ci_row = self.ci / (k_row.unsqueeze(1) ** 2)
                if (torch.isfinite(new_w_row).all() and torch.isfinite(new_value_scale).all()
                        and torch.isfinite(new_ci_row).all()):
                    self.w_stored = new_w_row
                    self.value_scale = new_value_scale
                    self.ci = new_ci_row
                    if self.fake_quantize_kind is not None:
                        self.w_stored = fake_quantize(self.w_stored, self.fake_quantize_kind,
                                                      stochastic=self.fake_quantize_stochastic)

    def _scale_update(self, name: str, g_agg: torch.Tensor, contrib_agg: torch.Tensor,
                       eff_lr: float, step: int) -> None:
        scale = getattr(self, name)
        state = getattr(self, f"{name}_state")
        if self.scale_invariant_chain_rule:
            # Multiplicative (log-space) step instead of additive: an
            # ADDITIVE step of size ~eff_lr is a huge RELATIVE change
            # once scale has shrunk far below 1 (which magnitude_scale
            # deliberately does) and negligible once it's grown large --
            # same "assumes scale stays near 1" bug as w_stored's own
            # update (see that fix's docstring above), just on scale
            # itself. d(loss)/d(log(scale)) = d(loss)/d(scale)*scale =
            # g_agg*scale (chain rule through scale=exp(log_scale));
            # RMSprop-normalizing THAT keeps the step a fixed RELATIVE
            # (percentage) size regardless of scale's own magnitude.
            # Bonus: scale can never cross zero this way (exp()>0),
            # unlike the additive step.
            log_grad = g_agg * scale
            log_contrib = contrib_agg * scale
            new_state = self.beta2 * state + (1.0 - self.beta2) * (log_grad * log_grad + log_contrib * log_contrib)
            bias_correction = 1.0 - self.beta2 ** step
            state_hat = new_state / bias_correction if bias_correction > 0 else new_state
            log_step = eff_lr * log_grad / (torch.sqrt(state_hat) + self.eps)
            new_scale = scale * torch.exp(-log_step)
        else:
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
