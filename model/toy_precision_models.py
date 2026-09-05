"""
sili_peridot/model/toy_precision_models.py
────────────────────────────────────────────
Precision-matched Adam-vs-importance comparison harness (Adam+fake-FP4
vs real DISLDOLayer FP4/energy).
See docs/research/toy_precision_models.rst:toy_precision_models.module_overview.
"""

from __future__ import annotations

import functools

import numpy as np
from sili.energy import EnergyDynamics
from sili.sparse_rnn import DISLDOLayer, DISLDOLayer8, DISLDOLayer8AdaMax, DISLDOLayer8Resync, DISLDOLayer32
from sili.tensor import Tensor, _acc, banded_attention, silu

from .toy_recall_models import AdamOptimizer, DenseTensorLinear, rmsnorm_tensor


def _toy_scale_energy() -> EnergyDynamics:
    """See docs/research/toy_precision_models.rst:toy_scale_energy.calibration."""
    return EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01, density=0.05, p=0.3)


def _apply_energy(energy: EnergyDynamics | None, attn: Tensor, T: int, hidden: int) -> tuple[Tensor, Tensor | None]:
    """Shared use_energy toggle helper (flatten/unflatten around EnergyDynamics), no-op when energy is None.
    See docs/research/toy_precision_models.rst:real_fp4_layer.inline_training."""
    if energy is None:
        return attn, None
    gated, aux_loss, _actual_p = energy(attn.reshape((T * hidden,)))
    return gated.reshape((T, hidden)), aux_loss


# See docs/research/toy_precision_models.rst:fake_quantize_fp4.straight_through_qat.
_FP4_POSITIVE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_TABLE = np.concatenate([-_FP4_POSITIVE[:0:-1], _FP4_POSITIVE]).astype(np.float32)
FP4_MAX = 6.0


def fake_quantize_fp4(w: Tensor) -> Tensor:
    """See docs/research/toy_precision_models.rst:fake_quantize_fp4.straight_through_qat."""
    row_scale = np.max(np.abs(w.data), axis=-1, keepdims=True) / FP4_MAX
    row_scale = np.maximum(row_scale, 1e-8)
    scaled = w.data / row_scale
    idx = np.argmin(np.abs(scaled[..., None] - FP4_TABLE[None, None, :]), axis=-1)
    quantized = (FP4_TABLE[idx] * row_scale).astype(np.float32)
    out = Tensor(quantized, (w,), "fake_quantize_fp4", w.backend)

    def _bwd():
        if out.grad is not None:
            _acc(w, out.grad)  # straight-through: identity gradient onto w

    out._backward = _bwd
    return out


class ArtificialFP4Linear:
    """fp32 master weight, matmuls against its own fake_quantize_fp4-quantized value every forward.
    See docs/research/toy_precision_models.rst:fake_quantize_fp4.straight_through_qat."""

    def __init__(self, in_features: int, out_features: int, scale: float = 0.1):
        self.weight = Tensor((np.random.randn(in_features, out_features) * scale).astype(np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return x @ fake_quantize_fp4(self.weight)

    def parameters(self) -> list[Tensor]:
        return [self.weight]


class _ArtificialFP4Layer:
    def __init__(self, hidden: int, mlp_hidden: int, use_energy: bool):
        self.q_proj = ArtificialFP4Linear(hidden, hidden)
        self.k_proj = ArtificialFP4Linear(hidden, hidden)
        self.v_proj = ArtificialFP4Linear(hidden, hidden)
        self.o_proj = ArtificialFP4Linear(hidden, hidden)
        self.gate_proj = ArtificialFP4Linear(hidden, mlp_hidden)
        self.up_proj = ArtificialFP4Linear(hidden, mlp_hidden)
        self.down_proj = ArtificialFP4Linear(mlp_hidden, hidden)
        self.energy = _toy_scale_energy() if use_energy else None
        self.input_ln = Tensor(np.ones(hidden, dtype=np.float32))
        self.post_ln = Tensor(np.ones(hidden, dtype=np.float32))

    def parameters(self) -> list[Tensor]:
        params = [self.input_ln, self.post_ln]
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.gate_proj, self.up_proj, self.down_proj):
            params += layer.parameters()
        return params


class ToySmallTransformerArtificialFP4:
    """Adam arm of the precision-matched comparison (ArtificialFP4Linear + ordinary AdamOptimizer).
    See docs/research/toy_precision_models.rst:toy_precision_models.module_overview
    and real_fp4_layer.inline_training."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        use_energy: bool = False,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
    ):
        self.hidden = hidden
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.layers = [_ArtificialFP4Layer(hidden, mlp_hidden, use_energy) for _ in range(n_layers)]
        self.lm_head = ArtificialFP4Linear(hidden, vocab_size)

    def parameters(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params + self.lm_head.parameters()

    def forward(self, embedded: np.ndarray) -> tuple[Tensor, Tensor | None]:
        T = embedded.shape[0]
        x = Tensor(embedded.astype(np.float32))
        aux_loss_total = None
        for layer in self.layers:
            normed = rmsnorm_tensor(x, layer.input_ln, self.rms_eps)
            q = layer.q_proj.forward(normed)
            k = layer.k_proj.forward(normed)
            v = layer.v_proj.forward(normed)
            attn = banded_attention(q, k, v, half_bandwidth=T, num_cpus=self.num_cpus, causal=True)
            attn, aux_loss = _apply_energy(layer.energy, attn, T, self.hidden)
            aux_loss_total = (
                aux_loss
                if aux_loss_total is None
                else (aux_loss_total if aux_loss is None else aux_loss_total + aux_loss)
            )
            attn = layer.o_proj.forward(attn)
            x = x + attn
            normed2 = rmsnorm_tensor(x, layer.post_ln, self.rms_eps)
            gate = layer.gate_proj.forward(normed2)
            up = layer.up_proj.forward(normed2)
            mlp_out = layer.down_proj.forward(silu(gate) * up)
            x = x + mlp_out
        logits = self.lm_head.forward(x)
        return logits, aux_loss_total


class _RealFP4Layer:
    """See docs/research/toy_precision_models.rst:real_fp4_layer.inline_training."""

    def __init__(
        self, hidden: int, mlp_hidden: int, max_weights: int, use_energy: bool, num_cpus: int, disldo_cls=DISLDOLayer
    ):
        self.q_proj = disldo_cls(hidden, hidden, max_weights, num_cpus)
        self.k_proj = disldo_cls(hidden, hidden, max_weights, num_cpus)
        self.v_proj = disldo_cls(hidden, hidden, max_weights, num_cpus)
        self.o_proj = disldo_cls(hidden, hidden, max_weights, num_cpus)
        self.gate_proj = disldo_cls(hidden, mlp_hidden, max_weights, num_cpus)
        self.up_proj = disldo_cls(hidden, mlp_hidden, max_weights, num_cpus)
        self.down_proj = disldo_cls(mlp_hidden, hidden, max_weights, num_cpus)
        self.energy = _toy_scale_energy() if use_energy else None
        self.input_ln = Tensor(np.ones(hidden, dtype=np.float32))
        self.post_ln = Tensor(np.ones(hidden, dtype=np.float32))

    def trainable_leaf_parameters(self) -> list[Tensor]:
        return [self.input_ln, self.post_ln]


class ToySmallTransformerRealFP4:
    """Importance arm of the precision-matched comparison (DISLDOLayer, inline-trained real FP4).
    See docs/research/toy_precision_models.rst:real_fp4_layer.inline_training."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        max_weights: int,
        use_energy: bool = False,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
        disldo_cls=DISLDOLayer,
    ):
        self.hidden = hidden
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.layers = [
            _RealFP4Layer(hidden, mlp_hidden, max_weights, use_energy, num_cpus, disldo_cls) for _ in range(n_layers)
        ]
        self.lm_head = disldo_cls(hidden, vocab_size, max_weights, num_cpus)

    def parameters_for_optimizer(self) -> list[Tensor]:
        """ONLY the plain leaf params (RMSNorm weights).
        See docs/research/toy_precision_models.rst:real_fp4_layer.inline_training."""
        params = []
        for layer in self.layers:
            params += layer.trainable_leaf_parameters()
        return params

    def forward(self, embedded: np.ndarray, learning_rate: float) -> tuple[Tensor, Tensor | None]:
        T = embedded.shape[0]
        x = Tensor(embedded.astype(np.float32))
        aux_loss_total = None
        for layer in self.layers:
            normed = rmsnorm_tensor(x, layer.input_ln, self.rms_eps)
            q = layer.q_proj.forward(normed, learning_rate)
            k = layer.k_proj.forward(normed, learning_rate)
            v = layer.v_proj.forward(normed, learning_rate)
            attn = banded_attention(q, k, v, half_bandwidth=T, num_cpus=self.num_cpus, causal=True)
            attn, aux_loss = _apply_energy(layer.energy, attn, T, self.hidden)
            aux_loss_total = (
                aux_loss
                if aux_loss_total is None
                else (aux_loss_total if aux_loss is None else aux_loss_total + aux_loss)
            )
            attn = layer.o_proj.forward(attn, learning_rate)
            x = x + attn
            normed2 = rmsnorm_tensor(x, layer.post_ln, self.rms_eps)
            gate = layer.gate_proj.forward(normed2, learning_rate)
            up = layer.up_proj.forward(normed2, learning_rate)
            mlp_out = layer.down_proj.forward(silu(gate) * up, learning_rate)
            x = x + mlp_out
        logits = self.lm_head.forward(x, learning_rate)
        return logits, aux_loss_total


class AdamRowScaleDISLDOLayer:
    """See docs/research/toy_precision_models.rst:adam_row_scale_disldo_layer.proxy_gradient_approximation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        rng: np.random.Generator | None = None,
    ):
        self._inner = DISLDOLayer(in_features, out_features, max_weights, num_cpus, rng=rng)
        self.in_features = in_features
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(in_features, dtype=np.float32)
        self.v = np.zeros(in_features, dtype=np.float32)
        self.t = 0

    def _row_scales(self) -> np.ndarray:
        return np.array([self._inner._c.get_value_scale(r) for r in range(self.in_features)], dtype=np.float32)

    def forward(self, x, learning_rate: float = 0.0) -> Tensor:
        before = self._row_scales()
        out = self._inner.forward(x, learning_rate)
        inner_bwd = out._backward

        def _bwd():
            inner_bwd()  # applies DISLDOLayer's own raw value_scale update
            if learning_rate == 0.0:
                return  # no training this call (e.g. eval forward) -- nothing to re-normalize
            after = self._row_scales()
            raw_delta = after - before
            grad_proxy = -raw_delta / learning_rate
            self.t += 1
            bc1 = 1.0 - self.beta1**self.t
            bc2 = 1.0 - self.beta2**self.t
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_proxy
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_proxy * grad_proxy)
            m_hat = self.m / bc1
            v_hat = self.v / bc2
            adjusted = before - learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            for r in range(self.in_features):
                self._inner._c.set_value_scale_raw(r, float(adjusted[r]))

        out._backward = _bwd
        return out

    def parameters(self) -> list[Tensor]:
        return []  # nothing here is an external-optimizer-trainable Tensor leaf


class AdamRank1DISLDOLayer:
    """See docs/research/toy_precision_models.rst:adam_rank1_disldo_layer.output_scale_activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        rng: np.random.Generator | None = None,
    ):
        self._inner = DISLDOLayer(in_features, out_features, max_weights, num_cpus, rng=rng)
        self.in_features = in_features
        self.out_features = out_features
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_row = np.zeros(in_features, dtype=np.float32)
        self.v_row = np.zeros(in_features, dtype=np.float32)
        self.m_col = np.zeros(out_features, dtype=np.float32)
        self.v_col = np.zeros(out_features, dtype=np.float32)
        self.t = 0
        for c in range(out_features):
            self._inner._c.set_output_scale_raw(c, 1.0)  # activates output_scale's own training

    def _row_scales(self) -> np.ndarray:
        return np.array([self._inner._c.get_value_scale(r) for r in range(self.in_features)], dtype=np.float32)

    def _col_scales(self) -> np.ndarray:
        return np.array([self._inner._c.get_output_scale(c) for c in range(self.out_features)], dtype=np.float32)

    def _adam_step(
        self, before: np.ndarray, after: np.ndarray, m: np.ndarray, v: np.ndarray, learning_rate: float
    ) -> np.ndarray:
        raw_delta = after - before
        grad_proxy = -raw_delta / learning_rate
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        m[:] = self.beta1 * m + (1.0 - self.beta1) * grad_proxy
        v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad_proxy * grad_proxy)
        m_hat = m / bc1
        v_hat = v / bc2
        return before - learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def forward(self, x, learning_rate: float = 0.0) -> Tensor:
        before_row = self._row_scales()
        before_col = self._col_scales()
        out = self._inner.forward(x, learning_rate)
        inner_bwd = out._backward

        def _bwd():
            inner_bwd()  # applies DISLDOLayer's own raw value_scale/output_scale updates
            if learning_rate == 0.0:
                return
            self.t += 1
            after_row = self._row_scales()
            adjusted_row = self._adam_step(before_row, after_row, self.m_row, self.v_row, learning_rate)
            for r in range(self.in_features):
                self._inner._c.set_value_scale_raw(r, float(adjusted_row[r]))
            after_col = self._col_scales()
            adjusted_col = self._adam_step(before_col, after_col, self.m_col, self.v_col, learning_rate)
            for c in range(self.out_features):
                self._inner._c.set_output_scale_raw(c, float(adjusted_col[c]))

        out._backward = _bwd
        return out

    def parameters(self) -> list[Tensor]:
        return []  # nothing here is an external-optimizer-trainable Tensor leaf


class ToySmallTransformerRealFP4RowScaleAdam(ToySmallTransformerRealFP4):
    """See docs/research/toy_precision_models.rst:adam_row_scale_disldo_layer.proxy_gradient_approximation."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        max_weights: int,
        use_energy: bool = False,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
    ):
        super().__init__(
            vocab_size,
            hidden,
            mlp_hidden,
            n_layers,
            max_weights,
            use_energy,
            num_cpus,
            rms_eps,
            disldo_cls=AdamRowScaleDISLDOLayer,
        )


class ToySmallTransformerRealFP4Rank1Adam(ToySmallTransformerRealFP4):
    """See docs/research/toy_precision_models.rst:adam_rank1_disldo_layer.output_scale_activation."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        max_weights: int,
        use_energy: bool = False,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
    ):
        super().__init__(
            vocab_size,
            hidden,
            mlp_hidden,
            n_layers,
            max_weights,
            use_energy,
            num_cpus,
            rms_eps,
            disldo_cls=AdamRank1DISLDOLayer,
        )


def row_scale_fake_quantize(vals: np.ndarray, ptrs: np.ndarray, bits: int) -> np.ndarray:
    """See docs/research/toy_precision_models.rst:rank1_fake_quantize.shared_scale_catastrophe_fix."""
    levels = 2 ** (bits - 1) - 1  # e.g. 7 for 4-bit, 127 for 8-bit
    out = vals.copy()
    for r in range(len(ptrs) - 1):
        s, e = int(ptrs[r]), int(ptrs[r + 1])
        if e <= s:
            continue
        seg = vals[s:e]
        max_abs = float(np.max(np.abs(seg)))
        if max_abs < 1e-12:
            continue
        scale = max_abs / levels
        out[s:e] = np.round(seg / scale) * scale
    return out


def rank1_fake_quantize(vals: np.ndarray, ptrs: np.ndarray, indices: np.ndarray, n_out: int, bits: int) -> np.ndarray:
    """See docs/research/toy_precision_models.rst:rank1_fake_quantize.shared_scale_catastrophe_fix."""
    levels = 2 ** (bits - 1) - 1
    n_in = len(ptrs) - 1
    abs_vals = np.abs(vals.astype(np.float64))
    row_of = np.repeat(np.arange(n_in), np.diff(ptrs).astype(np.int64))
    col_of = indices.astype(np.int64)

    row_scale = np.ones(n_in, dtype=np.float64)
    col_scale = np.ones(n_out, dtype=np.float64)
    for _ in range(3):
        col_max = np.zeros(n_out, dtype=np.float64)
        np.maximum.at(col_max, col_of, abs_vals / np.maximum(row_scale[row_of], 1e-12))
        col_scale = np.maximum(col_max, 1e-12)

        row_max = np.zeros(n_in, dtype=np.float64)
        np.maximum.at(row_max, row_of, abs_vals / np.maximum(col_scale[col_of], 1e-12))
        row_scale = np.maximum(row_max, 1e-12)

    envelope = row_scale[row_of] * col_scale[col_of]
    step = np.maximum(envelope / levels, 1e-12)
    out = np.round(vals.astype(np.float64) / step) * step
    return out.astype(np.float32)


def rankn_fake_quantize(
    vals: np.ndarray, ptrs: np.ndarray, indices: np.ndarray, n_out: int, bits: int, rank: int = 2
) -> np.ndarray:
    """See docs/research/toy_precision_models.rst:rankn_fake_quantize.magnitude_bucketed_columns."""
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    levels = 2 ** (bits - 1) - 1
    n_in = len(ptrs) - 1
    abs_vals = np.abs(vals.astype(np.float64))
    row_of = np.repeat(np.arange(n_in), np.diff(ptrs).astype(np.int64))
    col_of = indices.astype(np.int64)

    row_max = np.zeros(n_in, dtype=np.float64)
    np.maximum.at(row_max, row_of, abs_vals)
    order = np.argsort(row_max, kind="stable")
    bucket_of_sorted_row = np.minimum((np.arange(n_in) * rank) // max(n_in, 1), rank - 1)
    row_bucket = np.zeros(n_in, dtype=np.int64)
    row_bucket[order] = bucket_of_sorted_row
    entry_bucket = row_bucket[row_of]

    envelope = np.empty_like(abs_vals)
    for b in range(rank):
        mask = entry_bucket == b
        if not np.any(mask):
            continue
        r_sub, c_sub, v_sub = row_of[mask], col_of[mask], abs_vals[mask]

        row_scale = np.ones(n_in, dtype=np.float64)
        col_scale = np.ones(n_out, dtype=np.float64)
        for _ in range(3):
            col_max = np.zeros(n_out, dtype=np.float64)
            np.maximum.at(col_max, c_sub, v_sub / np.maximum(row_scale[r_sub], 1e-12))
            col_scale = np.maximum(col_max, 1e-12)

            row_max_pass = np.zeros(n_in, dtype=np.float64)
            np.maximum.at(row_max_pass, r_sub, v_sub / np.maximum(col_scale[c_sub], 1e-12))
            row_scale = np.maximum(row_max_pass, 1e-12)

        envelope[mask] = row_scale[r_sub] * col_scale[c_sub]

    step = np.maximum(envelope / levels, 1e-12)
    out = np.round(vals.astype(np.float64) / step) * step
    return out.astype(np.float32)


def residual_fake_quantize(
    vals: np.ndarray, ptrs: np.ndarray, indices: np.ndarray, n_out: int, bits_per_stage: int, n_stages: int
) -> np.ndarray:
    """See docs/research/toy_precision_models.rst:residual_fake_quantize.true_rvq_vs_rankn."""
    residual = vals.astype(np.float64).copy()
    reconstructed = np.zeros_like(residual)
    for _ in range(n_stages):
        stage_q = rank1_fake_quantize(residual.astype(np.float32), ptrs, indices, n_out, bits_per_stage).astype(
            np.float64
        )
        reconstructed += stage_q
        residual = residual - stage_q
    return reconstructed.astype(np.float32)


def fixed_digit_residual_quantize(
    vals: np.ndarray, bits_per_stage: int, n_stages: int, base: float = 12.0, e_shared: float = 1.0
) -> np.ndarray:
    """See docs/research/toy_precision_models.rst:fixed_digit_residual_quantize.base_and_e_shared_derivation."""
    levels = 2 ** (bits_per_stage - 1) - 1
    residual = vals.astype(np.float64).copy()
    reconstructed = np.zeros_like(residual)
    step = e_shared / levels
    for _ in range(n_stages):
        stage_q = np.round(residual / step) * step
        reconstructed += stage_q
        residual = residual - stage_q
        step /= base
    return reconstructed.astype(np.float32)


def _quantize_raw_digit_inplace(inner: DISLDOLayer32, bits: int, step: float) -> None:
    """See docs/research/toy_precision_models.rst:true_multi_digit_layer.independent_digit_architecture."""
    c = inner._c
    ptrs = np.array(c.ptrs, copy=True)
    indices = np.array(c.indices, copy=True)
    w = np.array(c.weights_vals, copy=True).astype(np.float64)
    imp = np.array(c.importance, copy=True).astype(np.float64)
    w_q = (np.round(w / step) * step).astype(np.float32)
    imp_q = (np.round(imp / step) * step).astype(np.float32)
    c.load_weights(ptrs, indices, w_q, imp_q)


class TrueMultiDigitLayer:
    """See docs/research/toy_precision_models.rst:true_multi_digit_layer.independent_digit_architecture."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        digit_cls=DISLDOLayer,
        bits_per_stage: int = 4,
        n_stages: int = 2,
        base: float = 12.0,
        e_shared: float | None = None,
        lr_power: float = 0.0,
        simulate_quantize: bool = False,
        share_connectivity: bool = False,
        dense: bool = False,
        scale_rank: int = 1,
        empty_init: bool = False,
        rng: np.random.Generator | None = None,
    ):
        # See docs/research/toy_precision_models.rst:true_multi_digit_layer.kwarg_forwarding_and_connectivity_sharing.
        digit_kwargs = {"rng": rng}
        if dense:
            digit_kwargs["dense"] = True
        if scale_rank != 1:
            digit_kwargs["scale_rank"] = scale_rank
        if empty_init:
            digit_kwargs["empty_init"] = True
        self.digits = [
            digit_cls(in_features, out_features, max_weights, num_cpus, **digit_kwargs) for _ in range(n_stages)
        ]
        # See docs/research/toy_precision_models.rst:true_multi_digit_layer.kwarg_forwarding_and_connectivity_sharing.
        if share_connectivity and n_stages > 1:
            base_c = self.digits[0]._c
            ptrs0 = np.asarray(base_c.ptrs)
            idx0 = np.asarray(base_c.indices)
            nnz = len(idx0)
            per_row = max(1, nnz // max(1, in_features))
            scale = 1.0 / np.sqrt(max(1, per_row))
            for digit in self.digits[1:]:
                dc = digit._c
                fresh_rng = rng if rng is not None else np.random.default_rng()
                values = fresh_rng.standard_normal(nnz).astype(np.float32) * scale
                dc.load_weights(ptrs0.astype(np.int32), idx0.astype(np.int32), values)
                if hasattr(dc, "equalize_to_capacity"):
                    dc.equalize_to_capacity(per_row)
        self.in_features = in_features
        self.out_features = out_features
        self.bits_per_stage = bits_per_stage
        self.n_stages = n_stages
        self.base = base
        self.lr_power = lr_power
        self.simulate_quantize = simulate_quantize
        levels = 2 ** (bits_per_stage - 1) - 1
        if e_shared is None:
            init_vals = np.abs(np.asarray(self.digits[0]._c.weights_vals, dtype=np.float64))
            e_shared = float(np.max(init_vals)) if init_vals.size and init_vals.max() > 1e-12 else 1.0
        self.e_shared = e_shared
        self._digit_step = e_shared / levels  # only used if simulate_quantize
        self._factors = [self.base ** (-i) for i in range(n_stages)]

    def forward(
        self,
        x,
        learning_rate: float = 0.0,
        lr_per_row_nnz: bool = True,
        damp_by_importance: bool = True,
        **synapse_kwargs,
    ) -> Tensor:
        # synapse_kwargs forwarded as-is to each digit.forward() (min_decay_frac/max_abs_delta/max_ci; see
        # sili.sparse_rnn.DISLDOLayer.forward).
        outs = []
        for i, digit in enumerate(self.digits):
            eff_lr = learning_rate / (self.base ** (self.lr_power * i))
            out_i = digit.forward(
                x, eff_lr, lr_per_row_nnz=lr_per_row_nnz, damp_by_importance=damp_by_importance, **synapse_kwargs
            )
            if learning_rate != 0.0 and self.simulate_quantize:
                inner_bwd = out_i._backward
                digit_i = digit

                def _make_bwd(inner_bwd=inner_bwd, digit_i=digit_i):
                    def _bwd():
                        inner_bwd()  # real RMSprop update for THIS digit only
                        _quantize_raw_digit_inplace(digit_i, self.bits_per_stage, self._digit_step)

                    return _bwd

                out_i._backward = _make_bwd()
            outs.append(out_i)
        total = outs[0] * self._factors[0]
        for i in range(1, self.n_stages):
            total = total + outs[i] * self._factors[i]
        return total

    def synaptogenesis(self, k: int, importance_cutoff: float):
        """Delegate to each digit's own real synaptogenesis, independently (no shared-connectivity coordination).
        See docs/research/toy_precision_models.rst:true_multi_digit_layer.kwarg_forwarding_and_connectivity_sharing."""
        for digit in self.digits:
            if hasattr(digit, "synaptogenesis"):
                digit.synaptogenesis(k, importance_cutoff, digit._max_row_weights)

    def magnitude_rescale_output(self, target: float, correction_rate: float, scale_invariant: bool = False) -> None:
        """Delegate to each digit's own real magnitude_rescale_output, same independent-per-digit pattern as
        synaptogenesis above."""
        for digit in self.digits:
            if hasattr(digit, "_c") and hasattr(digit._c, "magnitude_rescale_output"):
                digit.magnitude_rescale_output(target, correction_rate, scale_invariant)

    def parameters(self) -> list[Tensor]:
        return []


class TrueMultiDigitDenseLayer:
    """See docs/research/toy_precision_models.rst:true_multi_digit_dense_layer.architecture_isolation_control."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int | None = None,
        num_cpus: int | None = None,
        bits_per_stage: int = 4,
        n_stages: int = 2,
        base: float = 12.0,
        e_shared: float | None = None,
        lr_power: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        levels = 2 ** (bits_per_stage - 1) - 1
        init_scale = (e_shared / levels) if e_shared is not None else 0.1
        self.digits = [DenseTensorLinear(in_features, out_features, scale=init_scale) for _ in range(n_stages)]
        self.opt = AdamOptimizer()
        self.n_stages = n_stages
        self.base = base
        self.lr_power = lr_power
        self._factors = [self.base ** (-i) for i in range(n_stages)]

    def forward(
        self, x, learning_rate: float = 0.0, lr_per_row_nnz: bool = True, damp_by_importance: bool = True
    ) -> Tensor:
        outs = [d.forward(x) for d in self.digits]
        total = outs[0] * self._factors[0]
        for i in range(1, self.n_stages):
            total = total + outs[i] * self._factors[i]
        if learning_rate != 0.0:
            inner_bwd = total._backward

            def _bwd():
                inner_bwd()
                for i, d in enumerate(self.digits):
                    eff_lr = learning_rate / (self.base ** (self.lr_power * i))
                    self.opt.step(d.parameters(), lr=eff_lr)

            total._backward = _bwd
        return total

    def parameters(self) -> list[Tensor]:
        return []  # trained internally via self.opt.step(), not an external optimizer


def _quantize_disldo32_inplace(
    inner: DISLDOLayer32,
    bits: int,
    scheme: str,
    quantize_importance: bool,
    rank: int = 1,
    n_stages: int = 1,
    base: float = 12.0,
    e_shared: float = 1.0,
) -> None:
    c = inner._c
    ptrs = np.array(c.ptrs, copy=True)
    indices = np.array(c.indices, copy=True)
    w = np.array(c.weights_vals, copy=True)
    imp = np.array(c.importance, copy=True)
    n_out = c.n_outputs
    if scheme == "row":
        w_q = row_scale_fake_quantize(w, ptrs, bits)
        imp_q = row_scale_fake_quantize(imp, ptrs, bits) if quantize_importance else imp
    elif scheme == "rank1":
        w_q = rank1_fake_quantize(w, ptrs, indices, n_out, bits)
        imp_q = rank1_fake_quantize(imp, ptrs, indices, n_out, bits) if quantize_importance else imp
    elif scheme == "rankn":
        w_q = rankn_fake_quantize(w, ptrs, indices, n_out, bits, rank)
        imp_q = rankn_fake_quantize(imp, ptrs, indices, n_out, bits, rank) if quantize_importance else imp
    elif scheme == "residual":
        w_q = residual_fake_quantize(w, ptrs, indices, n_out, bits, n_stages)
        imp_q = residual_fake_quantize(imp, ptrs, indices, n_out, bits, n_stages) if quantize_importance else imp
    elif scheme == "fixed_digit_residual":
        w_q = fixed_digit_residual_quantize(w, bits, n_stages, base, e_shared)
        imp_q = fixed_digit_residual_quantize(imp, bits, n_stages, base, e_shared) if quantize_importance else imp
    else:
        raise ValueError(scheme)
    c.load_weights(ptrs, indices, w_q.astype(np.float32), imp_q.astype(np.float32))


class QuantizedDISLDOLayer32:
    """See docs/research/toy_precision_models.rst:quantized_disldo_layer32.rank1_scale_envelope."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        bits: int = 8,
        scheme: str = "rank1",
        quantize_importance: bool = True,
        rank: int = 1,
        n_stages: int = 1,
        base: float = 12.0,
        e_shared: float | None = None,
        rng: np.random.Generator | None = None,
    ):
        self._inner = DISLDOLayer32(in_features, out_features, max_weights, num_cpus, rng=rng)
        self.bits = bits
        self.scheme = scheme
        self.quantize_importance = quantize_importance
        self.rank = rank  # only consulted when scheme == "rankn"
        self.n_stages = n_stages  # only consulted when scheme in {"residual", "fixed_digit_residual"}
        self.base = base  # only consulted when scheme == "fixed_digit_residual"
        # See docs/research/toy_precision_models.rst:fixed_digit_residual_quantize.base_and_e_shared_derivation.
        self.e_shared = 1.0  # only consulted when scheme == "fixed_digit_residual"
        if scheme == "fixed_digit_residual":
            if e_shared is None:
                init_vals = np.asarray(self._inner._c.weights_vals, dtype=np.float64)
                init_abs = np.abs(init_vals)
                e_shared = float(np.max(init_abs)) if init_abs.size and init_abs.max() > 1e-12 else 1.0
            self.e_shared = e_shared

    def forward(
        self, x, learning_rate: float = 0.0, lr_per_row_nnz: bool = True, damp_by_importance: bool = True
    ) -> Tensor:
        out = self._inner.forward(
            x, learning_rate, lr_per_row_nnz=lr_per_row_nnz, damp_by_importance=damp_by_importance
        )
        inner_bwd = out._backward

        def _bwd():
            inner_bwd()  # real fp32 RMSprop update happens here first
            if learning_rate != 0.0:
                _quantize_disldo32_inplace(
                    self._inner,
                    self.bits,
                    self.scheme,
                    self.quantize_importance,
                    self.rank,
                    self.n_stages,
                    self.base,
                    self.e_shared,
                )

        out._backward = _bwd
        return out

    def parameters(self) -> list[Tensor]:
        return []  # nothing here is an external-optimizer-trainable Tensor leaf


def _seed_rank1_scale(inner_c, in_features: int, out_features: int) -> None:
    """See docs/research/toy_precision_models.rst:seed_rank1_scale.cold_start_diagnostic."""
    ptrs = np.array(inner_c.ptrs, copy=True)
    indices = np.array(inner_c.indices, copy=True)
    abs_vals = np.abs(np.array(inner_c.weights_vals, copy=True).astype(np.float64))
    row_of = np.repeat(np.arange(in_features), np.diff(ptrs).astype(np.int64))
    col_of = indices.astype(np.int64)

    row_scale = np.ones(in_features, dtype=np.float64)
    col_scale = np.ones(out_features, dtype=np.float64)
    for _ in range(3):
        col_max = np.zeros(out_features, dtype=np.float64)
        np.maximum.at(col_max, col_of, abs_vals / np.maximum(row_scale[row_of], 1e-12))
        col_scale = np.maximum(col_max, 1e-12)

        row_max = np.zeros(in_features, dtype=np.float64)
        np.maximum.at(row_max, row_of, abs_vals / np.maximum(col_scale[col_of], 1e-12))
        row_scale = np.maximum(row_max, 1e-12)

    for r in range(in_features):
        inner_c.set_value_scale_raw(r, float(row_scale[r]))
    for c in range(out_features):
        inner_c.set_output_scale_raw(c, float(col_scale[c]))


class SeededRank1DISLDOLayer8(DISLDOLayer8):
    """See docs/research/toy_precision_models.rst:seed_rank1_scale.cold_start_diagnostic."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(in_features, out_features, max_weights, num_cpus, rng=rng)
        _seed_rank1_scale(self._c, in_features, out_features)


class SeededDISLDOLayer8Resync(DISLDOLayer8Resync):
    """See docs/research/toy_precision_models.rst:seeded_disldo_layer8_resync.fair_comparison_rationale."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(in_features, out_features, max_weights, num_cpus, rng=rng)
        _seed_rank1_scale(self._c, in_features, out_features)


class SeededDISLDOLayer8AdaMax(DISLDOLayer8AdaMax):
    """See docs/research/toy_precision_models.rst:seeded_disldo_layer8_resync.fair_comparison_rationale."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(in_features, out_features, max_weights, num_cpus, rng=rng)
        _seed_rank1_scale(self._c, in_features, out_features)


class PeriodicSeedRank1DISLDOLayer8(DISLDOLayer8):
    """See docs/research/toy_precision_models.rst:periodic_seed_rank1_disldo_layer8.repeated_correction_test."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_weights: int,
        num_cpus: int = 4,
        reseed_every: int = 200,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(in_features, out_features, max_weights, num_cpus, rng=rng)
        self._reseed_in_features = in_features
        self._reseed_out_features = out_features
        self.reseed_every = reseed_every
        self._step_count = 0
        _seed_rank1_scale(self._c, in_features, out_features)

    def forward(
        self, x, learning_rate: float = 0.0, lr_per_row_nnz: bool = True, damp_by_importance: bool = True
    ) -> Tensor:
        out = super().forward(x, learning_rate, lr_per_row_nnz=lr_per_row_nnz, damp_by_importance=damp_by_importance)
        inner_bwd = out._backward

        def _bwd():
            inner_bwd()
            if learning_rate != 0.0:
                self._step_count += 1
                if self._step_count % self.reseed_every == 0:
                    _seed_rank1_scale(self._c, self._reseed_in_features, self._reseed_out_features)

        out._backward = _bwd
        return out


class ToySmallTransformerFP32Ref(ToySmallTransformerRealFP4):
    """Plain DISLDOLayer32 (fp32, no quantization), the reference ceiling for the quantized siblings below.
    See docs/research/toy_precision_models.rst:quantized_disldo_layer32.rank1_scale_envelope."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        max_weights: int,
        use_energy: bool = False,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
    ):
        super().__init__(
            vocab_size,
            hidden,
            mlp_hidden,
            n_layers,
            max_weights,
            use_energy,
            num_cpus,
            rms_eps,
            disldo_cls=DISLDOLayer32,
        )


class ToySmallTransformerQuant8Rank1(ToySmallTransformerRealFP4):
    """QuantizedDISLDOLayer32(bits=8, scheme=rank1) -- the validated winner config.
    See docs/research/toy_precision_models.rst:quantized_disldo_layer32.rank1_scale_envelope."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        max_weights: int,
        use_energy: bool = False,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
    ):
        cls = functools.partial(QuantizedDISLDOLayer32, bits=8, scheme="rank1", quantize_importance=True)
        super().__init__(
            vocab_size, hidden, mlp_hidden, n_layers, max_weights, use_energy, num_cpus, rms_eps, disldo_cls=cls
        )


class ToySmallTransformerQuant4Rank1(ToySmallTransformerRealFP4):
    """Same as ToySmallTransformerQuant8Rank1 but 4-bit -- known-worse (not broken) comparison point.
    See docs/research/toy_precision_models.rst:quantized_disldo_layer32.rank1_scale_envelope."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        max_weights: int,
        use_energy: bool = False,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
    ):
        cls = functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rank1", quantize_importance=True)
        super().__init__(
            vocab_size, hidden, mlp_hidden, n_layers, max_weights, use_energy, num_cpus, rms_eps, disldo_cls=cls
        )


class ToySmallTransformerQuant4Rank2(ToySmallTransformerRealFP4):
    """Same as ToySmallTransformerQuant4Rank1 but scheme=rankn(rank=2).
    See docs/research/toy_precision_models.rst:rankn_fake_quantize.magnitude_bucketed_columns."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        max_weights: int,
        use_energy: bool = False,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
    ):
        cls = functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rankn", rank=2, quantize_importance=True)
        super().__init__(
            vocab_size, hidden, mlp_hidden, n_layers, max_weights, use_energy, num_cpus, rms_eps, disldo_cls=cls
        )


class ToySmallTransformerQuant4Rank4(ToySmallTransformerRealFP4):
    """Same as ToySmallTransformerQuant4Rank2 but rank=4.
    See docs/research/toy_precision_models.rst:rankn_fake_quantize.magnitude_bucketed_columns."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        mlp_hidden: int,
        n_layers: int,
        max_weights: int,
        use_energy: bool = False,
        num_cpus: int = 2,
        rms_eps: float = 1e-6,
    ):
        cls = functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rankn", rank=4, quantize_importance=True)
        super().__init__(
            vocab_size, hidden, mlp_hidden, n_layers, max_weights, use_energy, num_cpus, rms_eps, disldo_cls=cls
        )


class _PeakEligibilityTrace:
    """See docs/research/toy_precision_models.rst:peak_eligibility_trace.signed_peak_hold_design."""

    def __init__(self, shape: tuple[int, ...], decay: float = 0.9):
        self.decay = decay
        self.peak = np.zeros(shape, dtype=np.float32)  # signed
        self.peak_mag = np.zeros(shape, dtype=np.float32)  # |peak|, tracked for comparison

    def update(self, x: np.ndarray) -> np.ndarray:
        """Call every forward(); returns a copy of the signed peak after this tick's update."""
        x_np = np.asarray(x, dtype=np.float32)
        decayed_mag = self.decay * self.peak_mag
        decayed_val = self.decay * self.peak
        replace = np.abs(x_np) > decayed_mag
        self.peak = np.where(replace, x_np, decayed_val)
        self.peak_mag = np.where(replace, np.abs(x_np), decayed_mag)
        return self.peak.copy()


class PeakEligibilityDISLDOLayer:
    """See docs/research/toy_precision_models.rst:peak_eligibility_disldo_layer.last_input_substitution_mechanism."""

    def __init__(
        self, in_features: int, out_features: int, max_weights: int, num_cpus: int = 4, trace_decay: float = 0.9
    ):
        self._inner = DISLDOLayer(in_features, out_features, max_weights, num_cpus)
        self.in_features = in_features
        self.trace_decay = trace_decay
        self.trace: _PeakEligibilityTrace | None = None  # lazily shaped to the first x

    def forward(self, x, learning_rate: float = 0.0) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=np.float32))
        x_np = np.asarray(x.data, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = x_np[None, :]  # match last_input's own always-2D [batch, cols] shape
        if self.trace is None:
            self.trace = _PeakEligibilityTrace(x_np.shape, decay=self.trace_decay)
        peak_snapshot = self.trace.update(x_np)

        out = self._inner.forward(x, learning_rate)
        if learning_rate != 0.0:
            self._inner._c.last_input[...] = peak_snapshot
        return out

    def parameters(self) -> list[Tensor]:
        return []  # nothing here is an external-optimizer-trainable Tensor leaf
