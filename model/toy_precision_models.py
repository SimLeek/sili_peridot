"""
sili_peridot/model/toy_precision_models.py
────────────────────────────────────────────
Adam (fp32-matched-to-FP4) vs importance+energy (real FP4), matched
precision -- see the approved plan (fuzzy-plotting-starlight.md) for
the full design rationale. Answers a question deliberately deferred
earlier this session: does this project's own importance/row-scale
training (`sili.sparse_rnn.DISLDOLayer`) beat a standard, well-tuned
optimizer (Adam), when BOTH sides represent weights at the SAME
FP4-level precision -- not fp32 vs FP4, which would just repeat the
precision/optimizer confound this session already spent real effort
disentangling (JOURNAL.md's two isolation controls).

`use_energy` is a SEPARATE, orthogonal toggle on BOTH model classes
(not baked into one arm only) -- per [[feedback_do_science_correctly]]:
an earlier version of this module gave `EnergyDynamics` to the
real-FP4 arm only, confounding "optimizer" with "energy's own added
noise/aux-loss." `EnergyDynamics` wraps ACTIVATIONS (the attention
output), not weight training, so it attaches identically regardless of
which linear-layer type is used -- each layer builds its OWN fresh
`EnergyDynamics` instance (`_toy_scale_energy()`) when `use_energy=True`,
since its `energy` running state is per-instance and must not be
shared across layers/models.

`fake_quantize_fp4`/`ArtificialFP4Linear` reproduce the REAL FP4
representation (`sili/lib/headers/fp4quant.hpp`'s 16-level table,
`sili/sparse_rnn.py`'s `per_row` calibration formula
`scale=max(|row|)/6.0`) as a straight-through fake-quantization op on
top of an ordinary Adam-trainable fp32 leaf -- NOT a reimplementation
of DISLDOLayer's own (also gradient-trained) row-scale, which stays
real/untouched on the other arm. `ToySmallTransformerRealFP4` uses
DISLDOLayer directly (per [[feedback_importance_is_already_the_optimizer]]:
its inline C++ update during backward() already IS an optimizer -- no
external AdamOptimizer.step() call on its own big weight matrices).

`AdamRowScaleDISLDOLayer`/`ToySmallTransformerRealFP4RowScaleAdam`:
per direct correction, individual FP4 weight VALUES should keep
importance as their training signal (believed to be the right
mechanism for synaptogenesis/pruning), but the coarser per-row
`value_scale` -- one float per row, not per weight -- is cheap to give
its own Adam-style adaptive normalization on top of importance's own
raw update, without replacing importance's role anywhere. See
AdamRowScaleDISLDOLayer's own docstring for the exact mechanism and
its one real approximation.
"""
from __future__ import annotations

import functools
from typing import List, Optional, Tuple

import numpy as np

from sili.tensor import Tensor, banded_attention, silu, _acc
from sili.sparse_rnn import (DISLDOLayer, DISLDOLayer32, DISLDOLayer8,
                             DISLDOLayer8Resync, DISLDOLayer8AdaMax)
from sili.energy import EnergyDynamics

from .toy_recall_models import rmsnorm_tensor, AdamOptimizer


def _toy_scale_energy() -> EnergyDynamics:
    """`sili_block.default_window_energy` is explicitly a placeholder
    calibrated for full-model-scale windows ("real tuning is Phase 5's
    job") -- checked directly against this toy scale (HIDDEN=12,
    T*HIDDEN~60) rather than assumed to transfer: its own aux_loss grew
    unbounded during training (5.5 -> 17.9 over 300 steps) and total
    loss diverged. This config instead matches sili__new's own small
    -scale EnergyDynamics test convention
    (tests/unit/python/test_sparse_rnn_cell.py's
    TestEnergyDynamicsKeptIndices, h sizes 20-64 -- comparable to this
    toy model's own T*hidden), verified directly to behave far better
    here (aux_loss stays 0.005 -> ~0.3, loss reaches a real minimum
    instead of diverging)."""
    return EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01,
                          density=0.05, p=0.3)


def _apply_energy(energy: Optional[EnergyDynamics], attn: Tensor,
                  T: int, hidden: int) -> Tuple[Tensor, Optional[Tensor]]:
    """Shared by both model classes below -- wraps attn [T, hidden]
    through EnergyDynamics (flatten/unflatten, matching
    model/tile_recurrence.py's own apply_tile_step convention) if
    `energy` is given, else passes attn through unchanged with no
    aux_loss. Keeps the with/without-energy code paths identical
    everywhere except this one call, so the two arms differ ONLY in
    whether this is a no-op."""
    if energy is None:
        return attn, None
    gated, aux_loss, _actual_p = energy(attn.reshape((T * hidden,)))
    return gated.reshape((T, hidden)), aux_loss


# 15 real, finite FP4 (OCP MXFP4 E2M1) levels -- verbatim copy of
# sili/lib/headers/fp4quant.hpp's FP4_TABLE (one of the 16 four-bit
# codes is repurposed as NaN there, leaving 15 usable values here):
# {0, .5, 1, 1.5, 2, 3, 4, 6} and their negatives (0 not duplicated).
_FP4_POSITIVE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_TABLE = np.concatenate([-_FP4_POSITIVE[:0:-1], _FP4_POSITIVE]).astype(np.float32)
FP4_MAX = 6.0


def fake_quantize_fp4(w: Tensor) -> Tensor:
    """Straight-through fake FP4 quantization -- forward rounds each
    row of `w` to the real 15-level FP4 table using the same per-row
    `scale=max(|row|)/6.0` calibration `sili/sparse_rnn.py`'s
    `per_row` mode uses; backward is identity (standard QAT practice --
    `w` itself, not the quantized output, is the thing Adam actually
    trains). Same custom-`_bwd`-closure pattern as
    `toy_recall_models.cross_entropy_sum`/`rmsnorm_tensor` -- no
    sili__new change needed."""
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
    """fp32 master weight (ordinary Adam-trainable leaf), matmul
    against its own fake-FP4-quantized value every forward -- see
    fake_quantize_fp4."""

    def __init__(self, in_features: int, out_features: int, scale: float = 0.1):
        self.weight = Tensor(
            (np.random.randn(in_features, out_features) * scale).astype(np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return x @ fake_quantize_fp4(self.weight)

    def parameters(self) -> List[Tensor]:
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

    def parameters(self) -> List[Tensor]:
        params = [self.input_ln, self.post_ln]
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj,
                      self.gate_proj, self.up_proj, self.down_proj):
            params += layer.parameters()
        return params


class ToySmallTransformerArtificialFP4:
    """Same shape as toy_recall_models.ToySmallTransformer, built from
    ArtificialFP4Linear instead of DenseTensorLinear -- the Adam arm
    of the precision-matched comparison. Trained via an ordinary
    AdamOptimizer over .parameters(), exactly like the dense toy
    baseline. `use_energy` toggles EnergyDynamics on the attention
    output (see module docstring) -- independent of the optimizer/
    precision question this class otherwise tests.

    forward() returns (logits, aux_loss) -- aux_loss is None when
    use_energy=False, else must be added to the task loss before
    .backward() (same convention as ToySmallTransformerRealFP4)."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int,
                 n_layers: int, use_energy: bool = False,
                 num_cpus: int = 2, rms_eps: float = 1e-6):
        self.hidden = hidden
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.layers = [_ArtificialFP4Layer(hidden, mlp_hidden, use_energy)
                       for _ in range(n_layers)]
        self.lm_head = ArtificialFP4Linear(hidden, vocab_size)

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params + self.lm_head.parameters()

    def forward(self, embedded: np.ndarray) -> Tuple[Tensor, Optional[Tensor]]:
        T = embedded.shape[0]
        x = Tensor(embedded.astype(np.float32))
        aux_loss_total = None
        for layer in self.layers:
            normed = rmsnorm_tensor(x, layer.input_ln, self.rms_eps)
            q = layer.q_proj.forward(normed)
            k = layer.k_proj.forward(normed)
            v = layer.v_proj.forward(normed)
            attn = banded_attention(q, k, v, half_bandwidth=T,
                                    num_cpus=self.num_cpus, causal=True)
            attn, aux_loss = _apply_energy(layer.energy, attn, T, self.hidden)
            aux_loss_total = aux_loss if aux_loss_total is None else (
                aux_loss_total if aux_loss is None else aux_loss_total + aux_loss)
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
    """DISLDOLayer's own big weight matrices train inline during
    backward() (learning_rate-driven, no external optimizer -- see
    module docstring); input_ln/post_ln are plain Tensor leaves,
    trained by a small separate AdamOptimizer step (see
    ToySmallTransformerRealFP4.parameters_for_optimizer -- these are
    the ONLY params that optimizer ever sees). `disldo_cls` lets
    ToySmallTransformerRealFP4RowScaleAdam reuse this exact layer
    shape with AdamRowScaleDISLDOLayer in place of plain DISLDOLayer."""

    def __init__(self, hidden: int, mlp_hidden: int, max_weights: int,
                use_energy: bool, num_cpus: int, disldo_cls=DISLDOLayer):
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

    def trainable_leaf_parameters(self) -> List[Tensor]:
        return [self.input_ln, self.post_ln]


class ToySmallTransformerRealFP4:
    """Same shape as toy_recall_models.ToySmallTransformer, built from
    DISLDOLayer (real inline-trained FP4, importance/row-scale as the
    optimizer) -- the importance arm of the precision-matched
    comparison. `use_energy` toggles EnergyDynamics on the attention
    output (matching model/tile_recurrence.py's own apply_tile_step
    pattern) -- independent of the optimizer/precision question this
    class otherwise tests (see module docstring -- an earlier version
    of this class always used energy, confounding the two).

    forward() returns (logits, aux_loss) -- aux_loss is None when
    use_energy=False, else must be added to the task loss before the
    single shared .backward() call: that one call both trains
    DISLDOLayer's weights (inline, via the gradient reaching each
    layer's own output) AND accumulates gradient for the plain leaf
    parameters (parameters_for_optimizer, below)."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int, n_layers: int,
                 max_weights: int, use_energy: bool = False,
                 num_cpus: int = 2, rms_eps: float = 1e-6, disldo_cls=DISLDOLayer):
        self.hidden = hidden
        self.rms_eps = rms_eps
        self.num_cpus = num_cpus
        self.layers = [_RealFP4Layer(hidden, mlp_hidden, max_weights, use_energy,
                                     num_cpus, disldo_cls)
                       for _ in range(n_layers)]
        self.lm_head = disldo_cls(hidden, vocab_size, max_weights, num_cpus)

    def parameters_for_optimizer(self) -> List[Tensor]:
        """ONLY the plain leaf params (RMSNorm weights) -- DISLDOLayer's
        own big weight matrices train inline during backward(), never
        via an external optimizer step (see module/class docstrings)."""
        params = []
        for layer in self.layers:
            params += layer.trainable_leaf_parameters()
        return params

    def forward(self, embedded: np.ndarray, learning_rate: float) -> Tuple[Tensor, Optional[Tensor]]:
        T = embedded.shape[0]
        x = Tensor(embedded.astype(np.float32))
        aux_loss_total = None
        for layer in self.layers:
            normed = rmsnorm_tensor(x, layer.input_ln, self.rms_eps)
            q = layer.q_proj.forward(normed, learning_rate)
            k = layer.k_proj.forward(normed, learning_rate)
            v = layer.v_proj.forward(normed, learning_rate)
            attn = banded_attention(q, k, v, half_bandwidth=T,
                                    num_cpus=self.num_cpus, causal=True)
            attn, aux_loss = _apply_energy(layer.energy, attn, T, self.hidden)
            aux_loss_total = aux_loss if aux_loss_total is None else (
                aux_loss_total if aux_loss is None else aux_loss_total + aux_loss)
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
    """A real DISLDOLayer, with its per-row `value_scale` re-normalized
    via Adam AFTER each backward() -- individual FP4 weight VALUES are
    completely untouched (importance keeps its existing role there
    entirely); only the coarser row-scale (one float per row) gets an
    adaptive step on top.

    Mechanism: `SparseLinearLayer.get_value_scale(row)`/
    `set_value_scale_raw(row, scale)` are real pybind accessors
    (`cpu_backend.cpp:1082-1097`) -- checked directly, not assumed.
    `forward()` snapshots `value_scale` for every row BEFORE the
    caller's `loss.backward()` runs (which is when DISLDOLayer's own
    `backward_dense` applies its raw importance-damped update to
    `value_scale`, inside the `_bwd` closure below); once that raw
    update has been applied, this treats the OBSERVED delta
    (`after - before`) as a proxy gradient signal
    (`grad ~= -raw_delta / learning_rate`) and runs one standard Adam
    moment-update step over that per-row vector, overwriting
    `value_scale` with the Adam-normalized step instead of the raw
    one.

    This is an approximation, documented as such rather than silently
    assumed correct: it re-normalizes an ALREADY-APPLIED delta rather
    than intercepting the true pre-damping gradient before
    DISLDOLayer's own importance-damping is applied to it -- a
    reasonable experimental probe for whether adaptive normalization
    on just the row-scale helps, not a claim that this is identical to
    running real Adam on the underlying gradient."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                num_cpus: int = 4, beta1: float = 0.9, beta2: float = 0.999,
                eps: float = 1e-8):
        self._inner = DISLDOLayer(in_features, out_features, max_weights, num_cpus)
        self.in_features = in_features
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(in_features, dtype=np.float32)
        self.v = np.zeros(in_features, dtype=np.float32)
        self.t = 0

    def _row_scales(self) -> np.ndarray:
        return np.array([self._inner._c.get_value_scale(r) for r in range(self.in_features)],
                        dtype=np.float32)

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
            bc1 = 1.0 - self.beta1 ** self.t
            bc2 = 1.0 - self.beta2 ** self.t
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_proxy
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad_proxy * grad_proxy)
            m_hat = self.m / bc1
            v_hat = self.v / bc2
            adjusted = before - learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            for r in range(self.in_features):
                self._inner._c.set_value_scale_raw(r, float(adjusted[r]))

        out._backward = _bwd
        return out

    def parameters(self) -> List[Tensor]:
        return []  # nothing here is an external-optimizer-trainable Tensor leaf


class AdamRank1DISLDOLayer:
    """AdamRowScaleDISLDOLayer's mechanism, extended from row-only to
    rank-1 (row `value_scale` AND column `output_scale`, both Adam
    -normalized independently) -- individual FP4 weight VALUES stay
    completely untouched, same as AdamRowScaleDISLDOLayer.

    `output_scale` is real (`get_output_scale(col)`/
    `set_output_scale_raw(col, scale)`, `cpu_backend.cpp:1105-1125`)
    but -- checked directly, not assumed -- the pybind docstring states
    it only becomes gradient-trainable in `backward_dense()` "after
    calling `set_output_scale_raw` at least once." `__init__` calls
    `set_output_scale_raw(c, 1.0)` for every column once (the
    documented default value) specifically to activate that -- without
    it, `output_scale` would never move at all and there'd be nothing
    for the Adam step to re-normalize.

    Same approximation as AdamRowScaleDISLDOLayer, documented there:
    re-normalizes an ALREADY-APPLIED delta, not the true pre-damping
    gradient."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                num_cpus: int = 4, beta1: float = 0.9, beta2: float = 0.999,
                eps: float = 1e-8):
        self._inner = DISLDOLayer(in_features, out_features, max_weights, num_cpus)
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
        return np.array([self._inner._c.get_value_scale(r) for r in range(self.in_features)],
                        dtype=np.float32)

    def _col_scales(self) -> np.ndarray:
        return np.array([self._inner._c.get_output_scale(c) for c in range(self.out_features)],
                        dtype=np.float32)

    def _adam_step(self, before: np.ndarray, after: np.ndarray, m: np.ndarray, v: np.ndarray,
                   learning_rate: float) -> np.ndarray:
        raw_delta = after - before
        grad_proxy = -raw_delta / learning_rate
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
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

    def parameters(self) -> List[Tensor]:
        return []  # nothing here is an external-optimizer-trainable Tensor leaf


class ToySmallTransformerRealFP4RowScaleAdam(ToySmallTransformerRealFP4):
    """ToySmallTransformerRealFP4 with AdamRowScaleDISLDOLayer in place
    of plain DISLDOLayer -- same architecture shape, same importance
    -driven per-weight training, only the row-scale gets Adam
    normalization on top. See AdamRowScaleDISLDOLayer's own docstring."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int, n_layers: int,
                 max_weights: int, use_energy: bool = False,
                 num_cpus: int = 2, rms_eps: float = 1e-6):
        super().__init__(vocab_size, hidden, mlp_hidden, n_layers, max_weights,
                         use_energy, num_cpus, rms_eps,
                         disldo_cls=AdamRowScaleDISLDOLayer)


class ToySmallTransformerRealFP4Rank1Adam(ToySmallTransformerRealFP4):
    """ToySmallTransformerRealFP4 with AdamRank1DISLDOLayer in place of
    plain DISLDOLayer -- row AND column scale both get Adam
    normalization. See AdamRank1DISLDOLayer's own docstring."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int, n_layers: int,
                 max_weights: int, use_energy: bool = False,
                 num_cpus: int = 2, rms_eps: float = 1e-6):
        super().__init__(vocab_size, hidden, mlp_hidden, n_layers, max_weights,
                         use_energy, num_cpus, rms_eps,
                         disldo_cls=AdamRank1DISLDOLayer)


def row_scale_fake_quantize(vals: np.ndarray, ptrs: np.ndarray, bits: int) -> np.ndarray:
    """Per-row max-abs scale, symmetric signed N-bit levels -- matches
    sili's own existing value_scale convention. Deterministic
    round-to-nearest (NOT stochastic -- keeps this isolated from FP4's
    own stochastic-rounding noise, a separate, already-characterized
    variable)."""
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


def rank1_fake_quantize(vals: np.ndarray, ptrs: np.ndarray, indices: np.ndarray,
                        n_out: int, bits: int) -> np.ndarray:
    """Row scale * col scale (rank-1 envelope, matching sili_peridot's
    own B5a fix for the shared-scale FP4 catastrophe): alternating
    max-fit, 3 passes. Fully vectorized (np.maximum.at scatter-max, no
    per-synapse Python loop) -- must stay cheap enough to run every
    training step at full density."""
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


def rankn_fake_quantize(vals: np.ndarray, ptrs: np.ndarray, indices: np.ndarray,
                        n_out: int, bits: int, rank: int = 2) -> np.ndarray:
    """Generalizes rank1_fake_quantize's single shared row_scale/col_scale
    pair to `rank` independently-fit column-scale profiles, one per
    row-magnitude bucket (rows bucketed by their own max |w| into `rank`
    equal-count quantile groups -- deterministic sort+split, no
    iterative clustering).

    An additive residual decomposition (fit rank-1, subtract, re-fit the
    leftover, matching e.g. matching-pursuit/greedy-SVD) was the first
    thing tried and does NOT work here, verified by hand before writing
    this: rank1_fake_quantize's envelope is a strict MAX-COVER (its
    alternating-max-fit guarantees row_scale[r]*col_scale[c] >= |v| for
    every synapse in the row/col, by construction of row_scale itself
    being a row max) -- the residual after subtracting it is <= 0
    everywhere, so a second additive term has nothing left to refine.
    This matters beyond being a dead end: an envelope is exactly what a
    real N-bit fixed-point scale must be (never let a stored value
    exceed what `levels` codes can represent) -- an approach that
    doesn't preserve the cover property would be simulating something
    real hardware couldn't actually do.

    Bucketing rows by magnitude is the degree of freedom rank-1 alone
    can't express: a single shared col_scale must cover the worst row
    sharing that column even when most rows sharing it are far smaller
    -- every small-magnitude row wastes precision matching a big
    outlier row it happens to share a column with. Splitting rows into
    magnitude buckets lets each bucket fit its own column envelope
    against only its own peers. Measured effect is real but modest
    (checked on a synthetic bimodal-magnitude case: rank-2 tightened
    the small-magnitude bucket's mean envelope/|value| ratio by ~22%,
    the large-magnitude bucket unchanged, as expected) -- it is bounded
    by how much true row/col scale correlation the data has, not a
    free lunch, and higher rank than the number of genuinely distinct
    magnitude regimes in the data won't keep helping.

    Reduces EXACTLY to rank1_fake_quantize when rank=1 (single bucket
    containing every row = the identical 3-pass alternating fit)."""
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


def residual_fake_quantize(vals: np.ndarray, ptrs: np.ndarray, indices: np.ndarray,
                           n_out: int, bits_per_stage: int, n_stages: int) -> np.ndarray:
    """True residual/cascaded quantization -- matching neural-audio-codec
    RVQ (e.g. EnCodec/SoundStream): quantize vals via rank1_fake_quantize
    at bits_per_stage, then quantize the ROUNDING RESIDUAL (vals - q1)
    with a FRESH rank1 envelope fit to the residual's own (much smaller)
    dynamic range, repeat n_stages times, sum every stage to reconstruct.

    NOT the same as the rank-n entry's documented dead end (JOURNAL.md):
    that attempt tried to refine the rank1 SCALE ENVELOPE itself
    (row_scale*col_scale, a max-cover bound -- provably nothing left to
    subtract a second time, since the envelope already upper-bounds
    every |v| in its row/col by construction). This instead refines the
    quantized VALUE's rounding error, an unrelated, always-nonzero
    quantity bounded by half a quantization step -- the actual mechanism
    real residual vector quantization exploits, and genuinely untested
    here before now.

    Total cost: n_stages * bits_per_stage bits/weight (plus n_stages
    independent row/col scale pairs -- small, O(rows+cols) overhead per
    stage) -- e.g. n_stages=2, bits_per_stage=4 is 8 bits/weight total,
    a fair, apples-to-apples comparison against
    rank1_fake_quantize(bits=8)'s single 8-bit code."""
    residual = vals.astype(np.float64).copy()
    reconstructed = np.zeros_like(residual)
    for _ in range(n_stages):
        stage_q = rank1_fake_quantize(residual.astype(np.float32), ptrs, indices,
                                      n_out, bits_per_stage).astype(np.float64)
        reconstructed += stage_q
        residual = residual - stage_q
    return reconstructed.astype(np.float32)


def fixed_digit_residual_quantize(vals: np.ndarray, bits_per_stage: int, n_stages: int,
                                  base: float = 4.0, e_shared: float = 1.0) -> np.ndarray:
    """Zero-scaling-VECTOR residual quantization, literal closed-form
    digit-place-value construction per direct design discussion:
    `fp(4n) ~= e_shared * sum_i digit_i * base**-i`. NO row/col fit
    anywhere (contrast `residual_fake_quantize`'s own per-stage
    `rank1_fake_quantize` calls, and `rank1_8bit`'s trained
    value_scale/output_scale) and NO per-call data-dependent
    computation either (contrast even a fresh-every-step global max
    like BitNet/XNOR-Net use) -- every stage's step size is a FIXED
    constant chosen before any data is seen, matching the literal
    "maybe choose/learn B or e_shared, but even then I'm not sure
    that's needed" framing directly.

    `base`: ratio between consecutive residual stages' resolution.
    Derived directly from the digit format's own mantissa bit count,
    not fit to data -- real sili FP4 (E2M1, 1 mantissa bit) has
    worst-case relative rounding error ~= 1/2**(mantissa_bits+1) = 1/4,
    so consecutive stages naturally shrink ~4x regardless of what the
    weights look like. Default 4.0 matches that directly.

    `e_shared`: a single FIXED scalar (not a per-row/per-col vector,
    not gradient-trained, never updated after being chosen) applied
    once to bring the whole layer's values into the digit format's
    representable floor -- has nothing to go stale relative to, since
    it never changes after construction. Real FP4 alone only covers
    ~[0.5, 6] before hitting its floor, and typical weight init
    (~1/sqrt(fan_in)) sits well below that -- this is the ONE thing a
    pure residual stack genuinely cannot fix on its own (each stage
    only refines PRECISION within the range the previous stage already
    covers, it can never extend the floor downward), so this parameter
    stays even in the otherwise fully zero-scale design. Default 1.0
    (no rescaling) -- real usage should derive this once from initial
    weight statistics at construction, never touch it again."""
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


def _quantize_disldo32_inplace(inner: DISLDOLayer32, bits: int, scheme: str,
                               quantize_importance: bool, rank: int = 1,
                               n_stages: int = 1, base: float = 4.0,
                               e_shared: float = 1.0) -> None:
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
        imp_q = (fixed_digit_residual_quantize(imp, bits, n_stages, base, e_shared)
                if quantize_importance else imp)
    else:
        raise ValueError(scheme)
    c.load_weights(ptrs, indices, w_q.astype(np.float32), imp_q.astype(np.float32))


class QuantizedDISLDOLayer32:
    """A real DISLDOLayer32 (fp32 DeltaCSRBiValues backend, same
    RMSprop-style importance formula as production DISLDOLayer) whose
    weight AND importance arrays get fake-quantized to `bits` right
    after every backward() call that actually trains (learning_rate !=
    0.0). Simulates "this layer's real storage is N-bit" while keeping
    the forward/backward ARITHMETIC itself exact fp32 -- isolates
    "does training survive N-bit storage" from "is the update-rule
    math itself precise", matching real quantization-aware-training
    simulators.

    Found empirically (see sili_peridot JOURNAL.md, the original
    single-RNN-task sweep in the quantization-exploration script):
    8-bit + rank-1 scale (row*col envelope), quantizing BOTH weight
    and importance, reaches near-FP32 convergence quality; plain
    per-row scale needs to leave importance in FP32 to do nearly as
    well; 4-bit (either scale scheme) converges but at a real quality
    cost even with rank-1 -- importance's dynamic range is shaped by
    BOTH forward and backward signal, unlike a weight, which is why it
    needs the extra rank-1 degree of freedom more. Defaults here
    (bits=8, scheme=rank1, quantize_importance=True) are that
    empirically-validated winner, not an arbitrary default -- this
    class is the vehicle for testing whether it generalizes across
    OTHER toy models/tasks before being worth a real sili__new C++
    variant.

    `scheme="rankn"` (with `rank` >= 2) is a follow-up being tested to
    see whether it helps 4-bit specifically -- see rankn_fake_quantize's
    own docstring for the mechanism and its honest, modest measured
    effect. `rank` is only consulted when scheme=="rankn".

    Same disldo_cls-pluggable call convention as DISLDOLayer/
    DISLDOLayer32/AdamRowScaleDISLDOLayer -- drops directly into
    ToySmallTransformerRealFP4/ToyTileRecurrenceRealFP4 with no
    changes needed there."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                num_cpus: int = 4, bits: int = 8, scheme: str = "rank1",
                quantize_importance: bool = True, rank: int = 1, n_stages: int = 1,
                base: float = 4.0, e_shared: Optional[float] = None,
                rng: Optional[np.random.Generator] = None):
        self._inner = DISLDOLayer32(in_features, out_features, max_weights, num_cpus, rng=rng)
        self.bits = bits
        self.scheme = scheme
        self.quantize_importance = quantize_importance
        self.rank = rank  # only consulted when scheme == "rankn"
        self.n_stages = n_stages  # only consulted when scheme in {"residual", "fixed_digit_residual"}
        self.base = base  # only consulted when scheme == "fixed_digit_residual"
        # Computed ONCE here from the initial preseeded weights, then frozen
        # for the rest of training (matching fixed_digit_residual_quantize's
        # own "chosen once, never touched again" design -- has nothing to go
        # stale relative to). e_shared=None (default) derives it from the
        # real initial weight magnitude; a caller can still pass a fixed
        # constant directly to skip that entirely.
        self.e_shared = 1.0  # only consulted when scheme == "fixed_digit_residual"
        if scheme == "fixed_digit_residual":
            if e_shared is None:
                init_vals = np.asarray(self._inner._c.weights_vals, dtype=np.float64)
                init_abs = np.abs(init_vals)
                e_shared = float(np.max(init_abs)) if init_abs.size and init_abs.max() > 1e-12 else 1.0
            self.e_shared = e_shared

    def forward(self, x, learning_rate: float = 0.0, lr_per_row_nnz: bool = True,
               damp_by_importance: bool = True) -> Tensor:
        out = self._inner.forward(x, learning_rate, lr_per_row_nnz=lr_per_row_nnz,
                                  damp_by_importance=damp_by_importance)
        inner_bwd = out._backward

        def _bwd():
            inner_bwd()  # real fp32 RMSprop update happens here first
            if learning_rate != 0.0:
                _quantize_disldo32_inplace(self._inner, self.bits, self.scheme,
                                           self.quantize_importance, self.rank,
                                           self.n_stages, self.base, self.e_shared)

        out._backward = _bwd
        return out

    def parameters(self) -> List[Tensor]:
        return []  # nothing here is an external-optimizer-trainable Tensor leaf


def _seed_rank1_scale(inner_c, in_features: int, out_features: int) -> None:
    """Seed value_scale/output_scale ONCE, from a real closed-form
    3-pass alternating max-cover fit of the layer's CURRENT (freshly
    -preseeded) weights -- same math as rank1_fake_quantize's own
    envelope fit, applied here only to set the starting point, not to
    quantize/round anything. Real ongoing training still uses
    DISLDOLayer8's own gradient-based value_scale/output_scale update
    (linear_disldo.hpp) after this -- isolates whether that slow,
    noisy, query-tick-only-gradient learning process was simply
    undertrained within a fixed step budget (found directly: real
    DISLDOLayer8 collapsed out-of-context, mean_acc=0.19, despite
    nominally using the identical 8-bit+rank1 scheme the toy
    fake-quantize simulation solved at mean_acc=0.97), vs the
    8-bit+rank1 REPRESENTATION itself being insufficient."""
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
    """Real DISLDOLayer8 (true C++ E4M3 storage, true disldo_forward/
    backward kernels -- NOT a fake-quantize simulation) whose
    value_scale/output_scale are seeded once at construction from a
    real closed-form rank-1 fit (see _seed_rank1_scale), instead of
    left at the default 1.0 for the slow gradient-based update to
    discover from scratch. Direct diagnostic for whether real
    DISLDOLayer8's out-of-context collapse is a cold-start/undertrained
    -scale problem rather than the 8-bit+rank-1 representation itself
    being insufficient (the toy simulation already showed the
    representation works when the envelope is well-fit)."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        super().__init__(in_features, out_features, max_weights, num_cpus, rng=rng)
        _seed_rank1_scale(self._c, in_features, out_features)


class SeededDISLDOLayer8Resync(DISLDOLayer8Resync):
    """Real DISLDOLayer8Resync (the DeferredScaleWrite fix -- see
    sili__new's ScalePolicy/disldo_backward docstrings) with
    value_scale/output_scale seeded once at construction, same as
    SeededRank1DISLDOLayer8. Seeding here isn't about cold-start (the
    fix under test is orthogonal to that) -- it's for a FAIR comparison
    against `fp8_seeded`: plain DISLDOLayer8's output_scale never
    trains at all unless something calls set_output_scale_raw at least
    once (confirmed directly: no code in DISLDOLayer8's own
    construction path does), so without seeding here too, any
    difference measured could just be "output_scale was active" rather
    than "the deferred-write fix helped"."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        super().__init__(in_features, out_features, max_weights, num_cpus, rng=rng)
        _seed_rank1_scale(self._c, in_features, out_features)


class SeededDISLDOLayer8AdaMax(DISLDOLayer8AdaMax):
    """Same as SeededDISLDOLayer8Resync, but the AdaMax-style scale
    update (see AdaMaxScalePolicy's docstring, sili__new's
    delta_csr_types.hpp) instead of RMSprop."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                num_cpus: int = 4, rng: Optional[np.random.Generator] = None):
        super().__init__(in_features, out_features, max_weights, num_cpus, rng=rng)
        _seed_rank1_scale(self._c, in_features, out_features)


class PeriodicSeedRank1DISLDOLayer8(DISLDOLayer8):
    """Like SeededRank1DISLDOLayer8, but re-seeds value_scale/output_scale
    from a fresh closed-form rank-1 fit every `reseed_every` training
    backward() calls, not just once at construction -- direct test of
    whether REPEATEDLY correcting the envelope (touching NOTHING about
    the real RMSprop weight-update math) can substitute for the
    simulation's every-step refit, or whether real DISLDOLayer8's own
    separate, nested value_scale/output_scale optimizer (see
    linear_disldo.hpp's `scale_eff_lr = learning_rate / nnz_row`,
    itself RMSprop-style via value_scale_importance) genuinely can't
    hold a good fit between corrections even when repeatedly given one."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                num_cpus: int = 4, reseed_every: int = 200,
                rng: Optional[np.random.Generator] = None):
        super().__init__(in_features, out_features, max_weights, num_cpus, rng=rng)
        self._reseed_in_features = in_features
        self._reseed_out_features = out_features
        self.reseed_every = reseed_every
        self._step_count = 0
        _seed_rank1_scale(self._c, in_features, out_features)

    def forward(self, x, learning_rate: float = 0.0, lr_per_row_nnz: bool = True,
               damp_by_importance: bool = True) -> Tensor:
        out = super().forward(x, learning_rate, lr_per_row_nnz=lr_per_row_nnz,
                              damp_by_importance=damp_by_importance)
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
    """ToySmallTransformerRealFP4 built from plain DISLDOLayer32 (fp32
    DeltaCSRBiValues, RMSprop importance, NO quantization) -- the
    reference ceiling this class's own quantized siblings below are
    measured against, on the SAME architecture/task (not the separate
    single-RNN-task reference the quantization scheme was originally
    validated on)."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int, n_layers: int,
                 max_weights: int, use_energy: bool = False,
                 num_cpus: int = 2, rms_eps: float = 1e-6):
        super().__init__(vocab_size, hidden, mlp_hidden, n_layers, max_weights,
                         use_energy, num_cpus, rms_eps,
                         disldo_cls=DISLDOLayer32)


class ToySmallTransformerQuant8Rank1(ToySmallTransformerRealFP4):
    """ToySmallTransformerRealFP4 with QuantizedDISLDOLayer32 (8-bit,
    rank-1 scale, weight+importance both quantized -- the validated
    winner config) in place of plain DISLDOLayer. See
    QuantizedDISLDOLayer32's own docstring."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int, n_layers: int,
                 max_weights: int, use_energy: bool = False,
                 num_cpus: int = 2, rms_eps: float = 1e-6):
        cls = functools.partial(QuantizedDISLDOLayer32, bits=8, scheme="rank1",
                                quantize_importance=True)
        super().__init__(vocab_size, hidden, mlp_hidden, n_layers, max_weights,
                         use_energy, num_cpus, rms_eps, disldo_cls=cls)


class ToySmallTransformerQuant4Rank1(ToySmallTransformerRealFP4):
    """Same as ToySmallTransformerQuant8Rank1 but 4-bit -- the
    known-worse (but not broken) config, kept as a comparison point,
    not the recommended default."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int, n_layers: int,
                 max_weights: int, use_energy: bool = False,
                 num_cpus: int = 2, rms_eps: float = 1e-6):
        cls = functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rank1",
                                quantize_importance=True)
        super().__init__(vocab_size, hidden, mlp_hidden, n_layers, max_weights,
                         use_energy, num_cpus, rms_eps, disldo_cls=cls)


class ToySmallTransformerQuant4Rank2(ToySmallTransformerRealFP4):
    """Same as ToySmallTransformerQuant4Rank1 but with rankn_fake_quantize
    (rank=2, magnitude-bucketed column scale) in place of plain rank1
    scaling -- tests whether the extra scale degree of freedom recovers
    some of 4-bit's real quality cost vs rank-1."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int, n_layers: int,
                 max_weights: int, use_energy: bool = False,
                 num_cpus: int = 2, rms_eps: float = 1e-6):
        cls = functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rankn", rank=2,
                                quantize_importance=True)
        super().__init__(vocab_size, hidden, mlp_hidden, n_layers, max_weights,
                         use_energy, num_cpus, rms_eps, disldo_cls=cls)


class ToySmallTransformerQuant4Rank4(ToySmallTransformerRealFP4):
    """Same as ToySmallTransformerQuant4Rank2 but rank=4 -- checks
    whether the trend (if any) continues past rank=2 or plateaus."""

    def __init__(self, vocab_size: int, hidden: int, mlp_hidden: int, n_layers: int,
                 max_weights: int, use_energy: bool = False,
                 num_cpus: int = 2, rms_eps: float = 1e-6):
        cls = functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rankn", rank=4,
                                quantize_importance=True)
        super().__init__(vocab_size, hidden, mlp_hidden, n_layers, max_weights,
                         use_energy, num_cpus, rms_eps, disldo_cls=cls)


class _PeakEligibilityTrace:
    """Leaky peak-hold tracker over the layer's FULL [batch, features]
    input shape (matching `SparseLinearLayer.last_input` exactly, not
    reduced) -- remembers, per (tile, feature) cell, the SIGNED value
    from whichever recent tick had the largest magnitude, decaying only
    until a new input exceeds the decayed peak (then replaced), rather
    than blurring all history into one running sum. Direct replacement
    for an earlier smooth-decaying-SUM design (`e = decay*e +
    activity`, no longer in this file) that was found, via the actual
    C++ update formula (`linear_disldo.hpp`'s disldo_backward:
    `dL/d(value_scale[r])` already bakes in the query tick's OWN input
    magnitude via `g = dy*iv`), to double-count activity magnitude when
    multiplied against DISLDOLayer's own already-input-weighted row
    gradient -- see JOURNAL.md's e-prop postmortem for the full
    diagnosis. SIGNED (not magnitude-only): the substitution this
    feeds (see PeakEligibilityDISLDOLayer) needs a real signed input
    value, since DISLDO's gradient math depends on input sign for
    direction, not just magnitude -- this formulation was worked out
    directly with the user."""

    def __init__(self, shape: Tuple[int, ...], decay: float = 0.9):
        self.decay = decay
        self.peak = np.zeros(shape, dtype=np.float32)      # signed
        self.peak_mag = np.zeros(shape, dtype=np.float32)  # |peak|, tracked for comparison

    def update(self, x: np.ndarray) -> np.ndarray:
        """Call every forward(), returns a SNAPSHOT (copy) of the
        signed peak as it stood right after this tick's update."""
        x_np = np.asarray(x, dtype=np.float32)
        decayed_mag = self.decay * self.peak_mag
        decayed_val = self.decay * self.peak
        replace = np.abs(x_np) > decayed_mag
        self.peak = np.where(replace, x_np, decayed_val)
        self.peak_mag = np.where(replace, np.abs(x_np), decayed_mag)
        return self.peak.copy()


class PeakEligibilityDISLDOLayer:
    """A real DISLDOLayer whose per-row `value_scale` credit-assignment
    uses REAL C++ gradient math, applied to a peak-substituted input --
    the replacement for the earlier, found-broken EPropDISLDOLayer/
    EPropAdamDISLDOLayer (removed; see JOURNAL.md's postmortem) AND for
    an even-earlier broadcast-`out.grad` version tried in between.

    Mechanism, worked out directly with the user: `SparseLinearLayer`
    caches its most recent forward input as `_last_input` and exposes
    it via the `last_input` property -- checked directly, not assumed
    (`cpu_backend.cpp:1199-1214`): this is a ZERO-COPY, WRITABLE numpy
    view straight onto the C++ buffer (verified: mutating the returned
    array from Python propagates into the object the C++ backward
    reads from). `backward_dense` doesn't take `x` as an argument at
    all -- it reads `_last_input` directly. So instead of trying to
    hand-derive a Python-side approximation of the gradient (two prior
    attempts, both found flawed -- see JOURNAL.md), this OVERWRITES
    `last_input` with the peak-held (signed) value right after
    forward, BEFORE backward ever fires, so DISLDO's OWN real
    `backward_dense` computes the row's (and, internally, each
    synapse's) gradient AS IF the input had been whichever recent tick
    was most salient for that cell -- zero Python-side gradient
    approximation, reusing the actual C++ math end to end.

    Known, accepted side effect: `dx` (accumulated into this layer's
    input Tensor's `.grad`, e.g. into `qkv_source.grad` in
    ToyTileRecurrenceRealFP4.step()) is ALSO computed from the
    substituted input, not the true one -- this contaminates gradient
    reaching upstream plain-Tensor leaves (input_ln/post_ln) a little,
    since backward_dense computes dx and the value_scale gradient from
    the same `_last_input` in one pass with no way to split them apart
    without a C++ change. Accepted as a small, documented tradeoff
    (input_ln/post_ln are secondary parameters, not the credit
    -assignment mechanism itself) rather than deferred silently.

    True per-SYNAPSE substitution (not just per-row) would need direct
    CSR access -- the "expensive, large core change" the user already
    flagged as a separate, later effort, not attempted here. If this
    layer's typical input activation ends up genuinely ~1-sparse (few
    nonzero rows), `SISLDOLayer`'s CSR forward/backward path would let
    this same substitution touch only the active indices instead of a
    full dense array -- a real future efficiency angle at true
    MiniCPM5 scale, not needed at this toy width."""

    def __init__(self, in_features: int, out_features: int, max_weights: int,
                num_cpus: int = 4, trace_decay: float = 0.9):
        self._inner = DISLDOLayer(in_features, out_features, max_weights, num_cpus)
        self.in_features = in_features
        self.trace_decay = trace_decay
        self.trace: Optional[_PeakEligibilityTrace] = None  # lazily shaped to the first x

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

    def parameters(self) -> List[Tensor]:
        return []  # nothing here is an external-optimizer-trainable Tensor leaf
