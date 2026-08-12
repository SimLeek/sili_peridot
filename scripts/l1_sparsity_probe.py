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
import sys, functools, statistics
sys.path.insert(0, ".")
import numpy as np

from sili import _cpu
from sili.tensor import Tensor, gaussian_attention, exp, reduce_sum, power, tensor_abs
from sili.sparse_rnn import DISLDOLayerDeterministic
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
    def __init__(self, seed, dense, o_proj_coef, all_layer_coef=0.0, l1_sparsity_coef=0.0,
                 all_zero_init=False, use_energy=False, energy_kwargs=None, scale_clip_max=None):
        if hasattr(_cpu, "seed_fp4_stochastic_rng"):
            _cpu.seed_fp4_stochastic_rng(seed)
        digit_cls = functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
                                      n_stages=3, base=12.0, lr_power=0.0, dense=dense)
        rng = np.random.default_rng(seed)
        self.state_width = EMBED_WIDTH * COLUMN_NEURONS
        sw = self.state_width
        self.q_proj = digit_cls(sw, sw, MAX_WEIGHTS, 1, rng=np.random.default_rng(rng.integers(2**31)))
        self.k_proj = digit_cls(sw, sw, MAX_WEIGHTS, 1, rng=np.random.default_rng(rng.integers(2**31)))
        self.v_proj = digit_cls(sw, sw, MAX_WEIGHTS, 1, rng=np.random.default_rng(rng.integers(2**31)))
        self.o_proj = digit_cls(sw, sw, MAX_WEIGHTS, 1, rng=np.random.default_rng(rng.integers(2**31)))
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

        self.energy = None
        if use_energy:
            from sili.energy import EnergyDynamics
            if energy_kwargs is not None:
                self.energy = EnergyDynamics(**energy_kwargs)
            else:
                from model.toy_precision_models import _toy_scale_energy
                self.energy = _toy_scale_energy()

        self.scale_clip_max = scale_clip_max

        if all_zero_init:
            n = sw * sw
            zeros = np.zeros(n, dtype=np.uint8)
            for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                for digit in layer.digits:
                    digit._c.load_dense_codes(zeros, zeros)
            n_lm = EMBED_WIDTH * VOCAB
            zeros_lm = np.zeros(n_lm, dtype=np.uint8)
            for digit in self.lm_head.digits:
                digit._c.load_dense_codes(zeros_lm, zeros_lm)
            # Neuron-level (input_ln/state_ln, centers/log_sigmas) stay at
            # their normal baseline -- zero-init is for SYNAPSES (weight
            # matrices) specifically, not neuron-level gain/threshold
            # params.

    def parameters_for_optimizer(self):
        return [self.input_ln, self.state_ln, self.centers, self.log_sigmas]

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

    def _ratio_penalty_split(self, layer, input_t: Tensor, lr: float, coef: float) -> Tensor:
        out_aux = layer.forward(input_t, lr, damp_by_importance=False)
        l2_in = float(np.linalg.norm(input_t.data)) + 1e-6
        ratio_t = self._l2(out_aux) * (1.0 / l2_in)
        return power(ratio_t - 1.0, 2) * coef

    def _l1_sparsity_split(self, layer, input_t: Tensor, lr: float, coef: float) -> Tensor:
        out_aux = layer.forward(input_t, lr, damp_by_importance=False)
        n = float(np.asarray(out_aux.data).size)
        return reduce_sum(tensor_abs(out_aux)) * (coef / n)

    def step(self, x_window, M_prev, lr):
        x_normed = rmsnorm_tensor(Tensor(x_window.astype(np.float32)), self.input_ln, 1e-6)
        m_normed = rmsnorm_tensor(Tensor(M_prev.astype(np.float32)), self.input_ln, 1e-6)
        qkv_source = x_normed + m_normed
        q = self.q_proj.forward(qkv_source, lr)
        k = self.k_proj.forward(qkv_source, lr)
        v = self.v_proj.forward(qkv_source, lr)
        sigmas = exp(self.log_sigmas)
        attn_raw = gaussian_attention(q, k, v, self.centers, sigmas, num_cpus=1, causal=False)
        attn_raw, energy_aux_loss = _apply_energy(self.energy, attn_raw, NUM_TILES, self.state_width)

        reg_terms = []
        if self.all_layer_coef > 0.0:
            reg_terms.append(self._ratio_penalty_split(self.q_proj, qkv_source, lr, self.all_layer_coef))
            reg_terms.append(self._ratio_penalty_split(self.k_proj, qkv_source, lr, self.all_layer_coef))
            reg_terms.append(self._ratio_penalty_split(self.v_proj, qkv_source, lr, self.all_layer_coef))

        raw = self.o_proj.forward(attn_raw, lr)
        if self.o_proj_coef > 0.0:
            raw_aux = self.o_proj.forward(attn_raw, lr, damp_by_importance=False)
            l2_in = float(np.linalg.norm(attn_raw.data)) + 1e-6
            ratio_t = self._l2(raw_aux) * (1.0 / l2_in)
            reg_terms.append(power(ratio_t - 1.0, 2) * self.o_proj_coef)

        if self.l1_sparsity_coef > 0.0:
            reg_terms.append(self._l1_sparsity_split(self.q_proj, qkv_source, lr, self.l1_sparsity_coef))
            reg_terms.append(self._l1_sparsity_split(self.k_proj, qkv_source, lr, self.l1_sparsity_coef))
            reg_terms.append(self._l1_sparsity_split(self.v_proj, qkv_source, lr, self.l1_sparsity_coef))
            reg_terms.append(self._l1_sparsity_split(self.o_proj, attn_raw, lr, self.l1_sparsity_coef))

        raw.data = np.clip(raw.data, -CLIP_RANGE, CLIP_RANGE)
        M_new_t = Tensor(M_prev.astype(np.float32)) + raw
        M_new_t = rmsnorm_tensor(M_new_t, self.state_ln, 1e-6)
        M_new_t.data = np.clip(M_new_t.data, -CLIP_RANGE, CLIP_RANGE)

        pooled = M_new_t.reshape((NUM_TILES, EMBED_WIDTH, COLUMN_NEURONS))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / COLUMN_NEURONS)
        logits = self.lm_head.forward(pooled, lr)

        total_aux = reg_terms[0] if reg_terms else None
        for extra in reg_terms[1:]:
            total_aux = total_aux + extra
        if energy_aux_loss is not None:
            total_aux = energy_aux_loss if total_aux is None else total_aux + energy_aux_loss
        return M_new_t.data, logits, total_aux


def run(model, n_steps, seed):
    task_rng = np.random.RandomState(seed)
    embed_table = task_rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3
    opt = AdamOptimizer()
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    skips = total = 0
    last_accs = []
    for step in range(1, n_steps + 1):
        lr = lr_schedule(step, n_steps, 0.002, 50)
        seq_len = min(2 + step // STEPS_PER_STAGE, NUM_TILES)
        tokens, pairs = generate_copy_sequence(task_rng, VOCAB, seq_len)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        correct = ntgt = 0
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, lr)
            if i in targets:
                pred = int(np.argmax(logits.data[NUM_TILES - 1]))
                ntgt += 1; correct += int(pred == targets[i])
                loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                if aux is not None:
                    loss = loss + aux
                loss.backward()
                total += 1
                n = clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
                if not np.isfinite(n):
                    skips += 1
                opt.step(model.parameters_for_optimizer(), lr=lr)
                if model.scale_clip_max is not None:
                    for layer in (model.q_proj, model.k_proj, model.v_proj, model.o_proj):
                        model.clip_scales(layer, model.scale_clip_max)
        if step % 200 == 0:
            last_accs.append(correct / max(ntgt, 1))
    return last_accs, skips, total


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
            accs, skips, total = run(model, N_STEPS, seed)
            per_seed.append(statistics.mean(accs[-3:]))
            tot_skips += skips; tot_calls += total
        print(f"l1_sparsity_coef={coef}  mean={statistics.mean(per_seed):.4f}  "
              f"std={statistics.stdev(per_seed):.4f}  per_seed={[round(v,4) for v in per_seed]}  "
              f"skip_rate={tot_skips/tot_calls:.3%}", flush=True)
