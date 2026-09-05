"""LANDMARK RESULT (2026-08-13): L1 output-sparsity on all 4 layers reaches
mean=1.0000 across 5 seeds at coef=0.05/0.07, beating spectral norm's
0.8858, with no hard rescale.
See docs/research/l1_sparsity_probe.rst:module_overview_landmark_result,
:split_backward_delivery_mechanism, :coefficient_sensitivity_goldilocks."""

import functools
import statistics
import sys
import time

sys.path.insert(0, ".")
import numpy as np
from sili import _cpu
from sili.sparse_rnn import DISLDOLayer, DISLDOLayerDeterministic
from sili.tensor import Tensor, exp, gaussian_attention, power, reduce_sum, tensor_abs

from model.toy_precision_models import TrueMultiDigitLayer, _apply_energy
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum, lr_schedule, rmsnorm_tensor
from scripts.train_tile_curriculum import _build_tile_window, generate_copy_sequence

VOCAB, EMBED_WIDTH, COLUMN_NEURONS, NUM_TILES, MAX_WEIGHTS = 10, 8, 4, 4, 128
STEPS_PER_STAGE = 500
CLIP_RANGE = 6.0


class OriginalArchModel:
    """Exact port of ToyTileRecurrenceRealFP4's architecture -- single
    v_proj on the combined qkv_source, single o_proj -- with the
    split-backward L1-sparsity (and L2-ratio, for comparison) mechanism
    wired onto all 4 layers instead of spectral_norm_target."""

    # See docs/research/l1_sparsity_probe.rst:uniform_energy_tap_design.
    _ENERGY_TAPS = ("input", "q", "k", "v", "attn", "raw", "logits")

    def __init__(
        self,
        seed,
        dense,
        o_proj_coef,
        all_layer_coef=0.0,
        l1_sparsity_coef=0.0,
        all_zero_init=False,
        zero_init_importance_code=1,
        use_energy=False,
        energy_kwargs=None,
        scale_clip_max=None,
        log_sigma_clip_max=None,
        stochastic_qkv=False,
        stochastic_o=False,
        lr_per_row_nnz=True,
        scale_rank=1,
        empty_init=False,
        synap_k=3,
        synap_importance_cutoff=0.0,
    ):
        # See docs/research/l1_sparsity_probe.rst:empty_init_synaptogenesis_design.
        self.empty_init = bool(empty_init)
        self.synap_k = int(synap_k)
        self.synap_importance_cutoff = float(synap_importance_cutoff)
        # See docs/research/l1_sparsity_probe.rst:lr_per_row_nnz_double_damping_bug.
        self.lr_per_row_nnz = bool(lr_per_row_nnz)
        if hasattr(_cpu, "seed_fp4_stochastic_rng"):
            _cpu.seed_fp4_stochastic_rng(seed)
        digit_cls = functools.partial(
            TrueMultiDigitLayer,
            digit_cls=DISLDOLayerDeterministic,
            n_stages=3,
            base=12.0,
            lr_power=0.0,
            dense=dense,
            scale_rank=scale_rank,
            empty_init=empty_init,
        )
        # See docs/research/l1_sparsity_probe.rst:stochastic_qkv_o_deterministic_rounding_floor.
        qkv_digit_cls = (
            functools.partial(
                TrueMultiDigitLayer,
                digit_cls=DISLDOLayer,
                n_stages=3,
                base=12.0,
                lr_power=0.0,
                dense=dense,
                scale_rank=scale_rank,
                empty_init=empty_init,
            )
            if stochastic_qkv
            else digit_cls
        )
        o_digit_cls = (
            functools.partial(
                TrueMultiDigitLayer,
                digit_cls=DISLDOLayer,
                n_stages=3,
                base=12.0,
                lr_power=0.0,
                dense=dense,
                scale_rank=scale_rank,
                empty_init=empty_init,
            )
            if stochastic_o
            else digit_cls
        )
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
        # See docs/research/l1_sparsity_probe.rst:split_backward_delivery_mechanism.
        self.l1_sparsity_coef = l1_sparsity_coef

        # See docs/research/l1_sparsity_probe.rst:uniform_energy_tap_design,
        # :fire_wake_gradient_lr_rescaling.
        self._fire_wake_gradient_base = None
        if energy_kwargs is not None and energy_kwargs.get("fire_wake_gradient") is not None:
            energy_kwargs = dict(energy_kwargs)
            self._fire_wake_gradient_base = float(energy_kwargs.pop("fire_wake_gradient"))

        self.energies = {}
        if use_energy:
            from sili.energy import EnergyDynamics

            def _make_energy():
                if energy_kwargs is not None:
                    # See docs/research/l1_sparsity_probe.rst:uniform_energy_tap_design.
                    ek = dict(energy_kwargs)
                    ek.setdefault("rng", np.random.default_rng(rng.integers(2**31)))
                    return EnergyDynamics(**ek)
                from model.toy_precision_models import _toy_scale_energy

                return _toy_scale_energy()

            for name in self._ENERGY_TAPS:
                self.energies[name] = _make_energy()

        self.scale_clip_max = scale_clip_max
        self.log_sigma_clip_max = log_sigma_clip_max

        if all_zero_init:
            # See docs/research/l1_sparsity_probe.rst:all_zero_init_importance_seeding.
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
        See docs/research/l1_sparsity_probe.rst:clip_log_sigmas_runaway_investigation."""
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
        # pre-division. See docs/research/l1_sparsity_probe.rst:lr_per_row_nnz_double_damping_bug.
        qkv_lr = o_lr = lmhead_lr = lr
        if self._fire_wake_gradient_base is not None and lr > 0.0:
            compensated = self._fire_wake_gradient_base / lr
            for energy in self.energies.values():
                energy.fire_wake_gradient = compensated
        energy_aux_terms = []

        def tap(name, tensor, n_hidden=self.state_width):
            # See docs/research/l1_sparsity_probe.rst:uniform_energy_tap_design.
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
            # See docs/research/l1_sparsity_probe.rst:lm_head_l1_deadlock_fix.
            reg_terms.append(self._l1_sparsity_split(self.lm_head, pooled, lmhead_lr, self.l1_sparsity_coef))
        logits = tap(
            "logits", self.lm_head.forward(pooled, lmhead_lr, lr_per_row_nnz=self.lr_per_row_nnz), n_hidden=VOCAB
        )

        total_aux = reg_terms[0] if reg_terms else None
        for extra in reg_terms[1:]:
            total_aux = total_aux + extra
        for extra in energy_aux_terms:
            total_aux = extra if total_aux is None else total_aux + extra
        return M_new_t.data, logits, total_aux


def evaluate(model, n_eval, seed, verbose=False):
    """Post-training capability check: n_eval FRESH, held-out sequences,
    forward-only (never trains). Reports plain correct/n_eval.
    See docs/research/l1_sparsity_probe.rst:evaluate_methodology_and_resolution.

    verbose: prints the last 5 (prediction, target) token-id pairs seen;
    does NOT change the return value."""
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
            M, logits, _aux = model.step(window, M, 0.0)
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


def run(model, n_steps, seed, verbose=False, periodic_eval_n=20, peak_lr=0.002):
    task_rng = np.random.RandomState(seed)
    embed_table = task_rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3
    opt = AdamOptimizer()
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    skips = total = 0
    last_accs = []
    t_start = time.time()
    for step in range(1, n_steps + 1):
        lr = lr_schedule(step, n_steps, peak_lr, 50)
        seq_len = min(2 + step // STEPS_PER_STAGE, NUM_TILES)
        tokens, pairs = generate_copy_sequence(task_rng, VOCAB, seq_len)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        correct = ntgt = 0
        # See docs/research/l1_sparsity_probe.rst:aux_accumulate_every_position,
        # :aux_averaged_by_seq_len.
        total_aux = None
        total_loss = None
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, lr)
            if aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux
            if i in targets:
                pred = int(np.argmax(logits.data[NUM_TILES - 1]))
                ntgt += 1
                correct += int(pred == targets[i])
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
                # Live quality trajectory; cheap small n_eval, forward-only.
                # See docs/research/l1_sparsity_probe.rst:evaluate_methodology_and_resolution.
                periodic_eval = evaluate(model, periodic_eval_n, seed) if periodic_eval_n > 0 else None
                eval_str = f" eval({periodic_eval_n})={periodic_eval:.3f}" if periodic_eval is not None else ""
                print(
                    f"  [seed={seed}] step={step}/{n_steps} "
                    f"avg_step={avg_step * 1000:.1f}ms elapsed={elapsed:.0f}s eta={eta:.0f}s{eval_str}",
                    flush=True,
                )
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
            model = OriginalArchModel(seed, dense=True, o_proj_coef=0.0, all_layer_coef=0.0, l1_sparsity_coef=coef)
            accs, skips, total, avg_step_time = run(model, N_STEPS, seed, verbose=True)
            per_seed.append(statistics.mean(accs[-3:]))
            tot_skips += skips
            tot_calls += total
        print(
            f"l1_sparsity_coef={coef}  mean={statistics.mean(per_seed):.4f}  "
            f"std={statistics.stdev(per_seed):.4f}  per_seed={[round(v, 4) for v in per_seed]}  "
            f"skip_rate={tot_skips / tot_calls:.3%}",
            flush=True,
        )
