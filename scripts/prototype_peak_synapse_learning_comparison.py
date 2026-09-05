"""
Small, fast (seconds) end-to-end learning comparison: does the
per-synapse peak-correction mechanism actually help a tiny recurrent
net LEARN the delayed-credit deviation-detection task, at realistic
learning rates and real repeated training.

See docs/research/prototype_peak_synapse_learning_comparison.rst for
the full narrative:
module_overview_and_latest_results (goal, latest result table,
verdict), bugs_found_before_trustworthy_run (4 real bugs fixed to get
a trustworthy run), selection_criterion_bptt_derivation (score formula
derivation), onehot_constant_magnitude_limitation (known limitation).

Run: python -m scripts.prototype_peak_synapse_learning_comparison
"""

import time

import numpy as np
from scipy import stats as scipy_stats
from sili import _cpu
from sili.energy import EnergyDynamics
from sili.sparse_rnn import DISLDOLayer
from sili.tensor import Tensor

from model.toy_beyond_context_task import VOCAB_SIZE, generate_deviation_sequence
from model.toy_recall_models import cross_entropy_sum, lr_schedule, predicted_token

W = 2  # in-context window (tiny, on purpose)
OUT_OF_CONTEXT_MAX = 6  # 3x the window
STATE_WIDTH = 16
# See docs/research/prototype_peak_synapse_learning_comparison.rst:
# num_cpus_single_thread_reproducibility.
NUM_CPUS = 1
IN_FEATURES = VOCAB_SIZE + STATE_WIDTH
# See docs/research/prototype_peak_synapse_learning_comparison.rst:
# max_weights_per_row_floor_fix.
MAX_WEIGHTS = IN_FEATURES * STATE_WIDTH
TRAIN_STEPS = 6000
WARMUP_STEPS = 300
STEPS_PER_LEVEL = 600
CURRICULUM_WINDOW = 2
PEAK_LR = 0.05
EVAL_SEQUENCES = 40
EVAL_N_VALUES = [2, 3, 4, 6]
# See docs/research/prototype_peak_synapse_learning_comparison.rst:
# gentle_energy_config_calibration.
GENTLE_ENERGY_CONFIG = {
    "drive": 0.005,
    "activation_cost": 0.01,
    "precision": 0.0002,
    "density": 0.75,
    "exploration": 0.0001,
    "p": 0.99,
}


def onehot(tok):
    v = np.zeros(VOCAB_SIZE, dtype=np.float32)
    v[tok] = 1.0
    return v


def decay_from_horizon(horizon: int, retain_fraction: float) -> float:
    """Minimum peak_decay retaining `retain_fraction` of a tag's score
    after `horizon` further ticks with no replacement. See
    docs/research/prototype_peak_synapse_learning_comparison.rst:
    decay_from_horizon_derivation."""
    if not (0.0 < retain_fraction < 1.0):
        raise ValueError(f"retain_fraction must be in (0,1), got {retain_fraction}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return retain_fraction ** (1.0 / horizon)


# See docs/research/prototype_peak_synapse_learning_comparison.rst:
# decay_from_horizon_derivation.
PEAK_DECAY = decay_from_horizon(OUT_OF_CONTEXT_MAX, 0.1)


def _sample_n_bits(rng, step):
    level = min(OUT_OF_CONTEXT_MAX, W + step // STEPS_PER_LEVEL)
    lo = max(2, level - CURRICULUM_WINDOW)
    return int(rng.randint(lo, level + 1))


class PlainCell:
    """Baseline: plain DISLDOLayer, no correction. See
    docs/research/prototype_peak_synapse_learning_comparison.rst:
    plain_cell_energy_and_aux_loss."""

    def __init__(self, seed=None, use_energy=True):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(None if seed is None else seed + 1)
        self.cell = DISLDOLayer(IN_FEATURES, STATE_WIDTH, MAX_WEIGHTS, NUM_CPUS, rng=rng1)
        self.head = DISLDOLayer(STATE_WIDTH, VOCAB_SIZE, MAX_WEIGHTS, NUM_CPUS, rng=rng2)
        self.energy = EnergyDynamics(**GENTLE_ENERGY_CONFIG) if use_energy else None

    def step(self, tok, M_prev, lr):
        x = np.concatenate([onehot(tok), M_prev])[None, :]
        delta = self.cell.forward(x, lr)
        if self.energy is not None and lr != 0.0:
            # See docs/research/prototype_peak_synapse_learning_comparison.rst:
            # energy_train_only_eval_skip_convention.
            delta, _aux, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta = delta.reshape((1, STATE_WIDTH))
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta
        logits = self.head.forward(M_new, lr)
        return M_new.data[0], logits

    def query_step(self, tok, M_prev, lr, answer):
        x = np.concatenate([onehot(tok), M_prev])[None, :]
        delta = self.cell.forward(x, lr)
        aux_loss = None
        if self.energy is not None and lr != 0.0:
            delta, aux_loss, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta = delta.reshape((1, STATE_WIDTH))
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta
        logits = self.head.forward(M_new, lr)
        loss = cross_entropy_sum(logits, [(0, answer)])
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()
        return M_new.data[0], logits


class PeakSynapseCell:
    """Cell + readout, cell ALSO gets a per-synapse peak correction at
    the query tick (see prototype_synapse_peak_credit.py for the
    mechanism itself). See
    docs/research/prototype_peak_synapse_learning_comparison.rst:
    peak_cell_correction_criterion."""

    def __init__(self, seed=None, peak_decay=PEAK_DECAY, correction_lr_mult=1.0, use_energy=True):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(None if seed is None else seed + 1)
        self.cell = DISLDOLayer(IN_FEATURES, STATE_WIDTH, MAX_WEIGHTS, NUM_CPUS, rng=rng1)
        self.head = DISLDOLayer(STATE_WIDTH, VOCAB_SIZE, MAX_WEIGHTS, NUM_CPUS, rng=rng2)
        self.energy = EnergyDynamics(**GENTLE_ENERGY_CONFIG) if use_energy else None
        self.peak_decay = peak_decay
        self.correction_lr_mult = correction_lr_mult
        self.peak = np.zeros(IN_FEATURES, dtype=np.float32)  # signed x_r at the winning tick
        self.peak_score = np.zeros(IN_FEATURES, dtype=np.float32)  # |x_r| * |state_change| at that tick

    def _update_peak(self, x_row, state_change_scale):
        score = np.abs(x_row) * state_change_scale
        decayed_val = self.peak_decay * self.peak
        decayed_score = self.peak_decay * self.peak_score
        replace = score > decayed_score
        self.peak = np.where(replace, x_row, decayed_val)
        self.peak_score = np.where(replace, score, decayed_score)

    def _cell_step(self, tok, M_prev, lr):
        """Shared forward: cell + energy gate. Returns (x_row,
        delta_raw, delta_gated) -- delta_gated IS this tick's state
        change (residual update: M_new = M_prev + delta_gated
        exactly), used both for the residual add AND as the
        state_change_scale for peak selection."""
        x_row = np.concatenate([onehot(tok), M_prev])
        x = x_row[None, :]
        delta = self.cell.forward(x, lr)
        delta_gated = delta
        aux_loss = None
        if self.energy is not None and lr != 0.0:
            # See docs/research/prototype_peak_synapse_learning_comparison.rst:
            # energy_train_only_eval_skip_convention.
            delta_gated, aux_loss, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta_gated = delta_gated.reshape((1, STATE_WIDTH))
        return x_row, delta, delta_gated, aux_loss

    def step(self, tok, M_prev, lr):
        x_row, _delta, delta_gated, _aux = self._cell_step(tok, M_prev, lr)
        state_change_scale = float(np.mean(np.abs(delta_gated.data)))
        self._update_peak(x_row, state_change_scale)
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta_gated
        logits = self.head.forward(M_new, lr)
        return M_new.data[0], logits

    def query_step(self, tok, M_prev, lr, answer):
        x_row, delta, delta_gated, aux_loss = self._cell_step(tok, M_prev, lr)
        state_change_scale = float(np.mean(np.abs(delta_gated.data)))
        self._update_peak(x_row, state_change_scale)
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta_gated
        logits = self.head.forward(M_new, lr)
        loss = cross_entropy_sum(logits, [(0, answer)])
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()

        # Extra per-synapse correction. See
        # docs/research/prototype_peak_synapse_learning_comparison.rst:
        # peak_cell_correction_criterion.
        if delta.grad is not None:
            dy = np.asarray(delta.grad, dtype=np.float32)[np.newaxis, :]
            dp, di, dv = _cpu.dense_to_top_k_csr(dy, dy.shape[1], NUM_CPUS)  # keep all columns (dense dy)
            ZERO_EPS = 1e-7
            for r in range(IN_FEATURES):
                if abs(x_row[r]) > ZERO_EPS:
                    continue  # row is currently active -- normal training already covers it
                if self.peak_score[r] <= ZERO_EPS:
                    continue  # no real historical peak to credit
                x_1hot = np.zeros((1, IN_FEATURES), dtype=np.float32)
                x_1hot[0, r] = self.peak[r]
                self.cell._c.backward_sparse(x_1hot, dp, di, dv, 1, lr * self.correction_lr_mult, lr_per_row_nnz=True)
        return M_new.data[0], logits


def train(cell_cls, seed):
    _cpu.seed_fp4_stochastic_rng(seed)  # 2nd uncontrolled RNG: FP4's own stochastic weight rounding
    rng = np.random.RandomState(seed)
    cell = cell_cls(seed=seed + 10_000)  # offset so wiring rng never collides with data rng
    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        n_bits = _sample_n_bits(rng, step)
        tokens, pairs = generate_deviation_sequence(rng, n_bits)
        query_pos, answer = pairs[0]
        M = np.zeros(STATE_WIDTH, dtype=np.float32)
        for i in range(query_pos):
            M, _ = cell.step(int(tokens[i]), M, lr)
        M, _ = cell.query_step(int(tokens[query_pos]), M, lr, answer)
    return cell


def evaluate(cell, seed):
    rng = np.random.RandomState(seed)
    results = {}
    for n_bits in EVAL_N_VALUES:
        correct = 0
        for _ in range(EVAL_SEQUENCES):
            tokens, pairs = generate_deviation_sequence(rng, n_bits)
            query_pos, answer = pairs[0]
            M = np.zeros(STATE_WIDTH, dtype=np.float32)
            for i in range(query_pos + 1):
                M, logits = cell.step(int(tokens[i]), M, 0.0)
            pred = predicted_token(logits, 0)
            correct += int(pred == answer)
        results[n_bits] = correct / EVAL_SEQUENCES
    return results


def main():
    t0 = time.time()
    N_SEEDS = 50
    plain_agg = {n: [] for n in EVAL_N_VALUES}
    peak_agg = {n: [] for n in EVAL_N_VALUES}
    for s in range(N_SEEDS):
        plain = train(PlainCell, seed=1000 + s)
        pr = evaluate(plain, seed=5000 + s)
        peak = train(PeakSynapseCell, seed=2000 + s)
        kr = evaluate(peak, seed=5000 + s)
        for n in EVAL_N_VALUES:
            plain_agg[n].append(pr[n])
            peak_agg[n].append(kr[n])
        print(
            f"seed {s}: plain={ {n: round(pr[n], 2) for n in EVAL_N_VALUES} }  "
            f"peak={ {n: round(kr[n], 2) for n in EVAL_N_VALUES} }"
        )

    # See docs/research/prototype_peak_synapse_learning_comparison.rst:
    # paired_hypothesis_test_design.
    print(
        f"\n{'n_bits':>8}  {'in_ctx':>7}  {'plain mean+-std':>17}  {'peak mean+-std':>17}  "
        f"{'diff':>7}  {'paired-t':>9}  {'p(t)':>7}  {'p(Wilcoxon)':>11}"
    )
    for n_bits in EVAL_N_VALUES:
        in_ctx = "yes" if n_bits <= W else "NO"
        p_arr = np.array(plain_agg[n_bits])
        k_arr = np.array(peak_agg[n_bits])
        pm, ps = p_arr.mean(), p_arr.std()
        km, ks = k_arr.mean(), k_arr.std()
        t_stat, t_p = scipy_stats.ttest_rel(k_arr, p_arr)
        try:
            _w_stat, w_p = scipy_stats.wilcoxon(k_arr, p_arr)
        except ValueError:
            w_p = float("nan")  # all-zero differences -- degenerate, not an error
        print(
            f"{n_bits:>8}  {in_ctx:>7}  {pm:>8.3f} +- {ps:<5.3f}  {km:>8.3f} +- {ks:<5.3f}  "
            f"{km - pm:>+7.4f}  {t_stat:>9.3f}  {t_p:>7.4f}  {w_p:>11.4f}"
        )
    print(
        f"\n(chance = 0.5, {N_SEEDS} seeds x {EVAL_SEQUENCES} eval sequences each; "
        f"p-values are two-sided, NOT corrected for testing {len(EVAL_N_VALUES)} points)"
    )
    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
