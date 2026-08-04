"""
Small, fast (seconds) end-to-end learning comparison: does the
per-synapse peak-correction mechanism (backward_sparse + 1-hot
substitution, verified to fire correctly in
prototype_synapse_peak_credit.py) actually help a tiny recurrent net
LEARN the delayed-credit deviation-detection task, at REALISTIC
learning rates and real repeated training -- not just "does the
gradient move once when pushed hard".

No attention, no MLP, no embedding table, no tile system -- one raw
recurrent DISLDOLayer cell (state_t = state_t-1 + cell([token_onehot,
state_t-1])) + one DISLDOLayer readout. Per direct instruction: verify
here, in seconds, before touching the full tile system.

Result as of the last real run (5 seeds x 40 eval sequences, both arms
using EnergyDynamics -- see below): NO clear win for the peak-synapse
mechanism over plain DISLDO -- every difference was within about 1
std. The isolated mechanism check (prototype_synapse_peak_credit.py)
still holds -- a silent-at-query-tick row genuinely gets real credit
that plain DISLDO cannot give it -- but that hasn't yet translated
into a measurable end-to-end learning benefit at this tiny scale.
Recorded honestly, not spun -- see JOURNAL.md for the full narrative
(RNG reproducibility bugs found/fixed along the way, EnergyDynamics
fixing a separate random-collapse instability, and the user's
follow-up critique that the row-level peak/backward_sparse hack still
isn't the "right" mechanism -- a true forward-contribution-weighted,
per-synapse eligibility trace needs new sili__new kernel work, not
more tuning of this prototype).

Run: python -m scripts.prototype_peak_synapse_learning_comparison
"""
import time
import numpy as np

from sili.sparse_rnn import DISLDOLayer
from sili import _cpu
from sili.tensor import Tensor
from sili.energy import EnergyDynamics

from model.toy_beyond_context_task import generate_deviation_sequence, VOCAB_SIZE
from model.toy_recall_models import cross_entropy_sum, predicted_token, lr_schedule

W = 2                    # in-context window (tiny, on purpose)
OUT_OF_CONTEXT_MAX = 6   # 3x the window
STATE_WIDTH = 8
# NUM_CPUS=1, not 2: _cpu.seed_fp4_stochastic_rng only reseeds the
# CALLING thread's RNG state (checked directly, not assumed -- its own
# docstring: "does not control a real (OpenMP-parallel) training run's
# outcome, only this one thread's RNG state"). With num_cpus>1, worker
# threads would keep independent, unseeded RNG state regardless of
# calling this -- single-threaded is what actually makes a run
# reproducible, appropriate anyway at this toy scale.
NUM_CPUS = 1
IN_FEATURES = VOCAB_SIZE + STATE_WIDTH
# DERIVED, not guessed: _preseed_random_sparse computes
# per_row = max(2, max_weights // n_inputs) and then k = per_row // 2 --
# for the cell layer (n_inputs=IN_FEATURES, n_outputs=STATE_WIDTH), the
# worst case, per_row was landing on the bare floor of 2 (k=1: literally
# ONE random connection per input row, zero redundancy) at the previous
# MAX_WEIGHTS=32. Setting max_weights so per_row can reach n_outputs
# (full column coverage for the widest layer) removes that floor-clamp
# cliff. This alone did NOT fix the random-collapse instability --
# turned out to be only ONE of two separate uncontrolled RNG sources
# (see seed_fp4_stochastic_rng call in train(), the second one: FP4's
# own stochastic weight rounding, used on every backward call, was
# ALSO never seeded).
MAX_WEIGHTS = IN_FEATURES * STATE_WIDTH
TRAIN_STEPS = 4000
WARMUP_STEPS = 200
STEPS_PER_LEVEL = 400
CURRICULUM_WINDOW = 2
PEAK_LR = 0.05
EVAL_SEQUENCES = 40
EVAL_N_VALUES = [2, 3, 4, 6]


def onehot(tok):
    v = np.zeros(VOCAB_SIZE, dtype=np.float32)
    v[tok] = 1.0
    return v


def _sample_n_bits(rng, step):
    level = min(OUT_OF_CONTEXT_MAX, W + step // STEPS_PER_LEVEL)
    lo = max(2, level - CURRICULUM_WINDOW)
    return int(rng.randint(lo, level + 1))


class PlainCell:
    """Baseline: plain DISLDOLayer, no correction. use_energy wraps the
    cell's own contribution (delta, before the residual add) through
    EnergyDynamics every tick -- same role EnergyDynamics plays on the
    attention output in the full tile-recurrence system, where it
    twice fixed this exact "collapse at one specific point" pattern
    (JOURNAL.md: n_bits=2 collapse, later n_bits=24 collapse) by
    keeping more neurons active, raising the odds a useful
    state-carrying pattern survives. aux_loss from non-query ticks is
    simply discarded (never reaches a .backward() call) -- same
    no-BPTT precedent as everywhere else in this project; only the
    query tick's own aux_loss is added to the real loss."""

    def __init__(self, seed=None, use_energy=True):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(None if seed is None else seed + 1)
        self.cell = DISLDOLayer(IN_FEATURES, STATE_WIDTH, MAX_WEIGHTS, NUM_CPUS, rng=rng1)
        self.head = DISLDOLayer(STATE_WIDTH, VOCAB_SIZE, MAX_WEIGHTS, NUM_CPUS, rng=rng2)
        self.energy = EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01,
                                     density=0.05, p=0.3) if use_energy else None

    def step(self, tok, M_prev, lr):
        x = np.concatenate([onehot(tok), M_prev])[None, :]
        delta = self.cell.forward(x, lr)
        if self.energy is not None:
            delta, _aux, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta = delta.reshape((1, STATE_WIDTH))
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta
        logits = self.head.forward(M_new, lr)
        return M_new.data[0], logits

    def query_step(self, tok, M_prev, lr, answer):
        x = np.concatenate([onehot(tok), M_prev])[None, :]
        delta = self.cell.forward(x, lr)
        aux_loss = None
        if self.energy is not None:
            delta, aux_loss, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta = delta.reshape((1, STATE_WIDTH))
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta
        logits = self.head.forward(M_new, lr)
        loss = cross_entropy_sum(logits, [(0, answer)])
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        return M_new.data[0], logits


class PeakSynapseCell:
    """Cell + readout, cell ALSO gets a per-synapse peak correction at
    the query tick (see prototype_synapse_peak_credit.py for the
    mechanism itself). Same energy wiring as PlainCell -- `delta`
    stays the RAW cell output (what the peak-correction's `delta.grad`
    check needs); `delta_gated` (after EnergyDynamics) is what actually
    goes into the residual add. `delta.grad` is still a real,
    correctly-backpropagated quantity when energy sits downstream of
    it -- just incorporating energy's own local derivative too."""

    def __init__(self, seed=None, peak_decay=0.9, correction_lr_mult=1.0, use_energy=True):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(None if seed is None else seed + 1)
        self.cell = DISLDOLayer(IN_FEATURES, STATE_WIDTH, MAX_WEIGHTS, NUM_CPUS, rng=rng1)
        self.head = DISLDOLayer(STATE_WIDTH, VOCAB_SIZE, MAX_WEIGHTS, NUM_CPUS, rng=rng2)
        self.energy = EnergyDynamics(drive=0.1, activation_cost=0.05, precision=0.01,
                                     density=0.05, p=0.3) if use_energy else None
        self.peak_decay = peak_decay
        self.correction_lr_mult = correction_lr_mult
        self.peak = np.zeros(IN_FEATURES, dtype=np.float32)
        self.peak_mag = np.zeros(IN_FEATURES, dtype=np.float32)

    def _update_peak(self, x_row):
        decayed_val = self.peak_decay * self.peak
        decayed_mag = self.peak_decay * self.peak_mag
        replace = np.abs(x_row) > decayed_mag
        self.peak = np.where(replace, x_row, decayed_val)
        self.peak_mag = np.where(replace, np.abs(x_row), decayed_mag)

    def step(self, tok, M_prev, lr):
        x_row = np.concatenate([onehot(tok), M_prev])
        self._update_peak(x_row)
        x = x_row[None, :]
        delta = self.cell.forward(x, lr)
        delta_gated = delta
        if self.energy is not None:
            delta_gated, _aux, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta_gated = delta_gated.reshape((1, STATE_WIDTH))
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta_gated
        logits = self.head.forward(M_new, lr)
        return M_new.data[0], logits

    def query_step(self, tok, M_prev, lr, answer):
        x_row = np.concatenate([onehot(tok), M_prev])
        self._update_peak(x_row)
        x = x_row[None, :]
        delta = self.cell.forward(x, lr)
        delta_gated = delta
        aux_loss = None
        if self.energy is not None:
            delta_gated, aux_loss, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta_gated = delta_gated.reshape((1, STATE_WIDTH))
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta_gated
        logits = self.head.forward(M_new, lr)
        loss = cross_entropy_sum(logits, [(0, answer)])
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()

        # extra per-synapse correction, using the REAL error that just
        # flowed into the cell's own output (delta.grad, populated by
        # the residual-add's backward before it reached the cell).
        # DERIVED criterion, not a tuned threshold: row r's normal
        # gradient is g = dy*iv (confirmed directly from
        # linear_disldo.hpp), exactly zero whenever x_row[r] is zero --
        # so only rows that are CURRENTLY zero (numerically, not "small")
        # are ones normal training structurally cannot touch this tick;
        # correcting anything else would duplicate/fight training that
        # already works. ZERO_EPS distinguishes exact-zero from nonzero
        # in float32, not a magnitude cutoff.
        if delta.grad is not None:
            dy = np.asarray(delta.grad, dtype=np.float32)[np.newaxis, :]
            dp, di, dv = _cpu.dense_to_top_k_csr(dy, dy.shape[1], NUM_CPUS)  # keep all columns (dense dy)
            ZERO_EPS = 1e-7
            for r in range(IN_FEATURES):
                if abs(x_row[r]) > ZERO_EPS:
                    continue  # row is currently active -- normal training already covers it
                if self.peak_mag[r] <= ZERO_EPS:
                    continue  # no real historical peak to credit
                x_1hot = np.zeros((1, IN_FEATURES), dtype=np.float32)
                x_1hot[0, r] = self.peak[r]
                self.cell._c.backward_sparse(x_1hot, dp, di, dv, 1, lr * self.correction_lr_mult,
                                             lr_per_row_nnz=True)
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
    N_SEEDS = 5
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
        print(f"seed {s}: plain={ {n: round(pr[n],2) for n in EVAL_N_VALUES} }  "
              f"peak={ {n: round(kr[n],2) for n in EVAL_N_VALUES} }")

    print(f"\n{'n_bits':>8}  {'in_ctx':>7}  {'plain (mean+-std)':>20}  {'peak-synapse (mean+-std)':>26}")
    for n_bits in EVAL_N_VALUES:
        in_ctx = "yes" if n_bits <= W else "NO"
        pm, ps = np.mean(plain_agg[n_bits]), np.std(plain_agg[n_bits])
        km, ks = np.mean(peak_agg[n_bits]), np.std(peak_agg[n_bits])
        print(f"{n_bits:>8}  {in_ctx:>7}  {pm:>8.3f} +- {ps:.3f}       {km:>8.3f} +- {ks:.3f}")
    print(f"\n(chance = 0.5, {N_SEEDS} seeds x {EVAL_SEQUENCES} eval sequences each)")
    print(f"total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
