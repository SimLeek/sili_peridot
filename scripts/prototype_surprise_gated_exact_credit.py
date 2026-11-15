"""Does engineering the proof's own precondition into the architecture --
rather than hoping a selection heuristic discovers it -- finally let a
cross-time credit mechanism rescue memory use on the deviation task?

Motivation (see bend_proofs/synaptic_hold_bptt_approximation in
sili__new, PR #55): synaptic-hold/e-prop-style identity-trace credit is
PROVABLY exact, not merely approximate, when the intermediate ticks
between a synapse's activity and the loss are INACTIVE (zero state
change). generate_deviation_sequence's body is already deterministic
and periodic (DEVIATION_BASE_PATTERN, period 3) with AT MOST ONE
deviating position -- "expected" is knowable exactly, by construction.
But measured directly (this session): the existing PlainCell's own
state-change magnitude (||delta||) is statistically indistinguishable
between regular and deviant ticks, both before AND after training
(untrained: 6.21 vs 6.51; trained 6000 steps: 85.3 vs 87.4) -- nothing
in the architecture or training signal currently exploits the task's
regularity to produce near-zero updates on expected input. The
proof's precondition is available in the TASK but not realized by the
MODEL.

This script closes that gap directly: SurpriseGatedExactCell
multiplicatively gates the state update by a deterministic surprise
indicator (tok != expected_tok), forcing EXACTLY zero state change on
every regular tick -- not learned, not approximate, a hard architectural
fact matching all_inactive_is_exact's precondition exactly. Since
generate_deviation_sequence guarantees <=1 deviation per sequence, this
means <=1 active tick ever -- and the proof's error term (which only
arises from PRODUCTS of 2+ active ticks' corrections) is then IDENTICALLY
zero, not just small: the identity-trace credit this cell applies
(holding the one active tick's input, crediting it with the query's own
gradient, zero decay) is exactly equal to true BPTT for this task, not
an approximation to it.

Reuses the existing 4-arm harness's task/eval machinery; only adds this
one new arm, compared against the same no_memory/plain/peak/energy_gated
baselines already measured (see prototype_energy_gated_synapse_credit.py).

Run: python -m scripts.prototype_surprise_gated_exact_credit
"""

import time

import numpy as np
from scipy import stats as scipy_stats
from sili.energy import EnergyDynamics
from sili.sparse_rnn import DISLDOLayer
from sili.tensor import Tensor

from model.toy_beyond_context_task import DEVIATION_BASE_PATTERN, VOCAB_SIZE, generate_deviation_sequence
from model.toy_recall_models import cross_entropy_sum
from scripts.prototype_energy_gated_synapse_credit import EnergyGatedCreditCell, train_nomem
from scripts.prototype_peak_synapse_learning_comparison import (
    EVAL_N_VALUES,
    EVAL_SEQUENCES,
    GENTLE_ENERGY_CONFIG,
    IN_FEATURES,
    MAX_WEIGHTS,
    NUM_CPUS,
    PEAK_LR,
    STATE_WIDTH,
    TRAIN_STEPS,
    WARMUP_STEPS,
    PeakSynapseCell,
    PlainCell,
    _sample_n_bits,
    lr_schedule,
    onehot,
    train,
)


def expected_tok(pos: int) -> int:
    """The task's own deterministic regularity -- known exactly, not
    learned, matching DEVIATION_BASE_PATTERN's period-3 construction."""
    return int(DEVIATION_BASE_PATTERN[pos % len(DEVIATION_BASE_PATTERN)])


class SurpriseGatedExactCell:
    """Body ticks: state update is multiplicatively gated by a hard
    surprise indicator (tok != expected_tok(pos)) -- EXACTLY zero on
    every regular tick, not approximately. At most one tick per
    sequence can ever be "active" (generate_deviation_sequence's own
    guarantee), so the single held active row is credited at query
    time with the query's own gradient at coefficient 1 (zero decay,
    zero threshold-tuning) -- exact per the proof's corollary, since the
    error term requires >=2 active ticks to be nonzero."""

    def __init__(self, seed=None, use_energy=True, correction_lr_mult=1.0):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(None if seed is None else seed + 1)
        self.cell = DISLDOLayer(IN_FEATURES, STATE_WIDTH, MAX_WEIGHTS, NUM_CPUS, rng=rng1)
        self.head = DISLDOLayer(STATE_WIDTH, VOCAB_SIZE, MAX_WEIGHTS, NUM_CPUS, rng=rng2)
        self.energy = EnergyDynamics(**GENTLE_ENERGY_CONFIG) if use_energy else None
        self.correction_lr_mult = correction_lr_mult
        self.held_x_row = None  # the one active tick's full input row, if any yet this sequence

    def step(self, tok, M_prev, lr, pos):
        x_row = np.concatenate([onehot(tok), M_prev])
        x = x_row[None, :]
        delta = self.cell.forward(x, lr)
        delta_gated = delta
        if self.energy is not None and lr != 0.0:
            delta_gated, _aux, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta_gated = delta_gated.reshape((1, STATE_WIDTH))
        surprise = 1.0 if tok != expected_tok(pos) else 0.0
        delta_final = delta_gated * surprise
        if surprise > 0.0:
            self.held_x_row = x_row  # at most one true event per sequence for this task
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta_final
        logits = self.head.forward(M_new, lr)
        return M_new.data[0], logits

    def query_step(self, tok, M_prev, lr, answer):
        """Query tick: ungated (the query token is never part of the
        body's periodic pattern, so "surprise" has no meaning here --
        memory must flow through to the head normally)."""
        x_row = np.concatenate([onehot(tok), M_prev])
        x = x_row[None, :]
        delta = self.cell.forward(x, lr)
        delta_gated = delta
        aux_loss = None
        if self.energy is not None and lr != 0.0:
            delta_gated, aux_loss, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta_gated = delta_gated.reshape((1, STATE_WIDTH))
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta_gated
        logits = self.head.forward(M_new, lr)
        loss = cross_entropy_sum(logits, [(0, answer)])
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()

        if delta.grad is not None and self.held_x_row is not None:
            dy = np.asarray(delta.grad, dtype=np.float32)
            dy = dy[np.newaxis, :] if dy.ndim == 1 else dy
            x_held = self.held_x_row[None, :].astype(np.float32)
            self.cell._c.backward_dense(x_held, dy, lr * self.correction_lr_mult, lr_per_row_nnz=True)
        self.held_x_row = None  # fresh sequence next time
        return M_new.data[0], logits


def train_surprise_gated(seed):
    """Same curriculum/shape as this module's shared train(), but needs
    to pass position `pos` into step() for the deterministic surprise
    gate -- kept as its own loop rather than overloading the generic
    train() signature."""
    from sili import _cpu

    _cpu.seed_fp4_stochastic_rng(seed)
    rng = np.random.RandomState(seed)
    cell = SurpriseGatedExactCell(seed=seed + 10_000)
    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        n_bits = _sample_n_bits(rng, step)
        tokens, pairs = generate_deviation_sequence(rng, n_bits)
        query_pos, answer = pairs[0]
        M = np.zeros(STATE_WIDTH, dtype=np.float32)
        for i in range(query_pos):
            M, _ = cell.step(int(tokens[i]), M, lr, i)
        M, _ = cell.query_step(int(tokens[query_pos]), M, lr, answer)
    return cell


def evaluate_surprise_gated(cell, seed):
    rng = np.random.RandomState(seed)
    results = {}
    for n_bits in EVAL_N_VALUES:
        correct = 0
        for _ in range(EVAL_SEQUENCES):
            tokens, pairs = generate_deviation_sequence(rng, n_bits)
            query_pos, answer = pairs[0]
            M = np.zeros(STATE_WIDTH, dtype=np.float32)
            for i in range(query_pos):
                M, _ = cell.step(int(tokens[i]), M, 0.0, i)
            _M, logits = cell.query_step(int(tokens[query_pos]), M, 0.0, answer)
            pred = int(np.argmax(logits.data[0]))
            correct += int(pred == answer)
        results[n_bits] = correct / EVAL_SEQUENCES
    return results


def main():
    t0 = time.time()
    N_SEEDS = 50
    W = 2
    agg = {
        name: {n: [] for n in EVAL_N_VALUES}
        for name in ("no_memory", "plain", "peak", "energy_gated", "surprise_gated")
    }
    for s in range(N_SEEDS):
        no_mem_cell = train_nomem(seed=1000 + s)
        plain_cell = train(PlainCell, seed=1000 + s)
        peak_cell = train(PeakSynapseCell, seed=2000 + s)
        energy_cell = train(EnergyGatedCreditCell, seed=1000 + s)
        sg_cell = train_surprise_gated(seed=1000 + s)

        from scripts.prototype_peak_synapse_learning_comparison import evaluate

        row = {
            "no_memory": evaluate(no_mem_cell, seed=5000 + s),
            "plain": evaluate(plain_cell, seed=5000 + s),
            "peak": evaluate(peak_cell, seed=5000 + s),
            "energy_gated": evaluate(energy_cell, seed=5000 + s),
            "surprise_gated": evaluate_surprise_gated(sg_cell, seed=5000 + s),
        }
        for name, res in row.items():
            for n in EVAL_N_VALUES:
                agg[name][n].append(res[n])
        print(
            f"seed {s}: " + "  ".join(f"{name}={ {n: round(row[name][n], 2) for n in EVAL_N_VALUES} }" for name in row)
        )

    print(
        f"\n{'n_bits':>8}  {'in_ctx':>7}  {'no_memory':>17}  {'plain':>17}  {'peak':>17}  {'energy_gated':>17}  "
        f"{'surprise_gated':>17}  {'sg vs nomem p':>13}  {'sg vs plain p':>13}  {'sg vs peak p':>12}  {'sg vs eg p':>10}"
    )
    for n_bits in EVAL_N_VALUES:
        in_ctx = "yes" if n_bits <= W else "NO"
        arrs = {name: np.array(agg[name][n_bits]) for name in agg}

        def fmt(a):
            return f"{a.mean():>8.3f} +- {a.std():<5.3f}"

        _t0, p0 = scipy_stats.ttest_rel(arrs["surprise_gated"], arrs["no_memory"])
        _t1, p1 = scipy_stats.ttest_rel(arrs["surprise_gated"], arrs["plain"])
        _t2, p2 = scipy_stats.ttest_rel(arrs["surprise_gated"], arrs["peak"])
        _t3, p3 = scipy_stats.ttest_rel(arrs["surprise_gated"], arrs["energy_gated"])
        print(
            f"{n_bits:>8}  {in_ctx:>7}  {fmt(arrs['no_memory'])}  {fmt(arrs['plain'])}  {fmt(arrs['peak'])}  "
            f"{fmt(arrs['energy_gated'])}  {fmt(arrs['surprise_gated'])}  {p0:>13.4f}  {p1:>13.4f}  {p2:>12.4f}  {p3:>10.4f}"
        )
    print(f"\n(no_memory is PROVABLY bound at the task prior (0.5); {N_SEEDS} seeds; p-values two-sided, uncorrected)")
    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
