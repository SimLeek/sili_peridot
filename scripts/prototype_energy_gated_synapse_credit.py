"""Does reusing EnergyDynamics' own homeostatic selection (the same
mechanism already validated for sleep annealing and Arm G/H backward
selection) as the WHICH-ROWS-TO-CREDIT gate beat the bespoke "silent +
peak_score > threshold" criterion prototype_peak_synapse_learning_
comparison.py uses? Direct motivation: that criterion is really just
magnitude-guided importance selection again, which has now lost twice
in this project (Arm C vs signal-guided on the column axis; blind
round-robin vs magnitude-ranking on the tile-cache row axis) -- and it
assumes one-hot inputs (exact zero = "silent"), which doesn't even
apply to ToyTileRecurrenceRMT's continuous embeddings. EnergyDynamics
is already a graded, decay-based, non-hard-zero tracker built for
continuous activations, so it sidesteps both problems at once.

Four arms, not two: `no_memory` (M always zero -- a PROVABLE, not just
empirical, chance-level floor: generate_deviation_sequence's QUERY_TOKEN
is a fixed constant, unconditional on the answer, so with M=0 the
query-time input is identical regardless of the true answer, bounding
accuracy at the prior by construction) and `plain` (has a real memory
channel, but NO mechanism connecting what gets written into memory at
one tick to whether that helped a later query -- query_step's single
loss.backward() only ever credits THAT tick's own weights, no BPTT) are
both expected to sit at/below chance on this task, for different,
clearly-separated reasons. `peak` and `energy_gated` are the two
candidate BPTT-approximating credit mechanisms being compared -- the
real question is whether EITHER rescues memory use relative to those
two chance-bound baselines, not just whether they beat each other.

Reuses this module's own task/eval/train harness unchanged -- only the
cell's correction mechanism differs, so the comparison isolates exactly
that one variable.

Run: python -m scripts.prototype_energy_gated_synapse_credit
"""

import time

import numpy as np
from scipy import stats as scipy_stats
from sili import _cpu
from sili.energy import EnergyDynamics
from sili.sparse_rnn import DISLDOLayer
from sili.tensor import Tensor

from model.toy_recall_models import cross_entropy_sum
from scripts.prototype_peak_synapse_learning_comparison import (
    EVAL_N_VALUES,
    GENTLE_ENERGY_CONFIG,
    IN_FEATURES,
    MAX_WEIGHTS,
    NUM_CPUS,
    PEAK_LR,
    STATE_WIDTH,
    TRAIN_STEPS,
    VOCAB_SIZE,
    WARMUP_STEPS,
    PeakSynapseCell,
    PlainCell,
    W,
    _sample_n_bits,
    evaluate,
    generate_deviation_sequence,
    lr_schedule,
    onehot,
    train,
)

# density*n_features ~= how many of IN_FEATURES=26 get selected per
# tick; p/density/drive/activation_cost satisfy EnergyDynamics' own
# constructor asserts (density <= p*0.8, etc). decay=0.0 -- no mean-
# reversion needed, this instance only ever tracks+selects, no forced
# activation change (its own aux_loss is never added to the task loss).
CREDIT_ENERGY_CONFIG = {
    "drive": 0.1,
    "activation_cost": 0.05,
    "precision": 0.01,
    "density": 0.15,
    "p": 0.3,
    "exploration": 0.01,
    "reactivity": 0.01,
    "decay": 0.0,
}


class EnergyGatedCreditCell:
    """Same cell/head/forward-gating energy as PeakSynapseCell. Credit
    selection differs: a SEPARATE EnergyDynamics instance tracks x_row
    (size IN_FEATURES) purely for its kept_indices ("the gate decision
    energy already made") -- no bespoke peak/decay formula. The
    remembered value to credit a selected row WITH is a plain EMA of
    x_row (x_ema), not a derived "score" -- EnergyDynamics' own state
    supplies the selection, the EMA supplies what value survived."""

    def __init__(
        self,
        seed=None,
        use_energy=True,
        credit_energy_kwargs=None,
        x_ema_decay=0.9,
        correction_lr_mult=1.0,
    ):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(None if seed is None else seed + 1)
        self.cell = DISLDOLayer(IN_FEATURES, STATE_WIDTH, MAX_WEIGHTS, NUM_CPUS, rng=rng1)
        self.head = DISLDOLayer(STATE_WIDTH, VOCAB_SIZE, MAX_WEIGHTS, NUM_CPUS, rng=rng2)
        self.energy = EnergyDynamics(**GENTLE_ENERGY_CONFIG) if use_energy else None
        self.credit_energy = EnergyDynamics(**(credit_energy_kwargs or CREDIT_ENERGY_CONFIG))
        self.x_ema = np.zeros(IN_FEATURES, dtype=np.float32)
        self.x_ema_decay = x_ema_decay
        self.correction_lr_mult = correction_lr_mult

    def _cell_step(self, tok, M_prev, lr):
        x_row = np.concatenate([onehot(tok), M_prev])
        x = x_row[None, :]
        delta = self.cell.forward(x, lr)
        delta_gated = delta
        aux_loss = None
        if self.energy is not None and lr != 0.0:
            delta_gated, aux_loss, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta_gated = delta_gated.reshape((1, STATE_WIDTH))
        if lr != 0.0:
            # Side effect only -- this call's own aux_loss is discarded,
            # never backpropped. We want kept_indices, not a forced
            # activation change on x_row itself.
            self.credit_energy(Tensor(x[0:1].astype(np.float32)))
        self.x_ema = self.x_ema_decay * self.x_ema + (1.0 - self.x_ema_decay) * x_row
        return x_row, delta, delta_gated, aux_loss

    def step(self, tok, M_prev, lr):
        _x_row, _delta, delta_gated, _aux = self._cell_step(tok, M_prev, lr)
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta_gated
        logits = self.head.forward(M_new, lr)
        return M_new.data[0], logits

    def query_step(self, tok, M_prev, lr, answer):
        _x_row, delta, delta_gated, aux_loss = self._cell_step(tok, M_prev, lr)
        M_new = Tensor(M_prev[None, :].astype(np.float32)) + delta_gated
        logits = self.head.forward(M_new, lr)
        loss = cross_entropy_sum(logits, [(0, answer)])
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()

        kept = self.credit_energy.kept_indices
        if delta.grad is not None and kept is not None and kept.size > 0:
            dy = np.asarray(delta.grad, dtype=np.float32)[np.newaxis, :]
            dp, di, dv = _cpu.dense_to_top_k_csr(dy, dy.shape[1], NUM_CPUS)
            for r in kept:
                x_1hot = np.zeros((1, IN_FEATURES), dtype=np.float32)
                x_1hot[0, r] = self.x_ema[r]
                self.cell._c.backward_sparse(x_1hot, dp, di, dv, 1, lr * self.correction_lr_mult, lr_per_row_nnz=True)
        return M_new.data[0], logits


class NoMemoryCell(PlainCell):
    """Chance-bound-by-construction control: M_prev is ALWAYS replaced
    with zero, never carries real history. generate_deviation_sequence's
    QUERY_TOKEN is a fixed constant unconditional on the answer, so with
    M=0 the query-time input is IDENTICAL regardless of the true answer
    -- a deterministic network fed an identical input must produce an
    identical output distribution, bounding accuracy at the task's prior
    (0.5 here) by construction, not merely "expected empirically." See
    this module's own docstring."""

    def step(self, tok, M_prev, lr):
        return super().step(tok, np.zeros(STATE_WIDTH, dtype=np.float32), lr)

    def query_step(self, tok, M_prev, lr, answer):
        return super().query_step(tok, np.zeros(STATE_WIDTH, dtype=np.float32), lr, answer)


def train_nomem(seed):
    """Same shape as this module's own train(), but NoMemoryCell takes
    no cell_cls arg (M_prev is ignored inside the cell itself, not at
    the call site) -- kept separate rather than overloading train()."""
    _cpu.seed_fp4_stochastic_rng(seed)
    rng = np.random.RandomState(seed)
    cell = NoMemoryCell(seed=seed + 10_000)
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


def main():
    t0 = time.time()
    N_SEEDS = 50
    arms = {"no_memory": None, "plain": PlainCell, "peak": PeakSynapseCell, "energy_gated": EnergyGatedCreditCell}
    agg = {name: {n: [] for n in EVAL_N_VALUES} for name in arms}
    for s in range(N_SEEDS):
        row = {}
        for name, cls in arms.items():
            if name == "no_memory":
                cell = train_nomem(seed=1000 + s)
            else:
                cell = train(cls, seed=1000 + s if name != "peak" else 2000 + s)
            res = evaluate(cell, seed=5000 + s)
            for n in EVAL_N_VALUES:
                agg[name][n].append(res[n])
            row[name] = {n: round(res[n], 2) for n in EVAL_N_VALUES}
        print(f"seed {s}: " + "  ".join(f"{name}={row[name]}" for name in arms))

    print(
        f"\n{'n_bits':>8}  {'in_ctx':>7}  {'no_memory':>17}  {'plain':>17}  {'peak':>17}  {'energy_gated':>17}  "
        f"{'plain vs nomem p':>16}  {'peak vs nomem p':>15}  {'eg vs nomem p':>13}  {'eg vs peak p':>12}"
    )
    for n_bits in EVAL_N_VALUES:
        in_ctx = "yes" if n_bits <= W else "NO"
        nm_arr = np.array(agg["no_memory"][n_bits])
        p_arr = np.array(agg["plain"][n_bits])
        k_arr = np.array(agg["peak"][n_bits])
        e_arr = np.array(agg["energy_gated"][n_bits])

        def fmt(a):
            return f"{a.mean():>8.3f} +- {a.std():<5.3f}"

        _t0, p0 = scipy_stats.ttest_rel(p_arr, nm_arr)
        _t1, p1 = scipy_stats.ttest_rel(k_arr, nm_arr)
        _t2, p2 = scipy_stats.ttest_rel(e_arr, nm_arr)
        _t3, p3 = scipy_stats.ttest_rel(e_arr, k_arr)
        print(
            f"{n_bits:>8}  {in_ctx:>7}  {fmt(nm_arr)}  {fmt(p_arr)}  {fmt(k_arr)}  {fmt(e_arr)}  "
            f"{p0:>16.4f}  {p1:>15.4f}  {p2:>13.4f}  {p3:>12.4f}"
        )
    print(f"\n(no_memory is PROVABLY bound at the task prior (0.5); {N_SEEDS} seeds; p-values two-sided, uncorrected)")
    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
