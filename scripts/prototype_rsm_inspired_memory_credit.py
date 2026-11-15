"""The validated result of the synaptic-hold/e-prop BPTT-approximation
investigation (see JOURNAL.md, 2026-11-10 entry, for the full 5-attempt
negative history this follows and the Bend proof it's grounded in --
`sili__new/bend_proofs/synaptic_hold_bptt_approximation`).

Five self-supervised attempts to replace the task's own oracle (which
knows DEVIATION_BASE_PATTERN) with a learned surprise/selection signal
all failed -- each tried to figure out WHICH past ticks to credit,
exactly the problem the oracle solved by cheating. Research (direct
instruction, after the 5th negative result: "research to find if
someone else found the answer") surfaced Srivastava/Van Hoesel/
Hawkins-style "Recurrent Sparse Memory" (RSM, arXiv:1905.11589,
"Learning distant cause and effect using only local and immediate
credit assignment") -- a different approach that sidesteps the
selection problem entirely: make the memory's OWN representation
combinatorially sparse via winner-take-all + REFRACTORY SELF-
INHIBITION (competition across TIME via a per-unit decaying trace, not
across space), trained with only a SHALLOW (backprop depth 2: decoder
then cell), purely LOCAL, self-supervised next-token loss every tick.
No held-row credit routing, no separate surprise predictor, no
query-time correction hack, no BPTT, no eligibility trace.

Confirmed at 30 seeds then REPLICATED at 50 seeds (both at a 1500-step
budget, k_active in {3,4,6} out of STATE_WIDTH=16): significantly beats
BOTH no_memory and plain at n_bits in {2,3,4} (n=3,4 genuinely
out-of-context, window=2) -- the first non-oracle mechanism in this
entire investigation to show a real, replicated win. Honest boundary:
does NOT extend to n=6 -- actively worse there for k=3/4, tied for
k=6. See JOURNAL.md for the full number table; this script reproduces
that comparison.

Run: python -m scripts.prototype_rsm_inspired_memory_credit
"""

import time

import numpy as np
from scipy import stats as scipy_stats
from sili.energy import EnergyDynamics
from sili.sparse_rnn import DISLDOLayer
from sili.tensor import Tensor

import scripts.prototype_surprise_gated_exact_credit as _sg_module
from model.toy_beyond_context_task import generate_deviation_sequence
from model.toy_recall_models import cross_entropy_sum
from scripts.prototype_energy_gated_synapse_credit import train_nomem
from scripts.prototype_peak_synapse_learning_comparison import (
    EVAL_N_VALUES,
    EVAL_SEQUENCES,
    GENTLE_ENERGY_CONFIG,
    IN_FEATURES,
    MAX_WEIGHTS,
    NUM_CPUS,
    PEAK_LR,
    STATE_WIDTH,
    VOCAB_SIZE,
    WARMUP_STEPS,
    PlainCell,
    _sample_n_bits,
    evaluate,
    lr_schedule,
    onehot,
    train,
)
from scripts.prototype_surprise_gated_exact_credit import evaluate_surprise_gated, train_surprise_gated

TRAIN_STEPS = 1500  # the validated scale -- not yet confirmed at the full 6000-step budget

# The oracle reference arm's own module defaults to the full 6000-step
# budget; override so it runs at the SAME scale as the RSM arms below.
_sg_module.TRAIN_STEPS = TRAIN_STEPS
_sg_module.WARMUP_STEPS = WARMUP_STEPS


class RSMInspiredCell:
    """Winner-take-all + refractory self-inhibition on the cell's own
    state-update magnitude, trained via a shallow local next-token
    decoder -- no credit routing of any kind."""

    def __init__(self, seed=None, k_active=4, refractory_decay=0.7, use_energy=False):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(None if seed is None else seed + 1)
        rng3 = np.random.default_rng(None if seed is None else seed + 2)
        self.cell = DISLDOLayer(IN_FEATURES, STATE_WIDTH, MAX_WEIGHTS, NUM_CPUS, rng=rng1)
        self.head = DISLDOLayer(STATE_WIDTH, VOCAB_SIZE, MAX_WEIGHTS, NUM_CPUS, rng=rng2)
        # Shallow local decoder: predicts the NEXT token directly from
        # this tick's sparse activity y -- backprop depth 2 (decoder
        # then cell), matching RSM's own discipline.
        self.decoder = DISLDOLayer(STATE_WIDTH, VOCAB_SIZE, STATE_WIDTH * VOCAB_SIZE, NUM_CPUS, rng=rng3)
        self.k_active = k_active
        self.refractory_decay = refractory_decay
        self.energy = EnergyDynamics(**GENTLE_ENERGY_CONFIG) if use_energy else None
        self.phi = np.zeros(STATE_WIDTH, dtype=np.float32)  # refractory trace, reset per sequence

    def reset_sequence(self):
        self.phi = np.zeros(STATE_WIDTH, dtype=np.float32)

    def _sparsify(self, delta_np):
        """Hard winner-take-all + refractory self-inhibition. Returns
        a 0/1 mask -- gradient flows through surviving units' VALUES
        only, not through this selection (the standard k-sparse /
        straight-through convention)."""
        activity = np.abs(delta_np)
        priority = (1.0 - self.phi) * activity
        top_idx = np.argsort(priority)[-self.k_active :]
        mask = np.zeros(STATE_WIDTH, dtype=np.float32)
        mask[top_idx] = 1.0
        self.phi = np.maximum(self.phi * self.refractory_decay, mask)
        return mask

    def step(self, tok, M_prev, lr, next_tok=None):
        x_row = np.concatenate([onehot(tok), M_prev])
        x = x_row[None, :]
        delta = self.cell.forward(x, lr)
        delta_np = np.asarray(delta.data[0], dtype=np.float32)
        mask = self._sparsify(delta_np)
        y = delta * Tensor(mask[None, :].astype(np.float32))

        if self.energy is not None and lr != 0.0:
            y, _aux, _p = self.energy(y.reshape((STATE_WIDTH,)))
            y = y.reshape((1, STATE_WIDTH))

        if lr != 0.0 and next_tok is not None:
            pred_logits = self.decoder.forward(y, lr)
            pred_loss = cross_entropy_sum(pred_logits, [(0, next_tok)])
            pred_loss.backward()

        M_new = Tensor(M_prev[None, :].astype(np.float32)) + y
        logits = self.head.forward(M_new, lr)
        return M_new.data[0], logits

    def query_step(self, tok, M_prev, lr, answer):
        """Query tick: ungated raw delta -- the query token isn't part
        of the body's regularity, don't artificially sparsify it."""
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
        return M_new.data[0], logits


def train_rsm(seed, k_active=4):
    from sili import _cpu

    _cpu.seed_fp4_stochastic_rng(seed)
    rng = np.random.RandomState(seed)
    cell = RSMInspiredCell(seed=seed + 10_000, k_active=k_active)
    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        n_bits = _sample_n_bits(rng, step)
        tokens, pairs = generate_deviation_sequence(rng, n_bits)
        query_pos, answer = pairs[0]
        cell.reset_sequence()
        M = np.zeros(STATE_WIDTH, dtype=np.float32)
        for i in range(query_pos):
            next_tok = int(tokens[i + 1]) if i + 1 < query_pos else None
            M, _ = cell.step(int(tokens[i]), M, lr, next_tok=next_tok)
        M, _ = cell.query_step(int(tokens[query_pos]), M, lr, answer)
    return cell


def evaluate_rsm(cell, seed):
    rng = np.random.RandomState(seed)
    results = {}
    for n_bits in EVAL_N_VALUES:
        correct = 0
        for _ in range(EVAL_SEQUENCES):
            tokens, pairs = generate_deviation_sequence(rng, n_bits)
            query_pos, answer = pairs[0]
            cell.reset_sequence()
            M = np.zeros(STATE_WIDTH, dtype=np.float32)
            for i in range(query_pos):
                M, _ = cell.step(int(tokens[i]), M, 0.0)
            _M, logits = cell.query_step(int(tokens[query_pos]), M, 0.0, answer)
            pred = int(np.argmax(logits.data[0]))
            correct += int(pred == answer)
        results[n_bits] = correct / EVAL_SEQUENCES
    return results


def main():
    t0 = time.time()
    N_SEEDS = 50
    k_configs = [3, 4, 6]
    names = ("no_memory", "plain", "surprise_gated_oracle", *(f"rsm_k{k}" for k in k_configs))
    agg = {name: {n: [] for n in EVAL_N_VALUES} for name in names}

    for s in range(N_SEEDS):
        cells = {
            "no_memory": train_nomem(seed=1000 + s),
            "plain": train(PlainCell, seed=1000 + s),
            "surprise_gated_oracle": train_surprise_gated(seed=1000 + s),
        }
        for k in k_configs:
            cells[f"rsm_k{k}"] = train_rsm(seed=1000 + s, k_active=k)
        row = {}
        for name, cell in cells.items():
            if name == "surprise_gated_oracle":
                res = evaluate_surprise_gated(cell, seed=5000 + s)
            elif name.startswith("rsm_k"):
                res = evaluate_rsm(cell, seed=5000 + s)
            else:
                res = evaluate(cell, seed=5000 + s)
            row[name] = res
            for n in EVAL_N_VALUES:
                agg[name][n].append(res[n])
        print(
            f"seed {s}: "
            + "  ".join(f"{name}={ {n: round(row[name][n], 2) for n in EVAL_N_VALUES} }" for name in names)
        )

    print(f"\n{'n_bits':>8}  " + "  ".join(f"{name:>21}" for name in names) + "  rsm_vs_nomem_p  rsm_vs_plain_p")
    for n_bits in EVAL_N_VALUES:
        arrs = {name: np.array(agg[name][n_bits]) for name in names}

        def fmt(a):
            return f"{a.mean():>8.3f} +- {a.std():<5.3f}"

        for cname in (f"rsm_k{k}" for k in k_configs):
            _t1, p_nomem = scipy_stats.ttest_rel(arrs[cname], arrs["no_memory"])
            _t2, p_plain = scipy_stats.ttest_rel(arrs[cname], arrs["plain"])
            print(f"{n_bits:>8}  {cname:>10}: {fmt(arrs[cname])}  vs_nomem_p={p_nomem:.4f}  vs_plain_p={p_plain:.4f}")
    print(f"\n(no_memory is PROVABLY bound at the task prior (0.5); {N_SEEDS} seeds; p-values two-sided, uncorrected)")
    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
