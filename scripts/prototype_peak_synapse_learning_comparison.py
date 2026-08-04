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

Result as of the LATEST run (50 seeds x 40 eval sequences,
STATE_WIDTH=16, TRAIN_STEPS=6000, GENTLE_ENERGY_CONFIG, eval-mode
energy bug fixed -- see below):

  n_bits  in_ctx  plain (mean+-std)  peak-synapse (mean+-std)
       2     yes     0.526 +- 0.223          0.539 +- 0.205
       3      NO     0.514 +- 0.168          0.542 +- 0.133
       4      NO     0.493 +- 0.187          0.502 +- 0.197
       6      NO     0.410 +- 0.197          0.452 +- 0.180

No single point clears its own noise (largest gap, n=6, is only
~1.1 combined SEM at N=50) -- NOT a proven effect. But peak-synapse is
nominally ahead at ALL FOUR points, including every out-of-context
one -- a consistent direction, unlike earlier (pre-fix) runs where the
sign flipped between reruns. Worth continued tracking, not yet worth
calling settled. Both arms remain at/below chance at n=6 specifically
-- the hardest, most out-of-context case is still the weak point for
both, independent of mechanism.

Multiple real bugs found and fixed along the way to get a run worth
trusting at all -- see JOURNAL.md for the full narrative: sili__new's
_preseed_random_sparse ignored np.random.seed() entirely (fixed
upstream, rng injection); FP4's own stochastic weight rounding was
ALSO unseeded (needs seed_fp4_stochastic_rng() per-thread, hence
NUM_CPUS=1 here); MAX_WEIGHTS was landing on a k=1 bare-floor
connection count; and EnergyDynamics -- calibrated at h sizes 20-64 --
was both (a) gating out ~69% of every state-write at this width by
default AND (b) being applied during EVAL calls too (no train/eval
mode of its own), corrupting every accuracy number measured before
these fixes. The isolated mechanism check
(prototype_synapse_peak_credit.py) still holds throughout -- a
silent-at-query-tick row genuinely gets real credit that plain DISLDO
cannot give it.

Selection criterion since revised, derived directly from the true BPTT
sum rather than guessed (see JOURNAL.md for the full derivation):
`dL/dW[r,c] = Sum_t [dL/dh_T . dh_T/dh_t] . x_r(t) . phi'(delta_c(t))`.
Because the state update is residual (h_t = h_{t-1} + phi(delta_t)),
`dh_T/dh_t` is dominated by an identity term regardless of T-t, so
CAPTURE (substituting a tagged x_r(t) into backward_sparse against the
real captured error at query time) is exactly the leading-order term
of that sum -- already validated, no kernel work needed, confirmed
directly against linear_disldo.hpp's disldo_forward: the per-synapse
`contrib = w * iv` is already materialized before the scatter-add, so
a max-tracking addition there would be nearly free, but turns out to
be unnecessary -- SELECTION doesn't need the weight at all. Comparing
candidate ticks for the SAME synapse needs to rank by
`x_r(t) . phi'(delta_c(t))`; comparing DIFFERENT rows competing for the
SAME output c at a FIXED tick has phi'(delta_c(t)) cancel out entirely
(shared by every row feeding c), leaving `|x_r(t)|` alone as correct
there -- so weight never enters either comparison. `phi'(delta_c(t))`
itself isn't cheaply available, but the residual structure gives an
exact stand-in: `state_change(t) = phi(delta_c(t))` IS the tick's
actual state delta (no need to compute a difference, delta_gated
already IS Δh for that tick) -- a reasonable proxy for phi' in the
near-linear regime. Net criterion: `score_r(t) = |x_r(t)| *
|state_change(t)|` (aggregated across output dims, since this
prototype tracks one peak per INPUT ROW, not per (row, col) synapse --
true per-synapse tracking is the bigger, deferred core change).

`decay_from_horizon(n, retain_fraction)`: derived, not tuned --
P(a peak set now survives n further ticks unbeaten) requires the
decayed threshold to stay low enough for that long; solving
`decay**n = retain_fraction` for decay gives the minimum decay rate
that keeps at least `retain_fraction` of the tag's original score
alive out to a target credit-assignment horizon of `n` ticks.

Known limitation, not yet worked around: one-hot token rows have
CONSTANT magnitude every time they fire, so any later occurrence of
the same token always beats a decayed peak of the same original
magnitude regardless of decay rate -- for those rows specifically, the
mechanism can't reach back further than the token's own most recent
occurrence. The STATE-portion rows (M_prev, continuously-valued,
carrying an additive trace of everything written so far thanks to the
residual update) don't have this problem the same way, though whether
the recurrent dynamics preserve or cancel an old write is an open,
state-width-dependent question -- plausibly safer with a wide
recurrent state (matching this project's own established
column-averaging design principle), not yet tested here.

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
STATE_WIDTH = 16
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
TRAIN_STEPS = 6000
WARMUP_STEPS = 300
STEPS_PER_LEVEL = 600
CURRICULUM_WINDOW = 2
PEAK_LR = 0.05
EVAL_SEQUENCES = 40
EVAL_N_VALUES = [2, 3, 4, 6]
# Verified directly (not guessed): the ORIGINAL calibration
# (drive=0.1, activation_cost=0.05, precision=0.01, density=0.05,
# p=0.3), taken from a config validated at h sizes 20-64, defaults to
# gating out ~69% of this tick's state-write at our much smaller
# STATE_WIDTH (measured: 5/16 neurons pass at init, actual_p=0.3125 --
# the model NEVER gets past its own hard ceiling `p`, since `density`
# is a target the gate only grows toward via training, not the
# starting point). That was catastrophic at this scale: a fixed
# n_bits=2 sanity check (the easiest possible version of this task)
# went from loss 0.009/100% accuracy with energy off to loss staying
# ABOVE chance-level and 22-45% final accuracy with the original
# config on. This config turns activation_cost/drive/precision/
# exploration down to their practical floors (activation_cost=0.01 is
# EnergyDynamics' own hard minimum) and p/density up to fully
# permissive (measured: 16/16 neurons pass at init, actual_p=1.0) --
# same fixed-n_bits=2 check recovers to 0.750+ accuracy with this
# config once the SEPARATE eval-mode bug (below) is also fixed.
#
# drive=0.005, not higher: a single still-collapsing seed's own
# `energy.energy` array, traced over its full training run at
# drive=0.005, showed mean energy trending NEGATIVE (settles ~-1.2 to
# -1.6, only 0-3/16 neurons near the firing threshold at any snapshot,
# down from all 16 at init) -- activation_cost*|h| drains energy
# proportional to a neuron's OWN real magnitude while drive accumulates
# it at a flat rate, so a neuron with genuinely large, useful output
# gets pushed toward the shutoff floor rather than toward firing.
# Raising drive to 0.05 DID flip that ONE seed's final mean energy
# positive (-1.10 -> +1.05) as predicted -- but made the real 8-seed
# stability check WORSE overall (means dropped from 0.41-0.60 to
# 0.34-0.35, MORE seeds hit exact 0.0 collapse, not fewer). A fix
# validated on one seed's own diagnostic doesn't necessarily generalize
# -- reverted to 0.005, the config that actually performed better
# across the full seed set, not the one that looked better on paper
# for a single case.
GENTLE_ENERGY_CONFIG = dict(drive=0.005, activation_cost=0.01, precision=0.0002,
                            density=0.75, exploration=0.0001, p=0.99)


def onehot(tok):
    v = np.zeros(VOCAB_SIZE, dtype=np.float32)
    v[tok] = 1.0
    return v


def decay_from_horizon(horizon: int, retain_fraction: float) -> float:
    """Derived, not tuned: the minimum peak_decay that keeps a tag's
    score at or above `retain_fraction` of its original value after
    `horizon` further ticks with no replacement (decay**horizon ==
    retain_fraction). E.g. decay_from_horizon(100, 0.1) ~= 0.977 --
    "I want tags to plausibly survive out to 100 ticks, retaining at
    least 10% of their original score.\""""
    if not (0.0 < retain_fraction < 1.0):
        raise ValueError(f"retain_fraction must be in (0,1), got {retain_fraction}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return retain_fraction ** (1.0 / horizon)


# Derived from the actual horizon this script tests (OUT_OF_CONTEXT_MAX
# ticks, the longest gap a tag might need to survive) and a 10% retain
# target -- replaces the earlier hardcoded peak_decay=0.9, which wasn't
# calibrated to any particular horizon at all.
PEAK_DECAY = decay_from_horizon(OUT_OF_CONTEXT_MAX, 0.1)


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
        self.energy = EnergyDynamics(**GENTLE_ENERGY_CONFIG) if use_energy else None

    def step(self, tok, M_prev, lr):
        x = np.concatenate([onehot(tok), M_prev])[None, :]
        delta = self.cell.forward(x, lr)
        if self.energy is not None and lr != 0.0:
            # EnergyDynamics is a TRAINING-time mechanism only -- not
            # designed for eval/inference (direct correction: no
            # train/eval mode of its own, so applying it during eval
            # runs the trained weights through a DIFFERENT, still-noisy
            # computational path than what any temperature-style
            # inference use would need real reparameterization for,
            # not just reuse). lr==0.0 is this codebase's own existing
            # convention for "this is an eval call" (matches every
            # other lr==0 skip elsewhere in this file and in DISLDO's
            # own backward) -- skip the gate entirely then, use the
            # raw, ungated delta.
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
    it -- just incorporating energy's own local derivative too.

    Selection score = |x_r(t)| * |state_change(t)| -- DERIVED (see
    module docstring), not the earlier magnitude-only |x_r(t)| version:
    `state_change` (= delta_gated for this tick, since the residual
    update makes Δh literally equal to the gated cell output, no
    separate subtraction needed) stands in for the otherwise-unavailable
    phi'(delta_c(t)) term in the true per-tick gradient contribution.
    The weight is deliberately NOT part of this score -- shown not to
    belong in either the same-row-across-time or same-tick-across-row
    comparison."""

    def __init__(self, seed=None, peak_decay=PEAK_DECAY, correction_lr_mult=1.0, use_energy=True):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(None if seed is None else seed + 1)
        self.cell = DISLDOLayer(IN_FEATURES, STATE_WIDTH, MAX_WEIGHTS, NUM_CPUS, rng=rng1)
        self.head = DISLDOLayer(STATE_WIDTH, VOCAB_SIZE, MAX_WEIGHTS, NUM_CPUS, rng=rng2)
        self.energy = EnergyDynamics(**GENTLE_ENERGY_CONFIG) if use_energy else None
        self.peak_decay = peak_decay
        self.correction_lr_mult = correction_lr_mult
        self.peak = np.zeros(IN_FEATURES, dtype=np.float32)         # signed x_r at the winning tick
        self.peak_score = np.zeros(IN_FEATURES, dtype=np.float32)   # |x_r| * |state_change| at that tick

    def _update_peak(self, x_row, state_change_scale):
        score = np.abs(x_row) * state_change_scale
        decayed_val = self.peak_decay * self.peak
        decayed_score = self.peak_decay * self.peak_score
        replace = score > decayed_score
        self.peak = np.where(replace, x_row, decayed_val)
        self.peak_score = np.where(replace, score, decayed_score)

    def _cell_step(self, tok, M_prev, lr):
        """Shared forward: cell + energy gate. Returns (x_row,
        delta_raw, delta_gated) -- delta_gated IS this tick's Δstate
        (residual update: M_new = M_prev + delta_gated exactly), used
        both for the residual add AND as the state_change_scale for
        peak selection."""
        x_row = np.concatenate([onehot(tok), M_prev])
        x = x_row[None, :]
        delta = self.cell.forward(x, lr)
        delta_gated = delta
        aux_loss = None
        if self.energy is not None and lr != 0.0:
            # See PlainCell.step's own comment: EnergyDynamics is
            # training-only, skip it entirely at eval (lr==0.0).
            delta_gated, aux_loss, _p = self.energy(delta.reshape((STATE_WIDTH,)))
            delta_gated = delta_gated.reshape((1, STATE_WIDTH))
        return x_row, delta, delta_gated, aux_loss

    def step(self, tok, M_prev, lr):
        x_row, delta, delta_gated, _aux = self._cell_step(tok, M_prev, lr)
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
                if self.peak_score[r] <= ZERO_EPS:
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
