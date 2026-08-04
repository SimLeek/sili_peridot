"""
scripts/train_toy_beyond_context_comparison.py
───────────────────────────────────────────────
Tier 1 of the out-of-context benchmark suite -- see the approved plan
(fuzzy-plotting-starlight.md) for the full three-tier design.
Running-parity task (model/toy_beyond_context_task.py): does
tile-recurrence's recurrent state genuinely extend effective range
BEYOND a fixed local window, where a structurally-windowed dense
transformer cannot -- the thing explicitly deferred twice earlier this
session (every MQAR run used num_tiles=seq_len, giving full visibility
by construction).

Both models get the SAME window `W`: `ToySmallTransformer(half_bandwidth=W,
n_layers=1)` (genuinely bounded causal visibility -- `n_layers=1` is
required for `half_bandwidth=W` to actually MEAN "structurally cannot
see past W positions": checked directly via a perturbation probe that
2 STACKED half_bandwidth=W layers give an effective receptive field of
2*W, not W -- a real bug in the first version of this script) vs
`ToyTileRecurrenceRealFP4(num_tiles=W)` (real sili, plain DISLDOLayer,
combined-input+state attention already fixed -- the just-validated
best config from the MQAR sweep).

Curriculum, per direct correction (the first version of this script
used two coarse phases -- in-context, then an abrupt jump to sampling
UNIFORMLY across the entire out-of-context range -- and never
converged): the max problem size must grow ONE INCREMENT AT A TIME
past the in-context ceiling (now [[feedback_gradual_out_of_context_curriculum]]).
Direct reasoning: "learning in context is O(1), learning in recurrent
state is O(1), but learning something that passed recurrent state is
O(random chance), and that chance is much less likely if the network
never learned stuff before" -- jumping straight to n_bits=16 (4x the
window) before the network had ever practiced even n_bits=W+1 (one
step beyond the window) needed several compounding, never-yet-trained
recurrent-carry steps to work simultaneously. `_sample_n_bits` instead
ramps a `level` ceiling from `W` up to `OUT_OF_CONTEXT_MAX` by exactly
1 every `STEPS_PER_LEVEL` steps, sampling from a small sliding window
just behind the current frontier (some review, but concentrated on the
newest increment) -- applies to every future tier (brackets, integer
math) built on this same pattern, not just this one.

Evaluation is an ACCURACY-VS-N CURVE (n_bits swept from in-context to
deeply out-of-context), not a single number -- the claim under test is
the SHAPE of the curve: dense should collapse toward chance (0.5) once
n_bits > W; tile-recurrence should stay meaningfully above chance if
it learned to carry running parity in M.

Run: python -m scripts.train_toy_beyond_context_comparison
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from model.toy_beyond_context_task import generate_parity_sequence, VOCAB_SIZE
from model.toy_recall_models import (
    ToySmallTransformer, cross_entropy_sum, predicted_token,
    AdamOptimizer, clip_grad_norm_, lr_schedule,
)
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from model.toy_precision_models import PeakEligibilityDISLDOLayer
from sili.sparse_rnn import DISLDOLayer

W = 4                       # the fixed window -- num_tiles for tile-recurrence,
                             # half_bandwidth for the dense baseline
TRAIN_STEPS = 120_000       # ~40x the first real run (3000) -- that run measured
                            # ~93s total training time across all 3 models, too
                            # quick to give each of the 13 curriculum levels enough
                            # dedicated practice (per direct decision: "give it an
                            # hour or so this time"). Scaled steps AND STEPS_PER_LEVEL/
                            # WARMUP_STEPS together, same ratios, same curriculum
                            # shape -- just far more repetitions at every level.
WARMUP_STEPS = 4_000
PEAK_LR = 0.02
MAX_GRAD_NORM = 1.0
OUT_OF_CONTEXT_MAX = 4 * W
STEPS_PER_LEVEL = 6_000     # steps of practice before advancing the ceiling by 1
                            # (2..OUT_OF_CONTEXT_MAX is (OUT_OF_CONTEXT_MAX-1) levels
                            # -- ramp finishes well before TRAIN_STEPS, leaving real
                            # consolidation time at the final out-of-context ceiling)
CURRICULUM_WINDOW = 3       # sample from [level-CURRICULUM_WINDOW, level], not just
                            # the exact frontier -- some review, concentrated recent
EVAL_SEQUENCES = 60
EVAL_N_VALUES = [2, 4, 8, 16, 24]

EMBED_WIDTH = 32
HIDDEN = 32
MLP_HIDDEN = 48
COLUMN_NEURONS = 8
TILE_MLP_HIDDEN = EMBED_WIDTH * COLUMN_NEURONS * 2
MAX_WEIGHTS_PER_LAYER = 4096  # genuine sparse budget -- see JOURNAL.md's
                              # "ToyTileRecurrence ported onto real sili" entry
                              # for why this must NOT be sized as in*out
NUM_CPUS = 4


def _sample_n_bits(rng: np.random.RandomState, step: int) -> int:
    """Gradual curriculum -- see module docstring's own
    [[feedback_gradual_out_of_context_curriculum]] rationale. `level`
    (the current ceiling) starts at W (in-context) and grows by
    exactly 1 every STEPS_PER_LEVEL steps up to OUT_OF_CONTEXT_MAX;
    n_bits is sampled from a small sliding window just behind that
    frontier, never jumping straight to a wide out-of-context range."""
    level = min(OUT_OF_CONTEXT_MAX, W + step // STEPS_PER_LEVEL)
    lo = max(2, level - CURRICULUM_WINDOW)
    return int(rng.randint(lo, level + 1))


def _build_window(embed_table: np.ndarray, tokens: np.ndarray, i: int,
                  num_tiles: int, M_prev: np.ndarray, column_neurons: int) -> np.ndarray:
    state_width = embed_table.shape[1] * column_neurons
    window = np.empty((num_tiles, state_width), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        window[j] = (np.repeat(embed_table[tokens[src]], column_neurons)
                     if src >= 0 else M_prev[j])
    return window


def _train_tile(disldo_cls, use_energy: bool, seed: int):
    """Trains one tile-recurrence variant. `disldo_cls` swaps the
    underlying per-weight training mechanism (plain DISLDOLayer,
    PeakEligibilityDISLDOLayer, ...) via ToyTileRecurrenceRealFP4's own
    existing disldo_cls= pattern -- see [[project_sili_bptt_alternatives]]
    for why a credit-assignment mechanism is being tested here (e-prop
    tried first, found flawed, replaced -- see JOURNAL.md's postmortem).
    `use_energy` toggles EnergyDynamics
    independently (see model/toy_tile_precision_models.py's own
    docstring and [[project_sili_bptt_or_chance]] for why energy is
    worth testing specifically on THIS task, unlike the MQAR sweep
    where it hurt)."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    tile = ToyTileRecurrenceRealFP4(VOCAB_SIZE, EMBED_WIDTH, COLUMN_NEURONS, TILE_MLP_HIDDEN,
                                    W, MAX_WEIGHTS_PER_LAYER, num_cpus=NUM_CPUS,
                                    disldo_cls=disldo_cls, use_energy=use_energy)
    tile_opt = AdamOptimizer()
    tile_embed = rng.randn(VOCAB_SIZE, EMBED_WIDTH).astype(np.float32) * 0.3
    state_width = EMBED_WIDTH * COLUMN_NEURONS

    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        n_bits = _sample_n_bits(rng, step)
        tokens, pairs = generate_parity_sequence(rng, n_bits)
        query_pos, answer = pairs[0]

        M = np.zeros((W, state_width), dtype=np.float32)
        for i in range(query_pos + 1):
            window = _build_window(tile_embed, tokens, i, W, M, COLUMN_NEURONS)
            M, logits_t, aux_loss = tile.step(window, M, lr)
            if i == query_pos:
                loss_t = cross_entropy_sum(logits_t, [(W - 1, answer)])
                if aux_loss is not None:
                    loss_t = loss_t + aux_loss
                loss_t.grad = np.array(1.0, dtype=np.float32)
                loss_t.backward()
                clip_grad_norm_(tile.parameters_for_optimizer(), MAX_GRAD_NORM)
                tile_opt.step(tile.parameters_for_optimizer(), lr=lr)

    return tile, tile_embed, rng


def train_dense():
    rng = np.random.RandomState(1)
    np.random.seed(1)
    dense = ToySmallTransformer(VOCAB_SIZE, HIDDEN, MLP_HIDDEN, n_layers=1,
                                num_cpus=2, half_bandwidth=W)
    dense_opt = AdamOptimizer()
    dense_embed = rng.randn(VOCAB_SIZE, HIDDEN).astype(np.float32) * 0.3

    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        n_bits = _sample_n_bits(rng, step)
        tokens, pairs = generate_parity_sequence(rng, n_bits)
        embedded = dense_embed[tokens]
        logits = dense.forward(embedded)
        loss = cross_entropy_sum(logits, pairs)
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        clip_grad_norm_(dense.parameters(), MAX_GRAD_NORM)
        dense_opt.step(dense.parameters(), lr=lr)

    return dense, dense_embed, rng


def evaluate_dense(dense, dense_embed, rng):
    results = {}
    for n_bits in EVAL_N_VALUES:
        correct = 0
        for _ in range(EVAL_SEQUENCES):
            tokens, pairs = generate_parity_sequence(rng, n_bits)
            query_pos, answer = pairs[0]
            embedded = dense_embed[tokens]
            logits = dense.forward(embedded)
            correct += int(predicted_token(logits, query_pos) == answer)
        results[n_bits] = correct / EVAL_SEQUENCES
    return results


def evaluate_tile(tile, tile_embed, rng):
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    results = {}
    for n_bits in EVAL_N_VALUES:
        correct = 0
        for _ in range(EVAL_SEQUENCES):
            tokens, pairs = generate_parity_sequence(rng, n_bits)
            query_pos, answer = pairs[0]
            M = np.zeros((W, state_width), dtype=np.float32)
            pred = None
            for i in range(query_pos + 1):
                window = _build_window(tile_embed, tokens, i, W, M, COLUMN_NEURONS)
                M, logits_t, _aux = tile.step(window, M, 0.0)
                if i == query_pos:
                    pred = predicted_token(logits_t, W - 1)
            correct += int(pred == answer)
        results[n_bits] = correct / EVAL_SEQUENCES
    return results


def main():
    print(f"W={W} train_steps={TRAIN_STEPS} warmup={WARMUP_STEPS} peak_lr={PEAK_LR} "
          f"steps_per_level={STEPS_PER_LEVEL} curriculum_window={CURRICULUM_WINDOW} "
          f"out_of_context_max={OUT_OF_CONTEXT_MAX} eval_sequences={EVAL_SEQUENCES}\n")

    t0 = time.time()
    dense, dense_embed, dense_rng = train_dense()
    t1 = time.time()
    dense_results = evaluate_dense(dense, dense_embed, dense_rng)
    print(f"dense (n_layers=1, half_bandwidth=W) trained ({t1 - t0:.1f}s)")

    t0 = time.time()
    tile_no_energy, tile_ne_embed, tile_ne_rng = _train_tile(DISLDOLayer, use_energy=False, seed=2)
    t1 = time.time()
    tile_no_energy_results = evaluate_tile(tile_no_energy, tile_ne_embed, tile_ne_rng)
    print(f"tile, plain (no energy) trained ({t1 - t0:.1f}s)")

    t0 = time.time()
    tile_energy, tile_e_embed, tile_e_rng = _train_tile(DISLDOLayer, use_energy=True, seed=3)
    t1 = time.time()
    tile_energy_results = evaluate_tile(tile_energy, tile_e_embed, tile_e_rng)
    print(f"tile, +energy trained ({t1 - t0:.1f}s)")

    t0 = time.time()
    tile_peak, tile_peak_embed, tile_peak_rng = _train_tile(
        PeakEligibilityDISLDOLayer, use_energy=False, seed=4)
    t1 = time.time()
    tile_peak_results = evaluate_tile(tile_peak, tile_peak_embed, tile_peak_rng)
    print(f"tile, peak-eligibility trained ({t1 - t0:.1f}s)\n")

    print(f"{'n_bits':>8}  {'in_ctx':>7}  {'dense':>7}  {'tile':>7}  "
          f"{'tile+egy':>9}  {'peak-elig':>10}")
    for n_bits in EVAL_N_VALUES:
        in_ctx = "yes" if n_bits <= W else "NO"
        print(f"{n_bits:>8}  {in_ctx:>7}  {dense_results[n_bits]:>7.2f}  "
              f"{tile_no_energy_results[n_bits]:>7.2f}  {tile_energy_results[n_bits]:>9.2f}  "
              f"{tile_peak_results[n_bits]:>10.2f}")
    print(f"\n(chance = 0.5 for a single binary answer bit)")
    return (dense_results, tile_no_energy_results, tile_energy_results, tile_peak_results)


if __name__ == "__main__":
    main()
