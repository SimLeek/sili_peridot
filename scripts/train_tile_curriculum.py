"""Fast (~seconds-to-minutes) curriculum test harness for
ToyTileRecurrenceRealFP4 -- built to answer "does ANY version of this
architecture actually learn anything" before scaling up again, per
direct instruction after the overnight run showed no real learning for
any arm (confirmed via scripts/learning_slope.py: PLATEAUED at a
near-chance biased collapse, not LEARNING).

Curriculum: seq_len starts at SEQ_LEN_START and grows by 1 every
STEPS_PER_STAGE steps up to SEQ_LEN_MAX (<=NUM_TILES, so this stays a
pure in-context test -- the tile window is always wide enough to hold
the whole sequence; out-of-context (seq_len > num_tiles) is a later
phase once in-context is solid, per direct instruction).

use_attention=False bypasses gaussian_attention entirely (see
ToyTileRecurrenceRealFP4's own docstring) -- an ablation to isolate
whether attention itself is the hard-to-learn part, tested BEFORE
assuming the whole architecture is broken.

Usage: python3 train_tile_curriculum.py <arm> <use_energy 0|1> <use_attention 0|1> <total_steps> [checkpoint_every] [seed]
  arm: rank1 | rank2 | fp8
"""
from __future__ import annotations

import sys
import time
import functools

import numpy as np

sys.path.insert(0, ".")

from sili.sparse_rnn import (DISLDOLayer, DISLDOLayer8, DISLDOLayer32,
                             DISLDOLayerResync, DISLDOLayerNoScale,
                             DISLDOLayerDeterministic, DISLDOLayerResyncDeterministic,
                             DISLDOLayerNoScaleDeterministic)
from sili import _cpu
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from model.toy_precision_models import (QuantizedDISLDOLayer32, SeededRank1DISLDOLayer8,
                                        PeriodicSeedRank1DISLDOLayer8,
                                        SeededDISLDOLayer8Resync, SeededDISLDOLayer8AdaMax,
                                        TrueMultiDigitLayer, TrueMultiDigitDenseLayer)
from model.toy_recall_models import cross_entropy_sum, predicted_token, AdamOptimizer, lr_schedule


def generate_copy_sequence(rng: np.random.RandomState, vocab: int, seq_len: int):
    """Simplest possible state-carrying task: token[0] is the "key",
    everything else is random filler; the ONLY thing to predict is
    token[0] again, queried at the FINAL tick. Works for any seq_len>=2
    (generate_mqar_sequence requires seq_len>=4*num_kv_pairs, which
    can't express seq_len=2/3 -- this is what "start at seq_len=2" per
    direct instruction actually needs). One (position, target) pair,
    always at the last position, matching this architecture's own
    "only the last tile produces logits" convention."""
    tokens = rng.randint(0, vocab, size=seq_len)
    pairs = [(seq_len - 1, int(tokens[0]))]
    return tokens, pairs

EMBED_WIDTH = 8
COLUMN_NEURONS = 4
MLP_HIDDEN_MULT = 2            # unused (MLP removed), kept for API compat
NUM_TILES = 4                  # ~= num_cores per direct suggestion; ALSO the
                               # curriculum's in-context ceiling (seq_len<=this)
VOCAB = 10                     # small vocab too -- chance=0.1, not 0.025, so a
                               # trivial task doesn't need thousands of steps
                               # just to beat noise
MAX_WEIGHTS_PER_LAYER = 128    # per_row=32 at state_width=32 -- generous at
                               # this tiny scale, not the bottleneck being tested
NUM_CPUS = 1
PEAK_LR = 0.002
WARMUP_STEPS = 50
EVAL_SEQUENCES = 60

SEQ_LEN_START = 2
SEQ_LEN_MAX = NUM_TILES
STEPS_PER_STAGE_DEFAULT = 500

ENERGY_KWARGS = dict(drive=0.00535, activation_cost=0.005, precision=0.001,
                     density=0.005, p=0.995, reactivity=0.0001)

ARMS = {
    "rank1": DISLDOLayer,
    "rank2": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rankn", rank=2,
                               quantize_importance=True),
    "fp8":   DISLDOLayer8,
    "fp32":  DISLDOLayer32,  # precision ceiling reference -- isolates FP4 quantization
                            # coarseness from architecture/training-dynamics limits
    "rank1_8bit": functools.partial(QuantizedDISLDOLayer32, bits=8, scheme="rank1",
                                    quantize_importance=True),  # the exact scheme that
                            # reached 1.0 at every out-of-context distance on the
                            # earlier tanh-cell task (JOURNAL.md) -- direct retest here
    # Alternative 4-bit scale representations -- same bit budget as "rank2"
    # (bits=4, scheme=rankn, rank=2 above), different envelope shape, to see
    # which specific scale scheme (not bit-depth) recovers out-of-context
    # accuracy at 4-bit.
    "row_4bit":   functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="row",
                                    quantize_importance=True),   # plain per-row max-abs
    "rank1_4bit": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rank1",
                                    quantize_importance=True),   # row x col envelope, rank1
    "rank4_4bit": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rankn", rank=4,
                                    quantize_importance=True),   # higher-rank envelope
    "multi_fp4": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="residual",
                                   n_stages=2, quantize_importance=True),  # true residual/
                            # cascaded quantization: 2 stages of real 4-bit each, summed
                            # -- 8 bits/weight total, a fair fight against rank1_8bit's
                            # single 8-bit code, not the earlier ruled-out envelope-of
                            # -envelope idea
    "fp8_seeded": SeededRank1DISLDOLayer8,  # real DISLDOLayer8 (true C++ E4M3 + rank-1
                            # value_scale/output_scale), scale seeded from a real
                            # closed-form fit at init instead of left at 1.0 -- tests
                            # whether real fp8's out-of-context collapse is a
                            # cold-start/undertrained-scale problem, not the
                            # representation itself
    "fp8_reseeded": functools.partial(PeriodicSeedRank1DISLDOLayer8, reseed_every=250),
                            # real DISLDOLayer8, scale re-seeded from a fresh closed
                            # -form fit every 250 training backward() calls -- tests
                            # whether REPEATED correction (no change to the real
                            # weight-update math) substitutes for the simulation's
                            # every-step refit
    "fp8_resync": SeededDISLDOLayer8Resync,  # REAL C++ fix (not a Python approximation):
                            # sili__new's disldo_backward now defers each touched
                            # entry's store until value_scale/output_scale are BOTH
                            # finalized for the call, instead of storing under the
                            # stale pre-update scale. Seeded like fp8_seeded for a
                            # fair comparison (isolates the deferred-write fix, not
                            # "was output_scale active at all").
    "fp8_adamax": SeededDISLDOLayer8AdaMax,  # same deferred-write fix, but
                            # value_scale/output_scale use an AdaMax-style decayed
                            # running-max update instead of RMSprop -- see
                            # AdaMaxScalePolicy's docstring in sili__new.
    "fixed_digit_2": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="fixed_digit_residual",
                                       n_stages=2, base=4.0, quantize_importance=True),
                            # ZERO trained/fitted scale anywhere (no row/col vector, no
                            # per-call data-dependent recompute) -- 2 fixed FP4 "digit"
                            # stages, base=4.0 derived directly from real FP4 (E2M1)'s own
                            # 1-mantissa-bit relative precision, e_shared derived ONCE from
                            # init weights and frozen for the whole run. 8 bits/weight
                            # total, same budget as rank1_8bit/multi_fp4 -- direct test of
                            # whether the whole trained-scale mechanism (and its staleness
                            # bug) can be skipped entirely, per direct design discussion.
    "fixed_digit_3": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="fixed_digit_residual",
                                       n_stages=3, base=4.0, quantize_importance=True),
                            # same, 3 stages / 12 bits -- checks whether more digits closes
                            # any remaining gap to rank1_8bit/multi_fp4's near-1.0 result.
    "fixed_digit_4": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="fixed_digit_residual",
                                       n_stages=4, base=4.0, quantize_importance=True),
                            # 4 stages / 16 bits -- finer quantization removes more of the
                            # implicit noise-filtering coarse rounding was providing (found
                            # directly: fixed_digit_3 needed half the PEAK_LR to stabilize),
                            # so this arm likely also needs a reduced --peak_lr to be fair.
    "true_multi_digit_lr0": functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                                              n_stages=3, base=4.0, lr_power=0.0),
                            # Genuinely SEPARATE per-digit training, REAL FP4 (DISLDOLayer)
                            # per digit -- no hidden fp32 shadow, no extra Python-side
                            # quantization step (real disldo_backward already stores FP4
                            # natively). lr_power=0: chain-rule-only scaling (the ordinary
                            # `out_i*factor_i` gradient reduction, nothing extra on top) --
                            # this is the natural baseline, NOT an unscaled naive one, see
                            # TrueMultiDigitLayer's own corrected docstring.
    "true_multi_digit_lr1": functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                                              n_stages=3, base=4.0, lr_power=1.0),
                            # lr_power=1: digit i's rate ALSO divided by base**i on top of
                            # the chain rule's own reduction -- extra damping beyond natural.
    "true_multi_digit_lr2": functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                                              n_stages=3, base=4.0, lr_power=2.0),
                            # lr_power=2: divided by base**(2i) on top -- more extra damping.
    "true_multi_digit_fp32_ref": functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer32,
                                                   n_stages=3, base=4.0, lr_power=0.0,
                                                   simulate_quantize=True, bits_per_stage=4),
                            # REFERENCE/ceiling control (per direct request: disldo vs a
                            # non-quantized-storage comparison catches whether real FP4
                            # storage itself, not the residual-digit architecture, explains
                            # any gap) -- same digit-residual structure and training
                            # dynamics, fp32-EXACT storage with a fake-quantize simulation
                            # step matching fixed_digit_residual_quantize's own grid.
    "true_multi_digit_dense": functools.partial(TrueMultiDigitDenseLayer,
                                                n_stages=3, base=4.0, lr_power=0.0),
                            # DISLDO vs ordinary-Adam-trained control (per direct request):
                            # same digit-residual architecture, dense fp32 weights, trained
                            # via a standard external AdamOptimizer instead of DISLDO's own
                            # inline importance-based update -- catches whether something
                            # about DISLDO's OWN mechanism (not the architecture) explains
                            # any gap vs a well-understood, trusted baseline optimizer.

    # Root-cause follow-up to true_multi_digit_lr0's chance-level collapse:
    # plain DISLDOLayer (FP4) was found to still call disldo_backward with
    # the DEFAULT ScalePolicy/DeferredScaleWrite=false args -- the exact
    # stale-value_scale bug fp8_resync fixed for FP8, never applied to FP4.
    # These two arms are the single-digit (n_stages=1, no residual
    # composition at all) direct test: is plain real FP4's out-of-context
    # collapse (row_4bit, etc.) explained by this same bug?
    "row_4bit_resync": DISLDOLayerResync,   # DeferredScaleWrite fix only
    "row_4bit_noscale": DISLDOLayerNoScale, # value_scale/output_scale forced
                            # to 1.0 forever -- direct real-hardware test of
                            # "zero trained scale" (per direct request:
                            # "Can we just add an option to remove the
                            # scaling too?"), not just fixing staleness.

    "true_multi_digit_resync": functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayerResync,
                                                 n_stages=3, base=4.0, lr_power=0.0),
                            # Same architecture as true_multi_digit_lr0, but each digit is a
                            # real FP4 DISLDOLayerResync instead of plain DISLDOLayer.
    "true_multi_digit_noscale": functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayerNoScale,
                                                  n_stages=3, base=4.0, lr_power=0.0),
                            # Same architecture, each digit's OWN internal value_scale/
                            # output_scale forced off -- our external per-digit factor_i
                            # composition is then the ONLY scale in play, matching the
                            # zero-trained-scale design intent exactly.

    # row_4bit_resync/noscale both STILL collapsed to chance -- value_scale
    # staleness/presence isn't the explanation after all. Remaining candidate:
    # real FP4 uses STOCHASTIC rounding on every store (fp4_quantize_stochastic,
    # real per-step noise); the things that DID succeed (fixed_digit_2,
    # true_multi_digit_fp32_ref's simulate_quantize) both use DETERMINISTIC
    # round-to-nearest. Direct single-variable isolation, single digit first:
    "row_4bit_deterministic": DISLDOLayerDeterministic,          # same RMSprop
                            # scale as plain DISLDOLayer, rounding only changed.
    "row_4bit_resync_deterministic": DISLDOLayerResyncDeterministic,
    "row_4bit_noscale_deterministic": DISLDOLayerNoScaleDeterministic,
                            # closest real-hardware match to fixed_digit_residual_
                            # quantize's own design: zero trained scale AND
                            # deterministic rounding together.

    # base=12.0: exact digit-range tiling (E2M1 math, see
    # fixed_digit_residual_quantize's docstring), the project's working
    # default. IMPORTANT: the single-seed sweep that first picked this
    # (JOURNAL.md 2026-08-10, mean_acc 0.9375 vs base=4's 0.8771) was run
    # BEFORE a real bug was found and fixed -- ToyTileRecurrenceRealFP4
    # never passed `rng=` down to disldo_cls, so every layer's initial
    # connectivity/weight values were genuinely unseeded regardless of
    # `seed` (confirmed directly: same seed, same command, gave
    # final-step accuracies of 0.70 then 0.65 across two back-to-back
    # runs). That single-seed comparison only shows "there exists a draw
    # where base=12 wins," not "usually wins" -- needs a proper
    # multi-seed re-run now that construction is actually reproducible
    # (see tests/test_residual_base_sweep.py) before this default is
    # fully trusted. Kept as the default in the meantime since the
    # theoretical argument (exact tiling) is independent of that bug.
    "true_multi_digit_deterministic": functools.partial(TrueMultiDigitLayer,
                                                        digit_cls=DISLDOLayerDeterministic,
                                                        n_stages=3, base=12.0, lr_power=0.0),
    # base=4.0 was the ORIGINAL default, derived from real FP4 (E2M1)'s
    # own worst-case relative rounding error (~1/4) -- kept as an explicit
    # comparison point now that base=12 is the default above. Same
    # digit_cls/n_stages/lr_power as true_multi_digit_deterministic --
    # base is the only varied axis.
    "true_multi_digit_deterministic_base4": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
        n_stages=3, base=4.0, lr_power=0.0),
    # base=6.0: halfway between base=4 (overlapping digit ranges) and
    # base=12 (exact tiling) -- added per direct request to fill in the
    # sparse 4/12/24 sweep with a point between the two closest-together
    # candidates.
    "true_multi_digit_deterministic_base6": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
        n_stages=3, base=6.0, lr_power=0.0),
    "true_multi_digit_deterministic_base24": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
        n_stages=3, base=24.0, lr_power=0.0),

    # lr_power retest under deterministic rounding (JOURNAL.md 2026-08-10
    # "Test 2"): the stochastic-rounding-era true_multi_digit_lr0/lr1/lr2
    # sweep found no real difference between lr_power values, predicted
    # to be because RMSprop's own eff_lr*g/sqrt(importance) self
    # -normalizes almost all of the extra per-digit factor_i damping away
    # on its own. Retesting under deterministic rounding (base=12, matching
    # the confirmed default) to confirm that prediction still holds now
    # that stochastic noise isn't swamping everything else.
    "true_multi_digit_deterministic_lr1": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
        n_stages=3, base=12.0, lr_power=1.0),
    "true_multi_digit_deterministic_lr2": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
        n_stages=3, base=12.0, lr_power=2.0),

    # _dense variants: fully dense connectivity (every synapse present,
    # loaded straight into block4 via sili__new's load_dense_codes) instead
    # of the random SPARSE "echo network" preseed every arm above uses --
    # per direct request, to test whether that random-connectivity-draw is
    # itself a significant source of the seed-to-seed variance seen even at
    # base=12 (std 0.043 across 5 seeds, JOURNAL.md 2026-08-10), independent
    # of base or bits. Same digit_cls/n_stages/lr_power/base as their
    # non-dense counterparts -- dense=True is the only varied axis, so this
    # is a direct paired comparison, not a new axis tangled with others.
    "true_multi_digit_deterministic_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
        n_stages=3, base=12.0, lr_power=0.0, dense=True),
    "true_multi_digit_deterministic_base4_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
        n_stages=3, base=4.0, lr_power=0.0, dense=True),
    "true_multi_digit_deterministic_base6_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
        n_stages=3, base=6.0, lr_power=0.0, dense=True),
    "true_multi_digit_deterministic_base24_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
        n_stages=3, base=24.0, lr_power=0.0, dense=True),

    "true_multi_digit_noscale_deterministic": functools.partial(TrueMultiDigitLayer,
                                                                digit_cls=DISLDOLayerNoScaleDeterministic,
                                                                n_stages=3, base=12.0, lr_power=0.0),

    # Direct hypothesis check: each digit's independent preseed/synaptogenesis
    # means digits' connectivity is essentially disjoint (verified: 0/20, 0/20,
    # 1/20 overlap on a fresh preseed) -- the residual-correction mechanism can
    # only fire where digits' connectivity actually coincides. share_connectivity
    # forces every digit onto digit 0's exact (ptrs, indices) at construction.
    "true_multi_digit_shared_conn": functools.partial(TrueMultiDigitLayer,
                                                       digit_cls=DISLDOLayerDeterministic,
                                                       n_stages=3, base=4.0, lr_power=0.0,
                                                       share_connectivity=True),
}


def _maybe_synaptogenesis(model, k: int = 4, importance_cutoff: float = 0.01):
    """Real structural growth+pruning (sili's actual synap_step/
    build_probes/equalizer_step, via _SparseLayerBase.synaptogenesis or
    TrueMultiDigitLayer's own per-digit delegate) on every disldo-family
    sublayer of the model, k=4 -- the established default elsewhere in
    the codebase (SparseRNNAgent's own synaptogenesis_k default; k=64
    was measured to saturate a 1000x1000 layer's connectivity in ONE
    call, k=4 grows gradually instead). Each sublayer's own already
    -stored `_max_row_weights` cap keeps nnz roughly STABLE over time
    (grow-and-prune balance against a fixed target, not unbounded
    growth) -- per direct instruction, not a new/growing budget.
    Dense/Adam-controlled sublayers (no `.synaptogenesis`, no sparse
    structure) are silently skipped."""
    for attr in ("q_proj", "k_proj", "v_proj", "lm_head"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        if hasattr(sub, "_max_row_weights"):
            sub.synaptogenesis(k, importance_cutoff, sub._max_row_weights)
        elif hasattr(sub, "synaptogenesis"):
            sub.synaptogenesis(k, importance_cutoff)
    o_proj = getattr(model, "o_proj", None)
    o_layers = o_proj if isinstance(o_proj, list) else ([o_proj] if o_proj is not None else [])
    for sub in o_layers:
        if hasattr(sub, "_max_row_weights"):
            sub.synaptogenesis(k, importance_cutoff, sub._max_row_weights)
        elif hasattr(sub, "synaptogenesis"):
            sub.synaptogenesis(k, importance_cutoff)


def _build_tile_window(embed_table: np.ndarray, tokens: np.ndarray, i: int,
                       num_tiles: int, column_neurons: int) -> np.ndarray:
    state_width = embed_table.shape[1] * column_neurons
    window = np.zeros((num_tiles, state_width), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        if src >= 0:
            window[j] = np.repeat(embed_table[tokens[src]], column_neurons)
    return window


def current_seq_len(step: int, steps_per_stage: int) -> int:
    stage = step // steps_per_stage
    return min(SEQ_LEN_START + stage, SEQ_LEN_MAX)


def evaluate(model, rng, embed_table: np.ndarray, seq_len: int) -> float:
    state_width = embed_table.shape[1] * COLUMN_NEURONS
    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, pairs = generate_copy_sequence(rng, VOCAB, seq_len)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, _aux = model.step(window, M, 0.0)
            if i in targets:
                pred = predicted_token(logits, NUM_TILES - 1)
                correct += int(pred == targets[i])
                total += 1
    return correct / total if total else 0.0


# value-bits/weight for each arm's stored weight representation -- used only
# for reporting an approximate memory footprint (index/overhead bits are the
# same across arms so they wash out of a *relative* comparison; this is an
# approximation, not a byte-exact accounting).
ARM_VALUE_BITS = {"rank1": 4, "rank2": 4, "fp8": 8, "fp32": 32, "rank1_8bit": 8,
                  "row_4bit": 4, "rank1_4bit": 4, "rank4_4bit": 4, "multi_fp4": 8,
                  "fp8_seeded": 8, "fp8_reseeded": 8, "fp8_resync": 8, "fp8_adamax": 8,
                  "fixed_digit_2": 8, "fixed_digit_3": 12, "fixed_digit_4": 16,
                  "true_multi_digit_lr0": 12, "true_multi_digit_lr1": 12, "true_multi_digit_lr2": 12,
                  "true_multi_digit_fp32_ref": 32, "true_multi_digit_dense": 32,
                  "row_4bit_resync": 4, "row_4bit_noscale": 4,
                  "true_multi_digit_resync": 12, "true_multi_digit_noscale": 12,
                  "row_4bit_deterministic": 4, "row_4bit_resync_deterministic": 4,
                  "row_4bit_noscale_deterministic": 4,
                  "true_multi_digit_deterministic": 12, "true_multi_digit_noscale_deterministic": 12,
                  "true_multi_digit_deterministic_base4": 12, "true_multi_digit_deterministic_base6": 12,
                  "true_multi_digit_deterministic_base24": 12,
                  "true_multi_digit_deterministic_lr1": 12, "true_multi_digit_deterministic_lr2": 12,
                  "true_multi_digit_deterministic_dense": 12, "true_multi_digit_deterministic_base4_dense": 12,
                  "true_multi_digit_deterministic_base6_dense": 12, "true_multi_digit_deterministic_base24_dense": 12,
                  "true_multi_digit_shared_conn": 12}


def estimate_value_bits(arm: str, state_width: int, embed_width: int, vocab: int,
                        max_weights: int, use_attention: bool) -> int:
    bits = ARM_VALUE_BITS[arm]
    o_proj = min(max_weights, state_width * state_width)
    lm_head = min(max_weights, embed_width * vocab)
    total_entries = o_proj + lm_head
    if use_attention:
        total_entries += 3 * min(max_weights, state_width * state_width)  # q,k,v
    return total_entries * bits


def main():
    global EMBED_WIDTH, COLUMN_NEURONS, MAX_WEIGHTS_PER_LAYER, SEQ_LEN_MAX, PEAK_LR

    arm = sys.argv[1]
    use_energy = bool(int(sys.argv[2]))
    use_attention = bool(int(sys.argv[3]))
    train_steps = int(sys.argv[4])
    checkpoint_every = int(sys.argv[5]) if len(sys.argv) > 5 else max(train_steps // 20, 50)
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
    steps_per_stage = int(sys.argv[7]) if len(sys.argv) > 7 else STEPS_PER_STAGE_DEFAULT
    # optional overrides -- used to build memory-footprint-matched comparisons
    # (e.g. a wider FP4 net whose *value* bits roughly match a narrower fp32
    # net's, per direct request) instead of always comparing arms at equal width.
    if len(sys.argv) > 8:
        EMBED_WIDTH = int(sys.argv[8])
    if len(sys.argv) > 9:
        COLUMN_NEURONS = int(sys.argv[9])
    if len(sys.argv) > 10:
        MAX_WEIGHTS_PER_LAYER = int(sys.argv[10])
    # seq_len_max > NUM_TILES is a real out-of-context test: once i exceeds
    # NUM_TILES-1, _build_tile_window's window no longer reaches back to
    # position 0, so recalling token[0] requires the info to have survived
    # in M_prev across ticks the window itself can no longer see.
    SEQ_LEN_MAX = int(sys.argv[11]) if len(sys.argv) > 11 else NUM_TILES
    # DISLDOLayer.forward's default lr_per_row_nnz=True divides the effective
    # rate by each row's connection count (nnz_this_row) -- real and
    # necessary when synaptogenesis makes degree vary, a silent crush at
    # fixed density. fp32 tolerates the crushed rate fine (continuous
    # updates); FP4 needs a large-enough step to move a value even one
    # quantization level, so this override compensates directly instead of
    # touching lr_per_row_nnz itself (which is buried inside disldo_cls's
    # own forward() call, not exposed through ToyTileRecurrenceRealFP4).
    if len(sys.argv) > 12:
        PEAK_LR = float(sys.argv[12])
    # o_proj_depth>1: N sequential FP4 sublayers instead of one wider layer --
    # a cascaded/residual-quantization-style test of whether composing coarse
    # stages recovers precision that widening alone doesn't, per direct idea.
    o_proj_depth = int(sys.argv[13]) if len(sys.argv) > 13 else 1
    # use_synaptogenesis: real dynamic growth+pruning (build_probes+
    # synap_step+equalizer_step via _maybe_synaptogenesis, k=4) every
    # outer step, instead of the static pre-seeded-only sparsity every
    # arm has used so far this session -- per direct request, testing
    # whether the residual digits want some OTHER connectivity pattern
    # discovered via real importance-driven growth/pruning, distinct
    # from both fully-independent-random and forced-identical (both
    # already tested). Default 0/off, backward-compatible with every
    # existing invocation.
    use_synaptogenesis = bool(int(sys.argv[14])) if len(sys.argv) > 14 else False

    state_width = EMBED_WIDTH * COLUMN_NEURONS
    mlp_hidden = state_width * MLP_HIDDEN_MULT

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    # Real DISLDOLayer-family (fp8/fp8_seeded/fp8_resync/fp8_adamax/rank1/
    # fp32) arms use stochastic rounding (fp4quant.hpp/fp8quant.hpp's
    # set_stochastic) whose RNG is thread-local and, by design, seeded
    # from the thread id at process start -- NOT controlled by `seed`
    # above, and NOT reproducible run-to-run without this call. Confirmed
    # directly: the SAME unchanged binary gave different single-step
    # results across separate process invocations (0.140625 vs 0.15625
    # for one stored weight) purely from this. Without pinning it here,
    # comparisons between arms (or before/after a C++ change) are
    # confounded by an extra, uncontrolled noise source on top of `seed`.
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    # Separate Generator (not the legacy RandomState `rng` above, used for
    # tokens/embed_table) for model construction -- ToyTileRecurrenceRealFP4
    # threads this down to each disldo_cls layer's initial connectivity/weight
    # values (see its own docstring for the bug this fixes: this was
    # previously never passed at all, so every layer's preseed was genuinely
    # unseeded regardless of `seed`).
    model_rng = np.random.default_rng(seed)
    model = ToyTileRecurrenceRealFP4(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, mlp_hidden, NUM_TILES, MAX_WEIGHTS_PER_LAYER,
        num_cpus=NUM_CPUS, disldo_cls=ARMS[arm],
        use_energy=use_energy, energy_kwargs=ENERGY_KWARGS if use_energy else None,
        use_attention=use_attention, o_proj_depth=o_proj_depth, rng=model_rng)
    opt = AdamOptimizer()
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

    value_bits = estimate_value_bits(arm, state_width, EMBED_WIDTH, VOCAB,
                                     MAX_WEIGHTS_PER_LAYER, use_attention)
    print(f"# arm={arm} use_energy={use_energy} use_attention={use_attention} "
          f"train_steps={train_steps} checkpoint_every={checkpoint_every} seed={seed} "
          f"vocab={VOCAB} num_tiles={NUM_TILES} embed_width={EMBED_WIDTH} "
          f"column_neurons={COLUMN_NEURONS} state_width={state_width} "
          f"max_weights={MAX_WEIGHTS_PER_LAYER} o_proj_depth={o_proj_depth} "
          f"peak_lr={PEAK_LR} est_value_bits={value_bits} "
          f"(~{value_bits/8:.0f} bytes) use_synaptogenesis={use_synaptogenesis} "
          f"seq_len={SEQ_LEN_START}->{SEQ_LEN_MAX} (+1/{steps_per_stage} steps)",
          flush=True)

    t0 = time.time()
    for step in range(1, train_steps + 1):
        seq_len = current_seq_len(step, steps_per_stage)
        lr = lr_schedule(step, train_steps, PEAK_LR, WARMUP_STEPS)
        tokens, pairs = generate_copy_sequence(rng, VOCAB, seq_len)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, lr)
            if i in targets:
                loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                if aux is not None:
                    loss = loss + aux
                loss.backward()
                opt.step(model.parameters_for_optimizer(), lr=lr)

        if use_synaptogenesis:
            _maybe_synaptogenesis(model)

        if step % checkpoint_every == 0:
            acc = evaluate(model, rng, embed_table, seq_len)
            elapsed = time.time() - t0
            print(f"step={step:>7}  seq_len={seq_len}  acc={acc:.4f}  "
                  f"({elapsed:.0f}s elapsed, {elapsed/step:.4f}s/step)", flush=True)

    print(f"# DONE arm={arm} use_energy={use_energy} use_attention={use_attention} "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
