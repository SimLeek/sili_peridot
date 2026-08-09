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

from sili.sparse_rnn import DISLDOLayer, DISLDOLayer8, DISLDOLayer32
from sili import _cpu
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from model.toy_precision_models import (QuantizedDISLDOLayer32, SeededRank1DISLDOLayer8,
                                        PeriodicSeedRank1DISLDOLayer8,
                                        SeededDISLDOLayer8Resync, SeededDISLDOLayer8AdaMax)
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
}


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
                  "fixed_digit_2": 8, "fixed_digit_3": 12}


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
    model = ToyTileRecurrenceRealFP4(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, mlp_hidden, NUM_TILES, MAX_WEIGHTS_PER_LAYER,
        num_cpus=NUM_CPUS, disldo_cls=ARMS[arm],
        use_energy=use_energy, energy_kwargs=ENERGY_KWARGS if use_energy else None,
        use_attention=use_attention, o_proj_depth=o_proj_depth)
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
          f"(~{value_bits/8:.0f} bytes) "
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

        if step % checkpoint_every == 0:
            acc = evaluate(model, rng, embed_table, seq_len)
            elapsed = time.time() - t0
            print(f"step={step:>7}  seq_len={seq_len}  acc={acc:.4f}  "
                  f"({elapsed:.0f}s elapsed, {elapsed/step:.4f}s/step)", flush=True)

    print(f"# DONE arm={arm} use_energy={use_energy} use_attention={use_attention} "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
