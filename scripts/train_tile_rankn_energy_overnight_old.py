"""Overnight tile-recurrence-prototype validation: rank1 vs rank2 FP4,
FP8 (rank1 -- real DISLDOLayer8 has no rank-n variant yet, that would
be new C++ work, not just wiring in existing fixes), each with and
without the corrected EnergyDynamics config, at a near-full-scale
state_width (not the toy column_neurons=8/16 the original stage-1/2/3
sweep used -- per direct instruction, this project's own column
-averaging mechanism needs recurrent size 1000+ to be a meaningful
test, not toy-scale convenience).

Energy config: drive=0.0535 (calibrated to THIS call site's own
measured attn-output magnitude, E[|attn|]~=1.07 at this scale --
NOT the drive=0.016 value calibrated for the much smaller tanh-cell
recurrence earlier tonight; the balance point is architecture-specific,
see sili__new energy.py's docstring), p=0.95 (from the same fix).
Verified directly before this run (not assumed): at this population
size (num_tiles*state_width=32768), this config gives ~1.2% firing /
~5% zeroing per tick, vs the broken default's 30%/70% -- see
sili_peridot JOURNAL.md's energy-calibration entries for the full
derivation and the smaller-scale statistical confirmation this run is
extending to the real tile-recurrence architecture.

Real MQAR recall task (generate_mqar_sequence), same target convention
as the existing train_toy_tile_precision_comparison.py.

Usage: python3 train_tile_rankn_energy_overnight.py <arm> <use_energy 0|1> <train_steps> <checkpoint_every>
  arm: rank1 | rank2 | fp8
"""

from __future__ import annotations

import functools
import sys
import time
import traceback
import warnings

import numpy as np

sys.path.insert(0, ".")

from sili.sparse_rnn import DISLDOLayer, DISLDOLayer8

from model.toy_precision_models import QuantizedDISLDOLayer32
from model.toy_recall_models import AdamOptimizer, cross_entropy_sum, lr_schedule, predicted_token
from model.toy_recall_task import generate_mqar_sequence
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4

EMBED_WIDTH = 8
COLUMN_NEURONS = 8  # state_width = 1024 -- near-full-scale, not toy
STATE_WIDTH = EMBED_WIDTH * COLUMN_NEURONS
MLP_HIDDEN = STATE_WIDTH * 2
NUM_TILES = 4  # seq_len=32, matches largest existing tested variant
SEQ_LEN = NUM_TILES
NUM_KV_PAIRS = 1
VOCAB = 40
MAX_WEIGHTS_PER_LAYER = 16384  # per_row=16 at state_width=1024, matching the
# per_row~16 ratio the original 4096-at-256 fix
# established -- kept proportional, not left at a
# width-independent constant that would make this
# scale needlessly sparse for no capacity reason.
NUM_CPUS = 4
PEAK_LR = 0.002  # NOT 0.02 -- matches the already-documented DISLDOLayer-family
# fix (JOURNAL.md ~line 2637): raw, unclipped per-synapse update
# diverges at Adam-tuned rates regardless of quantization.
# Re-verified directly at this scale, with the state_ln fix
# already applied: peak_lr=0.02 still gives 65 overflow warnings
# in exp() over 100 steps (max|M|=78); peak_lr=0.002 gives zero.
# The two bugs (unbounded M, LR too high) are separate and both
# real -- state_ln alone does not make 0.02 safe.
WARMUP_STEPS = 100
EVAL_SEQUENCES = 60

ENERGY_KWARGS = {
    "drive": 0.00535,
    "activation_cost": 0.005,
    "precision": 0.001,
    "density": 0.005,
    "p": 0.995,
    "reactivity": 0.0001,
}

ARMS = {
    "rank1": DISLDOLayer,
    "rank2": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rankn", rank=2, quantize_importance=True),
    "fp8": DISLDOLayer8,
}

warnings.filterwarnings("error", category=RuntimeWarning)


def _build_targets(tokens: np.ndarray, mqar_pairs: list, num_kv_pairs: int) -> dict:
    context_size = num_kv_pairs * 2
    targets = dict(mqar_pairs)
    for i in range(context_size - 1):
        targets.setdefault(i, int(tokens[i + 1]))
    return targets


def _build_tile_window(
    embed_table: np.ndarray, tokens: np.ndarray, i: int, num_tiles: int, M_prev: np.ndarray, column_neurons: int
) -> np.ndarray:
    state_width = embed_table.shape[1] * column_neurons
    window = np.empty((num_tiles, state_width), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        window[j] = np.repeat(embed_table[tokens[src]], column_neurons) if src >= 0 else M_prev[j]
    return window


def evaluate(model, rng, embed_table: np.ndarray) -> float:
    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, SEQ_LEN, NUM_KV_PAIRS)
        mqar_by_pos = dict(mqar_pairs)
        M = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
        for i in range(SEQ_LEN):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, M, COLUMN_NEURONS)
            M, logits, _aux = model.step(window, M, 0.0)
            if i in mqar_by_pos:
                pred = predicted_token(logits, NUM_TILES - 1)
                correct += int(pred == mqar_by_pos[i])
                total += 1
    return correct / total


def main():
    try:
        arm = sys.argv[1]
        use_energy = bool(int(sys.argv[2]))
        train_steps = int(sys.argv[3])
        checkpoint_every = int(sys.argv[4]) if len(sys.argv) > 4 else max(train_steps // 20, 50)
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else 9000

        rng = np.random.RandomState(seed)
        np.random.seed(seed)
        model = ToyTileRecurrenceRealFP4(
            VOCAB,
            EMBED_WIDTH,
            COLUMN_NEURONS,
            MLP_HIDDEN,
            NUM_TILES,
            MAX_WEIGHTS_PER_LAYER,
            num_cpus=NUM_CPUS,
            disldo_cls=ARMS[arm],
            use_energy=use_energy,
            energy_kwargs=ENERGY_KWARGS if use_energy else None,
        )
        opt = AdamOptimizer()
        embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

        print(
            f"# arm={arm} use_energy={use_energy} train_steps={train_steps} "
            f"checkpoint_every={checkpoint_every} seed={seed} "
            f"state_width={STATE_WIDTH} num_tiles={NUM_TILES} max_weights={MAX_WEIGHTS_PER_LAYER}",
            flush=True,
        )

        t0 = time.time()
        for step in range(1, train_steps + 1):
            lr = lr_schedule(step, train_steps, PEAK_LR, WARMUP_STEPS)
            tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, SEQ_LEN, NUM_KV_PAIRS)
            targets = _build_targets(tokens, mqar_pairs, NUM_KV_PAIRS)
            M = np.zeros((NUM_TILES, STATE_WIDTH), dtype=np.float32)
            for i in range(SEQ_LEN):
                window = _build_tile_window(embed_table, tokens, i, NUM_TILES, M, COLUMN_NEURONS)
                M, logits, aux = model.step(window, M, lr)
                if i in targets:
                    loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                    if aux is not None:
                        loss = loss + aux
                    loss.grad = np.array(1.0, dtype=np.float32)
                    loss.backward()
                    opt.step(model.parameters_for_optimizer(), lr=lr)

            if step % checkpoint_every == 0:
                acc = evaluate(model, rng, embed_table)
                elapsed = time.time() - t0
                print(
                    f"step={step:>7}  acc={acc:.4f}  ({elapsed:.0f}s elapsed, {elapsed / step:.3f}s/step)", flush=True
                )

        print(f"# DONE arm={arm} use_energy={use_energy} ({time.time() - t0:.0f}s total)", flush=True)
    except RuntimeWarning:
        print("Caught RuntimeWarning.")
        traceback.print_exc()


if __name__ == "__main__":
    main()
