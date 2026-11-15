"""Tests RSM's ACTUAL claim, in the regime its own paper claims it.

Why this exists: the RSM-inspired sparse memory write was measured negative on
MQAR across every axis tried (k_active 2%-100%, filler predictability, decoder
lr/schedule, stream redundancy 6%-97.5%, write-gated inactivity) -- see
JOURNAL.md's 2026-11-12 entry. The literature says that is the expected outcome,
on two independent axes each source states plainly:

  * RSM (arXiv:1905.11589) names generalization to UNSEEN input sequences as its
    most significant limitation (PTB perplexity 124, ~a basic RNN, vs LSTM
    58-86). MQAR draws fresh random key-value bindings every episode, so nearly
    every example is an unseen sequence.
  * Zoology (arXiv:2312.04927) finds the required hidden-state dimensionality
    GROWS with the number of key-value pairs to recall, and that scaling
    parameters does not substitute (a 70M attention model beats a 1.4B
    gated-convolution model). RSM's headline claim is ~an order of magnitude
    LESS memory than LSTM/GRU. Those are opposed objectives.

Our own MQAR data reproduced Zoology's scaling law: accuracy was monotonic in
usable state (k_active 2%: 0.333, 25%: 0.385, off: 0.401, 100%: 0.435). So
sparsifying the write strictly costs recall there. MQAR is close to a worst case
for this mechanism, and tuning it against MQAR was the wrong experiment.

What RSM actually claims, and what this measures instead:
  * FIXED generating process (its wins were Embedded Reber Grammar, bouncing-ball
    dynamics, a fixed maze -- processes it can memorize), not fresh bindings.
  * A COMPRESSIBLE, redundant stream (successive video frames are ~99%
    identical), which is also the proof's favourable regime: synaptic_hold_bptt_
    approximation's `all_inactive_is_exact` is exact only when intermediate ticks
    are INACTIVE, which requires the input to barely change.
  * The win condition is MEMORY FOOTPRINT AT EQUAL QUALITY, not peak accuracy.
    So this sweeps num_memory_slots and asks where each arm's quality collapses:
    RSM's claim is that it holds quality at FEWER slots than the dense baseline.

Task: next-token prediction over a fixed, memorizable, redundant stream. The
process is a fixed periodic pattern (so it is learnable and reused across
episodes, unlike MQAR) with run-length `RUN` repetition (so redundancy is
(RUN-1)/RUN and most ticks are genuinely inactive). Prediction target is the
next token at every position, which is RSM's own objective.

Honest scope: a negative here would be a much stronger result than the MQAR
negative, because it would be a failure in the mechanism's own claimed regime.
A positive here does NOT transfer back to MQAR, for the two reasons above.

Run: python -m scripts.prototype_rsm_fixed_process_memory
"""

import time

import numpy as np

from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum, predicted_token
from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT

VOCAB = 32
EMBED_WIDTH = 12
COLUMN_NEURONS = 8
NUM_TILES = 16
MAX_WEIGHTS = 4000
NUM_CPUS = 2
PEAK_LR = 0.015
SEQ_LEN = 64
RUN = 8  # run length => redundancy (RUN-1)/RUN = 87.5%
PERIOD = 12  # fixed generating process: a periodic pattern of distinct symbols
TRAIN_STEPS = 400
EVAL_SEQS = 20

# The fixed process. Reused across every episode, so the model CAN memorize it --
# unlike MQAR's fresh-per-episode bindings. Held RUN ticks per symbol, so most
# consecutive pairs are identical.
BASE_PATTERN = [(7 * i + 3) % VOCAB for i in range(PERIOD)]


def make_sequence(offset: int) -> np.ndarray:
    """One episode of the fixed redundant process, phase-shifted by `offset`.
    Phase varies so the model must track position rather than memorize one
    absolute string, but the PROCESS is identical every episode."""
    return np.array(
        [BASE_PATTERN[((i // RUN) + offset) % PERIOD] for i in range(SEQ_LEN)],
        dtype=np.int64,
    )


def _window(embed_table, tokens, i):
    rows = [embed_table[tokens[max(0, i - (NUM_TILES - 1) + j)]] for j in range(NUM_TILES)]
    return np.stack(rows, axis=0)


def run_arm(num_memory_slots, rsm_on, k_frac, seed, novelty=None):
    rng = np.random.default_rng(seed)
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    k_active = max(1, round(k_frac * num_memory_slots * state_width)) if rsm_on else None
    model = ToyTileRecurrenceRMT(
        VOCAB,
        EMBED_WIDTH,
        COLUMN_NEURONS,
        NUM_TILES,
        num_memory_slots,
        MAX_WEIGHTS,
        num_cpus=NUM_CPUS,
        qkvo_norm_enable=True,
        rsm_memory_write_enable=rsm_on,
        rsm_k_active=k_active,
        rsm_novelty_threshold=novelty,
        rng=rng,
    )
    opt = AdamOptimizer()
    embed_table = (rng.standard_normal((VOCAB, EMBED_WIDTH)) * 0.3).astype(np.float32)

    for step in range(TRAIN_STEPS):
        tokens = make_sequence(step % PERIOD)
        memory = np.zeros((num_memory_slots, state_width), dtype=np.float32)
        if rsm_on:
            model.rsm_reset_sequence()
        for i in range(SEQ_LEN - 1):
            mem_new, logits, aux = model.step(_window(embed_table, tokens, i), memory, PEAK_LR)
            memory = mem_new
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, int(tokens[i + 1]))])
            if aux is not None:
                loss = loss + aux
            loss.backward()
            clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
            opt.step(model.parameters_for_optimizer(), lr=PEAK_LR)

    correct = total = 0
    for e in range(EVAL_SEQS):
        tokens = make_sequence(e % PERIOD)
        memory = np.zeros((num_memory_slots, state_width), dtype=np.float32)
        if rsm_on:
            model.rsm_reset_sequence()
        for i in range(SEQ_LEN - 1):
            memory, logits, _aux = model.step(_window(embed_table, tokens, i), memory, 0.0)
            correct += int(predicted_token(logits, NUM_TILES - 1) == int(tokens[i + 1]))
            total += 1
    return correct / total


def main():
    t0 = time.time()
    print(f"fixed process: PERIOD={PERIOD} RUN={RUN} => redundancy {100 * (RUN - 1) / RUN:.1f}%, VOCAB={VOCAB}")
    print(f"chance = {1.0 / VOCAB:.4f};  memory footprint sweep, RSM's own win condition\n")
    print(f"{'slots':>6} {'dense (RSM off)':>16} {'RSM 25%':>10} {'RSM 25%+gate':>13}")
    for slots in (2, 4, 16):
        row = []
        for rsm_on, nov in ((False, None), (True, None), (True, 0.5)):
            accs = [run_arm(slots, rsm_on, 0.25, 1000 + s, novelty=nov) for s in range(2)]
            row.append(sum(accs) / len(accs))
        print(f"{slots:>6} {row[0]:>16.4f} {row[1]:>10.4f} {row[2]:>13.4f}", flush=True)
    print(f"\ntotal {time.time() - t0:.0f}s")
    print("Read it as: does RSM hold quality at FEWER slots than dense? That is the claim.")


if __name__ == "__main__":
    main()
