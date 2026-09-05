"""At what model scale (state_width) or INPUT/ACTIVATION sparsity does
stochastic rounding's inherent per-step noise become more "stable to
work with" -- and stop harming PEAK TRAINED accuracy as much as it
might at small scale/low input sparsity?

Uses an OUT-OF-CONTEXT task (seq_len = NUM_TILES + OOC_MARGIN), not the
in-context seq_len==NUM_TILES setup other probes in this investigation
used. generate_copy_sequence's own docstring and _build_tile_window's
sliding-window math both confirm: at seq_len<=NUM_TILES, the tile
window at the final (query) position reaches all the way back to
position 0 (the key) directly, so the task is solvable from local
attention alone with NO real dependency on the carried recurrent state
M -- a genuinely "in-context" task. Direct instruction: the stochastic-
rounding issues this whole investigation is chasing showed up on tasks
that specifically REQUIRE recurrence (state carried across tile
boundaries beyond the local window's reach), not this in-context
setup -- so this grid needs seq_len > NUM_TILES, forcing the key to be
carried forward through M past where the local window can see it
directly. Per project convention (grow past the in-context ceiling ONE
increment at a time, not a big out-of-context jump), OOC_MARGIN is
kept small (+2) rather than testing some large out-of-context gap.

Direct follow-up to the stuck-weights investigation: the (row,col)-
keyed check_stuck_weights redesign (model/eval_stuck_weights.py) just
confirmed stochastic rounding produces real nonzero movement on
already-live synapses where deterministic rounding is bit-exact-frozen
(mean_delta_w=0.002259 vs 0.000000, toy scale, 800 steps) -- so
stochastic rounding is doing real, useful work. But it's also, by
construction, noisy (every near-threshold update has some chance of
rounding either way each step). This script measures whether that
noise is a real cost in FINAL ACCURACY (not just a per-synapse
curiosity) and whether the cost shrinks as state_width or INPUT
sparsity changes.

IMPORTANT, corrected per direct feedback: an earlier version of this
script swept CONNECTIVITY/WEIGHT sparsity (dense=False, max_weights as
a fraction of n_in*n_out on the layer's own synapses) under the label
"density" -- that's a different axis from what was actually asked
("input sparsity"). Connectivity now stays FULLY DENSE (dense=True) at
every grid point; the swept axis here is genuine INPUT/ACTIVATION
sparsity instead: each vocab token's embedding is given a fixed random
subset of active (nonzero) dimensions at the requested input_density
fraction, so the SIGNAL a synapse ever sees is sparse, independent of
how densely connected the layer itself is. This is the intended
"sparse input" reading -- per direct correction, NOT weight sparsity.
Only vary the one intended axis per comparison: connectivity density
is now held fixed rather than co-varying with input sparsity.

Two axes, both swept independently against the SAME deterministic
baseline at each grid point (comparing stochastic to the deterministic
arm run at that exact width/input-sparsity, not to a single fixed
baseline):
  - state_width, via (embed_width, column_neurons) pairs, same
    convention as weight_update_magnitude_vs_width.py.
  - input_density: fraction of each vocab token's embedding dimensions
    that are nonzero (1.0 = fully dense input, same as before this
    axis existed; lower = sparser input signal, same active dims reused
    every time that token appears, so it's a fixed per-token sparse
    representation, not per-step dropout noise).

For each grid point, N_SEEDS independent seeds per arm (deterministic,
stochastic), each trained for N_STEPS then evaluated on held-out copy-
task accuracy (real metric, not a proxy). Reports per grid point:
  - stoch_mean, stoch_std (variance across seeds -- the "how noisy" signal)
  - det_mean (deterministic baseline at the same grid point)
  - gap = det_mean - stoch_mean (positive = stochastic underperforms;
    shrinking toward 0 as width/input_density increases is the
    "becomes safe" signal the user asked about)

Kept fast on purpose (per-project convention: these diagnostics should
run in 5-10 minutes, not hours) -- small grid, few seeds, short training.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/stochastic_stability_vs_scale_sparsity.py
"""

import functools

import numpy as np
from sili.sparse_rnn import DISLDOLayer, DISLDOLayerDeterministic

from model.toy_precision_models import TrueMultiDigitLayer
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum, predicted_token
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from scripts.train_tile_curriculum import _build_tile_window, generate_copy_sequence

VOCAB = 10
NUM_TILES = 4
OOC_MARGIN = 2  # seq_len = NUM_TILES + OOC_MARGIN -- past the in-context
# ceiling by one small, deliberate step (see module docstring)
SEQ_LEN = NUM_TILES + OOC_MARGIN
N_STEPS = 200
EVAL_SEQUENCES = 30
LR = 0.01
SEEDS = [1000, 1001, 1002]

# (embed_width, column_neurons) -> state_width = product.
WIDTHS = [(8, 4), (16, 8), (32, 16)]  # state_width 32, 128, 512
INPUT_DENSITIES = [1.0, 0.25, 0.0625]  # fraction of embedding dims left nonzero per token


def _sparse_embed_table(task_rng, seed, vocab, embed_width, input_density):
    """Dense random embedding table, then (if input_density<1.0) each
    vocab row gets a FIXED random subset of active dims zeroed out to
    the requested density -- same active dims every time that token is
    looked up, so this is a stable sparse input REPRESENTATION, not
    per-step dropout noise. Mask RNG is seeded independently of
    task_rng (token sequence generation) so the two axes don't
    entangle."""
    embed_table = task_rng.randn(vocab, embed_width).astype(np.float32) * 0.3
    if input_density >= 1.0:
        return embed_table
    mask_rng = np.random.RandomState(seed + 5000)
    n_active = max(1, round(input_density * embed_width))
    mask = np.zeros((vocab, embed_width), dtype=np.float32)
    for v in range(vocab):
        active_dims = mask_rng.choice(embed_width, size=n_active, replace=False)
        mask[v, active_dims] = 1.0
    return embed_table * mask


def _train_and_eval(embed_width, column_neurons, input_density, digit_backend, seed):
    state_width = embed_width * column_neurons
    max_weights = state_width * state_width  # connectivity always fully dense -- see module docstring

    digit_cls = functools.partial(
        TrueMultiDigitLayer, digit_cls=digit_backend, n_stages=3, base=12.0, lr_power=0.0, dense=True, scale_rank=1
    )
    rng = np.random.default_rng(seed)
    model = ToyTileRecurrenceRealFP4(
        VOCAB,
        embed_width,
        column_neurons,
        mlp_hidden=0,
        num_tiles=NUM_TILES,
        max_weights=max_weights,
        num_cpus=1,
        disldo_cls=digit_cls,
        rng=rng,
    )
    task_rng = np.random.RandomState(seed)
    embed_table = _sparse_embed_table(task_rng, seed, VOCAB, embed_width, input_density)
    opt = AdamOptimizer()

    for _step in range(N_STEPS):
        tokens, pairs = generate_copy_sequence(task_rng, VOCAB, SEQ_LEN)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        total_loss = None
        for i in range(SEQ_LEN):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, column_neurons)
            M, logits, _aux = model.step(window, M, LR)
            if i in targets:
                tgt_loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                total_loss = tgt_loss if total_loss is None else total_loss + tgt_loss
        if total_loss is not None:
            total_loss.backward()
            clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
            opt.step(model.parameters_for_optimizer(), lr=LR)

    eval_rng = np.random.RandomState(seed + 777)
    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, pairs = generate_copy_sequence(eval_rng, VOCAB, SEQ_LEN)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        for i in range(SEQ_LEN):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, column_neurons)
            M, logits, _aux = model.step(window, M, 0.0)
            if i in targets:
                pred = predicted_token(logits, NUM_TILES - 1)
                correct += int(pred == targets[i])
                total += 1
    return correct / total if total else 0.0


def run_grid_point(embed_width, column_neurons, input_density):
    det_accs = [_train_and_eval(embed_width, column_neurons, input_density, DISLDOLayerDeterministic, s) for s in SEEDS]
    stoch_accs = [_train_and_eval(embed_width, column_neurons, input_density, DISLDOLayer, s) for s in SEEDS]
    det_mean = float(np.mean(det_accs))
    stoch_mean = float(np.mean(stoch_accs))
    stoch_std = float(np.std(stoch_accs))
    gap = det_mean - stoch_mean
    return det_mean, stoch_mean, stoch_std, gap, det_accs, stoch_accs


if __name__ == "__main__":
    print(
        f"{'state_width':>12} {'input_density':>13} {'det_mean':>9} {'stoch_mean':>10} {'stoch_std':>9} {'gap':>8}",
        flush=True,
    )
    for ew, cn in WIDTHS:
        state_width = ew * cn
        for input_density in INPUT_DENSITIES:
            det_mean, stoch_mean, stoch_std, gap, det_accs, stoch_accs = run_grid_point(ew, cn, input_density)
            print(
                f"{state_width:>12} {input_density:>13.4f} {det_mean:>9.3f} {stoch_mean:>10.3f} "
                f"{stoch_std:>9.3f} {gap:>8.3f}   det={det_accs} stoch={stoch_accs}",
                flush=True,
            )
