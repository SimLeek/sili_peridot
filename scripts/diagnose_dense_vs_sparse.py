"""Diagnostic: dense connectivity trains at chance while sparse trains
fine (JOURNAL.md 2026-08-10) -- even after fixing two real init-scale
bugs, confirmed via a single-digit forward-output check (std=0.937,
healthy). This script compares the SAME model at the SAME seed, dense
vs sparse, at every pipeline stage (via ToyTileRecurrenceRealFP4.step's
new debug=True instrumentation) over the first several real training
steps, to find exactly where their value distributions diverge --
digging into the real math instead of guessing at another scale fix.

Usage: python3 scripts/diagnose_dense_vs_sparse.py [n_steps]
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from sili import _cpu
from sili.sparse_rnn import DISLDOLayerDeterministic
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4
from model.toy_precision_models import TrueMultiDigitLayer
from model.toy_recall_models import cross_entropy_sum, AdamOptimizer, lr_schedule, clip_grad_norm_
from scripts.train_tile_curriculum import generate_copy_sequence, _build_tile_window

import functools

VOCAB = 10
EMBED_WIDTH = 16
COLUMN_NEURONS = 8
NUM_TILES = 4
MAX_WEIGHTS = 1500
PEAK_LR = 0.002
MAX_GRAD_NORM = 1.0  # matches train_tile_curriculum.py's own fix -- see its docstring
WARMUP_STEPS = 50
SEED = 1000

STAGES = ["qkv_source", "q", "k", "v", "attn_raw", "attn_o_proj",
         "pre_clip", "clip_fraction", "post_clip", "logits"]


def build_model(dense: bool, seed: int) -> ToyTileRecurrenceRealFP4:
    # Same gotcha as feedback_seed_stochastic_rng_for_comparisons: the FP4
    # stochastic-rounding RNG is thread-local and unseeded by default,
    # separate from every `seed`/`rng` argument here -- even though
    # DISLDOLayerDeterministic's WEIGHT rounding is deterministic, leaving
    # this unseeded still makes two runs of this exact script disagree
    # (confirmed directly: two invocations gave NaN at different steps).
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    digit_cls = functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic,
                                  n_stages=3, base=12.0, lr_power=0.0, dense=dense)
    rng = np.random.default_rng(seed)
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    return ToyTileRecurrenceRealFP4(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, state_width * 2, NUM_TILES, MAX_WEIGHTS,
        num_cpus=1, disldo_cls=digit_cls, use_attention=True, rng=rng)


def _scale_stats(layer, n_in, n_out):
    """Direct read of a single digit's real value_scale/output_scale --
    RMSprop-trained, could be drifting badly over many steps even if
    the FIRST few steps' forward values look healthy."""
    vs = [layer._c.get_value_scale(r) for r in range(n_in)]
    os_ = [layer._c.get_output_scale(c) for c in range(n_out)]
    return {"value_scale_mean": float(np.mean(vs)), "value_scale_max": float(np.max(np.abs(vs))),
           "output_scale_mean": float(np.mean(os_)), "output_scale_max": float(np.max(np.abs(os_)))}


def run(model: ToyTileRecurrenceRealFP4, n_steps: int, seed: int, sample_every: int = 1,
        detail_range: tuple | None = None):
    """Mirrors train_tile_curriculum.py main()'s training loop closely
    enough to be representative, minimal enough to stay readable.

    `n_steps` sets the LR schedule's total length (lr_schedule's cosine
    decay is parameterized by it) -- comparing step 300 across two runs
    with DIFFERENT n_steps compares two different effective LRs, not
    the same point in training (confirmed directly: caused an apparent
    "different NaN onset" that was actually just this, not real
    nondeterminism). Always run the FULL n_steps; use `sample_every`/
    `detail_range` to control how much gets recorded, not how far the
    schedule itself progresses.

    `detail_range`: optional (start, end) step range (inclusive) to
    record EVERY step's stats within, regardless of `sample_every` --
    for zooming into a specific window (e.g. right around a divergence)
    without needing a separate, shorter (and therefore differently
    -scheduled) run.
    """
    task_rng = np.random.RandomState(seed)
    embed_table = task_rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3
    opt = AdamOptimizer()
    state_width = EMBED_WIDTH * COLUMN_NEURONS
    seq_len = 2  # fixed, in-context only -- this diagnostic doesn't need the curriculum

    history = []
    for step in range(1, n_steps + 1):
        lr = lr_schedule(step, n_steps, PEAK_LR, WARMUP_STEPS)
        tokens, pairs = generate_copy_sequence(task_rng, VOCAB, seq_len)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        in_detail = detail_range is not None and detail_range[0] <= step <= detail_range[1]
        do_debug = in_detail or (step % sample_every == 0) or step == 1
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, lr, debug=do_debug)
            if i in targets:
                loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                if aux is not None:
                    loss = loss + aux
                loss.backward()
                clip_grad_norm_(model.parameters_for_optimizer(), MAX_GRAD_NORM)
                opt.step(model.parameters_for_optimizer(), lr=lr)
        if do_debug:
            entry = dict(model._last_step_debug_stats)
            entry["step"] = step
            # Direct read of value_scale/output_scale AND raw weight
            # magnitude for every real layer (all 3 digits each) -- to
            # find which one's DISLDO-internal RMSprop-trained update
            # (untouched by any Python-level clip) is actually the one
            # exploding, now that clipping attn/M_new_t/gradients was
            # confirmed NOT sufficient to prevent the NaN divergence.
            layers = {"q": model.q_proj, "k": model.k_proj, "v": model.v_proj,
                     "o": model.o_proj, "lm": model.lm_head}
            entry["layer_scales"] = {}
            for name, layer in layers.items():
                digit_maxes = []
                for digit in layer.digits:
                    n_in, n_out = digit._c.n_inputs, digit._c.n_outputs
                    s = _scale_stats(digit, n_in, n_out)
                    w = np.asarray(digit._c.weights_vals)
                    digit_maxes.append((s["output_scale_max"], s["value_scale_max"],
                                        float(np.abs(w).max()) if w.size else 0.0))
                entry["layer_scales"][name] = digit_maxes
            # log_sigmas/sigmas: unbounded below in the C++ attention math
            # (gaussian_attention_backward's dSigmas has a 1/sigma^3 term,
            # attention.hpp:857, no floor anywhere) -- suspected runaway
            # -shrink feedback loop, untouched by any clip added so far.
            ls = model.log_sigmas.data
            entry["log_sigmas"] = (float(ls.min()), float(ls.max()))
            entry["sigmas"] = (float(np.exp(ls).min()), float(np.exp(ls).max()))
            # centers: plain Tensor param, updated via the external
            # AdamOptimizer with a clipped GRADIENT NORM but no clamp on
            # the resulting VALUE -- could in principle random-walk away
            # from the valid key range [0, num_tiles) over enough steps.
            c = model.centers.data
            entry["centers"] = (float(c.min()), float(c.max()))
            history.append(entry)
    return history


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sample_every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    detail_range = None
    if len(sys.argv) > 4:
        detail_range = (int(sys.argv[3]), int(sys.argv[4]))

    print(f"Building dense (seed={SEED}) and sparse (seed={SEED}) models...")
    dense_model = build_model(dense=True, seed=SEED)
    sparse_model = build_model(dense=False, seed=SEED)

    print(f"Running {n_steps} real training steps on each (sampled every {sample_every}"
          f"{f', detail {detail_range}' if detail_range else ''})...")
    dense_hist = run(dense_model, n_steps, SEED, sample_every, detail_range)
    sparse_hist = run(sparse_model, n_steps, SEED, sample_every, detail_range)

    if sample_every == 1:
        for step_idx in range(len(dense_hist)):
            print(f"\n=== step {step_idx + 1} ===")
            print(f"{'stage':<16} {'dense mean':<12} {'dense std':<12} {'dense |max|':<12} | "
                  f"{'sparse mean':<12} {'sparse std':<12} {'sparse |max|':<12}")
            for stage in STAGES:
                d = dense_hist[step_idx].get(stage)
                s = sparse_hist[step_idx].get(stage)
                if stage == "clip_fraction":
                    print(f"{stage:<16} {d:<12.4f} {'':<12} {'':<12} | {s:<12.4f}")
                    continue
                print(f"{stage:<16} {d['mean']:<12.4f} {d['std']:<12.4f} {d['abs_max']:<12.4f} | "
                      f"{s['mean']:<12.4f} {s['std']:<12.4f} {s['abs_max']:<12.4f}")
    else:
        # Longer-run trend view: logits std (collapsing to ~0 = dead model),
        # clip_fraction, and per-layer max(output_scale, value_scale, raw
        # weight magnitude) across all 3 digits -- to find which layer's
        # DISLDO-internal scale/weight update is the one actually exploding,
        # since Python-level activation/gradient clipping alone was
        # confirmed NOT sufficient to prevent the dense NaN divergence.
        # Weight/scale magnitudes confirmed already structurally bounded
        # (FP4 codes have a hard ceiling) and log_sigmas/sigmas confirmed
        # NOT collapsing -- neither is the runaway-growth source. Focus
        # instead on the stages that are genuinely UNCLIPPED today: q/k/v
        # (only attn_o_proj got a forward clip so far) and attn_raw.
        act_stages = ["qkv_source", "q", "k", "v", "attn_raw", "attn_o_proj"]
        header = f"{'step':<6} {'d.logstd':<10} {'d.clipf':<8}"
        for st in act_stages:
            header += f" {'d.'+st+'.|max|':<16}"
        header += f" {'d.centers[min,max]':<20}"
        print("\n" + header)
        for d in dense_hist:
            row = f"{d['step']:<6} {d['logits']['std']:<10.4f} {d['clip_fraction']:<8.4f}"
            for st in act_stages:
                row += f" {d[st]['abs_max']:<16.4f}"
            if "centers" in d:
                row += f" [{d['centers'][0]:.3f},{d['centers'][1]:.3f}]"
            print(row)


if __name__ == "__main__":
    sys.exit(main())
