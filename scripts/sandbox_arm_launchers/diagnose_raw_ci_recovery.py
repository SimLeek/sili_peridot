"""Diagnostic run (not a comparison arm), same base config as v3
(l2decay lambda=0.05 threshold=0.9 temperature=0.05) -- direct
instruction, after "sustained decrease to what number exactly... [ci]
potentially fighting more than just this decay": col_importance (the
only signal logged so far) is a SECOND EMA on top of the real
per-synapse ci accumulator, so its own recovery dynamics between
plasticity touches can't be measured from what's already logged. See
docs/research/toy_tile_recurrence_rmt.rst:raw_ci_landscape_capture for
the full derivation.

Two new captures, both opt-in and both verified safe/cheap against the
real model class before use:
1. plasticity_raw_importance_log=True -- each once-per-cycle snapshot
   also gets the raw (in_features, out_features) importance matrix, a
   full-run "landscape" (renderable as a lossless video afterward,
   stored as float32 .npz first).
2. raw_ci_sample_fn -- fires once per REAL weight update (the true
   update cadence), tracking a small fixed sample of individual
   synapses per layer, to directly resolve the 1-step-vs-many-step
   recovery question.

Run at the full 100000 steps, matching v2/v3's length -- a first
30000-step attempt was discarded: checked against v3's OWN step~30000
snapshot (not a later step), which was already fully saturated
(col_importance_mean 99.4-99.99, l2sat~0.99-1.0, l2decay~0.87-0.88
across all 4 pools) while the short diagnostic run was nowhere close
(means 1.6-19.5, l2decay=0.0) -- a real divergence at matched step
count, not an artifact of comparing different points in a schedule.
Direct instruction: relaunch at matching length instead of guessing at
why a truncated run differed."""

import os
import sys

sys.path.insert(0, ".")
import numpy as np

import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

MAX_STEPS = 100000
N_SAMPLES_PER_LAYER = 32
FLUSH_EVERY = 1000
SAMPLE_DIR = "logs/raw_ci_diagnostic/dense_lr_unscaled_v3_samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)

_sampled_indices: dict[str, np.ndarray] = {}
_buffers: dict[str, dict[str, list]] = {}
_flush_count: dict[str, int] = {}


def _flush(name):
    buf = _buffers[name]
    if not buf["steps"]:
        return
    n = _flush_count.get(name, 0)
    np.savez_compressed(
        os.path.join(SAMPLE_DIR, f"{name}.chunk{n:04d}.npz"),
        steps=np.array(buf["steps"], dtype=np.int64),
        values=np.stack(buf["values"]),
        sample_indices=_sampled_indices[name],
    )
    _flush_count[name] = n + 1
    buf["steps"] = []
    buf["values"] = []


def raw_ci_sample_fn(step, model):
    for name, layer in model._named_real_layers():
        raw = np.array(layer.importance)
        if name not in _sampled_indices:
            n = len(raw)
            rng = np.random.default_rng(abs(hash(name)) % (2**32))
            k = min(N_SAMPLES_PER_LAYER, n)
            _sampled_indices[name] = rng.choice(n, size=k, replace=False)
            _buffers[name] = {"steps": [], "values": []}
        idx = _sampled_indices[name]
        buf = _buffers[name]
        buf["steps"].append(step)
        buf["values"].append(raw[idx].copy())
        if len(buf["steps"]) >= FLUSH_EVERY:
            _flush(name)


def log_fn(
    step,
    vocab_size,
    k,
    phase,
    event,
    loss_ema,
    acc_ema,
    ranks=None,
    steps_per_sec=None,
    max_streak=None,
    dy_r_target=None,
    x_r_target=None,
    layer_timing=None,
    window_wall_s=None,
    plasticity_totals=None,
):
    loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
    acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
    tag = f"  [{event}]" if event else ""
    sps_s = f"  steps/sec={steps_per_sec:.1f}" if steps_per_sec is not None else ""
    plast_s = ""
    if plasticity_totals:
        worst_key, worst = max(plasticity_totals.items(), key=lambda kv: kv[1]["last_l2_decay_strength"])
        plast_s = (
            f"  l2decay[{worst_key}(sat={worst['last_l2_sat_ratio']:.2f},decay={worst['last_l2_decay_strength']:.2f})]"
        )
    print(
        f"  step={step:>7}  vocab={vocab_size:>4}  k={k:>3}  loss={loss_s}  acc={acc_s}{tag}{sps_s}{plast_s}",
        flush=True,
    )


print(
    f"# DIAGNOSTIC: raw ci recovery capture, same config as v3 (lambda=0.05 threshold=0.9 "
    f"temperature=0.05), max_steps={MAX_STEPS}, sampling {N_SAMPLES_PER_LAYER} synapses/layer "
    f"every real step -> {SAMPLE_DIR}, landscape frames in the usual column-log snapshot dir",
    flush=True,
)
r = m.train_curriculum(
    "fp32",
    MAX_STEPS,
    1000,
    0.015,
    16,
    10,
    embed_width=36,
    wrong_streak_threshold=100000000,
    k_first_target=3,
    plasticity_reset_enable=True,
    plasticity_reset_l2_decay_lambda=0.05,
    plasticity_reset_l2_decay_threshold=0.9,
    plasticity_reset_l2_decay_temperature=0.05,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v3_diagnostic",
    plasticity_raw_importance_log=True,
    raw_ci_sample_fn=raw_ci_sample_fn,
    log_every=250,
    log_fn=log_fn,
)
for name in list(_buffers):
    _flush(name)
print(
    f"\nFINAL final_vocab={r['final_vocab']} final_k={r['final_k']} "
    f"steps_per_sec={r['steps_per_sec']:.2f} ({r['elapsed_s']:.0f}s)",
    flush=True,
)
print("DONE_DIAGNOSE_RAW_CI_RECOVERY", flush=True)
