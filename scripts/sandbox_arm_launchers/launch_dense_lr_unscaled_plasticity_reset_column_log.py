"""Data-collection companion to launch_dense_lr_unscaled_plasticity_reset.py
(same v2 config: dead pool pruned, k=1.0, reset_fraction=0.01) --
direct instruction, after reviewing the log files this investigation
already produces (importance, loss, accuracy): could a real
reset-selection equation be fit from actual training data instead of a
hand-derived heuristic? Only intentional difference from the v2
launcher: plasticity_column_log_dir writes one small .npz per
completed cycle per layer/pool (col_importance/col_grad_slow/
col_grad_fast/col_grad_var/col_age/col_reset_active plus step/
loss_ema/acc_ema) under logs/plasticity_column_snapshots/dense_lr_unscaled/
-- see docs/research/toy_tile_recurrence_rmt.rst:plasticity_column_state.
NUM_CPUS=2 (lower than the other 3 concurrent runs' NUM_CPUS=4) since
this is a 4th run added on top of already-oversubscribed CPU (3 runs at
NUM_CPUS=4 = 12 requested vs 8 real cores); reusing the DENSE arm
specifically since it's the cleanest, most-already-understood config
(no polyak_lr or Arm C sine-gate confounds) for a first data-collection
pass."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 2
m.K_START = 2


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
    streak_s = f"  max_streak={max_streak:>2}/10" if max_streak is not None else ""
    plast_s = ""
    if plasticity_totals:
        n_reset = sum(t["n_reset"] for t in plasticity_totals.values())
        worst_key, worst = max(plasticity_totals.items(), key=lambda kv: kv[1]["last_deviation"])
        plast_s = (
            f"  plasticity[reset={n_reset} "
            f"worst={worst_key}(dev={worst['last_deviation']:.2f}"
            f"[{worst['last_min_deviation']:.2f},{worst['last_max_deviation']:.2f}]"
            f",imp={worst['last_importance']:.4f})]"
        )
    print(
        f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
        f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{streak_s}{plast_s}",
        flush=True,
    )


print(
    "# DENSE-UNSCALED + PLASTICITY_RESET v2 + COLUMN DATA COLLECTION "
    "(dead pool pruned, k=1.0, reset_fraction=0.01), K_START=2, "
    "write_time_aux_targets=False precision=fp32 "
    "max_steps=100000 seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=2 -- "
    "logging per-column state to logs/plasticity_column_snapshots/dense_lr_unscaled/ "
    "for offline reset-selection-equation fitting",
    flush=True,
)
r = m.train_curriculum(
    "fp32",
    100000,
    1000,
    0.015,
    16,
    10,
    embed_width=36,
    wrong_streak_threshold=100000000,
    k_first_target=3,
    plasticity_reset_enable=True,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled",
    log_every=250,
    log_fn=log_fn,
)
print(
    f"\nFINAL final_vocab={r['final_vocab']} final_k={r['final_k']} "
    f"final_phase={r['final_phase']} steps_per_sec={r['steps_per_sec']:.2f} "
    f"({r['elapsed_s']:.0f}s)",
    flush=True,
)
print(f"PEAK peak_vocab={r['peak_stage']['vocab']} peak_k={r['peak_stage']['k']}", flush=True)
print(f"STAGE_HISTORY {r['stage_history']}", flush=True)
print("DONE_DENSE_LR_UNSCALED_PLASTICITY_RESET_COLUMN_LOG", flush=True)
