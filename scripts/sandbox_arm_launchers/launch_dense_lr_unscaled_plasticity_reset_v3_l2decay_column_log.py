"""v3 relaunch, same base config as launch_dense_lr_unscaled_plasticity_reset.py
(v2) plus l2_decay_lambda=0.05/threshold=0.9/temperature=0.05 and column
logging enabled, at NUM_CPUS=4 (matching v2, not the data-collection
run's NUM_CPUS=2) for a clean comparison. See
docs/research/toy_tile_recurrence_rmt.rst:l2_saturation_decay for the
full derivation."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
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
            f",imp={worst['last_importance']:.4f},"
            f"l2sat={worst['last_l2_sat_ratio']:.2f},"
            f"l2decay={worst['last_l2_decay_strength']:.2f})]"
        )
    print(
        f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
        f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{streak_s}{plast_s}",
        flush=True,
    )


print(
    "# DENSE-UNSCALED + PLASTICITY_RESET v3 (dead pool pruned, k=1.0, reset_fraction=0.01, "
    "L2-saturation-gated decay lambda=0.05 threshold=0.9 temperature=0.05 [SOFT sigmoid, not "
    "a hard cutoff]) + COLUMN DATA COLLECTION, K_START=2, write_time_aux_targets=False "
    "precision=fp32 max_steps=100000 seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- "
    "testing whether opposing col_importance's real saturation at max_ci=100 breaks the "
    "v2 plateau (vocab=64/k=3 flat for its own final steps); logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v3_l2decay/",
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
    plasticity_reset_l2_decay_lambda=0.05,
    plasticity_reset_l2_decay_threshold=0.9,
    plasticity_reset_l2_decay_temperature=0.05,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v3_l2decay",
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
print("DONE_DENSE_LR_UNSCALED_PLASTICITY_RESET_V3_L2DECAY_COLUMN_LOG", flush=True)
