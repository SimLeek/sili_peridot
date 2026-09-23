"""v9: real-engine validation of layer_l2_decay -- the offline Python
sandbox's layer-wide, rank-preserving candidate (uniform multiplicative
shrink on the WHOLE population, strength driven by the population's own
L2-norm saturation ratio) is a faithful port of a mechanism that
ALREADY EXISTS in the real C++ engine (built earlier this session, but
never enabled in any run -- lambda was 0.0 everywhere so far). Direct
instruction, after the offline sandbox showed this candidate avoiding
saturation while preserving rank order (unlike ceiling_decay_step's
per-column clipping): "let's try both of them and record while seeing
if it helps reliably beat mqar rather than randomly stalling."

reset_fraction=0.0 isolates L2 decay as the ONLY active mechanism (no
individual per-column select_by_deviation/top-importance resets at
all) -- matching exactly what the sandbox's layer_l2_decay_step tested
in isolation, so this run is the real-engine analog of that specific
sandbox result, not a combination with anything else. lambda/threshold/
temperature match the sandbox test's own parameters exactly."""

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
    plast_s = ""
    if plasticity_totals:
        n_reset = sum(t["n_reset"] for t in plasticity_totals.values())
        worst_key, worst = max(plasticity_totals.items(), key=lambda kv: kv[1]["last_l2_decay_strength"])
        plast_s = (
            f"  plasticity[reset={n_reset} "
            f"worst={worst_key}(l2sat={worst['last_l2_sat_ratio']:.2f}"
            f",l2decay={worst['last_l2_decay_strength']:.2f}"
            f",imp={worst['last_importance']:.4f})]"
        )
    print(
        f"  step={step:>7}  vocab={vocab_size:>4}  k={k:>3}  loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{plast_s}",
        flush=True,
    )


print(
    "# DENSE-UNSCALED + PLASTICITY_RESET v9 (layer_l2_decay ONLY -- reset_fraction=0.0 disables "
    "all individual per-column resets, isolating the layer-wide L2-saturation-gated decay "
    "mechanism exactly as tested in the offline sandbox: lambda=0.05 threshold=0.9 "
    "temperature=0.05 max_ci=100.0), K_START=2, precision=fp32 max_steps=100000 seed=1000 "
    "embed_width=36 k_first_target=3 NUM_CPUS=4 -- testing whether this reliably beats MQAR "
    "instead of randomly stalling like v5-v8 did; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v9_layer_l2_decay/",
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
    plasticity_reset_reset_fraction=0.0,
    plasticity_reset_l2_decay_lambda=0.05,
    plasticity_reset_l2_decay_threshold=0.9,
    plasticity_reset_l2_decay_temperature=0.05,
    plasticity_reset_max_ci=100.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v9_layer_l2_decay",
    plasticity_raw_importance_log=True,
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
print("DONE_DENSE_LR_UNSCALED_PLASTICITY_RESET_V9_LAYER_L2_DECAY", flush=True)
