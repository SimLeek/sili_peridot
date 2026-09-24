"""v13b: replication of v13 (QKV-Norm + O-Norm) under a DIFFERENT seed,
to check the GRADUATED result (final_vocab=126, final_k=4 at step
13342, faster than v12) wasn't a lucky seed -- same discipline as
v12b, which is exactly what caught v12's own fragility (v12 seed=1000
GRADUATED, v12b seed=1001 stalled at vocab=64/k=2 for ~90k steps).

Identical config to v13 (qkvo_norm_enable=True, everything else
unchanged) EXCEPT seed=1001 (v13 used 1000, same seed v12b used for
its own v12 replication -- keeps the seed axis comparable across both
mechanisms). Writes to a NEW plasticity_column_log_dir and log file --
v13's own recordings at
logs/plasticity_column_snapshots/dense_lr_unscaled_v13_qkvo_norm/ and
logs/dense_lr_unscaled_plasticity_reset_v13_qkvo_norm_local.log are
left untouched."""

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
    qk_spectral_norm=None,
):
    loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
    acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
    tag = f"  [{event}]" if event else ""
    sps_s = f"  steps/sec={steps_per_sec:.1f}" if steps_per_sec is not None else ""
    specnorm_s = (
        f"  qk_specnorm[q={qk_spectral_norm['q_proj']:.2f} k={qk_spectral_norm['k_proj']:.2f}]"
        if qk_spectral_norm
        else ""
    )
    print(
        f"  step={step:>7}  vocab={vocab_size:>4}  k={k:>3}  loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{specnorm_s}",
        flush=True,
    )


print(
    "# DENSE-UNSCALED + QKVO_NORM v13b REPEAT (qkvo_norm_enable=True, seed=1001 -- same config as "
    "v13 but a different seed, to validate the GRADUATED result wasn't a lucky draw, same "
    "discipline as v12b), K_START=2, precision=fp32 max_steps=100000 embed_width=36 "
    "k_first_target=3 NUM_CPUS=4 -- v13's own recordings are untouched; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v13b_qkvo_norm_repeat/",
    flush=True,
)
r = m.train_curriculum(
    "fp32",
    100000,
    1001,
    0.015,
    16,
    10,
    embed_width=36,
    wrong_streak_threshold=100000000,
    k_first_target=3,
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v13b_qkvo_norm_repeat",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_V13B_REPEAT", flush=True)
