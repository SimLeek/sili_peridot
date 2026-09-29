"""v19b: REPLICATION run of v19 (identical config, identical seed=1001)
-- v19 GRADUATED at step 15026, matching v18's own graduation timing
almost exactly. Same rationale as v15/v15b and v18/v18b's own
precedent -- never trust a single-seed result, especially given v18's
own graduation did NOT reproduce under its own same-seed repeat
(v18b stalled at (64,3) with a weight-norm runaway WeightRenorm was
specifically built to fix).

Same config, same seed=1001 (checking this project's own documented
multi-threaded backward nondeterminism at NUM_CPUS>1 first, a separate
question from seed sensitivity -- see v19c, companion launcher, for the
new-seed check). Own separate log + plasticity_column_snapshots
directory so this run's data never mixes with v19's own recording."""

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
    "# DENSE-UNSCALED + QKVO_NORM + CI_AND_WEIGHT_RENORM v19b REPEAT (identical config to v19: "
    "qkvo_norm_enable=True, seed=1001, ci_renorm_enable=stable_region AND "
    "weight_renorm_enable=stable_region, max_abs_grad=8.0 still active by default via "
    "NOCAPS_KWARGS_FP32, NO centering; plasticity_reset_enable=True with reset_fraction=0.0 is a "
    "deliberate no-op, kept ONLY to reuse the existing column-log capture pathway), K_START=2, "
    "precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- verifying "
    "v19's own real result (GRADUATED at step 15026, vocab=126/k=4) reproduces under this "
    "project's known multi-threaded backward nondeterminism; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v19b_qkvo_norm_ci_and_weight_renorm_repeat/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v19b_qkvo_norm_ci_and_weight_renorm_repeat",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
    ci_renorm_enable="stable_region",
    weight_renorm_enable="stable_region",
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CI_AND_WEIGHT_RENORM_V19B_REPEAT", flush=True)
