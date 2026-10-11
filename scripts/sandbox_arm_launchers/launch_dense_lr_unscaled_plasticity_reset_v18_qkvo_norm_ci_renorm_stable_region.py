"""v18: real-engine validation of CiRenorm's StableRegion arm, companion
to v17 (TrustRatio) -- same v14 base config (qkvo_norm_enable=True +
max_abs_grad=8.0 default), deliberately WITHOUT centering, same harder
seed (1001).

StableRegion (this arm): affine (z-score) renormalize each layer's real
per-synapse ci toward an EMPIRICALLY-DERIVED per-layer (mean, std)
target, pulled directly from v13/v15's own real graduated-run data
(medians over each run's step>=3000 window): q_proj (0.09, 0.18),
k_proj (0.10, 0.13), v_proj (0.40, 0.35), input_proj (1.05, 1.47),
o_proj (0.20, 0.67), lm_head (0.09, 0.03). Direct instruction, after the
design was refined to control variance too, not just mean: "Ah, stable
average and variance too, as a run, in addition to just average."

See v17's own docstring and sili__new's
docs/research/delta_csr_types.rst:synapse_policy.ci_renorm for the full
shared motivation ("the average ci of a layer is literally inversely
proportional to the average plasticity of that layer... importance was
not a misnomer" -- rescale, don't reset) and the 4 real bugs found and
fixed while building this mechanism."""

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
    "# DENSE-UNSCALED + QKVO_NORM + CI_RENORM_STABLE_REGION v18 (qkvo_norm_enable=True, "
    "seed=1001 -- same harder seed as v13b/v14/v15/v16/v17 -- ci_renorm_enable=stable_region, "
    "max_abs_grad=8.0 still active by default via NOCAPS_KWARGS_FP32, NO centering (isolating "
    "CiRenorm's own effect); plasticity_reset_enable=True with reset_fraction=0.0 is a "
    "deliberate no-op, kept ONLY to reuse the existing column-log capture pathway), K_START=2, "
    "precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- direct "
    "3-way comparison against v14 (no CiRenorm) and v17 (TrustRatio arm): does renormalizing "
    "toward empirically-derived per-layer (mean,std) targets fix the v/o/input_proj saturation; "
    "logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v18_qkvo_norm_ci_renorm_stable_region/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v18_qkvo_norm_ci_renorm_stable_region",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
    ci_renorm_enable="stable_region",
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CI_RENORM_STABLE_REGION_V18", flush=True)
