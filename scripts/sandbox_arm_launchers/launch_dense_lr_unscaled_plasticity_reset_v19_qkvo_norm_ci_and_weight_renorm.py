"""v19: real-engine validation of CiRenorm StableRegion PLUS WeightRenorm
together, on the SAME v14/v18 base config (qkvo_norm_enable=True +
max_abs_grad=8.0 default), same harder seed (1001), still WITHOUT
centering.

Direct motivation: v18 (CiRenorm StableRegion alone) GRADUATED, but its
own same-seed repeat v18b stalled at (64,3). Deep-diving v18b's own
recorded raw_weight/raw_importance matrices found CiRenorm was
correctly holding ci near the StableRegion target throughout v18b's
entire run -- the mechanism worked exactly as designed -- but
v_proj/o_proj/input_proj's WEIGHT itself, left completely
unconstrained, ran away in a genuine positive-feedback loop (weight
norm climbing smoothly from ~17 to 360 over 95k steps). Mechanism:
pinning ci to a fixed target while the layer's true gradient magnitude
legitimately grows keeps the effective step large, letting weight grow
faster than warranted, which drives larger downstream gradients the
fixed ci target still won't track.

Direct instruction: "I guess we have to do a weight renorm in addition
to the ci renorm. Exact same method would probably be fine." This run
tests whether constraining weight ALONGSIDE ci (both toward
empirically-derived per-layer targets from v18/v18c's own healthy
early-training data) prevents the runaway that doomed v18b specifically,
without breaking what already worked for v18/v18c. See sili__new's
docs/research/delta_csr_types.rst:synapse_policy.weight_renorm for the
full mechanism and its own regression tests."""

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
    "# DENSE-UNSCALED + QKVO_NORM + CI_AND_WEIGHT_RENORM v19 (qkvo_norm_enable=True, seed=1001 "
    "-- same harder seed as v13b/v14/v15/v16/v17/v18 -- ci_renorm_enable=stable_region AND "
    "weight_renorm_enable=stable_region together, max_abs_grad=8.0 still active by default via "
    "NOCAPS_KWARGS_FP32, NO centering; plasticity_reset_enable=True with reset_fraction=0.0 is a "
    "deliberate no-op, kept ONLY to reuse the existing column-log capture pathway), K_START=2, "
    "precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- does "
    "constraining weight alongside ci prevent the weight-runaway positive-feedback loop that "
    "doomed v18b specifically; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v19_qkvo_norm_ci_and_weight_renorm/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v19_qkvo_norm_ci_and_weight_renorm",
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CI_AND_WEIGHT_RENORM_V19", flush=True)
