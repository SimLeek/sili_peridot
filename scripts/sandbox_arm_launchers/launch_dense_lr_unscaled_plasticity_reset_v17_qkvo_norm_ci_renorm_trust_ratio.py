"""v17: real-engine validation of CiRenorm's TrustRatio arm, on v14's
own base config (qkvo_norm_enable=True + max_abs_grad=8.0 default),
deliberately WITHOUT centering -- isolates CiRenorm's own contribution
against the same baseline v15/v16 were compared to, rather than
confounding it with centering's already-inconclusive effect ("do
science correctly": enumerate every difference before an A/B).

Motivated by a deep statistical pass across v13/v13b/v14/v15/v15b/v15c/
v16 (84,121 per-layer-per-cycle snapshots): graduated runs keep average
ci low in v_proj/o_proj/input_proj; every stalled run's average ci
climbs toward max_ci=100 and NEVER recovers spontaneously within 100k
steps. Direct instruction: "the average ci of a layer is literally
inversely proportional to the average plasticity of that layer...
importance was not a misnomer" -- rules out resetting/pruning high-ci
synapses (genuinely important) in favor of RESCALING the population
back into a healthy range while preserving relative ranking.

TrustRatio (this arm): LAMB-inspired (You et al. 2019) multiplicative
rescale toward a target ci implied by each column's own current weight
norm -- self-calibrating, no manual per-layer targets needed. v18
(companion launcher) is StableRegion (empirically-derived per-layer
targets from real graduated-run data).

Same harder seed (1001) as v13b/v14/v15/v16, so this is a direct A/B
against v14 (no CiRenorm) and v15/v16 (centering instead of CiRenorm).
See sili__new's
docs/research/delta_csr_types.rst:synapse_policy.ci_renorm for the
full mechanism derivation, including the 4 real bugs found and fixed
while building it."""

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
    "# DENSE-UNSCALED + QKVO_NORM + CI_RENORM_TRUST_RATIO v17 (qkvo_norm_enable=True, seed=1001 "
    "-- same harder seed as v13b/v14/v15/v16 -- ci_renorm_enable=trust_ratio, max_abs_grad=8.0 "
    "still active by default via NOCAPS_KWARGS_FP32, NO centering (isolating CiRenorm's own "
    "effect against v14's own baseline); plasticity_reset_enable=True with reset_fraction=0.0 is "
    "a deliberate no-op, kept ONLY to reuse the existing column-log capture pathway), K_START=2, "
    "precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- does rescaling "
    "(not resetting) a layer's ci back toward a weight-norm-implied target fix the v/o/input_proj "
    "saturation that regressed v14's curriculum progress; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v17_qkvo_norm_ci_renorm_trust_ratio/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v17_qkvo_norm_ci_renorm_trust_ratio",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
    ci_renorm_enable="trust_ratio",
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CI_RENORM_TRUST_RATIO_V17", flush=True)
