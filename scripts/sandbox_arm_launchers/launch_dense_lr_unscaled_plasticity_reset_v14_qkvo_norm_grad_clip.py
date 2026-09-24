"""v14: real-engine validation of qkvo_norm_enable + max_abs_grad clipping
together, on the SAME harder seed (1001) that exposed v13b's recurring
v_proj/o_proj/input_proj saturation. Direct instruction: "Let's set a
default, thread it through, and then run a real mqar run with it, v14.
After that, we can get back to the adabelief-style work."

max_abs_grad is NOT a launcher-level flag here -- it's now baked into
NOCAPS_KWARGS_FP32 (train_mqar_curriculum.py's module-level fp32 synapse
kwargs, alongside the pre-existing max_abs_delta/max_ci caps) as a real,
data-derived default (8.0, from v13's own recorded gradient-scale
distribution), active automatically for every fp32 training run,
including this one -- see sili__new's
docs/research/delta_csr_types.rst:synapse_policy.max_abs_grad_clip for
the full derivation and the TDD proof (a g=50 spike poisons ci by
2155x unclipped, only 1.86x clipped).

Everything else identical to v13b (qkvo_norm_enable=True, seed=1001,
same curriculum/embed_width/k_first_target) so the comparison is a
direct one: does grad clipping additionally fix the v/o/input_proj
saturation that recurred in v13b DESPITE v_norm_ln/o_norm_ln being
active? Own separate recordings -- v13/v13b's own logs are untouched."""

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
    "# DENSE-UNSCALED + QKVO_NORM + MAX_ABS_GRAD v14 (qkvo_norm_enable=True, seed=1001 -- same "
    "harder seed as v13b -- max_abs_grad=8.0 now active by default via NOCAPS_KWARGS_FP32, not a "
    "launcher flag -- plasticity_reset_enable=True with reset_fraction=0.0 is a deliberate no-op, "
    "kept ONLY to reuse the existing column-log capture pathway), K_START=2, precision=fp32 "
    "max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- testing whether grad clipping "
    "fixes the v/o/input_proj saturation that recurred in v13b; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v14_qkvo_norm_grad_clip/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v14_qkvo_norm_grad_clip",
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_GRAD_CLIP_V14", flush=True)
