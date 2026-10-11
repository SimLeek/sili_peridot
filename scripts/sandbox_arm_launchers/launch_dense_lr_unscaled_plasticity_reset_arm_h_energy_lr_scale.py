"""Arm H: v19's config (ci_renorm+weight_renorm+plasticity_reset,
seed=1001) plus a per-output-neuron POST-CLIP learning-rate multiplier
(energy_lr_scale, sili__new commit 88e6742) standalone -- no Arm G
selection, dy_sparsity_p=1.0 fallback routes through backward_sparse
so the multiplier has a path to apply.

Unlike an additive per-neuron loss term (found mathematically inert:
max_abs_delta clips the RMSprop-normalized update before eff_lr
applies, and real gradients run 4-5+ orders of magnitude past that
clip's saturation point), this multiplies the ALREADY-clipped,
sign-decided step -- proportional leverage, not a counterweight. Same
per-layer drive/decay calibration as Arm G (real 3000-step no-energy
magnitude probe, JOURNAL.md has the numbers), kappa=0.5 (default,
bounded automatically by energy's own decay-clamped range)."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

ENERGY_GATE_KWARGS = {
    "input_proj": {"drive": 0.00444, "activation_cost": 0.05, "decay": 0.000889},
    "q_proj": {"drive": 0.01977, "activation_cost": 0.05, "decay": 0.003954},
    "k_proj": {"drive": 0.02559, "activation_cost": 0.05, "decay": 0.005117},
    "v_proj": {"drive": 0.02526, "activation_cost": 0.05, "decay": 0.005052},
    "o_proj": {"drive": 0.01423, "activation_cost": 0.05, "decay": 0.002846},
}


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
    print(
        f"  step={step:>7}  vocab={vocab_size:>4}  k={k:>3}  loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}", flush=True
    )


print(
    "# ARM H: ENERGY LR-SCALE STANDALONE (qkvo_norm_enable=True, seed=1001, "
    "ci_renorm_enable=stable_region AND weight_renorm_enable=stable_region (v19's own config) "
    "PLUS dy_energy_lr_scale_enable=True (kappa=0.5 default) with per-layer GradSelectionEnergy "
    "drive/decay derived from a real 3000-step no-energy magnitude probe (same as Arm G) -- "
    "standalone post-clip lr multiplier, no Arm G selection, falls back to dy_sparsity_p=1.0, "
    "K_START=2, precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- "
    "does a proportional post-clip multiplier (not a selection mechanism) help escape a stall; "
    "logging to logs/plasticity_column_snapshots/dense_lr_unscaled_arm_h_energy_lr_scale/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_arm_h_energy_lr_scale",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
    ci_renorm_enable="stable_region",
    weight_renorm_enable="stable_region",
    dy_energy_lr_scale_enable=True,
    dy_energy_gate_kwargs=ENERGY_GATE_KWARGS,
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
print("DONE_ARM_H_ENERGY_LR_SCALE", flush=True)
