"""Arm G: v19's config (ci_renorm+weight_renorm+plasticity_reset,
seed=1001) plus BACKWARD/dy grad-selection by GradSelectionEnergy state
instead of Arm C's blind sine-wave time-gate. One tracker per wide
layer (input_proj/q_proj/k_proj/v_proj/o_proj), fed that layer's own
forward output every call; each step selects the density-fraction of
neurons with the HIGHEST (quietest) energy for dy_gate_mask.

drive/decay per layer derived from a real 3000-step no-energy probe's
measured mean|output| (JOURNAL.md has the full numbers): drive =
activation_cost(0.05) * mean|output|, decay = drive/5 (comfortable
margin above the drive>2*decay coverage-guarantee minimum). density=0.4
matches the middle of Arm C's own tested density range for a
same-sparsity comparison. Tests the hypothesis that measured
under-activity beats blind time-division for grad-selection fairness
(see docs/research/train_mqar_curriculum.rst's 2026-09-29 disambiguation
note)."""

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
    "# ARM G: ENERGY-BASED GRAD SELECTION (qkvo_norm_enable=True, seed=1001, "
    "ci_renorm_enable=stable_region AND weight_renorm_enable=stable_region (v19's own config) "
    "PLUS dy_energy_gate_density=0.4 with per-layer GradSelectionEnergy drive/decay derived from "
    "a real 3000-step no-energy magnitude probe -- top-k BACKWARD grad selection by per-neuron "
    "under-activity instead of Arm C's blind time-division, K_START=2, precision=fp32 "
    "max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- does measured under-activity "
    "beat blind time-division for grad-sparsity fairness; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_arm_g_energy_grad_selection/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_arm_g_energy_grad_selection",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
    ci_renorm_enable="stable_region",
    weight_renorm_enable="stable_region",
    dy_energy_gate_density=0.4,
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
print("DONE_ARM_G_ENERGY_GRAD_SELECTION", flush=True)
