"""v22: v19's config plus homeostatic energy gating WITHOUT
wake_gate_steps -- companion to v21 (wake_gate_steps=2500). Direct
correction: there shouldn't need a step-counter gate at all per the
original design; pure drive/activation_cost/noise dynamics should stay
quiescent in healthy regions and provide escape energy in a stuck
attractor if calibrated correctly.

Verified via a real ~20000-step probe (JOURNAL.md has the full
numbers): state/embed_input's real |h| genuinely differs between
healthy and stuck regimes, mostly-separably (embed_input in particular
goes nearly frozen once stuck). Under a drive calibrated neutral for
the healthy regime, the same drive gives real positive energy drift
once stuck -- fast enough to matter, slow enough not to misfire on
ordinary fluctuation.

drive = activation_cost * E[|h|] (activation_cost=0.05, matching this
project's existing validated config), E[|h|] measured directly under
the REAL v19 stack (not a simpler probe): state=0.342 (drive=0.0171),
embed_input=0.094 (drive=0.0047) -- corrects a calibration gap found
in v21's own drive (measured without CiRenorm/WeightRenorm active).
Smoke-tested before launch; may take a few attempts to tune right."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

ENERGY_KWARGS = {
    "state": {
        "drive": 0.0171,
        "activation_cost": 0.05,
        "precision": 0.01,
        "density": 0.5,
        "p": 0.9,
        # wake_gate_steps intentionally OMITTED (defaults to None) --
        # the whole point of this arm.
    },
    "embed_input": {
        "drive": 0.0047,
        "activation_cost": 0.05,
        "precision": 0.01,
        "density": 0.5,
        "p": 0.9,
    },
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
    "# DENSE-UNSCALED + QKVO_NORM + CI_WEIGHT_RENORM + ENERGY_NO_GATE v22 (qkvo_norm_enable=True, "
    "seed=1001 -- same harder seed as v13b/v14/.../v21 -- ci_renorm_enable=stable_region AND "
    "weight_renorm_enable=stable_region (v19's own config) PLUS use_energy=True on "
    "state/embed_input WITHOUT wake_gate_steps (companion to v21, which uses "
    "wake_gate_steps=2500), drive calibrated from a real 2000-step measurement under the ACTUAL "
    "v19 stack (state=0.0171, embed_input=0.0047), max_abs_grad=8.0 still active by default via "
    "NOCAPS_KWARGS_FP32, NO centering; plasticity_reset_enable=True with reset_fraction=0.0 is a "
    "deliberate no-op, kept ONLY to reuse the existing column-log capture pathway), K_START=2, "
    "precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- does the pure "
    "continuous energy mechanism (no artificial step-counter gate) naturally stay quiescent in "
    "healthy regions while escaping the stable-but-bad attractor that trapped v19b/v19c; logging "
    "to logs/plasticity_column_snapshots/dense_lr_unscaled_v22_qkvo_norm_ci_weight_renorm_energy_no_gate/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v22_qkvo_norm_ci_weight_renorm_energy_no_gate",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
    ci_renorm_enable="stable_region",
    weight_renorm_enable="stable_region",
    use_energy=True,
    energy_kwargs=ENERGY_KWARGS,
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CI_WEIGHT_RENORM_ENERGY_NO_GATE_V22", flush=True)
