"""v21: v19's own config (CiRenorm StableRegion + WeightRenorm, seed=1001,
qkvo_norm_enable=True, max_abs_grad=8.0 default, no centering) PLUS the
homeostatic energy gating mechanism (sili/energy.py's EnergyDynamics),
applied to the "embed_input"/"state" regions.

Direct motivation: v19/v19b/v19c (WeightRenorm confirmed to keep weight
bounded throughout, unlike v18b's own runaway) still showed 2 of 3 runs
getting PERMANENTLY STUCK right at a curriculum transition -- loss
spikes sharply within 50-250 steps of a level-up, then LOCKS FLAT at an
elevated plateau for the remaining 90k+ steps, never recovering (v19b:
loss locks ~3.8-4.7 after step 9201; v19c: loss locks ~4.4-4.7 after
step 7500). This is a genuine stable-but-bad attractor, not gradual
divergence -- checked and ruled out Q/K spectral norm as a distinguishing
factor at the fork point (nearly identical, ~14.6-14.8, in ALL THREE
runs regardless of outcome).

Direct instruction: "The energy system is exactly the thing that is
meant to escape stable attractors. The forced firing, energy
accumulation, all of it is meant to add a noise floor, escape lock step
behavior with continual symmetry breaking, etc., and the original repo
for it demonstrated escaping zero init and pathological attractor init
RNNs... we'd need to be careful to tune it so that it regularly settles
in the high quality solutions instead of the low quality ones."

Why the prior negative result (wide288 sparse arm, memory:
wide288_energy_rl_negative_result) does NOT predict failure here: that
run used nucleus top-k SPARSE selection, where which neurons fire
naturally rotates every step from content alone -- "hasn't fired in N
steps" was never a real pathology signal there, so energy's noise only
corrupted an otherwise-fine gradient. This entire v13-v21 family is
dense=True -- no organic rotation at all -- much closer to the
ORIGINAL zero-init/pathological-attractor-escape scenario energy was
built and validated against.

Tuning, direct instructions on each knob:
- wake_gate_steps=2500: safely past the longest HEALTHY post-transition
  recovery window observed (v19's own recovery took ~1000-1500 steps),
  but far short of the 90k-step stuck duration -- intervenes only once
  something looks genuinely locked, not during a normal dip.
- drive/activation_cost/precision/density/p: kept at this project's own
  existing validated values (launch_sparse_energy_remote.py's
  width=288 sparse-arm config), EXCEPT drive itself, which the energy.py
  docstring states must be calibrated per-region as
  drive=activation_cost*E[|h|] -- measured DIRECTLY at this toy scale
  (embed_width=36) rather than assumed from width=288's own number:
  state E[|h|]~=0.228 (drive=0.0114, matches width=288's calibration
  almost exactly), embed_input E[|h|]~=0.0497 (drive=0.0025, close to
  but not identical to width=288's 0.00294) -- RMSNorm keeps activation
  scale roughly width-independent, confirmed empirically, not assumed.
- stagger_wake_init=True: matches the existing validated config."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

ENERGY_KWARGS = {
    "state": {
        "drive": 0.0114,
        "activation_cost": 0.05,
        "precision": 0.01,
        "density": 0.5,
        "p": 0.9,
        "wake_gate_steps": 2500,
        "stagger_wake_init": True,
    },
    "embed_input": {
        "drive": 0.0025,
        "activation_cost": 0.05,
        "precision": 0.01,
        "density": 0.5,
        "p": 0.9,
        "wake_gate_steps": 2500,
        "stagger_wake_init": True,
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
    "# DENSE-UNSCALED + QKVO_NORM + CI_WEIGHT_RENORM + ENERGY v21 (qkvo_norm_enable=True, "
    "seed=1001 -- same harder seed as v13b/v14/.../v19 -- ci_renorm_enable=stable_region AND "
    "weight_renorm_enable=stable_region (v19's own config) PLUS use_energy=True on "
    "state/embed_input, wake_gate_steps=2500, drive calibrated from this toy scale's own "
    "measured E[|h|] (state=0.0114, embed_input=0.0025), max_abs_grad=8.0 still active by "
    "default via NOCAPS_KWARGS_FP32, NO centering; plasticity_reset_enable=True with "
    "reset_fraction=0.0 is a deliberate no-op, kept ONLY to reuse the existing column-log "
    "capture pathway), K_START=2, precision=fp32 max_steps=100000 embed_width=36 "
    "k_first_target=3 NUM_CPUS=4 -- does homeostatic energy gating's forced-firing/noise-floor "
    "escape the stable-but-bad attractor that trapped v19b/v19c right after a curriculum "
    "transition; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v21_qkvo_norm_ci_weight_renorm_energy/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v21_qkvo_norm_ci_weight_renorm_energy",
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CI_WEIGHT_RENORM_ENERGY_V21", flush=True)
