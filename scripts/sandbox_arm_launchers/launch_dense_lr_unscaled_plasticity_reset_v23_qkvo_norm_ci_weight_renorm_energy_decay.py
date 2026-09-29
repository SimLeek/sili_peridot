"""v23: v22's config (energy, no wake_gate_steps) plus EnergyDynamics'
new `decay` mean-reversion term -- fixes the random-walk-drift bug found
analyzing v21/v22 (energy concentrated nowhere near 0 even under
"neutral" calibration; see sili__new commit 5a3c2da).

drive/decay derived from a real ~30000-step no-energy baseline probe
(same seed=1001/ci_renorm/weight_renorm/plasticity_reset stack as
v19/v19b), which produced a genuine 18746-step stall at vocab=64/k=1
that it eventually escaped on its own -- a real pre-energy
bad-stable-attractor episode, not a synthetic one. JOURNAL.md has the
full numbers. embed_input showed a real, clean drop in mean|h| during
that stall (0.0844 healthy -> 0.0512 stuck); decay chosen so the
stuck-regime fixed point comfortably clears the fire threshold
(~3.3 > 2.0) while staying far below the healthy-regime crossing rate.
state showed no clean stall-vs-healthy separation in that same episode
-- its decay is chosen only from the concentration/bug-fix rationale
(healthy-regime variance), not tuned for escape."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

ENERGY_KWARGS = {
    "state": {
        "drive": 0.01702,
        "activation_cost": 0.05,
        "precision": 0.01,
        "density": 0.5,
        "p": 0.9,
        "decay": 0.002,
    },
    "embed_input": {
        "drive": 0.00422,
        "activation_cost": 0.05,
        "precision": 0.01,
        "density": 0.5,
        "p": 0.9,
        "decay": 0.0005,
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
    "# DENSE-UNSCALED + QKVO_NORM + CI_WEIGHT_RENORM + ENERGY_DECAY v23 (qkvo_norm_enable=True, "
    "seed=1001 -- same harder seed as v13b/v14/.../v22 -- ci_renorm_enable=stable_region AND "
    "weight_renorm_enable=stable_region (v19's own config) PLUS use_energy=True on "
    "state/embed_input WITHOUT wake_gate_steps (v22's config) PLUS decay=0.002/0.0005 "
    "(mean-reversion, sili__new commit 5a3c2da) -- drive/decay both derived from a real "
    "no-energy baseline probe's own 18746-step vocab=64/k=1 stall (JOURNAL.md has the full "
    "numbers), max_abs_grad=8.0 still active by default via NOCAPS_KWARGS_FP32, NO centering; "
    "plasticity_reset_enable=True with reset_fraction=0.0 is a deliberate no-op, kept ONLY to "
    "reuse the existing column-log capture pathway), K_START=2, precision=fp32 max_steps=100000 "
    "embed_width=36 k_first_target=3 NUM_CPUS=4 -- does mean-reverting energy dynamics, "
    "calibrated from a real observed stall episode, escape the bad-stable-attractor regime "
    "better than v22's plain (non-mean-reverting) energy; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v23_qkvo_norm_ci_weight_renorm_energy_decay/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v23_qkvo_norm_ci_weight_renorm_energy_decay",
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CI_WEIGHT_RENORM_ENERGY_DECAY_V23", flush=True)
