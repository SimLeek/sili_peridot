"""v27: same base config as v25 (v19c's exact config, seed=1000 --
the cleanest reliably-stuck run on record) but with sleep_gate_layers=
("q_proj", "k_proj") instead of a blind periodic schedule. Tests the
2026-10-04 "freeze-gated sleep" hypothesis: v25/v26's own data showed
col_importance going completely static (zero delta between amortized
cycles) is a real, purely intrinsic signal that differed in TIMING
between v19c (froze at step 13273, permanently stuck) and v24 (froze
at step 35848, only AFTER already reaching k=5) -- but blind periodic
sleep delayed this freeze almost identically in both v25 (helped) and
v26 (hurt), suggesting the periodic schedule was disrupting v24-style
runs mid-climb while they were never at risk of freezing yet. Gating
sleep on the freeze itself, rather than a schedule, should leave an
uninterrupted healthy climb alone (sleep never turns on) while still
rescuing a genuinely dead layer the moment it goes static -- and turn
back off automatically the moment col_importance starts moving again
(no separate wake-up logic needed, same reactive check both ways).

See JOURNAL.md's 2026-10-04 "freeze-gated sleep" entry for the full
v25/v26 analysis this is built on."""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

SLEEP_ENERGY_KWARGS = {
    name: {
        "drive": 0.0775,
        "activation_cost": 0.05,
        "decay": 0.0075,
        "precision": 0.01,
        "density": 0.3,
        "p": 0.6,
        "exploration": 0.01,
        "reactivity": 0.01,
        "fire_wake_gradient": 1.0,
        "wake_seed": 7,
    }
    for name in ["input_proj", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]
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
    "# v27: FREEZE-GATED SLEEP ON v19c's KNOWN-STUCK CONFIG (seed=1000, ci_renorm_enable=stable_region "
    "AND weight_renorm_enable=stable_region PLUS sleep_enable=True gated on sleep_gate_layers=(q_proj,k_proj) "
    "col_importance freeze, decay=0.0075 rotation rate on all 6 wide layers, fire_wake_gradient=1.0) PLUS "
    "continue_past_graduation=True, K_START=2, precision=fp32 max_steps=100000 embed_width=36 "
    "k_first_target=3 NUM_CPUS=4 -- does gating sleep on q/k's own intrinsic freeze (vs v25's blind "
    "every-5000-steps schedule) still rescue v19c's own stall; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v27_freeze_gated_sleep_on_stuck_v19c/",
    flush=True,
)
r = m.train_curriculum(
    "fp32",
    100000,
    1000,
    0.015,
    16,
    10,
    embed_width=36,
    wrong_streak_threshold=100000000,
    k_first_target=3,
    continue_past_graduation=True,
    sleep_enable=True,
    sleep_gate_layers=("q_proj", "k_proj"),
    sleep_energy_kwargs=SLEEP_ENERGY_KWARGS,
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v27_freeze_gated_sleep_on_stuck_v19c",
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
print("DONE_V27_FREEZE_GATED_SLEEP_ON_STUCK_V19C", flush=True)
