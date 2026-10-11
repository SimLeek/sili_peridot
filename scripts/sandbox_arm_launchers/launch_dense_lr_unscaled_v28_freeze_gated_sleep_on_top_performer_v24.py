"""v28: v26's companion with sleep_gate_layers instead of a blind
schedule (see v27's own docstring for the full rationale and
JOURNAL.md's 2026-10-04 "freeze-gated sleep" entry). v24's exact
config (seed=1001, reached k=5 unmodified) -- the critical test is
whether gating sleep on q_proj/k_proj's own col_importance freeze
(rather than every 5000 steps regardless of state) leaves v24's own
fast, uninterrupted k2->k5 climb alone, since that climb's natural
freeze point (step 35848 in the unmodified run) is well AFTER k=5 is
already reached -- v26's blind schedule fired its first window at
step 5000, mid-climb, and that run never got past k=2."""

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
    "# v28: FREEZE-GATED SLEEP ON v24's TOP-PERFORMER CONFIG (seed=1001, l2_decay_chunk_size=800/ "
    "l2_decay_adaptation_rate=0.3, ci_renorm_enable=stable_region AND weight_renorm_enable=stable_region "
    "PLUS sleep_enable=True gated on sleep_gate_layers=(q_proj,k_proj) col_importance freeze instead of "
    "v26's blind every-5000-steps schedule) PLUS continue_past_graduation=True, K_START=2, precision=fp32 "
    "max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- does leaving v24's own uninterrupted "
    "climb alone (sleep only engages once q/k actually freeze) let it reach k=5 like the unmodified run, "
    "unlike v26's disrupted k=2 stall; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v28_freeze_gated_sleep_on_top_performer_v24/",
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
    continue_past_graduation=True,
    l2_decay_chunk_size=800,
    l2_decay_adaptation_rate=0.3,
    sleep_enable=True,
    sleep_gate_layers=("q_proj", "k_proj"),
    sleep_energy_kwargs=SLEEP_ENERGY_KWARGS,
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v28_freeze_gated_sleep_on_top_performer_v24",
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
print("DONE_V28_FREEZE_GATED_SLEEP_ON_TOP_PERFORMER_V24", flush=True)
