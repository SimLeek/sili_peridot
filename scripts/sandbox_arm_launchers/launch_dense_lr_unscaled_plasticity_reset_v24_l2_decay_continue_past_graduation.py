"""v24: v19's config (ci_renorm+weight_renorm+plasticity_reset,
seed=1001) plus apply_amortized_l2_decay (a plain, closed-loop weight
decay toward target_rms=1/sqrt(fan_in) -- NEVER before combined with
CiRenorm/WeightRenorm, or tested at all beyond v9's much older,
unrelated plasticity_reset-internal l2decay arm). Also sets
continue_past_graduation=True: trains the full 100000 steps regardless
of whether/when it reaches vocab=126/k=4, instead of stopping at first
graduation (v19's own behavior) -- lets k keep climbing past 4 and
keeps recording plasticity_column_snapshots the whole way, for a fair,
same-duration comparison against the stalled runs (v19b/v19c/H/G/GH)
that already ran the full 100k by virtue of never graduating.

Mathematical check before building (JOURNAL.md has the full
derivation): l2_decay's own closed-loop target (1/sqrt(fan_in)) and
WeightRenorm's empirically-derived target_std for q/k/v_proj (0.059)
are the SAME value (1/sqrt(288)=0.0589) -- both mechanisms chase the
identical fixed point and both preserve rank order, so they should
compose without fighting over WHERE weights converge. The one thing
math alone couldn't resolve is whether l2_decay's own rms-based
self-tuning gets confounded by WeightRenorm's concurrent rescaling --
this run is the empirical check."""

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
    print(
        f"  step={step:>7}  vocab={vocab_size:>4}  k={k:>3}  loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}", flush=True
    )


print(
    "# v24: L2_DECAY + CI/WEIGHT_RENORM + CONTINUE_PAST_GRADUATION (qkvo_norm_enable=True, "
    "seed=1001, ci_renorm_enable=stable_region AND weight_renorm_enable=stable_region (v19's own "
    "config) PLUS l2_decay_chunk_size=800/l2_decay_adaptation_rate=0.3 (never before tested "
    "alongside CiRenorm/WeightRenorm) PLUS continue_past_graduation=True (trains full 100000 "
    "steps regardless of graduation, k keeps climbing past 4 if reached), K_START=2, "
    "precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- does plain "
    "closed-loop weight decay compose with or interfere with CiRenorm/WeightRenorm, and how far "
    "past graduation can the model push if training continues; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v24_l2_decay_continue_past_graduation/",
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
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v24_l2_decay_continue_past_graduation",
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
print("DONE_V24_L2_DECAY_CONTINUE_PAST_GRADUATION", flush=True)
