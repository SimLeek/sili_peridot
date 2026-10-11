"""v11: real-engine validation of qk_l1_sparsity_coef -- an L1
output-sparsity penalty (this project's own prior cheap proxy for
spectral normalization, project_hybrid_precision_plan.md) scoped ONLY
to q_proj/k_proj, independent of the existing layer-wide
l1_sparsity_coef.

Direct instruction, after v10 (L2 Init) showed q_proj/k_proj uniquely
collapsing to near-zero deviation std while staying saturated-high in
importance (a Q/K-specific homogenization not seen in v_proj/o_proj/
lm_head): "spectral normalization takes a ton of calculation, but we
used an L1 norm or something similar to give a similar effect... If you
can get a quick additional loss or other low-calculation method to move
the layers towards the spectral-normalization, then I'd like that to be
implemented and tested as well."

coef=0.05 is the Goldilocks-zone value from the original (project-wide,
all-4-layer) L1-sparsity finding (0.03-0.07 sweet spot; <=0.02 too weak,
>=0.1 degrades). l1_sparsity_coef=0.0 so qk_l1_sparsity_coef is the
ONLY active penalty this run. plasticity_reset_enable=True with
reset_fraction=0.0 is a deliberate no-op -- enabled ONLY to reuse the
existing column-log npz-writing pathway, same trick as v10.
qk_spectral_norm_diag_log=True watches the actual quantity this
mechanism is supposed to bound."""

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
    "# DENSE-UNSCALED + QK_L1_SPARSITY v11 (qk_l1_sparsity_coef=0.05, l1_sparsity_coef=0.0 -- "
    "plasticity_reset_enable=True with reset_fraction=0.0 is a deliberate no-op, kept ONLY to "
    "reuse the existing column-log capture pathway), K_START=2, precision=fp32 max_steps=100000 "
    "seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- testing whether a cheap L1 proxy on "
    "Q/K alone fixes the deviation-std collapse found in v10; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v11_qk_l1_sparsity/",
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
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v11_qk_l1_sparsity",
    plasticity_raw_importance_log=True,
    qk_l1_sparsity_coef=0.05,
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
print("DONE_DENSE_LR_UNSCALED_QK_L1_SPARSITY_V11", flush=True)
