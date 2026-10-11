"""v15: real-engine validation of AdaBelief-style COLUMN-ONLY centering on
top of v14's config (qkvo_norm_enable=True + max_abs_grad=8.0, still active
by default via NOCAPS_KWARGS_FP32 -- clipping and centering are
COMPLEMENTARY, not competing: clipping defends a one-off spike, centering
defends a SUSTAINED large gradient that clipping alone still lets ci settle
near max_abs_grad^2 and stay stuck at). Same harder seed (1001) as
v13b/v14, so this is a direct A/B against v14's own real result.

Direct instruction, after v14 regressed curriculum progress vs v13b despite
fixing the literal ci=100 saturation: "how would we actually get per column
clipping? Would per column adabelief work?" -- followed by the design
decision to build BOTH column-only and row+column arms and compare them
empirically: "Agreed on the proper design, eventually leading to new mqar
versions with column-only vs rowxcolumn." This is the column-only arm;
v16 (companion launcher) is row+column.

centering_col_enable=True only (centering_row_enable left at its False
default) -- see sili__new's
docs/research/delta_csr_types.rst:synapse_policy.adabelief_centering for
the full mechanism, the additive-vs-Adafactor-multiplicative design
decision, and the real double-counting bug found+fixed in the row+column
path (which column-only alone never triggers, since there's only one
axis)."""

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
    "# DENSE-UNSCALED + QKVO_NORM + CENTERING_COL v15 (qkvo_norm_enable=True, seed=1001 -- same "
    "harder seed as v13b/v14 -- centering_col_enable=True, max_abs_grad=8.0 still active by default "
    "via NOCAPS_KWARGS_FP32 -- complementary, not replaced; plasticity_reset_enable=True with "
    "reset_fraction=0.0 is a deliberate no-op, kept ONLY to reuse the existing column-log capture "
    "pathway), K_START=2, precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 "
    "-- direct A/B against v14 (grad-clip-only): does column-only AdaBelief centering fix the "
    "v/o/input_proj saturation / curriculum regression v14 showed; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v15_qkvo_norm_centering_col/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v15_qkvo_norm_centering_col",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
    centering_col_enable=True,
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CENTERING_COL_V15", flush=True)
