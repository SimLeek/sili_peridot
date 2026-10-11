"""v16: real-engine validation of AdaBelief-style ROW+COLUMN centering,
companion to v15 (column-only) -- same harder seed (1001), same v14 base
config (qkvo_norm_enable=True + max_abs_grad=8.0 still active by default
via NOCAPS_KWARGS_FP32, complementary to centering not replaced by it).

Direct design decision, after v14 regressed curriculum progress vs v13b:
row+column ("I'd say rowxcolumn since that includes per column and we may
be able to use some of the low rank stuff to implement it which may allow
us to expand it if needed") was chosen over column-only alone specifically
because it generalizes toward this project's existing AQRS rank-N
scale-tracking pattern (the same row-times-column factorization already
used for value_scale/output_scale), leaving room to expand to a real
low-rank factorization later. "Agreed on the proper design, eventually
leading to new mqar versions with column-only vs rowxcolumn." -- this is
that row+column arm; v15 (companion launcher) is column-only.

centering_row_enable=True AND centering_col_enable=True. Built on the
ADDITIVE m[i,j]=m_row[i]+m_col[j] design (not Adafactor's multiplicative
row/column reconstruction, whose optimality only holds for a nonnegative
target like g^2, not signed g) -- and on the residual-fit fix for a real
double-counting bug found via an end-to-end smoke test (each axis fitting
raw g independently made m_row+m_col overshoot toward 2g instead of g when
row/column effects are confounded; fixed by having each axis fit the
residual against the OTHER axis's OLD pre-touch value instead). See
sili__new's
docs/research/delta_csr_types.rst:synapse_policy.adabelief_centering for
the full derivation, the bug, the fix, and the verified numbers (buggy:
mean importance=16.8844, barely different from no-centering's 16.9364;
fixed: 0.4407, on par with column-only's 0.5327)."""

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
    "# DENSE-UNSCALED + QKVO_NORM + CENTERING_ROWCOL v16 (qkvo_norm_enable=True, seed=1001 -- same "
    "harder seed as v13b/v14/v15 -- centering_row_enable=True AND centering_col_enable=True, "
    "max_abs_grad=8.0 still active by default via NOCAPS_KWARGS_FP32 -- complementary, not "
    "replaced; plasticity_reset_enable=True with reset_fraction=0.0 is a deliberate no-op, kept "
    "ONLY to reuse the existing column-log capture pathway), K_START=2, precision=fp32 "
    "max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- direct 3-way comparison "
    "against v14 (grad-clip-only) and v15 (column-only centering): does the row+column additive "
    "baseline do better, worse, or the same as column-only; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v16_qkvo_norm_centering_rowcol/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v16_qkvo_norm_centering_rowcol",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
    centering_row_enable=True,
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CENTERING_ROWCOL_V16", flush=True)
