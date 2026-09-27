"""v15b: REPLICATION run of v15 (identical config, identical seed=1001) --
direct instruction after v15's own surprising result (GRADUATED at step
15063, vocab=126/k=4 -- far better than v14's stalled (126,2) and v16's
stalled (126,3)): "Hmm. I suppose we should launch v15 again to verify."

Same rationale as v12/v12b and v13/v13b's own precedent of never trusting
a single-seed result -- but this repeat deliberately keeps the SAME
seed=1001 rather than switching seeds, since this project has documented
real run-to-run threading nondeterminism at NUM_CPUS>1 (see
project_backward_sparse_threading_nondeterminism memory: "C++ update
gives different value_scale across repeated identical calls at
num_cpus=2") -- an independent question from seed sensitivity, and worth
checking first given v15's graduation was unusually fast/clean compared
to every other arm tried in this whole investigation.

Own separate log + plasticity_column_snapshots directory (v15b_repeat,
not v15) so this run's data never mixes with v15's own recording."""

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
    "# DENSE-UNSCALED + QKVO_NORM + CENTERING_COL v15b REPEAT (identical config to v15: "
    "qkvo_norm_enable=True, seed=1001, centering_col_enable=True, max_abs_grad=8.0 still active by "
    "default via NOCAPS_KWARGS_FP32; plasticity_reset_enable=True with reset_fraction=0.0 is a "
    "deliberate no-op, kept ONLY to reuse the existing column-log capture pathway), K_START=2, "
    "precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- verifying v15's "
    "own real result (GRADUATED at step 15063, vocab=126/k=4) reproduces under this project's known "
    "multi-threaded backward nondeterminism; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v15b_qkvo_norm_centering_col_repeat/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v15b_qkvo_norm_centering_col_repeat",
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_CENTERING_COL_V15B_REPEAT", flush=True)
