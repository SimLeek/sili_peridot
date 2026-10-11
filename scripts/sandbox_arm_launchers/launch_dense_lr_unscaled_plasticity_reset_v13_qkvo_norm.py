"""v13: real-engine validation of qkvo_norm_enable -- extends v12's
QK-Norm (Q/K only) to also normalize V (same pattern: normalize the
full tensor once, right after its own post-projection clip, before any
downstream gather) and o_proj's output (normalized right before the
residual add into the recurrent state / content residual, the closest
analog for a layer whose only consumer is a residual connection, not a
dot-product or weighted sum).

Direct instruction, after v12b (seed=1001) failed to replicate v12's
(seed=1000) GRADUATED result -- deviation-std check on v12b showed
q_proj/k_proj still avoiding v10's collapse, but v_proj/o_proj/
input_proj now FULLY saturated (col_importance exactly 100.00) instead,
a different failure mode than v10 ever showed. Asked whether this had
research precedent: yes -- Zhai et al. 2023 (already cited for QK-Norm)
found applying spectral-norm-reparametrization to ALL linear layers
(not just Q/K) performs comparably or better and is the simpler
choice; QKV-Norm (LayerNorm on Q, K, AND V) is now standard in Gemma 3/
OLMo 2/Qwen 3; and this project's own prior work found o_proj
specifically needed spectral-radius control in the older
ToyTileRecurrenceRealFP4 architecture, because it sits inside the
recurrent loop (eigenvalues compound across timesteps). "Please extend
the current qk_norm_enable into a qkvo_norm_enable, using the same
pattern that works for QK to fix the other layers, then launch that as
V13 and we'll see if that can perform robustly."

seed=1000 -- SAME seed as v12's own GRADUATED run, so this is a direct
apples-to-apples comparison against v12's exact trajectory, not a new
random draw. plasticity_reset_enable=True with reset_fraction=0.0 is a
deliberate no-op -- enabled ONLY to reuse the existing column-log
npz-writing pathway. qk_spectral_norm_diag_log=True still only reports
q_proj/k_proj (the diagnostic itself wasn't extended to v/o this pass);
the deviation-std verification against ALL SIX pools' recorded
raw_importance/raw_weight is done post-hoc from the column log, same
as v10/v11/v12/v12b."""

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
    "# DENSE-UNSCALED + QKVO_NORM v13 (qkvo_norm_enable=True -- RMSNorm on Q/K/V before the "
    "attention dot product/weighted sum AND on o_proj's output before the residual add, seed=1000 "
    "same as v12's own GRADUATED run -- plasticity_reset_enable=True with reset_fraction=0.0 is a "
    "deliberate no-op, kept ONLY to reuse the existing column-log capture pathway), K_START=2, "
    "precision=fp32 max_steps=100000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- testing whether "
    "extending QK-Norm to V/O fixes the saturation v12b showed shifting onto v_proj/o_proj/"
    "input_proj; logging to logs/plasticity_column_snapshots/dense_lr_unscaled_v13_qkvo_norm/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v13_qkvo_norm",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
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
print("DONE_DENSE_LR_UNSCALED_QKVO_NORM_V13", flush=True)
