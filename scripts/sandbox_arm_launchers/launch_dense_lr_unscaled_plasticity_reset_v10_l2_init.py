"""v10: real-engine validation of L2 Init (Kumar, Marklund & Van Roy,
"Maintaining Plasticity in Continual Learning via Regenerative
Regularization", CoLLAs 2025, arXiv:2308.11958) -- regularizes weight
toward its OWN initial value, found in the paper to mitigate loss of
plasticity more consistently than Shrink-and-Perturb or plain L2.

Genuinely NEW real engine mechanism (did not exist before this run --
only tested in the offline Python sandbox until now):
apply_amortized_l2_init/apply_amortized_block4_l2_init, built with the
same TDD discipline as every other mechanism this session. Direct
instruction, after the offline sandbox showed rate=0.0001 avoiding
importance saturation while keeping real accumulated importance (unlike
rate=0.01, which was confirmed too aggressive -- crushed importance to
near-zero everywhere): "let's try both of them and record while seeing
if it helps reliably beat mqar rather than randomly stalling."

plasticity_reset_enable=True with reset_fraction=0.0 is a deliberate
no-op (nothing ever gets selected/reset by IT) -- enabled ONLY to reuse
the existing column-log npz-writing pathway (currently coupled to
plasticity_reset_enable) so this run stays replayable the same way
every other arm has been. l2_init_enable=True is the only mechanism
actually touching weights here."""

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
):
    loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
    acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
    tag = f"  [{event}]" if event else ""
    sps_s = f"  steps/sec={steps_per_sec:.1f}" if steps_per_sec is not None else ""
    print(
        f"  step={step:>7}  vocab={vocab_size:>4}  k={k:>3}  loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}",
        flush=True,
    )


print(
    "# DENSE-UNSCALED + L2_INIT v10 (rate=0.0001 -- plasticity_reset_enable=True with "
    "reset_fraction=0.0 is a deliberate no-op, kept ONLY to reuse the existing column-log "
    "capture pathway; l2_init_enable=True is the only real mechanism), K_START=2, precision=fp32 "
    "max_steps=100000 seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- testing whether "
    "this reliably beats MQAR instead of randomly stalling like v5-v8 did; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v10_l2_init/",
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v10_l2_init",
    plasticity_raw_importance_log=True,
    l2_init_enable=True,
    l2_init_rate=0.0001,
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
print("DONE_DENSE_LR_UNSCALED_L2_INIT_V10", flush=True)
