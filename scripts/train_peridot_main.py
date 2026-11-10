"""THE canonical sili_peridot entry point -- run this to see the
project's current best-known-working recipe for training a sparse,
plasticity-preserving recurrent model on MQAR.

What this demonstrates (see README.md and JOURNAL.md for the full
research narrative):
  - sparse DISLDO layers (dense=True here; see docs/research for the
    sparse-weight arms) trained with the engine's own amortized
    per-synapse update rule ("importance is already the optimizer").
  - magnitude regularization (QKVO-Norm + CiRenorm + WeightRenorm +
    amortized l2_decay) that keeps weight/importance scale calibrated
    throughout a 100k+ step run without manual intervention.
  - a purely intrinsic, reactive "sleep" annealing mechanism
    (sleep_gate_layers) that reverses directional rank collapse in
    q_proj/k_proj by gating EnergyDynamics' forced-firing on those
    layers' own col_importance going static (zero delta between
    amortized cycles) -- no curriculum/task info, no blind schedule,
    self-limiting (turns off the moment the layer starts moving again).

Result this recipe reached (v28, seed=1001, commit history on branch
research/dense-vs-sparse-mqar-300k): vocab=126, k=7 -- well past k=5,
the highest level solvable from pure in-context lookup alone (the
16-token attention window structurally cannot span the key-query gap
k=6/k=7 require; see JOURNAL.md's "k=4/k=5 in-context ceiling" entry).
Reaching k=6/k=7 means the model is genuinely using its 2-slot
recurrent memory with real superposition, not just getting lucky at
the in-context ceiling.

Known issue, NOT fixed here (see JOURNAL.md's 2026-10 "overflow" note,
tracked for a follow-up PR): sili/cpu.py's generic elementwise
Backend.mul can overflow in fp32 during very long runs (observed once,
late in an extended k-plateau) without raising -- silent, not yet
root-caused. Didn't visibly corrupt the run it occurred in, but is a
real correctness gap worth closing before scaling up.

Usage: python3 scripts/train_peridot_main.py [max_steps] [seed]
"""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4
m.K_START = 2

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1001

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
    f"# sili_peridot main entry point: sparsity + plasticity recipe (seed={SEED}, "
    f"max_steps={MAX_STEPS}) -- qkvo_norm_enable=True, ci_renorm_enable=stable_region, "
    "weight_renorm_enable=stable_region, l2_decay_chunk_size=800/l2_decay_adaptation_rate=0.3, "
    "plasticity_reset_enable=True, sleep_enable=True with sleep_gate_layers=(q_proj,k_proj) "
    "freeze-gated annealing, continue_past_graduation=True, embed_width=36, k_first_target=3, "
    "K_START=2 -- logging to logs/plasticity_column_snapshots/train_peridot_main/",
    flush=True,
)
r = m.train_curriculum(
    "fp32",
    MAX_STEPS,
    SEED,
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
    plasticity_column_log_dir="logs/plasticity_column_snapshots/train_peridot_main",
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
print("DONE_TRAIN_PERIDOT_MAIN", flush=True)
