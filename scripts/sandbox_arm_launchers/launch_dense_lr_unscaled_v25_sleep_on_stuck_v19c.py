"""v25: v19c's EXACT config (ci_renorm+weight_renorm+plasticity_reset,
seed=1000 -- the cleanest reliably-stuck run on record, stalled at
vocab=126/k=2 for the full 100k steps) PLUS sleep_enable=True, testing
whether periodic energy-driven sleep annealing (JOURNAL.md's 2026-10-03
"energy-driven sleep annealing" entry) helps a run that's ALREADY stuck
escape, vs just letting it continue to stall.

sleep_energy_kwargs: one shared config applied to all 6 wide layers
(input_proj/q_proj/k_proj/v_proj/o_proj/lm_head), reusing the exact
decay=0.0075 rotation rate found to be the clean stability sweet spot
in the isolated single-layer burst test (faster rotation went noisy/
unstable, slower was clean but much slower to matter) -- fire_wake_gradient
(not the gentle energy_loss term) is the real lever, applied via
model.step()'s own output-gated sleep mechanism, WeightRenorm/CiRenorm/
l2_decay all kept running during sleep steps too (per direct
instruction to run WeightRenorm concurrently). drive/activation_cost
reuse the isolated test's own measured scale (h_mean~=0.79 under random
input) as a first approximation -- NOT separately re-measured per-layer
under this run's real activations, a known simplification to revisit if
results are ambiguous.

sleep_every_steps=5000, sleep_duration_steps=500 (10% duty cycle) --
one sleep rotation cycle (~400 steps to 95% convergence at this decay)
fits within each burst with some margin, not yet tuned against real
task-performance cost/benefit.

continue_past_graduation=True (v19c's own original launcher predates
this feature) -- full 100k-step trajectory regardless of outcome, for
a fair comparison against v19c's own already-recorded stuck trajectory."""

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
    "# v25: SLEEP ANNEALING ON v19c's KNOWN-STUCK CONFIG (seed=1000, ci_renorm_enable=stable_region "
    "AND weight_renorm_enable=stable_region PLUS sleep_enable=True every 5000 steps for 500 steps, "
    "decay=0.0075 rotation rate on all 6 wide layers, fire_wake_gradient=1.0) PLUS "
    "continue_past_graduation=True, K_START=2, precision=fp32 max_steps=100000 embed_width=36 "
    "k_first_target=3 NUM_CPUS=4 -- does periodic sleep annealing help v19c's own already-stuck "
    "config escape, vs its recorded stall at vocab=126/k=2; logging to "
    "logs/plasticity_column_snapshots/dense_lr_unscaled_v25_sleep_on_stuck_v19c/",
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
    sleep_every_steps=5000,
    sleep_duration_steps=500,
    sleep_energy_kwargs=SLEEP_ENERGY_KWARGS,
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v25_sleep_on_stuck_v19c",
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
print("DONE_V25_SLEEP_ON_STUCK_V19C", flush=True)
