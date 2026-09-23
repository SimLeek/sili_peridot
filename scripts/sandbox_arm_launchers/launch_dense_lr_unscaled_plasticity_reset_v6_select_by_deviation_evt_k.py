"""v6 relaunch of the select_by_deviation arm, same base config as v5
(dead pool pruned, reset_fraction=0.01, L2 decay OFF -- lambda=0.0
default) but now running against sili__new's FIXED engine: the gate's
threshold k is no longer the fixed k=1.0 passed in (that value is
IGNORED by the engine under select_by_deviation=True as of this fix --
see delta_csr_types.hpp's k_effective), replaced with
sqrt(2*ln(mature_population_size)), derived from Gaussian extreme
value theory rather than guessed.

v5 (same config, old broken gate) never progressed past vocab=16/k=3
even after 37,750+ steps -- confirmed via direct log analysis that its
gate opened 97.25% of cycles (vs 22.25% under top-importance
selection) because selection always returns the population's own
maximum deviation, defeating a gate calibrated against a non-maximal
reference distribution. This run tests whether the EVT-derived
threshold actually restores the mechanism's intended
"only-intervene-on-genuine-anomalies" behavior. See
docs/research/toy_tile_recurrence_rmt.rst:select_by_deviation_early_detection
and its k_derivation follow-up in
docs/research/delta_csr_types.rst (sili__new side) for the full
derivation and test coverage. Column logging enabled to keep
collecting comparable data."""

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
    plast_s = ""
    if plasticity_totals:
        n_reset = sum(t["n_reset"] for t in plasticity_totals.values())
        worst_key, worst = max(plasticity_totals.items(), key=lambda kv: kv[1]["last_deviation"])
        plast_s = (
            f"  plasticity[reset={n_reset} "
            f"worst={worst_key}(dev={worst['last_deviation']:.2f}"
            f"[{worst['last_min_deviation']:.2f},{worst['last_max_deviation']:.2f}]"
            f",imp={worst['last_importance']:.4f})]"
        )
    print(
        f"  step={step:>7}  vocab={vocab_size:>4}  k={k:>3}  loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{plast_s}",
        flush=True,
    )


print(
    "# DENSE-UNSCALED + PLASTICITY_RESET v6 (select_by_deviation=True, EVT-derived gate "
    "k=sqrt(2*ln(N)) replacing the old fixed k=1.0 that v5 showed was defeated by construction, "
    "dead pool pruned, reset_fraction=0.01, L2 decay OFF), K_START=2, write_time_aux_targets=False "
    "precision=fp32 max_steps=100000 seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- "
    "testing whether the EVT-derived gate threshold restores select_by_deviation's intended "
    "only-intervene-on-genuine-anomalies behavior after v5's gate stayed open 97.25% of cycles; "
    "logging to logs/plasticity_column_snapshots/dense_lr_unscaled_v6_select_by_deviation_evt_k/",
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
    plasticity_reset_select_by_deviation=True,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v6_select_by_deviation_evt_k",
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
print("DONE_DENSE_LR_UNSCALED_PLASTICITY_RESET_V6_SELECT_BY_DEVIATION_EVT_K", flush=True)
