"""v8: identical config to v7 (percentile-derived gate k, same base
dense-unscaled setup) -- NOT meant to run detached in the background.
Direct instruction after v7's counterintuitive result (a genuinely-
firing gate did WORSE than a permanently-inert one): "let's ... get
displayarray ... working and display all of the networks side by side
... and I'll just look at all of the synapses for the next run."

Run this yourself, in a terminal you can see, so the live window stays
up: `python scripts/sandbox_arm_launchers/launch_dense_lr_unscaled_plasticity_reset_v8_select_by_deviation_live_display.py`
Press ESC on the "synapses" window to stop watching (training keeps
running in the background after that -- the window just closes).

See scripts/live_synapse_display.py and
docs/research/toy_tile_recurrence_rmt.rst:live_synapse_display for what
each panel means: per pool, top to bottom -- label, weight heatmap
("the synapse"), the deviation/reset strip (red = this column's
z-score, green flag = selected/reset THIS cycle), importance heatmap."""

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
    "# DENSE-UNSCALED + PLASTICITY_RESET v8 (select_by_deviation=True, percentile-derived gate, "
    "identical config to v7) + LIVE SYNAPSE DISPLAY -- run this yourself in a visible terminal, "
    "not detached. max_steps=100000 seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=4",
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
    plasticity_live_display=True,
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
print("DONE_DENSE_LR_UNSCALED_PLASTICITY_RESET_V8_SELECT_BY_DEVIATION_LIVE_DISPLAY", flush=True)
