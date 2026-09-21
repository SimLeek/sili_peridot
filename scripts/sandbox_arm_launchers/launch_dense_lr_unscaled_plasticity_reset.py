"""v2 relaunch, same base config as launch_width288_nolevel_down_control.py
(the "dense-unscaled" reference stall: peak_lr=0.015, plain dense, no
sparsity mechanism, K_START=2, LEVEL_DOWN disabled; historical peak
vocab=64/k=3, flat for its final 86,257 steps). v1
(dead_fraction=0.01, k=2.0) UNDERPERFORMED the historical stall --
peaked at only vocab=32/k=2, with accuracy actively declining across
the whole plateau rather than converging to a stable floor like the
true historical stall did. Root cause (worked out from the finished
v1 logs + the engine's own formula, not just correlation): the frozen
pool's dev never crossed k=2.0 in any of 3 full 100k-step runs
(confirmed via grep, gate was a permanent no-op), leaving the
(ungated, full-strength) dead pool as the only active mechanism -- and
its own touch shrinks importance further, making a touched column
MORE likely to be re-selected next cycle, a self-reinforcing spiral
with no real protection for genuinely-useful-but-intermittently-active
columns (exactly what MQAR associative recall produces). v2: dead pool
PRUNED entirely, frozen pool's k recalibrated to 1.0 (from the real
dev distribution observed in v1, not a blind guess) so it can actually
fire. See docs/research/toy_tile_recurrence_rmt.rst:
plasticity_reset_design for the full derivation."""

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
    streak_s = f"  max_streak={max_streak:>2}/10" if max_streak is not None else ""
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
        f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
        f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{streak_s}{plast_s}",
        flush=True,
    )


print(
    "# DENSE-UNSCALED + PLASTICITY_RESET v2 (dead pool pruned, k=1.0, reset_fraction=0.01), "
    "K_START=2, write_time_aux_targets=False precision=fp32 "
    "max_steps=100000 seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- "
    "testing against the original's flat vocab=64/k=3 stall (86,257 steps flat)",
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
print("DONE_DENSE_LR_UNSCALED_PLASTICITY_RESET", flush=True)
