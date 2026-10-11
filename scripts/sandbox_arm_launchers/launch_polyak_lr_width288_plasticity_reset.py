"""v2 relaunch, same base config as launch_polyak_lr_width288.py (the
RECALIBRATED polyak_lr=True run -- f_star=0, c=0.0005, lr_max=0.05,
denom=Lbar; historical peak vocab=32/k=3, flat for ~94,000 steps). v1
(dead_fraction=0.01, k=2.0) MATCHED the historical stall exactly (same
peak vocab=32/k=3) but never broke past it, and showed accuracy
declining across the plateau rather than converging to a floor. Root
cause worked out from v1's finished logs + the engine's own update
formula (not just correlation): the frozen pool's dev never crossed
k=2.0 in any of 3 full 100k-step runs (a permanent no-op), leaving the
ungated, full-strength dead pool as the only active mechanism -- its
own touch shrinks importance further, making a touched column MORE
likely to be re-selected next cycle, a self-reinforcing spiral with no
protection for genuinely-useful-but-intermittently-active columns
(exactly what MQAR associative recall produces). v2: dead pool PRUNED
entirely, frozen pool's k recalibrated to 1.0 (from v1's real dev
distribution, not a blind guess) so it can actually fire. See
docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design for
the full derivation."""

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
    "# DENSE + POLYAK_LR + PLASTICITY_RESET v2 (dead pool pruned, k=1.0, reset_fraction=0.01), "
    "K_START=2, write_time_aux_targets=False precision=fp32 "
    "max_steps=100000 seed=1000 embed_width=36 k_first_target=3 NUM_CPUS=4 -- "
    "testing against the original's flat vocab=32/k=3 stall (~94,000 steps flat)",
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
    polyak_lr=True,
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
print("DONE_POLYAK_LR_WIDTH288_PLASTICITY_RESET", flush=True)
