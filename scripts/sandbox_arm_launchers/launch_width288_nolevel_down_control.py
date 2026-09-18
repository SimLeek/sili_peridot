"""Direct width-288 replica of the 2026-09-07 `arm_nolevel_down` result
(JOURNAL.md) that reached vocab=126 at embed_width=16/state_width=128:
K_START=2 (skip the k=1 shortcut stage) + wrong_streak_threshold
effectively infinite (LEVEL_DOWN fully disabled -- identified back then
as the actual blocker, not capacity/architecture/BPTT) + plain dense,
no sparsity mechanism at all. Never actually run at the current
300k-param width=288 config -- the existing dense reference used the
STANDARD curriculum (K_START=1), not this specific fixed one, so it
isn't a clean apples-to-apples comparison against the historical
success. Tests directly: does the bigger model reach vocab=126 in a
comparable or better step count, or is something about this specific
scaling degenerate, per direct instruction to investigate."""

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
):
    loss_s = f"{loss_ema:.4f}" if loss_ema is not None else "n/a"
    acc_s = f"{acc_ema:.4f}" if acc_ema is not None else "n/a"
    tag = f"  [{event}]" if event else ""
    sps_s = f"  steps/sec={steps_per_sec:.1f}" if steps_per_sec is not None else ""
    streak_s = f"  max_streak={max_streak:>2}/10" if max_streak is not None else ""
    print(
        f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
        f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{sps_s}{streak_s}",
        flush=True,
    )


print(
    "# WIDTH-288 REPLICA of the 2026-09-07 arm_nolevel_down success "
    "(embed_width=16 originally reached vocab=126) -- K_START=2, "
    "wrong_streak_threshold=1e8 (LEVEL_DOWN disabled), plain dense, no "
    "sparsity mechanism. precision=fp32 max_steps=100000 seed=1000 "
    "embed_width=36 k_first_target=3 NUM_CPUS=4",
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
print("DONE_WIDTH288_NOLEVEL_DOWN_CONTROL", flush=True)
