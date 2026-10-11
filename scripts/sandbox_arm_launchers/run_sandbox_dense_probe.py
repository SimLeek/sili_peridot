import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as tmc

tmc.K_START = 2


def _q(tokens, mqar_pairs, num_kv_pairs):
    return dict(mqar_pairs)


tmc._build_targets = _q


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
    sps_s = f"sps={steps_per_sec:.2f}" if steps_per_sec is not None else ""
    tag = f"  [{event}]" if event else ""
    print(
        f"step={step:>6} vocab={vocab_size:>3} k={k} phase={phase:<5} loss={loss_ema:.3f} acc={acc_ema:.3f}{tag} {sps_s}",
        flush=True,
    )


r = tmc.train_curriculum(
    precision="fp32",
    max_steps=3000,
    seed=0,
    peak_lr=0.015,
    num_tiles=8,
    k_max=8,
    log_every=500,
    log_fn=log_fn,
    embed_width=36,
    k_first_target=3,
    wrong_streak_threshold=float("inf"),
)
print(f"\nFINAL steps_per_sec={r['steps_per_sec']:.2f} elapsed_s={r['elapsed_s']:.0f}", flush=True)
