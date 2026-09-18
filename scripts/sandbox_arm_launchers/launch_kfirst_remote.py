import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

m.NUM_CPUS = 4

_SHORT_NAME = {"input_proj": "in", "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "lm_head": "lm"}
STREAK_THRESHOLD = 10
_TIMING_ORDER = ["input_proj", "q_proj", "k_proj", "v_proj", "o_proj", "attention", "lm_head", "critic_head"]


def _layer_timing_str(layer_timing, window_wall_s):
    if not layer_timing or window_wall_s is None:
        return ""
    parts = []
    accounted_s = 0.0
    for name in _TIMING_ORDER:
        rec = layer_timing.get(name)
        if rec is None:
            continue
        comp_s = rec["fwd_s"] + rec["bwd_s"]
        accounted_s += comp_s
        pct = 100.0 * comp_s / window_wall_s if window_wall_s > 0 else 0.0
        parts.append(f"{_SHORT_NAME.get(name, name)}={comp_s:.2f}s({pct:.0f}%)")
    other_s = max(0.0, window_wall_s - accounted_s)
    other_pct = 100.0 * other_s / window_wall_s if window_wall_s > 0 else 0.0
    parts.append(f"other={other_s:.2f}s({other_pct:.0f}%)")
    return "  t[" + " ".join(parts) + f" / {window_wall_s:.2f}s]"


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
    streak_s = f"  max_streak={max_streak:>2}/{STREAK_THRESHOLD}" if max_streak is not None else ""
    timing_s = _layer_timing_str(layer_timing, window_wall_s)

    def _r_target_str(label, d):
        if not d:
            return ""
        _set = {n: v for n, v in d.items() if v is not None}
        if not _set:
            return ""
        return "  " + label + "[" + " ".join(f"{_SHORT_NAME.get(n, n)}={v:.3f}" for n, v in _set.items()) + "]"

    dy_r_s = _r_target_str("dy_r_target", dy_r_target)
    x_r_s = _r_target_str("x_r_target", x_r_target)
    print(
        f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
        f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{timing_s}{sps_s}{streak_s}{dy_r_s}{x_r_s}",
        flush=True,
    )


def trajectory_log_fn_default(step, model) -> None:
    dy_parts = []
    for name, r_bar in model.dy_r_target.items():
        if r_bar is None:
            continue
        surprise = model._layer_surprise.get(name)
        sel = model.last_grad_selection.get(name)
        bits = [f"r{r_bar:.3f}"]
        if sel is not None:
            bits.append(f"R{sel['R_mean']:.3f}")
            bits.append(f"k{sel['k_mean']:.1f}")
        if surprise is not None:
            bits.append(f"E{surprise['E_t']:.2g}")
            bits.append(f"L{surprise['Lbar']:.2g}")
        dy_parts.append(f"{_SHORT_NAME.get(name, name)}=" + ",".join(bits))
    x_parts = []
    for name, x_target in model.x_r_target.items():
        if x_target is None:
            continue
        sel = model.last_input_selection.get(name)
        if sel is not None:
            x_parts.append(
                f"{_SHORT_NAME.get(name, name)}=target{x_target:.3f},R{sel['R_mean']:.3f},k{sel['k_mean']:.1f}"
            )
        else:
            x_parts.append(f"{_SHORT_NAME.get(name, name)}=target{x_target:.3f}")
    if dy_parts:
        print(f"    [TRAJ] step={step:>7} axis=dy " + " ".join(dy_parts), flush=True)
    if x_parts:
        print(f"    [TRAJ] step={step:>7} axis=x  " + " ".join(x_parts), flush=True)


print(
    "# MQAR curriculum (k_first_target odometer mode) precision=fp32 max_steps=100000 seed=1000 "
    "peak_lr=0.015 num_tiles=16 k_max=10 embed_width=36 r_target_min=0.5 "
    "dy_r_target=0.9 x_r_target=0.9 target_steps_per_sec=95.69 k_first_target=3",
    flush=True,
)

r = m.train_curriculum(
    "fp32",
    100000,
    1000,
    0.015,
    16,
    10,
    additive_rank=1,
    dynamic_rank_control=True,
    rank_grace_period_steps=50,
    embed_width=36,
    wrong_streak_threshold=100000000,
    dy_r_target=0.9,
    target_steps_per_sec=95.69,
    x_r_target=0.9,
    r_target_min=0.5,
    k_first_target=3,
    trajectory_log_every=2000,
    log_fn=log_fn,
    trajectory_log_fn=trajectory_log_fn_default,
)
print(
    f"\nFINAL final_vocab={r['final_vocab']} final_k={r['final_k']} "
    f"final_phase={r['final_phase']} steps_per_sec={r['steps_per_sec']:.1f} "
    f"({r['elapsed_s']:.0f}s)",
    flush=True,
)
print(f"PEAK peak_vocab={r['peak_stage']['vocab']} peak_k={r['peak_stage']['k']}", flush=True)
