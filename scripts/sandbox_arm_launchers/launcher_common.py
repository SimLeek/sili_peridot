"""Shared log_fn/trajectory_log_fn/final-summary helpers for the
sandbox_arm_launchers consolidation (launch.py/sweep.py) -- lifted
verbatim from train_mqar_curriculum.py's own main(), which already had
the complete, battle-tested superset of every log_fn variant the ~80
retired launcher scripts had separately copy-pasted (each printing a
different subset of the same available fields, never different data).
Guard-on-presence design: every optional field is printed if the
caller supplied it, omitted otherwise -- this is why one function
losslessly covers every prior variant instead of needing several.

See scripts/sandbox_arm_launchers/README.md for how launch.py/sweep.py
use this, and JOURNAL.md for the consolidation's own rationale.
"""

from __future__ import annotations

SHORT_NAME = {"input_proj": "in", "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "lm_head": "lm"}

TIMING_ORDER = ["input_proj", "q_proj", "k_proj", "v_proj", "o_proj", "attention", "lm_head", "critic_head"]


def ranks_str(ranks) -> str:
    if not ranks:
        return ""
    parts = [f"{SHORT_NAME.get(n, n)}={s}/{a}" for n, (s, a) in ranks.items()]
    return "  ranks[" + " ".join(parts) + "]"


def layer_timing_str(layer_timing, window_wall_s) -> str:
    if not layer_timing or window_wall_s is None:
        return ""
    parts = []
    accounted_s = 0.0
    for name in TIMING_ORDER:
        comp = layer_timing.get(name)
        if comp is None:
            continue
        comp_s = comp.get("fwd_s", 0.0) + comp.get("bwd_s", 0.0)
        accounted_s += comp_s
        pct = 100.0 * comp_s / window_wall_s if window_wall_s > 0 else 0.0
        parts.append(f"{SHORT_NAME.get(name, name)}={comp_s:.2f}s({pct:.0f}%)")
    other_s = max(0.0, window_wall_s - accounted_s)
    other_pct = 100.0 * other_s / window_wall_s if window_wall_s > 0 else 0.0
    parts.append(f"other={other_s:.2f}s({other_pct:.0f}%)")
    return "  t[" + " ".join(parts) + f" / {window_wall_s:.2f}s]"


def plasticity_totals_str(totals) -> str:
    if not totals:
        return ""
    n_reset = sum(t["n_reset"] for t in totals.values())
    worst_key, worst = max(totals.items(), key=lambda kv: kv[1]["last_deviation"])
    return (
        f"  plasticity[reset={n_reset} "
        f"worst={worst_key}(dev={worst['last_deviation']:.2f}"
        f"[{worst['last_min_deviation']:.2f},{worst['last_max_deviation']:.2f}]"
        f",imp={worst['last_importance']:.4f})]"
    )


def qk_spectral_norm_str(qk_spectral_norm) -> str:
    if not qk_spectral_norm:
        return ""
    parts = [f"{SHORT_NAME.get(n, n)}={v:.2f}" for n, v in qk_spectral_norm.items()]
    return "  qk_specnorm[" + " ".join(parts) + "]"


def r_target_str(label, d) -> str:
    if not d:
        return ""
    _set = {n: v for n, v in d.items() if v is not None}
    if not _set:
        return ""
    return "  " + label + "[" + " ".join(f"{SHORT_NAME.get(n, n)}={v:.3f}" for n, v in _set.items()) + "]"


def make_log_fn(streak_threshold: int = 10):
    """Returns a log_fn covering every field train_curriculum's log_fn
    callback can supply -- guarded on presence, so it's safe to pass as
    every preset's log_fn regardless of which optional features that
    preset's own config actually turns on."""

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
        streak_s = f"  max_streak={max_streak:>2}/{streak_threshold}" if max_streak is not None else ""
        timing_s = layer_timing_str(layer_timing, window_wall_s)
        plasticity_s = plasticity_totals_str(plasticity_totals)
        qk_specnorm_s = qk_spectral_norm_str(qk_spectral_norm)
        dy_r_s = r_target_str("dy_r_target", dy_r_target)
        x_r_s = r_target_str("x_r_target", x_r_target)
        print(
            f"  step={step:>7}  phase={phase:<5}  vocab={vocab_size:>4}  k={k:>3}  "
            f"loss_ema={loss_s}  acc_ema={acc_s}{tag}{timing_s}{sps_s}{streak_s}{dy_r_s}{x_r_s}"
            f"{ranks_str(ranks)}{plasticity_s}{qk_specnorm_s}",
            flush=True,
        )

    return log_fn


def make_trajectory_log_fn():
    """Returns the default fine-grained trajectory_log_fn -- same one
    train_mqar_curriculum.main() and launch_kfirst_remote.py each used
    to define separately."""

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
            dy_parts.append(f"{SHORT_NAME.get(name, name)}=" + ",".join(bits))
        x_parts = []
        for name, x_target in model.x_r_target.items():
            if x_target is None:
                continue
            sel = model.last_input_selection.get(name)
            if sel is not None:
                x_parts.append(
                    f"{SHORT_NAME.get(name, name)}=target{x_target:.3f},R{sel['R_mean']:.3f},k{sel['k_mean']:.1f}"
                )
            else:
                x_parts.append(f"{SHORT_NAME.get(name, name)}=target{x_target:.3f}")
        if dy_parts:
            print(f"    [TRAJ] step={step:>7} axis=dy " + " ".join(dy_parts), flush=True)
        if x_parts:
            print(f"    [TRAJ] step={step:>7} axis=x  " + " ".join(x_parts), flush=True)

    return trajectory_log_fn_default


def print_final_summary(r: dict, label: str = "") -> None:
    """Prints every field train_curriculum's return dict always has
    (steps_per_sec/peak_rss_mb/precision/final_vocab/final_k/
    final_phase/peak_stage/graduated/total_steps/elapsed_s/
    stage_history/...) -- every prior launcher just printed a subset
    of this same, always-present dict."""
    tag = f"[{label}] " if label else ""
    print(
        f"\n{tag}FINAL precision={r.get('precision')} final_vocab={r['final_vocab']} "
        f"final_k={r['final_k']} final_phase={r['final_phase']} graduated={r.get('graduated')} "
        f"total_steps={r.get('total_steps')} steps_per_sec={r['steps_per_sec']:.2f} "
        f"elapsed_s={r.get('elapsed_s', 0.0):.0f}",
        flush=True,
    )
    peak = r.get("peak_stage")
    if peak:
        print(f"{tag}PEAK peak_vocab={peak.get('vocab')} peak_k={peak.get('k')}", flush=True)
    if "stage_history" in r:
        print(f"{tag}STAGE_HISTORY {r['stage_history']}", flush=True)
    print(f"DONE_{label.upper()}" if label else "DONE", flush=True)
