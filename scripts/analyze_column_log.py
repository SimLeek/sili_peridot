"""Offline analysis of the per-column plasticity snapshots collected by
launch_dense_lr_unscaled_plasticity_reset_column_log.py (see
docs/research/toy_tile_recurrence_rmt.rst:plasticity_column_state).
One-off exploratory script -- findings get recorded in JOURNAL.md, this
file is disposable (see feedback_scripts_vs_tests_convention)."""

import glob
import os

import numpy as np

BASE = "logs/plasticity_column_snapshots/dense_lr_unscaled"


def load_pool(pool):
    files = sorted(glob.glob(os.path.join(BASE, pool, "*.npz")))
    steps, loss, acc, importance, grad_slow, grad_fast, grad_var, age, reset_active = ([] for _ in range(9))
    for f in files:
        d = np.load(f)
        steps.append(int(d["step"]))
        loss.append(float(d["loss_ema"]))
        acc.append(float(d["acc_ema"]))
        importance.append(d["col_importance"])
        grad_slow.append(d["col_grad_slow"])
        grad_fast.append(d["col_grad_fast"])
        grad_var.append(d["col_grad_var"])
        age.append(d["col_age"])
        reset_active.append(d["col_reset_active"])
    return {
        "steps": np.array(steps),
        "loss": np.array(loss),
        "acc": np.array(acc),
        "importance": np.stack(importance),
        "grad_slow": np.stack(grad_slow),
        "grad_fast": np.stack(grad_fast),
        "grad_var": np.stack(grad_var),
        "age": np.stack(age),
        "reset_active": np.stack(reset_active),
    }


def analyze_pool(name, data):
    n_cycles, n_cols = data["importance"].shape
    max_age = data["age"].max(axis=0)
    never_reset = data["reset_active"].sum(axis=0) == 0
    final_imp = data["importance"][-1]
    # Engine's real z-score deviation formula, NOT grad_fast/grad_slow
    # (that ratio blows up whenever grad_slow crosses zero, common since
    # it EMAs a signed per-cycle delta).
    std_dev = np.sqrt(np.maximum(0.0, data["grad_var"]))
    deviation = (data["grad_fast"] - data["grad_slow"]) / (std_dev + 1e-8)

    print(f"\n=== {name} ===  cycles={n_cycles} cols={n_cols}")
    print(f"  importance range across run: [{data['importance'].min():.5f}, {data['importance'].max():.5f}]")
    print(f"  deviation range across run:  [{deviation.min():.3f}, {deviation.max():.3f}]  mean={deviation.mean():.3f}")
    print(f"  max col_age reached: {max_age.max()} (median {np.median(max_age):.0f})")
    n_never = never_reset.sum()
    print(f"  columns NEVER reset (0 times across whole run): {n_never}/{n_cols}")
    if n_never:
        idx = np.where(never_reset)[0]
        imp_never = final_imp[idx]
        print(
            f"    their final importance: min={imp_never.min():.5f} "
            f"median={np.median(imp_never):.5f} max={imp_never.max():.5f}  "
            f"(population final importance median={np.median(final_imp):.5f})"
        )

    # idle-column check: low importance AND low grad activity AND old age,
    # sustained for a long stretch -- would the old dead-pool criterion
    # (bottom-K importance*|weight|, now unreplayable since weight isn't
    # logged) have found something top-importance selection structurally
    # cannot reach?
    low_imp_thresh = np.percentile(data["importance"], 5, axis=1, keepdims=True)
    persistently_low = (data["importance"] <= low_imp_thresh).mean(
        axis=0
    )  # fraction of cycles each col was in bottom 5%
    chronically_idle = np.where((persistently_low > 0.8) & never_reset)[0]
    print(f"  chronically-idle (bottom-5%-importance >80% of cycles, never reset): {len(chronically_idle)}/{n_cols}")

    return {
        "deviation": deviation,
        "never_reset": never_reset,
        "chronically_idle": chronically_idle,
        "final_imp": final_imp,
    }


def top_k_indices(score, k):
    return np.argsort(-score)[:k]


def rank_pct(values):
    """0..1 percentile rank per row, ties averaged."""
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    n = values.shape[1]
    row_idx = np.arange(values.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(n)[None, :]
    return ranks / (n - 1)


STALL_START_STEP = 18837  # last real level_up in STAGE_HISTORY; vocab=126/k=2
# held for the rest of the 100k-step run (the "critical fail period").

MAX_CI = 100.0  # matches kSynapsePolicyMaxCi / max_ci=100.0f clamp on the
# real ci accumulator (sili/cpu_backend.cpp) that col_importance EMAs.


def l2_saturation_decay(importance, l2_lambda=0.1, threshold=0.9):
    """L2-style decay on importance, gated by how close the population's
    L2 norm is to the fully-saturated ceiling (||100*ones(n)||_2 =
    max_ci*sqrt(n)) -- a function of CURRENT state only, no age/step
    counter. A counter-based decay exponent eventually overflows or
    silently changes meaning at large step counts; this project's
    plasticity mechanisms are required to stay well-defined at any
    horizon (see the design's own scale-invariance requirement), so an
    L2-norm saturation ratio is the right shape here, not col_age.
    decay_strength ramps 0->1 as the population's saturation ratio goes
    threshold->1.0 -- near-zero effect while comfortably below the
    ceiling ("not activating much when it actually is below it"), real
    proportional shrinkage once the whole vector is genuinely saturated.
    """
    n = importance.shape[-1]
    l2_norm = np.sqrt((importance**2).sum(axis=-1, keepdims=True))
    ceiling_norm = MAX_CI * np.sqrt(n)
    sat_ratio = l2_norm / ceiling_norm
    decay_strength = np.clip((sat_ratio - threshold) / (1.0 - threshold), 0.0, 1.0)
    return importance * (1.0 - l2_lambda * decay_strength)


def replay_selection(name, data, deviation, l2_lambda=0.1, l2_threshold=0.9):
    n_cycles, n_cols = data["importance"].shape
    k = max(1, round(0.01 * n_cols))

    imp_rank = rank_pct(data["importance"])
    dev_rank = rank_pct(deviation)
    combined_score = imp_rank + dev_rank

    # offline re-scoring only (not a true counterfactual -- see caveat
    # printed below): recomputed fresh from each cycle's already-logged
    # raw importance, not compounded across cycles, since the real
    # trajectory already reflects unmodified training.
    decayed_imp = l2_saturation_decay(data["importance"], l2_lambda, l2_threshold)

    is_stall = data["steps"] >= STALL_START_STEP

    sets = {crit: {"success": set(), "stall": set()} for crit in ("raw", "combined", "decayed")}
    prev_top = {"raw": None, "combined": None, "decayed": None}
    repeat_count = {
        "raw": {"success": 0, "stall": 0},
        "combined": {"success": 0, "stall": 0},
        "decayed": {"success": 0, "stall": 0},
    }
    cycle_count = {"success": 0, "stall": 0}

    for t in range(n_cycles):
        period = "stall" if is_stall[t] else "success"
        cycle_count[period] += 1
        tops = {
            "raw": frozenset(top_k_indices(data["importance"][t], k).tolist()),
            "combined": frozenset(top_k_indices(combined_score[t], k).tolist()),
            "decayed": frozenset(top_k_indices(decayed_imp[t], k).tolist()),
        }
        for crit, top in tops.items():
            sets[crit][period].update(top)
            if prev_top[crit] is not None and prev_top[crit] == top:
                repeat_count[crit][period] += 1
            prev_top[crit] = top

    n = importance_n = data["importance"].shape[1]
    l2_norm = np.sqrt((data["importance"] ** 2).sum(axis=1))
    sat_ratio = l2_norm / (MAX_CI * np.sqrt(importance_n))
    print(
        f"  [replay] k={k} per cycle, split at step {STALL_START_STEP} "
        f"({cycle_count['success']} success cycles, {cycle_count['stall']} stall cycles), "
        f"l2_lambda={l2_lambda} threshold={l2_threshold}"
    )
    print(
        f"    population L2 saturation ratio: success mean={sat_ratio[~is_stall].mean():.3f}  "
        f"stall mean={sat_ratio[is_stall].mean():.3f}  (1.0 = fully saturated at max_ci)"
    )
    for period in ("success", "stall"):
        n = cycle_count[period] or 1
        print(
            f"    {period}: distinct cols selected  raw={len(sets['raw'][period])}  "
            f"combined={len(sets['combined'][period])}  l2decay={len(sets['decayed'][period])}"
            f"   |  exact-same-top-k-as-prev-cycle  "
            f"raw={repeat_count['raw'][period]}/{n} ({100 * repeat_count['raw'][period] / n:.0f}%)  "
            f"combined={repeat_count['combined'][period]}/{n} ({100 * repeat_count['combined'][period] / n:.0f}%)  "
            f"l2decay={repeat_count['decayed'][period]}/{n} ({100 * repeat_count['decayed'][period] / n:.0f}%)"
        )


def discrimination_over_time(name, data):
    """Population std of raw importance per cycle, sampled at 5 points --
    shows whether the ranking signal's spread collapses as training
    progresses (saturation) vs stays informative."""
    n_cycles = data["importance"].shape[0]
    idxs = [int(n_cycles * f) for f in (0.05, 0.25, 0.5, 0.75, 0.99)]
    stds = [float(data["importance"][i].std()) for i in idxs]
    steps = [int(data["steps"][i]) for i in idxs]
    print(f"  [discrimination] importance std at steps {steps}: " + ", ".join(f"{s:.4f}" for s in stds))


def main():
    pools = sorted(d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)))
    results = {}
    for pool in pools:
        data = load_pool(pool)
        r = analyze_pool(pool, data)
        discrimination_over_time(pool, data)
        replay_selection(pool, data, r["deviation"])
        results[pool] = r


if __name__ == "__main__":
    main()
