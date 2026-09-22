"""Offline analysis of the raw-ci-recovery diagnostic run's per-step
synapse samples. One-off exploratory script (see
feedback_scripts_vs_tests_convention)."""

import glob
import os

import numpy as np

SAMPLE_DIR = "logs/raw_ci_diagnostic/dense_lr_unscaled_v3_samples"


def load_layer(name):
    files = sorted(
        glob.glob(os.path.join(SAMPLE_DIR, f"{name}.chunk*.npz")),
        key=lambda f: int(f.split("chunk")[1].split(".")[0]),
    )
    steps, values = [], []
    sample_indices = None
    for f in files:
        d = np.load(f)
        steps.append(d["steps"])
        values.append(d["values"])
        sample_indices = d["sample_indices"]
    return np.concatenate(steps), np.concatenate(values, axis=0), sample_indices


def find_drops_and_recovery(steps, series, drop_frac=0.05, recover_frac=0.98):
    """A 'drop' is a step-to-step decrease of at least drop_frac of the
    pre-drop value. Recovery time = how many subsequent SAMPLES (real
    per-step ticks in our data, not necessarily unique 'steps' since
    2 query targets can share one outer step) it takes to get back to
    recover_frac of the pre-drop value. Returns list of
    (step, drop_amount, recovery_ticks_or_None)."""
    events = []
    n = len(series)
    for t in range(1, n):
        prev, cur = series[t - 1], series[t]
        if prev <= 1e-9:
            continue
        rel_drop = (prev - cur) / prev
        if rel_drop >= drop_frac:
            target = prev * recover_frac
            recovery_ticks = None
            for k in range(t, min(t + 200, n)):
                if series[k] >= target:
                    recovery_ticks = k - t
                    break
            events.append((steps[t], prev, cur, rel_drop, recovery_ticks))
    return events


def main():
    for layer_name in ["v_proj", "input_proj", "q_proj", "k_proj", "o_proj", "lm_head"]:
        steps, values, sample_indices = load_layer(layer_name)
        n_ticks, n_synapses = values.shape
        print(
            f"\n=== {layer_name} === ticks={n_ticks} synapses_sampled={n_synapses} step_range=[{steps[0]},{steps[-1]}]"
        )

        all_events = []
        for j in range(n_synapses):
            series = values[:, j]
            events = find_drops_and_recovery(steps, series)
            for ev in events:
                all_events.append((j, *ev))

        print(f"  total drop events (>=5% single-tick decrease) across all sampled synapses: {len(all_events)}")
        if not all_events:
            continue

        recoveries = [ev[5] for ev in all_events if ev[5] is not None]
        never = sum(1 for ev in all_events if ev[5] is None)
        if recoveries:
            recoveries = np.array(recoveries)
            print("  recovery time to 98% of pre-drop value (in real per-step ticks):")
            print(f"    n_recovered={len(recoveries)}  never_recovered_within_200_ticks={never}")
            print(
                f"    min={recoveries.min()}  median={np.median(recoveries):.1f}  "
                f"mean={recoveries.mean():.1f}  max={recoveries.max()}"
            )
            print(f"    fraction recovering within 1 tick: {(recoveries <= 1).mean():.3f}")
            print(f"    fraction recovering within 5 ticks: {(recoveries <= 5).mean():.3f}")
            print(f"    fraction recovering within 10 ticks: {(recoveries <= 10).mean():.3f}")
        else:
            print(f"  no drops ever recovered within 200 ticks (never={never})")

        # show a few concrete example trajectories around a drop, for the
        # largest few drops observed
        all_events.sort(key=lambda e: -e[3])
        print("  largest drop examples (synapse_idx, step, prev, cur, rel_drop, recovery_ticks):")
        for ev in all_events[:5]:
            print(f"    {ev}")


if __name__ == "__main__":
    main()
