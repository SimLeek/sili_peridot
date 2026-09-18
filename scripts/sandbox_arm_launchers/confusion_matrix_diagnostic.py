"""Runs baseline + Arm A/B/C (and Arm A+C) at a short, equal, bounded
step budget and reports the two failure-mode proxies plus early outcome,
per the dense-vs-sparse-mqar-300k plan's confusion matrix.

Mode 1 (input starvation): fraction of forward-axis dims whose EMA'd
selection frequency (model._x_balance_freq, now tracked unconditionally
whenever x_r_target is active, not just under Arm A) stays below
MODE1_DEAD_THRESHOLD by the end of the run -- "how many neurons are
effectively dead on the forward axis."

Mode 2 (grad signal insufficiency): mean R (backward-axis energy
retention -- fraction of true gradient squared-magnitude kept after
selection, model.last_grad_selection[name]['R_mean']) across the wide
layers at the end of the run. Low R means most of the true gradient
signal is being thrown away regardless of how much magnitude survives.
Reused directly from the ALREADY-INSTRUMENTED last_grad_selection rather
than building a new paired dense-backward-control harness -- a real,
deliberate scope reduction (see plan doc), not silently incomplete.
"""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

STEPS = 3000
SEED = 1000
EMBED_WIDTH = 36
K_FIRST_TARGET = 3
MODE1_DEAD_THRESHOLD = 0.05

ROWS = {
    "baseline": {},
    "arm_a": {"x_balance_loss_coef": 0.01},
    "arm_b": {"x_balance_bias_step": 0.01},
    "arm_c": {"dy_time_gate_cutoff": 0.3, "dy_r_target": None},
    "arm_a_plus_c": {"x_balance_loss_coef": 0.01, "dy_time_gate_cutoff": 0.3, "dy_r_target": None},
}

BASE_KWARGS = {
    "embed_width": EMBED_WIDTH,
    "wrong_streak_threshold": 100000000,
    "k_first_target": K_FIRST_TARGET,
    "r_target_min": 0.3,
    "x_r_target": 0.9,
    "dy_r_target": 0.9,
    "log_every": STEPS,
}


def run_row(name: str, extra: dict) -> None:
    kwargs = dict(BASE_KWARGS)
    kwargs.update(extra)
    model_holder = {}

    def capture_model_fn(step, model):
        model_holder["model"] = model

    r = m.train_curriculum(
        "fp32",
        STEPS,
        SEED,
        0.015,
        16,
        10,
        trajectory_log_every=STEPS - 1,
        trajectory_log_fn=capture_model_fn,
        **kwargs,
    )
    model = model_holder.get("model")

    print(f"\n########## {name} ##########")
    print(
        f"  final: vocab={r['final_vocab']} k={r['final_k']} phase={r['final_phase']} "
        f"steps/sec={r['steps_per_sec']:.2f} peak={r['peak_stage']}"
    )
    if model is None:
        print("  (no model snapshot captured -- trajectory_log_fn never fired)")
        return

    print("  mode 1 (input starvation, frac dims dead):")
    for layer_name, freq in model._x_balance_freq.items():
        dead_frac = float((freq < MODE1_DEAD_THRESHOLD).mean())
        print(f"    {layer_name}: {dead_frac:.3f}")

    print("  mode 2 (grad signal, mean R retained):")
    for layer_name, sel in model.last_grad_selection.items():
        print(f"    {layer_name}: R={sel['R_mean']:.3f} k={sel['k_mean']:.1f}")


if __name__ == "__main__":
    for name, extra in ROWS.items():
        run_row(name, extra)
