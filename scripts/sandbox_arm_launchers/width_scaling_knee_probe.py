"""Cheap test of the "more neurons -> smaller required active FRACTION"
conjecture: run the same task/curriculum-stage at a few widths
(column_neurons swept, embed_width fixed), measure the knee-elbow R
(_knee_elbow_r, model/toy_tile_recurrence_rmt.py) and the k_mean at a
fixed x_r_target after a short warm-up, and report k_mean/state_width
per width. If that ratio shrinks as width grows, that's a direct,
empirical confirmation using the SAME measurement already built for the
adaptive x_r_target_auto mechanism -- no new theory needed, just a
rerun at a couple of sizes.
"""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

STEPS = 1500
SEED = 1000
EMBED_WIDTH = 36
K_FIRST_TARGET = 3
FIXED_X_R_TARGET = 0.9
COLUMN_NEURONS_SWEEP = (4, 8, 16, 32)


def run_width(column_neurons: int) -> None:
    model_holder = {}

    def capture_model_fn(step, model):
        model_holder["model"] = model

    # column_neurons isn't a train_curriculum() kwarg -- it's imported as
    # a module-level constant (from scripts.train_mqar_rmt_reference),
    # same pattern already used elsewhere (e.g. m.NUM_CPUS = 4) to vary a
    # "constant" per-call without changing train_curriculum's signature.
    m.COLUMN_NEURONS = column_neurons
    r = m.train_curriculum(
        "fp32",
        STEPS,
        SEED,
        0.015,
        16,
        10,
        embed_width=EMBED_WIDTH,
        wrong_streak_threshold=100000000,
        k_first_target=K_FIRST_TARGET,
        r_target_min=0.3,
        x_r_target=FIXED_X_R_TARGET,
        log_every=STEPS,
        trajectory_log_every=STEPS - 1,
        trajectory_log_fn=capture_model_fn,
    )
    model = model_holder.get("model")
    state_width = EMBED_WIDTH * column_neurons

    print(f"\n########## column_neurons={column_neurons} (state_width={state_width}) ##########")
    print(f"  final: vocab={r['final_vocab']} k={r['final_k']} steps/sec={r['steps_per_sec']:.2f}")
    if model is None:
        print("  (no model snapshot captured)")
        return
    for layer_name, sel in model.last_input_selection.items():
        k_mean = sel["k_mean"]
        elbow_r = model._x_knee_r_target.get(layer_name)
        elbow_s = f"{elbow_r:.3f}" if elbow_r is not None else "n/a"
        print(
            f"  {layer_name}: k_mean={k_mean:.1f}  k_mean/state_width={k_mean / state_width:.4f}  "
            f"R@x_r_target=0.9={sel['R_mean']:.3f}  knee_elbow_R={elbow_s}"
        )


if __name__ == "__main__":
    for cn in COLUMN_NEURONS_SWEEP:
        run_width(cn)
