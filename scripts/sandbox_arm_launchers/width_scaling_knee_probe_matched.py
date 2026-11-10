"""Controlled version of width_scaling_knee_probe.py: pins the
curriculum at a FIXED stage (k=2, vocab=16 -- the more complex, genuine
associative-binding case, not the degenerate k=1 shortcut) via
K_START=2 + an unreachable streak_threshold, so every width is compared
at matched task difficulty instead of wherever each happened to land
after a fixed step budget (the real confound in the first pass --
widths 144/288/576/1152 landed at k=2/2/3/1 respectively, confounding
the width-scaling read for q/k/v/o_proj). Takes MULTIPLE snapshots per
width across training on that fixed stage, so both the width trend and
the within-stage training-progress trend are visible.
"""

import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

STEPS = 2000
SEED = 1000
EMBED_WIDTH = 36
FIXED_X_R_TARGET = 0.9
COLUMN_NEURONS_SWEEP = (4, 8, 16, 32)
SNAPSHOT_EVERY = 400
K_FIRST_TARGET = 3  # k_first_vocab = seq_len_for_k(3)+4 = 16, matches vocab=16 convention


def run_width(column_neurons: int) -> None:
    snapshots = []

    def capture_model_fn(step, model):
        snapshots.append(
            (
                step,
                {name: dict(sel) for name, sel in model.last_input_selection.items()},
                dict(model._x_knee_r_target),
            )
        )

    m.COLUMN_NEURONS = column_neurons
    m.K_START = 2  # start (and, via streak_threshold below, stay) at k=2, not k=1
    r = m.train_curriculum(
        "fp32",
        STEPS,
        SEED,
        0.015,
        16,
        10,
        embed_width=EMBED_WIDTH,
        wrong_streak_threshold=100000000,
        streak_threshold=100000000,  # pin at k=2, vocab=16 -- never levels up
        k_first_target=K_FIRST_TARGET,
        x_r_target=FIXED_X_R_TARGET,
        log_every=STEPS,
        trajectory_log_every=SNAPSHOT_EVERY,
        trajectory_log_fn=capture_model_fn,
    )
    state_width = EMBED_WIDTH * column_neurons

    print(f"\n########## column_neurons={column_neurons} (state_width={state_width}) ##########")
    print(f"  final: vocab={r['final_vocab']} k={r['final_k']} steps/sec={r['steps_per_sec']:.2f}")
    for step, sel_by_layer, knee_by_layer in snapshots:
        for layer_name, sel in sel_by_layer.items():
            k_mean = sel["k_mean"]
            knee_r = knee_by_layer.get(layer_name)
            knee_s = f"{knee_r:.3f}" if knee_r is not None else "n/a"
            print(
                f"  step={step:>5}  {layer_name}: k_mean={k_mean:.1f}  "
                f"k_mean/state_width={k_mean / state_width:.4f}  "
                f"R@x_r_target=0.9={sel['R_mean']:.3f}  knee_elbow_R={knee_s}"
            )


if __name__ == "__main__":
    for cn in COLUMN_NEURONS_SWEEP:
        run_width(cn)
