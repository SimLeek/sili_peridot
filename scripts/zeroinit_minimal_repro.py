"""Minimal, fast repro: does a zero-init model actually learn anything, or
does it stay indistinguishable from a completely untrained model?

Built to check a specific finding (2026-08-16 session): the full 15k-step
landmark_checklist.py zero-init configs (baseline_zeroinit, zeroinit_energy,
zeroinit_rank1, zeroinit_rank2) all produced eval_acc numbers BIT-IDENTICAL
to a model that received ZERO training steps, with predictions stuck on a
single constant token regardless of input/target -- a degenerate-output
signature, not real (if imperfect) learning. Small/fast on purpose so it
can be re-run cheaply both to confirm the bug and, after a fix, to confirm
learning actually starts.

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/zeroinit_minimal_repro.py
"""
from scripts.l1_sparsity_probe import OriginalArchModel, run, evaluate

SEEDS = [1000, 1001, 1002]
N_STEPS = 1500
N_EVAL = 100


def main():
    print(f"zeroinit_minimal_repro: {len(SEEDS)} seeds x {N_STEPS} steps x {N_EVAL}-eval\n", flush=True)

    print("--- untrained (0 steps) reference ---", flush=True)
    untrained = {}
    for seed in SEEDS:
        model = OriginalArchModel(seed, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
                                   l1_sparsity_coef=0.05, use_energy=False, all_zero_init=True,
                                   scale_rank=1)
        acc = evaluate(model, N_EVAL, seed, verbose=True)
        untrained[seed] = acc
        print(f"seed={seed} untrained eval_acc={acc:.4f}", flush=True)

    print("\n--- trained (rank=1, no energy) ---", flush=True)
    for seed in SEEDS:
        model = OriginalArchModel(seed, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
                                   l1_sparsity_coef=0.05, use_energy=False, all_zero_init=True,
                                   scale_rank=1)
        run(model, N_STEPS, seed, verbose=False)
        acc = evaluate(model, N_EVAL, seed, verbose=True)
        delta = acc - untrained[seed]
        flag = "SAME AS UNTRAINED (bug reproduces)" if abs(delta) < 1e-9 else f"differs by {delta:+.4f}"
        print(f"seed={seed} trained(rank1) eval_acc={acc:.4f}  [{flag}]", flush=True)

    print("\n--- trained (rank=2, no energy) ---", flush=True)
    for seed in SEEDS:
        model = OriginalArchModel(seed, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
                                   l1_sparsity_coef=0.05, use_energy=False, all_zero_init=True,
                                   scale_rank=2)
        run(model, N_STEPS, seed, verbose=False)
        acc = evaluate(model, N_EVAL, seed, verbose=True)
        delta = acc - untrained[seed]
        flag = "SAME AS UNTRAINED (bug reproduces)" if abs(delta) < 1e-9 else f"differs by {delta:+.4f}"
        print(f"seed={seed} trained(rank2) eval_acc={acc:.4f}  [{flag}]", flush=True)


if __name__ == "__main__":
    main()
