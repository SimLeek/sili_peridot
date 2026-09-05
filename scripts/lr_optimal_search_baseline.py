"""Longer, fully-converging find_optimal_lr run on the `baseline` config
(the 8.5-min/11-trial run stopped via time_budget before reaching
log_tol, landing at lr=0.0483/24x/eval_acc=0.65 -- this gives it enough
budget to actually converge and confirm/refine that peak).

Usage: PYTHONPATH=<sili_peridot repo root> python scripts/lr_optimal_search_baseline.py
"""

from model.eval_lr import find_optimal_lr
from scripts.l1_sparsity_probe import OriginalArchModel, evaluate, run


def trial_fn(lr, seed):
    model = OriginalArchModel(
        seed,
        dense=True,
        o_proj_coef=0.0,
        all_layer_coef=0.0,
        l1_sparsity_coef=0.05,
        use_energy=False,
        all_zero_init=False,
    )
    run(model, 1500, seed, verbose=False, peak_lr=lr)
    return evaluate(model, 50, seed)


if __name__ == "__main__":
    result = find_optimal_lr(
        trial_fn,
        seeds=(1000, 1001),
        initial_lr=0.002,
        time_budget_s=2700.0,
        log_tol=0.05,
    )
    print(
        f"\nFINAL: best_lr={result.best_lr:.5f} (mult={result.best_lr / 0.002:.2f}x) "
        f"best_score={result.best_score:.4f} n_trials={result.n_trials} "
        f"elapsed={result.elapsed_s:.0f}s reason={result.stopped_reason}"
    )
    print("\nHistory (log_lr, score), sorted by log_lr:")
    for log_lr, score in sorted(result.history):
        import math

        lr = math.exp(log_lr)
        print(f"  lr={lr:.5f} (mult={lr / 0.002:6.2f}x) score={score:.4f}")
