"""LR range test (Smith-style LR finder): ramps peak_lr exponentially
from lr_min to lr_max over a short, fixed-difficulty run and reports
the LR of steepest loss descent -- a cheap (minutes, not hours)
alternative to bracketing a good peak_lr via several full 100k-step
runs. See docs/research/train_mqar_curriculum.rst:
train_curriculum.lr_override_fn_range_test.

Usage: python3 scripts/lr_range_test.py [embed_width] [num_steps] [lr_min] [lr_max]
"""

import math
import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m


def run_lr_range_test(
    embed_width: int = 36,
    num_steps: int = 1000,
    lr_min: float = 1e-5,
    lr_max: float = 1.0,
    seed: int = 1000,
    pin_vocab: int = 32,
    pin_k: int = 2,
    num_cpus: int = 4,
    smoothing_beta: float = 0.9,
) -> dict:
    """Trains a fresh model with lr(step) ramped log-linearly from
    lr_min to lr_max over num_steps, difficulty pinned at (pin_vocab,
    pin_k) via an effectively-infinite streak_threshold (never levels
    up mid-test -- see width_scaling_knee_probe_matched.py's own
    pinning precedent) so the loss trace reflects LR sensitivity at one
    fixed difficulty, not a moving target. Loss at very high LR can
    diverge to NaN/inf or crash the run outright (a real, expected
    range-test outcome, not a bug) -- both are handled: NaN/inf points
    are excluded from the recommendation search, and an exception mid-run
    is caught, analyzing whatever was captured up to the crash.

    Returns {"trace": [(step, lr, loss), ...], "recommended_lr": float
    | None, "min_loss_lr": float | None, "crashed_at_step": int | None}.
    """
    m.NUM_CPUS = num_cpus
    m.K_START = pin_k
    m.VOCAB_START = pin_vocab

    def lr_schedule(step: int) -> float:
        frac = step / num_steps
        return lr_min * (lr_max / lr_min) ** frac

    trace: list[tuple[int, float, float]] = []

    def log_fn(step, vocab_size, k, phase, event, loss_ema, acc_ema, **_kwargs):
        trace.append((step, lr_schedule(step), loss_ema))

    crashed_at_step = None
    try:
        m.train_curriculum(
            "fp32",
            num_steps,
            seed,
            lr_min,
            16,
            10,
            embed_width=embed_width,
            wrong_streak_threshold=10**9,
            streak_threshold=10**9,
            k_first_target=pin_k,
            lr_override_fn=lr_schedule,
            log_every=1,
            log_fn=log_fn,
        )
    except Exception as exc:
        crashed_at_step = trace[-1][0] if trace else None
        print(f"# range test raised {type(exc).__name__}: {exc} (at step {crashed_at_step})", flush=True)

    recommended_lr, min_loss_lr = _pick_lr(trace, smoothing_beta)
    return {
        "trace": trace,
        "recommended_lr": recommended_lr,
        "min_loss_lr": min_loss_lr,
        "crashed_at_step": crashed_at_step,
    }


def _pick_lr(trace: list[tuple[int, float, float]], smoothing_beta: float) -> tuple[float | None, float | None]:
    """Fastai-style LR finder heuristic: smooth the loss trace, find the
    global minimum, then within the region BEFORE that minimum pick the
    LR at the steepest negative d(loss)/d(log(lr)) -- the point of
    fastest improvement, typically well below where loss starts to
    visibly diverge. NaN/inf points are dropped first."""
    valid = [(s, lr, loss) for s, lr, loss in trace if loss is not None and math.isfinite(loss)]
    if len(valid) < 3:
        return None, None

    smoothed = [valid[0][2]]
    avg = valid[0][2]
    for _s, _lr, loss in valid[1:]:
        avg = smoothing_beta * avg + (1.0 - smoothing_beta) * loss
        smoothed.append(avg)

    min_idx = min(range(len(smoothed)), key=lambda i: smoothed[i])
    min_loss_lr = valid[min_idx][1]
    if min_idx < 2:
        return min_loss_lr, min_loss_lr

    best_idx, best_slope = 1, 0.0
    for i in range(1, min_idx):
        log_lr_prev = math.log(valid[i - 1][1])
        log_lr_cur = math.log(valid[i][1])
        if log_lr_cur == log_lr_prev:
            continue
        slope = (smoothed[i] - smoothed[i - 1]) / (log_lr_cur - log_lr_prev)
        if slope < best_slope:
            best_slope, best_idx = slope, i
    return valid[best_idx][1], min_loss_lr


def _print_report(result: dict) -> None:
    trace = result["trace"]
    print(f"\n{len(trace)} points logged.", flush=True)
    for i in range(0, len(trace), max(1, len(trace) // 20)):
        s, lr, loss = trace[i]
        loss_s = f"{loss:.4f}" if loss is not None and math.isfinite(loss) else "nan/inf"
        print(f"  step={s:>5}  lr={lr:.6f}  loss={loss_s}", flush=True)
    if result["crashed_at_step"] is not None:
        print(f"\nCRASHED at step {result['crashed_at_step']}", flush=True)
    print(f"\nMIN_LOSS_LR={result['min_loss_lr']}", flush=True)
    print(f"RECOMMENDED_LR={result['recommended_lr']}", flush=True)
    print("DONE_LR_RANGE_TEST", flush=True)


if __name__ == "__main__":
    _embed_width = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    _num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    _lr_min = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    _lr_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    print(
        f"# LR RANGE TEST embed_width={_embed_width} num_steps={_num_steps} "
        f"lr_min={_lr_min} lr_max={_lr_max} pin_vocab=32 pin_k=2",
        flush=True,
    )
    _result = run_lr_range_test(embed_width=_embed_width, num_steps=_num_steps, lr_min=_lr_min, lr_max=_lr_max)
    _print_report(_result)
