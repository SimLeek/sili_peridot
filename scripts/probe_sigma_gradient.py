"""Direct check (direct instruction): is log_sigmas.grad genuinely zero at
embed_width=32, or just small/mistimed? Distinguishes softmax saturation /
a real bug (gradient ~0 at the SOURCE, before any optimizer/clip touches
it) from an LR-mismatch explanation (gradient present but too small to
move the parameter over the step counts already tried).
"""

import sys

import numpy as np

from scripts.train_mqar_curriculum import train_curriculum


def run(
    label: str,
    embed_width: int,
    input_sparsity_p,
    wide_max_weights,
    output_dy_sparsity_p,
    max_steps: int,
    seed: int = 2001,
) -> None:
    grad_norms = []
    zero_count = 0

    def sigma_grad_debug_fn(step, log_sigmas_grad, centers_grad):
        nonlocal zero_count
        if log_sigmas_grad is None:
            return
        g = np.asarray(log_sigmas_grad, dtype=np.float64)
        n = float(np.linalg.norm(g))
        grad_norms.append(n)
        if n == 0.0:
            zero_count += 1

    print(f"=== {label}: embed_width={embed_width} steps={max_steps} ===", flush=True)
    train_curriculum(
        "fp4",
        max_steps,
        seed,
        0.015,
        16,
        10,
        log_every=max_steps + 1,
        additive_rank=1,
        dynamic_rank_control=True,
        rank_grace_period_steps=50,
        use_critic=False,
        recurrent_only_output=False,
        embed_width=embed_width,
        input_sparsity_p=input_sparsity_p,
        wide_max_weights=wide_max_weights,
        output_dy_sparsity_p=output_dy_sparsity_p,
        sigma_grad_debug_fn=sigma_grad_debug_fn,
    )

    arr = np.array(grad_norms)
    n = len(arr)
    print(
        f"  {label}: {n} backward calls captured, {zero_count} with EXACTLY zero "
        f"log_sigmas.grad ({100 * zero_count / max(n, 1):.1f}%)",
        flush=True,
    )
    if n > 0:
        nz = arr[arr > 0]
        print(f"  grad norm: mean={arr.mean():.3e} max={arr.max():.3e} median={np.median(arr):.3e}", flush=True)
        if len(nz) > 0:
            print(f"  grad norm (nonzero only): mean={nz.mean():.3e} min={nz.min():.3e}", flush=True)


def main():
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2001
    run(
        "base-16",
        embed_width=16,
        input_sparsity_p=None,
        wide_max_weights=None,
        output_dy_sparsity_p=0.5,
        max_steps=max_steps,
        seed=seed,
    )
    print(flush=True)
    run(
        "embed-32",
        embed_width=32,
        input_sparsity_p=0.5,
        wide_max_weights=2048,
        output_dy_sparsity_p=0.5,
        max_steps=max_steps,
        seed=seed,
    )


if __name__ == "__main__":
    main()
