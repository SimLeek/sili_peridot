"""Checkpoint/repro for the baseline_energy skip-rate investigation
(2026-08-12): OriginalArchModel(dense=True, use_energy=True,
l1_sparsity_coef=0.05), seed=1000, hits a non-finite gradient early
(observed at step=1959 in the original run; run() then goes on to skip
46.8% of steps over a full 15000-step run under this config).

CORRECTED, per direct user question -- this script is NOT reliably
reproducible step-for-step, despite using a fixed seed. Directly
tested: two back-to-back reruns of this exact file gave first_nonfinite
at step=2202 and step=1611 respectively -- neither matches the original
1959, and they don't match each other. This is the SAME residual
run-to-run nondeterminism already found at the full multi-seed landmark
scale (post block4 StochasticRounding fix, commit 900b318: seed=1001
flipped between 0.6667/1.0000 across two identical runs, baseline
skip_rate varied 0.011%-0.509%) -- it is NOT confined to multi-seed
loops; it shows up in a single isolated model over a long enough
(15000-step) run too. The stochastic-rounding fix genuinely fixed
short-timescale/few-step determinism (confirmed via multiple
independent checks) but did NOT eliminate whatever causes long-horizon
divergence; that source remains unidentified. Do not treat "step 1959"
(or any specific step number from this script) as a fixed reference --
treat this as "log_sigmas drifts monotonically negative and unbounded,
with no sign of leveling off, and this reliably (if not at a fixed
step) leads to a non-finite gradient somewhere in the first ~1500-2500
steps" instead.

State at the ORIGINAL first failure (kept for reference only, not as a
guaranteed-reproducible target) -- log this before touching log_sigmas
clamping, so the underlying mechanism can still be investigated after a
clamp fix lands and makes the RAW unbounded-drift failure unreproducible:
    step=1959 i=3: gradnorm non-finite! aux_loss=0.7474427819252014
    log_sigmas=[-1.3583835, -1.3720931, -1.0657132, -0.66924304]
    input_ln_absmax=1.0218 (crossed back above 1.0 in the same window
    log_sigmas has been drifting monotonically negative, unbounded,
    since ~step 20 with no sign of leveling off before the failure --
    sigmas=exp(log_sigmas) was NOT yet near-zero at the failure point
    (~0.26-0.51), so "sigma collapses to exactly 0" is not the direct
    trigger at this step, though the drift itself is still a real,
    unbounded problem worth fixing regardless of the precise NaN
    mechanism. input_ln_absmax reversing from a ~0.69 minimum (around
    step 1000-1100) back up through 1.0 in the same step window as the
    failure is a notable but NOT yet confirmed-causal coincidence.
"""
import numpy as np
from scripts.l1_sparsity_probe import (OriginalArchModel, VOCAB, EMBED_WIDTH, COLUMN_NEURONS,
                                        NUM_TILES, _build_tile_window, cross_entropy_sum,
                                        clip_grad_norm_, AdamOptimizer, STEPS_PER_STAGE,
                                        generate_copy_sequence, lr_schedule)

seed = 1000
N_STEPS = 15000
task_rng = np.random.RandomState(seed)
embed_table = task_rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3
state_width = EMBED_WIDTH * COLUMN_NEURONS

m = OriginalArchModel(seed, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
                       l1_sparsity_coef=0.05, use_energy=True)
opt = AdamOptimizer()

first_nonfinite = None
for step in range(1, N_STEPS + 1):
    lr = lr_schedule(step, N_STEPS, 0.002, 50)
    seq_len = min(2 + step // STEPS_PER_STAGE, NUM_TILES)
    tokens, pairs = generate_copy_sequence(task_rng, VOCAB, seq_len)
    targets = dict(pairs)
    M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
    aux_val = None
    for i in range(seq_len):
        window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
        M, logits, aux = m.step(window, M, lr)
        aux_val = float(aux.data) if aux is not None else None
        if not np.all(np.isfinite(M)) and first_nonfinite is None:
            first_nonfinite = step
            print(f"step={step} i={i}: M went non-finite! aux_loss={aux_val}")
        if i in targets:
            loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
            if aux is not None:
                loss = loss + aux
            loss.backward()
            n = clip_grad_norm_(m.parameters_for_optimizer(), 1.0)
            if not np.isfinite(n) and first_nonfinite is None:
                first_nonfinite = step
                print(f"step={step} i={i}: gradnorm non-finite! aux_loss={aux_val}")
            opt.step(m.parameters_for_optimizer(), lr=lr)
    if step % 100 == 0 or step <= 20 or (first_nonfinite and step <= first_nonfinite + 3):
        m_ok = np.all(np.isfinite(M))
        print(f"step={step}: aux_loss={aux_val} M_absmax={np.max(np.abs(M)) if m_ok else 'NaN'} "
              f"log_sigmas={m.log_sigmas.data} input_ln_absmax={np.max(np.abs(m.input_ln.data)):.4f}")
    if first_nonfinite is not None and step > first_nonfinite + 10:
        break

print(f"first_nonfinite step: {first_nonfinite}")
