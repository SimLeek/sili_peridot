"""Checkpoint/repro for the baseline_energy skip-rate investigation
(2026-08-12): OriginalArchModel(dense=True, use_energy=True,
l1_sparsity_coef=0.05), seed=1000, hits a non-finite gradient at
EXACTLY step=1959, i=3 (first occurrence; run() then goes on to skip
46.8% of steps over a full 15000-step run under this config).

Kept intentionally reproducible via seed alone -- now that sili__new's
block4 StochasticRounding bug is fixed (commit 900b318), this script's
output is deterministic run-to-run, so no model-weight snapshot is
needed to revisit this exact failure later; re-running this file will
always reach the same state at the same step.

State at first failure (log this before touching log_sigmas clamping,
so the underlying mechanism can still be investigated after a clamp
fix lands and makes the RAW unbounded-drift failure unreproducible):
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
