"""Checkpoint/repro for the baseline_energy skip-rate investigation (2026-08-12).

See docs/research/debug_energy_log_sigmas_runaway.rst:
module_overview_and_nonreproducibility for the run config, the observed
non-finite failure, and the CORRECTED finding that this script is NOT
reliably reproducible step-for-step despite a fixed seed.

See docs/research/debug_energy_log_sigmas_runaway.rst:
original_failure_state_snapshot for the state dump at the original
first failure, kept for reference only, not as a reproducible target.
"""

import numpy as np

from scripts.l1_sparsity_probe import (
    COLUMN_NEURONS,
    EMBED_WIDTH,
    NUM_TILES,
    STEPS_PER_STAGE,
    VOCAB,
    AdamOptimizer,
    OriginalArchModel,
    _build_tile_window,
    clip_grad_norm_,
    cross_entropy_sum,
    generate_copy_sequence,
    lr_schedule,
)

seed = 1000
N_STEPS = 15000
task_rng = np.random.RandomState(seed)
embed_table = task_rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3
state_width = EMBED_WIDTH * COLUMN_NEURONS

m = OriginalArchModel(seed, dense=True, o_proj_coef=0.0, all_layer_coef=0.0, l1_sparsity_coef=0.05, use_energy=True)
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
        print(
            f"step={step}: aux_loss={aux_val} M_absmax={np.max(np.abs(M)) if m_ok else 'NaN'} "
            f"log_sigmas={m.log_sigmas.data} input_ln_absmax={np.max(np.abs(m.input_ln.data)):.4f}"
        )
    if first_nonfinite is not None and step > first_nonfinite + 10:
        break

print(f"first_nonfinite step: {first_nonfinite}")
