"""
scripts/mqar_k1_precision_control.py
───────────────────────────────────────
Direct test of the user's hypothesis (see conversation): K=1's
saturation at ~15-27% accuracy (mqar_k1_calibration.py, 200000 steps,
dense FP4 multi-digit) might be a superposition-style precision/
capacity bottleneck (routing one of ~4032 possible key/value identities
through several FP4-quantized transforms), not an architecture,
attention, or step-budget problem.

Decisive test: same K=1 task, same step budget, FP32 in place of the
FP4 multi-digit stack (DISLDOLayer32 -- this project's own established
"precision ceiling reference", isolates quantization coarseness from
architecture/training-dynamics limits, matching e.g. train_tile_
curriculum.py's own "fp32" arm convention).

CAVEAT 1 (found while building this, not guessed): DISLDOLayer32 has no
`dense` kwarg -- it's a pure diagnostic class, sparse-echo connectivity
only, no block4/dense support. So an fp32-vs-fp4 comparison can only be
apples-to-apples at matched connectivity if the FP4 side is ALSO run
sparse (not dense=True, which is what mqar_k1_calibration.py used).
Runs BOTH sparse arms here (sparse-fp32 and sparse-fp4-multi-digit) so
precision is the only isolated variable between them -- the existing
dense-fp4 result from mqar_k1_calibration.py remains a separate,
already-available third reference point for the connectivity axis,
not directly compared against fp32 here.

CAVEAT 2 (also found while building this): sparse TrueMultiDigitLayer
at train_mqar_k_sweep.py's own EMBED_WIDTH=16/COLUMN_NEURONS=8
(state_width=128) crashes with a heap-corruption error ("free(): too
many chunks detected in tcache") -- reproduced directly, a real bug,
not a fluke (confirmed it does NOT reproduce at state_width=32). Out of
scope to root-cause here, so this control instead runs at EMBED_WIDTH=8/
COLUMN_NEURONS=4 (state_width=32) -- the scale used everywhere else in
this project (the copy-task curriculum, the model-level long-horizon
test), sidestepping the crash AND using the most validated
configuration for gaussian_attention itself. This means this control's
own numbers are NOT directly comparable to mqar_k1_calibration.py's
0.1667 (different state_width) -- it answers "is FP4 precision the
bottleneck at the well-validated scale," not "...at the specific scale
that plateaued.\""

Run: python3 scripts/mqar_k1_precision_control.py [train_steps] [seed] [eval_every]
"""
from __future__ import annotations

import sys
import time
import functools

sys.path.insert(0, ".")

import scripts.train_mqar_k_sweep as m
from sili.sparse_rnn import DISLDOLayer, DISLDOLayer32
from model.toy_precision_models import TrueMultiDigitLayer

LOG_PATH = "mqar_k1_precision_control.log"


def main():
    train_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    eval_every = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

    log_file = open(LOG_PATH, "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    # Sparse (non-dense) connectivity needs a real per-row weight budget --
    # 512 (this module's own default, sized for dense=True which ignores
    # it) caused a heap-corruption crash ("free(): unaligned chunk
    # detected in tcache") at state_width=128 sparse, found directly while
    # smoke-testing this script. 4096 matches train_toy_tile_precision_
    # comparison.py's own established budget at a comparable scale.
    m.MAX_WEIGHTS_PER_LAYER = 4096
    # state_width=128 (this module's own EMBED_WIDTH=16/COLUMN_NEURONS=8
    # default) ALSO crashes sparse TrueMultiDigitLayer with a DIFFERENT
    # heap-corruption error ("free(): too many chunks detected in
    # tcache") even at max_weights=4096 -- confirmed directly this does
    # NOT reproduce at state_width=32. Dropping to the scale used
    # everywhere else in this project (see CAVEAT 2 above) sidesteps it.
    m.EMBED_WIDTH = 8
    m.COLUMN_NEURONS = 4
    # num_cpus>1 hits an ALREADY-KNOWN pre-existing thread-safety race
    # (same class as test_stats_thread_safety.cpp's tracked segfault, and
    # the intermittent crash found earlier this session in
    # test_toy_tile_precision_models.py at num_cpus=2) -- confirmed
    # directly this is genuinely intermittent (1/3 runs crashed with the
    # SAME config, not deterministic), not something this control script
    # introduced. num_cpus=1 has no concurrency, sidesteps it entirely.
    m.NUM_CPUS = 1

    log(f"# K=1 precision control: train_steps={train_steps} seed={seed} eval_every={eval_every} "
        f"embed_width={m.EMBED_WIDTH} column_neurons={m.COLUMN_NEURONS} "
        f"state_width={m.EMBED_WIDTH*m.COLUMN_NEURONS} vocab={m.VOCAB} "
        f"l1_sparsity_coef={m.L1_SPARSITY_COEF} peak_lr={m.PEAK_LR}")

    arms = {
        "sparse_fp4_multi_digit": functools.partial(TrueMultiDigitLayer, digit_cls=DISLDOLayer,
                                                     n_stages=3, base=12.0, lr_power=0.0),  # dense=False (default)
        "sparse_fp32": DISLDOLayer32,
    }

    results = {}
    for name, cls in arms.items():
        log(f"\n=== {name} ===")
        m.DISLDO_CLS = cls

        def log_fn(k, step, total, elapsed, loss, acc, _name=name):
            acc_s = f"  acc={acc:.4f}" if acc is not None else ""
            log(f"  [{_name}] step={step:>6}/{total}  mean_query_loss={loss:.4f}{acc_s}  "
                f"({elapsed:.0f}s elapsed, {elapsed/step:.4f}s/step)")

        t0 = time.time()
        r = m.train_and_eval(1, seed, train_steps, log_fn=log_fn,
                             pool_size=1, refresh_every=1, eval_every=eval_every)
        log(f"{name} FINAL: {r}  (total {time.time()-t0:.0f}s)")
        results[name] = r

    log("\n# SUMMARY")
    log(f"{'arm':>25}  {'acc':>8}")
    for name, r in results.items():
        log(f"{name:>25}  {r['acc']:>8.4f}")
    log(f"{'dense_fp4 (prior run, 200000 steps)':>25}  {0.1667:>8.4f}")

    log_file.close()


if __name__ == "__main__":
    main()
