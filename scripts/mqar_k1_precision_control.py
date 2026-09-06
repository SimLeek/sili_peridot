"""
scripts/mqar_k1_precision_control.py
───────────────────────────────────────
See docs/research/mqar_k1_precision_control.rst:mqar_k1_precision_control.module_overview.
See docs/research/mqar_k1_precision_control.rst:mqar_k1_precision_control.sparse_only_comparison_design.
See docs/research/mqar_k1_precision_control.rst:mqar_k1_precision_control.state_width_crash_workaround.

Run: python3 scripts/mqar_k1_precision_control.py [train_steps] [seed] [eval_every]
"""

from __future__ import annotations

import functools
import sys
import time

sys.path.insert(0, ".")

from sili.sparse_rnn import DISLDOLayer, DISLDOLayer32

import scripts.train_mqar_k_sweep as m
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

    # See docs/research/mqar_k1_precision_control.rst:mqar_k1_precision_control.state_width_crash_workaround.
    m.MAX_WEIGHTS_PER_LAYER = 4096
    m.EMBED_WIDTH = 8
    m.COLUMN_NEURONS = 4
    # See docs/research/mqar_k1_precision_control.rst:mqar_k1_precision_control.num_cpus_race_workaround.
    m.NUM_CPUS = 1

    log(
        f"# K=1 precision control: train_steps={train_steps} seed={seed} eval_every={eval_every} "
        f"embed_width={m.EMBED_WIDTH} column_neurons={m.COLUMN_NEURONS} "
        f"state_width={m.EMBED_WIDTH * m.COLUMN_NEURONS} vocab={m.VOCAB} "
        f"l1_sparsity_coef={m.L1_SPARSITY_COEF} peak_lr={m.PEAK_LR}"
    )

    arms = {
        "sparse_fp4_multi_digit": functools.partial(
            TrueMultiDigitLayer, digit_cls=DISLDOLayer, n_stages=3, base=12.0, lr_power=0.0
        ),  # dense=False (default)
        "sparse_fp32": DISLDOLayer32,
    }

    results = {}
    for name, cls in arms.items():
        log(f"\n=== {name} ===")
        m.DISLDO_CLS = cls

        def log_fn(k, step, total, elapsed, loss, acc, _name=name):
            acc_s = f"  acc={acc:.4f}" if acc is not None else ""
            log(
                f"  [{_name}] step={step:>6}/{total}  mean_query_loss={loss:.4f}{acc_s}  "
                f"({elapsed:.0f}s elapsed, {elapsed / step:.4f}s/step)"
            )

        t0 = time.time()
        r = m.train_and_eval(1, seed, train_steps, log_fn=log_fn, pool_size=1, refresh_every=1, eval_every=eval_every)
        log(f"{name} FINAL: {r}  (total {time.time() - t0:.0f}s)")
        results[name] = r

    log("\n# SUMMARY")
    log(f"{'arm':>25}  {'acc':>8}")
    for name, r in results.items():
        log(f"{name:>25}  {r['acc']:>8.4f}")
    log(f"{'dense_fp4 (prior run, 200000 steps)':>25}  {0.1667:>8.4f}")

    log_file.close()


if __name__ == "__main__":
    main()
