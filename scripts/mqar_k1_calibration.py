"""See docs/research/mqar_k1_calibration.rst:module_overview."""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

import scripts.train_mqar_k_sweep as m

LOG_PATH = "mqar_k1_calibration.log"


def main():
    train_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    eval_every = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    log_file = open(LOG_PATH, "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(
        f"# K=1 calibration: train_steps={train_steps} seed={seed} eval_every={eval_every} "
        f"embed_width={m.EMBED_WIDTH} column_neurons={m.COLUMN_NEURONS} "
        f"state_width={m.EMBED_WIDTH * m.COLUMN_NEURONS} vocab={m.VOCAB} "
        f"l1_sparsity_coef={m.L1_SPARSITY_COEF} peak_lr={m.PEAK_LR}"
    )

    def log_fn(k, step, total, elapsed, loss, acc):
        acc_s = f"  acc={acc:.4f}" if acc is not None else ""
        log(
            f"  [K={k}] step={step:>6}/{total}  mean_query_loss={loss:.4f}{acc_s}  "
            f"({elapsed:.0f}s elapsed, {elapsed / step:.4f}s/step)"
        )

    t0 = time.time()
    result = m.train_and_eval(1, seed, train_steps, log_fn=log_fn, pool_size=1, refresh_every=1, eval_every=eval_every)
    log(f"\n# FINAL: {result}  (total {time.time() - t0:.0f}s)")
    log_file.close()


if __name__ == "__main__":
    main()
