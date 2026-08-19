"""
scripts/mqar_calibrated_sweep.py
──────────────────────────────────
Full K=[1,2,4,8] MQAR sweep using a per-K step budget calibrated from
the K=1 saturation run (mqar_k1_calibration.py, 200000 steps): loss
dropped sharply through step ~40000 then oscillated flat (3.58-3.90,
no further trend) for the remaining 160000 steps -- real saturation,
not continued improvement. K=1's own result from that run (acc=0.1667,
200000 steps) is reused directly here rather than rerun.

Per-step cost grows sharply with K (measured directly: K=1=0.040s,
K=2=0.135s, K=4=0.36s, K=8=1.02s/step -- longer sequences AND wider
num_tiles attention), so a uniform step count across K would make K=8
take days. Instead each K gets a step budget picked to cost roughly the
same WALL-CLOCK time (~2h), scaled inversely by its own per-step cost --
a practical overnight-budget compromise, explicitly NOT a claim that
every K reaches its own true saturation point the way K=1's calibration
run did (harder K likely needs proportionally MORE steps to cover its
own larger combinatorial key/value/gap space, which this budget does
not attempt to fully resolve).

Run: python3 scripts/mqar_calibrated_sweep.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

import scripts.train_mqar_k_sweep as m

SEED = 1000
LOG_PATH = "mqar_calibrated_sweep.log"

# (K, train_steps, eval_every) -- ~2h wall-clock budget per K, scaled by
# each K's own measured per-step cost.
PLAN = [
    (2, 50000, 5000),
    (4, 20000, 2000),
    (8, 7000, 700),
]

K1_RESULT = {"num_kv_pairs": 1, "seq_len": 4, "acc": 0.16666666666666666,
            "elapsed_s": 8103.2, "note": "reused from mqar_k1_calibration.py's 200000-step run"}


def main():
    log_file = open(LOG_PATH, "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"# Calibrated MQAR sweep, seed={SEED}, plan={PLAN}")
    log(f"K=1 (reused from prior calibration run): {K1_RESULT}")

    results = [dict(K1_RESULT, num_kv_pairs=1)]
    for k, train_steps, eval_every in PLAN:
        log(f"\n=== K={k} (seq_len={m.seq_len_for_k(k)}, train_steps={train_steps}) ===")

        def log_fn(kk, step, total, elapsed, loss, acc, _k=k):
            acc_s = f"  acc={acc:.4f}" if acc is not None else ""
            log(f"  [K={_k}] step={step:>6}/{total}  mean_query_loss={loss:.4f}{acc_s}  "
                f"({elapsed:.0f}s elapsed, {elapsed/step:.4f}s/step)")

        t0 = time.time()
        r = m.train_and_eval(k, SEED, train_steps, log_fn=log_fn,
                             pool_size=1, refresh_every=1, eval_every=eval_every)
        log(f"K={k} FINAL: {r}  (total {time.time()-t0:.0f}s)")
        results.append(r)

    log("\n# SUMMARY")
    log(f"{'K':>4}  {'seq_len':>8}  {'acc':>8}")
    for r in results:
        log(f"{r['num_kv_pairs']:>4}  {r['seq_len']:>8}  {r['acc']:>8.4f}")

    log_file.close()


if __name__ == "__main__":
    main()
