"""Attention-sharpness probe (direct instruction, explainable-AI investigation
into the embed_width=32 vocab ceiling): gaussian_attention's real score is

    score[q,j] = (Q[q].K[j])*scale - (j - center[q])^2 / (2*sigma[q]^2)

(sili__new/sili/lib/headers/attention.hpp:689) -- sigma directly sets how
content-driven vs. position-anchored a query's attention is. As sigma -> 0
the Gaussian bias dominates regardless of content (attention is FORCED onto
the query's own learned center, high reliability, low flexibility). As
sigma -> large the positional term vanishes and attention is free to be
driven purely by Q.K content -- more flexible, but also more surface area
for content-driven shortcuts (including the kind of boundary-hovering
behavior flagged as possible reward hacking) that a pure position-anchored
query can't develop at all.

Uses train_curriculum()'s query_debug_fn hook (added for this investigation)
to capture (correct, sigma) at EVERY real query step, using the exact same
task-generation/labeling code already validated -- not a hand-rolled probe
that risks diverging from it. Reports sigma distribution split by
correct/incorrect, for base-16 (the config that reached peak_vocab=64) vs
embed-32 (the config stuck at peak_vocab=16), same seed/steps/everything
else the same as the JOURNAL.md long_run_* comparisons.
"""
import sys
import numpy as np

from scripts.train_mqar_curriculum import train_curriculum
from scripts.train_mqar_rmt_reference import NUM_MEMORY_SLOTS


def run_probe(label: str, embed_width: int, input_sparsity_p, wide_max_weights,
              output_dy_sparsity_p, max_steps: int, seed: int = 2001) -> dict:
    records = []

    def query_debug_fn(step, correct, logit_row, last_debug):
        sigmas = last_debug.get("sigmas")
        if sigmas is None:
            return
        sigma_at_query = float(sigmas[NUM_MEMORY_SLOTS + logit_row])
        records.append((step, bool(correct), sigma_at_query))

    print(f"=== {label}: embed_width={embed_width} steps={max_steps} ===", flush=True)
    r = train_curriculum(
        "fp4", max_steps, seed, 0.015, 16, 10, log_every=max_steps + 1,
        additive_rank=1, dynamic_rank_control=True, rank_grace_period_steps=50,
        use_critic=False, recurrent_only_output=False,
        embed_width=embed_width, input_sparsity_p=input_sparsity_p,
        wide_max_weights=wide_max_weights, output_dy_sparsity_p=output_dy_sparsity_p,
        query_debug_fn=query_debug_fn)

    sigmas_all = np.array([s for _, _, s in records])
    correct_mask = np.array([c for _, c, _ in records])
    n = len(records)
    n_correct = int(correct_mask.sum())
    print(f"  {label}: {n} queries, {n_correct} correct ({100*n_correct/max(n,1):.1f}%), "
          f"peak_vocab={r['peak_stage']['vocab']} final_vocab={r['final_vocab']}", flush=True)
    if n > 0:
        print(f"  sigma (all):       mean={sigmas_all.mean():.4f} std={sigmas_all.std():.4f} "
              f"median={np.median(sigmas_all):.4f} p10={np.percentile(sigmas_all,10):.4f} "
              f"p90={np.percentile(sigmas_all,90):.4f}", flush=True)
    if n_correct > 0:
        s_correct = sigmas_all[correct_mask]
        print(f"  sigma (correct):   mean={s_correct.mean():.4f} std={s_correct.std():.4f} "
              f"median={np.median(s_correct):.4f}", flush=True)
    if n - n_correct > 0:
        s_wrong = sigmas_all[~correct_mask]
        print(f"  sigma (incorrect): mean={s_wrong.mean():.4f} std={s_wrong.std():.4f} "
              f"median={np.median(s_wrong):.4f}", flush=True)
    return {"label": label, "records": records, "peak_vocab": r["peak_stage"]["vocab"],
            "final_vocab": r["final_vocab"]}


def main():
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2001

    run_probe("base-16", embed_width=16, input_sparsity_p=None, wide_max_weights=None,
              output_dy_sparsity_p=0.5, max_steps=max_steps, seed=seed)
    print(flush=True)
    run_probe("embed-32", embed_width=32, input_sparsity_p=0.5, wide_max_weights=2048,
              output_dy_sparsity_p=0.5, max_steps=max_steps, seed=seed)


if __name__ == "__main__":
    main()
