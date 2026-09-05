"""
scripts/train_toy_beyond_context_peak_eligibility_only.py
────────────────────────────────────────────────────────────
Runs ONLY the new peak-eligibility arm (PeakEligibilityDISLDOLayer) of
the Tier 1 beyond-context comparison, at a REDUCED step budget.

Replaces the earlier e-prop-only script -- e-prop (both the plain and
Adam variants) was found structurally flawed (see JOURNAL.md's
postmortem: its delta-trick gradient proxy is provably zero for a row
silent at the query tick, which is exactly the row this mechanism most
needs to credit). PeakEligibilityDISLDOLayer replaces it: instead of
any Python-side gradient approximation, it substitutes a peak-held
(signed) value directly into SparseLinearLayer's own `last_input`
buffer before backward fires, so DISLDO's REAL C++ gradient math
computes the correction -- worked out directly with the user.

Same reduced-budget rationale as before: the 120k-step (~1hr) scale-up
was compensating for the no-BPTT/no-gradient-pathway problem;
peak-eligibility exists to provide exactly that missing pathway, so it
should not need the same 40x scale-up to show a signal.

The dense/tile/tile+energy numbers below are from the PRIOR
120,000-step run (JOURNAL.md, "Tier 1, ~1hr budget (40x steps)") and
are shown for CONTEXT ONLY -- different step budget, not a strict
equal-budget comparison.

Run: python -m scripts.train_toy_beyond_context_peak_eligibility_only
"""

from __future__ import annotations

import time

import scripts.train_toy_beyond_context_comparison as base
from model.toy_precision_models import PeakEligibilityDISLDOLayer
from scripts.train_toy_beyond_context_comparison import EVAL_N_VALUES, W

# Reduced budget: 30% of the 120k run, same curriculum ratios/shape as
# the earlier e-prop probe (WARMUP_STEPS/TRAIN_STEPS = 1/30,
# STEPS_PER_LEVEL/TRAIN_STEPS = 1/20). Monkeypatched onto the imported
# module since _train_tile/_sample_n_bits read these as module globals.
base.TRAIN_STEPS = 36_000
base.WARMUP_STEPS = 1_200
base.STEPS_PER_LEVEL = 1_800

_train_tile = base._train_tile
evaluate_tile = base.evaluate_tile

# From JOURNAL.md, "Tier 1, ~1hr budget (40x steps)" -- TRAIN_STEPS=120_000
# (3.3x this run's budget), same curriculum shape, seeds 1/2/3 for
# dense/tile-no-energy/tile+energy. Shown for CONTEXT ONLY (see docstring).
RECORDED_DENSE = {2: 0.53, 4: 0.38, 8: 0.45, 16: 0.63, 24: 0.53}
RECORDED_TILE_NO_ENERGY = {2: 0.72, 4: 0.45, 8: 0.43, 16: 0.47, 24: 0.15}
RECORDED_TILE_ENERGY = {2: 0.45, 4: 0.30, 8: 0.43, 16: 0.35, 24: 0.50}


def main():
    print(
        f"TRAIN_STEPS={base.TRAIN_STEPS} (30% of the 120k run) "
        f"warmup={base.WARMUP_STEPS} steps_per_level={base.STEPS_PER_LEVEL}\n"
    )

    t0 = time.time()
    tile_peak, tile_peak_embed, tile_peak_rng = _train_tile(PeakEligibilityDISLDOLayer, use_energy=False, seed=4)
    t1 = time.time()
    tile_peak_results = evaluate_tile(tile_peak, tile_peak_embed, tile_peak_rng)
    print(f"tile, peak-eligibility trained ({t1 - t0:.1f}s)\n")

    print(f"{'n_bits':>8}  {'in_ctx':>7}  {'dense*':>7}  {'tile*':>7}  {'tile+egy*':>10}  {'peak-elig':>10}")
    for n_bits in EVAL_N_VALUES:
        in_ctx = "yes" if n_bits <= W else "NO"
        print(
            f"{n_bits:>8}  {in_ctx:>7}  {RECORDED_DENSE[n_bits]:>7.2f}  "
            f"{RECORDED_TILE_NO_ENERGY[n_bits]:>7.2f}  {RECORDED_TILE_ENERGY[n_bits]:>10.2f}  "
            f"{tile_peak_results[n_bits]:>10.2f}"
        )
    print("\n(chance = 0.5 for a single binary answer bit)")
    print(
        "(* = recorded from the prior 120,000-step run -- 3.3x this run's "
        "budget, context only, not equal-budget -- see JOURNAL.md "
        "'Tier 1, ~1hr budget (40x steps)')"
    )
    return tile_peak_results


if __name__ == "__main__":
    main()
