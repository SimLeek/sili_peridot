"""See docs/research/train_toy_beyond_context_peak_eligibility_only.rst:module_overview."""

from __future__ import annotations

import time

import scripts.train_toy_beyond_context_comparison as base
from model.toy_precision_models import PeakEligibilityDISLDOLayer
from scripts.train_toy_beyond_context_comparison import EVAL_N_VALUES, W

# See docs/research/train_toy_beyond_context_peak_eligibility_only.rst:reduced_budget_monkeypatch.
base.TRAIN_STEPS = 36_000
base.WARMUP_STEPS = 1_200
base.STEPS_PER_LEVEL = 1_800

_train_tile = base._train_tile
evaluate_tile = base.evaluate_tile

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
