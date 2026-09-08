"""
scripts/estimate_wide_scaling.py

Pre-calculate the expected sparsity speedup at a TARGET state_width before
committing compute to actually training there. See
docs/research/estimate_wide_scaling.rst for the physical argument,
calibration-data provenance, and usage.

*ID:* ``estimate_wide_scaling.module_overview``
"""

from __future__ import annotations

import sys

COLUMN_NEURONS = 8
VOCAB = 128
CALIBRATION_WIDTH = 128  # embed_width=16 * COLUMN_NEURONS=8

# Noise-averaged shared baseline, not two separate readings.
# See docs/research/estimate_wide_scaling.rst:calibration_data
LM_HEAD_BASE_W128 = (0.0043 + 0.00086) / 2  # = 0.00258

# Per-step seconds at CALIBRATION_WIDTH, x_r_target=None (dense/no-op _to_sparse).
CALIBRATION_DENSE_W128 = {
    "input_proj": 0.01025,
    "wide_qkvo": 0.0179 + 0.02375 + 0.02405 + 0.0244,  # q+k+v+o
    "attention": 0.00985,
    "lm_head": LM_HEAD_BASE_W128,
    "other": 0.02355,  # pure orchestration -- no CSR construction happens
}

# Per-step seconds at CALIBRATION_WIDTH, x_r_target=dy_r_target=0.5 (both AT
# their floor -- steady state, not still ratcheting down).
CALIBRATION_SPARSE_W128 = {
    "input_proj": 0.00256,
    "wide_qkvo": 0.00938 + 0.01004 + 0.0097 + 0.0095,
    "attention": 0.00138,
    "lm_head": LM_HEAD_BASE_W128,
    "other": 0.04134,  # orchestration + CSR construction for 5 layers
}


def _orchestration_const() -> float:
    """The width-independent slice of "other" -- assumed equal to the
    DENSE run's other (no CSR construction happening there at all)."""
    return CALIBRATION_DENSE_W128["other"]


def _csr_overhead_at_calibration() -> float:
    """The EXTRA per-step cost sparsity's own CSR construction adds at
    CALIBRATION_WIDTH -- the real, measured "sparsity isn't free" number."""
    return CALIBRATION_SPARSE_W128["other"] - _orchestration_const()


def project_dense(width: int, vocab_exponent: float = 0.0) -> dict:
    scale2 = (width / CALIBRATION_WIDTH) ** 2
    scale1 = width / CALIBRATION_WIDTH
    scale_lm = (width / CALIBRATION_WIDTH) ** (1.0 + vocab_exponent)
    return {
        "input_proj": CALIBRATION_DENSE_W128["input_proj"] * scale2,
        "wide_qkvo": CALIBRATION_DENSE_W128["wide_qkvo"] * scale2,
        "attention": CALIBRATION_DENSE_W128["attention"] * scale1,
        "lm_head": CALIBRATION_DENSE_W128["lm_head"] * scale_lm,
        "other": _orchestration_const(),  # O(1) -- no CSR ever built
    }


def project_sparse(width: int, r_target: float = 0.5, vocab_exponent: float = 0.0) -> dict:
    """r_target only affects the DOCUMENTATION here -- the calibration
    data is specifically for r_target=0.5; using a different r_target
    changes the wide-layer reduction FACTOR (not modeled from first
    principles here, since that depends on the natural-sparsity curve's
    r-vs-dims-kept relationship, which itself may shift with width -- see
    project_natural_sparsity_curve.md's own caveat). This function
    extrapolates the MEASURED r=0.5 reduction factor to other widths at
    the SAME r_target; it does not extrapolate across r_target values.
    lm_head/critic_head are NOT currently sparsified by design (see
    JOURNAL/plan) so vocab_exponent affects them identically in both
    project_dense and project_sparse -- it changes the FIXED-COST floor
    both arms share, not the sparsity win itself."""
    if r_target != 0.5:
        raise NotImplementedError(
            "Calibration data is for r_target=0.5 only -- see docstring. "
            "Re-run the clean timing probe at a new r_target to add a new "
            "calibration point before projecting it."
        )
    scale2 = (width / CALIBRATION_WIDTH) ** 2
    scale1 = width / CALIBRATION_WIDTH
    scale_lm = (width / CALIBRATION_WIDTH) ** (1.0 + vocab_exponent)
    csr_overhead = _csr_overhead_at_calibration() * scale1
    return {
        "input_proj": CALIBRATION_SPARSE_W128["input_proj"] * scale2,
        "wide_qkvo": CALIBRATION_SPARSE_W128["wide_qkvo"] * scale2,
        "attention": CALIBRATION_SPARSE_W128["attention"] * scale1,
        "lm_head": CALIBRATION_SPARSE_W128["lm_head"] * scale_lm,
        "other": _orchestration_const() + csr_overhead,
    }


def report(width: int, vocab_exponent: float) -> None:
    dense = project_dense(width, vocab_exponent)
    sparse = project_sparse(width, vocab_exponent=vocab_exponent)
    dense_total = sum(dense.values())
    sparse_total = sum(sparse.values())
    speedup = dense_total / sparse_total
    other_frac = sparse["other"] / sparse_total
    lm_frac = sparse["lm_head"] / sparse_total
    embed_width = width // COLUMN_NEURONS
    print(
        f"width={width:>6} (embed_width={embed_width:>4})  "
        f"dense={1 / dense_total:>7.2f} sps  sparse={1 / sparse_total:>7.2f} sps  "
        f"speedup={speedup:>5.2f}x  sparse_other_frac={other_frac:>5.1%}  "
        f"sparse_lm_head_frac={lm_frac:>5.1%}"
    )


def main() -> None:
    widths = [int(a) for a in sys.argv[1:]] or [128, 256, 512, 1024, 2048, 4096, 8000]
    print(
        f"Calibrated from real W={CALIBRATION_WIDTH} measurements "
        f"(r_target=0.5, steady state). Wide layers ~O(W^2), "
        f"attention ~O(W), CSR overhead ~O(W), orchestration O(1).\n"
    )

    print("--- Scenario A: vocab_size stays fixed at 128 as width scales ---")
    print("(only valid if scaling width WITHOUT scaling task difficulty)")
    for w in widths:
        report(w, vocab_exponent=0.0)

    print("\n--- Scenario B: vocab_size scales LINEARLY with width ---")
    print("(more realistic if we scale width TO unlock more vocab capacity --")
    print(" vocab is already saturated at the current width, per task #318)")
    for w in widths:
        report(w, vocab_exponent=1.0)


if __name__ == "__main__":
    main()
