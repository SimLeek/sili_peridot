"""
sili_peridot/scripts/bench_dual_matrix_layers.py
──────────────────────────────────────────────────
Benchmark (not a pytest integration test, matching sili_v_torch.md's own
convention): for each real per-role weight tensor in the pruned, real
MiniCPM5-1B-Base checkpoint, build both plain disldo (existing production
`_cpu.SparseLinearLayer`, same construction convention as
`model/sili_block.py`'s `_build_step_layer_from_arrays` -- row=input) and
the new dual-matrix layer (`dual_layer_proto` from sili__new's
`prototypes/sili_ell/`, forward-only prototype -- row=output, its own
natural orientation) on the SAME real weights, time forward pass on both,
report per-role speedup.

Answers directly: "what speedup does the dual-matrix design get on
different parts of the real converted 1B model?" -- see conversation for
the design (y = disldo(x) + block4(x), a locally-dense 4x4-tile companion
matrix for sili's existing per-synapse disldo, validated on synthetic
uniform-random data down to ~10% local fill before this benchmark).

Requires `dual_layer_proto*.so` built in sili__new/prototypes/sili_ell/
(see that directory's own build commands in the conversation/commit
history -- not yet part of the stable sili._cpu extension).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sili__new', 'prototypes', 'sili_ell'))

import numpy as np

from sili import _cpu
import dual_layer_proto as dl

from model.checkpoint import load_minicpm5_checkpoint
from model.prune import prune_state_dict_by_role, DEFAULT_TARGET_SPARSITY_BY_ROLE

REAL_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'MiniCPM5-1B-Base')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'dual_matrix_vs_disldo.md')
NUM_CPUS = 8
MIN_FILL_FRAC = 0.10  # this machine's measured breakeven -- see sili__new/prototypes/sili_ell/BLOCK4_NOTES.md


def dense_mask_to_csr_vectorized(mask: np.ndarray, W: np.ndarray, n_rows: int):
    """Fully vectorized (no Python loop over nnz or over rows) -- the
    original dict-based version took a Python for-loop over every nonzero
    element, intractable for a real 130560x1536 tensor at 70% density
    (~140M nonzeros -- would have taken well over the 280s timeout, found
    empirically before this rewrite, not assumed)."""
    rows, cols = np.nonzero(mask)
    vals = W[rows, cols].astype(np.float32)
    order = np.lexsort((cols, rows))  # sort by row, then col within row
    rows, cols, vals = rows[order], cols[order], vals[order]
    ptrs = np.searchsorted(rows, np.arange(n_rows + 1)).astype(np.uint32)
    return ptrs, cols.astype(np.uint32), vals


def bench_tensor(name, W: np.ndarray, reps=20):
    """W: [n_out, n_in] dense (already pruned -- zeros are the sparsity).
    Returns a dict of results, or None if too dense/small to be meaningful."""
    n_out, n_in = W.shape
    n_in_pad = ((n_in + 3) // 4) * 4
    n_out_pad = ((n_out + 3) // 4) * 4
    if n_in_pad != n_in or n_out_pad != n_out:
        return None  # skip non-multiple-of-4 shapes for this forward-only prototype

    mask = W != 0
    density = float(mask.mean())
    nnz = int(mask.sum())
    if nnz == 0:
        return None

    # Natural (row=out, col=in) CSR -- dual_layer_proto's own convention.
    ptrs_out, idx_out, w_out = dense_mask_to_csr_vectorized(mask, W, n_out)
    imp_out = np.full_like(w_out, 0.5, dtype=np.float32)

    # Transposed (row=in, col=out) CSR -- disldo's own convention, matching
    # model/sili_block.py's real construction path exactly.
    ptrs_in, idx_in, w_in = dense_mask_to_csr_vectorized(mask.T, W.T, n_in)

    # ---- plain disldo (matches sili_block.py's _build_step_layer_from_arrays) ----
    layer = _cpu.SparseLinearLayer(n_in, n_out, int(nnz * 1.3) + 64, NUM_CPUS)
    vals = w_in.copy()
    row_of_nnz = np.repeat(np.arange(n_in, dtype=np.int64), np.diff(ptrs_in))
    abs_vals = np.abs(vals)
    row_max = np.zeros(n_in, dtype=np.float32)
    np.maximum.at(row_max, row_of_nnz, abs_vals)
    row_scales = np.ones(n_in, dtype=np.float32)
    nonzero_max = row_max > 0
    row_scales[nonzero_max] = row_max[nonzero_max] / 6.0
    vals = vals / row_scales[row_of_nnz]
    layer.load_weights(ptrs_in.astype(np.int32), idx_in.astype(np.int32), vals)
    for r in np.nonzero(row_scales != 1.0)[0]:
        layer.set_value_scale_raw(int(r), float(row_scales[r]))

    x = np.random.randn(n_in).astype(np.float32)
    layer.forward_dense(x, 0.0)  # warm
    best = 1e18
    for _ in range(reps):
        t0 = time.perf_counter()
        layer.forward_dense(x, 0.0)
        best = min(best, time.perf_counter() - t0)
    t_disldo = best

    # ---- dual-matrix (natural orientation, no row_scale -- see conversation's
    # open point: this prototype has no per-row calibration yet, unlike disldo's
    # real value_scale above; fine for FP4-native-range weights, a real quality
    # comparison on real small-magnitude weights would need this added first) ----
    dual = dl.build_dual_layer(ptrs_out, idx_out, w_out, imp_out,
                               n_in=n_in, n_out=n_out, min_fill_frac=MIN_FILL_FRAC, num_cpus=NUM_CPUS)
    dual.forward(x)  # warm
    best = 1e18
    for _ in range(reps):
        t0 = time.perf_counter()
        dual.forward(x)
        best = min(best, time.perf_counter() - t0)
    t_dual = best

    return dict(
        name=name, shape=f"{n_out}x{n_in}", density=density, nnz=nnz,
        t_disldo_ms=t_disldo * 1e3, t_dual_ms=t_dual * 1e3,
        speedup=t_disldo / t_dual,
        a_nnz=dual.a_nnz(), b_nnz=dual.b_nnz(),
        b_frac=dual.b_nnz() / max(nnz, 1),
    )


def main():
    print("Loading + pruning real checkpoint...")
    sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
    sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
    del sd

    # One representative tensor per role (layer 0), plus embed_tokens/lm_head.
    targets = []
    for k in sparse_state:
        if '.layers.0.' in k and any(s in k for s in
            ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            targets.append(k)
    for k in sparse_state:
        if 'embed_tokens' in k or 'lm_head' in k:
            targets.append(k)

    results = []
    for key in targets:
        entry = sparse_state[key]
        W = (entry['csr'].to_dense() if 'csr' in entry else entry['raw']).numpy().astype(np.float32)
        print(f"benchmarking {key} shape={W.shape} density={(W!=0).mean():.4f} ...")
        r = bench_tensor(key, W)
        if r is None:
            print(f"  skipped (non-multiple-of-4 shape or empty)")
            continue
        results.append(r)
        print(f"  disldo={r['t_disldo_ms']:.3f}ms  dual={r['t_dual_ms']:.3f}ms  "
              f"speedup={r['speedup']:.2f}x  b_frac={r['b_frac']:.2%}")

    with open(REPORT_PATH, 'w') as f:
        f.write("# Dual-matrix (disldo + dense-4x4-block) vs plain disldo: real MiniCPM5-1B-Base layers\n\n")
        f.write(f"Forward pass only (no backward/training in this prototype yet). "
                f"`min_fill_frac={MIN_FILL_FRAC}` (this machine's measured breakeven, "
                f"see sili__new/prototypes/sili_ell/BLOCK4_NOTES.md -- CPU- and "
                f"data-distribution-specific, not a universal constant). "
                f"num_cpus={NUM_CPUS}. One representative tensor per role (layer 0), "
                f"plus embed_tokens/lm_head, from the real, already-pruned checkpoint "
                f"(B3's `DEFAULT_TARGET_SPARSITY_BY_ROLE`).\n\n")
        f.write("| role | shape | density | nnz | disldo ms | dual ms | speedup | %nnz in block4 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['name']} | {r['shape']} | {r['density']:.4f} | {r['nnz']} | "
                    f"{r['t_disldo_ms']:.3f} | {r['t_dual_ms']:.3f} | {r['speedup']:.2f}x | "
                    f"{r['b_frac']:.1%} |\n")
        f.write("\nNo per-row value_scale calibration in the dual-matrix path yet "
                "(disldo's own comparison column DOES use it, matching real "
                "production usage) -- these weights are the real pruned checkpoint's "
                "own FP4-native-ish magnitudes, not verified small enough to need it "
                "for a fair speed comparison, but a real quality comparison would need "
                "this added first (see conversation).\n")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == '__main__':
    main()
