"""
Fast, throwaway verification: does a per-SYNAPSE (not just per-row
value_scale) peak-eligibility correction, via SparseLinearLayer's real
backward_sparse (dense-input, sparse-gradient path), give genuine
credit to a row's actual WEIGHT VALUES when that row peaked several
ticks ago but is silent at the query tick -- something plain DISLDO
structurally cannot do at all (g = dy*iv = 0 when iv=0).

Minimal on purpose, per direct instruction: no attention, no MLP, no
embedding table, no tile system -- one raw recurrent DISLDOLayer cell,
tiny state width, should run in well under a second. Only worth
building the C++ kernel / wiring into the real architecture if THIS
shows the effect first.
"""

import time

import numpy as np
from sili import _cpu
from sili.sparse_rnn import DISLDOLayer

VOCAB = 3  # 0, 1, '?' -- matches model/toy_beyond_context_task.py
STATE_WIDTH = 6
IN_FEATURES = VOCAB + STATE_WIDTH
MAX_WEIGHTS = 24
NUM_CPUS = 2


class PeakSynapseCell:
    """One recurrent DISLDOLayer cell (state_t = state_t-1 + cell(x_t,
    state_t-1)), with an EXTRA per-synapse peak-eligibility correction
    on top of the layer's own normal (current-tick-only) training.

    Normal path: exactly plain DISLDOLayer -- forward_dense/
    backward_dense, trains whatever fired THIS tick, same as always.

    Extra path (`correct_from_peaks`, called explicitly after the
    normal backward): for each row whose remembered peak is
    significant, build a 1-HOT dense array (only that row nonzero,
    value = its peak) and call `backward_sparse` directly on the SAME
    underlying SparseLinearLayer object -- `backward_sparse` takes the
    dense input EXPLICITLY as an argument (cpu_backend.cpp:476-489),
    not cached, so no last_input mutation trick needed here. Since
    each synapse's own gradient is `g = dy*iv` (row-local, confirmed
    directly from linear_disldo.hpp), zeroing every OTHER row in the
    1-hot array means ONLY this row's synapses get a nonzero, genuine
    per-synapse update -- real credit assignment, not a value_scale
    homeostatic knob."""

    def __init__(self, in_features, out_features, max_weights, num_cpus=2, peak_decay=0.9):
        self.layer = DISLDOLayer(in_features, out_features, max_weights, num_cpus)
        self.in_features = in_features
        self.peak_decay = peak_decay
        self.peak = np.zeros(in_features, dtype=np.float32)  # signed
        self.peak_mag = np.zeros(in_features, dtype=np.float32)  # |peak|

    def update_peak(self, x_row: np.ndarray) -> None:
        x_row = np.asarray(x_row, dtype=np.float32)
        decayed_val = self.peak_decay * self.peak
        decayed_mag = self.peak_decay * self.peak_mag
        replace = np.abs(x_row) > decayed_mag
        self.peak = np.where(replace, x_row, decayed_val)
        self.peak_mag = np.where(replace, np.abs(x_row), decayed_mag)

    def forward(self, x, learning_rate=0.0):
        self.update_peak(x[0] if x.ndim == 2 else x)
        return self.layer.forward(x, learning_rate)

    def correct_from_peaks(
        self, dy: np.ndarray, learning_rate: float, threshold: float = 0.3, backprop_p: float = 1.0
    ) -> int:
        """Call AFTER the normal backward has already fired. Returns
        the number of rows corrected (for reporting)."""
        dy = np.asarray(dy, dtype=np.float32)[np.newaxis, :]
        k = max(1, int(dy.shape[1] * backprop_p))
        dp, di, dv = _cpu.dense_to_top_k_csr(dy, k, NUM_CPUS)
        n_corrected = 0
        for r in range(self.in_features):
            if self.peak_mag[r] < threshold:
                continue
            x_1hot = np.zeros((1, self.in_features), dtype=np.float32)
            x_1hot[0, r] = self.peak[r]
            self.layer._c.backward_sparse(x_1hot, dp, di, dv, 1, learning_rate, lr_per_row_nnz=True)
            n_corrected += 1
        return n_corrected


def get_row_weight_snapshot(layer: DISLDOLayer, row: int) -> np.ndarray:
    """Sum of |stored weight values| touching this row's own
    connections -- a real per-SYNAPSE quantity (not value_scale),
    computed from the CSR structure (ptrs/indices index into the flat
    weights_vals array)."""
    ptrs = np.asarray(layer._c.ptrs)
    np.asarray(layer._c.indices)
    vals = np.asarray(layer._c.weights_vals)
    lo, hi = ptrs[row], ptrs[row + 1]
    return vals[lo:hi].copy()


def main():
    t0 = time.time()
    np.random.RandomState(0)

    cell = PeakSynapseCell(IN_FEATURES, STATE_WIDTH, MAX_WEIGHTS, NUM_CPUS)
    M = np.zeros(STATE_WIDTH, dtype=np.float32)

    def onehot(tok):
        v = np.zeros(VOCAB, dtype=np.float32)
        v[tok] = 1.0
        return v

    # Tick 0: token '1' (strong, distinctive input) -- row for that
    # onehot slot peaks here.
    x0 = np.concatenate([onehot(1), M])[None, :]
    out0 = cell.forward(x0, learning_rate=0.0)
    M = out0.data[0]  # M_new = cell output directly (toy: no residual, keep it minimal)

    # Ticks 1-3: token '0' every time -- the '1'-onehot row (index 1)
    # is now silent (its own onehot slot = 0.0) for the rest of the
    # sequence, exactly the credit-assignment gap under test.
    for _ in range(3):
        x = np.concatenate([onehot(0), M])[None, :]
        out = cell.forward(x, learning_rate=0.0)
        M = out.data[0]

    row_before = get_row_weight_snapshot(cell.layer, 1).copy()  # row 1 = the '1'-token onehot slot
    vs_before = cell.layer._c.get_value_scale(1)
    ptrs = np.asarray(cell.layer._c.ptrs)
    print(
        f"row 1 has {ptrs[2] - ptrs[1]} connection(s), to columns {np.asarray(cell.layer._c.indices)[ptrs[1] : ptrs[2]]}"
    )
    peak_row1 = cell.peak[1]
    print(f"row 1 (the '1'-token input) remembered peak: {peak_row1:.3f}")
    print(f"row 1 weight values BEFORE any correction: {row_before}, value_scale: {vs_before:.6f}")

    # Query tick: token '?' (query), row 1 (the '1' onehot slot) is
    # STILL silent here too (this tick's token is '?', not '1').
    x_query = np.concatenate([onehot(2), M])[None, :]
    out_query = cell.forward(x_query, learning_rate=0.05)
    out_query.grad = np.array([1.0, -1.0, 0.3], dtype=np.float32)  # a real, nonzero error signal
    out_query.backward()

    row_after_normal = get_row_weight_snapshot(cell.layer, 1).copy()
    print(f"row 1 weight values AFTER normal DISLDO backward (row silent -> should be UNCHANGED): {row_after_normal}")
    print(f"  unchanged: {np.allclose(row_before, row_after_normal)}")

    # NOW the extra per-synapse peak correction, using the SAME real
    # error that just fired (out_query.grad). Repeated + a large lr
    # for this diagnostic specifically, to disambiguate "mechanism
    # doesn't fire at all" from "fires but too small to cross an FP4
    # quantization boundary in one step".
    for _i in range(20):
        n = cell.correct_from_peaks(out_query.grad, learning_rate=1.0, threshold=0.3)
    row_after_peak_correction = get_row_weight_snapshot(cell.layer, 1).copy()
    vs_after = cell.layer._c.get_value_scale(1)
    print(f"rows corrected via peak mechanism (last call): {n}")
    print(
        f"row 1 weight values AFTER 20x peak correction (lr=1.0): {row_after_peak_correction}, value_scale: {vs_after:.6f}"
    )
    print(f"  weight changed from normal-only: {not np.allclose(row_after_normal, row_after_peak_correction)}")
    print(f"  value_scale changed: {vs_after != vs_before}")
    print(f"  finite: {np.all(np.isfinite(row_after_peak_correction))}")

    print(f"\ntotal time: {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
