import sys
import time

sys.path.insert(0, ".")
import numpy as np
from sili.sparse_rnn import DISLDOLayer32
from sili.tensor import Tensor, reduce_sum

LAYER_SHAPES = {
    "input_proj (36->288)": (36, 288),
    "q/k/v/o_proj (288->288)": (288, 288),
}
N_CALLS = 4500
NUM_CPUS = 4


def bench(n_in, n_out, n_calls=N_CALLS, dense=True):
    rng = np.random.default_rng(0)
    layer = DISLDOLayer32(n_in, n_out, n_in * n_out, NUM_CPUS, rng=rng, dense=dense)
    x_np = rng.standard_normal(n_in).astype(np.float32)
    # warmup
    for _ in range(20):
        x = Tensor(x_np.copy())
        y = layer.forward(x, learning_rate=1e-3)
        loss = reduce_sum(y, axis=None)
        loss.backward()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        x = Tensor(x_np.copy())
        y = layer.forward(x, learning_rate=1e-3)
        loss = reduce_sum(y, axis=None)
        loss.backward()
    t1 = time.perf_counter()
    total = t1 - t0
    return total, total / n_calls


print(f"sili dense benchmark, NUM_CPUS={NUM_CPUS}, N_CALLS per shape: {N_CALLS}\n")
for name, (n_in, n_out) in LAYER_SHAPES.items():
    total, per_call = bench(n_in, n_out, dense=True)
    print(
        f"  {name} [DENSE]: {total:.3f}s total / {N_CALLS} calls = {per_call * 1000:.4f} ms/call ({1 / per_call:.1f} calls/sec)"
    )
print()
for name, (n_in, n_out) in LAYER_SHAPES.items():
    total, per_call = bench(n_in, n_out, dense=False)
    print(
        f"  {name} [SPARSE init, no r_target]: {total:.3f}s total / {N_CALLS} calls = {per_call * 1000:.4f} ms/call ({1 / per_call:.1f} calls/sec)"
    )
