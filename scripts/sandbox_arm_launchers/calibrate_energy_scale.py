import collections
import sys

sys.path.insert(0, ".")
import numpy as np

import scripts.train_mqar_curriculum as m

stats = collections.defaultdict(list)
_orig_to_sparse = None


def patched_to_sparse(self, x, layer_name):
    if self.x_r_target.get(layer_name) is not None and len(stats[layer_name]) < 400:
        h = np.asarray(x.data)
        stats[layer_name].append(float(np.mean(np.abs(h))))
    return _orig_to_sparse(self, x, layer_name)


from model.toy_tile_recurrence_rmt import ToyTileRecurrenceRMT

_orig_to_sparse = ToyTileRecurrenceRMT._to_sparse
ToyTileRecurrenceRMT._to_sparse = patched_to_sparse

sys.argv = [
    "train_mqar_curriculum.py",
    "fp32",
    "1500",
    "1000",
    "0.015",
    "16",
    "10",
    "1",
    "1",
    "50",
    "0",
    "0",
    "36",
    "-1",
    "-1",
    "-1",
    "0",
    "-1",
    "100000000",
    "0.9",
    "0",
    "-1",
    "95.69",
    "-1",
    "0.9",
    "0",
    "-1",
    "0.5",
]
m.main()

print("\n=== E[|h|] just before each layer's _to_sparse call (last 400 samples) ===")
for name in ["input_proj", "q_proj", "k_proj", "v_proj", "o_proj"]:
    vals = stats[name]
    if vals:
        arr = np.array(vals[-400:])
        print(
            f"{name:12s} n={len(arr):4d}  mean(|h|)={arr.mean():.5f}  median={np.median(arr):.5f}  p90={np.percentile(arr, 90):.5f}"
        )
