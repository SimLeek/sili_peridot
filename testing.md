# Testing sili_peridot

## Before comparing against the reference commits (af8ceb2 / c653df8)

sili__new commit `54e29b5` (paired with this repo's `ebab9d0`) fixes a
real, previously-present sign-discarding bug in `linear_disldo.hpp`'s
`quant_floor` (was unconditionally `zero_escape_eps + |quant|` for
EVERY synapse, discarding `quant`'s real sign even when nonzero -- not
just at the intended `quant==0` escape-from-zero case; now gated to
`quant == 0` only, real signed `quant` used otherwise). This bug
predates and was present THROUGHOUT the `af8ceb2`/`c653df8` landmark
run and everything since, up until this session. **A numeric
difference against those reference commits is not automatically a
regression** -- it may simply be backprop correctly handling negative
weights for the first time. If a number moved, check the DIRECTION and
magnitude of the change (a sign-correctness fix should generically
help or be neutral, not cause wholesale collapse) before treating it
as a red flag.

Quick reference for verifying changes after a commit. Run these from
the repo root (`sili_peridot/`) with `PYTHONPATH=.` (or from inside the
repo, since several scripts do `sys.path.insert(0, ".")` themselves)
using the shared venv (`/home/simleek/claude_code/.venv/bin/python`).
sili_peridot depends on sili__new -- rebuild that first if you've
touched anything under `sili__new/sili/` (see sili__new/testing.md).

## 1. Standard pytest suite

```bash
/home/simleek/claude_code/.venv/bin/python -m pytest tests/ -q
```

Covers checkpointing, curriculum, pruning, quantization, tile
recurrence, precision models, and dense sweeps. This is the primary
regression gate -- run it before and after any change to
`model/*.py` or `sili` itself.

## 2. l1_sparsity_probe.py -- the L1-sparsity landmark result + zero/empty-init arms

`scripts/l1_sparsity_probe.py` is both a reference script (its module
docstring documents the established landmark result: L1-sparsity alone,
coef=0.05/0.07, mean=1.0000 across 5 seeds on the 15000-step
out-of-context curriculum) and the home of `OriginalArchModel`, which
every other script in this list imports and builds on.

Quick reproduction of the landmark result (takes a while -- this is
the __main__ block):
```bash
PYTHONPATH=. /home/simleek/claude_code/.venv/bin/python scripts/l1_sparsity_probe.py
```

Fast sanity check (seconds, not the full 15k-step run) -- useful after
touching `OriginalArchModel`, `step()`, energy wiring, or synaptogenesis:
```python
from scripts.l1_sparsity_probe import OriginalArchModel, run, evaluate
for use_energy in (False, True):
    for empty_init in (False, True):
        model = OriginalArchModel(1000, dense=not empty_init, o_proj_coef=0.0,
                                   all_layer_coef=0.0, l1_sparsity_coef=0.05,
                                   use_energy=use_energy, empty_init=empty_init, synap_k=3)
        run(model, 50, 1000, verbose=False)
        acc = evaluate(model, 20, 1000)
        print(f"use_energy={use_energy} empty_init={empty_init}: OK, eval_acc={acc:.4f} "
              f"n_taps_active={len(model.energies)}")
```
Should complete without exceptions; `n_taps_active` should be 0 when
`use_energy=False` and 7 when `use_energy=True` (one EnergyDynamics
instance per entry in `OriginalArchModel._ENERGY_TAPS`: input, q, k,
v, attn, raw, logits -- energy is applied uniformly to every
neuron-producing tensor, not a hand-curated subset).

## 3. landmark_checklist.py -- multi-config, multi-seed regression suite

```bash
PYTHONPATH=. /home/simleek/claude_code/.venv/bin/python scripts/landmark_checklist.py
```

Runs `CONFIGS` (baseline, baseline_energy, baseline_zeroinit,
zeroinit_energy, zeroinit_rank1/2, zeroinit_energy_rank1/2) at 5 seeds
x 15000 steps each and compares against the `REFERENCE` dict of known
numbers. **This is slow (the original all_zero_init-based zero-init
arms took ~3 hours)** -- see item 5 below for the much faster
empty-CSR-based alternative that's replacing it for zero-weight-init
testing specifically. `all_zero_init` (a dense grid pre-loaded at
weight=0/importance=1) and `empty_init` (genuinely zero connections,
grown via real synaptogenesis) are DIFFERENT, non-interchangeable
mechanisms -- see sili__new/testing.md's "Known findings" section.

## 4. rank2_baseline_probe.py -- isolated rank=2 vs rank=1 comparison

```bash
PYTHONPATH=. /home/simleek/claude_code/.venv/bin/python scripts/rank2_baseline_probe.py
```

Standalone (doesn't modify `landmark_checklist.py`'s `CONFIGS`, which
is read once at import time) -- tests `baseline_rank2`/
`baseline_energy_rank2` at 5 seeds x 15000 steps, non-zero-init.
Reference numbers from this session: `baseline_rank2` eval_acc_mean
0.1980 (regression vs rank1's 0.3180); `baseline_energy_rank2`
eval_acc_mean 0.1240 (improvement vs rank1's 0.0860).

## 5. empty_init_synaptogenesis_ladder.py -- fast graduated tier ladder (preferred for zero-weight-init work)

```bash
PYTHONPATH=. /home/simleek/claude_code/.venv/bin/python scripts/empty_init_synaptogenesis_ladder.py <tier>
# tier in {1, 2, 4, 8}
```

The genuine zero-weight-init design (empty CSR + real synaptogenesis
growth, NOT the `all_zero_init` dense-preload hack) at graduated,
honestly-measured wall-clock scales:
- `1`: ~1-2 min, 1 seed, 8000 steps -- smoke test, does growth+escape
  happen in the full 5-layer model without crashing.
- `2`: ~2-3 min, 3 seeds, 5500 steps each -- is escape-from-zero
  reproducible across seeds.
- `4`: ~4 min, 3 seeds, 12000 steps each (past curriculum-complete,
  ~1500 steps) -- does accuracy exceed a genuinely untrained (0-step)
  reference.
- `8`: ~8 min, 5 seeds x 15000 steps -- matches the established
  landmark scale exactly, first real head-to-head vs
  baseline/baseline_energy.

**Per direct correction, do not trust these minute labels as a
guarantee** -- they were calibrated by actually running and measuring,
not estimated from an early per-step rate (synaptogenesis grows the
network's live connectivity over a run, so per-step cost is NOT
constant; a naive "100 steps took Xms, so N steps takes N*Xms"
estimate undershoots real wall time -- confirmed directly: 8000 steps
projected at ~53s actually took 108.7s). If a tier drifts far from its
label on a different machine/config, shrink ITS OWN step/seed count
for next time -- do not interrupt a run mid-flight for a time budget,
a truncated run doesn't give a valid result.

Tiers 16/32/64 are NOT YET built -- reserved for the recurrent-scale
model (`ToyTileRecurrenceRealFP4`/`train_tile_curriculum.py`), which
does not yet have `empty_init`/synaptogenesis wired in at all (only
`OriginalArchModel` does, as of this session).

## 6. zeroinit_minimal_repro.py -- fast pre-flight check before a full training run

```bash
PYTHONPATH=. /home/simleek/claude_code/.venv/bin/python scripts/zeroinit_minimal_repro.py
```

3 seeds x 1500 steps x 100-eval, compares untrained vs trained
(rank1/rank2), all zero-init/no-energy. Built to catch a real
degenerate-output failure (predictions bit-identical to untrained)
cheaply before sinking hours into a full run -- rerun this FIRST
whenever touching the zero-init/rank-N/energy code paths, before
launching anything at landmark scale. Uses `evaluate(..., verbose=True)`
to print the last 5 (pred, target) pairs -- a bare accuracy number
can't distinguish "the model is degenerate/guessing a narrow set of
tokens" from "these particular held-out sequences happened to skew
toward one target."

## General diagnostic pattern used throughout this session

For any "is this layer actually training or just producing noise"
question, read the STORED WEIGHT directly rather than trusting
accuracy alone -- pure matmul layers (no bias term) cannot produce
constant/input-independent output unless something is genuinely
broken, so a direct weight probe is a stronger, more precise diagnostic:

```python
import numpy as np
from sili.tensor import Tensor

def max_abs_w(layer):  # layer = a TrueMultiDigitLayer (q_proj, o_proj, lm_head, ...)
    mx = 0.0
    for digit in layer.digits:
        n_in = digit._c.n_inputs
        eye = np.eye(n_in, dtype=np.float32)
        out = digit.forward(Tensor(eye), 0.0)
        mx = max(mx, float(np.max(np.abs(np.asarray(out.data)))))
    return mx
```

Always sanity-check a probe like this against a KNOWN-nonzero
(non-zero-init) control model first -- a probe that reads 0 for
EVERYTHING, including a control that should obviously be nonzero, is a
broken probe, not a broken model.
