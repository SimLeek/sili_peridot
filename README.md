# sili_peridot

Research project converting dense transformer models (ultimately
`MiniCPM5-1B-Base`) into sparse, recurrent models trained entirely on
[`sili__new`](https://github.com/SimLeek/sili__new) (no PyTorch at
inference). This repo is where that approach gets proven out on toy
tasks before scaling up -- see `todolist.md` for the full MiniCPM5
conversion plan.

## Main entry point

```
python3 scripts/train_peridot_main.py [max_steps] [seed]
```

This is the project's current best-known-working recipe: sparse DISLDO
layers trained on MQAR (multi-query associative recall), with the
magnitude/plasticity regularization this project found necessary to
keep a 100k+ step run from stalling out. It demonstrates, together:

- **Sparsity**: DISLDO's own amortized per-synapse update rule
  ("importance is already the optimizer" -- no separate Adam-style
  state needed per weight).
- **Magnitude plasticity**: QKVO-Norm + CiRenorm + WeightRenorm +
  amortized `l2_decay`, keeping weight/importance scale calibrated for
  the whole run without manual intervention.
- **Directional plasticity**: a purely intrinsic "sleep" annealing
  mechanism (`sleep_gate_layers`) -- `EnergyDynamics`' forced-firing
  gated reactively on whether `q_proj`/`k_proj`'s own `col_importance`
  has gone static (zero delta between amortized cycles), with no
  curriculum/task information and no blind schedule. It turns itself
  off the moment the layer starts moving again.

**Result**: this recipe reached `vocab=126, k=7` on MQAR (seed 1001).
`k=5` is the highest level solvable from pure in-context lookup -- the
model's 16-token attention window structurally cannot span the
key-query gap `k=6`/`k=7` require (see JOURNAL.md's "k=4/k=5 in-context
ceiling" entry for the exact derivation). Reaching `k=6`/`k=7` means
the model is genuinely using its 2-slot recurrent memory with real
superposition, not getting lucky at the in-context ceiling.

Full research narrative -- including the mechanisms that were tried
and didn't work, and why -- is in `JOURNAL.md`. Fastest-to-mastery
results across every tried config are tracked in `MQAR_LEADERBOARD.md`.

## Setup

```
python3 -m venv --system-site-packages .venv_peridot
source .venv_peridot/bin/activate
pip install -e /path/to/sili__new   # sili_peridot has no runtime deps of its own
```

Run the fast test suite before committing:

```
pytest tests/ -k "not integration"
```

## Known issues (tracked, not yet fixed)

- `sili__new`'s `sili/cpu.py` generic elementwise `Backend.mul` can
  overflow in fp32 during very long runs (observed once, late into an
  extended k-level plateau under `train_peridot_main.py`) without
  raising -- silent, not yet root-caused. Didn't visibly corrupt the
  run it occurred in, but is a real correctness gap worth closing
  before scaling up. See JOURNAL.md's 2026-10 entry.

## Roadmap (not started)

- Scale this recipe up with FP4 precision and harder MQAR tasks on a
  rented GPU (RTX 5090 class). Before scaling, re-run the same kind of
  small-model ablation testing used to find this recipe -- don't
  assume what worked at this scale ports unchanged.
- The sleep mechanism's cadence here is tied to training steps; a
  real-time deployment (e.g. a robot that should sleep on a wall-clock
  schedule, not "every N gradient steps") will likely need the
  steps/lr/gradient-sparsity balance re-tuned, not just the sleep gate
  itself.
- See `todolist.md` for the full MiniCPM5 conversion plan this toy-task
  work is building toward.
