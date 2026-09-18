# MQAR mastery leaderboard: fastest setup vs. parameter count

Running log of the best (and best-documented) results toward "MQAR
mastery" -- reaching `vocab=126` (`TASK_VOCAB_MAX`) **and** `k=3`
simultaneously, held (LEVEL_DOWN disabled in every run below, so once
reached it can't regress). Two separate "fastest" axes are tracked,
since they don't always agree: **steps** to the milestone, and
**wall-clock** to the milestone (estimated from the nearest logged
`steps/sec`, not a precise stopwatch -- and NOT comparable 1:1 across
machines/contention, see Caveats).

Update this table whenever a run lands a new best, or a notable
negative result, on either axis. Link the exact commit so any entry is
reproducible.

## Current record

**state_width=288 (embed_width=36, ~351k params), dense, `peak_lr=0.01`**
-- reached `vocab=126, k=3` at **step 13,601** (~30 min wall-clock est.),
commit [`30ca81a`](https://github.com/SimLeek/sili_peridot/commit/30ca81a),
`launch_dense_lr_scaled.py`, branch `research/dense-vs-sparse-mqar-300k`.

First run in this project's history (as far as this log's coverage
goes) to reach the full `vocab=126, k=3` milestone at all -- the
previous best (`arm_nolevel_down`, state_width=128, 2026-09-07) reached
`vocab=126` but only at `k=2`, after 43,286 steps (~94 min), and never
advanced to `k=3` within its own 100k-step budget. This run beat that
on BOTH steps and wall-clock while reaching a strictly harder milestone,
at over 2x the parameter count -- direct evidence the width=288
"degeneracy" was a fixed-`peak_lr` miscalibration, not a real
capacity/architecture regression. See
`docs/research/train_mqar_curriculum.rst:train_curriculum.width_scaling_lr_fanin_hypothesis`
and `JOURNAL.md` 2026-09-18 for the full writeup.

## Table

| Date | state_width (embed_width) | ~Params | Mechanism | peak_lr | Milestone reached | Steps | Wall-clock (est.) | steps/sec | Commit | Branch | Launcher | Status/notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-07 | 128 (16) | ~70k | dense-equiv. (Group A engine, no r_target set) | 0.015 | vocab=126, **k=2** | 43,286 | ~94 min | 7.69 | [`c7c4b47`](https://github.com/SimLeek/sili_peridot/commit/c7c4b47) | `research/mqar-leveldown-vocab126` (PR #24) | ad-hoc CLI invocation, not a saved launcher script | Ran to full 100k-step budget (~217 min) at vocab=126/k=2 without ever reaching k=3 -- full mastery not achieved |
| 2026-09-18 | 288 (36) | ~351k | dense-equiv., peak_lr **unscaled** | 0.015 | vocab=64, k=3 (stuck) | -- (never reached vocab=126) | 100,000 steps / ~13,524s (~3.76h) full budget | 7.39 | [`72663b9`](https://github.com/SimLeek/sili_peridot/commit/72663b9) | `research/dense-vs-sparse-mqar-300k` | `launch_width288_nolevel_down_control.py` | Stuck at vocab=64/k=3 for its last 86,257 steps -- this run is what exposed the width-scaling degeneracy |
| **2026-09-18** | **288 (36)** | **~351k** | **dense-equiv., peak_lr scaled (alpha~=0.5)** | **0.01** | **vocab=126, k=3** | **13,601** | **~30 min (est.)** | ~7.3-7.5 | [`30ca81a`](https://github.com/SimLeek/sili_peridot/commit/30ca81a) | `research/dense-vs-sparse-mqar-300k` | `launch_dense_lr_scaled.py` | **CURRENT BEST**, both axes. Still running past this point (climbing toward k=4) as of last check |
| 2026-09-18 | 288 (36) | ~351k | dense-equiv., peak_lr scaled (alpha=1.0, muP) | 0.0067 | in progress | TBD | TBD | TBD | [`ffe0a67`](https://github.com/SimLeek/sili_peridot/commit/ffe0a67) | `research/dense-vs-sparse-mqar-300k` | `launch_dense_lr_alpha1_scaled.py` | Third data point for the LR/fan-in exponent -- update on completion |
| 2026-09-18 | 288 (36) | ~351k | dense fwd + Arm C backward gate | 0.015 | vocab=64, k=2 (stuck) | -- (never reached vocab=126) | 100,000 steps / ~14,481s (~4.02h) full budget | 6.91 | [`72663b9`](https://github.com/SimLeek/sili_peridot/commit/72663b9) | `research/dense-vs-sparse-mqar-300k` | `launch_dense_forward_arm_c_backward.py` | Predates the LR fix -- worth rerunning at peak_lr~=0.01 |
| 2026-09-18 | 288 (36) | ~351k | Arm C backward + knee-adaptive forward (4 variants: standard, skip-k1, margin=0, margin=0.15) | 0.015 | in progress (arch-sandbox, all 4 slower than vocab=126 as of last check) | TBD | TBD | ~1.0-2.1 (contended, 4-way on 16 threads) | [`72663b9`](https://github.com/SimLeek/sili_peridot/commit/72663b9) | `research/dense-vs-sparse-mqar-300k` | `launch_arm_c_plus_knee*.py` | Predates the LR fix too -- worth rerunning at peak_lr~=0.01 once these finish/are judged stuck |

## Caveats

- **Wall-clock is estimated**, not stopwatch-measured: computed from
  the `steps/sec` figure in the nearest log line to the milestone, not
  a captured timestamp. Good to within a few percent for a single run,
  not precise enough to rank two near-tied entries against each other.
- **Cross-machine/contention comparisons are not apples-to-apples.**
  Local (8-core laptop) runs alone get ~7-8.6 steps/sec; arch-sandbox
  runs sharing 16 threads 4-ways at `num_cpus=4` each get ~1-2
  steps/sec from contention alone, independent of the mechanism being
  tested (see `project_sili_arch_sandbox_ccx_topology_thread_wall`
  memory). Don't read a slow arch-sandbox wall-clock as a slow
  mechanism without checking how many jobs it was sharing the box with.
- **"~Params" is the rough formula** `4*state_width^2 +
  embed_width*state_width + embed_width*126 + 128*embed_width` (4x
  `state_width x state_width` q/k/v/o_proj + input_proj + lm_head +
  embed table), not an exact count from the actual parameter arrays.
- Historical (pre-2026-09-18) entries are backfilled from `JOURNAL.md`
  and may be missing runs that weren't as clearly documented there --
  this table's coverage starts solid from the dense-vs-sparse-mqar-300k
  investigation onward; treat anything before that as partial.
