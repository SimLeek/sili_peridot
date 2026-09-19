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
| **2026-09-18** | **288 (36)** | **~351k** | **dense-equiv., peak_lr scaled (alpha~=0.5)** | **0.01** | **vocab=126, k=3** | **13,601** | **~30 min (est.)** | ~7.3-7.5 (final 6.95) | [`30ca81a`](https://github.com/SimLeek/sili_peridot/commit/30ca81a) | `research/dense-vs-sparse-mqar-300k` | `launch_dense_lr_scaled.py` | **CURRENT BEST**, both axes. Run completed (100k/100k steps, 14,399s total): held vocab=126/k=3 for the remaining 86,399 steps with no regression, never reached k=4 -- consistent with k=3 being this config's expected architectural ceiling, not a new stall |
| 2026-09-18 | 288 (36) | ~351k | dense-equiv., peak_lr scaled (alpha=1.0, muP) | 0.0067 | vocab=32, k=3 (stuck) | -- (never reached vocab=126) | 100,000 steps / 14,068s (~3.91h) full budget | 7.11 | [`ffe0a67`](https://github.com/SimLeek/sili_peridot/commit/ffe0a67) | `research/dense-vs-sparse-mqar-300k` | `launch_dense_lr_alpha1_scaled.py` | **Worse than the unscaled 0.015 baseline** (which reached vocab=64) -- reached vocab=32/k=3 by step 10,283 then flat for 89,717 steps. Third LR data point reveals a real sweet spot, not "lower is always better": 0.015 too high, 0.0067 too low, 0.01 near-optimal |
| 2026-09-18 | 288 (36) | ~351k | dense fwd + Arm C backward gate | 0.015 | vocab=64, k=2 (stuck) | -- (never reached vocab=126) | 100,000 steps / ~14,481s (~4.02h) full budget | 6.91 | [`72663b9`](https://github.com/SimLeek/sili_peridot/commit/72663b9) | `research/dense-vs-sparse-mqar-300k` | `launch_dense_forward_arm_c_backward.py` | Predates the LR fix -- worth rerunning at peak_lr~=0.01 |
| 2026-09-18 | 288 (36) | ~351k | Arm C backward + knee-adaptive forward, **standard curriculum (K_START=1, margin=0.05)** | 0.015 | vocab=32, k=1 (stuck) | -- (never reached vocab=126) | 100,000 steps / 48,301s (~13.4h) full budget | 2.07 (contended, arch-sandbox 4-way) | [`72663b9`](https://github.com/SimLeek/sili_peridot/commit/72663b9) | `research/dense-vs-sparse-mqar-300k` | `launch_arm_c_plus_knee.py` | Weakest completed result so far: reached vocab=32/k=1 at step 6,848, then flat for the remaining 93,152 steps. Predates the LR fix -- worth rerunning at peak_lr~=0.01 |
| 2026-09-19 | 288 (36) | ~351k | Arm C backward + knee-adaptive forward, K_START=2, **margin=0.0** | 0.015 | vocab=32, k=2 (stuck) | 4,483 (then flat) | 100,000 steps / 84,828s (~23.6h) full budget | 1.18 (contended, arch-sandbox 4-way) | [`72663b9`](https://github.com/SimLeek/sili_peridot/commit/72663b9) | `research/dense-vs-sparse-mqar-300k` | `launch_arm_c_plus_knee_margin0.py` | Another weak result, flat for 95,517 of 100,000 steps. Predates the LR fix -- worth rerunning at peak_lr~=0.01 |
| 2026-09-19 | 288 (36) | ~351k | Arm C backward + knee-adaptive forward, K_START=2 (skip k=1), margin=0.05 | 0.015 | vocab=64, k=2 (stuck) | 34,813 (then flat) | 100,000 steps / 87,622s (~24.3h) full budget | 1.14 (contended, arch-sandbox 4-way) | [`72663b9`](https://github.com/SimLeek/sili_peridot/commit/72663b9) | `research/dense-vs-sparse-mqar-300k` | `launch_arm_c_plus_knee_skip_k1.py` | Best of the 4 original knee variants (reached vocab=64 vs the others' vocab=32) but still stuck, never reached k=3. Predates the LR fix -- worth rerunning at peak_lr~=0.01 |
| 2026-09-19 | 288 (36) | ~351k | Arm C backward + knee-adaptive forward, K_START=2, **margin=0.15** | 0.015 | vocab=126, k=2 (stuck) | 34,660 (then flat, never reached k=3) | 100,000 steps / 92,118s (~25.6h) full budget | 1.09 (contended, arch-sandbox 4-way) | [`72663b9`](https://github.com/SimLeek/sili_peridot/commit/72663b9) | `research/dense-vs-sparse-mqar-300k` | `launch_arm_c_plus_knee_margin15.py` | **Best of the 4 original knee variants** -- only one to reach full vocab=126, though never k=3. All 4 now complete, all predate the LR fix; margin=0.15 is the natural candidate to rerun at peak_lr~=0.01 |
| 2026-09-18 | 128 (16) | ~70k | dy_r_target + fixed absolute per-step backward count (dy_k_min=dy_k_max=64), dense=True | 0.015 | vocab=16, k=3 (stuck) | 925 (then flat) | 100,000 steps / 9,080s (~2.52h) full budget | 11.01 | [`4c378fc`](https://github.com/SimLeek/sili_peridot/commit/4c378fc) | `research/dense-vs-sparse-mqar-300k` | `launch_dy_fixed_count_width128.py` | Weakest completed result of the investigation. Confirmed NOT width-specific -- see width=288 companion below, also weak |
| 2026-09-19 | 288 (36) | ~351k | dy_r_target + fixed absolute per-step backward count (dy_k_min=dy_k_max=64), dense=True | 0.015 | vocab=32, k=2 (stuck) | 25,688 (then flat) | 100,000 steps / 51,825s (~14.4h) full budget | 1.93 (contended, arch-sandbox) | [`4c378fc`](https://github.com/SimLeek/sili_peridot/commit/4c378fc) | `research/dense-vs-sparse-mqar-300k` | `launch_dy_fixed_count_width288.py` | **Test 2 complete (negative at both widths)**: got further than width=128 (vocab=32 vs vocab=16) before stalling, but still far short of the dense/Arm C results at either width. Forcing an exact dy_k_min=dy_k_max=64 clamp -- overriding dy_r_target's own natural selection entirely -- looks actively harmful, not a safe dynamic stand-in for the abandoned structural fan-in cap |
| 2026-09-18 | 288 (36) | ~351k | dense fwd + Arm C backward gate (cutoff=0.0, ~50% density), peak_lr **unscaled** | 0.015 | **vocab=126, k=3** | 24,331 | 100,000 steps / 19,803s (~5.5h) full budget; ~80 min to milestone (est.) | 5.05 | [`4c378fc`](https://github.com/SimLeek/sili_peridot/commit/4c378fc) | `research/dense-vs-sparse-mqar-300k` | `launch_armc_gatemid_lr_unscaled.py` | Real confirmation Arm C tolerates the SAME unscaled LR that made dense stall permanently at vocab=64/k=3 -- slower than dense's own LR-fixed record (13,601 steps/~30min) but succeeds where unscaled dense fails outright |
| 2026-09-18 | 288 (36) | ~351k | dense fwd + Arm C backward gate (cutoff=0.0, ~50% density), peak_lr **scaled** | 0.01 | vocab=126, k=2 (stuck) | 20,642 (then flat, never reached k=3) | 100,000 steps / 14,117s (~3.92h) full budget | 7.08 | [`4c378fc`](https://github.com/SimLeek/sili_peridot/commit/4c378fc) | `research/dense-vs-sparse-mqar-300k` | `launch_armc_gatemid_lr_scaled.py` | **Reversal**: dense's own best LR (0.01) did WORSE here than the unscaled 0.015 row above (later to vocab=126, never reached k=3) -- for Arm C, lowering LR the way dense needed appears actively counterproductive, not just unnecessary |
| 2026-09-19 | 288 (36) | ~351k | dense fwd + Arm C backward gate (cutoff=0.6, ~29.5% density), peak_lr **unscaled** | 0.015 | vocab=126, k=2 (stuck) | 30,097 (then flat, never reached k=3) | 100,000 steps / 13,153s (~3.65h) full budget | 7.60 | [`4c378fc`](https://github.com/SimLeek/sili_peridot/commit/4c378fc) | `research/dense-vs-sparse-mqar-300k` | `launch_armc_gatesparse_lr_unscaled.py` | **Sparser gate did worse, not better**, at the SAME unscaled LR: later to vocab=126 (30,097 vs 17,146) and never reached k=3, unlike the ~50%-density sibling above. Cuts against a naive "sparser gate = more LR tolerance = better" reading -- within {~50%, ~29.5%} at this LR, denser won on both axes |
| 2026-09-19 | 288 (36) | ~351k | dense fwd + Arm C backward gate (cutoff=0.6, ~29.5% density), peak_lr **scaled** | 0.01 | **vocab=126, k=3** | 43,905 | 100,000 steps / 17,120s (~4.76h) full budget | 5.84 | [`4c378fc`](https://github.com/SimLeek/sili_peridot/commit/4c378fc) | `research/dense-vs-sparse-mqar-300k` | `launch_armc_gatesparse_lr_scaled.py` | **Test 3 grid complete (4/4)**: this cell + the ~50%/unscaled cell both reached k=3; both cross-combinations (~50%/scaled, ~29.5%/unscaled) got stuck at k=2. A genuine crossover, not a monotonic density-vs-LR trend either direction -- reads as each density having its own narrow LR sweet spot (consistent with how tight the pure-dense sweet spot already was) rather than "sparser needs more/less LR" as a general rule. Slowest Arm C milestone yet (43,905 steps) despite succeeding |
| 2026-09-19 | 288 (36) | ~351k | dense, per-layer Polyak dynamic LR (c=0.5, lr_max=0.1, denom=raw E_t -- **uncalibrated**) | dynamic | vocab=16, k=3 (stuck) | 713 (then completely flat) | 100,000 steps / 14,702s (~4.08h) full budget | 6.80 | [`ac5edb6`](https://github.com/SimLeek/sili_peridot/commit/ac5edb6) | `research/dense-vs-sparse-mqar-300k` | `launch_polyak_lr_width288.py` | **Failed to learn at all.** Diagnosed: raw per-call E_t swings by orders of magnitude, constantly saturating lr against the old lr_max=0.1 -- itself already unstable per the range test. Root-caused and fixed same day (Lbar instead of E_t, c/lr_max recalibrated ~1000x down) -- see the recalibrated row below |
| 2026-09-19 | 288 (36) | ~351k | dense, per-layer Polyak dynamic LR (c=0.0005, lr_max=0.05, denom=Lbar -- **recalibrated**) | dynamic | in progress (local) | TBD | TBD | TBD | [`c8bd3eb`](https://github.com/SimLeek/sili_peridot/commit/c8bd3eb) | `research/dense-vs-sparse-mqar-300k` | `launch_polyak_lr_width288.py` | Re-run after the E_t->Lbar + c/lr_max fix. Short smoke test showed sane decreasing lr and real progress (loss 3.8->1.8, reached k=3) instead of the frozen pattern -- this is the real validation against the peak_lr=0.01 record |

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
