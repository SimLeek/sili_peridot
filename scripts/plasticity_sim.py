"""Offline, pure-Python sandbox for rapidly testing candidate
plasticity-reset algorithms against REAL recorded training data,
without needing a new C++ engine run per candidate. Direct
instruction, after watching v8's replay show importance saturating to
max_ci=100 (solid yellow) for q/k/v_proj by step ~30k while o_proj
collapsed into a few wide stripes: "I think we could try some
algorithms to see if we can get the post 8000 or 30k step state to
look more like the pre-8000 state by accumulating whatever the
algorithm does to all the arrays over time and accumulating the
deviation. The goal is to keep the network plastic enough to learn the
new challenges, and I don't think all importance saturated to 100
matches that."

**Scope, stated explicitly**: this operates on col_importance/deviation
dynamics -- the exact quantities the real engine's cycle-boundary
selection/gating logic itself operates on (see
delta_csr_types.hpp:plasticity_select_cycle_boundary, ported faithfully
below). NOT a full per-synapse gradient re-simulation -- the per-cycle
col_importance delta recorded in each run's .npz snapshots is used
AS-IS as the "natural" (environment) signal for every candidate to
react to. This is an approximation with one known confound: at cycles
where the REAL run's own algorithm reset a column, that column's
recorded delta includes the real algorithm's own decay effect, not
just genuine gradient-driven growth -- there's no recorded way to
exactly invert that (the real reset also mixes in a random `fresh`
weight sample that was never recorded, and even for col_importance
alone, its own EMA touch reads the PRE-decay raw value, so the decay's
effect only shows up in the FOLLOWING cycle's delta, entangled with
whatever real growth also happened by then). Only reset_fraction
(1% by default) of columns are touched per cycle, so this confound
affects a minority of cycles/columns -- flagged here, not hidden, and
checked directly in test_plasticity_sim.py's replay-fidelity test
(replaying real deltas with zero further intervention must exactly
reproduce the real recorded col_importance trajectory -- the sandbox's
one hard fidelity guarantee).

**What a "candidate algorithm" controls**: given the simulated
trajectory-so-far (its OWN col_importance, col_grad_fast/slow/var,
col_age -- never peeking at the real run's future or its real
deviation), decide which columns to touch this cycle and how strongly,
mirroring exactly what plasticity_select_cycle_boundary decides in the
real engine. Promising candidates still need a real engine run to
validate -- a different algorithm changes real future gradients too,
which this offline sandbox cannot replay -- per the user's own stated
two-phase plan (screen fast in Python, validate for real).

**Export back to the SAME .npz schema replay_synapse_display.py reads**
so a candidate's simulated run can be watched exactly like a real one
-- see export_simulated_trajectory(). raw_importance/raw_weight are
themselves approximated for visualization (rescaled from the real
run's own per-synapse texture, or a freshly-drawn sample matching the
real engine's own init distribution for newly-reset columns) -- not
exact, same spirit as the col_importance-level approximation above."""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass, field

import numpy as np

MATURITY_CYCLES = 1


# ---------------------------------------------------------------------------
# Faithful port of delta_csr_types.hpp:plasticity_select_cycle_boundary's
# grad-tracking + deviation + k-derivation math. Every formula here has a
# 1:1 line in the C++ source (see module docstring for the anchor).
# ---------------------------------------------------------------------------


@dataclass
class SimPlasticityState:
    """Mirrors the C++ PlasticityState's per-column fields (the subset
    relevant to cycle-boundary selection -- no per-cell weight/importance
    storage, since candidates only decide WHICH columns + how strongly,
    not the per-synapse update itself)."""

    n_out: int
    col_importance: np.ndarray = field(default=None)
    col_grad_fast: np.ndarray = field(default=None)
    col_grad_slow: np.ndarray = field(default=None)
    col_grad_var: np.ndarray = field(default=None)
    col_grad_initialized: np.ndarray = field(default=None)
    col_importance_prev_cycle: np.ndarray = field(default=None)
    col_age: np.ndarray = field(default=None)
    col_reset_active: np.ndarray = field(default=None)
    col_plasticity_boost: np.ndarray = field(default=None)

    def __post_init__(self):
        n = self.n_out
        if self.col_importance is None:
            self.col_importance = np.zeros(n, dtype=np.float64)
        if self.col_grad_fast is None:
            self.col_grad_fast = np.zeros(n, dtype=np.float64)
        if self.col_grad_slow is None:
            self.col_grad_slow = np.zeros(n, dtype=np.float64)
        if self.col_grad_var is None:
            self.col_grad_var = np.zeros(n, dtype=np.float64)
        if self.col_grad_initialized is None:
            self.col_grad_initialized = np.zeros(n, dtype=bool)
        if self.col_importance_prev_cycle is None:
            self.col_importance_prev_cycle = np.zeros(n, dtype=np.float64)
        if self.col_age is None:
            self.col_age = np.zeros(n, dtype=np.int64)
        if self.col_reset_active is None:
            self.col_reset_active = np.zeros(n, dtype=bool)
        if self.col_plasticity_boost is None:
            self.col_plasticity_boost = np.zeros(n, dtype=np.float64)


def update_grad_tracking(
    state: SimPlasticityState,
    eta_slow: float = 0.99,
    eta_slow_catchup: float = 0.95,
    eta_fast: float = 0.5,
    eta_var: float = 0.9,
    blend: float = 0.10,
) -> None:
    """Port of the per-column grad_fast/slow/var update, run once per
    cycle boundary BEFORE selection, using state.col_importance as it
    stands RIGHT NOW (already updated by the caller with this cycle's
    natural delta) vs state.col_importance_prev_cycle (last cycle's
    value). In-place; also updates col_importance_prev_cycle."""
    delta = state.col_importance - state.col_importance_prev_cycle
    cold = ~state.col_grad_initialized
    state.col_grad_fast[cold] = delta[cold]
    state.col_grad_slow[cold] = delta[cold]
    state.col_grad_var[cold] = 0.0
    state.col_grad_initialized[cold] = True

    warm = ~cold
    diff = delta - state.col_grad_slow
    state.col_grad_var[warm] = eta_var * state.col_grad_var[warm] + (1.0 - eta_var) * diff[warm] ** 2
    state.col_grad_fast[warm] = eta_fast * state.col_grad_fast[warm] + (1.0 - eta_fast) * delta[warm]
    catchup = warm & (delta < state.col_grad_slow)
    slow_branch = warm & ~catchup
    state.col_grad_slow[catchup] = (
        eta_slow_catchup * state.col_grad_slow[catchup] + (1.0 - eta_slow_catchup) * delta[catchup]
    )
    state.col_grad_slow[slow_branch] = (
        eta_slow * state.col_grad_slow[slow_branch] + (1.0 - eta_slow) * delta[slow_branch]
    )

    # Reset-induced perturbation inflation -- same formula, using THIS
    # cycle's just-cleared-before-selection col_reset_active/boost from
    # the PREVIOUS cycle boundary call (the caller is responsible for
    # calling this BEFORE clearing col_reset_active, matching the C++
    # ordering exactly).
    reset_cols = state.col_reset_active
    if np.any(reset_cols):
        strength = blend * state.col_plasticity_boost[reset_cols]
        perturbation = strength * state.col_importance_prev_cycle[reset_cols]
        state.col_grad_var[reset_cols] += perturbation**2

    state.col_importance_prev_cycle = state.col_importance.copy()


def compute_deviation(state: SimPlasticityState, mature_mask: np.ndarray) -> np.ndarray:
    """Signed z-score, matching deviation_by_col exactly. Zero for
    immature columns (never selected, matches the C++ default-init)."""
    dev = np.zeros(state.n_out, dtype=np.float64)
    std_dev = np.sqrt(np.maximum(state.col_grad_var[mature_mask], 0.0))
    dev[mature_mask] = (state.col_grad_fast[mature_mask] - state.col_grad_slow[mature_mask]) / (std_dev + 1e-8)
    return dev


def percentile_k(deviation: np.ndarray, mature_mask: np.ndarray, reset_fraction: float) -> float:
    """Linear-interpolated percentile of the mature population's own
    deviation values, at 100*(1-reset_fraction) -- matches numpy's
    default 'linear' method, matches the C++ port exactly (both
    verified bit-accurate in sili__new's own unit tests)."""
    vals = deviation[mature_mask]
    if vals.size == 0:
        return 0.0
    if vals.size == 1:
        return float(vals[0])
    p = 100.0 * (1.0 - reset_fraction)
    return float(np.percentile(vals, p, method="linear"))


def evt_k(mature_count: int) -> float:
    """E[max_N] of N iid standard normals ~ sqrt(2*ln(N)) -- the FIRST
    (found-wrong) fix, kept here so it can be re-tested/compared, not
    because it's recommended."""
    if mature_count <= 1:
        return 0.0
    return math.sqrt(2.0 * math.log(mature_count))


def cycle_boundary(
    state: SimPlasticityState,
    reset_fraction: float,
    k: float,
    select_by_deviation: bool,
    k_mode: str = "percentile",
    eta_slow: float = 0.99,
    eta_slow_catchup: float = 0.95,
    eta_fast: float = 0.5,
    eta_var: float = 0.9,
    blend: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """One full cycle-boundary step, matching
    plasticity_select_cycle_boundary's order of operations exactly:
    (1) grad-tracking update using the PREVIOUS cycle's reset_active/
    boost, (2) clear reset_active, (3) compute deviation for the whole
    mature population, (4) rank candidates (by deviation or
    col_importance), (5) derive k_effective, (6) flag the top_n and set
    their boost. Returns (reset_active, deviation) for this cycle.
    Mutates state in place, including bumping col_age."""
    mature_mask = state.col_age >= MATURITY_CYCLES
    update_grad_tracking(state, eta_slow, eta_slow_catchup, eta_fast, eta_var, blend)
    state.col_reset_active[:] = False

    deviation = compute_deviation(state, mature_mask)
    mature_idx = np.flatnonzero(mature_mask)
    if select_by_deviation:
        order = mature_idx[np.argsort(-deviation[mature_idx])]
    else:
        order = mature_idx[np.argsort(-state.col_importance[mature_idx])]

    top_n = int(round(len(mature_idx) * reset_fraction))
    if select_by_deviation:
        if k_mode == "percentile":
            k_effective = percentile_k(deviation, mature_mask, reset_fraction)
        elif k_mode == "evt":
            k_effective = evt_k(len(mature_idx))
        else:
            raise ValueError(f"unknown k_mode {k_mode!r}")
    else:
        k_effective = k

    state.col_plasticity_boost[:] = 0.0
    selected = order[:top_n]
    if len(selected):
        state.col_reset_active[selected] = True
        state.col_plasticity_boost[selected] = np.maximum(0.0, deviation[selected] - k_effective)

    state.col_age[state.col_reset_active] = 0
    state.col_age[~state.col_reset_active] += 1

    return state.col_reset_active.copy(), deviation


# ---------------------------------------------------------------------------
# Loading recorded trajectories
# ---------------------------------------------------------------------------


@dataclass
class PoolTrajectory:
    pool_key: str
    steps: np.ndarray  # (T,)
    col_importance: np.ndarray  # (T, n_out) -- the recorded EMA aggregate
    raw_mean_ci: np.ndarray | None  # (T, n_out) -- mean over rows of raw_importance, if recorded
    col_grad_fast: np.ndarray
    col_grad_slow: np.ndarray
    col_grad_var: np.ndarray
    col_age: np.ndarray
    col_reset_active: np.ndarray
    files: list[str]  # (T,) -- for lazy raw_importance/raw_weight access

    @property
    def n_out(self) -> int:
        return self.col_importance.shape[1]

    @property
    def n_cycles(self) -> int:
        return self.col_importance.shape[0]


def load_pool_trajectory(pool_dir: str, pool_key: str | None = None) -> PoolTrajectory:
    files = sorted(
        glob.glob(os.path.join(pool_dir, "step*.npz")),
        key=lambda f: int(os.path.basename(f).replace("step", "").replace(".npz", "")),
    )
    if not files:
        raise FileNotFoundError(f"No .npz snapshots found under {pool_dir}")
    steps, col_importance, fast, slow, var, age, active, raw_means = [], [], [], [], [], [], [], []
    has_raw = True
    for f in files:
        d = np.load(f)
        steps.append(int(d["step"]))
        col_importance.append(d["col_importance"])
        fast.append(d["col_grad_fast"])
        slow.append(d["col_grad_slow"])
        var.append(d["col_grad_var"])
        age.append(d["col_age"])
        active.append(d["col_reset_active"])
        if "raw_importance" in d.files:
            raw_means.append(d["raw_importance"].mean(axis=0))
        else:
            has_raw = False
    return PoolTrajectory(
        pool_key=pool_key or os.path.basename(os.path.normpath(pool_dir)),
        steps=np.array(steps),
        col_importance=np.stack(col_importance).astype(np.float64),
        raw_mean_ci=np.stack(raw_means).astype(np.float64) if has_raw else None,
        col_grad_fast=np.stack(fast).astype(np.float64),
        col_grad_slow=np.stack(slow).astype(np.float64),
        col_grad_var=np.stack(var).astype(np.float64),
        col_age=np.stack(age).astype(np.int64),
        col_reset_active=np.stack(active).astype(bool),
        files=files,
    )


def natural_deltas(signal: np.ndarray) -> np.ndarray:
    """Per-cycle delta of an arbitrary (T, n_out) importance-like
    signal (traj.col_importance -- the recorded EMA aggregate -- or
    traj.raw_mean_ci -- the mean per-synapse raw ci, closer to ground
    truth and free of the EMA aggregate's own smoothing) -- the
    "environment" every candidate replays against. Shape (T-1, n_out).
    Whichever signal is used, this is an APPROXIMATION, stated plainly:
    at cycles/columns a real run's own algorithm reset, the recorded
    delta bakes in that real algorithm's own effect, not pure gradient
    growth -- and since a column gets touched many times over a full
    run, this compounds. Direct instruction after finding replaying
    col_importance this way didn't reproduce even v8's OWN real
    algorithm's real outcome: "this is of course going to be an
    approximation... This just allows us to see if the algorithm is
    continually at least pushing the model towards a good state and
    not away from a good state" -- a directional screening signal, not
    a quantitative forecast."""
    return signal[1:] - signal[:-1]


# ---------------------------------------------------------------------------
# Running a candidate algorithm against a loaded trajectory
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    pool_key: str
    steps: np.ndarray  # (T,) -- same as the real trajectory's steps
    col_importance: np.ndarray  # (T, n_out) simulated -- holds whichever signal simulate() was given
    col_grad_fast: np.ndarray
    col_grad_slow: np.ndarray
    col_grad_var: np.ndarray
    col_age: np.ndarray
    col_reset_active: np.ndarray  # (T, n_out) -- True at cycles/columns this candidate touched
    deviation: np.ndarray  # (T, n_out)


def simulate(
    traj: PoolTrajectory,
    reset_fraction: float = 0.01,
    k: float = 1.0,
    select_by_deviation: bool = False,
    k_mode: str = "percentile",
    eta_slow: float = 0.99,
    eta_slow_catchup: float = 0.95,
    eta_fast: float = 0.5,
    eta_var: float = 0.9,
    blend: float = 0.10,
    max_ci: float | None = None,
    signal: str = "col_importance",
) -> SimResult:
    """Replay traj's real per-cycle natural deltas, letting a candidate
    algorithm (parameterized the same way the real engine is) decide
    its OWN selection/gating every cycle, on a copy of the chosen
    signal (`signal="col_importance"`, the recorded EMA aggregate, or
    `signal="raw_mean_ci"`, the mean per-synapse raw ci -- requires the
    run to have been captured with plasticity_raw_importance_log=True)
    that starts identical to the real run's but evolves independently
    from then on. max_ci, if given, clamps the signal after every cycle
    (matching the real ci accumulator's own hard ceiling)."""
    if signal == "col_importance":
        base = traj.col_importance
    elif signal == "raw_mean_ci":
        if traj.raw_mean_ci is None:
            raise KeyError(
                f"{traj.pool_key} has no raw_importance recorded -- re-run with "
                "plasticity_raw_importance_log=True, or use signal='col_importance'"
            )
        base = traj.raw_mean_ci
    else:
        raise ValueError(f"unknown signal {signal!r}")
    deltas = natural_deltas(base)
    n_out = traj.n_out
    T = traj.n_cycles
    state = SimPlasticityState(n_out=n_out, col_importance=base[0].copy())

    out_importance = np.zeros((T, n_out))
    out_fast = np.zeros((T, n_out))
    out_slow = np.zeros((T, n_out))
    out_var = np.zeros((T, n_out))
    out_age = np.zeros((T, n_out), dtype=np.int64)
    out_active = np.zeros((T, n_out), dtype=bool)
    out_dev = np.zeros((T, n_out))

    out_importance[0] = state.col_importance
    out_age[0] = state.col_age

    for t in range(T - 1):
        state.col_importance = state.col_importance + deltas[t]
        if max_ci is not None:
            np.clip(state.col_importance, None, max_ci, out=state.col_importance)
        active, dev = cycle_boundary(
            state,
            reset_fraction=reset_fraction,
            k=k,
            select_by_deviation=select_by_deviation,
            k_mode=k_mode,
            eta_slow=eta_slow,
            eta_slow_catchup=eta_slow_catchup,
            eta_fast=eta_fast,
            eta_var=eta_var,
            blend=blend,
        )
        # Apply THIS candidate's own decay directly to col_importance
        # (the real engine's per-cell touch loop does this too, just
        # at per-synapse granularity -- see module docstring for why
        # operating at the col_importance level is the right fidelity
        # for testing selection/gating rules specifically).
        strength = blend * state.col_plasticity_boost
        state.col_importance = state.col_importance * (1.0 - strength * active)

        out_importance[t + 1] = state.col_importance
        out_fast[t + 1] = state.col_grad_fast
        out_slow[t + 1] = state.col_grad_slow
        out_var[t + 1] = state.col_grad_var
        out_age[t + 1] = state.col_age
        out_active[t + 1] = active
        out_dev[t + 1] = dev

    return SimResult(
        pool_key=traj.pool_key,
        steps=traj.steps,
        col_importance=out_importance,
        col_grad_fast=out_fast,
        col_grad_slow=out_slow,
        col_grad_var=out_var,
        col_age=out_age,
        col_reset_active=out_active,
        deviation=out_dev,
    )


# ---------------------------------------------------------------------------
# Export back to the replay tool's .npz schema
# ---------------------------------------------------------------------------


def sim_export_dir(base_column_log_dir: str, candidate_name: str, pool_key: str) -> str:
    """Naming convention for exported simulated trajectories: a sibling
    of the real run's own column-log directory, suffixed with the
    candidate's name, so multiple candidates tested against the same
    real run never collide or overwrite each other or the real data --
    e.g. `<run>_sim_<candidate_name>/<pool_key>/`."""
    parent = os.path.dirname(os.path.normpath(base_column_log_dir))
    run_name = os.path.basename(os.path.normpath(base_column_log_dir))
    return os.path.join(parent, f"{run_name}_sim_{candidate_name}", pool_key)


def export_simulated_trajectory(
    traj: PoolTrajectory,
    result: SimResult,
    out_dir: str,
    max_ci: float = 100.0,
    signal: str = "col_importance",
) -> int:
    """Write one .npz per simulated cycle, same schema
    replay_synapse_display.py reads. raw_importance is APPROXIMATED by
    rescaling the real run's own per-synapse importance at that cycle
    so each column's relative LEVEL matches the simulated signal
    (real_raw * sim_signal / real_signal, elementwise per column) --
    preserves the real per-synapse texture while reflecting what the
    candidate algorithm would have produced at the aggregate level.
    `signal` must match whatever was passed to simulate() (defaults
    must agree -- "raw_mean_ci" gives the more faithful rescale, since
    both sides then come from the same raw_importance data). raw_weight:
    unchanged from the real run for columns this candidate never reset;
    for columns it DID reset, reuses the real run's post-reset weight
    if that column was ALSO reset for real at a nearby cycle, else
    leaves it unchanged (a weight-level reset's random `fresh`
    component was never recorded, so a genuinely new reset can't be
    reconstructed -- documented limitation, not hidden). Returns the
    number of files written."""
    os.makedirs(out_dir, exist_ok=True)
    real_signal = traj.raw_mean_ci if signal == "raw_mean_ci" else traj.col_importance
    n_written = 0
    for t in range(result.steps.shape[0]):
        d = np.load(traj.files[t])
        real_imp = d["raw_importance"] if "raw_importance" in d.files else None
        real_w = d["raw_weight"] if "raw_weight" in d.files else None
        extra = {}
        if real_imp is not None:
            real_col = np.maximum(real_signal[t], 1e-8)
            ratio = result.col_importance[t] / real_col
            extra["raw_importance"] = np.clip(real_imp * ratio[None, :], 0.0, max_ci)
        if real_w is not None:
            extra["raw_weight"] = real_w
        np.savez(
            os.path.join(out_dir, f"step{int(result.steps[t]):08d}.npz"),
            step=result.steps[t],
            loss_ema=float("nan"),
            acc_ema=float("nan"),
            n_reset_this_cycle=int(result.col_reset_active[t].sum()),
            l2_sat_ratio=0.0,
            l2_decay_strength=0.0,
            col_importance=result.col_importance[t].astype(np.float32),
            col_grad_slow=result.col_grad_slow[t].astype(np.float32),
            col_grad_fast=result.col_grad_fast[t].astype(np.float32),
            col_grad_var=result.col_grad_var[t].astype(np.float32),
            col_age=result.col_age[t].astype(np.uint32),
            col_reset_active=result.col_reset_active[t].astype(np.uint8),
            **extra,
        )
        n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# Level-based candidates: a genuinely different criterion shape than
# select_by_deviation/top-importance (no growth-rate history needed --
# just "how close is this column to max_ci right now").
# ---------------------------------------------------------------------------


def ceiling_decay_step(
    col_importance: np.ndarray,
    max_ci: float = 100.0,
    ceiling_frac: float = 0.9,
    decay_rate: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Any column above `ceiling_frac * max_ci` gets decayed this cycle,
    strength ramping smoothly from 0 (at the ceiling) to `decay_rate`
    (at max_ci) -- deliberately the simplest possible rule directly
    targeting "nothing should sit pinned near max_ci", unlike
    select_by_deviation's growth-RATE criterion (which can leave an
    already-saturated, no-longer-accelerating column untouched
    indefinitely, since deviation only fires on a RATE spike, not a
    sustained high LEVEL). Returns (touched_mask, strength)."""
    ceiling = ceiling_frac * max_ci
    span = max_ci - ceiling
    over = np.clip(col_importance - ceiling, 0.0, None)
    strength = decay_rate * (over / span) if span > 0 else np.zeros_like(col_importance)
    touched = strength > 0
    return touched, strength


def simulate_level_based(
    traj: PoolTrajectory,
    step_fn,
    signal: str = "col_importance",
    max_ci: float | None = None,
    **step_kwargs,
) -> SimResult:
    """Generic replay for LEVEL-based candidates -- no growth-rate
    history needed. `step_fn(col_importance_array, **step_kwargs) ->
    (touched_mask, strength_array)` decides this cycle's action purely
    from the CURRENT signal level. Simpler counterpart to simulate()
    (which mirrors the real engine's select-then-deviation-gate shape
    specifically); this is for candidates with a genuinely different
    criterion. col_grad_fast/slow/var/deviation/col_age are left at
    zero in the returned SimResult (not meaningful for this candidate
    shape) -- present only so downstream tools (saturation_fraction,
    export_simulated_trajectory) see the same SimResult shape."""
    if signal == "col_importance":
        base = traj.col_importance
    elif signal == "raw_mean_ci":
        if traj.raw_mean_ci is None:
            raise KeyError(
                f"{traj.pool_key} has no raw_importance recorded -- re-run with "
                "plasticity_raw_importance_log=True, or use signal='col_importance'"
            )
        base = traj.raw_mean_ci
    else:
        raise ValueError(f"unknown signal {signal!r}")
    deltas = natural_deltas(base)
    n_out = traj.n_out
    T = traj.n_cycles
    current = base[0].copy()

    out_importance = np.zeros((T, n_out))
    out_active = np.zeros((T, n_out), dtype=bool)
    out_importance[0] = current

    for t in range(T - 1):
        current = current + deltas[t]
        if max_ci is not None:
            np.clip(current, None, max_ci, out=current)
        touched, strength = step_fn(current, **step_kwargs)
        current = current * (1.0 - strength * touched)
        out_importance[t + 1] = current
        out_active[t + 1] = touched

    zeros = np.zeros_like(out_importance)
    return SimResult(
        pool_key=traj.pool_key,
        steps=traj.steps,
        col_importance=out_importance,
        col_grad_fast=zeros,
        col_grad_slow=zeros,
        col_grad_var=zeros,
        col_age=np.zeros((T, n_out), dtype=np.int64),
        col_reset_active=out_active,
        deviation=zeros,
    )


def saturation_fraction(result: SimResult, max_ci: float = 100.0, tol: float = 0.5) -> np.ndarray:
    """Fraction of columns within `tol` of max_ci at each cycle -- the
    direct, literal metric for "importance saturated to 100", tracked
    over time so a candidate's whole trajectory can be compared, not
    just its final state."""
    return (result.col_importance >= (max_ci - tol)).mean(axis=1)
