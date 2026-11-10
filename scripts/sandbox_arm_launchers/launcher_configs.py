"""Preset configs for launch.py/sweep.py, replacing ~68 retired
launch_*.py files that each hardcoded one train_curriculum(...) call.
Each PRESETS entry is a plain dict of train_curriculum kwargs (plus
the special "num_cpus"/"k_start" keys launch.py pops off and applies
as module-level overrides, matching every original file's own
`m.NUM_CPUS = ...`/`m.K_START = ...` convention -- the true underlying
defaults are NUM_CPUS=16, K_START=1, NOT the commonly-overridden 4/2,
so every preset states its own value explicitly rather than relying on
a guessed default).

Families are built as a BASE dict plus derived variants via
`{**BASE, "key": new_value}` -- this is deliberately visible, not
hidden behind a generic parameterization, since "X is just Y with Z
changed" (e.g. v19b = v19 with the same seed re-run; v19c = v19 with a
new seed) is exactly the relationship worth preserving, not erasing.

Every entry's comment names the file it replaces, for anyone following
a JOURNAL.md/MQAR_LEADERBOARD.md reference to an exact historical
filename that no longer exists as its own file (recoverable either way
via git history, but this avoids needing to dig for the common case).
"""

from __future__ import annotations

import math

PRESETS: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# arm_b_biasfree.py
# ---------------------------------------------------------------------------
PRESETS["arm_b_biasfree"] = {
    "precision": "fp32",
    "max_steps": 20000,
    "seed": 1000,
    "peak_lr": 0.015,
    "num_tiles": 16,
    "k_max": 10,
    "additive_rank": 1,
    "dynamic_rank_control": True,
    "rank_grace_period_steps": 50,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "dy_r_target": 0.9,
    "target_steps_per_sec": 95.69,
    "x_r_target": 0.9,
    "r_target_min": 0.5,
    "k_first_target": 3,
    "x_balance_bias_step": 0.01,
    "log_every": 250,
    "num_cpus": 4,
}

# ---------------------------------------------------------------------------
# dense_control_remote.py
# ---------------------------------------------------------------------------
PRESETS["dense_control_remote"] = {
    "precision": "fp32",
    "max_steps": 3000,
    "seed": 1000,
    "peak_lr": 0.015,
    "num_tiles": 16,
    "k_max": 10,
    "additive_rank": 1,
    "dynamic_rank_control": True,
    "rank_grace_period_steps": 50,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "k_first_target": 3,
    "log_every": 25,
}

# ---------------------------------------------------------------------------
# launch_dense_remote.py / launch_sparse_remote.py / launch_sparse_gradlog_remote.py
# -- originally called m.main() via a hand-built sys.argv list; main() does
# nothing beyond argv-parsing + an internal log_fn/trajectory_log_fn +
# calling train_curriculum(**these same kwargs), so folding them in as
# ordinary presets is behavior-preserving, not an approximation.
# ---------------------------------------------------------------------------
PRESETS["dense_remote"] = {
    "precision": "fp32",
    "max_steps": 100000,
    "seed": 1000,
    "peak_lr": 0.015,
    "num_tiles": 16,
    "k_max": 10,
    "additive_rank": 1,
    "dynamic_rank_control": True,
    "rank_grace_period_steps": 50,
    "use_critic": False,
    "recurrent_only_output": False,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "dy_k_min": 0,
    "x_k_min": 0,
}
PRESETS["sparse_gradlog_remote"] = dict(
    PRESETS["dense_remote"],
    dy_r_target=0.9,
    target_steps_per_sec=95.69,
    x_r_target=0.9,
    trajectory_log_every=2000,
    r_target_min=0.5,
    num_cpus=4,
)
PRESETS["sparse_remote"] = {k: v for k, v in PRESETS["sparse_gradlog_remote"].items() if k != "num_cpus"}

# ---------------------------------------------------------------------------
# launch_sparse_energy_remote.py -- note: NO k_first_target, NO num_cpus/
# k_start override at all (confirmed directly, this is the one exception).
# ---------------------------------------------------------------------------
PRESETS["sparse_energy_remote"] = {
    "precision": "fp32",
    "max_steps": 100000,
    "seed": 1000,
    "peak_lr": 0.015,
    "num_tiles": 16,
    "k_max": 10,
    "additive_rank": 1,
    "dynamic_rank_control": True,
    "rank_grace_period_steps": 50,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "dy_r_target": 0.9,
    "target_steps_per_sec": 95.69,
    "x_r_target": 0.9,
    "r_target_min": 0.5,
    "use_energy": True,
    "energy_kwargs": {
        "state": {
            "drive": 0.0114,
            "activation_cost": 0.05,
            "precision": 0.01,
            "density": 0.5,
            "p": 0.9,
            "wake_gate_steps": 2000,
            "stagger_wake_init": True,
        },
        "embed_input": {
            "drive": 0.00294,
            "activation_cost": 0.05,
            "precision": 0.01,
            "density": 0.5,
            "p": 0.9,
            "wake_gate_steps": 2000,
            "stagger_wake_init": True,
        },
    },
}

# ---------------------------------------------------------------------------
# sweep_r0.8.py / sweep_r0.9.py -- K_START left at true default (1), only
# NUM_CPUS=4 overridden (verified directly, not assumed).
# ---------------------------------------------------------------------------
_R_TARGET_MIN_BASE = {
    "precision": "fp32",
    "max_steps": 100000,
    "seed": 1000,
    "peak_lr": 0.015,
    "num_tiles": 16,
    "k_max": 10,
    "additive_rank": 1,
    "dynamic_rank_control": True,
    "rank_grace_period_steps": 50,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "dy_r_target": 0.9,
    "target_steps_per_sec": 95.69,
    "x_r_target": 0.9,
    "k_first_target": 3,
    "log_every": 1000,
    "num_cpus": 4,
}
PRESETS["sweep_r0.8"] = dict(_R_TARGET_MIN_BASE, r_target_min=0.8)
PRESETS["sweep_r0.9"] = dict(_R_TARGET_MIN_BASE, r_target_min=0.9)

# ---------------------------------------------------------------------------
# Arm C family (dy_time_gate_cutoff) -- launch_armc_*.py
# ---------------------------------------------------------------------------
_ARMC_BASE = {
    "precision": "fp32",
    "max_steps": 100000,
    "seed": 1000,
    "num_tiles": 16,
    "k_max": 10,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "k_first_target": 3,
    "log_every": 250,
    "num_cpus": 4,
    "k_start": 2,
}
PRESETS["armc_gatemid_lr_scaled"] = dict(_ARMC_BASE, peak_lr=0.01, dy_time_gate_cutoff=0.0)
PRESETS["armc_gatemid_lr_unscaled"] = dict(_ARMC_BASE, peak_lr=0.015, dy_time_gate_cutoff=0.0)
PRESETS["armc_gatesparse_lr_scaled"] = dict(_ARMC_BASE, peak_lr=0.01, dy_time_gate_cutoff=0.6)
PRESETS["armc_gatesparse_lr_unscaled"] = dict(_ARMC_BASE, peak_lr=0.015, dy_time_gate_cutoff=0.6)
# 1/p and 1/sqrt(p) LR-compensation brackets for cutoff=0.3's reduced update
# frequency -- computed via the ORIGINAL files' own exact formula, not a
# rounded literal (an earlier draft of this file used 0.0248/0.0158,
# truncated from the real 0.024813.../0.015752... -- fixed after a
# verification pass flagged the discrepancy).
_GATE_DENSITY_P = 0.5 - math.asin(0.3) / math.pi  # ~0.403
PRESETS["armc_lr_scaled_invp"] = dict(_ARMC_BASE, peak_lr=0.01 / _GATE_DENSITY_P, dy_time_gate_cutoff=0.3)
PRESETS["armc_lr_scaled_invsqrtp"] = dict(
    _ARMC_BASE, peak_lr=0.01 / math.sqrt(_GATE_DENSITY_P), dy_time_gate_cutoff=0.3
)
PRESETS["dense_forward_arm_c_backward"] = dict(_ARMC_BASE, peak_lr=0.015, dy_time_gate_cutoff=0.3)

# ---------------------------------------------------------------------------
# Arm C + knee-adaptive x_r_target family -- launch_arm_c_plus_knee*.py
# NOTE: launch_arm_c_plus_knee.py (the plain base) never touches K_START --
# it stays at the TRUE default (1), unlike every *_skip_k1/_margin* sibling
# which explicitly sets K_START=2. Verified directly, not assumed -- this
# is exactly the kind of difference that's easy to silently lose.
# ---------------------------------------------------------------------------
PRESETS["arm_c_plus_knee"] = {
    "precision": "fp32",
    "max_steps": 100000,
    "seed": 1000,
    "peak_lr": 0.015,
    "num_tiles": 16,
    "k_max": 10,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "k_first_target": 3,
    "dy_time_gate_cutoff": 0.3,
    "x_r_target_auto": True,
    "x_r_target_auto_margin": 0.05,
    "log_every": 250,
    "trajectory_log_every": 2000,
    "num_cpus": 4,
    # k_start intentionally omitted -- true default (1), verified directly.
}
PRESETS["arm_c_plus_knee_skip_k1"] = dict(
    PRESETS["arm_c_plus_knee"],
    k_start=2,  # skip k=1 only -- see original file's own docstring
)
PRESETS["arm_c_plus_knee_margin0"] = dict(
    PRESETS["arm_c_plus_knee_skip_k1"],
    x_r_target_auto_margin=0.0,
)
PRESETS["arm_c_plus_knee_margin15"] = dict(
    PRESETS["arm_c_plus_knee_skip_k1"],
    x_r_target_auto_margin=0.15,
)
PRESETS["arm_c_plus_knee_margin15_loss_decay"] = dict(
    PRESETS["arm_c_plus_knee_margin15"],
    loss_adjusted_decay_enable=True,
)
PRESETS["arm_c_plus_knee_margin15_plasticity_reset"] = dict(
    PRESETS["arm_c_plus_knee_margin15"],
    plasticity_reset_enable=True,
)

# ---------------------------------------------------------------------------
# Width-288 dense family -- launch_dense_lr_*scaled*.py, launch_dense_maxgradnorm_scaled.py
# ---------------------------------------------------------------------------
_WIDTH288_BASE = {
    "precision": "fp32",
    "max_steps": 100000,
    "seed": 1000,
    "num_tiles": 16,
    "k_max": 10,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "k_first_target": 3,
    "log_every": 250,
    "num_cpus": 4,
    "k_start": 2,
}
PRESETS["width288_nolevel_down_control"] = dict(_WIDTH288_BASE, peak_lr=0.015)  # the reference/control baseline
PRESETS["dense_lr_scaled"] = dict(_WIDTH288_BASE, peak_lr=0.01)
PRESETS["dense_lr_alpha1_scaled"] = dict(_WIDTH288_BASE, peak_lr=0.0067)
PRESETS["dense_lr_unscaled_loss_decay"] = dict(_WIDTH288_BASE, peak_lr=0.015, loss_adjusted_decay_enable=True)
PRESETS["dense_maxgradnorm_scaled"] = dict(_WIDTH288_BASE, peak_lr=0.015, max_grad_norm=1.5)

# ---------------------------------------------------------------------------
# dy_k_min/dy_k_max fixed-count family -- launch_dy_fixed_count_width*.py
# ---------------------------------------------------------------------------
PRESETS["dy_fixed_count_width128"] = dict(
    _WIDTH288_BASE,
    peak_lr=0.015,
    embed_width=16,  # width=128 arm -- deliberately NOT the usual 36
    dy_r_target=0.9,
    dy_k_min=64,
    dy_k_max=64,
)
PRESETS["dy_fixed_count_width288"] = dict(
    _WIDTH288_BASE,
    peak_lr=0.015,
    dy_r_target=0.9,
    dy_k_min=64,
    dy_k_max=64,
)

# ---------------------------------------------------------------------------
# Polyak LR family -- launch_polyak_lr_width288*.py
# ---------------------------------------------------------------------------
PRESETS["polyak_lr_width288"] = dict(_WIDTH288_BASE, peak_lr=0.015, polyak_lr=True)
PRESETS["polyak_lr_width288_loss_decay"] = dict(PRESETS["polyak_lr_width288"], loss_adjusted_decay_enable=True)
PRESETS["polyak_lr_width288_plasticity_reset"] = dict(PRESETS["polyak_lr_width288"], plasticity_reset_enable=True)

# ---------------------------------------------------------------------------
# Plasticity-reset v-series -- the big QKVO-Norm/CiRenorm/WeightRenorm/
# energy/sleep lineage, v3 through v28. All share the same base shape:
# plasticity_reset_enable=True + a per-run plasticity_column_log_dir.
# ---------------------------------------------------------------------------
_PLASTICITY_BASE = {
    "precision": "fp32",
    "max_steps": 100000,
    "seed": 1000,
    "peak_lr": 0.015,
    "num_tiles": 16,
    "k_max": 10,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "k_first_target": 3,
    "log_every": 250,
    "num_cpus": 4,
    "k_start": 2,
}

PRESETS["dense_lr_unscaled_plasticity_reset"] = dict(_PLASTICITY_BASE, plasticity_reset_enable=True)
PRESETS["dense_lr_unscaled_plasticity_reset_column_log"] = dict(
    _PLASTICITY_BASE,
    num_cpus=2,  # the one exception: this file used 2, not 4 -- verified directly
    plasticity_reset_enable=True,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v3_l2decay_column_log"] = dict(
    _PLASTICITY_BASE,
    plasticity_reset_enable=True,
    plasticity_reset_l2_decay_lambda=0.05,
    plasticity_reset_l2_decay_threshold=0.9,
    plasticity_reset_l2_decay_temperature=0.05,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v3_l2decay",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v5_select_by_deviation"] = dict(
    _PLASTICITY_BASE,
    plasticity_reset_enable=True,
    plasticity_reset_select_by_deviation=True,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v5_select_by_deviation",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v6_select_by_deviation_evt_k"] = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v5_select_by_deviation"],
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v6_select_by_deviation_evt_k",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v7_select_by_deviation_percentile_k"] = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v5_select_by_deviation"],
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v7_select_by_deviation_percentile_k",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v8_select_by_deviation_replay_capture"] = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v7_select_by_deviation_percentile_k"],
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v8_select_by_deviation_replay_capture",
    plasticity_raw_importance_log=True,
)
PRESETS["dense_lr_unscaled_plasticity_reset_v9_layer_l2_decay"] = dict(
    _PLASTICITY_BASE,
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_reset_l2_decay_lambda=0.05,
    plasticity_reset_l2_decay_threshold=0.9,
    plasticity_reset_l2_decay_temperature=0.05,
    plasticity_reset_max_ci=100.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v9_layer_l2_decay",
    plasticity_raw_importance_log=True,
)
PRESETS["dense_lr_unscaled_plasticity_reset_v10_l2_init"] = dict(
    _PLASTICITY_BASE,
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v10_l2_init",
    plasticity_raw_importance_log=True,
    l2_init_enable=True,
    l2_init_rate=0.0001,
)
PRESETS["dense_lr_unscaled_plasticity_reset_v11_qk_l1_sparsity"] = dict(
    _PLASTICITY_BASE,
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v11_qk_l1_sparsity",
    plasticity_raw_importance_log=True,
    qk_l1_sparsity_coef=0.05,
    qk_spectral_norm_diag_log=True,
)
# v12/v12b: DELETED (launch_dense_lr_unscaled_plasticity_reset_v12_qk_norm.py,
# launch_dense_lr_unscaled_plasticity_reset_v12b_qk_norm_repeat.py), not
# folded in. Both called qk_norm_enable=True, a Q/K-ONLY normalization
# flag removed from train_curriculum's signature once v12/v12b's own
# real validation found it insufficient -- per train_mqar_curriculum.py's
# own comment, qkvo_norm_enable "supersedes the earlier qk_norm_enable...
# v12/v12b's real validation found Q/K-only normalization just shifted
# the same saturation pattern onto v_proj/o_proj/input_proj instead of
# removing it," which is exactly why v13 (qkvo, broader) exists and
# superseded it. Deleted rather than kept broken: the underlying
# capability is gone (not a rename -- qkvo_norm_enable is a materially
# different, broader config, substituting it would misrepresent what
# v12 tested), and the negative finding that made it obsolete is
# already captured in train_mqar_curriculum.py's own comment and
# JOURNAL.md (2026-11-09 entry). See JOURNAL.md for the full note.
PRESETS["dense_lr_unscaled_plasticity_reset_v13_qkvo_norm"] = dict(
    _PLASTICITY_BASE,
    seed=1000,
    plasticity_reset_enable=True,
    plasticity_reset_reset_fraction=0.0,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v13_qkvo_norm",
    plasticity_raw_importance_log=True,
    qkvo_norm_enable=True,
    qk_spectral_norm_diag_log=True,
)
PRESETS["dense_lr_unscaled_plasticity_reset_v13b_qkvo_norm_repeat"] = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v13_qkvo_norm"],
    seed=1001,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v13b_qkvo_norm_repeat",
)
# v14 onward: seed=1001 is the "hard seed" baseline every subsequent entry
# in this lineage (through v24+) inherits unless it's an explicit new-seed
# ("...seed1000"/"c") replication.
_QKVO_BASE = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v13b_qkvo_norm_repeat"],
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v14_qkvo_norm_grad_clip",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v14_qkvo_norm_grad_clip"] = dict(_QKVO_BASE)
PRESETS["dense_lr_unscaled_plasticity_reset_v15_qkvo_norm_centering_col"] = dict(
    _QKVO_BASE,
    centering_col_enable=True,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v15_qkvo_norm_centering_col",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v15b_qkvo_norm_centering_col_repeat"] = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v15_qkvo_norm_centering_col"],
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v15b_qkvo_norm_centering_col_repeat",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v15c_qkvo_norm_centering_col_seed1000"] = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v15_qkvo_norm_centering_col"],
    seed=1000,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v15c_qkvo_norm_centering_col_seed1000",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v16_qkvo_norm_centering_rowcol"] = dict(
    _QKVO_BASE,
    centering_row_enable=True,
    centering_col_enable=True,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v16_qkvo_norm_centering_rowcol",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v17_qkvo_norm_ci_renorm_trust_ratio"] = dict(
    _QKVO_BASE,
    ci_renorm_enable="trust_ratio",
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v17_qkvo_norm_ci_renorm_trust_ratio",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v18_qkvo_norm_ci_renorm_stable_region"] = dict(
    _QKVO_BASE,
    ci_renorm_enable="stable_region",
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v18_qkvo_norm_ci_renorm_stable_region",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v18b_qkvo_norm_ci_renorm_stable_region_repeat"] = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v18_qkvo_norm_ci_renorm_stable_region"],
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v18b_qkvo_norm_ci_renorm_stable_region_repeat",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v18c_qkvo_norm_ci_renorm_stable_region_seed1000"] = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v18_qkvo_norm_ci_renorm_stable_region"],
    seed=1000,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v18c_qkvo_norm_ci_renorm_stable_region_seed1000",
)
# v19: CiRenorm StableRegion + WeightRenorm together -- the config every
# v21+ (energy/sleep) entry below is built on top of.
_V19_BASE = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v18_qkvo_norm_ci_renorm_stable_region"],
    weight_renorm_enable="stable_region",
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v19_qkvo_norm_ci_and_weight_renorm",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v19_qkvo_norm_ci_and_weight_renorm"] = dict(_V19_BASE)
PRESETS["dense_lr_unscaled_plasticity_reset_v19b_qkvo_norm_ci_and_weight_renorm_repeat"] = dict(
    _V19_BASE,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v19b_qkvo_norm_ci_and_weight_renorm_repeat",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v19c_qkvo_norm_ci_and_weight_renorm_seed1000"] = dict(
    _V19_BASE,
    seed=1000,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v19c_qkvo_norm_ci_and_weight_renorm_seed1000",
)

# v21/v22/v23: v19 + homeostatic EnergyDynamics, three successive
# calibration corrections (wake_gate_steps -> no gate -> no gate + decay).
PRESETS["dense_lr_unscaled_plasticity_reset_v21_qkvo_norm_ci_weight_renorm_energy"] = dict(
    _V19_BASE,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v21_qkvo_norm_ci_weight_renorm_energy",
    use_energy=True,
    energy_kwargs={
        "state": {
            "drive": 0.0114,
            "activation_cost": 0.05,
            "precision": 0.01,
            "density": 0.5,
            "p": 0.9,
            "wake_gate_steps": 2500,
            "stagger_wake_init": True,
        },
        "embed_input": {
            "drive": 0.0025,
            "activation_cost": 0.05,
            "precision": 0.01,
            "density": 0.5,
            "p": 0.9,
            "wake_gate_steps": 2500,
            "stagger_wake_init": True,
        },
    },
)
PRESETS["dense_lr_unscaled_plasticity_reset_v22_qkvo_norm_ci_weight_renorm_energy_no_gate"] = dict(
    _V19_BASE,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v22_qkvo_norm_ci_weight_renorm_energy_no_gate",
    use_energy=True,
    energy_kwargs={
        "state": {"drive": 0.0171, "activation_cost": 0.05, "precision": 0.01, "density": 0.5, "p": 0.9},
        "embed_input": {"drive": 0.0047, "activation_cost": 0.05, "precision": 0.01, "density": 0.5, "p": 0.9},
    },
)
PRESETS["dense_lr_unscaled_plasticity_reset_v23_qkvo_norm_ci_weight_renorm_energy_decay"] = dict(
    _V19_BASE,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v23_qkvo_norm_ci_weight_renorm_energy_decay",
    use_energy=True,
    energy_kwargs={
        "state": {
            "drive": 0.01702,
            "activation_cost": 0.05,
            "precision": 0.01,
            "density": 0.5,
            "p": 0.9,
            "decay": 0.002,
        },
        "embed_input": {
            "drive": 0.00422,
            "activation_cost": 0.05,
            "precision": 0.01,
            "density": 0.5,
            "p": 0.9,
            "decay": 0.0005,
        },
    },
)

# v24: v19 config with amortized l2 decay and continue_past_graduation
# enabled -- the best result in the whole investigation, reaching a k
# of 5 at seed 1001.
_V24_BASE = dict(
    _V19_BASE,
    seed=1001,
    continue_past_graduation=True,
    l2_decay_chunk_size=800,
    l2_decay_adaptation_rate=0.3,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v24_l2_decay_continue_past_graduation",
)
PRESETS["dense_lr_unscaled_plasticity_reset_v24_l2_decay_continue_past_graduation"] = dict(_V24_BASE)

_SLEEP_ENERGY_KWARGS = {
    name: {
        "drive": 0.0775,
        "activation_cost": 0.05,
        "decay": 0.0075,
        "precision": 0.01,
        "density": 0.3,
        "p": 0.6,
        "exploration": 0.01,
        "reactivity": 0.01,
        "fire_wake_gradient": 1.0,
        "wake_seed": 7,
    }
    for name in ["input_proj", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]
}
# v25: sleep annealing on v19c's known-stuck config (seed=1000).
PRESETS["dense_lr_unscaled_v25_sleep_on_stuck_v19c"] = dict(
    PRESETS["dense_lr_unscaled_plasticity_reset_v19c_qkvo_norm_ci_and_weight_renorm_seed1000"],
    continue_past_graduation=True,
    sleep_enable=True,
    sleep_every_steps=5000,
    sleep_duration_steps=500,
    sleep_energy_kwargs=_SLEEP_ENERGY_KWARGS,
    plasticity_reset_reset_fraction=0.0,
    plasticity_raw_importance_log=True,
    qk_spectral_norm_diag_log=True,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v25_sleep_on_stuck_v19c",
)
# v26: same sleep config on v24's top-performer config (seed=1001).
PRESETS["dense_lr_unscaled_v26_sleep_on_top_performer_v24"] = dict(
    _V24_BASE,
    sleep_enable=True,
    sleep_every_steps=5000,
    sleep_duration_steps=500,
    sleep_energy_kwargs=_SLEEP_ENERGY_KWARGS,
    plasticity_reset_reset_fraction=0.0,
    plasticity_raw_importance_log=True,
    qk_spectral_norm_diag_log=True,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v26_sleep_on_top_performer_v24",
)
# v27/v28: freeze-gated sleep (sleep_gate_layers) instead of blind periodic.
PRESETS["dense_lr_unscaled_v27_freeze_gated_sleep_on_stuck_v19c"] = dict(
    {
        k: v
        for k, v in PRESETS["dense_lr_unscaled_v25_sleep_on_stuck_v19c"].items()
        if k not in ("sleep_every_steps", "sleep_duration_steps")
    },
    sleep_gate_layers=("q_proj", "k_proj"),
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v27_freeze_gated_sleep_on_stuck_v19c",
)
PRESETS["dense_lr_unscaled_v28_freeze_gated_sleep_on_top_performer_v24"] = dict(
    {
        k: v
        for k, v in PRESETS["dense_lr_unscaled_v26_sleep_on_top_performer_v24"].items()
        if k not in ("sleep_every_steps", "sleep_duration_steps")
    },
    sleep_gate_layers=("q_proj", "k_proj"),
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_v28_freeze_gated_sleep_on_top_performer_v24",
)

# ---------------------------------------------------------------------------
# Arm G/H/GH energy-based grad-selection/lr-scale family.
# ---------------------------------------------------------------------------
_ENERGY_GATE_KWARGS = {
    "input_proj": {"drive": 0.00444, "activation_cost": 0.05, "decay": 0.000889},
    "q_proj": {"drive": 0.01977, "activation_cost": 0.05, "decay": 0.003954},
    "k_proj": {"drive": 0.02559, "activation_cost": 0.05, "decay": 0.005117},
    "v_proj": {"drive": 0.02526, "activation_cost": 0.05, "decay": 0.005052},
    "o_proj": {"drive": 0.01423, "activation_cost": 0.05, "decay": 0.002846},
}
_ARM_GH_BASE = {
    "precision": "fp32",
    "max_steps": 100000,
    "seed": 1001,
    "peak_lr": 0.015,
    "num_tiles": 16,
    "k_max": 10,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "k_first_target": 3,
    "plasticity_reset_enable": True,
    "plasticity_reset_reset_fraction": 0.0,
    "plasticity_raw_importance_log": True,
    "qkvo_norm_enable": True,
    "qk_spectral_norm_diag_log": True,
    "ci_renorm_enable": "stable_region",
    "weight_renorm_enable": "stable_region",
    "log_every": 250,
    "num_cpus": 4,
    "k_start": 2,
}
PRESETS["dense_lr_unscaled_plasticity_reset_arm_g_energy_grad_selection"] = dict(
    _ARM_GH_BASE,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_arm_g_energy_grad_selection",
    dy_energy_gate_density=0.4,
    dy_energy_gate_kwargs=_ENERGY_GATE_KWARGS,
)
PRESETS["dense_lr_unscaled_plasticity_reset_arm_h_energy_lr_scale"] = dict(
    _ARM_GH_BASE,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_arm_h_energy_lr_scale",
    dy_energy_lr_scale_enable=True,
    dy_energy_gate_kwargs=_ENERGY_GATE_KWARGS,
)
PRESETS["dense_lr_unscaled_plasticity_reset_arm_gh_combined"] = dict(
    _ARM_GH_BASE,
    plasticity_column_log_dir="logs/plasticity_column_snapshots/dense_lr_unscaled_arm_gh_combined",
    dy_energy_gate_density=0.4,
    dy_energy_lr_scale_enable=True,
    dy_energy_gate_kwargs=_ENERGY_GATE_KWARGS,
)

# ---------------------------------------------------------------------------
# train_peridot_main.py -- NOT retired (it's the canonical entry point,
# README-documented separately), but its config is worth having here too
# for consistency/sweep-ability. Mirrors scripts/train_peridot_main.py
# exactly; keep both in sync if either changes.
# ---------------------------------------------------------------------------
PRESETS["peridot_main"] = {
    "precision": "fp32",
    "max_steps": 100000,
    "seed": 1001,
    "peak_lr": 0.015,
    "num_tiles": 16,
    "k_max": 10,
    "embed_width": 36,
    "wrong_streak_threshold": 100000000,
    "k_first_target": 3,
    "continue_past_graduation": True,
    "l2_decay_chunk_size": 800,
    "l2_decay_adaptation_rate": 0.3,
    "sleep_enable": True,
    "sleep_gate_layers": ("q_proj", "k_proj"),
    "sleep_energy_kwargs": _SLEEP_ENERGY_KWARGS,
    "plasticity_reset_enable": True,
    "plasticity_reset_reset_fraction": 0.0,
    "plasticity_column_log_dir": "logs/plasticity_column_snapshots/train_peridot_main",
    "plasticity_raw_importance_log": True,
    "qkvo_norm_enable": True,
    "qk_spectral_norm_diag_log": True,
    "ci_renorm_enable": "stable_region",
    "weight_renorm_enable": "stable_region",
    "log_every": 250,
    "num_cpus": 4,
    "k_start": 2,
}

# ---------------------------------------------------------------------------
# Sweep-axis definitions for sweep.py (replaces the 4-5 genuine multi-run
# sweep scripts that couldn't fold into PRESETS as single-call entries).
# Each value is (base_preset_name, [override_dict, ...]).
# ---------------------------------------------------------------------------
SWEEPS: dict[str, tuple[str, list[dict]]] = {
    # knee_margin_sweep.py: base preset must be "arm_c_plus_knee" (true
    # default k_start=1), NOT "arm_c_plus_knee_skip_k1" -- the original
    # file never touches K_START at all (verified directly by reading
    # it; an earlier draft of this dict used the wrong base preset).
    "knee_margin": (
        "arm_c_plus_knee",
        [
            {"max_steps": 4000, "x_r_target_auto_margin": m, "log_every": 4000}
            for m in (0.0, 0.03, 0.05, 0.08, 0.12, 0.2)
        ],
    ),
    # armc_gate_density_lr_seed_sweep.py: seeds are (1000, 2000, 3000),
    # NOT (1000, 1001, 1002) -- verified directly by reading the
    # original file's own SEEDS constant (an earlier draft of this
    # dict had the wrong seed values).
    "armc_gate_density_lr_seed": (
        "armc_gatemid_lr_unscaled",
        [
            {"max_steps": 25000, "seed": seed, "peak_lr": lr, "dy_time_gate_cutoff": cutoff, "log_every": 25000}
            for cutoff in (0.0, 0.6)
            for lr in (0.015, 0.01)
            for seed in (1000, 2000, 3000)
        ],
    ),
}
