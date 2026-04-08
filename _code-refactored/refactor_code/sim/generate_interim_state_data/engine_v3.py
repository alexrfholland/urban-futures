from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.sim.setup import a_scenario_initialiseDS, params_v3  # noqa: E402
from refactor_code.sim.setup.constants import (  # noqa: E402
    COLONISE_FULL_GROUND,
    DECAY_FULL,
    DECAY_PARTIAL,
    DEPLOY_FULL_LOG,
    DEPLOY_FULL_POLE,
    RECRUIT_FULL,
    RECRUIT_PARTIAL,
    RELEASECONTROL_FULL,
    RELEASECONTROL_PARTIAL,
)

from refactor_code.paths import (  # noqa: E402
    scenario_log_df_path,
    scenario_pole_df_path,
    scenario_tree_df_path,
)
from refactor_code.sim.setup.structure_ids import (  # noqa: E402
    assign_log_structure_ids,
    assign_pole_structure_ids,
    assign_tree_structure_ids,
    collect_structure_ids,
    replacement_structure_ids,
)


ENGINE_PULSE_INTERVAL = 10
RECRUIT_INTERVAL = 30
ABSENT_TREE_SIZES = {"gone", "early-tree-death"}
NON_OCCUPYING_TREE_SIZES = {"snag", "fallen", "decayed", *ABSENT_TREE_SIZES}
DISTANCE_THRESHOLD_METERS = 5.0
RECRUIT_SPACING_THRESHOLD_METERS = 1.5
RELAXED_RECRUIT_SPACING_THRESHOLD_METERS = 0.5
BUFFER_RECRUIT_PARENT_OFFSET_METERS = 2.5
LOW_CONTROL_STATE = "low-control"
REDUCE_PRUNING_TO_PARK_YEARS = 20
ELIMINATE_PRUNING_TO_PARK_YEARS = 20
ELIMINATE_PRUNING_TO_LOW_YEARS = 40
PRUNING_TARGET_ORDER = {
    "standard-pruning": 0,
    RELEASECONTROL_PARTIAL: 1,
    RELEASECONTROL_FULL: 2,
}
UNDER_NODE_TREATMENT_ORDER = {
    "paved": 0,
    "none": 0,
    "None": 0,
    "exoskeleton": 1,
    "footprint-depaved": 2,
    "node-rewilded": 3,
}
RELEASE_SUPPORT_BY_REWILDING = {
    "node-rewilded": RELEASECONTROL_FULL,
    "footprint-depaved": RELEASECONTROL_PARTIAL,
}
DECAY_INTERVENTION_BY_TREATMENT = {
    "node-rewilded": DECAY_FULL,
    "footprint-depaved": DECAY_FULL,
    "exoskeleton": DECAY_PARTIAL,
}
RELEASE_TREATMENT_BY_INTERVENTION = {
    RELEASECONTROL_PARTIAL: "footprint-depaved",
    RELEASECONTROL_FULL: "node-rewilded",
    "standard-pruning": "paved",
    "none": "paved",
}
# Le Roux-shaped mortality curves, anchored to the current annual mortality
# defaults. Only cohorts up to 70-80 cm are used because this mortality pass
# only applies to trees currently classed as small or medium.
URBAN_MORTALITY_CURVE_FACTORS = np.array(
    [1.0, 0.75, 0.5833333333, 0.5, 0.4333333333, 0.35, 0.25, 0.1666666667],
    dtype=float,
)
RESERVE_MORTALITY_CURVE_FACTORS = np.array(
    [1.0, 0.8, 0.6333333333, 0.5, 0.4, 0.3, 0.2333333333, 0.1666666667],
    dtype=float,
)


def remap_values(values, old_min, old_max, new_min, new_max):
    if old_max == old_min:
        return np.full_like(values, new_max, dtype=float)
    return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def _sample_triangular_years(spec: dict, size: int) -> np.ndarray:
    minimum = float(spec["min"])
    mode = float(spec["mode"])
    maximum = float(spec["max"])
    samples = np.random.triangular(minimum, mode, maximum, size=size)
    return np.rint(samples).astype(int)


def _as_object_series(values, length, default):
    if values is None:
        return pd.Series([default] * length, dtype="object")
    series = pd.Series(values)
    return series.fillna(default).astype(str)


def _blank_like(series: pd.Series, extra_tokens: set[str] | None = None) -> pd.Series:
    tokens = {"", "nan", "none", "None"}
    if extra_tokens:
        tokens |= extra_tokens
    normalized = series.fillna("").astype(str).str.strip()
    return normalized.isin(tokens)


def _numeric_series(values, default: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.fillna(default).astype(float)


def _under_node_treatment_rank(values: pd.Series) -> pd.Series:
    return values.map(UNDER_NODE_TREATMENT_ORDER).fillna(0).astype(int)


def _merge_under_node_treatments(current: pd.Series, proposed: pd.Series) -> pd.Series:
    current_series = current.fillna("paved").astype(str)
    proposed_series = proposed.fillna("paved").astype(str)
    choose_proposed = _under_node_treatment_rank(proposed_series) > _under_node_treatment_rank(current_series)
    return pd.Series(
        np.where(choose_proposed.to_numpy(dtype=bool), proposed_series.to_numpy(dtype=object), current_series.to_numpy(dtype=object)),
        index=current.index,
        dtype="object",
    )


def _legacy_control_to_target(control_value: str) -> str:
    if control_value == "park-tree":
        return RELEASECONTROL_PARTIAL
    if control_value in {"reserve-tree", "improved-tree"}:
        return RELEASECONTROL_FULL
    return "standard-pruning"


def _legacy_control_to_realized(control_value: str) -> str:
    if control_value in {"reserve-tree", "improved-tree"}:
        return LOW_CONTROL_STATE
    if control_value == "park-tree":
        return "park-tree"
    return "street-tree"


def _legacy_control_to_years(control_value: str) -> int:
    if control_value == "park-tree":
        return REDUCE_PRUNING_TO_PARK_YEARS
    if control_value in {"reserve-tree", "improved-tree"}:
        return ELIMINATE_PRUNING_TO_LOW_YEARS
    return 0


def _export_control_value(realized_control: str, size_value: str) -> str:
    if size_value in {"senescing", "snag", "fallen", "decayed"}:
        return "improved-tree"
    if realized_control == LOW_CONTROL_STATE:
        return "reserve-tree"
    return realized_control


def _seeded_years(*parts: object, minimum: int, maximum: int) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    seed_value = int(digest[:16], 16)
    return minimum + (seed_value % (maximum - minimum + 1))


def _seeded_unit_float(*parts: object) -> float:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    seed_value = int(digest[:16], 16)
    return seed_value / float(16**16 - 1)


def _seeded_triangular_year(*parts: object, minimum: float, mode: float, maximum: float) -> int:
    if maximum <= minimum:
        return int(round(minimum))

    unit = _seeded_unit_float(*parts)
    split = (mode - minimum) / (maximum - minimum) if maximum > minimum else 0.5

    if unit <= split:
        sample = minimum + np.sqrt(unit * (maximum - minimum) * (mode - minimum))
    else:
        sample = maximum - np.sqrt((1.0 - unit) * (maximum - minimum) * (maximum - mode))
    return int(round(sample))


def _seeded_year_within_window(*parts: object, window_start: int, window_end: int) -> int:
    if window_end <= window_start:
        return int(window_end)
    return _seeded_years(*parts, minimum=int(window_start), maximum=int(window_end))


def _site_specific_structures(site: str, log_df: pd.DataFrame | None, pole_df: pd.DataFrame | None):
    if site == "trimmed-parade":
        return None, None
    if site == "uni":
        log_df = None
    if site == "city":
        pole_df = None
    return log_df, pole_df


def _with_node_type(df: pd.DataFrame | None, node_type: str) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    df["nodeType"] = node_type
    return df


def _refresh_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    legacy_schema_columns = [
        "rewilded",
        "lifecycle_decision",
        "control_realized",
        "pruning_target",
        "pruning_target_years",
        "autonomy_years",
    ]
    present_legacy_columns = [column for column in legacy_schema_columns if column in df.columns]
    if present_legacy_columns:
        raise ValueError(
            "Legacy v3/v2 schema columns are no longer supported. "
            f"Regenerate the inputs with the renamed fields instead: {present_legacy_columns}"
        )

    if "useful_life_expectency" in df.columns and "useful_life_expectancy" not in df.columns:
        df.rename(columns={"useful_life_expectency": "useful_life_expectancy"}, inplace=True)
    if "fallen_remove_after_years" in df.columns and "fallen_decay_after_years" not in df.columns:
        df.rename(columns={"fallen_remove_after_years": "fallen_decay_after_years"}, inplace=True)

    for column, default in {
        "nodeType": "tree",
        "under-node-treatment": "paved",
        "action": "None",
        "replacement_reason": "none",
        "isNewTree": False,
        "isRewildedTree": False,
        "hasbeenReplanted": False,
        "unmanagedCount": 0.0,
        "proposal-decay_decision": "not-assessed",
        "proposal-decay_intervention": "none",
        "proposal-release-control_decision": "not-assessed",
        "proposal-release-control_intervention": "none",
        "proposal-release-control_target_years": np.nan,
        "proposal-release-control_years": np.nan,
        "proposal-recruit_decision": "not-assessed",
        "proposal-recruit_intervention": "none",
        "recruit_intervention_type": "none",
        "recruit_source_id": "none",
        "recruit_year": np.nan,
        "proposal-colonise_decision": "not-assessed",
        "proposal-colonise_intervention": "none",
        "proposal-deploy-structure_decision": "not-assessed",
        "proposal-deploy-structure_intervention": "none",
        "control_reached": None,
        "lifecycle_state": None,
        "early_death_at_year": np.nan,
        "became_large_at_year": np.nan,
        "became_senescing_at_year": np.nan,
        "became_snag_at_year": np.nan,
        "senescing_duration_years": np.nan,
        "snag_duration_years": np.nan,
        "became_fallen_at_year": np.nan,
        "became_decayed_at_year": np.nan,
        "became_gone_at_year": np.nan,
        "fallen_decay_after_years": np.nan,
        "decayed_remove_after_years": np.nan,
        "structureID": np.nan,
    }.items():
        if column not in df.columns:
            df[column] = default

    df["size"] = _as_object_series(df.get("size"), len(df), "small")
    df["control"] = _as_object_series(df.get("control"), len(df), "street-tree")
    df["under-node-treatment"] = _as_object_series(df.get("under-node-treatment"), len(df), "paved")
    df["action"] = _as_object_series(df.get("action"), len(df), "None")
    df["replacement_reason"] = _as_object_series(df.get("replacement_reason"), len(df), "none")
    for column, default in {
        "proposal-decay_decision": "not-assessed",
        "proposal-decay_intervention": "none",
        "proposal-release-control_decision": "not-assessed",
        "proposal-release-control_intervention": "none",
        "proposal-recruit_decision": "not-assessed",
        "proposal-recruit_intervention": "none",
        "proposal-colonise_decision": "not-assessed",
        "proposal-colonise_intervention": "none",
        "proposal-deploy-structure_decision": "not-assessed",
        "proposal-deploy-structure_intervention": "none",
    }.items():
        df[column] = _as_object_series(df.get(column), len(df), default)
    df["recruit_intervention_type"] = _as_object_series(df.get("recruit_intervention_type"), len(df), "none")
    df["recruit_source_id"] = _as_object_series(df.get("recruit_source_id"), len(df), "none")
    df["recruit_mechanism"] = _as_object_series(df.get("recruit_mechanism"), len(df), "none")

    df["under-node-treatment"] = df["under-node-treatment"].replace({"none": "paved", "None": "paved"})
    if df["control_reached"].isna().any():
        missing = df["control_reached"].isna()
        df.loc[missing, "control_reached"] = df.loc[missing, "control"].map(_legacy_control_to_realized)
    df["control_reached"] = _as_object_series(df["control_reached"], len(df), "street-tree")

    living_mask = df["size"].isin(["small", "medium", "large"])

    blank_decay_decision = _blank_like(df["proposal-decay_decision"], {"not-assessed"})
    df.loc[blank_decay_decision, "proposal-decay_decision"] = np.where(
        df.loc[blank_decay_decision, "size"].isin(["senescing", "snag", "fallen", "decayed"]),
        "proposal-decay_accepted",
        "not-assessed",
    )
    blank_decay_intervention = _blank_like(df["proposal-decay_intervention"], {"not-assessed"})
    blank_colonise_intervention = _blank_like(df["proposal-colonise_intervention"], {"not-assessed"})
    blank_colonise_decision = _blank_like(df["proposal-colonise_decision"], {"not-assessed"})
    df.loc[blank_colonise_decision, "proposal-colonise_decision"] = np.where(
        df.loc[blank_colonise_decision, "proposal-colonise_intervention"].eq(COLONISE_FULL_GROUND),
        "proposal-colonise_accepted",
        "proposal-colonise_rejected",
    )
    blank_recruit_intervention = _blank_like(df["proposal-recruit_intervention"], {"not-assessed"})
    blank_recruit_decision = _blank_like(df["proposal-recruit_decision"], {"not-assessed"})
    df.loc[blank_recruit_decision, "proposal-recruit_decision"] = np.where(
        df.loc[blank_recruit_decision, "proposal-recruit_intervention"].eq("none"),
        "not-assessed",
        "proposal-recruit_accepted",
    )
    blank_deploy_intervention = _blank_like(df["proposal-deploy-structure_intervention"], {"not-assessed"})
    blank_deploy_decision = _blank_like(df["proposal-deploy-structure_decision"], {"not-assessed"})
    df.loc[blank_deploy_decision, "proposal-deploy-structure_decision"] = np.where(
        df.loc[blank_deploy_decision, "proposal-deploy-structure_intervention"].eq("none"),
        "not-assessed",
        "proposal-deploy-structure_accepted",
    )

    if df["lifecycle_state"].isna().any():
        missing = df["lifecycle_state"].isna()
        state = np.where(
            df["size"].isin(["senescing", "snag", "fallen", "decayed"]),
            df["size"],
            "standing",
        )
        df.loc[missing, "lifecycle_state"] = state[missing]
    df["lifecycle_state"] = _as_object_series(df["lifecycle_state"], len(df), "standing")
    df.drop(
        columns=[
            "decay_support",
            "release_control_support",
            "recruit_support",
            "colonise_support",
            "deploy_structure_support",
        ],
        errors="ignore",
        inplace=True,
    )

    for numeric_column, default in {
        "useful_life_expectancy": 120.0,
        "diameter_breast_height": 10.0,
        "CanopyArea": 1.0,
        "sim_NodesArea": np.nan,
        "CanopyResistance": np.nan,
        "sim_NodesVoxels": 1.0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    }.items():
        if numeric_column not in df.columns:
            df[numeric_column] = default

    # Some legacy site inputs carry the column but leave row values blank.
    ule_series = pd.to_numeric(df["useful_life_expectancy"], errors="coerce")
    size_based_ule = df["size"].map({
        "small": 80.0,
        "medium": 50.0,
        "large": 10.0,
    })
    df["useful_life_expectancy"] = ule_series.fillna(size_based_ule).fillna(120.0)

    df["sim_NodesArea"] = df["sim_NodesArea"].fillna(df["CanopyArea"])
    default_release_years = pd.to_numeric(df["control"].map(_legacy_control_to_years), errors="coerce").fillna(0.0)
    df["proposal-release-control_target_years"] = pd.to_numeric(
        df["proposal-release-control_target_years"], errors="coerce"
    ).fillna(default_release_years)
    df["proposal-release-control_years"] = pd.to_numeric(
        df["proposal-release-control_years"], errors="coerce"
    ).fillna(default_release_years)

    non_living_mask = ~df["size"].isin(["small", "medium", "large"])
    if non_living_mask.any():
        df.loc[non_living_mask, "proposal-release-control_decision"] = "not-assessed"
        df.loc[non_living_mask, "proposal-release-control_intervention"] = "none"
        df.loc[non_living_mask, "proposal-release-control_target_years"] = 0.0
        df.loc[non_living_mask, "proposal-release-control_years"] = 0.0

    df["recruit_year"] = pd.to_numeric(df["recruit_year"], errors="coerce")
    df["early_death_at_year"] = pd.to_numeric(df["early_death_at_year"], errors="coerce")
    df["became_large_at_year"] = pd.to_numeric(df["became_large_at_year"], errors="coerce")
    df["became_senescing_at_year"] = pd.to_numeric(df["became_senescing_at_year"], errors="coerce")
    df["became_snag_at_year"] = pd.to_numeric(df["became_snag_at_year"], errors="coerce")
    df["senescing_duration_years"] = pd.to_numeric(df["senescing_duration_years"], errors="coerce")
    df["snag_duration_years"] = pd.to_numeric(df["snag_duration_years"], errors="coerce")
    df["became_fallen_at_year"] = pd.to_numeric(df["became_fallen_at_year"], errors="coerce")
    df["became_decayed_at_year"] = pd.to_numeric(df["became_decayed_at_year"], errors="coerce")
    df["became_gone_at_year"] = pd.to_numeric(df["became_gone_at_year"], errors="coerce")
    df["fallen_decay_after_years"] = pd.to_numeric(df["fallen_decay_after_years"], errors="coerce")
    df["decayed_remove_after_years"] = pd.to_numeric(df["decayed_remove_after_years"], errors="coerce")
    return df


def get_timestep_params(site: str, scenario: str, absolute_year: int, previous_year: int) -> dict:
    params = params_v3.get_params_for_year(site, scenario, absolute_year)
    params["absolute_year"] = absolute_year
    params["previous_year"] = previous_year
    params["step_years"] = max(0, absolute_year - previous_year)
    return params


def split_window_into_pulses(previous_year: int, absolute_year: int, max_pulse_years: int = ENGINE_PULSE_INTERVAL):
    if absolute_year <= previous_year:
        return []

    pulses = []
    pulse_start = previous_year
    while pulse_start < absolute_year:
        pulse_end = min(pulse_start + max_pulse_years, absolute_year)
        pulses.append((pulse_start, pulse_end))
        pulse_start = pulse_end
    return pulses


def _grow_dbh_constant(dbh: np.ndarray, step_years: float, params: dict) -> np.ndarray:
    """Original constant growth: mean of growth_factor_range per year."""
    growth_factor = np.mean(params["growth_factor_range"])
    return dbh + growth_factor * step_years


def _grow_dbh_fischer(dbh: np.ndarray, step_years: float, _params: dict) -> np.ndarray:
    """Fischer SI (yellow box): Age = 0.0197135 * pi * (DBH/2)^2.

    Virtual-age round-trip: DBH -> age -> advance -> new DBH.
    Used by Le Roux for their cohort model.
    """
    k = 0.0197135 * np.pi / 4.0  # 0.01548
    virtual_age = k * dbh ** 2
    new_age = virtual_age + step_years
    return np.sqrt(new_age / k)


def _grow_dbh_sideroxylon(dbh: np.ndarray, step_years: float, _params: dict) -> np.ndarray:
    """E. sideroxylon (USFS InlEmp): DBH = 4.85 + 1.82*Age - 0.011*Age^2.

    Quadratic, valid to ~80 cm DBH.  R^2 = 0.963.
    """
    a, b, c = 4.85016, 1.82135, -0.01093
    disc = b ** 2 - 4 * (-c) * (dbh - a)
    disc = np.maximum(disc, 0.0)
    virtual_age = (b - np.sqrt(disc)) / (2 * (-c))
    virtual_age = np.maximum(virtual_age, 0.0)
    new_age = virtual_age + step_years
    new_dbh = a + b * new_age + c * new_age ** 2
    return np.maximum(new_dbh, dbh)  # never shrink


def _grow_dbh_banks(dbh: np.ndarray, step_years: float, _params: dict) -> np.ndarray:
    """Banks power law (yellow box): DBH = 4.50 * Age^0.6.

    Empirical fit to Canberra yellow box cohorts.
    """
    coeff, exp = 4.50, 0.6
    virtual_age = (dbh / coeff) ** (1.0 / exp)
    new_age = virtual_age + step_years
    return coeff * new_age ** exp


def _grow_dbh_ulmus(dbh: np.ndarray, step_years: float, _params: dict) -> np.ndarray:
    """Ulmus americana (USFS PacfNW): DBH = -0.707 + 1.817*Age - 0.005*Age^2.

    Quadratic, valid to ~138 cm DBH.  R^2 = 0.982.
    """
    a, b, c = -0.70738, 1.81652, -0.00501
    disc = b ** 2 - 4 * (-c) * (dbh - a)
    disc = np.maximum(disc, 0.0)
    virtual_age = (b - np.sqrt(disc)) / (2 * (-c))
    virtual_age = np.maximum(virtual_age, 0.0)
    new_age = virtual_age + step_years
    new_dbh = a + b * new_age + c * new_age ** 2
    return np.maximum(new_dbh, dbh)  # never shrink


GROWTH_MODELS = {
    "constant": _grow_dbh_constant,
    "fischer": _grow_dbh_fischer,
    "sideroxylon": _grow_dbh_sideroxylon,
    "banks": _grow_dbh_banks,
    "ulmus": _grow_dbh_ulmus,
}


def age_trees(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    step_years = params["step_years"]
    if step_years <= 0:
        return df

    growth_model = params.get("growth_model", "constant")
    living_mask = df["size"].isin(["small", "medium", "large"])
    previous_size = df["size"].copy()
    dbh = df.loc[living_mask, "diameter_breast_height"].values.astype(float)

    # Per-tree growth noise: each tree experiences ±15% variation in effective
    # growing time, representing natural variation in site conditions.
    seed = int(params.get("seed", 42))
    growth_rng = np.random.default_rng(seed + 6000 + int(params["absolute_year"]))
    n_living = int(living_mask.sum())
    effective_step = step_years * growth_rng.uniform(0.85, 1.15, size=n_living)

    if growth_model == "split":
        # Colonial trees use elm curve, precolonial use eucalypt curve
        euc_model = params.get("growth_model_precolonial", "fischer")
        elm_model = params.get("growth_model_colonial", "ulmus")
        precol = df.loc[living_mask, "precolonial"].values.astype(bool)
        new_dbh = dbh.copy()
        if precol.any():
            new_dbh[precol] = GROWTH_MODELS[euc_model](dbh[precol], effective_step[precol], params)
        if (~precol).any():
            new_dbh[~precol] = GROWTH_MODELS[elm_model](dbh[~precol], effective_step[~precol], params)
        df.loc[living_mask, "diameter_breast_height"] = new_dbh
    else:
        grow_fn = GROWTH_MODELS[growth_model]
        df.loc[living_mask, "diameter_breast_height"] = grow_fn(dbh, effective_step, params)

    growth_mask = df["size"].isin(["small", "medium"])
    df.loc[growth_mask, "size"] = pd.cut(
        df.loc[growth_mask, "diameter_breast_height"],
        bins=[-10, 30, 80, float("inf")],
        labels=["small", "medium", "large"],
    ).astype(str)
    newly_large_mask = growth_mask & previous_size.ne("large") & df["size"].eq("large")
    df.loc[newly_large_mask & df["became_large_at_year"].isna(), "became_large_at_year"] = int(params["absolute_year"])

    df["useful_life_expectancy"] = df["useful_life_expectancy"] - step_years
    df.loc[df["size"].isin(["small", "medium", "large"]), "lifecycle_state"] = "standing"
    return df


def _mortality_dbh_cohorts(dbh_values: pd.Series) -> pd.Series:
    """Return 10 cm DBH cohort indices for mortality thinning."""
    dbh = _numeric_series(dbh_values, default=0.0).clip(lower=0.0)
    cohort_index = np.floor(dbh / 10.0).astype(int).clip(lower=0)
    return pd.Series(cohort_index, index=dbh_values.index, dtype="int64")


def _annual_mortality_rate_for_cohort(cohort: int, *, reserve_like: bool, params: dict) -> float:
    anchor = float(
        params.get(
            "annual_tree_death_nature-reserves" if reserve_like else "annual_tree_death_urban",
            0.03 if reserve_like else 0.06,
        )
    )
    mortality_model = params.get("mortality_model", "shaped")
    if mortality_model == "flat":
        # Flat Le Roux: same anchor rate for all cohorts.
        return float(np.clip(anchor, 0.0, 1.0))
    # Default shaped: anchor × cohort-specific factor.
    factors = RESERVE_MORTALITY_CURVE_FACTORS if reserve_like else URBAN_MORTALITY_CURVE_FACTORS
    cohort_idx = int(np.clip(cohort, 0, len(factors) - 1))
    return float(np.clip(anchor * factors[cohort_idx], 0.0, 1.0))


def apply_annual_tree_mortality(df: pd.DataFrame, params: dict, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    step_years = int(params["step_years"])
    if step_years <= 0:
        return df

    candidate_mask = df["size"].isin(["small", "medium", "large"])
    if not candidate_mask.any():
        return df

    reserve_like_mask = candidate_mask & (
        df["proposal-recruit_intervention"].eq(RECRUIT_FULL)
        | df["recruit_intervention_type"].eq(RECRUIT_FULL)
    )
    dbh_cohorts = _mortality_dbh_cohorts(df["diameter_breast_height"])
    rng = np.random.default_rng(seed + 2000 + int(params["absolute_year"]))
    mortality_mask = pd.Series(False, index=df.index, dtype=bool)

    # Thin small/medium/large trees by DBH cohort so each cohort retains a
    # stable surviving proportion, rather than rolling every row independently.
    for is_reserve_like in (False, True):
        context_mask = candidate_mask & reserve_like_mask.eq(is_reserve_like)
        if not context_mask.any():
            continue
        context_cohorts = dbh_cohorts.loc[context_mask]

        for cohort in sorted(context_cohorts.unique()):
            cohort_index = context_cohorts.index[context_cohorts.eq(int(cohort))]
            cohort_size = int(len(cohort_index))
            if cohort_size == 0:
                continue

            annual_rate = _annual_mortality_rate_for_cohort(int(cohort), reserve_like=is_reserve_like, params=params)
            # Large trees (cohort 8+ = 80cm+) get half the mortality rate
            if cohort >= 8:
                annual_rate *= 0.5
            step_death_probability = 1.0 - np.power(1.0 - annual_rate, step_years)
            survivor_count = int(np.rint(cohort_size * (1.0 - step_death_probability)))
            survivor_count = max(0, min(cohort_size, survivor_count))
            death_count = cohort_size - survivor_count
            if death_count <= 0:
                continue

            dead_index = rng.choice(cohort_index.to_numpy(), size=death_count, replace=False)
            mortality_mask.loc[dead_index] = True

    if not mortality_mask.any():
        return df

    recruit_mortality_mask = mortality_mask & df["isNewTree"].fillna(False)
    existing_dieback_mask = mortality_mask & ~df["isNewTree"].fillna(False)

    df.loc[recruit_mortality_mask, "size"] = "early-tree-death"
    df.loc[recruit_mortality_mask, "lifecycle_state"] = "early-tree-death"
    df.loc[recruit_mortality_mask, "replacement_reason"] = "annual-tree-death"
    df.loc[recruit_mortality_mask, "early_death_at_year"] = int(params["absolute_year"])
    df.loc[recruit_mortality_mask, "action"] = "None"
    df.loc[recruit_mortality_mask, "under-node-treatment"] = "paved"
    df.loc[recruit_mortality_mask, "proposal-decay_decision"] = "not-assessed"
    df.loc[recruit_mortality_mask, "proposal-decay_intervention"] = "none"
    df.loc[recruit_mortality_mask, "proposal-release-control_decision"] = "not-assessed"
    df.loc[recruit_mortality_mask, "proposal-release-control_intervention"] = "none"
    df.loc[recruit_mortality_mask, "proposal-release-control_target_years"] = 0.0
    df.loc[recruit_mortality_mask, "proposal-release-control_years"] = 0.0
    df.loc[recruit_mortality_mask, "proposal-recruit_decision"] = "not-assessed"
    df.loc[recruit_mortality_mask, "proposal-recruit_intervention"] = "none"
    df.loc[recruit_mortality_mask, "recruit_intervention_type"] = "none"
    df.loc[recruit_mortality_mask, "recruit_source_id"] = "none"
    df.loc[recruit_mortality_mask, "proposal-colonise_decision"] = "not-assessed"
    df.loc[recruit_mortality_mask, "proposal-colonise_intervention"] = "none"

    df.loc[existing_dieback_mask, "action"] = "REPLACE"
    df.loc[existing_dieback_mask, "lifecycle_state"] = "standing"
    df.loc[existing_dieback_mask, "replacement_reason"] = "early-dieback-replant"
    df.loc[existing_dieback_mask, "proposal-decay_decision"] = "not-assessed"
    df.loc[existing_dieback_mask, "proposal-decay_intervention"] = "none"
    return df


def determine_proposal_decay(df: pd.DataFrame, params: dict, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    np.random.seed(seed + int(params["absolute_year"]))

    living_mask = df["size"].isin(["small", "medium", "large"])
    preexisting_replace_mask = living_mask & df["action"].eq("REPLACE")
    lifecycle_living_mask = living_mask & ~preexisting_replace_mask
    df.loc[lifecycle_living_mask, "action"] = "None"
    df.loc[lifecycle_living_mask, "replacement_reason"] = "none"
    df.loc[lifecycle_living_mask, "proposal-decay_decision"] = "not-assessed"
    df.loc[lifecycle_living_mask, "proposal-decay_intervention"] = "none"

    if not lifecycle_living_mask.any():
        return df

    senescing_ramp_start = params["lifecycle_senescing_ramp_start"]
    df["proposal_decay_chance"] = remap_values(
        df["useful_life_expectancy"].to_numpy(dtype=float),
        old_min=senescing_ramp_start,
        old_max=0,
        new_min=100,
        new_max=0,
    ).clip(0, 100)
    df["proposal_decay_roll"] = np.random.uniform(0, 100, len(df))

    proposal_decay_mask = lifecycle_living_mask & (df["proposal_decay_roll"] < df["proposal_decay_chance"])
    age_in_place_mask = proposal_decay_mask & (
        df["CanopyResistance"] < params["minimal-tree-support-threshold"]
    )
    replace_mask = proposal_decay_mask & ~age_in_place_mask

    df.loc[age_in_place_mask, "action"] = "AGE-IN-PLACE"
    df.loc[age_in_place_mask, "proposal-decay_decision"] = "proposal-decay_accepted"
    df.loc[replace_mask, "action"] = "REPLACE"
    df.loc[replace_mask, "proposal-decay_decision"] = "proposal-decay_rejected"
    df.loc[replace_mask, "_decay_rejected_this_state"] = True
    return df


def apply_proposal_decay_accepted_lifecycle_changes(
    df: pd.DataFrame,
    params: dict,
    seed: int = 42,
) -> pd.DataFrame:
    df = df.copy()
    np.random.seed(seed + 1000 + int(params["absolute_year"]))
    current_year = int(params["absolute_year"])

    age_in_place_mask = (
        df["proposal-decay_decision"].eq("proposal-decay_accepted")
        | df["size"].isin(["senescing", "snag", "fallen", "decayed"])
    )

    new_senescing_mask = age_in_place_mask & df["size"].isin(["small", "medium", "large"])
    df.loc[new_senescing_mask, "size"] = "senescing"
    df.loc[new_senescing_mask & df["became_senescing_at_year"].isna(), "became_senescing_at_year"] = current_year
    missing_senescing_duration_mask = df["size"].eq("senescing") & df["senescing_duration_years"].isna()
    if missing_senescing_duration_mask.any():
        df.loc[missing_senescing_duration_mask, "senescing_duration_years"] = _sample_triangular_years(
            params["senescing_duration_years"],
            int(missing_senescing_duration_mask.sum()),
        )

    senescing_to_snag_mask = (
        df["size"].eq("senescing")
        & df["became_senescing_at_year"].notna()
        & df["senescing_duration_years"].notna()
        & ((current_year - df["became_senescing_at_year"]) >= df["senescing_duration_years"])
    )
    df.loc[senescing_to_snag_mask, "size"] = "snag"
    df.loc[senescing_to_snag_mask & df["became_snag_at_year"].isna(), "became_snag_at_year"] = current_year

    new_snag_duration_mask = df["size"].eq("snag") & df["snag_duration_years"].isna()
    if new_snag_duration_mask.any():
        df.loc[new_snag_duration_mask, "snag_duration_years"] = _sample_triangular_years(
            params["snag_duration_years"],
            int(new_snag_duration_mask.sum()),
        )

    snag_to_fallen_mask = (
        df["size"].eq("snag")
        & df["became_snag_at_year"].notna()
        & df["snag_duration_years"].notna()
        & ((current_year - df["became_snag_at_year"]) >= df["snag_duration_years"])
    )
    brace_collapse_mask = snag_to_fallen_mask & df["proposal-decay_intervention"].eq(DECAY_PARTIAL)
    df.loc[snag_to_fallen_mask & ~brace_collapse_mask, "size"] = "fallen"
    df.loc[brace_collapse_mask, "action"] = "REPLACE"
    df.loc[brace_collapse_mask, "replacement_reason"] = "brace-collapse"

    state_values = np.where(
        df["size"].isin(["senescing", "snag", "fallen", "decayed"]),
        df["size"],
        "standing",
    )
    df["lifecycle_state"] = state_values
    df.loc[df["size"].isin(["senescing", "snag", "fallen", "decayed"]), "proposal-decay_decision"] = "proposal-decay_accepted"
    return df


def apply_proposal_decay_rejected_changes(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    replace_mask = df["action"].eq("REPLACE")
    if not replace_mask.any():
        return df

    step_years = params["step_years"]
    growth_factor = np.mean(params["growth_factor_range"])
    current_ule = df.loc[replace_mask, "useful_life_expectancy"].to_numpy(dtype=float)
    replacement_growth_years = np.clip(-np.minimum(0, current_ule), 0, step_years)
    # Stagger replant DBH using Fischer growth with random arrival within pulse.
    fischer_k = 0.0197135 * np.pi / 4.0
    seed = int(params.get("seed", 42))
    replant_rng = np.random.default_rng(seed + 5000 + int(params["absolute_year"]))
    arrival_offset = replant_rng.uniform(0, step_years, size=replace_mask.sum())
    growth_time = np.maximum(replacement_growth_years, step_years - arrival_offset)
    start_age = fischer_k * 4.0  # 2cm start
    replacement_dbh = np.sqrt((start_age + growth_time) / fischer_k)

    df.loc[replace_mask, "diameter_breast_height"] = np.round(replacement_dbh, 1)
    df.loc[replace_mask, "action"] = "None"
    df.loc[replace_mask, "precolonial"] = True
    df.loc[replace_mask, "useful_life_expectancy"] = 120 - replacement_growth_years
    df.loc[replace_mask, "hasbeenReplanted"] = True
    df.loc[replace_mask, "size"] = pd.cut(
        df.loc[replace_mask, "diameter_breast_height"],
        bins=[-10, 30, 80, float("inf")],
        labels=["small", "medium", "large"],
    ).astype(str)
    df.loc[replace_mask, "lifecycle_state"] = "standing"
    # Ground treatment persists — the site improvement survives the tree.
    # under-node-treatment is NOT reset to "paved".
    df.loc[replace_mask, "proposal-decay_decision"] = "not-assessed"
    df.loc[replace_mask, "proposal-decay_intervention"] = "none"
    # Clear release-control accumulators so the new sapling starts fresh,
    # but the ground treatment (under-node-treatment) is inherited.
    df.loc[replace_mask, "proposal-release-control_decision"] = "not-assessed"
    df.loc[replace_mask, "proposal-release-control_intervention"] = "none"
    df.loc[replace_mask, "proposal-release-control_target_years"] = 0.0
    df.loc[replace_mask, "proposal-release-control_years"] = 0.0
    # Clear recruit/colonise state — new tree, fresh assessment.
    df.loc[replace_mask, "proposal-recruit_decision"] = "not-assessed"
    df.loc[replace_mask, "proposal-recruit_intervention"] = "none"
    df.loc[replace_mask, "recruit_intervention_type"] = "none"
    df.loc[replace_mask, "recruit_source_id"] = "none"
    df.loc[replace_mask, "proposal-colonise_decision"] = "not-assessed"
    df.loc[replace_mask, "proposal-colonise_intervention"] = "none"
    df.loc[replace_mask, "early_death_at_year"] = np.nan
    df.loc[replace_mask, "became_large_at_year"] = np.nan
    df.loc[replace_mask, "became_senescing_at_year"] = np.nan
    df.loc[replace_mask, "became_snag_at_year"] = np.nan
    df.loc[replace_mask, "senescing_duration_years"] = np.nan
    df.loc[replace_mask, "snag_duration_years"] = np.nan
    df.loc[replace_mask, "became_fallen_at_year"] = np.nan
    df.loc[replace_mask, "became_decayed_at_year"] = np.nan
    df.loc[replace_mask, "became_gone_at_year"] = np.nan
    df.loc[replace_mask, "fallen_decay_after_years"] = np.nan
    df.loc[replace_mask, "decayed_remove_after_years"] = np.nan
    used_ids = collect_structure_ids(df.loc[~replace_mask])
    df.loc[replace_mask, "structureID"] = replacement_structure_ids(
        df,
        replace_mask,
        site=params["site"],
        scenario=params["scenario"],
        absolute_year=int(params["absolute_year"]),
        used_ids=used_ids,
    )
    return df


def assign_decay_interventions(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    df["proposal-decay_intervention"] = "none"
    decay_mask = df["proposal-decay_decision"].eq("proposal-decay_accepted")
    if not decay_mask.any():
        return df

    proposed_treatment = pd.cut(
        df.loc[decay_mask, "CanopyResistance"],
        bins=[
            -float("inf"),
            params["maximum-tree-support-threshold"],
            params["moderate-tree-support-threshold"],
            params["minimal-tree-support-threshold"],
            float("inf"),
        ],
        labels=["node-rewilded", "footprint-depaved", "exoskeleton", "None"],
    )
    proposed_treatment = pd.Series(proposed_treatment, index=df.index[decay_mask]).astype("object").fillna("paved")
    proposed_treatment = proposed_treatment.replace({"None": "paved"})
    current_treatment = df.loc[decay_mask, "under-node-treatment"].astype(str)
    merged_treatment = _merge_under_node_treatments(current_treatment, proposed_treatment)
    df.loc[decay_mask, "under-node-treatment"] = merged_treatment
    df.loc[decay_mask, "proposal-decay_intervention"] = merged_treatment.map(DECAY_INTERVENTION_BY_TREATMENT).fillna("none")
    return df


def refresh_colonise_support(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proposal-colonise_intervention"] = "none"
    ground_mask = df["under-node-treatment"].isin(["node-rewilded", "footprint-depaved"])
    df.loc[ground_mask, "proposal-colonise_intervention"] = COLONISE_FULL_GROUND
    df["proposal-colonise_decision"] = np.where(
        ground_mask.to_numpy(dtype=bool),
        "proposal-colonise_accepted",
        "proposal-colonise_rejected",
    )
    return df


def _target_rank(values: pd.Series) -> pd.Series:
    return values.map(PRUNING_TARGET_ORDER).fillna(0).astype(int)


def _realized_control_from_years(release_intervention: str, release_years: float) -> str:
    if release_intervention == RELEASECONTROL_FULL:
        if release_years >= ELIMINATE_PRUNING_TO_LOW_YEARS:
            return LOW_CONTROL_STATE
        if release_years >= ELIMINATE_PRUNING_TO_PARK_YEARS:
            return "park-tree"
        return "street-tree"
    if release_intervention == RELEASECONTROL_PARTIAL:
        if release_years >= REDUCE_PRUNING_TO_PARK_YEARS:
            return "park-tree"
        return "street-tree"
    return "street-tree"


def apply_release_control(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    living_mask = df["size"].isin(["small", "medium", "large"])
    non_living_mask = ~living_mask
    df.loc[non_living_mask, "proposal-release-control_decision"] = "not-assessed"
    df.loc[non_living_mask, "proposal-release-control_intervention"] = "none"
    df.loc[non_living_mask, "proposal-release-control_target_years"] = 0.0
    df.loc[non_living_mask, "proposal-release-control_years"] = 0.0
    df.loc[living_mask, "proposal-release-control_decision"] = "proposal-release-control_rejected"

    if living_mask.any():
        proposed_treatment = pd.cut(
            df.loc[living_mask, "CanopyResistance"],
            bins=[
                -float("inf"),
                params["maximum-tree-support-threshold"],
                params["moderate-tree-support-threshold"],
                float("inf"),
            ],
            labels=["node-rewilded", "footprint-depaved", "None"],
        )
        proposed_treatment = pd.Series(proposed_treatment, index=df.index[living_mask]).astype("object").fillna("paved")
        proposed_treatment = proposed_treatment.replace({"None": "paved"})
        support_series = proposed_treatment.map(RELEASE_SUPPORT_BY_REWILDING).fillna("none")
        current_intervention = df.loc[living_mask, "proposal-release-control_intervention"].fillna("none").astype(str)
        current_target = current_intervention.replace({"none": "standard-pruning"})
        proposed_target = support_series.replace({"none": "standard-pruning"})

        current_rank = _target_rank(current_target)
        proposed_rank = _target_rank(proposed_target)
        chosen_rank = np.maximum(current_rank.to_numpy(dtype=int), proposed_rank.to_numpy(dtype=int))
        reverse_map = {value: key for key, value in PRUNING_TARGET_ORDER.items()}
        new_target = pd.Series(chosen_rank).map(reverse_map).to_numpy()

        changed_mask = current_target.to_numpy() != new_target
        current_target_years = df.loc[living_mask, "proposal-release-control_target_years"].to_numpy(dtype=float)
        release_years = df.loc[living_mask, "proposal-release-control_years"].to_numpy(dtype=float)
        step_years = params["step_years"]

        new_target_years = np.where(
            new_target == "standard-pruning",
            0.0,
            np.where(
                current_target.to_numpy() == "standard-pruning",
                step_years,
                current_target_years + step_years,
            ),
        )
        new_release_years = np.where(
            new_target == "standard-pruning",
            0.0,
            release_years + step_years,
        )

        df.loc[living_mask, "proposal-release-control_target_years"] = new_target_years
        df.loc[living_mask, "proposal-release-control_years"] = new_release_years
        intervention_from_target = pd.Series(new_target, index=df.index[living_mask]).replace({"standard-pruning": "none"})
        df.loc[living_mask, "proposal-release-control_intervention"] = intervention_from_target
        df.loc[living_mask, "proposal-release-control_decision"] = np.where(
            intervention_from_target.eq("none"),
            "proposal-release-control_rejected",
            "proposal-release-control_accepted",
        )
        merged_treatment = _merge_under_node_treatments(
            df.loc[living_mask, "under-node-treatment"].astype(str),
            intervention_from_target.map(RELEASE_TREATMENT_BY_INTERVENTION).fillna("paved"),
        )
        df.loc[living_mask, "under-node-treatment"] = merged_treatment

    support_from_target = {
        "standard-pruning": "none",
        RELEASECONTROL_PARTIAL: RELEASECONTROL_PARTIAL,
        RELEASECONTROL_FULL: RELEASECONTROL_FULL,
    }
    accepted_release_mask = df["proposal-release-control_decision"].eq("proposal-release-control_accepted")
    df.loc[accepted_release_mask, "proposal-release-control_intervention"] = (
        df.loc[accepted_release_mask, "proposal-release-control_intervention"].replace(support_from_target).fillna("none")
    )
    df.loc[living_mask, "proposal-release-control_intervention"] = df.loc[living_mask, "proposal-release-control_intervention"].fillna("none")
    df.loc[df["proposal-release-control_decision"].ne("proposal-release-control_accepted"), "proposal-release-control_intervention"] = "none"
    df.loc[non_living_mask, "proposal-release-control_decision"] = "not-assessed"
    df.loc[non_living_mask, "proposal-release-control_intervention"] = "none"
    df.loc[non_living_mask, "proposal-release-control_target_years"] = 0.0
    df.loc[non_living_mask, "proposal-release-control_years"] = 0.0

    df["control_reached"] = [
        _realized_control_from_years(intervention, years)
        for intervention, years in zip(
            df["proposal-release-control_intervention"],
            df["proposal-release-control_years"],
            strict=False,
        )
    ]
    senescent_mask = df["size"].isin(["senescing", "snag", "fallen", "decayed"])
    df.loc[senescent_mask, "control_reached"] = LOW_CONTROL_STATE
    df["control"] = [
        _export_control_value(realized, size)
        for realized, size in zip(df["control_reached"], df["size"], strict=False)
    ]
    return df


def update_fallen_tracking(
    df: pd.DataFrame,
    site: str,
    scenario: str,
    pulse_start_year: int,
    current_year: int,
    params: dict,
) -> pd.DataFrame:
    df = df.copy()
    fallen_mask = df["size"].eq("fallen")
    newly_fallen_mask = fallen_mask & df["fallen_decay_after_years"].isna()
    if newly_fallen_mask.any():
        df.loc[newly_fallen_mask & df["became_fallen_at_year"].isna(), "became_fallen_at_year"] = current_year

        decay_years = []
        for index, row in df.loc[newly_fallen_mask].iterrows():
            identity = row.get("structureID", row.get("NodeID", row.get("tree_number", index)))
            decay_years.append(
                _seeded_triangular_year(
                    site,
                    scenario,
                    identity,
                    current_year,
                    "fallen-to-decayed",
                    minimum=float(params["fallen_duration_years"]["min"]),
                    mode=float(params["fallen_duration_years"]["mode"]),
                    maximum=float(params["fallen_duration_years"]["max"]),
                )
            )
        df.loc[newly_fallen_mask, "fallen_decay_after_years"] = decay_years

    decayed_transition_mask = (
        fallen_mask
        & df["became_fallen_at_year"].notna()
        & df["fallen_decay_after_years"].notna()
        & ((current_year - df["became_fallen_at_year"]) >= df["fallen_decay_after_years"])
    )
    if decayed_transition_mask.any():
        df.loc[decayed_transition_mask, "size"] = "decayed"
        df.loc[decayed_transition_mask, "lifecycle_state"] = "decayed"
        became_decayed_years = []
        for index, row in df.loc[decayed_transition_mask].iterrows():
            identity = row.get("structureID", row.get("NodeID", row.get("tree_number", index)))
            became_decayed_years.append(
                _seeded_year_within_window(
                    site,
                    scenario,
                    identity,
                    current_year,
                    "fallen-to-decayed-observed-year",
                    window_start=pulse_start_year,
                    window_end=current_year,
                )
            )
        df.loc[decayed_transition_mask, "became_decayed_at_year"] = became_decayed_years

        remove_years = []
        for index, row in df.loc[decayed_transition_mask].iterrows():
            identity = row.get("structureID", row.get("NodeID", row.get("tree_number", index)))
            remove_years.append(
                _seeded_triangular_year(
                    site,
                    scenario,
                    identity,
                    current_year,
                    "decayed-remove",
                    minimum=float(params["decayed_duration_years"]["min"]),
                    mode=float(params["decayed_duration_years"]["mode"]),
                    maximum=float(params["decayed_duration_years"]["max"]),
                )
            )
        df.loc[decayed_transition_mask, "decayed_remove_after_years"] = remove_years

    decayed_mask = df["size"].eq("decayed")
    newly_decayed_mask = decayed_mask & df["became_decayed_at_year"].isna()
    if newly_decayed_mask.any():
        became_decayed_years = []
        for index, row in df.loc[newly_decayed_mask].iterrows():
            identity = row.get("structureID", row.get("NodeID", row.get("tree_number", index)))
            became_decayed_years.append(
                _seeded_year_within_window(
                    site,
                    scenario,
                    identity,
                    current_year,
                    "decayed-observed-year",
                    window_start=pulse_start_year,
                    window_end=current_year,
                )
            )
        df.loc[newly_decayed_mask, "became_decayed_at_year"] = became_decayed_years

        remove_years = []
        for index, row in df.loc[newly_decayed_mask].iterrows():
            identity = row.get("structureID", row.get("NodeID", row.get("tree_number", index)))
            remove_years.append(
                _seeded_triangular_year(
                    site,
                    scenario,
                    identity,
                    current_year,
                    "decayed-remove",
                    minimum=float(params["decayed_duration_years"]["min"]),
                    mode=float(params["decayed_duration_years"]["mode"]),
                    maximum=float(params["decayed_duration_years"]["max"]),
                )
            )
        df.loc[newly_decayed_mask, "decayed_remove_after_years"] = remove_years

    removable_mask = (
        decayed_mask
        & df["became_decayed_at_year"].notna()
        & df["decayed_remove_after_years"].notna()
        & ((current_year - df["became_decayed_at_year"]) >= df["decayed_remove_after_years"])
    )
    if removable_mask.any():
        df.loc[removable_mask, "size"] = "gone"
        df.loc[removable_mask, "lifecycle_state"] = "gone"
        df.loc[removable_mask & df["became_gone_at_year"].isna(), "became_gone_at_year"] = current_year
        df.loc[removable_mask, "action"] = "None"
        df.loc[removable_mask, "under-node-treatment"] = "paved"
        df.loc[removable_mask, "proposal-recruit_decision"] = "not-assessed"
        df.loc[removable_mask, "proposal-recruit_intervention"] = "none"
        df.loc[removable_mask, "recruit_intervention_type"] = "none"
        df.loc[removable_mask, "recruit_source_id"] = "none"
        df.loc[removable_mask, "proposal-colonise_decision"] = "not-assessed"
        df.loc[removable_mask, "proposal-colonise_intervention"] = "none"

    return df


def _build_voxel_tree(ds: xr.Dataset):
    points = np.vstack([
        ds["centroid_x"].values,
        ds["centroid_y"].values,
        ds["centroid_z"].values,
    ]).T
    return points, cKDTree(points)


def _resolve_candidate_voxel(ds: xr.Dataset, tree: cKDTree, position: tuple[float, float, float]) -> int:
    _, voxel_index = tree.query(np.array(position))
    return int(voxel_index)


def _resolve_candidate_from_indices(
    points: np.ndarray,
    tree: cKDTree | None,
    voxel_indices: np.ndarray,
    position: tuple[float, float, float],
) -> int | None:
    if tree is None or len(voxel_indices) == 0:
        return None
    _, local_index = tree.query(np.array(position))
    return int(voxel_indices[int(local_index)])


def _recruit_active_mask(df: pd.DataFrame) -> pd.Series:
    return df["size"].isin(["small", "medium", "large", "senescing"])


def _append_recruit_telemetry_csv(path: Path | str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    telemetry_path = Path(path)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "site",
        "scenario",
        "assessed_year",
        "pulse_start_year",
        "pulse_end_year",
        "recruit_type",
        "source_id",
        "source_treatment",
        "candidate_count",
        "quota",
        "starting_occupancy",
        "remaining_capacity",
        "placed",
        "unfilled",
        "rejected_spacing",
        "rejected_attempt_limit",
        "attempts",
    ]
    write_header = not telemetry_path.exists()
    with telemetry_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _format_source_id(value: object, fallback: object) -> str:
    candidate = fallback if pd.isna(value) else value
    if isinstance(candidate, (int, np.integer)):
        return str(int(candidate))
    if isinstance(candidate, (float, np.floating)) and np.isfinite(candidate):
        if float(candidate).is_integer():
            return str(int(candidate))
        return str(candidate)
    return str(candidate)


def _valid_source_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return np.isfinite(value) and float(value) >= 0
    string_value = str(value).strip().lower()
    return string_value not in {"", "none", "nan", "-1"}


def _pick_source_id(*values: object, fallback: object) -> str:
    for value in values:
        if _valid_source_value(value):
            return _format_source_id(value, fallback)
    return _format_source_id(fallback, fallback)


def _recruit_occupancy_counts(df: pd.DataFrame, intervention_type: str) -> dict[str, int]:
    occupied_mask = (
        df["isNewTree"].fillna(False)
        & df["recruit_intervention_type"].eq(intervention_type)
        & df["recruit_source_id"].ne("none")
        & ~df["size"].isin(NON_OCCUPYING_TREE_SIZES)
    )
    if not occupied_mask.any():
        return {}
    return (
        df.loc[occupied_mask, "recruit_source_id"]
        .astype(str)
        .value_counts()
        .to_dict()
    )


def _new_tree_template(df: pd.DataFrame):
    template = {}
    for column in df.columns:
        if pd.api.types.is_bool_dtype(df[column]):
            template[column] = False
        elif pd.api.types.is_numeric_dtype(df[column]):
            template[column] = np.nan
        else:
            template[column] = None
    return template


def _make_new_tree_record(
    df_template: dict,
    ds: xr.Dataset,
    voxel_index: int,
    position: tuple[float, float, float],
    recruit_intervention: str,
    recruit_intervention_type: str,
    recruit_source_id: str,
    recruit_year: int,
    under_node_treatment: str = "none",
    recruit_mechanism: str = "none",
) -> dict:
    record = dict(df_template)
    x, y, z = position
    record.update(
        {
            "x": x,
            "y": y,
            "z": z,
            "size": "small",
            "diameter_breast_height": 2.0,
            "precolonial": True,
            "useful_life_expectancy": 120.0,
            "CanopyArea": 1.0,
            "sim_NodesArea": 1.0,
            "sim_NodesVoxels": 1.0,
            "CanopyResistance": float(ds["analysis_combined_resistance"].values[voxel_index]),
            "voxel_index": voxel_index,
            "tree_id": -1,
            "tree_number": -1,
            "NodeID": -1,
            "debugNodeID": -1,
            "isNewTree": True,
            "isRewildedTree": True,
            "hasbeenReplanted": False,
            "under-node-treatment": under_node_treatment,
            "recruit_mechanism": recruit_mechanism,
            "action": "None",
            "replacement_reason": "none",
            "proposal-decay_decision": "not-assessed",
            "proposal-decay_intervention": "none",
            "proposal-release-control_decision": "proposal-release-control_accepted",
            "proposal-release-control_intervention": RELEASECONTROL_FULL,
            "proposal-release-control_target_years": float(ELIMINATE_PRUNING_TO_LOW_YEARS),
            "proposal-release-control_years": float(ELIMINATE_PRUNING_TO_LOW_YEARS),
            "proposal-recruit_decision": "proposal-recruit_accepted",
            "proposal-recruit_intervention": recruit_intervention,
            "recruit_intervention_type": recruit_intervention_type,
            "recruit_source_id": recruit_source_id,
            "recruit_year": recruit_year,
            "proposal-colonise_decision": "proposal-colonise_rejected",
            "proposal-colonise_intervention": "none",
            "proposal-deploy-structure_decision": "not-assessed",
            "proposal-deploy-structure_intervention": "none",
            "control_reached": LOW_CONTROL_STATE,
            "control": "reserve-tree",
            "lifecycle_state": "standing",
            "early_death_at_year": np.nan,
            "became_large_at_year": np.nan,
            "became_senescing_at_year": np.nan,
            "became_snag_at_year": np.nan,
            "senescing_duration_years": np.nan,
            "snag_duration_years": np.nan,
            "became_fallen_at_year": np.nan,
            "fallen_decay_after_years": np.nan,
            "became_decayed_at_year": np.nan,
            "decayed_remove_after_years": np.nan,
            "became_gone_at_year": np.nan,
        }
    )
    return record


def calculate_under_node_treatment_status(df: pd.DataFrame, ds: xr.Dataset, params: dict):
    ds = ds.copy(deep=True)
    absolute_year = int(params.get("absolute_year", params.get("years_passed", 0)))
    if absolute_year <= 0:
        ds["scenario_rewildingEnabled"] = xr.full_like(ds["node_CanopyID"], -1)
        ds["scenario_rewildingPlantings"] = xr.full_like(ds["node_CanopyID"], -1)
        return df, ds

    active_df = df.loc[_recruit_active_mask(df)].copy()

    depaved_threshold = params["sim_TurnsThreshold"]
    depaved_mask = (ds["sim_Turns"] <= depaved_threshold) & (ds["sim_Turns"] >= 0)
    terrain_mask = (ds["site_building_element"] != "facade") & (ds["site_building_element"] != "roof")
    combined_mask = depaved_mask & terrain_mask

    ds["scenario_rewildingEnabled"] = xr.where(combined_mask, absolute_year, -1)

    voxel_positions = np.vstack([
        ds["centroid_x"].values[combined_mask],
        ds["centroid_y"].values[combined_mask],
        ds["centroid_z"].values[combined_mask],
    ]).T
    filtered_voxel_ids = np.arange(ds.sizes["voxel"])[combined_mask]

    if len(active_df) == 0 or len(voxel_positions) == 0:
        ds["scenario_rewildingPlantings"] = xr.where(combined_mask, absolute_year, -1)
        return df, ds

    tree_locations = np.vstack([active_df["x"], active_df["y"], active_df["z"]]).T
    tree_kdtree = cKDTree(tree_locations)
    distances, _ = tree_kdtree.query(voxel_positions, distance_upper_bound=RECRUIT_SPACING_THRESHOLD_METERS)
    filtered_proximity_mask = distances > RECRUIT_SPACING_THRESHOLD_METERS

    proximity_mask = np.full(ds.sizes["voxel"], False)
    proximity_mask[filtered_voxel_ids] = filtered_proximity_mask

    final_mask = depaved_mask & terrain_mask & proximity_mask
    ds["scenario_rewildingPlantings"] = xr.where(final_mask, absolute_year, -1)
    return df, ds


def apply_recruit(
    df: pd.DataFrame,
    ds: xr.Dataset,
    params: dict,
    telemetry_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    step_years = params["step_years"]
    if step_years <= 0:
        return df

    seed = int(params.get("seed", 42))
    np.random.seed(seed + 3000 + int(params["absolute_year"]))

    gone_mask = df["size"].isin(ABSENT_TREE_SIZES)
    if gone_mask.any():
        df.loc[gone_mask, "proposal-recruit_decision"] = "not-assessed"
        df.loc[gone_mask, "proposal-recruit_intervention"] = "none"
        df.loc[gone_mask, "recruit_intervention_type"] = "none"
        df.loc[gone_mask, "recruit_source_id"] = "none"

    existing_recruit_intervention = df["proposal-recruit_intervention"].fillna("none").astype(str)
    df["proposal-recruit_decision"] = np.where(
        existing_recruit_intervention.eq("none"),
        "not-assessed",
        "proposal-recruit_accepted",
    )

    absolute_year = int(params["absolute_year"])
    site = str(params["site"])
    scenario = str(params["scenario"])
    pulse_factor = step_years / RECRUIT_INTERVAL
    planting_density_sqm = params["plantingDensity"] / 10000
    df_template = _new_tree_template(df)
    voxel_points, voxel_tree = _build_voxel_tree(ds)
    planting_mask = ds["scenario_rewildingPlantings"].values == params["absolute_year"]
    available_voxel_indices = np.arange(ds.sizes["voxel"])[planting_mask]
    eligible_voxel_points = voxel_points[available_voxel_indices] if len(available_voxel_indices) else np.empty((0, 3))
    eligible_voxel_tree = cKDTree(eligible_voxel_points) if len(eligible_voxel_points) else None

    active_df = df.loc[_recruit_active_mask(df)]
    current_positions = np.vstack([active_df["x"], active_df["y"], active_df["z"]]).T if len(active_df) else np.empty((0, 3))
    accepted_positions = [tuple(position) for position in current_positions]

    node_candidates: list[dict] = []
    node_telemetry: dict[str, dict[str, object]] = {}
    existing_mask = (~df["isNewTree"].fillna(False)) & (~df["size"].isin(ABSENT_TREE_SIZES))
    node_support_mask = existing_mask & df["under-node-treatment"].isin(["node-rewilded", "footprint-depaved"])
    node_rewilded_mask = node_support_mask & df["under-node-treatment"].eq("node-rewilded")
    footprint_depaved_mask = node_support_mask & df["under-node-treatment"].eq("footprint-depaved")
    df.loc[node_support_mask, "proposal-recruit_decision"] = "proposal-recruit_accepted"
    df.loc[node_rewilded_mask, "proposal-recruit_intervention"] = RECRUIT_FULL
    df.loc[footprint_depaved_mask, "proposal-recruit_intervention"] = RECRUIT_PARTIAL
    node_occupancy_full = _recruit_occupancy_counts(df, RECRUIT_FULL)
    node_occupancy_partial = _recruit_occupancy_counts(df, RECRUIT_PARTIAL)
    if node_support_mask.any():
        temp_area = np.where(
            df.loc[node_support_mask, "under-node-treatment"].eq("node-rewilded"),
            _numeric_series(df.loc[node_support_mask, "sim_NodesArea"]).to_numpy(dtype=float),
            _numeric_series(df.loc[node_support_mask, "CanopyArea"]).to_numpy(dtype=float),
        )
        temp_area = np.nan_to_num(temp_area, nan=0.0, posinf=0.0, neginf=0.0)
        temp_area = np.clip(temp_area, a_min=0.0, a_max=None)
        node_quota = np.ceil(temp_area * planting_density_sqm * pulse_factor).astype(int)
        for (_, parent_row), quota in zip(df.loc[node_support_mask].iterrows(), node_quota, strict=False):
            source_id = _pick_source_id(
                parent_row.get("NodeID"),
                parent_row.get("debugNodeID"),
                fallback="unknown",
            )
            source_treatment = str(parent_row.get("under-node-treatment", "none"))
            is_node_rewilded = source_treatment == "node-rewilded"
            recruit_type = RECRUIT_FULL if is_node_rewilded else RECRUIT_PARTIAL
            node_occupancy = node_occupancy_full if is_node_rewilded else node_occupancy_partial
            parent_position = (
                float(parent_row["x"]),
                float(parent_row["y"]),
                float(parent_row["z"]),
            )
            starting_occupancy = int(node_occupancy.get(source_id, 0))
            remaining_capacity = max(0, int(quota) - node_occupancy.get(source_id, 0))
            telemetry = node_telemetry.setdefault(
                source_id,
                {
                    "site": site,
                    "scenario": scenario,
                    "assessed_year": absolute_year,
                    "pulse_start_year": int(params["previous_year"]),
                    "pulse_end_year": absolute_year,
                    "recruit_type": recruit_type,
                    "source_id": source_id,
                    "source_treatment": source_treatment,
                    "candidate_count": 0,
                    "quota": 0,
                    "starting_occupancy": starting_occupancy,
                    "remaining_capacity": 0,
                    "placed": 0,
                    "unfilled": 0,
                    "rejected_spacing": 0,
                    "rejected_attempt_limit": 0,
                    "attempts": 0,
                },
            )
            telemetry["quota"] += int(quota)
            telemetry["remaining_capacity"] += int(remaining_capacity)
            if remaining_capacity <= 0:
                continue
            planted_here = 0
            attempts = 0
            attempt_cap = max(remaining_capacity * 8, 8)
            spacing_threshold = RECRUIT_SPACING_THRESHOLD_METERS
            while planted_here < remaining_capacity and attempts < attempt_cap:
                attempts += 1
                telemetry["candidate_count"] += 1
                candidate_position = (
                    float(parent_position[0] + np.random.uniform(-BUFFER_RECRUIT_PARENT_OFFSET_METERS, BUFFER_RECRUIT_PARENT_OFFSET_METERS)),
                    float(parent_position[1] + np.random.uniform(-BUFFER_RECRUIT_PARENT_OFFSET_METERS, BUFFER_RECRUIT_PARENT_OFFSET_METERS)),
                    float(parent_position[2]),
                )
                if accepted_positions:
                    spacing_positions = accepted_positions
                    if parent_position in accepted_positions:
                        spacing_positions = [position for position in accepted_positions if position != parent_position]
                    if spacing_positions:
                        candidate_tree = cKDTree(np.array(spacing_positions))
                        distance, _ = candidate_tree.query(
                            np.array(candidate_position),
                            distance_upper_bound=spacing_threshold,
                        )
                        if np.isfinite(distance) and distance <= spacing_threshold:
                            telemetry["rejected_spacing"] += 1
                            continue
                voxel_index = _resolve_candidate_from_indices(
                    eligible_voxel_points,
                    eligible_voxel_tree,
                    available_voxel_indices,
                    candidate_position,
                )
                if voxel_index is None:
                    telemetry["rejected_attempt_limit"] += 1
                    continue
                node_candidates.append(
                    _make_new_tree_record(
                        df_template=df_template,
                        ds=ds,
                        voxel_index=voxel_index,
                        position=candidate_position,
                        recruit_intervention=recruit_type,
                        recruit_intervention_type=recruit_type,
                        recruit_source_id=source_id,
                        recruit_year=absolute_year,
                        under_node_treatment=source_treatment,
                        recruit_mechanism="node",
                    )
                )
                accepted_positions.append(candidate_position)
                planted_here += 1
                telemetry["placed"] += 1
                node_occupancy[source_id] = node_occupancy.get(source_id, 0) + 1
            if planted_here < remaining_capacity and attempts >= attempt_cap:
                spacing_threshold = RELAXED_RECRUIT_SPACING_THRESHOLD_METERS
                extra_attempts = 0
                while planted_here < remaining_capacity and extra_attempts < attempt_cap:
                    extra_attempts += 1
                    telemetry["candidate_count"] += 1
                    candidate_position = (
                        float(parent_position[0] + np.random.uniform(-BUFFER_RECRUIT_PARENT_OFFSET_METERS, BUFFER_RECRUIT_PARENT_OFFSET_METERS)),
                        float(parent_position[1] + np.random.uniform(-BUFFER_RECRUIT_PARENT_OFFSET_METERS, BUFFER_RECRUIT_PARENT_OFFSET_METERS)),
                        float(parent_position[2]),
                    )
                    if accepted_positions:
                        spacing_positions = accepted_positions
                        if parent_position in accepted_positions:
                            spacing_positions = [position for position in accepted_positions if position != parent_position]
                        if spacing_positions:
                            candidate_tree = cKDTree(np.array(spacing_positions))
                            distance, _ = candidate_tree.query(
                                np.array(candidate_position),
                                distance_upper_bound=spacing_threshold,
                            )
                            if np.isfinite(distance) and distance <= spacing_threshold:
                                telemetry["rejected_spacing"] += 1
                                continue
                    voxel_index = _resolve_candidate_from_indices(
                        eligible_voxel_points,
                        eligible_voxel_tree,
                        available_voxel_indices,
                        candidate_position,
                    )
                    if voxel_index is None:
                        continue
                    node_candidates.append(
                        _make_new_tree_record(
                            df_template=df_template,
                            ds=ds,
                            voxel_index=voxel_index,
                            position=candidate_position,
                            recruit_intervention=recruit_type,
                            recruit_intervention_type=recruit_type,
                            recruit_source_id=source_id,
                            recruit_year=absolute_year,
                            under_node_treatment=source_treatment,
                            recruit_mechanism="node",
                        )
                    )
                    accepted_positions.append(candidate_position)
                    planted_here += 1
                    telemetry["placed"] += 1
                    node_occupancy[source_id] = node_occupancy.get(source_id, 0) + 1
                attempts += extra_attempts
            telemetry["attempts"] += int(attempts)
            telemetry["unfilled"] += max(0, remaining_capacity - planted_here)
            if planted_here < remaining_capacity and attempts >= attempt_cap:
                telemetry["rejected_attempt_limit"] += max(0, remaining_capacity - planted_here)

    turn_candidates: list[dict] = []
    ground_telemetry: dict[str, dict[str, object]] = {}
    if planting_mask.any():
        sim_node_values = ds["sim_Nodes"].values if "sim_Nodes" in ds.variables else np.full(ds.sizes["voxel"], np.nan)
        analysis_node_values = (
            ds["analysis_nodeID"].values if "analysis_nodeID" in ds.variables else np.full(ds.sizes["voxel"], np.nan)
        )
        canopy_node_values = (
            ds["node_CanopyID"].values if "node_CanopyID" in ds.variables else np.full(ds.sizes["voxel"], np.nan)
        )
        ground_source_to_voxels: dict[str, list[int]] = {}
        for voxel_index in available_voxel_indices:
            source_id = _pick_source_id(
                sim_node_values[int(voxel_index)],
                analysis_node_values[int(voxel_index)],
                canopy_node_values[int(voxel_index)],
                fallback=f"voxel-{int(voxel_index)}",
            )
            ground_source_to_voxels.setdefault(source_id, []).append(int(voxel_index))

        ground_occupancy = _recruit_occupancy_counts(df, RECRUIT_FULL)
        voxel_area_sqm = ds.attrs["voxel_size"] * ds.attrs["voxel_size"]

        for source_id, voxel_indices in ground_source_to_voxels.items():
            source_area_sqm = len(voxel_indices) * voxel_area_sqm
            source_quota = int(np.round(source_area_sqm * planting_density_sqm * pulse_factor))
            starting_occupancy = int(ground_occupancy.get(source_id, 0))
            remaining_capacity = max(0, source_quota - ground_occupancy.get(source_id, 0))
            telemetry = ground_telemetry.setdefault(
                source_id,
                {
                    "site": site,
                    "scenario": scenario,
                    "assessed_year": absolute_year,
                    "pulse_start_year": int(params["previous_year"]),
                    "pulse_end_year": absolute_year,
                    "recruit_type": RECRUIT_FULL,
                    "source_id": source_id,
                    "source_treatment": "scenario_rewildingPlantings",
                    "candidate_count": int(len(voxel_indices)),
                    "quota": int(source_quota),
                    "starting_occupancy": starting_occupancy,
                    "remaining_capacity": int(remaining_capacity),
                    "placed": 0,
                    "unfilled": 0,
                    "rejected_spacing": 0,
                    "rejected_attempt_limit": 0,
                    "attempts": 0,
                },
            )
            if remaining_capacity <= 0:
                continue

            candidate_indices = np.array(voxel_indices, dtype=int)
            np.random.shuffle(candidate_indices)

            for voxel_index in candidate_indices:
                telemetry["attempts"] += 1
                if ground_occupancy.get(source_id, 0) >= source_quota:
                    break
                candidate_position = tuple(voxel_points[int(voxel_index)])
                if accepted_positions:
                    candidate_tree = cKDTree(np.array(accepted_positions))
                    distance, _ = candidate_tree.query(
                        np.array(candidate_position),
                        distance_upper_bound=RECRUIT_SPACING_THRESHOLD_METERS,
                    )
                    if np.isfinite(distance) and distance <= RECRUIT_SPACING_THRESHOLD_METERS:
                        telemetry["rejected_spacing"] += 1
                        continue
                turn_candidates.append(
                    _make_new_tree_record(
                        df_template=df_template,
                        ds=ds,
                        voxel_index=int(voxel_index),
                        position=candidate_position,
                        recruit_intervention=RECRUIT_FULL,
                        recruit_intervention_type=RECRUIT_FULL,
                        recruit_source_id=source_id,
                        recruit_year=absolute_year,
                        under_node_treatment="scenario-rewilded",
                        recruit_mechanism="ground",
                    )
                )
                accepted_positions.append(candidate_position)
                telemetry["placed"] += 1
                ground_occupancy[source_id] = ground_occupancy.get(source_id, 0) + 1
            telemetry["unfilled"] += max(0, source_quota - ground_occupancy.get(source_id, 0))

    if node_candidates or turn_candidates:
        new_trees = pd.DataFrame(node_candidates + turn_candidates)
        # Stagger recruit DBH by simulating random arrival times within the pulse.
        # Each recruit arrives at a uniform random point in [0, step_years] and
        # pre-grows from 2cm using Fischer for the remaining time in the pulse.
        n_new = len(new_trees)
        if n_new > 0 and step_years > 0:
            fischer_k = 0.0197135 * np.pi / 4.0
            start_dbh = 2.0
            stagger_rng = np.random.default_rng(seed + 4000 + int(params["absolute_year"]))
            arrival_offset = stagger_rng.uniform(0, step_years, size=n_new)
            growth_time = step_years - arrival_offset
            start_age = fischer_k * start_dbh ** 2
            new_trees["diameter_breast_height"] = np.round(
                np.sqrt((start_age + growth_time) / fischer_k), 1
            )
        df = pd.concat([df, new_trees], ignore_index=True)

    if telemetry_rows is not None:
        telemetry_rows.extend(node_telemetry.values())
        telemetry_rows.extend(ground_telemetry.values())

    return _refresh_schema(df)


def assign_logs(log_df: pd.DataFrame | None, params: dict) -> pd.DataFrame | None:
    if log_df is None:
        return None

    log_df = _with_node_type(log_df, "log")
    turn_threshold = params["sim_TurnsThreshold"]
    resistance_threshold = params.get("sim_averageResistance", 0)
    mask = (log_df["sim_averageResistance"] <= resistance_threshold) & (log_df["sim_Turns"] <= turn_threshold)
    log_df["isEnabled"] = mask
    log_df["proposal-deploy-structure_decision"] = np.where(
        mask,
        "proposal-deploy-structure_accepted",
        "not-assessed",
    )
    log_df["proposal-deploy-structure_intervention"] = np.where(mask, DEPLOY_FULL_LOG, "none")
    return log_df[log_df["isEnabled"]].copy()


def assign_poles(pole_df: pd.DataFrame | None, params: dict) -> pd.DataFrame | None:
    if pole_df is None:
        return None

    pole_df = _with_node_type(pole_df, "pole")
    resistance_threshold = params.get("sim_averageResistance", 0)
    mask = pole_df["sim_averageResistance"] < resistance_threshold
    pole_df["isEnabled"] = mask
    pole_df["proposal-deploy-structure_decision"] = np.where(
        mask,
        "proposal-deploy-structure_accepted",
        "not-assessed",
    )
    pole_df["proposal-deploy-structure_intervention"] = np.where(mask, DEPLOY_FULL_POLE, "none")
    return pole_df[pole_df["isEnabled"]].copy()


def _run_single_pulse(
    df: pd.DataFrame,
    ds: xr.Dataset,
    site: str,
    scenario: str,
    params: dict,
    telemetry_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    df = _refresh_schema(df)
    df = age_trees(df, params)
    df = apply_annual_tree_mortality(df, params)
    df = determine_proposal_decay(df, params)
    df = assign_decay_interventions(df, params)
    df = apply_proposal_decay_accepted_lifecycle_changes(df, params)
    params = dict(params)
    params["site"] = site
    params["scenario"] = scenario
    df = apply_proposal_decay_rejected_changes(df, params)
    df = apply_release_control(df, params)
    df = refresh_colonise_support(df)
    df = update_fallen_tracking(df, site, scenario, params["previous_year"], params["absolute_year"], params)

    _, pulse_ds = calculate_under_node_treatment_status(df, ds, params)
    df = apply_recruit(df, pulse_ds, params, telemetry_rows=telemetry_rows)
    df = refresh_colonise_support(df)
    return _refresh_schema(df)


def _year_zero_state(
    site: str,
    scenario: str,
    year: int,
    voxel_size: int,
    tree_df: pd.DataFrame,
    log_df: pd.DataFrame | None,
    pole_df: pd.DataFrame | None,
    output_mode: str | None,
):
    tree_df = _refresh_schema(tree_df)
    tree_df["action"] = "None"
    tree_df["replacement_reason"] = "none"
    tree_df["proposal-decay_decision"] = "not-assessed"
    tree_df["proposal-decay_intervention"] = "none"
    tree_df["proposal-recruit_decision"] = "not-assessed"
    tree_df["proposal-recruit_intervention"] = "none"
    tree_df["proposal-colonise_decision"] = "not-assessed"
    tree_df["proposal-colonise_intervention"] = "none"
    tree_df["proposal-deploy-structure_decision"] = "not-assessed"
    tree_df["proposal-deploy-structure_intervention"] = "none"
    tree_df["proposal-release-control_decision"] = "not-assessed"
    tree_df["proposal-release-control_intervention"] = "none"
    tree_df["lifecycle_state"] = "standing"
    tree_df["under-node-treatment"] = "paved"
    tree_df["recruit_mechanism"] = "none"
    tree_df["control_reached"] = tree_df["control"].map(_legacy_control_to_realized)

    params = get_timestep_params(site, scenario, year, previous_year=0)
    log_df, pole_df = _site_specific_structures(site, log_df, pole_df)
    log_df = assign_logs(log_df, params)
    pole_df = assign_poles(pole_df, params)

    used_ids: set[int] = set()
    tree_df = assign_tree_structure_ids(tree_df, site=site, scenario=scenario, used_ids=used_ids)
    log_df = assign_log_structure_ids(log_df, site=site, used_ids=used_ids)
    pole_df = assign_pole_structure_ids(pole_df, site=site, used_ids=used_ids)

    output_tree_path = scenario_tree_df_path(site, scenario, year, voxel_size, output_mode)
    tree_df.to_csv(output_tree_path, index=False)

    if log_df is not None:
        log_df.to_csv(scenario_log_df_path(site, scenario, year, voxel_size, output_mode), index=False)
    if pole_df is not None:
        pole_df.to_csv(scenario_pole_df_path(site, scenario, year, voxel_size, output_mode), index=False)

    return tree_df, log_df, pole_df


def run_timestep(
    site: str,
    scenario: str,
    year: int,
    previous_year: int,
    voxel_size: int,
    tree_df: pd.DataFrame,
    possibility_space_ds: xr.Dataset,
    log_df: pd.DataFrame | None = None,
    pole_df: pd.DataFrame | None = None,
    output_mode: str | None = None,
    write_outputs: bool = True,
    recruit_telemetry_path: Path | str | None = None,
):
    if year == 0:
        return _year_zero_state(site, scenario, year, voxel_size, tree_df, log_df, pole_df, output_mode)

    tree_state = _refresh_schema(tree_df)
    tree_state["_decay_rejected_this_state"] = False
    recruit_telemetry_rows: list[dict[str, object]] = []
    for pulse_start, pulse_end in split_window_into_pulses(previous_year, year):
        pulse_params = get_timestep_params(site, scenario, pulse_end, pulse_start)
        tree_state = _run_single_pulse(
            tree_state,
            possibility_space_ds.copy(deep=True),
            site,
            scenario,
            pulse_params,
            telemetry_rows=recruit_telemetry_rows,
        )

    # Integrate decay rejections accumulated across all pulses in this state.
    # A tree rejected in an earlier pulse (then replanted) still shows as rejected in output.
    rejected_mask = tree_state["_decay_rejected_this_state"]
    tree_state.loc[rejected_mask, "proposal-decay_decision"] = "proposal-decay_rejected"
    tree_state = tree_state.drop(columns=["_decay_rejected_this_state"])

    final_params = get_timestep_params(site, scenario, year, previous_year)
    log_df, pole_df = _site_specific_structures(site, log_df, pole_df)
    log_df = assign_logs(log_df, final_params)
    pole_df = assign_poles(pole_df, final_params)

    tree_state = _refresh_schema(tree_state)
    used_ids: set[int] = set()
    tree_state = assign_tree_structure_ids(tree_state, site=site, scenario=scenario, used_ids=used_ids)
    log_df = assign_log_structure_ids(log_df, site=site, used_ids=used_ids)
    pole_df = assign_pole_structure_ids(pole_df, site=site, used_ids=used_ids)

    if write_outputs:
        tree_state.to_csv(scenario_tree_df_path(site, scenario, year, voxel_size, output_mode), index=False)
        if log_df is not None:
            log_df.to_csv(scenario_log_df_path(site, scenario, year, voxel_size, output_mode), index=False)
        if pole_df is not None:
            pole_df.to_csv(scenario_pole_df_path(site, scenario, year, voxel_size, output_mode), index=False)
        if recruit_telemetry_path is not None and recruit_telemetry_rows:
            _append_recruit_telemetry_csv(recruit_telemetry_path, recruit_telemetry_rows)

    return tree_state, log_df, pole_df


def run_scenario(
    site,
    scenario,
    year,
    voxel_size,
    treeDF,
    possibility_space_ds,
    logDF=None,
    poleDF=None,
    previous_year=None,
    output_mode=None,
    write_outputs=True,
    recruit_telemetry_path: Path | str | None = None,
):
    previous_year = 0 if previous_year is None else previous_year

    if logDF is not None:
        logDF = a_scenario_initialiseDS.log_processing(logDF, possibility_space_ds)
    if poleDF is not None:
        poleDF = a_scenario_initialiseDS.pole_processing(poleDF, None, possibility_space_ds)

    return run_timestep(
        site=site,
        scenario=scenario,
        year=year,
        previous_year=previous_year,
        voxel_size=voxel_size,
        tree_df=treeDF,
        possibility_space_ds=possibility_space_ds,
        log_df=logDF,
        pole_df=poleDF,
        output_mode=output_mode,
        write_outputs=write_outputs,
        recruit_telemetry_path=recruit_telemetry_path,
    )
