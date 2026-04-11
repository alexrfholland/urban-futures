from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent
FINAL_DIR = REPO_ROOT / "final"
for import_root in (CODE_ROOT, FINAL_DIR):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from _futureSim_refactored.sim.setup import a_scenario_initialiseDS  # noqa: E402
import a_scenario_params  # noqa: E402

from _futureSim_refactored.paths import (  # noqa: E402
    scenario_log_df_path,
    scenario_pole_df_path,
    scenario_tree_df_path,
)
from _futureSim_refactored.sim.setup.structure_ids import (  # noqa: E402
    assign_log_structure_ids,
    assign_pole_structure_ids,
    assign_tree_structure_ids,
    collect_structure_ids,
    replacement_structure_ids,
)


MAX_RECRUIT_PULSE_YEARS = 30
DISTANCE_THRESHOLD_METERS = 5.0
LOW_CONTROL_STATE = "low-control"
REDUCE_PRUNING_TO_PARK_YEARS = 20
ELIMINATE_PRUNING_TO_PARK_YEARS = 20
ELIMINATE_PRUNING_TO_LOW_YEARS = 40
FALLEN_TO_DECAY_RANGE_YEARS = (50, 150)
DECAYED_REMOVE_RANGE_YEARS = (50, 100)
PRUNING_TARGET_ORDER = {
    "standard-pruning": 0,
    "reduce-pruning": 1,
    "eliminate-pruning": 2,
}
RELEASE_SUPPORT_BY_REWILDING = {
    "node-rewilded": "eliminate-pruning",
    "footprint-depaved": "reduce-pruning",
}


def remap_values(values, old_min, old_max, new_min, new_max):
    if old_max == old_min:
        return np.full_like(values, new_max, dtype=float)
    return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def _as_object_series(values, length, default):
    if values is None:
        return pd.Series([default] * length, dtype="object")
    series = pd.Series(values)
    return series.fillna(default).astype(str)


def _legacy_control_to_target(control_value: str) -> str:
    if control_value == "park-tree":
        return "reduce-pruning"
    if control_value in {"reserve-tree", "improved-tree"}:
        return "eliminate-pruning"
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


def _site_specific_structures(site: str, log_df: pd.DataFrame | None, pole_df: pd.DataFrame | None):
    if site == "trimmed-parade":
        return None, None
    if site == "uni":
        log_df = None
    if site == "city":
        pole_df = None
    return log_df, pole_df


def _refresh_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "useful_life_expectency" in df.columns and "useful_life_expectancy" not in df.columns:
        df.rename(columns={"useful_life_expectency": "useful_life_expectancy"}, inplace=True)
    if "fallen_remove_after_years" in df.columns and "fallen_decay_after_years" not in df.columns:
        df.rename(columns={"fallen_remove_after_years": "fallen_decay_after_years"}, inplace=True)

    for column, default in {
        "rewilded": "paved",
        "action": "None",
        "isNewTree": False,
        "isRewildedTree": False,
        "hasbeenReplanted": False,
        "unmanagedCount": 0.0,
        "decay_support": "none",
        "release_control_support": "none",
        "recruit_support": "none",
        "recruit_intervention_type": "none",
        "recruit_source_id": "none",
        "recruit_year": np.nan,
        "colonise_support": "none",
        "deploy_structure_support": "none",
        "pruning_target": None,
        "pruning_target_years": np.nan,
        "autonomy_years": np.nan,
        "control_realized": None,
        "lifecycle_decision": None,
        "lifecycle_state": None,
        "fallen_since_year": np.nan,
        "fallen_decay_after_years": np.nan,
        "decayed_since_year": np.nan,
        "decayed_remove_after_years": np.nan,
        "structureID": np.nan,
    }.items():
        if column not in df.columns:
            df[column] = default

    df["size"] = _as_object_series(df.get("size"), len(df), "small")
    df["control"] = _as_object_series(df.get("control"), len(df), "street-tree")
    df["rewilded"] = _as_object_series(df.get("rewilded"), len(df), "paved")
    df["action"] = _as_object_series(df.get("action"), len(df), "None")
    df["recruit_intervention_type"] = _as_object_series(df.get("recruit_intervention_type"), len(df), "none")
    df["recruit_source_id"] = _as_object_series(df.get("recruit_source_id"), len(df), "none")

    if df["pruning_target"].isna().any():
        missing = df["pruning_target"].isna()
        df.loc[missing, "pruning_target"] = df.loc[missing, "control"].map(_legacy_control_to_target)
    df["pruning_target"] = _as_object_series(df["pruning_target"], len(df), "standard-pruning")

    if df["control_realized"].isna().any():
        missing = df["control_realized"].isna()
        df.loc[missing, "control_realized"] = df.loc[missing, "control"].map(_legacy_control_to_realized)
    df["control_realized"] = _as_object_series(df["control_realized"], len(df), "street-tree")

    if df["autonomy_years"].isna().any():
        missing = df["autonomy_years"].isna()
        df.loc[missing, "autonomy_years"] = df.loc[missing, "control"].map(_legacy_control_to_years)
    if df["pruning_target_years"].isna().any():
        missing = df["pruning_target_years"].isna()
        df.loc[missing, "pruning_target_years"] = df.loc[missing, "control"].map(_legacy_control_to_years)

    if df["release_control_support"].eq("none").all():
        support_map = {
            "standard-pruning": "none",
            "reduce-pruning": "reduce-pruning",
            "eliminate-pruning": "eliminate-pruning",
        }
        df["release_control_support"] = df["pruning_target"].map(support_map).fillna("none")

    if df["lifecycle_decision"].isna().any():
        missing = df["lifecycle_decision"].isna()
        action_map = {
            "AGE-IN-PLACE": "age-in-place",
            "REPLACE": "replace",
            "SENESCENT": "senescing",
            "None": "stable",
        }
        df.loc[missing, "lifecycle_decision"] = df.loc[missing, "action"].map(action_map).fillna("stable")
    df["lifecycle_decision"] = _as_object_series(df["lifecycle_decision"], len(df), "stable")

    if df["lifecycle_state"].isna().any():
        missing = df["lifecycle_state"].isna()
        state = np.where(
            df["size"].isin(["senescing", "snag", "fallen", "decayed"]),
            df["size"],
            "standing",
        )
        df.loc[missing, "lifecycle_state"] = state[missing]
    df["lifecycle_state"] = _as_object_series(df["lifecycle_state"], len(df), "standing")

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

    df["sim_NodesArea"] = df["sim_NodesArea"].fillna(df["CanopyArea"])
    df["autonomy_years"] = pd.to_numeric(df["autonomy_years"], errors="coerce").fillna(0.0)
    df["pruning_target_years"] = pd.to_numeric(df["pruning_target_years"], errors="coerce").fillna(0.0)
    df["recruit_year"] = pd.to_numeric(df["recruit_year"], errors="coerce")
    df["fallen_since_year"] = pd.to_numeric(df["fallen_since_year"], errors="coerce")
    df["fallen_decay_after_years"] = pd.to_numeric(df["fallen_decay_after_years"], errors="coerce")
    df["decayed_since_year"] = pd.to_numeric(df["decayed_since_year"], errors="coerce")
    df["decayed_remove_after_years"] = pd.to_numeric(df["decayed_remove_after_years"], errors="coerce")

    return df


def get_timestep_params(site: str, scenario: str, absolute_year: int, previous_year: int) -> dict:
    params = a_scenario_params.get_params_for_year(site, scenario, absolute_year)
    params["absolute_year"] = absolute_year
    params["previous_year"] = previous_year
    params["step_years"] = max(0, absolute_year - previous_year)
    return params


def split_window_into_pulses(previous_year: int, absolute_year: int, max_pulse_years: int = MAX_RECRUIT_PULSE_YEARS):
    if absolute_year <= previous_year:
        return []

    pulses = []
    pulse_start = previous_year
    while pulse_start < absolute_year:
        pulse_end = min(pulse_start + max_pulse_years, absolute_year)
        pulses.append((pulse_start, pulse_end))
        pulse_start = pulse_end
    return pulses


def age_trees(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    step_years = params["step_years"]
    if step_years <= 0:
        return df

    growth_factor_per_year = np.mean(params["growth_factor_range"])
    living_mask = df["size"].isin(["small", "medium", "large"])
    df.loc[living_mask, "diameter_breast_height"] = (
        df.loc[living_mask, "diameter_breast_height"] + (growth_factor_per_year * step_years)
    )

    growth_mask = df["size"].isin(["small", "medium"])
    df.loc[growth_mask, "size"] = pd.cut(
        df.loc[growth_mask, "diameter_breast_height"],
        bins=[-10, 30, 80, float("inf")],
        labels=["small", "medium", "large"],
    ).astype(str)

    df["useful_life_expectancy"] = df["useful_life_expectancy"] - step_years
    df.loc[df["size"].isin(["small", "medium", "large"]), "lifecycle_state"] = "standing"
    return df


def determine_lifecycle_decisions(df: pd.DataFrame, params: dict, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    np.random.seed(seed + int(params["absolute_year"]))

    living_mask = df["size"].isin(["small", "medium", "large"])
    df.loc[living_mask, "action"] = "None"
    df.loc[living_mask, "lifecycle_decision"] = "stable"

    if not living_mask.any():
        return df

    senesce_threshold = params["senescingThreshold"]
    df["senesceChance"] = remap_values(
        df["useful_life_expectancy"].to_numpy(dtype=float),
        old_min=senesce_threshold,
        old_max=0,
        new_min=100,
        new_max=0,
    ).clip(0, 100)
    df["senesceRoll"] = np.random.uniform(0, 100, len(df))

    senesce_mask = living_mask & (df["senesceRoll"] < df["senesceChance"])
    age_in_place_mask = senesce_mask & (df["CanopyResistance"] < params["ageInPlaceThreshold"])
    replace_mask = senesce_mask & ~age_in_place_mask

    df.loc[age_in_place_mask, "action"] = "AGE-IN-PLACE"
    df.loc[age_in_place_mask, "lifecycle_decision"] = "age-in-place"
    df.loc[replace_mask, "action"] = "REPLACE"
    df.loc[replace_mask, "lifecycle_decision"] = "replace"
    return df


def apply_senescence_states(df: pd.DataFrame, params: dict, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    np.random.seed(seed + 1000 + int(params["absolute_year"]))

    age_in_place_mask = (
        df["lifecycle_decision"].eq("age-in-place")
        | df["size"].isin(["senescing", "snag", "fallen", "decayed"])
    )

    new_senescing_mask = age_in_place_mask & df["size"].isin(["small", "medium", "large"])
    df.loc[new_senescing_mask, "size"] = "senescing"

    df["snagChance"] = remap_values(
        df["useful_life_expectancy"].to_numpy(dtype=float),
        old_min=params["snagThreshold"],
        old_max=0,
        new_min=100,
        new_max=0,
    ).clip(0, 100)
    df["collapseChance"] = remap_values(
        df["useful_life_expectancy"].to_numpy(dtype=float),
        old_min=params["collapsedThreshold"],
        old_max=0,
        new_min=100,
        new_max=0,
    ).clip(0, 100)
    df["snagRoll"] = np.random.uniform(0, 100, len(df))
    df["collapseRoll"] = np.random.uniform(0, 100, len(df))

    snag_mask = df["size"].eq("senescing") & (df["snagRoll"] < df["snagChance"])
    df.loc[snag_mask, "size"] = "snag"

    collapse_mask = df["size"].isin(["senescing", "snag"]) & (df["collapseRoll"] < df["collapseChance"])
    df.loc[collapse_mask, "size"] = "fallen"

    state_values = np.where(
        df["size"].isin(["senescing", "snag", "fallen", "decayed"]),
        df["size"],
        "standing",
    )
    df["lifecycle_state"] = state_values
    return df


def handle_replace_trees(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    replace_mask = df["action"].eq("REPLACE")
    if not replace_mask.any():
        return df

    step_years = params["step_years"]
    growth_factor = np.mean(params["growth_factor_range"])
    current_ule = df.loc[replace_mask, "useful_life_expectancy"].to_numpy(dtype=float)
    replacement_growth_years = np.clip(-np.minimum(0, current_ule), 0, step_years)
    replacement_dbh = 2 + (growth_factor * replacement_growth_years)

    df.loc[replace_mask, "diameter_breast_height"] = replacement_dbh
    df.loc[replace_mask, "precolonial"] = True
    df.loc[replace_mask, "useful_life_expectancy"] = 120 - replacement_growth_years
    df.loc[replace_mask, "size"] = pd.cut(
        df.loc[replace_mask, "diameter_breast_height"],
        bins=[-10, 30, 80, float("inf")],
        labels=["small", "medium", "large"],
    ).astype(str)
    df.loc[replace_mask, "lifecycle_state"] = "standing"
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


def assign_decay_support(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    df["decay_support"] = "none"
    decay_mask = df["lifecycle_decision"].eq("age-in-place")
    if not decay_mask.any():
        return df

    decay_rewilded = pd.cut(
        df.loc[decay_mask, "CanopyResistance"],
        bins=[
            -float("inf"),
            params["rewildThreshold"],
            params["plantThreshold"],
            params["ageInPlaceThreshold"],
            float("inf"),
        ],
        labels=["node-rewilded", "footprint-depaved", "exoskeleton", "None"],
    )
    decay_rewilded = pd.Series(decay_rewilded, index=df.index[decay_mask]).astype("object").fillna("None")
    df.loc[decay_mask, "rewilded"] = decay_rewilded
    df.loc[decay_mask & df["rewilded"].isin(["node-rewilded", "footprint-depaved"]), "decay_support"] = "buffer-feature"
    df.loc[decay_mask & df["rewilded"].eq("exoskeleton"), "decay_support"] = "brace-feature"
    return df


def refresh_colonise_support(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["colonise_support"] = "none"
    ground_mask = df["rewilded"].isin(["node-rewilded", "footprint-depaved"])
    df.loc[ground_mask, "colonise_support"] = "rewild-ground"
    return df


def _target_rank(values: pd.Series) -> pd.Series:
    return values.map(PRUNING_TARGET_ORDER).fillna(0).astype(int)


def _realized_control_from_years(pruning_target: str, autonomy_years: float) -> str:
    if pruning_target == "eliminate-pruning":
        if autonomy_years >= ELIMINATE_PRUNING_TO_LOW_YEARS:
            return LOW_CONTROL_STATE
        if autonomy_years >= ELIMINATE_PRUNING_TO_PARK_YEARS:
            return "park-tree"
        return "street-tree"
    if pruning_target == "reduce-pruning":
        if autonomy_years >= REDUCE_PRUNING_TO_PARK_YEARS:
            return "park-tree"
        return "street-tree"
    return "street-tree"


def apply_release_control(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    living_mask = df["size"].isin(["small", "medium", "large"])

    if living_mask.any():
        living_rewilded = pd.cut(
            df.loc[living_mask, "CanopyResistance"],
            bins=[-float("inf"), params["rewildThreshold"], params["plantThreshold"], float("inf")],
            labels=["node-rewilded", "footprint-depaved", "None"],
        )
        living_rewilded = pd.Series(living_rewilded, index=df.index[living_mask]).astype("object").fillna("None")
        df.loc[living_mask, "rewilded"] = living_rewilded

        support_series = living_rewilded.map(RELEASE_SUPPORT_BY_REWILDING).fillna("none")
        current_target = df.loc[living_mask, "pruning_target"].fillna("standard-pruning").astype(str)
        proposed_target = support_series.replace({"none": "standard-pruning"})

        current_rank = _target_rank(current_target)
        proposed_rank = _target_rank(proposed_target)
        chosen_rank = np.maximum(current_rank.to_numpy(dtype=int), proposed_rank.to_numpy(dtype=int))
        reverse_map = {value: key for key, value in PRUNING_TARGET_ORDER.items()}
        new_target = pd.Series(chosen_rank).map(reverse_map).to_numpy()

        changed_mask = current_target.to_numpy() != new_target
        df.loc[living_mask, "pruning_target"] = new_target

        current_target_years = df.loc[living_mask, "pruning_target_years"].to_numpy(dtype=float)
        autonomy_years = df.loc[living_mask, "autonomy_years"].to_numpy(dtype=float)
        step_years = params["step_years"]

        new_target_years = np.where(
            new_target == "standard-pruning",
            0.0,
            np.where(changed_mask, step_years, current_target_years + step_years),
        )
        new_autonomy_years = np.where(
            new_target == "standard-pruning",
            0.0,
            autonomy_years + step_years,
        )

        df.loc[living_mask, "pruning_target_years"] = new_target_years
        df.loc[living_mask, "autonomy_years"] = new_autonomy_years

    support_from_target = {
        "standard-pruning": "none",
        "reduce-pruning": "reduce-pruning",
        "eliminate-pruning": "eliminate-pruning",
    }
    df["release_control_support"] = df["pruning_target"].map(support_from_target).fillna("none")

    df["control_realized"] = [
        _realized_control_from_years(target, years)
        for target, years in zip(df["pruning_target"], df["autonomy_years"], strict=False)
    ]
    senescent_mask = df["size"].isin(["senescing", "snag", "fallen", "decayed"])
    df.loc[senescent_mask, "control_realized"] = LOW_CONTROL_STATE
    df["control"] = [
        _export_control_value(realized, size)
        for realized, size in zip(df["control_realized"], df["size"], strict=False)
    ]
    return df


def update_fallen_tracking(df: pd.DataFrame, site: str, scenario: str, current_year: int) -> pd.DataFrame:
    df = df.copy()
    fallen_mask = df["size"].eq("fallen")
    newly_fallen_mask = fallen_mask & df["fallen_since_year"].isna()
    if newly_fallen_mask.any():
        df.loc[newly_fallen_mask, "fallen_since_year"] = current_year

        decay_years = []
        for index, row in df.loc[newly_fallen_mask].iterrows():
            identity = row.get("structureID", row.get("NodeID", row.get("tree_number", index)))
            decay_years.append(
                _seeded_years(
                    site,
                    scenario,
                    identity,
                    current_year,
                    "fallen-to-decayed",
                    minimum=FALLEN_TO_DECAY_RANGE_YEARS[0],
                    maximum=FALLEN_TO_DECAY_RANGE_YEARS[1],
                )
            )
        df.loc[newly_fallen_mask, "fallen_decay_after_years"] = decay_years

    decayed_transition_mask = (
        fallen_mask
        & df["fallen_since_year"].notna()
        & df["fallen_decay_after_years"].notna()
        & ((current_year - df["fallen_since_year"]) >= df["fallen_decay_after_years"])
    )
    if decayed_transition_mask.any():
        df.loc[decayed_transition_mask, "size"] = "decayed"
        df.loc[decayed_transition_mask, "lifecycle_state"] = "decayed"
        df.loc[decayed_transition_mask, "decayed_since_year"] = current_year

        remove_years = []
        for index, row in df.loc[decayed_transition_mask].iterrows():
            identity = row.get("structureID", row.get("NodeID", row.get("tree_number", index)))
            remove_years.append(
                _seeded_years(
                    site,
                    scenario,
                    identity,
                    current_year,
                    "decayed-remove",
                    minimum=DECAYED_REMOVE_RANGE_YEARS[0],
                    maximum=DECAYED_REMOVE_RANGE_YEARS[1],
                )
            )
        df.loc[decayed_transition_mask, "decayed_remove_after_years"] = remove_years

    decayed_mask = df["size"].eq("decayed")
    newly_decayed_mask = decayed_mask & df["decayed_since_year"].isna()
    if newly_decayed_mask.any():
        df.loc[newly_decayed_mask, "decayed_since_year"] = current_year

        remove_years = []
        for index, row in df.loc[newly_decayed_mask].iterrows():
            identity = row.get("structureID", row.get("NodeID", row.get("tree_number", index)))
            remove_years.append(
                _seeded_years(
                    site,
                    scenario,
                    identity,
                    current_year,
                    "decayed-remove",
                    minimum=DECAYED_REMOVE_RANGE_YEARS[0],
                    maximum=DECAYED_REMOVE_RANGE_YEARS[1],
                )
            )
        df.loc[newly_decayed_mask, "decayed_remove_after_years"] = remove_years

    removable_mask = (
        decayed_mask
        & df["decayed_since_year"].notna()
        & df["decayed_remove_after_years"].notna()
        & ((current_year - df["decayed_since_year"]) >= df["decayed_remove_after_years"])
    )
    if removable_mask.any():
        df = df.loc[~removable_mask].copy()

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
        & ~df["size"].eq("fallen")
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
    recruit_support: str,
    recruit_intervention_type: str,
    recruit_source_id: str,
    recruit_year: int,
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
            "rewilded": "paved",
            "action": "None",
            "decay_support": "none",
            "release_control_support": "eliminate-pruning",
            "recruit_support": recruit_support,
            "recruit_intervention_type": recruit_intervention_type,
            "recruit_source_id": recruit_source_id,
            "recruit_year": recruit_year,
            "colonise_support": "rewild-ground",
            "deploy_structure_support": "none",
            "pruning_target": "eliminate-pruning",
            "pruning_target_years": float(ELIMINATE_PRUNING_TO_LOW_YEARS),
            "autonomy_years": float(ELIMINATE_PRUNING_TO_LOW_YEARS),
            "control_realized": LOW_CONTROL_STATE,
            "control": "reserve-tree",
            "lifecycle_decision": "stable",
            "lifecycle_state": "standing",
            "fallen_since_year": np.nan,
            "fallen_decay_after_years": np.nan,
            "decayed_since_year": np.nan,
            "decayed_remove_after_years": np.nan,
        }
    )
    return record


def calculate_rewilded_status(df: pd.DataFrame, ds: xr.Dataset, params: dict):
    ds = ds.copy(deep=True)
    absolute_year = int(params.get("absolute_year", params.get("years_passed", 0)))
    if absolute_year <= 0:
        ds["scenario_rewildingEnabled"] = xr.full_like(ds["node_CanopyID"], -1)
        ds["scenario_rewildingPlantings"] = xr.full_like(ds["node_CanopyID"], -1)
        return df, ds

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

    if len(df) == 0 or len(voxel_positions) == 0:
        ds["scenario_rewildingPlantings"] = xr.where(combined_mask, absolute_year, -1)
        return df, ds

    tree_locations = np.vstack([df["x"], df["y"], df["z"]]).T
    tree_kdtree = cKDTree(tree_locations)
    distances, _ = tree_kdtree.query(voxel_positions, distance_upper_bound=DISTANCE_THRESHOLD_METERS)
    filtered_proximity_mask = distances > DISTANCE_THRESHOLD_METERS

    proximity_mask = np.full(ds.sizes["voxel"], False)
    proximity_mask[filtered_voxel_ids] = filtered_proximity_mask

    final_mask = depaved_mask & terrain_mask & proximity_mask
    ds["scenario_rewildingPlantings"] = xr.where(final_mask, absolute_year, -1)
    return df, ds


def apply_recruit(df: pd.DataFrame, ds: xr.Dataset, params: dict) -> pd.DataFrame:
    df = df.copy()
    step_years = params["step_years"]
    if step_years <= 0:
        return df

    absolute_year = int(params["absolute_year"])
    pulse_factor = step_years / MAX_RECRUIT_PULSE_YEARS
    planting_density_sqm = params["plantingDensity"] / 10000
    df_template = _new_tree_template(df)
    voxel_points, voxel_tree = _build_voxel_tree(ds)

    current_positions = np.vstack([df["x"], df["y"], df["z"]]).T if len(df) else np.empty((0, 3))
    accepted_positions = [tuple(position) for position in current_positions]

    node_candidates: list[dict] = []
    existing_mask = ~df["isNewTree"].fillna(False)
    node_support_mask = existing_mask & df["rewilded"].isin(["node-rewilded", "footprint-depaved"])
    node_occupancy = _recruit_occupancy_counts(df, "buffer-feature")
    if node_support_mask.any():
        temp_area = np.where(
            df.loc[node_support_mask, "rewilded"].eq("node-rewilded"),
            df.loc[node_support_mask, "sim_NodesArea"],
            df.loc[node_support_mask, "CanopyArea"],
        )
        node_quota = np.ceil(temp_area * planting_density_sqm * pulse_factor).astype(int)
        for (_, parent_row), quota in zip(df.loc[node_support_mask].iterrows(), node_quota, strict=False):
            source_id = _pick_source_id(
                parent_row.get("NodeID"),
                parent_row.get("debugNodeID"),
                fallback="unknown",
            )
            remaining_capacity = max(0, int(quota) - node_occupancy.get(source_id, 0))
            if remaining_capacity <= 0:
                continue
            planted_here = 0
            attempts = 0
            while planted_here < remaining_capacity and attempts < max(remaining_capacity * 8, 8):
                attempts += 1
                candidate_position = (
                    float(parent_row["x"] + np.random.uniform(-2.5, 2.5)),
                    float(parent_row["y"] + np.random.uniform(-2.5, 2.5)),
                    float(parent_row["z"]),
                )
                if accepted_positions:
                    candidate_tree = cKDTree(np.array(accepted_positions))
                    distance, _ = candidate_tree.query(np.array(candidate_position), distance_upper_bound=DISTANCE_THRESHOLD_METERS)
                    if np.isfinite(distance) and distance <= DISTANCE_THRESHOLD_METERS:
                        continue
                voxel_index = _resolve_candidate_voxel(ds, voxel_tree, candidate_position)
                node_candidates.append(
                    _make_new_tree_record(
                        df_template=df_template,
                        ds=ds,
                        voxel_index=voxel_index,
                        position=candidate_position,
                        recruit_support="buffer-feature",
                        recruit_intervention_type="buffer-feature",
                        recruit_source_id=source_id,
                        recruit_year=absolute_year,
                    )
                )
                accepted_positions.append(candidate_position)
                planted_here += 1
                node_occupancy[source_id] = node_occupancy.get(source_id, 0) + 1

    turn_candidates: list[dict] = []
    planting_mask = ds["scenario_rewildingPlantings"].values == params["absolute_year"]
    if planting_mask.any():
        available_voxel_indices = np.arange(ds.sizes["voxel"])[planting_mask]
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

        ground_occupancy = _recruit_occupancy_counts(df, "rewild-ground")
        voxel_area_sqm = ds.attrs["voxel_size"] * ds.attrs["voxel_size"]

        for source_id, voxel_indices in ground_source_to_voxels.items():
            source_area_sqm = len(voxel_indices) * voxel_area_sqm
            source_quota = int(np.round(source_area_sqm * planting_density_sqm * pulse_factor))
            remaining_capacity = max(0, source_quota - ground_occupancy.get(source_id, 0))
            if remaining_capacity <= 0:
                continue

            candidate_indices = np.array(voxel_indices, dtype=int)
            np.random.shuffle(candidate_indices)

            for voxel_index in candidate_indices:
                if ground_occupancy.get(source_id, 0) >= source_quota:
                    break
                candidate_position = tuple(voxel_points[int(voxel_index)])
                if accepted_positions:
                    candidate_tree = cKDTree(np.array(accepted_positions))
                    distance, _ = candidate_tree.query(np.array(candidate_position), distance_upper_bound=DISTANCE_THRESHOLD_METERS)
                    if np.isfinite(distance) and distance <= DISTANCE_THRESHOLD_METERS:
                        continue
                turn_candidates.append(
                    _make_new_tree_record(
                        df_template=df_template,
                        ds=ds,
                        voxel_index=int(voxel_index),
                        position=candidate_position,
                        recruit_support="rewild-ground",
                        recruit_intervention_type="rewild-ground",
                        recruit_source_id=source_id,
                        recruit_year=absolute_year,
                    )
                )
                accepted_positions.append(candidate_position)
                ground_occupancy[source_id] = ground_occupancy.get(source_id, 0) + 1

    if node_candidates or turn_candidates:
        df = pd.concat([df, pd.DataFrame(node_candidates + turn_candidates)], ignore_index=True)

    return _refresh_schema(df)


def assign_logs(log_df: pd.DataFrame | None, params: dict) -> pd.DataFrame | None:
    if log_df is None:
        return None

    log_df = log_df.copy()
    turn_threshold = params["sim_TurnsThreshold"]
    resistance_threshold = params.get("sim_averageResistance", 0)
    mask = (log_df["sim_averageResistance"] <= resistance_threshold) & (log_df["sim_Turns"] <= turn_threshold)
    log_df["isEnabled"] = mask
    return log_df[log_df["isEnabled"]].copy()


def assign_poles(pole_df: pd.DataFrame | None, params: dict) -> pd.DataFrame | None:
    if pole_df is None:
        return None

    pole_df = pole_df.copy()
    resistance_threshold = params.get("sim_averageResistance", 0)
    mask = pole_df["sim_averageResistance"] < resistance_threshold
    pole_df["isEnabled"] = mask
    return pole_df[pole_df["isEnabled"]].copy()


def _run_single_pulse(
    df: pd.DataFrame,
    ds: xr.Dataset,
    site: str,
    scenario: str,
    params: dict,
) -> pd.DataFrame:
    df = _refresh_schema(df)
    df = age_trees(df, params)
    df = determine_lifecycle_decisions(df, params)
    df = apply_senescence_states(df, params)
    params = dict(params)
    params["site"] = site
    params["scenario"] = scenario
    df = handle_replace_trees(df, params)
    df = assign_decay_support(df, params)
    df = apply_release_control(df, params)
    df = refresh_colonise_support(df)
    df = update_fallen_tracking(df, site, scenario, params["absolute_year"])

    _, pulse_ds = calculate_rewilded_status(df, ds, params)
    df = apply_recruit(df, pulse_ds, params)
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
    tree_df["decay_support"] = "none"
    tree_df["recruit_support"] = "none"
    tree_df["colonise_support"] = "none"
    tree_df["deploy_structure_support"] = "none"
    tree_df["lifecycle_decision"] = "stable"
    tree_df["lifecycle_state"] = "standing"
    tree_df["rewilded"] = "paved"
    tree_df["control_realized"] = tree_df["control"].map(_legacy_control_to_realized)

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
    subset_ds: xr.Dataset,
    log_df: pd.DataFrame | None = None,
    pole_df: pd.DataFrame | None = None,
    output_mode: str | None = None,
    write_outputs: bool = True,
):
    if year == 0:
        return _year_zero_state(site, scenario, year, voxel_size, tree_df, log_df, pole_df, output_mode)

    tree_state = _refresh_schema(tree_df)
    for pulse_start, pulse_end in split_window_into_pulses(previous_year, year):
        pulse_params = get_timestep_params(site, scenario, pulse_end, pulse_start)
        tree_state = _run_single_pulse(tree_state, subset_ds.copy(deep=True), site, scenario, pulse_params)

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

    return tree_state, log_df, pole_df


def run_scenario(
    site,
    scenario,
    year,
    voxel_size,
    treeDF,
    subsetDS,
    logDF=None,
    poleDF=None,
    previous_year=None,
    output_mode=None,
    write_outputs=True,
):
    previous_year = 0 if previous_year is None else previous_year

    if logDF is not None:
        logDF = a_scenario_initialiseDS.log_processing(logDF, subsetDS)
    if poleDF is not None:
        poleDF = a_scenario_initialiseDS.pole_processing(poleDF, None, subsetDS)

    return run_timestep(
        site=site,
        scenario=scenario,
        year=year,
        previous_year=previous_year,
        voxel_size=voxel_size,
        tree_df=treeDF,
        subset_ds=subsetDS,
        log_df=logDF,
        pole_df=poleDF,
        output_mode=output_mode,
        write_outputs=write_outputs,
    )
