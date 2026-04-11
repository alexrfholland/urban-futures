"""
a_info_proposal_interventions.py
================================
Compute proposal opportunities and proposal-to-intervention conversation metrics
from existing scenario tree CSVs and urban-feature VTK files.

Outputs:
    - {site}_{voxel_size}_proposal_opportunities.csv
    - {site}_{voxel_size}_proposal_interventions.csv
    - {site}_{voxel_size}_proposal_qc.csv
    - all_sites_{voxel_size}_proposal_opportunities.csv
    - all_sites_{voxel_size}_proposal_interventions.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent

if str(CODE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT.parent))

from _futureSim_refactored.paths import (
    engine_output_state_vtk_path,
    normalize_output_mode,
    refactor_statistics_root,
    scenario_log_df_path,
    scenario_output_root,
    scenario_pole_df_path,
    scenario_tree_df_path,
)


SITES = ["trimmed-parade", "city", "uni"]
DEFAULT_SCENARIOS = ["positive", "trending"]
SITE_REPORT_LABELS = {
    "trimmed-parade": "PARADE",
    "uni": "STREET",
    "city": "CITY",
}
SITE_REPORT_ORDER = ["trimmed-parade", "uni", "city"]
DEFAULT_TABLE_YEARS = [0, 10, 30, 60, 90, 120, 150, 180]
RECRUIT_DISTANCE_M = 20.0
BUILDING_URBAN_VALUES = {"facade", "green roof", "brown roof"}

COLONISE_PROPOSAL_VALUES = {
    "brownroof",
    "greenroof",
    "livingfacade",
    "footprint-depaved",
    "node-rewilded",
    "otherground",
    "rewilded",
}
COLONISE_REWILD_VALUES = {"node-rewilded", "footprint-depaved", "rewilded"}
COLONISE_ENRICH_VALUES = {"greenroof"}
COLONISE_ROUGHEN_VALUES = {"brownroof", "livingfacade"}

DECAY_BUFFER_VALUES = {"node-rewilded", "footprint-depaved"}
RECRUIT_BUFFER_VALUES = {"node-rewilded", "footprint-depaved"}
RECRUIT_REWILD_VALUES = {"otherground", "rewilded"}

PROPOSAL_COUNT_EXPORTS = {
    "deploy_structure": [
        ("opportunity_tree_count", "utility poles"),
        ("opportunity_voxel_count", "artificial canopy voxels"),
    ],
    "decay": [
        ("opportunity_tree_count", "trees reaching senescence"),
        ("opportunity_voxel_count", "decay opportunity voxels"),
    ],
    "recruit": [
        ("opportunity_voxel_count", "recruitable ground voxels"),
    ],
    "colonise": [
        ("opportunity_voxel_count", "colonisable surface voxels"),
    ],
    "release_control": [
        ("opportunity_voxel_count", "arboreal voxels"),
    ],
}

INTERVENTION_SUPPORT_LEVELS = {
    ("deploy_structure", "Adapt-Utility-Pole"): "full",
    ("deploy_structure", "Upgrade-Feature"): "full",
    ("decay", "Buffer-Feature"): "full",
    ("decay", "Brace-Feature"): "partial",
    ("recruit", "Buffer-Feature"): "partial",
    ("recruit", "Rewild-Ground"): "full",
    ("colonise", "Rewild-Ground"): "mixed",
    ("colonise", "Enrich-Envelope"): "full",
    ("colonise", "Roughen-Envelope"): "partial",
    ("release_control", "Eliminate-Pruning"): "full",
    ("release_control", "Reduce-Pruning"): "partial",
}

INTERVENTION_COUNT_EXPORTS = {
    ("deploy_structure", "Adapt-Utility-Pole"): [
        ("supported_tree_count", "utility poles"),
        ("supported_voxel_count", "artificial canopy voxels"),
    ],
    ("deploy_structure", "Upgrade-Feature"): [
        ("supported_voxel_count", "upgraded-feature voxels"),
    ],
    ("decay", "Buffer-Feature"): [
        ("supported_tree_count", "senescing trees"),
        ("supported_voxel_count", "buffer-feature voxels"),
    ],
    ("decay", "Brace-Feature"): [
        ("supported_tree_count", "senescing trees"),
        ("supported_voxel_count", "brace-feature voxels"),
    ],
    ("recruit", "Buffer-Feature"): [
        ("supported_voxel_count", "recruit grassland voxels"),
    ],
    ("recruit", "Rewild-Ground"): [
        ("supported_voxel_count", "recruit grassland voxels"),
    ],
    ("colonise", "Rewild-Ground"): [
        ("supported_voxel_count", "rewilded ground voxels"),
    ],
    ("colonise", "Enrich-Envelope"): [
        ("supported_tree_count", "enabled rooftop logs"),
        ("supported_voxel_count", "green roof voxels"),
    ],
    ("colonise", "Roughen-Envelope"): [
        ("supported_voxel_count", "roughened envelope voxels"),
    ],
    ("release_control", "Eliminate-Pruning"): [
        ("supported_voxel_count", "arboreal voxels"),
    ],
    ("release_control", "Reduce-Pruning"): [
        ("supported_voxel_count", "arboreal voxels"),
    ],
}

PROPOSAL_IDS = [
    "deploy_structure",
    "decay",
    "recruit",
    "colonise",
    "release_control",
]

PROPOSAL_LABELS = {
    "deploy_structure": "Deploy-Structure",
    "decay": "Decay",
    "recruit": "Recruit",
    "colonise": "Colonise",
    "release_control": "Release-Control",
}

PROPOSAL_ALIASES = {
    "deploy": "deploy_structure",
    "deploy structure": "deploy_structure",
    "deploy_structure": "deploy_structure",
    "deploy-structure": "deploy_structure",
    "decay": "decay",
    "recruit": "recruit",
    "colonise": "colonise",
    "colonize": "colonise",
    "release control": "release_control",
    "release_control": "release_control",
    "release-control": "release_control",
}

INTERVENTION_ALIASES = {
    "buffer": "Buffer-Feature",
    "buffer feature": "Buffer-Feature",
    "buffer-feature": "Buffer-Feature",
    "brace": "Brace-Feature",
    "brace feature": "Brace-Feature",
    "brace-feature": "Brace-Feature",
    "rewild ground": "Rewild-Ground",
    "rewild-ground": "Rewild-Ground",
    "adapt utility pole": "Adapt-Utility-Pole",
    "adapt-utility-pole": "Adapt-Utility-Pole",
    "upgrade feature": "Upgrade-Feature",
    "upgrade-feature": "Upgrade-Feature",
    "enrich envelope": "Enrich-Envelope",
    "enrich-envelope": "Enrich-Envelope",
    "roughen envelope": "Roughen-Envelope",
    "roughen-envelope": "Roughen-Envelope",
    "reduce pruning": "Reduce-Pruning",
    "reduce-pruning": "Reduce-Pruning",
    "eliminate pruning": "Eliminate-Pruning",
    "eliminate-pruning": "Eliminate-Pruning",
}

OPPORTUNITY_COLUMNS = [
    "site",
    "scenario",
    "year",
    "proposal_id",
    "proposal_label",
    "opportunity_tree_count",
    "opportunity_voxel_count",
    "status",
    "notes",
]

INTERVENTION_COLUMNS = [
    "site",
    "scenario",
    "year",
    "proposal_id",
    "support_level",
    "support_label",
    "status",
    "supported_tree_count",
    "supported_tree_pct",
    "supported_voxel_count",
    "supported_voxel_pct",
    "proposal_tree_count",
    "proposal_voxel_count",
    "notes",
]

QC_COLUMNS = [
    "site",
    "scenario",
    "year",
    "check_name",
    "passed",
    "value",
    "details",
]

REQUIRED_TREE_COLUMNS = [
    "action",
    "size",
    "control",
    "under-node-treatment",
    "isNewTree",
    "precolonial",
]

REQUIRED_VTK_ARRAYS = [
    "search_bioavailable",
    "search_urban_elements",
    "scenario_outputs",
    "scenario_under-node-treatment",
    "scenario_bioEnvelope",
    "forest_control",
    "forest_precolonial",
    "forest_size",
    "indicator_Tree_generations_grassland",
    "indicator_Bird_self_peeling",
]

# -----------------------------------------------------------------------------
# INTERPRETATION NOTES (proposal -> intervention mapping)
# -----------------------------------------------------------------------------
# This script does not infer proposals/interventions from text. It maps them
# directly to existing model fields so each metric is reproducible:
#
# Deploy-Structure opportunity / Adapt-Utility-Pole:
#   pole_df.isEnabled == True AND pole_df.size == 'artificial'
#   AND pole_df.precolonial == False
# Upgrade-Feature:
#   vtk.forest_precolonial == False AND vtk.indicator_Bird_self_peeling == True
#
# Decay opportunity:
#   tree_df.isNewTree == False AND tree_df.action in {'AGE-IN-PLACE','SENESCENT'}
# Decay Buffer-Feature:
#   tree_df.rewilded in {'node-rewilded','footprint-depaved'}
#   vtk.scenario_bioEnvelope in {'node-rewilded','footprint-depaved'}
# Decay Brace-Feature:
#   tree_df.rewilded == 'exoskeleton'
#   vtk.scenario_bioEnvelope == 'exoskeleton'
#
# Recruit opportunity:
#   all ground_only voxels within 20m of a canopy-feature
# Recruit Buffer-Feature:
#   vtk.indicator_Tree_generations_grassland == True
#   AND vtk.scenario_bioEnvelope in {'node-rewilded','footprint-depaved'}
# Recruit Rewild-Ground:
#   vtk.indicator_Tree_generations_grassland == True
#   AND vtk.scenario_bioEnvelope in {'otherGround','rewilded'}
#
# Colonise opportunity:
#   vtk.scenario_outputs in {
#       'brownRoof','greenRoof','livingFacade','footprint-depaved',
#       'node-rewilded','otherGround','rewilded'
#   }
# Colonise Rewild-Ground:
#   vtk.scenario_outputs in {'node-rewilded','footprint-depaved','rewilded'}
# Colonise Enrich-Envelope:
#   vtk.scenario_outputs == 'greenRoof'
# Colonise Roughen-Envelope:
#   vtk.scenario_outputs in {'brownRoof','livingFacade'}
#
# Release-Control opportunity:
#   vtk.search_bioavailable == 'arboreal'
# Release-Control Eliminate-Pruning:
#   vtk.forest_control in {'reserve-tree','improved-tree'}
# Release-Control Reduce-Pruning:
#   vtk.forest_control == 'park-tree'
# -----------------------------------------------------------------------------


def normalize_token(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def canonical_proposal(value: str | None) -> str | None:
    if value is None:
        return None
    key = normalize_token(value).replace("_", " ").replace("-", " ")
    if key not in PROPOSAL_ALIASES:
        raise ValueError(f"Unknown proposal filter: {value}")
    return PROPOSAL_ALIASES[key]


def canonical_intervention(value: str | None) -> str | None:
    if value is None:
        return None
    key = normalize_token(value)
    if key not in INTERVENTION_ALIASES:
        raise ValueError(f"Unknown intervention filter: {value}")
    return INTERVENTION_ALIASES[key]


def parse_years_arg(value: str | None) -> list[int]:
    if value is None:
        return list(DEFAULT_TABLE_YEARS)
    text = value.strip().lower()
    if text in {"all", "default"}:
        return list(DEFAULT_TABLE_YEARS)
    years: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        years.append(int(token))
    return sorted(set(years))


def format_voxel_size(voxel_size: float) -> str:
    if float(voxel_size).is_integer():
        return str(int(voxel_size))
    return str(voxel_size)


def get_output_dir(output_mode: str | None) -> Path:
    return refactor_statistics_root(output_mode) / "csv"


def get_indicator_output_dir(output_mode: str | None) -> Path:
    return refactor_statistics_root(output_mode).parent


def normalize_scalar(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def vtk_str_array(values) -> np.ndarray:
    # Deliberately avoid per-value Python normalization loops.
    return np.asarray(values).astype(str)


def vtk_bool_array(values) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == bool:
        return array
    if np.issubdtype(array.dtype, np.number):
        return array.astype(float) != 0
    lowered = np.char.lower(array.astype(str))
    return np.isin(lowered, ["true", "1", "yes", "y", "t"])


def normalize_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_scalar)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    normalized = normalize_series(series).str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def pct_or_na(numerator, denominator):
    if denominator is None or pd.isna(denominator) or float(denominator) == 0:
        return pd.NA
    if numerator is None or pd.isna(numerator):
        return pd.NA
    return round(float(numerator) / float(denominator) * 100.0, 6)


def qc_row(site, scenario, year, name, passed, value, details):
    return {
        "site": site,
        "scenario": scenario,
        "year": year,
        "check_name": name,
        "passed": bool(passed),
        "value": value,
        "details": details,
    }


def opportunity_row(
    site,
    scenario,
    year,
    proposal_id,
    tree_count,
    voxel_count,
    status="computed",
    notes="",
):
    return {
        "site": site,
        "scenario": scenario,
        "year": year,
        "proposal_id": proposal_id,
        "proposal_label": PROPOSAL_LABELS[proposal_id],
        "opportunity_tree_count": tree_count,
        "opportunity_voxel_count": voxel_count,
        "status": status,
        "notes": notes,
    }


def intervention_row(
    site,
    scenario,
    year,
    proposal_id,
    support_level,
    support_label,
    supported_tree_count,
    supported_voxel_count,
    proposal_tree_count,
    proposal_voxel_count,
    status="computed",
    notes="",
):
    resolved_support_level = support_level or INTERVENTION_SUPPORT_LEVELS.get((proposal_id, support_label), "")
    return {
        "site": site,
        "scenario": scenario,
        "year": year,
        "proposal_id": proposal_id,
        "support_level": resolved_support_level,
        "support_label": support_label,
        "status": status,
        "supported_tree_count": supported_tree_count,
        "supported_tree_pct": pct_or_na(supported_tree_count, proposal_tree_count),
        "supported_voxel_count": supported_voxel_count,
        "supported_voxel_pct": pct_or_na(supported_voxel_count, proposal_voxel_count),
        "proposal_tree_count": proposal_tree_count,
        "proposal_voxel_count": proposal_voxel_count,
        "notes": notes,
    }


def get_tree_path(site: str, scenario: str, year: int, voxel_size: float, output_mode: str | None) -> Path:
    return scenario_tree_df_path(site, scenario, year, voxel_size, output_mode)


def get_vtk_path(site: str, scenario: str, year: int, voxel_size: float, output_mode: str | None) -> Path:
    voxel = format_voxel_size(voxel_size)
    site_base = scenario_output_root(output_mode) / site
    output_base = get_indicator_output_dir(output_mode)
    candidates = [
        engine_output_state_vtk_path(site, scenario, year, voxel_size, output_mode),
        # Backward compatibility fallbacks.
        output_base / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features_with_indicators.vtk",
        site_base / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features_with_indicators.vtk",
        output_base / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features.vtk",
        site_base / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features.vtk",
        site_base / f"{site}_{voxel}_{scenario}_scenarioYR{year}_urban_features.vtk",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def get_pole_path(site: str, scenario: str, year: int, voxel_size: float, output_mode: str | None) -> Path:
    return scenario_pole_df_path(site, scenario, year, voxel_size, output_mode)


def get_log_path(site: str, scenario: str, year: int, voxel_size: float, output_mode: str | None) -> Path:
    return scenario_log_df_path(site, scenario, year, voxel_size, output_mode)


def discover_scenarios(site: str, voxel_size: float, output_mode: str | None) -> list[str]:
    voxel = format_voxel_size(voxel_size)
    base = scenario_output_root(output_mode) / site
    if not base.exists():
        return []
    pattern = re.compile(rf"^{re.escape(site)}_(.+)_{re.escape(voxel)}_treeDF_(\d+)\.csv$")
    scenarios: set[str] = set()
    for path in base.glob(f"{site}_*_{voxel}_treeDF_*.csv"):
        match = pattern.match(path.name)
        if match:
            scenarios.add(match.group(1))
    return sorted(scenarios)


def discover_years(site: str, scenario: str, voxel_size: float, output_mode: str | None) -> list[int]:
    voxel = format_voxel_size(voxel_size)
    base = scenario_output_root(output_mode) / site
    if not base.exists():
        return []
    pattern = re.compile(
        rf"^{re.escape(site)}_{re.escape(scenario)}_{re.escape(voxel)}_treeDF_(\d+)\.csv$"
    )
    years: set[int] = set()
    for path in base.glob(f"{site}_{scenario}_{voxel}_treeDF_*.csv"):
        match = pattern.match(path.name)
        if match:
            years.add(int(match.group(1)))
    return sorted(years)


def should_include_proposal(proposal_id: str, proposal_filter: str | None) -> bool:
    return proposal_filter is None or proposal_filter == proposal_id


def should_include_intervention(label: str, intervention_filter: str | None) -> bool:
    return intervention_filter is None or intervention_filter == label


def build_building_mask(urban_values: np.ndarray) -> np.ndarray:
    lower = np.char.lower(urban_values.astype(str))
    return np.isin(lower, list(BUILDING_URBAN_VALUES))


def points_within_distance(points: np.ndarray, reference_mask: np.ndarray, distance_m: float) -> np.ndarray:
    if points.size == 0 or not np.any(reference_mask):
        return np.zeros(points.shape[0], dtype=bool)
    tree = cKDTree(points[reference_mask])
    distances, _ = tree.query(points, k=1)
    return distances <= distance_m


def compute_metrics_for_combo(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float,
    proposal_filter: str | None,
    intervention_filter: str | None,
    include_stubs: bool,
    output_mode: str | None,
):
    opportunity_rows = []
    intervention_rows = []
    qc_rows = []

    tree_path = get_tree_path(site, scenario, year, voxel_size, output_mode)
    vtk_path = get_vtk_path(site, scenario, year, voxel_size, output_mode)

    missing_paths = [str(p) for p in [tree_path, vtk_path] if not p.exists()]
    if missing_paths:
        qc_rows.append(
            qc_row(
                site,
                scenario,
                year,
                "input_files_exist",
                False,
                ",".join(missing_paths),
                "Tree CSV and/or urban_features VTK is missing for this timestep.",
            )
        )
        return opportunity_rows, intervention_rows, qc_rows

    qc_rows.append(qc_row(site, scenario, year, "input_files_exist", True, "ok", "Required files found."))

    tree_df = pd.read_csv(tree_path)
    missing_tree_cols = [c for c in REQUIRED_TREE_COLUMNS if c not in tree_df.columns]
    if missing_tree_cols:
        qc_rows.append(
            qc_row(
                site,
                scenario,
                year,
                "required_tree_columns_present",
                False,
                ",".join(missing_tree_cols),
                "Tree dataframe is missing required columns.",
            )
        )
        return opportunity_rows, intervention_rows, qc_rows
    qc_rows.append(
        qc_row(site, scenario, year, "required_tree_columns_present", True, "ok", "All required tree columns present.")
    )

    poly = pv.read(str(vtk_path))
    missing_arrays = [a for a in REQUIRED_VTK_ARRAYS if a not in poly.point_data]
    if missing_arrays:
        qc_rows.append(
            qc_row(
                site,
                scenario,
                year,
                "required_vtk_arrays_present",
                False,
                ",".join(missing_arrays),
                "Urban features VTK is missing required arrays.",
            )
        )
        return opportunity_rows, intervention_rows, qc_rows
    qc_rows.append(
        qc_row(site, scenario, year, "required_vtk_arrays_present", True, "ok", "All required VTK arrays present.")
    )

    pole_path = get_pole_path(site, scenario, year, voxel_size, output_mode)
    log_path = get_log_path(site, scenario, year, voxel_size, output_mode)
    pole_df = pd.read_csv(pole_path) if pole_path.exists() else pd.DataFrame()
    log_df = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    action = normalize_series(tree_df["action"])
    under_node_treatment = normalize_series(tree_df["under-node-treatment"])
    is_new_tree = bool_series(tree_df["isNewTree"])

    scenario_under_node_treatment = vtk_str_array(poly.point_data["scenario_under-node-treatment"])
    scenario_bio_envelope = vtk_str_array(poly.point_data["scenario_bioEnvelope"])
    scenario_outputs = vtk_str_array(poly.point_data["scenario_outputs"])
    search_bioavailable = vtk_str_array(poly.point_data["search_bioavailable"])
    search_urban_elements = vtk_str_array(poly.point_data["search_urban_elements"])
    forest_control = vtk_str_array(poly.point_data["forest_control"])
    forest_precolonial = vtk_bool_array(poly.point_data["forest_precolonial"])
    forest_size = vtk_str_array(poly.point_data["forest_size"])
    recruit_indicator = vtk_bool_array(poly.point_data["indicator_Tree_generations_grassland"])
    peeling_indicator = vtk_bool_array(poly.point_data["indicator_Bird_self_peeling"])

    scenario_under_node_treatment_lower = np.char.lower(scenario_under_node_treatment)
    scenario_bio_envelope_lower = np.char.lower(scenario_bio_envelope)
    scenario_outputs_lower = np.char.lower(scenario_outputs)
    search_bioavailable_lower = np.char.lower(search_bioavailable)
    search_urban_elements_lower = np.char.lower(search_urban_elements)
    forest_control_lower = np.char.lower(forest_control)
    forest_size_lower = np.char.lower(forest_size)

    # ------------------------------------------------------------------
    # DEPLOY-STRUCTURE
    # ------------------------------------------------------------------
    if should_include_proposal("deploy_structure", proposal_filter):
        if not pole_df.empty and {"isEnabled", "size", "precolonial"}.issubset(pole_df.columns):
            pole_is_enabled = bool_series(pole_df["isEnabled"])
            pole_size = normalize_series(pole_df["size"])
            pole_precolonial = bool_series(pole_df["precolonial"])
            deploy_pole_mask = pole_is_enabled & (pole_size == "artificial") & (~pole_precolonial)
            deploy_structure_count = int(deploy_pole_mask.sum())
        else:
            deploy_structure_count = 0

        deploy_voxel_mask = (forest_size_lower == "artificial") & (~forest_precolonial)
        deploy_voxel_count = int(np.sum(deploy_voxel_mask))

        opportunity_rows.append(
            opportunity_row(
                site,
                scenario,
                year,
                "deploy_structure",
                deploy_structure_count,
                deploy_voxel_count,
                status="computed",
                notes="Number of utility poles designed for artificial canopies this turn.",
            )
        )

        if should_include_intervention("Adapt-Utility-Pole", intervention_filter):
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "deploy_structure",
                    "",
                    "Adapt-Utility-Pole",
                    deploy_structure_count,
                    deploy_voxel_count,
                    deploy_structure_count,
                    deploy_voxel_count,
                    status="computed",
                    notes="Measured from enabled artificial poles in poleDF.",
                )
            )

        if should_include_intervention("Upgrade-Feature", intervention_filter):
            upgrade_voxel_mask = (~forest_precolonial) & peeling_indicator
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "deploy_structure",
                    "",
                    "Upgrade-Feature",
                    pd.NA,
                    int(np.sum(upgrade_voxel_mask)),
                    deploy_structure_count,
                    deploy_voxel_count,
                    status="computed",
                    notes=(
                        "Assumes all peeling bark in elms is artificial bark installation; "
                        "currently does not track artificial hollows in upgraded canopies."
                    ),
                )
            )
        else:
            upgrade_voxel_mask = np.zeros(poly.n_points, dtype=bool)

    # ------------------------------------------------------------------
    # DECAY
    # ------------------------------------------------------------------
    if should_include_proposal("decay", proposal_filter):
        decay_tree_opp = (~is_new_tree) & action.isin(["AGE-IN-PLACE", "SENESCENT"])
        decay_tree_count = int(decay_tree_opp.sum())
        decay_voxel_opp = np.isin(
            scenario_under_node_treatment_lower,
            ["exoskeleton", "footprint-depaved", "node-rewilded", "rewilded"],
        )
        decay_voxel_count = int(np.sum(decay_voxel_opp))

        opportunity_rows.append(
            opportunity_row(
                site,
                scenario,
                year,
                "decay",
                decay_tree_count,
                decay_voxel_count,
                status="computed",
                notes="Number of trees reaching senescence this turn.",
            )
        )

        if should_include_intervention("Buffer-Feature", intervention_filter):
            tree_mask = decay_tree_opp & under_node_treatment.isin(list(DECAY_BUFFER_VALUES))
            voxel_mask = np.isin(scenario_bio_envelope_lower, list(DECAY_BUFFER_VALUES))
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "decay",
                    "",
                    "Buffer-Feature",
                    int(tree_mask.sum()),
                    int(np.sum(voxel_mask)),
                    decay_tree_count,
                    decay_voxel_count,
                    status="computed",
                    notes="Ageing feature can senesce and collapse in place.",
                )
            )

        if should_include_intervention("Brace-Feature", intervention_filter):
            tree_mask = decay_tree_opp & (under_node_treatment == "exoskeleton")
            voxel_mask = scenario_bio_envelope_lower == "exoskeleton"
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "decay",
                    "",
                    "Brace-Feature",
                    int(tree_mask.sum()),
                    int(np.sum(voxel_mask)),
                    decay_tree_count,
                    decay_voxel_count,
                    status="computed",
                    notes="Ageing feature retained in place without collapse zone.",
                )
            )

    # ------------------------------------------------------------------
    # RECRUIT
    # ------------------------------------------------------------------
    if should_include_proposal("recruit", proposal_filter):
        canopy_feature_mask = ~np.isin(forest_size_lower, ["", "nan", "none"])
        if "stat_fallen log" in poly.point_data:
            fallen_log = np.asarray(poly.point_data["stat_fallen log"])
            if np.issubdtype(fallen_log.dtype, np.number):
                canopy_feature_mask |= fallen_log > 0
        building_mask = build_building_mask(search_urban_elements)
        recruit_voxel_opp = points_within_distance(poly.points, canopy_feature_mask, RECRUIT_DISTANCE_M) & (~building_mask)
        recruit_voxel_count = int(np.sum(recruit_voxel_opp))

        opportunity_rows.append(
            opportunity_row(
                site,
                scenario,
                year,
                "recruit",
                pd.NA,
                recruit_voxel_count,
                status="computed",
                notes="All ground_only voxels within 20m of a canopy-feature.",
            )
        )

        if should_include_intervention("Buffer-Feature", intervention_filter):
            voxel_mask = recruit_indicator & np.isin(scenario_bio_envelope_lower, list(RECRUIT_BUFFER_VALUES))
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "recruit",
                    "",
                    "Buffer-Feature",
                    pd.NA,
                    int(np.sum(voxel_mask)),
                    pd.NA,
                    recruit_voxel_count,
                    status="computed",
                    notes="Recruit grassland in node-rewilded or footprint-depaved bioEnvelope states.",
                )
            )

        if should_include_intervention("Rewild-Ground", intervention_filter):
            voxel_mask = recruit_indicator & np.isin(scenario_bio_envelope_lower, list(RECRUIT_REWILD_VALUES))
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "recruit",
                    "",
                    "Rewild-Ground",
                    pd.NA,
                    int(np.sum(voxel_mask)),
                    pd.NA,
                    recruit_voxel_count,
                    status="computed",
                    notes="Recruit grassland in otherGround or rewilded bioEnvelope states.",
                )
            )

    # ------------------------------------------------------------------
    # RELEASE CONTROL
    # ------------------------------------------------------------------
    if should_include_proposal("release_control", proposal_filter):
        release_voxel_opp = search_bioavailable_lower == "arboreal"

        release_tree_count = pd.NA
        release_voxel_count = int(np.sum(release_voxel_opp))

        opportunity_rows.append(
            opportunity_row(
                site,
                scenario,
                year,
                "release_control",
                release_tree_count,
                release_voxel_count,
                status="computed",
                notes="All arboreal voxels.",
            )
        )

        if should_include_intervention("Eliminate-Pruning", intervention_filter):
            voxel_mask = release_voxel_opp & np.isin(
                forest_control_lower, ["reserve-tree", "reserve tree", "improved-tree", "improved tree"]
            )
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "release_control",
                    "",
                    "Eliminate-Pruning",
                    pd.NA,
                    int(np.sum(voxel_mask)),
                    release_tree_count,
                    release_voxel_count,
                    status="computed",
                    notes="Assumes pruning is fully withdrawn from canopy.",
                )
            )

        if should_include_intervention("Reduce-Pruning", intervention_filter):
            voxel_mask = release_voxel_opp & np.isin(forest_control_lower, ["park-tree", "park tree"])
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "release_control",
                    "",
                    "Reduce-Pruning",
                    pd.NA,
                    int(np.sum(voxel_mask)),
                    release_tree_count,
                    release_voxel_count,
                    status="computed",
                    notes="Assumes pruning is reduced but not eliminated.",
                )
            )

    # ------------------------------------------------------------------
    # COLONISE
    # ------------------------------------------------------------------
    if should_include_proposal("colonise", proposal_filter):
        colonise_voxel_opp = np.isin(scenario_outputs_lower, list(COLONISE_PROPOSAL_VALUES))
        colonise_tree_count = pd.NA
        colonise_voxel_count = int(np.sum(colonise_voxel_opp))
        opportunity_rows.append(
            opportunity_row(
                site,
                scenario,
                year,
                "colonise",
                colonise_tree_count,
                colonise_voxel_count,
                status="computed",
                notes="Voxels designated as brownRoof, greenRoof, livingFacade, footprint-depaved, node-rewilded, otherGround, rewilded.",
            )
        )

        if should_include_intervention("Rewild-Ground", intervention_filter):
            voxel_mask = colonise_voxel_opp & np.isin(scenario_outputs_lower, list(COLONISE_REWILD_VALUES))
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "colonise",
                    "",
                    "Rewild-Ground",
                    pd.NA,
                    int(np.sum(voxel_mask)),
                    colonise_tree_count,
                    colonise_voxel_count,
                    status="computed",
                    notes="Rewilded ground grouped from node-rewilded, footprint-depaved, and rewilded.",
                )
            )

        if should_include_intervention("Enrich-Envelope", intervention_filter):
            voxel_mask = colonise_voxel_opp & np.isin(scenario_outputs_lower, list(COLONISE_ENRICH_VALUES))
            if not log_df.empty and {"isEnabled", "roofID"}.issubset(log_df.columns):
                log_enabled = bool_series(log_df["isEnabled"])
                roof_present = log_df["roofID"].notna()
                enrich_feature_count = int((log_enabled & roof_present).sum())
            else:
                enrich_feature_count = pd.NA
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "colonise",
                    "",
                    "Enrich-Envelope",
                    enrich_feature_count,
                    int(np.sum(voxel_mask)),
                    colonise_tree_count,
                    colonise_voxel_count,
                    status="computed",
                    notes=(
                        "Green roof envelope counted here; rooftop logs are measured from logDF and "
                        "still need a dedicated field for explicit Blender highlighting."
                    ),
                )
            )

        if should_include_intervention("Roughen-Envelope", intervention_filter):
            voxel_mask = colonise_voxel_opp & np.isin(scenario_outputs_lower, list(COLONISE_ROUGHEN_VALUES))
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "colonise",
                    "",
                    "Roughen-Envelope",
                    pd.NA,
                    int(np.sum(voxel_mask)),
                    colonise_tree_count,
                    colonise_voxel_count,
                    status="computed",
                    notes="Brown roof and living facade roughening counted together.",
                )
            )

    # QC checks specific to filtered brace test path
    if proposal_filter is not None and intervention_filter is not None:
        computed_rows = [
            row
            for row in intervention_rows
            if row["status"] == "computed"
            and row["proposal_id"] == proposal_filter
            and row["support_label"] == intervention_filter
        ]
        qc_rows.append(
            qc_row(
                site,
                scenario,
                year,
                "filtered_intervention_row_count",
                len(computed_rows) == 1,
                len(computed_rows),
                "Expected exactly one computed intervention row for proposal+intervention filter.",
            )
        )

    qc_rows.append(
        qc_row(
            site,
            scenario,
            year,
            "non_negative_supported_voxel_counts",
            all(
                (pd.isna(row["supported_voxel_count"]) or float(row["supported_voxel_count"]) >= 0)
                for row in intervention_rows
            ),
            "ok",
            "All supported voxel counts are non-negative.",
        )
    )

    return opportunity_rows, intervention_rows, qc_rows


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def process_site(
    site: str,
    scenario: str | None,
    year: int | None,
    voxel_size: float,
    proposal_filter: str | None,
    intervention_filter: str | None,
    include_stubs: bool,
    output_mode: str | None,
):
    scenarios = [scenario] if scenario else discover_scenarios(site, voxel_size, output_mode)
    if not scenarios:
        scenarios = DEFAULT_SCENARIOS

    site_opportunity_rows = []
    site_intervention_rows = []
    site_qc_rows = []

    for current_scenario in scenarios:
        years = [year] if year is not None else discover_years(site, current_scenario, voxel_size, output_mode)
        if not years and year is not None:
            years = [year]
        for current_year in years:
            opp_rows, int_rows, qc_rows = compute_metrics_for_combo(
                site=site,
                scenario=current_scenario,
                year=current_year,
                voxel_size=voxel_size,
                proposal_filter=proposal_filter,
                intervention_filter=intervention_filter,
                include_stubs=include_stubs,
                output_mode=output_mode,
            )
            site_opportunity_rows.extend(opp_rows)
            site_intervention_rows.extend(int_rows)
            site_qc_rows.extend(qc_rows)

    opportunity_df = ensure_columns(pd.DataFrame(site_opportunity_rows), OPPORTUNITY_COLUMNS)
    intervention_df = ensure_columns(pd.DataFrame(site_intervention_rows), INTERVENTION_COLUMNS)
    qc_df = ensure_columns(pd.DataFrame(site_qc_rows), QC_COLUMNS)

    return opportunity_df, intervention_df, qc_df


def save_site_outputs(
    site: str,
    voxel_size: float,
    opportunity_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    output_mode: str | None,
):
    output_dir = get_output_dir(output_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel = format_voxel_size(voxel_size)
    opportunity_path = output_dir / f"{site}_{voxel}_proposal_opportunities.csv"
    intervention_path = output_dir / f"{site}_{voxel}_proposal_interventions.csv"
    qc_path = output_dir / f"{site}_{voxel}_proposal_qc.csv"

    opportunity_df.to_csv(opportunity_path, index=False)
    intervention_df.to_csv(intervention_path, index=False)
    qc_df.to_csv(qc_path, index=False)

    print(f"Saved: {opportunity_path}")
    print(f"Saved: {intervention_path}")
    print(f"Saved: {qc_path}")


def combine_all_sites(voxel_size: float, output_mode: str | None):
    voxel = format_voxel_size(voxel_size)
    output_dir = get_output_dir(output_mode)
    all_opp = []
    all_int = []
    for site in SITES:
        opp_path = output_dir / f"{site}_{voxel}_proposal_opportunities.csv"
        int_path = output_dir / f"{site}_{voxel}_proposal_interventions.csv"
        if opp_path.exists():
            all_opp.append(pd.read_csv(opp_path))
        if int_path.exists():
            all_int.append(pd.read_csv(int_path))

    combined_opp = ensure_columns(pd.concat(all_opp, ignore_index=True), OPPORTUNITY_COLUMNS) if all_opp else ensure_columns(pd.DataFrame(), OPPORTUNITY_COLUMNS)
    combined_int = ensure_columns(pd.concat(all_int, ignore_index=True), INTERVENTION_COLUMNS) if all_int else ensure_columns(pd.DataFrame(), INTERVENTION_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_opp = output_dir / f"all_sites_{voxel}_proposal_opportunities.csv"
    out_int = output_dir / f"all_sites_{voxel}_proposal_interventions.csv"
    combined_opp.to_csv(out_opp, index=False)
    combined_int.to_csv(out_int, index=False)
    print(f"Saved: {out_opp}")
    print(f"Saved: {out_int}")
    return combined_opp, combined_int


def build_refactor_raw_tables(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float,
    output_mode: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proposal_rows: list[dict] = []
    intervention_rows: list[dict] = []

    for site in sites:
        for scenario in scenarios:
            for year in years:
                opportunity_rows, intervention_metric_rows, _ = compute_metrics_for_combo(
                    site=site,
                    scenario=scenario,
                    year=year,
                    voxel_size=voxel_size,
                    proposal_filter=None,
                    intervention_filter=None,
                    include_stubs=False,
                    output_mode=output_mode,
                )
                for row in opportunity_rows:
                    for metric_column, measure in PROPOSAL_COUNT_EXPORTS.get(row["proposal_id"], []):
                        proposal_rows.append(
                            {
                                "site": site,
                                "scenario": scenario,
                                "year": year,
                                "proposal_id": row["proposal_id"],
                                "proposal_label": row["proposal_label"],
                                "measure": measure,
                                "metric_column": metric_column,
                                "value": row.get(metric_column, pd.NA),
                                "notes": row.get("notes", ""),
                            }
                        )
                for row in intervention_metric_rows:
                    key = (row["proposal_id"], row["support_label"])
                    for metric_column, measure in INTERVENTION_COUNT_EXPORTS.get(key, []):
                        intervention_rows.append(
                            {
                                "site": site,
                                "scenario": scenario,
                                "year": year,
                                "proposal_id": row["proposal_id"],
                                "proposal_label": PROPOSAL_LABELS[row["proposal_id"]],
                                "intervention": row["support_label"],
                                "support": INTERVENTION_SUPPORT_LEVELS.get(key, ""),
                                "measure": measure,
                                "metric_column": metric_column,
                                "value": row.get(metric_column, pd.NA),
                                "notes": row.get("notes", ""),
                            }
                        )

    raw_proposals = pd.DataFrame(
        proposal_rows,
        columns=[
            "site",
            "scenario",
            "year",
            "proposal_id",
            "proposal_label",
            "measure",
            "metric_column",
            "value",
            "notes",
        ],
    )
    raw_interventions = pd.DataFrame(
        intervention_rows,
        columns=[
            "site",
            "scenario",
            "year",
            "proposal_id",
            "proposal_label",
            "intervention",
            "support",
            "measure",
            "metric_column",
            "value",
            "notes",
        ],
    )
    return raw_proposals, raw_interventions


def refactor_statistics_dir(*parts: str, output_mode: str | None = None) -> Path:
    path = refactor_statistics_root(output_mode)
    for part in parts:
        path /= part
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def refactor_site_raw_path(site: str, kind: str, output_mode: str | None = None) -> Path:
    return refactor_statistics_dir("raw", site, f"{kind}.csv", output_mode=output_mode)


def refactor_aggregate_path(folder: str, kind: str, output_mode: str | None = None) -> Path:
    return refactor_statistics_dir(folder, f"{kind}.csv", output_mode=output_mode)


def update_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return ensure_columns(pd.read_csv(path), columns)


def replace_scope_rows(
    existing_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    scope_columns: list[str],
    output_columns: list[str],
) -> pd.DataFrame:
    existing_df = ensure_columns(existing_df.copy(), output_columns)
    incoming_df = ensure_columns(incoming_df.copy(), output_columns)

    if scope_df.empty:
        return existing_df if incoming_df.empty else ensure_columns(incoming_df, output_columns)

    scope_df = scope_df[scope_columns].drop_duplicates()
    existing_without_replaced = existing_df.merge(
        scope_df,
        on=scope_columns,
        how="left",
        indicator=True,
    )
    existing_without_replaced = existing_without_replaced[
        existing_without_replaced["_merge"] == "left_only"
    ].drop(columns=["_merge"])

    combined = pd.concat([existing_without_replaced, incoming_df], ignore_index=True)
    sort_columns = [column for column in output_columns if column != "notes"]
    combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return ensure_columns(combined, output_columns)


def collect_all_site_raw_tables(kind: str, columns: list[str], output_mode: str | None = None) -> pd.DataFrame:
    raw_root = refactor_statistics_dir("raw", output_mode=output_mode)
    site_paths = sorted(raw_root.glob(f"*/{kind}.csv"))
    frames = [ensure_columns(pd.read_csv(path), columns) for path in site_paths if path.exists()]
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    return ensure_columns(combined, columns)


def build_comparison_table(
    raw_df: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=key_columns + [
            "positive_value",
            "trending_value",
            "delta_trending_minus_positive",
            "trending_pct_of_positive",
            "positive_multiple_of_trending",
        ])

    comparison = raw_df[key_columns].drop_duplicates().reset_index(drop=True)
    positive_values = (
        raw_df[raw_df["scenario"] == "positive"][key_columns + ["value"]]
        .drop_duplicates(subset=key_columns, keep="first")
        .rename(columns={"value": "positive_value"})
    )
    trending_values = (
        raw_df[raw_df["scenario"] == "trending"][key_columns + ["value"]]
        .drop_duplicates(subset=key_columns, keep="first")
        .rename(columns={"value": "trending_value"})
    )
    comparison = comparison.merge(positive_values, on=key_columns, how="left")
    comparison = comparison.merge(trending_values, on=key_columns, how="left")

    def pct_of_positive(row):
        positive = row["positive_value"]
        trending = row["trending_value"]
        if pd.isna(positive) or pd.isna(trending) or float(positive) == 0:
            return pd.NA
        return round(float(trending) / float(positive) * 100.0, 6)

    def positive_multiple(row):
        positive = row["positive_value"]
        trending = row["trending_value"]
        if pd.isna(positive) or pd.isna(trending) or float(trending) == 0:
            return pd.NA
        return round(float(positive) / float(trending), 6)

    comparison["delta_trending_minus_positive"] = comparison["trending_value"] - comparison["positive_value"]
    comparison["trending_pct_of_positive"] = comparison.apply(pct_of_positive, axis=1)
    comparison["positive_multiple_of_trending"] = comparison.apply(positive_multiple, axis=1)
    return comparison


def build_highlights_table(comparison_df: pd.DataFrame, key_columns: list[str], top_n: int = 15) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame(columns=["highlight_type"] + key_columns + [
            "positive_value",
            "trending_value",
            "delta_trending_minus_positive",
            "trending_pct_of_positive",
            "positive_multiple_of_trending",
        ])

    positive_dominates = comparison_df[
        comparison_df["positive_value"].fillna(0) > comparison_df["trending_value"].fillna(0)
    ].copy()
    positive_dominates = positive_dominates.sort_values(
        ["trending_pct_of_positive", "positive_multiple_of_trending"],
        ascending=[True, False],
        na_position="last",
    ).head(top_n)
    positive_dominates["highlight_type"] = "positive_dominates"

    trending_dominates = comparison_df[
        comparison_df["trending_value"].fillna(0) > comparison_df["positive_value"].fillna(0)
    ].copy()
    trending_dominates = trending_dominates.sort_values(
        ["delta_trending_minus_positive"],
        ascending=[False],
        na_position="last",
    ).head(top_n)
    trending_dominates["highlight_type"] = "trending_dominates"

    ordered_columns = ["highlight_type"] + key_columns + [
        "positive_value",
        "trending_value",
        "delta_trending_minus_positive",
        "trending_pct_of_positive",
        "positive_multiple_of_trending",
    ]
    return pd.concat([positive_dominates, trending_dominates], ignore_index=True)[ordered_columns]


def save_refactor_statistics_exports(
    raw_proposals: pd.DataFrame,
    raw_interventions: pd.DataFrame,
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    output_mode: str | None = None,
):
    proposal_raw_columns = [
        "site",
        "scenario",
        "year",
        "proposal_id",
        "proposal_label",
        "measure",
        "metric_column",
        "value",
        "notes",
        "last_updated",
    ]
    intervention_raw_columns = [
        "site",
        "scenario",
        "year",
        "proposal_id",
        "proposal_label",
        "intervention",
        "support",
        "measure",
        "metric_column",
        "value",
        "notes",
        "last_updated",
    ]

    timestamp = update_timestamp()
    raw_proposals = raw_proposals.copy()
    raw_interventions = raw_interventions.copy()
    raw_proposals["last_updated"] = timestamp
    raw_interventions["last_updated"] = timestamp

    scope_columns = ["site", "scenario", "year"]
    requested_scope = pd.DataFrame(
        [(site, scenario, year) for site in sites for scenario in scenarios for year in years],
        columns=scope_columns,
    )
    updated_sites = sorted(set(sites))

    for site in updated_sites:
        site_proposals = ensure_columns(
            raw_proposals[raw_proposals["site"] == site].copy(),
            proposal_raw_columns,
        )
        site_interventions = ensure_columns(
            raw_interventions[raw_interventions["site"] == site].copy(),
            intervention_raw_columns,
        )

        proposals_raw_path = refactor_site_raw_path(site, "proposals", output_mode=output_mode)
        interventions_raw_path = refactor_site_raw_path(site, "interventions", output_mode=output_mode)

        existing_site_proposals = read_csv_or_empty(proposals_raw_path, proposal_raw_columns)
        existing_site_interventions = read_csv_or_empty(interventions_raw_path, intervention_raw_columns)
        site_scope = requested_scope[requested_scope["site"] == site]

        upserted_site_proposals = replace_scope_rows(
            existing_site_proposals,
            site_proposals,
            site_scope,
            scope_columns,
            proposal_raw_columns,
        )
        upserted_site_interventions = replace_scope_rows(
            existing_site_interventions,
            site_interventions,
            site_scope,
            scope_columns,
            intervention_raw_columns,
        )

        upserted_site_proposals.to_csv(proposals_raw_path, index=False)
        upserted_site_interventions.to_csv(interventions_raw_path, index=False)
        print(f"Saved: {proposals_raw_path}")
        print(f"Saved: {interventions_raw_path}")

    legacy_raw_paths = [
        refactor_statistics_dir("raw", "proposals.csv", output_mode=output_mode),
        refactor_statistics_dir("raw", "interventions.csv", output_mode=output_mode),
    ]
    for legacy_path in legacy_raw_paths:
        if legacy_path.exists():
            legacy_path.unlink()

    proposal_keys = ["site", "year", "proposal_id", "proposal_label", "measure", "metric_column"]
    intervention_keys = [
        "site",
        "year",
        "proposal_id",
        "proposal_label",
        "intervention",
        "support",
        "measure",
        "metric_column",
    ]
    all_raw_proposals = collect_all_site_raw_tables("proposals", proposal_raw_columns, output_mode=output_mode)
    all_raw_interventions = collect_all_site_raw_tables("interventions", intervention_raw_columns, output_mode=output_mode)

    proposal_comparisons = build_comparison_table(all_raw_proposals, proposal_keys)
    intervention_comparisons = build_comparison_table(all_raw_interventions, intervention_keys)
    proposal_highlights = build_highlights_table(proposal_comparisons, proposal_keys)
    intervention_highlights = build_highlights_table(intervention_comparisons, intervention_keys)

    proposals_comparison_path = refactor_aggregate_path("comparison", "proposals", output_mode=output_mode)
    interventions_comparison_path = refactor_aggregate_path("comparison", "interventions", output_mode=output_mode)
    proposals_highlight_path = refactor_aggregate_path("highlights", "proposals", output_mode=output_mode)
    interventions_highlight_path = refactor_aggregate_path("highlights", "interventions", output_mode=output_mode)

    proposal_comparisons.to_csv(proposals_comparison_path, index=False)
    intervention_comparisons.to_csv(interventions_comparison_path, index=False)
    proposal_highlights.to_csv(proposals_highlight_path, index=False)
    intervention_highlights.to_csv(interventions_highlight_path, index=False)

    print(f"Saved: {proposals_comparison_path}")
    print(f"Saved: {interventions_comparison_path}")
    print(f"Saved: {proposals_highlight_path}")
    print(f"Saved: {interventions_highlight_path}")


def compute_tree_only_decay_counts(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float,
    output_mode: str | None,
) -> dict:
    tree_path = get_tree_path(site, scenario, year, voxel_size, output_mode)
    if not tree_path.exists():
        return {
            "site": site,
            "scenario": scenario,
            "year": year,
            "source_exists": False,
            "total_trees": pd.NA,
            "decay_tree_count": pd.NA,
            "brace_tree_count": pd.NA,
            "buffer_tree_count": pd.NA,
            "status": "missing_tree_csv",
            "notes": f"Missing file: {tree_path}",
        }

    tree_df = pd.read_csv(tree_path)
    required = ["action", "under-node-treatment", "isNewTree"]
    missing_cols = [c for c in required if c not in tree_df.columns]
    if missing_cols:
        return {
            "site": site,
            "scenario": scenario,
            "year": year,
            "source_exists": True,
            "total_trees": len(tree_df),
            "decay_tree_count": pd.NA,
            "brace_tree_count": pd.NA,
            "buffer_tree_count": pd.NA,
            "status": "missing_required_columns",
            "notes": "Missing columns: " + ",".join(missing_cols),
        }

    action = normalize_series(tree_df["action"])
    under_node_treatment = normalize_series(tree_df["under-node-treatment"])
    is_new_tree = bool_series(tree_df["isNewTree"])

    decay_tree_opp = (~is_new_tree) & action.isin(["AGE-IN-PLACE", "SENESCENT"])
    brace_mask = decay_tree_opp & (under_node_treatment == "exoskeleton")
    buffer_mask = decay_tree_opp & under_node_treatment.isin(["node-rewilded", "footprint-depaved"])

    return {
        "site": site,
        "scenario": scenario,
        "year": year,
        "source_exists": True,
        "total_trees": int(len(tree_df)),
        "decay_tree_count": int(decay_tree_opp.sum()),
        "brace_tree_count": int(brace_mask.sum()),
        "buffer_tree_count": int(buffer_mask.sum()),
        "status": "computed",
        "notes": "",
    }


def build_tree_only_decay_df(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float,
    output_mode: str | None,
) -> pd.DataFrame:
    rows = []
    for site in sites:
        for scenario in scenarios:
            for year in years:
                rows.append(compute_tree_only_decay_counts(site, scenario, year, voxel_size, output_mode))
    columns = [
        "site",
        "scenario",
        "year",
        "source_exists",
        "total_trees",
        "decay_tree_count",
        "brace_tree_count",
        "buffer_tree_count",
        "status",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def aggregate_all_sites_tree_only(stats_df: pd.DataFrame, scenarios: list[str], years: list[int]) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        for year in years:
            subset = stats_df[
                (stats_df["scenario"] == scenario) & (stats_df["year"] == year) & (stats_df["source_exists"] == True)
            ]
            if subset.empty:
                rows.append(
                    {
                        "site": "all-sites",
                        "scenario": scenario,
                        "year": year,
                        "source_exists": False,
                        "total_trees": pd.NA,
                        "decay_tree_count": pd.NA,
                        "brace_tree_count": pd.NA,
                        "buffer_tree_count": pd.NA,
                        "status": "missing_tree_csv",
                        "notes": "No site records available for this scenario/year.",
                    }
                )
                continue

            rows.append(
                {
                    "site": "all-sites",
                    "scenario": scenario,
                    "year": year,
                    "source_exists": True,
                    "total_trees": int(subset["total_trees"].fillna(0).sum()),
                    "decay_tree_count": int(subset["decay_tree_count"].fillna(0).sum()),
                    "brace_tree_count": int(subset["brace_tree_count"].fillna(0).sum()),
                    "buffer_tree_count": int(subset["buffer_tree_count"].fillna(0).sum()),
                    "status": "computed",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def format_ratio_cell(numerator, denominator, allow_na_zero: bool = True) -> str:
    if pd.isna(numerator) or pd.isna(denominator):
        return "NA"
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator == 0:
        if allow_na_zero:
            return f"{numerator}/{denominator} (NA)"
        return f"{numerator}/{denominator} (0.0%)"
    pct = numerator / denominator * 100.0
    return f"{numerator}/{denominator} ({pct:.1f}%)"


def sum_with_na(series: pd.Series) -> int | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return int(valid.astype(float).sum())


def format_count_cell(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{int(value):,}"


def format_count_pct_cell(numerator, denominator) -> str:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator):
        return "NA"
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator == 0:
        return f"{numerator:,} (NA)"
    pct = numerator / denominator * 100.0
    return f"{numerator:,} ({pct:.2f}%)"


def render_single_decay_table(title: str, subset: pd.DataFrame, years: list[int]) -> str:
    year_to_row = {int(row["year"]): row for _, row in subset.iterrows()}
    cells_decay = []
    cells_brace = []
    cells_buffer = []

    for year in years:
        row = year_to_row.get(year)
        if row is None:
            cells_decay.append("NA")
            cells_brace.append("NA")
            cells_buffer.append("NA")
            continue
        cells_decay.append(format_ratio_cell(row["decay_tree_count"], row["total_trees"], allow_na_zero=False))
        cells_brace.append(format_ratio_cell(row["brace_tree_count"], row["decay_tree_count"], allow_na_zero=True))
        cells_buffer.append(format_ratio_cell(row["buffer_tree_count"], row["decay_tree_count"], allow_na_zero=True))

    total_trees = sum_with_na(subset["total_trees"])
    total_decay = sum_with_na(subset["decay_tree_count"])
    total_brace = sum_with_na(subset["brace_tree_count"])
    total_buffer = sum_with_na(subset["buffer_tree_count"])

    decay_total_cell = (
        "NA"
        if total_trees is None or total_decay is None
        else format_ratio_cell(total_decay, total_trees, allow_na_zero=False)
    )
    brace_total_cell = (
        "NA"
        if total_decay is None or total_brace is None
        else format_ratio_cell(total_brace, total_decay, allow_na_zero=True)
    )
    buffer_total_cell = (
        "NA"
        if total_decay is None or total_buffer is None
        else format_ratio_cell(total_buffer, total_decay, allow_na_zero=True)
    )

    header = "| Metric | " + " | ".join(str(y) for y in years) + " | Total |"
    divider = "|---|" + "|".join(["---:"] * (len(years) + 1)) + "|"
    row_decay = (
        "| trees proposing decay / total trees | " + " | ".join(cells_decay) + f" | {decay_total_cell} |"
    )
    row_brace = (
        "| trees granted brace / total decay proposals | " + " | ".join(cells_brace) + f" | {brace_total_cell} |"
    )
    row_buffer = (
        "| trees granted buffer / total decay proposals | "
        + " | ".join(cells_buffer)
        + f" | {buffer_total_cell} |"
    )
    return "\n".join([title, header, divider, row_decay, row_brace, row_buffer, ""])


def generate_decay_comparison_tables(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float,
    output_mode: str | None,
):
    per_site_df = build_tree_only_decay_df(sites, scenarios, years, voxel_size, output_mode)
    all_sites_df = aggregate_all_sites_tree_only(per_site_df, scenarios, years)

    sections: list[str] = []
    for site in sites:
        site_label = SITE_REPORT_LABELS.get(site, site.upper())
        for scenario in scenarios:
            subset = per_site_df[(per_site_df["site"] == site) & (per_site_df["scenario"] == scenario)].sort_values("year")
            title = f"{site_label} - {scenario.upper()}"
            sections.append(render_single_decay_table(title, subset, years))

    for scenario in scenarios:
        subset = all_sites_df[all_sites_df["scenario"] == scenario].sort_values("year")
        title = f"ALL SITES - {scenario.upper()}"
        sections.append(render_single_decay_table(title, subset, years))

    markdown = "\n".join(sections).strip() + "\n"
    return per_site_df, all_sites_df, markdown


def save_decay_comparison_outputs(
    voxel_size: float,
    per_site_df: pd.DataFrame,
    all_sites_df: pd.DataFrame,
    markdown: str,
    output_mode: str | None,
):
    output_dir = get_output_dir(output_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel = format_voxel_size(voxel_size)

    by_site_path = output_dir / f"decay_tree_comparison_by_site_scenario_year_{voxel}.csv"
    all_sites_path = output_dir / f"decay_tree_comparison_all_sites_scenario_year_{voxel}.csv"
    markdown_path = output_dir / f"decay_tree_comparison_tables_{voxel}.md"

    per_site_df.to_csv(by_site_path, index=False)
    all_sites_df.to_csv(all_sites_path, index=False)
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved: {by_site_path}")
    print(f"Saved: {all_sites_path}")
    print(f"Saved: {markdown_path}")


def compute_release_control_voxel_counts(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float,
    output_mode: str | None,
) -> dict:
    vtk_path = get_vtk_path(site, scenario, year, voxel_size, output_mode)
    if not vtk_path.exists():
        return {
            "site": site,
            "scenario": scenario,
            "year": year,
            "source_exists": False,
            "all_canopy_voxel_count": pd.NA,
            "proposal_voxel_count": pd.NA,
            "reduce_pruning_voxel_count": pd.NA,
            "eliminate_pruning_voxel_count": pd.NA,
            "status": "missing_vtk",
            "notes": f"Missing file: {vtk_path}",
        }

    poly = pv.read(str(vtk_path))
    required_arrays = ["search_bioavailable", "forest_size", "forest_control"]
    missing_arrays = [a for a in required_arrays if a not in poly.point_data]
    if missing_arrays:
        return {
            "site": site,
            "scenario": scenario,
            "year": year,
            "source_exists": True,
            "all_canopy_voxel_count": pd.NA,
            "proposal_voxel_count": pd.NA,
            "reduce_pruning_voxel_count": pd.NA,
            "eliminate_pruning_voxel_count": pd.NA,
            "status": "missing_required_arrays",
            "notes": "Missing arrays: " + ",".join(missing_arrays),
        }

    search_bioavailable = vtk_str_array(poly.point_data["search_bioavailable"])
    forest_control = vtk_str_array(poly.point_data["forest_control"])

    # Proposal = all arboreal voxels.
    search_bioavailable_lower = np.char.lower(search_bioavailable)
    forest_control_lower = np.char.lower(forest_control)

    canopy_mask = search_bioavailable_lower == "arboreal"
    proposal_mask = canopy_mask
    reduce_mask = proposal_mask & np.isin(forest_control_lower, ["park-tree", "park tree"])
    eliminate_mask = proposal_mask & np.isin(
        forest_control_lower,
        ["reserve-tree", "reserve tree", "improved-tree", "improved tree"],
    )

    return {
        "site": site,
        "scenario": scenario,
        "year": year,
        "source_exists": True,
        "all_canopy_voxel_count": int(np.sum(canopy_mask)),
        "proposal_voxel_count": int(np.sum(proposal_mask)),
        "reduce_pruning_voxel_count": int(np.sum(reduce_mask)),
        "eliminate_pruning_voxel_count": int(np.sum(eliminate_mask)),
        "status": "computed",
        "notes": "",
    }


def build_release_control_voxel_df(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float,
    output_mode: str | None,
) -> pd.DataFrame:
    rows = []
    for site in sites:
        for scenario in scenarios:
            for year in years:
                rows.append(compute_release_control_voxel_counts(site, scenario, year, voxel_size, output_mode))
    columns = [
        "site",
        "scenario",
        "year",
        "source_exists",
        "all_canopy_voxel_count",
        "proposal_voxel_count",
        "reduce_pruning_voxel_count",
        "eliminate_pruning_voxel_count",
        "status",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def aggregate_all_sites_release_control_voxels(
    stats_df: pd.DataFrame,
    scenarios: list[str],
    years: list[int],
) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        for year in years:
            subset = stats_df[
                (stats_df["scenario"] == scenario) & (stats_df["year"] == year) & (stats_df["source_exists"] == True)
            ]
            if subset.empty:
                rows.append(
                    {
                        "site": "all-sites",
                        "scenario": scenario,
                        "year": year,
                        "source_exists": False,
                        "all_canopy_voxel_count": pd.NA,
                        "proposal_voxel_count": pd.NA,
                        "reduce_pruning_voxel_count": pd.NA,
                        "eliminate_pruning_voxel_count": pd.NA,
                        "status": "missing_vtk",
                        "notes": "No site records available for this scenario/year.",
                    }
                )
                continue

            rows.append(
                {
                    "site": "all-sites",
                    "scenario": scenario,
                    "year": year,
                    "source_exists": True,
                    "all_canopy_voxel_count": int(subset["all_canopy_voxel_count"].fillna(0).sum()),
                    "proposal_voxel_count": int(subset["proposal_voxel_count"].fillna(0).sum()),
                    "reduce_pruning_voxel_count": int(subset["reduce_pruning_voxel_count"].fillna(0).sum()),
                    "eliminate_pruning_voxel_count": int(subset["eliminate_pruning_voxel_count"].fillna(0).sum()),
                    "status": "computed",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def render_single_release_control_table(title: str, subset: pd.DataFrame, years: list[int]) -> str:
    year_to_row = {int(row["year"]): row for _, row in subset.iterrows()}
    cells_proposal = []
    cells_reduce = []
    cells_eliminate = []

    for year in years:
        row = year_to_row.get(year)
        if row is None:
            cells_proposal.append("NA")
            cells_reduce.append("NA")
            cells_eliminate.append("NA")
            continue
        cells_proposal.append(
            format_ratio_cell(row["proposal_voxel_count"], row["all_canopy_voxel_count"], allow_na_zero=False)
        )
        cells_reduce.append(
            format_ratio_cell(row["reduce_pruning_voxel_count"], row["proposal_voxel_count"], allow_na_zero=True)
        )
        cells_eliminate.append(
            format_ratio_cell(row["eliminate_pruning_voxel_count"], row["proposal_voxel_count"], allow_na_zero=True)
        )

    total_canopy = sum_with_na(subset["all_canopy_voxel_count"])
    total_proposal = sum_with_na(subset["proposal_voxel_count"])
    total_reduce = sum_with_na(subset["reduce_pruning_voxel_count"])
    total_eliminate = sum_with_na(subset["eliminate_pruning_voxel_count"])

    proposal_total_cell = (
        "NA"
        if total_canopy is None or total_proposal is None
        else format_ratio_cell(total_proposal, total_canopy, allow_na_zero=False)
    )
    reduce_total_cell = (
        "NA"
        if total_proposal is None or total_reduce is None
        else format_ratio_cell(total_reduce, total_proposal, allow_na_zero=True)
    )
    eliminate_total_cell = (
        "NA"
        if total_proposal is None or total_eliminate is None
        else format_ratio_cell(total_eliminate, total_proposal, allow_na_zero=True)
    )

    header = "| Metric | " + " | ".join(str(y) for y in years) + " | Total |"
    divider = "|---|" + "|".join(["---:"] * (len(years) + 1)) + "|"
    row_proposal = (
        "| canopy voxels proposing release control / all canopy voxels | "
        + " | ".join(cells_proposal)
        + f" | {proposal_total_cell} |"
    )
    row_reduce = (
        "| canopy voxels granted reduce pruning / total release-control proposals | "
        + " | ".join(cells_reduce)
        + f" | {reduce_total_cell} |"
    )
    row_eliminate = (
        "| canopy voxels granted eliminate pruning / total release-control proposals | "
        + " | ".join(cells_eliminate)
        + f" | {eliminate_total_cell} |"
    )
    return "\n".join([title, header, divider, row_proposal, row_reduce, row_eliminate, ""])


def generate_release_control_comparison_tables(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float,
    output_mode: str | None,
):
    per_site_df = build_release_control_voxel_df(sites, scenarios, years, voxel_size, output_mode)
    all_sites_df = aggregate_all_sites_release_control_voxels(per_site_df, scenarios, years)

    sections: list[str] = []
    for site in sites:
        site_label = SITE_REPORT_LABELS.get(site, site.upper())
        for scenario in scenarios:
            subset = per_site_df[(per_site_df["site"] == site) & (per_site_df["scenario"] == scenario)].sort_values("year")
            title = f"{site_label} - {scenario.upper()}"
            sections.append(render_single_release_control_table(title, subset, years))

    for scenario in scenarios:
        subset = all_sites_df[all_sites_df["scenario"] == scenario].sort_values("year")
        title = f"ALL SITES - {scenario.upper()}"
        sections.append(render_single_release_control_table(title, subset, years))

    markdown = "\n".join(sections).strip() + "\n"
    return per_site_df, all_sites_df, markdown


def save_release_control_comparison_outputs(
    voxel_size: float,
    per_site_df: pd.DataFrame,
    all_sites_df: pd.DataFrame,
    markdown: str,
    output_mode: str | None,
):
    output_dir = get_output_dir(output_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel = format_voxel_size(voxel_size)

    by_site_path = output_dir / f"release_control_voxel_comparison_by_site_scenario_year_{voxel}.csv"
    all_sites_path = output_dir / f"release_control_voxel_comparison_all_sites_scenario_year_{voxel}.csv"
    markdown_path = output_dir / f"release_control_voxel_comparison_tables_{voxel}.md"

    per_site_df.to_csv(by_site_path, index=False)
    all_sites_df.to_csv(all_sites_path, index=False)
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved: {by_site_path}")
    print(f"Saved: {all_sites_path}")
    print(f"Saved: {markdown_path}")


def compute_connect_voxel_counts(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float,
    output_mode: str | None,
) -> dict:
    vtk_path = get_vtk_path(site, scenario, year, voxel_size, output_mode)
    if not vtk_path.exists():
        return {
            "site": site,
            "scenario": scenario,
            "year": year,
            "source_exists": False,
            "proposal_voxel_count": pd.NA,
            "rewild_ground_voxel_count": pd.NA,
            "enrich_envelope_voxel_count": pd.NA,
            "roughen_envelope_voxel_count": pd.NA,
            "status": "missing_vtk",
            "notes": f"Missing file: {vtk_path}",
        }

    poly = pv.read(str(vtk_path))
    if "scenario_outputs" not in poly.point_data:
        return {
            "site": site,
            "scenario": scenario,
            "year": year,
            "source_exists": True,
            "proposal_voxel_count": pd.NA,
            "rewild_ground_voxel_count": pd.NA,
            "enrich_envelope_voxel_count": pd.NA,
            "roughen_envelope_voxel_count": pd.NA,
            "status": "missing_required_arrays",
            "notes": "Missing arrays: scenario_outputs",
        }

    scenario_outputs = vtk_str_array(poly.point_data["scenario_outputs"])
    scenario_outputs_lower = np.char.lower(scenario_outputs)

    proposal_mask = np.isin(scenario_outputs_lower, list(COLONISE_PROPOSAL_VALUES))
    rewild_mask = proposal_mask & np.isin(scenario_outputs_lower, list(COLONISE_REWILD_VALUES))
    enrich_mask = proposal_mask & np.isin(scenario_outputs_lower, list(COLONISE_ENRICH_VALUES))
    roughen_mask = proposal_mask & np.isin(scenario_outputs_lower, list(COLONISE_ROUGHEN_VALUES))

    return {
        "site": site,
        "scenario": scenario,
        "year": year,
        "source_exists": True,
        "proposal_voxel_count": int(np.sum(proposal_mask)),
        "rewild_ground_voxel_count": int(np.sum(rewild_mask)),
        "enrich_envelope_voxel_count": int(np.sum(enrich_mask)),
        "roughen_envelope_voxel_count": int(np.sum(roughen_mask)),
        "status": "computed",
        "notes": "",
    }


def build_connect_voxel_df(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float,
    output_mode: str | None,
) -> pd.DataFrame:
    rows = []
    for site in sites:
        for scenario in scenarios:
            for year in years:
                rows.append(compute_connect_voxel_counts(site, scenario, year, voxel_size, output_mode))
    columns = [
        "site",
        "scenario",
        "year",
        "source_exists",
        "proposal_voxel_count",
        "rewild_ground_voxel_count",
        "enrich_envelope_voxel_count",
        "roughen_envelope_voxel_count",
        "status",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def aggregate_all_sites_connect_voxels(
    stats_df: pd.DataFrame,
    scenarios: list[str],
    years: list[int],
) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        for year in years:
            subset = stats_df[
                (stats_df["scenario"] == scenario) & (stats_df["year"] == year) & (stats_df["source_exists"] == True)
            ]
            if subset.empty:
                rows.append(
                    {
                        "site": "all-sites",
                        "scenario": scenario,
                        "year": year,
                        "source_exists": False,
                        "proposal_voxel_count": pd.NA,
                        "rewild_ground_voxel_count": pd.NA,
                        "enrich_envelope_voxel_count": pd.NA,
                        "roughen_envelope_voxel_count": pd.NA,
                        "status": "missing_vtk",
                        "notes": "No site records available for this scenario/year.",
                    }
                )
                continue

            rows.append(
                {
                    "site": "all-sites",
                    "scenario": scenario,
                    "year": year,
                    "source_exists": True,
                    "proposal_voxel_count": int(subset["proposal_voxel_count"].fillna(0).sum()),
                    "rewild_ground_voxel_count": int(subset["rewild_ground_voxel_count"].fillna(0).sum()),
                    "enrich_envelope_voxel_count": int(subset["enrich_envelope_voxel_count"].fillna(0).sum()),
                    "roughen_envelope_voxel_count": int(subset["roughen_envelope_voxel_count"].fillna(0).sum()),
                    "status": "computed",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def render_single_connect_table(title: str, subset: pd.DataFrame, years: list[int]) -> str:
    year_to_row = {int(row["year"]): row for _, row in subset.iterrows()}
    proposal_cells = []
    rewild_cells = []
    enrich_cells = []
    roughen_cells = []

    for year in years:
        row = year_to_row.get(year)
        if row is None:
            proposal_cells.append("NA")
            rewild_cells.append("NA")
            enrich_cells.append("NA")
            roughen_cells.append("NA")
            continue
        proposal_cells.append(format_count_cell(row["proposal_voxel_count"]))
        rewild_cells.append(format_count_pct_cell(row["rewild_ground_voxel_count"], row["proposal_voxel_count"]))
        enrich_cells.append(format_count_pct_cell(row["enrich_envelope_voxel_count"], row["proposal_voxel_count"]))
        roughen_cells.append(format_count_pct_cell(row["roughen_envelope_voxel_count"], row["proposal_voxel_count"]))

    total_proposal = sum_with_na(subset["proposal_voxel_count"])
    total_rewild = sum_with_na(subset["rewild_ground_voxel_count"])
    total_enrich = sum_with_na(subset["enrich_envelope_voxel_count"])
    total_roughen = sum_with_na(subset["roughen_envelope_voxel_count"])

    proposal_total_cell = format_count_cell(total_proposal)
    rewild_total_cell = format_count_pct_cell(total_rewild, total_proposal)
    enrich_total_cell = format_count_pct_cell(total_enrich, total_proposal)
    roughen_total_cell = format_count_pct_cell(total_roughen, total_proposal)

    header = "| Metric | " + " | ".join(str(y) for y in years) + " | Total |"
    divider = "|---|" + "|".join(["---:"] * (len(years) + 1)) + "|"
    row_proposal = "| Proposal voxels [x] | " + " | ".join(proposal_cells) + f" | {proposal_total_cell} |"
    row_rewild = "| Rewild-Ground | " + " | ".join(rewild_cells) + f" | {rewild_total_cell} |"
    row_enrich = "| Enrich-Envelope | " + " | ".join(enrich_cells) + f" | {enrich_total_cell} |"
    row_roughen = "| Roughen-Envelope | " + " | ".join(roughen_cells) + f" | {roughen_total_cell} |"
    return "\n".join([title, header, divider, row_proposal, row_rewild, row_enrich, row_roughen, ""])


def generate_connect_comparison_tables(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float,
    output_mode: str | None,
):
    per_site_df = build_connect_voxel_df(sites, scenarios, years, voxel_size, output_mode)
    all_sites_df = aggregate_all_sites_connect_voxels(per_site_df, scenarios, years)

    sections: list[str] = []
    for site in sites:
        site_label = SITE_REPORT_LABELS.get(site, site.upper())
        for scenario in scenarios:
            subset = per_site_df[(per_site_df["site"] == site) & (per_site_df["scenario"] == scenario)].sort_values("year")
            title = f"{site_label} - {scenario.upper()}"
            sections.append(render_single_connect_table(title, subset, years))

    for scenario in scenarios:
        subset = all_sites_df[all_sites_df["scenario"] == scenario].sort_values("year")
        title = f"ALL SITES - {scenario.upper()}"
        sections.append(render_single_connect_table(title, subset, years))

    markdown = "\n".join(sections).strip() + "\n"
    return per_site_df, all_sites_df, markdown


def save_connect_comparison_outputs(
    voxel_size: float,
    per_site_df: pd.DataFrame,
    all_sites_df: pd.DataFrame,
    markdown: str,
    output_mode: str | None,
):
    output_dir = get_output_dir(output_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel = format_voxel_size(voxel_size)

    by_site_path = output_dir / f"connect_voxel_comparison_by_site_scenario_year_{voxel}.csv"
    all_sites_path = output_dir / f"connect_voxel_comparison_all_sites_scenario_year_{voxel}.csv"
    markdown_path = output_dir / f"connect_voxel_comparison_tables_{voxel}.md"

    per_site_df.to_csv(by_site_path, index=False)
    all_sites_df.to_csv(all_sites_path, index=False)
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved: {by_site_path}")
    print(f"Saved: {all_sites_path}")
    print(f"Saved: {markdown_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute proposal/intervention metrics from scenario outputs.")
    parser.add_argument("--site", type=str, default="all", help='Site name or "all"')
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name (e.g. positive, trending)")
    parser.add_argument("--year", type=int, default=None, help="Single year/timestep")
    parser.add_argument("--voxel-size", type=float, default=1)
    parser.add_argument("--proposal", type=str, default=None, help="Proposal filter")
    parser.add_argument("--intervention", type=str, default=None, help="Intervention filter")
    parser.add_argument("--include-stubs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-combine", action="store_true", help="Skip all-sites combine output")
    parser.add_argument(
        "--output-mode",
        type=str,
        default=None,
        choices=["canonical", "validation"],
        help="Scenario/engine output root mode for reads and CSV writes.",
    )
    parser.add_argument(
        "--decay-comparison-tables",
        action="store_true",
        help=(
            "Generate tree-only Decay/Brace/Buffer comparison tables "
            "(site-scenario and all-sites totals) and exit."
        ),
    )
    parser.add_argument(
        "--table-years",
        type=str,
        default="0,10,30,60,90,120,150,180",
        help='Comma-separated years for decay comparison tables (default "0,10,30,60,90,120,150,180").',
    )
    parser.add_argument(
        "--release-control-comparison-tables",
        action="store_true",
        help=(
            "Generate VTK-level Release Control comparison tables "
            "(site-scenario and all-sites totals) and exit."
        ),
    )
    parser.add_argument(
        "--connect-comparison-tables",
        action="store_true",
        help=(
            "Generate VTK-level Connect comparison tables from scenario_outputs "
            "(site-scenario and all-sites totals) and exit."
        ),
    )
    parser.add_argument(
        "--export-refactor-statistics",
        action="store_true",
        help=(
            "Upsert per-site long-format raw proposal/intervention tables with last_updated, "
            "then rebuild aggregate comparison and highlights tables in "
            "_statistics-refactored."
        ),
    )

    args = parser.parse_args()
    output_mode = normalize_output_mode(args.output_mode)

    if args.decay_comparison_tables:
        report_sites = SITE_REPORT_ORDER if args.site == "all" else [args.site]
        scenarios = [args.scenario] if args.scenario else list(DEFAULT_SCENARIOS)
        years = parse_years_arg(args.table_years)
        per_site_df, all_sites_df, markdown = generate_decay_comparison_tables(
            sites=report_sites,
            scenarios=scenarios,
            years=years,
            voxel_size=args.voxel_size,
            output_mode=output_mode,
        )
        save_decay_comparison_outputs(args.voxel_size, per_site_df, all_sites_df, markdown, output_mode)
        print("\n" + markdown)
        return

    if args.release_control_comparison_tables:
        report_sites = SITE_REPORT_ORDER if args.site == "all" else [args.site]
        scenarios = [args.scenario] if args.scenario else list(DEFAULT_SCENARIOS)
        years = parse_years_arg(args.table_years)
        per_site_df, all_sites_df, markdown = generate_release_control_comparison_tables(
            sites=report_sites,
            scenarios=scenarios,
            years=years,
            voxel_size=args.voxel_size,
            output_mode=output_mode,
        )
        save_release_control_comparison_outputs(args.voxel_size, per_site_df, all_sites_df, markdown, output_mode)
        print("\n" + markdown)
        return

    if args.connect_comparison_tables:
        report_sites = SITE_REPORT_ORDER if args.site == "all" else [args.site]
        scenarios = [args.scenario] if args.scenario else list(DEFAULT_SCENARIOS)
        years = parse_years_arg(args.table_years)
        per_site_df, all_sites_df, markdown = generate_connect_comparison_tables(
            sites=report_sites,
            scenarios=scenarios,
            years=years,
            voxel_size=args.voxel_size,
            output_mode=output_mode,
        )
        save_connect_comparison_outputs(args.voxel_size, per_site_df, all_sites_df, markdown, output_mode)
        print("\n" + markdown)
        return

    if args.export_refactor_statistics:
        report_sites = SITE_REPORT_ORDER if args.site == "all" else [args.site]
        scenarios = [args.scenario] if args.scenario else list(DEFAULT_SCENARIOS)
        years = parse_years_arg(args.table_years)
        raw_proposals, raw_interventions = build_refactor_raw_tables(
            sites=report_sites,
            scenarios=scenarios,
            years=years,
            voxel_size=args.voxel_size,
            output_mode=output_mode,
        )
        save_refactor_statistics_exports(
            raw_proposals,
            raw_interventions,
            sites=report_sites,
            scenarios=scenarios,
            years=years,
            output_mode=output_mode,
        )
        return

    sites = SITES if args.site == "all" else [args.site]

    proposal_filter = canonical_proposal(args.proposal)
    intervention_filter = canonical_intervention(args.intervention)

    for site in sites:
        print(f"\nProcessing {site} ...")
        opportunity_df, intervention_df, qc_df = process_site(
            site=site,
            scenario=args.scenario,
            year=args.year,
            voxel_size=args.voxel_size,
            proposal_filter=proposal_filter,
            intervention_filter=intervention_filter,
            include_stubs=args.include_stubs,
            output_mode=output_mode,
        )
        save_site_outputs(site, args.voxel_size, opportunity_df, intervention_df, qc_df, output_mode)
        print(
            f"  Rows -> opportunities: {len(opportunity_df)}, interventions: {len(intervention_df)}, qc: {len(qc_df)}"
        )

    if not args.no_combine:
        combine_all_sites(args.voxel_size, output_mode)


if __name__ == "__main__":
    main()
