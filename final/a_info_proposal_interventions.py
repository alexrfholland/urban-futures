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
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree


SITES = ["trimmed-parade", "city", "uni"]
DEFAULT_SCENARIOS = ["positive", "trending"]
OUTPUT_DIR = Path("data/revised/final/output/csv")
DATA_DIR = Path("data/revised/final")
INDICATOR_VTK_DIR = DATA_DIR / "output"
SITE_REPORT_LABELS = {
    "trimmed-parade": "PARADE",
    "uni": "STREET",
    "city": "CITY",
}
SITE_REPORT_ORDER = ["trimmed-parade", "uni", "city"]
DEFAULT_TABLE_YEARS = [0, 10, 30, 60, 90, 120, 150, 180]

COLONISE_DISTANCE_M = 5.0
EXCLUDED_URBAN_VALUES = {
    "open space",
    "roadway",
    "busy roadway",
    "parking",
    "other street potential",
}

CONNECT_PROPOSAL_VALUES = {
    "brownroof",
    "greenroof",
    "livingfacade",
    "footprint-depaved",
    "node-rewilded",
    "otherground",
    "rewilded",
}
CONNECT_FULL_VALUES = {"greenroof", "node-rewilded", "rewilded"}
CONNECT_PARTIAL_VALUES = {"brownroof", "footprint-depaved", "livingfacade"}

PROPOSAL_IDS = [
    "decay",
    "release_control",
    "colonise",
    "recruit",
    "deploy",
    "translocate",
]

PROPOSAL_LABELS = {
    "decay": "Decay",
    "release_control": "Release Control",
    "colonise": "Colonise",
    "recruit": "Recruit",
    "deploy": "Deploy",
    "translocate": "Translocate",
}

PROPOSAL_ALIASES = {
    "decay": "decay",
    "release control": "release_control",
    "release_control": "release_control",
    "release-control": "release_control",
    "colonise": "colonise",
    "colonize": "colonise",
    "recruit": "recruit",
    "deploy": "deploy",
    "translocate": "translocate",
}

INTERVENTION_ALIASES = {
    "buffer": "Buffer",
    "brace": "Brace",
    "eliminate pruning": "Eliminate pruning",
    "reduce pruning": "Reduce pruning",
    "connect (green envelopes)": "Connect (green envelopes)",
    "depave": "Depave",
    "connect (brown envelopes)": "Connect (brown envelopes)",
    "connect (full)": "Connect (full)",
    "connect full": "Connect (full)",
    "connect (partial)": "Connect (partial)",
    "connect partial": "Connect (partial)",
    "stub": "Stub",
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
    "rewilded",
    "isNewTree",
    "precolonial",
]

REQUIRED_VTK_ARRAYS = [
    "search_bioavailable",
    "search_urban_elements",
    "search_design_action",
    "scenario_outputs",
    "scenario_rewilded",
    "scenario_bioEnvelope",
    "forest_control",
    "forest_precolonial",
]

# -----------------------------------------------------------------------------
# INTERPRETATION NOTES (proposal -> intervention mapping)
# -----------------------------------------------------------------------------
# This script does not infer proposals/interventions from text. It maps them
# directly to existing model fields so each metric is reproducible:
#
# Decay opportunity:
#   tree_df.isNewTree == False AND tree_df.action in {'AGE-IN-PLACE','SENESCENT'}
# Decay partial support (Buffer):
#   tree_df.rewilded in {'node-rewilded','footprint-depaved'}
#   vtk.scenario_rewilded in {'node-rewilded','footprint-depaved','rewilded'}
# Decay full support (Brace):
#   tree_df.rewilded == 'exoskeleton'
#   vtk.scenario_rewilded == 'exoskeleton'
#
# Release Control opportunity:
#   vtk.search_bioavailable == 'arboreal'
# Release Control full support (Eliminate pruning):
#   vtk.forest_control in {'reserve-tree','improved-tree'}
# Release Control partial support (Reduce pruning):
#   vtk.forest_control == 'park-tree'
#
# Connect (stored under proposal_id='colonise') opportunity:
#   vtk.scenario_outputs in {
#       'brownRoof','greenRoof','livingFacade','footprint-depaved',
#       'node-rewilded','otherGround','rewilded'
#   }
# Connect full support:
#   vtk.scenario_outputs in {'greenRoof','node-rewilded','rewilded'}
# Connect partial support:
#   vtk.scenario_outputs in {'brownRoof','footprint-depaved','livingFacade'}
#
# Recruit / Deploy / Translocate:
#   Explicit stub rows in this pass (status='stub').
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
    return {
        "site": site,
        "scenario": scenario,
        "year": year,
        "proposal_id": proposal_id,
        "support_level": support_level,
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


def get_tree_path(site: str, scenario: str, year: int, voxel_size: float) -> Path:
    voxel = format_voxel_size(voxel_size)
    return DATA_DIR / site / f"{site}_{scenario}_{voxel}_treeDF_{year}.csv"


def get_vtk_path(site: str, scenario: str, year: int, voxel_size: float) -> Path:
    voxel = format_voxel_size(voxel_size)
    site_base = DATA_DIR / site
    output_base = INDICATOR_VTK_DIR
    candidates = [
        # Prefer latest indicator-enriched output VTKs.
        output_base / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features_with_indicators.vtk",
        site_base / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features_with_indicators.vtk",
        # Backward compatibility fallbacks.
        output_base / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features.vtk",
        site_base / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features.vtk",
        site_base / f"{site}_{voxel}_{scenario}_scenarioYR{year}_urban_features.vtk",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def discover_scenarios(site: str, voxel_size: float) -> list[str]:
    voxel = format_voxel_size(voxel_size)
    base = DATA_DIR / site
    if not base.exists():
        return []
    pattern = re.compile(rf"^{re.escape(site)}_(.+)_{re.escape(voxel)}_treeDF_(\d+)\.csv$")
    scenarios: set[str] = set()
    for path in base.glob(f"{site}_*_{voxel}_treeDF_*.csv"):
        match = pattern.match(path.name)
        if match:
            scenarios.add(match.group(1))
    return sorted(scenarios)


def discover_years(site: str, scenario: str, voxel_size: float) -> list[int]:
    voxel = format_voxel_size(voxel_size)
    base = DATA_DIR / site
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


def build_excluded_urban_mask(urban_values: np.ndarray) -> np.ndarray:
    lower = np.char.lower(urban_values.astype(str))
    base_mask = np.isin(lower, list(EXCLUDED_URBAN_VALUES))
    truncated_mask = np.char.startswith(lower, "other street potenti")
    return base_mask | truncated_mask


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
):
    opportunity_rows = []
    intervention_rows = []
    qc_rows = []

    tree_path = get_tree_path(site, scenario, year, voxel_size)
    vtk_path = get_vtk_path(site, scenario, year, voxel_size)

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

    action = normalize_series(tree_df["action"])
    rewilded = normalize_series(tree_df["rewilded"])
    is_new_tree = bool_series(tree_df["isNewTree"])

    scenario_rewilded = vtk_str_array(poly.point_data["scenario_rewilded"])
    scenario_bio_envelope = vtk_str_array(poly.point_data["scenario_bioEnvelope"])
    scenario_outputs = vtk_str_array(poly.point_data["scenario_outputs"])
    search_bioavailable = vtk_str_array(poly.point_data["search_bioavailable"])
    search_urban_elements = vtk_str_array(poly.point_data["search_urban_elements"])
    forest_control = vtk_str_array(poly.point_data["forest_control"])

    # ------------------------------------------------------------------
    # DECAY
    # ------------------------------------------------------------------
    if should_include_proposal("decay", proposal_filter):
        decay_tree_opp = (~is_new_tree) & action.isin(["AGE-IN-PLACE", "SENESCENT"])
        decay_tree_count = int(decay_tree_opp.sum())
        decay_voxel_opp = np.isin(
            scenario_rewilded,
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
                notes="Opportunity trees: non-new trees with action AGE-IN-PLACE or SENESCENT.",
            )
        )

        if should_include_intervention("Buffer", intervention_filter):
            tree_mask = decay_tree_opp & rewilded.isin(["node-rewilded", "footprint-depaved"])
            voxel_mask = np.isin(scenario_rewilded, ["node-rewilded", "footprint-depaved", "rewilded"])
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "decay",
                    "partial",
                    "Buffer",
                    int(tree_mask.sum()),
                    int(np.sum(voxel_mask)),
                    decay_tree_count,
                    decay_voxel_count,
                    status="computed",
                    notes="Partial support mapped to node/footprint/generic rewilded.",
                )
            )

        if should_include_intervention("Brace", intervention_filter):
            tree_mask = decay_tree_opp & (rewilded == "exoskeleton")
            voxel_mask = scenario_rewilded == "exoskeleton"
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "decay",
                    "full",
                    "Brace",
                    int(tree_mask.sum()),
                    int(np.sum(voxel_mask)),
                    decay_tree_count,
                    decay_voxel_count,
                    status="computed",
                    notes="Full support mapped to exoskeleton.",
                )
            )

    # ------------------------------------------------------------------
    # RELEASE CONTROL
    # ------------------------------------------------------------------
    if should_include_proposal("release_control", proposal_filter):
        search_bioavailable_lower = np.char.lower(search_bioavailable)
        forest_control_lower = np.char.lower(forest_control)
        release_voxel_opp = search_bioavailable_lower == "arboreal"

        # Release control is voxel-defined in this framing.
        release_tree_opp = pd.Series([False] * len(tree_df), index=tree_df.index)
        release_tree_count = 0
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
                notes="Opportunity voxels: all arboreal voxels.",
            )
        )

        if should_include_intervention("Eliminate pruning", intervention_filter):
            tree_mask = release_tree_opp
            voxel_mask = release_voxel_opp & np.isin(
                forest_control_lower, ["reserve-tree", "reserve tree", "improved-tree", "improved tree"]
            )
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "release_control",
                    "full",
                    "Eliminate pruning",
                    int(tree_mask.sum()),
                    int(np.sum(voxel_mask)),
                    release_tree_count,
                    release_voxel_count,
                    status="computed",
                    notes="Full support where forest_control is reserve-tree or improved-tree.",
                )
            )

        if should_include_intervention("Reduce pruning", intervention_filter):
            tree_mask = release_tree_opp
            voxel_mask = release_voxel_opp & np.isin(forest_control_lower, ["park-tree", "park tree"])
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "release_control",
                    "partial",
                    "Reduce pruning",
                    int(tree_mask.sum()),
                    int(np.sum(voxel_mask)),
                    release_tree_count,
                    release_voxel_count,
                    status="computed",
                    notes="Partial support where forest_control is park-tree.",
                )
            )

    # ------------------------------------------------------------------
    # COLONISE
    # ------------------------------------------------------------------
    if should_include_proposal("colonise", proposal_filter):
        scenario_outputs_lower = np.char.lower(scenario_outputs)
        colonise_voxel_opp = np.isin(scenario_outputs_lower, list(CONNECT_PROPOSAL_VALUES))

        colonise_tree_count = 0
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
                notes=(
                    "Connect proposal voxels from scenario_outputs in "
                    "{brownRoof,greenRoof,livingFacade,footprint-depaved,node-rewilded,otherGround,rewilded}."
                ),
            )
        )

        if should_include_intervention("Connect (full)", intervention_filter):
            voxel_mask = colonise_voxel_opp & np.isin(scenario_outputs_lower, list(CONNECT_FULL_VALUES))
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "colonise",
                    "full",
                    "Connect (full)",
                    0,
                    int(np.sum(voxel_mask)),
                    colonise_tree_count,
                    colonise_voxel_count,
                    status="computed",
                    notes="Full support where scenario_outputs is greenRoof, node-rewilded, or rewilded.",
                )
            )

        if should_include_intervention("Connect (partial)", intervention_filter):
            voxel_mask = colonise_voxel_opp & np.isin(scenario_outputs_lower, list(CONNECT_PARTIAL_VALUES))
            intervention_rows.append(
                intervention_row(
                    site,
                    scenario,
                    year,
                    "colonise",
                    "partial",
                    "Connect (partial)",
                    0,
                    int(np.sum(voxel_mask)),
                    colonise_tree_count,
                    colonise_voxel_count,
                    status="computed",
                    notes="Partial support where scenario_outputs is brownRoof, footprint-depaved, or livingFacade.",
                )
            )

    # ------------------------------------------------------------------
    # STUBS
    # ------------------------------------------------------------------
    if include_stubs:
        for proposal_id in ["recruit", "deploy", "translocate"]:
            if not should_include_proposal(proposal_id, proposal_filter):
                continue
            opportunity_rows.append(
                opportunity_row(
                    site,
                    scenario,
                    year,
                    proposal_id,
                    pd.NA,
                    pd.NA,
                    status="stub",
                    notes="Stub placeholder; metric definition deferred.",
                )
            )
            if should_include_intervention("Stub", intervention_filter):
                intervention_rows.append(
                    intervention_row(
                        site,
                        scenario,
                        year,
                        proposal_id,
                        "stub",
                        "Stub",
                        pd.NA,
                        pd.NA,
                        pd.NA,
                        pd.NA,
                        status="stub",
                        notes="Stub placeholder; intervention mapping deferred.",
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
):
    scenarios = [scenario] if scenario else discover_scenarios(site, voxel_size)
    if not scenarios:
        scenarios = DEFAULT_SCENARIOS

    site_opportunity_rows = []
    site_intervention_rows = []
    site_qc_rows = []

    for current_scenario in scenarios:
        years = [year] if year is not None else discover_years(site, current_scenario, voxel_size)
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
            )
            site_opportunity_rows.extend(opp_rows)
            site_intervention_rows.extend(int_rows)
            site_qc_rows.extend(qc_rows)

    opportunity_df = ensure_columns(pd.DataFrame(site_opportunity_rows), OPPORTUNITY_COLUMNS)
    intervention_df = ensure_columns(pd.DataFrame(site_intervention_rows), INTERVENTION_COLUMNS)
    qc_df = ensure_columns(pd.DataFrame(site_qc_rows), QC_COLUMNS)

    return opportunity_df, intervention_df, qc_df


def save_site_outputs(site: str, voxel_size: float, opportunity_df: pd.DataFrame, intervention_df: pd.DataFrame, qc_df: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voxel = format_voxel_size(voxel_size)
    opportunity_path = OUTPUT_DIR / f"{site}_{voxel}_proposal_opportunities.csv"
    intervention_path = OUTPUT_DIR / f"{site}_{voxel}_proposal_interventions.csv"
    qc_path = OUTPUT_DIR / f"{site}_{voxel}_proposal_qc.csv"

    opportunity_df.to_csv(opportunity_path, index=False)
    intervention_df.to_csv(intervention_path, index=False)
    qc_df.to_csv(qc_path, index=False)

    print(f"Saved: {opportunity_path}")
    print(f"Saved: {intervention_path}")
    print(f"Saved: {qc_path}")


def combine_all_sites(voxel_size: float):
    voxel = format_voxel_size(voxel_size)
    all_opp = []
    all_int = []
    for site in SITES:
        opp_path = OUTPUT_DIR / f"{site}_{voxel}_proposal_opportunities.csv"
        int_path = OUTPUT_DIR / f"{site}_{voxel}_proposal_interventions.csv"
        if opp_path.exists():
            all_opp.append(pd.read_csv(opp_path))
        if int_path.exists():
            all_int.append(pd.read_csv(int_path))

    combined_opp = ensure_columns(pd.concat(all_opp, ignore_index=True), OPPORTUNITY_COLUMNS) if all_opp else ensure_columns(pd.DataFrame(), OPPORTUNITY_COLUMNS)
    combined_int = ensure_columns(pd.concat(all_int, ignore_index=True), INTERVENTION_COLUMNS) if all_int else ensure_columns(pd.DataFrame(), INTERVENTION_COLUMNS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_opp = OUTPUT_DIR / f"all_sites_{voxel}_proposal_opportunities.csv"
    out_int = OUTPUT_DIR / f"all_sites_{voxel}_proposal_interventions.csv"
    combined_opp.to_csv(out_opp, index=False)
    combined_int.to_csv(out_int, index=False)
    print(f"Saved: {out_opp}")
    print(f"Saved: {out_int}")
    return combined_opp, combined_int


def compute_tree_only_decay_counts(site: str, scenario: str, year: int, voxel_size: float) -> dict:
    tree_path = get_tree_path(site, scenario, year, voxel_size)
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
    required = ["action", "rewilded", "isNewTree"]
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
    rewilded = normalize_series(tree_df["rewilded"])
    is_new_tree = bool_series(tree_df["isNewTree"])

    decay_tree_opp = (~is_new_tree) & action.isin(["AGE-IN-PLACE", "SENESCENT"])
    brace_mask = decay_tree_opp & (rewilded == "exoskeleton")
    buffer_mask = decay_tree_opp & rewilded.isin(["node-rewilded", "footprint-depaved"])

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
) -> pd.DataFrame:
    rows = []
    for site in sites:
        for scenario in scenarios:
            for year in years:
                rows.append(compute_tree_only_decay_counts(site, scenario, year, voxel_size))
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
):
    per_site_df = build_tree_only_decay_df(sites, scenarios, years, voxel_size)
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
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voxel = format_voxel_size(voxel_size)

    by_site_path = OUTPUT_DIR / f"decay_tree_comparison_by_site_scenario_year_{voxel}.csv"
    all_sites_path = OUTPUT_DIR / f"decay_tree_comparison_all_sites_scenario_year_{voxel}.csv"
    markdown_path = OUTPUT_DIR / f"decay_tree_comparison_tables_{voxel}.md"

    per_site_df.to_csv(by_site_path, index=False)
    all_sites_df.to_csv(all_sites_path, index=False)
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved: {by_site_path}")
    print(f"Saved: {all_sites_path}")
    print(f"Saved: {markdown_path}")


def compute_release_control_voxel_counts(site: str, scenario: str, year: int, voxel_size: float) -> dict:
    vtk_path = get_vtk_path(site, scenario, year, voxel_size)
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
) -> pd.DataFrame:
    rows = []
    for site in sites:
        for scenario in scenarios:
            for year in years:
                rows.append(compute_release_control_voxel_counts(site, scenario, year, voxel_size))
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
):
    per_site_df = build_release_control_voxel_df(sites, scenarios, years, voxel_size)
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
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voxel = format_voxel_size(voxel_size)

    by_site_path = OUTPUT_DIR / f"release_control_voxel_comparison_by_site_scenario_year_{voxel}.csv"
    all_sites_path = OUTPUT_DIR / f"release_control_voxel_comparison_all_sites_scenario_year_{voxel}.csv"
    markdown_path = OUTPUT_DIR / f"release_control_voxel_comparison_tables_{voxel}.md"

    per_site_df.to_csv(by_site_path, index=False)
    all_sites_df.to_csv(all_sites_path, index=False)
    markdown_path.write_text(markdown, encoding="utf-8")

    print(f"Saved: {by_site_path}")
    print(f"Saved: {all_sites_path}")
    print(f"Saved: {markdown_path}")


def compute_connect_voxel_counts(site: str, scenario: str, year: int, voxel_size: float) -> dict:
    vtk_path = get_vtk_path(site, scenario, year, voxel_size)
    if not vtk_path.exists():
        return {
            "site": site,
            "scenario": scenario,
            "year": year,
            "source_exists": False,
            "proposal_voxel_count": pd.NA,
            "full_connect_voxel_count": pd.NA,
            "partial_connect_voxel_count": pd.NA,
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
            "full_connect_voxel_count": pd.NA,
            "partial_connect_voxel_count": pd.NA,
            "status": "missing_required_arrays",
            "notes": "Missing arrays: scenario_outputs",
        }

    scenario_outputs = vtk_str_array(poly.point_data["scenario_outputs"])
    scenario_outputs_lower = np.char.lower(scenario_outputs)

    proposal_mask = np.isin(scenario_outputs_lower, list(CONNECT_PROPOSAL_VALUES))
    full_mask = proposal_mask & np.isin(scenario_outputs_lower, list(CONNECT_FULL_VALUES))
    partial_mask = proposal_mask & np.isin(scenario_outputs_lower, list(CONNECT_PARTIAL_VALUES))

    return {
        "site": site,
        "scenario": scenario,
        "year": year,
        "source_exists": True,
        "proposal_voxel_count": int(np.sum(proposal_mask)),
        "full_connect_voxel_count": int(np.sum(full_mask)),
        "partial_connect_voxel_count": int(np.sum(partial_mask)),
        "status": "computed",
        "notes": "",
    }


def build_connect_voxel_df(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float,
) -> pd.DataFrame:
    rows = []
    for site in sites:
        for scenario in scenarios:
            for year in years:
                rows.append(compute_connect_voxel_counts(site, scenario, year, voxel_size))
    columns = [
        "site",
        "scenario",
        "year",
        "source_exists",
        "proposal_voxel_count",
        "full_connect_voxel_count",
        "partial_connect_voxel_count",
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
                        "full_connect_voxel_count": pd.NA,
                        "partial_connect_voxel_count": pd.NA,
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
                    "full_connect_voxel_count": int(subset["full_connect_voxel_count"].fillna(0).sum()),
                    "partial_connect_voxel_count": int(subset["partial_connect_voxel_count"].fillna(0).sum()),
                    "status": "computed",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def render_single_connect_table(title: str, subset: pd.DataFrame, years: list[int]) -> str:
    year_to_row = {int(row["year"]): row for _, row in subset.iterrows()}
    proposal_cells = []
    full_cells = []
    partial_cells = []

    for year in years:
        row = year_to_row.get(year)
        if row is None:
            proposal_cells.append("NA")
            full_cells.append("NA")
            partial_cells.append("NA")
            continue
        proposal_cells.append(format_count_cell(row["proposal_voxel_count"]))
        full_cells.append(format_count_pct_cell(row["full_connect_voxel_count"], row["proposal_voxel_count"]))
        partial_cells.append(format_count_pct_cell(row["partial_connect_voxel_count"], row["proposal_voxel_count"]))

    total_proposal = sum_with_na(subset["proposal_voxel_count"])
    total_full = sum_with_na(subset["full_connect_voxel_count"])
    total_partial = sum_with_na(subset["partial_connect_voxel_count"])

    proposal_total_cell = format_count_cell(total_proposal)
    full_total_cell = format_count_pct_cell(total_full, total_proposal)
    partial_total_cell = format_count_pct_cell(total_partial, total_proposal)

    header = "| Metric | " + " | ".join(str(y) for y in years) + " | Total |"
    divider = "|---|" + "|".join(["---:"] * (len(years) + 1)) + "|"
    row_proposal = "| Proposal voxels [x] | " + " | ".join(proposal_cells) + f" | {proposal_total_cell} |"
    row_full = "| Full Connect | " + " | ".join(full_cells) + f" | {full_total_cell} |"
    row_partial = "| Partial Connect | " + " | ".join(partial_cells) + f" | {partial_total_cell} |"
    return "\n".join([title, header, divider, row_proposal, row_full, row_partial, ""])


def generate_connect_comparison_tables(
    sites: list[str],
    scenarios: list[str],
    years: list[int],
    voxel_size: float,
):
    per_site_df = build_connect_voxel_df(sites, scenarios, years, voxel_size)
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
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voxel = format_voxel_size(voxel_size)

    by_site_path = OUTPUT_DIR / f"connect_voxel_comparison_by_site_scenario_year_{voxel}.csv"
    all_sites_path = OUTPUT_DIR / f"connect_voxel_comparison_all_sites_scenario_year_{voxel}.csv"
    markdown_path = OUTPUT_DIR / f"connect_voxel_comparison_tables_{voxel}.md"

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

    args = parser.parse_args()

    if args.decay_comparison_tables:
        report_sites = SITE_REPORT_ORDER if args.site == "all" else [args.site]
        scenarios = [args.scenario] if args.scenario else list(DEFAULT_SCENARIOS)
        years = parse_years_arg(args.table_years)
        per_site_df, all_sites_df, markdown = generate_decay_comparison_tables(
            sites=report_sites,
            scenarios=scenarios,
            years=years,
            voxel_size=args.voxel_size,
        )
        save_decay_comparison_outputs(args.voxel_size, per_site_df, all_sites_df, markdown)
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
        )
        save_release_control_comparison_outputs(args.voxel_size, per_site_df, all_sites_df, markdown)
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
        )
        save_connect_comparison_outputs(args.voxel_size, per_site_df, all_sites_df, markdown)
        print("\n" + markdown)
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
        )
        save_site_outputs(site, args.voxel_size, opportunity_df, intervention_df, qc_df)
        print(
            f"  Rows -> opportunities: {len(opportunity_df)}, interventions: {len(intervention_df)}, qc: {len(qc_df)}"
        )

    if not args.no_combine:
        combine_all_sites(args.voxel_size)


if __name__ == "__main__":
    main()
