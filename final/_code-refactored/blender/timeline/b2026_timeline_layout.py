from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_scene_contract as scene_contract


TIMELINE_YEARS = (0, 10, 30, 60, 180)
VISUAL_STRIP_POSITION_OVERRIDES = {
    "city": {
        0: 180,
        10: 60,
        30: 30,
        60: 10,
        180: 0,
    },
}

DATA_BUNDLE_ROOT_CANDIDATES = (
    Path(r"D:\2026 Arboreal Futures\urban-futures\_data-refactored\v3engine_outputs"),
    Path(r"D:\2026 Arboreal Futures\refactored-data"),
    Path(r"D:\2026 Arboreal Futures\data-refactored"),
    Path(r"Z:\MF 2026 Arboreal Futures\data"),
    Path(r"D:\2026 Arboreal Futures\data"),
)

REPO_ROOT = Path(__file__).resolve().parents[4]
REPO_DATA_ROOT = REPO_ROOT / "data" / "revised" / "final"

TIMELINE_TEST_MODE = os.environ.get("B2026_TIMELINE_TEST_MODE", "").strip().lower()
TIMELINE_TEST_MODE_SPECS = {
    "parade_positive180_everywhere": {
        "sites": ("trimmed-parade",),
        "scenarios": ("positive", "trending"),
        "years": TIMELINE_YEARS,
        "source_scenario": "positive",
        "source_year": 180,
    },
    "trimmed_parade_positive180_everywhere": {
        "sites": ("trimmed-parade",),
        "scenarios": ("positive", "trending"),
        "years": TIMELINE_YEARS,
        "source_scenario": "positive",
        "source_year": 180,
    },
}


TIMELINE_SITE_SPECS = {
    "trimmed-parade": {
        "scene_name": "parade",
        "world_collection_name": scene_contract.get_collection_name("trimmed-parade", "timeline_base", legacy=True),
        "manager_collection_name": scene_contract.get_collection_name("trimmed-parade", "manager", legacy=True),
        "source_world_objects": (
            scene_contract.SITE_CONTRACTS["trimmed-parade"]["world_objects"]["base"],
            scene_contract.SITE_CONTRACTS["trimmed-parade"]["world_objects"]["road"],
        ),
        "box_length": (280.875, 112.0, 50.0),
        "timeline_years": TIMELINE_YEARS,
        "strips": (
            {
                "year": 0,
                "label": "yr0",
                "box_position": (-89.26000000000931, 279.06, 42.0),
                "translate": (0.0, 0.0, 0.0),
            },
            {
                "year": 10,
                "label": "yr10",
                "box_position": (-89.26000000000931, 166.86, 42.0),
                "translate": (0.0, -5.0, 0.0),
            },
            {
                "year": 30,
                "label": "yr30",
                "box_position": (-89.26000000000931, 54.66, 42.0),
                "translate": (0.0, -10.0, 0.0),
            },
            {
                "year": 60,
                "label": "yr60",
                "box_position": (-89.26000000000931, -57.54, 42.0),
                "translate": (0.0, -15.0, 0.0),
            },
            {
                "year": 180,
                "label": "yr180",
                "box_position": (-89.26000000000931, -169.74, 42.0),
                "translate": (0.0, -20.0, 0.0),
            },
        ),
    },
    "city": {
        "scene_name": "city",
        "world_collection_name": scene_contract.get_collection_name("city", "timeline_base", legacy=True),
        "manager_collection_name": scene_contract.get_collection_name("city", "manager", legacy=True),
        "source_world_objects": (
            scene_contract.SITE_CONTRACTS["city"]["world_objects"]["base"],
            scene_contract.SITE_CONTRACTS["city"]["world_objects"]["road"],
        ),
        "box_length": (281.0, 112.0, 209.0),
        "timeline_years": TIMELINE_YEARS,
        "strips": (
            {
                "year": 0,
                "label": "yr0",
                "box_position": (-75.8041194739053, 97.4443366928, 23.5),
                "translate": (0.0, 0.0, 0.0),
            },
            {
                "year": 10,
                "label": "yr10",
                "box_position": (-75.8041194739053, -14.5556633072, 23.5),
                "translate": (0.0, -5.0, 0.0),
            },
            {
                "year": 30,
                "label": "yr30",
                "box_position": (-75.8041194739053, -126.5556633072, 23.5),
                "translate": (0.0, -10.0, 0.0),
            },
            {
                "year": 60,
                "label": "yr60",
                "box_position": (-75.8041194739053, -238.5556633072, 23.5),
                "translate": (0.0, -15.0, 0.0),
            },
            {
                "year": 180,
                "label": "yr180",
                "box_position": (-75.8041194739053, -350.5556633071974, 23.5),
                "translate": (0.0, -20.0, 0.0),
            },
        ),
    },
    "uni": {
        "scene_name": "uni",
        "world_collection_name": scene_contract.get_collection_name("uni", "timeline_base", legacy=True),
        "manager_collection_name": scene_contract.get_collection_name("uni", "manager", legacy=True),
        "source_world_objects": (
            scene_contract.SITE_CONTRACTS["uni"]["world_objects"]["base"],
            scene_contract.SITE_CONTRACTS["uni"]["world_objects"]["road"],
        ),
        "box_length": (112.2, 281.0, 77.0),
        "timeline_years": TIMELINE_YEARS,
        "strips": (
            {
                "year": 0,
                "label": "yr0",
                "box_position": (-300.29, -88.6363, 32.5),
                "translate": (0.0, 0.0, 0.0),
            },
            {
                "year": 10,
                "label": "yr10",
                "box_position": (-188.09, -88.6363, 32.5),
                "translate": (5.0, 0.0, 0.0),
            },
            {
                "year": 30,
                "label": "yr30",
                "box_position": (-75.89, -88.6363, 32.5),
                "translate": (10.0, 0.0, 0.0),
            },
            {
                "year": 60,
                "label": "yr60",
                "box_position": (36.31, -88.6363, 32.5),
                "translate": (15.0, 0.0, 0.0),
            },
            {
                "year": 180,
                "label": "yr180",
                "box_position": (148.51, -88.6363, 32.5),
                "translate": (20.0, 0.0, 0.0),
            },
        ),
    },
    "street": {
        "scene_name": "street",
        "world_collection_name": scene_contract.get_collection_name("street", "timeline_base", legacy=True),
        "manager_collection_name": scene_contract.get_collection_name("street", "manager", legacy=True),
        "world_objects": (
            scene_contract.SITE_CONTRACTS["street"]["world_objects"]["base"],
            scene_contract.SITE_CONTRACTS["street"]["world_objects"]["road"],
        ),
        "box_length": (112.2, 281.0, 77.0),
        "timeline_years": TIMELINE_YEARS,
        "strips": (
            {
                "year": 0,
                "label": "yr0",
                "box_position": (-300.29, -88.6363, 32.5),
                "translate": (0.0, 0.0, 0.0),
            },
            {
                "year": 10,
                "label": "yr10",
                "box_position": (-188.09, -88.6363, 32.5),
                "translate": (5.0, 0.0, 0.0),
            },
            {
                "year": 30,
                "label": "yr30",
                "box_position": (-75.89, -88.6363, 32.5),
                "translate": (10.0, 0.0, 0.0),
            },
            {
                "year": 60,
                "label": "yr60",
                "box_position": (36.31, -88.6363, 32.5),
                "translate": (15.0, 0.0, 0.0),
            },
            {
                "year": 180,
                "label": "yr180",
                "box_position": (148.51, -88.6363, 32.5),
                "translate": (20.0, 0.0, 0.0),
            },
        ),
    },
}

ASSET_SITE_ALIASES = {
    "street": "uni",
}


def canonicalize_asset_site(site: str) -> str:
    return ASSET_SITE_ALIASES.get(site, site)


def get_timeline_site_spec(site: str) -> dict | None:
    return TIMELINE_SITE_SPECS.get(site)


def get_position_year(site: str, display_year: int) -> int:
    return VISUAL_STRIP_POSITION_OVERRIDES.get(site, {}).get(display_year, display_year)


def timeline_mode_supported(site: str) -> bool:
    return get_timeline_site_spec(site) is not None


def iter_existing_bundle_roots():
    for candidate in DATA_BUNDLE_ROOT_CANDIDATES:
        if candidate.exists():
            yield candidate


def iter_candidate_paths(root: Path, relative_paths: tuple[Path, ...]):
    for relative_path in relative_paths:
        candidate = root / relative_path
        if candidate.exists():
            yield candidate


def get_active_timeline_test_mode() -> str | None:
    if not TIMELINE_TEST_MODE:
        return None
    if TIMELINE_TEST_MODE in TIMELINE_TEST_MODE_SPECS:
        return TIMELINE_TEST_MODE
    raise ValueError(
        f"Unknown B2026_TIMELINE_TEST_MODE '{TIMELINE_TEST_MODE}'. "
        f"Expected one of: {', '.join(sorted(TIMELINE_TEST_MODE_SPECS))}"
    )


def resolve_source_asset_request(site: str, scenario: str, year: int) -> tuple[str, int]:
    mode_name = get_active_timeline_test_mode()
    if mode_name is None:
        return scenario, year

    mode_spec = TIMELINE_TEST_MODE_SPECS[mode_name]
    if site not in mode_spec["sites"]:
        return scenario, year
    if scenario not in mode_spec["scenarios"]:
        return scenario, year
    if year not in mode_spec["years"]:
        return scenario, year
    return mode_spec["source_scenario"], mode_spec["source_year"]


def resolve_tree_ply_folder() -> Path:
    for root in iter_existing_bundle_roots():
        candidate = root / "treeMeshesPly"
        if candidate.exists():
            return candidate
    return REPO_DATA_ROOT / "treeMeshesPly"


def resolve_log_ply_folder() -> Path:
    for root in iter_existing_bundle_roots():
        for folder_name in ("logMeshesPly", "logMeshesPLY"):
            candidate = root / folder_name
            if candidate.exists():
                return candidate
    return REPO_DATA_ROOT / "logMeshesPly"


def resolve_feature_csv_path(site: str, scenario: str, year: int) -> Path:
    asset_site = canonicalize_asset_site(site)
    source_scenario, source_year = resolve_source_asset_request(site, scenario, year)
    bundle_name = f"{asset_site}_{source_scenario}_1_nodeDF_yr{source_year}.csv"
    for root in iter_existing_bundle_roots():
        relative_paths = (
            Path("feature-locations") / asset_site / bundle_name,
            Path("node-dfs") / asset_site / bundle_name,
            Path("nodeDFs") / asset_site / bundle_name,
            Path("node_dfs") / asset_site / bundle_name,
            Path(asset_site) / "feature-locations" / bundle_name,
            Path(asset_site) / "node-dfs" / bundle_name,
            Path(asset_site) / "nodeDFs" / bundle_name,
            Path(asset_site) / "node_dfs" / bundle_name,
        )
        for candidate in iter_candidate_paths(root, relative_paths):
            return candidate

    for filename in (
        f"{asset_site}_{source_scenario}_1_nodeDF_{source_year}.csv",
        f"{asset_site}_{source_scenario}_1_nodeDF_yr{source_year}.csv",
    ):
        candidate = REPO_DATA_ROOT / asset_site / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find feature-location CSV for "
        f"site={site}, scenario={scenario}, year={year} "
        f"(resolved source: scenario={source_scenario}, year={source_year})"
    )


def resolve_bioenvelope_ply_path(site: str, scenario: str, year: int) -> Path:
    asset_site = canonicalize_asset_site(site)
    source_scenario, source_year = resolve_source_asset_request(site, scenario, year)
    bundle_name = f"{asset_site}_{source_scenario}_1_envelope_scenarioYR{source_year}.ply"
    for root in iter_existing_bundle_roots():
        relative_paths = (
            Path("bioenvelopes") / asset_site / bundle_name,
            Path("envelopes") / asset_site / bundle_name,
            Path(asset_site) / "bioenvelopes" / bundle_name,
            Path(asset_site) / "envelopes" / bundle_name,
        )
        for candidate in iter_candidate_paths(root, relative_paths):
            return candidate

    candidate = REPO_DATA_ROOT / asset_site / "ply" / bundle_name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find bioenvelope PLY for "
        f"site={site}, scenario={scenario}, year={year} "
        f"(resolved source: scenario={source_scenario}, year={source_year})"
    )


def resolve_state_vtk_path(site: str, scenario: str, year: int) -> Path:
    asset_site = canonicalize_asset_site(site)
    source_scenario, source_year = resolve_source_asset_request(site, scenario, year)
    bundle_name = f"{asset_site}_{source_scenario}_1_scenarioYR{source_year}.vtk"
    refactored_bundle_name = (
        f"{asset_site}_{source_scenario}_1_yr{source_year}_state_with_indicators.vtk"
    )
    for root in iter_existing_bundle_roots():
        relative_paths = (
            Path("vtks") / asset_site / bundle_name,
            Path("vtks") / asset_site / refactored_bundle_name,
            Path("scenario-vtks") / asset_site / bundle_name,
            Path("scenario-vtks") / asset_site / refactored_bundle_name,
            Path("state-vtks") / asset_site / bundle_name,
            Path("state-vtks") / asset_site / refactored_bundle_name,
            Path(asset_site) / "vtks" / bundle_name,
            Path(asset_site) / "vtks" / refactored_bundle_name,
            Path(asset_site) / "scenario-vtks" / bundle_name,
            Path(asset_site) / "scenario-vtks" / refactored_bundle_name,
            Path(asset_site) / "state-vtks" / bundle_name,
            Path(asset_site) / "state-vtks" / refactored_bundle_name,
        )
        for candidate in iter_candidate_paths(root, relative_paths):
            return candidate

    candidate = REPO_DATA_ROOT / asset_site / f"{asset_site}_{source_scenario}_1_scenarioYR{source_year}.vtk"
    if candidate.exists():
        return candidate
    candidate = REPO_DATA_ROOT / asset_site / refactored_bundle_name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find state VTK for "
        f"site={site}, scenario={scenario}, year={year} "
        f"(resolved source: scenario={source_scenario}, year={source_year})"
    )


def strip_bounds(strip_spec: dict, site_spec: dict) -> tuple[np.ndarray, np.ndarray]:
    mins = np.asarray(strip_spec["box_position"], dtype=np.float64)
    lengths = np.asarray(site_spec["box_length"], dtype=np.float64)
    return mins, mins + lengths


def clip_mask_for_points(points: np.ndarray, strip_spec: dict, site_spec: dict) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    mins, maxs = strip_bounds(strip_spec, site_spec)
    return np.all((points >= mins) & (points <= maxs), axis=1)


def translate_points(points: np.ndarray, strip_spec: dict) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    offset = np.asarray(strip_spec["translate"], dtype=np.float64)
    return points + offset


def filter_dataframe_to_strip(df: pd.DataFrame, strip_spec: dict, site_spec: dict) -> pd.DataFrame:
    mins, maxs = strip_bounds(strip_spec, site_spec)
    mask = (
        df["x"].between(mins[0], maxs[0])
        & df["y"].between(mins[1], maxs[1])
        & df["z"].between(mins[2], maxs[2])
    )
    clipped = df.loc[mask].copy()
    if clipped.empty:
        return clipped

    clipped["x"] = clipped["x"] + strip_spec["translate"][0]
    clipped["y"] = clipped["y"] + strip_spec["translate"][1]
    clipped["z"] = clipped["z"] + strip_spec["translate"][2]
    clipped["year"] = strip_spec["year"]
    clipped["timeline_year"] = strip_spec["year"]
    clipped["timeline_label"] = strip_spec["label"]
    return clipped


def build_timeline_dataframe(site: str, scenario: str, years: tuple[int, ...] | None = None):
    site_spec = get_timeline_site_spec(site)
    if site_spec is None:
        raise ValueError(f"Timeline mode is not configured for site '{site}'")

    requested_years = years or tuple(site_spec["timeline_years"])
    timeline_frames = []
    source_paths = []
    used_strips = []

    strips_by_year = {strip["year"]: strip for strip in site_spec["strips"]}
    for year in requested_years:
        position_year = get_position_year(site, year)
        strip_spec = strips_by_year.get(position_year)
        if strip_spec is None:
            raise ValueError(f"No timeline strip spec defined for display year {year} in site '{site}'")

        source_scenario, source_year = resolve_source_asset_request(site, scenario, year)
        csv_path = resolve_feature_csv_path(site, scenario, year)
        source_df = pd.read_csv(csv_path)
        clipped_df = filter_dataframe_to_strip(source_df, strip_spec, site_spec)
        if not clipped_df.empty:
            clipped_df["year"] = year
            clipped_df["timeline_year"] = year
            clipped_df["source_scenario"] = source_scenario
            clipped_df["source_timeline_year"] = source_year
            clipped_df["position_timeline_year"] = position_year
            clipped_df["position_timeline_label"] = strip_spec["label"]
        timeline_frames.append(clipped_df)
        source_paths.append(csv_path)
        used_strips.append(strip_spec)

    if timeline_frames:
        combined = pd.concat(timeline_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()
    return combined, source_paths, used_strips
