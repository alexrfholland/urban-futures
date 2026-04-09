"""Build bV2 tree/log/pole instancers for the active scene."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import bpy
import numpy as np
import pandas as pd

try:
    from .bV2_scene_contract import (
        GLOBAL_RULES,
        NODE_GROUP_NAMES,
        get_instancer_families,
        get_mode_year_token,
        make_models_collection_name,
        make_position_object_name,
    )
    from .bV2_paths import iter_blender_input_roots
    from refactor_code.blender.bexport.proposal_framebuffers import (
        DEFAULT_OUTPUT_COLUMNS as PROPOSAL_FRAMEBUFFER_OUTPUT_COLUMNS,
        build_blender_proposal_framebuffer_columns,
    )
except ImportError:
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from bV2_scene_contract import (  # type: ignore
        GLOBAL_RULES,
        NODE_GROUP_NAMES,
        get_instancer_families,
        get_mode_year_token,
        make_models_collection_name,
        make_position_object_name,
    )
    from bV2_paths import iter_blender_input_roots  # type: ignore
    from refactor_code.blender.bexport.proposal_framebuffers import (  # type: ignore
        DEFAULT_OUTPUT_COLUMNS as PROPOSAL_FRAMEBUFFER_OUTPUT_COLUMNS,
        build_blender_proposal_framebuffer_columns,
    )


CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
REPO_ROOT = CODE_ROOT.parent
REPO_DATA_ROOT = REPO_ROOT / "data" / "revised" / "final"

TIMELINE_YEARS = (0, 10, 30, 60, 180)
TIMELINE_OFFSET_STEP = 5.0
VISUAL_STRIP_POSITION_OVERRIDES = {
    "city": {
        0: 180,
        10: 60,
        30: 30,
        60: 10,
        180: 0,
    },
}


def cumulative_timeline_translate(axis: str, offset_index: int) -> tuple[float, float, float]:
    distance = float(offset_index) * TIMELINE_OFFSET_STEP
    if axis == "x":
        return (distance, 0.0, 0.0)
    if axis == "y":
        return (0.0, -distance, 0.0)
    raise ValueError(f"Unsupported timeline offset axis: {axis!r}")


TIMELINE_SITE_SPECS = {
    "trimmed-parade": {
        "box_length": (280.875, 112.0, 50.0),
        "strips": {
            0: {"label": "yr0", "box_position": (-89.26000000000931, 279.06, 42.0), "translate": cumulative_timeline_translate("y", 0)},
            10: {"label": "yr10", "box_position": (-89.26000000000931, 166.86, 42.0), "translate": cumulative_timeline_translate("y", 1)},
            30: {"label": "yr30", "box_position": (-89.26000000000931, 54.66, 42.0), "translate": cumulative_timeline_translate("y", 2)},
            60: {"label": "yr60", "box_position": (-89.26000000000931, -57.54, 42.0), "translate": cumulative_timeline_translate("y", 3)},
            180: {"label": "yr180", "box_position": (-89.26000000000931, -169.74, 42.0), "translate": cumulative_timeline_translate("y", 4)},
        },
    },
    "city": {
        "box_length": (281.0, 112.0, 209.0),
        "strips": {
            0: {"label": "yr0", "box_position": (-75.8041194739053, 97.4443366928, 23.5), "translate": cumulative_timeline_translate("y", 0)},
            10: {"label": "yr10", "box_position": (-75.8041194739053, -14.5556633072, 23.5), "translate": cumulative_timeline_translate("y", 1)},
            30: {"label": "yr30", "box_position": (-75.8041194739053, -126.5556633072, 23.5), "translate": cumulative_timeline_translate("y", 2)},
            60: {"label": "yr60", "box_position": (-75.8041194739053, -238.5556633072, 23.5), "translate": cumulative_timeline_translate("y", 3)},
            180: {"label": "yr180", "box_position": (-75.8041194739053, -350.5556633071974, 23.5), "translate": cumulative_timeline_translate("y", 4)},
        },
    },
    "uni": {
        "box_length": (112.2, 281.0, 77.0),
        "strips": {
            0: {"label": "yr0", "box_position": (-300.29, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 0)},
            10: {"label": "yr10", "box_position": (-188.09, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 1)},
            30: {"label": "yr30", "box_position": (-75.89, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 2)},
            60: {"label": "yr60", "box_position": (36.31, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 3)},
            180: {"label": "yr180", "box_position": (148.51, -88.6363, 32.5), "translate": cumulative_timeline_translate("x", 4)},
        },
    },
}

DATA_BUNDLE_ROOT_ENV_NAMES = (
    "BV2_DATA_BUNDLE_ROOTS",
    "BV2_DATA_BUNDLE_ROOT",
    "B2026_DATA_BUNDLE_ROOTS",
    "B2026_DATA_BUNDLE_ROOT",
)

ASSET_SITE_ALIASES = {"street": "uni"}

STATE_TO_COLLECTION_ROLE = {
    "positive": "positive_instances",
    "positive_priority": "positive_priority_instances",
    "trending": "trending_instances",
}
FAMILY_TO_NODE_TYPE = {
    "trees": "tree",
    "logs": "log",
    "poles": "pole",
}
FAMILY_TO_PLY_ROOT_ROLE = {
    "trees": "tree",
    "logs": "log",
    "poles": "tree",
}

IGNORED_TREE_SIZES = {"gone", "early-tree-death"}
PRIORITY_TREE_SIZES = {"senescing", "snag", "fallen"}
POLE_FALLBACK_PLY = "artificial_precolonial.False_size.snag_control.improved-tree_id.10.ply"
HIDE_IMPORTED_MODEL_OBJECTS = True
MODEL_PASS_INDEX = 3
POINT_CLOUD_PASS_INDEX = 3
SOURCE_YEAR_DEFAULT = int(GLOBAL_RULES["source_year_initial_value"])

TREE_PROPOSAL_COLUMNS = tuple(PROPOSAL_FRAMEBUFFER_OUTPUT_COLUMNS.values())
TREE_RESOURCE_COLUMNS = (
    "resource_other",
    "resource_dead",
    "resource_peeling",
    "resource_perch",
    "resource_epiphyte",
    "resource_fallen",
    "resource_hollow",
)
LOG_PATH = os.environ.get("BV2_LOG_PATH", "").strip()
SOURCE_YEAR_DEBUG_COLORS = (
    (0, (0.894, 0.102, 0.110, 1.0)),
    (10, (0.216, 0.494, 0.722, 1.0)),
    (30, (0.302, 0.686, 0.290, 1.0)),
    (60, (0.596, 0.306, 0.639, 1.0)),
    (180, (1.000, 0.498, 0.000, 1.0)),
)


def log(*args) -> None:
    message = " ".join(str(arg) for arg in args)
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


def get_instancer_display_material_name() -> str:
    mode = os.environ.get("BV2_INSTANCER_DISPLAY_MODE", "").strip().lower()
    if mode in {"source_year", "source-year", "debug_source_years", "debug-source-years"}:
        return str(GLOBAL_RULES.get("debug_source_years_material", "debug-source-years"))
    return str(GLOBAL_RULES["instancer_material"])


def canonicalize_asset_site(site: str) -> str:
    return ASSET_SITE_ALIASES.get(site, site)


def iter_existing_bundle_roots() -> Iterable[Path]:
    seen: set[Path] = set()
    for env_name in DATA_BUNDLE_ROOT_ENV_NAMES:
        raw_value = os.environ.get(env_name, "").strip()
        if not raw_value:
            continue
        for raw_path in raw_value.split(os.pathsep):
            candidate = Path(raw_path.strip())
            if not raw_path.strip() or candidate in seen:
                continue
            if candidate.exists():
                seen.add(candidate)
                yield candidate

    for candidate in iter_blender_input_roots():
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def resolve_feature_csv_path(site: str, scenario: str, year: int) -> Path:
    asset_site = canonicalize_asset_site(site)
    bundle_name = f"{asset_site}_{scenario}_1_nodeDF_yr{year}.csv"
    for root in iter_existing_bundle_roots():
        for relative in (
            Path("feature-locations") / asset_site / bundle_name,
            Path("node-dfs") / asset_site / bundle_name,
            Path("nodeDFs") / asset_site / bundle_name,
            Path("node_dfs") / asset_site / bundle_name,
            Path(asset_site) / "feature-locations" / bundle_name,
            Path(asset_site) / "node-dfs" / bundle_name,
            Path(asset_site) / "nodeDFs" / bundle_name,
            Path(asset_site) / "node_dfs" / bundle_name,
        ):
            candidate = root / relative
            if candidate.exists():
                return candidate

    for filename in (f"{asset_site}_{scenario}_1_nodeDF_{year}.csv", bundle_name):
        candidate = REPO_DATA_ROOT / asset_site / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve node CSV for site={site}, scenario={scenario}, year={year}")


def resolve_tree_ply_folder() -> Path:
    for root in iter_existing_bundle_roots():
        candidate = root / "treeMeshesPly"
        if candidate.exists():
            return candidate
    for candidate in (
        Path(r"D:\2026 Arboreal Futures\data\treeMeshesPly"),
        REPO_DATA_ROOT / "treeMeshesPly",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve treeMeshesPly folder")


def resolve_log_ply_folder() -> Path:
    for root in iter_existing_bundle_roots():
        for folder_name in ("logMeshesPly", "logMeshesPLY"):
            candidate = root / folder_name
            if candidate.exists():
                return candidate
    for candidate in (
        Path(r"D:\2026 Arboreal Futures\data\logMeshesPly"),
        Path(r"D:\2026 Arboreal Futures\data\logMeshesPLY"),
        REPO_DATA_ROOT / "logMeshesPly",
        REPO_DATA_ROOT / "logMeshesPLY",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve logMeshesPly folder")


def get_active_scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    if scene is None:
        raise RuntimeError("No active Blender scene")
    return scene


def get_runtime_config(scene: bpy.types.Scene | None = None) -> dict[str, object]:
    scene = scene or get_active_scene()
    site = str(scene.get("bV2_site", "")).strip()
    mode = str(scene.get("bV2_mode", "")).strip()
    year_raw = str(scene.get("bV2_year", "")).strip()
    year = int(year_raw) if year_raw else None
    if not site or not mode:
        raise RuntimeError("Scene is missing bV2 runtime metadata (`bV2_site`, `bV2_mode`)")
    return {
        "scene": scene,
        "site": site,
        "mode": mode,
        "year": year,
        "year_token": get_mode_year_token(mode, year),
        "instancer_families": get_instancer_families(site),
    }


def get_active_years(site: str, mode: str, year: int | None) -> tuple[int, ...]:
    if mode == "timeline":
        return TIMELINE_YEARS
    if year is None:
        raise ValueError(f"{mode} instancer build requires a year")
    return (int(year),)


def get_position_year(site: str, display_year: int) -> int:
    return VISUAL_STRIP_POSITION_OVERRIDES.get(site, {}).get(display_year, display_year)


def get_timeline_strip_spec(site: str, display_year: int) -> dict:
    site_spec = TIMELINE_SITE_SPECS.get(site)
    if site_spec is None:
        raise ValueError(f"No timeline strip spec configured for site '{site}'")
    position_year = get_position_year(site, display_year)
    strip_spec = site_spec["strips"].get(position_year)
    if strip_spec is None:
        raise ValueError(f"No timeline strip defined for site='{site}' year={display_year} (position year {position_year})")
    return strip_spec


def filter_dataframe_to_strip(df: pd.DataFrame, site: str, display_year: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    site_spec = TIMELINE_SITE_SPECS.get(site)
    if site_spec is None:
        raise ValueError(f"No timeline site spec configured for site '{site}'")
    strip_spec = get_timeline_strip_spec(site, display_year)
    mins = np.asarray(strip_spec["box_position"], dtype=np.float64)
    lengths = np.asarray(site_spec["box_length"], dtype=np.float64)
    maxs = mins + lengths
    mask = (
        df["x"].between(mins[0], maxs[0])
        & df["y"].between(mins[1], maxs[1])
        & df["z"].between(mins[2], maxs[2])
    )
    clipped = df.loc[mask].copy()
    if clipped.empty:
        return clipped
    clipped["timeline_label"] = strip_spec["label"]
    clipped["position_timeline_year"] = get_position_year(site, display_year)
    return clipped


def get_series_or_default(df: pd.DataFrame, column_name: str, default_value) -> pd.Series:
    if column_name in df.columns:
        return df[column_name]
    return pd.Series([default_value] * len(df), index=df.index)


def convert_control(value) -> pd.Series:
    control_map = {
        "street-tree": 1,
        "park-tree": 2,
        "reserve-tree": 3,
        "improved-tree": 4,
    }
    return pd.Series(value).fillna("").astype(str).str.lower().map(control_map).fillna(-1).astype(np.int32)


def convert_precolonial(value) -> pd.Series:
    normalized = pd.Series(value).fillna("").astype(str).str.strip().str.lower()
    precolonial_map = {
        "false": 1,
        "0": 1,
        "no": 1,
        "true": 2,
        "1": 2,
        "yes": 2,
    }
    return normalized.map(precolonial_map).fillna(-1).astype(np.int32)


def convert_size(value) -> pd.Series:
    size_map = {
        "small": 1,
        "medium": 2,
        "large": 3,
        "senescing": 4,
        "snag": 5,
        "fallen": 6,
        "decayed": 7,
    }
    return pd.Series(value).fillna("").astype(str).str.lower().map(size_map).fillna(-1).astype(np.int32)


def convert_improvement(value) -> pd.Series:
    normalized = pd.Series(value).fillna("").astype(str).str.strip().str.lower()
    truthy = {"true", "yes", "1", "y"}
    falsy = {"false", "no", "0", "n", ""}
    result = pd.Series(np.full(len(normalized), -1, dtype=np.int32), index=normalized.index)
    result[normalized.isin(truthy)] = 1
    result[normalized.isin(falsy)] = 0
    return result.astype(np.int32)


def stable_hash_index(parts: tuple[object, ...], modulo: int) -> int:
    if modulo <= 0:
        return 0
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def apply_site_specific_fixes(df: pd.DataFrame, site: str) -> pd.DataFrame:
    adjusted = df.copy()
    if site == "trimmed-parade" and "nodeType" not in adjusted.columns:
        adjusted["nodeType"] = "tree"
    return adjusted


def ensure_proposal_framebuffer_columns(df: pd.DataFrame) -> pd.DataFrame:
    adjusted = df.copy()
    missing = [column for column in TREE_PROPOSAL_COLUMNS if column not in adjusted.columns]
    if not missing:
        return adjusted
    proposal_columns = build_blender_proposal_framebuffer_columns(adjusted)
    for attr_name in TREE_PROPOSAL_COLUMNS:
        adjusted[attr_name] = proposal_columns[attr_name].to_numpy(dtype=np.int32)
    return adjusted


def normalize_node_dataframe(df: pd.DataFrame, site: str, source_year: int, offset: tuple[float, float, float]) -> pd.DataFrame:
    adjusted = ensure_proposal_framebuffer_columns(apply_site_specific_fixes(df, site))
    adjusted = adjusted.copy()
    for axis in ("x", "y", "z"):
        adjusted[axis] = pd.to_numeric(get_series_or_default(adjusted, axis, 0.0), errors="coerce").fillna(0.0)

    adjusted["x"] = adjusted["x"] + float(offset[0])
    adjusted["y"] = adjusted["y"] + float(offset[1])
    adjusted["z"] = adjusted["z"] + float(offset[2])
    adjusted["source-year"] = int(source_year)
    adjusted["nodeType"] = get_series_or_default(adjusted, "nodeType", "tree").fillna("tree").astype(str).str.lower()
    adjusted["tree_id"] = pd.to_numeric(get_series_or_default(adjusted, "tree_id", -1), errors="coerce").fillna(-1).astype(np.int32)
    adjusted["structureID"] = pd.to_numeric(get_series_or_default(adjusted, "structureID", -1), errors="coerce").fillna(-1).astype(np.int32)
    adjusted["NodeID"] = pd.to_numeric(get_series_or_default(adjusted, "NodeID", -1), errors="coerce").fillna(-1).astype(np.int32)
    adjusted["rotateZ"] = pd.to_numeric(get_series_or_default(adjusted, "rotateZ", 0.0), errors="coerce").fillna(0.0).astype(np.float32)
    return adjusted


def load_scenario_dataframe(site: str, scenario: str, mode: str, year: int | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    active_years = tuple(get_active_years(site, mode, year))
    log("LOAD_SCENARIO_START", "site=", site, "scenario=", scenario, "mode=", mode, "years=", active_years)
    for source_year in active_years:
        csv_path = resolve_feature_csv_path(site, scenario, source_year)
        log("LOAD_SCENARIO_YEAR_START", scenario, source_year, csv_path)
        source_df = pd.read_csv(csv_path)
        if mode == "timeline":
            strip_spec = get_timeline_strip_spec(site, source_year)
            source_df = filter_dataframe_to_strip(source_df, site, source_year)
            offset = strip_spec["translate"]
        else:
            offset = (0.0, 0.0, 0.0)
        frames.append(normalize_node_dataframe(source_df, site, source_year, offset))
        log("LOAD_SCENARIO_YEAR_DONE", scenario, source_year, "rows=", len(frames[-1]))
    if not frames:
        log("LOAD_SCENARIO_DONE", scenario, "rows=0")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    log("LOAD_SCENARIO_DONE", scenario, "rows=", len(combined))
    return combined


def drop_ignored_tree_sizes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "size" not in df.columns:
        return df
    normalized_sizes = df["size"].fillna("").astype(str).str.lower()
    ignore_mask = df["nodeType"].eq("tree") & normalized_sizes.isin(IGNORED_TREE_SIZES)
    if not ignore_mask.any():
        return df
    filtered = df.loc[~ignore_mask].copy()
    log(f"Ignoring {int(ignore_mask.sum())} tree rows with sizes {sorted(set(normalized_sizes.loc[ignore_mask]))}")
    return filtered


def derive_priority_dataframe(positive_df: pd.DataFrame) -> pd.DataFrame:
    if positive_df.empty:
        return positive_df.copy()
    tree_mask = positive_df["nodeType"].eq("tree") & positive_df["size"].fillna("").astype(str).str.lower().isin(PRIORITY_TREE_SIZES)
    log_mask = positive_df["nodeType"].eq("log")
    return positive_df.loc[tree_mask | log_mask].copy()


def split_family_dataframes(df: pd.DataFrame, site: str) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    valid_families = set(get_instancer_families(site))
    for family, node_type in FAMILY_TO_NODE_TYPE.items():
        if family not in valid_families:
            continue
        result[family] = df.loc[df["nodeType"].eq(node_type)].copy()
    return result


def parse_ply_filenames(filenames: Iterable[str], family: str) -> pd.DataFrame:
    values = pd.Series(list(filenames), dtype="object")
    if values.empty:
        return pd.DataFrame()

    if family in {"trees", "poles"}:
        parts = values.str[:-4].str.split("_", expand=True)
        return pd.DataFrame(
            {
                "precolonial": parts[0].str.split(".").str[1],
                "size": parts[1].str.split(".").str[1].str.lower(),
                "control": parts[2].str.split(".").str[1],
                "id": pd.to_numeric(parts[3].str.split(".").str[1], errors="coerce").fillna(-1).astype(np.int32),
                "filename": values,
            }
        )

    parts = values.str[:-4].str.split(".", expand=True)
    return pd.DataFrame(
        {
            "size": parts[1].str.lower(),
            "id": pd.to_numeric(parts[3], errors="coerce").fillna(-1).astype(np.int32),
            "filename": values,
        }
    )


def build_available_tree_id_map(ply_folder: Path) -> dict[tuple[str, str], tuple[int, ...]]:
    tree_id_map: dict[tuple[str, str], set[int]] = {}
    for filename in os.listdir(ply_folder):
        if not filename.endswith(".ply") or filename.startswith("artificial_"):
            continue
        try:
            precolonial_part, size_part, _control_part, id_part = filename[:-4].split("_")
            precolonial = precolonial_part.split(".")[1]
            size = size_part.split(".")[1].lower()
            tree_id = int(id_part.split(".")[1])
        except Exception:
            continue
        tree_id_map.setdefault((precolonial, size), set()).add(tree_id)
    return {key: tuple(sorted(values)) for key, values in tree_id_map.items()}


def remap_tree_ids_to_available_models(df: pd.DataFrame, ply_folder: Path) -> pd.DataFrame:
    if df.empty or "tree_id" not in df.columns:
        return df
    available_tree_ids = build_available_tree_id_map(ply_folder)
    remapped_ids: list[int] = []
    for row in df.itertuples():
        precolonial = str(getattr(row, "precolonial", "False"))
        size = str(getattr(row, "size", "")).lower()
        available_ids = available_tree_ids.get((precolonial, size))
        try:
            original_id = int(getattr(row, "tree_id", -1))
        except Exception:
            original_id = -1
        if not available_ids or original_id in available_ids:
            remapped_ids.append(original_id)
            continue
        chosen_id = available_ids[
            stable_hash_index(
                (
                    getattr(row, "structureID", getattr(row, "Index", 0)),
                    getattr(row, "x", 0.0),
                    getattr(row, "y", 0.0),
                    precolonial,
                    size,
                    original_id,
                ),
                len(available_ids),
            )
        ]
        remapped_ids.append(chosen_id)
    adjusted = df.copy()
    adjusted["tree_id"] = np.asarray(remapped_ids, dtype=np.int32)
    return adjusted


def choose_stable_match(matches: pd.DataFrame, needed: pd.Series) -> str:
    if matches.empty:
        raise ValueError("No candidate matches supplied")
    idx = stable_hash_index(tuple(needed.astype(str).tolist()), len(matches))
    return str(matches.iloc[idx]["filename"])


def build_filename_requests(df: pd.DataFrame, family: str) -> pd.DataFrame:
    adjusted = df.copy()
    if family in {"trees", "poles"}:
        adjusted["filename"] = (
            "precolonial."
            + adjusted["precolonial"].astype(str).str.capitalize()
            + "_size."
            + adjusted["size"].astype(str)
            + "_control."
            + adjusted["control"].astype(str)
            + "_id."
            + adjusted["tree_id"].astype(str)
            + ".ply"
        )
    else:
        adjusted["filename"] = (
            "size."
            + adjusted["size"].astype(str)
            + ".log."
            + adjusted["tree_id"].astype(str)
            + ".ply"
        )
    return adjusted


def build_filename_fallback_map(df: pd.DataFrame, family: str, ply_folder: Path) -> dict[str, str]:
    unique_filenames = sorted(df["filename"].unique().tolist())
    if family == "poles" and POLE_FALLBACK_PLY:
        fallback_path = ply_folder / POLE_FALLBACK_PLY
        if fallback_path.exists():
            return {filename: POLE_FALLBACK_PLY for filename in unique_filenames}

    available_plys = [filename for filename in os.listdir(ply_folder) if filename.endswith(".ply")]
    available_templates = parse_ply_filenames(available_plys, family)
    needed_templates = parse_ply_filenames(unique_filenames, family)
    fallback_map: dict[str, str] = {}

    for _, needed in needed_templates.iterrows():
        if family in {"trees", "poles"}:
            exact = available_templates[
                (available_templates["precolonial"] == needed["precolonial"])
                & (available_templates["size"] == str(needed["size"]).lower())
                & (available_templates["control"] == needed["control"])
                & (available_templates["id"] == int(needed["id"]))
            ]
            best = available_templates[
                (available_templates["precolonial"] == needed["precolonial"])
                & (available_templates["size"] == str(needed["size"]).lower())
                & (available_templates["control"] == needed["control"])
            ]
            precolonial_size = available_templates[
                (available_templates["precolonial"] == needed["precolonial"])
                & (available_templates["size"] == str(needed["size"]).lower())
            ]
            size_only = available_templates[available_templates["size"] == str(needed["size"]).lower()]
            matches = exact if not exact.empty else best if not best.empty else precolonial_size if not precolonial_size.empty else size_only
        else:
            exact = available_templates[
                (available_templates["size"] == str(needed["size"]).lower())
                & (available_templates["id"] == int(needed["id"]))
            ]
            size_only = available_templates[available_templates["size"] == str(needed["size"]).lower()]
            matches = exact if not exact.empty else size_only

        if matches.empty:
            matches = available_templates
        fallback_map[str(needed["filename"])] = choose_stable_match(matches, needed)

    return fallback_map


def make_model_object_name(index: int, family: str, resolved_filename: str) -> str:
    return f"instanceID.{index}_{family}__{Path(resolved_filename).stem}"


def import_ply_object(filepath: Path) -> bpy.types.Object | None:
    before_names = set(bpy.data.objects.keys())
    bpy.ops.wm.ply_import(filepath=str(filepath))
    new_names = [name for name in bpy.data.objects.keys() if name not in before_names]
    new_objects = [bpy.data.objects[name] for name in new_names]
    mesh_objects = [obj for obj in new_objects if obj.type == "MESH"]
    if mesh_objects:
        return mesh_objects[0]
    if new_objects:
        return new_objects[0]
    return None


def configure_imported_model_visibility(obj: bpy.types.Object) -> None:
    if not HIDE_IMPORTED_MODEL_OBJECTS:
        return
    obj.hide_viewport = True
    obj.hide_render = True
    obj.hide_select = True
    obj.display_type = "BOUNDS"


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_debug_source_years_material() -> bpy.types.Material:
    material_name = str(GLOBAL_RULES.get("debug_source_years_material", "debug-source-years"))
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    material.use_fake_user = True

    node_tree = material.node_tree
    node_tree.nodes.clear()
    nodes = node_tree.nodes

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (980, 0)

    attr_geometry = nodes.new("ShaderNodeAttribute")
    attr_geometry.location = (-920, 120)
    attr_geometry.attribute_name = "source-year"
    if hasattr(attr_geometry, "attribute_type"):
        attr_geometry.attribute_type = "GEOMETRY"

    attr_instancer = nodes.new("ShaderNodeAttribute")
    attr_instancer.location = (-920, -60)
    attr_instancer.attribute_name = "source-year"
    if hasattr(attr_instancer, "attribute_type"):
        attr_instancer.attribute_type = "INSTANCER"

    combine_year = nodes.new("ShaderNodeMath")
    combine_year.operation = "MAXIMUM"
    combine_year.location = (-660, 40)

    fallback = nodes.new("ShaderNodeRGB")
    fallback.location = (-260, 220)
    fallback.outputs[0].default_value = (0.3, 0.3, 0.3, 1.0)

    last_color_socket = fallback.outputs["Color"]
    x_position = -420
    y_base = 120
    for index, (year_value, color_value) in enumerate(SOURCE_YEAR_DEBUG_COLORS):
        compare = nodes.new("ShaderNodeMath")
        compare.location = (x_position, y_base - index * 170)
        compare.operation = "COMPARE"
        compare.inputs[1].default_value = float(year_value)
        compare.inputs[2].default_value = 0.5

        color_node = nodes.new("ShaderNodeRGB")
        color_node.location = (x_position + 220, y_base - index * 170)
        color_node.outputs[0].default_value = color_value

        mix_rgb = nodes.new("ShaderNodeMixRGB")
        mix_rgb.location = (x_position + 460, y_base - index * 170)
        mix_rgb.blend_type = "MIX"

        ensure_link(node_tree, combine_year.outputs["Value"], compare.inputs[0])
        ensure_link(node_tree, compare.outputs["Value"], mix_rgb.inputs["Fac"])
        ensure_link(node_tree, last_color_socket, mix_rgb.inputs["Color1"])
        ensure_link(node_tree, color_node.outputs["Color"], mix_rgb.inputs["Color2"])
        last_color_socket = mix_rgb.outputs["Color"]

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (280, 0)
    emission.inputs["Strength"].default_value = 1.0
    ensure_link(node_tree, attr_geometry.outputs["Fac"], combine_year.inputs[0])
    ensure_link(node_tree, attr_instancer.outputs["Fac"], combine_year.inputs[1])
    ensure_link(node_tree, last_color_socket, emission.inputs["Color"])
    ensure_link(node_tree, emission.outputs["Emission"], output.inputs["Surface"])
    return material


def ensure_model_material(obj: bpy.types.Object, material_name: str | None = None) -> None:
    material_name = material_name or get_instancer_display_material_name()
    if material_name == str(GLOBAL_RULES.get("debug_source_years_material", "debug-source-years")):
        material = ensure_debug_source_years_material()
    else:
        material = bpy.data.materials.get(material_name)
    if material is None or obj.type != "MESH":
        return
    mesh = obj.data
    if len(mesh.materials) == 0:
        mesh.materials.append(material)
    else:
        for index in range(len(mesh.materials)):
            mesh.materials[index] = material


def ensure_constant_mesh_int_attribute(mesh: bpy.types.Mesh, name: str, value: int) -> None:
    if not hasattr(mesh, "attributes"):
        return
    attr = mesh.attributes.get(name)
    if attr is None:
        attr = mesh.attributes.new(name=name, type="INT", domain="POINT")
    values = np.full(len(attr.data), int(value), dtype=np.int32)
    attr.data.foreach_set("value", values)
    mesh.update()


def shift_int_resource_on_imported_models(node_objects: dict[int, bpy.types.Object], delta: int) -> None:
    for obj in node_objects.values():
        mesh = getattr(obj, "data", None)
        if mesh is None or not hasattr(mesh, "attributes"):
            continue
        attr = mesh.attributes.get("int_resource")
        if attr is None:
            continue
        values = np.empty(len(attr.data), dtype=np.float32)
        attr.data.foreach_get("value", values)
        values = np.rint(values).astype(np.int32) + int(delta)
        if getattr(attr, "data_type", "") == "INT":
            attr.data.foreach_set("value", values)
        else:
            attr.data.foreach_set("value", values.astype(np.float32))
        mesh.update()


def cleanup_models_collection(parent_collection: bpy.types.Collection, models_collection_name: str) -> None:
    existing = parent_collection.children.get(models_collection_name)
    if existing is None:
        return
    for obj in list(existing.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    parent_collection.children.unlink(existing)
    bpy.data.collections.remove(existing)


def cleanup_point_object(point_name: str) -> None:
    obj = bpy.data.objects.get(point_name)
    if obj is None:
        return
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def cleanup_node_group(node_group_name: str) -> None:
    node_group = bpy.data.node_groups.get(node_group_name)
    if node_group is None:
        return
    bpy.data.node_groups.remove(node_group, do_unlink=True)


def ensure_model_cache_collection(helpers_collection: bpy.types.Collection) -> bpy.types.Collection:
    cache_collection = helpers_collection.children.get("model_cache")
    if cache_collection is None:
        cache_collection = bpy.data.collections.new("model_cache")
        helpers_collection.children.link(cache_collection)
    return cache_collection


def get_cached_model_object(
    cache_collection: bpy.types.Collection,
    *,
    family: str,
    actual_filename: str,
) -> bpy.types.Object | None:
    for obj in cache_collection.objects:
        if obj.get("bV2_source_ply") == actual_filename and obj.get("bV2_model_family") == family:
            return obj
    return None


def cache_model_from_ply(
    cache_collection: bpy.types.Collection,
    *,
    family: str,
    actual_filename: str,
    filepath: Path,
) -> bpy.types.Object | None:
    log("IMPORT_START", family, actual_filename)
    t0 = time.perf_counter()
    imported = import_ply_object(filepath)
    if imported is None:
        log(f"PLY import did not return an object for {filepath}")
        return None

    imported.name = f"cache__{family}__{Path(actual_filename).stem}"
    if imported.data is not None:
        imported.data.name = f"{imported.name}_mesh"
    if imported.users_collection:
        for collection in list(imported.users_collection):
            collection.objects.unlink(imported)
    cache_collection.objects.link(imported)
    imported["bV2_source_ply"] = actual_filename
    imported["bV2_model_family"] = family
    imported.pass_index = MODEL_PASS_INDEX
    configure_imported_model_visibility(imported)
    ensure_model_material(imported)
    if imported.data is not None:
        ensure_constant_mesh_int_attribute(imported.data, "source-year", SOURCE_YEAR_DEFAULT)
    shift_int_resource_on_imported_models({0: imported}, 1)
    log("IMPORT_DONE", family, actual_filename, "seconds", round(time.perf_counter() - t0, 2))
    return imported


def instantiate_cached_model(
    cached_source: bpy.types.Object,
    *,
    target_name: str,
    models_collection: bpy.types.Collection,
) -> bpy.types.Object:
    instance = cached_source.copy()
    if cached_source.data is not None:
        instance.data = cached_source.data
    instance.name = target_name
    instance["bV2_source_ply"] = cached_source.get("bV2_source_ply", "")
    instance["bV2_model_family"] = cached_source.get("bV2_model_family", "")
    instance.pass_index = MODEL_PASS_INDEX
    configure_imported_model_visibility(instance)
    models_collection.objects.link(instance)
    return instance


def build_model_library(
    scene: bpy.types.Scene,
    df: pd.DataFrame,
    family: str,
    ply_folder: Path,
    models_collection: bpy.types.Collection,
) -> tuple[pd.DataFrame, dict[int, bpy.types.Object]]:
    adjusted = build_filename_requests(df, family)
    if family in {"trees", "poles"}:
        adjusted = remap_tree_ids_to_available_models(adjusted, ply_folder)
        adjusted = build_filename_requests(adjusted, family)

    fallback_map = build_filename_fallback_map(adjusted, family, ply_folder)
    unique_requested = sorted(adjusted["filename"].unique().tolist())
    instance_map = pd.Series(range(len(unique_requested)), index=unique_requested)
    adjusted["model_index"] = adjusted["filename"].map(instance_map).astype(np.int32)
    adjusted["resolved_filename"] = adjusted["filename"].map(lambda value: fallback_map.get(value, value))

    helpers_collection = find_collection_by_suffix(scene, "helpers")
    cache_collection = ensure_model_cache_collection(helpers_collection)
    node_objects: dict[int, bpy.types.Object] = {}
    for index, requested_filename in enumerate(unique_requested):
        actual_filename = fallback_map.get(requested_filename, requested_filename)
        filepath = ply_folder / actual_filename
        if not filepath.exists():
            log(f"Missing model file for {family}: {filepath}")
            continue
        cached_source = get_cached_model_object(
            cache_collection,
            family=family,
            actual_filename=actual_filename,
        )
        if cached_source is None:
            cached_source = cache_model_from_ply(
                cache_collection,
                family=family,
                actual_filename=actual_filename,
                filepath=filepath,
            )
        else:
            log("CACHE_HIT", family, actual_filename)
        if cached_source is None:
            continue
        target_name = make_model_object_name(index, family, actual_filename)
        instance = instantiate_cached_model(
            cached_source,
            target_name=target_name,
            models_collection=models_collection,
        )
        node_objects[index] = instance

    return adjusted, node_objects


def add_point_attribute(mesh: bpy.types.Mesh, name: str, attr_type: str, values: np.ndarray) -> None:
    attr = mesh.attributes.new(name=name, type=attr_type, domain="POINT")
    if attr_type == "FLOAT":
        attr.data.foreach_set("value", values.astype(np.float32))
    else:
        attr.data.foreach_set("value", values.astype(np.int32))
    if hasattr(attr, "is_runtime_only"):
        attr.is_runtime_only = False


def build_attribute_payloads(df: pd.DataFrame, family: str) -> dict[str, tuple[str, np.ndarray]]:
    payloads: dict[str, tuple[str, np.ndarray]] = {
        "rotation": ("FLOAT", pd.to_numeric(get_series_or_default(df, "rotateZ", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)),
        "structure_id": ("INT", pd.to_numeric(get_series_or_default(df, "structureID", -1), errors="coerce").fillna(-1).to_numpy(dtype=np.int32)),
        "precolonial": ("INT", convert_precolonial(get_series_or_default(df, "precolonial", False)).to_numpy(dtype=np.int32)),
        "size": ("INT", convert_size(get_series_or_default(df, "size", "")).to_numpy(dtype=np.int32)),
        "instanceID": ("INT", pd.to_numeric(get_series_or_default(df, "model_index", -1), errors="coerce").fillna(-1).to_numpy(dtype=np.int32)),
        "improvement": ("INT", convert_improvement(get_series_or_default(df, "Improvement", "")).to_numpy(dtype=np.int32)),
        "canopy_resistance": ("FLOAT", pd.to_numeric(get_series_or_default(df, "CanopyResistance", -1.0), errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32)),
        "node_id": ("INT", pd.to_numeric(get_series_or_default(df, "NodeID", -1), errors="coerce").fillna(-1).to_numpy(dtype=np.int32)),
        "bioEnvelopeType": ("INT", pd.to_numeric(get_series_or_default(df, "bioEnvelopeType", -1), errors="coerce").fillna(-1).to_numpy(dtype=np.int32)),
        "sim_Turns": ("INT", pd.to_numeric(get_series_or_default(df, "sim_Turns", -1), errors="coerce").fillna(-1).to_numpy(dtype=np.int32)),
        "source-year": ("INT", pd.to_numeric(get_series_or_default(df, "source-year", SOURCE_YEAR_DEFAULT), errors="coerce").fillna(SOURCE_YEAR_DEFAULT).to_numpy(dtype=np.int32)),
    }

    for proposal_column in TREE_PROPOSAL_COLUMNS:
        payloads[proposal_column] = (
            "INT",
            pd.to_numeric(get_series_or_default(df, proposal_column, 0), errors="coerce").fillna(0).to_numpy(dtype=np.int32),
        )

    if family in {"trees", "poles"}:
        payloads["control"] = ("INT", convert_control(get_series_or_default(df, "control", "")).to_numpy(dtype=np.int32))

    for column_name in TREE_RESOURCE_COLUMNS:
        payloads[column_name] = (
            "INT",
            pd.to_numeric(get_series_or_default(df, column_name, 0), errors="coerce").fillna(0).to_numpy(dtype=np.int32),
        )

    return payloads


def ensure_geometry_nodes_modifier(point_cloud: bpy.types.Object, modifier_name: str, node_group: bpy.types.NodeTree) -> bpy.types.Modifier:
    existing = point_cloud.modifiers.get(modifier_name)
    if existing is not None:
        point_cloud.modifiers.remove(existing)
    modifier = point_cloud.modifiers.new(name=modifier_name, type="NODES")
    modifier.node_group = node_group
    return modifier


def wire_instance_template(point_cloud: bpy.types.Object, models_collection: bpy.types.Collection, node_group_name: str) -> None:
    template_name = NODE_GROUP_NAMES["instancers"]
    template_group = bpy.data.node_groups.get(template_name)
    if template_group is None:
        raise RuntimeError(f"Could not find required node group {template_name!r}")

    cleanup_node_group(node_group_name)
    node_group = template_group.copy()
    node_group.name = node_group_name
    ensure_geometry_nodes_modifier(point_cloud, node_group_name, node_group)

    display_material_name = get_instancer_display_material_name()
    if display_material_name == str(GLOBAL_RULES.get("debug_source_years_material", "debug-source-years")):
        display_material = ensure_debug_source_years_material()
    else:
        display_material = bpy.data.materials.get(display_material_name)

    configured = False
    for node in node_group.nodes:
        if node.type == "COLLECTION_INFO":
            if hasattr(node, "inputs") and "Collection" in node.inputs:
                node.inputs["Collection"].default_value = models_collection
                configured = True
        if node.type == "SET_MATERIAL" and display_material is not None:
            if hasattr(node, "inputs") and "Material" in node.inputs:
                node.inputs["Material"].default_value = display_material
    if not configured:
        raise RuntimeError(f"Geometry node group {node_group_name!r} does not contain a configurable Collection Info node")


def find_collection_by_suffix(scene: bpy.types.Scene, suffix: str) -> bpy.types.Collection:
    queue = list(scene.collection.children)
    partial_match: bpy.types.Collection | None = None
    while queue:
        collection = queue.pop(0)
        role_name = str(collection.get("bV2_role", collection.name.split("::")[-1]))
        if role_name == suffix:
            return collection
        if role_name.startswith(suffix) or suffix.startswith(role_name):
            partial_match = partial_match or collection
        queue.extend(collection.children)
    if partial_match is not None:
        return partial_match
    raise RuntimeError(f"Could not find collection with suffix {suffix!r} in scene {scene.name!r}")


def build_family_state_instancer(
    scene: bpy.types.Scene,
    *,
    site: str,
    mode: str,
    year: int | None,
    family: str,
    state: str,
    df: pd.DataFrame,
) -> dict[str, object] | None:
    if df.empty:
        log(f"Skipping {family}/{state}: no rows")
        return None

    log("BUILD_STATE_START", "family=", family, "state=", state, "rows=", len(df))

    state_collection = find_collection_by_suffix(scene, STATE_TO_COLLECTION_ROLE[state])
    point_name = make_position_object_name(family, site, mode, state, year)
    models_collection_name = make_models_collection_name(family, site, mode, state, year)
    node_group_name = f"{point_name}__gn"

    cleanup_point_object(point_name)
    cleanup_models_collection(state_collection, models_collection_name)
    cleanup_node_group(node_group_name)

    models_collection = bpy.data.collections.new(models_collection_name)
    state_collection.children.link(models_collection)

    ply_folder = resolve_tree_ply_folder() if FAMILY_TO_PLY_ROOT_ROLE[family] == "tree" else resolve_log_ply_folder()
    adjusted_df, model_objects = build_model_library(scene, df, family, ply_folder, models_collection)
    log(
        "BUILD_STATE_MODELS_DONE",
        "family=",
        family,
        "state=",
        state,
        "rows=",
        len(adjusted_df),
        "models=",
        len(model_objects),
    )

    points = adjusted_df.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64)
    mesh = bpy.data.meshes.new(point_name)
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()

    point_cloud = bpy.data.objects.new(point_name, mesh)
    point_cloud.pass_index = POINT_CLOUD_PASS_INDEX
    state_collection.objects.link(point_cloud)

    for attr_name, (attr_type, values) in build_attribute_payloads(adjusted_df, family).items():
        add_point_attribute(mesh, attr_name, attr_type, values)

    wire_instance_template(point_cloud, models_collection, node_group_name)
    log(
        "BUILD_STATE_DONE",
        "family=",
        family,
        "state=",
        state,
        "point_name=",
        point_cloud.name,
        "points=",
        len(points),
    )

    return {
        "family": family,
        "state": state,
        "row_count": int(len(adjusted_df)),
        "model_count": int(len(model_objects)),
        "point_object": point_cloud.name,
        "models_collection": models_collection.name,
    }


def build_instancers(
    scene: bpy.types.Scene | None = None,
    *,
    site: str | None = None,
    mode: str | None = None,
    year: int | None = None,
) -> dict[str, object]:
    scene = scene or get_active_scene()
    runtime = get_runtime_config(scene)
    site = site or str(runtime["site"])
    mode = mode or str(runtime["mode"])
    if year is None:
        year = runtime["year"]  # type: ignore[assignment]

    is_baseline = mode == "baseline"

    log(f"Building bV2 instancers for scene={scene.name}, site={site}, mode={mode}, year={year}")

    if is_baseline:
        log("BUILD_INSTANCERS_LOAD_BASELINE_START")
        baseline_df = drop_ignored_tree_sizes(load_scenario_dataframe(site, "baseline", mode, year))
        log("BUILD_INSTANCERS_LOAD_BASELINE_DONE", "rows=", len(baseline_df))

        positive_df = baseline_df
        priority_df = derive_priority_dataframe(positive_df)

        frames_by_state = {
            "positive": split_family_dataframes(positive_df, site),
            "positive_priority": split_family_dataframes(priority_df, site),
        }
    else:
        log("BUILD_INSTANCERS_LOAD_POSITIVE_START")
        positive_df = drop_ignored_tree_sizes(load_scenario_dataframe(site, "positive", mode, year))
        log("BUILD_INSTANCERS_LOAD_POSITIVE_DONE", "rows=", len(positive_df))

        log("BUILD_INSTANCERS_LOAD_TRENDING_START")
        trending_df = drop_ignored_tree_sizes(load_scenario_dataframe(site, "trending", mode, year))
        log("BUILD_INSTANCERS_LOAD_TRENDING_DONE", "rows=", len(trending_df))

        log("BUILD_INSTANCERS_DERIVE_PRIORITY_START")
        priority_df = derive_priority_dataframe(positive_df)
        log("BUILD_INSTANCERS_DERIVE_PRIORITY_DONE", "rows=", len(priority_df))

        frames_by_state = {
            "positive": split_family_dataframes(positive_df, site),
            "positive_priority": split_family_dataframes(priority_df, site),
            "trending": split_family_dataframes(trending_df, site),
        }

    results: list[dict[str, object]] = []
    for state, family_frames in frames_by_state.items():
        log("BUILD_INSTANCERS_STATE_START", state, "families=", ",".join(sorted(family_frames.keys())))
        for family in get_instancer_families(site):
            family_df = family_frames.get(family, pd.DataFrame())
            log("BUILD_INSTANCERS_FAMILY_START", "state=", state, "family=", family, "rows=", len(family_df))
            result = build_family_state_instancer(
                scene,
                site=site,
                mode=mode,
                year=year,
                family=family,
                state=state,
                df=family_df,
            )
            if result is not None:
                results.append(result)
            log("BUILD_INSTANCERS_FAMILY_DONE", "state=", state, "family=", family)
        log("BUILD_INSTANCERS_STATE_DONE", state)

    summary = {
        "scene": scene.name,
        "site": site,
        "mode": mode,
        "year": year,
        "results": results,
        "positive_rows": int(len(positive_df)),
        "priority_rows": int(len(priority_df)),
        "trending_rows": 0 if is_baseline else int(len(trending_df)),
    }
    scene["bV2_instancers_built"] = True
    log(f"Instancer build summary: {summary}")
    return summary


def main() -> None:
    build_instancers()


if __name__ == "__main__":
    main()
