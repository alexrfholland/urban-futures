from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
import xarray as xr
from scipy.spatial import cKDTree

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
REPO_ROOT = CODE_ROOT.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.sim.voxel import voxel_a_helper_functions as a_helper_functions  # noqa: E402
from refactor_code.input_processing.tree_processing import a_resource_distributor_dataframes  # noqa: E402

from refactor_code.blender.bexport.proposal_framebuffers import build_blender_proposal_framebuffer_columns  # noqa: E402
from refactor_code.blender.bexport.proposal_framebuffers_vtk import build_blender_proposal_framebuffer_arrays  # noqa: E402
from refactor_code.paths import (  # noqa: E402
    engine_output_nodedf_path,
    engine_output_baseline_terrain_vtk_path,
    engine_output_baseline_trees_csv_path,
    engine_output_root,
    format_voxel_size,
    site_subset_dataset_path,
    site_world_reference_vtk_path,
    scenario_node_df_path,
    scenario_output_root,
    scenario_baseline_combined_vtk_path,
    scenario_baseline_resources_vtk_path,
    scenario_baseline_terrain_vtk_path,
    scenario_baseline_trees_csv_path,
)
from refactor_code.sim.setup.structure_ids import assign_baseline_tree_structure_ids  # noqa: E402
from refactor_code.sim.setup.constants import (  # noqa: E402
    COLONISE_FULL_GROUND,
    DECAY_FULL,
    RECRUIT_FULL,
    RELEASECONTROL_FULL,
)


APPROVED_CANONICAL_TEMPLATE_ROOT = (
    REPO_ROOT
    / "_data-refactored"
    / "model-inputs"
    / "tree_variants"
    / "template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen"
    / "trees"
)
DEFAULT_TEMPLATE_BASE_ROOT = REPO_ROOT / "_data-refactored" / "model-inputs" / "tree_libraries" / "base" / "trees"
DEFAULT_TEMPLATE_VARIANTS_ROOT = REPO_ROOT / "_data-refactored" / "model-inputs" / "tree_variants"
DEFAULT_V3_SCENARIO_OUTPUT_ROOT = REPO_ROOT / "data" / "revised" / "final-v3"
DEFAULT_V3_ENGINE_OUTPUT_ROOT = REPO_ROOT / "_data-refactored" / "v3engine_outputs"
BASELINE_DENSITY_CSV = REPO_ROOT / "data" / "csvs" / "tree-baseline-density.csv"
DEFAULT_TOTAL_DEADWOOD_TARGET_M3_PER_HA = 208.3
DEFAULT_WOOD_DENSITY_T_PER_M3 = 0.60
DEFAULT_FALLEN_SHARE = 0.70
DEFAULT_DECAYED_SHARE = 0.30
DEFAULT_SENESCING_SHARE = 0.65
DEFAULT_SNAG_SHARE = 0.30
DEFAULT_DEADWOOD_DBH_CM = 80.0
BASE_TEMPLATE_GEOMETRY_STEP_M = 0.1
BASE_TEMPLATE_LOOKUP_FILENAME = "template-edits_base_geometry_volume_lookup.csv"
BASELINE_RANDOM_SEED = 42
BASELINE_NODE_SCENARIO = "baseline"
BASELINE_NODE_YEAR = -180
ACTIVE_TREE_SIZES = {"small", "medium", "large", "senescing"}
DECAY_ACCEPTED_SIZES = {"senescing", "snag", "fallen", "decayed"}
RELEASE_ACCEPTED_SIZES = {"small", "medium", "large"}
DEPLOY_ACCEPTED_SIZES = {"fallen", "decayed"}
PROPOSAL_FAMILY_SPECS = [
    ("proposal-decay", "proposal_decay", "proposal_decayV3"),
    ("proposal-release-control", "proposal_release_control", "proposal_release_controlV3"),
    ("proposal-recruit", "proposal_recruit", "proposal_recruitV3"),
    ("proposal-colonise", "proposal_colonise", "proposal_coloniseV3"),
    ("proposal-deploy-structure", "proposal_deploy_structure", "proposal_deploy_structureV3"),
]


@dataclass(frozen=True)
class BaselineTargets:
    area_ha: float
    total_deadwood_target_m3_per_ha: float
    total_deadwood_target_m3: float
    fallen_share: float
    decayed_share: float
    fallen_target_m3: float
    decayed_target_m3: float
    total_deadwood_target_t: float
    total_deadwood_target_kg: float
    fallen_target_t: float
    fallen_target_kg: float
    decayed_target_t: float
    decayed_target_kg: float
    wood_density_t_per_m3: float


@dataclass(frozen=True)
class PreflightArtifacts:
    site: str
    voxel_size: float
    template_root: Path
    template_table_path: Path
    template_lookup_path: Path
    scenario_output_root: Path
    engine_output_root: Path
    candidate_table: pd.DataFrame
    targets: BaselineTargets
    candidate_csv_path: Path
    target_json_path: Path


@dataclass(frozen=True)
class GeneratedBaselineArtifacts:
    trees_csv_path: Path
    resource_vtk_path: Path
    terrain_vtk_path: Path
    combined_vtk_path: Path
    refactored_trees_csv_path: Path
    refactored_terrain_vtk_path: Path
    legacy_nodedf_path: Path
    refactored_nodedf_path: Path
    allocation_csv_path: Path
    metadata_json_path: Path
    candidate_csv_path: Path
    target_json_path: Path
    combined_polydata: pv.PolyData


def _v3_output_roots(
    run_output_root: str | Path | None = None,
) -> tuple[Path, Path]:
    if run_output_root is not None:
        os.environ["REFACTOR_RUN_OUTPUT_ROOT"] = str(Path(run_output_root).resolve())
    scenario_root = scenario_output_root().resolve()
    engine_root = engine_output_root().resolve()
    return scenario_root, engine_root


def require_tree_template_root(template_root: str | Path | None = None) -> Path:
    if template_root is not None:
        os.environ["TREE_TEMPLATE_ROOT"] = str(Path(template_root).resolve())

    configured = os.environ.get("TREE_TEMPLATE_ROOT")
    if not configured:
        from refactor_code.paths import tree_template_root
        resolved = tree_template_root().resolve()
    else:
        resolved = Path(configured).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"TREE_TEMPLATE_ROOT does not exist: {resolved}")
    return resolved


def _template_base_root() -> Path:
    override = os.environ.get("TREE_TEMPLATE_BASE_ROOT") or os.environ.get("BASE_TREE_TEMPLATES_ROOT")
    if override:
        return Path(override).resolve()
    return DEFAULT_TEMPLATE_BASE_ROOT.resolve()


def _input_subset_path(site: str, voxel_size: float | int) -> Path:
    return site_subset_dataset_path(site, voxel_size)


def _input_terrain_vtk_path(site: str) -> Path:
    return site_world_reference_vtk_path(site, "road")


def _baseline_support_dir(site: str) -> Path:
    output_dir = engine_output_baseline_trees_csv_path(site).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _support_file(site: str, voxel_size: float | int, stem: str, suffix: str) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _baseline_support_dir(site) / f"{site}_baseline_v3_{stem}_{voxel}{suffix}"


def _load_site_dataset(site: str, voxel_size: float | int) -> xr.Dataset:
    path = _input_subset_path(site, voxel_size)
    if not path.exists():
        raise FileNotFoundError(f"Subset dataset not found: {path}")
    return xr.open_dataset(path)


def _load_terrain_polydata(site: str) -> pv.PolyData:
    path = _input_terrain_vtk_path(site)
    if not path.exists():
        raise FileNotFoundError(f"Terrain VTK not found: {path}")
    return pv.read(path)


def calculate_site_area_ha(xarray_dataset: xr.Dataset) -> float:
    xmin, xmax, ymin, ymax = xarray_dataset.attrs["bounds"][:4]
    return float(((xmax - xmin) * (ymax - ymin)) / 10000.0)


def calculate_deadwood_targets(
    area_ha: float,
    *,
    total_deadwood_target_m3_per_ha: float = DEFAULT_TOTAL_DEADWOOD_TARGET_M3_PER_HA,
    fallen_share: float = DEFAULT_FALLEN_SHARE,
    decayed_share: float = DEFAULT_DECAYED_SHARE,
    wood_density_t_per_m3: float = DEFAULT_WOOD_DENSITY_T_PER_M3,
) -> BaselineTargets:
    if not math.isclose(fallen_share + decayed_share, 1.0, abs_tol=1e-9):
        raise ValueError("fallen_share + decayed_share must equal 1.0")

    total_deadwood_target_m3 = area_ha * total_deadwood_target_m3_per_ha
    fallen_target_m3 = total_deadwood_target_m3 * fallen_share
    decayed_target_m3 = total_deadwood_target_m3 * decayed_share
    total_deadwood_target_t = total_deadwood_target_m3 * wood_density_t_per_m3
    total_deadwood_target_kg = total_deadwood_target_t * 1000.0
    fallen_target_t = fallen_target_m3 * wood_density_t_per_m3
    fallen_target_kg = fallen_target_t * 1000.0
    decayed_target_t = decayed_target_m3 * wood_density_t_per_m3
    decayed_target_kg = decayed_target_t * 1000.0

    return BaselineTargets(
        area_ha=area_ha,
        total_deadwood_target_m3_per_ha=total_deadwood_target_m3_per_ha,
        total_deadwood_target_m3=total_deadwood_target_m3,
        fallen_share=fallen_share,
        decayed_share=decayed_share,
        fallen_target_m3=fallen_target_m3,
        decayed_target_m3=decayed_target_m3,
        total_deadwood_target_t=total_deadwood_target_t,
        total_deadwood_target_kg=total_deadwood_target_kg,
        fallen_target_t=fallen_target_t,
        fallen_target_kg=fallen_target_kg,
        decayed_target_t=decayed_target_t,
        decayed_target_kg=decayed_target_kg,
        wood_density_t_per_m3=wood_density_t_per_m3,
    )


def _load_voxel_template_table(template_root: Path, voxel_size: float | int) -> tuple[pd.DataFrame, Path]:
    voxel_prefix = format_voxel_size(voxel_size)
    candidates = [
        template_root / f"{voxel_prefix}_combined_voxel_templateDF.pkl",
        template_root / f"combined_voxelSize_{voxel_prefix}_templateDF.pkl",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_pickle(path), path
    raise FileNotFoundError(
        f"Could not find voxel template table for voxel_size={voxel_size} in {template_root}. "
        f"Tried: {[path.name for path in candidates]}"
    )


def _template_lookup_path(template_root: Path) -> Path:
    return template_root / BASE_TEMPLATE_LOOKUP_FILENAME


def build_base_deadwood_volume_lookup(
    *,
    template_root: Path,
    wood_density_t_per_m3: float = DEFAULT_WOOD_DENSITY_T_PER_M3,
) -> tuple[pd.DataFrame, Path]:
    candidates = [
        template_root / "template-library.selected-overrides.pkl",
        template_root / "template-library.overrides-applied.pkl",
    ]
    template_table_path = next((path for path in candidates if path.exists()), None)
    if template_table_path is None:
        raise FileNotFoundError(
            "Deadwood template table not found. "
            f"Tried: {[str(path) for path in candidates]}"
        )

    template_df = pd.read_pickle(template_table_path)

    rows: list[dict[str, Any]] = []
    for template_row in template_df.itertuples(index=False):
        size_value = str(template_row.size).strip()
        if size_value not in {"fallen", "decayed", "snag"}:
            continue

        template = template_row.template.copy()
        unique_xyz = template.loc[:, ["x", "y", "z"]].drop_duplicates().astype(float)
        axis_steps: dict[str, float] = {}
        for axis_name in ("x", "y", "z"):
            axis_values = np.sort(unique_xyz[axis_name].drop_duplicates().to_numpy(dtype=float))
            step_values = np.diff(axis_values)
            step_values = step_values[step_values > 1e-9]
            if len(step_values) == 0:
                raise ValueError(
                    f"Could not determine base geometry step for {size_value} "
                    f"{template_row.precolonial=} {template_row.control=} {template_row.tree_id=}"
                )
            axis_steps[axis_name] = float(np.min(np.round(step_values, 10)))

        base_cell_volume_m3 = axis_steps["x"] * axis_steps["y"] * axis_steps["z"]
        unique_xyz_cells = int(unique_xyz.shape[0])
        volume_m3 = float(unique_xyz_cells) * base_cell_volume_m3
        biomass_t = volume_m3 * wood_density_t_per_m3
        biomass_kg = biomass_t * 1000.0

        rows.append(
            {
                "precolonial": bool(template_row.precolonial),
                "size": size_value,
                "control": str(template_row.control).strip(),
                "tree_id": int(template_row.tree_id),
                "base_x_step_m": axis_steps["x"],
                "base_y_step_m": axis_steps["y"],
                "base_z_step_m": axis_steps["z"],
                "base_cell_volume_m3": base_cell_volume_m3,
                "base_geometry_unique_xyz_cells": unique_xyz_cells,
                "template_volume_m3": volume_m3,
                "template_biomass_t": biomass_t,
                "template_biomass_kg": biomass_kg,
                "wood_density_t_per_m3": wood_density_t_per_m3,
                "all_axes_base_0.1": bool(
                    math.isclose(axis_steps["x"], BASE_TEMPLATE_GEOMETRY_STEP_M, abs_tol=1e-9)
                    and math.isclose(axis_steps["y"], BASE_TEMPLATE_GEOMETRY_STEP_M, abs_tol=1e-9)
                    and math.isclose(axis_steps["z"], BASE_TEMPLATE_GEOMETRY_STEP_M, abs_tol=1e-9)
                ),
            }
        )

    candidate_table = pd.DataFrame(rows).sort_values(
        ["size", "template_volume_m3", "precolonial", "tree_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    if candidate_table.empty:
        raise ValueError(f"No fallen/decayed candidates found in template table {template_table_path}")

    lookup_path = _template_lookup_path(template_root)
    candidate_table.to_csv(lookup_path, index=False)
    return candidate_table, lookup_path


def build_deadwood_candidate_table(
    *,
    template_root: Path,
    voxel_size: float | int,
    wood_density_t_per_m3: float = DEFAULT_WOOD_DENSITY_T_PER_M3,
) -> tuple[pd.DataFrame, Path, Path]:
    template_df, voxel_template_table_path = _load_voxel_template_table(template_root, voxel_size)
    volume_lookup_df, template_lookup_path = build_base_deadwood_volume_lookup(
        template_root=template_root,
        wood_density_t_per_m3=wood_density_t_per_m3,
    )

    voxel_key_df = (
        template_df.loc[template_df["size"].isin(["fallen", "decayed"]), ["precolonial", "size", "control", "tree_id"]]
        .drop_duplicates()
        .copy()
    )
    candidate_table = volume_lookup_df.loc[
        volume_lookup_df["size"].isin(["fallen", "decayed"]) & volume_lookup_df["precolonial"].eq(True)
    ].merge(
        voxel_key_df,
        on=["precolonial", "size", "control", "tree_id"],
        how="inner",
    )
    candidate_table = candidate_table.sort_values(
        ["size", "template_volume_m3", "precolonial", "tree_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    if candidate_table.empty:
        raise ValueError(f"No fallen/decayed candidates found for voxel table {voxel_template_table_path}")
    return candidate_table, voxel_template_table_path, template_lookup_path


def write_preflight_artifacts(
    *,
    site: str,
    voxel_size: float | int,
    template_root: Path,
    template_table_path: Path,
    template_lookup_path: Path,
    scenario_output_root: Path,
    engine_output_root: Path,
    candidate_table: pd.DataFrame,
    targets: BaselineTargets,
) -> tuple[Path, Path]:
    candidate_csv_path = _support_file(site, voxel_size, "deadwood_candidates", ".csv")
    target_json_path = _support_file(site, voxel_size, "deadwood_targets", ".json")

    candidate_table.to_csv(candidate_csv_path, index=False)
    target_payload = {
        "site": site,
        "voxel_size": float(voxel_size),
        "tree_template_root": str(template_root),
        "template_table_path": str(template_table_path),
        "template_lookup_path": str(template_lookup_path),
        "scenario_output_root": str(scenario_output_root),
        "engine_output_root": str(engine_output_root),
        "targets": asdict(targets),
    }
    target_json_path.write_text(json.dumps(target_payload, indent=2), encoding="utf-8")
    return candidate_csv_path, target_json_path


def run_preflight(
    *,
    site: str,
    voxel_size: float | int = 1,
    template_root: str | Path | None = None,
    run_output_root: str | Path | None = None,
    total_deadwood_target_m3_per_ha: float = DEFAULT_TOTAL_DEADWOOD_TARGET_M3_PER_HA,
    fallen_share: float = DEFAULT_FALLEN_SHARE,
    decayed_share: float = DEFAULT_DECAYED_SHARE,
    wood_density_t_per_m3: float = DEFAULT_WOOD_DENSITY_T_PER_M3,
) -> PreflightArtifacts:
    resolved_template_root = require_tree_template_root(template_root)
    resolved_scenario_root, resolved_engine_root = _v3_output_roots(
        run_output_root=run_output_root,
    )

    site_dataset = _load_site_dataset(site, voxel_size)
    area_ha = calculate_site_area_ha(site_dataset)
    targets = calculate_deadwood_targets(
        area_ha,
        total_deadwood_target_m3_per_ha=total_deadwood_target_m3_per_ha,
        fallen_share=fallen_share,
        decayed_share=decayed_share,
        wood_density_t_per_m3=wood_density_t_per_m3,
    )
    candidate_table, template_table_path, template_lookup_path = build_deadwood_candidate_table(
        template_root=resolved_template_root,
        voxel_size=voxel_size,
        wood_density_t_per_m3=wood_density_t_per_m3,
    )
    candidate_csv_path, target_json_path = write_preflight_artifacts(
        site=site,
        voxel_size=voxel_size,
        template_root=resolved_template_root,
        template_table_path=template_table_path,
        template_lookup_path=template_lookup_path,
        scenario_output_root=resolved_scenario_root,
        engine_output_root=resolved_engine_root,
        candidate_table=candidate_table,
        targets=targets,
    )
    site_dataset.close()

    return PreflightArtifacts(
        site=site,
        voxel_size=float(voxel_size),
        template_root=resolved_template_root,
        template_table_path=template_table_path,
        template_lookup_path=template_lookup_path,
        scenario_output_root=resolved_scenario_root,
        engine_output_root=resolved_engine_root,
        candidate_table=candidate_table,
        targets=targets,
        candidate_csv_path=candidate_csv_path,
        target_json_path=target_json_path,
    )


def _get_ground_voxels(xarray_dataset: xr.Dataset, terrain_polydata: pv.PolyData) -> pd.DataFrame:
    voxel_df = pd.DataFrame(
        {
            "voxel_I": xarray_dataset["voxel_I"].values,
            "voxel_J": xarray_dataset["voxel_J"].values,
            "voxel_K": xarray_dataset["voxel_K"].values,
            "centroid_x": xarray_dataset["centroid_x"].values,
            "centroid_y": xarray_dataset["centroid_y"].values,
            "centroid_z": xarray_dataset["centroid_z"].values,
        }
    )
    voxel_df["isGround"] = False

    terrain_points = terrain_polydata.points
    tree = cKDTree(terrain_points, leafsize=16)
    voxel_points = voxel_df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(dtype=float)
    distances, _ = tree.query(
        voxel_points,
        k=1,
        eps=0.1,
        distance_upper_bound=1.0,
    )
    voxel_df.loc[distances <= 1.0, "isGround"] = True
    return voxel_df


def _assign_positions(baseline_tree_df: pd.DataFrame, terrain_df: pd.DataFrame, seed: int = BASELINE_RANDOM_SEED) -> pd.DataFrame:
    np.random.seed(seed)
    placed_df = baseline_tree_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    ground_df = terrain_df.loc[terrain_df["isGround"]].reset_index()

    if len(placed_df) > len(ground_df):
        raise ValueError(f"More baseline objects ({len(placed_df)}) than ground voxels ({len(ground_df)})")

    selected_rows = ground_df.sample(n=len(placed_df), random_state=seed).reset_index(drop=True)
    placed_df["voxelID"] = selected_rows["index"].to_numpy(dtype=int)
    placed_df["x"] = selected_rows["centroid_x"].to_numpy(dtype=float)
    placed_df["y"] = selected_rows["centroid_y"].to_numpy(dtype=float)
    placed_df["z"] = selected_rows["centroid_z"].to_numpy(dtype=float)
    placed_df["voxel_I"] = selected_rows["voxel_I"].to_numpy(dtype=int)
    placed_df["voxel_J"] = selected_rows["voxel_J"].to_numpy(dtype=int)
    placed_df["voxel_K"] = selected_rows["voxel_K"].to_numpy(dtype=int)
    return placed_df


def _calculate_useful_life_expectancy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    growth_factor = float(np.mean([0.37, 0.51]))
    df["age"] = df["diameter_breast_height"] / growth_factor

    max_dbh_age = 80.0 / growth_factor
    senescing_addition = 600.0 - max_dbh_age

    senescing_mask = df["size"].eq("senescing")
    if senescing_mask.any():
        rng = np.random.default_rng(BASELINE_RANDOM_SEED)
        df.loc[senescing_mask, "age"] = df.loc[senescing_mask, "age"].to_numpy(dtype=float) + rng.uniform(
            0.0,
            senescing_addition,
            senescing_mask.sum(),
        )

    snag_mask = df["size"].eq("snag")
    if snag_mask.any():
        rng = np.random.default_rng(BASELINE_RANDOM_SEED + 1)
        df.loc[snag_mask, "age"] = rng.uniform(450.0, 500.0, snag_mask.sum())

    fallen_mask = df["size"].eq("fallen")
    if fallen_mask.any():
        rng = np.random.default_rng(BASELINE_RANDOM_SEED + 2)
        df.loc[fallen_mask, "age"] = rng.uniform(500.0, 600.0, fallen_mask.sum())

    decayed_mask = df["size"].eq("decayed")
    if decayed_mask.any():
        rng = np.random.default_rng(BASELINE_RANDOM_SEED + 3)
        df.loc[decayed_mask, "age"] = rng.uniform(600.0, 700.0, decayed_mask.sum())

    df["useful_life_expectancy"] = 600.0 - df["age"]
    return df


def _get_terrain_polydata(terrain_df: pd.DataFrame) -> pv.PolyData:
    ground_points = terrain_df.loc[terrain_df["isGround"]]
    terrain_points = np.column_stack(
        (
            ground_points["centroid_x"].to_numpy(dtype=float),
            ground_points["centroid_y"].to_numpy(dtype=float),
            ground_points["centroid_z"].to_numpy(dtype=float),
        )
    )
    terrain_cloud = pv.PolyData(terrain_points)

    bounds = terrain_cloud.bounds
    x_min, x_max, y_min, y_max, _, _ = bounds
    noise = pv.perlin_noise(0.02, (2, 2, 1), (0, 0, 0))
    noise_values = np.zeros(len(terrain_points), dtype=float)

    x_norm = (terrain_points[:, 0] - x_min) / (x_max - x_min)
    y_norm = (terrain_points[:, 1] - y_min) / (y_max - y_min)

    for index, (x_value, y_value) in enumerate(zip(x_norm, y_norm, strict=False)):
        noise_values[index] = noise.EvaluateFunction(x_value * 10.0, y_value * 10.0, 0.0)

    noise_values = (noise_values - noise_values.min()) / (noise_values.max() - noise_values.min())
    terrain_cloud.point_data["noise"] = noise_values
    return terrain_cloud


def _combine_polydata(resource_poly: pv.PolyData, terrain_poly: pv.PolyData) -> pv.PolyData:
    resource_poly = resource_poly.copy(deep=True)
    terrain_poly = terrain_poly.copy(deep=True)

    resource_poly.point_data["search_bioavailable"] = np.full(resource_poly.n_points, "arboreal", dtype="<U20")
    terrain_poly.point_data["search_bioavailable"] = np.full(terrain_poly.n_points, "low-vegetation", dtype="<U20")

    for attr in ["useful_life_expectancy", "precolonial", "size", "control"]:
        if attr in resource_poly.point_data:
            resource_poly.point_data[f"forest_{attr}"] = resource_poly.point_data[attr].copy()
            del resource_poly.point_data[attr]

    combined_poly = resource_poly.copy()
    terrain_poly = a_helper_functions.initialize_polydata_variables_generic_auto(terrain_poly, resource_poly)
    resource_poly = a_helper_functions.initialize_polydata_variables_generic_auto(resource_poly, terrain_poly)
    combined_poly = combined_poly.append_polydata(terrain_poly)
    return combined_poly


def _proposal_alias_column(name: str) -> str:
    return name.replace("-", "_")


def _append_proposal_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliased = df.copy()
    for family, _, _ in PROPOSAL_FAMILY_SPECS:
        for suffix in ("decision", "intervention"):
            source = f"{family}_{suffix}"
            aliased[_proposal_alias_column(source)] = aliased[source]
    return aliased


def _apply_baseline_proposal_rules(baseline_tree_df: pd.DataFrame, site_dataset: xr.Dataset) -> pd.DataFrame:
    df = baseline_tree_df.copy()
    size_lower = df["size"].astype(str).str.lower()

    resistance = (
        site_dataset["analysis_combined_resistance"].values
        if "analysis_combined_resistance" in site_dataset
        else None
    )
    if resistance is not None:
        voxel_ids = pd.to_numeric(df["voxelID"], errors="coerce").fillna(-1).astype(int).to_numpy()
        canopy_resistance = np.full(len(df), np.nan, dtype=float)
        valid_mask = (voxel_ids >= 0) & (voxel_ids < len(resistance))
        canopy_resistance[valid_mask] = resistance[voxel_ids[valid_mask]]
        df["CanopyResistance"] = canopy_resistance
    elif "CanopyResistance" not in df.columns:
        df["CanopyResistance"] = np.nan

    df["CanopyArea"] = 1.0
    df["sim_NodesVoxels"] = 1.0
    df["sim_NodesArea"] = 1.0
    df["isNewTree"] = False
    df["isRewildedTree"] = False
    df["hasbeenReplanted"] = False
    df["unmanagedCount"] = 0
    df["action"] = "None"
    df["nodeType"] = "tree"
    df["under-node-treatment"] = "paved"
    df["replacement_reason"] = "none"
    df["proposal-release-control_target_years"] = 0
    df["proposal-release-control_years"] = 0
    df["control_reached"] = df["control"].astype(str)
    df["lifecycle_state"] = size_lower.to_numpy()
    df["fallen_since_year"] = -1
    df["fallen_decay_after_years"] = -1
    df["decayed_since_year"] = -1
    df["decayed_remove_after_years"] = -1
    df["nodeTypeInt"] = 0

    df["proposal-decay_decision"] = np.where(
        size_lower.isin(DECAY_ACCEPTED_SIZES),
        "proposal-decay_accepted",
        "not-assessed",
    )
    df["proposal-release-control_decision"] = np.where(
        size_lower.isin(RELEASE_ACCEPTED_SIZES),
        "proposal-release-control_accepted",
        "not-assessed",
    )
    df["proposal-recruit_decision"] = "not-assessed"
    df["proposal-colonise_decision"] = "not-assessed"
    df["proposal-deploy-structure_decision"] = np.where(
        size_lower.isin(DEPLOY_ACCEPTED_SIZES),
        "proposal-deploy-structure_accepted",
        "not-assessed",
    )

    for family, _, _ in PROPOSAL_FAMILY_SPECS:
        df[f"{family}_intervention"] = "none"

    return df


def _build_baseline_node_df(baseline_tree_df: pd.DataFrame, site_dataset: xr.Dataset) -> pd.DataFrame:
    node_df = _apply_baseline_proposal_rules(baseline_tree_df, site_dataset)
    node_df = _append_proposal_alias_columns(node_df)
    blender_columns = build_blender_proposal_framebuffer_columns(node_df)
    for column_name in blender_columns.columns:
        node_df[column_name] = blender_columns[column_name]
    return node_df



def _annotate_baseline_combined_polydata(
    combined_poly: pv.PolyData,
    *,
    node_df: pd.DataFrame,
    resource_df: pd.DataFrame,
    terrain_poly: pv.PolyData,
) -> pv.PolyData:
    combined_poly = combined_poly.copy(deep=True)
    resource_count = len(resource_df)
    terrain_count = terrain_poly.n_points
    total_points = combined_poly.n_points

    if resource_count + terrain_count != total_points:
        raise ValueError(
            "Baseline combined polydata size mismatch: "
            f"{resource_count} resource + {terrain_count} terrain != {total_points} total"
        )

    # --- read forest_size from the combined polydata ---
    forest_size = np.asarray(combined_poly.point_data.get("forest_size", np.full(total_points, "nan", dtype="<U20")), dtype="<U20")
    forest_size_lower = np.char.lower(forest_size)

    # --- terrain mask (terrain points are appended after resource points) ---
    terrain_mask = np.zeros(total_points, dtype=bool)
    terrain_mask[resource_count:] = True

    # --- build masks ---
    decay_mask_arboreal = np.isin(forest_size_lower, ["senescing", "snag", "fallen", "decayed"])
    decay_arboreal_points = combined_poly.points[decay_mask_arboreal]
    if len(decay_arboreal_points) > 0:
        decay_tree = cKDTree(decay_arboreal_points[:, :2])
        terrain_xy = combined_poly.points[terrain_mask, :2]
        decay_distances, _ = decay_tree.query(terrain_xy, k=1)
        decay_mask_ground = np.zeros(total_points, dtype=bool)
        decay_mask_ground[terrain_mask] = decay_distances <= 2.0
    else:
        decay_mask_ground = np.zeros(total_points, dtype=bool)
    decay_mask = decay_mask_arboreal | decay_mask_ground

    release_control_mask = np.isin(forest_size_lower, ["small", "medium", "large"])

    active_trees = node_df[node_df["size"].isin(["small", "medium", "large"])]
    if len(active_trees) > 0:
        active_tree_xy = active_trees[["x", "y"]].to_numpy()
        active_tree_kd = cKDTree(active_tree_xy)
        terrain_xy = combined_poly.points[terrain_mask, :2]
        recruit_distances, _ = active_tree_kd.query(terrain_xy, k=1)
        recruit_mask = np.zeros(total_points, dtype=bool)
        recruit_mask[terrain_mask] = recruit_distances > 1.5
    else:
        recruit_mask = terrain_mask.copy()

    colonise_mask = terrain_mask

    # --- build numpy arrays, then assign to point_data in one shot ---
    decay_decision = np.full(total_points, "not-assessed", dtype="<U64")
    decay_intervention = np.full(total_points, "none", dtype="<U64")
    decay_decision[decay_mask] = "proposal-decay_accepted"
    decay_intervention[decay_mask] = DECAY_FULL

    release_decision = np.full(total_points, "not-assessed", dtype="<U64")
    release_intervention = np.full(total_points, "none", dtype="<U64")
    release_decision[release_control_mask] = "proposal-release-control_accepted"
    release_intervention[release_control_mask] = RELEASECONTROL_FULL

    recruit_decision = np.full(total_points, "not-assessed", dtype="<U64")
    recruit_intervention = np.full(total_points, "none", dtype="<U64")
    recruit_decision[recruit_mask] = "proposal-recruit_accepted"
    recruit_intervention[recruit_mask] = RECRUIT_FULL

    colonise_decision = np.full(total_points, "not-assessed", dtype="<U64")
    colonise_intervention = np.full(total_points, "none", dtype="<U64")
    colonise_decision[colonise_mask] = "proposal-colonise_accepted"
    colonise_intervention[colonise_mask] = COLONISE_FULL_GROUND

    combined_poly.point_data["proposal_decayV4"] = decay_decision
    combined_poly.point_data["proposal_decayV4_intervention"] = decay_intervention
    combined_poly.point_data["proposal_release_controlV4"] = release_decision
    combined_poly.point_data["proposal_release_controlV4_intervention"] = release_intervention
    combined_poly.point_data["proposal_recruitV4"] = recruit_decision
    combined_poly.point_data["proposal_recruitV4_intervention"] = recruit_intervention
    combined_poly.point_data["proposal_coloniseV4"] = colonise_decision
    combined_poly.point_data["proposal_coloniseV4_intervention"] = colonise_intervention
    combined_poly.point_data["proposal_deploy_structureV4"] = np.full(total_points, "not-assessed", dtype="<U64")
    combined_poly.point_data["proposal_deploy_structureV4_intervention"] = np.full(total_points, "none", dtype="<U64")

    # --- blender framebuffers ---
    blender_arrays = build_blender_proposal_framebuffer_arrays(combined_poly.point_data)
    for name, values in blender_arrays.items():
        combined_poly.point_data[name] = values

    return combined_poly


def _build_live_baseline_tree_df(area_ha: float) -> pd.DataFrame:
    baseline_densities = pd.read_csv(BASELINE_DENSITY_CSV).rename(columns={"Size": "diameter_breast_height"})
    rows: list[pd.DataFrame] = []
    for density_row in baseline_densities.itertuples(index=False):
        dbh = float(density_row.diameter_breast_height)
        flat_density = float(density_row.Flat)
        num_points = int((area_ha * flat_density) / 0.1)
        if num_points <= 0:
            continue

        rows.append(
            pd.DataFrame(
                {
                    "x": np.nan,
                    "y": np.nan,
                    "z": np.nan,
                    "voxelID": np.nan,
                    "voxel_I": -1,
                    "voxel_J": -1,
                    "voxel_K": -1,
                    "precolonial": True,
                    "control": "reserve-tree",
                    "diameter_breast_height": dbh,
                    "tree_id": -1,
                    "useful_life_expectancy": -1.0,
                    "baseline_volume_m3": np.nan,
                    "baseline_biomass_kg": np.nan,
                },
                index=np.arange(num_points),
            )
        )

    baseline_tree_df = pd.concat(rows, ignore_index=True)
    baseline_tree_df["size"] = "small"
    baseline_tree_df.loc[baseline_tree_df["diameter_breast_height"] >= 50.0, "size"] = "medium"
    baseline_tree_df.loc[baseline_tree_df["diameter_breast_height"] >= 80.0, "size"] = "large"
    return baseline_tree_df


def _apply_senescing_and_snag(
    baseline_tree_df: pd.DataFrame,
    *,
    senescing_share: float = DEFAULT_SENESCING_SHARE,
    snag_share: float = DEFAULT_SNAG_SHARE,
    seed: int = BASELINE_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    updated_df = baseline_tree_df.copy()
    large_indices = updated_df.index[updated_df["size"].eq("large")].to_numpy()
    live_large_total = int(len(large_indices))
    if live_large_total == 0:
        empty_snags = updated_df.iloc[0:0].copy()
        return updated_df, empty_snags, 0

    rng = np.random.default_rng(seed)
    senescing_count = int(round(live_large_total * senescing_share))
    if senescing_count > 0:
        chosen_indices = rng.choice(large_indices, size=senescing_count, replace=False)
        updated_df.loc[chosen_indices, "size"] = "senescing"
        updated_df.loc[chosen_indices, "control"] = "improved-tree"

    snag_count = int(round(live_large_total * snag_share))
    large_source_df = baseline_tree_df.loc[large_indices].copy()
    if snag_count <= 0:
        snag_df = large_source_df.iloc[0:0].copy()
    else:
        donor_positions = rng.choice(
            np.arange(len(large_source_df)),
            size=snag_count,
            replace=snag_count > len(large_source_df),
        )
        snag_df = large_source_df.iloc[donor_positions].copy().reset_index(drop=True)
        snag_df["size"] = "snag"
        snag_df["control"] = "improved-tree"

    return updated_df, snag_df, live_large_total


def _solve_minimum_overshoot_allocation(weights: list[int], target_weight: int) -> list[int]:
    if target_weight <= 0:
        return []
    if not weights:
        raise ValueError("No candidate weights were provided")

    max_weight = max(weights)
    limit = target_weight + max_weight
    inf = limit + 10_000
    best_item_counts = [inf] * (limit + 1)
    previous_total = [-1] * (limit + 1)
    previous_choice = [-1] * (limit + 1)
    best_item_counts[0] = 0

    for running_total in range(limit + 1):
        if best_item_counts[running_total] == inf:
            continue
        for candidate_index, weight in enumerate(weights):
            next_total = running_total + weight
            if next_total > limit:
                continue
            candidate_item_count = best_item_counts[running_total] + 1
            if candidate_item_count < best_item_counts[next_total]:
                best_item_counts[next_total] = candidate_item_count
                previous_total[next_total] = running_total
                previous_choice[next_total] = candidate_index

    chosen_total = next((value for value in range(target_weight, limit + 1) if best_item_counts[value] != inf), None)
    if chosen_total is None:
        raise ValueError(f"Could not allocate a combination to reach target weight {target_weight}")

    chosen_indices: list[int] = []
    current_total = chosen_total
    while current_total > 0:
        chosen_indices.append(previous_choice[current_total])
        current_total = previous_total[current_total]
    chosen_indices.reverse()
    return chosen_indices


def _generate_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]

    compositions: list[tuple[int, ...]] = []
    for value in range(total + 1):
        for remainder in _generate_compositions(total - value, parts - 1):
            compositions.append((value,) + remainder)
    return compositions


def _solve_even_distribution_allocation(
    weights: list[int],
    target_weight: int,
    *,
    max_extra_items: int = 12,
) -> list[int]:
    if target_weight <= 0:
        return []
    if not weights:
        raise ValueError("No candidate weights were provided")

    min_total_count = int(math.ceil(target_weight / max(weights)))
    search_limit = min_total_count + max_extra_items
    best_solution: tuple[int, int, float, int, int, tuple[int, ...]] | None = None

    for total_count in range(min_total_count, search_limit + 1):
        for counts in _generate_compositions(total_count, len(weights)):
            total = sum(weight * count for weight, count in zip(weights, counts, strict=False))
            if total < target_weight:
                continue

            missing_templates = sum(count == 0 for count in counts)
            imbalance = max(counts) - min(counts)
            mean_count = total_count / len(weights)
            variance = sum((count - mean_count) ** 2 for count in counts)
            overshoot = total - target_weight
            candidate = (missing_templates, imbalance, variance, overshoot, total_count, counts)
            if best_solution is None or candidate < best_solution:
                best_solution = candidate

        if best_solution is not None and best_solution[0] == 0 and best_solution[1] <= 1:
            break

    if best_solution is None:
        raise ValueError(f"Could not find an even allocation to reach target weight {target_weight}")

    counts = best_solution[-1]
    chosen_indices: list[int] = []
    active = True
    round_robin_counts = list(counts)
    while active:
        active = False
        for candidate_index, count in enumerate(round_robin_counts):
            if count <= 0:
                continue
            chosen_indices.append(candidate_index)
            round_robin_counts[candidate_index] -= 1
            active = True
    return chosen_indices


def allocate_deadwood_candidates(
    candidate_table: pd.DataFrame,
    *,
    size_value: str,
    target_m3: float,
) -> pd.DataFrame:
    size_candidates = candidate_table.loc[candidate_table["size"].eq(size_value)].copy()
    if size_candidates.empty:
        raise ValueError(f"No candidates available for size={size_value}")

    size_candidates = size_candidates.sort_values(
        ["template_volume_m3", "precolonial", "tree_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    target_cells = int(math.ceil(target_m3 / (BASE_TEMPLATE_GEOMETRY_STEP_M ** 3)))
    weights = size_candidates["base_geometry_unique_xyz_cells"].astype(int).tolist()
    if size_value == "fallen":
        chosen_indices = _solve_even_distribution_allocation(weights, target_cells)
    else:
        chosen_indices = _solve_minimum_overshoot_allocation(weights, target_cells)
    allocation_df = size_candidates.iloc[chosen_indices].copy().reset_index(drop=True)
    allocation_df.insert(0, "allocation_sequence", np.arange(1, len(allocation_df) + 1))
    allocation_df["target_m3"] = float(target_m3)
    allocation_df["target_base_geometry_cells"] = target_cells
    allocation_df["allocated_base_geometry_cells_cumulative"] = allocation_df["base_geometry_unique_xyz_cells"].cumsum()
    allocation_df["allocated_biomass_kg_cumulative"] = allocation_df["template_biomass_kg"].cumsum()
    allocation_df["allocated_volume_m3_cumulative"] = allocation_df["template_volume_m3"].cumsum()
    return allocation_df


def _allocation_rows_to_tree_df(allocation_df: pd.DataFrame) -> pd.DataFrame:
    if allocation_df.empty:
        return pd.DataFrame(
            columns=[
                "x",
                "y",
                "z",
                "voxelID",
                "voxel_I",
                "voxel_J",
                "voxel_K",
                "precolonial",
                "control",
                "diameter_breast_height",
                "tree_id",
                "useful_life_expectancy",
                "size",
                "baseline_volume_m3",
                "baseline_biomass_kg",
                "baseline_allocation_sequence",
            ]
        )

    return pd.DataFrame(
        {
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
            "voxelID": np.nan,
            "voxel_I": -1,
            "voxel_J": -1,
            "voxel_K": -1,
            "precolonial": allocation_df["precolonial"].astype(bool).to_numpy(),
            "control": allocation_df["control"].astype(str).to_numpy(),
            "diameter_breast_height": DEFAULT_DEADWOOD_DBH_CM,
            "tree_id": allocation_df["tree_id"].astype(int).to_numpy(),
            "useful_life_expectancy": -1.0,
            "size": allocation_df["size"].astype(str).to_numpy(),
            "baseline_volume_m3": allocation_df["template_volume_m3"].astype(float).to_numpy(),
            "baseline_biomass_kg": allocation_df["template_biomass_kg"].astype(float).to_numpy(),
            "baseline_allocation_sequence": allocation_df["allocation_sequence"].astype(int).to_numpy(),
        }
    )


def generate_baseline(
    *,
    site: str,
    voxel_size: float | int = 1,
    template_root: str | Path | None = None,
    run_output_root: str | Path | None = None,
    total_deadwood_target_m3_per_ha: float = DEFAULT_TOTAL_DEADWOOD_TARGET_M3_PER_HA,
    fallen_share: float = DEFAULT_FALLEN_SHARE,
    decayed_share: float = DEFAULT_DECAYED_SHARE,
    senescing_share: float = DEFAULT_SENESCING_SHARE,
    snag_share: float = DEFAULT_SNAG_SHARE,
    wood_density_t_per_m3: float = DEFAULT_WOOD_DENSITY_T_PER_M3,
    visualize: bool = False,
) -> GeneratedBaselineArtifacts:
    preflight = run_preflight(
        site=site,
        voxel_size=voxel_size,
        template_root=template_root,
        run_output_root=run_output_root,
        total_deadwood_target_m3_per_ha=total_deadwood_target_m3_per_ha,
        fallen_share=fallen_share,
        decayed_share=decayed_share,
        wood_density_t_per_m3=wood_density_t_per_m3,
    )

    site_dataset = _load_site_dataset(site, voxel_size)
    terrain_polydata = _load_terrain_polydata(site)
    terrain_df = _get_ground_voxels(site_dataset, terrain_polydata)

    live_baseline_df = _build_live_baseline_tree_df(preflight.targets.area_ha)
    live_baseline_df, snag_df, live_large_total = _apply_senescing_and_snag(
        live_baseline_df,
        senescing_share=senescing_share,
        snag_share=snag_share,
        seed=BASELINE_RANDOM_SEED,
    )
    fallen_allocation_df = allocate_deadwood_candidates(
        preflight.candidate_table,
        size_value="fallen",
        target_m3=preflight.targets.fallen_target_m3,
    )
    decayed_allocation_df = allocate_deadwood_candidates(
        preflight.candidate_table,
        size_value="decayed",
        target_m3=preflight.targets.decayed_target_m3,
    )

    deadwood_df = pd.concat(
        [
            _allocation_rows_to_tree_df(fallen_allocation_df),
            _allocation_rows_to_tree_df(decayed_allocation_df),
        ],
        ignore_index=True,
    )

    baseline_tree_df = pd.concat([live_baseline_df, snag_df, deadwood_df], ignore_index=True)
    baseline_tree_df["tree_number"] = np.arange(len(baseline_tree_df), dtype=int)
    baseline_tree_df["NodeID"] = baseline_tree_df["tree_number"]
    baseline_tree_df["structureID"] = np.nan
    rng = np.random.default_rng(BASELINE_RANDOM_SEED)
    baseline_tree_df["rotateZ"] = rng.uniform(0.0, 360.0, len(baseline_tree_df))

    baseline_tree_df = _assign_positions(baseline_tree_df, terrain_df, seed=BASELINE_RANDOM_SEED)
    baseline_tree_df = assign_baseline_tree_structure_ids(baseline_tree_df, site=site)
    baseline_tree_df = _calculate_useful_life_expectancy(baseline_tree_df)
    baseline_node_df = _build_baseline_node_df(baseline_tree_df, site_dataset)
    baseline_tree_df = baseline_node_df.copy()

    baseline_tree_df, resource_df = a_resource_distributor_dataframes.process_all_trees(
        baseline_tree_df,
        voxel_size=voxel_size,
    )
    resource_df = a_resource_distributor_dataframes.rotate_resource_structures(baseline_tree_df, resource_df)
    resource_poly = a_resource_distributor_dataframes.convertToPoly(resource_df)

    for column in ["tree_number", "NodeID", "structureID", "diameter_breast_height", "baseline_biomass_kg", "baseline_volume_m3"]:
        if column in resource_df.columns:
            resource_poly.point_data[f"forest_{column}"] = resource_df[column].to_numpy()

    terrain_output_poly = _get_terrain_polydata(terrain_df)
    combined_poly = _combine_polydata(resource_poly, terrain_output_poly)
    combined_poly = _annotate_baseline_combined_polydata(
        combined_poly,
        node_df=baseline_tree_df,
        resource_df=resource_df,
        terrain_poly=terrain_output_poly,
    )

    resource_vtk_path = scenario_baseline_resources_vtk_path(site, voxel_size)
    trees_csv_path = scenario_baseline_trees_csv_path(site)
    terrain_vtk_path = scenario_baseline_terrain_vtk_path(site, voxel_size)
    combined_vtk_path = scenario_baseline_combined_vtk_path(site, voxel_size)
    refactored_trees_csv_path = engine_output_baseline_trees_csv_path(site)
    refactored_terrain_vtk_path = engine_output_baseline_terrain_vtk_path(site, voxel_size)
    legacy_nodedf_path = scenario_node_df_path(site, BASELINE_NODE_SCENARIO, BASELINE_NODE_YEAR, voxel_size)
    refactored_nodedf_path = engine_output_nodedf_path(site, BASELINE_NODE_SCENARIO, BASELINE_NODE_YEAR, voxel_size)

    resource_vtk_path.parent.mkdir(parents=True, exist_ok=True)
    resource_poly.save(resource_vtk_path)
    baseline_tree_df.to_csv(trees_csv_path, index=False)
    baseline_tree_df.to_csv(refactored_trees_csv_path, index=False)
    baseline_tree_df.to_csv(legacy_nodedf_path, index=False)
    baseline_tree_df.to_csv(refactored_nodedf_path, index=False)
    terrain_output_poly.save(terrain_vtk_path)
    terrain_output_poly.save(refactored_terrain_vtk_path)
    combined_poly.save(combined_vtk_path)

    allocation_csv_path = _support_file(site, voxel_size, "deadwood_allocation", ".csv")
    allocation_df = pd.concat([fallen_allocation_df, decayed_allocation_df], ignore_index=True)
    allocation_df.to_csv(allocation_csv_path, index=False)

    metadata_json_path = _support_file(site, voxel_size, "metadata", ".json")
    metadata_payload = {
        "site": site,
        "voxel_size": float(voxel_size),
        "tree_template_root": str(preflight.template_root),
        "template_table_path": str(preflight.template_table_path),
        "template_lookup_path": str(preflight.template_lookup_path),
        "baseline_density_csv": str(BASELINE_DENSITY_CSV),
        "scenario_output_root": str(preflight.scenario_output_root),
        "engine_output_root": str(preflight.engine_output_root),
        "baseline_node_scenario": BASELINE_NODE_SCENARIO,
        "baseline_node_year": BASELINE_NODE_YEAR,
        "legacy_nodedf_path": str(legacy_nodedf_path),
        "refactored_nodedf_path": str(refactored_nodedf_path),
        "targets": asdict(preflight.targets),
        "shares": {
            "senescing_share": senescing_share,
            "snag_share": snag_share,
            "fallen_share": fallen_share,
            "decayed_share": decayed_share,
        },
        "counts": baseline_tree_df["size"].value_counts().sort_index().to_dict(),
        "live_large_total": live_large_total,
        "fallen_allocated_volume_m3": float(fallen_allocation_df["template_volume_m3"].sum()),
        "decayed_allocated_volume_m3": float(decayed_allocation_df["template_volume_m3"].sum()),
        "fallen_allocated_biomass_kg": float(fallen_allocation_df["template_biomass_kg"].sum()),
        "decayed_allocated_biomass_kg": float(decayed_allocation_df["template_biomass_kg"].sum()),
        "fallen_target_m3": float(preflight.targets.fallen_target_m3),
        "decayed_target_m3": float(preflight.targets.decayed_target_m3),
        "fallen_target_kg": float(preflight.targets.fallen_target_kg),
        "decayed_target_kg": float(preflight.targets.decayed_target_kg),
        "fallen_allocation_count": int(len(fallen_allocation_df)),
        "decayed_allocation_count": int(len(decayed_allocation_df)),
    }
    metadata_json_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    if visualize:
        combined_poly.plot(scalars="resource_fallen log", render_points_as_spheres=True)

    site_dataset.close()

    return GeneratedBaselineArtifacts(
        trees_csv_path=trees_csv_path,
        resource_vtk_path=resource_vtk_path,
        terrain_vtk_path=terrain_vtk_path,
        combined_vtk_path=combined_vtk_path,
        refactored_trees_csv_path=refactored_trees_csv_path,
        refactored_terrain_vtk_path=refactored_terrain_vtk_path,
        legacy_nodedf_path=legacy_nodedf_path,
        refactored_nodedf_path=refactored_nodedf_path,
        allocation_csv_path=allocation_csv_path,
        metadata_json_path=metadata_json_path,
        candidate_csv_path=preflight.candidate_csv_path,
        target_json_path=preflight.target_json_path,
        combined_polydata=combined_poly,
    )
