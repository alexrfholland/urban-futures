from __future__ import annotations

import os
from pathlib import Path


# 1. Versioning

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATA_ROOT = REPO_ROOT / "data"
REFACTORED_DATA_ROOT = REPO_ROOT / "_data-refactored"
MODEL_INPUTS_ROOT = REFACTORED_DATA_ROOT / "model-inputs"
MODEL_OUTPUTS_ROOT = REFACTORED_DATA_ROOT / "model-outputs"
GENERATED_STATE_RUNS_ROOT = MODEL_OUTPUTS_ROOT / "generated-states"
BLENDER_ROOT = REFACTORED_DATA_ROOT / "blender"
BLENDER_INPUTS_ROOT = BLENDER_ROOT / "inputs"
BLENDER_WORLD_ROOT = BLENDER_INPUTS_ROOT / "world"
TREE_LIBRARY_ROOT = MODEL_INPUTS_ROOT / "tree_libraries"
TREE_LIBRARY_BASE_ROOT = TREE_LIBRARY_ROOT / "base" / "trees"
TREE_LIBRARY_EXPORT_ROOT = MODEL_INPUTS_ROOT / "tree_library_exports"
TREE_MESH_VTK_ROOT = TREE_LIBRARY_EXPORT_ROOT / "treeMeshes"
TREE_MESH_PLY_ROOT = TREE_LIBRARY_EXPORT_ROOT / "treeMeshesPly"
LOG_MESH_VTK_ROOT = TREE_LIBRARY_EXPORT_ROOT / "logMeshes"
LOG_MESH_PLY_ROOT = TREE_LIBRARY_EXPORT_ROOT / "logMeshesPly"
TREE_VARIANTS_ROOT = MODEL_INPUTS_ROOT / "tree_variants"
APPROVED_TREE_TEMPLATE_ROOT = (
    TREE_VARIANTS_ROOT
    / "template-edits__fallens-nonpre-direct__snags-elm-snags-old__decayed-small-fallen"
    / "trees"
)

CANONICAL_VERSION = "v3"
CANONICAL_SCENARIO_OUTPUT_ROOT = LEGACY_DATA_ROOT / "revised" / "final-v3"
CANONICAL_ENGINE_OUTPUT_ROOT = REFACTORED_DATA_ROOT / "v3engine_outputs"
CANONICAL_REFACTORED_STATISTICS_ROOT = REPO_ROOT / "_statistics-refactored-v3"

LEGACY_VERSION_ROOTS = {
    "v1": {
        "branch": "engine-v1",
        "scenario": LEGACY_DATA_ROOT / "revised" / "final",
        "engine": REFACTORED_DATA_ROOT / "final-hooks",
        "statistics": REPO_ROOT / "_statistics-refactored",
    },
    "v2": {
        "branch": "master",
        "scenario": LEGACY_DATA_ROOT / "revised" / "final-v2",
        "engine": REFACTORED_DATA_ROOT / "v2engine_outputs",
        "statistics": REPO_ROOT / "_statistics-refactored-v2",
    },
    "v3": {
        "branch": "engine-v3",
        "scenario": CANONICAL_SCENARIO_OUTPUT_ROOT,
        "engine": CANONICAL_ENGINE_OUTPUT_ROOT,
        "statistics": CANONICAL_REFACTORED_STATISTICS_ROOT,
    },
}

VALIDATION_SCENARIO_OUTPUT_ROOT = CANONICAL_SCENARIO_OUTPUT_ROOT
VALIDATION_ENGINE_OUTPUT_ROOT = CANONICAL_ENGINE_OUTPUT_ROOT
VALIDATION_REFACTORED_STATISTICS_ROOT = CANONICAL_REFACTORED_STATISTICS_ROOT

DEFAULT_OUTPUT_MODE = "canonical"
VALID_OUTPUT_MODES = {"canonical", "validation"}


def format_voxel_size(voxel_size: float | int) -> str:
    numeric = float(voxel_size)
    if numeric.is_integer():
        return str(int(numeric))
    return str(voxel_size)


def _normalize_relpath(relpath: str | Path) -> Path:
    return relpath if isinstance(relpath, Path) else Path(relpath)


def normalize_output_mode(output_mode: str | None = None) -> str:
    mode = output_mode or os.environ.get("REFACTOR_OUTPUT_MODE", DEFAULT_OUTPUT_MODE)
    if mode not in VALID_OUTPUT_MODES:
        raise ValueError(f"Unknown output mode: {mode}")
    return mode


def canonical_version_roots(version: str = CANONICAL_VERSION) -> dict[str, Path | str]:
    try:
        return LEGACY_VERSION_ROOTS[version]
    except KeyError as exc:
        raise ValueError(f"Unknown simulation version: {version}") from exc


def generated_state_run_root(run_name: str) -> Path:
    return GENERATED_STATE_RUNS_ROOT / run_name


def simv3_run_root(run_name: str | int) -> Path:
    name = str(run_name)
    if name.startswith("simv3-"):
        return generated_state_run_root(name)
    return generated_state_run_root(f"simv3-{name}")


def mediaflux_versioned_destination(run_name: str) -> str:
    root = os.environ.get("MEDIAFLUX_SIM_DESTINATION_ROOT", "urban-futures/simulation-runs")
    return f"{root.rstrip('/')}/{run_name}"


def refactor_run_output_root(output_mode: str | None = None) -> Path | None:
    override = os.environ.get("REFACTOR_RUN_OUTPUT_ROOT")
    if override:
        return Path(override)
    # Fall back to the last explicitly-set output root from the run log
    from refactor_code.sim.run.run_log import get_last_output_root
    last = get_last_output_root()
    if last and last != "default":
        return Path(last)
    return None


def _unified_interim_root(output_mode: str | None = None) -> Path | None:
    run_root = refactor_run_output_root(output_mode)
    if run_root is None:
        return None
    return run_root / "temp" / "interim-data"


def _unified_validation_root(output_mode: str | None = None) -> Path | None:
    run_root = refactor_run_output_root(output_mode)
    if run_root is None:
        return None
    return run_root / "temp" / "validation"


def _unified_postprocess_root(output_mode: str | None = None) -> Path | None:
    run_root = refactor_run_output_root(output_mode)
    if run_root is None:
        return None
    return run_root / "output"


def _unified_statistics_root(output_mode: str | None = None) -> Path | None:
    run_root = refactor_run_output_root(output_mode)
    if run_root is None:
        return None
    return run_root / "output" / "stats"


def scenario_output_root(output_mode: str | None = None) -> Path:
    unified_root = _unified_interim_root(output_mode)
    if unified_root is not None:
        return unified_root
    mode = normalize_output_mode(output_mode)
    return VALIDATION_SCENARIO_OUTPUT_ROOT if mode == "validation" else CANONICAL_SCENARIO_OUTPUT_ROOT


def engine_output_root(output_mode: str | None = None) -> Path:
    unified_root = _unified_postprocess_root(output_mode)
    if unified_root is not None:
        return unified_root
    mode = normalize_output_mode(output_mode)
    return VALIDATION_ENGINE_OUTPUT_ROOT if mode == "validation" else CANONICAL_ENGINE_OUTPUT_ROOT


def refactor_statistics_root(output_mode: str | None = None) -> Path:
    unified_root = _unified_statistics_root(output_mode)
    if unified_root is not None:
        return unified_root
    mode = normalize_output_mode(output_mode)
    return (
        VALIDATION_REFACTORED_STATISTICS_ROOT
        if mode == "validation"
        else CANONICAL_REFACTORED_STATISTICS_ROOT
    )


def refactored_data_read_path(relpath: str | Path) -> Path:
    relpath = _normalize_relpath(relpath)
    refactored_path = REFACTORED_DATA_ROOT / relpath
    if refactored_path.exists():
        return refactored_path
    return LEGACY_DATA_ROOT / relpath


def refactored_data_write_path(relpath: str | Path) -> Path:
    relpath = _normalize_relpath(relpath)
    output_path = REFACTORED_DATA_ROOT / relpath
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _scenario_write_path(relpath: str | Path, output_mode: str | None = None) -> Path:
    relpath = _normalize_relpath(relpath)
    output_path = scenario_output_root(output_mode) / relpath
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _engine_output_write_path(relpath: str | Path, output_mode: str | None = None) -> Path:
    relpath = _normalize_relpath(relpath)
    output_path = engine_output_root(output_mode) / relpath
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


# 2. Site Information

def site_inputs_dir(site: str) -> Path:
    return LEGACY_DATA_ROOT / "revised" / "final" / site


def site_subset_dataset_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return site_inputs_dir(site) / f"{site}_{voxel}_subsetForScenarios.nc"


def site_rewilding_voxel_array_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return site_inputs_dir(site) / f"{site}_{voxel}_voxelArray_RewildingNodes.nc"


def site_tree_locations_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return site_inputs_dir(site) / f"{site}_{voxel}_treeDF.csv"


def site_pole_locations_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return site_inputs_dir(site) / f"{site}_{voxel}_poleDF.csv"


def site_log_locations_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return site_inputs_dir(site) / f"{site}_{voxel}_logDF.csv"


def site_voxel_array_with_logs_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return site_inputs_dir(site) / f"{site}_{voxel}_voxelArray_withLogs.nc"


def site_extra_tree_locations_path(site: str) -> Path:
    return site_inputs_dir(site) / f"{site}-extraTreeDF.csv"


def site_extra_pole_locations_path(site: str) -> Path:
    return site_inputs_dir(site) / f"{site}-extraPoleDF.csv"


def site_world_reference_vtk_path(site: str, kind: str) -> Path:
    suffix_map = {
        "site": f"{site}-siteVoxels-masked.vtk",
        "road": f"{site}-roadVoxels-coloured.vtk",
    }
    if kind not in suffix_map:
        raise ValueError(f"Unknown world reference kind: {kind}")
    return LEGACY_DATA_ROOT / "revised" / "final" / suffix_map[kind]


def scenario_site_dir(site: str, output_mode: str | None = None) -> Path:
    return _scenario_write_path(site, output_mode)


def scenario_tree_df_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(Path(site) / f"{site}_{scenario}_{voxel}_treeDF_{year}.csv", output_mode)


def scenario_log_df_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(Path(site) / f"{site}_{scenario}_{voxel}_logDF_{year}.csv", output_mode)


def scenario_pole_df_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(Path(site) / f"{site}_{scenario}_{voxel}_poleDF_{year}.csv", output_mode)


def scenario_node_df_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(Path(site) / f"{site}_{scenario}_{voxel}_nodeDF_{year}.csv", output_mode)


def legacy_world_reference_vtk_path(site: str, kind: str) -> Path:
    return site_world_reference_vtk_path(site, kind)


# 3. Baselines

def scenario_baseline_dir(output_mode: str | None = None) -> Path:
    return _scenario_write_path("baselines", output_mode)


def scenario_baseline_resources_vtk_path(
    site: str,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(Path("baselines") / f"{site}_baseline_resources_{voxel}.vtk", output_mode)


def scenario_baseline_trees_csv_path(
    site: str,
    output_mode: str | None = None,
) -> Path:
    return _scenario_write_path(Path("baselines") / f"{site}_baseline_trees.csv", output_mode)


def scenario_baseline_terrain_vtk_path(
    site: str,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(Path("baselines") / f"{site}_baseline_terrain_{voxel}.vtk", output_mode)


def scenario_baseline_combined_vtk_path(
    site: str,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(Path("baselines") / f"{site}_baseline_combined_{voxel}.vtk", output_mode)


def scenario_baseline_urban_features_vtk_path(
    site: str,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(
        Path("baselines") / f"{site}_baseline_combined_{voxel}_urban_features.vtk",
        output_mode,
    )


def engine_output_baseline_state_vtk_path(
    site: str,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _engine_output_write_path(
        Path("vtks") / site / f"{site}_baseline_{voxel}_state_with_indicators.vtk",
        output_mode,
    )


def engine_output_baseline_indicator_counts_path(
    site: str,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _engine_output_write_path(
        Path("stats") / "per-state" / site / f"{site}_baseline_{voxel}_indicator_counts.csv",
        output_mode,
    )


def engine_output_baseline_action_counts_path(
    site: str,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _engine_output_write_path(
        Path("stats") / "per-state" / site / f"{site}_baseline_{voxel}_action_counts.csv",
        output_mode,
    )


def engine_output_baseline_trees_csv_path(site: str, output_mode: str | None = None) -> Path:
    return _engine_output_write_path(Path("baselines") / site / f"{site}_baseline_trees.csv", output_mode)


def engine_output_baseline_terrain_vtk_path(
    site: str,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _engine_output_write_path(
        Path("baselines") / site / f"{site}_baseline_terrain_{voxel}.vtk",
        output_mode,
    )


def hook_baseline_state_vtk_path(site: str, voxel_size: float | int = 1) -> Path:
    return engine_output_baseline_state_vtk_path(site, voxel_size, output_mode="canonical")


def hook_baseline_trees_csv_path(site: str) -> Path:
    return engine_output_baseline_trees_csv_path(site, output_mode="canonical")


def hook_baseline_terrain_vtk_path(site: str, voxel_size: float | int = 1) -> Path:
    return engine_output_baseline_terrain_vtk_path(site, voxel_size, output_mode="canonical")


def hook_baseline_terrain_ply_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    output_path = BLENDER_INPUTS_ROOT / "baselines" / site / f"{site}_baseline_terrain_{voxel}.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


# 4. Tree Library

def tree_template_base_root() -> Path:
    override = os.environ.get("TREE_TEMPLATE_BASE_ROOT") or os.environ.get("BASE_TREE_TEMPLATES_ROOT")
    return Path(override).expanduser() if override else TREE_LIBRARY_BASE_ROOT


def tree_template_variants_root() -> Path:
    override = os.environ.get("TREE_TEMPLATE_VARIANTS_ROOT")
    return Path(override).expanduser() if override else TREE_VARIANTS_ROOT


def tree_template_root() -> Path:
    override = os.environ.get("TREE_TEMPLATE_ROOT")
    return Path(override).expanduser() if override else APPROVED_TREE_TEMPLATE_ROOT


def hook_tree_ply_library_dir() -> Path:
    output_dir = TREE_LIBRARY_EXPORT_ROOT / "treePlyLibrary"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def hook_log_ply_library_dir() -> Path:
    output_dir = TREE_LIBRARY_EXPORT_ROOT / "logPlyLibrary"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def tree_template_resource_dict_path() -> Path:
    return tree_template_base_root() / "resource_dicDF.csv"


def tree_template_combined_edits_path() -> Path:
    return tree_template_base_root() / "combined_editsDF.pkl"


def tree_template_selected_overrides_path() -> Path:
    return tree_template_base_root() / "template-library.selected-overrides.pkl"


def tree_template_overrides_applied_path() -> Path:
    return tree_template_base_root() / "template-library.overrides-applied.pkl"


def tree_mesh_vtk_dir() -> Path:
    TREE_MESH_VTK_ROOT.mkdir(parents=True, exist_ok=True)
    return TREE_MESH_VTK_ROOT


def tree_mesh_ply_dir() -> Path:
    TREE_MESH_PLY_ROOT.mkdir(parents=True, exist_ok=True)
    return TREE_MESH_PLY_ROOT


def log_mesh_vtk_dir() -> Path:
    LOG_MESH_VTK_ROOT.mkdir(parents=True, exist_ok=True)
    return LOG_MESH_VTK_ROOT


def log_mesh_ply_dir() -> Path:
    LOG_MESH_PLY_ROOT.mkdir(parents=True, exist_ok=True)
    return LOG_MESH_PLY_ROOT


# 5. Model Outputs

def scenario_state_vtk_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(Path(site) / f"{site}_{scenario}_{voxel}_scenarioYR{year}.vtk", output_mode)


def scenario_urban_features_vtk_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _scenario_write_path(
        Path(site) / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features.vtk",
        output_mode,
    )


def engine_output_state_vtk_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _engine_output_write_path(
        Path("vtks") / site / f"{site}_{scenario}_{voxel}_yr{year}_state_with_indicators.vtk",
        output_mode,
    )


def engine_output_nodedf_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _engine_output_write_path(
        Path("feature-locations") / site / f"{site}_{scenario}_{voxel}_nodeDF_yr{year}.csv",
        output_mode,
    )


def engine_output_validation_dir(output_mode: str | None = None) -> Path:
    unified_root = _unified_validation_root(output_mode)
    if unified_root is not None:
        unified_root.mkdir(parents=True, exist_ok=True)
        return unified_root
    return _engine_output_write_path("validation", output_mode)


def engine_output_bioenvelope_ply_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _engine_output_write_path(
        Path("bioenvelopes") / site / f"{site}_{scenario}_{voxel}_envelope_scenarioYR{year}.ply",
        output_mode,
    )


def hook_state_vtk_latest_path(site: str, scenario: str, year: int, voxel_size: float | int = 1) -> Path:
    return engine_output_state_vtk_path(site, scenario, year, voxel_size, output_mode="canonical")


def hook_state_nodedf_path(site: str, scenario: str, year: int, voxel_size: float | int = 1) -> Path:
    return engine_output_nodedf_path(site, scenario, year, voxel_size, output_mode="canonical")


def hook_bioenvelope_ply_path(site: str, scenario: str, year: int, voxel_size: float | int = 1) -> Path:
    return engine_output_bioenvelope_ply_path(site, scenario, year, voxel_size, output_mode="canonical")


# 6. Assessed/Postprocessed Outputs

def scenario_visualization_dir(output_mode: str | None = None) -> Path:
    output_dir = engine_output_validation_dir(output_mode) / "search-variable-visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# 7. Statistics

def statistics_csv_dir(output_mode: str | None = None) -> Path:
    output_dir = refactor_statistics_root(output_mode) / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def statistics_debug_dir(output_mode: str | None = None) -> Path:
    output_dir = refactor_statistics_root(output_mode) / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def tree_library_resource_counts_path(output_mode: str | None = None) -> Path:
    output_path = refactor_statistics_root(output_mode) / "tree-library" / "template_resource_counts.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def engine_output_state_indicator_counts_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _engine_output_write_path(
        Path("stats") / "per-state" / site / f"{site}_{scenario}_{voxel}_yr{year}_indicator_counts.csv",
        output_mode,
    )


def engine_output_state_action_counts_path(
    site: str,
    scenario: str,
    year: int,
    voxel_size: float | int = 1,
    output_mode: str | None = None,
) -> Path:
    voxel = format_voxel_size(voxel_size)
    return _engine_output_write_path(
        Path("stats") / "per-state" / site / f"{site}_{scenario}_{voxel}_yr{year}_action_counts.csv",
        output_mode,
    )


# 8. Blender

def hook_world_buildings_ply_path(site: str) -> Path:
    output_path = BLENDER_WORLD_ROOT / site / f"{site}_buildings.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def hook_world_road_ply_path(site: str) -> Path:
    output_path = BLENDER_WORLD_ROOT / site / f"{site}_road.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
