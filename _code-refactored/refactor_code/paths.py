from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATA_ROOT = REPO_ROOT / "data"
REFACTORED_DATA_ROOT = REPO_ROOT / "_data-refactored"
FINAL_HOOKS_ROOT = REFACTORED_DATA_ROOT / "final-hooks"
V2_ENGINE_OUTPUT_ROOT = REFACTORED_DATA_ROOT / "v2engine_outputs"
CANONICAL_REFACTORED_STATISTICS_ROOT = REPO_ROOT / "_statistics-refactored"
VALIDATION_REFACTORED_STATISTICS_ROOT = REPO_ROOT / "_statistics-refactored-v2"
CANONICAL_SCENARIO_OUTPUT_ROOT = LEGACY_DATA_ROOT / "revised" / "final"
VALIDATION_SCENARIO_OUTPUT_ROOT = LEGACY_DATA_ROOT / "revised" / "final-v2"

DEFAULT_OUTPUT_MODE = "validation"
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


def refactor_run_output_root(output_mode: str | None = None) -> Path | None:
    override = os.environ.get("REFACTOR_RUN_OUTPUT_ROOT")
    if override:
        return Path(override)
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
    return V2_ENGINE_OUTPUT_ROOT if mode == "validation" else FINAL_HOOKS_ROOT


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


def hook_state_vtk_latest_path(site: str, scenario: str, year: int, voxel_size: float | int = 1) -> Path:
    return engine_output_state_vtk_path(site, scenario, year, voxel_size, output_mode="canonical")


def hook_baseline_state_vtk_path(site: str, voxel_size: float | int = 1) -> Path:
    return engine_output_baseline_state_vtk_path(site, voxel_size, output_mode="canonical")


def hook_state_nodedf_path(site: str, scenario: str, year: int, voxel_size: float | int = 1) -> Path:
    return engine_output_nodedf_path(site, scenario, year, voxel_size, output_mode="canonical")


def hook_tree_ply_library_dir() -> Path:
    output_dir = FINAL_HOOKS_ROOT / "feature-libraries" / "treePlyLibrary"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def hook_log_ply_library_dir() -> Path:
    output_dir = FINAL_HOOKS_ROOT / "feature-libraries" / "logPlyLibrary"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def hook_world_buildings_ply_path(site: str) -> Path:
    return refactored_data_write_path(
        Path("final-hooks")
        / "world"
        / site
        / f"{site}_buildings.ply"
    )


def hook_world_road_ply_path(site: str) -> Path:
    return refactored_data_write_path(
        Path("final-hooks")
        / "world"
        / site
        / f"{site}_road.ply"
    )


def hook_bioenvelope_ply_path(site: str, scenario: str, year: int, voxel_size: float | int = 1) -> Path:
    return engine_output_bioenvelope_ply_path(site, scenario, year, voxel_size, output_mode="canonical")


def hook_baseline_trees_csv_path(site: str) -> Path:
    return engine_output_baseline_trees_csv_path(site, output_mode="canonical")


def hook_baseline_terrain_vtk_path(site: str, voxel_size: float | int = 1) -> Path:
    return engine_output_baseline_terrain_vtk_path(site, voxel_size, output_mode="canonical")


def hook_baseline_terrain_ply_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return refactored_data_write_path(
        Path("final-hooks")
        / "baselines"
        / site
        / f"{site}_baseline_terrain_{voxel}.ply"
    )


def legacy_world_reference_vtk_path(site: str, kind: str) -> Path:
    suffix_map = {
        "site": f"{site}-siteVoxels-masked.vtk",
        "road": f"{site}-roadVoxels-coloured.vtk",
    }
    if kind not in suffix_map:
        raise ValueError(f"Unknown world reference kind: {kind}")
    return LEGACY_DATA_ROOT / "revised" / "final" / suffix_map[kind]
