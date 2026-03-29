from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATA_ROOT = REPO_ROOT / "data"
REFACTORED_DATA_ROOT = REPO_ROOT / "_data-refactored"
FINAL_HOOKS_ROOT = REFACTORED_DATA_ROOT / "final-hooks"


def format_voxel_size(voxel_size: float | int) -> str:
    numeric = float(voxel_size)
    if numeric.is_integer():
        return str(int(numeric))
    return str(voxel_size)


def _normalize_relpath(relpath: str | Path) -> Path:
    return relpath if isinstance(relpath, Path) else Path(relpath)


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


def hook_state_vtk_latest_path(site: str, scenario: str, year: int, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return refactored_data_write_path(
        Path("final-hooks")
        / "vtks"
        / site
        / f"{site}_{scenario}_{voxel}_yr{year}_state_with_indicators.vtk"
    )


def hook_baseline_state_vtk_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return refactored_data_write_path(
        Path("final-hooks")
        / "vtks"
        / site
        / f"{site}_baseline_{voxel}_state_with_indicators.vtk"
    )


def hook_state_nodedf_path(site: str, scenario: str, year: int, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return refactored_data_write_path(
        Path("final-hooks")
        / "feature-locations"
        / site
        / f"{site}_{scenario}_{voxel}_nodeDF_yr{year}.csv"
    )


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
    voxel = format_voxel_size(voxel_size)
    return refactored_data_write_path(
        Path("final-hooks")
        / "bioenvelopes"
        / site
        / f"{site}_{scenario}_{voxel}_envelope_scenarioYR{year}.ply"
    )


def hook_baseline_trees_csv_path(site: str) -> Path:
    return refactored_data_write_path(
        Path("final-hooks")
        / "baselines"
        / site
        / f"{site}_baseline_trees.csv"
    )


def hook_baseline_terrain_vtk_path(site: str, voxel_size: float | int = 1) -> Path:
    voxel = format_voxel_size(voxel_size)
    return refactored_data_write_path(
        Path("final-hooks")
        / "baselines"
        / site
        / f"{site}_baseline_terrain_{voxel}.vtk"
    )


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
