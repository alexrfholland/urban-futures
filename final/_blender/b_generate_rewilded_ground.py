import sys
from pathlib import Path

import pyvista as pv

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import a_vtk_to_ply
from b_rewilded_surface_shell import (
    SHELL_DISTANCE,
    SURFACE_VOXEL_SIZE,
    SURFACE_VOXEL_SPACING,
    extract_isosurface_from_points,
    select_explicit_export_keys,
    surface_mesh_to_voxel_shell,
    transfer_point_data_nearest,
)


SITE = 'uni'
SCENARIOS = ['positive', 'trending']
VOXEL_SIZE = 1
YEARS = [10, 30, 60, 180]
OUTPUT_SUBDIR = 'ply'
REQUIRED_ATTRS = ['scenario_rewilded', 'sim_Turns', 'sim_averageResistance']
REPO_ROOT = Path(__file__).resolve().parents[2]


def format_voxel_size(voxel_size: float | int) -> str:
    numeric = float(voxel_size)
    if numeric.is_integer():
        return str(int(numeric))
    return str(voxel_size)


def hook_state_vtk_latest_path(site: str, scenario: str, year: int, voxel_size: int) -> Path:
    voxel = format_voxel_size(voxel_size)
    return REPO_ROOT / "_data-refactored" / "final-hooks" / "vtks" / site / f"{site}_{scenario}_{voxel}_yr{year}_state_with_indicators.vtk"


def resolve_scenario_vtk_path(site: str, voxel_size: int, year: int, scenario: str | None = None) -> Path:
    if scenario:
        vtk_path = hook_state_vtk_latest_path(site, scenario, year, voxel_size)
        if vtk_path.exists():
            return vtk_path

    voxel = format_voxel_size(voxel_size)
    legacy_assessed_candidates = []
    if scenario:
        legacy_assessed_candidates.append(
            Path("data/revised/final/output") / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features_with_indicators.vtk"
        )

    file_path = Path(f'data/revised/final/{site}')
    candidates = []
    if scenario:
        candidates.append(file_path / f'{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk')
    candidates.append(file_path / f'{site}_{voxel_size}_scenarioYR{year}.vtk')

    for candidate in legacy_assessed_candidates + candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f'No scenario VTK found for site={site}, scenario={scenario}, year={year}')


def select_ground_points(poly: pv.PolyData) -> pv.PolyData:
    if 'maskForRewilding' in poly.point_data:
        mask = poly.point_data['maskForRewilding'].astype(bool)
        print(f"Using maskForRewilding for ground selection: {mask.sum()} points")
    else:
        mask = poly.point_data['scenario_rewilded'] != 'none'
        print(f"Using scenario_rewilded fallback mask: {mask.sum()} points")

    return poly.extract_points(mask)


def _backfill_missing_attrs(target_poly: pv.PolyData, fallback_poly: pv.PolyData, required_attrs: list[str]) -> pv.PolyData:
    missing_attrs = [attr for attr in required_attrs if attr in fallback_poly.point_data and attr not in target_poly.point_data]
    if missing_attrs:
        print(f"Backfilling missing attrs from fallback source: {missing_attrs}")
        target_poly = transfer_point_data_nearest(
            fallback_poly,
            target_poly,
            numeric_only=False,
            explicit_keys=missing_attrs,
        )
    return target_poly


def generate_rewilded_ground(site: str, voxel_size: int, year: int, scenario: str | None = None):
    vtk_path = resolve_scenario_vtk_path(site, voxel_size, year, scenario)
    print(f'loading polydata from {vtk_path}')
    poly = pv.read(vtk_path)

    ground_points = select_ground_points(poly)
    if ground_points.n_points == 0:
        return None, None

    surface_mesh = extract_isosurface_from_points(
        ground_points,
        spacing=SURFACE_VOXEL_SPACING,
    )
    if surface_mesh is None or surface_mesh.n_points == 0:
        return None, None

    explicit_keys = select_explicit_export_keys(ground_points.point_data.keys())
    surface_mesh = transfer_point_data_nearest(
        ground_points,
        surface_mesh,
        numeric_only=True,
        explicit_keys=explicit_keys,
    )

    shell_points = surface_mesh_to_voxel_shell(
        surface_mesh,
        voxel_size=SURFACE_VOXEL_SIZE,
        shell_distance=SHELL_DISTANCE,
    )
    if shell_points is None or shell_points.n_points == 0:
        return surface_mesh, None

    shell_points = transfer_point_data_nearest(
        surface_mesh,
        shell_points,
        numeric_only=True,
        explicit_keys=REQUIRED_ATTRS,
    )
    shell_points = _backfill_missing_attrs(shell_points, ground_points, REQUIRED_ATTRS)
    return surface_mesh, shell_points


def main():
    site = SITE
    scenarios = SCENARIOS
    voxel_size = VOXEL_SIZE
    years = YEARS

    file_path = Path(f'data/revised/final/{site}')
    output_dir = file_path / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        for year in years:
            try:
                surface_mesh, shell_points = generate_rewilded_ground(
                    site=site,
                    voxel_size=voxel_size,
                    year=year,
                    scenario=scenario,
                )
            except FileNotFoundError as exc:
                print(exc)
                continue

            if surface_mesh is None or shell_points is None:
                print(f"No ground output for year {year} and {scenario}")
                continue

            attributes = ['scenario_rewilded', 'sim_Turns']
            if 'sim_averageResistance' in shell_points.point_data:
                attributes.append('sim_averageResistance')

            output_name = f'{site}_{scenario}_{voxel_size}_ground_scenarioYR{year}' if scenario else f'{site}_{voxel_size}_ground_scenarioYR{year}'
            output_base = output_dir / output_name
            a_vtk_to_ply.export_polydata_points_to_ply(
                shell_points,
                str(output_base.with_suffix('.ply')),
                attributesToTransfer=attributes,
            )
            surface_mesh.save(str(output_base.with_suffix('.vtk')))
            print(f"Saved ground shell points to {output_base.with_suffix('.ply')}")
            print(f"Saved ground surface mesh to {output_base.with_suffix('.vtk')}")


if __name__ == "__main__":
    main()
