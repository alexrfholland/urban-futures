import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

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
REQUIRED_ATTRS = ['scenario_bioEnvelope', 'sim_Turns', 'sim_averageResistance']
WRITE_LEGACY_OUTPUTS = True
REPO_ROOT = Path(__file__).resolve().parents[2]


def format_voxel_size(voxel_size: float | int) -> str:
    numeric = float(voxel_size)
    if numeric.is_integer():
        return str(int(numeric))
    return str(voxel_size)


def hook_state_vtk_latest_path(site: str, scenario: str, year: int, voxel_size: int) -> Path:
    voxel = format_voxel_size(voxel_size)
    return REPO_ROOT / "_data-refactored" / "final-hooks" / "vtks" / site / f"{site}_{scenario}_{voxel}_yr{year}_state_with_indicators.vtk"


def hook_bioenvelope_ply_path(site: str, scenario: str, year: int, voxel_size: int) -> Path:
    voxel = format_voxel_size(voxel_size)
    path = REPO_ROOT / "_data-refactored" / "final-hooks" / "bioenvelopes" / site / f"{site}_{scenario}_{voxel}_envelope_scenarioYR{year}.ply"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def legacy_world_reference_vtk_path(site: str, kind: str) -> Path:
    suffix_map = {
        "site": f"{site}-siteVoxels-masked.vtk",
        "road": f"{site}-roadVoxels-coloured.vtk",
    }
    if kind not in suffix_map:
        raise ValueError(f"Unknown world reference kind: {kind}")
    return REPO_ROOT / "data" / "revised" / "final" / suffix_map[kind]


def resolve_scenario_vtk_path(site: str, scenario: str, year: int, voxel_size: int) -> Path:
    vtk_path = hook_state_vtk_latest_path(site, scenario, year, voxel_size)
    if vtk_path.exists():
        return vtk_path

    voxel = format_voxel_size(voxel_size)
    legacy_path = Path("data/revised/final/output") / f"{site}_{scenario}_{voxel}_scenarioYR{year}_urban_features_with_indicators.vtk"
    if legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(f"No assessed state VTK found for site={site}, scenario={scenario}, year={year}")


def legacy_envelope_output_dir(site: str) -> Path:
    return Path(f"data/revised/final/{site}") / OUTPUT_SUBDIR


def load_reference_data(site):
    print(f"Loading site and road voxels for {site}...")
    site_voxels = pv.read(legacy_world_reference_vtk_path(site, 'site'))
    road_voxels = pv.read(legacy_world_reference_vtk_path(site, 'road'))

    site_points = site_voxels.points
    road_points = road_voxels.points
    combined_points = np.concatenate((site_points, road_points), axis=0)

    site_normals = np.column_stack(
        (
            site_voxels.point_data['orig_normal_x'],
            site_voxels.point_data['orig_normal_y'],
            site_voxels.point_data['orig_normal_z'],
        )
    )
    road_normals = np.full((road_points.shape[0], 3), [0, 0, 1], dtype=float)
    combined_normals = np.concatenate((site_normals, road_normals), axis=0)

    print(f"Building KD-tree for {len(combined_points)} reference points...")
    return cKDTree(combined_points), combined_points, combined_normals


def generate_normals(voxel_polydata: pv.PolyData, tree: cKDTree, combined_normals: np.ndarray):
    print("Generating reference normals for envelope points...")
    distances, indices = tree.query(voxel_polydata.points, k=100, distance_upper_bound=1.0)
    valid_mask = distances < np.inf
    normals = np.zeros((voxel_polydata.n_points, 3), dtype=float)
    valid_points = valid_mask.any(axis=1)

    if np.any(valid_points):
        averaged = np.array(
            [
                np.mean(combined_normals[indices[idx][valid_mask[idx]]], axis=0)
                for idx in np.where(valid_points)[0]
            ]
        )
        magnitudes = np.linalg.norm(averaged, axis=1)
        non_zero = magnitudes > 0
        averaged[non_zero] /= magnitudes[non_zero][:, np.newaxis]
        normals[valid_points] = averaged

    normals[~valid_points] = [0, 0, 1]
    voxel_polydata.point_data['normals'] = normals
    voxel_polydata.point_data['normal_magnitude'] = np.linalg.norm(normals, axis=1)
    return voxel_polydata


def categorize_surface_by_normal(poly):
    normals = poly.point_data['normals']
    threshold = np.cos(np.radians(45))
    surface_category = np.select(
        [
            normals[:, 0] > threshold,
            normals[:, 0] < -threshold,
            normals[:, 1] > threshold,
            normals[:, 1] < -threshold,
        ],
        ['x_pos', 'x_neg', 'y_pos', 'y_neg'],
        default='up',
    )
    poly.point_data['surface_category'] = surface_category
    return poly


def _merge_polydata(poly_list: list[pv.PolyData]) -> pv.PolyData | None:
    if not poly_list:
        return None

    merged = poly_list[0].copy()
    for poly in poly_list[1:]:
        merged = merged.merge(poly)
    return merged.clean()


def _select_valid_bioenvelope_points(voxel_polydata: pv.PolyData) -> pv.PolyData:
    if 'bioMask' in voxel_polydata.point_data:
        valid_mask = voxel_polydata.point_data['bioMask'].astype(bool)
        print(f"Using bioMask for envelope filtering: {np.sum(valid_mask)} points")
    else:
        valid_mask = voxel_polydata.point_data['scenario_bioEnvelope'] != 'none'
        print(f"Using scenario_bioEnvelope fallback mask: {np.sum(valid_mask)} points")

    return voxel_polydata.extract_points(valid_mask)


def _build_envelope_surface(valid_polydata: pv.PolyData) -> pv.PolyData | None:
    processed_surfaces = []
    explicit_keys = select_explicit_export_keys(valid_polydata.point_data.keys()) + ['surface_category']

    for bioenv in np.unique(valid_polydata.point_data['scenario_bioEnvelope']):
        points_subset = valid_polydata.extract_points(valid_polydata.point_data['scenario_bioEnvelope'] == bioenv)
        if points_subset.n_points == 0:
            continue

        if bioenv == 'livingFacade' and 'surface_category' in points_subset.point_data:
            surface_categories = np.unique(points_subset.point_data['surface_category'])
        else:
            surface_categories = ['up']

        for surface_category in surface_categories:
            if bioenv == 'livingFacade':
                category_points = points_subset.extract_points(points_subset.point_data['surface_category'] == surface_category)
            else:
                category_points = points_subset

            if category_points.n_points == 0:
                continue

            surface = extract_isosurface_from_points(
                category_points,
                spacing=SURFACE_VOXEL_SPACING,
                extra_point_data={
                    'scenario_bioEnvelope': bioenv,
                    'surface_category': surface_category,
                },
            )
            if surface is None or surface.n_points == 0:
                continue

            surface = transfer_point_data_nearest(
                category_points,
                surface,
                numeric_only=True,
                explicit_keys=explicit_keys,
            )
            processed_surfaces.append(surface)

    if not processed_surfaces:
        return None

    combined_surface = _merge_polydata(processed_surfaces)
    return transfer_point_data_nearest(
        valid_polydata,
        combined_surface,
        numeric_only=True,
        explicit_keys=explicit_keys,
    )


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


def generate_rewilded_envelopes(
    voxel_polydata: pv.PolyData,
    site: str,
    tree: cKDTree,
    combined_points: np.ndarray,
    combined_normals: np.ndarray,
):
    valid_polydata = _select_valid_bioenvelope_points(voxel_polydata)
    if valid_polydata.n_points == 0:
        return None, None

    valid_polydata = generate_normals(valid_polydata, tree, combined_normals)
    valid_polydata = categorize_surface_by_normal(valid_polydata)

    surface_mesh = _build_envelope_surface(valid_polydata)
    if surface_mesh is None or surface_mesh.n_points == 0:
        return None, None

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
    shell_points = _backfill_missing_attrs(shell_points, valid_polydata, REQUIRED_ATTRS)
    return surface_mesh, shell_points


def scenario_bioenvelope_map_to_int_simple(poly):
    category_map = {
        'exoskeleton': 1,
        'brownRoof': 2,
        'otherGround': 3,
        'node-rewilded': 4,
        'rewilded': 4,
        'footprint-depaved': 5,
        'livingFacade': 6,
        'greenRoof': 7,
    }
    simplified_map = {
        'brownRoof': 2,
        'livingFacade': 3,
        'greenRoof': 4,
    }

    scenario_bioenvelope = pd.Series(poly.point_data['scenario_bioEnvelope'])
    scenario_bioenvelope_int = scenario_bioenvelope.map(category_map).fillna(0).astype(np.int32).to_numpy()
    scenario_bioenvelope_simple = scenario_bioenvelope.map(simplified_map).fillna(1).astype(np.int32).to_numpy()

    poly.point_data['scenario_bioEnvelope_int'] = scenario_bioenvelope_int
    poly.point_data['scenario_bioEnvelope_simple_int'] = scenario_bioenvelope_simple
    return poly


def main():
    site = SITE
    scenarios = SCENARIOS
    voxel_size = VOXEL_SIZE
    years = YEARS

    output_dir = legacy_envelope_output_dir(site)
    if WRITE_LEGACY_OUTPUTS:
        output_dir.mkdir(parents=True, exist_ok=True)

    tree, combined_points, combined_normals = load_reference_data(site)

    for scenario in scenarios:
        for year in years:
            vtk_path = resolve_scenario_vtk_path(site, scenario, year, voxel_size)
            print(f'loading polydata from {vtk_path}')
            voxel_polydata = pv.read(vtk_path)

            surface_mesh, shell_points = generate_rewilded_envelopes(
                voxel_polydata,
                site,
                tree,
                combined_points,
                combined_normals,
            )
            if surface_mesh is None or shell_points is None:
                print(f"No envelope output for year {year} and {scenario}")
                continue

            shell_points = scenario_bioenvelope_map_to_int_simple(shell_points)

            attributes = ['scenario_bioEnvelope_int', 'scenario_bioEnvelope_simple_int', 'sim_Turns']
            if 'sim_averageResistance' in shell_points.point_data:
                attributes.append('sim_averageResistance')

            refactored_output_path = hook_bioenvelope_ply_path(site, scenario, year, voxel_size)
            a_vtk_to_ply.export_polydata_points_to_ply(
                shell_points,
                str(refactored_output_path),
                attributesToTransfer=attributes,
            )
            print(f"Saved refactored envelope shell points to {refactored_output_path}")
            if WRITE_LEGACY_OUTPUTS:
                output_base = output_dir / f'{site}_{scenario}_{voxel_size}_envelope_scenarioYR{year}'
                a_vtk_to_ply.export_polydata_points_to_ply(
                    shell_points,
                    str(output_base.with_suffix('.ply')),
                    attributesToTransfer=attributes,
                )
                surface_mesh.save(str(output_base.with_suffix('.vtk')))
                print(f"Saved legacy envelope shell points to {output_base.with_suffix('.ply')}")
                print(f"Saved legacy envelope surface mesh to {output_base.with_suffix('.vtk')}")


if __name__ == "__main__":
    main()
