import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree


CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent
BLENDER_EXPORT_DIR = Path(__file__).resolve().parent
FINAL_DIR = REPO_ROOT / "final"
TREE_PROCESSING_DIR = CODE_ROOT / "input_processing" / "tree_processing"

for import_root in (BLENDER_EXPORT_DIR, TREE_PROCESSING_DIR, FINAL_DIR, CODE_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import vtk_to_ply as a_vtk_to_ply

from _futureSim_refactored.paths import (
    engine_output_bioenvelope_ply_path,
    engine_output_state_vtk_path,
    legacy_world_reference_vtk_path,
)


SITES = ["trimmed-parade", "city", "uni"]
BLENDER_PROPOSAL_ATTRS = [
    "blender_proposal-deploy-structure",
    "blender_proposal-decay",
    "blender_proposal-recruit",
    "blender_proposal-colonise",
    "blender_proposal-release-control",
]

_REFERENCE_CACHE: dict[str, tuple[cKDTree, np.ndarray]] = {}


def _get_reference_data(site: str) -> tuple[cKDTree, np.ndarray]:
    cached = _REFERENCE_CACHE.get(site)
    if cached is None:
        cached = load_reference_data(site)
        _REFERENCE_CACHE[site] = cached
    return cached


def load_reference_data(site: str):
    print(f"Loading site and road voxels for {site}...")
    site_voxels = pv.read(legacy_world_reference_vtk_path(site, "site"))
    road_voxels = pv.read(legacy_world_reference_vtk_path(site, "road"))

    site_points = site_voxels.points
    road_points = road_voxels.points
    combined_points = np.concatenate((site_points, road_points), axis=0)

    site_normals = np.column_stack(
        (
            site_voxels.point_data["orig_normal_x"],
            site_voxels.point_data["orig_normal_y"],
            site_voxels.point_data["orig_normal_z"],
        )
    )
    road_normals = np.full((road_points.shape[0], 3), [0, 0, 1], dtype=float)
    combined_normals = np.concatenate((site_normals, road_normals), axis=0)

    print(f"Building KD-tree for {len(combined_points)} points...")
    return cKDTree(combined_points), combined_normals


def generate_normals(voxel_polydata: pv.PolyData, tree: cKDTree, combined_normals: np.ndarray) -> pv.PolyData:
    print("Finding nearest neighbors and averaging normals...")
    distances, indices = tree.query(
        voxel_polydata.points,
        k=100,
        distance_upper_bound=1.0,
        workers=-1,
    )
    valid_mask = distances < np.inf
    normals = np.zeros((voxel_polydata.n_points, 3), dtype=float)
    valid_points = valid_mask.any(axis=1)
    n_defaults = np.sum(~valid_points)
    print(f"Assigning default normal to {n_defaults} points with no valid neighbors")

    if np.any(valid_points):
        safe_indices = indices.copy()
        safe_indices[~valid_mask] = 0
        neighbor_normals = combined_normals[safe_indices]
        neighbor_normals[~valid_mask] = 0.0

        counts = valid_mask.sum(axis=1, keepdims=True)
        averaged = neighbor_normals.sum(axis=1)
        averaged[valid_points] /= counts[valid_points]

        magnitudes = np.linalg.norm(averaged[valid_points], axis=1)
        non_zero = magnitudes > 0
        averaged_valid = averaged[valid_points]
        averaged_valid[non_zero] /= magnitudes[non_zero][:, np.newaxis]
        normals[valid_points] = averaged_valid

    normals[~valid_points] = [0, 0, 1]
    voxel_polydata.point_data["normals"] = normals
    voxel_polydata.point_data["normal_magnitude"] = np.linalg.norm(normals, axis=1)
    return voxel_polydata


def extract_isosurface_from_polydata(
    polydata: pv.PolyData,
    spacing: tuple[float, float, float],
    bioenv: str,
    surface_cat: str,
) -> pv.PolyData | None:
    category = f"{bioenv}-{surface_cat}"
    print(f"{category} polydata has {polydata.n_points} points")

    if polydata is None or polydata.n_points == 0:
        return None

    points = polydata.points
    x, y, z = points.T
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    dims = (
        int((x_max - x_min) / spacing[0]) + 1,
        int((y_max - y_min) / spacing[1]) + 1,
        int((z_max - z_min) / spacing[2]) + 1,
    )

    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=(x_min, y_min, z_min))
    scalars = np.zeros(grid.n_points)

    for px, py, pz in points:
        ix = int((px - x_min) / spacing[0])
        iy = int((py - y_min) / spacing[1])
        iz = int((pz - z_min) / spacing[2])
        grid_idx = ix + iy * dims[0] + iz * dims[0] * dims[1]
        if grid_idx < len(scalars):
            scalars[grid_idx] = 2

    grid.point_data["values"] = scalars
    isosurface = grid.contour(isosurfaces=[0.5], scalars="values", method="flying_edges", compute_normals=True)
    surface = isosurface.extract_surface()
    if surface.n_points == 0:
        return None

    surface.point_data["scenario_bioEnvelope"] = np.full(surface.n_points, bioenv)
    surface.point_data["surfaceCat"] = np.full(surface.n_points, surface_cat)
    return surface


def transfer_point_data(original_poly: pv.PolyData, target_poly: pv.PolyData, attributes: list[str] | None = None) -> pv.PolyData:
    if original_poly is None or target_poly is None:
        return target_poly

    if attributes is None:
        attributes = list(original_poly.point_data.keys())

    print(f"Transferring attributes from original ({original_poly.n_points} points) to target ({target_poly.n_points} points)")
    tree = cKDTree(original_poly.points)
    _, indices = tree.query(target_poly.points)

    for attr in attributes:
        if attr in original_poly.point_data:
            print(f"Transferring {attr}")
            target_poly.point_data[attr] = original_poly.point_data[attr][indices]
        else:
            print(f"Warning: {attr} not found in original polydata")

    return target_poly


def categorize_surface_by_normal(poly: pv.PolyData) -> pv.PolyData:
    normals = poly.point_data["normals"]
    threshold = np.cos(np.radians(45))

    poly.point_data["surface_category"] = np.select(
        [
            normals[:, 0] > threshold,
            normals[:, 0] < -threshold,
            normals[:, 1] > threshold,
            normals[:, 1] < -threshold,
        ],
        ["x_pos", "x_neg", "y_pos", "y_neg"],
        default="up",
    )
    return poly


def generate_rewilded_envelopes(voxel_polydata: pv.PolyData, site: str, tree: cKDTree, combined_normals: np.ndarray) -> pv.PolyData | None:
    if "bioMask" in voxel_polydata.point_data:
        valid_mask = voxel_polydata.point_data["bioMask"].astype(bool)
        print(f"Using bioMask for envelope filtering: {valid_mask.sum()} points")
    else:
        valid_mask = voxel_polydata.point_data["scenario_bioEnvelope"] != "none"
        print(f"Using scenario_bioEnvelope fallback mask: {valid_mask.sum()} points")

    valid_polydata = voxel_polydata.extract_points(valid_mask)
    print(f"Found {len(valid_polydata.points)} valid bioenvelope points")
    if valid_polydata.n_points == 0:
        return None

    valid_polydata = generate_normals(valid_polydata, tree, combined_normals)
    valid_polydata = categorize_surface_by_normal(valid_polydata)

    processed_surfaces = []
    unique_bioenvelopes = np.unique(valid_polydata.point_data["scenario_bioEnvelope"])

    for bioenv in unique_bioenvelopes:
        print(f"\nProcessing {bioenv}")
        bioenv_mask = valid_polydata.point_data["scenario_bioEnvelope"] == bioenv
        points_subset = valid_polydata.extract_points(bioenv_mask)

        if bioenv == "livingFacade":
            unique_surfaces = np.unique(points_subset.point_data["surface_category"])
            for surf_cat in unique_surfaces:
                cat_mask = points_subset.point_data["surface_category"] == surf_cat
                cat_points = points_subset.extract_points(cat_mask)

                if cat_points.n_points == 0:
                    continue

                surface = extract_isosurface_from_polydata(
                    cat_points,
                    (1.0, 1.0, 1.0),
                    bioenv,
                    surf_cat,
                )
                if surface is not None:
                    processed_surfaces.append(surface)
        else:
            if points_subset.n_points == 0:
                continue

            surface = extract_isosurface_from_polydata(
                points_subset,
                (1.0, 1.0, 1.0),
                bioenv,
                "up",
            )
            if surface is not None:
                processed_surfaces.append(surface)

    if not processed_surfaces:
        return None

    combined = processed_surfaces[0].merge(processed_surfaces[1:])
    return transfer_point_data(valid_polydata, combined)


def scenario_bioenvelope_map_to_int_simple(iso_surface: pv.PolyData) -> pv.PolyData:
    category_map = {
        "exoskeleton": 1,
        "brownRoof": 2,
        "otherGround": 3,
        "node-rewilded": 4,
        "rewilded": 4,
        "footprint-depaved": 5,
        "livingFacade": 6,
        "greenRoof": 7,
        "footprint-depaved-connected": 8,
    }
    simplified_map = {
        "brownRoof": 2,
        "livingFacade": 3,
        "greenRoof": 4,
    }

    scenario_bioenvelope = pd.Series(iso_surface.point_data["scenario_bioEnvelope"])
    scenario_bioenvelope_int = scenario_bioenvelope.map(category_map).fillna(0).astype(np.int32).to_numpy()
    scenario_bioenvelope_simple = scenario_bioenvelope.map(simplified_map).fillna(1).astype(np.int32).to_numpy()

    iso_surface.point_data["scenario_bioEnvelope_int"] = scenario_bioenvelope_int
    iso_surface.point_data["scenario_bioEnvelope_simple_int"] = scenario_bioenvelope_simple
    return iso_surface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rewilded bioenvelope PLYs from state VTKs.")
    parser.add_argument("--site", default="all", help="Site key or 'all'.")
    parser.add_argument("--scenario", default="all", help="Scenario key or 'all'.")
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=[0, 1, 10, 30, 60, 90, 120, 150, 180],
        help="Years to export.",
    )
    parser.add_argument("--voxel-size", type=int, default=1, help="Voxel size. Default: 1.")
    parser.add_argument("--output-mode", default="validation", choices=["canonical", "validation"])
    return parser.parse_args()


def iter_targets(args: argparse.Namespace):
    sites = SITES if args.site == "all" else [args.site]
    scenarios = ["positive", "trending"] if args.scenario == "all" else [args.scenario]

    for site in sites:
        if site not in SITES:
            raise KeyError(f"Unknown site: {site}")
        for scenario in scenarios:
            for year in args.years:
                vtk_path = engine_output_state_vtk_path(site, scenario, year, args.voxel_size, args.output_mode)
                if vtk_path.exists():
                    yield site, scenario, year, vtk_path
                else:
                    print(f"Skipping missing VTK: {vtk_path}")


def export_target(
    site: str,
    scenario: str,
    year: int,
    vtk_path: Path | None,
    output_mode: str,
    *,
    voxel_size: int = 1,
    mesh: pv.PolyData | None = None,
) -> Path | None:
    """Export bioenvelope PLY. Pass *mesh* to skip disk read."""
    if mesh is None:
        mesh = pv.read(vtk_path)

    tree, combined_normals = _get_reference_data(site)

    iso_surface = generate_rewilded_envelopes(mesh, site, tree, combined_normals)
    if iso_surface is None:
        print(f"No envelope output for {site}/{scenario}/yr{year}")
        return None

    iso_surface = scenario_bioenvelope_map_to_int_simple(iso_surface)
    attributes = [
        "scenario_bioEnvelope_int",
        "scenario_bioEnvelope_simple_int",
        "sim_Turns",
        *BLENDER_PROPOSAL_ATTRS,
    ]
    if "sim_averageResistance" in iso_surface.point_data:
        attributes.append("sim_averageResistance")

    output_path = engine_output_bioenvelope_ply_path(site, scenario, year, voxel_size, output_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    a_vtk_to_ply.export_polydata_to_ply(
        iso_surface,
        str(output_path),
        attributesToTransfer=attributes,
    )
    print(f"Saved bioenvelope surface to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    for site, scenario, year, vtk_path in iter_targets(args):
        try:
            export_target(site, scenario, year, vtk_path, args.output_mode, voxel_size=args.voxel_size)
        except Exception as exc:
            print(f"  Bioenvelope export failed ({site}/{scenario}/yr{year}): {exc}")


if __name__ == "__main__":
    main()
