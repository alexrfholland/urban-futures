import math

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


SURFACE_VOXEL_SIZE = 0.25
SURFACE_VOXEL_SPACING = (0.25, 0.25, 0.25)
SHELL_DISTANCE = math.sqrt(3.0) * SURFACE_VOXEL_SIZE / 2.0
MAX_DISTANCE_CHUNK_POINTS = 250_000
EXPLICIT_EXPORT_PRIORITY = (
    "sim_Turns",
    "sim_averageResistance",
    "scenario_bioEnvelope",
    "scenario_rewilded",
    "bioMask",
    "maskForRewilding",
)


def select_explicit_export_keys(source_keys) -> list[str]:
    source_key_set = set(source_keys)
    return [key for key in EXPLICIT_EXPORT_PRIORITY if key in source_key_set]


def extract_isosurface_from_points(
    polydata: pv.PolyData,
    spacing: tuple[float, float, float] = SURFACE_VOXEL_SPACING,
    isovalue: float = 0.5,
    extra_point_data: dict | None = None,
    preserve_source_lattice: bool = False,
) -> pv.PolyData | None:
    if polydata is None or polydata.n_points == 0:
        return None

    spacing_array = np.asarray(spacing, dtype=float)
    points = np.asarray(polydata.points, dtype=float)

    if preserve_source_lattice:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        dims = np.maximum(((maxs - mins) / spacing_array).astype(int) + 1, 2)
    else:
        mins = np.floor(points.min(axis=0) / spacing_array) * spacing_array
        maxs = np.ceil(points.max(axis=0) / spacing_array) * spacing_array
        dims = np.maximum(((maxs - mins) / spacing_array).astype(int) + 2, 2)

    grid = pv.ImageData(dimensions=tuple(int(dim) for dim in dims), spacing=tuple(spacing_array), origin=tuple(mins))
    if preserve_source_lattice:
        scalars = np.zeros(grid.n_points, dtype=float)
        occupied_value = 2.0
    else:
        scalars = np.zeros(grid.n_points, dtype=np.uint8)
        occupied_value = 1

    ijk = np.floor((points - mins) / spacing_array).astype(int)
    valid_ijk_mask = np.all((ijk >= 0) & (ijk < np.asarray(dims)), axis=1)
    if not np.any(valid_ijk_mask):
        return None

    flat_indices = np.ravel_multi_index(ijk[valid_ijk_mask].T, dims)
    scalars[flat_indices] = occupied_value

    grid.point_data["values"] = scalars
    isosurface = grid.contour(
        isosurfaces=[isovalue],
        scalars="values",
        method="flying_edges",
        compute_normals=True,
    )
    surface = isosurface.extract_surface()
    if not preserve_source_lattice:
        surface = surface.triangulate().clean()

    if surface.n_points == 0:
        return None

    if extra_point_data:
        for key, value in extra_point_data.items():
            if np.isscalar(value):
                surface.point_data[key] = np.full(surface.n_points, value)
            else:
                array = np.asarray(value)
                if len(array) == surface.n_points:
                    surface.point_data[key] = array

    return surface


def transfer_point_data_nearest(
    source_poly: pv.PolyData,
    target_poly: pv.PolyData,
    numeric_only: bool = False,
    explicit_keys: list[str] | tuple[str, ...] | None = None,
) -> pv.PolyData | None:
    if source_poly is None or source_poly.n_points == 0 or target_poly is None or target_poly.n_points == 0:
        return target_poly

    explicit_key_set = set(explicit_keys or [])
    kd_tree = cKDTree(np.asarray(source_poly.points))
    _, indices = kd_tree.query(np.asarray(target_poly.points))

    for key in source_poly.point_data.keys():
        array = np.asarray(source_poly.point_data[key])

        should_copy = key in explicit_key_set
        if not should_copy:
            is_numeric = np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
            should_copy = is_numeric if numeric_only else (
                is_numeric or np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.object_)
            )

        if should_copy:
            target_poly.point_data[key] = source_poly.point_data[key][indices]

    return target_poly


def _aligned_axis(min_value: float, max_value: float, voxel_size: float) -> np.ndarray:
    start = math.floor(min_value / voxel_size) * voxel_size
    end = math.ceil(max_value / voxel_size) * voxel_size
    return np.arange(start, end + voxel_size * 0.5, voxel_size, dtype=float)


def surface_mesh_to_voxel_shell(
    mesh: pv.PolyData,
    voxel_size: float = SURFACE_VOXEL_SIZE,
    shell_distance: float | None = None,
) -> pv.PolyData | None:
    if mesh is None or mesh.n_points == 0:
        return None

    surface = mesh.extract_surface().triangulate().clean()
    if surface.n_points == 0:
        return None

    shell_distance = shell_distance if shell_distance is not None else math.sqrt(3.0) * voxel_size / 2.0

    xmin, xmax, ymin, ymax, zmin, zmax = surface.bounds
    xs = _aligned_axis(xmin - shell_distance, xmax + shell_distance, voxel_size)
    ys = _aligned_axis(ymin - shell_distance, ymax + shell_distance, voxel_size)
    zs = _aligned_axis(zmin - shell_distance, zmax + shell_distance, voxel_size)

    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        return None

    chunk_x = max(1, MAX_DISTANCE_CHUNK_POINTS // max(len(ys), 1))
    kept_points = []

    for z in zs:
        z_column = np.full(len(ys), z, dtype=float)
        for start in range(0, len(xs), chunk_x):
            xs_chunk = xs[start : start + chunk_x]
            xx, yy = np.meshgrid(xs_chunk, ys, indexing="xy")
            points = np.column_stack((xx.ravel(), yy.ravel(), np.repeat(z_column, len(xs_chunk))))

            probe = pv.PolyData(points)
            probe.compute_implicit_distance(surface, inplace=True)
            distances = np.abs(np.asarray(probe.point_data["implicit_distance"]))
            keep_mask = distances <= shell_distance
            if np.any(keep_mask):
                kept_points.append(points[keep_mask])

    if not kept_points:
        return None

    quantized = np.round(np.vstack(kept_points) / voxel_size).astype(np.int64)
    unique_quantized = np.unique(quantized, axis=0)
    shell_points = unique_quantized.astype(np.float64) * voxel_size

    return pv.PolyData(shell_points)
