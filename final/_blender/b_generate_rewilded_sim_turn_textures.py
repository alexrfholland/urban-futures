import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image
from matplotlib import colormaps


TARGET_SITES = ("trimmed-parade", "city")
TARGET_YEAR = 180
VOXEL_SIZE = 1.0
VALUE_KEY = "sim_Turns"
TEXTURE_SUBDIR = Path("textures/rewilded_sim_turns")
PLY_SUBDIR = Path("ply")
COLORMAP_NAME = "viridis"

PROJECTION_SPECS = {
    "z_pos": {"u_axis": "x", "v_axis": "y", "depth_axis": "z", "extreme": "max"},
    "z_neg": {"u_axis": "x", "v_axis": "y", "depth_axis": "z", "extreme": "min"},
    "x_pos": {"u_axis": "y", "v_axis": "z", "depth_axis": "x", "extreme": "max"},
    "x_neg": {"u_axis": "y", "v_axis": "z", "depth_axis": "x", "extreme": "min"},
    "y_pos": {"u_axis": "x", "v_axis": "z", "depth_axis": "y", "extreme": "max"},
    "y_neg": {"u_axis": "x", "v_axis": "z", "depth_axis": "y", "extreme": "min"},
}

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def quantize_index(coords: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor((coords / voxel_size) + 0.5).astype(np.int32)


def normalize_values(values: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    if value_max <= value_min:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values.astype(np.float32) - value_min) / (value_max - value_min)
    return np.clip(normalized, 0.0, 1.0)


def parse_scenario_from_stem(site: str, stem: str) -> str | None:
    prefix = f"{site}_"
    suffix = f"_scenarioYR{TARGET_YEAR}"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        return None

    middle = stem[len(prefix) : -len(suffix)]
    if middle.endswith("_1"):
        scenario = middle[:-2]
        return scenario or None
    return None


def matching_ply_candidates(site: str, scenario: str | None, surface_kind: str) -> list[str]:
    if scenario:
        basename = f"{site}_{scenario}_1_{surface_kind}_scenarioYR{TARGET_YEAR}.ply"
    else:
        basename = f"{site}_1_{surface_kind}_scenarioYR{TARGET_YEAR}.ply"
    return [basename]


def existing_ply_candidates(vtk_path: Path, site: str, scenario: str | None, surface_kind: str) -> list[str]:
    ply_dir = vtk_path.parent / PLY_SUBDIR
    return [
        basename
        for basename in matching_ply_candidates(site, scenario, surface_kind)
        if (ply_dir / basename).exists()
    ]


def select_surface_mask(poly: pv.PolyData, surface_kind: str) -> np.ndarray | None:
    if VALUE_KEY not in poly.point_data:
        return None

    values = np.asarray(poly.point_data[VALUE_KEY], dtype=float)
    value_mask = np.isfinite(values) & (values >= 0)

    if surface_kind == "envelope":
        if "scenario_bioEnvelope" not in poly.point_data:
            return None
        mask = np.asarray(poly.point_data["scenario_bioEnvelope"]) != "none"
    elif surface_kind == "ground":
        if "maskForRewilding" in poly.point_data:
            mask = np.asarray(poly.point_data["maskForRewilding"]).astype(bool)
        elif "scenario_rewilded" in poly.point_data:
            mask = np.asarray(poly.point_data["scenario_rewilded"]) != "none"
        else:
            return None
    else:
        return None

    return mask & value_mask


def build_projection(points: np.ndarray, values: np.ndarray, spec: dict) -> dict | None:
    u_idx = AXIS_INDEX[spec["u_axis"]]
    v_idx = AXIS_INDEX[spec["v_axis"]]
    d_idx = AXIS_INDEX[spec["depth_axis"]]

    u_indices = quantize_index(points[:, u_idx], VOXEL_SIZE)
    v_indices = quantize_index(points[:, v_idx], VOXEL_SIZE)
    depth = points[:, d_idx]

    frame = pd.DataFrame(
        {
            "u_idx": u_indices,
            "v_idx": v_indices,
            "depth": depth,
            "value": values,
        }
    )

    grouped = frame.groupby(["u_idx", "v_idx"], sort=False)["depth"]
    best = grouped.idxmax() if spec["extreme"] == "max" else grouped.idxmin()
    selected = frame.loc[best].copy()
    if selected.empty:
        return None

    u_min = int(selected["u_idx"].min())
    v_min = int(selected["v_idx"].min())
    u_max = int(selected["u_idx"].max())
    v_max = int(selected["v_idx"].max())
    width = u_max - u_min + 1
    height = v_max - v_min + 1

    values_grid = np.full((height, width), np.nan, dtype=np.float32)
    alpha_grid = np.zeros((height, width), dtype=np.uint8)

    cols = (selected["u_idx"].to_numpy() - u_min).astype(np.int32)
    rows = (height - 1 - (selected["v_idx"].to_numpy() - v_min)).astype(np.int32)
    values_grid[rows, cols] = selected["value"].to_numpy(dtype=np.float32)
    alpha_grid[rows, cols] = 255

    return {
        "values_grid": values_grid,
        "alpha_grid": alpha_grid,
        "u_min_index": u_min,
        "v_min_index": v_min,
        "width": width,
        "height": height,
    }


def save_projection_images(
    output_dir: Path,
    texture_prefix: str,
    projection_name: str,
    values_grid: np.ndarray,
    alpha_grid: np.ndarray,
    value_min: float,
    value_max: float,
) -> tuple[str, str]:
    finite_mask = np.isfinite(values_grid)
    filled_values = np.where(finite_mask, values_grid, value_min)
    normalized = normalize_values(filled_values, value_min, value_max)

    raw_u16 = np.round(normalized * 65535.0).astype(np.uint16)
    raw_image_path = output_dir / f"{texture_prefix}_{projection_name}.png"
    Image.fromarray(raw_u16, mode="I;16").save(raw_image_path)

    mask_path = output_dir / f"{texture_prefix}_{projection_name}_mask.png"
    Image.fromarray(alpha_grid, mode="L").save(mask_path)

    colormap = colormaps[COLORMAP_NAME]
    rgba = (colormap(normalized) * 255.0).astype(np.uint8)
    rgba[..., 3] = alpha_grid
    preview_path = output_dir / f"{texture_prefix}_{projection_name}_preview.png"
    Image.fromarray(rgba, mode="RGBA").save(preview_path)

    return raw_image_path.name, preview_path.name, mask_path.name


def write_texture_set(
    vtk_path: Path,
    surface_kind: str,
    points: np.ndarray,
    values: np.ndarray,
    output_dir: Path,
    site: str,
    scenario: str | None,
    ply_candidates: list[str],
) -> None:
    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))
    texture_prefix = f"{vtk_path.stem}_{surface_kind}_{VALUE_KEY}"

    metadata = {
        "site": site,
        "scenario": scenario,
        "year": TARGET_YEAR,
        "surface_kind": surface_kind,
        "value_key": VALUE_KEY,
        "voxel_size": VOXEL_SIZE,
        "colormap": COLORMAP_NAME,
        "value_min": value_min,
        "value_max": value_max,
        "source_vtk": str(vtk_path),
        "ply_candidates": ply_candidates,
        "projections": {},
    }

    for projection_name, spec in PROJECTION_SPECS.items():
        projection = build_projection(points, values, spec)
        if projection is None:
            continue

        raw_filename, preview_filename, mask_filename = save_projection_images(
            output_dir,
            texture_prefix,
            projection_name,
            projection["values_grid"],
            projection["alpha_grid"],
            value_min,
            value_max,
        )

        metadata["projections"][projection_name] = {
            "u_axis": spec["u_axis"],
            "v_axis": spec["v_axis"],
            "depth_axis": spec["depth_axis"],
            "extreme": spec["extreme"],
            "u_min_index": projection["u_min_index"],
            "v_min_index": projection["v_min_index"],
            "width": projection["width"],
            "height": projection["height"],
            "raw_texture": raw_filename,
            "mask_texture": mask_filename,
            "preview_texture": preview_filename,
        }

    metadata_path = output_dir / f"{texture_prefix}_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {metadata_path}")


def process_vtk(vtk_path: Path) -> None:
    site = vtk_path.parent.name
    scenario = parse_scenario_from_stem(site, vtk_path.stem)
    print(f"Processing {vtk_path.name} for site={site}, scenario={scenario}")

    poly = pv.read(vtk_path)
    values = np.asarray(poly.point_data[VALUE_KEY], dtype=float)
    output_dir = vtk_path.parent / TEXTURE_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for surface_kind in ("envelope", "ground"):
        ply_candidates = existing_ply_candidates(vtk_path, site, scenario, surface_kind)
        if not ply_candidates:
            print(f"Skipping {surface_kind}: no matching PLY candidates on disk")
            continue

        mask = select_surface_mask(poly, surface_kind)
        if mask is None or not np.any(mask):
            print(f"Skipping {surface_kind}: no valid mask")
            continue

        surface_points = np.asarray(poly.points[mask], dtype=float)
        surface_values = values[mask]
        print(f"Generating {surface_kind} textures from {len(surface_points)} voxels")
        write_texture_set(
            vtk_path,
            surface_kind,
            surface_points,
            surface_values,
            output_dir,
            site,
            scenario,
            ply_candidates,
        )


def main():
    for site in TARGET_SITES:
        site_dir = Path(f"data/revised/final/{site}")
        for vtk_path in sorted(site_dir.glob(f"*scenarioYR{TARGET_YEAR}.vtk")):
            process_vtk(vtk_path)


if __name__ == "__main__":
    main()
