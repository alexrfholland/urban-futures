from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv


def _derive_node_csv_path(vtk_path: Path) -> Path:
    stem = vtk_path.name
    match = re.match(r"(?P<prefix>.+)_yr(?P<year>\d+)_state_with_indicators\.vtk$", stem)
    if not match:
        raise ValueError(
            "Could not derive nodeDF path from VTK name. "
            "Pass --node-csv explicitly."
        )
    site_dir = vtk_path.parent.name
    run_root = vtk_path.parents[2]
    prefix = match.group("prefix")
    year = match.group("year")
    return run_root / "feature-locations" / site_dir / f"{prefix}_nodeDF_yr{year}.csv"


def _normalize_forest_size(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values).astype(str)
    normalized = np.char.lower(np.char.strip(normalized))
    normalized[np.isin(normalized, ["", "nan", "none"])] = "none"
    return normalized


def _node_resistance_lookup(node_df: pd.DataFrame) -> pd.Series:
    required = {"NodeID", "CanopyResistance"}
    missing = required - set(node_df.columns)
    if missing:
        raise ValueError(f"nodeDF is missing required columns: {sorted(missing)}")

    lookup_df = node_df.loc[:, ["NodeID", "CanopyResistance"]].copy()
    lookup_df["NodeID"] = pd.to_numeric(lookup_df["NodeID"], errors="coerce")
    lookup_df["CanopyResistance"] = pd.to_numeric(lookup_df["CanopyResistance"], errors="coerce")
    lookup_df = lookup_df.dropna(subset=["NodeID"]).drop_duplicates(subset=["NodeID"], keep="last")
    lookup_df["NodeID"] = lookup_df["NodeID"].astype(int)
    return lookup_df.set_index("NodeID")["CanopyResistance"]


def build_plotter(
    vtk_path: Path,
    node_csv_path: Path,
    point_size: float,
    nan_color: str = "red",
):
    mesh = pv.read(vtk_path)
    node_df = pd.read_csv(node_csv_path)

    if "forest_size" not in mesh.point_data:
        raise ValueError("VTK must contain forest_size point-data.")

    if "forest_NodeID" in mesh.point_data:
        vtk_node_ids = pd.to_numeric(pd.Series(mesh.point_data["forest_NodeID"]), errors="coerce")
    elif "sim_Nodes" in mesh.point_data:
        vtk_node_ids = pd.to_numeric(pd.Series(mesh.point_data["sim_Nodes"]), errors="coerce")
        vtk_node_ids = vtk_node_ids.where(vtk_node_ids >= 0)
    else:
        raise ValueError("VTK must contain either forest_NodeID or sim_Nodes point-data.")

    forest_size = _normalize_forest_size(mesh.point_data["forest_size"])
    tree_mask = vtk_node_ids.notna().to_numpy() & ~np.isin(
        forest_size,
        ["none", "gone", "early-tree-death"],
    )

    resistance_lookup = _node_resistance_lookup(node_df)
    canopy_resistance = vtk_node_ids.map(resistance_lookup).to_numpy(dtype=float)

    valid_tree_mask = tree_mask & np.isfinite(canopy_resistance)
    missing_tree_mask = tree_mask & ~np.isfinite(canopy_resistance)
    background_mask = ~tree_mask

    mesh = mesh.copy()
    mesh["tree_canopy_resistance"] = np.nan_to_num(canopy_resistance, nan=-1.0)

    plotter = pv.Plotter()
    plotter.set_background("white")

    if background_mask.any():
        plotter.add_mesh(
            mesh.extract_points(background_mask, include_cells=False),
            color="lightgrey",
            opacity=0.18,
            point_size=point_size,
            render_points_as_spheres=True,
        )

    if valid_tree_mask.any():
        plotter.add_mesh(
            mesh.extract_points(valid_tree_mask, include_cells=False),
            scalars="tree_canopy_resistance",
            cmap="viridis",
            clim=(0, 100),
            point_size=point_size,
            render_points_as_spheres=True,
            scalar_bar_args={"title": "CanopyResistance"},
        )

    if missing_tree_mask.any():
        plotter.add_mesh(
            mesh.extract_points(missing_tree_mask, include_cells=False),
            color=nan_color,
            point_size=point_size * 1.2,
            render_points_as_spheres=True,
        )

    plotter.enable_eye_dome_lighting()
    plotter.add_text(
        "\n".join(
            [
                vtk_path.name,
                f"tree voxels: {int(tree_mask.sum()):,}",
                f"valid resistance voxels: {int(valid_tree_mask.sum()):,}",
                f"missing resistance voxels: {int(missing_tree_mask.sum()):,}",
            ]
        ),
        position="upper_left",
        font_size=10,
    )

    return plotter, {
        "tree_voxels": int(tree_mask.sum()),
        "valid_tree_voxels": int(valid_tree_mask.sum()),
        "missing_tree_voxels": int(missing_tree_mask.sum()),
        "node_csv_path": str(node_csv_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Interactive PyVista plot of tree voxels colored by CanopyResistance, with missing values in red."
    )
    parser.add_argument("vtk", type=Path, help="Path to state_with_indicators.vtk")
    parser.add_argument("--node-csv", type=Path, help="Matching integrated nodeDF CSV. If omitted, derive from the VTK path.")
    parser.add_argument("--point-size", type=float, default=4.0, help="Point size for rendering.")
    parser.add_argument("--screenshot", type=Path, help="Optional screenshot output path. Uses off-screen rendering.")
    args = parser.parse_args()

    vtk_path = args.vtk.resolve()
    node_csv_path = args.node_csv.resolve() if args.node_csv else _derive_node_csv_path(vtk_path)

    plotter, summary = build_plotter(
        vtk_path=vtk_path,
        node_csv_path=node_csv_path,
        point_size=float(args.point_size),
    )

    print(f"VTK: {vtk_path}")
    print(f"nodeDF: {summary['node_csv_path']}")
    print(f"tree_voxels: {summary['tree_voxels']}")
    print(f"valid_tree_voxels: {summary['valid_tree_voxels']}")
    print(f"missing_tree_voxels: {summary['missing_tree_voxels']}")

    if args.screenshot:
        screenshot_path = args.screenshot.resolve()
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot_path), auto_close=True)
        print(f"screenshot: {screenshot_path}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
