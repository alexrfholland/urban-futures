from __future__ import annotations

import argparse
import ast
import colorsys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import xarray as xr
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = REPO_ROOT / "_data-refactored/model-outputs/generated-states/simv3-7/output/feature-locations"
DEFAULT_DATA_ROOT = REPO_ROOT / "data/revised/final"


def _derive_node_csv_path(site: str, scenario: str, year: int) -> Path:
    return DEFAULT_RUN_ROOT / site / f"{site}_{scenario}_1_nodeDF_yr{year}.csv"


def _derive_with_resistance_path(site: str) -> Path:
    return DEFAULT_DATA_ROOT / site / f"{site}_1_voxelArray_withResistance.nc"


def _build_highlight_table(node_df: pd.DataFrame, ds: xr.Dataset) -> pd.DataFrame:
    required = {"tree_number", "NodeID", "x", "y", "z", "CanopyResistance"}
    missing = required - set(node_df.columns)
    if missing:
        raise ValueError(f"nodeDF is missing required columns: {sorted(missing)}")

    na_df = node_df[node_df["CanopyResistance"].isna()].copy().reset_index(drop=True)
    if na_df.empty:
        return na_df

    canopy_mask = (
        (ds["site_canopy_isCanopyCorrected"].values == 1.0)
        | (ds["road_canopy_isCanopy"].values == 1.0)
    )
    canopy_xy = np.column_stack(
        [ds["centroid_x"].values[canopy_mask], ds["centroid_y"].values[canopy_mask]]
    )
    canopy_node_ids = ds["node_CanopyID"].values[canopy_mask]
    canopy_kdtree = cKDTree(canopy_xy)

    target_node_ids: list[int | None] = []
    nearby_group_lists: list[list[int]] = []
    diagnoses: list[str] = []

    for _, row in na_df.iterrows():
        x = float(row["x"])
        y = float(row["y"])
        node_id = int(row["NodeID"])

        idxs = canopy_kdtree.query_ball_point([x, y], r=10.0)
        nearby_ids = canopy_node_ids[idxs] if idxs else np.array([], dtype=int)
        assigned_ids = sorted({int(v) for v in nearby_ids.tolist() if int(v) > 0})
        nearby_group_lists.append(assigned_ids)

        target_node_id = None
        if assigned_ids:
            dists = np.sqrt((canopy_xy[idxs, 0] - x) ** 2 + (canopy_xy[idxs, 1] - y) ** 2)
            assigned_mask = nearby_ids > 0
            if assigned_mask.any():
                local_idx = np.argmin(dists[assigned_mask])
                target_node_id = int(nearby_ids[assigned_mask][local_idx])

        if not idxs:
            diagnosis = "no canopy-mask voxels within 10m"
        elif target_node_id == node_id:
            diagnosis = "has canopy-mask voxels assigned to own node"
        elif target_node_id is not None:
            diagnosis = "nearby canopy-mask voxels assigned to other node(s) only"
        else:
            diagnosis = "nearby canopy-mask voxels exist but none assigned to any node"

        target_node_ids.append(target_node_id)
        diagnoses.append(diagnosis)

    na_df["target_canopy_node_id"] = target_node_ids
    na_df["nearby_canopy_node_ids"] = nearby_group_lists
    na_df["diagnosis"] = diagnoses
    return na_df


def _color_table(count: int) -> list[str]:
    colors = []
    for i in range(max(count, 1)):
        hue = i / max(count, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors.append("#%02x%02x%02x" % tuple(int(c * 255) for c in rgb))
    return colors


def _subset_polydata(points: np.ndarray) -> pv.PolyData:
    poly = pv.PolyData(points)
    return poly


def build_plotter(
    site: str,
    scenario: str,
    year: int,
    node_csv_path: Path,
    with_resistance_path: Path,
    point_size: float,
    tree_point_size: float,
):
    node_df = pd.read_csv(node_csv_path)
    ds = xr.open_dataset(with_resistance_path)

    highlight_df = _build_highlight_table(node_df, ds)
    if highlight_df.empty:
        raise ValueError("No NA CanopyResistance trees found in the requested nodeDF.")

    points = np.column_stack(
        [ds["centroid_x"].values, ds["centroid_y"].values, ds["centroid_z"].values]
    )
    node_canopy_ids = ds["node_CanopyID"].values

    plotter = pv.Plotter()
    plotter.set_background("white")

    plotter.add_mesh(
        _subset_polydata(points),
        color="lightgrey",
        opacity=0.08,
        point_size=point_size,
        render_points_as_spheres=True,
    )

    colors = _color_table(len(highlight_df))
    legend_entries: list[tuple[str, str]] = []
    summary_lines = [
        f"{site} / {scenario} / yr{year}",
        f"NA CanopyResistance trees: {len(highlight_df)}",
    ]

    for color, (_, row) in zip(colors, highlight_df.iterrows()):
        tree_number = int(row["tree_number"])
        node_id = int(row["NodeID"])
        target_node_id = row["target_canopy_node_id"]

        tree_points = np.array([[float(row["x"]), float(row["y"]), float(row["z"])]])
        plotter.add_mesh(
            _subset_polydata(tree_points),
            color=color,
            point_size=tree_point_size,
            render_points_as_spheres=True,
        )

        if pd.notna(target_node_id):
            canopy_mask = node_canopy_ids == int(target_node_id)
            if canopy_mask.any():
                plotter.add_mesh(
                    _subset_polydata(points[canopy_mask]),
                    color=color,
                    point_size=point_size * 1.25,
                    render_points_as_spheres=True,
                )

        label = (
            f"tree {tree_number} / node {node_id} -> canopy {int(target_node_id)}"
            if pd.notna(target_node_id)
            else f"tree {tree_number} / node {node_id} -> no canopy group"
        )
        summary_lines.append(label)
        legend_entries.append((label[:60], color))

    plotter.enable_eye_dome_lighting()
    plotter.add_text("\n".join(summary_lines[:18]), position="upper_left", font_size=10)
    plotter.add_legend(legend_entries[:20], bcolor="white", face="circle")

    summary = {
        "site": site,
        "scenario": scenario,
        "year": year,
        "node_csv_path": str(node_csv_path),
        "with_resistance_path": str(with_resistance_path),
        "na_tree_count": int(len(highlight_df)),
        "highlight_rows": highlight_df[
            ["tree_number", "NodeID", "target_canopy_node_id", "diagnosis", "nearby_canopy_node_ids"]
        ].to_dict(orient="records"),
    }
    ds.close()
    return plotter, summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive PyVista plot for year-0 trees with NA CanopyResistance. "
            "Each NA tree gets its own colour, and the canopy group it falls into is shown in the same colour."
        )
    )
    parser.add_argument("--site", required=True, choices=["trimmed-parade", "city", "uni"])
    parser.add_argument("--scenario", default="positive")
    parser.add_argument("--year", type=int, default=0)
    parser.add_argument("--node-csv", type=Path)
    parser.add_argument("--with-resistance", type=Path)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--tree-point-size", type=float, default=18.0)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()

    node_csv_path = args.node_csv.resolve() if args.node_csv else _derive_node_csv_path(args.site, args.scenario, args.year)
    with_resistance_path = (
        args.with_resistance.resolve() if args.with_resistance else _derive_with_resistance_path(args.site)
    )

    plotter, summary = build_plotter(
        site=args.site,
        scenario=args.scenario,
        year=int(args.year),
        node_csv_path=node_csv_path,
        with_resistance_path=with_resistance_path,
        point_size=float(args.point_size),
        tree_point_size=float(args.tree_point_size),
    )

    print(f"nodeDF: {summary['node_csv_path']}")
    print(f"withResistance: {summary['with_resistance_path']}")
    print(f"NA trees: {summary['na_tree_count']}")
    for row in summary["highlight_rows"]:
        print(
            f"tree {row['tree_number']} node {row['NodeID']} -> canopy {row['target_canopy_node_id']} | "
            f"{row['diagnosis']} | nearby {row['nearby_canopy_node_ids']}"
        )

    if args.screenshot:
        screenshot_path = args.screenshot.resolve()
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot_path), auto_close=True)
        print(f"screenshot: {screenshot_path}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
