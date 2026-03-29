from __future__ import annotations

import os
import sys
from pathlib import Path

import pyvista as pv


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

from refactor_code.paths import hook_baseline_terrain_ply_path, hook_baseline_terrain_vtk_path

DEFAULT_SITE = os.environ.get("B2026_BASELINE_SITE", "city")
DEFAULT_VOXEL_SIZE = int(os.environ.get("B2026_BASELINE_VOXEL_SIZE", "1"))
DEFAULT_INPUT_VTK = hook_baseline_terrain_vtk_path(DEFAULT_SITE, DEFAULT_VOXEL_SIZE)
DEFAULT_OUTPUT_PLY = hook_baseline_terrain_ply_path(DEFAULT_SITE, DEFAULT_VOXEL_SIZE)


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return Path(raw).expanduser()


INPUT_VTK = env_path("B2026_BASELINE_INPUT_VTK", DEFAULT_INPUT_VTK)
OUTPUT_PLY = env_path("B2026_BASELINE_OUTPUT_PLY", DEFAULT_OUTPUT_PLY)


def write_ascii_point_ply(points, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("element face 0\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{float(x):.9g} {float(y):.9g} {float(z):.9g}\n")


def main() -> None:
    if not INPUT_VTK.exists():
        raise FileNotFoundError(f"Input VTK was not found: {INPUT_VTK}")

    poly = pv.read(INPUT_VTK)
    if poly.n_points == 0:
        raise ValueError(f"No points were found in {INPUT_VTK}")

    write_ascii_point_ply(poly.points, OUTPUT_PLY)
    print(f"[baseline_vtk_export] input: {INPUT_VTK}")
    print(f"[baseline_vtk_export] points: {poly.n_points}")
    print(f"[baseline_vtk_export] output: {OUTPUT_PLY}")


if __name__ == "__main__":
    main()
