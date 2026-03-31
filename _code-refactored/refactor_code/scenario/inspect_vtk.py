from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyvista as pv

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.paths import scenario_state_vtk_path


def parse_args():
    parser = argparse.ArgumentParser(description="Open or screenshot a scenario VTK for manual review.")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--site", default="trimmed-parade")
    parser.add_argument("--scenario", default="positive")
    parser.add_argument("--year", type=int, default=180)
    parser.add_argument("--voxel-size", type=int, default=1)
    parser.add_argument("--output-mode", default="validation", choices=["canonical", "validation"])
    parser.add_argument("--scalar", default="scenario_rewilded")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    vtk_path = args.path or scenario_state_vtk_path(
        args.site,
        args.scenario,
        args.year,
        args.voxel_size,
        args.output_mode,
    )
    mesh = pv.read(vtk_path)
    print(vtk_path)
    print(mesh.array_names)

    plotter = pv.Plotter(off_screen=bool(args.screenshot))
    scalars = args.scalar if args.scalar in mesh.array_names else None
    plotter.add_mesh(mesh, scalars=scalars, render_points_as_spheres=True, point_size=3)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(args.screenshot))
        print(args.screenshot)
    else:
        plotter.show()


if __name__ == "__main__":
    main()
