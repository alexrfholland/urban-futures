from __future__ import annotations

"""
Debug renderer for release-control status broadcast to VTK canopy voxels.

Colours living tree voxels by their release-control status:
  - rejected                 (living, CR too high)
  - reduce-canopy-pruning    (moderated)
  - eliminate-canopy-pruning (autonomous)

Non-living (snag, fallen, etc.) and non-tree voxels are left white.
Output goes to:

    {render_root}/debugReleaseControl/{site}_{scenario}_yr{year}_release_control_with-legend.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
if str(CODE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT.parent))

from _futureSim_refactored.paths import (
    engine_output_baseline_state_vtk_path,
    engine_output_state_vtk_path,
    engine_output_validation_dir,
)
from _futureSim_refactored.outputs.report.render_common import (
    CAMERAS,
    CUSTOM_RENDER_SETTINGS,
    load_font,
    render_png,
)


# Three distinct, saturated colours for the release-control statuses
RELEASE_COLOURS: dict[str, tuple[int, int, int]] = {
    "rejected":                 (220,  50,  50),  # red
    "reduce-canopy-pruning":    (255, 165,   0),  # orange
    "eliminate-canopy-pruning": ( 40, 170,  70),  # green
}


def colour_release_control(mesh: pv.PolyData) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Colour voxels by release-control status. Non-assessed/other = white."""
    rgb = np.full((mesh.n_points, 3), 255, dtype=np.uint8)

    decision = np.asarray(mesh["proposal_release_control"]).astype(str)
    intervention = np.asarray(mesh["proposal_release_control_intervention"]).astype(str)

    accepted = np.char.find(decision, "accepted") >= 0
    rejected = np.char.find(decision, "rejected") >= 0

    reduce_mask = accepted & (intervention == "reduce-canopy-pruning")
    eliminate_mask = accepted & (intervention == "eliminate-canopy-pruning")

    rgb[rejected] = np.asarray(RELEASE_COLOURS["rejected"], dtype=np.uint8)
    rgb[reduce_mask] = np.asarray(RELEASE_COLOURS["reduce-canopy-pruning"], dtype=np.uint8)
    rgb[eliminate_mask] = np.asarray(RELEASE_COLOURS["eliminate-canopy-pruning"], dtype=np.uint8)

    legend = [
        (f"rejected ({int(rejected.sum()):,} voxels)", RELEASE_COLOURS["rejected"]),
        (f"reduce-canopy-pruning ({int(reduce_mask.sum()):,} voxels)", RELEASE_COLOURS["reduce-canopy-pruning"]),
        (f"eliminate-canopy-pruning ({int(eliminate_mask.sum()):,} voxels)", RELEASE_COLOURS["eliminate-canopy-pruning"]),
    ]
    return rgb, legend


# ── rendering ──────────────────────────────────────────────────────────────

def compose_with_legend(
    base_image_path: Path,
    title: str,
    legend_entries: list[tuple[str, tuple[int, int, int]]],
    output_path: Path,
) -> None:
    base = Image.open(base_image_path).convert("RGB")
    title_font = load_font(36)
    label_font = load_font(19)
    swatch = 20
    row_gap = 8

    legend_height = 60 + len(legend_entries) * (swatch + row_gap)
    title_height = 72

    canvas = Image.new("RGB", (base.width, base.height + title_height + legend_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((28, 18), title, fill=(0, 0, 0), font=title_font)
    canvas.paste(base, (0, title_height))

    y = title_height + base.height + 20
    x = 28
    for label, colour in legend_entries:
        draw.rectangle([x, y + 4, x + swatch, y + 4 + swatch], fill=colour, outline=(0, 0, 0))
        draw.text((x + swatch + 10, y), label, fill=(0, 0, 0), font=label_font)
        y += swatch + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render debug-release-control diagnostic views.")
    parser.add_argument("--site", default="trimmed-parade", help="Site key or 'all'.")
    parser.add_argument("--scenario", default="positive", help="Scenario key or 'all'.")
    parser.add_argument("--years", nargs="*", type=int, default=[10, 60, 180], help="Years to render.")
    parser.add_argument("--output-mode", default="validation", choices=["canonical", "validation"])
    return parser.parse_args()


def iter_targets(args: argparse.Namespace):
    sites = list(CAMERAS.keys()) if args.site == "all" else [args.site]
    scenarios = ["positive", "trending", "baseline"] if args.scenario == "all" else [args.scenario]

    for site in sites:
        if site not in CAMERAS:
            raise KeyError(f"Unknown site camera preset: {site}")
        for scenario in scenarios:
            if scenario == "baseline":
                vtk_path = engine_output_baseline_state_vtk_path(site, 1, args.output_mode)
                if vtk_path.exists():
                    yield site, scenario, 0, vtk_path
                else:
                    print(f"Skipping missing VTK: {vtk_path}")
                continue
            for year in args.years:
                vtk_path = engine_output_state_vtk_path(site, scenario, year, 1, args.output_mode)
                if vtk_path.exists():
                    yield site, scenario, year, vtk_path
                else:
                    print(f"Skipping missing VTK: {vtk_path}")


def render_debug_target(
    site: str,
    scenario: str,
    year: int,
    vtk_path: Path | None,
    output_mode: str,
    mesh: pv.PolyData | None = None,
) -> list[Path]:
    """Render debug release-control layer. Pass *mesh* to skip disk read."""
    if mesh is None:
        mesh = pv.read(vtk_path)

    render_root = engine_output_validation_dir(output_mode) / "renders" / "debugReleaseControl"
    outputs: list[Path] = []

    required = {"proposal_release_control", "proposal_release_control_intervention"}
    if not required.issubset(set(mesh.point_data.keys())):
        print(f"  Skipping — missing release-control arrays")
        return outputs

    rgb, legend_entries = colour_release_control(mesh)

    stem = f"{site}_{scenario}_yr{year}_release_control"
    raw_path = render_root / f"{stem}.__temp__.png"
    final_path = render_root / f"{stem}_with-legend.png"

    title = f"{site} {scenario} yr{year} — release control"
    render_png(mesh, site, raw_path, rgb)
    compose_with_legend(raw_path, title, legend_entries, final_path)
    raw_path.unlink(missing_ok=True)
    print(final_path)
    outputs.append(final_path)

    return outputs


def main() -> None:
    args = parse_args()
    for site, scenario, year, vtk_path in iter_targets(args):
        render_debug_target(site, scenario, year, vtk_path, args.output_mode)


if __name__ == "__main__":
    main()
