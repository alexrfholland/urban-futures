from __future__ import annotations

"""
Render the custom v3 proposal schema images from assessed VTKs.

This is the checked-in renderer for the custom proposal images previously made
ad hoc under:
    _data-refactored/v3engine_outputs/validation/renders/custom

It uses:
    - the `blender_proposal-*` framebuffer arrays
    - the fixed deadwood base colours
    - the accepted camera presets
    - the accepted PyVista settings:
        render_points_as_spheres = False
        lighting = False
        eye_dome_lighting = True

Outputs:
    - {site}_{scenario}_yr{year}_engine3-proposals_interventions_with-legend.png
    - {site}_{scenario}_yr{year}_engine3-proposals.png
Optional extra variants:
    - {site}_{scenario}_yr{year}_engine3-proposals_interventions.png
    - {site}_{scenario}_yr{year}_engine3-proposals_with-legend.png

Meaning:
    - default:
      - `engine3-proposals_interventions_with-legend`
      - `engine3-proposals`
    - `engine3-proposals_interventions` shows intervention-specific accepted states
    - `engine3-proposals` shows proposal presence only
      any framebuffer value except `0` (not-assessed) and `1` (rejected)
"""

import argparse
import colorsys
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont


CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.paths import (
    engine_output_baseline_state_vtk_path,
    engine_output_state_vtk_path,
    engine_output_validation_dir,
)
from refactor_code.outputs.report.render_forest_size_views import CAMERAS
from refactor_code.outputs.report.render_proposal_schema_v3 import (
    CUSTOM_RENDER_SETTINGS,
    DEADWOOD_BASE_HEX,
    FOREST_SIZE_HEX,
    PROPOSAL_HEX,
    PROPOSAL_INT_MAPPING,
    PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP,
    RELEASE_CONTROL_FOREST_SIZE_KEYS,
    RELEASE_CONTROL_SATURATION,
    WHITE_RGB,
)


INTERVENTIONS_OUTPUT_STEM = "engine3-proposals_interventions"
PROPOSALS_ONLY_OUTPUT_STEM = "engine3-proposals"
TITLE_TEXT_TEMPLATE = "{site} {scenario} yr{year}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render custom v3 proposal schema views.")
    parser.add_argument("--site", default="trimmed-parade", help="Site key or 'all'.")
    parser.add_argument("--scenario", default="positive", help="Scenario key or 'all'.")
    parser.add_argument("--years", nargs="*", type=int, default=[180], help="Years to render.")
    parser.add_argument("--output-mode", default="validation", choices=["canonical", "validation"])
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Also write the plain interventions image and the proposal-only image with legend.",
    )
    parser.add_argument(
        "--with-legend",
        action="store_true",
        help="Deprecated alias for --all-variants.",
    )
    return parser.parse_args()


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


FOREST_SIZE_RGB = {key: _hex_rgb(value) for key, value in FOREST_SIZE_HEX.items()}
DEADWOOD_BASE_RGB = {key: _hex_rgb(value) for key, value in DEADWOOD_BASE_HEX.items()}
PROPOSAL_RGB = {
    array_name: {state: _hex_rgb(hex_value) for state, hex_value in state_map.items()}
    for array_name, state_map in PROPOSAL_HEX.items()
}
PROPOSALS_ONLY_RGB = {
    "blender_proposal-deploy-structure": _hex_rgb("#C05E5E"),
    "blender_proposal-decay": _hex_rgb("#B83B6B"),
    "blender_proposal-recruit": _hex_rgb("#5CB85C"),
    "blender_proposal-colonise": _hex_rgb("#8CCC4F"),
    "blender_proposal-release-control": _hex_rgb("#D4882B"),
}


def _normalize_str_array(values: np.ndarray) -> np.ndarray:
    return np.char.lower(np.asarray(values).astype(str))


def _rgb_to_uint8_array(rgb: tuple[int, int, int], count: int) -> np.ndarray:
    array = np.empty((count, 3), dtype=np.uint8)
    array[:] = np.asarray(rgb, dtype=np.uint8)
    return array


def _release_control_rgb(forest_size: str, state: int) -> tuple[int, int, int] | None:
    if state == 0:
        return None
    base = FOREST_SIZE_RGB.get(forest_size)
    if base is None:
        return None
    saturation_scale = RELEASE_CONTROL_SATURATION.get(state)
    if saturation_scale is None:
        return None
    r, g, b = [component / 255.0 for component in base]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, s * saturation_scale))
    rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
    return (round(rr * 255), round(gg * 255), round(bb * 255))


def interventions_schema_rgb(mesh: pv.PolyData) -> np.ndarray:
    required_arrays = set(PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP)
    missing_arrays = [name for name in required_arrays if name not in mesh.point_data]
    if missing_arrays:
        raise KeyError(f"Missing required proposal framebuffer arrays: {sorted(missing_arrays)}")

    rgb = _rgb_to_uint8_array(WHITE_RGB, mesh.n_points)

    forest_size = _normalize_str_array(mesh["forest_size"]) if "forest_size" in mesh.point_data else np.full(mesh.n_points, "", dtype="<U32")

    for label, color in DEADWOOD_BASE_RGB.items():
        rgb[forest_size == label] = np.asarray(color, dtype=np.uint8)

    release_name = "blender_proposal-release-control"
    if release_name in mesh.point_data:
        release_values = np.asarray(mesh.point_data[release_name]).astype(int)
        for forest_key in RELEASE_CONTROL_FOREST_SIZE_KEYS:
            forest_mask = forest_size == forest_key
            if not np.any(forest_mask):
                continue
            for state in [1, 2, 3]:
                state_mask = forest_mask & (release_values == state)
                if not np.any(state_mask):
                    continue
                color = _release_control_rgb(forest_key, state)
                if color is not None:
                    rgb[state_mask] = np.asarray(color, dtype=np.uint8)

    for array_name in PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP:
        if array_name == release_name or array_name not in mesh.point_data:
            continue
        values = np.asarray(mesh.point_data[array_name]).astype(int)
        color_map = PROPOSAL_RGB.get(array_name, {})
        for state, color in color_map.items():
            mask = values == state
            if np.any(mask):
                rgb[mask] = np.asarray(color, dtype=np.uint8)

    return rgb


def proposals_only_schema_rgb(mesh: pv.PolyData) -> np.ndarray:
    required_arrays = set(PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP)
    missing_arrays = [name for name in required_arrays if name not in mesh.point_data]
    if missing_arrays:
        raise KeyError(f"Missing required proposal framebuffer arrays: {sorted(missing_arrays)}")

    rgb = _rgb_to_uint8_array(WHITE_RGB, mesh.n_points)
    forest_size = _normalize_str_array(mesh["forest_size"]) if "forest_size" in mesh.point_data else np.full(mesh.n_points, "", dtype="<U32")

    for label, color in DEADWOOD_BASE_RGB.items():
        rgb[forest_size == label] = np.asarray(color, dtype=np.uint8)

    for array_name in PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP:
        values = np.asarray(mesh.point_data[array_name]).astype(int)
        proposal_mask = (values != 0) & (values != 1)
        if not np.any(proposal_mask):
            continue
        color = PROPOSALS_ONLY_RGB.get(array_name)
        if color is None:
            continue
        rgb[proposal_mask] = np.asarray(color, dtype=np.uint8)

    return rgb


def render_png(mesh: pv.PolyData, site: str, output_path: Path, rgb_values: np.ndarray) -> None:
    settings = CUSTOM_RENDER_SETTINGS
    camera = CAMERAS[site]
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(int(settings["window_width"]), int(settings["window_height"])),
    )
    plotter.set_background(str(settings["background"]))
    plotter.add_mesh(
        mesh,
        scalars=rgb_values,
        rgb=True,
        render_points_as_spheres=bool(settings["render_points_as_spheres"]),
        point_size=float(settings["point_size"]),
        lighting=bool(settings["lighting"]),
    )
    if bool(settings["eye_dome_lighting"]):
        plotter.enable_eye_dome_lighting()
    plotter.camera_position = [
        camera["position"],
        camera["focal_point"],
        camera["view_up"],
    ]
    plotter.camera.view_angle = camera["view_angle"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(output_path))
    plotter.close()


def _interventions_legend_sections() -> list[tuple[str, list[tuple[str, tuple[int, int, int]]]]]:
    return [
        (
            "Deploy-Structure",
            [
                ("Adapt-Utility-Pole", PROPOSAL_RGB["blender_proposal-deploy-structure"][2]),
                ("Translocated-Log", PROPOSAL_RGB["blender_proposal-deploy-structure"][3]),
                ("Upgrade-Feature", PROPOSAL_RGB["blender_proposal-deploy-structure"][4]),
            ],
        ),
        (
            "Decay",
            [
                ("Buffer-Feature", PROPOSAL_RGB["blender_proposal-decay"][2]),
                ("Brace-Feature", PROPOSAL_RGB["blender_proposal-decay"][3]),
            ],
        ),
        (
            "Recruit",
            [
                ("Buffer-Feature", PROPOSAL_RGB["blender_proposal-recruit"][2]),
                ("Rewild-Ground", PROPOSAL_RGB["blender_proposal-recruit"][3]),
            ],
        ),
        (
            "Colonise",
            [
                ("Rewild-Ground", PROPOSAL_RGB["blender_proposal-colonise"][2]),
                ("Enrich-Envelope", PROPOSAL_RGB["blender_proposal-colonise"][3]),
                ("Roughen-Envelope", PROPOSAL_RGB["blender_proposal-colonise"][4]),
            ],
        ),
        (
            "Deadwood Base",
            [
                ("Fallen", DEADWOOD_BASE_RGB["fallen"]),
                ("Decayed", DEADWOOD_BASE_RGB["decayed"]),
            ],
        ),
    ]


def _proposals_only_legend_sections() -> list[tuple[str, list[tuple[str, tuple[int, int, int]]]]]:
    return [
        (
            "Proposal Families",
            [
                ("Deploy-Structure", PROPOSALS_ONLY_RGB["blender_proposal-deploy-structure"]),
                ("Decay", PROPOSALS_ONLY_RGB["blender_proposal-decay"]),
                ("Recruit", PROPOSALS_ONLY_RGB["blender_proposal-recruit"]),
                ("Colonise", PROPOSALS_ONLY_RGB["blender_proposal-colonise"]),
                ("Release-Control", PROPOSALS_ONLY_RGB["blender_proposal-release-control"]),
            ],
        ),
        (
            "Deadwood Base",
            [
                ("Fallen", DEADWOOD_BASE_RGB["fallen"]),
                ("Decayed", DEADWOOD_BASE_RGB["decayed"]),
            ],
        ),
    ]


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ["Aptos.ttf", "Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_release_control_matrix(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    section_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> int:
    draw.text((x0, y0), "Release-Control", fill=(0, 0, 0), font=section_font)
    y = y0 + 28

    row_labels = ["small", "medium", "large", "senescing", "snag"]
    col_labels = [("rejected", 1), ("reduce", 2), ("eliminate", 3)]
    cell_w = 28
    cell_h = 18
    label_w = 68

    for col_idx, (label, _state) in enumerate(col_labels):
        draw.text((x0 + label_w + col_idx * (cell_w + 8), y), label, fill=(0, 0, 0), font=label_font)
    y += 20

    for row_label in row_labels:
        draw.text((x0, y), row_label, fill=(0, 0, 0), font=label_font)
        for col_idx, (_label, state) in enumerate(col_labels):
            color = _release_control_rgb(row_label, state)
            if color is None:
                color = (255, 255, 255)
            cx = x0 + label_w + col_idx * (cell_w + 8)
            draw.rectangle([cx, y + 2, cx + cell_w, y + 2 + cell_h], fill=color, outline=(0, 0, 0))
        y += cell_h + 10
    return y


def compose_with_legend(
    base_image_path: Path,
    site: str,
    scenario: str,
    year: int,
    output_path: Path,
    *,
    legend_sections: list[tuple[str, list[tuple[str, tuple[int, int, int]]]]],
    include_release_control_matrix: bool,
) -> None:
    base = Image.open(base_image_path).convert("RGB")
    title_font = _load_font(36)
    section_font = _load_font(22)
    label_font = _load_font(19)

    title_height = 72
    legend_width = 360
    padding = 28
    swatch = 18
    row_gap = 10
    section_gap = 18

    canvas = Image.new("RGB", (base.width + legend_width, base.height + title_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 18), TITLE_TEXT_TEMPLATE.format(site=site, scenario=scenario, year=year), fill=(0, 0, 0), font=title_font)
    canvas.paste(base, (0, title_height))

    x0 = base.width + 16
    y = title_height + 20
    for section_name, entries in legend_sections:
        draw.text((x0, y), section_name, fill=(0, 0, 0), font=section_font)
        y += 30
        for label, color in entries:
            draw.rectangle([x0, y + 4, x0 + swatch, y + 4 + swatch], fill=color, outline=(0, 0, 0))
            draw.text((x0 + swatch + 10, y), label, fill=(0, 0, 0), font=label_font)
            y += swatch + row_gap
        y += section_gap

    if include_release_control_matrix:
        y = _draw_release_control_matrix(draw, x0, y, section_font, label_font)
        y += section_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


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


def render_target(site: str, scenario: str, year: int, vtk_path: Path, output_mode: str, all_variants: bool) -> list[Path]:
    mesh = pv.read(vtk_path)
    render_root = engine_output_validation_dir(output_mode) / "renders" / "custom"
    outputs: list[Path] = []

    interventions_stem = f"{site}_{scenario}_yr{year}_{INTERVENTIONS_OUTPUT_STEM}"
    interventions_rgb = interventions_schema_rgb(mesh)
    interventions_legend_path = render_root / f"{interventions_stem}_with-legend.png"
    interventions_base_for_legend = render_root / f"{interventions_stem}.__temp__.png"
    render_png(mesh, site, interventions_base_for_legend, interventions_rgb)
    compose_with_legend(
        interventions_base_for_legend,
        site,
        scenario,
        year,
        interventions_legend_path,
        legend_sections=_interventions_legend_sections(),
        include_release_control_matrix=True,
    )
    outputs.append(interventions_legend_path)
    interventions_base_for_legend.unlink(missing_ok=True)
    if all_variants:
        interventions_base_path = render_root / f"{interventions_stem}.png"
        render_png(mesh, site, interventions_base_path, interventions_rgb)
        outputs.append(interventions_base_path)

    proposals_only_stem = f"{site}_{scenario}_yr{year}_{PROPOSALS_ONLY_OUTPUT_STEM}"
    proposals_only_base_path = render_root / f"{proposals_only_stem}.png"
    proposals_only_rgb = proposals_only_schema_rgb(mesh)
    render_png(mesh, site, proposals_only_base_path, proposals_only_rgb)
    outputs.append(proposals_only_base_path)
    if all_variants:
        proposals_only_legend_path = render_root / f"{proposals_only_stem}_with-legend.png"
        compose_with_legend(
            proposals_only_base_path,
            site,
            scenario,
            year,
            proposals_only_legend_path,
            legend_sections=_proposals_only_legend_sections(),
            include_release_control_matrix=False,
        )
        outputs.append(proposals_only_legend_path)
    return outputs


def main() -> None:
    args = parse_args()
    all_variants = args.all_variants or args.with_legend
    for site, scenario, year, vtk_path in iter_targets(args):
        for output in render_target(site, scenario, year, vtk_path, args.output_mode, all_variants):
            print(output)


if __name__ == "__main__":
    main()
