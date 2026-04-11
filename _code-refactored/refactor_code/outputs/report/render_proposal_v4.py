from __future__ import annotations

"""
Render v4 proposal-and-interventions views from assessed VTKs.

Main variant (always rendered):
    - {site}_{scenario}_yr{year}_proposal-and-interventions_with-legend.png

Decay-pathway voxels (senescing, snag, fallen, decayed) are coloured by
`forest_size`; proposal interventions are overlaid on non-decay voxels
only. Release-control is shown as saturation shifts on living tree
sizes. The legend is composed into the bottom whitespace of the render.

Optional variant (`--all-variants`):
    - {site}_{scenario}_yr{year}_proposal-families-only.png
    - {site}_{scenario}_yr{year}_proposal-families-only_with-legend.png

Flat colour per proposal family with no per-intervention detail.
"""

import argparse
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
from refactor_code.outputs.report.render_common import (
    CAMERAS,
    DEADWOOD_BASE_HEX,
    DEADWOOD_BASE_RGB,
    FOREST_SIZE_HEX,
    FOREST_SIZE_RGB,
    PROPOSAL_HEX,
    PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP,
    PROPOSAL_RGB,
    RELEASE_CONTROL_FOREST_SIZE_KEYS,
    RELEASE_CONTROL_SATURATION,
    WHITE_RGB,
    hex_rgb,
    load_font,
    normalize_str_array,
    release_control_rgb,
    render_png,
    rgb_to_uint8_array,
)


OUTPUT_STEM = "proposal-and-interventions"
FAMILIES_ONLY_OUTPUT_STEM = "proposal-families-only"
TITLE_TEXT_TEMPLATE = "{site} {scenario} yr{year}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render v4 proposal-and-interventions views.")
    parser.add_argument("--site", default="trimmed-parade", help="Site key or 'all'.")
    parser.add_argument("--scenario", default="positive", help="Scenario key or 'all'.")
    parser.add_argument("--years", nargs="*", type=int, default=[0, 1, 10, 30, 60, 90, 120, 150, 180], help="Years to render.")
    parser.add_argument("--output-mode", default="validation", choices=["canonical", "validation"])
    parser.add_argument(
        "--also-families-only",
        action="store_true",
        help="Also render the flat proposal-families-only variant (one colour per family).",
    )
    parser.add_argument("--model-base-y", type=int, default=None, help="Pre-computed shared model base y.")
    parser.add_argument("--target-model-width", type=int, default=None, help="Pre-computed shared target model width.")
    return parser.parse_args()


PROPOSAL_FAMILIES_ONLY_RGB: dict[str, tuple[int, int, int]] = {
    "blender_proposal-deploy-structure": hex_rgb("#C05E5E"),
    "blender_proposal-decay": hex_rgb("#B83B6B"),
    "blender_proposal-recruit": hex_rgb("#5CB85C"),
    "blender_proposal-colonise": hex_rgb("#8CCC4F"),
    "blender_proposal-release-control": hex_rgb("#D4882B"),
}
DECAY_PATHWAY_SIZES = ["senescing", "snag", "fallen", "decayed"]


def proposal_families_only_schema_rgb(mesh: pv.PolyData) -> np.ndarray:
    required_arrays = set(PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP)
    missing_arrays = [name for name in required_arrays if name not in mesh.point_data]
    if missing_arrays:
        raise KeyError(f"Missing required proposal framebuffer arrays: {sorted(missing_arrays)}")

    rgb = rgb_to_uint8_array(WHITE_RGB, mesh.n_points)
    forest_size = normalize_str_array(mesh["forest_size"]) if "forest_size" in mesh.point_data else np.full(mesh.n_points, "", dtype="<U32")

    for label, color in DEADWOOD_BASE_RGB.items():
        rgb[forest_size == label] = np.asarray(color, dtype=np.uint8)

    for array_name in PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP:
        values = np.asarray(mesh.point_data[array_name]).astype(int)
        proposal_mask = (values != 0) & (values != 1)
        if not np.any(proposal_mask):
            continue
        color = PROPOSAL_FAMILIES_ONLY_RGB.get(array_name)
        if color is None:
            continue
        rgb[proposal_mask] = np.asarray(color, dtype=np.uint8)

    return rgb


def proposal_and_interventions_rgb(mesh: pv.PolyData) -> np.ndarray:
    """Colour the full decay pathway (senescing/snag/fallen/decayed) by
    forest_size and overlay proposal interventions on the non-decay voxels."""
    required_arrays = set(PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP)
    missing_arrays = [name for name in required_arrays if name not in mesh.point_data]
    if missing_arrays:
        raise KeyError(f"Missing required proposal framebuffer arrays: {sorted(missing_arrays)}")

    rgb = rgb_to_uint8_array(WHITE_RGB, mesh.n_points)

    forest_size = normalize_str_array(mesh["forest_size"]) if "forest_size" in mesh.point_data else np.full(mesh.n_points, "", dtype="<U32")

    # Base layer: color all decay-pathway sizes by their forest_size colour
    for label in DECAY_PATHWAY_SIZES:
        color = FOREST_SIZE_RGB.get(label)
        if color is not None:
            rgb[forest_size == label] = np.asarray(color, dtype=np.uint8)

    # Mask out decay pathway voxels — they keep their forest_size colour only
    decay_mask = np.zeros(mesh.n_points, dtype=bool)
    for label in DECAY_PATHWAY_SIZES:
        decay_mask |= forest_size == label
    non_decay = ~decay_mask

    # Release-control layer (saturation shift on living sizes)
    release_name = "blender_proposal-release-control"
    if release_name in mesh.point_data:
        release_values = np.asarray(mesh.point_data[release_name]).astype(int)
        for forest_key in RELEASE_CONTROL_FOREST_SIZE_KEYS:
            forest_mask = forest_size == forest_key
            if not np.any(forest_mask):
                continue
            for state in [1, 2, 3]:
                state_mask = forest_mask & (release_values == state) & non_decay
                if not np.any(state_mask):
                    continue
                color = release_control_rgb(forest_key, state)
                if color is not None:
                    rgb[state_mask] = np.asarray(color, dtype=np.uint8)

    # Proposal layers on top (only non-decay voxels)
    for array_name in PROPOSAL_LAYER_ORDER_BOTTOM_TO_TOP:
        if array_name == release_name or array_name not in mesh.point_data:
            continue
        values = np.asarray(mesh.point_data[array_name]).astype(int)
        color_map = PROPOSAL_RGB.get(array_name, {})
        for state, color in color_map.items():
            mask = (values == state) & non_decay
            if np.any(mask):
                rgb[mask] = np.asarray(color, dtype=np.uint8)

    return rgb


def _proposal_and_interventions_legend_sections() -> list[tuple[str, list[tuple[str, tuple[int, int, int]]]]]:
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
                ("Smaller-Patches-Rewild\n(higher mortality)", PROPOSAL_RGB["blender_proposal-recruit"][2]),
                ("Larger-Patches-Rewild\n(lower mortality)", PROPOSAL_RGB["blender_proposal-recruit"][3]),
            ],
        ),
        (
            "Colonise",
            [
                ("Larger-Patches-Rewild\n(lower mortality)", PROPOSAL_RGB["blender_proposal-colonise"][2]),
                ("Enrich-Envelope", PROPOSAL_RGB["blender_proposal-colonise"][3]),
                ("Roughen-Envelope", PROPOSAL_RGB["blender_proposal-colonise"][4]),
            ],
        ),
    ]


def _proposal_families_only_legend_sections() -> list[tuple[str, list[tuple[str, tuple[int, int, int]]]]]:
    return [
        (
            "Proposal Families",
            [
                ("Deploy-Structure", PROPOSAL_FAMILIES_ONLY_RGB["blender_proposal-deploy-structure"]),
                ("Decay", PROPOSAL_FAMILIES_ONLY_RGB["blender_proposal-decay"]),
                ("Recruit", PROPOSAL_FAMILIES_ONLY_RGB["blender_proposal-recruit"]),
                ("Colonise", PROPOSAL_FAMILIES_ONLY_RGB["blender_proposal-colonise"]),
                ("Release-Control", PROPOSAL_FAMILIES_ONLY_RGB["blender_proposal-release-control"]),
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


def _draw_release_control_matrix(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    section_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> int:
    draw.text((x0, y0), "Release-Control", fill=(0, 0, 0), font=section_font)
    y = y0 + 28

    # Columns: forest sizes (vertical text), Rows: intervention labels
    col_sizes = ["small", "medium", "large", "senescing", "snag"]
    row_interventions = [
        ("Rejected", 1),
        ("Reduce-Canopy-Pruning", 2),
        ("Eliminate-Canopy-Pruning", 3),
    ]
    cell_w = 24
    cell_h = 16
    col_spacing = cell_w + 4
    small_font = load_font(13)
    vertical_text_height = 58

    # Measure row label width from longest intervention name
    row_label_bbox = label_font.getbbox(row_interventions[2][0])  # longest
    label_w = (row_label_bbox[2] - row_label_bbox[0]) + 10
    cells_x0 = x0 + label_w

    # Vertical column headers
    for col_idx, size_label in enumerate(col_sizes):
        cx = cells_x0 + col_idx * col_spacing + cell_w // 2
        txt_img = Image.new("RGBA", (vertical_text_height, 16), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((2, 0), size_label.capitalize(), fill=(0, 0, 0), font=small_font)
        rotated = txt_img.rotate(90, expand=True)
        draw._image.paste(rotated, (cx - rotated.width // 2, y), rotated)

    y += vertical_text_height

    for row_label, state in row_interventions:
        draw.text((x0, y), row_label, fill=(0, 0, 0), font=label_font)
        for col_idx, size_label in enumerate(col_sizes):
            color = release_control_rgb(size_label, state)
            if color is None:
                color = (255, 255, 255)
            cx = cells_x0 + col_idx * col_spacing
            draw.rectangle([cx, y + 2, cx + cell_w, y + 2 + cell_h], fill=color, outline=(0, 0, 0))
        y += cell_h + 8

    return y


def _draw_decay_lifecycle_matrix(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    section_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> int:
    """Draw a matrix of lifecycle phases enabled under Brace / Buffer.

    Columns are senescing, snag, fallen, decayed with vertical headers.
    Enabled cells show the forest_size colour; disabled cells are empty.
    """
    col_phases = ["senescing", "snag", "fallen", "decayed"]
    rows = [
        ("Lifecycle phases enabled\nunder Brace", [True, True, False, False]),
        ("Lifecycle phases enabled\nunder Buffer", [True, True, True, True]),
    ]
    cell_w = 24
    cell_h = 16
    col_spacing = cell_w + 4
    small_font = load_font(13)
    vertical_text_height = 58
    line_height = 16
    row_text_lines = 2  # each row label is 2 lines
    row_height = max(cell_h, row_text_lines * line_height)
    y = y0

    # Measure label width from longest line in row labels
    max_line_w = 0
    for row_label, _ in rows:
        for line in row_label.split("\n"):
            bbox = small_font.getbbox(line)
            max_line_w = max(max_line_w, bbox[2] - bbox[0])
    label_w = max_line_w + 10
    cells_x0 = x0 + label_w

    # Vertical column headers (rotated text)
    for col_idx, phase in enumerate(col_phases):
        cx = cells_x0 + col_idx * col_spacing + cell_w // 2
        txt_img = Image.new("RGBA", (vertical_text_height, 16), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((2, 0), phase.capitalize(), fill=(0, 0, 0), font=small_font)
        rotated = txt_img.rotate(90, expand=True)
        draw._image.paste(rotated, (cx - rotated.width // 2, y), rotated)

    y += vertical_text_height

    for row_label, enabled in rows:
        # Draw multi-line label in small font
        for line_idx, line in enumerate(row_label.split("\n")):
            draw.text((x0, y + line_idx * line_height), line, fill=(0, 0, 0), font=small_font)
        # Draw cells vertically centred with text block
        cell_y = y + (row_text_lines * line_height - cell_h) // 2
        for col_idx, (phase, is_enabled) in enumerate(zip(col_phases, enabled)):
            cx = cells_x0 + col_idx * col_spacing
            if is_enabled:
                color = FOREST_SIZE_RGB.get(phase, (200, 200, 200))
                draw.rectangle([cx, cell_y, cx + cell_w, cell_y + cell_h], fill=color, outline=(0, 0, 0))
            else:
                draw.rectangle([cx, cell_y, cx + cell_w, cell_y + cell_h], fill=(255, 255, 255), outline=(180, 180, 180))
        y += row_height + 8

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
    include_decay_lifecycle_matrix: bool = False,
) -> None:
    """Compose a rendered image with a legend panel on the right.

    When *include_decay_lifecycle_matrix* is True the lifecycle-phase
    matrix is drawn inline immediately after the Decay section entries
    (identified by section name starting with "Decay").
    """
    base = Image.open(base_image_path).convert("RGB")
    title_font = load_font(36)
    section_font = load_font(22)
    label_font = load_font(19)

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
        if include_decay_lifecycle_matrix and section_name.startswith("Decay"):
            y += 4
            y = _draw_decay_lifecycle_matrix(draw, x0, y, label_font, label_font)
        y += section_gap

    if include_release_control_matrix:
        y = _draw_release_control_matrix(draw, x0, y, section_font, label_font)
        y += section_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _measure_raw_model_bounds(base_image_path: Path) -> dict[str, int]:
    """Return bounds of the non-white content in a raw render."""
    arr = np.array(Image.open(base_image_path).convert("RGB"))
    non_white_rows = np.where(np.any(arr < 250, axis=(1, 2)))[0]
    non_white_cols = np.where(np.any(arr < 250, axis=(0, 2)))[0]
    return {
        "top": int(non_white_rows[0]) if len(non_white_rows) else 0,
        "bottom": int(non_white_rows[-1]) if len(non_white_rows) else 800,
        "left": int(non_white_cols[0]) if len(non_white_cols) else 0,
        "right": int(non_white_cols[-1]) if len(non_white_cols) else 2200,
    }


# Fixed model-base y-position so all sites align their bottom here.
# Set from the tallest site (city) during render_target.
_SHARED_MODEL_BASE_Y: int | None = None


def compose_with_bottom_legend(
    base_image_path: Path,
    site: str,
    scenario: str,
    year: int,
    output_path: Path,
    *,
    legend_sections: list[tuple[str, list[tuple[str, tuple[int, int, int]]]]],
    include_release_control_matrix: bool,
    include_decay_lifecycle_matrix: bool = False,
    model_base_y: int | None = None,
    target_model_width: int | None = None,
) -> None:
    """Compose legend into the existing whitespace of the rendered image.

    Draws a title in the top whitespace and the legend columns in the
    bottom whitespace. No canvas expansion — stays at render resolution.

    If *model_base_y* is given, the image is shifted so the model's bottom
    row lands at that y-coordinate, giving consistent vertical alignment
    across sites.

    If *target_model_width* is given, the base image is uniformly scaled
    so its model content width matches the target, keeping the model
    visually consistent across sites with different camera distances.
    """
    base = Image.open(base_image_path).convert("RGB")
    arr = np.array(base)

    # Find where model content starts/ends vertically and horizontally
    non_white_rows = np.where(np.any(arr < 250, axis=(1, 2)))[0]
    non_white_cols = np.where(np.any(arr < 250, axis=(0, 2)))[0]
    content_top = int(non_white_rows[0]) if len(non_white_rows) else 0
    content_bottom = int(non_white_rows[-1]) if len(non_white_rows) else base.height // 2
    content_left = int(non_white_cols[0]) if len(non_white_cols) else 50
    content_right = int(non_white_cols[-1]) if len(non_white_cols) else base.width - 50

    # Scale the base image if target_model_width is set
    if target_model_width is not None:
        current_width = content_right - content_left
        if current_width > 0 and current_width != target_model_width:
            scale = target_model_width / current_width
            new_w = round(base.width * scale)
            new_h = round(base.height * scale)
            base = base.resize((new_w, new_h), Image.LANCZOS)
            # Re-measure after scaling
            arr = np.array(base)
            non_white_rows = np.where(np.any(arr < 250, axis=(1, 2)))[0]
            non_white_cols = np.where(np.any(arr < 250, axis=(0, 2)))[0]
            content_top = int(non_white_rows[0]) if len(non_white_rows) else 0
            content_bottom = int(non_white_rows[-1]) if len(non_white_rows) else base.height // 2
            content_left = int(non_white_cols[0]) if len(non_white_cols) else 50
            content_right = int(non_white_cols[-1]) if len(non_white_cols) else base.width - 50

    content_height = content_bottom - content_top

    title_font = load_font(36)
    section_font = load_font(22)
    label_font = load_font(19)
    swatch = 20
    row_gap = 8

    # Compute legend height from the tallest column
    entry_h = swatch + row_gap  # 28px per entry
    section_header_h = 30
    col_heights = []
    for section_name, entries in legend_sections:
        h = section_header_h + len(entries) * entry_h
        if include_decay_lifecycle_matrix and section_name.startswith("Decay"):
            h += 6 + 58 + 2 * (16 + 8)  # vertical text + 2 rows
        col_heights.append(h)
    if include_release_control_matrix:
        # header 28 + vertical text 58 + 3 rows × (16+8)
        col_heights.append(28 + 58 + 3 * 24)
    legend_height = max(col_heights) if col_heights else 0

    bottom_pad = 100

    if model_base_y is not None:
        # Shift so model bottom lands at model_base_y
        shift_up = content_bottom - model_base_y
        gap_pad = 80
    else:
        # Auto layout
        spare = base.height - content_height - legend_height - bottom_pad
        top_pad = max(60, spare // 2)
        gap_pad = max(60, spare - top_pad)
        shift_up = content_top - top_pad

    canvas_w, canvas_h = 2200, 1600
    # Centre the (possibly scaled) image horizontally
    x_offset = (canvas_w - base.width) // 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(base, (x_offset, -shift_up))
    draw = ImageDraw.Draw(canvas)

    content_bottom -= shift_up
    content_left += x_offset
    content_right += x_offset

    # Title above content, aligned with model left edge
    draw.text((content_left, 20), TITLE_TEXT_TEMPLATE.format(site=site, scenario=scenario, year=year), fill=(0, 0, 0), font=title_font)

    # Legend starts gap_pad below model content
    legend_top = content_bottom + gap_pad

    # Distribute legend columns evenly across model width
    num_cols = len(legend_sections) + (1 if include_release_control_matrix else 0)
    available_width = content_right - content_left
    col_width = available_width // num_cols if num_cols else available_width

    for col_idx_iter, (section_name, entries) in enumerate(legend_sections):
        x = content_left + col_idx_iter * col_width
        y = legend_top
        draw.text((x, y), section_name, fill=(0, 0, 0), font=section_font)
        y += 30
        for label, color in entries:
            draw.rectangle([x, y + 4, x + swatch, y + 4 + swatch], fill=color, outline=(0, 0, 0))
            lines = label.split("\n")
            draw.text((x + swatch + 10, y), lines[0], fill=(0, 0, 0), font=label_font)
            if len(lines) > 1:
                for extra_line in lines[1:]:
                    y += 20
                    draw.text((x + swatch + 10, y), extra_line, fill=(0, 0, 0), font=label_font)
            y += swatch + row_gap
        if include_decay_lifecycle_matrix and section_name.startswith("Decay"):
            y += 6
            y = _draw_decay_lifecycle_matrix(draw, x, y, label_font, label_font)

    if include_release_control_matrix:
        x = content_left + len(legend_sections) * col_width
        y = legend_top
        _draw_release_control_matrix(draw, x, y, section_font, label_font)

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


def render_target(
    site: str,
    scenario: str,
    year: int,
    vtk_path: Path | None,
    output_mode: str,
    *,
    also_families_only: bool = False,
    model_base_y: int | None = None,
    target_model_width: int | None = None,
    mesh: pv.PolyData | None = None,
) -> list[Path]:
    if mesh is None:
        mesh = pv.read(vtk_path)
    render_root = engine_output_validation_dir(output_mode) / "renders"
    outputs: list[Path] = []

    # Main variant: proposal-and-interventions with bottom legend
    main_stem = f"{site}_{scenario}_yr{year}_{OUTPUT_STEM}"
    main_rgb = proposal_and_interventions_rgb(mesh)
    main_base_for_legend = render_root / f"{main_stem}.__temp__.png"
    main_legend_path = render_root / f"{main_stem}_with-legend.png"
    render_png(mesh, site, main_base_for_legend, main_rgb)
    compose_with_bottom_legend(
        main_base_for_legend,
        site,
        scenario,
        year,
        main_legend_path,
        legend_sections=_proposal_and_interventions_legend_sections(),
        include_release_control_matrix=True,
        include_decay_lifecycle_matrix=True,
        model_base_y=model_base_y,
        target_model_width=target_model_width,
    )
    outputs.append(main_legend_path)
    main_base_for_legend.unlink(missing_ok=True)

    # Optional: proposal-families-only variant (flat colour per family)
    if also_families_only:
        families_stem = f"{site}_{scenario}_yr{year}_{FAMILIES_ONLY_OUTPUT_STEM}"
        families_base_path = render_root / f"{families_stem}.png"
        families_rgb = proposal_families_only_schema_rgb(mesh)
        render_png(mesh, site, families_base_path, families_rgb)
        outputs.append(families_base_path)
        families_legend_path = render_root / f"{families_stem}_with-legend.png"
        compose_with_legend(
            families_base_path,
            site,
            scenario,
            year,
            families_legend_path,
            legend_sections=_proposal_families_only_legend_sections(),
            include_release_control_matrix=False,
        )
        outputs.append(families_legend_path)

    return outputs


def main() -> None:
    args = parse_args()

    targets = list(iter_targets(args))
    if not targets:
        return

    # Use pre-computed values if provided, otherwise measure from main variant
    if args.model_base_y is not None and args.target_model_width is not None:
        model_base_y = args.model_base_y
        target_model_width = args.target_model_width
    else:
        render_root = engine_output_validation_dir(args.output_mode) / "renders"
        max_bottom = 0
        max_width = 0
        for site, scenario, year, vtk_path in targets:
            stem = f"{site}_{scenario}_yr{year}_{OUTPUT_STEM}"
            raw_path = render_root / f"{stem}.__measure__.png"
            mesh = pv.read(vtk_path)
            rgb = proposal_and_interventions_rgb(mesh)
            render_png(mesh, site, raw_path, rgb)
            bounds = _measure_raw_model_bounds(raw_path)
            max_bottom = max(max_bottom, bounds["bottom"])
            model_w = bounds["right"] - bounds["left"]
            max_width = max(max_width, model_w)
            raw_path.unlink(missing_ok=True)

        multi = len(targets) > 1
        model_base_y = (max_bottom - 100) if multi else None
        target_model_width = max_width if multi else None

    for site, scenario, year, vtk_path in targets:
        for output in render_target(
            site,
            scenario,
            year,
            vtk_path,
            args.output_mode,
            also_families_only=args.also_families_only,
            model_base_y=model_base_y,
            target_model_width=target_model_width,
        ):
            print(output)


if __name__ == "__main__":
    main()
