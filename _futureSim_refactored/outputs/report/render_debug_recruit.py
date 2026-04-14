from __future__ import annotations

"""
Debug renderer for recruit_ diagnostic arrays broadcast to VTK canopy voxels.

Produces one image per recruit_ variable, colouring voxels by that variable
and leaving everything else white.  Output goes to:

    {render_root}/debugRecruit/{site}_{scenario}_yr{year}_{array_name}_with-legend.png

Uses the same camera presets, render settings and compose pipeline as the
main proposal renderer.
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
    WHITE_RGB,
    load_font,
    render_png,
)


# ── colour palettes ────────────────────────────────────────────────────────

# Categorical variables — hand-picked distinct colours
CATEGORICAL_PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    "recruit_isNewTree": {
        "true":  (46, 139, 87),    # sea green
        "false": (200, 200, 200),  # light grey (original trees)
    },
    "recruit_hasbeenReplanted": {
        "true":  (65, 105, 225),   # royal blue
        "false": (200, 200, 200),
    },
    "recruit_mechanism": {
        "node-rewild":         (255, 127, 14),  # orange
        "under-canopy":        (148, 103, 189), # purple
        "under-canopy-linked": (80, 200, 180),  # teal
        "ground":              (44, 160, 44),   # green
        "none":                (200, 200, 200),
    },
}

# For recruit_year: a sequential colour ramp (blue → red)
# For recruit_mortality_cohort: a sequential colour ramp (green → red)
# For recruit_mortality_rate: a sequential colour ramp (yellow → dark red)

def _lerp_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ── colouring functions ────────────────────────────────────────────────────

def _colour_categorical(mesh: pv.PolyData, array_name: str) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Colour by a categorical string array. Returns (rgb, legend_entries)."""
    palette = CATEGORICAL_PALETTES.get(array_name, {})
    # Normalise to lowercase strings so True/False booleans match "true"/"false"
    values = np.char.lower(np.asarray(mesh[array_name]).astype(str))
    rgb = np.full((mesh.n_points, 3), 255, dtype=np.uint8)

    legend = []
    for label, colour in palette.items():
        mask = values == label
        if np.any(mask):
            rgb[mask] = np.asarray(colour, dtype=np.uint8)
            legend.append((label, colour))

    # Catch any values not in the palette (skip nan/none/false — leave white)
    known = set(palette.keys())
    skip = {"none", "false", "nan"}
    for val in np.unique(values):
        if val not in known and val not in skip:
            rgb[values == val] = np.asarray((128, 128, 128), dtype=np.uint8)

    return rgb, legend


def _colour_recruit_year(mesh: pv.PolyData) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Colour by recruit_year. NaN/none = white, else blue→red ramp by year."""
    raw = np.asarray(mesh["recruit_year"])
    rgb = np.full((mesh.n_points, 3), 255, dtype=np.uint8)

    # Handle both numeric and string representations
    if np.issubdtype(raw.dtype, np.floating):
        years = raw.copy()
        valid = ~np.isnan(years)
    else:
        str_vals = raw.astype(str)
        valid = np.zeros(mesh.n_points, dtype=bool)
        years = np.zeros(mesh.n_points, dtype=float)
        for i in range(mesh.n_points):
            if str_vals[i] not in ("none", "nan", ""):
                try:
                    years[i] = float(str_vals[i])
                    valid[i] = True
                except ValueError:
                    pass

    if not np.any(valid):
        return rgb, [("no recruits", (200, 200, 200))]

    valid_years = years[valid]
    ymin, ymax = valid_years.min(), valid_years.max()

    c_low = (50, 100, 200)   # blue — early recruits
    c_high = (200, 50, 50)   # red — late recruits

    if ymax > ymin:
        t = (years - ymin) / (ymax - ymin)
    else:
        t = np.zeros_like(years)

    for i in np.where(valid)[0]:
        rgb[i] = np.asarray(_lerp_rgb(c_low, c_high, t[i]), dtype=np.uint8)

    legend = [
        (f"yr {int(ymin)}", c_low),
        (f"yr {int(ymax)}", c_high),
    ]
    return rgb, legend


def _colour_numeric(mesh: pv.PolyData, array_name: str, c_low: tuple[int, int, int], c_high: tuple[int, int, int], fmt: str = ".2f") -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Colour by a numeric float array. NaN = white, else ramp c_low→c_high."""
    values = np.asarray(mesh[array_name], dtype=float)
    rgb = np.full((mesh.n_points, 3), 255, dtype=np.uint8)

    valid = ~np.isnan(values)
    if not np.any(valid):
        return rgb, [("no data", (200, 200, 200))]

    vmin, vmax = values[valid].min(), values[valid].max()
    if vmax > vmin:
        t = (values - vmin) / (vmax - vmin)
    else:
        t = np.zeros_like(values)

    for i in np.where(valid)[0]:
        rgb[i] = np.asarray(_lerp_rgb(c_low, c_high, t[i]), dtype=np.uint8)

    legend = [
        (f"{vmin:{fmt}}", c_low),
        (f"{vmax:{fmt}}", c_high),
    ]
    return rgb, legend


def _colour_cohort(mesh: pv.PolyData) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Colour by recruit_mortality_cohort. Discrete integer bins with distinct colours."""
    values = np.asarray(mesh["recruit_mortality_cohort"], dtype=float)
    rgb = np.full((mesh.n_points, 3), 255, dtype=np.uint8)

    valid = ~np.isnan(values)
    if not np.any(valid):
        return rgb, [("no data", (200, 200, 200))]

    unique_cohorts = sorted(set(int(v) for v in values[valid]))

    # Generate distinct colours using a hue spread
    n = len(unique_cohorts)
    cohort_colours = {}
    for idx, cohort in enumerate(unique_cohorts):
        # Spread across hue from green (0.33) → red (0.0)
        import colorsys
        hue = 0.33 - (0.33 * idx / max(n - 1, 1))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.85)
        colour = (int(r * 255), int(g * 255), int(b * 255))
        cohort_colours[cohort] = colour

    int_values = np.where(valid, np.nan_to_num(values, nan=-1).astype(int), -1)
    for cohort, colour in cohort_colours.items():
        mask = valid & (int_values == cohort)
        if np.any(mask):
            rgb[mask] = np.asarray(colour, dtype=np.uint8)

    # Legend: show min, mid, max cohort
    legend = []
    for cohort in unique_cohorts:
        dbh_lo = (cohort - 1) * 10
        dbh_hi = cohort * 10
        legend.append((f"cohort {cohort} ({dbh_lo}-{dbh_hi}cm)", cohort_colours[cohort]))

    return rgb, legend


# ── all debug layers ───────────────────────────────────────────────────────

DEBUG_LAYERS = [
    "recruit_isNewTree",
    "recruit_hasbeenReplanted",
    "recruit_mechanism",
    "recruit_year",
    "recruit_mortality_rate",
    "recruit_mortality_cohort",
    "ground_recruitment",
    "node_rewild_recruitment",
    "under_canopy_recruitment",
    "under_canopy_linked_recruitment",
    "sim_nodes_zones",
]


def _colour_sim_nodes(mesh: pv.PolyData) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Colour each sim_Nodes zone with a random, visually distinct colour. -1 = white."""
    sim_nodes = np.asarray(mesh["sim_Nodes"], dtype=int)
    rgb = np.full((mesh.n_points, 3), 255, dtype=np.uint8)

    unique_ids = sorted(set(sim_nodes[sim_nodes >= 0]))
    if not unique_ids:
        return rgb, [("no sim_Nodes zones", (200, 200, 200))]

    # Generate maximally distinct colours via golden-ratio hue spacing
    import colorsys
    rng = np.random.RandomState(42)
    # Shuffle so adjacent IDs don't get similar colours
    order = list(range(len(unique_ids)))
    rng.shuffle(order)

    id_colours: dict[int, tuple[int, int, int]] = {}
    for rank, idx in enumerate(order):
        hue = (rank * 0.618033988749895) % 1.0  # golden ratio
        sat = 0.6 + rng.random() * 0.3           # 0.6-0.9
        val = 0.65 + rng.random() * 0.25         # 0.65-0.9
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        id_colours[unique_ids[idx]] = (int(r * 255), int(g * 255), int(b * 255))

    for node_id, colour in id_colours.items():
        mask = sim_nodes == node_id
        rgb[mask] = np.asarray(colour, dtype=np.uint8)

    # Legend: show count + a few sample IDs
    legend = [
        (f"{len(unique_ids)} unique sim_Nodes zones", (100, 100, 100)),
        (f"{int((sim_nodes >= 0).sum()):,} voxels coloured", (150, 150, 150)),
    ]
    # Show up to 8 largest zones
    from collections import Counter
    counts = Counter(sim_nodes[sim_nodes >= 0].tolist())
    top = counts.most_common(8)
    for node_id, count in top:
        legend.append((f"ID {node_id} ({count:,} voxels)", id_colours[node_id]))

    return rgb, legend


def _colour_zone_recruitment(
    mesh: pv.PolyData,
    zone_array_name: str,
    mechanism_value: str,
    zone_label: str,
    c_tree: tuple[int, int, int],
    c_zone: tuple[int, int, int],
) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Generic zone-recruitment renderer: zone voxels + recruited tree canopies."""
    rgb = np.full((mesh.n_points, 3), 255, dtype=np.uint8)

    # Zone voxels (>= 0 means active at some year)
    zone_arr = np.asarray(mesh[zone_array_name])
    zone_mask = zone_arr >= 0
    rgb[zone_mask] = np.asarray(c_zone, dtype=np.uint8)

    # Recruited tree canopies on top
    mech = np.char.lower(np.asarray(mesh["recruit_mechanism"]).astype(str))
    tree_mask = mech == mechanism_value
    rgb[tree_mask] = np.asarray(c_tree, dtype=np.uint8)

    legend = [
        (f"{mechanism_value} canopy ({int(tree_mask.sum()):,} voxels)", c_tree),
        (f"{zone_label} ({int(zone_mask.sum()):,} voxels)", c_zone),
    ]
    return rgb, legend


def _colour_ground_recruitment(mesh: pv.PolyData) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Ground-recruited tree canopies + recruitable ground voxels."""
    return _colour_zone_recruitment(
        mesh,
        zone_array_name="scenario_rewildGroundRecruitZone",
        mechanism_value="ground",
        zone_label="recruitable ground",
        c_tree=(230, 120, 30),     # orange
        c_zone=(120, 190, 80),     # green
    )


def _colour_node_rewild_recruitment(mesh: pv.PolyData) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Node-rewild recruited canopies + node-rewild zone voxels."""
    return _colour_zone_recruitment(
        mesh,
        zone_array_name="scenario_nodeRewildRecruitZone",
        mechanism_value="node-rewild",
        zone_label="node-rewild zone (sim_Nodes)",
        c_tree=(200, 50, 50),      # red
        c_zone=(100, 160, 220),    # blue
    )


def _colour_under_canopy_recruitment(mesh: pv.PolyData) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Under-canopy recruited canopies + under-canopy zone voxels."""
    return _colour_zone_recruitment(
        mesh,
        zone_array_name="scenario_underCanopyRecruitZone",
        mechanism_value="under-canopy",
        zone_label="under-canopy zone (node_CanopyID)",
        c_tree=(180, 50, 180),     # magenta
        c_zone=(170, 140, 210),    # lavender
    )


def _colour_under_canopy_linked_recruitment(mesh: pv.PolyData) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    """Under-canopy-linked recruited canopies + linked zone voxels."""
    return _colour_zone_recruitment(
        mesh,
        zone_array_name="scenario_underCanopyLinkedRecruitZone",
        mechanism_value="under-canopy-linked",
        zone_label="under-canopy-linked zone (node_CanopyID)",
        c_tree=(30, 150, 130),     # deep teal
        c_zone=(150, 220, 210),    # pale teal
    )


def colour_for_layer(mesh: pv.PolyData, layer: str) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]:
    if layer in CATEGORICAL_PALETTES:
        return _colour_categorical(mesh, layer)
    elif layer == "recruit_year":
        return _colour_recruit_year(mesh)
    elif layer == "recruit_mortality_rate":
        return _colour_numeric(mesh, layer, (255, 255, 100), (180, 0, 0), fmt=".3f")
    elif layer == "recruit_mortality_cohort":
        return _colour_cohort(mesh)
    elif layer == "ground_recruitment":
        return _colour_ground_recruitment(mesh)
    elif layer == "node_rewild_recruitment":
        return _colour_node_rewild_recruitment(mesh)
    elif layer == "under_canopy_recruitment":
        return _colour_under_canopy_recruitment(mesh)
    elif layer == "under_canopy_linked_recruitment":
        return _colour_under_canopy_linked_recruitment(mesh)
    elif layer == "sim_nodes_zones":
        return _colour_sim_nodes(mesh)
    else:
        raise ValueError(f"Unknown debug layer: {layer}")


# ── rendering ──────────────────────────────────────────────────────────────

def compose_with_legend(
    base_image_path: Path,
    title: str,
    legend_entries: list[tuple[str, tuple[int, int, int]]],
    output_path: Path,
) -> None:
    """Add a title and legend strip to the bottom of a rendered image."""
    base = Image.open(base_image_path).convert("RGB")
    title_font = load_font(36)
    label_font = load_font(19)
    swatch = 20
    row_gap = 8

    # Legend height
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
    parser = argparse.ArgumentParser(description="Render debug-recruit diagnostic views.")
    parser.add_argument("--site", default="trimmed-parade", help="Site key or 'all'.")
    parser.add_argument("--scenario", default="positive", help="Scenario key or 'all'.")
    parser.add_argument("--years", nargs="*", type=int, default=[10, 60, 180], help="Years to render.")
    parser.add_argument("--output-mode", default="validation", choices=["canonical", "validation"])
    parser.add_argument("--layers", nargs="*", default=None, help="Specific layers to render. Default: all.")
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
    layers: list[str] | None = None,
    mesh: pv.PolyData | None = None,
) -> list[Path]:
    """Render debug recruit layers. Pass *mesh* to skip disk read."""
    if mesh is None:
        mesh = pv.read(vtk_path)
    if layers is None:
        layers = DEBUG_LAYERS

    render_root = engine_output_validation_dir(output_mode) / "renders" / "debugRecruit"
    outputs: list[Path] = []

    available = set(mesh.point_data.keys())
    _composite_layers = {"ground_recruitment", "node_rewild_recruitment", "under_canopy_recruitment", "under_canopy_linked_recruitment", "sim_nodes_zones"}
    for layer in layers:
        if layer not in available and layer not in _composite_layers:
            print(f"  Skipping {layer} — not in VTK")
            continue

        rgb, legend_entries = colour_for_layer(mesh, layer)

        stem = f"{site}_{scenario}_yr{year}_{layer}"
        raw_path = render_root / f"{stem}.__temp__.png"
        final_path = render_root / f"{stem}_with-legend.png"

        title = f"{site} {scenario} yr{year} — {layer}"
        render_png(mesh, site, raw_path, rgb)
        compose_with_legend(raw_path, title, legend_entries, final_path)
        raw_path.unlink(missing_ok=True)
        print(final_path)
        outputs.append(final_path)

    return outputs


def main() -> None:
    args = parse_args()
    layers = args.layers if args.layers else DEBUG_LAYERS

    for site, scenario, year, vtk_path in iter_targets(args):
        render_debug_target(site, scenario, year, vtk_path, args.output_mode, layers=layers)


if __name__ == "__main__":
    main()
