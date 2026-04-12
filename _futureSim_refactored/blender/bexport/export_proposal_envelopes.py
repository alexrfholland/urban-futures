"""Export proposal-based bioenvelope classification and PLY mesh.

Classifies every voxel in a state VTK by its highest-priority proposal
intervention, then extracts per-category isosurface meshes, merges them,
and writes a single PLY with:

- ``intervention_bioenvelope_ply-string``  (label)
- ``intervention_bioenvelope_ply-int``     (integer code)
- per-family proposal framebuffers         (blender_proposal-*)

Drop-in replacement for export_rewilded_envelopes.py — same output path
and filename convention.

Priority (highest to lowest):
    1. buffer-feature+depaved   (decay buffer on depaved ground)
    2. RECRUIT_FULL
    3. RECRUIT_PARTIAL
    4. COLONISE_FULL_ENVELOPE
    5. COLONISE_PARTIAL_ENVELOPE
    6. COLONISE_FULL_GROUND
    7. DECAY_FULL
    8. deploy-any
    -  none
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import convolve

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent
BLENDER_EXPORT_DIR = Path(__file__).resolve().parent

if str(CODE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT.parent))
if str(BLENDER_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(BLENDER_EXPORT_DIR))

import vtk_to_ply as a_vtk_to_ply

from _futureSim_refactored.sim.setup.constants import (
    BIOENVELOPE_PLY_COLORS,
    BIOENVELOPE_PLY_INT,
    COLONISE_FULL_ENVELOPE,
    COLONISE_FULL_GROUND,
    COLONISE_PARTIAL_ENVELOPE,
    DECAY_FULL,
    RECRUIT_FULL,
    RECRUIT_PARTIAL,
)
from _futureSim_refactored.paths import (
    engine_output_bioenvelope_ply_path,
    engine_output_state_vtk_path,
)
from _futureSim_refactored.blender.bexport.export_rewilded_envelopes import (
    transfer_point_data,
)

# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------
LABEL_NONE = "none"
LABEL_DEPLOY_ANY = "deploy-any"
LABEL_DECAY_DEPAVED = "buffer-feature+depaved"

PRIORITY_LABELS: list[str] = [
    LABEL_DEPLOY_ANY,
    DECAY_FULL,
    COLONISE_FULL_GROUND,
    COLONISE_PARTIAL_ENVELOPE,
    COLONISE_FULL_ENVELOPE,
    RECRUIT_PARTIAL,
    RECRUIT_FULL,
    LABEL_DECAY_DEPAVED,
]

SITES = ["trimmed-parade", "city", "uni"]

BLENDER_PROPOSAL_ATTRS = [
    "blender_proposal-deploy-structure",
    "blender_proposal-decay",
    "blender_proposal-recruit",
    "blender_proposal-colonise",
    "blender_proposal-release-control",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(arr: np.ndarray) -> np.ndarray:
    """Lowercase-strip a string array, coercing object dtype to fixed-width."""
    out = np.asarray(arr).astype("<U64")
    return np.char.lower(np.char.strip(out))


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_intervention_bioenvelope(mesh: pv.PolyData) -> pv.PolyData:
    """Add ``intervention_bioenvelope_ply-string`` and ``-int`` to *mesh*.

    Classifies every point by its highest-priority proposal intervention.
    Operates in-place and returns the same mesh.
    """
    n = mesh.n_points
    result = np.full(n, LABEL_NONE, dtype="<U64")

    # Gate: only voxels inside the bioenvelope are eligible.
    if "scenario_bioEnvelope" in mesh.point_data:
        bio = _norm(mesh.point_data["scenario_bioEnvelope"])
        eligible = bio != "none"
    else:
        eligible = np.ones(n, dtype=bool)

    # --- 8. deploy-any (lowest) ---
    if "proposal_deploy_structure" in mesh.point_data:
        decision = _norm(mesh.point_data["proposal_deploy_structure"])
        result[eligible & (np.char.find(decision, "accepted") >= 0)] = LABEL_DEPLOY_ANY

    # --- 7. decay buffer-feature ---
    if "proposal_decay_intervention" in mesh.point_data:
        decay_interv = _norm(mesh.point_data["proposal_decay_intervention"])
        result[eligible & (decay_interv == DECAY_FULL)] = DECAY_FULL

    # --- 6-4. colonise ---
    if "proposal_colonise_intervention" in mesh.point_data:
        col_interv = _norm(mesh.point_data["proposal_colonise_intervention"])
        result[eligible & (col_interv == COLONISE_FULL_GROUND)] = COLONISE_FULL_GROUND
        result[eligible & (col_interv == COLONISE_PARTIAL_ENVELOPE)] = COLONISE_PARTIAL_ENVELOPE
        result[eligible & (col_interv == COLONISE_FULL_ENVELOPE)] = COLONISE_FULL_ENVELOPE

    # --- 3-2. recruit ---
    if "proposal_recruit_intervention" in mesh.point_data:
        rec_interv = _norm(mesh.point_data["proposal_recruit_intervention"])
        result[eligible & (rec_interv == RECRUIT_PARTIAL)] = RECRUIT_PARTIAL
        result[eligible & (rec_interv == RECRUIT_FULL)] = RECRUIT_FULL

    # --- 1. decay buffer-feature + depaved (highest) ---
    if "proposal_decay_intervention" in mesh.point_data and "scenario_bioEnvelope" in mesh.point_data:
        depaved = np.isin(bio, ["footprint-depaved", "footprint-depaved-connected"])
        result[eligible & (decay_interv == DECAY_FULL) & depaved] = LABEL_DECAY_DEPAVED

    mesh.point_data["intervention_bioenvelope_ply-string"] = result

    # Integer encoding
    int_result = np.zeros(n, dtype=np.int16)
    for label, code in BIOENVELOPE_PLY_INT.items():
        int_result[result == label] = code
    mesh.point_data["intervention_bioenvelope_ply-int"] = int_result

    return mesh


# ---------------------------------------------------------------------------
# Isosurface extraction (with gap filling)
# ---------------------------------------------------------------------------

# 6-connected face-neighbor kernel for 3D convolution
_FACE_KERNEL = np.zeros((3, 3, 3), dtype=int)
_FACE_KERNEL[1, 1, 0] = 1; _FACE_KERNEL[1, 1, 2] = 1
_FACE_KERNEL[1, 0, 1] = 1; _FACE_KERNEL[1, 2, 1] = 1
_FACE_KERNEL[0, 1, 1] = 1; _FACE_KERNEL[2, 1, 1] = 1


def extract_isosurface_filled(
    polydata: pv.PolyData,
    spacing: tuple[float, float, float],
    label: str,
    *,
    min_neighbors: int = 2,
) -> pv.PolyData | None:
    """Grid voxels, fill isolated gaps, then extract isosurface.

    Empty cells with >= *min_neighbors* filled face-neighbors (out of 6)
    are filled before contouring.  This plugs single-voxel holes without
    the destructive erosion step of a morphological close.
    """
    print(f"{label} polydata has {polydata.n_points} points")
    if polydata is None or polydata.n_points == 0:
        return None

    points = polydata.points
    x, y, z = points.T
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    nx = int((x_max - x_min) / spacing[0]) + 1
    ny = int((y_max - y_min) / spacing[1]) + 1
    nz = int((z_max - z_min) / spacing[2]) + 1

    # Build 3D binary volume
    vol = np.zeros((nx, ny, nz), dtype=np.uint8)
    ix = np.clip(((x - x_min) / spacing[0]).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / spacing[1]).astype(int), 0, ny - 1)
    iz = np.clip(((z - z_min) / spacing[2]).astype(int), 0, nz - 1)
    vol[ix, iy, iz] = 1

    # Fill empty cells with enough filled face-neighbors
    filled_before = int(vol.sum())
    neighbor_count = convolve(vol.astype(int), _FACE_KERNEL, mode="constant", cval=0)
    fill_mask = (vol == 0) & (neighbor_count >= min_neighbors)
    vol[fill_mask] = 1
    filled_after = int(vol.sum())
    if filled_after > filled_before:
        print(f"  Gap fill: {filled_after - filled_before} voxels added (min_neighbors={min_neighbors})")

    # Flatten to ImageData -- PyVista uses Fortran (column-major) order
    dims = (nx, ny, nz)
    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=(x_min, y_min, z_min))
    grid.point_data["values"] = (vol.flatten(order="F") * 2).astype(float)

    isosurface = grid.contour(
        isosurfaces=[0.5], scalars="values",
        method="flying_edges", compute_normals=True,
    )
    surface = isosurface.extract_surface()
    if surface.n_points == 0:
        return None

    surface.point_data["scenario_bioEnvelope"] = np.full(surface.n_points, label)
    return surface


# ---------------------------------------------------------------------------
# Isosurface assembly
# ---------------------------------------------------------------------------

def generate_proposal_envelopes(
    voxel_polydata: pv.PolyData,
) -> pv.PolyData | None:
    """Extract per-category isosurfaces, merge, and transfer attributes once."""
    labels = np.asarray(voxel_polydata.point_data["intervention_bioenvelope_ply-string"])
    valid_mask = labels != LABEL_NONE
    print(f"Eligible voxels for proposal envelope: {valid_mask.sum()}")

    if not valid_mask.any():
        return {}

    valid_polydata = voxel_polydata.extract_points(valid_mask)
    unique_labels = np.unique(valid_polydata.point_data["intervention_bioenvelope_ply-string"])

    surfaces: list[pv.PolyData] = []
    for label in unique_labels:
        if label == LABEL_NONE:
            continue
        print(f"\nExtracting isosurface: {label}")
        label_mask = valid_polydata.point_data["intervention_bioenvelope_ply-string"] == label
        subset = valid_polydata.extract_points(label_mask)

        if subset.n_points == 0:
            continue

        surface = extract_isosurface_filled(
            subset, (1.0, 1.0, 1.0), label,
            min_neighbors=2,
        )
        if surface is not None:
            surfaces.append(surface)

    if not surfaces:
        return None

    # Merge first, then ONE KNN transfer from the full eligible set
    combined = surfaces[0].merge(surfaces[1:]) if len(surfaces) > 1 else surfaces[0]
    transfer_attrs = [
        "intervention_bioenvelope_ply-int",
        *BLENDER_PROPOSAL_ATTRS,
        "sim_Turns",
        "sim_averageResistance",
    ]
    transfer_attrs = [a for a in transfer_attrs if a in valid_polydata.point_data]
    combined = transfer_point_data(valid_polydata, combined, transfer_attrs)
    return combined


# ---------------------------------------------------------------------------
# Export (drop-in replacement for export_rewilded_envelopes.export_target)
# ---------------------------------------------------------------------------

def export_target(
    site: str,
    scenario: str,
    year: int,
    vtk_path: Path | None,
    output_mode: str,
    *,
    voxel_size: int = 1,
    mesh: pv.PolyData | None = None,
) -> Path | None:
    """Classify, extract per-category isosurfaces, merge, and write one PLY.

    Same signature and output path as export_rewilded_envelopes.export_target
    so it can be used as a drop-in replacement.
    """
    if mesh is None:
        mesh = pv.read(vtk_path)

    classify_intervention_bioenvelope(mesh)
    print_summary(mesh)

    combined = generate_proposal_envelopes(mesh)

    if combined is None:
        print(f"No proposal envelope output for {site}/{scenario}/yr{year}")
        return None

    print(f"\nCombined mesh: {combined.n_points:,} verts, {combined.n_cells:,} faces")

    candidates = [
        "intervention_bioenvelope_ply-int",
        *BLENDER_PROPOSAL_ATTRS,
        "sim_Turns",
        "sim_averageResistance",
    ]
    attributes = [a for a in candidates if a in combined.point_data]

    # Same output path as old export_rewilded_envelopes
    output_path = engine_output_bioenvelope_ply_path(site, scenario, year, voxel_size, output_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    a_vtk_to_ply.export_polydata_to_ply(
        combined,
        str(output_path),
        attributesToTransfer=attributes,
    )
    print(f"Saved proposal envelope PLY to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def build_rgb_array(mesh: pv.PolyData) -> np.ndarray:
    """Build an (n, 3) uint8 RGB array from the classification field."""
    labels = mesh.point_data["intervention_bioenvelope_ply-string"]
    rgb = np.full((mesh.n_points, 3), 255, dtype=np.uint8)
    for label, color in BIOENVELOPE_PLY_COLORS.items():
        mask = np.asarray(labels) == label
        if mask.any():
            rgb[mask] = np.asarray(color, dtype=np.uint8)
    return rgb


def plot_interactive(mesh: pv.PolyData, *, title: str = "") -> None:
    """Show an interactive 3-D PyVista plot coloured by intervention class."""
    rgb = build_rgb_array(mesh)
    mesh.point_data["RGB"] = rgb

    plotter = pv.Plotter(window_size=(2200, 1600), title=title or "intervention bioenvelope")
    plotter.set_background("white")
    plotter.enable_eye_dome_lighting()
    plotter.add_mesh(
        mesh,
        scalars="RGB",
        rgb=True,
        point_size=4.0,
        render_points_as_spheres=False,
        lighting=False,
    )

    legend_entries = []
    labels = np.asarray(mesh.point_data["intervention_bioenvelope_ply-string"])
    for label in [LABEL_NONE] + PRIORITY_LABELS:
        count = int((labels == label).sum())
        if count > 0:
            r, g, b = BIOENVELOPE_PLY_COLORS[label]
            legend_entries.append([f"{label} ({count:,})", [r / 255, g / 255, b / 255, 1.0]])
    plotter.add_legend(
        legend_entries,
        bcolor=(1, 1, 1, 0.85),
        face="circle",
        size=(0.30, 0.30),
    )

    plotter.show()


def print_summary(mesh: pv.PolyData) -> None:
    """Print counts per intervention label."""
    labels = np.asarray(mesh.point_data["intervention_bioenvelope_ply-string"])
    print(f"\nIntervention bioenvelope classification -- {mesh.n_points:,} points")
    print("-" * 60)
    for label in [LABEL_NONE] + PRIORITY_LABELS:
        count = int((labels == label).sum())
        pct = 100.0 * count / mesh.n_points if mesh.n_points else 0
        print(f"  {label:40s}  {count:>8,}  ({pct:5.1f}%)")
    print("-" * 60)


# ---------------------------------------------------------------------------
# CLI (standalone)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify VTK voxels by proposal-intervention priority, extract isosurface, and export PLY.",
    )
    parser.add_argument("--vtk", type=str, default=None, help="Path to a single state VTK file.")
    parser.add_argument("--site", default="all", help="Site key or 'all'.")
    parser.add_argument("--scenario", default="all", help="Scenario key or 'all'.")
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=[0, 1, 10, 30, 60, 90, 120, 150, 180],
        help="Years to export.",
    )
    parser.add_argument("--voxel-size", type=int, default=1, help="Voxel size. Default: 1.")
    parser.add_argument("--output-mode", default="validation", choices=["canonical", "validation"])
    parser.add_argument("--no-plot", action="store_true", help="Skip interactive plot.")
    parser.add_argument("--plot-only", action="store_true", help="Only plot, skip PLY export.")
    return parser.parse_args()


def iter_targets(args: argparse.Namespace):
    sites = SITES if args.site == "all" else [args.site]
    scenarios = ["positive", "trending"] if args.scenario == "all" else [args.scenario]

    for site in sites:
        if site not in SITES:
            raise KeyError(f"Unknown site: {site}")
        for scenario in scenarios:
            for year in args.years:
                vtk_path = engine_output_state_vtk_path(site, scenario, year, args.voxel_size, args.output_mode)
                if vtk_path.exists():
                    yield site, scenario, year, vtk_path
                else:
                    print(f"Skipping missing VTK: {vtk_path}")


def main() -> None:
    args = parse_args()

    # Single VTK mode
    if args.vtk:
        mesh = pv.read(args.vtk)
        print(f"Loaded {mesh.n_points:,} points from {args.vtk}")
        classify_intervention_bioenvelope(mesh)
        print_summary(mesh)
        if args.plot_only or not args.no_plot:
            site = args.site if args.site != "all" else "unknown"
            scenario = args.scenario if args.scenario != "all" else "unknown"
            plot_interactive(mesh, title=f"{site} / {scenario}")
        return

    # Plot-only mode (single site/scenario/year from resolved path)
    if args.plot_only:
        site = args.site if args.site != "all" else "city"
        scenario = args.scenario if args.scenario != "all" else "positive"
        year = args.years[0] if args.years else 180
        vtk_path = engine_output_state_vtk_path(site, scenario, year, args.voxel_size, args.output_mode)
        mesh = pv.read(str(vtk_path))
        print(f"Loaded {mesh.n_points:,} points")
        classify_intervention_bioenvelope(mesh)
        print_summary(mesh)
        plot_interactive(mesh, title=f"{site} / {scenario} / yr{year}")
        return

    # Batch export mode
    for site, scenario, year, vtk_path in iter_targets(args):
        try:
            export_target(site, scenario, year, vtk_path, args.output_mode, voxel_size=args.voxel_size)
        except Exception as exc:
            print(f"  Proposal envelope export failed ({site}/{scenario}/yr{year}): {exc}")


if __name__ == "__main__":
    main()
