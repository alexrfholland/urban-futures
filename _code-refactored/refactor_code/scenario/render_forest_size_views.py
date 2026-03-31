from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.paths import (
    engine_output_baseline_state_vtk_path,
    engine_output_state_vtk_path,
    engine_output_validation_dir,
)


GREY_REST = (242, 242, 242, 255)
FOREST_SIZE_RGBA = {
    "small": (170, 219, 94, 255),
    "medium": (154, 185, 222, 255),
    "large": (249, 159, 118, 255),
    "senescing": (235, 155, 197, 255),
    "snag": (252, 227, 88, 255),
    "fallen": (130, 203, 185, 255),
    "artificial": (255, 0, 0, 255),
}
BIOENVELOPE_RGBA = {
    "none": (242, 242, 242, 255),
    "exoskeleton": (217, 99, 140, 255),
    "brownroof": (184, 122, 56, 255),
    "otherground": (117, 163, 196, 255),
    "rewilded": (92, 184, 87, 255),
    "footprint-depaved": (235, 191, 84, 255),
    "livingfacade": (76, 168, 153, 255),
    "greenroof": (140, 204, 79, 255),
    "node-rewilded": (92, 184, 87, 255),
}
PROPOSAL_RGBA = {
    "none": GREY_REST,
    "decay-other": (168, 76, 115, 255),
    "decay_buffer-feature": (217, 106, 147, 255),
    "decay_brace-feature": (234, 178, 198, 255),
    "recruit-other": (77, 113, 142, 255),
    "recruit_rewild-ground": (79, 147, 194, 255),
    "recruit_buffer-feature": (155, 195, 226, 255),
    "release-control-other": (154, 106, 37, 255),
    "release-control_eliminate-pruning": (217, 139, 43, 255),
    "release-control_reduce-pruning": (241, 194, 125, 255),
    "colonise-other": (78, 124, 84, 255),
    "colonise_rewild-ground": (92, 184, 92, 255),
    "colonise_enrich-envelope": (111, 196, 106, 255),
    "colonise_roughen-envelope": (166, 216, 155, 255),
    "deploy-structure-other": (168, 74, 74, 255),
    "deploy-structure_adapt-utility-pole": (214, 92, 92, 255),
    "deploy-structure_upgrade-feature": (231, 138, 122, 255),
}
PROPOSAL_HYBRID_RGBA = {
    "none": GREY_REST,
    "decay_buffer-feature": (210, 88, 128, 255),
    "decay_brace-feature": (236, 179, 199, 255),
    "recruit_rewild-ground": (63, 130, 191, 255),
    "recruit_buffer-feature": (158, 199, 230, 255),
    "release-control_eliminate-pruning": (212, 136, 34, 255),
    "release-control_reduce-pruning": (241, 198, 122, 255),
    "colonise_rewild-ground": (67, 168, 92, 255),
    "colonise_enrich-envelope": (118, 198, 107, 255),
    "colonise_roughen-envelope": (181, 221, 155, 255),
    "deploy-structure_adapt-utility-pole": (204, 83, 83, 255),
    "deploy-structure_upgrade-feature": (229, 145, 120, 255),
}
PROPOSAL_PRIORITY = [
    "proposal_colonise",
    "proposal_recruit",
    "proposal_release_control",
    "proposal_decay",
    "proposal_deploy_structure",
]

CAMERAS = {
    "trimmed-parade": {
        "position": (-710.5999, 155.0484, 780.0399),
        "focal_point": (52.8332, 109.8565, 57.0),
        "view_up": (0.6873, -0.0101, 0.7263),
        "view_angle": 30.0,
    },
    "city": {
        "position": (827.5661, 49.1329, 880.7060),
        "focal_point": (288.2490, -22.3298, 333.8400),
        "view_up": (-0.7116, -0.0063, 0.7026),
        "view_angle": 28.1718,
    },
    "uni": {
        "position": (-76.7619, -879.4925, 863.1532),
        "focal_point": (-13.3853, 44.2705, 63.1832),
        "view_up": (-0.0381, 0.6557, 0.7541),
        "view_angle": 30.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render classic, merged, and proposal view image sequences.")
    parser.add_argument("--site", default="all", help="Site key or 'all'.")
    parser.add_argument("--scenario", default="all", help="Scenario key, 'baseline', or 'all'.")
    parser.add_argument("--years", nargs="*", type=int, default=None, help="Years to render.")
    parser.add_argument("--voxel-size", type=int, default=1)
    parser.add_argument("--output-mode", default="validation", choices=["canonical", "validation"])
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--window-width", type=int, default=2200)
    parser.add_argument("--window-height", type=int, default=1600)
    return parser.parse_args()


def _normalize_str_array(values: np.ndarray) -> np.ndarray:
    return np.char.lower(np.asarray(values).astype(str))


def _empty_rgba(size: int, color: tuple[int, int, int, int]) -> np.ndarray:
    rgba = np.zeros((size, 4), dtype=np.uint8)
    rgba[:] = np.asarray(color, dtype=np.uint8)
    return rgba


def _forest_mask(forest_size_values: np.ndarray) -> np.ndarray:
    normalized = _normalize_str_array(forest_size_values)
    return np.isin(normalized, list(FOREST_SIZE_RGBA.keys()))


def classic_rgba(mesh: pv.PolyData) -> np.ndarray:
    forest_size = _normalize_str_array(mesh["forest_size"])
    rgba = _empty_rgba(mesh.n_points, GREY_REST)
    for label, color in FOREST_SIZE_RGBA.items():
        rgba[forest_size == label] = np.asarray(color, dtype=np.uint8)
    return rgba


def merged_rgba(mesh: pv.PolyData) -> np.ndarray:
    forest_size = _normalize_str_array(mesh["forest_size"])
    if "scenario_bioEnvelope" in mesh.point_data:
        bio_envelope = _normalize_str_array(mesh["scenario_bioEnvelope"])
    else:
        bio_envelope = np.full(mesh.n_points, "none", dtype="<U32")
    rgba = _empty_rgba(mesh.n_points, GREY_REST)
    tree_mask = _forest_mask(mesh["forest_size"])

    for label, color in BIOENVELOPE_RGBA.items():
        if label == "none":
            continue
        rgba[(bio_envelope == label) & (~tree_mask)] = np.asarray(color, dtype=np.uint8)

    for label, color in FOREST_SIZE_RGBA.items():
        rgba[forest_size == label] = np.asarray(color, dtype=np.uint8)

    return rgba


def proposal_rgba(mesh: pv.PolyData) -> np.ndarray:
    rgba = _empty_rgba(mesh.n_points, GREY_REST)
    labels = np.full(mesh.n_points, "none", dtype="<U64")
    unset_mask = np.ones(mesh.n_points, dtype=bool)

    for array_name in reversed(PROPOSAL_PRIORITY):
        if array_name not in mesh.point_data:
            continue
        values = np.asarray(mesh.point_data[array_name]).astype(str)
        active_mask = values != "none"
        labels[active_mask & unset_mask] = values[active_mask & unset_mask]
        unset_mask &= ~active_mask

    for label, color in PROPOSAL_RGBA.items():
        rgba[labels == label] = np.asarray(color, dtype=np.uint8)

    return rgba


def proposal_hybrid_rgba(mesh: pv.PolyData) -> np.ndarray:
    rgba = _empty_rgba(mesh.n_points, GREY_REST)
    labels = np.full(mesh.n_points, "none", dtype="<U64")
    unset_mask = np.ones(mesh.n_points, dtype=bool)

    for array_name in reversed(PROPOSAL_PRIORITY):
        if array_name not in mesh.point_data:
            continue
        values = np.asarray(mesh.point_data[array_name]).astype(str)
        active_mask = (values != "none") & (~np.char.endswith(values, "-other"))
        labels[active_mask & unset_mask] = values[active_mask & unset_mask]
        unset_mask &= ~active_mask

    for label, color in PROPOSAL_HYBRID_RGBA.items():
        rgba[labels == label] = np.asarray(color, dtype=np.uint8)

    forest_size = _normalize_str_array(mesh["forest_size"])
    for lifecycle_label in ["senescing", "snag", "fallen"]:
        rgba[forest_size == lifecycle_label] = np.asarray(FOREST_SIZE_RGBA[lifecycle_label], dtype=np.uint8)

    return rgba


def add_legend(plotter: pv.Plotter, entries: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    legend_entries = []
    for label, rgba in entries:
        color = "#{:02x}{:02x}{:02x}".format(rgba[0], rgba[1], rgba[2])
        legend_entries.append([label, color])
    plotter.add_legend(legend_entries, bcolor="#ffffff", face="circle", size=(0.22, 0.28))


def view_entries(view_name: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    if view_name == "classic":
        ordered = [
            ("rest", GREY_REST),
            ("small", FOREST_SIZE_RGBA["small"]),
            ("medium", FOREST_SIZE_RGBA["medium"]),
            ("large", FOREST_SIZE_RGBA["large"]),
            ("senescing", FOREST_SIZE_RGBA["senescing"]),
            ("snag", FOREST_SIZE_RGBA["snag"]),
            ("fallen", FOREST_SIZE_RGBA["fallen"]),
            ("artificial", FOREST_SIZE_RGBA["artificial"]),
        ]
        return ordered
    if view_name == "merged":
        ordered = [
            ("rest", GREY_REST),
            ("exoskeleton", BIOENVELOPE_RGBA["exoskeleton"]),
            ("brownRoof", BIOENVELOPE_RGBA["brownroof"]),
            ("otherGround", BIOENVELOPE_RGBA["otherground"]),
            ("rewilded", BIOENVELOPE_RGBA["rewilded"]),
            ("footprintDepaved", BIOENVELOPE_RGBA["footprint-depaved"]),
            ("livingFacade", BIOENVELOPE_RGBA["livingfacade"]),
            ("greenRoof", BIOENVELOPE_RGBA["greenroof"]),
            ("small", FOREST_SIZE_RGBA["small"]),
            ("medium", FOREST_SIZE_RGBA["medium"]),
            ("large", FOREST_SIZE_RGBA["large"]),
            ("senescing", FOREST_SIZE_RGBA["senescing"]),
            ("snag", FOREST_SIZE_RGBA["snag"]),
            ("fallen", FOREST_SIZE_RGBA["fallen"]),
            ("artificial", FOREST_SIZE_RGBA["artificial"]),
        ]
        return ordered
    if view_name == "proposal-hybrid":
        return [
            ("rest", GREY_REST),
            ("senescing", FOREST_SIZE_RGBA["senescing"]),
            ("snag", FOREST_SIZE_RGBA["snag"]),
            ("fallen", FOREST_SIZE_RGBA["fallen"]),
            ("decay buffer", PROPOSAL_HYBRID_RGBA["decay_buffer-feature"]),
            ("decay brace", PROPOSAL_HYBRID_RGBA["decay_brace-feature"]),
            ("recruit rewild", PROPOSAL_HYBRID_RGBA["recruit_rewild-ground"]),
            ("recruit buffer", PROPOSAL_HYBRID_RGBA["recruit_buffer-feature"]),
            ("release eliminate", PROPOSAL_HYBRID_RGBA["release-control_eliminate-pruning"]),
            ("release reduce", PROPOSAL_HYBRID_RGBA["release-control_reduce-pruning"]),
            ("colonise rewild", PROPOSAL_HYBRID_RGBA["colonise_rewild-ground"]),
            ("colonise enrich", PROPOSAL_HYBRID_RGBA["colonise_enrich-envelope"]),
            ("colonise roughen", PROPOSAL_HYBRID_RGBA["colonise_roughen-envelope"]),
            ("deploy adapt", PROPOSAL_HYBRID_RGBA["deploy-structure_adapt-utility-pole"]),
            ("deploy upgrade", PROPOSAL_HYBRID_RGBA["deploy-structure_upgrade-feature"]),
        ]
    return [
        ("rest", GREY_REST),
        ("decay other", PROPOSAL_RGBA["decay-other"]),
        ("decay buffer", PROPOSAL_RGBA["decay_buffer-feature"]),
        ("decay brace", PROPOSAL_RGBA["decay_brace-feature"]),
        ("recruit other", PROPOSAL_RGBA["recruit-other"]),
        ("recruit rewild", PROPOSAL_RGBA["recruit_rewild-ground"]),
        ("recruit buffer", PROPOSAL_RGBA["recruit_buffer-feature"]),
        ("release other", PROPOSAL_RGBA["release-control-other"]),
        ("release eliminate", PROPOSAL_RGBA["release-control_eliminate-pruning"]),
        ("release reduce", PROPOSAL_RGBA["release-control_reduce-pruning"]),
        ("colonise other", PROPOSAL_RGBA["colonise-other"]),
        ("colonise rewild", PROPOSAL_RGBA["colonise_rewild-ground"]),
        ("colonise enrich", PROPOSAL_RGBA["colonise_enrich-envelope"]),
        ("colonise roughen", PROPOSAL_RGBA["colonise_roughen-envelope"]),
        ("deploy other", PROPOSAL_RGBA["deploy-structure-other"]),
        ("deploy adapt", PROPOSAL_RGBA["deploy-structure_adapt-utility-pole"]),
        ("deploy upgrade", PROPOSAL_RGBA["deploy-structure_upgrade-feature"]),
    ]


def render_view(
    mesh: pv.PolyData,
    site: str,
    scenario: str,
    year: int,
    view_name: str,
    output_path: Path,
    point_size: float,
    window_width: int,
    window_height: int,
) -> None:
    camera = CAMERAS[site]

    if view_name == "classic":
        rgba = classic_rgba(mesh)
    elif view_name == "merged":
        rgba = merged_rgba(mesh)
    elif view_name == "proposal-hybrid":
        rgba = proposal_hybrid_rgba(mesh)
    elif view_name == "proposal":
        rgba = proposal_rgba(mesh)
    else:
        raise ValueError(f"Unknown view: {view_name}")

    plotter = pv.Plotter(off_screen=True, window_size=(window_width, window_height))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars=rgba,
        rgb=True,
        render_points_as_spheres=True,
        point_size=point_size,
    )
    plotter.enable_eye_dome_lighting()
    plotter.add_text(f"{site} | {scenario} | yr{year} | {view_name}", position="upper_left", font_size=14, color="black")
    add_legend(plotter, view_entries(view_name))
    plotter.camera_position = [
        camera["position"],
        camera["focal_point"],
        camera["view_up"],
    ]
    plotter.camera.view_angle = camera["view_angle"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(output_path))
    plotter.close()


def iter_targets(args: argparse.Namespace):
    sites = list(CAMERAS.keys()) if args.site == "all" else [args.site]
    scenarios = ["baseline", "positive", "trending"] if args.scenario == "all" else [args.scenario]
    years = args.years or [0, 10, 30, 60, 180]

    for site in sites:
        if site not in CAMERAS:
            raise KeyError(f"Unknown site camera preset: {site}")
        for scenario in scenarios:
            if scenario == "baseline":
                vtk_path = engine_output_baseline_state_vtk_path(site, args.voxel_size, args.output_mode)
                if vtk_path.exists():
                    yield site, scenario, 0, vtk_path
                else:
                    print(f"Skipping missing VTK: {vtk_path}")
                continue
            for year in years:
                vtk_path = engine_output_state_vtk_path(site, scenario, year, args.voxel_size, args.output_mode)
                if vtk_path.exists():
                    yield site, scenario, year, vtk_path
                else:
                    print(f"Skipping missing VTK: {vtk_path}")


def main() -> None:
    args = parse_args()
    render_root = engine_output_validation_dir(args.output_mode) / "renders"

    for site, scenario, year, vtk_path in iter_targets(args):
        mesh = pv.read(vtk_path)
        base_name = f"{site}_{scenario}_yr{year}"
        print(f"Rendering {base_name} from {vtk_path}")
        for view_name in ["classic", "merged", "proposal-hybrid"]:
            output_path = render_root / view_name / f"{base_name}_{view_name}.png"
            render_view(
                mesh,
                site,
                scenario,
                year,
                view_name,
                output_path,
                args.point_size,
                args.window_width,
                args.window_height,
            )
            print(output_path)


if __name__ == "__main__":
    main()
