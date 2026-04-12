"""Render per-intervention proposal PNGs from a single EXR.

Builds the compositor graph inline and renders each intervention via the
Composite node.  Blender 4.2 headless does NOT fire File Output nodes
reliably, so we render 13 passes within a single session — the EXR is
loaded once and each pass just writes one PNG (~3 s each at 8K).

The canonical blend (from _build_proposal_and_interventions.py) exists for
visual inspection; this script is the render-time equivalent.

Usage::

    blender --background --factory-startup --python \
        _futureSim_refactored/blender/compositor/scripts/render_edge_lab_current_proposal_and_interventions.py

Env vars::

    COMPOSITOR_PROPOSAL_INTERVENTIONS_EXR   — path to the input EXR
    COMPOSITOR_PROPOSAL_INTERVENTIONS_DIR   — output directory (auto-derived if omitted)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
REPO_ROOT = next(
    p.parent for p in _THIS.parents if p.name == "_futureSim_refactored"
)
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _exr_header import read_exr_dimensions  # noqa: E402

OUTPUT_BASE = REPO_ROOT / "_data-refactored" / "compositor" / "outputs"
DEFAULT_EXR = (
    REPO_ROOT / "_data-refactored" / "blenderv2" / "output"
    / "4.10" / "city_timeline" / "city_timeline__positive_state__8k64s.exr"
)

# Same spec as _build_proposal_and_interventions.py — source of truth is
# proposal_framebuffers.FRAMEBUFFER_STATE_MAPPINGS + constants.py
INTERVENTION_SPECS: list[tuple[str, list[tuple[int, str, str]]]] = [
    ("proposal-decay", [
        (2, "buffer-feature", "#B83B6B"),
        (3, "brace-feature", "#D9638C"),
    ]),
    ("proposal-release-control", [
        (1, "rejected", "#333333"),
        (2, "reduce-canopy-pruning", "#808080"),
        (3, "eliminate-canopy-pruning", "#FFFFFF"),
    ]),
    ("proposal-recruit", [
        (2, "rewild-smaller-patch", "#C5E28E"),
        (3, "rewild-larger-patch", "#5CB85C"),
    ]),
    ("proposal-colonise", [
        (2, "larger-patches-rewild", "#5CB85C"),
        (3, "enrich-envelope", "#8CCC4F"),
        (4, "roughen-envelope", "#B87A38"),
    ]),
    ("proposal-deploy-structure", [
        (2, "adapt-utility-pole", "#FF0000"),
        (3, "translocate-log", "#8F89BF"),
        (4, "upgrade-feature", "#CE6DD9"),
    ]),
]


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


EXR_PATH = env_path("COMPOSITOR_PROPOSAL_INTERVENTIONS_EXR", str(DEFAULT_EXR))


def derive_output_dir(exr_path: Path) -> Path:
    exr_family = exr_path.parent.name
    sim_root = exr_path.parent.parent.name
    parts = exr_path.stem.split("__")
    view_layer = parts[1] if len(parts) >= 3 else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_BASE / sim_root / exr_family / f"proposal_and_interventions__{timestamp}__{view_layer}"


OUTPUT_DIR = (
    env_path("COMPOSITOR_PROPOSAL_INTERVENTIONS_DIR", "")
    if os.environ.get("COMPOSITOR_PROPOSAL_INTERVENTIONS_DIR")
    else derive_output_dir(EXR_PATH)
)


def log(msg: str) -> None:
    print(f"[render_proposal_and_interventions] {msg}")


def srgb_to_linear(c: int) -> float:
    n = c / 255.0
    return n / 12.92 if n <= 0.04045 else ((n + 0.055) / 1.055) ** 2.4


def hex_to_rgba(h: str) -> tuple[float, float, float, float]:
    h = h.lstrip("#")
    return (
        srgb_to_linear(int(h[0:2], 16)),
        srgb_to_linear(int(h[2:4], 16)),
        srgb_to_linear(int(h[4:6], 16)),
        1.0,
    )


def ensure_link(tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        tree.links.remove(link)
    tree.links.new(from_socket, to_socket)


def main() -> None:
    if not EXR_PATH.exists():
        raise FileNotFoundError(f"EXR not found: {EXR_PATH}")

    width, height = read_exr_dimensions(str(EXR_PATH))
    log(f"exr: {EXR_PATH.name}  resolution: {width}x{height}")

    # Fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "Current"
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    # Scene config
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    try:
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # EXR input
    exr = tree.nodes.new("CompositorNodeImage")
    exr.name = "EXR"
    exr.label = "EXR"
    exr.location = (0, 0)
    exr.image = bpy.data.images.load(str(EXR_PATH), check_existing=True)

    # Composite + Viewer sinks (reused per intervention)
    composite = tree.nodes.new("CompositorNodeComposite")
    composite.location = (1600, 400)
    viewer = tree.nodes.new("CompositorNodeViewer")
    viewer.location = (1600, 540)

    # Build per-intervention chains
    wired: list[tuple[str, bpy.types.NodeSocket]] = []
    y = 200.0

    for family_aov, interventions in INTERVENTION_SPECS:
        family_short = family_aov.replace("proposal-", "")
        for val, slug, hex_color in interventions:
            # Value match: subtract → abs → less-than 0.5
            sub = tree.nodes.new("CompositorNodeMath")
            sub.name = f"{family_aov}::{slug}::sub"
            sub.operation = "SUBTRACT"
            sub.inputs[1].default_value = float(val)
            sub.location = (400, y)

            abso = tree.nodes.new("CompositorNodeMath")
            abso.name = f"{family_aov}::{slug}::abs"
            abso.operation = "ABSOLUTE"
            abso.location = (600, y)

            lt = tree.nodes.new("CompositorNodeMath")
            lt.name = f"{family_aov}::{slug}::lt"
            lt.operation = "LESS_THAN"
            lt.inputs[1].default_value = 0.5
            lt.location = (800, y)

            ensure_link(tree, exr.outputs[family_aov], sub.inputs[0])
            ensure_link(tree, sub.outputs["Value"], abso.inputs[0])
            ensure_link(tree, abso.outputs["Value"], lt.inputs[0])

            # RGB + SetAlpha
            rgb = tree.nodes.new("CompositorNodeRGB")
            rgb.name = f"{family_aov}::{slug}::rgb"
            rgb.label = slug
            rgb.outputs[0].default_value = hex_to_rgba(hex_color)
            rgb.location = (1000, y + 60)

            sa = tree.nodes.new("CompositorNodeSetAlpha")
            sa.name = f"{family_aov}::{slug}::rgba"
            sa.mode = "APPLY"
            sa.location = (1200, y)

            ensure_link(tree, rgb.outputs[0], sa.inputs["Image"])
            ensure_link(tree, lt.outputs["Value"], sa.inputs["Alpha"])

            wired.append((f"proposal-{family_short}-{slug}", sa.outputs["Image"]))
            y -= 160.0
        y -= 200.0

    # Render each intervention by wiring it to Composite and writing one PNG.
    # Blender 4.2 headless File Output nodes never fire, so this per-socket
    # approach is the only reliable method.  The EXR is already loaded and the
    # compositor caches intermediate results, so each pass is fast (~3 s at 8K).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (slug, socket) in enumerate(wired):
        ensure_link(tree, socket, composite.inputs[0])
        ensure_link(tree, socket, viewer.inputs[0])
        out_path = OUTPUT_DIR / f"{slug}.png"
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True, scene=scene.name)
        log(f"[{i+1}/{len(wired)}] {slug}.png")

    log(f"done -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
