"""Ad-hoc: emit a size-colored silhouette PNG for a per-tree library EXR.

Since the library EXRs have no `size` AOV, the size bucket comes from the
filename — the asset token embeds `size.<label>`. The instancer's size->int
mapping lives in bV2_build_instancers.py convert_size():
    small=1, medium=2, large=3, senescing=4, snag=5, fallen=6, decayed=7

This script:
  1. Loads the library EXR
  2. Builds a tiny compositor graph: RGB(size_color) + SetAlpha(from EXR Alpha)
  3. Writes one PNG where RGB = size color, A = tree silhouette

Env vars:
    COMPOSITOR_LIBRARY_EXR  — path to one library EXR
    COMPOSITOR_OUTPUT_DIR   — short dir (Blender MAX_PATH workaround)
    COMPOSITOR_SIZE_LABEL   — one of small|medium|large|senescing|snag|fallen|decayed|artificial

Writes:
    <output_dir>/size_colored.png

The palette matches compositor_sizes.blend (see SIZE_PALETTE docstring below).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _exr_header import read_exr_dimensions  # noqa: E402


# Canonical linear-RGB palette lifted directly from compositor_sizes.blend.
# Every `Sizes::*_<size> :: grouped` node group carries a 'Flat Colour' input
# with the same value across all four workflow branches (existing / pathway /
# priority / trending). These are the authoritative per-size colors.
#
# Source: _futureSim_refactored/blender/compositor/canonical_templates/compositor_sizes.blend
#   inspected via _inspect_sizes_palette.py on 2026-04-11.
#
# Current lifecycle mapping:
#   small=1 medium=2 large=3 senescing=4 snag=5 fallen=6 decayed=7 artificial=-1
# The compositor palette below follows that contract directly.
SIZE_PALETTE: dict[str, tuple[float, float, float]] = {
    "small":      (0.401978, 0.708376, 0.111932),
    "medium":     (0.258183, 0.610496, 0.283149),
    "large":      (0.947307, 0.346704, 0.181164),
    "senescing":  (0.830770, 0.327778, 0.558340),
    "snag":       (0.973445, 0.768151, 0.097587),
    "fallen":     (0.274677, 0.250158, 0.520996),
    "decayed":    (0.114435, 0.238398, 0.208637),
    "artificial": (1.000000, 0.000000, 0.000000),
}


def log(message: str) -> None:
    print(f"[render_library_size_colored] {message}")


def env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing env: {name}")
    return value


def find_socket(node: bpy.types.Node, name: str):
    for s in node.outputs:
        if s.name == name:
            return s
    return None


def set_standard_view(scene: bpy.types.Scene) -> None:
    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def rename_outputs(output_dir: Path) -> None:
    for rendered_path in sorted(output_dir.glob("*_0001.png")):
        final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")


def main() -> None:
    exr_path = Path(env_required("COMPOSITOR_LIBRARY_EXR"))
    output_dir = Path(env_required("COMPOSITOR_OUTPUT_DIR"))
    size_label = env_required("COMPOSITOR_SIZE_LABEL").strip().lower()

    if not exr_path.exists():
        raise FileNotFoundError(exr_path)
    if size_label not in SIZE_PALETTE:
        raise ValueError(
            f"unknown size label {size_label!r}; allowed: {sorted(SIZE_PALETTE.keys())}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    r, g, b = SIZE_PALETTE[size_label]
    log(f"EXR: {exr_path}")
    log(f"Output dir: {output_dir}")
    log(f"Size label: {size_label}  RGB(linear)=({r}, {g}, {b})")

    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)

    img = bpy.data.images.load(str(exr_path))
    img_node = tree.nodes.new("CompositorNodeImage")
    img_node.image = img
    img_node.location = (0, 0)

    alpha_sock = find_socket(img_node, "Alpha")
    if alpha_sock is None:
        raise RuntimeError(f"EXR has no 'Alpha' socket: {exr_path}")

    rgb = tree.nodes.new("CompositorNodeRGB")
    rgb.outputs[0].default_value = (r, g, b, 1.0)
    rgb.location = (300, 100)

    set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
    set_alpha.mode = "REPLACE_ALPHA"
    set_alpha.location = (550, 50)
    tree.links.new(rgb.outputs[0], set_alpha.inputs["Image"])
    tree.links.new(alpha_sock, set_alpha.inputs["Alpha"])

    comp = tree.nodes.new("CompositorNodeComposite")
    comp.location = (850, 200)
    viewer = tree.nodes.new("CompositorNodeViewer")
    viewer.location = (850, 50)
    tree.links.new(set_alpha.outputs["Image"], comp.inputs[0])
    tree.links.new(set_alpha.outputs["Image"], viewer.inputs[0])

    out = tree.nodes.new("CompositorNodeOutputFile")
    out.location = (850, -200)
    out.base_path = str(output_dir)
    out.format.file_format = "PNG"
    out.format.color_mode = "RGBA"
    out.format.color_depth = "8"
    while len(out.file_slots) > 0:
        out.file_slots.remove(out.inputs[0])
    out.file_slots.new("size_colored_")
    tree.links.new(set_alpha.outputs["Image"], out.inputs[0])

    width, height = read_exr_dimensions(str(exr_path))
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)

    scene.render.filepath = str(output_dir / "_discard_render.png")
    bpy.ops.render.render(write_still=True, scene=scene.name)
    rename_outputs(output_dir)

    discard = output_dir / "_discard_render.png"
    if discard.exists():
        discard.unlink()
    log(f"DONE ({size_label})")


if __name__ == "__main__":
    main()
