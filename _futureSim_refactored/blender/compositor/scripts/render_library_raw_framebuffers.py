"""Ad-hoc: extract raw framebuffers (AO + resource_* masks) from per-tree
library EXRs, without the canonical state-aware compositor blends.

Per-tree library EXRs are single-object tree renders — no ground, no buildings
— so the IndexOB masking that compositor_ao / compositor_resources /
compositor_sizes use to isolate trees is redundant. The AO and resource_*_mask
passes can be saved straight from the EXR.

Env vars (all required except COMPOSITOR_TARGET_SOCKETS):
    COMPOSITOR_LIBRARY_EXR   — path to one library EXR
    COMPOSITOR_OUTPUT_DIR    — short dir (avoids Windows MAX_PATH in Blender)
    COMPOSITOR_SCENE_NAME    — ignored; this runner builds its own scene
    COMPOSITOR_TARGET_SOCKETS — optional comma list; default is:
        AO,resource_colour,resource_dead_branch_mask,resource_epiphyte_mask,
        resource_fallen_log_mask,resource_hollow_mask,resource_none_mask,
        resource_peeling_bark_mask,resource_perch_branch_mask

Usage:
    blender -b --factory-startup -P render_library_raw_framebuffers.py
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


DEFAULT_SOCKETS = (
    "AO",
    "resource_colour",
    "resource_dead_branch_mask",
    "resource_epiphyte_mask",
    "resource_fallen_log_mask",
    "resource_hollow_mask",
    "resource_none_mask",
    "resource_peeling_bark_mask",
    "resource_perch_branch_mask",
)


def log(message: str) -> None:
    print(f"[render_library_raw_framebuffers] {message}")


def env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing env: {name}")
    return value


def target_sockets() -> tuple[str, ...]:
    raw = os.environ.get("COMPOSITOR_TARGET_SOCKETS", "")
    if not raw.strip():
        return DEFAULT_SOCKETS
    return tuple(s.strip() for s in raw.split(",") if s.strip())


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


def find_socket(node: bpy.types.Node, name: str):
    for s in node.outputs:
        if s.name == name:
            return s
    return None


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
    if not exr_path.exists():
        raise FileNotFoundError(exr_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    sockets_wanted = target_sockets()
    log(f"EXR: {exr_path}")
    log(f"Output dir: {output_dir}")
    log(f"Sockets: {', '.join(sockets_wanted)}")

    # Start from a clean blend (factory startup already did)
    # Clear existing scenes except the default, then rebuild from scratch
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    # Clear any pre-existing nodes from the default startup
    for n in list(tree.nodes):
        tree.nodes.remove(n)

    # Load the EXR
    img = bpy.data.images.load(str(exr_path))
    img_node = tree.nodes.new("CompositorNodeImage")
    img_node.image = img
    img_node.location = (0, 0)

    # Composite + Viewer are required so the render pipeline has a sink
    comp = tree.nodes.new("CompositorNodeComposite")
    comp.location = (600, 300)
    viewer = tree.nodes.new("CompositorNodeViewer")
    viewer.location = (600, 150)

    # Always route something into Composite to keep Blender happy
    combined = find_socket(img_node, "Combined")
    if combined is not None:
        tree.links.new(combined, comp.inputs[0])
        tree.links.new(combined, viewer.inputs[0])

    # Output file node — one slot per target socket
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.location = (600, -200)
    out.base_path = str(output_dir)
    out.format.file_format = "PNG"
    out.format.color_mode = "RGBA"
    out.format.color_depth = "8"
    # Remove default slot and add named ones
    while len(out.file_slots) > 0:
        out.file_slots.remove(out.inputs[0])

    available = {s.name for s in img_node.outputs}
    wired = []
    skipped = []
    for name in sockets_wanted:
        if name not in available:
            skipped.append(name)
            continue
        slot = out.file_slots.new(f"{name}_")
        tree.links.new(
            find_socket(img_node, name),
            out.inputs[len(out.file_slots) - 1],
        )
        wired.append(name)

    if skipped:
        log(f"SKIPPED (not in EXR): {skipped}")
    if not wired:
        raise RuntimeError(f"no target sockets were wired; wanted={sockets_wanted}")

    # Resolution from EXR header
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

    log(f"DONE: wired={wired}")


if __name__ == "__main__":
    main()
