"""Thin runner: compositor_base.blend on parade single-state yr180 EXRs (latest timestamp).

Opens the canonical base blend, repaths the single EXR input, renders each
labeled reroute individually through the Composite node, exits without saving.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _exr_header import read_exr_dimensions

CANONICAL_BLEND = (
    REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
    / "canonical_templates" / "compositor_base.blend"
)
EXR_DIR = (
    REPO_ROOT / "_data-refactored" / "blenderv2" / "output"
    / "20260412_232449_parade_single-state_yr180_8k"
)
EXISTING_EXR = EXR_DIR / "parade_single-state_yr180__existing_condition_positive__8k.exr"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    REPO_ROOT / "_data-refactored" / "compositor" / "outputs"
    / "4.10" / "parade_single-state_yr180" / f"base__{stamp}"
)

# The base blend uses labeled reroutes for outputs
REROUTE_LABELS = [
    "base_rgb",
    "base_outlines",
    "base_sim-turns",
    "base_sim-nodes",
    "base_sim-turns_ripple-effect",
    "base_depth_windowed_balanced_refined",
    "base_depth_windowed_internal_refined",
    "base_depth_windowed_internal_dense",
    "base_depth_windowed_balanced_dense",
]


def log(msg: str) -> None:
    print(f"[run_base_parade] {msg}")


def require_node(tree, name):
    n = tree.nodes.get(name)
    if n is None:
        raise ValueError(f"Missing node: {name}")
    return n


def require_node_by_type(tree, bl_idname):
    for n in tree.nodes:
        if n.bl_idname == bl_idname:
            return n
    raise ValueError(f"Missing node type: {bl_idname}")


def require_node_by_label(tree, label, bl_idname=None):
    for n in tree.nodes:
        if n.label != label:
            continue
        if bl_idname is not None and n.bl_idname != bl_idname:
            continue
        return n
    raise ValueError(f"Missing node with label: {label}")


def set_standard_view(scene):
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


def exr_image_socket(node):
    image = node.outputs.get("Image")
    if image is not None and getattr(image, "enabled", True):
        return image
    combined = node.outputs.get("Combined")
    if combined is not None and getattr(combined, "enabled", True):
        return combined
    if image is not None:
        return image
    raise ValueError(f"Node '{node.name}' has no enabled Image or Combined output")


def render_socket_to_png(scene, tree, socket, out_path):
    composite = require_node_by_type(tree, "CompositorNodeComposite")
    for link in list(composite.inputs[0].links):
        tree.links.remove(link)
    tree.links.new(socket, composite.inputs[0])
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    log(f"Wrote {out_path}")


def main():
    if not EXISTING_EXR.exists():
        raise FileNotFoundError(EXISTING_EXR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(CANONICAL_BLEND))
    scene = bpy.data.scenes.get("Current")
    if scene is None or scene.node_tree is None:
        raise ValueError("Scene 'Current' not found")

    tree = scene.node_tree
    set_standard_view(scene)

    # Repath the single EXR input
    existing = require_node(tree, "Current Base Outputs :: EXR Existing")
    img = bpy.data.images.load(str(EXISTING_EXR), check_existing=True)
    existing.image = img
    log(f"Loaded {EXISTING_EXR.name}")

    # Reconnect group input
    base_group = require_node(tree, "Current Base Outputs :: Group")
    group_input = base_group.inputs.get("Existing Image")
    if group_input is not None:
        for link in list(group_input.links):
            tree.links.remove(link)
        tree.links.new(exr_image_socket(existing), group_input)
        log("Reconnected Existing Image -> group")

    # Detect resolution
    w, h = read_exr_dimensions(str(EXISTING_EXR))
    log(f"Resolution {w}x{h}")
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    # Mute all File Output nodes (if any)
    for n in tree.nodes:
        if n.bl_idname == "CompositorNodeOutputFile":
            n.mute = True

    # Render each labeled reroute individually
    for label in REROUTE_LABELS:
        reroute = require_node_by_label(tree, label, "NodeReroute")
        render_socket_to_png(scene, tree, reroute.outputs[0], OUTPUT_DIR / f"{label}.png")

    log("done")


if __name__ == "__main__":
    main()
