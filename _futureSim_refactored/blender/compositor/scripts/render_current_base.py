"""Fast runner for compositor_base.blend.

Special case: compositor_base.blend has NO File Output node — it exposes
10 labeled reroutes instead. Each reroute is rendered through the Composite
node individually. write_still is fine here because Composite always writes
`scene.render.filepath` when rendering a single frame; the animation=True
rule from COMPOSITOR_TEMPLATE_CONTRACT.md applies only to File Output slots.

Single-input runner: existing_condition_positive EXR.

Environment variables:
    COMPOSITOR_EXISTING_EXR   existing_condition_positive EXR (required)
    COMPOSITOR_OUTPUT_DIR     output directory (required)
    COMPOSITOR_BLEND_PATH     optional override
    COMPOSITOR_SCENE_NAME     optional override (default: Current)
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
from _fast_runner_core import CANONICAL_ROOT  # noqa: E402

NAME = "render_current_base"
BLEND_DEFAULT = CANONICAL_ROOT / "compositor_base.blend"
SCENE_DEFAULT = "Current"
EXR_NODE = "Current Base Outputs :: EXR Existing"
GROUP_NODE = "Current Base Outputs :: Group"
GROUP_INPUT_NAME = "Existing Image"

REROUTE_LABELS = [
    "base_rgb",
    "base_white_render",
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
    print(f"[{NAME}] {msg}")


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
    raise ValueError(f"Node {node.name!r} has no enabled Image or Combined output")


def render_socket_to_png(scene, tree, socket, out_path):
    composite = require_node_by_type(tree, "CompositorNodeComposite")
    for link in list(composite.inputs[0].links):
        tree.links.remove(link)
    tree.links.new(socket, composite.inputs[0])
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    log(f"wrote {out_path.name}")


def main() -> None:
    existing = os.environ.get("COMPOSITOR_EXISTING_EXR", "")
    out = os.environ.get("COMPOSITOR_OUTPUT_DIR", "")
    if not (existing and out):
        raise ValueError("Required env vars: COMPOSITOR_EXISTING_EXR, COMPOSITOR_OUTPUT_DIR")
    blend = Path(os.environ.get("COMPOSITOR_BLEND_PATH", str(BLEND_DEFAULT)))
    scene_name = os.environ.get("COMPOSITOR_SCENE_NAME", SCENE_DEFAULT)

    existing_path = Path(existing)
    output_dir = Path(out)
    if not blend.exists():
        raise FileNotFoundError(blend)
    if not existing_path.exists():
        raise FileNotFoundError(existing_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.data.scenes.get(scene_name)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene {scene_name!r} not found in {blend}")
    tree = scene.node_tree
    set_standard_view(scene)

    exr_node = require_node(tree, EXR_NODE)
    if exr_node.image is None:
        exr_node.image = bpy.data.images.load(str(existing_path), check_existing=True)
    else:
        exr_node.image.filepath = str(existing_path)
        exr_node.image.reload()
    log(f"loaded {existing_path.name} -> {EXR_NODE!r}")

    group = require_node(tree, GROUP_NODE)
    group_input = group.inputs.get(GROUP_INPUT_NAME)
    if group_input is not None:
        for link in list(group_input.links):
            tree.links.remove(link)
        tree.links.new(exr_image_socket(exr_node), group_input)

    w, h = read_exr_dimensions(str(existing_path))
    log(f"resolution {w}x{h}")
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

    for n in tree.nodes:
        if n.bl_idname == "CompositorNodeOutputFile":
            n.mute = True

    missing: list[str] = []
    for label in REROUTE_LABELS:
        reroute = require_node_by_label(tree, label, "NodeReroute")
        out_path = output_dir / f"{label}.png"
        render_socket_to_png(scene, tree, reroute.outputs[0], out_path)
        if not out_path.exists():
            missing.append(label)

    if missing:
        raise RuntimeError(f"{NAME}: reroutes missing on disk: {missing}")
    log(f"{len(REROUTE_LABELS)}/{len(REROUTE_LABELS)} reroutes present")


if __name__ == "__main__":
    main()
