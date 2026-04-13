"""Thin runner for compositor_intervention_int.blend.

Opens the canonical template, repaths a single bioenvelope EXR input,
wires ``intervention_bioenvelope_ply-int`` to the normalize node,
and renders one RGBA PNG.

This is a runtime runner, not a template-edit script. It does not modify
the canonical graph.

Environment variables (all overridable):

    COMPOSITOR_BLEND_PATH       path to compositor_intervention_int.blend
    COMPOSITOR_OUTPUT_DIR       output directory for PNG
    COMPOSITOR_EXR              bioenvelope EXR to render
    COMPOSITOR_SCENE_NAME       scene name (default: Current)

Usage:

    export COMPOSITOR_EXR="path/to/bioenvelope_positive__8k.exr"
    export COMPOSITOR_OUTPUT_DIR="path/to/output"
    blender --background --python render_edge_lab_current_intervention_int.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy

# Derive repo root from __file__ so defaults work on any machine.
_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _exr_header import read_exr_dimensions  # noqa: E402

COMPOSITOR_ROOT = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"
OUTPUT_BASE = REPO_ROOT / "_data-refactored" / "compositor" / "outputs"


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


BLEND_PATH = env_path(
    "COMPOSITOR_BLEND_PATH",
    str(CANONICAL_ROOT / "compositor_intervention_int.blend"),
)
OUTPUT_DIR = env_path(
    "COMPOSITOR_OUTPUT_DIR",
    str(OUTPUT_BASE / "intervention_int" / "current"),
)
EXR_PATH = env_path("COMPOSITOR_EXR", "")
SCENE_NAME = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")

# Node names must match the rebuild script.
EXR_NODE_NAME = "InterventionInt::EXR Input"
RAW_HUB_NODE_NAME = "InterventionInt::Raw"
FILE_OUTPUT_NAME = "InterventionInt::Outputs"

# The AOV socket name as it appears on the EXR Image node.
# Blender truncates long socket names, so the full
# "intervention_bioenvelope_ply-int" may appear as
# "intervention_bioenvelope_ply-".  Match on a stable prefix.
AOV_SOCKET_PREFIX = "intervention_bioenvelope_ply"


def log(message: str) -> None:
    print(f"[render_intervention_int] {message}")


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


def require_node(node_tree: bpy.types.NodeTree, name: str) -> bpy.types.Node:
    node = node_tree.nodes.get(name)
    if node is None:
        raise ValueError(f"Missing node: {name}")
    return node


def require_node_by_type(node_tree: bpy.types.NodeTree, bl_idname: str) -> bpy.types.Node:
    for node in node_tree.nodes:
        if node.bl_idname == bl_idname:
            return node
    raise ValueError(f"Missing node of type: {bl_idname}")


def detect_resolution(exr_path: Path) -> tuple[int, int]:
    """Read the true displayWindow directly from the EXR header.

    Per the Input Resolution Rule in COMPOSITOR_TEMPLATE_CONTRACT.md, this
    must NOT fall back to a hardcoded resolution. Raise on failure.
    """
    if not exr_path.exists():
        raise FileNotFoundError(f"EXR not found: {exr_path}")
    w, h = read_exr_dimensions(str(exr_path))
    return w, h


def repath_and_wire(
    node_tree: bpy.types.NodeTree,
    exr_path: Path,
) -> bpy.types.Image:
    """Load an EXR into the Image node and wire the AOV socket to the raw hub."""
    exr_node = require_node(node_tree, EXR_NODE_NAME)
    raw_hub = require_node(node_tree, RAW_HUB_NODE_NAME)

    # Load the image.
    img = bpy.data.images.load(str(exr_path), check_existing=True)
    exr_node.image = img
    log(f"loaded {exr_path.name}")

    # Find the AOV socket (Blender may truncate the name).
    aov_socket = None
    for sock in exr_node.outputs:
        if sock.name.startswith(AOV_SOCKET_PREFIX):
            aov_socket = sock
            break
    if aov_socket is None:
        available = [s.name for s in exr_node.outputs]
        raise ValueError(
            f"EXR has no socket starting with '{AOV_SOCKET_PREFIX}'. "
            f"Available: {available}"
        )

    # Clear existing links to raw hub, then wire.
    for link in list(raw_hub.inputs[0].links):
        node_tree.links.remove(link)
    node_tree.links.new(aov_socket, raw_hub.inputs[0])
    log(f"wired {aov_socket.name} -> {RAW_HUB_NODE_NAME}")
    return img


def render_socket_to_png(
    scene: bpy.types.Scene,
    node_tree: bpy.types.NodeTree,
    image_socket,
    output_path: Path,
) -> None:
    """Render a single compositor socket to a PNG via the Composite node.

    Blender 4.2 intermittently skips File Output slots when no Render Layers
    node is present.  This renders each slot individually through the Composite
    node as the established workaround (same pattern as the mist runner).
    """
    composite = require_node_by_type(node_tree, "CompositorNodeComposite")
    viewer = require_node_by_type(node_tree, "CompositorNodeViewer")
    for link in list(composite.inputs[0].links):
        node_tree.links.remove(link)
    for link in list(viewer.inputs[0].links):
        node_tree.links.remove(link)
    node_tree.links.new(image_socket, composite.inputs[0])
    node_tree.links.new(image_socket, viewer.inputs[0])
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    log(f"Wrote {output_path}")


def rename_family_outputs(output_dir: Path) -> None:
    for rendered_path in sorted(output_dir.glob("*_0001.png")):
        final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(f"Canonical blend not found: {BLEND_PATH}")

    exr_path = Path(str(EXR_PATH))
    if not exr_path.exists():
        raise FileNotFoundError(
            f"EXR not found: {exr_path}. Set COMPOSITOR_EXR to a bioenvelope EXR."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")

    node_tree = scene.node_tree
    set_standard_view(scene)
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    # Repath and wire the single EXR input.
    repath_and_wire(node_tree, exr_path)
    width, height = detect_resolution(exr_path)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    log(f"resolution {width}x{height} from {exr_path.name}")

    # File Output node — only set the base path and unmute.
    # Format and color management are owned by the canonical template.
    workflow_output = require_node(node_tree, FILE_OUTPUT_NAME)
    workflow_output.base_path = str(OUTPUT_DIR)
    workflow_output.mute = False

    # Mute all other file output nodes.
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile" and node.name != FILE_OUTPUT_NAME:
            node.mute = True

    scene.render.filepath = str(OUTPUT_DIR / "_discard_render.png")
    bpy.ops.render.render(write_still=True, scene=scene.name)
    rename_family_outputs(OUTPUT_DIR)

    # Blender 4.2 intermittently skips File Output slots when no Render
    # Layers node is present.  Re-render missing slots individually through
    # the Composite node (same workaround as the mist runner).
    workflow_output.mute = True
    for index, slot in enumerate(workflow_output.file_slots):
        stem = slot.path.rstrip("_")
        out_path = OUTPUT_DIR / f"{stem}.png"
        if not out_path.exists() and workflow_output.inputs[index].links:
            render_socket_to_png(
                scene,
                node_tree,
                workflow_output.inputs[index].links[0].from_socket,
                out_path,
            )
    rename_family_outputs(OUTPUT_DIR)

    # Clean up discard.
    discard = OUTPUT_DIR / "_discard_render.png"
    if discard.exists():
        discard.unlink()

    # Report what was written.
    for slot in workflow_output.file_slots:
        stem = slot.path.rstrip("_")
        out_path = OUTPUT_DIR / f"{stem}.png"
        if out_path.exists():
            log(f"Wrote {out_path}")
        else:
            log(f"MISSING {out_path}")
    log("done")


if __name__ == "__main__":
    main()
