"""Build a contract-aligned working-copy blend for the v8_t10 mist outline graph.

This creates a reusable compositor working copy:

- blend name: ``mist_complex_outlines.blend``
- graph label: ``whole_forest_outline_v8_t10``

The saved blend follows the compositor template contract as closely as Blender
allows for a saved EXR-driven graph:

- explicit workflow-local EXR hook naming
- explicit semantic mask naming for the positive-state mask

Blender only exposes the multilayer EXR pass sockets once a real EXR is bound
to the Image node, so this working copy keeps a repo-local placeholder EXR
bound to the hook node. Thin runners should repath that node at runtime.

The graph itself stays identical to the experimental v8_t10 logic:

Mist -> Kirsch -> hard threshold(0.10) -> BW -> mask -> coloured alpha
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DEFAULT_EXR = (
    REPO_ROOT
    / "_data-refactored"
    / "blenderv2"
    / "output"
    / "4.10"
    / "parade_timeline"
    / "parade_timeline__positive_state__8k.exr"
)
OUTPUT_BLEND = (
    REPO_ROOT
    / "_data-refactored"
    / "compositor"
    / "temp_blends"
    / "template_development"
    / "mist_complex_outlines.blend"
)

TREE_ID = 3
THRESHOLD = 0.10
GROUP_NAME = "whole_forest_outline_v8_t10"
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)
WORKFLOW_FRAME = "Mist Complex Outlines"
EXR_NODE_NAME = "Mist Complex Outlines :: EXR Positive"
MASK_NODE_NAME = "arboreal_positive_mask"


def log(message: str) -> None:
    print(f"[extract_mist_complex_outlines] {message}")


def clear_tree(node_tree: bpy.types.NodeTree) -> None:
    for link in list(node_tree.links):
        node_tree.links.remove(link)
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def link(node_tree: bpy.types.NodeTree, source, target) -> None:
    for existing in list(target.links):
        node_tree.links.remove(existing)
    node_tree.links.new(source, target)


def add_node(node_tree: bpy.types.NodeTree, node_type: str, name: str, location):
    node = node_tree.nodes.new(node_type)
    node.name = name
    node.label = name
    node.location = location
    return node


def detect_resolution(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["oiiotool", "--info", "-v", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+)\s*x\s*(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not read EXR dimensions from {path}")
    return int(match.group(1)), int(match.group(2))


def build_group() -> bpy.types.NodeTree:
    if GROUP_NAME in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[GROUP_NAME], do_unlink=True)

    group = bpy.data.node_groups.new(GROUP_NAME, "CompositorNodeTree")
    group.interface.new_socket(
        "positive_state_mist", in_out="INPUT", socket_type="NodeSocketColor"
    )
    group.interface.new_socket(
        "arboreal_positive_mask", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    group.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    group.interface.new_socket("Alpha", in_out="OUTPUT", socket_type="NodeSocketFloat")

    nodes = group.nodes
    links = group.links

    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-1200, 0)
    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (650, 0)

    kirsch = nodes.new("CompositorNodeFilter")
    kirsch.name = f"{GROUP_NAME}_kirsch"
    kirsch.label = kirsch.name
    kirsch.location = (-950, 0)
    kirsch.filter_type = "KIRSCH"

    ramp = nodes.new("CompositorNodeValToRGB")
    ramp.name = f"{GROUP_NAME}_threshold"
    ramp.label = ramp.name
    ramp.location = (-700, 0)
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = THRESHOLD
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)

    bw = nodes.new("CompositorNodeRGBToBW")
    bw.name = f"{GROUP_NAME}_bw"
    bw.label = bw.name
    bw.location = (-430, 0)

    masked = nodes.new("CompositorNodeMath")
    masked.name = f"{GROUP_NAME}_masked"
    masked.label = masked.name
    masked.location = (-180, 0)
    masked.operation = "MULTIPLY"
    masked.use_clamp = True

    rgb = nodes.new("CompositorNodeRGB")
    rgb.name = f"{GROUP_NAME}_rgb"
    rgb.label = rgb.name
    rgb.location = (30, -180)
    rgb.outputs[0].default_value = EDGE_COLOR_LINEAR

    set_alpha = nodes.new("CompositorNodeSetAlpha")
    set_alpha.name = GROUP_NAME
    set_alpha.label = GROUP_NAME
    set_alpha.location = (250, 0)
    set_alpha.mode = "REPLACE_ALPHA"

    links.new(group_in.outputs["positive_state_mist"], kirsch.inputs["Image"])
    links.new(kirsch.outputs["Image"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Image"], bw.inputs["Image"])
    links.new(bw.outputs["Val"], masked.inputs[0])
    links.new(group_in.outputs["arboreal_positive_mask"], masked.inputs[1])
    links.new(rgb.outputs[0], set_alpha.inputs["Image"])
    links.new(masked.outputs["Value"], set_alpha.inputs["Alpha"])
    links.new(set_alpha.outputs["Image"], group_out.inputs["Image"])
    links.new(masked.outputs["Value"], group_out.inputs["Alpha"])

    return group


def main() -> None:
    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_EXR.exists():
        raise FileNotFoundError(f"Placeholder EXR not found: {DEFAULT_EXR}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "mist_complex_outlines"
    scene.use_nodes = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    width, height = detect_resolution(DEFAULT_EXR)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    try:
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    node_tree = scene.node_tree
    clear_tree(node_tree)

    workflow_frame = add_node(node_tree, "NodeFrame", WORKFLOW_FRAME, (-1520, 280))
    workflow_frame.label = WORKFLOW_FRAME

    exr = add_node(node_tree, "CompositorNodeImage", EXR_NODE_NAME, (-1400, 200))
    exr.image = bpy.data.images.load(str(DEFAULT_EXR), check_existing=True)
    exr.parent = workflow_frame

    id_mask = add_node(node_tree, "CompositorNodeIDMask", MASK_NODE_NAME, (-1400, -120))
    id_mask.index = TREE_ID
    id_mask.use_antialiasing = True
    id_mask.parent = workflow_frame
    link(node_tree, exr.outputs["IndexOB"], id_mask.inputs["ID value"])

    group = build_group()
    outline = add_node(node_tree, "CompositorNodeGroup", GROUP_NAME, (-700, 100))
    outline.node_tree = group
    outline.parent = workflow_frame

    composite = add_node(node_tree, "CompositorNodeComposite", "Composite", (420, 120))
    viewer = add_node(node_tree, "CompositorNodeViewer", "Viewer", (420, -60))
    composite.parent = workflow_frame
    viewer.parent = workflow_frame

    link(node_tree, exr.outputs["Mist"], outline.inputs["positive_state_mist"])
    link(node_tree, id_mask.outputs["Alpha"], outline.inputs["arboreal_positive_mask"])
    link(node_tree, outline.outputs["Image"], composite.inputs["Image"])
    link(node_tree, outline.outputs["Image"], viewer.inputs["Image"])

    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    log(f"Saved {OUTPUT_BLEND}")


if __name__ == "__main__":
    main()
