"""Build the generic canonical mist-complex-outlines compositor blend.

This canonical is intentionally single-input:

- one EXR input hook
- one generic arboreal mask
- one output slot for ``whole_forest_outline_v8_t10``

Runners should repath the EXR input and derive the final PNG filename from the
input EXR name at runtime.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
CANONICAL_BLEND = (
    REPO_ROOT
    / "_futureSim_refactored"
    / "blender"
    / "compositor"
    / "canonical_templates"
    / "compositor_mist_complex_outlines.blend"
)
DEFAULT_EXR = (
    REPO_ROOT
    / "_data-refactored"
    / "blenderv2"
    / "output"
    / "4.10"
    / "parade_timeline"
    / "parade_timeline__positive_state__8k.exr"
)

TREE_ID = 3
THRESHOLD = 0.10
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)
GROUP_NAME = "MistComplexOutlines::whole_forest_outline_v8_t10_group"


def log(message: str) -> None:
    print(f"[build_compositor_mist_complex_outlines] {message}")


def clear_tree(node_tree: bpy.types.NodeTree) -> None:
    for link in list(node_tree.links):
        node_tree.links.remove(link)
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


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
    existing = bpy.data.node_groups.get(GROUP_NAME)
    if existing is not None:
        bpy.data.node_groups.remove(existing, do_unlink=True)

    group = bpy.data.node_groups.new(GROUP_NAME, "CompositorNodeTree")
    group.interface.new_socket(
        "mist", in_out="INPUT", socket_type="NodeSocketColor"
    )
    group.interface.new_socket(
        "arboreal_mask", in_out="INPUT", socket_type="NodeSocketFloat"
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
    kirsch.name = "MistComplexOutlines::kirsch"
    kirsch.label = kirsch.name
    kirsch.location = (-950, 0)
    kirsch.filter_type = "KIRSCH"

    ramp = nodes.new("CompositorNodeValToRGB")
    ramp.name = "MistComplexOutlines::threshold"
    ramp.label = ramp.name
    ramp.location = (-700, 0)
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = THRESHOLD
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)

    bw = nodes.new("CompositorNodeRGBToBW")
    bw.name = "MistComplexOutlines::bw"
    bw.label = bw.name
    bw.location = (-430, 0)

    masked = nodes.new("CompositorNodeMath")
    masked.name = "MistComplexOutlines::masked"
    masked.label = masked.name
    masked.location = (-180, 0)
    masked.operation = "MULTIPLY"
    masked.use_clamp = True

    rgb = nodes.new("CompositorNodeRGB")
    rgb.name = "MistComplexOutlines::rgb"
    rgb.label = rgb.name
    rgb.location = (30, -180)
    rgb.outputs[0].default_value = EDGE_COLOR_LINEAR

    set_alpha = nodes.new("CompositorNodeSetAlpha")
    set_alpha.name = "MistComplexOutlines::whole_forest_outline_v8_t10"
    set_alpha.label = set_alpha.name
    set_alpha.location = (250, 0)
    set_alpha.mode = "REPLACE_ALPHA"

    links.new(group_in.outputs["mist"], kirsch.inputs["Image"])
    links.new(kirsch.outputs["Image"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Image"], bw.inputs["Image"])
    links.new(bw.outputs["Val"], masked.inputs[0])
    links.new(group_in.outputs["arboreal_mask"], masked.inputs[1])
    links.new(rgb.outputs[0], set_alpha.inputs["Image"])
    links.new(masked.outputs["Value"], set_alpha.inputs["Alpha"])
    links.new(set_alpha.outputs["Image"], group_out.inputs["Image"])
    links.new(masked.outputs["Value"], group_out.inputs["Alpha"])

    return group


def main() -> None:
    if not DEFAULT_EXR.exists():
        raise FileNotFoundError(f"Placeholder EXR not found: {DEFAULT_EXR}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "Current"
    scene.use_nodes = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
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

    family_frame = add_node(
        node_tree, "NodeFrame", "MistComplexOutlines::FamilyFrame", (-1510, 250)
    )

    exr = add_node(
        node_tree, "CompositorNodeImage", "MistComplexOutlines::EXR Input", (-1380, 180)
    )
    exr.parent = family_frame
    exr.image = bpy.data.images.load(str(DEFAULT_EXR), check_existing=True)

    mask_source = add_node(
        node_tree,
        "CompositorNodeIDMask",
        "MistComplexOutlines::arboreal_mask_source",
        (-1380, -120),
    )
    mask_source.parent = family_frame
    mask_source.index = TREE_ID
    mask_source.use_antialiasing = True
    node_tree.links.new(exr.outputs["IndexOB"], mask_source.inputs["ID value"])

    mask_reroute = add_node(
        node_tree,
        "NodeReroute",
        "MistComplexOutlines::arboreal_mask",
        (-1120, -40),
    )
    mask_reroute.parent = family_frame
    node_tree.links.new(mask_source.outputs["Alpha"], mask_reroute.inputs[0])

    group = build_group()
    workflow = add_node(
        node_tree, "CompositorNodeGroup", "MistComplexOutlines::viewlayer_group", (-720, 90)
    )
    workflow.parent = family_frame
    workflow.node_tree = group
    node_tree.links.new(exr.outputs["Mist"], workflow.inputs["mist"])
    node_tree.links.new(mask_reroute.outputs[0], workflow.inputs["arboreal_mask"])

    output = add_node(
        node_tree, "CompositorNodeOutputFile", "MistComplexOutlines::Outputs", (250, 120)
    )
    output.parent = family_frame
    output.base_path = "//"
    output.format.file_format = "PNG"
    output.format.color_mode = "RGBA"
    output.format.color_depth = "8"
    while len(output.file_slots) > 1:
        output.file_slots.remove(output.file_slots[-1])
    output.file_slots[0].path = "whole_forest_outline_v8_t10_"
    node_tree.links.new(workflow.outputs["Image"], output.inputs[0])

    composite = add_node(node_tree, "CompositorNodeComposite", "MistComplexOutlines::Composite", (250, -40))
    composite.parent = family_frame
    viewer = add_node(node_tree, "CompositorNodeViewer", "MistComplexOutlines::Viewer", (250, -180))
    viewer.parent = family_frame
    node_tree.links.new(workflow.outputs["Image"], composite.inputs["Image"])
    node_tree.links.new(workflow.outputs["Image"], viewer.inputs["Image"])

    CANONICAL_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(CANONICAL_BLEND))
    log(f"Saved {CANONICAL_BLEND}")


if __name__ == "__main__":
    main()
