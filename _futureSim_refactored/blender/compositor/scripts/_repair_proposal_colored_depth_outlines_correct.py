"""Repair a renderable copy of proposal_colored_depth_outlines_CORRECT.blend.

This script reads the user-validated visual layout from the CORRECT blend and
writes a new blend with the structural fixes needed for headless rendering:

1. Proposal RGB colors are normalized to the approved saturated palette.
2. A fresh 5-slot `ProposalColoredDepthOutput` node is created and wired to the
   masked proposal outputs instead of the single broken saved File Output node.
3. A Composite sink is ensured so Blender 4.2 actually evaluates the graph.

The input blend is never modified in place.
"""
from __future__ import annotations

import os
from pathlib import Path

import bpy

INPUT_BLEND = Path(os.environ["REPAIR_INPUT_BLEND"])
OUTPUT_BLEND = Path(os.environ["REPAIR_OUTPUT_BLEND"])
SCENE_NAME = "ProposalColoredDepthOutlines"
PRESERVE_COLORS = os.environ.get("REPAIR_PRESERVE_COLORS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

FRAME_COLORS = {
    "proposal- colonise": (1.000, 0.550, 0.000, 1.0),
    "proposal - decay": (0.900, 0.150, 0.150, 1.0),
    "proposal - deploy structure": (0.150, 0.350, 1.000, 1.0),
    "proposal - recruit": (0.150, 0.750, 0.150, 1.0),
    "proposal - release control": (0.550, 0.100, 0.850, 1.0),
}

OUTPUT_WIRING = [
    ("proposal-depth-colour_colonise_", "Set Alpha"),
    ("proposal-depth-colour_decay_", "Set Alpha.002"),
    ("proposal-depth-colour_deploy-structure_", "Set Alpha.003"),
    ("proposal-depth-colour_recruit_", "Set Alpha.004"),
    ("proposal-depth-colour_release_control_", "Set Alpha.001"),
]


def log(message: str) -> None:
    print(f"[repair] {message}")


def unlink_all_inputs(tree: bpy.types.NodeTree, node: bpy.types.Node) -> None:
    for sock in node.inputs:
        for link in list(sock.links):
            tree.links.remove(link)


def apply_frame_colors(tree: bpy.types.NodeTree) -> None:
    frame_by_name = {n.name: n for n in tree.nodes if n.bl_idname == "NodeFrame"}
    rgb_by_frame: dict[str, list[bpy.types.Node]] = {name: [] for name in frame_by_name}
    for node in tree.nodes:
        if node.bl_idname != "CompositorNodeRGB":
            continue
        if node.parent is not None and node.parent.name in rgb_by_frame:
            rgb_by_frame[node.parent.name].append(node)

    for frame_name, frame in frame_by_name.items():
        color = FRAME_COLORS.get(frame.label)
        if color is None:
            log(f"skip unmapped frame {frame.label!r}")
            continue
        for rgb in rgb_by_frame[frame_name]:
            old = tuple(rgb.outputs[0].default_value)
            rgb.outputs[0].default_value = color
            log(
                f"color {frame.label!r}: "
                f"({old[0]:.3f},{old[1]:.3f},{old[2]:.3f}) -> "
                f"({color[0]:.3f},{color[1]:.3f},{color[2]:.3f})"
            )


def ensure_rebuilt_file_output(tree: bpy.types.NodeTree) -> bpy.types.Node:
    for name in ("ProposalColoredDepthOutput", "File Output"):
        old = tree.nodes.get(name)
        if old is None or old.bl_idname != "CompositorNodeOutputFile":
            continue
        unlink_all_inputs(tree, old)
        old.name = f"_orphan_{name.replace(' ', '_')}"
        old.label = "_orphan"
        old.mute = True
        log(f"orphaned legacy output node {name!r}")

    output = tree.nodes.new("CompositorNodeOutputFile")
    output.name = "ProposalColoredDepthOutput"
    output.label = "ProposalColoredDepthOutput"
    output.format.file_format = "PNG"
    output.format.color_mode = "RGBA"
    output.format.color_depth = "8"
    output.location = (900, -250)

    while len(output.file_slots) > 1:
        output.file_slots.remove(output.file_slots[-1])
    output.file_slots[0].path = OUTPUT_WIRING[0][0]
    for path, _node_name in OUTPUT_WIRING[1:]:
        output.file_slots.new(path)

    for index, (path, node_name) in enumerate(OUTPUT_WIRING):
        source = tree.nodes.get(node_name)
        if source is None:
            raise RuntimeError(f"missing expected node {node_name!r}")
        output.file_slots[index].path = path
        tree.links.new(source.outputs["Image"], output.inputs[index])
        log(f"slot[{index}] {path!r} <- {node_name!r}.Image")

    return output


def ensure_composite_sink(tree: bpy.types.NodeTree) -> None:
    source = tree.nodes.get("Set Alpha")
    if source is None:
        raise RuntimeError("missing expected colonise node 'Set Alpha'")

    composite = next(
        (node for node in tree.nodes if node.bl_idname == "CompositorNodeComposite"),
        None,
    )
    if composite is None:
        composite = tree.nodes.new("CompositorNodeComposite")
        composite.name = "Composite"
        composite.location = (1150, -800)
        log("created Composite sink")
    unlink_all_inputs(tree, composite)
    tree.links.new(source.outputs["Image"], composite.inputs["Image"])
    log("wired Composite <- 'Set Alpha'.Image")

    viewer = next(
        (node for node in tree.nodes if node.bl_idname == "CompositorNodeViewer"),
        None,
    )
    if viewer is None:
        viewer = tree.nodes.new("CompositorNodeViewer")
        viewer.name = "Viewer"
        viewer.location = (1150, -1020)
        log("created Viewer sink")
    unlink_all_inputs(tree, viewer)
    tree.links.new(source.outputs["Image"], viewer.inputs["Image"])
    log("wired Viewer <- 'Set Alpha'.Image")


def main() -> None:
    log(f"open {INPUT_BLEND}")
    bpy.ops.wm.open_mainfile(filepath=str(INPUT_BLEND))

    scene = bpy.data.scenes[SCENE_NAME]
    tree = scene.node_tree

    if PRESERVE_COLORS:
        log("preserving existing RGB node colors from source blend")
    else:
        apply_frame_colors(tree)
    ensure_rebuilt_file_output(tree)
    ensure_composite_sink(tree)

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND), copy=False)
    log(f"saved repaired copy -> {OUTPUT_BLEND}")


if __name__ == "__main__":
    main()
