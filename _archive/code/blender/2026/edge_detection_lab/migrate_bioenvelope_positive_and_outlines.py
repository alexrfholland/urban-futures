from __future__ import annotations

import os
from pathlib import Path

import bpy


BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_BLEND_PATH",
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/compositor_bioenvelope.blend",
    )
)


POSITIVE_RENAMES = {
    "bioenvelope_brownroof": "positive_bioenvelope_brownroof",
    "bioenvelope_exoskeleton": "positive_bioenvelope_exoskeleton",
    "bioenvelope_footprintdepaved": "positive_bioenvelope_footprintdepaved",
    "bioenvelope_full_image": "positive_bioenvelope_full_image",
    "bioenvelope_greenroof": "positive_bioenvelope_greenroof",
    "bioenvelope_livingfacade": "positive_bioenvelope_livingfacade",
    "bioenvelope_otherground": "positive_bioenvelope_otherground",
    "bioenvelope_rewilded": "positive_bioenvelope_rewilded",
    "bioenvelope_outlines-depth": "positive_bioenvelope_outlines-depth",
    "bioenvelope_outlines-simple": "positive_bioenvelope_outlines-simple",
}


NEW_INPUTS = (
    ("existing__Depth", "NodeSocketFloat"),
    ("existing__IndexOB", "NodeSocketFloat"),
    ("trending__Depth", "NodeSocketFloat"),
    ("trending__IndexOB", "NodeSocketFloat"),
)


NEW_OUTPUTS = (
    ("base_bioenvelope_outlines-depth", "NodeSocketColor"),
    ("base_bioenvelope_outlines-simple", "NodeSocketColor"),
    ("trending_bioenvelope_outlines-depth", "NodeSocketColor"),
    ("trending_bioenvelope_outlines-simple", "NodeSocketColor"),
)


OUTPUT_SLOT_STEMS = [
    "envelope_voxels",
    "trending_bioenvelope_greenroof",
    "trending_bioenvelope_livingfacade",
    "trending_bioenvelope_footprintdepaved",
    "trending_bioenvelope_rewilded",
    "trending_bioenvelope_otherground",
    "trending_bioenvelope_brownroof",
    "trending_bioenvelope_exoskeleton",
    "trending_bioenvelope_full-image",
    "positive_bioenvelope_greenroof",
    "positive_bioenvelope_livingfacade",
    "positive_bioenvelope_footprintdepaved",
    "positive_bioenvelope_rewilded",
    "positive_bioenvelope_otherground",
    "positive_bioenvelope_brownroof",
    "positive_bioenvelope_exoskeleton",
    "positive_bioenvelope_full-image",
    "base_bioenvelope_greenroof",
    "base_bioenvelope_livingfacade",
    "base_bioenvelope_footprintdepaved",
    "base_bioenvelope_rewilded",
    "base_bioenvelope_otherground",
    "base_bioenvelope_brownroof",
    "base_bioenvelope_exoskeleton",
    "base_bioenvelope_full-image",
    "positive_bioenvelope_outlines-depth",
    "positive_bioenvelope_outlines-simple",
    "base_bioenvelope_outlines-depth",
    "base_bioenvelope_outlines-simple",
    "trending_bioenvelope_outlines-depth",
    "trending_bioenvelope_outlines-simple",
]


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def interface_socket_by_name(
    node_tree: bpy.types.NodeTree,
    name: str,
    in_out: str,
):
    for item in node_tree.interface.items_tree:
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        if getattr(item, "in_out", None) != in_out:
            continue
        if item.name == name:
            return item
    return None


def ensure_interface_socket(
    node_tree: bpy.types.NodeTree,
    name: str,
    in_out: str,
    socket_type: str,
):
    item = interface_socket_by_name(node_tree, name, in_out)
    if item is not None:
        return item
    return node_tree.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)


def clone_color_ramp(src: bpy.types.Node, dst: bpy.types.Node) -> None:
    while len(dst.color_ramp.elements) > 1:
        dst.color_ramp.elements.remove(dst.color_ramp.elements[-1])
    src_elements = list(src.color_ramp.elements)
    dst_elements = list(dst.color_ramp.elements)
    while len(dst_elements) < len(src_elements):
        dst.color_ramp.elements.new(src_elements[len(dst_elements)].position)
        dst_elements = list(dst.color_ramp.elements)
    for src_el, dst_el in zip(src_elements, dst_elements):
        dst_el.position = src_el.position
        dst_el.color = src_el.color
        dst_el.alpha = src_el.alpha
    dst.color_ramp.interpolation = src.color_ramp.interpolation
    dst.color_ramp.color_mode = src.color_ramp.color_mode
    dst.color_ramp.hue_interpolation = src.color_ramp.hue_interpolation


def clone_node(
    node_tree: bpy.types.NodeTree,
    src: bpy.types.Node,
    name: str,
    label: str,
    location: tuple[float, float],
):
    node = node_tree.nodes.new(src.bl_idname)
    node.name = name
    node.label = label
    node.location = location
    node.width = src.width
    node.height = src.height
    node.hide = src.hide
    node.use_custom_color = src.use_custom_color
    if src.use_custom_color:
        node.color = src.color

    for prop in ("operation", "use_clamp", "distance", "falloff", "mode", "filter_type", "index", "use_antialiasing"):
        if hasattr(src, prop) and hasattr(node, prop):
            try:
                setattr(node, prop, getattr(src, prop))
            except Exception:
                pass

    if src.bl_idname == "CompositorNodeGroup":
        node.node_tree = src.node_tree

    if src.bl_idname == "CompositorNodeValToRGB":
        clone_color_ramp(src, node)

    for src_socket, dst_socket in zip(src.inputs, node.inputs):
        if src_socket.is_linked:
            continue
        try:
            dst_socket.default_value = src_socket.default_value
        except Exception:
            pass
    return node


def rename_positive_outputs(group_tree: bpy.types.NodeTree) -> None:
    for old_name, new_name in POSITIVE_RENAMES.items():
        item = interface_socket_by_name(group_tree, old_name, "OUTPUT")
        if item is not None:
            item.name = new_name

    for old_name, new_name in POSITIVE_RENAMES.items():
        node = group_tree.nodes.get(f"Current BioEnvelope :: {old_name}")
        if node is not None:
            node.name = f"Current BioEnvelope :: {new_name}"
            node.label = new_name


def add_outline_branch(
    group_tree: bpy.types.NodeTree,
    branch_prefix: str,
    depth_input_socket,
    index_input_socket,
    value_mask_socket,
    y_shift: float,
    output_depth_name: str,
    output_simple_name: str,
    use_rewild_value_mask: bool = False,
) -> None:
    src_mask = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Mask"]
    src_depth_norm = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Depth Normalize"]
    src_depth_prepped = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Depth Prepped"]
    src_kirsch = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Depth Kirsch"]
    src_ramp = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Depth Ramp"]
    src_bw = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Depth BW"]
    src_depth_alpha = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Depth Alpha"]
    src_depth_group = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Depth Group"]
    src_dilate = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Simple Dilate"]
    src_subtract = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Simple Subtract"]
    src_simple_alpha = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Simple Alpha"]
    src_simple_group = group_tree.nodes["Current BioEnvelope :: Rewild Outline :: Simple Group"]

    if use_rewild_value_mask:
        mask = group_tree.nodes.new("CompositorNodeMath")
        mask.name = f"Current BioEnvelope :: {branch_prefix} Outline :: Mask"
        mask.label = f"{branch_prefix} Rewilded Mask"
        mask.location = (src_mask.location.x, src_mask.location.y + y_shift)
        mask.operation = "COMPARE"
        mask.inputs[1].default_value = 4.0
        mask.inputs[2].default_value = 0.1
        mask.use_custom_color = True
        mask.color = src_mask.color
    else:
        mask = clone_node(
            group_tree,
            src_mask,
            f"Current BioEnvelope :: {branch_prefix} Outline :: Mask",
            f"{branch_prefix} Rewilded Mask",
            (src_mask.location.x, src_mask.location.y + y_shift),
        )
    depth_norm = clone_node(
        group_tree,
        src_depth_norm,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Depth Normalize",
        f"{branch_prefix} depth normalize",
        (src_depth_norm.location.x, src_depth_norm.location.y + y_shift),
    )
    depth_prepped = clone_node(
        group_tree,
        src_depth_prepped,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Depth Prepped",
        f"{branch_prefix} depth prepped",
        (src_depth_prepped.location.x, src_depth_prepped.location.y + y_shift),
    )
    kirsch = clone_node(
        group_tree,
        src_kirsch,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Depth Kirsch",
        f"{branch_prefix} depth kirsch",
        (src_kirsch.location.x, src_kirsch.location.y + y_shift),
    )
    ramp = clone_node(
        group_tree,
        src_ramp,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Depth Ramp",
        f"{branch_prefix} depth ramp",
        (src_ramp.location.x, src_ramp.location.y + y_shift),
    )
    bw = clone_node(
        group_tree,
        src_bw,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Depth BW",
        f"{branch_prefix} depth bw",
        (src_bw.location.x, src_bw.location.y + y_shift),
    )
    depth_alpha = clone_node(
        group_tree,
        src_depth_alpha,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Depth Alpha",
        f"{branch_prefix} depth alpha",
        (src_depth_alpha.location.x, src_depth_alpha.location.y + y_shift),
    )
    depth_group = clone_node(
        group_tree,
        src_depth_group,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Depth Group",
        f"{branch_prefix} Depth Group",
        (src_depth_group.location.x, src_depth_group.location.y + y_shift),
    )
    dilate = clone_node(
        group_tree,
        src_dilate,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Simple Dilate",
        f"{branch_prefix} simple dilate",
        (src_dilate.location.x, src_dilate.location.y + y_shift),
    )
    subtract = clone_node(
        group_tree,
        src_subtract,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Simple Subtract",
        f"{branch_prefix} simple subtract",
        (src_subtract.location.x, src_subtract.location.y + y_shift),
    )
    simple_alpha = clone_node(
        group_tree,
        src_simple_alpha,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Simple Alpha",
        f"{branch_prefix} simple alpha",
        (src_simple_alpha.location.x, src_simple_alpha.location.y + y_shift),
    )
    simple_group = clone_node(
        group_tree,
        src_simple_group,
        f"Current BioEnvelope :: {branch_prefix} Outline :: Simple Group",
        f"{branch_prefix} Simple Group",
        (src_simple_group.location.x, src_simple_group.location.y + y_shift),
    )

    depth_reroute = group_tree.nodes.new("NodeReroute")
    depth_reroute.name = f"Current BioEnvelope :: {output_depth_name}"
    depth_reroute.label = output_depth_name
    depth_reroute.location = (depth_group.location.x + 260.0, depth_group.location.y)

    simple_reroute = group_tree.nodes.new("NodeReroute")
    simple_reroute.name = f"Current BioEnvelope :: {output_simple_name}"
    simple_reroute.label = output_simple_name
    simple_reroute.location = (simple_group.location.x + 260.0, simple_group.location.y)

    ensure_link(group_tree, depth_input_socket, depth_norm.inputs[0])
    if use_rewild_value_mask:
        ensure_link(group_tree, value_mask_socket, mask.inputs[0])
        mask_socket = mask.outputs["Value"]
    else:
        ensure_link(group_tree, index_input_socket, mask.inputs["ID value"])
        mask_socket = mask.outputs["Alpha"]
    ensure_link(group_tree, depth_norm.outputs["Value"], depth_prepped.inputs["Image"])
    ensure_link(group_tree, mask_socket, depth_prepped.inputs["Alpha"])
    ensure_link(group_tree, depth_prepped.outputs["Image"], kirsch.inputs["Image"])
    ensure_link(group_tree, kirsch.outputs["Image"], ramp.inputs["Fac"])
    ensure_link(group_tree, ramp.outputs["Image"], bw.inputs["Image"])
    ensure_link(group_tree, bw.outputs["Val"], depth_alpha.inputs[0])
    ensure_link(group_tree, mask_socket, depth_alpha.inputs[1])
    ensure_link(group_tree, depth_alpha.outputs["Value"], depth_group.inputs["Mask Alpha"])
    ensure_link(group_tree, depth_group.outputs["Image"], depth_reroute.inputs[0])

    ensure_link(group_tree, mask_socket, dilate.inputs["Mask"])
    ensure_link(group_tree, dilate.outputs["Mask"], subtract.inputs[0])
    ensure_link(group_tree, mask_socket, subtract.inputs[1])
    ensure_link(group_tree, subtract.outputs["Value"], simple_alpha.inputs[0])
    ensure_link(group_tree, simple_alpha.outputs["Value"], simple_group.inputs["Mask Alpha"])
    ensure_link(group_tree, simple_group.outputs["Image"], simple_reroute.inputs[0])

    group_output = next(node for node in group_tree.nodes if node.bl_idname == "NodeGroupOutput")
    ensure_link(group_tree, depth_reroute.outputs[0], group_output.inputs[output_depth_name])
    ensure_link(group_tree, simple_reroute.outputs[0], group_output.inputs[output_simple_name])


def rebuild_top_output_node(
    node_tree: bpy.types.NodeTree,
    workflow_node: bpy.types.Node,
    old_output_node: bpy.types.Node,
) -> bpy.types.Node:
    parent = old_output_node.parent
    location = tuple(old_output_node.location)
    base_path = old_output_node.base_path
    node_tree.nodes.remove(old_output_node)

    output = node_tree.nodes.new("CompositorNodeOutputFile")
    output.name = "Current BioEnvelope ::Outputs"
    output.label = "Current BioEnvelope ::Outputs"
    output.location = location
    output.parent = parent
    output.base_path = base_path
    output.format.file_format = "PNG"
    output.format.color_mode = "RGBA"
    output.format.color_depth = "8"

    while len(output.file_slots) > 1:
        output.file_slots.remove(output.file_slots[-1])
    output.file_slots[0].path = f"{OUTPUT_SLOT_STEMS[0]}_"
    for stem in OUTPUT_SLOT_STEMS[1:]:
        output.file_slots.new(stem)
        output.file_slots[-1].path = f"{stem}_"

    for link in list(output.inputs[0].links):
        node_tree.links.remove(link)
    for i, stem in enumerate(OUTPUT_SLOT_STEMS):
        socket_name = stem.replace("-", "_")
        source = workflow_node.outputs.get(stem) or workflow_node.outputs.get(socket_name)
        if source is None:
            raise ValueError(f"Missing workflow output socket for {stem}")
        ensure_link(node_tree, source, output.inputs[i])
    return output


def reconnect_group_inputs(node_tree: bpy.types.NodeTree, workflow_node: bpy.types.Node) -> None:
    existing = node_tree.nodes["Current BioEnvelope :: EXR Existing"]
    trending = node_tree.nodes["Current BioEnvelope :: EXR Trending"]

    ensure_link(node_tree, existing.outputs["Depth"], workflow_node.inputs["existing__Depth"])
    ensure_link(node_tree, existing.outputs["IndexOB"], workflow_node.inputs["existing__IndexOB"])
    ensure_link(node_tree, trending.outputs["Depth"], workflow_node.inputs["trending__Depth"])
    ensure_link(node_tree, trending.outputs["IndexOB"], workflow_node.inputs["trending__IndexOB"])


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(BLEND_PATH)

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes["Current"]
    node_tree = scene.node_tree
    workflow_node = node_tree.nodes["Current BioEnvelope :: Workflow Group"]
    group_tree = workflow_node.node_tree

    for name, socket_type in NEW_INPUTS:
        ensure_interface_socket(group_tree, name, "INPUT", socket_type)
    for name, socket_type in NEW_OUTPUTS:
        ensure_interface_socket(group_tree, name, "OUTPUT", socket_type)

    rename_positive_outputs(group_tree)

    group_input = next(node for node in group_tree.nodes if node.bl_idname == "NodeGroupInput")

    add_outline_branch(
        group_tree,
        "Base",
        group_input.outputs["existing__Depth"],
        group_input.outputs["existing__IndexOB"],
        group_input.outputs["existing__world_design_bioenvelope"],
        600.0,
        "base_bioenvelope_outlines-depth",
        "base_bioenvelope_outlines-simple",
        use_rewild_value_mask=True,
    )
    add_outline_branch(
        group_tree,
        "Trending",
        group_input.outputs["trending__Depth"],
        group_input.outputs["trending__IndexOB"],
        None,
        -600.0,
        "trending_bioenvelope_outlines-depth",
        "trending_bioenvelope_outlines-simple",
    )

    old_output = node_tree.nodes["Current BioEnvelope ::Outputs"]
    rebuild_top_output_node(node_tree, workflow_node, old_output)
    reconnect_group_inputs(node_tree, workflow_node)

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    print(f"[migrate_bioenvelope_positive_and_outlines] Updated {BLEND_PATH}")


if __name__ == "__main__":
    main()
