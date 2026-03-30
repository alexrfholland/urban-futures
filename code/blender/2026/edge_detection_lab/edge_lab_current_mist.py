from __future__ import annotations

import bpy


TREE_ID = 3
MIST_VARIANTS = (
    "mist_kirsch_thin",
    "mist_kirsch_fine",
    "mist_kirsch_extra_thin",
)


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def new_node(
    node_tree: bpy.types.NodeTree,
    bl_idname: str,
    name: str,
    label: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None = None,
    color: tuple[float, float, float] | None = None,
):
    existing = node_tree.nodes.get(name)
    if existing is not None:
        node_tree.nodes.remove(existing)
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if parent is not None:
        node.parent = parent
    if color is not None:
        node.use_custom_color = True
        node.color = color
    return node


def ensure_frame(
    node_tree: bpy.types.NodeTree,
    name: str,
    label: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None = None,
    color: tuple[float, float, float] | None = None,
) -> bpy.types.Node:
    frame = node_tree.nodes.get(name)
    if frame is None:
        frame = node_tree.nodes.new("NodeFrame")
        frame.name = name
    frame.label = label
    frame.location = location
    frame.label_size = 18
    frame.shrink = False
    frame.parent = parent
    if color is not None:
        frame.use_custom_color = True
        frame.color = color
    return frame


def remove_node_if_exists(node_tree: bpy.types.NodeTree, name: str) -> None:
    node = node_tree.nodes.get(name)
    if node is not None:
        node_tree.nodes.remove(node)


def require_node(node_tree: bpy.types.NodeTree, name: str) -> bpy.types.Node:
    node = node_tree.nodes.get(name)
    if node is None:
        raise ValueError(f"Missing node: {name}")
    return node


def set_alpha_node(
    node_tree: bpy.types.NodeTree,
    image_socket,
    alpha_socket,
    name: str,
    label: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None,
) -> bpy.types.Node:
    node = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        name,
        label,
        location,
        parent=parent,
        color=(0.16, 0.20, 0.16),
    )
    node.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    ensure_link(node_tree, alpha_socket, node.inputs["Alpha"])
    return node


def normalize_node(
    node_tree: bpy.types.NodeTree,
    value_socket,
    name: str,
    label: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None,
) -> bpy.types.Node:
    node = new_node(
        node_tree,
        "CompositorNodeNormalize",
        name,
        label,
        location,
        parent=parent,
        color=(0.16, 0.18, 0.20),
    )
    ensure_link(node_tree, value_socket, node.inputs[0])
    return node


def id_mask_node(
    node_tree: bpy.types.NodeTree,
    index_socket,
    name: str,
    label: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None,
) -> bpy.types.Node:
    node = new_node(
        node_tree,
        "CompositorNodeIDMask",
        name,
        label,
        location,
        parent=parent,
        color=(0.18, 0.18, 0.10),
    )
    node.index = TREE_ID
    node.use_antialiasing = True
    ensure_link(node_tree, index_socket, node.inputs["ID value"])
    return node


def math_node(
    node_tree: bpy.types.NodeTree,
    operation: str,
    name: str,
    label: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None,
    clamp: bool = True,
) -> bpy.types.Node:
    node = new_node(
        node_tree,
        "CompositorNodeMath",
        name,
        label,
        location,
        parent=parent,
        color=(0.18, 0.16, 0.20),
    )
    node.operation = operation
    node.use_clamp = clamp
    return node


def quantize_value_socket(
    node_tree: bpy.types.NodeTree,
    value_socket,
    name_prefix: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None,
):
    scale_up = math_node(
        node_tree,
        "MULTIPLY",
        f"{name_prefix} ScaleUp",
        f"{name_prefix} ScaleUp",
        location,
        parent=parent,
        clamp=False,
    )
    scale_up.inputs[1].default_value = 255.0
    ensure_link(node_tree, value_socket, scale_up.inputs[0])

    rounded = math_node(
        node_tree,
        "ROUND",
        f"{name_prefix} Round",
        f"{name_prefix} Round",
        (location[0] + 220.0, location[1]),
        parent=parent,
        clamp=False,
    )
    ensure_link(node_tree, scale_up.outputs["Value"], rounded.inputs[0])

    scale_down = math_node(
        node_tree,
        "DIVIDE",
        f"{name_prefix} ScaleDown",
        f"{name_prefix} ScaleDown",
        (location[0] + 440.0, location[1]),
        parent=parent,
        clamp=False,
    )
    scale_down.inputs[1].default_value = 255.0
    ensure_link(node_tree, rounded.outputs["Value"], scale_down.inputs[0])
    return scale_down.outputs["Value"]


def quantize_rgba_socket(
    node_tree: bpy.types.NodeTree,
    image_socket,
    name_prefix: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None,
):
    separate = new_node(
        node_tree,
        "CompositorNodeSepRGBA",
        f"{name_prefix} Separate",
        f"{name_prefix} Separate",
        location,
        parent=parent,
        color=(0.14, 0.18, 0.20),
    )
    ensure_link(node_tree, image_socket, separate.inputs["Image"])

    r = quantize_value_socket(node_tree, separate.outputs["R"], f"{name_prefix} R", (location[0] + 240.0, location[1] + 180.0), parent)
    g = quantize_value_socket(node_tree, separate.outputs["G"], f"{name_prefix} G", (location[0] + 240.0, location[1] + 60.0), parent)
    b = quantize_value_socket(node_tree, separate.outputs["B"], f"{name_prefix} B", (location[0] + 240.0, location[1] - 60.0), parent)
    a = quantize_value_socket(node_tree, separate.outputs["A"], f"{name_prefix} A", (location[0] + 240.0, location[1] - 180.0), parent)

    combine = new_node(
        node_tree,
        "CompositorNodeCombRGBA",
        f"{name_prefix} Combine",
        f"{name_prefix} Combine",
        (location[0] + 920.0, location[1]),
        parent=parent,
        color=(0.14, 0.18, 0.20),
    )
    ensure_link(node_tree, r, combine.inputs["R"])
    ensure_link(node_tree, g, combine.inputs["G"])
    ensure_link(node_tree, b, combine.inputs["B"])
    ensure_link(node_tree, a, combine.inputs["A"])
    return combine.outputs["Image"]


def connect_scene_outputs(
    node_tree: bpy.types.NodeTree,
    scene_name: str,
    visible_socket,
    mist_visible_socket,
) -> None:
    for variant in MIST_VARIANTS:
        base = f"MistOutlines::{scene_name}_{variant}"
        edge_rgb = require_node(node_tree, f"{base}_edge_rgb")
        mist_bw = require_node(node_tree, f"{base}_mist_bw")
        mist_rgba = require_node(node_tree, f"{base}_mist_rgba")
        ensure_link(node_tree, visible_socket, edge_rgb.inputs[1])
        ensure_link(node_tree, mist_visible_socket, mist_bw.inputs["Image"])
        ensure_link(node_tree, mist_visible_socket, mist_rgba.inputs["Image"])


def ensure_current_mist_exr_branch(scene: bpy.types.Scene) -> None:
    if scene.node_tree is None:
        raise ValueError("Scene has no compositor node tree")

    node_tree = scene.node_tree
    family = require_node(node_tree, "MistOutlines::FamilyFrame")
    exr_ref_pathway = require_node(node_tree, "AO::EXR Pathway")
    exr_ref_priority = require_node(node_tree, "AO::EXR Priority")
    exr_ref_trending = require_node(node_tree, "Resources::EXR Trending")

    legacy_prep = node_tree.nodes.get("MistOutlines::Frame Static Prep Inputs")
    if legacy_prep is not None:
        legacy_prep.label = "Legacy PNG Prep Inputs (unused)"

    exr_frame = ensure_frame(
        node_tree,
        "MistOutlines::Frame EXR Inputs",
        "EXR Inputs",
        (-3780.0, 1380.0),
        parent=family,
        color=(0.16, 0.16, 0.16),
    )
    mask_frame = ensure_frame(
        node_tree,
        "MistOutlines::Frame EXR Masks",
        "Visible Arboreal Masks",
        (-3340.0, 1380.0),
        parent=family,
        color=(0.18, 0.18, 0.12),
    )
    prep_frame = ensure_frame(
        node_tree,
        "MistOutlines::Frame EXR Prep",
        "Mist Prep",
        (-2820.0, 1380.0),
        parent=family,
        color=(0.14, 0.18, 0.20),
    )

    exr_pathway = new_node(
        node_tree,
        "CompositorNodeImage",
        "MistOutlines::EXR Pathway",
        "EXR Pathway",
        (-3640.0, 1080.0),
        parent=exr_frame,
        color=(0.12, 0.18, 0.10),
    )
    exr_pathway.image = exr_ref_pathway.image

    exr_priority = new_node(
        node_tree,
        "CompositorNodeImage",
        "MistOutlines::EXR Priority",
        "EXR Priority",
        (-3640.0, 700.0),
        parent=exr_frame,
        color=(0.12, 0.18, 0.10),
    )
    exr_priority.image = exr_ref_priority.image

    exr_trending = new_node(
        node_tree,
        "CompositorNodeImage",
        "MistOutlines::EXR Trending",
        "EXR Trending",
        (-3640.0, 320.0),
        parent=exr_frame,
        color=(0.12, 0.18, 0.10),
    )
    exr_trending.image = exr_ref_trending.image

    mask_pathway = id_mask_node(
        node_tree,
        exr_pathway.outputs["IndexOB"],
        "MistOutlines::mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (-3220.0, 980.0),
        parent=mask_frame,
    )
    mask_priority_all = id_mask_node(
        node_tree,
        exr_priority.outputs["IndexOB"],
        "MistOutlines::mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (-3220.0, 600.0),
        parent=mask_frame,
    )
    mask_priority_visible = math_node(
        node_tree,
        "MULTIPLY",
        "MistOutlines::mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (-2980.0, 600.0),
        parent=mask_frame,
    )
    ensure_link(node_tree, mask_priority_all.outputs["Alpha"], mask_priority_visible.inputs[0])
    ensure_link(node_tree, mask_pathway.outputs["Alpha"], mask_priority_visible.inputs[1])

    mask_trending = id_mask_node(
        node_tree,
        exr_trending.outputs["IndexOB"],
        "MistOutlines::mask_visible-arboreal_trending",
        "mask_visible-arboreal_trending",
        (-3220.0, 220.0),
        parent=mask_frame,
    )

    prep_specs = (
        ("pathway", exr_pathway, mask_pathway.outputs["Alpha"], 980.0),
        ("priority", exr_priority, mask_priority_visible.outputs["Value"], 600.0),
        ("trending", exr_trending, mask_trending.outputs["Alpha"], 220.0),
    )
    for scene_name, exr_node, alpha_socket, y in prep_specs:
        visible = set_alpha_node(
            node_tree,
            exr_node.outputs["Image"],
            alpha_socket,
            f"MistOutlines::{scene_name}_visible_arboreal",
            f"{scene_name}_visible_arboreal",
            (-2760.0, y + 60.0),
            parent=prep_frame,
        )
        mist_normalized = normalize_node(
            node_tree,
            exr_node.outputs["Mist"],
            f"MistOutlines::{scene_name}_mist_normalized",
            f"{scene_name}_mist_normalized",
            (-2760.0, y - 150.0),
            parent=prep_frame,
        )
        mist_visible = set_alpha_node(
            node_tree,
            mist_normalized.outputs[0],
            alpha_socket,
            f"MistOutlines::{scene_name}_mist_normalized_visible",
            f"{scene_name}_mist_normalized_visible",
            (-2520.0, y - 150.0),
            parent=prep_frame,
        )
        mist_quantized = quantize_rgba_socket(
            node_tree,
            mist_visible.outputs["Image"],
            f"MistOutlines::{scene_name}_mist_quantized",
            (-2260.0, y - 150.0),
            prep_frame,
        )
        connect_scene_outputs(
            node_tree,
            scene_name,
            visible.outputs["Image"],
            mist_quantized,
        )

    for scene_name in ("pathway", "priority", "trending"):
        remove_node_if_exists(node_tree, f"MistOutlines::{scene_name}_visible_png")
        remove_node_if_exists(node_tree, f"MistOutlines::{scene_name}_mist_visible_png")
