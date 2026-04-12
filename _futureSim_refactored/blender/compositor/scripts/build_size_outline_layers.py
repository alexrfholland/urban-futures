from __future__ import annotations

from pathlib import Path

import bpy

REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
COMPOSITOR_ROOT = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"
REFERENCE_EXR = (
    REPO_ROOT
    / "_data-refactored"
    / "blenderv2"
    / "output"
    / "4.9"
    / "city_baseline"
    / "city_baseline__positive_state__8k64s.exr"
)
BLEND_PATH = CANONICAL_ROOT / "size_outline_layers.blend"

OUTLINE_RGBA = (0.015996, 0.006048822, 0.043735046, 1.0)
SIZE_CLASSES = (
    ("size-outline_small", (1.0,)),
    ("size-outline_medium", (2.0,)),
    ("size-outline_large", (3.0,)),
    ("size-outline_senescing", (4.0,)),
    ("size-outline_snag", (5.0,)),
    ("size-outline_fallen", (6.0,)),
    ("size-outline_decayed", (7.0,)),
    ("size-outline_artificial", (-1.0,)),
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
):
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    return node


def socket_by_name(node: bpy.types.Node, preferred_name: str):
    for socket in node.outputs:
        if socket.name == preferred_name:
            return socket
    normalized = preferred_name.lower().replace(" ", "_")
    for socket in node.outputs:
        if socket.name.lower().replace(" ", "_") == normalized:
            return socket
    raise KeyError(f"Missing socket {preferred_name} on {node.name}")


def detect_resolution(image: bpy.types.Image) -> tuple[int, int]:
    width, height = image.size[:]
    if width > 0 and height > 0:
        return width, height
    return 7680, 4320


def configure_scene(scene: bpy.types.Scene, width: int, height: int) -> None:
    scene.use_nodes = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
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


def build_class_band_mask(node_tree, value_socket, slug: str, target_values: tuple[float, ...], x: float, y: float):
    band_outputs = []
    for index, target_value in enumerate(target_values):
        offset_y = y + index * 120.0
        lower = new_node(
            node_tree,
            "CompositorNodeMath",
            f"{slug}::lower-bound::{index}",
            "lower-bound",
            (x, offset_y - 40.0),
        )
        lower.operation = "GREATER_THAN"
        lower.inputs[1].default_value = target_value - 0.5
        ensure_link(node_tree, value_socket, lower.inputs[0])

        upper = new_node(
            node_tree,
            "CompositorNodeMath",
            f"{slug}::upper-bound::{index}",
            "upper-bound",
            (x, offset_y + 40.0),
        )
        upper.operation = "LESS_THAN"
        upper.inputs[1].default_value = target_value + 0.5
        ensure_link(node_tree, value_socket, upper.inputs[0])

        inside_band = new_node(
            node_tree,
            "CompositorNodeMath",
            f"{slug}::inside-band::{index}",
            "inside-band",
            (x + 260.0, offset_y),
        )
        inside_band.operation = "MULTIPLY"
        inside_band.use_clamp = True
        ensure_link(node_tree, lower.outputs["Value"], inside_band.inputs[0])
        ensure_link(node_tree, upper.outputs["Value"], inside_band.inputs[1])
        band_outputs.append(inside_band.outputs["Value"])

    if len(band_outputs) == 1:
        return band_outputs[0]

    add = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{slug}::combined-band",
        "combined-band",
        (x + 520.0, y + 60.0),
    )
    add.operation = "ADD"
    add.use_clamp = True
    ensure_link(node_tree, band_outputs[0], add.inputs[0])
    ensure_link(node_tree, band_outputs[1], add.inputs[1])
    return add.outputs["Value"]


def build_outline_mask(node_tree, mask_socket, slug: str, x: float, y: float):
    dilate = new_node(
        node_tree,
        "CompositorNodeDilateErode",
        f"{slug}::dilate",
        "dilate",
        (x, y - 40.0),
    )
    dilate.mode = "DISTANCE"
    dilate.distance = 1
    ensure_link(node_tree, mask_socket, dilate.inputs["Mask"])

    erode = new_node(
        node_tree,
        "CompositorNodeDilateErode",
        f"{slug}::erode",
        "erode",
        (x, y + 80.0),
    )
    erode.mode = "DISTANCE"
    erode.distance = -1
    ensure_link(node_tree, mask_socket, erode.inputs["Mask"])

    edge_band = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{slug}::edge-band",
        "edge-band",
        (x + 260.0, y),
    )
    edge_band.operation = "SUBTRACT"
    edge_band.use_clamp = True
    ensure_link(node_tree, dilate.outputs["Mask"], edge_band.inputs[0])
    ensure_link(node_tree, erode.outputs["Mask"], edge_band.inputs[1])

    edge_binary = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{slug}::edge-binary",
        "edge-binary",
        (x + 500.0, y),
    )
    edge_binary.operation = "GREATER_THAN"
    edge_binary.inputs[1].default_value = 0.0
    ensure_link(node_tree, edge_band.outputs["Value"], edge_binary.inputs[0])

    rgb = new_node(node_tree, "CompositorNodeRGB", f"{slug}::rgb", "rgb", (x + 760.0, y))
    rgb.outputs[0].default_value = OUTLINE_RGBA

    rgba = new_node(node_tree, "CompositorNodeSetAlpha", f"{slug}::rgba", "rgba", (x + 1020.0, y))
    rgba.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, rgb.outputs[0], rgba.inputs["Image"])
    ensure_link(node_tree, edge_binary.outputs["Value"], rgba.inputs["Alpha"])
    return rgba.outputs["Image"]


def add_output_node(node_tree: bpy.types.NodeTree, rendered):
    output = new_node(
        node_tree,
        "CompositorNodeOutputFile",
        "SizeOutlineOutput",
        "SizeOutlineOutput",
        (1760.0, 0.0),
    )
    output.base_path = "//"
    output.format.file_format = "PNG"
    output.format.color_mode = "RGBA"
    output.format.color_depth = "8"

    while len(output.file_slots) > 1:
        output.file_slots.remove(output.file_slots[-1])
    output.file_slots[0].path = f"{rendered[0][0]}_"

    for slug, _socket in rendered[1:]:
        output.file_slots.new(slug)

    for index, (slug, socket) in enumerate(rendered):
        output.file_slots[index].path = f"{slug}_"
        ensure_link(node_tree, socket, output.inputs[index])

    return output


def main() -> None:
    if not REFERENCE_EXR.exists():
        raise FileNotFoundError(REFERENCE_EXR)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "SizeOutline"
    scene.use_nodes = True
    node_tree = scene.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    exr_node = new_node(node_tree, "CompositorNodeImage", "EXR", "EXR", (0.0, 0.0))
    exr_node.image = bpy.data.images.load(str(REFERENCE_EXR), check_existing=True)
    width, height = detect_resolution(exr_node.image)
    configure_scene(scene, width, height)

    rendered = []
    size_socket = socket_by_name(exr_node, "size")
    for row, (slug, target_values) in enumerate(SIZE_CLASSES):
        y = 720.0 - row * 260.0
        mask_socket = build_class_band_mask(node_tree, size_socket, slug, target_values, 220.0, y)
        image_socket = build_outline_mask(node_tree, mask_socket, slug, 520.0, y)
        rendered.append((slug, image_socket))

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (1760.0, 280.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (1760.0, 140.0))
    ensure_link(node_tree, rendered[0][1], composite.inputs[0])
    ensure_link(node_tree, rendered[0][1], viewer.inputs[0])
    add_output_node(node_tree, rendered)

    BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))


if __name__ == "__main__":
    main()
