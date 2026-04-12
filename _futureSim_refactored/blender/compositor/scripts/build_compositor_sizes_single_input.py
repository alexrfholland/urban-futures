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
BLEND_PATH = CANONICAL_ROOT / "compositor_sizes_single_input.blend"

SIZE_CLASSES = (
    ("size_small", (1.0,), (0.401978, 0.708376, 0.111932, 1.0)),
    ("size_medium", (2.0,), (0.323143, 0.485150, 0.730461, 1.0)),
    ("size_large", (3.0,), (0.947307, 0.346704, 0.181164, 1.0)),
    ("size_senescing", (4.0,), (0.830770, 0.327778, 0.558340, 1.0)),
    ("size_snag", (5.0,), (0.973445, 0.768151, 0.097587, 1.0)),
    ("size_fallen", (6.0,), (0.274677, 0.250158, 0.520996, 1.0)),
    ("size_decayed", (7.0,), (0.114435, 0.238398, 0.208637, 1.0)),
    ("size_artificial", (-1.0,), (1.000000, 0.000000, 0.000000, 1.0)),
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


def build_colored_mask(node_tree, mask_socket, slug: str, rgba: tuple[float, float, float, float], x: float, y: float):
    rgb = new_node(node_tree, "CompositorNodeRGB", f"{slug}::rgb", "rgb", (x, y))
    rgb.outputs[0].default_value = rgba

    set_alpha = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{slug}::rgba",
        "rgba",
        (x + 260.0, y),
    )
    set_alpha.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, rgb.outputs[0], set_alpha.inputs["Image"])
    ensure_link(node_tree, mask_socket, set_alpha.inputs["Alpha"])
    return set_alpha.outputs["Image"]


def build_combined(node_tree, rendered, x: float, y: float):
    current_socket = rendered[0][1]
    for index, (slug, socket) in enumerate(rendered[1:], start=1):
        alpha_over = new_node(
            node_tree,
            "CompositorNodeAlphaOver",
            f"size_combined::alpha_over_{index}",
            "alpha-over",
            (x + index * 260.0, y),
        )
        alpha_over.premul = 1.0
        ensure_link(node_tree, current_socket, alpha_over.inputs[1])
        ensure_link(node_tree, socket, alpha_over.inputs[2])
        current_socket = alpha_over.outputs["Image"]
    return current_socket


def add_output_node(node_tree: bpy.types.NodeTree, rendered, combined_socket):
    output = new_node(
        node_tree,
        "CompositorNodeOutputFile",
        "SizesSingle::Outputs",
        "SizesSingle::Outputs",
        (2740.0, 0.0),
    )
    output.base_path = "//"
    output.format.file_format = "PNG"
    output.format.color_mode = "RGBA"
    output.format.color_depth = "8"

    while len(output.file_slots) > 1:
        output.file_slots.remove(output.file_slots[-1])
    output.file_slots[0].path = "size_combined_"

    for slug, _socket in rendered:
        output.file_slots.new(slug)

    ensure_link(node_tree, combined_socket, output.inputs[0])
    for index, (slug, socket) in enumerate(rendered, start=1):
        output.file_slots[index].path = f"{slug}_"
        ensure_link(node_tree, socket, output.inputs[index])

    return output


def main() -> None:
    if not REFERENCE_EXR.exists():
        raise FileNotFoundError(REFERENCE_EXR)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = "SizesSingleInput"
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
    for row, (slug, target_values, rgba) in enumerate(SIZE_CLASSES):
        y = 760.0 - row * 240.0
        mask_socket = build_class_band_mask(node_tree, size_socket, slug, target_values, 240.0, y)
        image_socket = build_colored_mask(node_tree, mask_socket, slug, rgba, 620.0, y)
        rendered.append((slug, image_socket))

    combined_socket = build_combined(node_tree, rendered, 1120.0, 200.0)

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (2740.0, 320.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (2740.0, 180.0))
    ensure_link(node_tree, combined_socket, composite.inputs[0])
    ensure_link(node_tree, combined_socket, viewer.inputs[0])
    add_output_node(node_tree, rendered, combined_socket)

    BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))


if __name__ == "__main__":
    main()
