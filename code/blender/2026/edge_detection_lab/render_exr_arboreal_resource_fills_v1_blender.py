from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = Path(
    os.environ.get(
        "EDGE_LAB_EXR_ROOT",
        str(REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "EDGE_LAB_OUTPUT_DIR",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_arboreal_resource_fills_v1"),
    )
)
BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_BLEND_PATH",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_arboreal_resource_fills_v1.blend"),
    )
)

PATHWAY_EXR = Path(os.environ.get("EDGE_LAB_PATHWAY_EXR", str(EXR_ROOT / "pathway_state.exr")))
PRIORITY_EXR = Path(os.environ.get("EDGE_LAB_PRIORITY_EXR", str(EXR_ROOT / "priority.exr")))
TRENDING_EXR = Path(os.environ.get("EDGE_LAB_TRENDING_EXR", str(EXR_ROOT / "trending_state.exr")))

TREE_ID = 3

RESOURCE_SPECS = (
    {"slug": "hollow", "mask_socket": "resource_hollow_mask", "hex": "#ce6dd9"},
    {"slug": "epiphyte", "mask_socket": "resource_epiphyte_mask", "hex": "#c5e28e"},
    {"slug": "fallen", "mask_socket": "resource_fallen_log_mask", "hex": "#8f89bf"},
    {"slug": "peeling", "mask_socket": "resource_peeling_bark_mask", "hex": "#ff85be"},
    {"slug": "dead", "mask_socket": "resource_dead_branch_mask", "hex": "#ffcc01"},
    {"slug": "perch", "mask_socket": "resource_perch_branch_mask", "hex": "#ffcb00"},
    {"slug": "other", "mask_socket": "resource_none_mask", "hex": "#cecece"},
)

STACK_TOP_TO_BOTTOM = ("hollow", "epiphyte", "fallen", "peeling", "dead", "perch", "other")


def detect_resolution_from_exr(path: Path) -> tuple[int, int]:
    try:
        info = subprocess.check_output(
            ["oiiotool", "--info", "-v", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        match = re.search(r":\s+(\d+)\s+x\s+(\d+),", info)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    image = bpy.data.images.load(str(path), check_existing=True)
    width, height = image.size[:]
    if width > 0 and height > 0:
        return int(width), int(height)
    return 3840, 2160


def log(message: str) -> None:
    print(f"[render_exr_arboreal_resource_fills_v1_blender] {message}")


def clear_node_tree(node_tree: bpy.types.NodeTree) -> None:
    for link in list(node_tree.links):
        node_tree.links.remove(link)
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


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
    color: tuple[float, float, float] | None = None,
):
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if color is not None:
        node.use_custom_color = True
        node.color = color
    return node


def frame_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], color: tuple[float, float, float]):
    node = new_node(node_tree, "NodeFrame", name, label, location, color=color)
    node.label_size = 18
    node.shrink = False
    return node


def image_node(node_tree: bpy.types.NodeTree, path: Path, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeImage", name, label, location, color=(0.12, 0.18, 0.10))
    node.image = bpy.data.images.load(str(path), check_existing=True)
    return node


def id_mask_node(node_tree: bpy.types.NodeTree, index_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeIDMask", name, label, location, color=(0.18, 0.18, 0.10))
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
    clamp: bool = True,
):
    node = new_node(node_tree, "CompositorNodeMath", name, label, location, color=(0.18, 0.16, 0.20))
    node.operation = operation
    node.use_clamp = clamp
    return node


def rgb_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], rgba: tuple[float, float, float, float]):
    node = new_node(node_tree, "CompositorNodeRGB", name, label, location, color=(0.18, 0.12, 0.20))
    node.outputs[0].default_value = rgba
    return node


def solid_color_image(node_tree: bpy.types.NodeTree, source_socket, name: str, label: str, location: tuple[float, float], rgba: tuple[float, float, float, float]):
    color = rgb_node(node_tree, f"{name}_rgb", f"{label}_rgb", (location[0] - 220.0, location[1]), rgba)
    mix = new_node(node_tree, "CompositorNodeMixRGB", name, label, location, color=(0.18, 0.12, 0.20))
    mix.blend_type = "MIX"
    mix.inputs[0].default_value = 1.0
    ensure_link(node_tree, source_socket, mix.inputs[1])
    ensure_link(node_tree, color.outputs[0], mix.inputs[2])
    return mix.outputs["Image"]


def set_alpha_node(node_tree: bpy.types.NodeTree, image_socket, alpha_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeSetAlpha", name, label, location, color=(0.16, 0.20, 0.16))
    node.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    ensure_link(node_tree, alpha_socket, node.inputs["Alpha"])
    return node


def alpha_over_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], bottom_socket, top_socket):
    node = new_node(node_tree, "CompositorNodeAlphaOver", name, label, location, color=(0.14, 0.20, 0.16))
    node.premul = 1.0
    ensure_link(node_tree, bottom_socket, node.inputs[1])
    ensure_link(node_tree, top_socket, node.inputs[2])
    return node


def output_node(node_tree: bpy.types.NodeTree, image_socket, stem: str, location: tuple[float, float]) -> Path:
    node = new_node(node_tree, "CompositorNodeOutputFile", f"Output {stem}", f"Output {stem}", location, color=(0.12, 0.20, 0.14))
    node.base_path = str(OUTPUT_DIR)
    node.format.file_format = "PNG"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "8"
    node.file_slots[0].path = f"{stem}_"
    ensure_link(node_tree, image_socket, node.inputs[0])
    return OUTPUT_DIR / f"{stem}_0001.png"


def rename_output(rendered_path: Path) -> Path:
    final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
    if rendered_path.exists():
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")
    return final_path


def configure_scene(scene: bpy.types.Scene) -> None:
    render_width, render_height = detect_resolution_from_exr(PATHWAY_EXR)
    scene.use_nodes = True
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
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
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def srgb_channel_to_linear(value: int) -> float:
    normalized = value / 255.0
    if normalized <= 0.04045:
        return normalized / 12.92
    return ((normalized + 0.055) / 1.055) ** 2.4


def hex_to_linear_rgba(hex_value: str) -> tuple[float, float, float, float]:
    value = hex_value.lstrip("#")
    r = srgb_channel_to_linear(int(value[0:2], 16))
    g = srgb_channel_to_linear(int(value[2:4], 16))
    b = srgb_channel_to_linear(int(value[4:6], 16))
    return (r, g, b, 1.0)


def socket_by_name(node, preferred_name: str):
    for socket in node.outputs:
        if socket.name == preferred_name:
            return socket
    normalized_preferred = preferred_name.lower().replace(" ", "_")
    for socket in node.outputs:
        if socket.name.lower().replace(" ", "_") == normalized_preferred:
            return socket
    raise KeyError(f"Socket '{preferred_name}' not found on {node.name}. Available: {[socket.name for socket in node.outputs]}")


def multiply_sockets(node_tree: bpy.types.NodeTree, left_socket, right_socket, name: str, label: str, location: tuple[float, float]):
    node = math_node(node_tree, "MULTIPLY", name, label, location, clamp=True)
    ensure_link(node_tree, left_socket, node.inputs[0])
    ensure_link(node_tree, right_socket, node.inputs[1])
    return node.outputs["Value"]


def build_scene_layer(
    node_tree: bpy.types.NodeTree,
    slug: str,
    exr_node,
    source_image_socket,
    resource_base_mask_socket,
    rendered_paths: list[Path],
    y: float,
    parent,
):
    colored_outputs: dict[str, object] = {}

    for index, resource in enumerate(RESOURCE_SPECS):
        resource_y = y - index * 160.0
        resource_mask_socket = socket_by_name(exr_node, resource["mask_socket"])
        masked_resource = multiply_sockets(
            node_tree,
            resource_base_mask_socket,
            resource_mask_socket,
            f"{slug}_{resource['slug']}_mask",
            f"{slug}_{resource['slug']}_mask",
            (-620.0, resource_y),
        )
        node_tree.nodes[f"{slug}_{resource['slug']}_mask"].parent = parent
        color_image = solid_color_image(
            node_tree,
            source_image_socket,
            f"{slug}_{resource['slug']}_fill_color",
            f"{slug}_{resource['slug']}_fill_color",
            (-360.0, resource_y),
            hex_to_linear_rgba(resource["hex"]),
        )
        node_tree.nodes[f"{slug}_{resource['slug']}_fill_color_rgb"].parent = parent
        node_tree.nodes[f"{slug}_{resource['slug']}_fill_color"].parent = parent
        rgba = set_alpha_node(
            node_tree,
            color_image,
            masked_resource,
            f"{slug}_{resource['slug']}_rgba",
            f"{slug}_{resource['slug']}_rgba",
            (-100.0, resource_y),
        )
        rgba.parent = parent
        rendered_paths.append(
            output_node(
                node_tree,
                rgba.outputs["Image"],
                f"{slug}_{resource['slug']}",
                (180.0, resource_y),
            )
        )
        node_tree.nodes[f"Output {slug}_{resource['slug']}"].parent = parent
        colored_outputs[resource["slug"]] = rgba.outputs["Image"]

    stack_bottom_to_top = list(reversed(STACK_TOP_TO_BOTTOM))
    combined_socket = colored_outputs[stack_bottom_to_top[0]]
    combined_y = y - len(RESOURCE_SPECS) * 160.0 - 80.0
    for index, resource_slug in enumerate(stack_bottom_to_top[1:], start=1):
        over = alpha_over_node(
            node_tree,
            f"{slug}_combine_{resource_slug}",
            f"{slug}_combine_{resource_slug}",
            (180.0 + index * 240.0, combined_y),
            combined_socket,
            colored_outputs[resource_slug],
        )
        over.parent = parent
        combined_socket = over.outputs["Image"]

    rendered_paths.append(
        output_node(
            node_tree,
            combined_socket,
            f"{slug}_resource_combined",
            (2100.0, combined_y),
        )
    )
    node_tree.nodes[f"Output {slug}_resource_combined"].parent = parent
    return combined_socket


def build_scene(scene: bpy.types.Scene) -> list[Path]:
    configure_scene(scene)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    inputs_frame = frame_node(node_tree, "Frame Inputs", "EXR Inputs", (-1820.0, 1180.0), (0.16, 0.16, 0.16))
    masks_frame = frame_node(node_tree, "Frame Masks", "Visible Arboreal Masks", (-1360.0, 1120.0), (0.18, 0.18, 0.14))
    pathway_frame = frame_node(node_tree, "Frame Pathway", "Pathway Resource Fills", (-760.0, 1120.0), (0.16, 0.20, 0.16))
    priority_frame = frame_node(node_tree, "Frame Priority", "Priority Resource Fills", (-760.0, -180.0), (0.16, 0.20, 0.16))
    trending_frame = frame_node(node_tree, "Frame Trending", "Trending Resource Fills", (-760.0, -1480.0), (0.16, 0.20, 0.16))

    pathway = image_node(node_tree, PATHWAY_EXR, "EXR Pathway", "EXR Pathway", (-1700.0, 960.0))
    pathway.parent = inputs_frame
    priority = image_node(node_tree, PRIORITY_EXR, "EXR Priority", "EXR Priority", (-1700.0, -340.0))
    priority.parent = inputs_frame
    trending = None
    if TRENDING_EXR.exists():
        trending = image_node(node_tree, TRENDING_EXR, "EXR Trending", "EXR Trending", (-1700.0, -1640.0))
        trending.parent = inputs_frame

    mask_pathway = id_mask_node(
        node_tree,
        pathway.outputs["IndexOB"],
        "mask_visible_arboreal_pathway",
        "mask_visible_arboreal_pathway",
        (-1280.0, 920.0),
    )
    mask_pathway.parent = masks_frame
    mask_priority_all = id_mask_node(
        node_tree,
        priority.outputs["IndexOB"],
        "mask_all_arboreal_priority",
        "mask_all_arboreal_priority",
        (-1280.0, -380.0),
    )
    mask_priority_all.parent = masks_frame
    mask_priority_visible = multiply_sockets(
        node_tree,
        mask_priority_all.outputs["Alpha"],
        mask_pathway.outputs["Alpha"],
        "mask_visible_arboreal_priority",
        "mask_visible_arboreal_priority",
        (-1040.0, -380.0),
    )
    node_tree.nodes["mask_visible_arboreal_priority"].parent = masks_frame
    mask_trending = None
    if trending is not None:
        mask_trending = id_mask_node(
            node_tree,
            trending.outputs["IndexOB"],
            "mask_visible_arboreal_trending",
            "mask_visible_arboreal_trending",
            (-1280.0, -1680.0),
        )
        mask_trending.parent = masks_frame

    rendered_paths: list[Path] = []
    composite_source = build_scene_layer(
        node_tree,
        "pathway",
        pathway,
        pathway.outputs["Image"],
        mask_pathway.outputs["Alpha"],
        rendered_paths,
        760.0,
        pathway_frame,
    )
    build_scene_layer(
        node_tree,
        "priority",
        priority,
        priority.outputs["Image"],
        mask_priority_visible,
        rendered_paths,
        -540.0,
        priority_frame,
    )
    if trending is not None and mask_trending is not None:
        build_scene_layer(
            node_tree,
            "trending",
            trending,
            trending.outputs["Image"],
            mask_trending.outputs["Alpha"],
            rendered_paths,
            -1840.0,
            trending_frame,
        )

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (2400.0, 200.0))
    composite.parent = pathway_frame
    ensure_link(node_tree, composite_source, composite.inputs[0])
    return rendered_paths


def main() -> None:
    scene = bpy.context.scene
    rendered_paths = build_scene(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    log(f"Saved {BLEND_PATH}")
    bpy.ops.render.render(write_still=False)
    for path in rendered_paths:
        rename_output(path)


if __name__ == "__main__":
    main()
