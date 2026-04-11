from __future__ import annotations

from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"
OUTPUT_DIR = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_v2" / "03_trending_depth_edges"
BLEND_PATH = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_trending_depth_edges_v2.blend"
PURPLE_EDGE_PATH = OUTPUT_DIR / "trending_depth_kirsch_purple_edges.png"

TRENDING_EXR = EXR_ROOT / "city-trending_state.exr"
TREE_ID = 3
EDGE_COLOR = (0.133, 0.071, 0.231, 1.0)


def log(message: str) -> None:
    print(f"[render_exr_trending_depth_edges_v2_blender] {message}")


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


def set_alpha_node(
    node_tree: bpy.types.NodeTree,
    image_socket,
    alpha_socket,
    name: str,
    label: str,
    location: tuple[float, float],
):
    node = new_node(node_tree, "CompositorNodeSetAlpha", name, label, location, color=(0.16, 0.20, 0.16))
    node.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    ensure_link(node_tree, alpha_socket, node.inputs["Alpha"])
    return node


def threshold_ramp(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], low: float, high: float):
    node = new_node(node_tree, "CompositorNodeValToRGB", name, label, location, color=(0.18, 0.14, 0.14))
    node.color_ramp.interpolation = "LINEAR"
    node.color_ramp.elements[0].position = low
    node.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node.color_ramp.elements[1].position = high
    node.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    return node


def rgb_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], rgba: tuple[float, float, float, float]):
    node = new_node(node_tree, "CompositorNodeRGB", name, label, location, color=(0.18, 0.12, 0.20))
    node.outputs[0].default_value = rgba
    return node


def output_node(
    node_tree: bpy.types.NodeTree,
    image_socket,
    name: str,
    label: str,
    location: tuple[float, float],
    stem: str,
):
    node = new_node(node_tree, "CompositorNodeOutputFile", name, label, location, color=(0.12, 0.20, 0.14))
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


def build_scene(scene: bpy.types.Scene) -> list[Path]:
    scene.use_nodes = True
    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    trending = image_node(node_tree, TRENDING_EXR, "EXR Trending", "EXR Trending", (-1500.0, 300.0))
    mask_trending = id_mask_node(
        node_tree,
        trending.outputs["IndexOB"],
        "mask_visible-arboreal_trending",
        "mask_visible-arboreal_trending",
        (-1160.0, 80.0),
    )

    normalize_depth = new_node(
        node_tree,
        "CompositorNodeNormalize",
        "depth_normalized_trending",
        "depth_normalized_trending",
        (-1160.0, 300.0),
        color=(0.16, 0.18, 0.20),
    )
    ensure_link(node_tree, trending.outputs["Depth"], normalize_depth.inputs[0])

    depth_masked = set_alpha_node(
        node_tree,
        normalize_depth.outputs[0],
        mask_trending.outputs["Alpha"],
        "depth_normalized_visible-arboreal_trending",
        "depth_normalized_visible-arboreal_trending",
        (-860.0, 300.0),
    )

    kirsch = new_node(
        node_tree,
        "CompositorNodeFilter",
        "edge_trending_kirsch",
        "edge_trending_kirsch",
        (-560.0, 300.0),
        color=(0.18, 0.16, 0.10),
    )
    kirsch.filter_type = "KIRSCH"
    ensure_link(node_tree, depth_masked.outputs["Image"], kirsch.inputs[0])

    normalize_edges = new_node(
        node_tree,
        "CompositorNodeNormalize",
        "edge_trending_normalized",
        "edge_trending_normalized",
        (-300.0, 300.0),
        color=(0.16, 0.18, 0.20),
    )
    ensure_link(node_tree, kirsch.outputs[0], normalize_edges.inputs[0])

    threshold = threshold_ramp(
        node_tree,
        "edge_trending_threshold",
        "edge_trending_threshold",
        (-40.0, 300.0),
        0.12,
        0.32,
    )
    ensure_link(node_tree, normalize_edges.outputs[0], threshold.inputs["Fac"])

    edge_color = rgb_node(
        node_tree,
        "edge_trending_purple_rgb",
        "edge_trending_purple_rgb",
        (220.0, 120.0),
        EDGE_COLOR,
    )
    purple_edges = set_alpha_node(
        node_tree,
        edge_color.outputs[0],
        threshold.outputs["Alpha"],
        "edge_trending_purple",
        "edge_trending_purple",
        (220.0, 300.0),
    )

    rendered_paths = [
        output_node(
            node_tree,
            depth_masked.outputs["Image"],
            "Output Trending Depth Normalized",
            "Output Trending Depth Normalized",
            (520.0, 300.0),
            "trending_depth_normalized_visible_arboreal",
        ),
    ]

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (860.0, 300.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (860.0, 80.0))
    ensure_link(node_tree, purple_edges.outputs["Image"], composite.inputs[0])
    ensure_link(node_tree, purple_edges.outputs["Image"], viewer.inputs[0])
    return rendered_paths


def main() -> None:
    scene = bpy.context.scene
    rendered_paths = build_scene(scene)
    scene.render.filepath = str(PURPLE_EDGE_PATH)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    log(f"Saved {BLEND_PATH}")
    bpy.ops.render.render(write_still=True)
    for path in rendered_paths:
        rename_output(path)


if __name__ == "__main__":
    main()
