from __future__ import annotations

from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"
OUTPUT_DIR = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_v6" / "01_control_size_masks"
BLEND_PATH = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_control_size_masks_v6.blend"

PATHWAY_EXR = EXR_ROOT / "city-pathway_state.exr"
PRIORITY_EXR = EXR_ROOT / "city-city_priority.exr"
TRENDING_EXR = EXR_ROOT / "city-trending_state.exr"

TREE_ID = 3
MATCH_EPSILON = 0.1

SIZE_OPACITY_MAP = {
    1: 0.30,  # small
    2: 0.50,  # medium
    3: 0.70,  # large
    4: 1.00,  # senescing
    5: 1.00,  # snag
    6: 1.00,  # fallen
}

CONTROL_OPACITY_MAP = {
    1: 0.30,  # street-tree
    2: 0.60,  # park-tree
    3: 1.00,  # reserve-tree
    4: 1.00,  # improved-tree
}


def log(message: str) -> None:
    print(f"[render_exr_control_size_masks_v6_blender] {message}")


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


def combine_rgba_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float]):
    return new_node(node_tree, "CompositorNodeCombRGBA", name, label, location, color=(0.14, 0.18, 0.20))


def file_output_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float], stem: str) -> Path:
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


def build_match_weight(
    node_tree: bpy.types.NodeTree,
    value_socket,
    target_value: int,
    weight: float,
    prefix: str,
    y: float,
):
    subtract = math_node(node_tree, "SUBTRACT", f"{prefix}_sub_{target_value}", f"{prefix}_sub_{target_value}", (-220.0, y), clamp=False)
    subtract.inputs[1].default_value = float(target_value)
    ensure_link(node_tree, value_socket, subtract.inputs[0])

    absolute = math_node(node_tree, "ABSOLUTE", f"{prefix}_abs_{target_value}", f"{prefix}_abs_{target_value}", (20.0, y), clamp=False)
    ensure_link(node_tree, subtract.outputs["Value"], absolute.inputs[0])

    less_than = math_node(node_tree, "LESS_THAN", f"{prefix}_match_{target_value}", f"{prefix}_match_{target_value}", (260.0, y), clamp=True)
    less_than.inputs[1].default_value = MATCH_EPSILON
    ensure_link(node_tree, absolute.outputs["Value"], less_than.inputs[0])

    weighted = math_node(node_tree, "MULTIPLY", f"{prefix}_weight_{target_value}", f"{prefix}_weight_{target_value}", (500.0, y), clamp=True)
    weighted.inputs[1].default_value = weight
    ensure_link(node_tree, less_than.outputs["Value"], weighted.inputs[0])
    return weighted.outputs["Value"]


def build_weighted_mask(
    node_tree: bpy.types.NodeTree,
    value_socket,
    opacity_map: dict[int, float],
    prefix: str,
    start_y: float,
):
    weighted_sockets = []
    for index, (target_value, weight) in enumerate(opacity_map.items()):
        weighted_sockets.append(
            build_match_weight(
                node_tree,
                value_socket,
                target_value,
                weight,
                prefix,
                start_y - index * 110.0,
            )
        )

    current_socket = weighted_sockets[0]
    combine_x = 760.0
    combine_y = start_y - 55.0
    for index, socket in enumerate(weighted_sockets[1:], start=1):
        add = math_node(node_tree, "ADD", f"{prefix}_add_{index}", f"{prefix}_add_{index}", (combine_x + (index - 1) * 220.0, combine_y), clamp=True)
        ensure_link(node_tree, current_socket, add.inputs[0])
        ensure_link(node_tree, socket, add.inputs[1])
        current_socket = add.outputs["Value"]
    return current_socket


def build_mask_output(
    node_tree: bpy.types.NodeTree,
    value_socket,
    alpha_socket,
    prefix: str,
    label: str,
    y: float,
) -> Path:
    masked_value = math_node(node_tree, "MULTIPLY", f"{prefix}_masked_value", f"{prefix}_masked_value", (1860.0, y), clamp=True)
    ensure_link(node_tree, value_socket, masked_value.inputs[0])
    ensure_link(node_tree, alpha_socket, masked_value.inputs[1])

    grayscale = combine_rgba_node(node_tree, f"{prefix}_rgba", f"{prefix}_rgba", (2100.0, y))
    ensure_link(node_tree, masked_value.outputs["Value"], grayscale.inputs["R"])
    ensure_link(node_tree, masked_value.outputs["Value"], grayscale.inputs["G"])
    ensure_link(node_tree, masked_value.outputs["Value"], grayscale.inputs["B"])
    grayscale.inputs["A"].default_value = 1.0

    with_alpha = set_alpha_node(
        node_tree,
        grayscale.outputs["Image"],
        alpha_socket,
        f"{prefix}_with_alpha",
        f"{prefix}_with_alpha",
        (2340.0, y),
    )
    return file_output_node(
        node_tree,
        with_alpha.outputs["Image"],
        f"Output {label}",
        f"Output {label}",
        (2580.0, y),
        prefix,
    )


def configure_scene(scene: bpy.types.Scene) -> None:
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
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass


def build_scene(scene: bpy.types.Scene) -> list[Path]:
    configure_scene(scene)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    pathway = image_node(node_tree, PATHWAY_EXR, "EXR Pathway", "EXR Pathway", (-1700.0, 560.0))
    priority = image_node(node_tree, PRIORITY_EXR, "EXR Priority", "EXR Priority", (-1700.0, 80.0))
    trending = image_node(node_tree, TRENDING_EXR, "EXR Trending", "EXR Trending", (-1700.0, -400.0))

    mask_visible_pathway = id_mask_node(
        node_tree,
        pathway.outputs["IndexOB"],
        "mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (-1420.0, 420.0),
    )
    mask_visible_trending = id_mask_node(
        node_tree,
        trending.outputs["IndexOB"],
        "mask_visible-arboreal_trending",
        "mask_visible-arboreal_trending",
        (-1420.0, -540.0),
    )
    mask_all_priority = id_mask_node(
        node_tree,
        priority.outputs["IndexOB"],
        "mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (-1420.0, -60.0),
    )
    mask_visible_priority = math_node(
        node_tree,
        "MULTIPLY",
        "mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (-1180.0, -60.0),
    )
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])

    rendered_paths: list[Path] = []
    layer_specs = [
        ("pathway", pathway, mask_visible_pathway.outputs["Alpha"], 560.0),
        ("priority", priority, mask_visible_priority.outputs["Value"], 80.0),
        ("trending", trending, mask_visible_trending.outputs["Alpha"], -400.0),
    ]

    for prefix, exr_node, alpha_socket, y in layer_specs:
        size_mask = build_weighted_mask(
            node_tree,
            exr_node.outputs["size"],
            SIZE_OPACITY_MAP,
            f"{prefix}_size",
            y + 120.0,
        )
        control_mask = build_weighted_mask(
            node_tree,
            exr_node.outputs["control"],
            CONTROL_OPACITY_MAP,
            f"{prefix}_control",
            y - 420.0,
        )
        rendered_paths.append(
            build_mask_output(
                node_tree,
                size_mask,
                alpha_socket,
                f"{prefix}_size_mask",
                f"{prefix}_size_mask",
                y + 20.0,
            )
        )
        rendered_paths.append(
            build_mask_output(
                node_tree,
                control_mask,
                alpha_socket,
                f"{prefix}_control_mask",
                f"{prefix}_control_mask",
                y - 520.0,
            )
        )

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (2900.0, 20.0))
    ensure_link(node_tree, trending.outputs["Image"], composite.inputs[0])
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
