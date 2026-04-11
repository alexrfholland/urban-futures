from __future__ import annotations

from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_PATH = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city" / "city-trending_state0000.exr"
OUTPUT_DIR = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_v7" / "01_trending_objectid5"
BLEND_PATH = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_trending_objectid5_mask_v7.blend"
OBJECT_ID = 5


def log(message: str) -> None:
    print(f"[render_trending_objectid5_mask_v7_blender] {message}")


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


def build_scene(scene: bpy.types.Scene) -> Path:
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

    image_node = new_node(node_tree, "CompositorNodeImage", "Trending EXR", "Trending EXR", (-900.0, 120.0), color=(0.12, 0.18, 0.10))
    image_node.image = bpy.data.images.load(str(EXR_PATH), check_existing=True)

    id_mask = new_node(node_tree, "CompositorNodeIDMask", "mask_objectid5", "mask_objectid5", (-620.0, 120.0), color=(0.18, 0.18, 0.10))
    id_mask.index = OBJECT_ID
    id_mask.use_antialiasing = True
    ensure_link(node_tree, image_node.outputs["IndexOB"], id_mask.inputs["ID value"])

    rgba = new_node(node_tree, "CompositorNodeCombRGBA", "mask_objectid5_rgba", "mask_objectid5_rgba", (-360.0, 120.0), color=(0.14, 0.18, 0.20))
    ensure_link(node_tree, id_mask.outputs["Alpha"], rgba.inputs["R"])
    ensure_link(node_tree, id_mask.outputs["Alpha"], rgba.inputs["G"])
    ensure_link(node_tree, id_mask.outputs["Alpha"], rgba.inputs["B"])
    rgba.inputs["A"].default_value = 1.0

    set_alpha = new_node(node_tree, "CompositorNodeSetAlpha", "mask_objectid5_with_alpha", "mask_objectid5_with_alpha", (-120.0, 120.0), color=(0.16, 0.20, 0.16))
    set_alpha.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, rgba.outputs["Image"], set_alpha.inputs["Image"])
    ensure_link(node_tree, id_mask.outputs["Alpha"], set_alpha.inputs["Alpha"])

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (360.0, 120.0))
    ensure_link(node_tree, set_alpha.outputs["Image"], composite.inputs[0])

    return output_node(node_tree, set_alpha.outputs["Image"], "trending_objectid5_mask", (120.0, 120.0))


def main() -> None:
    scene = bpy.context.scene
    rendered_path = build_scene(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    log(f"Saved {BLEND_PATH}")
    bpy.ops.render.render(write_still=False)
    rename_output(rendered_path)


if __name__ == "__main__":
    main()
