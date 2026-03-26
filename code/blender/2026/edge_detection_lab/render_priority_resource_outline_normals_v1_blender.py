from __future__ import annotations

from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EDGE_LAB_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
OUTPUT_DIR = EDGE_LAB_ROOT / "outputs" / "exr_city_blender_priority_resource_outline_normals_v1"
BLEND_PATH = EDGE_LAB_ROOT / "exr_city_blender_priority_resource_outline_normals_v1.blend"

RESOURCE_COMBINED = EDGE_LAB_ROOT / "outputs" / "exr_city_blender_arboreal_resource_fills_v1" / "priority_resource_combined.png"
OUTLINE_FINE = EDGE_LAB_ROOT / "outputs" / "exr_city_blender_arboreal_mist_kirsch_sizes" / "priority_mist_kirsch_fine.png"
NORMAL_X = EDGE_LAB_ROOT / "outputs" / "exr_city_blender_normals_xyz_all_layers" / "priority_normal_x.png"
NORMAL_Y = EDGE_LAB_ROOT / "outputs" / "exr_city_blender_normals_xyz_all_layers" / "priority_normal_y.png"
NORMAL_Z = EDGE_LAB_ROOT / "outputs" / "exr_city_blender_normals_xyz_all_layers" / "priority_normal_z.png"

def log(message: str) -> None:
    print(f"[render_priority_resource_outline_normals_v1_blender] {message}")


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


def rgb_to_bw_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeRGBToBW", name, label, location, color=(0.14, 0.16, 0.20))
    ensure_link(node_tree, image_socket, node.inputs["Image"])
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


def value_to_grayscale(
    node_tree: bpy.types.NodeTree,
    value_socket,
    prefix: str,
    location: tuple[float, float],
    low_pos: float,
    high_pos: float,
    low_value: float,
    high_value: float,
):
    ramp = new_node(node_tree, "CompositorNodeValToRGB", f"{prefix}_ramp", f"{prefix}_ramp", location, color=(0.18, 0.14, 0.14))
    ramp.color_ramp.interpolation = "LINEAR"
    ramp.color_ramp.elements[0].position = low_pos
    ramp.color_ramp.elements[0].color = (low_value, low_value, low_value, 1.0)
    ramp.color_ramp.elements[1].position = high_pos
    ramp.color_ramp.elements[1].color = (high_value, high_value, high_value, 1.0)
    ensure_link(node_tree, value_socket, ramp.inputs["Fac"])
    return ramp.outputs["Image"]


def set_alpha_node(node_tree: bpy.types.NodeTree, image_socket, alpha_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeSetAlpha", name, label, location, color=(0.16, 0.20, 0.16))
    node.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    ensure_link(node_tree, alpha_socket, node.inputs["Alpha"])
    return node


def mix_node(node_tree: bpy.types.NodeTree, blend_type: str, fac: float, name: str, label: str, location: tuple[float, float], color1_socket, color2_socket):
    node = new_node(node_tree, "CompositorNodeMixRGB", name, label, location, color=(0.18, 0.12, 0.20))
    node.blend_type = blend_type
    node.inputs[0].default_value = fac
    ensure_link(node_tree, color1_socket, node.inputs[1])
    ensure_link(node_tree, color2_socket, node.inputs[2])
    return node


def build_axis_shading(
    node_tree: bpy.types.NodeTree,
    axis_slug: str,
    normal_image_socket,
    resource_image_socket,
    resource_alpha_socket,
    outline_image_socket,
    y: float,
):
    is_y = axis_slug == "y"
    normal_bw = rgb_to_bw_node(
        node_tree,
        normal_image_socket,
        f"Priority Normal {axis_slug.upper()} BW",
        f"Priority Normal {axis_slug.upper()} BW",
        (-1320.0, y),
    )
    shaded_gray = value_to_grayscale(
        node_tree,
        normal_bw.outputs["Val"],
        f"priority_normal_{axis_slug}_light",
        (-1020.0, y),
        low_pos=0.02 if is_y else 0.08 if axis_slug != "z" else 0.14,
        high_pos=0.98 if is_y else 0.92 if axis_slug != "z" else 0.98,
        low_value=0.0 if is_y else 0.04 if axis_slug != "z" else 0.10,
        high_value=1.00,
    )
    shaded_rgba = set_alpha_node(
        node_tree,
        shaded_gray,
        resource_alpha_socket,
        f"Priority Normal {axis_slug.upper()} RGBA",
        f"Priority Normal {axis_slug.upper()} RGBA",
        (-760.0, y),
    )
    resource_shaded_mix = mix_node(
        node_tree,
        "OVERLAY",
        1.0,
        f"Priority Resource {axis_slug.upper()} Overlay",
        f"Priority Resource {axis_slug.upper()} Overlay",
        (-500.0, y),
        resource_image_socket,
        shaded_rgba.outputs["Image"],
    )
    if is_y:
        resource_shaded_mix = mix_node(
            node_tree,
            "LINEAR_LIGHT",
            0.5,
            "Priority Resource Y Boost",
            "Priority Resource Y Boost",
            (-360.0, y + 120.0),
            resource_shaded_mix.outputs["Image"],
            shaded_rgba.outputs["Image"],
        )
    resource_shaded = set_alpha_node(
        node_tree,
        resource_shaded_mix.outputs["Image"],
        resource_alpha_socket,
        f"Priority Resource {axis_slug.upper()} Shaded",
        f"Priority Resource {axis_slug.upper()} Shaded",
        (-240.0, y),
    )
    resource_outline = alpha_over_node(
        node_tree,
        f"Priority Resource {axis_slug.upper()} Outline",
        f"Priority Resource {axis_slug.upper()} Outline",
        (20.0, y),
        resource_shaded.outputs["Image"],
        outline_image_socket,
    )
    return shaded_rgba.outputs["Image"], resource_shaded.outputs["Image"], resource_outline.outputs["Image"]


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

    resource = image_node(node_tree, RESOURCE_COMBINED, "Priority Resource Combined", "Priority Resource Combined", (-1560.0, 420.0))
    outline = image_node(node_tree, OUTLINE_FINE, "Priority Outline Fine", "Priority Outline Fine", (-1560.0, 80.0))
    normal_x = image_node(node_tree, NORMAL_X, "Priority Normal X", "Priority Normal X", (-1560.0, -380.0))
    normal_y = image_node(node_tree, NORMAL_Y, "Priority Normal Y", "Priority Normal Y", (-1560.0, -620.0))
    normal_z = image_node(node_tree, NORMAL_Z, "Priority Normal Z", "Priority Normal Z", (-1560.0, -860.0))

    resource_alpha = rgb_to_bw_node(node_tree, resource.outputs["Alpha"], "Priority Resource Alpha", "Priority Resource Alpha", (-1320.0, 620.0))
    light_x, resource_x, resource_x_outline = build_axis_shading(
        node_tree,
        "x",
        normal_x.outputs["Image"],
        resource.outputs["Image"],
        resource_alpha.outputs["Val"],
        outline.outputs["Image"],
        360.0,
    )
    light_y, resource_y, resource_y_outline = build_axis_shading(
        node_tree,
        "y",
        normal_y.outputs["Image"],
        resource.outputs["Image"],
        resource_alpha.outputs["Val"],
        outline.outputs["Image"],
        -40.0,
    )
    light_z, resource_z, resource_z_outline = build_axis_shading(
        node_tree,
        "z",
        normal_z.outputs["Image"],
        resource.outputs["Image"],
        resource_alpha.outputs["Val"],
        outline.outputs["Image"],
        -440.0,
    )

    rendered_paths = [
        output_node(node_tree, resource.outputs["Image"], "priority_resource_plain", (1480.0, 720.0)),
        output_node(node_tree, light_x, "priority_normallight_x", (1260.0, 360.0)),
        output_node(node_tree, resource_x, "priority_resource_normallit_x", (1480.0, 360.0)),
        output_node(node_tree, resource_x_outline, "priority_resource_normallit_x_outline_fine", (1720.0, 360.0)),
        output_node(node_tree, light_y, "priority_normallight_y", (1260.0, -40.0)),
        output_node(node_tree, resource_y, "priority_resource_normallit_y", (1480.0, -40.0)),
        output_node(node_tree, resource_y_outline, "priority_resource_normallit_y_outline_fine", (1720.0, -40.0)),
        output_node(node_tree, light_z, "priority_normallight_z", (1260.0, -440.0)),
        output_node(node_tree, resource_z, "priority_resource_normallit_z", (1480.0, -440.0)),
        output_node(node_tree, resource_z_outline, "priority_resource_normallit_z_outline_fine", (1720.0, -440.0)),
    ]

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (1980.0, -440.0))
    ensure_link(node_tree, resource_z_outline, composite.inputs[0])
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
