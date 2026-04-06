from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


EXR_ROOT = env_path(
    "EDGE_LAB_EXR_ROOT",
    REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city",
)
OUTPUT_DIR = env_path(
    "EDGE_LAB_OUTPUT_DIR",
    REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_normals_xyz_all_layers",
)
BLEND_PATH = env_path(
    "EDGE_LAB_BLEND_PATH",
    REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_normals_xyz_all_layers.blend",
)


def default_view_layer_specs() -> tuple[tuple[str, Path], ...]:
    return (
        ("pathway", env_path("EDGE_LAB_PATHWAY_EXR", EXR_ROOT / "city-pathway_state.exr")),
        ("priority", env_path("EDGE_LAB_PRIORITY_EXR", EXR_ROOT / "city-city_priority.exr")),
        ("trending", env_path("EDGE_LAB_TRENDING_EXR", EXR_ROOT / "city-trending_state.exr")),
        ("existing_condition", env_path("EDGE_LAB_EXISTING_EXR", EXR_ROOT / "city-existing_condition.exr")),
        ("bioenvelope", env_path("EDGE_LAB_BIOENVELOPE_EXR", EXR_ROOT / "city-city_bioenvelope.exr")),
    )


def parse_view_layer_specs() -> tuple[tuple[str, Path], ...]:
    raw = os.environ.get("EDGE_LAB_VIEW_LAYER_SPECS", "").strip()
    if not raw:
        return default_view_layer_specs()

    specs: list[tuple[str, Path]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "EDGE_LAB_VIEW_LAYER_SPECS entries must use 'slug=/absolute/or/relative/path.exr'"
            )
        slug, path_text = item.split("=", 1)
        slug = slug.strip()
        if not slug:
            raise ValueError("EDGE_LAB_VIEW_LAYER_SPECS contains an empty slug")
        specs.append((slug, Path(path_text.strip()).expanduser()))

    if not specs:
        raise ValueError("EDGE_LAB_VIEW_LAYER_SPECS did not resolve to any layers")
    return tuple(specs)


def log(message: str) -> None:
    print(f"[render_exr_normals_xyz_all_layers_blender] {message}")


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


def separate_rgba_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeSepRGBA", name, label, location, color=(0.14, 0.18, 0.20))
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


def grayscale_from_value(node_tree: bpy.types.NodeTree, value_socket, prefix: str, location: tuple[float, float]):
    ramp = new_node(node_tree, "CompositorNodeValToRGB", f"{prefix}_ramp", f"{prefix}_ramp", location, color=(0.18, 0.14, 0.14))
    ramp.color_ramp.interpolation = "LINEAR"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    ensure_link(node_tree, value_socket, ramp.inputs["Fac"])
    return ramp.outputs["Image"]


def set_alpha_node(node_tree: bpy.types.NodeTree, image_socket, alpha_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeSetAlpha", name, label, location, color=(0.16, 0.20, 0.16))
    node.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    ensure_link(node_tree, alpha_socket, node.inputs["Alpha"])
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
    view_layer_specs = parse_view_layer_specs()
    render_width, render_height = detect_resolution_from_exr(view_layer_specs[0][1])
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


def remap_normal_component(node_tree: bpy.types.NodeTree, value_socket, prefix: str, y: float):
    add = math_node(node_tree, "ADD", f"{prefix}_add", f"{prefix}_add", (-320.0, y), clamp=False)
    ensure_link(node_tree, value_socket, add.inputs[0])
    add.inputs[1].default_value = 1.0
    scale = math_node(node_tree, "MULTIPLY", f"{prefix}_scale", f"{prefix}_scale", (-80.0, y), clamp=True)
    ensure_link(node_tree, add.outputs["Value"], scale.inputs[0])
    scale.inputs[1].default_value = 0.5
    return scale.outputs["Value"]


def build_scene(scene: bpy.types.Scene) -> list[Path]:
    configure_scene(scene)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    rendered_paths: list[Path] = []
    composite_source = None
    view_layer_specs = parse_view_layer_specs()

    for idx, (slug, exr_path) in enumerate(view_layer_specs):
        if not exr_path.exists():
            raise FileNotFoundError(f"EXR not found for layer '{slug}': {exr_path}")
        y = 860.0 - idx * 420.0
        exr = image_node(node_tree, exr_path, f"{slug}_exr", f"{slug}_exr", (-1560.0, y))
        normal = separate_rgba_node(node_tree, exr.outputs["Normal"], f"{slug}_normal_sep", f"{slug}_normal_sep", (-1300.0, y))

        for axis_name, socket_name, offset in (("x", "R", 120.0), ("y", "G", 0.0), ("z", "B", -120.0)):
            remapped = remap_normal_component(node_tree, normal.outputs[socket_name], f"{slug}_{axis_name}", y + offset)
            grayscale = grayscale_from_value(node_tree, remapped, f"{slug}_{axis_name}", (160.0, y + offset))
            rgba = set_alpha_node(
                node_tree,
                grayscale,
                exr.outputs["Alpha"],
                f"{slug}_{axis_name}_rgba",
                f"{slug}_{axis_name}_rgba",
                (380.0, y + offset),
            )
            rendered_paths.append(
                output_node(
                    node_tree,
                    rgba.outputs["Image"],
                    f"{slug}_normal_{axis_name}",
                    (620.0, y + offset),
                )
            )
            if composite_source is None:
                composite_source = rgba.outputs["Image"]

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (900.0, 180.0))
    if composite_source is not None:
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
