from __future__ import annotations

import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"
OUTPUT_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender"
BLEND_PATH = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_lab.blend"

PATHWAY_EXR = EXR_ROOT / "city-pathway_state.exr"
PRIORITY_EXR = EXR_ROOT / "city-city_priority.exr"
TRENDING_EXR = EXR_ROOT / "city-trending_state.exr"
EXISTING_EXR = EXR_ROOT / "city-existing_condition.exr"

TREE_ID = 3
EDGE_COLOR = (0.133, 0.071, 0.231, 1.0)


def log(message: str) -> None:
    print(f"[render_exr_edge_variants_blender] {message}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_node_tree(node_tree: bpy.types.NodeTree) -> None:
    for link in list(node_tree.links):
        node_tree.links.remove(link)
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def new_node(
    node_tree: bpy.types.NodeTree,
    bl_idname: str,
    name: str,
    location: tuple[float, float],
    label: str | None = None,
    color: tuple[float, float, float] | None = None,
):
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label or name
    node.location = location
    if color is not None:
        node.use_custom_color = True
        node.color = color
    return node


def load_image(path: Path, colorspace: str | None = None) -> bpy.types.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    image = bpy.data.images.load(str(path), check_existing=True)
    if colorspace:
        try:
            image.colorspace_settings.name = colorspace
        except Exception:
            pass
    return image


def image_input(node_tree: bpy.types.NodeTree, path: Path, name: str, location: tuple[float, float]) -> bpy.types.CompositorNodeImage:
    node = new_node(node_tree, "CompositorNodeImage", name, location, color=(0.12, 0.18, 0.10))
    node.image = load_image(path)
    return node


def id_mask(node_tree: bpy.types.NodeTree, index_socket, object_id: int, name: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeIDMask", name, location, color=(0.18, 0.18, 0.10))
    node.index = object_id
    node.use_antialiasing = True
    ensure_link(node_tree, index_socket, node.inputs["ID value"])
    return node


def set_alpha(
    node_tree: bpy.types.NodeTree,
    image_socket,
    alpha_socket,
    name: str,
    location: tuple[float, float],
    mode: str = "REPLACE_ALPHA",
):
    node = new_node(node_tree, "CompositorNodeSetAlpha", name, location, color=(0.16, 0.20, 0.16))
    node.mode = mode
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    ensure_link(node_tree, alpha_socket, node.inputs["Alpha"])
    return node


def math_node(node_tree: bpy.types.NodeTree, operation: str, name: str, location: tuple[float, float], use_clamp: bool = False):
    node = new_node(node_tree, "CompositorNodeMath", name, location, color=(0.18, 0.16, 0.20))
    node.operation = operation
    node.use_clamp = use_clamp
    return node


def blur_node(node_tree: bpy.types.NodeTree, name: str, location: tuple[float, float], pixels: int):
    node = new_node(node_tree, "CompositorNodeBlur", name, location, color=(0.14, 0.16, 0.20))
    node.filter_type = "GAUSS"
    node.use_relative = False
    node.size_x = pixels
    node.size_y = pixels
    return node


def threshold_ramp(
    node_tree: bpy.types.NodeTree,
    name: str,
    location: tuple[float, float],
    low: float,
    high: float,
):
    node = new_node(node_tree, "CompositorNodeValToRGB", name, location, color=(0.18, 0.14, 0.14))
    node.color_ramp.interpolation = "LINEAR"
    node.color_ramp.elements[0].position = low
    node.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node.color_ramp.elements[1].position = high
    node.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    return node


def dilate_erode(node_tree: bpy.types.NodeTree, name: str, location: tuple[float, float], distance: int):
    node = new_node(node_tree, "CompositorNodeDilateErode", name, location, color=(0.16, 0.14, 0.20))
    node.mode = "DISTANCE"
    node.distance = distance
    return node


def rgb_constant(node_tree: bpy.types.NodeTree, name: str, location: tuple[float, float], rgba: tuple[float, float, float, float]):
    node = new_node(node_tree, "CompositorNodeRGB", name, location, color=(0.18, 0.12, 0.20))
    node.outputs[0].default_value = rgba
    return node


def alpha_over(node_tree: bpy.types.NodeTree, name: str, location: tuple[float, float], bottom_socket, top_socket):
    node = new_node(node_tree, "CompositorNodeAlphaOver", name, location, color=(0.14, 0.20, 0.16))
    node.premul = 1.0
    ensure_link(node_tree, bottom_socket, node.inputs[1])
    ensure_link(node_tree, top_socket, node.inputs[2])
    return node


def file_output(
    node_tree: bpy.types.NodeTree,
    name: str,
    location: tuple[float, float],
    directory: Path,
    stem: str,
    source_socket,
) -> tuple[bpy.types.CompositorNodeOutputFile, Path]:
    node = new_node(node_tree, "CompositorNodeOutputFile", name, location, color=(0.12, 0.20, 0.14))
    node.base_path = str(ensure_dir(directory))
    node.format.file_format = "PNG"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "8"
    node.file_slots[0].path = f"{stem}_"
    ensure_link(node_tree, source_socket, node.inputs[0])
    return node, directory / f"{stem}_0001.png"


def rename_outputs(paths: list[Path]) -> None:
    for rendered in paths:
        final = rendered.with_name(rendered.name.replace("_0001", ""))
        if not rendered.exists():
            continue
        if final.exists():
            final.unlink()
        rendered.replace(final)
        log(f"Renamed {rendered.name} -> {final.name}")


def build_ao_branch(node_tree: bpy.types.NodeTree, image_node, label: str, y: float, directory: Path, rendered_paths: list[Path]) -> None:
    mask = id_mask(node_tree, image_node.outputs["IndexOB"], TREE_ID, f"{label} Mask", (-1100.0, y))
    rgba = set_alpha(
        node_tree,
        image_node.outputs["AO"],
        mask.outputs["Alpha"],
        f"{label} AO RGBA",
        (-860.0, y),
        mode="REPLACE_ALPHA",
    )
    _, rendered = file_output(node_tree, f"{label} AO Output", (-620.0, y), directory, label.lower().replace(" ", "_"), rgba.outputs["Image"])
    rendered_paths.append(rendered)


def build_edge_width(
    node_tree: bpy.types.NodeTree,
    normalized_socket,
    mask_socket,
    base_name: str,
    y: float,
    low: float,
    high: float,
    strong_low: float,
    strong_high: float,
    core_distance: int,
    wide_distance: int,
    blur_pixels: int,
):
    presence = threshold_ramp(node_tree, f"{base_name} Presence", (80.0, y + 80.0), low, high)
    strong = threshold_ramp(node_tree, f"{base_name} Strong", (80.0, y - 80.0), strong_low, strong_high)
    ensure_link(node_tree, normalized_socket, presence.inputs["Fac"])
    ensure_link(node_tree, normalized_socket, strong.inputs["Fac"])

    core = dilate_erode(node_tree, f"{base_name} Core", (320.0, y + 80.0), core_distance)
    wide = dilate_erode(node_tree, f"{base_name} Wide", (320.0, y - 80.0), wide_distance)
    ensure_link(node_tree, presence.outputs["Image"], core.inputs[0])
    ensure_link(node_tree, strong.outputs["Image"], wide.inputs[0])

    combine = math_node(node_tree, "MAXIMUM", f"{base_name} Combine", (560.0, y), use_clamp=True)
    ensure_link(node_tree, core.outputs[0], combine.inputs[0])
    ensure_link(node_tree, wide.outputs[0], combine.inputs[1])

    soften = blur_node(node_tree, f"{base_name} Soften", (800.0, y), blur_pixels)
    ensure_link(node_tree, combine.outputs[0], soften.inputs[0])

    mask_mul = math_node(node_tree, "MULTIPLY", f"{base_name} Masked", (1040.0, y), use_clamp=True)
    ensure_link(node_tree, soften.outputs[0], mask_mul.inputs[0])
    ensure_link(node_tree, mask_socket, mask_mul.inputs[1])
    return mask_mul.outputs[0]


def build_trending_kirsch(node_tree: bpy.types.NodeTree, trending_node, existing_node, directory: Path, rendered_paths: list[Path]):
    base_y = -220.0
    tree_mask = id_mask(node_tree, trending_node.outputs["IndexOB"], TREE_ID, "Trending Tree Mask", (-1500.0, base_y))
    tree_rgba = set_alpha(
        node_tree,
        trending_node.outputs["Image"],
        tree_mask.outputs["Alpha"],
        "Trending Tree RGBA",
        (-1260.0, base_y),
        mode="REPLACE_ALPHA",
    )

    depth_filter = new_node(node_tree, "CompositorNodeFilter", "Trending Depth Kirsch", (-1260.0, base_y - 260.0), color=(0.18, 0.16, 0.10))
    depth_filter.filter_type = "KIRSCH"
    ensure_link(node_tree, trending_node.outputs["Depth"], depth_filter.inputs[0])

    normalize = new_node(node_tree, "CompositorNodeNormalize", "Trending Depth Normalize", (-1020.0, base_y - 260.0), color=(0.16, 0.18, 0.20))
    ensure_link(node_tree, depth_filter.outputs[0], normalize.inputs[0])

    thin_alpha = build_edge_width(
        node_tree,
        normalize.outputs[0],
        tree_mask.outputs["Alpha"],
        "Trending Kirsch Thin",
        base_y - 420.0,
        low=0.08,
        high=0.24,
        strong_low=0.20,
        strong_high=0.42,
        core_distance=1,
        wide_distance=2,
        blur_pixels=1,
    )
    regular_alpha = build_edge_width(
        node_tree,
        normalize.outputs[0],
        tree_mask.outputs["Alpha"],
        "Trending Kirsch Regular",
        base_y - 680.0,
        low=0.08,
        high=0.24,
        strong_low=0.18,
        strong_high=0.38,
        core_distance=2,
        wide_distance=4,
        blur_pixels=2,
    )

    edge_rgb = rgb_constant(node_tree, "Edge Color", (-1260.0, base_y - 980.0), EDGE_COLOR)
    thin_rgba = set_alpha(
        node_tree,
        edge_rgb.outputs[0],
        thin_alpha,
        "Trending Thin Edge RGBA",
        (-1020.0, base_y - 540.0),
        mode="REPLACE_ALPHA",
    )
    regular_rgba = set_alpha(
        node_tree,
        edge_rgb.outputs[0],
        regular_alpha,
        "Trending Regular Edge RGBA",
        (-1020.0, base_y - 800.0),
        mode="REPLACE_ALPHA",
    )

    tree_over_base = alpha_over(node_tree, "Trending Trees Over Base", (-760.0, base_y), existing_node.outputs["Image"], tree_rgba.outputs["Image"])
    thin_comp = alpha_over(node_tree, "Trending Thin Composite", (-500.0, base_y - 540.0), tree_over_base.outputs["Image"], thin_rgba.outputs["Image"])
    regular_comp = alpha_over(node_tree, "Trending Regular Composite", (-500.0, base_y - 800.0), tree_over_base.outputs["Image"], regular_rgba.outputs["Image"])

    _, rendered = file_output(node_tree, "Trending Trees Output", (-220.0, base_y), directory, "trending_visible_trees", tree_rgba.outputs["Image"])
    rendered_paths.append(rendered)
    _, rendered = file_output(node_tree, "Trending Thin Edge Output", (-220.0, base_y - 540.0), directory, "trending_kirsch_thin_edges", thin_rgba.outputs["Image"])
    rendered_paths.append(rendered)
    _, rendered = file_output(node_tree, "Trending Thin Composite Output", (20.0, base_y - 540.0), directory, "trending_kirsch_thin_composite", thin_comp.outputs["Image"])
    rendered_paths.append(rendered)
    _, rendered = file_output(node_tree, "Trending Regular Edge Output", (-220.0, base_y - 800.0), directory, "trending_kirsch_regular_edges", regular_rgba.outputs["Image"])
    rendered_paths.append(rendered)
    _, rendered = file_output(node_tree, "Trending Regular Composite Output", (20.0, base_y - 800.0), directory, "trending_kirsch_regular_composite", regular_comp.outputs["Image"])
    rendered_paths.append(rendered)

    return regular_comp.outputs["Image"]


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

    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    ao_dir = ensure_dir(OUTPUT_ROOT / "01_ao")
    trending_dir = ensure_dir(OUTPUT_ROOT / "02_trending_trees")
    rendered_paths: list[Path] = []

    pathway = image_input(node_tree, PATHWAY_EXR, "Pathway EXR", (-1500.0, 860.0))
    priority = image_input(node_tree, PRIORITY_EXR, "Priority EXR", (-1500.0, 620.0))
    trending = image_input(node_tree, TRENDING_EXR, "Trending EXR", (-1500.0, 380.0))
    existing = image_input(node_tree, EXISTING_EXR, "Existing EXR", (-1500.0, 120.0))

    build_ao_branch(node_tree, pathway, "Pathway Tree AO", 860.0, ao_dir, rendered_paths)
    build_ao_branch(node_tree, priority, "Priority Tree AO", 620.0, ao_dir, rendered_paths)
    build_ao_branch(node_tree, trending, "Trending Visible Tree AO", 380.0, ao_dir, rendered_paths)

    final_socket = build_trending_kirsch(node_tree, trending, existing, trending_dir, rendered_paths)

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", (280.0, -980.0), color=(0.12, 0.16, 0.20))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", (280.0, -1220.0), color=(0.12, 0.16, 0.20))
    ensure_link(node_tree, final_socket, composite.inputs[0])
    ensure_link(node_tree, final_socket, viewer.inputs[0])
    return rendered_paths


def save_blend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    log(f"Saved {path}")


def main() -> None:
    scene = bpy.context.scene
    rendered_paths = build_scene(scene)
    save_blend(BLEND_PATH)
    bpy.ops.render.render(write_still=False)
    rename_outputs(rendered_paths)
    log(f"AO dir: {OUTPUT_ROOT / '01_ao'}")
    log(f"Trending dir: {OUTPUT_ROOT / '02_trending_trees'}")


if __name__ == "__main__":
    main()
