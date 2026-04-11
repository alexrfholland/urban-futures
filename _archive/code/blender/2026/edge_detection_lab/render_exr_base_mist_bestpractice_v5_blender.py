from __future__ import annotations

import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"
OUTPUT_DIR = Path(
    os.environ.get(
        "EDGE_LAB_OUTPUT_DIR",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_v5" / "01_base_mist"),
    )
)
BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_BLEND_PATH",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_base_mist_v5.blend"),
    )
)

PATHWAY_EXR = Path(os.environ.get("EDGE_LAB_PATHWAY_EXR", str(EXR_ROOT / "city-pathway_state.exr")))
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)
BASE_IDS = (1, 2)

VARIANT_SPECS = (
    {
        "name": "base_mist_kirsch_thin",
        "filter_type": "KIRSCH",
        "presence_low": 0.06,
        "presence_high": 0.20,
        "strong_low": 0.18,
        "strong_high": 0.34,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
    },
    {
        "name": "base_mist_kirsch_regular",
        "filter_type": "KIRSCH",
        "presence_low": 0.06,
        "presence_high": 0.20,
        "strong_low": 0.16,
        "strong_high": 0.30,
        "core_distance": 2,
        "wide_distance": 4,
        "blur_pixels": 2,
    },
    {
        "name": "base_mist_sobel_clean",
        "filter_type": "SOBEL",
        "presence_low": 0.05,
        "presence_high": 0.16,
        "strong_low": 0.12,
        "strong_high": 0.24,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
    },
)


def log(message: str) -> None:
    print(f"[render_exr_base_mist_bestpractice_v5_blender] {message}")


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


def id_mask_node(node_tree: bpy.types.NodeTree, index_socket, object_id: int, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeIDMask", name, label, location, color=(0.18, 0.18, 0.10))
    node.index = object_id
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


def normalize_node(node_tree: bpy.types.NodeTree, value_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeNormalize", name, label, location, color=(0.16, 0.18, 0.20))
    ensure_link(node_tree, value_socket, node.inputs[0])
    return node


def rgb_to_bw_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeRGBToBW", name, label, location, color=(0.14, 0.16, 0.20))
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


def filter_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float], filter_type: str):
    node = new_node(node_tree, "CompositorNodeFilter", name, label, location, color=(0.18, 0.16, 0.10))
    node.filter_type = filter_type
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


def threshold_ramp(
    node_tree: bpy.types.NodeTree,
    value_socket,
    name: str,
    label: str,
    location: tuple[float, float],
    low: float,
    high: float,
):
    node = new_node(node_tree, "CompositorNodeValToRGB", name, label, location, color=(0.18, 0.14, 0.14))
    node.color_ramp.interpolation = "LINEAR"
    node.color_ramp.elements[0].position = low
    node.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node.color_ramp.elements[1].position = high
    node.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    ensure_link(node_tree, value_socket, node.inputs["Fac"])
    return node


def blur_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], pixels: int):
    node = new_node(node_tree, "CompositorNodeBlur", name, label, location, color=(0.14, 0.16, 0.20))
    node.filter_type = "GAUSS"
    node.use_relative = False
    node.size_x = pixels
    node.size_y = pixels
    return node


def dilate_erode_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], distance: int):
    node = new_node(node_tree, "CompositorNodeDilateErode", name, label, location, color=(0.16, 0.14, 0.20))
    node.mode = "DISTANCE"
    node.distance = distance
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


def alpha_over_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], bottom_socket, top_socket):
    node = new_node(node_tree, "CompositorNodeAlphaOver", name, label, location, color=(0.14, 0.20, 0.16))
    node.premul = 1.0
    ensure_link(node_tree, bottom_socket, node.inputs[1])
    ensure_link(node_tree, top_socket, node.inputs[2])
    return node


def separate_rgba_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeSepRGBA", name, label, location, color=(0.14, 0.18, 0.20))
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


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


def build_base_mask(node_tree: bpy.types.NodeTree, pathway_node, y: float):
    ground = id_mask_node(node_tree, pathway_node.outputs["IndexOB"], BASE_IDS[0], "mask_base_ground", "mask_base_ground", (-1240.0, y + 120.0))
    buildings = id_mask_node(node_tree, pathway_node.outputs["IndexOB"], BASE_IDS[1], "mask_base_buildings", "mask_base_buildings", (-1240.0, y - 60.0))
    combined = math_node(node_tree, "MAXIMUM", "mask_visible-base", "mask_visible-base", (-1000.0, y + 30.0))
    ensure_link(node_tree, ground.outputs["Alpha"], combined.inputs[0])
    ensure_link(node_tree, buildings.outputs["Alpha"], combined.inputs[1])
    return combined.outputs["Value"]


def build_prep_stage(scene: bpy.types.Scene) -> list[Path]:
    configure_scene(scene)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    pathway = image_node(node_tree, PATHWAY_EXR, "EXR Pathway", "EXR Pathway", (-1500.0, 220.0))
    base_mask = build_base_mask(node_tree, pathway, 120.0)

    base_visible = set_alpha_node(
        node_tree,
        pathway.outputs["Image"],
        base_mask,
        "base_visible_layer",
        "base_visible_layer",
        (-760.0, 220.0),
    )
    mist_normalized = normalize_node(
        node_tree,
        pathway.outputs["Mist"],
        "base_mist_normalized",
        "base_mist_normalized",
        (-760.0, -60.0),
    )
    mist_visible = set_alpha_node(
        node_tree,
        mist_normalized.outputs[0],
        base_mask,
        "base_mist_normalized_visible",
        "base_mist_normalized_visible",
        (-500.0, -60.0),
    )

    rendered_paths = [
        file_output_node(
            node_tree,
            base_visible.outputs["Image"],
            "Output Base Visible Layer",
            "Output Base Visible Layer",
            (-220.0, 220.0),
            "base_visible_layer",
        ),
        file_output_node(
            node_tree,
            mist_visible.outputs["Image"],
            "Output Base Mist Visible",
            "Output Base Mist Visible",
            (-220.0, -60.0),
            "base_mist_normalized_visible",
        ),
    ]

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (120.0, 220.0))
    ensure_link(node_tree, base_visible.outputs["Image"], composite.inputs[0])
    return rendered_paths


def build_edge_width(node_tree: bpy.types.NodeTree, normalized_socket, mask_socket, prefix: str, spec: dict, y: float):
    presence = threshold_ramp(
        node_tree,
        normalized_socket,
        f"{prefix}_presence",
        f"{prefix}_presence",
        (-180.0, y + 80.0),
        spec["presence_low"],
        spec["presence_high"],
    )
    strong = threshold_ramp(
        node_tree,
        normalized_socket,
        f"{prefix}_strong",
        f"{prefix}_strong",
        (-180.0, y - 80.0),
        spec["strong_low"],
        spec["strong_high"],
    )
    core = dilate_erode_node(
        node_tree,
        f"{prefix}_core",
        f"{prefix}_core",
        (80.0, y + 80.0),
        spec["core_distance"],
    )
    wide = dilate_erode_node(
        node_tree,
        f"{prefix}_wide",
        f"{prefix}_wide",
        (80.0, y - 80.0),
        spec["wide_distance"],
    )
    ensure_link(node_tree, presence.outputs["Image"], core.inputs[0])
    ensure_link(node_tree, strong.outputs["Image"], wide.inputs[0])

    combine = math_node(node_tree, "MAXIMUM", f"{prefix}_combined", f"{prefix}_combined", (320.0, y))
    ensure_link(node_tree, core.outputs[0], combine.inputs[0])
    ensure_link(node_tree, wide.outputs[0], combine.inputs[1])

    soften = blur_node(
        node_tree,
        f"{prefix}_soften",
        f"{prefix}_soften",
        (560.0, y),
        spec["blur_pixels"],
    )
    ensure_link(node_tree, combine.outputs[0], soften.inputs[0])

    masked = math_node(node_tree, "MULTIPLY", f"{prefix}_masked", f"{prefix}_masked", (800.0, y))
    ensure_link(node_tree, soften.outputs[0], masked.inputs[0])
    ensure_link(node_tree, mask_socket, masked.inputs[1])
    return masked.outputs["Value"]


def build_variant_stage(scene: bpy.types.Scene, spec: dict, base_visible_path: Path, mist_visible_path: Path) -> Path:
    configure_scene(scene)
    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    base_visible = image_node(node_tree, base_visible_path, "base_visible_png", "base_visible_png", (-1400.0, 220.0))
    mist_visible = image_node(node_tree, mist_visible_path, "base_mist_visible_png", "base_mist_visible_png", (-1400.0, -60.0))
    mist_bw = rgb_to_bw_node(node_tree, mist_visible.outputs["Image"], "base_mist_bw", "base_mist_bw", (-1140.0, -60.0))
    mist_rgba = separate_rgba_node(node_tree, mist_visible.outputs["Image"], "base_mist_rgba", "base_mist_rgba", (-1140.0, -260.0))

    edge_filter = filter_node(
        node_tree,
        mist_bw.outputs["Val"],
        f"{spec['name']}_filter",
        f"{spec['name']}_filter",
        (-900.0, -60.0),
        spec["filter_type"],
    )
    edge_normalized = normalize_node(
        node_tree,
        edge_filter.outputs[0],
        f"{spec['name']}_normalized",
        f"{spec['name']}_normalized",
        (-660.0, -60.0),
    )
    edge_alpha = build_edge_width(node_tree, edge_normalized.outputs[0], mist_rgba.outputs["A"], spec["name"], spec, -60.0)

    edge_rgb = solid_color_image(
        node_tree,
        base_visible.outputs["Image"],
        f"{spec['name']}_edge_rgb",
        f"{spec['name']}_edge_rgb",
        (1060.0, -180.0),
        EDGE_COLOR_LINEAR,
    )
    edge_rgba = set_alpha_node(
        node_tree,
        edge_rgb,
        edge_alpha,
        f"{spec['name']}_edge_rgba",
        f"{spec['name']}_edge_rgba",
        (1280.0, -60.0),
    )
    composite_image = alpha_over_node(
        node_tree,
        f"{spec['name']}_composite",
        f"{spec['name']}_composite",
        (1520.0, 80.0),
        base_visible.outputs["Image"],
        edge_rgba.outputs["Image"],
    )

    edge_only_rendered = file_output_node(
        node_tree,
        edge_rgba.outputs["Image"],
        f"Output {spec['name']} Edge",
        f"Output {spec['name']} Edge",
        (1520.0, -180.0),
        f"{spec['name']}_edges",
    )
    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (1760.0, 80.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (1760.0, -80.0))
    ensure_link(node_tree, composite_image.outputs["Image"], composite.inputs[0])
    ensure_link(node_tree, composite_image.outputs["Image"], viewer.inputs[0])

    scene.render.filepath = str(OUTPUT_DIR / f"{spec['name']}_composite.png")
    bpy.ops.render.render(write_still=True)
    rename_output(edge_only_rendered)
    composite_path = OUTPUT_DIR / f"{spec['name']}_composite.png"
    log(f"Rendered {composite_path.name}")
    return composite_path


def render_prep(scene: bpy.types.Scene) -> tuple[Path, Path]:
    rendered_paths = build_prep_stage(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    log(f"Saved {BLEND_PATH}")
    bpy.ops.render.render(write_still=False)
    final_paths = [rename_output(path) for path in rendered_paths]
    return final_paths[0], final_paths[1]


def main() -> None:
    scene = bpy.context.scene
    base_visible_path, mist_visible_path = render_prep(scene)
    for spec in VARIANT_SPECS:
        build_variant_stage(scene, spec, base_visible_path, mist_visible_path)


if __name__ == "__main__":
    main()
