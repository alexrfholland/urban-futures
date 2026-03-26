from __future__ import annotations

from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"
OUTPUT_DIR = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_base_lines_v4_tuned"
BLEND_PATH = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_base_lines_v4_tuned.blend"

BASE_EXR = EXR_ROOT / "city-existing_condition.exr"
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)
GROUND_ID = 1
BUILDING_ID = 2

VARIANT_SPECS = (
    {
        "name": "base_depth_windowed_balanced_refined",
        "ground_filter": "SOBEL",
        "building_windows": (
            {"min_depth": 88.0, "max_depth": 152.0, "filter": "KIRSCH", "low": 0.018, "high": 0.064, "core": 1, "wide": 1, "blur": 1},
            {"min_depth": 132.0, "max_depth": 245.0, "filter": "KIRSCH", "low": 0.030, "high": 0.082, "core": 1, "wide": 1, "blur": 1},
        ),
        "ground_window": {"min_depth": 88.0, "max_depth": 235.0, "low": 0.050, "high": 0.128, "core": 0, "wide": 0, "blur": 1},
    },
    {
        "name": "base_depth_windowed_internal_refined",
        "ground_filter": "SOBEL",
        "building_windows": (
            {"min_depth": 88.0, "max_depth": 145.0, "filter": "KIRSCH", "low": 0.014, "high": 0.052, "core": 1, "wide": 1, "blur": 1},
            {"min_depth": 125.0, "max_depth": 215.0, "filter": "KIRSCH", "low": 0.024, "high": 0.070, "core": 1, "wide": 1, "blur": 1},
            {"min_depth": 188.0, "max_depth": 320.0, "filter": "SOBEL", "low": 0.036, "high": 0.096, "core": 0, "wide": 1, "blur": 1},
        ),
        "ground_window": {"min_depth": 88.0, "max_depth": 255.0, "low": 0.056, "high": 0.140, "core": 0, "wide": 0, "blur": 1},
    },
    {
        "name": "base_depth_windowed_internal_dense",
        "ground_filter": "SOBEL",
        "building_windows": (
            {"min_depth": 88.0, "max_depth": 145.0, "filter": "KIRSCH", "low": 0.012, "high": 0.046, "core": 1, "wide": 1, "blur": 1},
            {"min_depth": 122.0, "max_depth": 205.0, "filter": "KIRSCH", "low": 0.022, "high": 0.062, "core": 1, "wide": 1, "blur": 1},
            {"min_depth": 185.0, "max_depth": 305.0, "filter": "KIRSCH", "low": 0.034, "high": 0.090, "core": 0, "wide": 1, "blur": 1},
        ),
        "ground_window": {"min_depth": 88.0, "max_depth": 245.0, "low": 0.058, "high": 0.146, "core": 0, "wide": 0, "blur": 1},
    },
    {
        "name": "base_depth_windowed_balanced_dense",
        "ground_filter": "SOBEL",
        "building_windows": (
            {"min_depth": 88.0, "max_depth": 154.0, "filter": "KIRSCH", "low": 0.016, "high": 0.058, "core": 1, "wide": 1, "blur": 1},
            {"min_depth": 132.0, "max_depth": 255.0, "filter": "KIRSCH", "low": 0.026, "high": 0.074, "core": 1, "wide": 1, "blur": 1},
        ),
        "ground_window": {"min_depth": 88.0, "max_depth": 238.0, "low": 0.046, "high": 0.118, "core": 0, "wide": 0, "blur": 1},
    },
)


def log(message: str) -> None:
    print(f"[render_exr_base_lines_v4_tuned_blender] {message}")


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


def rgb_to_bw_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeRGBToBW", name, label, location, color=(0.14, 0.16, 0.20))
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


def normalize_node(node_tree: bpy.types.NodeTree, value_socket, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeNormalize", name, label, location, color=(0.16, 0.18, 0.20))
    ensure_link(node_tree, value_socket, node.inputs[0])
    return node


def filter_node(node_tree: bpy.types.NodeTree, image_socket, name: str, label: str, location: tuple[float, float], filter_type: str):
    node = new_node(node_tree, "CompositorNodeFilter", name, label, location, color=(0.18, 0.16, 0.10))
    node.filter_type = filter_type
    ensure_link(node_tree, image_socket, node.inputs["Image"])
    return node


def threshold_ramp(node_tree: bpy.types.NodeTree, value_socket, name: str, label: str, location: tuple[float, float], low: float, high: float):
    node = new_node(node_tree, "CompositorNodeValToRGB", name, label, location, color=(0.18, 0.14, 0.14))
    node.color_ramp.interpolation = "LINEAR"
    node.color_ramp.elements[0].position = low
    node.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node.color_ramp.elements[1].position = high
    node.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    ensure_link(node_tree, value_socket, node.inputs["Fac"])
    return node


def dilate_erode_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], distance: int):
    node = new_node(node_tree, "CompositorNodeDilateErode", name, label, location, color=(0.16, 0.14, 0.20))
    node.mode = "DISTANCE"
    node.distance = distance
    return node


def blur_node(node_tree: bpy.types.NodeTree, name: str, label: str, location: tuple[float, float], pixels: int):
    node = new_node(node_tree, "CompositorNodeBlur", name, label, location, color=(0.14, 0.16, 0.20))
    node.filter_type = "GAUSS"
    node.use_relative = False
    node.size_x = pixels
    node.size_y = pixels
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


def masked_scalar_to_image(node_tree: bpy.types.NodeTree, scalar_socket, mask_socket, name: str, label: str, location: tuple[float, float]):
    multiply = math_node(node_tree, "MULTIPLY", f"{name}_masked", f"{label}_masked", location, clamp=False)
    ensure_link(node_tree, scalar_socket, multiply.inputs[0])
    ensure_link(node_tree, mask_socket, multiply.inputs[1])
    rgb = new_node(node_tree, "CompositorNodeCombRGBA", f"{name}_rgba", f"{label}_rgba", (location[0] + 240.0, location[1]), color=(0.14, 0.18, 0.20))
    ensure_link(node_tree, multiply.outputs["Value"], rgb.inputs["R"])
    ensure_link(node_tree, multiply.outputs["Value"], rgb.inputs["G"])
    ensure_link(node_tree, multiply.outputs["Value"], rgb.inputs["B"])
    rgb.inputs["A"].default_value = 1.0
    return rgb.outputs["Image"]


def windowed_scalar_to_image(
    node_tree: bpy.types.NodeTree,
    scalar_socket,
    mask_socket,
    name: str,
    label: str,
    location: tuple[float, float],
    min_depth: float,
    max_depth: float,
):
    shifted = math_node(node_tree, "SUBTRACT", f"{name}_shift", f"{label}_shift", location, clamp=False)
    ensure_link(node_tree, scalar_socket, shifted.inputs[0])
    shifted.inputs[1].default_value = min_depth

    scaled = math_node(node_tree, "DIVIDE", f"{name}_scale", f"{label}_scale", (location[0] + 220.0, location[1]), clamp=True)
    ensure_link(node_tree, shifted.outputs["Value"], scaled.inputs[0])
    scaled.inputs[1].default_value = max(max_depth - min_depth, 0.001)

    masked = math_node(node_tree, "MULTIPLY", f"{name}_masked", f"{label}_masked", (location[0] + 440.0, location[1]), clamp=True)
    ensure_link(node_tree, scaled.outputs["Value"], masked.inputs[0])
    ensure_link(node_tree, mask_socket, masked.inputs[1])

    rgb = new_node(node_tree, "CompositorNodeCombRGBA", f"{name}_rgba", f"{label}_rgba", (location[0] + 680.0, location[1]), color=(0.14, 0.18, 0.20))
    ensure_link(node_tree, masked.outputs["Value"], rgb.inputs["R"])
    ensure_link(node_tree, masked.outputs["Value"], rgb.inputs["G"])
    ensure_link(node_tree, masked.outputs["Value"], rgb.inputs["B"])
    rgb.inputs["A"].default_value = 1.0
    return rgb.outputs["Image"]


def build_filtered_alpha(
    node_tree: bpy.types.NodeTree,
    source_socket,
    mask_socket,
    prefix: str,
    y: float,
    filter_type: str,
    low: float,
    high: float,
    core_distance: int,
    wide_distance: int,
    blur_pixels: int,
):
    source_bw = rgb_to_bw_node(node_tree, source_socket, f"{prefix}_bw", f"{prefix}_bw", (-760.0, y))
    filtered = filter_node(node_tree, source_bw.outputs["Val"], f"{prefix}_filter", f"{prefix}_filter", (-520.0, y), filter_type)
    normalized = normalize_node(node_tree, filtered.outputs["Image"], f"{prefix}_norm", f"{prefix}_norm", (-260.0, y))

    presence = threshold_ramp(node_tree, normalized.outputs[0], f"{prefix}_presence", f"{prefix}_presence", (0.0, y + 70.0), low, high)
    strong = threshold_ramp(node_tree, normalized.outputs[0], f"{prefix}_strong", f"{prefix}_strong", (0.0, y - 70.0), low * 0.8, high * 1.35)

    core = dilate_erode_node(node_tree, f"{prefix}_core", f"{prefix}_core", (260.0, y + 70.0), core_distance)
    wide = dilate_erode_node(node_tree, f"{prefix}_wide", f"{prefix}_wide", (260.0, y - 70.0), wide_distance)
    ensure_link(node_tree, presence.outputs["Image"], core.inputs[0])
    ensure_link(node_tree, strong.outputs["Image"], wide.inputs[0])

    merged = math_node(node_tree, "MAXIMUM", f"{prefix}_merged", f"{prefix}_merged", (520.0, y), clamp=True)
    ensure_link(node_tree, core.outputs[0], merged.inputs[0])
    ensure_link(node_tree, wide.outputs[0], merged.inputs[1])

    softened = blur_node(node_tree, f"{prefix}_blur", f"{prefix}_blur", (780.0, y), blur_pixels)
    ensure_link(node_tree, merged.outputs[0], softened.inputs[0])

    masked = math_node(node_tree, "MULTIPLY", f"{prefix}_masked_alpha", f"{prefix}_masked_alpha", (1040.0, y), clamp=True)
    ensure_link(node_tree, softened.outputs[0], masked.inputs[0])
    ensure_link(node_tree, mask_socket, masked.inputs[1])
    return masked.outputs["Value"]


def build_windowed_alpha(
    node_tree: bpy.types.NodeTree,
    depth_socket,
    mask_socket,
    prefix: str,
    y: float,
    spec: dict,
):
    window_image = windowed_scalar_to_image(
        node_tree,
        depth_socket,
        mask_socket,
        f"{prefix}_window",
        f"{prefix}_window",
        (-1260.0, y),
        spec["min_depth"],
        spec["max_depth"],
    )
    contrast = threshold_ramp(
        node_tree,
        rgb_to_bw_node(node_tree, window_image, f"{prefix}_window_bw", f"{prefix}_window_bw", (-320.0, y)).outputs["Val"],
        f"{prefix}_contrast",
        f"{prefix}_contrast",
        (-80.0, y),
        0.02,
        0.98,
    )
    return build_filtered_alpha(
        node_tree,
        contrast.outputs["Image"],
        mask_socket,
        prefix,
        y,
        spec["filter"],
        spec["low"],
        spec["high"],
        spec["core"],
        spec["wide"],
        spec["blur"],
    )


def build_multiwindow_building_alpha(
    node_tree: bpy.types.NodeTree,
    depth_socket,
    building_mask_socket,
    prefix: str,
    y: float,
    window_specs: tuple[dict, ...],
):
    combined = None
    for idx, window_spec in enumerate(window_specs):
        branch = build_windowed_alpha(
            node_tree,
            depth_socket,
            building_mask_socket,
            f"{prefix}_w{idx + 1}",
            y - idx * 180.0,
            window_spec,
        )
        if combined is None:
            combined = branch
        else:
            max_node = math_node(node_tree, "MAXIMUM", f"{prefix}_max_{idx + 1}", f"{prefix}_max_{idx + 1}", (1320.0, y - idx * 90.0), clamp=True)
            ensure_link(node_tree, combined, max_node.inputs[0])
            ensure_link(node_tree, branch, max_node.inputs[1])
            combined = max_node.outputs["Value"]
    return combined


def build_scene(scene: bpy.types.Scene) -> list[Path]:
    configure_scene(scene)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    exr = image_node(node_tree, BASE_EXR, "EXR Existing Condition", "EXR Existing Condition", (-1800.0, 260.0))
    ground_mask = id_mask_node(node_tree, exr.outputs["IndexOB"], GROUND_ID, "mask_ground", "mask_ground", (-1500.0, 420.0))
    building_mask = id_mask_node(node_tree, exr.outputs["IndexOB"], BUILDING_ID, "mask_buildings", "mask_buildings", (-1500.0, 120.0))
    base_mask = math_node(node_tree, "MAXIMUM", "mask_base", "mask_base", (-1260.0, 260.0))
    ensure_link(node_tree, ground_mask.outputs["Alpha"], base_mask.inputs[0])
    ensure_link(node_tree, building_mask.outputs["Alpha"], base_mask.inputs[1])

    depth_base_image = masked_scalar_to_image(node_tree, exr.outputs["Depth"], base_mask.outputs["Value"], "base_depth_image", "base_depth_image", (-1500.0, -160.0))

    rendered_paths: list[Path] = []
    composite_source = None

    for index, spec in enumerate(VARIANT_SPECS):
        y = -220.0 - index * 520.0

        building_alpha = build_multiwindow_building_alpha(
            node_tree,
            exr.outputs["Depth"],
            building_mask.outputs["Alpha"],
            f"{spec['name']}_building",
            y,
            spec["building_windows"],
        )
        ground_alpha = build_windowed_alpha(
            node_tree,
            exr.outputs["Depth"],
            ground_mask.outputs["Alpha"],
            f"{spec['name']}_ground",
            y - 220.0,
            {
                "min_depth": spec["ground_window"]["min_depth"],
                "max_depth": spec["ground_window"]["max_depth"],
                "filter": spec["ground_filter"],
                "low": spec["ground_window"]["low"],
                "high": spec["ground_window"]["high"],
                "core": spec["ground_window"]["core"],
                "wide": spec["ground_window"]["wide"],
                "blur": spec["ground_window"]["blur"],
            },
        )

        combined_alpha = math_node(node_tree, "MAXIMUM", f"{spec['name']}_alpha", f"{spec['name']}_alpha", (1620.0, y - 110.0), clamp=True)
        ensure_link(node_tree, building_alpha, combined_alpha.inputs[0])
        ensure_link(node_tree, ground_alpha, combined_alpha.inputs[1])

        edge_color = solid_color_image(
            node_tree,
            depth_base_image,
            f"{spec['name']}_edge_rgb",
            f"{spec['name']}_edge_rgb",
            (1860.0, y - 110.0),
            EDGE_COLOR_LINEAR,
        )
        edge_rgba = set_alpha_node(
            node_tree,
            edge_color,
            combined_alpha.outputs["Value"],
            f"{spec['name']}_edge_rgba",
            f"{spec['name']}_edge_rgba",
            (2100.0, y - 110.0),
        )
        rendered_paths.append(
            file_output_node(
                node_tree,
                edge_rgba.outputs["Image"],
                f"Output {spec['name']}",
                f"Output {spec['name']}",
                (2340.0, y - 110.0),
                spec["name"],
            )
        )
        if composite_source is None:
            composite_source = edge_rgba.outputs["Image"]

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (2600.0, -220.0))
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
