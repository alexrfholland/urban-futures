from __future__ import annotations

from array import array
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = Path(
    os.environ.get(
        "EDGE_LAB_EXR_ROOT",
        str(REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"),
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "EDGE_LAB_OUTPUT_DIR",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_0000_arboreal_mist_v2_screenlift"),
    )
)
BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_BLEND_PATH",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_0000_arboreal_mist_v2_screenlift.blend"),
    )
)

PATHWAY_EXR = Path(os.environ.get("EDGE_LAB_PATHWAY_EXR", str(EXR_ROOT / "pathway_state.exr")))
PRIORITY_EXR = Path(os.environ.get("EDGE_LAB_PRIORITY_EXR", str(EXR_ROOT / "priority.exr")))
TRENDING_EXR = Path(os.environ.get("EDGE_LAB_TRENDING_EXR", str(EXR_ROOT / "trending_state.exr")))

TREE_ID = 3
EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)
VARIANT_PRESET = os.environ.get("EDGE_LAB_VARIANT_PRESET", "screenlift").strip().lower()
RENDER_MODE = os.environ.get("EDGE_LAB_RENDER_MODE", "full").strip().lower()
WORKFLOW_ID = os.environ.get(
    "EDGE_LAB_WORKFLOW_ID",
    (
        "blender_exr_arboreal_mist_v2"
        if VARIANT_PRESET == "screenlift"
        else "blender_exr_arboreal_mist_v2_extrathin"
        if VARIANT_PRESET == "extrathin"
        else "blender_exr_arboreal_mist_kirschsizes_v1"
        if VARIANT_PRESET == "kirschsizes"
        else "blender_exr_arboreal_mist_kirschremap_v1"
        if VARIANT_PRESET == "kirschremap"
        else "blender_exr_arboreal_mist_v3"
        if VARIANT_PRESET == "localratio"
        else "blender_exr_arboreal_mist_v4"
    ),
)
WORKFLOW_NOTES = os.environ.get(
    "EDGE_LAB_WORKFLOW_NOTES",
    "Adds a bottom-weighted screen-space gain mask before thresholding to reduce vertical edge attenuation in mist-derived arboreal outlines."
    if VARIANT_PRESET == "screenlift"
    else "Reuses the screen-lift arboreal mist workflow but pushes thresholds and width shaping toward a finer extra-thin line."
    if VARIANT_PRESET == "extrathin"
    else "Exports only the Kirsch mist edge PNGs in three widths: thin, fine, and extra-thin."
    if VARIANT_PRESET == "kirschsizes"
    else "Exports only the isolated remapped Kirsch mist edge PNGs as a separate experimental branch."
    if VARIANT_PRESET == "kirschremap"
    else "Adds local mist normalization before edge detection, with one hybrid variant that also uses a mild bottom-weighted screen-space gain."
    if VARIANT_PRESET == "localratio"
    else "Adds local normalization to the filtered edge-response field before width shaping, with one mild hybrid screen-lift variant for comparison.",
)
PREP_ROOT = Path(os.environ.get("EDGE_LAB_PREP_OUTPUT_DIR", str(OUTPUT_ROOT / "_prep")))


def exr_image_socket(node: bpy.types.Node) -> bpy.types.NodeSocket:
    for socket_name in ("Image", "Combined"):
        socket = node.outputs.get(socket_name)
        if socket is not None:
            return socket
    raise KeyError(f'EXR node "{node.name}" has neither "Image" nor "Combined" output')


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
    return 3840, 2160


RENDER_WIDTH = int(os.environ.get("EDGE_LAB_RESOLUTION_X", detect_resolution_from_exr(PATHWAY_EXR)[0]))
RENDER_HEIGHT = int(os.environ.get("EDGE_LAB_RESOLUTION_Y", detect_resolution_from_exr(PATHWAY_EXR)[1]))

SCREENLIFT_VARIANTS = (
    {
        "name": "mist_kirsch_thin_screenlift",
        "filter_type": "KIRSCH",
        "presence_low": 0.07,
        "presence_high": 0.22,
        "strong_low": 0.18,
        "strong_high": 0.34,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
        "screen_gain_bottom": 1.42,
        "screen_gain_power": 1.8,
    },
    {
        "name": "mist_kirsch_balanced_screenlift",
        "filter_type": "KIRSCH",
        "presence_low": 0.06,
        "presence_high": 0.20,
        "strong_low": 0.16,
        "strong_high": 0.30,
        "core_distance": 1,
        "wide_distance": 4,
        "blur_pixels": 2,
        "screen_gain_bottom": 1.55,
        "screen_gain_power": 2.0,
    },
    {
        "name": "mist_sobel_clean_screenlift",
        "filter_type": "SOBEL",
        "presence_low": 0.05,
        "presence_high": 0.16,
        "strong_low": 0.12,
        "strong_high": 0.24,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
        "screen_gain_bottom": 1.35,
        "screen_gain_power": 1.8,
    },
)

EXTRATHIN_VARIANTS = (
    {
        "name": "mist_kirsch_extra_thin_screenlift",
        "filter_type": "KIRSCH",
        "presence_low": 0.10,
        "presence_high": 0.28,
        "strong_low": 0.24,
        "strong_high": 0.42,
        "core_distance": 0,
        "wide_distance": 1,
        "blur_pixels": 1,
        "screen_gain_bottom": 1.34,
        "screen_gain_power": 1.7,
    },
)

KIRSCHSIZE_VARIANTS = (
    {
        "name": "mist_kirsch_thin",
        "filter_type": "KIRSCH",
        "presence_low": 0.07,
        "presence_high": 0.22,
        "strong_low": 0.18,
        "strong_high": 0.34,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
        "screen_gain_bottom": 1.42,
        "screen_gain_power": 1.8,
    },
    {
        "name": "mist_kirsch_fine",
        "filter_type": "KIRSCH",
        "presence_low": 0.085,
        "presence_high": 0.25,
        "strong_low": 0.21,
        "strong_high": 0.38,
        "core_distance": 1,
        "wide_distance": 1,
        "blur_pixels": 1,
        "screen_gain_bottom": 1.38,
        "screen_gain_power": 1.75,
    },
    {
        "name": "mist_kirsch_extra_thin",
        "filter_type": "KIRSCH",
        "presence_low": 0.10,
        "presence_high": 0.28,
        "strong_low": 0.24,
        "strong_high": 0.42,
        "core_distance": 0,
        "wide_distance": 1,
        "blur_pixels": 1,
        "screen_gain_bottom": 1.34,
        "screen_gain_power": 1.7,
    },
)

REMAP_VARIANTS = (
    {
        "name": "mist_kirsch_remapped",
        "filter_type": "KIRSCH",
        "edge_remap_low": 0.08,
        "edge_remap_high": 0.55,
        "presence_low": 0.10,
        "presence_high": 0.28,
        "strong_low": 0.24,
        "strong_high": 0.42,
        "core_distance": 0,
        "wide_distance": 0,
        "blur_pixels": 0,
        "screen_gain_bottom": 1.12,
        "screen_gain_power": 1.45,
        "alpha_gain": 0.45,
    },
)

LOCALRATIO_VARIANTS = (
    {
        "name": "mist_kirsch_thin_localratio",
        "filter_type": "KIRSCH",
        "presence_low": 0.06,
        "presence_high": 0.20,
        "strong_low": 0.16,
        "strong_high": 0.30,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
        "local_blur_pixels": 180,
        "local_floor": 0.10,
    },
    {
        "name": "mist_kirsch_balanced_localratio",
        "filter_type": "KIRSCH",
        "presence_low": 0.05,
        "presence_high": 0.18,
        "strong_low": 0.14,
        "strong_high": 0.26,
        "core_distance": 1,
        "wide_distance": 3,
        "blur_pixels": 1,
        "local_blur_pixels": 220,
        "local_floor": 0.10,
    },
    {
        "name": "mist_kirsch_thin_localratio_screenlift",
        "filter_type": "KIRSCH",
        "presence_low": 0.06,
        "presence_high": 0.20,
        "strong_low": 0.16,
        "strong_high": 0.30,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
        "local_blur_pixels": 180,
        "local_floor": 0.10,
        "screen_gain_bottom": 1.18,
        "screen_gain_power": 1.8,
    },
    {
        "name": "mist_sobel_clean_localratio",
        "filter_type": "SOBEL",
        "presence_low": 0.04,
        "presence_high": 0.14,
        "strong_low": 0.10,
        "strong_high": 0.20,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
        "local_blur_pixels": 200,
        "local_floor": 0.12,
    },
)

EDGELOCALRATIO_VARIANTS = (
    {
        "name": "mist_kirsch_thin_edgelocalratio",
        "filter_type": "KIRSCH",
        "presence_low": 0.06,
        "presence_high": 0.20,
        "strong_low": 0.16,
        "strong_high": 0.30,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
        "edge_local_blur_pixels": 220,
        "edge_local_floor": 0.12,
    },
    {
        "name": "mist_kirsch_balanced_edgelocalratio",
        "filter_type": "KIRSCH",
        "presence_low": 0.05,
        "presence_high": 0.18,
        "strong_low": 0.14,
        "strong_high": 0.26,
        "core_distance": 1,
        "wide_distance": 3,
        "blur_pixels": 1,
        "edge_local_blur_pixels": 260,
        "edge_local_floor": 0.12,
    },
    {
        "name": "mist_kirsch_thin_edgelocalratio_screenlift",
        "filter_type": "KIRSCH",
        "presence_low": 0.06,
        "presence_high": 0.20,
        "strong_low": 0.16,
        "strong_high": 0.30,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
        "edge_local_blur_pixels": 220,
        "edge_local_floor": 0.12,
        "screen_gain_bottom": 1.08,
        "screen_gain_power": 1.6,
    },
    {
        "name": "mist_sobel_clean_edgelocalratio",
        "filter_type": "SOBEL",
        "presence_low": 0.04,
        "presence_high": 0.14,
        "strong_low": 0.10,
        "strong_high": 0.20,
        "core_distance": 1,
        "wide_distance": 2,
        "blur_pixels": 1,
        "edge_local_blur_pixels": 240,
        "edge_local_floor": 0.14,
    },
)

if VARIANT_PRESET == "screenlift":
    VARIANT_SPECS = SCREENLIFT_VARIANTS
elif VARIANT_PRESET == "extrathin":
    VARIANT_SPECS = EXTRATHIN_VARIANTS
elif VARIANT_PRESET == "kirschsizes":
    VARIANT_SPECS = KIRSCHSIZE_VARIANTS
elif VARIANT_PRESET == "kirschremap":
    VARIANT_SPECS = REMAP_VARIANTS
elif VARIANT_PRESET == "localratio":
    VARIANT_SPECS = LOCALRATIO_VARIANTS
else:
    VARIANT_SPECS = EDGELOCALRATIO_VARIANTS


def log(message: str) -> None:
    print(f"[render_exr_arboreal_mist_variants_v2_blender] {message}")


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


def frame_node(
    node_tree: bpy.types.NodeTree,
    name: str,
    label: str,
    location: tuple[float, float],
    color: tuple[float, float, float],
):
    frame = node_tree.nodes.get(name)
    if frame is None:
        frame = node_tree.nodes.new("NodeFrame")
        frame.name = name
    frame.label = label
    frame.location = location
    frame.label_size = 18
    frame.use_custom_color = True
    frame.color = color
    frame.shrink = False
    return frame


def parent_nodes(frame: bpy.types.Node, *nodes: bpy.types.Node) -> None:
    for node in nodes:
        if node is not None:
            node.parent = frame


def image_node(node_tree: bpy.types.NodeTree, path: Path, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeImage", name, label, location, color=(0.12, 0.18, 0.10))
    node.image = bpy.data.images.load(str(path), check_existing=True)
    return node


def datablock_image_node(node_tree: bpy.types.NodeTree, image: bpy.types.Image, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeImage", name, label, location, color=(0.12, 0.18, 0.10))
    node.image = image
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


def value_node(node_tree: bpy.types.NodeTree, value: float, name: str, label: str, location: tuple[float, float]):
    node = new_node(node_tree, "CompositorNodeValue", name, label, location, color=(0.18, 0.12, 0.20))
    node.outputs[0].default_value = value
    return node


def remap_value_socket(
    node_tree: bpy.types.NodeTree,
    value_socket,
    prefix: str,
    location: tuple[float, float],
    low: float,
    high: float,
):
    subtract = math_node(
        node_tree,
        "SUBTRACT",
        f"{prefix}_remap_subtract",
        f"{prefix}_remap_subtract",
        location,
        clamp=False,
    )
    ensure_link(node_tree, value_socket, subtract.inputs[0])
    subtract.inputs[1].default_value = low

    divide = math_node(
        node_tree,
        "DIVIDE",
        f"{prefix}_remap_divide",
        f"{prefix}_remap_divide",
        (location[0] + 240.0, location[1]),
        clamp=True,
    )
    ensure_link(node_tree, subtract.outputs["Value"], divide.inputs[0])
    divide.inputs[1].default_value = max(high - low, 1e-6)
    return divide.outputs["Value"]


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


def file_output_node(node_tree: bpy.types.NodeTree, image_socket, output_dir: Path, name: str, label: str, location: tuple[float, float], stem: str) -> Path:
    node = new_node(node_tree, "CompositorNodeOutputFile", name, label, location, color=(0.12, 0.20, 0.14))
    node.base_path = str(ensure_dir(output_dir))
    node.format.file_format = "PNG"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "8"
    node.file_slots[0].path = f"{stem}_"
    ensure_link(node_tree, image_socket, node.inputs[0])
    return output_dir / f"{stem}_0001.png"


def rename_output(rendered_path: Path) -> Path:
    final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
    if rendered_path.exists():
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")
    return final_path


def resolve_existing_render_path(path: Path) -> Path:
    if path.exists():
        return path
    numbered = path.with_name(path.stem + "_0001" + path.suffix)
    if numbered.exists():
        return numbered
    for _ in range(20):
        if path.exists():
            return path
        if numbered.exists():
            return numbered
        time.sleep(0.1)
    return path


def render_scene(scene: bpy.types.Scene, *, write_still: bool = False) -> None:
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=write_still, scene=scene.name)


def configure_scene(scene: bpy.types.Scene) -> None:
    scene.use_nodes = True
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
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


def vertical_gain_image(name: str, bottom_gain: float, power: float) -> bpy.types.Image:
    existing = bpy.data.images.get(name)
    if existing is not None and (existing.size[0] != RENDER_WIDTH or existing.size[1] != RENDER_HEIGHT):
        bpy.data.images.remove(existing)
        existing = None
    image = existing or bpy.data.images.new(name=name, width=RENDER_WIDTH, height=RENDER_HEIGHT, alpha=True, float_buffer=True)
    pixels = array("f")
    width = RENDER_WIDTH
    height = RENDER_HEIGHT
    for y in range(height):
        fac = 1.0 - (y / (height - 1))
        gain = 1.0 + (bottom_gain - 1.0) * (fac**power)
        pixels.extend([gain, gain, gain, 1.0] * width)
    image.pixels.foreach_set(pixels)
    image.update()
    return image


def local_ratio_normalize(node_tree: bpy.types.NodeTree, value_socket, prefix: str, blur_pixels: int, floor_value: float, y: float):
    envelope = blur_node(
        node_tree,
        f"{prefix}_local_env",
        f"{prefix}_local_env",
        (-1040.0, y - 180.0),
        blur_pixels,
    )
    ensure_link(node_tree, value_socket, envelope.inputs[0])
    envelope_floor = math_node(
        node_tree,
        "ADD",
        f"{prefix}_local_env_floor",
        f"{prefix}_local_env_floor",
        (-800.0, y - 180.0),
        clamp=False,
    )
    ensure_link(node_tree, envelope.outputs[0], envelope_floor.inputs[0])
    envelope_floor.inputs[1].default_value = floor_value
    ratio = math_node(
        node_tree,
        "DIVIDE",
        f"{prefix}_local_ratio",
        f"{prefix}_local_ratio",
        (-800.0, y),
        clamp=False,
    )
    ensure_link(node_tree, value_socket, ratio.inputs[0])
    ensure_link(node_tree, envelope_floor.outputs["Value"], ratio.inputs[1])
    normalized = normalize_node(
        node_tree,
        ratio.outputs["Value"],
        f"{prefix}_local_ratio_normalized",
        f"{prefix}_local_ratio_normalized",
        (-560.0, y),
    )
    return normalized.outputs[0]


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
    if "alpha_gain" not in spec:
        return masked.outputs["Value"]
    gain = value_node(
        node_tree,
        spec["alpha_gain"],
        f"{prefix}_alpha_gain",
        f"{prefix}_alpha_gain",
        (1040.0, y),
    )
    gained = math_node(node_tree, "MULTIPLY", f"{prefix}_alpha_scaled", f"{prefix}_alpha_scaled", (1280.0, y))
    ensure_link(node_tree, masked.outputs["Value"], gained.inputs[0])
    ensure_link(node_tree, gain.outputs[0], gained.inputs[1])
    return gained.outputs["Value"]


def build_prep_stage(scene: bpy.types.Scene) -> dict[str, tuple[Path, Path]]:
    configure_scene(scene)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    exr_frame = frame_node(node_tree, "Frame Prep EXR Inputs", "EXR Inputs", (-1700.0, 820.0), (0.16, 0.16, 0.16))
    mask_frame = frame_node(node_tree, "Frame Prep Masks", "Visible Arboreal Masks", (-1380.0, 740.0), (0.18, 0.18, 0.12))
    prep_frame = frame_node(node_tree, "Frame Prep Mist", "Mist Prep", (-920.0, 760.0), (0.14, 0.18, 0.20))
    output_frame = frame_node(node_tree, "Frame Prep Outputs", "Prep Outputs", (-320.0, 760.0), (0.12, 0.20, 0.14))

    pathway = image_node(node_tree, PATHWAY_EXR, "EXR Pathway", "EXR Pathway", (-1540.0, 560.0))
    priority = image_node(node_tree, PRIORITY_EXR, "EXR Priority", "EXR Priority", (-1540.0, 200.0))
    trending = image_node(node_tree, TRENDING_EXR, "EXR Trending", "EXR Trending", (-1540.0, -160.0)) if TRENDING_EXR.exists() else None
    parent_nodes(exr_frame, pathway, priority, trending)

    mask_visible_pathway = id_mask_node(
        node_tree,
        pathway.outputs["IndexOB"],
        "mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (-1180.0, 420.0),
    )
    mask_all_priority = id_mask_node(
        node_tree,
        priority.outputs["IndexOB"],
        "mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (-1180.0, 60.0),
    )
    mask_visible_priority = math_node(
        node_tree,
        "MULTIPLY",
        "mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (-940.0, 60.0),
    )
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])
    parent_nodes(mask_frame, mask_visible_pathway, mask_all_priority, mask_visible_priority)

    scene_specs = [
        ("pathway", pathway, mask_visible_pathway.outputs["Alpha"], 520.0),
        ("priority", priority, mask_visible_priority.outputs["Value"], 160.0),
    ]
    if trending is not None:
        mask_visible_trending = id_mask_node(
            node_tree,
            trending.outputs["IndexOB"],
            "mask_visible-arboreal_trending",
            "mask_visible-arboreal_trending",
            (-1180.0, -300.0),
        )
        parent_nodes(mask_frame, mask_visible_trending)
        scene_specs.append(("trending", trending, mask_visible_trending.outputs["Alpha"], -200.0))

    rendered_paths: list[Path] = []
    final_paths: dict[str, tuple[Path, Path]] = {}

    for scene_name, exr_node, alpha_socket, y in scene_specs:
        prep_mode = RENDER_MODE in {"edges_only", "prep_only_static"}
        scene_dir = ensure_dir((PREP_ROOT if prep_mode else OUTPUT_ROOT) / scene_name)
        visible = set_alpha_node(
            node_tree,
            exr_image_socket(exr_node),
            alpha_socket,
            f"{scene_name}_visible_arboreal",
            f"{scene_name}_visible_arboreal",
            (-760.0, y + 80.0),
        )
        mist_normalized = normalize_node(
            node_tree,
            exr_node.outputs["Mist"],
            f"{scene_name}_mist_normalized",
            f"{scene_name}_mist_normalized",
            (-760.0, y - 110.0),
        )
        mist_visible = set_alpha_node(
            node_tree,
            mist_normalized.outputs[0],
            alpha_socket,
            f"{scene_name}_mist_normalized_visible",
            f"{scene_name}_mist_normalized_visible",
            (-500.0, y - 110.0),
        )
        parent_nodes(prep_frame, visible, mist_normalized, mist_visible)

        visible_path = file_output_node(
            node_tree,
            visible.outputs["Image"],
            scene_dir,
            f"Output {scene_name} Visible",
            f"Output {scene_name} Visible",
            (-220.0, y + 80.0),
            f"{scene_name}_visible_arboreal",
        )
        node_tree.nodes[f"Output {scene_name} Visible"].parent = output_frame
        mist_path = file_output_node(
            node_tree,
            mist_visible.outputs["Image"],
            scene_dir,
            f"Output {scene_name} Mist Visible",
            f"Output {scene_name} Mist Visible",
            (-220.0, y - 110.0),
            f"{scene_name}_mist_normalized_visible",
        )
        node_tree.nodes[f"Output {scene_name} Mist Visible"].parent = output_frame
        rendered_paths.extend([visible_path, mist_path])
        final_paths[scene_name] = (visible_path, mist_path)

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (120.0, 240.0))
    ensure_link(node_tree, exr_image_socket(pathway), composite.inputs[0])
    parent_nodes(output_frame, composite)

    render_scene(scene, write_still=False)

    for scene_name, paths in final_paths.items():
        final_paths[scene_name] = (rename_output(paths[0]), rename_output(paths[1]))
    return final_paths


def build_variant_stage(scene: bpy.types.Scene, scene_name: str, spec: dict, visible_path: Path, mist_visible_path: Path, output_dir: Path) -> dict[str, str]:
    configure_scene(scene)
    node_tree = scene.node_tree
    clear_node_tree(node_tree)

    input_frame = frame_node(node_tree, "Frame Variant Inputs", "Variant Inputs", (-1720.0, 360.0), (0.16, 0.16, 0.16))
    signal_frame = frame_node(node_tree, "Frame Variant Signal", "Edge Signal", (-1160.0, 320.0), (0.18, 0.16, 0.10))
    width_frame = frame_node(node_tree, "Frame Variant Width", "Width Shaping", (-260.0, 320.0), (0.18, 0.14, 0.18))
    output_frame = frame_node(node_tree, "Frame Variant Output", "Composite And Outputs", (1120.0, 320.0), (0.12, 0.20, 0.14))

    visible = image_node(node_tree, visible_path, f"{scene_name}_visible_png", f"{scene_name}_visible_png", (-1560.0, 160.0))
    mist_visible = image_node(node_tree, mist_visible_path, f"{scene_name}_mist_visible_png", f"{scene_name}_mist_visible_png", (-1560.0, -120.0))
    mist_bw = rgb_to_bw_node(node_tree, mist_visible.outputs["Image"], f"{scene_name}_mist_bw", f"{scene_name}_mist_bw", (-1300.0, -120.0))
    mist_rgba = separate_rgba_node(node_tree, mist_visible.outputs["Image"], f"{scene_name}_mist_rgba", f"{scene_name}_mist_rgba", (-1300.0, -320.0))
    parent_nodes(input_frame, visible, mist_visible, mist_bw, mist_rgba)
    variant_name = f"{scene_name}_{spec['name']}"

    filter_input = mist_bw.outputs["Val"]
    filter_input_x = -1060.0
    if "input_remap_low" in spec and "input_remap_high" in spec:
        filter_input = remap_value_socket(
            node_tree,
            filter_input,
            f"{variant_name}_input",
            (-1060.0, -300.0),
            spec["input_remap_low"],
            spec["input_remap_high"],
        )
        filter_input_x = -580.0
    if VARIANT_PRESET == "localratio":
        filter_input = local_ratio_normalize(
            node_tree,
            mist_bw.outputs["Val"],
            variant_name,
            spec["local_blur_pixels"],
            spec["local_floor"],
            -60.0,
        )
        filter_input_x = -320.0

    edge_filter = filter_node(
        node_tree,
        filter_input,
        f"{variant_name}_filter",
        f"{variant_name}_filter",
        (filter_input_x, -60.0),
        spec["filter_type"],
    )
    edge_normalized = normalize_node(
        node_tree,
        edge_filter.outputs[0],
        f"{variant_name}_normalized",
        f"{variant_name}_normalized",
        (filter_input_x + 240.0, -60.0),
    )
    parent_nodes(signal_frame, edge_filter, edge_normalized)

    edge_signal = edge_normalized.outputs[0]
    if "edge_remap_low" in spec and "edge_remap_high" in spec:
        edge_signal = remap_value_socket(
            node_tree,
            edge_signal,
            f"{variant_name}_edge",
            (filter_input_x + 480.0, 120.0),
            spec["edge_remap_low"],
            spec["edge_remap_high"],
        )
    if VARIANT_PRESET == "edgelocalratio":
        edge_signal = local_ratio_normalize(
            node_tree,
            edge_normalized.outputs[0],
            f"{variant_name}_edge",
            spec["edge_local_blur_pixels"],
            spec["edge_local_floor"],
            -60.0,
        )
    if "screen_gain_bottom" in spec:
        gain_image = vertical_gain_image(f"{variant_name}_vertical_gain", spec["screen_gain_bottom"], spec["screen_gain_power"])
        gain = datablock_image_node(node_tree, gain_image, f"{variant_name}_vertical_gain", f"{variant_name}_vertical_gain", (filter_input_x + 480.0, -320.0))
        gain_rgba = separate_rgba_node(node_tree, gain.outputs["Image"], f"{variant_name}_gain_rgba", f"{variant_name}_gain_rgba", (filter_input_x + 720.0, -320.0))
        edge_lifted = math_node(
            node_tree,
            "MULTIPLY",
            f"{variant_name}_screen_lift",
            f"{variant_name}_screen_lift",
            (filter_input_x + 720.0, -60.0),
            clamp=False,
        )
        ensure_link(node_tree, edge_signal, edge_lifted.inputs[0])
        ensure_link(node_tree, gain_rgba.outputs["R"], edge_lifted.inputs[1])
        edge_screen_normalized = normalize_node(
            node_tree,
            edge_lifted.outputs["Value"],
            f"{variant_name}_screen_lift_normalized",
            f"{variant_name}_screen_lift_normalized",
            (filter_input_x + 960.0, -60.0),
        )
        edge_signal = edge_screen_normalized.outputs[0]
        parent_nodes(signal_frame, gain, gain_rgba, edge_lifted, edge_screen_normalized)

    edge_alpha = build_edge_width(node_tree, edge_signal, mist_rgba.outputs["A"], variant_name, spec, 260.0)

    edge_image = solid_color_image(
        node_tree,
        visible.outputs["Image"],
        f"{variant_name}_edge_rgb",
        f"{variant_name}_edge_rgb",
        (1180.0, -180.0),
        EDGE_COLOR_LINEAR,
    )
    edge_rgb_node = node_tree.nodes[f"{variant_name}_edge_rgb"]
    edge_rgb_rgb_node = node_tree.nodes[f"{variant_name}_edge_rgb_rgb"]
    edge_rgba = set_alpha_node(
        node_tree,
        edge_image,
        edge_alpha,
        f"{variant_name}_edge_rgba",
        f"{variant_name}_edge_rgba",
        (1400.0, -60.0),
    )
    composite_image = alpha_over_node(
        node_tree,
        f"{variant_name}_composite",
        f"{variant_name}_composite",
        (1640.0, 80.0),
        visible.outputs["Image"],
        edge_rgba.outputs["Image"],
    )
    for node_name in (
        f"{variant_name}_presence",
        f"{variant_name}_strong",
        f"{variant_name}_core",
        f"{variant_name}_wide",
        f"{variant_name}_combined",
        f"{variant_name}_soften",
        f"{variant_name}_masked",
    ):
        node = node_tree.nodes.get(node_name)
        if node is not None:
            node.parent = width_frame
    parent_nodes(output_frame, edge_rgb_node, edge_rgb_rgb_node, edge_rgba, composite_image)

    edge_only_rendered = file_output_node(
        node_tree,
        edge_rgba.outputs["Image"],
        output_dir,
        f"Output {variant_name} Edge",
        f"Output {variant_name} Edge",
        (1640.0, -180.0),
        variant_name if RENDER_MODE == "edges_only" else f"{variant_name}_edges",
    )
    node_tree.nodes[f"Output {variant_name} Edge"].parent = output_frame
    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (1880.0, 80.0))
    parent_nodes(output_frame, composite)
    if RENDER_MODE == "edges_only":
        ensure_link(node_tree, edge_rgba.outputs["Image"], composite.inputs[0])
        bpy.ops.render.render(write_still=False)
        edge_path = rename_output(edge_only_rendered)
        log(f"Rendered {edge_path.name}")
        return {"edge": str(edge_path)}

    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (1880.0, -80.0))
    ensure_link(node_tree, composite_image.outputs["Image"], composite.inputs[0])
    ensure_link(node_tree, composite_image.outputs["Image"], viewer.inputs[0])
    parent_nodes(output_frame, viewer, composite)

    composite_path = output_dir / f"{variant_name}_composite.png"
    scene.render.filepath = str(composite_path)
    bpy.ops.render.render(write_still=True)
    edge_path = rename_output(edge_only_rendered)
    log(f"Rendered {composite_path.name}")
    return {
        "edge": str(edge_path),
        "composite": str(composite_path),
    }


def build_static_variant_branch(
    node_tree: bpy.types.NodeTree,
    scene_name: str,
    spec: dict,
    visible_socket,
    mist_visible_socket,
    base_x: float,
    base_y: float,
    output_dir: Path,
):
    variant_name = f"{scene_name}_{spec['name']}"
    variant_frame = frame_node(
        node_tree,
        f"Frame {variant_name}",
        variant_name,
        (base_x - 120.0, base_y + 220.0),
        (0.18, 0.14, 0.18),
    )
    signal_frame = frame_node(
        node_tree,
        f"Frame {variant_name} Signal",
        f"{variant_name} signal",
        (base_x + 120.0, base_y + 220.0),
        (0.18, 0.16, 0.10),
    )
    width_frame = frame_node(
        node_tree,
        f"Frame {variant_name} Width",
        f"{variant_name} width",
        (base_x + 1160.0, base_y + 220.0),
        (0.18, 0.14, 0.18),
    )
    output_frame = frame_node(
        node_tree,
        f"Frame {variant_name} Output",
        f"{variant_name} output",
        (base_x + 2440.0, base_y + 220.0),
        (0.12, 0.20, 0.14),
    )

    mist_bw = rgb_to_bw_node(
        node_tree,
        mist_visible_socket,
        f"{variant_name}_mist_bw",
        f"{variant_name}_mist_bw",
        (base_x, base_y),
    )
    mist_rgba = separate_rgba_node(
        node_tree,
        mist_visible_socket,
        f"{variant_name}_mist_rgba",
        f"{variant_name}_mist_rgba",
        (base_x, base_y - 220.0),
    )
    parent_nodes(variant_frame, mist_bw, mist_rgba)

    filter_input = mist_bw.outputs["Val"]
    filter_input_x = base_x + 260.0
    if "input_remap_low" in spec and "input_remap_high" in spec:
        filter_input = remap_value_socket(
            node_tree,
            filter_input,
            f"{variant_name}_input",
            (base_x + 260.0, base_y - 280.0),
            spec["input_remap_low"],
            spec["input_remap_high"],
        )
        filter_input_x = base_x + 740.0
    if "local_blur_pixels" in spec and "local_floor" in spec:
        filter_input = local_ratio_normalize(
            node_tree,
            mist_bw.outputs["Val"],
            variant_name,
            spec["local_blur_pixels"],
            spec["local_floor"],
            base_y,
        )
        filter_input_x = base_x + 1020.0

    edge_filter = filter_node(
        node_tree,
        filter_input,
        f"{variant_name}_filter",
        f"{variant_name}_filter",
        (filter_input_x, base_y),
        spec["filter_type"],
    )
    edge_normalized = normalize_node(
        node_tree,
        edge_filter.outputs[0],
        f"{variant_name}_normalized",
        f"{variant_name}_normalized",
        (filter_input_x + 240.0, base_y),
    )
    parent_nodes(signal_frame, edge_filter, edge_normalized)

    edge_signal = edge_normalized.outputs[0]
    if "edge_remap_low" in spec and "edge_remap_high" in spec:
        edge_signal = remap_value_socket(
            node_tree,
            edge_signal,
            f"{variant_name}_edge",
            (filter_input_x + 480.0, base_y + 220.0),
            spec["edge_remap_low"],
            spec["edge_remap_high"],
        )
    if "edge_local_blur_pixels" in spec and "edge_local_floor" in spec:
        edge_signal = local_ratio_normalize(
            node_tree,
            edge_normalized.outputs[0],
            f"{variant_name}_edge",
            spec["edge_local_blur_pixels"],
            spec["edge_local_floor"],
            base_y,
        )

    if "screen_gain_bottom" in spec:
        gain_image = vertical_gain_image(
            f"{variant_name}_vertical_gain",
            spec["screen_gain_bottom"],
            spec["screen_gain_power"],
        )
        gain = datablock_image_node(
            node_tree,
            gain_image,
            f"{variant_name}_vertical_gain",
            f"{variant_name}_vertical_gain",
            (filter_input_x + 480.0, base_y - 260.0),
        )
        gain_rgba = separate_rgba_node(
            node_tree,
            gain.outputs["Image"],
            f"{variant_name}_gain_rgba",
            f"{variant_name}_gain_rgba",
            (filter_input_x + 720.0, base_y - 260.0),
        )
        edge_lifted = math_node(
            node_tree,
            "MULTIPLY",
            f"{variant_name}_screen_lift",
            f"{variant_name}_screen_lift",
            (filter_input_x + 720.0, base_y),
            clamp=False,
        )
        ensure_link(node_tree, edge_signal, edge_lifted.inputs[0])
        ensure_link(node_tree, gain_rgba.outputs["R"], edge_lifted.inputs[1])
        edge_screen_normalized = normalize_node(
            node_tree,
            edge_lifted.outputs["Value"],
            f"{variant_name}_screen_lift_normalized",
            f"{variant_name}_screen_lift_normalized",
            (filter_input_x + 960.0, base_y),
        )
        edge_signal = edge_screen_normalized.outputs[0]
        parent_nodes(signal_frame, gain, gain_rgba, edge_lifted, edge_screen_normalized)

    edge_alpha = build_edge_width(
        node_tree,
        edge_signal,
        mist_rgba.outputs["A"],
        variant_name,
        spec,
        base_y,
    )

    edge_image = solid_color_image(
        node_tree,
        visible_socket,
        f"{variant_name}_edge_rgb",
        f"{variant_name}_edge_rgb",
        (base_x + 2280.0, base_y - 180.0),
        EDGE_COLOR_LINEAR,
    )
    edge_rgb_node = node_tree.nodes[f"{variant_name}_edge_rgb"]
    edge_rgb_rgb_node = node_tree.nodes[f"{variant_name}_edge_rgb_rgb"]
    edge_rgba = set_alpha_node(
        node_tree,
        edge_image,
        edge_alpha,
        f"{variant_name}_edge_rgba",
        f"{variant_name}_edge_rgba",
        (base_x + 2520.0, base_y),
    )
    parent_nodes(output_frame, edge_rgb_node, edge_rgb_rgb_node, edge_rgba)

    for node_name in (
        f"{variant_name}_presence",
        f"{variant_name}_strong",
        f"{variant_name}_core",
        f"{variant_name}_wide",
        f"{variant_name}_combined",
        f"{variant_name}_soften",
        f"{variant_name}_masked",
    ):
        node = node_tree.nodes.get(node_name)
        if node is not None:
            node.parent = width_frame

    edge_only_rendered = file_output_node(
        node_tree,
        edge_rgba.outputs["Image"],
        output_dir,
        f"Output {variant_name} Edge",
        f"Output {variant_name} Edge",
        (base_x + 2760.0, base_y),
        variant_name,
    )
    node_tree.nodes[f"Output {variant_name} Edge"].parent = output_frame
    return edge_only_rendered, edge_rgba.outputs["Image"]


def build_scene(scene: bpy.types.Scene) -> list[Path]:
    original_render_mode = RENDER_MODE
    original_prep_root = PREP_ROOT
    prep_cache_root = OUTPUT_ROOT / "_prep_cache"
    prep_scene = bpy.data.scenes.new(f"{scene.name}__PrepScratch")
    try:
        globals()["RENDER_MODE"] = "prep_only_static"
        globals()["PREP_ROOT"] = prep_cache_root
        prepared = build_prep_stage(prep_scene)
    finally:
        globals()["RENDER_MODE"] = original_render_mode
        globals()["PREP_ROOT"] = original_prep_root
        if prep_scene.name in bpy.data.scenes:
            bpy.data.scenes.remove(prep_scene)

    configure_scene(scene)
    node_tree = scene.node_tree
    clear_node_tree(node_tree)
    ensure_dir(OUTPUT_ROOT)

    input_frame = frame_node(node_tree, "Frame Static Prep Inputs", "Prep PNG Inputs", (-3600.0, 1180.0), (0.16, 0.16, 0.16))
    header_frame = frame_node(node_tree, "Frame Static Header", "Mist Kirsch Sizes", (-420.0, 1460.0), (0.18, 0.14, 0.20))

    header = new_node(
        node_tree,
        "NodeFrame",
        "Frame Static Note",
        "Outputs: pathway / priority / trending (optional)",
        (-260.0, 1380.0),
        color=(0.16, 0.16, 0.16),
    )
    header.label_size = 20
    header.use_custom_color = True
    header.color = (0.16, 0.16, 0.16)
    header.shrink = False
    header.parent = header_frame

    rendered_paths: list[Path] = []
    composite_sockets = []
    variant_offsets = {
        "mist_kirsch_thin": -1600.0,
        "mist_kirsch_fine": 0.0,
        "mist_kirsch_extra_thin": 1600.0,
        "mist_kirsch_remapped": 3200.0,
    }
    scene_y = {
        "pathway": 760.0,
        "priority": 20.0,
        "trending": -720.0,
    }

    for scene_name, (visible_path, mist_visible_path) in prepared.items():
        visible_path = resolve_existing_render_path(visible_path)
        mist_visible_path = resolve_existing_render_path(mist_visible_path)
        y = scene_y.get(scene_name, 0.0)
        visible_png = image_node(
            node_tree,
            visible_path,
            f"{scene_name}_visible_png",
            f"{scene_name}_visible_png",
            (-3440.0, y + 80.0),
        )
        mist_visible_png = image_node(
            node_tree,
            mist_visible_path,
            f"{scene_name}_mist_visible_png",
            f"{scene_name}_mist_visible_png",
            (-3440.0, y - 160.0),
        )
        parent_nodes(input_frame, visible_png, mist_visible_png)

        for spec in KIRSCHSIZE_VARIANTS:
            base_x = variant_offsets[spec["name"]]
            rendered_path, composite_socket = build_static_variant_branch(
                    node_tree,
                    scene_name,
                    spec,
                    visible_png.outputs["Image"],
                    mist_visible_png.outputs["Image"],
                    base_x,
                    y,
                    OUTPUT_ROOT,
                )
            rendered_paths.append(rendered_path)
            composite_sockets.append(composite_socket)

    composite = new_node(node_tree, "CompositorNodeComposite", "Composite", "Composite", (3180.0, 1120.0))
    viewer = new_node(node_tree, "CompositorNodeViewer", "Viewer", "Viewer", (3180.0, 920.0))
    parent_nodes(header_frame, composite, viewer)
    if composite_sockets:
        transparent = rgb_node(
            node_tree,
            "static_mist_composite_base",
            "static_mist_composite_base",
            (2100.0, 1180.0),
            (0.0, 0.0, 0.0, 0.0),
        )
        parent_nodes(header_frame, transparent)
        running_socket = transparent.outputs[0]
        x = 2340.0
        for idx, edge_socket in enumerate(composite_sockets):
            mix = alpha_over_node(
                node_tree,
                f"static_mist_composite_{idx}",
                f"static_mist_composite_{idx}",
                (x, 1180.0),
                running_socket,
                edge_socket,
            )
            parent_nodes(header_frame, mix)
            running_socket = mix.outputs["Image"]
            x += 220.0
        ensure_link(node_tree, running_socket, composite.inputs[0])
        ensure_link(node_tree, running_socket, viewer.inputs[0])
    return rendered_paths


def run_output_workflow(scene: bpy.types.Scene) -> None:
    prepared = build_prep_stage(scene)

    if RENDER_MODE == "edges_only":
        for scene_name, (visible_path, mist_visible_path) in prepared.items():
            for spec in VARIANT_SPECS:
                build_variant_stage(
                    scene,
                    scene_name,
                    spec,
                    visible_path,
                    mist_visible_path,
                    OUTPUT_ROOT,
                )
        if PREP_ROOT.exists():
            shutil.rmtree(PREP_ROOT)
        return

    summary = {
        "workflow_id": WORKFLOW_ID,
        "reused_workflow": "blender_exr_arboreal_mist_v1",
        "reused_pattern": "render_exr_base_mist_bestpractice_v5_blender.py",
        "notes": WORKFLOW_NOTES,
        "variant_preset": VARIANT_PRESET,
        "sources": {
            "pathway": str(PATHWAY_EXR),
            "priority": str(PRIORITY_EXR),
            "trending": str(TRENDING_EXR),
        },
        "variants": list(VARIANT_SPECS),
        "outputs": {},
    }

    for scene_name, (visible_path, mist_visible_path) in prepared.items():
        scene_dir = ensure_dir(OUTPUT_ROOT / scene_name)
        summary["outputs"][scene_name] = {
            "visible": str(visible_path),
            "mist_visible": str(mist_visible_path),
            "variants": {},
        }
        for spec in VARIANT_SPECS:
            summary["outputs"][scene_name]["variants"][spec["name"]] = build_variant_stage(
                scene,
                scene_name,
                spec,
                visible_path,
                mist_visible_path,
                scene_dir,
            )

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Wrote {summary_path}")


def main() -> None:
    scene = bpy.context.scene
    run_output_workflow(scene)
    BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    log(f"Saved {BLEND_PATH}")


if __name__ == "__main__":
    main()
