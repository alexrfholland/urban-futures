from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"


def load_current_mist_helper():
    module_path = Path(__file__).with_name("edge_lab_current_mist.py")
    spec = importlib.util.spec_from_file_location("edge_lab_current_mist", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


CURRENT_SOURCE_BLEND = env_path(
    "EDGE_LAB_CURRENT_SOURCE_BLEND",
    DATA_ROOT / "edge_lab_output_suite_refined.blend",
)
LEGACY_SOURCE_BLEND = env_path(
    "EDGE_LAB_LEGACY_SOURCE_BLEND",
    DATA_ROOT / "city_exr_compositor_lightweight_city_final.blend",
)
OUTPUT_BLEND = env_path(
    "EDGE_LAB_FINAL_TEMPLATE_BLEND",
    DATA_ROOT / "edge_lab_final_template.blend",
)
PROPOSAL_PATHWAY_EXR = env_path(
    "EDGE_LAB_PROPOSAL_PATHWAY_EXR",
    DATA_ROOT / "inputs" / "parade_8k_network_20260402" / "parade_pathway_state_8k.exr",
)
PROPOSAL_PRIORITY_EXR = env_path(
    "EDGE_LAB_PROPOSAL_PRIORITY_EXR",
    DATA_ROOT / "inputs" / "parade_8k_network_20260402" / "parade_priority_state_8k.exr",
)
PROPOSAL_TRENDING_EXR = env_path(
    "EDGE_LAB_PROPOSAL_TRENDING_EXR",
    DATA_ROOT / "inputs" / "parade_8k_network_20260402" / "parade_trending_state_8k.exr",
)

CURRENT_SOURCE_SCENE = os.environ.get("EDGE_LAB_CURRENT_SOURCE_SCENE", "Suite")
LEGACY_SOURCE_SCENE = os.environ.get("EDGE_LAB_LEGACY_SOURCE_SCENE", "City")


def log(message: str) -> None:
    print(f"[build_edge_lab_final_template_blend] {message}")


def clear_state() -> None:
    scratch = bpy.data.scenes[0]
    scratch.name = "Scratch"
    for scene in list(bpy.data.scenes)[1:]:
        bpy.data.scenes.remove(scene)
    for world in list(bpy.data.worlds):
        bpy.data.worlds.remove(world)


def append_scene(blend_path: Path, source_scene_name: str) -> bpy.types.Scene:
    if not blend_path.exists():
        raise FileNotFoundError(blend_path)
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        if source_scene_name not in data_from.scenes:
            raise ValueError(f"Scene '{source_scene_name}' not found in {blend_path}")
        data_to.scenes = [source_scene_name]
    scene = data_to.scenes[0]
    if scene is None:
        raise RuntimeError(f"Failed to append '{source_scene_name}' from {blend_path}")
    return scene


def normalize_scene(scene: bpy.types.Scene) -> None:
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
    parent: bpy.types.Node | None = None,
    color: tuple[float, float, float] | None = None,
):
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if parent is not None:
        node.parent = parent
    if color is not None:
        node.use_custom_color = True
        node.color = color
    return node


def remove_prefixed_nodes(node_tree: bpy.types.NodeTree, prefix: str) -> None:
    for node in list(node_tree.nodes):
        if node.name.startswith(prefix):
            node_tree.nodes.remove(node)


def find_ao_group() -> bpy.types.NodeTree:
    group = bpy.data.node_groups.get("_AO SHADING.001")
    if group is None:
        raise ValueError("Missing _AO SHADING.001 node group")
    return group


def find_required_group(name: str) -> bpy.types.NodeTree:
    group = bpy.data.node_groups.get(name)
    if group is None:
        raise ValueError(f"Missing {name} node group")
    return group


BIOENVELOPE_PALETTE_DEFAULTS: dict[str, tuple[float, float, float, float]] = {
    "0 Unmatched": (0.0, 0.0, 0.0, 0.0),
    "1 Exoskeleton": (0.85, 0.39, 0.55, 1.0),
    "2 BrownRoof": (0.72, 0.48, 0.22, 1.0),
    "3 OtherGround": (0.46, 0.64, 0.77, 1.0),
    "4 Rewilded": (0.36, 0.72, 0.34, 1.0),
    "5 FootprintDepaved": (0.92, 0.75, 0.33, 1.0),
    "6 LivingFacade": (0.30, 0.66, 0.60, 1.0),
    "7 GreenRoof": (0.55, 0.80, 0.31, 1.0),
}

TREE_ID = 3
MATCH_EPSILON = 0.1
GROUND_ID = 1
BUILDING_ID = 2
BASE_DEPTH_EDGE_COLOR_LINEAR = (0.015996, 0.006048822, 0.043735046, 1.0)

BASE_DEPTH_VARIANT_SPECS: tuple[dict[str, object], ...] = (
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

SIZE_SPECS: tuple[dict[str, object], ...] = (
    {"slug": "small", "value": 1, "hex": "#AADB5E"},
    {"slug": "medium", "value": 2, "hex": "#9AB9DE"},
    {"slug": "large", "value": 3, "hex": "#F99F76"},
    {"slug": "senescing", "value": 4, "hex": "#EB9BC5"},
    {"slug": "snag", "value": 5, "hex": "#FCE358"},
    {"slug": "fallen", "value": 6, "hex": "#8F89BF"},
    {"slug": "artificial", "value": 7, "hex": "#FF0000"},
)

PROPOSAL_BRANCH_SPECS: tuple[tuple[str, tuple[tuple[int, str, str], ...]], ...] = (
    (
        "proposal-deploy-structure",
        (
            (2, "adapt-utility-pole", "#FF0000"),
            (3, "translocated-log", "#8F89BF"),
            (4, "upgrade-feature", "#CE6DD9"),
        ),
    ),
    (
        "proposal-decay",
        (
            (2, "buffer-feature", "#B83B6B"),
            (3, "brace-feature", "#D9638C"),
        ),
    ),
    (
        "proposal-recruit",
        (
            (2, "buffer-feature", "#C5E28E"),
            (3, "rewild-ground", "#5CB85C"),
        ),
    ),
    (
        "proposal-colonise",
        (
            (2, "rewild-ground", "#5CB85C"),
            (3, "enrich-envelope", "#8CCC4F"),
            (4, "roughen-envelope", "#B87A38"),
        ),
    ),
    (
        "proposal-release-control",
        (
            (1, "rejected", "#333333"),
            (2, "reduce-pruning", "#808080"),
            (3, "eliminate-pruning", "#FFFFFF"),
        ),
    ),
)


def release_control_hex(intensity: float) -> str:
    value = max(0, min(255, round(255.0 * intensity)))
    return f"#{value:02X}{value:02X}{value:02X}"


def apply_bioenvelope_palette_defaults(node: bpy.types.Node) -> None:
    for input_name, value in BIOENVELOPE_PALETTE_DEFAULTS.items():
        socket = node.inputs.get(input_name)
        if socket is not None and not socket.is_linked:
            socket.default_value = value


def srgb_channel_to_linear(value: int) -> float:
    normalized = value / 255.0
    if normalized <= 0.04045:
        return normalized / 12.92
    return ((normalized + 0.055) / 1.055) ** 2.4


def hex_to_linear_rgba(hex_value: str) -> tuple[float, float, float, float]:
    value = hex_value.lstrip("#")
    return (
        srgb_channel_to_linear(int(value[0:2], 16)),
        srgb_channel_to_linear(int(value[2:4], 16)),
        srgb_channel_to_linear(int(value[4:6], 16)),
        1.0,
    )


def socket_by_name(node: bpy.types.Node, preferred_name: str):
    for socket in node.outputs:
        if socket.name == preferred_name:
            return socket
    normalized_preferred = preferred_name.lower().replace(" ", "_")
    for socket in node.outputs:
        if socket.name.lower().replace(" ", "_") == normalized_preferred:
            return socket
    raise KeyError(
        f"Socket '{preferred_name}' not found on {node.name}. "
        f"Available: {[socket.name for socket in node.outputs]}"
    )


def build_value_match_socket(
    node_tree: bpy.types.NodeTree,
    value_socket,
    target_value: int,
    base_name: str,
    x: float,
    y: float,
    parent: bpy.types.Node,
):
    compare = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{base_name} :: compare",
        "compare",
        (x, y),
        parent=parent,
        color=(0.18, 0.16, 0.20),
    )
    compare.operation = "COMPARE"
    compare.use_clamp = True
    compare.inputs[1].default_value = float(target_value)
    compare.inputs[2].default_value = 0.0
    ensure_link(node_tree, value_socket, compare.inputs[0])
    return compare.outputs["Value"]


def build_multiply_socket(
    node_tree: bpy.types.NodeTree,
    left_socket,
    right_socket,
    base_name: str,
    x: float,
    y: float,
    parent: bpy.types.Node,
):
    multiply = new_node(
        node_tree,
        "CompositorNodeMath",
        base_name,
        base_name.rsplit("::", 1)[-1].strip(),
        (x, y),
        parent=parent,
        color=(0.18, 0.16, 0.20),
    )
    multiply.operation = "MULTIPLY"
    multiply.use_clamp = True
    ensure_link(node_tree, left_socket, multiply.inputs[0])
    ensure_link(node_tree, right_socket, multiply.inputs[1])
    return multiply.outputs["Value"]


def build_color_alpha_image(
    node_tree: bpy.types.NodeTree,
    rgba: tuple[float, float, float, float],
    alpha_socket,
    base_name: str,
    x: float,
    y: float,
    parent: bpy.types.Node,
):
    rgb = new_node(
        node_tree,
        "CompositorNodeRGB",
        f"{base_name} :: rgb",
        "rgb",
        (x, y),
        parent=parent,
        color=(0.18, 0.12, 0.20),
    )
    rgb.outputs[0].default_value = rgba

    set_alpha = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{base_name} :: rgba",
        "rgba",
        (x + 220.0, y),
        parent=parent,
        color=(0.16, 0.20, 0.16),
    )
    set_alpha.mode = "APPLY"
    ensure_link(node_tree, rgb.outputs[0], set_alpha.inputs["Image"])
    ensure_link(node_tree, alpha_socket, set_alpha.inputs["Alpha"])
    return set_alpha.outputs["Image"]


def configure_png_output_node(node: bpy.types.Node, base_path: str, stem: str) -> None:
    node.base_path = base_path
    node.format.file_format = "PNG"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "8"
    while len(node.file_slots) > 1:
        node.file_slots.remove(node.file_slots[-1])
    node.file_slots[0].path = f"{stem}_"


def rebuild_group(name: str) -> bpy.types.NodeTree:
    existing = bpy.data.node_groups.get(name)
    if existing is not None:
        bpy.data.node_groups.remove(existing)
    return bpy.data.node_groups.new(name=name, type="CompositorNodeTree")


def add_interface_socket(
    node_tree: bpy.types.NodeTree,
    name: str,
    in_out: str,
    socket_type: str,
):
    return node_tree.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)


def group_math_node(
    node_tree: bpy.types.NodeTree,
    operation: str,
    name: str,
    location: tuple[float, float],
    clamp: bool = True,
) -> bpy.types.Node:
    node = node_tree.nodes.new("CompositorNodeMath")
    node.name = name
    node.label = name
    node.location = location
    node.operation = operation
    node.use_clamp = clamp
    return node


def group_rgb_to_bw_node(
    node_tree: bpy.types.NodeTree,
    image_socket,
    name: str,
    location: tuple[float, float],
) -> bpy.types.Node:
    node = node_tree.nodes.new("CompositorNodeRGBToBW")
    node.name = name
    node.label = name
    node.location = location
    node_tree.links.new(image_socket, node.inputs["Image"])
    return node


def group_normalize_node(
    node_tree: bpy.types.NodeTree,
    value_socket,
    name: str,
    location: tuple[float, float],
) -> bpy.types.Node:
    node = node_tree.nodes.new("CompositorNodeNormalize")
    node.name = name
    node.label = name
    node.location = location
    node_tree.links.new(value_socket, node.inputs[0])
    return node


def group_filter_node(
    node_tree: bpy.types.NodeTree,
    image_socket,
    name: str,
    location: tuple[float, float],
    filter_type: str,
) -> bpy.types.Node:
    node = node_tree.nodes.new("CompositorNodeFilter")
    node.name = name
    node.label = name
    node.location = location
    node.filter_type = filter_type
    node_tree.links.new(image_socket, node.inputs["Image"])
    return node


def group_threshold_ramp(
    node_tree: bpy.types.NodeTree,
    value_socket,
    name: str,
    location: tuple[float, float],
    low: float,
    high: float,
) -> bpy.types.Node:
    node = node_tree.nodes.new("CompositorNodeValToRGB")
    node.name = name
    node.label = name
    node.location = location
    node.color_ramp.interpolation = "LINEAR"
    node.color_ramp.elements[0].position = low
    node.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node.color_ramp.elements[1].position = high
    node.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    node_tree.links.new(value_socket, node.inputs["Fac"])
    return node


def group_dilate_erode_node(
    node_tree: bpy.types.NodeTree,
    name: str,
    location: tuple[float, float],
    distance: int,
) -> bpy.types.Node:
    node = node_tree.nodes.new("CompositorNodeDilateErode")
    node.name = name
    node.label = name
    node.location = location
    node.mode = "DISTANCE"
    node.distance = distance
    return node


def group_blur_node(
    node_tree: bpy.types.NodeTree,
    name: str,
    location: tuple[float, float],
    pixels: int,
) -> bpy.types.Node:
    node = node_tree.nodes.new("CompositorNodeBlur")
    node.name = name
    node.label = name
    node.location = location
    node.filter_type = "GAUSS"
    node.use_relative = False
    node.size_x = pixels
    node.size_y = pixels
    return node


def build_base_depth_filtered_alpha_group(
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
    source_bw = group_rgb_to_bw_node(node_tree, source_socket, f"{prefix}_bw", (-760.0, y))
    filtered = group_filter_node(node_tree, source_bw.outputs["Val"], f"{prefix}_filter", (-520.0, y), filter_type)
    normalized = group_normalize_node(node_tree, filtered.outputs["Image"], f"{prefix}_norm", (-260.0, y))

    presence = group_threshold_ramp(node_tree, normalized.outputs[0], f"{prefix}_presence", (0.0, y + 70.0), low, high)
    strong = group_threshold_ramp(node_tree, normalized.outputs[0], f"{prefix}_strong", (0.0, y - 70.0), low * 0.8, high * 1.35)

    core = group_dilate_erode_node(node_tree, f"{prefix}_core", (260.0, y + 70.0), core_distance)
    wide = group_dilate_erode_node(node_tree, f"{prefix}_wide", (260.0, y - 70.0), wide_distance)
    node_tree.links.new(presence.outputs["Image"], core.inputs[0])
    node_tree.links.new(strong.outputs["Image"], wide.inputs[0])

    merged = group_math_node(node_tree, "MAXIMUM", f"{prefix}_merged", (520.0, y), clamp=True)
    node_tree.links.new(core.outputs[0], merged.inputs[0])
    node_tree.links.new(wide.outputs[0], merged.inputs[1])

    softened = group_blur_node(node_tree, f"{prefix}_blur", (780.0, y), blur_pixels)
    node_tree.links.new(merged.outputs[0], softened.inputs[0])

    masked = group_math_node(node_tree, "MULTIPLY", f"{prefix}_masked_alpha", (1040.0, y), clamp=True)
    node_tree.links.new(softened.outputs[0], masked.inputs[0])
    node_tree.links.new(mask_socket, masked.inputs[1])
    return masked.outputs["Value"]


def build_base_depth_windowed_alpha_group(
    node_tree: bpy.types.NodeTree,
    depth_socket,
    mask_socket,
    prefix: str,
    y: float,
    spec: dict[str, object],
):
    shifted = group_math_node(node_tree, "SUBTRACT", f"{prefix}_shift", (-1260.0, y), clamp=False)
    node_tree.links.new(depth_socket, shifted.inputs[0])
    shifted.inputs[1].default_value = float(spec["min_depth"])

    scaled = group_math_node(node_tree, "DIVIDE", f"{prefix}_scale", (-1040.0, y), clamp=True)
    node_tree.links.new(shifted.outputs["Value"], scaled.inputs[0])
    scaled.inputs[1].default_value = max(float(spec["max_depth"]) - float(spec["min_depth"]), 0.001)

    masked = group_math_node(node_tree, "MULTIPLY", f"{prefix}_window_masked", (-820.0, y), clamp=True)
    node_tree.links.new(scaled.outputs["Value"], masked.inputs[0])
    node_tree.links.new(mask_socket, masked.inputs[1])

    rgba = node_tree.nodes.new("CompositorNodeCombRGBA")
    rgba.name = f"{prefix}_rgba"
    rgba.label = f"{prefix}_rgba"
    rgba.location = (-580.0, y)
    node_tree.links.new(masked.outputs["Value"], rgba.inputs["R"])
    node_tree.links.new(masked.outputs["Value"], rgba.inputs["G"])
    node_tree.links.new(masked.outputs["Value"], rgba.inputs["B"])
    rgba.inputs["A"].default_value = 1.0

    window_bw = group_rgb_to_bw_node(node_tree, rgba.outputs["Image"], f"{prefix}_window_bw", (-320.0, y))
    contrast = group_threshold_ramp(node_tree, window_bw.outputs["Val"], f"{prefix}_contrast", (-80.0, y), 0.02, 0.98)
    return build_base_depth_filtered_alpha_group(
        node_tree,
        contrast.outputs["Image"],
        mask_socket,
        prefix,
        y,
        str(spec["filter"]),
        float(spec["low"]),
        float(spec["high"]),
        int(spec["core"]),
        int(spec["wide"]),
        int(spec["blur"]),
    )


def build_base_depth_multiwindow_building_alpha_group(
    node_tree: bpy.types.NodeTree,
    depth_socket,
    building_mask_socket,
    prefix: str,
    y: float,
    window_specs: tuple[dict[str, object], ...],
):
    combined = None
    for idx, window_spec in enumerate(window_specs):
        branch = build_base_depth_windowed_alpha_group(
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
            max_node = group_math_node(node_tree, "MAXIMUM", f"{prefix}_max_{idx + 1}", (1320.0, y - idx * 90.0), clamp=True)
            node_tree.links.new(combined, max_node.inputs[0])
            node_tree.links.new(branch, max_node.inputs[1])
            combined = max_node.outputs["Value"]
    if combined is None:
        raise ValueError(f"No building windows configured for {prefix}")
    return combined


def build_base_outputs_group() -> bpy.types.NodeTree:
    outliner_group = find_required_group("_OUTLINER.001")

    group = rebuild_group("EDGE_CURRENT_BASE_OUTPUTS")
    add_interface_socket(group, "Existing Image", "INPUT", "NodeSocketColor")
    add_interface_socket(group, "Existing AO", "INPUT", "NodeSocketFloat")
    add_interface_socket(group, "Existing Normal", "INPUT", "NodeSocketVector")
    add_interface_socket(group, "Existing Depth", "INPUT", "NodeSocketFloat")
    add_interface_socket(group, "Existing IndexOB", "INPUT", "NodeSocketFloat")
    add_interface_socket(group, "world_sim_turns", "INPUT", "NodeSocketFloat")
    add_interface_socket(group, "world_sim_nodes", "INPUT", "NodeSocketFloat")

    add_interface_socket(group, "base_rgb", "OUTPUT", "NodeSocketColor")
    add_interface_socket(group, "base_white_render", "OUTPUT", "NodeSocketFloat")
    add_interface_socket(group, "base_outlines", "OUTPUT", "NodeSocketColor")
    add_interface_socket(group, "base_sim-turns", "OUTPUT", "NodeSocketFloat")
    add_interface_socket(group, "base_sim-nodes", "OUTPUT", "NodeSocketFloat")
    add_interface_socket(group, "base_sim-turns_ripple-effect", "OUTPUT", "NodeSocketColor")
    for spec in BASE_DEPTH_VARIANT_SPECS:
        add_interface_socket(group, str(spec["name"]), "OUTPUT", "NodeSocketColor")

    nodes = group.nodes
    links = group.links

    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-1600.0, 0.0)
    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (1600.0, 0.0)

    depth_norm = nodes.new("CompositorNodeNormalize")
    depth_norm.name = "base_outlines_depth_normalize"
    depth_norm.location = (-1180.0, -40.0)
    links.new(group_in.outputs["Existing Depth"], depth_norm.inputs[0])

    outlines = nodes.new("CompositorNodeGroup")
    outlines.name = "base_outlines"
    outlines.label = "base_outlines"
    outlines.node_tree = outliner_group
    outlines.location = (-900.0, -40.0)
    links.new(depth_norm.outputs[0], outlines.inputs["Image"])

    sim_turns_norm = nodes.new("CompositorNodeNormalize")
    sim_turns_norm.name = "base_sim_turns"
    sim_turns_norm.label = "base_sim-turns"
    sim_turns_norm.location = (-880.0, -300.0)
    links.new(group_in.outputs["world_sim_turns"], sim_turns_norm.inputs[0])

    sim_nodes_norm = nodes.new("CompositorNodeNormalize")
    sim_nodes_norm.name = "base_sim_nodes"
    sim_nodes_norm.label = "base_sim-nodes"
    sim_nodes_norm.location = (-880.0, -520.0)
    links.new(group_in.outputs["world_sim_nodes"], sim_nodes_norm.inputs[0])

    turns_divide = nodes.new("CompositorNodeMath")
    turns_divide.name = "world_sim_turns_to_unit"
    turns_divide.label = "Turns to 0-1"
    turns_divide.operation = "DIVIDE"
    turns_divide.inputs[1].default_value = 100.0
    turns_divide.location = (-1180.0, -780.0)
    links.new(group_in.outputs["world_sim_turns"], turns_divide.inputs[0])

    turns_spread = nodes.new("CompositorNodeMath")
    turns_spread.name = "world_sim_turns_spread"
    turns_spread.label = "Distribution spread"
    turns_spread.operation = "POWER"
    turns_spread.inputs[1].default_value = 0.35
    turns_spread.location = (-900.0, -780.0)
    links.new(turns_divide.outputs[0], turns_spread.inputs[0])

    turns_ramp = nodes.new("CompositorNodeValToRGB")
    turns_ramp.name = "world_sim_turns_greyscale"
    turns_ramp.label = "Greyscale ramp"
    turns_ramp.location = (-600.0, -780.0)
    links.new(turns_spread.outputs[0], turns_ramp.inputs["Fac"])

    ground_mask = nodes.new("CompositorNodeIDMask")
    ground_mask.name = "base_depth_ground_mask"
    ground_mask.label = "base_depth_ground_mask"
    ground_mask.location = (-1560.0, -1180.0)
    ground_mask.index = GROUND_ID
    ground_mask.use_antialiasing = True
    links.new(group_in.outputs["Existing IndexOB"], ground_mask.inputs["ID value"])

    building_mask = nodes.new("CompositorNodeIDMask")
    building_mask.name = "base_depth_building_mask"
    building_mask.label = "base_depth_building_mask"
    building_mask.location = (-1560.0, -1460.0)
    building_mask.index = BUILDING_ID
    building_mask.use_antialiasing = True
    links.new(group_in.outputs["Existing IndexOB"], building_mask.inputs["ID value"])

    edge_rgb = nodes.new("CompositorNodeRGB")
    edge_rgb.name = "base_depth_edge_rgb"
    edge_rgb.label = "base_depth_edge_rgb"
    edge_rgb.location = (1860.0, -1360.0)
    edge_rgb.outputs[0].default_value = BASE_DEPTH_EDGE_COLOR_LINEAR

    for index, spec in enumerate(BASE_DEPTH_VARIANT_SPECS):
        spec_name = str(spec["name"])
        y = -1320.0 - index * 520.0
        building_alpha = build_base_depth_multiwindow_building_alpha_group(
            group,
            group_in.outputs["Existing Depth"],
            building_mask.outputs["Alpha"],
            f"{spec_name}_building",
            y,
            tuple(spec["building_windows"]),
        )
        ground_alpha = build_base_depth_windowed_alpha_group(
            group,
            group_in.outputs["Existing Depth"],
            ground_mask.outputs["Alpha"],
            f"{spec_name}_ground",
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
        combined_alpha = group_math_node(group, "MAXIMUM", f"{spec_name}_alpha", (1620.0, y - 110.0), clamp=True)
        links.new(building_alpha, combined_alpha.inputs[0])
        links.new(ground_alpha, combined_alpha.inputs[1])

        edge_rgba = nodes.new("CompositorNodeSetAlpha")
        edge_rgba.name = f"{spec_name}_edge_rgba"
        edge_rgba.label = f"{spec_name}_edge_rgba"
        edge_rgba.location = (2100.0, y - 110.0)
        edge_rgba.mode = "REPLACE_ALPHA"
        links.new(edge_rgb.outputs[0], edge_rgba.inputs["Image"])
        links.new(combined_alpha.outputs["Value"], edge_rgba.inputs["Alpha"])
        links.new(edge_rgba.outputs["Image"], group_out.inputs[spec_name])

    links.new(group_in.outputs["Existing Image"], group_out.inputs["base_rgb"])
    links.new(group_in.outputs["Existing AO"], group_out.inputs["base_white_render"])
    links.new(outlines.outputs["Image"], group_out.inputs["base_outlines"])
    links.new(sim_turns_norm.outputs[0], group_out.inputs["base_sim-turns"])
    links.new(sim_nodes_norm.outputs[0], group_out.inputs["base_sim-nodes"])
    links.new(turns_ramp.outputs["Image"], group_out.inputs["base_sim-turns_ripple-effect"])

    return group


def add_current_base_outputs_branch(scene: bpy.types.Scene) -> None:
    if scene.node_tree is None:
        raise ValueError("Current scene has no compositor node tree")

    prefix = "Current Base Outputs :: "
    node_tree = scene.node_tree
    remove_prefixed_nodes(node_tree, prefix)
    group_tree = build_base_outputs_group()

    existing = node_tree.nodes.get("AO::EXR Existing")
    if existing is None:
        raise ValueError("Current scene is missing AO::EXR Existing for base outputs branch")

    frame = new_node(
        node_tree,
        "NodeFrame",
        f"{prefix}Frame",
        "base_outputs",
        (3800.0, -2500.0),
    )
    frame.use_custom_color = True
    frame.color = (0.12, 0.12, 0.12)

    base_group = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{prefix}Group",
        "Base outputs",
        (3980.0, -2360.0),
        parent=frame,
        color=(0.16, 0.18, 0.22),
    )
    base_group.node_tree = group_tree
    ensure_link(node_tree, existing.outputs["Image"], base_group.inputs["Existing Image"])
    ensure_link(node_tree, existing.outputs["AO"], base_group.inputs["Existing AO"])
    ensure_link(node_tree, existing.outputs["Normal"], base_group.inputs["Existing Normal"])
    ensure_link(node_tree, existing.outputs["Depth"], base_group.inputs["Existing Depth"])
    ensure_link(node_tree, existing.outputs["IndexOB"], base_group.inputs["Existing IndexOB"])
    ensure_link(node_tree, existing.outputs["world_sim_turns"], base_group.inputs["world_sim_turns"])
    ensure_link(node_tree, existing.outputs["world_sim_nodes"], base_group.inputs["world_sim_nodes"])

    output_specs = (
        ("base_rgb", "base_rgb", -2100.0),
        ("base_white_render", "base_white_render", -2250.0),
        ("base_outlines", "base_outlines", -2400.0),
        ("base_sim-turns", "base_sim-turns", -2550.0),
        ("base_sim-nodes", "base_sim-nodes", -2700.0),
        ("base_sim-turns_ripple-effect", "base_sim-turns_ripple-effect", -2850.0),
        ("base_depth_windowed_balanced_refined", "base_depth_windowed_balanced_refined", -3000.0),
        ("base_depth_windowed_internal_refined", "base_depth_windowed_internal_refined", -3150.0),
        ("base_depth_windowed_internal_dense", "base_depth_windowed_internal_dense", -3300.0),
        ("base_depth_windowed_balanced_dense", "base_depth_windowed_balanced_dense", -3450.0),
    )
    for socket_name, label, y in output_specs:
        reroute = new_node(
            node_tree,
            "NodeReroute",
            f"{prefix}{label}",
            label,
            (4700.0, y),
            parent=frame,
        )
        ensure_link(node_tree, base_group.outputs[socket_name], reroute.inputs[0])


def add_current_sizes_branch(scene: bpy.types.Scene) -> None:
    if scene.node_tree is None:
        raise ValueError("Current scene has no compositor node tree")

    prefix = "Sizes::"
    node_tree = scene.node_tree
    remove_prefixed_nodes(node_tree, prefix)

    pathway = node_tree.nodes.get("AO::EXR Pathway")
    priority = node_tree.nodes.get("AO::EXR Priority")
    existing = node_tree.nodes.get("AO::EXR Existing")
    trending = node_tree.nodes.get("Resources::EXR Trending")
    if pathway is None or priority is None or existing is None or trending is None:
        raise ValueError("Current scene is missing EXR inputs for sizes branch")

    frame = new_node(
        node_tree,
        "NodeFrame",
        f"{prefix}FamilyFrame",
        "Sizes",
        (5400.0, 1200.0),
    )
    frame.use_custom_color = True
    frame.color = (0.16, 0.18, 0.14)

    mask_visible_pathway = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{prefix}mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (5600.0, 920.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_visible_pathway.index = TREE_ID
    mask_visible_pathway.use_antialiasing = True
    ensure_link(node_tree, pathway.outputs["IndexOB"], mask_visible_pathway.inputs["ID value"])

    mask_all_priority = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{prefix}mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (5600.0, 640.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_all_priority.index = TREE_ID
    mask_all_priority.use_antialiasing = True
    ensure_link(node_tree, priority.outputs["IndexOB"], mask_all_priority.inputs["ID value"])

    mask_visible_priority = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{prefix}mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (5840.0, 760.0),
        parent=frame,
        color=(0.18, 0.16, 0.20),
    )
    mask_visible_priority.operation = "MULTIPLY"
    mask_visible_priority.use_clamp = True
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])

    mask_visible_existing = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{prefix}mask_visible-arboreal_existing-condition",
        "mask_visible-arboreal_existing-condition",
        (5600.0, 360.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_visible_existing.index = TREE_ID
    mask_visible_existing.use_antialiasing = True
    ensure_link(node_tree, existing.outputs["IndexOB"], mask_visible_existing.inputs["ID value"])

    mask_visible_trending = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{prefix}mask_visible-arboreal_trending",
        "mask_visible-arboreal_trending",
        (5600.0, 80.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_visible_trending.index = TREE_ID
    mask_visible_trending.use_antialiasing = True
    ensure_link(node_tree, trending.outputs["IndexOB"], mask_visible_trending.inputs["ID value"])

    phase_specs = (
        ("existing_condition", existing, mask_visible_existing.outputs["Alpha"], 0.0),
        ("pathway", pathway, mask_visible_pathway.outputs["Alpha"], -980.0),
        ("priority", priority, mask_visible_priority.outputs["Value"], -1960.0),
        ("trending", trending, mask_visible_trending.outputs["Alpha"], -2940.0),
    )
    output_base_path = "//outputs/current/sizes"

    for phase_slug, exr_node, visible_mask_socket, phase_y in phase_specs:
        phase_frame = new_node(
            node_tree,
            "NodeFrame",
            f"{prefix}{phase_slug}::Frame",
            phase_slug,
            (6120.0, phase_y + 720.0),
            parent=frame,
        )
        phase_frame.use_custom_color = True
        phase_frame.color = (0.12, 0.14, 0.12)

        size_socket = socket_by_name(exr_node, "size")
        colored_outputs: dict[str, object] = {}

        for index, spec in enumerate(SIZE_SPECS):
            class_y = phase_y + 520.0 - index * 120.0
            base_name = f"{prefix}{phase_slug}_{spec['slug']}"
            matched_socket = build_value_match_socket(
                node_tree,
                size_socket,
                int(spec["value"]),
                base_name,
                6320.0,
                class_y,
                phase_frame,
            )
            masked_socket = build_multiply_socket(
                node_tree,
                visible_mask_socket,
                matched_socket,
                f"{base_name} :: masked",
                7040.0,
                class_y,
                phase_frame,
            )
            color_socket = build_color_alpha_image(
                node_tree,
                hex_to_linear_rgba(str(spec["hex"])),
                masked_socket,
                base_name,
                7280.0,
                class_y,
                phase_frame,
            )
            colored_outputs[str(spec["slug"])] = color_socket

            output_node = new_node(
                node_tree,
                "CompositorNodeOutputFile",
                f"{prefix}Output :: {phase_slug}_{spec['slug']}",
                f"Output :: {phase_slug}_{spec['slug']}",
                (7800.0, class_y),
                parent=phase_frame,
                color=(0.12, 0.20, 0.14),
            )
            configure_png_output_node(output_node, output_base_path, f"{phase_slug}_{spec['slug']}")
            ensure_link(node_tree, color_socket, output_node.inputs[0])

        combined_socket = colored_outputs[str(SIZE_SPECS[-1]["slug"])]
        combine_y = phase_y - 420.0
        for index, spec in enumerate(reversed(SIZE_SPECS[:-1]), start=1):
            combine_slug = str(spec["slug"])
            alpha_over = new_node(
                node_tree,
                "CompositorNodeAlphaOver",
                f"{prefix}{phase_slug} :: combine_{combine_slug}",
                f"combine_{combine_slug}",
                (7520.0 + (index - 1) * 220.0, combine_y),
                parent=phase_frame,
                color=(0.14, 0.20, 0.16),
            )
            alpha_over.premul = 1.0
            ensure_link(node_tree, combined_socket, alpha_over.inputs[1])
            ensure_link(node_tree, colored_outputs[combine_slug], alpha_over.inputs[2])
            combined_socket = alpha_over.outputs["Image"]

        combined_output = new_node(
            node_tree,
            "CompositorNodeOutputFile",
            f"{prefix}Output :: {phase_slug}_size_combined",
            f"Output :: {phase_slug}_size_combined",
            (9120.0, combine_y),
            parent=phase_frame,
            color=(0.12, 0.20, 0.14),
        )
        configure_png_output_node(combined_output, output_base_path, f"{phase_slug}_size_combined")
        ensure_link(node_tree, combined_socket, combined_output.inputs[0])


def add_current_proposals_branch(scene: bpy.types.Scene) -> None:
    if scene.node_tree is None:
        raise ValueError("Current scene has no compositor node tree")

    prefix = "Proposals::"
    node_tree = scene.node_tree
    remove_prefixed_nodes(node_tree, prefix)

    mask_pathway_source = node_tree.nodes.get("AO::EXR Pathway")
    mask_priority_source = node_tree.nodes.get("AO::EXR Priority")
    mask_trending_source = node_tree.nodes.get("Resources::EXR Trending")
    if mask_pathway_source is None or mask_priority_source is None or mask_trending_source is None:
        raise ValueError("Current scene is missing EXR inputs for proposal visibility masks")

    frame = new_node(
        node_tree,
        "NodeFrame",
        f"{prefix}FamilyFrame",
        "Proposals",
        (5400.0, 3400.0),
    )
    frame.use_custom_color = True
    frame.color = (0.18, 0.16, 0.12)

    proposal_pathway = new_node(
        node_tree,
        "CompositorNodeImage",
        f"{prefix}EXR Pathway",
        "EXR Pathway",
        (5600.0, 3380.0),
        parent=frame,
        color=(0.12, 0.18, 0.10),
    )
    proposal_pathway.image = bpy.data.images.load(str(PROPOSAL_PATHWAY_EXR), check_existing=True)

    proposal_priority = new_node(
        node_tree,
        "CompositorNodeImage",
        f"{prefix}EXR Priority",
        "EXR Priority",
        (5600.0, 3300.0),
        parent=frame,
        color=(0.12, 0.18, 0.10),
    )
    proposal_priority.image = bpy.data.images.load(str(PROPOSAL_PRIORITY_EXR), check_existing=True)

    proposal_trending = new_node(
        node_tree,
        "CompositorNodeImage",
        f"{prefix}EXR Trending",
        "EXR Trending",
        (5600.0, 3220.0),
        parent=frame,
        color=(0.12, 0.18, 0.10),
    )
    proposal_trending.image = bpy.data.images.load(str(PROPOSAL_TRENDING_EXR), check_existing=True)

    mask_visible_pathway = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{prefix}mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (5600.0, 3120.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_visible_pathway.index = TREE_ID
    mask_visible_pathway.use_antialiasing = True
    ensure_link(node_tree, mask_pathway_source.outputs["IndexOB"], mask_visible_pathway.inputs["ID value"])

    mask_all_priority = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{prefix}mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (5600.0, 2840.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_all_priority.index = TREE_ID
    mask_all_priority.use_antialiasing = True
    ensure_link(node_tree, mask_priority_source.outputs["IndexOB"], mask_all_priority.inputs["ID value"])

    mask_visible_priority = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{prefix}mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (5840.0, 2960.0),
        parent=frame,
        color=(0.18, 0.16, 0.20),
    )
    mask_visible_priority.operation = "MULTIPLY"
    mask_visible_priority.use_clamp = True
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])

    mask_visible_trending = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{prefix}mask_visible-arboreal_trending",
        "mask_visible-arboreal_trending",
        (5600.0, 2560.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_visible_trending.index = TREE_ID
    mask_visible_trending.use_antialiasing = True
    ensure_link(node_tree, mask_trending_source.outputs["IndexOB"], mask_visible_trending.inputs["ID value"])

    phase_specs = (
        ("pathway", proposal_pathway, mask_visible_pathway.outputs["Alpha"], 0.0),
        ("priority", proposal_priority, mask_visible_priority.outputs["Value"], -1700.0),
        ("trending", proposal_trending, mask_visible_trending.outputs["Alpha"], -3400.0),
    )

    output_base_path = "//outputs/current/proposals"

    for phase_slug, exr_node, visible_mask_socket, phase_y in phase_specs:
        phase_frame = new_node(
            node_tree,
            "NodeFrame",
            f"{prefix}{phase_slug}::Frame",
            phase_slug,
            (6120.0, phase_y + 2920.0),
            parent=frame,
        )
        phase_frame.use_custom_color = True
        phase_frame.color = (0.12, 0.14, 0.12)

        channel_y = phase_y + 2720.0
        for family_name, interventions in PROPOSAL_BRANCH_SPECS:
            family_frame = new_node(
                node_tree,
                "NodeFrame",
                f"{prefix}{phase_slug}::{family_name}::Frame",
                family_name,
                (6280.0, channel_y + 80.0),
                parent=phase_frame,
            )
            family_frame.use_custom_color = True
            family_frame.color = (0.14, 0.12, 0.12)

            proposal_socket = socket_by_name(exr_node, family_name)
            for idx, (value, intervention_slug, hex_value) in enumerate(interventions):
                class_y = channel_y - idx * 120.0
                base_name = f"{prefix}{phase_slug}::{family_name}::{intervention_slug}"
                matched_socket = build_value_match_socket(
                    node_tree,
                    proposal_socket,
                    int(value),
                    base_name,
                    6480.0,
                    class_y,
                    family_frame,
                )
                masked_socket = build_multiply_socket(
                    node_tree,
                    visible_mask_socket,
                    matched_socket,
                    f"{base_name} :: masked",
                    7200.0,
                    class_y,
                    family_frame,
                )
                color_socket = build_color_alpha_image(
                    node_tree,
                    hex_to_linear_rgba(str(hex_value)),
                    masked_socket,
                    base_name,
                    7440.0,
                    class_y,
                    family_frame,
                )
                stem = f"{phase_slug}/proposal-{family_name.replace('proposal-', '')}-{intervention_slug}"
                output_node = new_node(
                    node_tree,
                    "CompositorNodeOutputFile",
                    f"{prefix}Output :: {phase_slug}::{family_name}::{intervention_slug}",
                    f"Output :: {phase_slug}::{family_name}::{intervention_slug}",
                    (7960.0, class_y),
                    parent=family_frame,
                    color=(0.12, 0.20, 0.14),
                )
                configure_png_output_node(output_node, output_base_path, stem)
                ensure_link(node_tree, color_socket, output_node.inputs[0])
            channel_y -= 460.0


def add_current_bioenvelope_branch(
    current: bpy.types.Scene,
    legacy: bpy.types.Scene,
) -> None:
    if current.node_tree is None or legacy.node_tree is None:
        raise ValueError("Current or Legacy scene has no compositor node tree")

    prefix = "Current BioEnvelope :: "
    node_tree = current.node_tree
    remove_prefixed_nodes(node_tree, prefix)

    existing = node_tree.nodes.get("AO::EXR Existing")
    if existing is None:
        raise ValueError("Current scene is missing AO::EXR Existing for bioenvelope branch")

    legacy_nodes = legacy.node_tree.nodes
    legacy_bio = legacy_nodes.get("EXR :: bioenvelope")
    legacy_trending = legacy_nodes.get("EXR :: trending_state")
    palette_group = find_required_group("WORLD_BIOENVELOPE_PALETTE")
    if legacy_bio is None or legacy_bio.image is None:
        raise ValueError("Legacy scene is missing EXR :: bioenvelope image")
    if legacy_trending is None or legacy_trending.image is None:
        raise ValueError("Legacy scene is missing EXR :: trending_state image")

    frame = new_node(
        node_tree,
        "NodeFrame",
        f"{prefix}Frame",
        "bioenvelope_outputs",
        (5400.0, -2500.0),
    )
    frame.use_custom_color = True
    frame.color = (0.10, 0.12, 0.10)

    exr_bio = new_node(
        node_tree,
        "CompositorNodeImage",
        f"{prefix}EXR BioEnvelope",
        "EXR BioEnvelope",
        (5580.0, -2100.0),
        parent=frame,
        color=(0.16, 0.18, 0.22),
    )
    exr_bio.image = legacy_bio.image

    exr_trending = new_node(
        node_tree,
        "CompositorNodeImage",
        f"{prefix}EXR Trending",
        "EXR Trending",
        (5580.0, -2380.0),
        parent=frame,
        color=(0.16, 0.18, 0.22),
    )
    exr_trending.image = legacy_trending.image

    existing_palette = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{prefix}Base BioEnvelope Palette",
        "Base BioEnvelope Palette",
        (5880.0, -1980.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    existing_palette.node_tree = palette_group
    apply_bioenvelope_palette_defaults(existing_palette)
    ensure_link(
        node_tree,
        existing.outputs["world_design_bioenvelope"],
        existing_palette.inputs["WorldBioEnvelope"],
    )

    envelope_palette = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{prefix}Envelope Palette",
        "Envelope Palette",
        (5880.0, -2240.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    envelope_palette.node_tree = palette_group
    apply_bioenvelope_palette_defaults(envelope_palette)
    ensure_link(node_tree, exr_bio.outputs["bioEnvelopeType"], envelope_palette.inputs["WorldBioEnvelope"])

    trending_palette = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{prefix}Trending Palette",
        "Trending Palette",
        (5880.0, -2500.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    trending_palette.node_tree = palette_group
    apply_bioenvelope_palette_defaults(trending_palette)
    ensure_link(
        node_tree,
        exr_trending.outputs["bioEnvelopeType"],
        trending_palette.inputs["WorldBioEnvelope"],
    )

    env_mask = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{prefix}Envelope Voxels Mask",
        "Envelope Voxels Mask",
        (5880.0, -2780.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    env_mask.operation = "GREATER_THAN"
    env_mask.inputs[1].default_value = 0.0
    ensure_link(
        node_tree,
        existing.outputs["world_design_bioenvelope"],
        env_mask.inputs[0],
    )

    envelope_voxels = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{prefix}Envelope Voxels",
        "Envelope Voxels",
        (6160.0, -2780.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    envelope_voxels.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, existing.outputs["Image"], envelope_voxels.inputs["Image"])
    ensure_link(node_tree, env_mask.outputs["Value"], envelope_voxels.inputs["Alpha"])

    palette_socket_specs = (
        ("Image", "full-image"),
        ("1 Exoskeleton", "exoskeleton"),
        ("2 BrownRoof", "brownroof"),
        ("3 OtherGround", "otherground"),
        ("4 Rewilded", "rewilded"),
        ("5 FootprintDepaved", "footprintdepaved"),
        ("6 LivingFacade", "livingfacade"),
        ("7 GreenRoof", "greenroof"),
    )

    reroute_x = 6460.0
    current_y = -1880.0
    for palette_node, stem_prefix in (
        (existing_palette, "base_bioenvelope"),
        (envelope_palette, "bioenvelope"),
        (trending_palette, "trending_bioenvelope"),
    ):
        for socket_name, suffix in palette_socket_specs:
            label = f"{stem_prefix}_{suffix}"
            reroute = new_node(
                node_tree,
                "NodeReroute",
                f"{prefix}{label}",
                label,
                (reroute_x, current_y),
                parent=frame,
            )
            ensure_link(node_tree, palette_node.outputs[socket_name], reroute.inputs[0])
            current_y -= 110.0
        current_y -= 40.0

    envelope_voxels_reroute = new_node(
        node_tree,
        "NodeReroute",
        f"{prefix}envelope_voxels",
        "envelope_voxels",
        (reroute_x, current_y),
        parent=frame,
    )
    ensure_link(node_tree, envelope_voxels.outputs["Image"], envelope_voxels_reroute.inputs[0])

    def configure_output_node(node: bpy.types.Node, slot_name: str) -> None:
        node.base_path = "//outputs/current/bioenvelope"
        node.format.file_format = "PNG"
        node.format.color_mode = "RGBA"
        node.format.color_depth = "8"
        while len(node.file_slots) > 1:
            node.file_slots.remove(node.file_slots[-1])
        node.file_slots[0].path = f"{slot_name}_"

    output_x = 6760.0
    output_y = -1880.0
    for palette_node, stem_prefix in (
        (existing_palette, "base_bioenvelope"),
        (envelope_palette, "bioenvelope"),
        (trending_palette, "trending_bioenvelope"),
    ):
        for socket_name, suffix in palette_socket_specs:
            stem = f"{stem_prefix}_{suffix}"
            output_node = new_node(
                node_tree,
                "CompositorNodeOutputFile",
                f"{prefix}Output :: {stem}",
                f"Output :: {stem}",
                (output_x, output_y),
                parent=frame,
                color=(0.12, 0.20, 0.14),
            )
            configure_output_node(output_node, stem)
            ensure_link(node_tree, palette_node.outputs[socket_name], output_node.inputs[0])
            output_y -= 110.0
        output_y -= 40.0

    envelope_voxels_output = new_node(
        node_tree,
        "CompositorNodeOutputFile",
        f"{prefix}Output :: envelope_voxels",
        "Output :: envelope_voxels",
        (output_x, output_y),
        parent=frame,
        color=(0.12, 0.20, 0.14),
    )
    configure_output_node(envelope_voxels_output, "envelope_voxels")
    ensure_link(node_tree, envelope_voxels.outputs["Image"], envelope_voxels_output.inputs[0])


def add_current_shading_branch(scene: bpy.types.Scene) -> None:
    if scene.node_tree is None:
        raise ValueError("Current scene has no compositor node tree")

    prefix = "Current Shading :: "
    node_tree = scene.node_tree
    remove_prefixed_nodes(node_tree, prefix)
    ao_group = find_ao_group()

    pathway = node_tree.nodes.get("AO::EXR Pathway")
    priority = node_tree.nodes.get("AO::EXR Priority")
    existing = node_tree.nodes.get("AO::EXR Existing")
    if pathway is None or priority is None or existing is None:
        raise ValueError("Current scene is missing AO EXR inputs for shading branch")

    frame = new_node(
        node_tree,
        "NodeFrame",
        f"{prefix}Frame",
        "Shading",
        (3800.0, -1400.0),
    )
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)

    mask_visible_pathway = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{prefix}mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (4020.0, -1020.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_visible_pathway.index = 3
    mask_visible_pathway.use_antialiasing = True
    ensure_link(node_tree, pathway.outputs["IndexOB"], mask_visible_pathway.inputs["ID value"])

    mask_all_priority = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{prefix}mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (4020.0, -1280.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_all_priority.index = 3
    mask_all_priority.use_antialiasing = True
    ensure_link(node_tree, priority.outputs["IndexOB"], mask_all_priority.inputs["ID value"])

    mask_visible_priority = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{prefix}mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (4260.0, -1160.0),
        parent=frame,
        color=(0.18, 0.16, 0.20),
    )
    mask_visible_priority.operation = "MULTIPLY"
    mask_visible_priority.use_clamp = True
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])

    pathway_shading = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{prefix}Pathway shading",
        "Pathway shading",
        (4500.0, -960.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    pathway_shading.node_tree = ao_group
    ensure_link(node_tree, pathway.outputs["AO"], pathway_shading.inputs["Image"])
    ensure_link(node_tree, pathway.outputs["Normal"], pathway_shading.inputs["Normal"])
    ensure_link(node_tree, pathway.outputs["Alpha"], pathway_shading.inputs["Alpha"])

    priority_shading = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{prefix}Priority shading",
        "Priority shading",
        (4500.0, -1200.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    priority_shading.node_tree = ao_group
    ensure_link(node_tree, priority.outputs["AO"], priority_shading.inputs["Image"])
    ensure_link(node_tree, priority.outputs["Normal"], priority_shading.inputs["Normal"])
    ensure_link(node_tree, priority.outputs["Alpha"], priority_shading.inputs["Alpha"])

    existing_shading = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{prefix}Existing Condition shading",
        "Existing Condition shading",
        (4500.0, -1440.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    existing_shading.node_tree = ao_group
    ensure_link(node_tree, existing.outputs["AO"], existing_shading.inputs["Image"])
    ensure_link(node_tree, existing.outputs["Normal"], existing_shading.inputs["Normal"])
    ensure_link(node_tree, existing.outputs["Alpha"], existing_shading.inputs["Alpha"])

    pathway_masked = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{prefix}Pathway shading masked",
        "Pathway shading masked",
        (4740.0, -960.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    pathway_masked.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, pathway_shading.outputs[0], pathway_masked.inputs["Image"])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], pathway_masked.inputs["Alpha"])

    priority_masked = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{prefix}Priority shading masked",
        "Priority shading masked",
        (4740.0, -1200.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    priority_masked.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, priority_shading.outputs[0], priority_masked.inputs["Image"])
    ensure_link(node_tree, mask_visible_priority.outputs["Value"], priority_masked.inputs["Alpha"])

    existing_masked = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{prefix}Existing Condition shading masked",
        "Existing Condition shading masked",
        (4740.0, -1440.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    existing_masked.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, existing_shading.outputs[0], existing_masked.inputs["Image"])
    ensure_link(node_tree, existing.outputs["Alpha"], existing_masked.inputs["Alpha"])

    base_ao = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{prefix}Base AO",
        "Existing Condition AO Full",
        (4500.0, -1680.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    base_ao.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, existing.outputs["AO"], base_ao.inputs["Image"])
    ensure_link(node_tree, existing.outputs["Alpha"], base_ao.inputs["Alpha"])

    outputs = (
        (pathway_masked.outputs["Image"], "pathway_shading", -960.0),
        (priority_masked.outputs["Image"], "priority_shading", -1200.0),
        (existing_masked.outputs["Image"], "existing_condition_shading", -1440.0),
        (base_ao.outputs["Image"], "existing_condition_ao_full", -1680.0),
    )
    for image_socket, stem, y in outputs:
        output_node = new_node(
            node_tree,
            "CompositorNodeOutputFile",
            f"{prefix}Output :: {stem}",
            f"Output :: {stem}",
            (5000.0, y),
            parent=frame,
            color=(0.12, 0.20, 0.14),
        )
        output_node.base_path = "//outputs/current/shading"
        output_node.format.file_format = "PNG"
        output_node.format.color_mode = "RGBA"
        output_node.format.color_depth = "8"
        output_node.file_slots[0].path = f"{stem}_"
        ensure_link(node_tree, image_socket, output_node.inputs[0])


def purge_unused_data() -> None:
    for _ in range(5):
        result = bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_linked_ids=True,
            do_recursive=True,
        )
        if result != {"FINISHED"}:
            break


def main() -> None:
    clear_state()
    mist_helper = load_current_mist_helper()

    current = append_scene(CURRENT_SOURCE_BLEND, CURRENT_SOURCE_SCENE)
    current.name = "Current"
    current["edge_lab_role"] = "current_layout_reference"
    current["edge_lab_source_blend"] = str(CURRENT_SOURCE_BLEND)
    current["edge_lab_source_scene"] = CURRENT_SOURCE_SCENE
    normalize_scene(current)

    legacy = append_scene(LEGACY_SOURCE_BLEND, LEGACY_SOURCE_SCENE)
    legacy.name = "Legacy"
    legacy["edge_lab_role"] = "legacy_classic_lightweight"
    legacy["edge_lab_source_blend"] = str(LEGACY_SOURCE_BLEND)
    legacy["edge_lab_source_scene"] = LEGACY_SOURCE_SCENE
    normalize_scene(legacy)

    add_current_base_outputs_branch(current)
    add_current_sizes_branch(current)
    add_current_proposals_branch(current)
    add_current_bioenvelope_branch(current, legacy)
    add_current_shading_branch(current)
    mist_helper.ensure_current_mist_exr_branch(current)

    for scene in list(bpy.data.scenes):
        if scene not in {current, legacy}:
            bpy.data.scenes.remove(scene)

    bpy.context.window.scene = current
    purge_unused_data()

    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))
    log(f"Saved {OUTPUT_BLEND}")


if __name__ == "__main__":
    main()
