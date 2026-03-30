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


def apply_bioenvelope_palette_defaults(node: bpy.types.Node) -> None:
    for input_name, value in BIOENVELOPE_PALETTE_DEFAULTS.items():
        socket = node.inputs.get(input_name)
        if socket is not None and not socket.is_linked:
            socket.default_value = value


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


def build_base_outputs_group() -> bpy.types.NodeTree:
    outliner_group = find_required_group("_OUTLINER.001")

    group = rebuild_group("EDGE_CURRENT_BASE_OUTPUTS")
    add_interface_socket(group, "Existing Image", "INPUT", "NodeSocketColor")
    add_interface_socket(group, "Existing AO", "INPUT", "NodeSocketFloat")
    add_interface_socket(group, "Existing Normal", "INPUT", "NodeSocketVector")
    add_interface_socket(group, "Existing Depth", "INPUT", "NodeSocketFloat")
    add_interface_socket(group, "world_sim_turns", "INPUT", "NodeSocketFloat")
    add_interface_socket(group, "world_sim_nodes", "INPUT", "NodeSocketFloat")

    add_interface_socket(group, "base_rgb", "OUTPUT", "NodeSocketColor")
    add_interface_socket(group, "base_white_render", "OUTPUT", "NodeSocketFloat")
    add_interface_socket(group, "base_outlines", "OUTPUT", "NodeSocketColor")
    add_interface_socket(group, "base_sim-turns", "OUTPUT", "NodeSocketFloat")
    add_interface_socket(group, "base_sim-nodes", "OUTPUT", "NodeSocketFloat")
    add_interface_socket(group, "base_sim-turns_ripple-effect", "OUTPUT", "NodeSocketColor")

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
    ensure_link(node_tree, existing.outputs["world_sim_turns"], base_group.inputs["world_sim_turns"])
    ensure_link(node_tree, existing.outputs["world_sim_nodes"], base_group.inputs["world_sim_nodes"])

    output_specs = (
        ("base_rgb", "base_rgb", -2100.0),
        ("base_white_render", "base_white_render", -2250.0),
        ("base_outlines", "base_outlines", -2400.0),
        ("base_sim-turns", "base_sim-turns", -2550.0),
        ("base_sim-nodes", "base_sim-nodes", -2700.0),
        ("base_sim-turns_ripple-effect", "base_sim-turns_ripple-effect", -2850.0),
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
