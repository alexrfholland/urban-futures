from pathlib import Path
import sys

import bpy


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

from refactor_code.paths import hook_tree_ply_library_dir

DEFAULT_BLEND_PATH = REPO_ROOT / "data/blender/2026/2026 futures heroes5.blend"
TREE_PLY_FOLDER = hook_tree_ply_library_dir()

TARGET_SCENES = ("city",)
PRIMARY_VIEW_LAYER = "pathway_state"
BACKUP_MATERIAL_NAME = "SS_RESOURCES"
RESOURCE_MATERIAL_NAME = "RESOURCES"
SCAN_TREE_PLYS = False

MATERIAL_PREFIX = "SS Resource Binary :: "
COMP_PREFIX = "SS Resource Binary Stack :: "
COMP_FRAME_NAME = f"{COMP_PREFIX}Frame"
COMP_FRAME_LABEL = "Resource Binary Masks"
MATERIAL_FRAME_NAME = f"{MATERIAL_PREFIX}Frame"
MATERIAL_FRAME_LABEL = "Resource Binary AOVs"

RESOURCE_SPECS = [
    {
        "label": "None",
        "slug": "none",
        "display_name": "None",
        "value": 1.0,
        "ply_property": "resource_other",
        "aov_name": "resource_none_mask",
        "color": (0.619608, 0.619608, 0.619608, 1.0),
    },
    {
        "label": "Dead Branch",
        "slug": "dead_branch",
        "display_name": "Dead Branch",
        "value": 2.0,
        "ply_property": "resource_dead branch",
        "aov_name": "resource_dead_branch_mask",
        "color": (0.129412, 0.588235, 0.952941, 1.0),
    },
    {
        "label": "Peeling Bark",
        "slug": "peeling_bark",
        "display_name": "Peeling Bark",
        "value": 3.0,
        "ply_property": "resource_peeling bark",
        "aov_name": "resource_peeling_bark_mask",
        "color": (1.0, 0.921569, 0.231373, 1.0),
    },
    {
        "label": "Perch Branch",
        "slug": "perch_branch",
        "display_name": "Perch Branch",
        "value": 4.0,
        "ply_property": "resource_perch branch",
        "aov_name": "resource_perch_branch_mask",
        "color": (1.0, 0.596078, 0.0, 1.0),
    },
    {
        "label": "Epiphyte",
        "slug": "epiphyte",
        "display_name": "Epiphyte",
        "value": 5.0,
        "ply_property": "resource_epiphyte",
        "aov_name": "resource_epiphyte_mask",
        "color": (0.545098, 0.764706, 0.290196, 1.0),
    },
    {
        "label": "Fallen Log",
        "slug": "fallen_log",
        "display_name": "Fallen Log",
        "value": 6.0,
        "ply_property": "resource_fallen log",
        "aov_name": "resource_fallen_log_mask",
        "color": (0.47451, 0.333333, 0.282353, 1.0),
    },
    {
        "label": "Hollow",
        "slug": "hollow",
        "display_name": "Hollow",
        "value": 7.0,
        "ply_property": "resource_hollow",
        "aov_name": "resource_hollow_mask",
        "color": (0.611765, 0.152941, 0.690196, 1.0),
    },
]

TREE_MASK_AOV_NAME = "resource_tree_mask"
RESOURCE_SCALAR_AOV_NAME = "resource"
RESOURCE_COLOUR_AOV_NAME = "resource_colour"


def get_blend_path():
    for arg in reversed(sys.argv):
        if arg.endswith(".blend"):
            return Path(arg)
    return DEFAULT_BLEND_PATH


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_node(node_tree, bl_idname, name, label, location, parent=None):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != bl_idname:
        if node is not None:
            node_tree.nodes.remove(node)
        node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if parent is not None:
        node.parent = parent
    return node


def ensure_rgb_node(node_tree, name, label, location, color, parent=None):
    node = ensure_node(node_tree, "ShaderNodeRGB", name, label, location, parent)
    node.outputs[0].default_value = color
    return node


def ensure_math_threshold(node_tree, name, label, location, operation, threshold, parent=None):
    node = ensure_node(node_tree, "ShaderNodeMath", name, label, location, parent)
    node.operation = operation
    node.inputs[1].default_value = threshold
    return node


def get_socket(node, socket_name, fallback_index=0, output=True):
    sockets = node.outputs if output else node.inputs
    socket = sockets.get(socket_name)
    if socket is not None:
        return socket
    return sockets[fallback_index]


def read_ply_vertex_properties(ply_path):
    props = []
    with open(ply_path, "rb") as handle:
        in_vertex_element = False
        while True:
            raw = handle.readline()
            if not raw:
                break
            line = raw.decode("latin1", errors="replace").strip()
            if line == "end_header":
                break
            if line.startswith("element "):
                in_vertex_element = line == "element vertex " + line.split()[-1]
                continue
            if in_vertex_element and line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    props.append(" ".join(parts[2:]))
    return props


def confirm_tree_ply_binary_resources():
    tree_plys = sorted(TREE_PLY_FOLDER.glob("*.ply"))
    if not tree_plys:
        raise ValueError(f"No tree PLYs found in {TREE_PLY_FOLDER}")
    required = [spec["ply_property"] for spec in RESOURCE_SPECS]
    with_raw = []
    missing_raw = []
    for ply_path in tree_plys:
        props = read_ply_vertex_properties(ply_path)
        missing = [name for name in required if name not in props]
        if missing:
            missing_raw.append((ply_path, missing))
        else:
            with_raw.append(ply_path)

    print(f"Tree PLYs with raw resource binary properties: {len(with_raw)}")
    print(f"Tree PLYs missing some/all raw resource binary properties: {len(missing_raw)}")
    if with_raw:
        print(f"Sample tree PLY with raw binaries: {with_raw[0]}")
    if missing_raw:
        print(f"Sample tree PLY requiring fallback: {missing_raw[0][0]}")
    return {
        "with_raw": with_raw,
        "missing_raw": missing_raw,
    }


def backup_resources_material():
    src = bpy.data.materials.get(RESOURCE_MATERIAL_NAME)
    if src is None:
        raise ValueError(f"Material '{RESOURCE_MATERIAL_NAME}' not found.")

    backup = bpy.data.materials.get(BACKUP_MATERIAL_NAME)
    if backup is None:
        backup = src.copy()
        backup.name = BACKUP_MATERIAL_NAME
        print(f"Created material backup: {BACKUP_MATERIAL_NAME}")
    else:
        print(f"Material backup already exists: {BACKUP_MATERIAL_NAME}")
    backup.use_fake_user = True
    return src, backup


def ensure_view_layer_aov(view_layer, aov_name, aov_type):
    for existing in view_layer.aovs:
        if existing.name == aov_name:
            if existing.type != aov_type:
                existing.type = aov_type
            return existing
    item = view_layer.aovs.add()
    item.name = aov_name
    item.type = aov_type
    return item


def ensure_all_view_layer_aovs():
    value_names = [RESOURCE_SCALAR_AOV_NAME, TREE_MASK_AOV_NAME] + [spec["aov_name"] for spec in RESOURCE_SPECS]
    for scene_name in TARGET_SCENES:
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            continue
        view_layer = scene.view_layers.get(PRIMARY_VIEW_LAYER)
        if view_layer is None:
            continue
        ensure_view_layer_aov(view_layer, RESOURCE_COLOUR_AOV_NAME, "COLOR")
        for name in value_names:
            ensure_view_layer_aov(view_layer, name, "VALUE")


def cleanup_material_nodes(node_tree):
    prefixes = (
        "SS Resource ",
        MATERIAL_PREFIX,
    )
    for node in list(node_tree.nodes):
        if node.name.startswith(prefixes):
            node_tree.nodes.remove(node)


def patch_resources_material(material):
    if not material.use_nodes or material.node_tree is None:
        raise ValueError(f"Material '{material.name}' does not use nodes.")

    nt = material.node_tree
    cleanup_material_nodes(nt)
    frame = ensure_node(
        nt,
        "NodeFrame",
        MATERIAL_FRAME_NAME,
        MATERIAL_FRAME_LABEL,
        (840.0, 640.0),
    )
    normalize_frame(frame)

    attr = next(
        (n for n in nt.nodes if n.bl_idname == "ShaderNodeAttribute" and n.attribute_name == "int_resource"),
        None,
    )
    if attr is None:
        attr = ensure_node(
            nt,
            "ShaderNodeAttribute",
            f"{MATERIAL_PREFIX}int_resource",
            "int_resource",
            (0.0, 0.0),
            frame,
        )
        attr.attribute_name = "int_resource"
    else:
        attr.parent = frame
        attr.location = (0.0, 0.0)

    resource_aov = next(
        (n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputAOV" and n.aov_name == RESOURCE_SCALAR_AOV_NAME),
        None,
    )
    if resource_aov is None:
        resource_aov = ensure_node(
            nt,
            "ShaderNodeOutputAOV",
            f"{MATERIAL_PREFIX}resource",
            "resource",
            (1240.0, 440.0),
            frame,
        )
    resource_aov.aov_name = RESOURCE_SCALAR_AOV_NAME
    ensure_link(nt, attr.outputs["Fac"], resource_aov.inputs["Value"])

    resource_colour_aov = next(
        (n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputAOV" and n.aov_name == RESOURCE_COLOUR_AOV_NAME),
        None,
    )
    if resource_colour_aov is None:
        resource_colour_aov = ensure_node(
            nt,
            "ShaderNodeOutputAOV",
            f"{MATERIAL_PREFIX}resource_colour",
            "resource_colour",
            (1240.0, 360.0),
            frame,
        )
    resource_colour_aov.aov_name = RESOURCE_COLOUR_AOV_NAME

    x0 = 260.0
    y0 = 420.0
    y_step = -170.0

    black = ensure_rgb_node(
        nt,
        f"{MATERIAL_PREFIX}Black",
        "Base black",
        (x0 - 350.0, y0 + 60.0),
        (0.0, 0.0, 0.0, 1.0),
        frame,
    )
    current_color_socket = black.outputs["Color"]

    for index, spec in enumerate(RESOURCE_SPECS):
        y = y0 + (index * y_step)

        attr_mask = ensure_node(
            nt,
            "ShaderNodeAttribute",
            f"{MATERIAL_PREFIX}Attr::{spec['slug']}",
            spec["ply_property"],
            (x0, y),
            frame,
        )
        attr_mask.attribute_name = spec["ply_property"]

        rgb = ensure_rgb_node(
            nt,
            f"{MATERIAL_PREFIX}Colour::{spec['slug']}",
            f"{spec['display_name']} colour",
            (x0 + 220.0, y),
            spec["color"],
            frame,
        )

        mix = ensure_node(
            nt,
            "ShaderNodeMix",
            f"{MATERIAL_PREFIX}Mix::{spec['slug']}",
            f"Mix {spec['display_name']}",
            (x0 + 470.0, y),
            frame,
        )
        mix.data_type = "RGBA"
        mix.blend_type = "MIX"
        ensure_link(nt, attr_mask.outputs["Fac"], mix.inputs["Factor"])
        ensure_link(nt, current_color_socket, mix.inputs["A"])
        ensure_link(nt, rgb.outputs["Color"], mix.inputs["B"])
        current_color_socket = mix.outputs["Result"]

        aov = ensure_node(
            nt,
            "ShaderNodeOutputAOV",
            f"{MATERIAL_PREFIX}AOV::{spec['slug']}",
            spec["aov_name"],
            (x0 + 980.0, y),
            frame,
        )
        aov.aov_name = spec["aov_name"]
        ensure_link(nt, attr_mask.outputs["Fac"], aov.inputs["Value"])

    tree_mask = ensure_math_threshold(
        nt,
        f"{MATERIAL_PREFIX}TreeMask",
        "All trees mask",
        (x0 + 730.0, y0 - (len(RESOURCE_SPECS) * 90.0)),
        "GREATER_THAN",
        0.5,
        frame,
    )
    ensure_link(nt, attr.outputs["Fac"], tree_mask.inputs[0])

    tree_mask_aov = ensure_node(
        nt,
        "ShaderNodeOutputAOV",
        f"{MATERIAL_PREFIX}AOV::tree_mask",
        TREE_MASK_AOV_NAME,
        (x0 + 980.0, y0 - (len(RESOURCE_SPECS) * 90.0)),
        frame,
    )
    tree_mask_aov.aov_name = TREE_MASK_AOV_NAME
    ensure_link(nt, tree_mask.outputs["Value"], tree_mask_aov.inputs["Value"])
    ensure_link(nt, current_color_socket, resource_colour_aov.inputs["Color"])


def normalize_frame(frame):
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)
    frame.shrink = False


def normalize_existing_resource_frames(node_tree):
    for node in node_tree.nodes:
        if node.bl_idname != "NodeFrame":
            continue
        text = f"{node.name} {node.label}"
        if "Resource" in text or "Mask From Value List" in text or "Size Weighted Colour Mix" in text:
            normalize_frame(node)


def find_primary_render_layers_node(node_tree):
    preferred = [
        node for node in node_tree.nodes
        if node.bl_idname == "CompositorNodeRLayers"
        and getattr(node, "layer", "") == PRIMARY_VIEW_LAYER
        and not node.name.startswith("Snapshot Setup")
    ]
    if preferred:
        return preferred[0]

    fallback = [node for node in node_tree.nodes if node.bl_idname == "CompositorNodeRLayers"]
    if fallback:
        return fallback[0]

    raise ValueError("No Render Layers node found in compositor.")


def socket_or_fail(node, socket_name):
    socket = node.outputs.get(socket_name)
    if socket is None:
        raise ValueError(f"Socket '{socket_name}' not found on node '{node.name}'.")
    return socket


def remove_generated_compositor_nodes(node_tree):
    for node in list(node_tree.nodes):
        if node.name.startswith(COMP_PREFIX):
            node_tree.nodes.remove(node)


def patch_scene_compositor(scene):
    if not scene.use_nodes or scene.node_tree is None:
        raise ValueError(f"Scene '{scene.name}' does not use compositor nodes.")

    nt = scene.node_tree
    normalize_existing_resource_frames(nt)
    remove_generated_compositor_nodes(nt)

    render_layers = find_primary_render_layers_node(nt)
    base_x = render_layers.location.x + 900.0
    base_y = render_layers.location.y + 650.0
    row_step = -180.0

    frame = ensure_node(
        nt,
        "NodeFrame",
        COMP_FRAME_NAME,
        COMP_FRAME_LABEL,
        (base_x - 180.0, base_y + 220.0),
    )
    normalize_frame(frame)

    tree_mask_in = ensure_node(
        nt,
        "NodeReroute",
        f"{COMP_PREFIX}all_trees_mask_input",
        "all_trees_mask",
        (0.0, row_step * (len(RESOURCE_SPECS) + 1)),
        frame,
    )
    ensure_link(nt, socket_or_fail(render_layers, TREE_MASK_AOV_NAME), tree_mask_in.inputs[0])
    tree_mask_out = ensure_node(
        nt,
        "NodeReroute",
        f"{COMP_PREFIX}all_trees_mask_out",
        "all_trees_mask_out",
        (1140.0, row_step * (len(RESOURCE_SPECS) + 1)),
        frame,
    )
    ensure_link(nt, tree_mask_in.outputs[0], tree_mask_out.inputs[0])

    display_specs = list(reversed(RESOURCE_SPECS))
    coloured_outputs = {}
    row_positions = {}
    for index, spec in enumerate(display_specs):
        y = row_step * index
        row_positions[spec["slug"]] = y

        mask_in = ensure_node(
            nt,
            "NodeReroute",
            f"{COMP_PREFIX}{spec['slug']}::mask",
            f"{spec['slug']}_mask",
            (0.0, y),
            frame,
        )
        ensure_link(nt, socket_or_fail(render_layers, spec["aov_name"]), mask_in.inputs[0])

        color = ensure_node(
            nt,
            "CompositorNodeRGB",
            f"{COMP_PREFIX}{spec['slug']}::colour",
            f"{spec['display_name']} colour",
            (210.0, y),
            frame,
        )
        color.outputs[0].default_value = spec["color"]

        set_alpha = ensure_node(
            nt,
            "CompositorNodeSetAlpha",
            f"{COMP_PREFIX}{spec['slug']}::set_alpha",
            spec["display_name"],
            (430.0, y),
            frame,
        )
        set_alpha.mode = "APPLY"
        ensure_link(nt, get_socket(color, "Image", 0, output=True), get_socket(set_alpha, "Image", 0, output=False))
        ensure_link(nt, mask_in.outputs[0], get_socket(set_alpha, "Alpha", 1, output=False))

        coloured_out = ensure_node(
            nt,
            "NodeReroute",
            f"{COMP_PREFIX}{spec['slug']}::out",
            spec["slug"],
            (660.0, y),
            frame,
        )
        ensure_link(nt, set_alpha.outputs["Image"], coloured_out.inputs[0])
        coloured_outputs[spec["slug"]] = coloured_out.outputs[0]

    combined_socket = coloured_outputs["none"]
    for spec in RESOURCE_SPECS[1:]:
        y = row_positions[spec["slug"]]
        alpha_over = ensure_node(
            nt,
            "CompositorNodeAlphaOver",
            f"{COMP_PREFIX}{spec['slug']}::alpha_over",
            f"{spec['display_name']} over stack",
            (890.0, y),
            frame,
        )
        alpha_over.inputs[0].default_value = 1.0
        ensure_link(nt, combined_socket, get_socket(alpha_over, "Image", 1, output=False))
        ensure_link(nt, coloured_outputs[spec["slug"]], get_socket(alpha_over, "Image", 2, output=False))
        combined_socket = get_socket(alpha_over, "Image", 0, output=True)

    if combined_socket is None:
        raise ValueError("No resource rows were built in compositor.")

    combined_out = ensure_node(
        nt,
        "NodeReroute",
        f"{COMP_PREFIX}combined",
        "resources_combined",
        (1140.0, row_step * 1.5),
        frame,
    )
    ensure_link(nt, combined_socket, combined_out.inputs[0])


def main():
    blend_path = get_blend_path()
    ply_scan = {"with_raw": [], "missing_raw": []}
    if SCAN_TREE_PLYS:
        ply_scan = confirm_tree_ply_binary_resources()

    resources_material, _ = backup_resources_material()
    ensure_all_view_layer_aovs()
    patch_resources_material(resources_material)

    for scene_name in TARGET_SCENES:
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            print(f"Skipping missing scene: {scene_name}")
            continue
        patch_scene_compositor(scene)

    print(f"Patched current in-session blend: {blend_path}")
    if SCAN_TREE_PLYS:
        print(f"Tree PLYs with raw resource binaries: {len(ply_scan['with_raw'])}")
        print(f"Tree PLYs requiring int_resource fallback: {len(ply_scan['missing_raw'])}")
    else:
        print("Tree PLY scan skipped.")
    print("Save the blend manually if the result looks correct.")


if __name__ == "__main__":
    main()
