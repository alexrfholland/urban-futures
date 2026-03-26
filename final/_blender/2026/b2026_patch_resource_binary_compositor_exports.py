from pathlib import Path
import re

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
TARGET_SCENE_NAMES = ("city", "Habitat features - close ups")
PREFERRED_VIEW_LAYER_NAME = "pathway_state"
OUTPUT_BASE_PATH = REPO_ROOT / "data" / "blender" / "2026" / "resource_layer_pngs"

PREFIX = "SS Resource Export :: "
STACK_FRAME_NAME = f"{PREFIX}Stack Frame"
STACK_FRAME_LABEL = "Resource Binary Stack"
EXPORT_FRAME_NAME = f"{PREFIX}Export Frame"
EXPORT_FRAME_LABEL = "Arboreal Resource PNG Exports"
FILE_OUTPUT_NODE_NAME = f"{PREFIX}File Output"

RESOURCE_SPECS = (
    {
        "display_name": "None",
        "slug": "none",
        "aov_name": "resource_none_mask",
        "color": (0.619608, 0.619608, 0.619608, 1.0),
    },
    {
        "display_name": "Dead Branch",
        "slug": "dead_branch",
        "aov_name": "resource_dead_branch_mask",
        "color": (0.129412, 0.588235, 0.952941, 1.0),
    },
    {
        "display_name": "Peeling Bark",
        "slug": "peeling_bark",
        "aov_name": "resource_peeling_bark_mask",
        "color": (1.0, 0.921569, 0.231373, 1.0),
    },
    {
        "display_name": "Perch Branch",
        "slug": "perch_branch",
        "aov_name": "resource_perch_branch_mask",
        "color": (1.0, 0.596078, 0.0, 1.0),
    },
    {
        "display_name": "Epiphyte",
        "slug": "epiphyte",
        "aov_name": "resource_epiphyte_mask",
        "color": (0.545098, 0.764706, 0.290196, 1.0),
    },
    {
        "display_name": "Fallen Log",
        "slug": "fallen_log",
        "aov_name": "resource_fallen_log_mask",
        "color": (0.47451, 0.333333, 0.282353, 1.0),
    },
    {
        "display_name": "Hollow",
        "slug": "hollow",
        "aov_name": "resource_hollow_mask",
        "color": (0.611765, 0.152941, 0.690196, 1.0),
    },
)

RESOURCE_COLOUR_AOV = "resource_colour"
RESOURCE_SCALAR_AOV = "resource"
RESOURCE_TREE_MASK_AOV = "resource_tree_mask"


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


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


def normalize_frame(frame, color=(0.14, 0.14, 0.14)):
    frame.use_custom_color = True
    frame.color = color
    frame.shrink = False


def ensure_frame(node_tree, name, label, location, color):
    frame = ensure_node(node_tree, "NodeFrame", name, label, location)
    normalize_frame(frame, color)
    return frame


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


def ensure_scene_aovs(scene):
    value_names = [RESOURCE_SCALAR_AOV, RESOURCE_TREE_MASK_AOV] + [spec["aov_name"] for spec in RESOURCE_SPECS]
    for view_layer in scene.view_layers:
        ensure_view_layer_aov(view_layer, RESOURCE_COLOUR_AOV, "COLOR")
        for aov_name in value_names:
            ensure_view_layer_aov(view_layer, aov_name, "VALUE")


def find_scene(scene_name):
    exact = bpy.data.scenes.get(scene_name)
    if exact is not None:
        return exact
    folded = scene_name.casefold()
    for scene in bpy.data.scenes:
        if scene.name.casefold() == folded:
            return scene
    return None


def choose_view_layer(scene):
    preferred = scene.view_layers.get(PREFERRED_VIEW_LAYER_NAME)
    if preferred is not None:
        return preferred
    if scene.view_layers:
        return scene.view_layers[0]
    raise ValueError(f"Scene '{scene.name}' has no view layers.")


def remove_generated_nodes(node_tree):
    for node in list(node_tree.nodes):
        if node.name.startswith(PREFIX):
            node_tree.nodes.remove(node)


def find_or_create_render_layers_node(node_tree, view_layer_name):
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeRLayers" and getattr(node, "layer", None) == view_layer_name:
            return node

    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeRLayers":
            node.layer = view_layer_name
            return node

    node = node_tree.nodes.new("CompositorNodeRLayers")
    node.layer = view_layer_name
    node.location = (-300.0, 300.0)
    return node


def find_reference_output_node(node_tree):
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeComposite":
            return node
    return None


def socket_or_fail(node, socket_name):
    socket = node.outputs.get(socket_name)
    if socket is None:
        raise ValueError(f"Socket '{socket_name}' not found on node '{node.name}'.")
    return socket


def get_socket(node, socket_name, fallback_index=0, output=True):
    sockets = node.outputs if output else node.inputs
    socket = sockets.get(socket_name)
    if socket is not None:
        return socket
    return sockets[fallback_index]


def configure_file_output_node(file_output, scene_slug):
    OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)
    file_output.base_path = str(OUTPUT_BASE_PATH)
    file_output.format.file_format = "PNG"
    file_output.format.color_mode = "RGBA"
    file_output.format.color_depth = "8"
    file_output.format.compression = 15

    while len(file_output.file_slots) > 0:
        file_output.file_slots.remove(file_output.inputs[0])

    for spec in RESOURCE_SPECS:
        file_output.file_slots.new(spec["display_name"])
        file_output.file_slots[-1].path = f"{scene_slug}_resource_{spec['slug']}_"


def patch_scene(scene):
    ensure_scene_aovs(scene)

    if not scene.use_nodes:
        scene.use_nodes = True

    node_tree = scene.node_tree
    if node_tree is None:
        raise ValueError(f"Scene '{scene.name}' has no compositor node tree.")

    remove_generated_nodes(node_tree)

    view_layer = choose_view_layer(scene)
    render_layers = find_or_create_render_layers_node(node_tree, view_layer.name)
    output_reference = find_reference_output_node(node_tree)

    stack_origin_x = render_layers.location.x + 950.0
    stack_origin_y = render_layers.location.y + 640.0
    row_step = -190.0

    stack_frame = ensure_frame(
        node_tree,
        STACK_FRAME_NAME,
        STACK_FRAME_LABEL,
        (stack_origin_x - 220.0, stack_origin_y + 240.0),
        (0.14, 0.14, 0.14),
    )

    coloured_outputs = {}
    row_y_by_slug = {}
    reroute_one_x = 740.0
    alpha_x = 980.0

    for index, spec in enumerate(RESOURCE_SPECS):
        y = row_step * index
        row_y_by_slug[spec["slug"]] = y

        mask_reroute = ensure_node(
            node_tree,
            "NodeReroute",
            f"{PREFIX}Mask::{spec['slug']}",
            spec["display_name"],
            (0.0, y),
            stack_frame,
        )
        ensure_link(node_tree, socket_or_fail(render_layers, spec["aov_name"]), mask_reroute.inputs[0])

        colour_node = ensure_node(
            node_tree,
            "CompositorNodeRGB",
            f"{PREFIX}Colour::{spec['slug']}",
            f"{spec['display_name']} colour",
            (220.0, y),
            stack_frame,
        )
        colour_node.outputs[0].default_value = spec["color"]

        set_alpha = ensure_node(
            node_tree,
            "CompositorNodeSetAlpha",
            f"{PREFIX}SetAlpha::{spec['slug']}",
            spec["display_name"],
            (430.0, y),
            stack_frame,
        )
        set_alpha.mode = "APPLY"
        ensure_link(
            node_tree,
            get_socket(colour_node, "Image", 0, output=True),
            get_socket(set_alpha, "Image", 0, output=False),
        )
        ensure_link(
            node_tree,
            mask_reroute.outputs[0],
            get_socket(set_alpha, "Alpha", 1, output=False),
        )

        detour_one = ensure_node(
            node_tree,
            "NodeReroute",
            f"{PREFIX}Layer::{spec['slug']}",
            spec["display_name"],
            (reroute_one_x, y),
            stack_frame,
        )
        ensure_link(node_tree, get_socket(set_alpha, "Image", 0, output=True), detour_one.inputs[0])
        coloured_outputs[spec["slug"]] = detour_one.outputs[0]

    combined_socket = coloured_outputs["none"]
    for spec in RESOURCE_SPECS[1:]:
        alpha_over = ensure_node(
            node_tree,
            "CompositorNodeAlphaOver",
            f"{PREFIX}AlphaOver::{spec['slug']}",
            f"{spec['display_name']} over",
            (alpha_x, row_y_by_slug[spec["slug"]]),
            stack_frame,
        )
        alpha_over.inputs[0].default_value = 1.0
        ensure_link(node_tree, combined_socket, get_socket(alpha_over, "Image", 1, output=False))
        ensure_link(node_tree, coloured_outputs[spec["slug"]], get_socket(alpha_over, "Image", 2, output=False))
        combined_socket = get_socket(alpha_over, "Image", 0, output=True)

    combined_reroute = ensure_node(
        node_tree,
        "NodeReroute",
        f"{PREFIX}Combined",
        "resources_combined",
        (alpha_x + 250.0, row_step * 1.5),
        stack_frame,
    )
    ensure_link(node_tree, combined_socket, combined_reroute.inputs[0])

    if output_reference is not None:
        export_origin_x = output_reference.location.x - 460.0
        export_origin_y = output_reference.location.y + 460.0
    else:
        export_origin_x = stack_origin_x + 1650.0
        export_origin_y = stack_origin_y + 180.0

    export_frame = ensure_frame(
        node_tree,
        EXPORT_FRAME_NAME,
        EXPORT_FRAME_LABEL,
        (export_origin_x - 260.0, export_origin_y + 180.0),
        (0.16, 0.16, 0.16),
    )

    scene_slug = slugify(scene.name)
    file_output = ensure_node(
        node_tree,
        "CompositorNodeOutputFile",
        FILE_OUTPUT_NODE_NAME,
        "Arboreal Resource PNGs",
        (420.0, row_step * 2.2),
        export_frame,
    )
    configure_file_output_node(file_output, scene_slug)

    for index, spec in enumerate(RESOURCE_SPECS):
        y = row_step * index
        detour_two = ensure_node(
            node_tree,
            "NodeReroute",
            f"{PREFIX}Export::{spec['slug']}",
            spec["display_name"],
            (0.0, y),
            export_frame,
        )
        ensure_link(node_tree, coloured_outputs[spec["slug"]], detour_two.inputs[0])
        ensure_link(node_tree, detour_two.outputs[0], file_output.inputs[index])

    return {
        "scene": scene.name,
        "view_layer": view_layer.name,
        "output_dir": str(OUTPUT_BASE_PATH),
        "file_prefix": scene_slug,
    }


def main():
    results = []
    for scene_name in TARGET_SCENE_NAMES:
        scene = find_scene(scene_name)
        if scene is None:
            print(f"Skipped missing scene: {scene_name}")
            continue
        results.append(patch_scene(scene))

    if not results:
        raise ValueError("No target scenes were found.")

    for result in results:
        print(
            f"Patched scene '{result['scene']}' on view layer '{result['view_layer']}' "
            f"with resource exports to {result['output_dir']} using prefix {result['file_prefix']}_resource_*"
        )


if __name__ == "__main__":
    main()
