import bpy


TARGET_SCENE_NAME = "city"
PRIMARY_VIEW_LAYER = "pathway_state"
COMP_PREFIX = "SS Resource Binary Stack :: "
COMP_FRAME_NAME = f"{COMP_PREFIX}Frame"
COMP_FRAME_LABEL = "Resource Binary Masks"

RESOURCE_SPECS = [
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
]

TREE_MASK_AOV_NAME = "resource_tree_mask"


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


def normalize_frame(frame):
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)
    frame.shrink = False


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


def remove_generated_compositor_nodes(node_tree):
    for node in list(node_tree.nodes):
        if node.name.startswith(COMP_PREFIX):
            node_tree.nodes.remove(node)


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


def patch_scene_compositor(scene):
    if not scene.use_nodes or scene.node_tree is None:
        raise ValueError(f"Scene '{scene.name}' does not use compositor nodes.")

    nt = scene.node_tree
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
    scene = bpy.data.scenes.get(TARGET_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{TARGET_SCENE_NAME}' not found.")
    patch_scene_compositor(scene)
    print(f"Patched city compositor resource binary frame in scene '{scene.name}'.")


if __name__ == "__main__":
    main()
