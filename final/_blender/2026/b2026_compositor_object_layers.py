import bpy


TARGET_SCENE_NAME = "parade-senescent"
RENDER_NODE_NAME = "Render Layers.001"
FRAME_NAME_PREFIX = "SplitLayers:"
PARENT_FRAME_NAME = f"{FRAME_NAME_PREFIX}Frame"
PARENT_FRAME_LABEL = "Object Index Split Layers"
CREATE_FILE_OUTPUT = False
FILE_OUTPUT_NODE_NAME = f"{FRAME_NAME_PREFIX}File Output"
FILE_OUTPUT_BASE_PATH = "//compositor_layers"

LAYER_SPECS = (
    {
        "key": "rewilded",
        "label": "Rewilded",
        "pass_index": 5,
        "color": (0.18, 0.28, 0.14),
    },
    {
        "key": "road",
        "label": "Road",
        "pass_index": 1,
        "color": (0.24, 0.20, 0.12),
    },
    {
        "key": "base",
        "label": "Base",
        "pass_index": 2,
        "color": (0.14, 0.18, 0.28),
    },
)


def require_scene(scene_name):
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' was not found.")
    if not scene.use_nodes or scene.node_tree is None:
        raise ValueError(f"Scene '{scene_name}' does not have compositor nodes enabled.")
    return scene


def require_render_layers_node(node_tree):
    node = node_tree.nodes.get(RENDER_NODE_NAME)
    if node is not None and node.bl_idname == "CompositorNodeRLayers":
        return node

    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeRLayers":
            return node

    raise ValueError("No Render Layers node was found in the compositor tree.")


def remove_existing_split_nodes(node_tree):
    for node in list(node_tree.nodes):
        if node.name.startswith(FRAME_NAME_PREFIX):
            node_tree.nodes.remove(node)


def new_node(node_tree, bl_idname, name, label, location, parent=None, color=None):
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


def ensure_parent_frame(node_tree, location):
    frame = new_node(
        node_tree,
        "NodeFrame",
        PARENT_FRAME_NAME,
        PARENT_FRAME_LABEL,
        location,
    )
    frame.use_custom_color = True
    frame.color = (0.13, 0.13, 0.13)
    frame.shrink = False
    return frame


def add_output_slot(file_output, slot_name):
    existing = file_output.file_slots.get(slot_name)
    if existing is not None:
        return existing
    return file_output.file_slots.new(slot_name)


def ensure_file_output(node_tree, parent_frame, location):
    if not CREATE_FILE_OUTPUT:
        return None

    file_output = new_node(
        node_tree,
        "CompositorNodeOutputFile",
        FILE_OUTPUT_NODE_NAME,
        "Split Layer EXRs",
        location,
        parent_frame,
        (0.16, 0.22, 0.14),
    )
    file_output.base_path = FILE_OUTPUT_BASE_PATH

    while len(file_output.file_slots) > 1:
        file_output.file_slots.remove(file_output.file_slots[-1])

    first_slot = file_output.file_slots[0]
    first_slot.path = "rewilded"
    first_slot.name = "rewilded"

    add_output_slot(file_output, "road").path = "road"
    add_output_slot(file_output, "base").path = "base"
    return file_output


def build_layer_branch(node_tree, render_node, parent_frame, spec, origin):
    frame = new_node(
        node_tree,
        "NodeFrame",
        f"{FRAME_NAME_PREFIX}{spec['label']} Frame",
        spec["label"],
        origin,
        parent_frame,
        spec["color"],
    )
    frame.shrink = False

    id_mask = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{FRAME_NAME_PREFIX}{spec['label']} Mask",
        f"{spec['label']} Mask",
        (origin[0] + 10, origin[1] - 20),
        frame,
        spec["color"],
    )
    id_mask.index = spec["pass_index"]
    id_mask.use_antialiasing = True

    set_alpha = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{FRAME_NAME_PREFIX}{spec['label']} Layer",
        f"{spec['label']} Layer",
        (origin[0] + 250, origin[1] - 20),
        frame,
        spec["color"],
    )
    set_alpha.mode = "APPLY"

    viewer = new_node(
        node_tree,
        "CompositorNodeViewer",
        f"{FRAME_NAME_PREFIX}{spec['label']} Viewer",
        f"{spec['label']} Viewer",
        (origin[0] + 500, origin[1] - 20),
        frame,
        spec["color"],
    )

    links = node_tree.links
    links.new(render_node.outputs["IndexOB"], id_mask.inputs["ID value"])
    links.new(render_node.outputs["Image"], set_alpha.inputs["Image"])
    links.new(id_mask.outputs["Alpha"], set_alpha.inputs["Alpha"])
    links.new(set_alpha.outputs["Image"], viewer.inputs["Image"])

    return set_alpha


def main():
    scene = require_scene(TARGET_SCENE_NAME)
    node_tree = scene.node_tree
    render_node = require_render_layers_node(node_tree)

    if "IndexOB" not in render_node.outputs:
        raise ValueError("Render Layers node does not expose IndexOB. Enable Object Index on the View Layer first.")

    remove_existing_split_nodes(node_tree)

    base_x = render_node.location.x + 450
    base_y = render_node.location.y + 500
    parent_frame = ensure_parent_frame(node_tree, (base_x - 80, base_y + 120))

    output_node = ensure_file_output(node_tree, parent_frame, (base_x + 760, base_y - 260))
    branch_outputs = {}

    for row, spec in enumerate(LAYER_SPECS):
        origin = (base_x, base_y - (row * 260))
        set_alpha = build_layer_branch(node_tree, render_node, parent_frame, spec, origin)
        branch_outputs[spec["key"]] = set_alpha

    if output_node is not None:
        for key, set_alpha in branch_outputs.items():
            slot_input = output_node.inputs.get(key)
            if slot_input is not None:
                node_tree.links.new(set_alpha.outputs["Image"], slot_input)

    print(f"Built compositor split layers in scene: {scene.name}")
    print(f"Render source node: {render_node.name}")
    for spec in LAYER_SPECS:
        print(f"- {spec['label']}: pass_index {spec['pass_index']}")
    if output_node is not None:
        print(f"File output base path: {output_node.base_path}")
    else:
        print("File Output node disabled; branches were added as viewer-ready nodes only.")


if __name__ == "__main__":
    main()
