import bpy


SAVE_FILE = False


SCENE_AOV_MAP = {
    "parade": [
        "structure_id",
        "resource",
        "size",
        "control",
        "node_type",
        "tree_interventions",
        "tree_proposals",
        "improvement",
        "canopy_resistance",
        "node_id",
        "instance_id",
        "isSenescent",
        "resource_colour",
        "isTerminal",
        "bioEnvelopeType",
        "sim_Turns",
        "bioSimple",
        "world_sim_turns",
        "world_sim_nodes",
        "world_design_bioenvelope",
        "world_design_bioenvelope_simple",
        "world_sim_matched",
    ],
    "city": [
        "structure_id",
        "resource",
        "size",
        "control",
        "node_type",
        "tree_interventions",
        "tree_proposals",
        "improvement",
        "canopy_resistance",
        "node_id",
        "instance_id",
        "isSenescent",
        "resource_colour",
        "isTerminal",
        "bioEnvelopeType",
        "sim_Turns",
        "bioSimple",
        "world_sim_turns",
        "world_sim_nodes",
        "world_design_bioenvelope",
        "world_design_bioenvelope_simple",
        "world_sim_matched",
    ],
}


def ensure_frame(node_tree, frame_name, label, location):
    frame = node_tree.nodes.get(frame_name)
    if frame is None or frame.bl_idname != "NodeFrame":
        frame = node_tree.nodes.new("NodeFrame")
    frame.name = frame_name
    frame.label = label
    frame.location = location
    return frame


def ensure_reroute(node_tree, name, label, location, parent):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != "NodeReroute":
        node = node_tree.nodes.new("NodeReroute")
    node.name = name
    node.label = label
    node.location = location
    node.parent = parent
    return node


def ensure_link(node_tree, from_socket, to_socket):
    for link in to_socket.links:
        if link.from_socket == from_socket:
            return
    node_tree.links.new(from_socket, to_socket)


def expose_aovs_for_render_layer_node(node_tree, render_node, aov_names):
    frame_name = f"AOV Inputs {render_node.name}"
    frame = ensure_frame(
        node_tree,
        frame_name,
        f"AOV Inputs - {render_node.layer}",
        (render_node.location.x + 350, render_node.location.y + 100),
    )

    exposed = []
    for index, aov_name in enumerate(aov_names):
        output_socket = render_node.outputs.get(aov_name)
        if output_socket is None:
            continue

        reroute = ensure_reroute(
            node_tree,
            f"{render_node.name}::{aov_name}",
            aov_name,
            (render_node.location.x + 520, render_node.location.y - (index * 70)),
            frame,
        )
        ensure_link(node_tree, output_socket, reroute.inputs[0])
        exposed.append(aov_name)

    return exposed


def patch_scene(scene_name, aov_names):
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        return []

    if not scene.use_nodes:
        scene.use_nodes = True

    if scene.node_tree is None:
        return []

    exposed = []
    for node in scene.node_tree.nodes:
        if node.type != "R_LAYERS":
            continue
        exposed.append((node.name, node.layer, expose_aovs_for_render_layer_node(scene.node_tree, node, aov_names)))
    return exposed


def main():
    results = {}
    for scene_name, aov_names in SCENE_AOV_MAP.items():
        exposed = patch_scene(scene_name, aov_names)
        if exposed:
            results[scene_name] = exposed

    for scene_name, node_results in results.items():
        print(f"Scene: {scene_name}")
        for node_name, layer_name, sockets in node_results:
            print(f"  Render Layers node: {node_name} ({layer_name})")
            print(f"  Exposed sockets: {sockets}")

    if SAVE_FILE:
        bpy.ops.wm.save_mainfile()
        print("Saved current blend file")


if __name__ == "__main__":
    main()
