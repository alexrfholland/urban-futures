import bpy
from pathlib import Path


TARGET_SCENE = "city"
SAVE_FILE = False
REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
CITY_BIOENVELOPE_OUTPUT_DIR = REPO_ROOT / "data" / "blender" / "2026" / "city_bioenvelope"

COLOURED_DEFAULTS = {
    "0 Unmatched": (0.95, 0.95, 0.95, 1.0),
    "1 Exoskeleton": (0.85, 0.39, 0.55, 1.0),
    "2 BrownRoof": (0.72, 0.48, 0.22, 1.0),
    "3 OtherGround": (0.46, 0.64, 0.77, 1.0),
    "4 Rewilded": (0.36, 0.72, 0.34, 1.0),
    "5 FootprintDepaved": (0.92, 0.75, 0.33, 1.0),
    "6 LivingFacade": (0.30, 0.66, 0.60, 1.0),
    "7 GreenRoof": (0.55, 0.80, 0.31, 1.0),
}

PALETTE_NAMES = (
    "0 Unmatched",
    "1 Exoskeleton",
    "2 BrownRoof",
    "3 OtherGround",
    "4 Rewilded",
    "5 FootprintDepaved",
    "6 LivingFacade",
    "7 GreenRoof",
)

PALETTE_SPECS = tuple(
    {
        "index": index,
        "name": name,
        "slug": name.split(" ", 1)[1].replace(" ", "_").lower(),
    }
    for index, name in enumerate(PALETTE_NAMES)
)


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_frame(node_tree, name, label, location):
    frame = node_tree.nodes.get(name)
    if frame is None or frame.bl_idname != "NodeFrame":
        frame = node_tree.nodes.new("NodeFrame")
    frame.name = name
    frame.label = label
    frame.location = location
    return frame


def ensure_reroute(node_tree, name, label, location, parent, input_socket):
    reroute = node_tree.nodes.get(name)
    if reroute is None or reroute.bl_idname != "NodeReroute":
        reroute = node_tree.nodes.new("NodeReroute")
    reroute.name = name
    reroute.label = label
    reroute.location = location
    reroute.parent = parent
    ensure_link(node_tree, input_socket, reroute.inputs[0])
    return reroute


def ensure_world_bio_palette_group():
    group_name = "WORLD_BIOENVELOPE_PALETTE"
    existing = bpy.data.node_groups.get(group_name)
    if existing is not None:
        bpy.data.node_groups.remove(existing)

    node_group = bpy.data.node_groups.new(group_name, "CompositorNodeTree")
    interface = node_group.interface
    interface.new_socket("WorldBioEnvelope", in_out="INPUT", socket_type="NodeSocketFloat")
    for input_name in PALETTE_NAMES:
        interface.new_socket(input_name, in_out="INPUT", socket_type="NodeSocketColor")
    interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    for spec in PALETTE_SPECS:
        interface.new_socket(spec["name"], in_out="OUTPUT", socket_type="NodeSocketColor")

    nodes = node_group.nodes
    group_input = nodes.new("NodeGroupInput")
    group_input.location = (-1600.0, 0.0)
    group_output = nodes.new("NodeGroupOutput")
    group_output.location = (1400.0, 0.0)

    previous_image_socket = None
    for spec in PALETTE_SPECS:
        index = spec["index"]
        input_name = spec["name"]

        compare_node = nodes.new("CompositorNodeMath")
        compare_node.name = f"Compare {input_name}"
        compare_node.label = compare_node.name
        compare_node.operation = "COMPARE"
        compare_node.inputs[1].default_value = float(index)
        compare_node.inputs[2].default_value = 0.1
        compare_node.location = (-1180.0, -120.0 - (index * 180.0))

        set_alpha_node = nodes.new("CompositorNodeSetAlpha")
        set_alpha_node.name = f"Layer {input_name}"
        set_alpha_node.label = set_alpha_node.name
        set_alpha_node.mode = "APPLY"
        set_alpha_node.location = (-860.0, -120.0 - (index * 180.0))

        ensure_link(node_group, group_input.outputs["WorldBioEnvelope"], compare_node.inputs[0])
        ensure_link(node_group, group_input.outputs[input_name], set_alpha_node.inputs["Image"])
        ensure_link(node_group, compare_node.outputs["Value"], set_alpha_node.inputs["Alpha"])
        ensure_link(node_group, set_alpha_node.outputs["Image"], group_output.inputs[input_name])

        if previous_image_socket is None:
            previous_image_socket = set_alpha_node.outputs["Image"]
        else:
            alpha_over = nodes.new("CompositorNodeAlphaOver")
            alpha_over.name = f"Composite {input_name}"
            alpha_over.label = alpha_over.name
            alpha_over.inputs[0].default_value = 1.0
            alpha_over.location = (-480.0 + (index * 220.0), -60.0)
            ensure_link(node_group, previous_image_socket, alpha_over.inputs[1])
            ensure_link(node_group, set_alpha_node.outputs["Image"], alpha_over.inputs[2])
            previous_image_socket = alpha_over.outputs["Image"]

    ensure_link(node_group, previous_image_socket, group_output.inputs["Image"])
    return node_group


def ensure_group_node(node_tree, node_name, node_group, location, parent):
    node = node_tree.nodes.get(node_name)
    if node is None or node.bl_idname != "CompositorNodeGroup":
        node = node_tree.nodes.new("CompositorNodeGroup")
    node.name = node_name
    node.label = node_name
    node.node_tree = node_group
    node.location = location
    node.parent = parent
    return node


def set_group_defaults(group_node):
    for input_name, value in COLOURED_DEFAULTS.items():
        socket = group_node.inputs.get(input_name)
        if socket is not None:
            socket.default_value = value


def patch_city_bioenvelope_exports():
    scene = bpy.data.scenes.get(TARGET_SCENE)
    if scene is None or not scene.use_nodes or scene.node_tree is None:
        raise ValueError(f"Scene '{TARGET_SCENE}' does not exist or has compositor nodes disabled.")

    node_tree = scene.node_tree
    bio_reroute = node_tree.nodes.get("Render Layers Existing Condition::world_design_bioenvelope")
    if bio_reroute is None:
        raise ValueError(
            f"Existing-condition world bioenvelope reroute was not found in scene '{TARGET_SCENE}'."
        )

    palette_group = ensure_world_bio_palette_group()

    coloured_frame = ensure_frame(
        node_tree,
        "WorldExistingCondition::BioEnvelopeColour",
        "world_bioenvelope_coloured",
        (14950.0, -1650.0),
    )
    coloured_group = ensure_group_node(
        node_tree,
        "world_bioenvelope_coloured",
        palette_group,
        (0.0, 0.0),
        coloured_frame,
    )
    set_group_defaults(coloured_group)
    ensure_link(node_tree, bio_reroute.outputs[0], coloured_group.inputs["WorldBioEnvelope"])
    ensure_reroute(
        node_tree,
        "world_bioenvelope_coloured_output",
        "world_bioenvelope_coloured_output",
        (320.0, -200.0),
        coloured_frame,
        coloured_group.outputs["Image"],
    )

    CITY_BIOENVELOPE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_frame = ensure_frame(
        node_tree,
        "WorldExistingCondition::BioEnvelopeExports",
        "world_bioenvelope_exports",
        (15480.0, -4600.0),
    )
    file_output = node_tree.nodes.get("world_bioenvelope_file_output")
    if file_output is None or file_output.bl_idname != "CompositorNodeOutputFile":
        if file_output is not None:
            node_tree.nodes.remove(file_output)
        file_output = node_tree.nodes.new("CompositorNodeOutputFile")
    file_output.name = "world_bioenvelope_file_output"
    file_output.label = "world_bioenvelope_file_output"
    file_output.location = (420.0, 0.0)
    file_output.parent = export_frame
    file_output.base_path = str(CITY_BIOENVELOPE_OUTPUT_DIR)
    file_output.format.file_format = "PNG"
    file_output.format.color_mode = "RGBA"
    file_output.format.color_depth = "8"

    while len(file_output.file_slots) > 0:
        file_output.file_slots.remove(file_output.inputs[0])

    for spec in PALETTE_SPECS:
        file_output.file_slots.new(spec["name"])
        file_output.file_slots[-1].path = f"bioenvelope_{spec['slug']}_"

    for index, spec in enumerate(PALETTE_SPECS):
        reroute = ensure_reroute(
            node_tree,
            f"world_bioenvelope_{spec['slug']}_output",
            f"world_bioenvelope_{spec['slug']}_output",
            (120.0, -(index * 140.0)),
            export_frame,
            coloured_group.outputs[spec["name"]],
        )
        ensure_link(node_tree, reroute.outputs[0], file_output.inputs[index])


def main():
    patch_city_bioenvelope_exports()
    print(f"Patched city bioenvelope palette and exports in scene '{TARGET_SCENE}'")
    if SAVE_FILE:
        bpy.ops.wm.save_mainfile()
        print("Saved current blend file")


if __name__ == "__main__":
    main()
