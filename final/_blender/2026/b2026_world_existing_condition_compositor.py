import bpy
from pathlib import Path


TARGET_SCENES = ("parade", "city")
SAVE_FILE = True
SIM_TURNS_MAX = 5000.0
SIM_TURNS_POWER = 0.35
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

BW_DEFAULTS = {
    "0 Unmatched": (0.95, 0.95, 0.95, 1.0),
    "1 Exoskeleton": (0.12, 0.12, 0.12, 1.0),
    "2 BrownRoof": (0.22, 0.22, 0.22, 1.0),
    "3 OtherGround": (0.34, 0.34, 0.34, 1.0),
    "4 Rewilded": (0.48, 0.48, 0.48, 1.0),
    "5 FootprintDepaved": (0.62, 0.62, 0.62, 1.0),
    "6 LivingFacade": (0.76, 0.76, 0.76, 1.0),
    "7 GreenRoof": (0.88, 0.88, 0.88, 1.0),
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
    node_group = bpy.data.node_groups.get(group_name)
    if node_group is not None:
        bpy.data.node_groups.remove(node_group)

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

    layer_outputs = {}
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
        layer_outputs[input_name] = set_alpha_node.outputs["Image"]

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


def set_group_defaults(group_node, defaults):
    for input_name, value in defaults.items():
        socket = group_node.inputs.get(input_name)
        if socket is not None:
            socket.default_value = value


def ensure_sim_turns_frame(node_tree, source_socket):
    frame = ensure_frame(node_tree, "WorldExistingCondition::SimTurns", "sim turns", (13550.0, -1600.0))

    node = node_tree.nodes.get("World Sim Turns To Hundred")
    if node is None or node.bl_idname != "CompositorNodeMapRange":
        node = node_tree.nodes.new("CompositorNodeMapRange")
    node.name = "World Sim Turns To Hundred"
    node.label = "Turns to 0-100"
    node.location = (0.0, 0.0)
    node.parent = frame
    node.use_clamp = True
    node.inputs["From Min"].default_value = 0.0
    node.inputs["From Max"].default_value = SIM_TURNS_MAX
    node.inputs["To Min"].default_value = 0.0
    node.inputs["To Max"].default_value = 100.0

    divide = node_tree.nodes.get("World Sim Turns To Unit")
    if divide is None or divide.bl_idname != "CompositorNodeMath":
        divide = node_tree.nodes.new("CompositorNodeMath")
    divide.name = "World Sim Turns To Unit"
    divide.label = "Turns to 0-1"
    divide.location = (260.0, 0.0)
    divide.parent = frame
    divide.operation = "DIVIDE"
    divide.inputs[1].default_value = 100.0

    power = node_tree.nodes.get("World Sim Turns Spread")
    if power is None or power.bl_idname != "CompositorNodeMath":
        power = node_tree.nodes.new("CompositorNodeMath")
    power.name = "World Sim Turns Spread"
    power.label = "Distribution spread"
    power.location = (520.0, 0.0)
    power.parent = frame
    power.operation = "POWER"
    power.inputs[1].default_value = SIM_TURNS_POWER
    power.use_clamp = True

    ramp = node_tree.nodes.get("World Sim Turns Greyscale")
    if ramp is None or ramp.bl_idname != "CompositorNodeValToRGB":
        ramp = node_tree.nodes.new("CompositorNodeValToRGB")
    ramp.name = "World Sim Turns Greyscale"
    ramp.label = "Greyscale ramp"
    ramp.location = (820.0, 0.0)
    ramp.parent = frame
    elements = ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = (0.0, 0.0, 0.0, 1.0)
    elements[1].position = 1.0
    elements[1].color = (1.0, 1.0, 1.0, 1.0)

    ensure_link(node_tree, source_socket, node.inputs["Value"])
    ensure_link(node_tree, node.outputs["Value"], divide.inputs[0])
    ensure_link(node_tree, divide.outputs["Value"], power.inputs[0])
    ensure_link(node_tree, power.outputs["Value"], ramp.inputs["Fac"])

    ensure_reroute(
        node_tree,
        "World Sim Turns 0-100",
        "world_sim_turns_0_100",
        (220.0, -160.0),
        frame,
        node.outputs["Value"],
    )
    ensure_reroute(
        node_tree,
        "World Sim Turns Equalised",
        "world_sim_turns_equalised",
        (500.0, -160.0),
        frame,
        power.outputs["Value"],
    )
    ensure_reroute(
        node_tree,
        "World Sim Turns Greyscale Output",
        "world_sim_turns_greyscale",
        (1080.0, -160.0),
        frame,
        ramp.outputs["Image"],
    )


def ensure_bioenvelope_groups(node_tree, source_socket, scene_name):
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
    set_group_defaults(coloured_group, COLOURED_DEFAULTS)
    ensure_link(node_tree, source_socket, coloured_group.inputs["WorldBioEnvelope"])
    ensure_reroute(
        node_tree,
        "world_bioenvelope_coloured_output",
        "world_bioenvelope_coloured_output",
        (320.0, -200.0),
        coloured_frame,
        coloured_group.outputs["Image"],
    )

    if scene_name == "city":
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

    bw_frame = ensure_frame(
        node_tree,
        "WorldExistingCondition::BioEnvelopeBW",
        "world_bioenvelope_bw",
        (14950.0, -3050.0),
    )
    bw_group = ensure_group_node(
        node_tree,
        "world_bioenvelope_bw",
        palette_group,
        (0.0, 0.0),
        bw_frame,
    )
    set_group_defaults(bw_group, BW_DEFAULTS)
    ensure_link(node_tree, source_socket, bw_group.inputs["WorldBioEnvelope"])
    ensure_reroute(
        node_tree,
        "world_bioenvelope_bw_output",
        "world_bioenvelope_bw_output",
        (320.0, -200.0),
        bw_frame,
        bw_group.outputs["Image"],
    )


def patch_scene(scene_name):
    scene = bpy.data.scenes.get(scene_name)
    if scene is None or not scene.use_nodes or scene.node_tree is None:
        return False

    node_tree = scene.node_tree
    turns_reroute = node_tree.nodes.get("Render Layers Existing Condition::world_sim_turns")
    bio_reroute = node_tree.nodes.get("Render Layers Existing Condition::world_design_bioenvelope")
    if turns_reroute is None or bio_reroute is None:
        raise ValueError(f"Existing-condition world reroutes were not found in scene '{scene_name}'.")

    ensure_sim_turns_frame(node_tree, turns_reroute.outputs[0])
    ensure_bioenvelope_groups(node_tree, bio_reroute.outputs[0], scene_name)
    return True


def main():
    patched = []
    for scene_name in TARGET_SCENES:
        if patch_scene(scene_name):
            patched.append(scene_name)

    print(f"Patched scenes: {patched}")

    if SAVE_FILE:
        bpy.ops.wm.save_mainfile()
        print("Saved current blend file")


if __name__ == "__main__":
    main()
