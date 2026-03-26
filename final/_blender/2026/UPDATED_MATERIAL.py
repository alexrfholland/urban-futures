import bpy


SOURCE_MATERIAL_NAME = "RESOURCES"
TARGET_MATERIAL_NAME = "UPDATED_MATERIAL"
FRAME_PREFIX = "UPDATED_MATERIAL :: "

FRAME_LOCATION = (900.0, 520.0)
ATTR_LOCATION = (-780.0, 360.0)
RESOURCE_AOV_LOCATION = (-420.0, 360.0)
BASE_RGB_LOCATION = (-780.0, 120.0)
ROW_COMPARE_X = -520.0
ROW_RGB_X = -260.0
ROW_MIX_X = 60.0
ROW_Y_START = 120.0
ROW_Y_STEP = -150.0
RESOURCE_COLOUR_AOV_LOCATION = (420.0, 280.0)
EMISSION_LOCATION = (420.0, 80.0)

# Imported models are shifted by +1 in the instancer.
RESOURCE_VALUES = {
    "None": 1.0,
    "Dead Branch": 2.0,
    "Peeling Bark": 3.0,
    "Perch Branch": 4.0,
    "Epiphyte": 5.0,
    "Fallen Log": 6.0,
    "Hollow": 7.0,
}

RESOURCE_COLOURS = {
    "None": (0.619608, 0.619608, 0.619608, 1.0),
    "Dead Branch": (0.129412, 0.588235, 0.952941, 1.0),
    "Peeling Bark": (1.0, 0.921569, 0.231373, 1.0),
    "Perch Branch": (1.0, 0.596078, 0.0, 1.0),
    "Epiphyte": (0.545098, 0.764706, 0.290196, 1.0),
    "Fallen Log": (0.47451, 0.333333, 0.282353, 1.0),
    "Hollow": (0.611765, 0.152941, 0.690196, 1.0),
}


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
    node.outputs["Color"].default_value = color
    return node


def ensure_compare_node(node_tree, name, label, location, compare_value, parent=None):
    node = ensure_node(node_tree, "ShaderNodeMath", name, label, location, parent)
    node.operation = "COMPARE"
    node.inputs[1].default_value = compare_value
    node.inputs[2].default_value = 0.1
    return node


def cleanup_generated_nodes(node_tree):
    for node in list(node_tree.nodes):
        if node.name.startswith(FRAME_PREFIX):
            node_tree.nodes.remove(node)


def require_material(name):
    material = bpy.data.materials.get(name)
    if material is None:
        raise ValueError(f"Material '{name}' not found.")
    if not material.use_nodes or material.node_tree is None:
        raise ValueError(f"Material '{name}' does not use nodes.")
    return material


def ensure_target_material():
    source = require_material(SOURCE_MATERIAL_NAME)
    target = bpy.data.materials.get(TARGET_MATERIAL_NAME)
    if target is None:
        target = source.copy()
        target.name = TARGET_MATERIAL_NAME
        print(f"Created {TARGET_MATERIAL_NAME} from {SOURCE_MATERIAL_NAME}")
    else:
        print(f"Updating existing {TARGET_MATERIAL_NAME}")
    target.use_fake_user = True
    return target


def patch_material(material):
    nt = material.node_tree
    cleanup_generated_nodes(nt)

    attr = next(
        (node for node in nt.nodes if node.bl_idname == "ShaderNodeAttribute" and node.attribute_name == "int_resource"),
        None,
    )
    if attr is None:
        raise ValueError("Could not find Attribute node reading 'int_resource'.")

    material_output = next((node for node in nt.nodes if node.bl_idname == "ShaderNodeOutputMaterial"), None)
    resource_aov = next(
        (node for node in nt.nodes if node.bl_idname == "ShaderNodeOutputAOV" and node.aov_name == "resource"),
        None,
    )
    resource_colour_aov = next(
        (node for node in nt.nodes if node.bl_idname == "ShaderNodeOutputAOV" and node.aov_name == "resource_colour"),
        None,
    )
    if material_output is None or resource_aov is None or resource_colour_aov is None:
        raise ValueError("Could not find Material Output, resource AOV, or resource_colour AOV.")

    frame = ensure_node(
        nt,
        "NodeFrame",
        f"{FRAME_PREFIX}Frame",
        "UPDATED_MATERIAL Resource Colour",
        FRAME_LOCATION,
    )
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)
    frame.shrink = False

    attr.parent = frame
    attr.location = ATTR_LOCATION
    resource_aov.parent = frame
    resource_aov.location = RESOURCE_AOV_LOCATION
    resource_colour_aov.parent = frame
    resource_colour_aov.location = RESOURCE_COLOUR_AOV_LOCATION

    # Keep the scalar resource AOV as the raw int_resource value.
    ensure_link(nt, attr.outputs["Fac"], resource_aov.inputs["Value"])

    none_rgb = ensure_rgb_node(
        nt,
        f"{FRAME_PREFIX}RGB None",
        "None",
        BASE_RGB_LOCATION,
        RESOURCE_COLOURS["None"],
        frame,
    )

    current_color_socket = none_rgb.outputs["Color"]
    order = [
        "Dead Branch",
        "Peeling Bark",
        "Perch Branch",
        "Epiphyte",
        "Fallen Log",
        "Hollow",
    ]

    for index, label in enumerate(order):
        y = ROW_Y_START + ((index + 1) * ROW_Y_STEP)
        compare = ensure_compare_node(
            nt,
            f"{FRAME_PREFIX}Compare {label}",
            f"Compare {label}",
            (ROW_COMPARE_X, y),
            RESOURCE_VALUES[label],
            frame,
        )
        ensure_link(nt, attr.outputs["Fac"], compare.inputs[0])

        rgb = ensure_rgb_node(
            nt,
            f"{FRAME_PREFIX}RGB {label}",
            label,
            (ROW_RGB_X, y),
            RESOURCE_COLOURS[label],
            frame,
        )

        mix = ensure_node(
            nt,
            "ShaderNodeMix",
            f"{FRAME_PREFIX}Mix {label}",
            f"Mix {label}",
            (ROW_MIX_X, y),
            frame,
        )
        mix.data_type = "RGBA"
        mix.blend_type = "MIX"
        ensure_link(nt, compare.outputs["Value"], mix.inputs["Factor"])
        ensure_link(nt, current_color_socket, mix.inputs["A"])
        ensure_link(nt, rgb.outputs["Color"], mix.inputs["B"])
        current_color_socket = mix.outputs["Result"]

    emission = ensure_node(
        nt,
        "ShaderNodeEmission",
        f"{FRAME_PREFIX}Preview Emission",
        "Resource Colour Preview",
        EMISSION_LOCATION,
        frame,
    )
    emission.inputs["Strength"].default_value = 1.0
    ensure_link(nt, current_color_socket, emission.inputs["Color"])

    ensure_link(nt, current_color_socket, resource_colour_aov.inputs["Color"])
    ensure_link(nt, emission.outputs["Emission"], material_output.inputs["Surface"])


def main():
    material = ensure_target_material()
    patch_material(material)

    print(f"Patched material: {material.name}")
    print("Kept existing AOV outputs already present on copied RESOURCES.")
    print("Rebuilt resource_colour from shifted one-based int_resource.")
    print("resource AOV remains raw int_resource.")


if __name__ == "__main__":
    main()
