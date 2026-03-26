import bpy


TARGET_MATERIAL_NAME = "RESOURCE_MASK_DEBUG"
PREFIX = "RESOURCE_MASK_DEBUG :: "

RESOURCE_ATTRS = (
    "resource_hollow",
    "resource_epiphyte",
    "resource_dead branch",
    "resource_perch branch",
    "resource_peeling bark",
    "resource_fallen log",
    "resource_other",
)


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
    node.parent = parent
    return node


def ensure_frame(node_tree, name, label, location, color):
    frame = ensure_node(node_tree, "NodeFrame", name, label, location)
    frame.use_custom_color = True
    frame.color = color
    frame.shrink = False
    return frame


def ensure_target_material():
    material = bpy.data.materials.get(TARGET_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(TARGET_MATERIAL_NAME)
        print(f"Created material: {TARGET_MATERIAL_NAME}")
    else:
        print(f"Updating material: {TARGET_MATERIAL_NAME}")
    material.use_nodes = True
    material.use_fake_user = True
    return material


def clear_node_tree(node_tree):
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def patch_material(material):
    nt = material.node_tree
    clear_node_tree(nt)

    source_frame = ensure_frame(
        nt,
        f"{PREFIX}Frame Sources",
        "Resource Mask Sources",
        (-1000.0, 360.0),
        (0.14, 0.14, 0.14),
    )
    preview_frame = ensure_frame(
        nt,
        f"{PREFIX}Frame Preview",
        "Preview",
        (-80.0, 360.0),
        (0.18, 0.18, 0.18),
    )

    source_y = 0.0
    source_step = -150.0
    default_source = None
    for index, attr_name in enumerate(RESOURCE_ATTRS):
        node = ensure_node(
            nt,
            "ShaderNodeAttribute",
            f"{PREFIX}Attr {attr_name}",
            attr_name,
            (0.0, source_y + (index * source_step)),
            source_frame,
        )
        node.attribute_name = attr_name
        if default_source is None:
            default_source = node

    active_mask = ensure_node(
        nt,
        "NodeReroute",
        f"{PREFIX}Active Mask",
        "Active Mask",
        (0.0, 0.0),
        preview_frame,
    )

    note = ensure_node(
        nt,
        "NodeFrame",
        f"{PREFIX}Note",
        "Connect the resource_* attribute you want to inspect into Active Mask",
        (-260.0, 180.0),
        preview_frame,
    )
    note.use_custom_color = True
    note.color = (0.20, 0.16, 0.08)
    note.shrink = False

    if default_source is not None:
        ensure_link(nt, default_source.outputs["Fac"], active_mask.inputs[0])

    multiply = ensure_node(
        nt,
        "ShaderNodeMath",
        f"{PREFIX}Preview Strength",
        "Preview Strength",
        (220.0, 0.0),
        preview_frame,
    )
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = 1.0
    ensure_link(nt, active_mask.outputs[0], multiply.inputs[0])

    combine = ensure_node(
        nt,
        "ShaderNodeCombineColor",
        f"{PREFIX}Combine",
        "Monochrome Preview",
        (460.0, 0.0),
        preview_frame,
    )
    ensure_link(nt, multiply.outputs["Value"], combine.inputs["Red"])
    ensure_link(nt, multiply.outputs["Value"], combine.inputs["Green"])
    ensure_link(nt, multiply.outputs["Value"], combine.inputs["Blue"])

    emission = ensure_node(
        nt,
        "ShaderNodeEmission",
        f"{PREFIX}Emission",
        "Emission",
        (720.0, 0.0),
        preview_frame,
    )
    emission.inputs["Strength"].default_value = 1.0
    ensure_link(nt, combine.outputs["Color"], emission.inputs["Color"])

    output = ensure_node(
        nt,
        "ShaderNodeOutputMaterial",
        f"{PREFIX}Material Output",
        "Material Output",
        (980.0, 0.0),
    )
    ensure_link(nt, emission.outputs["Emission"], output.inputs["Surface"])


def main():
    material = ensure_target_material()
    patch_material(material)
    print(f"Material ready: {material.name}")
    print("Connect any resource_* Attribute node into 'Active Mask' to inspect it.")


if __name__ == "__main__":
    main()
