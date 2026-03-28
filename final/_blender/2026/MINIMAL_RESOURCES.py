import bpy


SOURCE_MATERIAL_NAME = "RESOURCES"
TARGET_MATERIAL_NAME = "MINIMAL_RESOURCES"
PREFIX = "MINIMAL_RESOURCES :: "

FRAME_X = {
    "instancer": -1800.0,
    "geometry": -980.0,
    "core": -180.0,
    "masks": 520.0,
    "preview": 1420.0,
}
FRAME_Y = 760.0
ATTR_X = 0.0
AOV_X = 420.0
ROW_STEP = -155.0

RESOURCE_COLOURS = {
    "None": (0.807843, 0.807843, 0.807843, 1.0),
    "Dead Branch": (1.0, 0.8, 0.003922, 1.0),
    "Peeling Bark": (1.0, 0.521569, 0.745098, 1.0),
    "Perch Branch": (1.0, 0.796078, 0.0, 1.0),
    "Epiphyte": (0.772549, 0.886275, 0.556863, 1.0),
    "Fallen Log": (0.560784, 0.537255, 0.74902, 1.0),
    "Hollow": (0.807843, 0.427451, 0.85098, 1.0),
}

PASSTHROUGH_AOV_SPECS = (
    {"aov_name": "structure_id", "attr_name": "structure_id", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "size", "attr_name": "size", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "control", "attr_name": "control", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "node_type", "attr_name": "node_type", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "tree_interventions", "attr_name": "tree_interventions", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "tree_proposals", "attr_name": "tree_proposals", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "improvement", "attr_name": "improvement", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "canopy_resistance", "attr_name": "canopy_resistance", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "node_id", "attr_name": "node_id", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "instanceID", "attr_name": "instanceID", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "instance_id", "attr_name": "instanceID", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "precolonial", "attr_name": "precolonial", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "bioEnvelopeType", "attr_name": "bioEnvelopeType", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "bioSimple", "attr_name": "bioSimple", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "sim_Turns", "attr_name": "sim_Turns", "type": "VALUE", "attribute_type": "INSTANCER"},
    {"aov_name": "isSenescent", "attr_name": "isSenescent", "type": "VALUE", "attribute_type": "GEOMETRY"},
    {"aov_name": "isTerminal", "attr_name": "isTerminal", "type": "VALUE", "attribute_type": "GEOMETRY"},
)

RESOURCE_MASK_SPECS = (
    {"label": "None", "slug": "none", "attr_name": "resource_other", "aov_name": "resource_none_mask", "color": RESOURCE_COLOURS["None"]},
    {"label": "Dead Branch", "slug": "dead_branch", "attr_name": "resource_dead", "aov_name": "resource_dead_branch_mask", "color": RESOURCE_COLOURS["Dead Branch"]},
    {"label": "Peeling Bark", "slug": "peeling_bark", "attr_name": "resource_peeling", "aov_name": "resource_peeling_bark_mask", "color": RESOURCE_COLOURS["Peeling Bark"]},
    {"label": "Perch Branch", "slug": "perch_branch", "attr_name": "resource_perch", "aov_name": "resource_perch_branch_mask", "color": RESOURCE_COLOURS["Perch Branch"]},
    {"label": "Epiphyte", "slug": "epiphyte", "attr_name": "resource_epiphyte", "aov_name": "resource_epiphyte_mask", "color": RESOURCE_COLOURS["Epiphyte"]},
    {"label": "Fallen Log", "slug": "fallen_log", "attr_name": "resource_fallen", "aov_name": "resource_fallen_log_mask", "color": RESOURCE_COLOURS["Fallen Log"]},
    {"label": "Hollow", "slug": "hollow", "attr_name": "resource_hollow", "aov_name": "resource_hollow_mask", "color": RESOURCE_COLOURS["Hollow"]},
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


def ensure_attribute_node(node_tree, name, label, location, attr_name, attribute_type=None, parent=None):
    node = ensure_node(node_tree, "ShaderNodeAttribute", name, label, location, parent)
    node.attribute_name = attr_name
    if attribute_type is not None and hasattr(node, "attribute_type"):
        node.attribute_type = attribute_type
    return node


def ensure_aov_node(node_tree, name, label, location, aov_name, parent=None):
    node = ensure_node(node_tree, "ShaderNodeOutputAOV", name, label, location, parent)
    node.aov_name = aov_name
    return node


def ensure_rgb_node(node_tree, name, label, location, color, parent=None):
    node = ensure_node(node_tree, "ShaderNodeRGB", name, label, location, parent)
    node.outputs["Color"].default_value = color
    return node


def ensure_math_node(node_tree, name, label, location, operation, parent=None):
    node = ensure_node(node_tree, "ShaderNodeMath", name, label, location, parent)
    node.operation = operation
    return node


def copy_material_settings(source, target):
    if source is None:
        return
    for attr_name in (
        "blend_method",
        "shadow_method",
        "alpha_threshold",
        "use_backface_culling",
        "use_backface_culling_shadow",
        "show_transparent_back",
        "use_screen_refraction",
        "refraction_depth",
        "preview_render_type",
    ):
        if hasattr(source, attr_name) and hasattr(target, attr_name):
            setattr(target, attr_name, getattr(source, attr_name))


def ensure_target_material():
    source = bpy.data.materials.get(SOURCE_MATERIAL_NAME)
    target = bpy.data.materials.get(TARGET_MATERIAL_NAME)
    if target is None:
        target = bpy.data.materials.new(TARGET_MATERIAL_NAME)
        print(f"Created material: {TARGET_MATERIAL_NAME}")
    else:
        print(f"Updating material: {TARGET_MATERIAL_NAME}")

    target.use_nodes = True
    target.use_fake_user = True
    copy_material_settings(source, target)
    return target


def clear_node_tree(node_tree):
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def build_aov_frame(node_tree, frame_name, frame_label, frame_x, frame_color, specs):
    frame = ensure_frame(
        node_tree,
        frame_name,
        frame_label,
        (frame_x, FRAME_Y),
        frame_color,
    )
    output_sockets = {}
    for index, spec in enumerate(specs):
        y = index * ROW_STEP
        attr = ensure_attribute_node(
            node_tree,
            f"{PREFIX}Attr {spec['aov_name']}",
            spec["attr_name"],
            (ATTR_X, y),
            spec["attr_name"],
            spec["attribute_type"],
            frame,
        )
        aov = ensure_aov_node(
            node_tree,
            f"{PREFIX}AOV {spec['aov_name']}",
            spec["aov_name"],
            (AOV_X, y),
            spec["aov_name"],
            frame,
        )
        ensure_link(node_tree, attr.outputs["Fac"], aov.inputs["Value"])
        output_sockets[spec["aov_name"]] = attr.outputs["Fac"]
    return output_sockets


def build_resource_frames(node_tree, material_output):
    core_frame = ensure_frame(
        node_tree,
        f"{PREFIX}Frame Core",
        "Resource Core",
        (FRAME_X["core"], FRAME_Y),
        (0.16, 0.16, 0.16),
    )
    mask_frame = ensure_frame(
        node_tree,
        f"{PREFIX}Frame Masks",
        "Resource Binary Masks",
        (FRAME_X["masks"], FRAME_Y),
        (0.16, 0.16, 0.16),
    )
    preview_frame = ensure_frame(
        node_tree,
        f"{PREFIX}Frame Preview",
        "Resource Colour Preview",
        (FRAME_X["preview"], FRAME_Y),
        (0.18, 0.18, 0.18),
    )

    int_attr = ensure_attribute_node(
        node_tree,
        f"{PREFIX}Attr int_resource",
        "int_resource",
        (ATTR_X, 0.0),
        "int_resource",
        None,
        core_frame,
    )
    resource_aov = ensure_aov_node(
        node_tree,
        f"{PREFIX}AOV resource",
        "resource",
        (AOV_X, 0.0),
        "resource",
        core_frame,
    )
    ensure_link(node_tree, int_attr.outputs["Fac"], resource_aov.inputs["Value"])

    tree_mask = ensure_math_node(
        node_tree,
        f"{PREFIX}Math tree_mask",
        "resource_tree_mask",
        (ATTR_X, -180.0),
        "GREATER_THAN",
        core_frame,
    )
    tree_mask.inputs[1].default_value = 0.5
    ensure_link(node_tree, int_attr.outputs["Fac"], tree_mask.inputs[0])

    tree_mask_aov = ensure_aov_node(
        node_tree,
        f"{PREFIX}AOV resource_tree_mask",
        "resource_tree_mask",
        (AOV_X, -180.0),
        "resource_tree_mask",
        core_frame,
    )
    ensure_link(node_tree, tree_mask.outputs["Value"], tree_mask_aov.inputs["Value"])

    mask_attr_x = ATTR_X
    mask_aov_x = AOV_X
    preview_rgb_x = 0.0
    preview_mix_x = 360.0

    black = ensure_rgb_node(
        node_tree,
        f"{PREFIX}Preview Base",
        "Preview Base",
        (preview_rgb_x, 120.0),
        (0.0, 0.0, 0.0, 1.0),
        preview_frame,
    )
    current_color = black.outputs["Color"]
    mask_outputs = {}

    for index, spec in enumerate(RESOURCE_MASK_SPECS):
        y = index * ROW_STEP
        attr = ensure_attribute_node(
            node_tree,
            f"{PREFIX}Mask Attr {spec['slug']}",
            spec["attr_name"],
            (mask_attr_x, y),
            spec["attr_name"],
            None,
            mask_frame,
        )
        aov = ensure_aov_node(
            node_tree,
            f"{PREFIX}Mask AOV {spec['slug']}",
            spec["aov_name"],
            (mask_aov_x, y),
            spec["aov_name"],
            mask_frame,
        )
        ensure_link(node_tree, attr.outputs["Fac"], aov.inputs["Value"])
        mask_outputs[spec["slug"]] = attr.outputs["Fac"]

        rgb = ensure_rgb_node(
            node_tree,
            f"{PREFIX}Preview RGB {spec['slug']}",
            spec["label"],
            (preview_rgb_x, y),
            spec["color"],
            preview_frame,
        )
        mix = ensure_node(
            node_tree,
            "ShaderNodeMix",
            f"{PREFIX}Preview Mix {spec['slug']}",
            f"Mix {spec['label']}",
            (preview_mix_x, y),
            preview_frame,
        )
        mix.data_type = "RGBA"
        mix.blend_type = "MIX"
        ensure_link(node_tree, mask_outputs[spec["slug"]], mix.inputs["Factor"])
        ensure_link(node_tree, current_color, mix.inputs["A"])
        ensure_link(node_tree, rgb.outputs["Color"], mix.inputs["B"])
        current_color = mix.outputs["Result"]

    colour_aov = ensure_aov_node(
        node_tree,
        f"{PREFIX}AOV resource_colour",
        "resource_colour",
        (AOV_X, -360.0),
        "resource_colour",
        core_frame,
    )
    ensure_link(node_tree, current_color, colour_aov.inputs["Color"])

    emission = ensure_node(
        node_tree,
        "ShaderNodeEmission",
        f"{PREFIX}Preview Emission",
        "Preview Emission",
        (760.0, -120.0),
        preview_frame,
    )
    emission.inputs["Strength"].default_value = 1.0
    ensure_link(node_tree, current_color, emission.inputs["Color"])
    ensure_link(node_tree, emission.outputs["Emission"], material_output.inputs["Surface"])


def patch_material(material):
    node_tree = material.node_tree
    clear_node_tree(node_tree)

    material_output = ensure_node(
        node_tree,
        "ShaderNodeOutputMaterial",
        f"{PREFIX}Material Output",
        "Material Output",
        (2440.0, 220.0),
    )

    instancer_specs = [spec for spec in PASSTHROUGH_AOV_SPECS if spec["attribute_type"] == "INSTANCER"]
    geometry_specs = [spec for spec in PASSTHROUGH_AOV_SPECS if spec["attribute_type"] == "GEOMETRY"]
    build_aov_frame(
        node_tree,
        f"{PREFIX}Frame Instancer",
        "Instancer AOVs",
        FRAME_X["instancer"],
        (0.13, 0.13, 0.13),
        instancer_specs,
    )
    build_aov_frame(
        node_tree,
        f"{PREFIX}Frame Geometry",
        "Geometry AOVs",
        FRAME_X["geometry"],
        (0.13, 0.13, 0.13),
        geometry_specs,
    )
    build_resource_frames(node_tree, material_output)


def main():
    material = ensure_target_material()
    patch_material(material)
    print(f"Material ready: {material.name}")
    print("Built from scratch with clean frames and no frame shrink.")


if __name__ == "__main__":
    main()
