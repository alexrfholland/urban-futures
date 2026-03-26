from pathlib import Path

import bpy


TARGET_SCENE_NAME = "city"
PRIMARY_VIEW_LAYER = "pathway_state"
SOURCE_MATERIAL_NAME = "RESOURCES"
TARGET_MATERIAL_NAME = "MATERIAL_BINARIES"
MATERIAL_PREFIX = "Material Binaries :: "
MATERIAL_FRAME_NAME = f"{MATERIAL_PREFIX}Frame"
MATERIAL_FRAME_LABEL = "Material Binaries AOVs"

RESOURCE_SPECS = [
    {
        "display_name": "None",
        "slug": "none",
        "ply_property": "resource_other",
        "aov_name": "resource_none_mask",
        "color": (0.619608, 0.619608, 0.619608, 1.0),
    },
    {
        "display_name": "Dead Branch",
        "slug": "dead_branch",
        "ply_property": "resource_dead branch",
        "aov_name": "resource_dead_branch_mask",
        "color": (0.129412, 0.588235, 0.952941, 1.0),
    },
    {
        "display_name": "Peeling Bark",
        "slug": "peeling_bark",
        "ply_property": "resource_peeling bark",
        "aov_name": "resource_peeling_bark_mask",
        "color": (1.0, 0.921569, 0.231373, 1.0),
    },
    {
        "display_name": "Perch Branch",
        "slug": "perch_branch",
        "ply_property": "resource_perch branch",
        "aov_name": "resource_perch_branch_mask",
        "color": (1.0, 0.596078, 0.0, 1.0),
    },
    {
        "display_name": "Epiphyte",
        "slug": "epiphyte",
        "ply_property": "resource_epiphyte",
        "aov_name": "resource_epiphyte_mask",
        "color": (0.545098, 0.764706, 0.290196, 1.0),
    },
    {
        "display_name": "Fallen Log",
        "slug": "fallen_log",
        "ply_property": "resource_fallen log",
        "aov_name": "resource_fallen_log_mask",
        "color": (0.47451, 0.333333, 0.282353, 1.0),
    },
    {
        "display_name": "Hollow",
        "slug": "hollow",
        "ply_property": "resource_hollow",
        "aov_name": "resource_hollow_mask",
        "color": (0.611765, 0.152941, 0.690196, 1.0),
    },
]

TREE_MASK_AOV_NAME = "resource_tree_mask"
RESOURCE_SCALAR_AOV_NAME = "resource"
RESOURCE_COLOUR_AOV_NAME = "resource_colour"


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


def cleanup_generated_nodes(node_tree):
    for node in list(node_tree.nodes):
        if node.name.startswith(MATERIAL_PREFIX):
            node_tree.nodes.remove(node)


def normalize_frame(frame):
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)
    frame.shrink = False


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


def ensure_city_view_layer_aovs():
    scene = bpy.data.scenes.get(TARGET_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{TARGET_SCENE_NAME}' not found.")
    view_layer = scene.view_layers.get(PRIMARY_VIEW_LAYER)
    if view_layer is None:
        raise ValueError(f"View layer '{PRIMARY_VIEW_LAYER}' not found in scene '{TARGET_SCENE_NAME}'.")

    ensure_view_layer_aov(view_layer, RESOURCE_COLOUR_AOV_NAME, "COLOR")
    ensure_view_layer_aov(view_layer, RESOURCE_SCALAR_AOV_NAME, "VALUE")
    ensure_view_layer_aov(view_layer, TREE_MASK_AOV_NAME, "VALUE")
    for spec in RESOURCE_SPECS:
        ensure_view_layer_aov(view_layer, spec["aov_name"], "VALUE")
    return view_layer


def ensure_target_material():
    source_material = bpy.data.materials.get(SOURCE_MATERIAL_NAME)
    if source_material is None:
        raise ValueError(f"Material '{SOURCE_MATERIAL_NAME}' not found.")

    target_material = bpy.data.materials.get(TARGET_MATERIAL_NAME)
    if target_material is None:
        target_material = source_material.copy()
        target_material.name = TARGET_MATERIAL_NAME
        print(f"Created material copy: {TARGET_MATERIAL_NAME}")
    else:
        print(f"Updating existing material: {TARGET_MATERIAL_NAME}")

    target_material.use_fake_user = True
    return target_material


def patch_material(material):
    if not material.use_nodes or material.node_tree is None:
        raise ValueError(f"Material '{material.name}' does not use nodes.")

    nt = material.node_tree
    cleanup_generated_nodes(nt)

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


def patch_instance_template_material(material):
    node_group = bpy.data.node_groups.get("instance_template")
    if node_group is None:
        raise ValueError("Node group 'instance_template' not found.")

    patched_nodes = []
    for node in node_group.nodes:
        if node.bl_idname != "GeometryNodeSetMaterial":
            continue
        material_socket = node.inputs.get("Material")
        if material_socket is None:
            continue
        material_socket.default_value = material
        patched_nodes.append(node.name)

    if not patched_nodes:
        raise ValueError("No GeometryNodeSetMaterial nodes were found in 'instance_template'.")

    return patched_nodes


def main():
    ensure_city_view_layer_aovs()
    target_material = ensure_target_material()
    patch_material(target_material)
    patched_nodes = patch_instance_template_material(target_material)

    print(f"Material ready: {target_material.name}")
    print(f"Instance template Set Material nodes patched: {patched_nodes}")
    return {
        "material_name": target_material.name,
        "patched_nodes": patched_nodes,
    }


if __name__ == "__main__":
    main()
