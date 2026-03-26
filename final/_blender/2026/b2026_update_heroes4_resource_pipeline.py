from pathlib import Path
import re
import sys

import bpy
import numpy as np


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DEFAULT_BLEND_PATH = REPO_ROOT / "data/blender/2026/2026 futures heroes4.blend"
INSTANCER_PATH = REPO_ROOT / "final/_blender/2026/b2026_instancer.py"
TARGET_SCENES = ("city", "parade")
RESOURCE_GROUP_NAMES = {"_RESOUrCE COLOURS", "_RESOUrCE COLOURS.001"}

RESOURCE_COLOURS = {
    "other": (158, 158, 158),
    "perch branch": (255, 152, 0),
    "dead branch": (33, 150, 243),
    "peeling bark": (255, 235, 59),
    "epiphyte": (139, 195, 74),
    "fallen log": (121, 85, 72),
    "hollow": (156, 39, 176),
}


RESOURCE_INT_MAP = {
    "none": 1.0,
    "dead branch": 2.0,
    "peeling bark": 3.0,
    "perch branch": 4.0,
    "epiphyte": 5.0,
    "fallen log": 6.0,
    "hollow": 7.0,
}


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def rgba_from_255(rgb):
    return tuple(channel / 255.0 for channel in rgb) + (1.0,)


def clear_node_tree(node_tree):
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def ensure_text_block(name, content):
    text = bpy.data.texts.get(name)
    if text is None:
        text = bpy.data.texts.new(name)
    text.clear()
    text.write(content)
    return text


def ensure_value_aov(view_layer, name):
    for aov in view_layer.aovs:
        if aov.name == name:
            aov.type = "VALUE"
            return aov
    aov = view_layer.aovs.add()
    aov.name = name
    aov.type = "VALUE"
    return aov


def ensure_all_view_layer_aovs():
    required_value_aovs = {
        "resource",
        "structure_id",
        "size",
        "instanceID",
        "isSenescent",
        "isTerminal",
        "control",
        "node_type",
        "tree_interventions",
        "tree_proposals",
        "improvement",
        "canopy_resistance",
        "node_id",
        "precolonial",
    }
    required_color_aovs = {"resource_colour"}

    for scene in bpy.data.scenes:
        for view_layer in scene.view_layers:
            for name in sorted(required_value_aovs):
                ensure_value_aov(view_layer, name)
            for name in sorted(required_color_aovs):
                existing = next((aov for aov in view_layer.aovs if aov.name == name), None)
                if existing is None:
                    aov = view_layer.aovs.add()
                    aov.name = name
                    aov.type = "COLOR"
                else:
                    existing.type = "COLOR"


def find_or_create_attribute_node(node_tree, attribute_name, name, location):
    node = next(
        (n for n in node_tree.nodes if n.bl_idname == "ShaderNodeAttribute" and n.attribute_name == attribute_name),
        None,
    )
    if node is None:
        node = node_tree.nodes.new("ShaderNodeAttribute")
    node.attribute_name = attribute_name
    node.name = name
    node.label = name
    node.location = location
    return node


def find_or_create_aov_node(node_tree, aov_name, name, location):
    node = next(
        (n for n in node_tree.nodes if n.bl_idname == "ShaderNodeOutputAOV" and n.aov_name == aov_name),
        None,
    )
    if node is None:
        node = node_tree.nodes.new("ShaderNodeOutputAOV")
    node.aov_name = aov_name
    node.name = name
    node.label = name
    node.location = location
    return node


def patch_resource_material(material):
    if not material.use_nodes or material.node_tree is None:
        return False

    node_tree = material.node_tree
    int_resource_attr = next(
        (node for node in node_tree.nodes if node.bl_idname == "ShaderNodeAttribute" and node.attribute_name == "int_resource"),
        None,
    )
    resource_aov = next(
        (node for node in node_tree.nodes if node.bl_idname == "ShaderNodeOutputAOV" and node.aov_name == "resource"),
        None,
    )
    if int_resource_attr is None or resource_aov is None:
        return False

    ensure_link(node_tree, int_resource_attr.outputs["Fac"], resource_aov.inputs["Value"])

    for node in node_tree.nodes:
        if node.bl_idname != "ShaderNodeMath" or getattr(node, "operation", None) != "COMPARE":
            continue
        if not node.inputs[0].links:
            continue
        source_node = node.inputs[0].links[0].from_node
        if source_node == int_resource_attr:
            continue
        ensure_link(node_tree, int_resource_attr.outputs["Fac"], node.inputs[0])

    precolonial_attr = find_or_create_attribute_node(
        node_tree,
        "precolonial",
        "AOV Attribute precolonial",
        (-900.0, -950.0),
    )
    precolonial_aov = find_or_create_aov_node(
        node_tree,
        "precolonial",
        "AOV Output precolonial",
        (-620.0, -950.0),
    )
    ensure_link(node_tree, precolonial_attr.outputs["Fac"], precolonial_aov.inputs["Value"])
    return True


def build_new_resource_colours_and_mask_group():
    group = bpy.data.node_groups.get("new_resource_colours_and_mask")
    if group is None:
        group = bpy.data.node_groups.new("new_resource_colours_and_mask", "CompositorNodeTree")
    clear_node_tree(group)

    interface = group.interface
    while interface.items_tree:
        interface.remove(interface.items_tree[0])

    interface.new_socket("Resource Scalar", in_out="INPUT", socket_type="NodeSocketFloat")
    for socket_name in [
        "None",
        "Dead Branch",
        "Peeling Bark",
        "Perch Branch",
        "Epiphyte",
        "Fallen Log",
        "Hollow",
    ]:
        interface.new_socket(socket_name, in_out="INPUT", socket_type="NodeSocketColor")
    interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    interface.new_socket("Mask", in_out="OUTPUT", socket_type="NodeSocketFloat")

    nodes = group.nodes
    links = group.links
    group_input = nodes.new("NodeGroupInput")
    group_input.location = (-1200.0, 0.0)
    group_output = nodes.new("NodeGroupOutput")
    group_output.location = (1160.0, 0.0)

    input_defaults = {
        "None": rgba_from_255(RESOURCE_COLOURS["other"]),
        "Dead Branch": rgba_from_255(RESOURCE_COLOURS["dead branch"]),
        "Peeling Bark": rgba_from_255(RESOURCE_COLOURS["peeling bark"]),
        "Perch Branch": rgba_from_255(RESOURCE_COLOURS["perch branch"]),
        "Epiphyte": rgba_from_255(RESOURCE_COLOURS["epiphyte"]),
        "Fallen Log": rgba_from_255(RESOURCE_COLOURS["fallen log"]),
        "Hollow": rgba_from_255(RESOURCE_COLOURS["hollow"]),
    }
    for socket_name, color in input_defaults.items():
        group_input.outputs[socket_name].default_value = color

    mask_math = nodes.new("CompositorNodeMath")
    mask_math.operation = "GREATER_THAN"
    mask_math.location = (-920.0, -260.0)
    mask_math.inputs[1].default_value = 1.5
    links.new(group_input.outputs["Resource Scalar"], mask_math.inputs[0])

    color_mix_nodes = []
    current_color_socket = group_input.outputs["None"]
    value_order = [
        ("Dead Branch", RESOURCE_INT_MAP["dead branch"]),
        ("Peeling Bark", RESOURCE_INT_MAP["peeling bark"]),
        ("Perch Branch", RESOURCE_INT_MAP["perch branch"]),
        ("Epiphyte", RESOURCE_INT_MAP["epiphyte"]),
        ("Fallen Log", RESOURCE_INT_MAP["fallen log"]),
        ("Hollow", RESOURCE_INT_MAP["hollow"]),
    ]
    x = -940.0
    for index, (label, compare_value) in enumerate(value_order):
        compare = nodes.new("CompositorNodeMath")
        compare.operation = "COMPARE"
        compare.location = (x, 260.0 - (index * 120.0))
        compare.inputs[1].default_value = compare_value
        compare.inputs[2].default_value = 0.1
        links.new(group_input.outputs["Resource Scalar"], compare.inputs[0])

        mix = nodes.new("CompositorNodeMixRGB")
        mix.blend_type = "MIX"
        mix.location = (x + 280.0, 40.0)
        links.new(compare.outputs["Value"], mix.inputs["Fac"])
        links.new(current_color_socket, mix.inputs[1])
        links.new(group_input.outputs[label], mix.inputs[2])
        current_color_socket = mix.outputs["Image"]
        color_mix_nodes.append((compare, mix))
        x += 280.0

    set_alpha = nodes.new("CompositorNodeSetAlpha")
    set_alpha.mode = "APPLY"
    set_alpha.location = (900.0, 0.0)
    links.new(current_color_socket, set_alpha.inputs[0])
    links.new(mask_math.outputs["Value"], set_alpha.inputs[1])
    links.new(set_alpha.outputs["Image"], group_output.inputs["Image"])
    links.new(mask_math.outputs["Value"], group_output.inputs["Mask"])
    return group


def build_mask_from_value_list_group():
    group = bpy.data.node_groups.get("mask_from_value_list")
    if group is None:
        group = bpy.data.node_groups.new("mask_from_value_list", "CompositorNodeTree")
    clear_node_tree(group)

    interface = group.interface
    while interface.items_tree:
        interface.remove(interface.items_tree[0])

    interface.new_socket("Scalar", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket("Tolerance", in_out="INPUT", socket_type="NodeSocketFloat")
    for index in range(1, 9):
        interface.new_socket(f"Value {index}", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket("Mask", in_out="OUTPUT", socket_type="NodeSocketFloat")

    nodes = group.nodes
    links = group.links
    group_input = nodes.new("NodeGroupInput")
    group_input.location = (-1000.0, 0.0)
    group_output = nodes.new("NodeGroupOutput")
    group_output.location = (1040.0, 0.0)
    group_input.outputs["Tolerance"].default_value = 0.1
    for index in range(1, 9):
        group_input.outputs[f"Value {index}"].default_value = -999.0

    last_mask = None
    max_x = -240.0
    for index in range(1, 9):
        compare = nodes.new("CompositorNodeMath")
        compare.operation = "COMPARE"
        compare.location = (-720.0, 260.0 - (index * 120.0))
        links.new(group_input.outputs["Scalar"], compare.inputs[0])
        links.new(group_input.outputs[f"Value {index}"], compare.inputs[1])
        links.new(group_input.outputs["Tolerance"], compare.inputs[2])

        if last_mask is None:
            last_mask = compare.outputs["Value"]
        else:
            maximum = nodes.new("CompositorNodeMath")
            maximum.operation = "MAXIMUM"
            maximum.location = (max_x, 20.0)
            links.new(last_mask, maximum.inputs[0])
            links.new(compare.outputs["Value"], maximum.inputs[1])
            last_mask = maximum.outputs["Value"]
            max_x += 180.0

    links.new(last_mask, group_output.inputs["Mask"])
    return group


def build_weighted_scalar_mix_group():
    group = bpy.data.node_groups.get("weighted_scalar_mix")
    if group is None:
        group = bpy.data.node_groups.new("weighted_scalar_mix", "CompositorNodeTree")
    clear_node_tree(group)

    interface = group.interface
    while interface.items_tree:
        interface.remove(interface.items_tree[0])

    interface.new_socket("Base Image", in_out="INPUT", socket_type="NodeSocketColor")
    interface.new_socket("Modified Image", in_out="INPUT", socket_type="NodeSocketColor")
    interface.new_socket("Scalar", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket("Tolerance", in_out="INPUT", socket_type="NodeSocketFloat")
    for index in range(1, 9):
        interface.new_socket(f"Weight {index}", in_out="INPUT", socket_type="NodeSocketFloat")
    interface.new_socket("Weight", in_out="OUTPUT", socket_type="NodeSocketFloat")
    interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    nodes = group.nodes
    links = group.links
    group_input = nodes.new("NodeGroupInput")
    group_input.location = (-1200.0, 0.0)
    group_output = nodes.new("NodeGroupOutput")
    group_output.location = (1340.0, 0.0)

    group_input.outputs["Tolerance"].default_value = 0.1
    default_weights = {
        1: 1.0,
        2: 0.6,
        3: 0.3,
        4: 1.0,
        5: 1.0,
        6: 1.0,
        7: 1.0,
        8: 1.0,
    }
    for index, default_value in default_weights.items():
        group_input.outputs[f"Weight {index}"].default_value = default_value

    weighted_outputs = []
    for index in range(1, 9):
        compare = nodes.new("CompositorNodeMath")
        compare.operation = "COMPARE"
        compare.location = (-900.0, 320.0 - (index * 120.0))
        compare.inputs[1].default_value = float(index)
        links.new(group_input.outputs["Scalar"], compare.inputs[0])
        links.new(group_input.outputs["Tolerance"], compare.inputs[2])

        multiply = nodes.new("CompositorNodeMath")
        multiply.operation = "MULTIPLY"
        multiply.location = (-640.0, 320.0 - (index * 120.0))
        links.new(compare.outputs["Value"], multiply.inputs[0])
        links.new(group_input.outputs[f"Weight {index}"], multiply.inputs[1])
        weighted_outputs.append(multiply.outputs["Value"])

    current_weight = weighted_outputs[0]
    add_x = -320.0
    for output_socket in weighted_outputs[1:]:
        add = nodes.new("CompositorNodeMath")
        add.operation = "ADD"
        add.use_clamp = True
        add.location = (add_x, 0.0)
        links.new(current_weight, add.inputs[0])
        links.new(output_socket, add.inputs[1])
        current_weight = add.outputs["Value"]
        add_x += 180.0

    mix = nodes.new("CompositorNodeMixRGB")
    mix.blend_type = "MIX"
    mix.location = (920.0, 0.0)
    links.new(current_weight, mix.inputs["Fac"])
    links.new(group_input.outputs["Base Image"], mix.inputs[1])
    links.new(group_input.outputs["Modified Image"], mix.inputs[2])

    links.new(current_weight, group_output.inputs["Weight"])
    links.new(mix.outputs["Image"], group_output.inputs["Image"])
    return group


def ensure_frame(node_tree, name, label, location):
    frame = node_tree.nodes.get(name)
    if frame is None or frame.bl_idname != "NodeFrame":
        frame = node_tree.nodes.new("NodeFrame")
    frame.name = name
    frame.label = label
    frame.location = location
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)
    frame.shrink = True
    return frame


def ensure_reroute(node_tree, name, label, location, parent):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != "NodeReroute":
        if node is not None:
            node_tree.nodes.remove(node)
        node = node_tree.nodes.new("NodeReroute")
    node.name = name
    node.label = label
    node.location = location
    node.parent = parent
    return node


def ensure_group_node(node_tree, name, label, node_group, location, parent):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != "CompositorNodeGroup":
        if node is not None:
            node_tree.nodes.remove(node)
        node = node_tree.nodes.new("CompositorNodeGroup")
    node.name = name
    node.label = label
    node.node_tree = node_group
    node.location = location
    node.parent = parent
    return node


def patch_group_nodes_to_new_resource_group(new_group):
    input_defaults = {
        "None": rgba_from_255(RESOURCE_COLOURS["other"]),
        "Dead Branch": rgba_from_255(RESOURCE_COLOURS["dead branch"]),
        "Peeling Bark": rgba_from_255(RESOURCE_COLOURS["peeling bark"]),
        "Perch Branch": rgba_from_255(RESOURCE_COLOURS["perch branch"]),
        "Epiphyte": rgba_from_255(RESOURCE_COLOURS["epiphyte"]),
        "Fallen Log": rgba_from_255(RESOURCE_COLOURS["fallen log"]),
        "Hollow": rgba_from_255(RESOURCE_COLOURS["hollow"]),
    }
    for scene in bpy.data.scenes:
        if not scene.use_nodes or scene.node_tree is None:
            continue
        for node in scene.node_tree.nodes:
            if node.bl_idname != "CompositorNodeGroup" or node.node_tree is None:
                continue
            if node.node_tree.name in RESOURCE_GROUP_NAMES or node.node_tree == new_group or node.name == "Group.015":
                node.node_tree = new_group
                node.label = "new_resource_colours_and_mask"
                for socket_name, color in input_defaults.items():
                    socket = node.inputs.get(socket_name)
                    if socket is not None:
                        socket.default_value = color


def find_primary_render_layers_node(node_tree):
    node = node_tree.nodes.get("Render Layers")
    if node is not None and node.bl_idname == "CompositorNodeRLayers":
        return node
    for candidate in node_tree.nodes:
        if candidate.bl_idname == "CompositorNodeRLayers" and getattr(candidate, "layer", None) == "pathway_state":
            return candidate
    return None


def add_mask_list_frame(node_tree):
    group = build_mask_from_value_list_group()
    frame = ensure_frame(
        node_tree,
        "ResourceTools::MaskFromValueList::Frame",
        "Mask From Value List",
        (14850.0, 300.0),
    )
    scalar_in = ensure_reroute(
        node_tree,
        "ResourceTools::MaskFromValueList::Scalar",
        "scalar_input",
        (0.0, 40.0),
        frame,
    )
    group_node = ensure_group_node(
        node_tree,
        "ResourceTools::MaskFromValueList::Group",
        "mask_from_value_list",
        group,
        (220.0, 0.0),
        frame,
    )
    output = ensure_reroute(
        node_tree,
        "ResourceTools::MaskFromValueList::Output",
        "mask_from_values_output",
        (540.0, 40.0),
        frame,
    )
    ensure_link(node_tree, scalar_in.outputs[0], group_node.inputs["Scalar"])
    ensure_link(node_tree, group_node.outputs["Mask"], output.inputs[0])


def add_weighted_mix_frame(node_tree):
    group = build_weighted_scalar_mix_group()
    frame = ensure_frame(
        node_tree,
        "ResourceTools::WeightedScalarMix::Frame",
        "Size Weighted Colour Mix",
        (14850.0, -520.0),
    )
    primary_render_layers = find_primary_render_layers_node(node_tree)

    base_in = ensure_reroute(
        node_tree,
        "ResourceTools::WeightedScalarMix::Base",
        "base_image",
        (0.0, 120.0),
        frame,
    )
    modified_in = ensure_reroute(
        node_tree,
        "ResourceTools::WeightedScalarMix::Modified",
        "modified_image",
        (0.0, -20.0),
        frame,
    )
    scalar_in = ensure_reroute(
        node_tree,
        "ResourceTools::WeightedScalarMix::Scalar",
        "size_scalar",
        (0.0, -180.0),
        frame,
    )
    group_node = ensure_group_node(
        node_tree,
        "ResourceTools::WeightedScalarMix::Group",
        "weighted_scalar_mix",
        group,
        (260.0, 0.0),
        frame,
    )
    weight_out = ensure_reroute(
        node_tree,
        "ResourceTools::WeightedScalarMix::WeightOutput",
        "size_weight_mask",
        (620.0, 100.0),
        frame,
    )
    image_out = ensure_reroute(
        node_tree,
        "ResourceTools::WeightedScalarMix::ImageOutput",
        "size_weighted_image",
        (620.0, -20.0),
        frame,
    )

    ensure_link(node_tree, base_in.outputs[0], group_node.inputs["Base Image"])
    ensure_link(node_tree, modified_in.outputs[0], group_node.inputs["Modified Image"])
    ensure_link(node_tree, scalar_in.outputs[0], group_node.inputs["Scalar"])
    ensure_link(node_tree, group_node.outputs["Weight"], weight_out.inputs[0])
    ensure_link(node_tree, group_node.outputs["Image"], image_out.inputs[0])

    if primary_render_layers is not None:
        size_socket = primary_render_layers.outputs.get("size")
        if size_socket is not None:
            ensure_link(node_tree, size_socket, scalar_in.inputs[0])


def patch_live_scenes():
    for scene_name in TARGET_SCENES:
        scene = bpy.data.scenes.get(scene_name)
        if scene is None or not scene.use_nodes or scene.node_tree is None:
            continue
        add_mask_list_frame(scene.node_tree)
        add_weighted_mix_frame(scene.node_tree)


def sync_embedded_instancer():
    if not INSTANCER_PATH.exists():
        raise FileNotFoundError(f"Missing instancer script: {INSTANCER_PATH}")
    ensure_text_block("Instancer", INSTANCER_PATH.read_text())


def shift_mesh_int_attribute_if_zero(mesh, attr_name):
    attr = mesh.attributes.get(attr_name)
    if attr is None:
        return False
    values = np.empty(len(attr.data), dtype=np.int32)
    attr.data.foreach_get("value", values)
    if not np.any(values == 0):
        return False
    values[values >= 0] += 1
    attr.data.foreach_set("value", values)
    mesh.update()
    return True


def patch_existing_point_cloud_attributes():
    instance_pattern = re.compile(r"instanceID\.(\d+)_.*?precolonial\.(True|False)")
    for obj in bpy.data.objects:
        if not (obj.name.startswith("TreePositions_") or obj.name.startswith("LogPositions_")):
            continue
        mesh = getattr(obj, "data", None)
        if mesh is None or not hasattr(mesh, "attributes"):
            continue

        shift_mesh_int_attribute_if_zero(mesh, "size")
        shift_mesh_int_attribute_if_zero(mesh, "control")

        if mesh.attributes.get("precolonial") is not None:
            continue

        instance_attr = mesh.attributes.get("instanceID")
        if instance_attr is None:
            continue

        models_collection = None
        for mod in obj.modifiers:
            if mod.type != "NODES" or mod.node_group is None:
                continue
            collection_info = next(
                (node for node in mod.node_group.nodes if node.bl_idname == "GeometryNodeCollectionInfo"),
                None,
            )
            if collection_info is not None:
                models_collection = collection_info.inputs["Collection"].default_value
                break
        if models_collection is None:
            continue

        instance_map = {}
        for model_obj in models_collection.objects:
            match = instance_pattern.search(model_obj.name)
            if match is None:
                continue
            instance_id = int(match.group(1))
            instance_map[instance_id] = 2 if match.group(2) == "True" else 1

        if not instance_map:
            continue

        instance_values = np.empty(len(instance_attr.data), dtype=np.int32)
        instance_attr.data.foreach_get("value", instance_values)
        precolonial_values = np.full(len(instance_values), -1, dtype=np.int32)
        for index, instance_id in enumerate(instance_values):
            mapped_value = instance_map.get(int(instance_id))
            if mapped_value is not None:
                precolonial_values[index] = mapped_value

        precolonial_attr = mesh.attributes.new(name="precolonial", type="INT", domain="POINT")
        precolonial_attr.data.foreach_set("value", precolonial_values)
        mesh.update()


def patch_existing_resource_meshes():
    processed_mesh_names = set()
    for collection in bpy.data.collections:
        if not collection.name.endswith("_plyModels"):
            continue

        is_log_collection = collection.name.startswith("log_")
        for obj in collection.objects:
            mesh = getattr(obj, "data", None)
            if mesh is None or mesh.name in processed_mesh_names or not hasattr(mesh, "attributes"):
                continue
            processed_mesh_names.add(mesh.name)

            attr = mesh.attributes.get("int_resource")
            if attr is None and is_log_collection:
                attr = mesh.attributes.new(name="int_resource", type="INT", domain="POINT")
            elif attr is None:
                continue

            values = np.empty(len(attr.data), dtype=np.int32)
            if len(attr.data):
                attr.data.foreach_get("value", values)
            else:
                continue

            if is_log_collection:
                values[:] = 6
            elif np.any(values == 0):
                values[values >= 0] += 1
            else:
                continue

            attr.data.foreach_set("value", values)
            mesh.update()


def main():
    blend_path = Path(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].endswith(".blend") else DEFAULT_BLEND_PATH
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    sync_embedded_instancer()
    ensure_all_view_layer_aovs()
    patch_existing_resource_meshes()
    patch_existing_point_cloud_attributes()

    patched_materials = []
    for material in bpy.data.materials:
        if patch_resource_material(material):
            patched_materials.append(material.name)

    new_resource_group = build_new_resource_colours_and_mask_group()
    build_mask_from_value_list_group()
    build_weighted_scalar_mix_group()
    patch_group_nodes_to_new_resource_group(new_resource_group)
    patch_live_scenes()

    bpy.ops.wm.save_mainfile(filepath=str(blend_path))
    print("Patched materials:", patched_materials)
    print("Updated blend:", blend_path)


if __name__ == "__main__":
    main()
