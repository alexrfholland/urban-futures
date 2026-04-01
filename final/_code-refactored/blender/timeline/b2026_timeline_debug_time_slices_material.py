from __future__ import annotations

import bpy


MATERIAL_NAME = "debug_time-slices"
TIME_SLICE_STOPS = (
    (0.0, (0.894, 0.102, 0.110, 1.0)),  # yr0
    (0.2, (0.216, 0.494, 0.722, 1.0)),  # yr10
    (0.4, (0.302, 0.686, 0.290, 1.0)),  # yr30
    (0.6, (0.596, 0.306, 0.639, 1.0)),  # yr60
    (0.8, (1.000, 0.498, 0.000, 1.0)),  # yr180
    (1.0, (1.000, 0.498, 0.000, 1.0)),
)
TARGET_NODE_GROUP_PREFIXES = ("tree_", "log_", "pole_")
YEAR_COLOR_MAP = (
    (0.0, (0.894, 0.102, 0.110, 1.0)),
    (10.0, (0.216, 0.494, 0.722, 1.0)),
    (30.0, (0.302, 0.686, 0.290, 1.0)),
    (60.0, (0.596, 0.306, 0.639, 1.0)),
    (180.0, (1.000, 0.498, 0.000, 1.0)),
)


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_material():
    material = bpy.data.materials.get(MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(MATERIAL_NAME)
    material.use_nodes = True
    material.use_fake_user = True

    node_tree = material.node_tree
    node_tree.nodes.clear()
    nodes = node_tree.nodes

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (980, 0)

    attr_geometry = nodes.new("ShaderNodeAttribute")
    attr_geometry.location = (-920, 120)
    attr_geometry.attribute_name = "year"
    if hasattr(attr_geometry, "attribute_type"):
        attr_geometry.attribute_type = "GEOMETRY"

    attr_instancer = nodes.new("ShaderNodeAttribute")
    attr_instancer.location = (-920, -60)
    attr_instancer.attribute_name = "year"
    if hasattr(attr_instancer, "attribute_type"):
        attr_instancer.attribute_type = "INSTANCER"

    combine_year = nodes.new("ShaderNodeMath")
    combine_year.operation = "MAXIMUM"
    combine_year.location = (-660, 40)

    fallback = nodes.new("ShaderNodeRGB")
    fallback.location = (-260, 220)
    fallback.outputs[0].default_value = YEAR_COLOR_MAP[-1][1]

    last_color_socket = fallback.outputs["Color"]
    x_position = -420
    y_base = 120
    for index, (year_value, color_value) in enumerate(YEAR_COLOR_MAP):
        compare = nodes.new("ShaderNodeMath")
        compare.location = (x_position, y_base - index * 170)
        compare.operation = "COMPARE"
        compare.inputs[1].default_value = year_value
        compare.inputs[2].default_value = 0.5

        color_node = nodes.new("ShaderNodeRGB")
        color_node.location = (x_position + 220, y_base - index * 170)
        color_node.outputs[0].default_value = color_value

        mix_rgb = nodes.new("ShaderNodeMixRGB")
        mix_rgb.location = (x_position + 460, y_base - index * 170)
        mix_rgb.blend_type = "MIX"

        ensure_link(node_tree, combine_year.outputs["Value"], compare.inputs[0])
        ensure_link(node_tree, compare.outputs["Value"], mix_rgb.inputs["Fac"])
        ensure_link(node_tree, last_color_socket, mix_rgb.inputs["Color1"])
        ensure_link(node_tree, color_node.outputs["Color"], mix_rgb.inputs["Color2"])
        last_color_socket = mix_rgb.outputs["Color"]

    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.location = (280, 40)
    emission = nodes.new("ShaderNodeEmission")
    emission.location = (280, -140)
    emission.inputs["Strength"].default_value = 1.0
    light_path = nodes.new("ShaderNodeLightPath")
    light_path.location = (520, -40)
    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (760, 0)

    ensure_link(node_tree, attr_geometry.outputs["Fac"], combine_year.inputs[0])
    ensure_link(node_tree, attr_instancer.outputs["Fac"], combine_year.inputs[1])
    ensure_link(node_tree, last_color_socket, diffuse.inputs["Color"])
    ensure_link(node_tree, last_color_socket, emission.inputs["Color"])
    ensure_link(node_tree, light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    ensure_link(node_tree, diffuse.outputs["BSDF"], mix.inputs[1])
    ensure_link(node_tree, emission.outputs["Emission"], mix.inputs[2])
    ensure_link(node_tree, mix.outputs["Shader"], output.inputs["Surface"])

    return material


def apply_to_instancer_node_groups(material):
    updated = []
    for node_group in bpy.data.node_groups:
        if not node_group.name.startswith(TARGET_NODE_GROUP_PREFIXES):
            continue
        changed = False
        for node in node_group.nodes:
            if node.bl_idname != "GeometryNodeSetMaterial":
                continue
            if "Material" not in node.inputs:
                continue
            node.inputs["Material"].default_value = material
            changed = True
        if changed:
            updated.append(node_group.name)

    instance_template = bpy.data.node_groups.get("instance_template")
    if instance_template is not None:
        for node in instance_template.nodes:
            if node.bl_idname == "GeometryNodeSetMaterial" and "Material" in node.inputs:
                node.inputs["Material"].default_value = material
                updated.append(instance_template.name)
                break
    return sorted(set(updated))


def apply_to_bioenvelopes(material):
    updated = []
    for collection in bpy.data.collections:
        if "_timeline_bioenvelope_" not in collection.name:
            continue
        for obj in collection.objects:
            if obj.type != "MESH" or obj.data is None:
                continue
            if not obj.data.materials:
                obj.data.materials.append(material)
            else:
                obj.data.materials[0] = material
            updated.append(obj.name)
    return sorted(updated)


def main():
    material = ensure_material()
    node_groups = apply_to_instancer_node_groups(material)
    bio_objects = apply_to_bioenvelopes(material)
    if bpy.data.filepath:
        bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath, copy=False)
    print(f"Prepared debug material: {material.name}")
    print(f"Updated instancer node groups: {node_groups}")
    print(f"Updated bioenvelope objects: {bio_objects}")


if __name__ == "__main__":
    main()
