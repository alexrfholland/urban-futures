from pathlib import Path
import sys

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DEFAULT_BLEND_PATH = REPO_ROOT / "data/blender/2026/2026 futures heroes5.blend"

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


def get_blend_path():
    for arg in reversed(sys.argv):
        if arg.endswith(".blend"):
            return Path(arg)
    return DEFAULT_BLEND_PATH


def ensure_link(node_tree, from_socket, to_socket):
    for link in list(to_socket.links):
        if link.from_socket == from_socket:
            return
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def ensure_node(node_tree, bl_idname, name, label, location):
    node = node_tree.nodes.get(name)
    if node is None or node.bl_idname != bl_idname:
        if node is not None:
            node_tree.nodes.remove(node)
        node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    return node


def ensure_rgb_node(node_tree, name, label, location, color):
    node = ensure_node(node_tree, "ShaderNodeRGB", name, label, location)
    node.outputs[0].default_value = color
    return node


def ensure_compare_node(node_tree, name, label, location, compare_value):
    node = ensure_node(node_tree, "ShaderNodeMath", name, label, location)
    node.operation = "COMPARE"
    node.inputs[1].default_value = compare_value
    node.inputs[2].default_value = 0.1
    return node


def backup_material():
    src = bpy.data.materials.get("RESOURCES")
    if src is None:
        raise ValueError("Material 'RESOURCES' not found.")

    backup = bpy.data.materials.get("SS_RESOURCES")
    if backup is None:
        backup = src.copy()
        backup.name = "SS_RESOURCES"
    return src, backup


def patch_resources_material(material):
    if not material.use_nodes or material.node_tree is None:
        raise ValueError("Material 'RESOURCES' does not use nodes.")

    nt = material.node_tree
    attr = next(
        (n for n in nt.nodes if n.bl_idname == "ShaderNodeAttribute" and n.attribute_name == "int_resource"),
        None,
    )
    if attr is None:
        raise ValueError("Could not find Attribute node reading 'int_resource'.")

    material_output = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    resource_aov = next(
        (n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputAOV" and n.aov_name == "resource"),
        None,
    )
    resource_colour_aov = next(
        (n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputAOV" and n.aov_name == "resource_colour"),
        None,
    )
    if material_output is None or resource_aov is None or resource_colour_aov is None:
        raise ValueError("Could not find one of: Material Output, resource AOV, resource_colour AOV.")

    ensure_link(nt, attr.outputs["Fac"], resource_aov.inputs["Value"])

    x0 = -250.0
    y0 = 300.0

    none_rgb = ensure_rgb_node(
        nt,
        "SS Resource None",
        "SS Resource None",
        (x0, y0 + 180.0),
        RESOURCE_COLOURS["None"],
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

    x = x0
    for index, label in enumerate(order):
        compare = ensure_compare_node(
            nt,
            f"SS Resource Compare {label}",
            f"SS Compare {label}",
            (x, y0 - index * 140.0),
            RESOURCE_VALUES[label],
        )
        ensure_link(nt, attr.outputs["Fac"], compare.inputs[0])

        rgb = ensure_rgb_node(
            nt,
            f"SS Resource RGB {label}",
            f"SS {label}",
            (x + 220.0, y0 - index * 140.0),
            RESOURCE_COLOURS[label],
        )

        mix = ensure_node(
            nt,
            "ShaderNodeMix",
            f"SS Resource Mix {label}",
            f"SS Mix {label}",
            (x + 480.0, y0 - index * 140.0),
        )
        mix.data_type = "RGBA"
        mix.blend_type = "MIX"
        mix.inputs["Factor"].default_value = 0.0
        ensure_link(nt, compare.outputs["Value"], mix.inputs["Factor"])
        ensure_link(nt, current_color_socket, mix.inputs["A"])
        ensure_link(nt, rgb.outputs["Color"], mix.inputs["B"])
        current_color_socket = mix.outputs["Result"]
        x += 120.0

    emission = ensure_node(
        nt,
        "ShaderNodeEmission",
        "SS Resource Emission",
        "SS Resource Emission",
        (x + 720.0, y0 - 180.0),
    )
    emission.inputs["Strength"].default_value = 1.0
    ensure_link(nt, current_color_socket, emission.inputs["Color"])

    ensure_link(nt, current_color_socket, resource_colour_aov.inputs["Color"])
    ensure_link(nt, emission.outputs["Emission"], material_output.inputs["Surface"])


def main():
    blend_path = get_blend_path()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    resources, backup = backup_material()
    patch_resources_material(resources)
    bpy.ops.wm.save_mainfile(filepath=str(blend_path))
    print("Backed up material as:", backup.name)
    print("Patched material:", resources.name)
    print("Updated blend:", blend_path)


if __name__ == "__main__":
    main()
