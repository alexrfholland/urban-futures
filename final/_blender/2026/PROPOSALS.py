from __future__ import annotations

import importlib.util
from pathlib import Path

import bpy


TARGET_MATERIAL_NAME = "PROPOSALS"
PREFIX = "PROPOSALS :: "

FRAME_X = {
    "attrs": -1800.0,
    "states": -980.0,
    "masked": -260.0,
    "add": 520.0,
    "shader": 1260.0,
}
FRAME_Y = 760.0
ROW_STEP = -180.0

STATE_COLOURS = {
    ("proposal-decay", "rejected"): (0.85, 0.12, 0.12, 1.0),
    ("proposal-decay", "buffer-feature"): (1.00, 0.56, 0.16, 1.0),
    ("proposal-decay", "brace-feature"): (1.00, 0.82, 0.18, 1.0),
    ("proposal-release-control", "rejected"): (0.85, 0.12, 0.12, 1.0),
    ("proposal-release-control", "reduce-pruning"): (0.30, 0.70, 1.00, 1.0),
    ("proposal-release-control", "eliminate-pruning"): (0.06, 0.36, 0.96, 1.0),
    ("proposal-recruit", "rejected"): (0.85, 0.12, 0.12, 1.0),
    ("proposal-recruit", "buffer-feature"): (0.65, 0.95, 0.18, 1.0),
    ("proposal-recruit", "rewild-ground"): (0.20, 0.76, 0.22, 1.0),
    ("proposal-colonise", "rejected"): (0.85, 0.12, 0.12, 1.0),
    ("proposal-colonise", "rewild-ground"): (0.00, 0.72, 0.60, 1.0),
    ("proposal-colonise", "enrich-envelope"): (0.60, 0.38, 0.96, 1.0),
    ("proposal-colonise", "roughen-envelope"): (0.92, 0.38, 0.82, 1.0),
    ("proposal-deploy-structure", "rejected"): (0.85, 0.12, 0.12, 1.0),
    ("proposal-deploy-structure", "adapt-utility-pole"): (0.56, 0.44, 0.95, 1.0),
    ("proposal-deploy-structure", "translocated-log"): (0.61, 0.39, 0.18, 1.0),
    ("proposal-deploy-structure", "upgrade-feature"): (0.96, 0.55, 0.75, 1.0),
}


def load_proposal_framebuffer_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / "_futureSim_refactored"
        / "blender"
        / "proposal_framebuffers.py"
    )
    spec = importlib.util.spec_from_file_location("b2026_proposal_framebuffers", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load proposal framebuffer builder from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def ensure_attribute_node(node_tree, name, label, location, attr_name, attribute_type, parent=None):
    node = ensure_node(node_tree, "ShaderNodeAttribute", name, label, location, parent)
    node.attribute_name = attr_name
    if hasattr(node, "attribute_type"):
        node.attribute_type = attribute_type
    return node


def ensure_math_node(node_tree, name, label, location, operation, parent=None):
    node = ensure_node(node_tree, "ShaderNodeMath", name, label, location, parent)
    node.operation = operation
    return node


def ensure_rgb_node(node_tree, name, label, location, color, parent=None):
    node = ensure_node(node_tree, "ShaderNodeRGB", name, label, location, parent)
    node.outputs["Color"].default_value = color
    return node


def ensure_mixrgb_node(node_tree, name, label, location, blend_type, parent=None):
    node = ensure_node(node_tree, "ShaderNodeMixRGB", name, label, location, parent)
    node.blend_type = blend_type
    node.use_clamp = True
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
    source = bpy.data.materials.get("WORLD_AOV") or bpy.data.materials.get("MINIMAL_RESOURCES")
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


def build_material():
    module = load_proposal_framebuffer_module()
    mappings = module.FRAMEBUFFER_STATE_MAPPINGS

    material = ensure_target_material()
    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    attr_frame = ensure_frame(
        node_tree,
        f"{PREFIX}Frame Attrs",
        "Proposal Attributes",
        (FRAME_X["attrs"], FRAME_Y),
        (0.14, 0.14, 0.14),
    )
    state_frame = ensure_frame(
        node_tree,
        f"{PREFIX}Frame States",
        "Proposal States",
        (FRAME_X["states"], FRAME_Y),
        (0.18, 0.18, 0.18),
    )
    masked_frame = ensure_frame(
        node_tree,
        f"{PREFIX}Frame Masked",
        "Masked Colours",
        (FRAME_X["masked"], FRAME_Y),
        (0.20, 0.20, 0.20),
    )
    add_frame = ensure_frame(
        node_tree,
        f"{PREFIX}Frame Add",
        "Accumulation",
        (FRAME_X["add"], FRAME_Y),
        (0.16, 0.16, 0.16),
    )

    output = ensure_node(
        node_tree,
        "ShaderNodeOutputMaterial",
        f"{PREFIX}Output",
        "Material Output",
        (FRAME_X["shader"] + 340.0, 0.0),
    )
    emission = ensure_node(
        node_tree,
        "ShaderNodeEmission",
        f"{PREFIX}Emission",
        "Emission",
        (FRAME_X["shader"], 0.0),
    )
    emission.inputs["Strength"].default_value = 1.0
    ensure_link(node_tree, emission.outputs["Emission"], output.inputs["Surface"])

    black = ensure_rgb_node(
        node_tree,
        f"{PREFIX}RGB Black",
        "Black",
        (FRAME_X["add"] - 260.0, 180.0),
        (0.0, 0.0, 0.0, 1.0),
        add_frame,
    )

    accumulator_socket = black.outputs["Color"]
    row = 0
    for family, state_mapping in mappings.items():
        attr_name = module.DEFAULT_OUTPUT_COLUMNS[family]
        family_y = row * ROW_STEP

        geometry_attr = ensure_attribute_node(
            node_tree,
            f"{PREFIX}Attr {family} Geometry",
            f"{family} geometry",
            (FRAME_X["attrs"], family_y + 60.0),
            attr_name,
            "GEOMETRY",
            attr_frame,
        )
        instancer_attr = ensure_attribute_node(
            node_tree,
            f"{PREFIX}Attr {family} Instancer",
            f"{family} instancer",
            (FRAME_X["attrs"], family_y - 60.0),
            attr_name,
            "INSTANCER",
            attr_frame,
        )
        max_node = ensure_math_node(
            node_tree,
            f"{PREFIX}Math {family} Max",
            f"{family} max",
            (FRAME_X["attrs"] + 260.0, family_y),
            "MAXIMUM",
            attr_frame,
        )
        ensure_link(node_tree, geometry_attr.outputs["Fac"], max_node.inputs[0])
        ensure_link(node_tree, instancer_attr.outputs["Fac"], max_node.inputs[1])

        for state_name, encoded_value in state_mapping.items():
            if state_name == "not-assessed":
                continue
            color = STATE_COLOURS.get((family, state_name))
            if color is None:
                raise ValueError(f"Missing preview colour for {(family, state_name)}")

            state_slug = f"{family}:{state_name}"
            compare = ensure_math_node(
                node_tree,
                f"{PREFIX}Compare {state_slug}",
                state_slug,
                (FRAME_X["states"], family_y),
                "COMPARE",
                state_frame,
            )
            compare.inputs[1].default_value = float(encoded_value)
            compare.inputs[2].default_value = 0.1
            ensure_link(node_tree, max_node.outputs["Value"], compare.inputs[0])

            color_node = ensure_rgb_node(
                node_tree,
                f"{PREFIX}RGB {state_slug}",
                state_slug,
                (FRAME_X["states"], family_y - 70.0),
                color,
                state_frame,
            )
            masked_color = ensure_mixrgb_node(
                node_tree,
                f"{PREFIX}Mask {state_slug}",
                state_slug,
                (FRAME_X["masked"], family_y),
                "MIX",
                masked_frame,
            )
            masked_color.inputs["Color1"].default_value = (0.0, 0.0, 0.0, 1.0)
            ensure_link(node_tree, compare.outputs["Value"], masked_color.inputs["Fac"])
            ensure_link(node_tree, color_node.outputs["Color"], masked_color.inputs["Color2"])

            add_node = ensure_mixrgb_node(
                node_tree,
                f"{PREFIX}Add {state_slug}",
                state_slug,
                (FRAME_X["add"], family_y),
                "ADD",
                add_frame,
            )
            add_node.inputs["Fac"].default_value = 1.0
            ensure_link(node_tree, accumulator_socket, add_node.inputs["Color1"])
            ensure_link(node_tree, masked_color.outputs["Color"], add_node.inputs["Color2"])
            accumulator_socket = add_node.outputs["Color"]
            row += 1

        row += 1

    ensure_link(node_tree, accumulator_socket, emission.inputs["Color"])
    return material


def main():
    build_material()


if __name__ == "__main__":
    main()
