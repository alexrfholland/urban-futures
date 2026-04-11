"""
Enable per-resource PNG exports for the baseline lightweight compositor blend.

This stays within the edge-lab/lightweight compositor area. It updates the
pathway and priority `SS Resource Export :: File Output*` nodes so each resource
reroute gets its own PNG slot, then saves the blend in place.
"""

from __future__ import annotations

from pathlib import Path

import bpy


BLEND_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/city_exr_compositor_lightweight_baseline.blend"
)
SCENE_NAME = "City"

RESOURCE_SLOTS = (
    ("none", "habitat_features_close_ups_resource_none_"),
    ("dead_branch", "habitat_features_close_ups_resource_dead_branch_"),
    ("peeling_bark", "habitat_features_close_ups_resource_peeling_bark_"),
    ("perch_branch", "habitat_features_close_ups_resource_perch_branch_"),
    ("epiphyte", "habitat_features_close_ups_resource_epiphyte_"),
    ("fallen_log", "habitat_features_close_ups_resource_fallen_log_"),
    ("hollow", "habitat_features_close_ups_resource_hollow_"),
)

RESOURCE_COLOURS = {
    "hollow": "#ce6dd9",
    "epiphyte": "#c5e28e",
    "dead_branch": "#ffcc01",
    "peeling_bark": "#ff85be",
    "fallen_log": "#8f89bf",
    "perch_branch": "#ffcb00",
    "none": "#cecece",
}

OUTPUT_NODE_SPECS = (
    ("SS Resource Export :: File Output", ""),
    ("SS Resource Export :: File Output.001", ".001"),
)


def log(message: str) -> None:
    print(f"[enable_baseline_lightweight_resource_exports] {message}")


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def clear_file_slots(output_node: bpy.types.Node) -> None:
    while len(output_node.inputs):
        output_node.file_slots.remove(output_node.inputs[0])


def require_node(node_tree: bpy.types.NodeTree, node_name: str) -> bpy.types.Node:
    node = node_tree.nodes.get(node_name)
    if node is None:
        raise ValueError(f"Missing node '{node_name}'")
    return node


def srgb_channel_to_linear(value: int) -> float:
    normalized = value / 255.0
    if normalized <= 0.04045:
        return normalized / 12.92
    return ((normalized + 0.055) / 1.055) ** 2.4


def hex_to_linear_rgba(hex_value: str) -> tuple[float, float, float, float]:
    value = hex_value.lstrip("#")
    return (
        srgb_channel_to_linear(int(value[0:2], 16)),
        srgb_channel_to_linear(int(value[2:4], 16)),
        srgb_channel_to_linear(int(value[4:6], 16)),
        1.0,
    )


def configure_colour_nodes(node_tree: bpy.types.NodeTree, suffix: str) -> None:
    for slug, hex_value in RESOURCE_COLOURS.items():
        colour_node = require_node(node_tree, f"SS Resource Export :: Colour::{slug}{suffix}")
        colour_node.outputs[0].default_value = hex_to_linear_rgba(hex_value)


def configure_output_node(node_tree: bpy.types.NodeTree, output_node_name: str, suffix: str) -> None:
    output_node = require_node(node_tree, output_node_name)
    clear_file_slots(output_node)

    for slug, slot_path in RESOURCE_SLOTS:
        output_node.file_slots.new(slug)
        slot = output_node.file_slots[-1]
        slot.path = slot_path

    for slug, _slot_path in RESOURCE_SLOTS:
        export_node_name = f"SS Resource Export :: Export::{slug}{suffix}"
        export_node = require_node(node_tree, export_node_name)
        target_socket = output_node.inputs.get(slug)
        if target_socket is None:
            raise ValueError(f"Missing output slot '{slug}' on '{output_node_name}'")
        ensure_link(node_tree, export_node.outputs[0], target_socket)

    output_node.mute = False
    log(f"Configured {output_node_name} with {len(RESOURCE_SLOTS)} resource slots")


def configure_scene(scene: bpy.types.Scene) -> None:
    try:
        scene.display_settings.display_device = "sRGB"
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    try:
        scene.sequencer_colorspace_settings.name = "sRGB"
    except Exception:
        pass


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(f"Blend not found: {BLEND_PATH}")

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' with compositor node tree not found in {BLEND_PATH}")

    configure_scene(scene)
    for output_node_name, suffix in OUTPUT_NODE_SPECS:
        configure_colour_nodes(scene.node_tree, suffix)
        configure_output_node(scene.node_tree, output_node_name, suffix)

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    log(f"Saved {BLEND_PATH}")


if __name__ == "__main__":
    main()
