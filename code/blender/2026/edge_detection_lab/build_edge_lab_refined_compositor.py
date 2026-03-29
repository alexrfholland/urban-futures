from __future__ import annotations

import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"

SOURCE_BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_SOURCE_COMBINED_BLEND",
        str(DATA_ROOT / "edge_lab_output_suite_combined.blend"),
    )
)
CLASSIC_BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_CLASSIC_BLEND",
        str(DATA_ROOT / "city_exr_compositor_lightweight.blend"),
    )
)
OUTPUT_BLEND_PATH = Path(
    os.environ.get(
        "EDGE_LAB_REFINED_BLEND_PATH",
        str(DATA_ROOT / "edge_lab_output_suite_refined.blend"),
    )
)

SOURCE_SCENES = ("AO", "Normals", "Resources", "DepthOutliner", "MistOutlines")
VERTICAL_PADDING = 760.0
FRAME_PADDING_X = 220.0
FRAME_PADDING_Y = 180.0
FAMILY_COLORS = {
    "AO": (0.20, 0.18, 0.14),
    "Normals": (0.14, 0.18, 0.22),
    "Resources": (0.16, 0.20, 0.14),
    "DepthOutliner": (0.20, 0.16, 0.16),
    "MistOutlines": (0.18, 0.14, 0.20),
}


def log(message: str) -> None:
    print(f"[build_edge_lab_refined_compositor] {message}")


def normalize_scene_color_management(scene: bpy.types.Scene) -> None:
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


def clear_node_tree(node_tree: bpy.types.NodeTree) -> None:
    for link in list(node_tree.links):
        node_tree.links.remove(link)
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)


def scene_bounds(scene: bpy.types.Scene) -> tuple[float, float, float, float]:
    xs = [node.location.x for node in scene.node_tree.nodes]
    ys = [node.location.y for node in scene.node_tree.nodes]
    return min(xs), max(xs), min(ys), max(ys)


def copy_simple_properties(src, dst, skip: set[str] | None = None) -> None:
    skip_ids = {
        "name",
        "location",
        "label",
        "parent",
        "rna_type",
        "type",
        "select",
        "dimensions",
        "inputs",
        "outputs",
        "internal_links",
        "color_ramp",
        "image_user",
        "interface",
        "file_slots",
        "layer_slots",
        "view_center",
    }
    if skip:
        skip_ids |= skip
    for prop in src.bl_rna.properties:
        identifier = prop.identifier
        if identifier in skip_ids or prop.is_readonly:
            continue
        try:
            value = getattr(src, identifier)
        except Exception:
            continue
        try:
            if prop.type in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}:
                setattr(dst, identifier, value)
            elif prop.type == "POINTER" and identifier in {"image", "node_tree"}:
                setattr(dst, identifier, value)
        except Exception:
            continue


def copy_color_ramp(src_node, dst_node) -> None:
    src = src_node.color_ramp
    dst = dst_node.color_ramp
    while len(dst.elements) > len(src.elements):
        dst.elements.remove(dst.elements[-1])
    while len(dst.elements) < len(src.elements):
        dst.elements.new(1.0)
    dst.color_mode = src.color_mode
    dst.interpolation = src.interpolation
    dst.hue_interpolation = src.hue_interpolation
    for src_el, dst_el in zip(src.elements, dst.elements):
        dst_el.position = src_el.position
        dst_el.color = tuple(src_el.color)


def copy_output_file_slots(src_node, dst_node) -> None:
    while len(dst_node.file_slots) > len(src_node.file_slots):
        dst_node.file_slots.remove(dst_node.file_slots[-1])
    while len(dst_node.file_slots) < len(src_node.file_slots):
        dst_node.file_slots.new("Image")
    copy_simple_properties(src_node.format, dst_node.format)
    for src_slot, dst_slot in zip(src_node.file_slots, dst_node.file_slots):
        try:
            dst_slot.path = src_slot.path
        except Exception:
            pass
        try:
            dst_slot.use_node_format = src_slot.use_node_format
        except Exception:
            pass
        try:
            copy_simple_properties(src_slot.format, dst_slot.format)
        except Exception:
            pass


def copy_socket_defaults(src_node, dst_node) -> None:
    for src_input, dst_input in zip(src_node.inputs, dst_node.inputs):
        if src_input.is_linked:
            continue
        if hasattr(src_input, "default_value") and hasattr(dst_input, "default_value"):
            try:
                dst_input.default_value = src_input.default_value
            except Exception:
                pass
    for src_output, dst_output in zip(src_node.outputs, dst_node.outputs):
        if hasattr(src_output, "default_value") and hasattr(dst_output, "default_value"):
            try:
                dst_output.default_value = src_output.default_value
            except Exception:
                pass


def clone_scene_into_suite(
    source_scene: bpy.types.Scene,
    target_tree: bpy.types.NodeTree,
    family_label: str,
    offset: tuple[float, float],
) -> None:
    node_map: dict[bpy.types.Node, bpy.types.Node] = {}
    family_prefix = family_label.replace(" ", "_")

    outer_frame = target_tree.nodes.new("NodeFrame")
    outer_frame.name = f"{family_prefix}::FamilyFrame"
    outer_frame.label = family_label
    outer_frame.label_size = 24
    outer_frame.shrink = False
    outer_frame.use_custom_color = True
    outer_frame.color = FAMILY_COLORS.get(family_label, (0.18, 0.18, 0.18))

    min_x, max_x, min_y, max_y = scene_bounds(source_scene)
    outer_frame.location = (
        offset[0] + min_x - FRAME_PADDING_X,
        offset[1] + max_y + FRAME_PADDING_Y,
    )

    for src_node in source_scene.node_tree.nodes:
        dst_node = target_tree.nodes.new(src_node.bl_idname)
        dst_node.name = f"{family_prefix}::{src_node.name}"
        dst_node.label = src_node.label
        dst_node.location = (src_node.location.x + offset[0], src_node.location.y + offset[1])
        copy_simple_properties(src_node, dst_node)
        copy_socket_defaults(src_node, dst_node)

        if src_node.bl_idname == "NodeFrame":
            dst_node.label_size = src_node.label_size
            dst_node.shrink = src_node.shrink
        elif src_node.bl_idname == "CompositorNodeValToRGB":
            copy_color_ramp(src_node, dst_node)
        elif src_node.bl_idname == "CompositorNodeOutputFile":
            copy_output_file_slots(src_node, dst_node)

        node_map[src_node] = dst_node

    for src_node, dst_node in node_map.items():
        if src_node.parent is not None:
            dst_node.parent = node_map[src_node.parent]
        else:
            dst_node.parent = outer_frame

    for src_link in source_scene.node_tree.links:
        src_from = node_map[src_link.from_node]
        src_to = node_map[src_link.to_node]
        from_socket = src_from.outputs[src_link.from_socket.name]
        to_socket = src_to.inputs[src_link.to_socket.name]
        target_tree.links.new(from_socket, to_socket)


def append_classic_legacy_scene() -> None:
    if not CLASSIC_BLEND_PATH.exists():
        log(f"Classic blend missing: {CLASSIC_BLEND_PATH}")
        return
    with bpy.data.libraries.load(str(CLASSIC_BLEND_PATH), link=False) as (data_from, data_to):
        if "City" not in data_from.scenes:
            raise ValueError(f"'City' not found in {CLASSIC_BLEND_PATH}")
        data_to.scenes = ["City"]
    scene = data_to.scenes[0]
    if scene is None:
        raise RuntimeError(f"Failed to append City from {CLASSIC_BLEND_PATH}")
    scene.name = "ClassicLegacy"
    normalize_scene_color_management(scene)
    scene["edge_lab_role"] = "legacy_classic_compositor"
    log("Appended ClassicLegacy scene")


def purge_unused_data() -> None:
    for _ in range(5):
        result = bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_linked_ids=True,
            do_recursive=True,
        )
        if result != {"FINISHED"}:
            break


def build_suite_scene() -> bpy.types.Scene:
    source_reference = bpy.data.scenes[SOURCE_SCENES[0]]
    suite = bpy.data.scenes.new("Suite")
    suite.use_nodes = True
    clear_node_tree(suite.node_tree)
    suite.render.resolution_x = source_reference.render.resolution_x
    suite.render.resolution_y = source_reference.render.resolution_y
    suite.render.resolution_percentage = source_reference.render.resolution_percentage
    suite.render.image_settings.file_format = source_reference.render.image_settings.file_format
    suite.render.image_settings.color_mode = source_reference.render.image_settings.color_mode
    suite.render.film_transparent = source_reference.render.film_transparent
    suite.render.use_compositing = True
    suite.render.use_sequencer = False
    suite.frame_start = 1
    suite.frame_end = 1
    suite.frame_current = 1
    normalize_scene_color_management(suite)

    title = suite.node_tree.nodes.new("NodeFrame")
    title.name = "Suite::Header"
    title.label = "Edge Lab Output Suite"
    title.label_size = 28
    title.shrink = False
    title.use_custom_color = True
    title.color = (0.15, 0.15, 0.15)
    title.location = (-260.0, 1860.0)

    current_top = 1200.0
    for name in SOURCE_SCENES:
        source_scene = bpy.data.scenes[name]
        min_x, max_x, min_y, max_y = scene_bounds(source_scene)
        offset = (0.0 - min_x, current_top - max_y)
        clone_scene_into_suite(source_scene, suite.node_tree, name, offset)
        height = (max_y - min_y) + FRAME_PADDING_Y * 2.0
        current_top -= height + VERTICAL_PADDING

    suite["edge_lab_role"] = "refined_suite"
    return suite


def main() -> None:
    if not SOURCE_BLEND_PATH.exists():
        raise FileNotFoundError(SOURCE_BLEND_PATH)

    bpy.ops.wm.open_mainfile(filepath=str(SOURCE_BLEND_PATH))

    build_suite_scene()
    append_classic_legacy_scene()

    for scene in list(bpy.data.scenes):
        if scene.name not in {"Suite", "ClassicLegacy"}:
            bpy.data.scenes.remove(scene)

    purge_unused_data()

    bpy.context.window.scene = bpy.data.scenes["Suite"]
    OUTPUT_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND_PATH))
    log(f"Saved {OUTPUT_BLEND_PATH}")


if __name__ == "__main__":
    main()
