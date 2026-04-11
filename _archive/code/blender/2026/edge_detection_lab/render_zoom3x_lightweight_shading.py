from __future__ import annotations

from pathlib import Path

import bpy


BLEND_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/city_exr_compositor_lightweight_baseline_zoom3x_8k.blend"
)
OUTPUT_DIR = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/outputs/exr_city_blender_baseline_zoom3x_8k_20260327/shading"
)
PATHWAY_EXR = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_baseline-city_zoom3x_8k/city-pathway_state.exr"
)
PRIORITY_EXR = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_baseline-city_zoom3x_8k/city-city_priority.exr"
)
EXISTING_EXR = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_baseline-city_zoom3x_8k/city-existing_condition.exr"
)
SCENE_NAME = "City"
NODE_PREFIX = "Zoom3x Shading :: "
TREE_ID = 3
RENDER_PRESET = "8k"
RENDER_SIZES = {
    "4k": (3840, 2160),
    "8k": (7680, 4320),
}


def log(message: str) -> None:
    print(f"[render_zoom3x_lightweight_shading] {message}")


def require_node(node_tree: bpy.types.NodeTree, name: str) -> bpy.types.Node:
    node = node_tree.nodes.get(name)
    if node is None:
        raise ValueError(f"Missing node '{name}'")
    return node


def ensure_link(node_tree: bpy.types.NodeTree, from_socket, to_socket) -> None:
    for link in list(to_socket.links):
        node_tree.links.remove(link)
    node_tree.links.new(from_socket, to_socket)


def new_node(
    node_tree: bpy.types.NodeTree,
    bl_idname: str,
    name: str,
    label: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None = None,
    color: tuple[float, float, float] | None = None,
):
    node = node_tree.nodes.new(bl_idname)
    node.name = name
    node.label = label
    node.location = location
    if parent is not None:
        node.parent = parent
    if color is not None:
        node.use_custom_color = True
        node.color = color
    return node


def remove_existing_helper_nodes(node_tree: bpy.types.NodeTree) -> None:
    for node in list(node_tree.nodes):
        if node.name.startswith(NODE_PREFIX):
            node_tree.nodes.remove(node)


def repath_exr(node: bpy.types.Node, filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"Missing EXR: {filepath}")
    image = node.image
    if image is None:
        raise ValueError(f"Node '{node.name}' has no image")
    image.filepath = str(filepath)
    image.reload()


def set_standard_view(scene: bpy.types.Scene) -> None:
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


def rename_output(rendered_path: Path) -> Path:
    final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
    if rendered_path.exists():
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")
    return final_path


def output_node(node_tree: bpy.types.NodeTree, image_socket, stem: str, location: tuple[float, float], parent) -> tuple[bpy.types.Node, Path]:
    node = new_node(
        node_tree,
        "CompositorNodeOutputFile",
        f"{NODE_PREFIX}Output :: {stem}",
        f"Output :: {stem}",
        location,
        parent=parent,
        color=(0.12, 0.20, 0.14),
    )
    node.base_path = str(OUTPUT_DIR)
    node.format.file_format = "PNG"
    node.format.color_mode = "RGBA"
    node.format.color_depth = "8"
    node.file_slots[0].path = f"{stem}_"
    ensure_link(node_tree, image_socket, node.inputs[0])
    return node, OUTPUT_DIR / f"{stem}_0001.png"


def ensure_ao_shading_group(node_tree: bpy.types.NodeTree) -> bpy.types.NodeTree:
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeGroup" and getattr(node, "node_tree", None) and node.node_tree.name == "_AO SHADING.001":
            group_tree = node.node_tree
            for item in group_tree.interface.items_tree:
                if getattr(item, "in_out", None) == "OUTPUT" and item.name == "Image":
                    item.name = "shading"
            return group_tree
    raise ValueError("Missing _AO SHADING.001")


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(f"Blend not found: {BLEND_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")
    node_tree = scene.node_tree
    set_standard_view(scene)
    remove_existing_helper_nodes(node_tree)

    pathway = require_node(node_tree, "City EXR :: pathway_state")
    priority = require_node(node_tree, "City EXR :: city_priority")
    existing = require_node(node_tree, "City EXR :: existing_condition")
    repath_exr(pathway, PATHWAY_EXR)
    repath_exr(priority, PRIORITY_EXR)
    repath_exr(existing, EXISTING_EXR)

    ao_group = ensure_ao_shading_group(node_tree)

    frame = new_node(
        node_tree,
        "NodeFrame",
        f"{NODE_PREFIX}Frame",
        "Zoom3x Shading Exports",
        (-2300.0, -1200.0),
    )
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)

    mask_pathway = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{NODE_PREFIX}mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (-2060.0, -620.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_pathway.index = TREE_ID
    mask_pathway.use_antialiasing = True
    ensure_link(node_tree, pathway.outputs["IndexOB"], mask_pathway.inputs["ID value"])

    mask_all_priority = new_node(
        node_tree,
        "CompositorNodeIDMask",
        f"{NODE_PREFIX}mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (-2060.0, -900.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_all_priority.index = TREE_ID
    mask_all_priority.use_antialiasing = True
    ensure_link(node_tree, priority.outputs["IndexOB"], mask_all_priority.inputs["ID value"])

    mask_visible_priority = new_node(
        node_tree,
        "CompositorNodeMath",
        f"{NODE_PREFIX}mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (-1820.0, -780.0),
        parent=frame,
        color=(0.18, 0.16, 0.20),
    )
    mask_visible_priority.operation = "MULTIPLY"
    mask_visible_priority.use_clamp = True
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])

    pathway_shading = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{NODE_PREFIX}Pathway shading",
        "Pathway shading",
        (-1540.0, -620.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    pathway_shading.node_tree = ao_group
    ensure_link(node_tree, pathway.outputs["AO"], pathway_shading.inputs["Image"])
    ensure_link(node_tree, pathway.outputs["Normal"], pathway_shading.inputs["Normal"])
    ensure_link(node_tree, pathway.outputs["Alpha"], pathway_shading.inputs["Alpha"])

    priority_shading = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{NODE_PREFIX}Priority shading",
        "Priority shading",
        (-1540.0, -880.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    priority_shading.node_tree = ao_group
    ensure_link(node_tree, priority.outputs["AO"], priority_shading.inputs["Image"])
    ensure_link(node_tree, priority.outputs["Normal"], priority_shading.inputs["Normal"])
    ensure_link(node_tree, priority.outputs["Alpha"], priority_shading.inputs["Alpha"])

    pathway_masked = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{NODE_PREFIX}Pathway shading masked",
        "Pathway shading masked",
        (-1280.0, -620.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    pathway_masked.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, pathway_shading.outputs[0], pathway_masked.inputs["Image"])
    ensure_link(node_tree, mask_pathway.outputs["Alpha"], pathway_masked.inputs["Alpha"])

    priority_masked = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{NODE_PREFIX}Priority shading masked",
        "Priority shading masked",
        (-1280.0, -880.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    priority_masked.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, priority_shading.outputs[0], priority_masked.inputs["Image"])
    ensure_link(node_tree, mask_visible_priority.outputs["Value"], priority_masked.inputs["Alpha"])

    outputs = []
    rendered = []
    for socket, stem, y in (
        (pathway_masked.outputs["Image"], "pathway_shading", -620.0),
        (priority_masked.outputs["Image"], "priority_shading", -880.0),
    ):
        node, path = output_node(node_tree, socket, stem, (-1060.0, y), frame)
        outputs.append(node)
        rendered.append(path)

    previous_mute_states = {}
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            previous_mute_states[node.name] = node.mute
            node.mute = True
    for node in outputs:
        node.mute = False

    width, height = RENDER_SIZES[RENDER_PRESET]
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    bpy.ops.render.render(write_still=False, scene=scene.name)

    final_paths = [rename_output(path) for path in rendered]

    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile" and node.name in previous_mute_states:
            node.mute = previous_mute_states[node.name]

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    for path in final_paths:
        log(f"Wrote {path}")


if __name__ == "__main__":
    main()
