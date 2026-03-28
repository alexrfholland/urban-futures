from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bpy


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


BLEND_PATH = env_path(
    "EDGE_LAB_BLEND_PATH",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/city_exr_compositor_lightweight_baseline.blend",
)
OUTPUT_DIR = env_path(
    "EDGE_LAB_OUTPUT_DIR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/edge_detection_lab/outputs/exr_city_blender_lightweight_baseline_ao",
)
PATHWAY_EXR = env_path(
    "EDGE_LAB_PATHWAY_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_baseline-city/city-pathway_state_8k.exr",
)
PRIORITY_EXR = env_path(
    "EDGE_LAB_PRIORITY_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_baseline-city/city-city_priority_8k.exr",
)
EXISTING_EXR = env_path(
    "EDGE_LAB_EXISTING_EXR",
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_baseline-city/city-existing_condition_8k.exr",
)
SCENE_NAME = "City"
NODE_PREFIX = "Baseline AO :: "
TREE_ID = 3


def log(message: str) -> None:
    print(f"[add_named_masks_and_render_baseline_lightweight_ao] {message}")


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


def rename_output(rendered_path: Path) -> Path:
    final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
    if rendered_path.exists():
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")
    return final_path


def file_output_node(
    node_tree: bpy.types.NodeTree,
    image_socket,
    stem: str,
    location: tuple[float, float],
    parent: bpy.types.Node | None,
) -> tuple[bpy.types.Node, Path]:
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


def ensure_ao_shading_output_name(node_tree: bpy.types.NodeTree) -> bpy.types.NodeTree:
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeGroup" and getattr(node, "node_tree", None) and node.node_tree.name == "_AO SHADING.001":
            group_tree = node.node_tree
            for item in group_tree.interface.items_tree:
                if getattr(item, "in_out", None) == "OUTPUT" and item.name == "Image":
                    item.name = "shading"
            return group_tree
    raise ValueError("Missing _AO SHADING.001 group in lightweight baseline blend")


def repath_exr_node(node: bpy.types.Node, filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"EXR not found: {filepath}")
    image = node.image
    if image is None:
        raise ValueError(f"Node '{node.name}' has no image")
    image.filepath = str(filepath)
    image.reload()


def detect_resolution(exr_paths: list[Path], images: list[bpy.types.Image]) -> tuple[int, int]:
    for path in exr_paths:
        try:
            result = subprocess.run(
                ["oiiotool", "--info", "-v", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            match = re.search(r"(\d+)\s*x\s*(\d+)", result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
        except Exception:
            continue
    for image in images:
        width, height = image.size[:]
        if width > 0 and height > 0:
            return width, height
    return 3840, 2160


def set_render_size(scene: bpy.types.Scene, width: int, height: int) -> None:
    scene.render.resolution_x = width
    scene.render.resolution_y = height


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(f"Blend not found: {BLEND_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' with compositor node tree not found in {BLEND_PATH}")

    node_tree = scene.node_tree
    set_standard_view(scene)
    remove_existing_helper_nodes(node_tree)
    ao_shading_group = ensure_ao_shading_output_name(node_tree)

    pathway = require_node(node_tree, "City EXR :: pathway_state")
    priority = require_node(node_tree, "City EXR :: city_priority")
    existing = require_node(node_tree, "City EXR :: existing_condition")
    repath_exr_node(pathway, PATHWAY_EXR)
    repath_exr_node(priority, PRIORITY_EXR)
    repath_exr_node(existing, EXISTING_EXR)

    frame = new_node(
        node_tree,
        "NodeFrame",
        f"{NODE_PREFIX}Frame",
        "Baseline AO Masks + Outputs",
        (-2200.0, -1200.0),
    )
    frame.use_custom_color = True
    frame.color = (0.14, 0.14, 0.14)

    mask_visible_pathway = new_node(
        node_tree,
        "CompositorNodeIDMask",
        "mask_visible-arboreal_pathway",
        "mask_visible-arboreal_pathway",
        (-2000.0, -680.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_visible_pathway.index = TREE_ID
    mask_visible_pathway.use_antialiasing = True
    ensure_link(node_tree, pathway.outputs["IndexOB"], mask_visible_pathway.inputs["ID value"])

    mask_all_priority = new_node(
        node_tree,
        "CompositorNodeIDMask",
        "mask_all-arboreal_priority",
        "mask_all-arboreal_priority",
        (-2000.0, -940.0),
        parent=frame,
        color=(0.18, 0.18, 0.10),
    )
    mask_all_priority.index = TREE_ID
    mask_all_priority.use_antialiasing = True
    ensure_link(node_tree, priority.outputs["IndexOB"], mask_all_priority.inputs["ID value"])

    mask_visible_priority = new_node(
        node_tree,
        "CompositorNodeMath",
        "mask_visible-arboreal_priority",
        "mask_visible-arboreal_priority",
        (-1760.0, -820.0),
        parent=frame,
        color=(0.18, 0.16, 0.20),
    )
    mask_visible_priority.operation = "MULTIPLY"
    mask_visible_priority.use_clamp = True
    ensure_link(node_tree, mask_all_priority.outputs["Alpha"], mask_visible_priority.inputs[0])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], mask_visible_priority.inputs[1])

    pathway_shading = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{NODE_PREFIX}Pathway shading",
        "Pathway shading",
        (-1520.0, -620.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    pathway_shading.node_tree = ao_shading_group
    ensure_link(node_tree, pathway.outputs["AO"], pathway_shading.inputs["Image"])
    ensure_link(node_tree, pathway.outputs["Normal"], pathway_shading.inputs["Normal"])
    ensure_link(node_tree, pathway.outputs["Alpha"], pathway_shading.inputs["Alpha"])

    priority_shading = new_node(
        node_tree,
        "CompositorNodeGroup",
        f"{NODE_PREFIX}Priority shading",
        "Priority shading",
        (-1520.0, -860.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    priority_shading.node_tree = ao_shading_group
    ensure_link(node_tree, priority.outputs["AO"], priority_shading.inputs["Image"])
    ensure_link(node_tree, priority.outputs["Normal"], priority_shading.inputs["Normal"])
    ensure_link(node_tree, priority.outputs["Alpha"], priority_shading.inputs["Alpha"])

    pathway_shading_masked = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{NODE_PREFIX}Pathway shading masked",
        "Pathway shading masked",
        (-1260.0, -620.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    pathway_shading_masked.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, pathway_shading.outputs[0], pathway_shading_masked.inputs["Image"])
    ensure_link(node_tree, mask_visible_pathway.outputs["Alpha"], pathway_shading_masked.inputs["Alpha"])

    priority_shading_masked = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{NODE_PREFIX}Priority shading masked",
        "Priority shading masked",
        (-1260.0, -860.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    priority_shading_masked.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, priority_shading.outputs[0], priority_shading_masked.inputs["Image"])
    ensure_link(node_tree, mask_visible_priority.outputs["Value"], priority_shading_masked.inputs["Alpha"])

    base_ao = new_node(
        node_tree,
        "CompositorNodeSetAlpha",
        f"{NODE_PREFIX}Base AO",
        "Existing Condition AO Full",
        (-1480.0, -1100.0),
        parent=frame,
        color=(0.16, 0.20, 0.16),
    )
    base_ao.mode = "REPLACE_ALPHA"
    ensure_link(node_tree, existing.outputs["AO"], base_ao.inputs["Image"])
    ensure_link(node_tree, existing.outputs["Alpha"], base_ao.inputs["Alpha"])

    output_nodes: list[bpy.types.Node] = []
    rendered_paths: list[Path] = []
    for image_socket, stem, y in (
        (pathway_shading_masked.outputs["Image"], "pathway_shading", -620.0),
        (priority_shading_masked.outputs["Image"], "priority_shading", -860.0),
        (base_ao.outputs["Image"], "existing_condition_ao_full", -1100.0),
    ):
        output_node, rendered_path = file_output_node(
            node_tree,
            image_socket,
            stem,
            (-1180.0, y),
            frame,
        )
        output_nodes.append(output_node)
        rendered_paths.append(rendered_path)

    previous_mute_states = {}
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            previous_mute_states[node.name] = node.mute
            node.mute = True
    for output_node in output_nodes:
        output_node.mute = False

    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    width, height = detect_resolution(
        [PATHWAY_EXR, PRIORITY_EXR, EXISTING_EXR],
        [pathway.image, priority.image, existing.image],
    )
    set_render_size(scene, width, height)
    scene.render.resolution_percentage = 100
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    bpy.ops.render.render(write_still=False, scene=scene.name)

    final_paths = [rename_output(path) for path in rendered_paths]

    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile" and node.name in previous_mute_states:
            node.mute = previous_mute_states[node.name]

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))

    for path in final_paths:
        log(f"Wrote {path}")


if __name__ == "__main__":
    main()
