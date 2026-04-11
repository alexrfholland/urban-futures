from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bpy

REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
COMPOSITOR_ROOT = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"
OUTPUT_BASE = REPO_ROOT / "_data-refactored" / "compositor" / "outputs"
DEFAULT_DATASET_ROOT = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "edge_detection_lab"
    / "inputs"
    / "LATEST_REMOTE_EXRS"
    / "simv3-7_20260405_8k64s_simv3-7"
    / "city_timeline"
)

def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


BLEND_PATH = env_path(
    "COMPOSITOR_BLEND_PATH",
    str(CANONICAL_ROOT / "edge_lab_final_template_safe_rebuild_20260405.blend"),
)
OUTPUT_DIR = env_path(
    "COMPOSITOR_OUTPUT_DIR",
    str(OUTPUT_BASE / "edge_lab_final_template" / "current" / "proposals"),
)
PATHWAY_EXR = env_path(
    "COMPOSITOR_PATHWAY_EXR",
    str(DEFAULT_DATASET_ROOT / "city_timeline__positive_state__8k64s.exr"),
)
PRIORITY_EXR = env_path(
    "COMPOSITOR_PRIORITY_EXR",
    str(DEFAULT_DATASET_ROOT / "city_timeline__positive_priority_state__8k64s.exr"),
)
TRENDING_EXR = env_path(
    "COMPOSITOR_TRENDING_EXR",
    str(DEFAULT_DATASET_ROOT / "city_timeline__trending_state__8k64s.exr"),
)
SCENE_NAME = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")
PHASE_FILTER = {
    item.strip().lower()
    for item in os.environ.get("COMPOSITOR_PHASE_FILTER", "").split(",")
    if item.strip()
}


def log(message: str) -> None:
    print(f"[render_edge_lab_current_proposals] {message}")


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


def require_node(node_tree: bpy.types.NodeTree, name: str) -> bpy.types.Node:
    node = node_tree.nodes.get(name)
    if node is None:
        raise ValueError(f"Missing node: {name}")
    return node


def require_node_by_type(node_tree: bpy.types.NodeTree, bl_idname: str) -> bpy.types.Node:
    for node in node_tree.nodes:
        if node.bl_idname == bl_idname:
            return node
    raise ValueError(f"Missing node of type: {bl_idname}")


def repath_exr_node(node: bpy.types.Node, filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    if node.image is None:
        raise ValueError(f"Node '{node.name}' has no image")
    node.image.filepath = str(filepath)
    node.image.reload()


def find_output_socket(node: bpy.types.Node, name: str):
    for socket in node.outputs:
        if socket.name == name:
            return socket
    return None


def preferred_image_socket(node: bpy.types.Node):
    image = find_output_socket(node, "Image")
    if image is not None and getattr(image, "enabled", True):
        return image
    combined = find_output_socket(node, "Combined")
    if combined is not None and getattr(combined, "enabled", True):
        return combined
    if image is not None:
        return image
    return None


def reconnect_image_links(node_tree: bpy.types.NodeTree, node: bpy.types.Node) -> None:
    image = find_output_socket(node, "Image")
    preferred = preferred_image_socket(node)
    if image is None or preferred is None or image == preferred:
        return
    targets = [link.to_socket for link in list(image.links)]
    for link in list(image.links):
        node_tree.links.remove(link)
    for target in targets:
        node_tree.links.new(preferred, target)


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


def configure_scene(scene: bpy.types.Scene, exr_paths: list[Path], images: list[bpy.types.Image]) -> None:
    width, height = detect_resolution(exr_paths, images)
    scene.use_nodes = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)


def mute_all_output_nodes(node_tree: bpy.types.NodeTree) -> None:
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            node.mute = True


def render_socket_to_png(
    scene: bpy.types.Scene,
    node_tree: bpy.types.NodeTree,
    image_socket,
    output_path: Path,
) -> None:
    composite = require_node_by_type(node_tree, "CompositorNodeComposite")
    viewer = require_node_by_type(node_tree, "CompositorNodeViewer")
    for link in list(composite.inputs[0].links):
        node_tree.links.remove(link)
    for link in list(viewer.inputs[0].links):
        node_tree.links.remove(link)
    node_tree.links.new(image_socket, composite.inputs[0])
    node_tree.links.new(image_socket, viewer.inputs[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    log(f"Wrote {output_path}")


def output_nodes(node_tree: bpy.types.NodeTree) -> list[bpy.types.Node]:
    nodes = [
        node
        for node in node_tree.nodes
        if node.bl_idname == "CompositorNodeOutputFile" and node.name.startswith("Proposals::Output :: ")
    ]
    if not nodes:
        raise ValueError("No Proposals output nodes found")
    if PHASE_FILTER:
        filtered = []
        for node in nodes:
            stem = output_stem(node)
            phase = stem.split("/", 1)[0].lower()
            if phase in PHASE_FILTER:
                filtered.append(node)
        if not filtered:
            raise ValueError(f"No Proposals outputs matched COMPOSITOR_PHASE_FILTER={sorted(PHASE_FILTER)}")
        nodes = filtered
    return sorted(nodes, key=lambda node: node.name)


def output_stem(node: bpy.types.Node) -> str:
    if not node.file_slots:
        raise ValueError(f"Output node '{node.name}' has no file slot")
    return node.file_slots[0].path.rstrip("_")


def linked_source_socket(node: bpy.types.Node):
    if not node.inputs[0].links:
        raise ValueError(f"Output node '{node.name}' is not linked")
    return node.inputs[0].links[0].from_socket


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(BLEND_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")

    node_tree = scene.node_tree
    pathway = require_node(node_tree, "Proposals::EXR Pathway")
    priority = require_node(node_tree, "Proposals::EXR Priority")
    trending = require_node(node_tree, "Proposals::EXR Trending")

    repath_exr_node(pathway, PATHWAY_EXR)
    repath_exr_node(priority, PRIORITY_EXR)
    repath_exr_node(trending, TRENDING_EXR)
    reconnect_image_links(node_tree, pathway)
    reconnect_image_links(node_tree, priority)
    reconnect_image_links(node_tree, trending)

    configure_scene(
        scene,
        [PATHWAY_EXR, PRIORITY_EXR, TRENDING_EXR],
        [node.image for node in (pathway, priority, trending) if node.image is not None],
    )
    mute_all_output_nodes(node_tree)

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    for node in output_nodes(node_tree):
        render_socket_to_png(
            scene,
            node_tree,
            linked_source_socket(node),
            OUTPUT_DIR / f"{output_stem(node)}.png",
        )


if __name__ == "__main__":
    main()
