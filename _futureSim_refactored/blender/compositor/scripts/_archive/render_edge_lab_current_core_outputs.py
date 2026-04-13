from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import bpy

# Derive repo root from __file__ so defaults work on any machine.
_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _exr_header import read_exr_dimensions  # noqa: E402
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
OUTPUT_ROOT = env_path(
    "COMPOSITOR_OUTPUT_DIR",
    str(OUTPUT_BASE / "edge_lab_final_template" / "current"),
)
PATHWAY_EXR = env_path(
    "COMPOSITOR_PATHWAY_EXR",
    str(DEFAULT_DATASET_ROOT / "city_timeline__positive_state__8k64s.exr"),
)
PRIORITY_EXR = env_path(
    "COMPOSITOR_PRIORITY_EXR",
    str(DEFAULT_DATASET_ROOT / "city_timeline__positive_priority_state__8k64s.exr"),
)
EXISTING_EXR = env_path(
    "COMPOSITOR_EXISTING_EXR",
    str(DEFAULT_DATASET_ROOT / "city_timeline__existing_condition_positive__8k64s.exr"),
)
TRENDING_EXR = env_path(
    "COMPOSITOR_TRENDING_EXR",
    str(DEFAULT_DATASET_ROOT / "city_timeline__trending_state__8k64s.exr"),
)
SCENE_NAME = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")
FAMILY_FILTER = {
    item.strip().lower()
    for item in os.environ.get("COMPOSITOR_RENDER_FAMILIES", "").split(",")
    if item.strip()
}


FAMILY_SPECS: tuple[dict[str, object], ...] = (
    {
        "name": "AO",
        "output_dir": "ao",
        "images": {
            "AO::EXR Pathway": PATHWAY_EXR,
            "AO::EXR Priority": PRIORITY_EXR,
            "AO::EXR Existing": EXISTING_EXR,
        },
    },
    {
        "name": "Normals",
        "output_dir": "normals",
        "images": {
            "Normals::EXR Pathway": PATHWAY_EXR,
            "Normals::EXR Priority": PRIORITY_EXR,
            "Normals::EXR Existing": EXISTING_EXR,
        },
    },
    {
        "name": "Resources",
        "output_dir": "resources",
        "images": {
            "Resources::EXR Pathway": PATHWAY_EXR,
            "Resources::EXR Priority": PRIORITY_EXR,
            "Resources::EXR Trending": TRENDING_EXR,
        },
    },
)


def log(message: str) -> None:
    print(f"[render_edge_lab_current_core_outputs] {message}")


def selected_families() -> tuple[dict[str, object], ...]:
    if not FAMILY_FILTER:
        return FAMILY_SPECS
    selected = []
    for family in FAMILY_SPECS:
        family_name = str(family["name"]).lower()
        output_dir = str(family["output_dir"]).lower()
        if family_name in FAMILY_FILTER or output_dir in FAMILY_FILTER:
            selected.append(family)
    if not selected:
        raise ValueError(f"No families matched COMPOSITOR_RENDER_FAMILIES={sorted(FAMILY_FILTER)}")
    return tuple(selected)


def require_node(node_tree: bpy.types.NodeTree, name: str) -> bpy.types.Node:
    node = node_tree.nodes.get(name)
    if node is None:
        raise ValueError(f"Missing node: {name}")
    return node


def require_output_nodes(node_tree: bpy.types.NodeTree, family_name: str) -> list[bpy.types.Node]:
    prefix = f"{family_name}::"
    nodes = [
        node
        for node in node_tree.nodes
        if node.bl_idname == "CompositorNodeOutputFile" and node.name.startswith(prefix)
    ]
    if not nodes:
        raise ValueError(f"Missing output nodes for family: {family_name}")
    return nodes


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


def repath_image_node(node: bpy.types.Node, filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    if node.image is None:
        raise ValueError(f"Node '{node.name}' has no image")
    node.image.filepath = str(filepath)
    node.image.reload()


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


def detect_resolution(image_paths: list[Path], images: list[bpy.types.Image]) -> tuple[int, int]:
    """Read the true displayWindow directly from the EXR header.

    Per the Input Resolution Rule in COMPOSITOR_TEMPLATE_CONTRACT.md, this
    must NOT fall back to a hardcoded resolution or to image.size (which
    returns (0, 0) in Blender 4.x until pixels are forced to decode).
    Raise loudly instead.
    """
    seen: list[tuple[int, int]] = []
    for path in image_paths:
        if not path.exists():
            continue
        w, h = read_exr_dimensions(str(path))
        seen.append((w, h))
    if not seen:
        raise RuntimeError(
            f"detect_resolution: could not read any EXR headers from {image_paths}"
        )
    if len(set(seen)) != 1:
        raise RuntimeError(
            f"detect_resolution: EXR dimensions disagree across inputs: {seen}"
        )
    return seen[0]


def configure_scene(scene: bpy.types.Scene, images: list[bpy.types.Image], image_paths: list[Path]) -> None:
    width, height = detect_resolution(image_paths, images)
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


def configure_family_outputs(output_nodes: list[bpy.types.Node], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for node in output_nodes:
        node.base_path = str(output_dir)
        node.format.file_format = "PNG"
        node.format.color_mode = "RGBA"
        node.format.color_depth = "8"
        node.mute = False


def rename_family_outputs(output_dir: Path) -> None:
    for rendered_path in sorted(output_dir.glob("*_0001.png")):
        final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(BLEND_PATH)

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")

    node_tree = scene.node_tree
    image_paths: list[Path] = []
    images: list[bpy.types.Image] = []
    for family in selected_families():
        for node_name, filepath in family["images"].items():
            node = require_node(node_tree, node_name)
            repath_image_node(node, filepath)
            reconnect_image_links(node_tree, node)
            image_paths.append(filepath)
            if node.image is not None:
                images.append(node.image)

    configure_scene(scene, images, image_paths)
    mute_all_output_nodes(node_tree)

    for family in selected_families():
        family_name = str(family["name"])
        output_dir = OUTPUT_ROOT / str(family["output_dir"])
        output_nodes = require_output_nodes(node_tree, family_name)
        configure_family_outputs(output_nodes, output_dir)
        bpy.ops.render.render(write_still=False, scene=scene.name)
        rename_family_outputs(output_dir)
        log(f"Rendered {family_name} -> {output_dir}")
        for node in output_nodes:
            node.mute = True


if __name__ == "__main__":
    main()
