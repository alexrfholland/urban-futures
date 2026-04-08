from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import bpy

REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
COMPOSITOR_ROOT = REPO_ROOT / "_code-refactored" / "refactor_code" / "blender" / "compositor"
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
    "EDGE_LAB_BLEND_PATH",
    str(CANONICAL_ROOT / "edge_lab_final_template_safe_rebuild_20260405.blend"),
)
OUTPUT_DIR = env_path(
    "EDGE_LAB_OUTPUT_DIR",
    str(OUTPUT_BASE / "edge_lab_final_template" / "current" / "base"),
)
EXISTING_EXR = env_path(
    "EDGE_LAB_EXISTING_EXR",
    str(DEFAULT_DATASET_ROOT / "city_timeline__existing_condition_positive__8k64s.exr"),
)
SCENE_NAME = os.environ.get("EDGE_LAB_SCENE_NAME", "Current")
OUTPUT_FILTER = {
    item.strip()
    for item in os.environ.get("EDGE_LAB_OUTPUT_FILTER", "").split(",")
    if item.strip()
}


def log(message: str) -> None:
    print(f"[render_edge_lab_current_base] {message}")


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


def require_any_node(node_tree: bpy.types.NodeTree, names: list[str]) -> bpy.types.Node:
    for name in names:
        node = node_tree.nodes.get(name)
        if node is not None:
            return node
    raise ValueError(f"Missing node from candidates: {names}")


def require_node_by_type(node_tree: bpy.types.NodeTree, bl_idname: str) -> bpy.types.Node:
    for node in node_tree.nodes:
        if node.bl_idname == bl_idname:
            return node
    raise ValueError(f"Missing node of type: {bl_idname}")


def require_node_by_label(
    node_tree: bpy.types.NodeTree,
    label: str,
    bl_idname: str | None = None,
) -> bpy.types.Node:
    for node in node_tree.nodes:
        if node.label != label:
            continue
        if bl_idname is not None and node.bl_idname != bl_idname:
            continue
        return node
    raise ValueError(f"Missing node with label: {label}")


def repath_exr_node(node: bpy.types.Node, filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"EXR not found: {filepath}")
    if node.image is None:
        raise ValueError(f"Node '{node.name}' has no image")
    node.image.filepath = str(filepath)
    node.image.reload()


def exr_image_socket(node: bpy.types.Node):
    image = node.outputs.get("Image")
    if image is not None and getattr(image, "enabled", True):
        return image
    combined = node.outputs.get("Combined")
    if combined is not None and getattr(combined, "enabled", True):
        return combined
    if image is not None:
        return image
    raise ValueError(f"Node '{node.name}' has neither an enabled Image nor Combined output")


def reconnect_group_input(node_tree: bpy.types.NodeTree, group_node: bpy.types.Node, input_name: str, socket) -> None:
    input_socket = group_node.inputs.get(input_name)
    if input_socket is None:
        raise ValueError(f"Missing group input '{input_name}' on node '{group_node.name}'")
    for link in list(input_socket.links):
        node_tree.links.remove(link)
    node_tree.links.new(socket, input_socket)


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


def render_socket_to_png(
    scene: bpy.types.Scene,
    node_tree: bpy.types.NodeTree,
    image_socket,
    output_path: Path,
) -> None:
    composite = require_node_by_type(node_tree, "CompositorNodeComposite")
    for link in list(composite.inputs[0].links):
        node_tree.links.remove(link)
    node_tree.links.new(image_socket, composite.inputs[0])

    original_filepath = scene.render.filepath
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    scene.render.filepath = original_filepath
    log(f"Wrote {output_path}")


def safe_save_mainfile(filepath: Path) -> None:
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(filepath))
    except RuntimeError as exc:
        log(f"Skipping save for {filepath}: {exc}")


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(BLEND_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")

    node_tree = scene.node_tree
    set_standard_view(scene)

    existing = require_any_node(node_tree, ["Current Base Outputs :: EXR Existing", "AO::EXR Existing"])
    repath_exr_node(existing, EXISTING_EXR)
    base_group = require_node(node_tree, "Current Base Outputs :: Group")
    reconnect_group_input(node_tree, base_group, "Existing Image", exr_image_socket(existing))

    outputs = [
        ("base_rgb", OUTPUT_DIR / "base_rgb.png"),
        ("base_outlines", OUTPUT_DIR / "base_outlines.png"),
        ("base_sim-turns", OUTPUT_DIR / "base_sim-turns.png"),
        ("base_sim-nodes", OUTPUT_DIR / "base_sim-nodes.png"),
        ("base_sim-turns_ripple-effect", OUTPUT_DIR / "base_sim-turns_ripple-effect.png"),
        ("base_depth_windowed_balanced_refined", OUTPUT_DIR / "base_depth_windowed_balanced_refined.png"),
        ("base_depth_windowed_internal_refined", OUTPUT_DIR / "base_depth_windowed_internal_refined.png"),
        ("base_depth_windowed_internal_dense", OUTPUT_DIR / "base_depth_windowed_internal_dense.png"),
        ("base_depth_windowed_balanced_dense", OUTPUT_DIR / "base_depth_windowed_balanced_dense.png"),
    ]
    if OUTPUT_FILTER:
        outputs = [(label, path) for label, path in outputs if label in OUTPUT_FILTER]

    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            node.mute = True

    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    width, height = detect_resolution([EXISTING_EXR], [existing.image])
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)

    safe_save_mainfile(BLEND_PATH)

    for label, output_path in outputs:
        reroute = require_node_by_label(node_tree, label, "NodeReroute")
        render_socket_to_png(scene, node_tree, reroute.outputs[0], output_path)

    safe_save_mainfile(BLEND_PATH)


if __name__ == "__main__":
    main()
