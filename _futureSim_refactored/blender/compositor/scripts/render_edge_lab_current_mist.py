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
    str(OUTPUT_BASE / "edge_lab_final_template" / "current" / "outlines_mist"),
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
MIST_EXR = env_path("COMPOSITOR_MIST_EXR", str(PATHWAY_EXR))
SCENE_NAME = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")


def log(message: str) -> None:
    print(f"[render_edge_lab_current_mist] {message}")


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


def repath_exr_node(node: bpy.types.Node, filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"EXR not found: {filepath}")
    if node.image is None:
        raise ValueError(f"Node '{node.name}' has no image")
    node.image.filepath = str(filepath)
    node.image.reload()


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


def rename_family_outputs(output_dir: Path) -> None:
    for rendered_path in sorted(output_dir.glob("*_0001.png")):
        final_path = rendered_path.with_name(rendered_path.name.replace("_0001", ""))
        if final_path.exists():
            final_path.unlink()
        rendered_path.replace(final_path)
        log(f"Renamed {rendered_path.name} -> {final_path.name}")


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
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    log(f"Wrote {output_path}")


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

    workflow_output = require_any_node(node_tree, ["MistOutlines::Outputs"])
    single_input = node_tree.nodes.get("MistOutlines::EXR Input") is not None

    if single_input:
        mist_input = require_any_node(node_tree, ["MistOutlines::EXR Input"])
        repath_exr_node(mist_input, MIST_EXR)
        exr_paths = [MIST_EXR]
        images = [mist_input.image]
    else:
        pathway = require_any_node(node_tree, ["MistOutlines::EXR Pathway"])
        priority = require_any_node(node_tree, ["MistOutlines::EXR Priority"])
        trending = require_any_node(node_tree, ["MistOutlines::EXR Trending"])
        repath_exr_node(pathway, PATHWAY_EXR)
        repath_exr_node(priority, PRIORITY_EXR)
        repath_exr_node(trending, TRENDING_EXR)
        exr_paths = [PATHWAY_EXR, PRIORITY_EXR, TRENDING_EXR]
        images = [pathway.image, priority.image, trending.image]

    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    width, height = detect_resolution(exr_paths, images)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)

    previous_mute_states = {}
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            previous_mute_states[node.name] = node.mute
            node.mute = True

    workflow_output.base_path = str(OUTPUT_DIR)
    workflow_output.format.file_format = "PNG"
    workflow_output.format.color_mode = "RGBA"
    workflow_output.format.color_depth = "8"
    workflow_output.mute = False

    scene.render.filepath = str(OUTPUT_DIR / "_discard_render.png")
    bpy.ops.render.render(write_still=True, scene=scene.name)
    rename_family_outputs(OUTPUT_DIR)

    # Mist's saved file output node is structurally correct now, but Blender
    # still intermittently skips writing these slots. Fall back to direct
    # socket renders from the saved template contract when needed.
    workflow_output.mute = True
    for index, slot in enumerate(workflow_output.file_slots):
        stem = slot.path.rstrip("_")
        out_path = OUTPUT_DIR / f"{stem}.png"
        if not out_path.exists() and workflow_output.inputs[index].links:
            render_socket_to_png(
                scene,
                node_tree,
                workflow_output.inputs[index].links[0].from_socket,
                out_path,
            )

    rename_family_outputs(OUTPUT_DIR)
    discard_path = OUTPUT_DIR / "_discard_render.png"
    if discard_path.exists():
        discard_path.unlink()

    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile" and node.name in previous_mute_states:
            node.mute = previous_mute_states[node.name]

    for slot in workflow_output.file_slots:
        stem = slot.path.rstrip("_")
        out_path = OUTPUT_DIR / f"{stem}.png"
        if out_path.exists():
            log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
