from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import bpy

_THIS = Path(__file__).resolve()
REPO_ROOT = next(p.parent for p in _THIS.parents if p.name == "_futureSim_refactored")
_SCRIPTS_DIR = _THIS.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _exr_header import read_exr_dimensions  # noqa: E402

COMPOSITOR_ROOT = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"
OUTPUT_BASE = REPO_ROOT / "_data-refactored" / "compositor" / "outputs"


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


BLEND_PATH = env_path(
    "COMPOSITOR_BLEND_PATH",
    str(CANONICAL_ROOT / "compositor_mist_complex_outlines.blend"),
)
EXR_PATH = env_path("COMPOSITOR_MIST_COMPLEX_EXR", "")
OUTPUT_DIR = env_path(
    "COMPOSITOR_OUTPUT_DIR",
    str(OUTPUT_BASE / "edge_lab_final_template" / "current" / "mist_complex_outlines"),
)
SCENE_NAME = os.environ.get("COMPOSITOR_SCENE_NAME", "Current")


def log(message: str) -> None:
    print(f"[render_edge_lab_current_mist_complex_outlines] {message}")


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


def detect_resolution(exr_path: Path) -> tuple[int, int]:
    width, height = read_exr_dimensions(str(exr_path))
    return width, height


def derive_output_name(exr_path: Path) -> str:
    stem = exr_path.stem
    stem = re.sub(r"__(?:2k|4k|8k)(?:64s)?$", "", stem)
    parts = stem.split("__")
    if len(parts) >= 2:
        derived = "__".join(parts[1:])
    else:
        derived = stem
    if not derived:
        raise ValueError(f"Could not derive output name from EXR path: {exr_path}")
    return derived


def rename_outputs(output_dir: Path) -> None:
    for rendered in sorted(output_dir.glob("*_0001.png")):
        final = rendered.with_name(rendered.name.replace("_0001", ""))
        if final.exists():
            final.unlink()
        rendered.replace(final)
        log(f"Renamed {rendered.name} -> {final.name}")


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
    exr_raw = os.environ.get("COMPOSITOR_MIST_COMPLEX_EXR", "").strip()
    if not exr_raw:
        raise ValueError("COMPOSITOR_MIST_COMPLEX_EXR is required")
    exr_path = Path(exr_raw).expanduser()
    if not exr_path.exists() or not exr_path.is_file():
        raise FileNotFoundError(f"EXR not found: {exr_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None or scene.node_tree is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {BLEND_PATH}")

    node_tree = scene.node_tree
    set_standard_view(scene)

    exr_node = require_node(node_tree, "MistComplexOutlines::EXR Input")
    output_node = require_node(node_tree, "MistComplexOutlines::Outputs")
    repath_exr_node(exr_node, exr_path)

    width, height = detect_resolution(exr_path)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_current = 1
    set_standard_view(scene)

    output_name = derive_output_name(exr_path)
    output_stem = f"{output_name}__whole_forest_outline_v8_t10"
    slot = output_node.file_slots[0]
    slot.path = f"{output_stem}_"
    output_node.base_path = str(OUTPUT_DIR)
    output_node.format.file_format = "PNG"
    output_node.format.color_mode = "RGBA"
    output_node.format.color_depth = "8"

    previous_mute_states = {}
    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            previous_mute_states[node.name] = node.mute
            node.mute = True
    output_node.mute = False

    scene.render.filepath = str(OUTPUT_DIR / "_discard_render.png")
    bpy.ops.render.render(write_still=True, scene=scene.name)
    rename_outputs(OUTPUT_DIR)

    output_path = OUTPUT_DIR / f"{output_stem}.png"
    if not output_path.exists() and output_node.inputs[0].links:
        output_node.mute = True
        render_socket_to_png(
            scene,
            node_tree,
            output_node.inputs[0].links[0].from_socket,
            output_path,
        )

    rename_outputs(OUTPUT_DIR)

    discard_path = OUTPUT_DIR / "_discard_render.png"
    if discard_path.exists():
        discard_path.unlink()

    for node in node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile" and node.name in previous_mute_states:
            node.mute = previous_mute_states[node.name]

    if output_path.exists():
        log(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
