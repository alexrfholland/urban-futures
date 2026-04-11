"""
Build a baseline lightweight compositor blend from the existing city EXR compositor.

This variant stays inside the edge-lab/lightweight-compositor workflow instead of
rebuilding from the heavy source scene. It opens the current lightweight compositor
blend, reattaches the three baseline-backed EXR image nodes, and saves the result
as a separate compositor-only blend.

The untouched view-layer image nodes remain as they were in the template because
no baseline EXRs were provided for `city_bioenvelope` or `trending_state`.
"""

from __future__ import annotations

import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"

DEFAULT_TEMPLATE_BLEND_PATH = DATA_ROOT / "city_exr_compositor_lightweight.blend"
DEFAULT_OUTPUT_BLEND_PATH = DATA_ROOT / "city_exr_compositor_lightweight_baseline.blend"
DEFAULT_TEST_RENDER_PATH = DATA_ROOT / "city_exr_compositor_lightweight_baseline_test.png"

TARGET_SCENE_NAME = os.environ.get("B2026_TARGET_SCENE_NAME", "City")
TEMPLATE_BLEND_PATH = Path(os.environ.get("B2026_TEMPLATE_BLEND_PATH", str(DEFAULT_TEMPLATE_BLEND_PATH)))
OUTPUT_BLEND_PATH = Path(os.environ.get("B2026_OUTPUT_BLEND_PATH", str(DEFAULT_OUTPUT_BLEND_PATH)))
TEST_RENDER_PATH = Path(os.environ.get("B2026_TEST_RENDER_PATH", str(DEFAULT_TEST_RENDER_PATH)))
RUN_TEST = os.environ.get("B2026_RUN_TEST", "1").strip().lower() in {"1", "true", "yes", "on"}
IMAGE_NODE_PREFIX = "City EXR :: "

BASELINE_EXR_MAP = {
    "pathway_state": Path(
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_baseline-city/city-pathway_state.exr"
    ),
    "city_priority": Path(
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_baseline-city/city-city_priority.exr"
    ),
    "existing_condition": Path(
        "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes6_baseline-city/city-existing_condition.exr"
    ),
}


def log(message: str) -> None:
    print(f"[build_city_exr_compositor_lightweight_baseline] {message}")


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_scene(name: str) -> bpy.types.Scene:
    exact = bpy.data.scenes.get(name)
    if exact is not None:
        return exact

    lowered = name.casefold()
    for scene in bpy.data.scenes:
        if scene.name.casefold() == lowered:
            return scene

    available = ", ".join(scene.name for scene in bpy.data.scenes)
    raise ValueError(f"Scene '{name}' was not found. Available scenes: {available}")


def require_node(scene: bpy.types.Scene, layer_name: str) -> bpy.types.Node:
    node_name = f"{IMAGE_NODE_PREFIX}{layer_name}"
    node = scene.node_tree.nodes.get(node_name)
    if node is None or node.bl_idname != "CompositorNodeImage":
        raise ValueError(f"Expected EXR image node '{node_name}' in scene '{scene.name}'.")
    return node


def load_linear_exr(exr_path: Path) -> bpy.types.Image:
    image = bpy.data.images.load(str(exr_path), check_existing=True)
    try:
        image.colorspace_settings.name = "Linear"
    except Exception:
        pass
    return image


def attach_baseline_exrs(scene: bpy.types.Scene) -> list[str]:
    repathed: list[str] = []
    for layer_name, exr_path in BASELINE_EXR_MAP.items():
        require_file(exr_path, f"Baseline EXR for {layer_name}")
        image_node = require_node(scene, layer_name)
        image = load_linear_exr(exr_path)
        image_node.image = image
        image_node.label = f"{image_node.name} [baseline]"
        repathed.append(f"{layer_name} -> {exr_path}")
    return repathed


def stamp_scene_metadata(scene: bpy.types.Scene) -> None:
    scene["b2026_baseline_template_blend_path"] = str(TEMPLATE_BLEND_PATH)
    scene["b2026_baseline_attached_layers"] = ",".join(BASELINE_EXR_MAP.keys())
    for layer_name, exr_path in BASELINE_EXR_MAP.items():
        scene[f"b2026_baseline_exr__{layer_name}"] = str(exr_path)


def save_output_blend() -> None:
    OUTPUT_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND_PATH), copy=True)


def verify(scene_name: str) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(OUTPUT_BLEND_PATH))
    scene = resolve_scene(scene_name)

    for layer_name, exr_path in BASELINE_EXR_MAP.items():
        node = require_node(scene, layer_name)
        if node.image is None or Path(bpy.path.abspath(node.image.filepath)) != exr_path:
            raise RuntimeError(f"Node '{node.name}' is not attached to expected baseline EXR '{exr_path}'.")

    if RUN_TEST:
        scene.render.filepath = str(TEST_RENDER_PATH)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        bpy.ops.render.render(write_still=True, scene=scene.name)
        if not TEST_RENDER_PATH.exists():
            raise RuntimeError(f"Test render did not write: {TEST_RENDER_PATH}")
        log(f"Test render: {TEST_RENDER_PATH}")


def main() -> None:
    require_file(TEMPLATE_BLEND_PATH, "Template lightweight compositor blend")
    for layer_name, exr_path in BASELINE_EXR_MAP.items():
        require_file(exr_path, f"Baseline EXR for {layer_name}")

    bpy.ops.wm.open_mainfile(filepath=str(TEMPLATE_BLEND_PATH))
    scene = resolve_scene(TARGET_SCENE_NAME)
    if scene.node_tree is None:
        raise ValueError(f"Scene '{scene.name}' does not have a compositor node tree.")

    repathed = attach_baseline_exrs(scene)
    stamp_scene_metadata(scene)
    save_output_blend()

    log(f"Template blend: {TEMPLATE_BLEND_PATH}")
    log(f"Output blend: {OUTPUT_BLEND_PATH}")
    for item in repathed:
        log(f"Repathed {item}")
    log("Untouched image nodes remain on their template EXRs: city_bioenvelope, trending_state")

    verify(scene.name)


if __name__ == "__main__":
    main()
