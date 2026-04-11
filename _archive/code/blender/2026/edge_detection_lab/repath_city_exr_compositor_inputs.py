"""
Repath the lightweight city EXR compositor to another EXR input set.

This is the generic adapter for variants such as baseline or alternate camera
exports. The compositor graph stays the same. Only the multilayer EXR image
nodes are repointed.

Expected node names in the input blend:
  EXR :: pathway_state
  EXR :: existing_condition
  EXR :: priority
  EXR :: bioenvelope
  EXR :: trending_state

Usage patterns:
1. Set B2026_EXR_ROOT to repath all matching layers from canonical filenames:
     {exr_root}/pathway_state.exr
     {exr_root}/priority.exr
     {exr_root}/existing_condition.exr
     {exr_root}/bioenvelope.exr
     {exr_root}/trending_state.exr
2. Set one or more B2026_EXR_PATH__<VIEW_LAYER> overrides for explicit files,
   for example:
     B2026_EXR_PATH__PATHWAY_STATE=/path/to/pathway_state.exr

`bioenvelope` and `trending_state` are optional. The baseline case usually only
provides:
  pathway_state.exr
  priority.exr
  existing_condition.exr

This script does not distinguish zoom3x vs worldcam. Camera choice belongs to
the heavy scene that rendered the EXRs.
"""

from __future__ import annotations

import os
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"

IMAGE_NODE_PREFIX = "EXR :: "
LEGACY_IMAGE_NODE_PREFIX = "City EXR :: "
OVERRIDE_PREFIX = "B2026_EXR_PATH__"

INPUT_BLEND_PATH = Path(
    os.environ.get(
        "B2026_INPUT_BLEND_PATH",
        str(DATA_ROOT / "city_exr_compositor_lightweight.blend"),
    )
)
OUTPUT_BLEND_PATH = Path(
    os.environ.get(
        "B2026_OUTPUT_BLEND_PATH",
        str(DATA_ROOT / "city_exr_compositor_lightweight_repathed.blend"),
    )
)
TARGET_SCENE_NAME = os.environ.get("B2026_TARGET_SCENE_NAME", "City")
EXR_ROOT = os.environ.get("B2026_EXR_ROOT", "").strip()
RUN_TEST = os.environ.get("B2026_RUN_TEST", "0").strip().lower() in {"1", "true", "yes", "on"}
TEST_RENDER_PATH = Path(
    os.environ.get(
        "B2026_TEST_RENDER_PATH",
        str(DATA_ROOT / "city_exr_compositor_lightweight_repathed_test.png"),
    )
)

LAYER_FILENAME_STEMS = {
    "pathway_state": "pathway_state",
    "existing_condition": "existing_condition",
    "city_priority": "priority",
    "city_bioenvelope": "bioenvelope",
    "trending_state": "trending_state",
    "priority": "priority",
    "bioenvelope": "bioenvelope",
}

CANONICAL_LAYER_NAMES = {
    "pathway_state": "pathway_state",
    "existing_condition": "existing_condition",
    "city_priority": "priority",
    "city_bioenvelope": "bioenvelope",
    "trending_state": "trending_state",
    "priority": "priority",
    "bioenvelope": "bioenvelope",
}
IMAGE_LAYOUT_ORDER = (
    "pathway_state",
    "priority",
    "existing_condition",
    "bioenvelope",
    "trending_state",
)


def log(message: str) -> None:
    print(f"[repath_city_exr_compositor_inputs] {message}")


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


def scene_image_nodes(scene: bpy.types.Scene) -> dict[str, bpy.types.Node]:
    if scene.node_tree is None:
        raise ValueError(f"Scene '{scene.name}' does not have a compositor node tree.")

    nodes: dict[str, bpy.types.Node] = {}
    for node in scene.node_tree.nodes:
        if node.bl_idname != "CompositorNodeImage":
            continue
        layer_name = None
        if node.name.startswith(IMAGE_NODE_PREFIX):
            layer_name = node.name.removeprefix(IMAGE_NODE_PREFIX)
        elif node.name.startswith(LEGACY_IMAGE_NODE_PREFIX):
            layer_name = node.name.removeprefix(LEGACY_IMAGE_NODE_PREFIX)
        elif node.label.startswith(IMAGE_NODE_PREFIX):
            layer_name = node.label.removeprefix(IMAGE_NODE_PREFIX).split(" [", 1)[0]
        elif node.label.startswith(LEGACY_IMAGE_NODE_PREFIX):
            layer_name = node.label.removeprefix(LEGACY_IMAGE_NODE_PREFIX).split(" [", 1)[0]
        if layer_name is None:
            continue
        canonical_layer_name = CANONICAL_LAYER_NAMES.get(layer_name, layer_name)
        node.name = f"{IMAGE_NODE_PREFIX}{canonical_layer_name}"
        node.label = f"{IMAGE_NODE_PREFIX}{canonical_layer_name}"
        nodes[canonical_layer_name] = node
    if not nodes:
        raise ValueError(
            f"Scene '{scene.name}' does not contain any EXR image nodes starting with '{IMAGE_NODE_PREFIX}' or '{LEGACY_IMAGE_NODE_PREFIX}'."
        )
    return nodes


def ensure_input_frame(scene: bpy.types.Scene) -> bpy.types.Node:
    node_tree = scene.node_tree
    assert node_tree is not None
    frame = node_tree.nodes.get("Frame EXR Inputs")
    if frame is None:
        frame = node_tree.nodes.new("NodeFrame")
        frame.name = "Frame EXR Inputs"
    frame.label = "EXR Inputs"
    frame.label_size = 18
    frame.use_custom_color = True
    frame.color = (0.16, 0.16, 0.16)
    frame.location = (-2200.0, 700.0)
    frame.shrink = False
    return frame


def layout_image_nodes(scene: bpy.types.Scene, image_nodes: dict[str, bpy.types.Node]) -> None:
    frame = ensure_input_frame(scene)
    ordered_layers = [layer for layer in IMAGE_LAYOUT_ORDER if layer in image_nodes]
    ordered_layers.extend(layer for layer in sorted(image_nodes) if layer not in ordered_layers)
    start_y = 500.0
    step_y = -260.0
    for index, layer_name in enumerate(ordered_layers):
        node = image_nodes[layer_name]
        node.parent = frame
        node.location = (-1960.0, start_y + step_y * index)
        node.width = max(node.width, 260.0)


def load_linear_exr(exr_path: Path) -> bpy.types.Image:
    image = bpy.data.images.load(str(exr_path), check_existing=True)
    try:
        image.colorspace_settings.name = "Linear"
    except Exception:
        pass
    return image


def env_override_map() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for key, value in os.environ.items():
        if not key.startswith(OVERRIDE_PREFIX):
            continue
        suffix = key.removeprefix(OVERRIDE_PREFIX).strip()
        if not suffix or not value.strip():
            continue
        layer_name = suffix.lower()
        mapping[layer_name] = Path(value.strip())
    return mapping


def candidate_exr_paths(root: Path, layer_name: str) -> list[Path]:
    stem = LAYER_FILENAME_STEMS.get(layer_name, layer_name)
    return [
        root / f"{stem}.exr",
        root / f"{stem}0000.exr",
    ]


def resolve_root_exr(root: Path, layer_name: str) -> Path | None:
    candidates = [path for path in candidate_exr_paths(root, layer_name) if path.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def repath_scene_inputs(scene: bpy.types.Scene) -> tuple[list[str], list[str], list[str]]:
    image_nodes = scene_image_nodes(scene)
    layout_image_nodes(scene, image_nodes)
    overrides = env_override_map()
    exr_root = Path(EXR_ROOT) if EXR_ROOT else None

    if not overrides and exr_root is None:
        raise ValueError(
            "Provide B2026_EXR_ROOT or one or more B2026_EXR_PATH__<VIEW_LAYER> overrides."
        )

    repathed: list[str] = []
    untouched: list[str] = []
    missing: list[str] = []

    for layer_name, node in image_nodes.items():
        explicit_path = overrides.get(layer_name)
        target_path: Path | None = None

        if explicit_path is not None:
            if not explicit_path.exists():
                missing.append(f"{layer_name} -> {explicit_path}")
                continue
            target_path = explicit_path
        elif exr_root is not None:
            target_path = resolve_root_exr(exr_root, layer_name)

        if target_path is None:
            untouched.append(layer_name)
            continue

        node.image = load_linear_exr(target_path)
        node.label = f"{node.name} [{target_path.name}]"
        repathed.append(f"{layer_name} -> {target_path}")

    return repathed, untouched, missing


def stamp_scene_metadata(scene: bpy.types.Scene, repathed: list[str], untouched: list[str]) -> None:
    scene["b2026_input_blend_path"] = str(INPUT_BLEND_PATH)
    scene["b2026_repath_exr_root"] = EXR_ROOT
    scene["b2026_repathed_layers"] = " | ".join(repathed)
    scene["b2026_untouched_layers"] = ",".join(untouched)


def save_output_blend() -> None:
    OUTPUT_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND_PATH), copy=True)


def verify(scene_name: str, repathed_layers: list[str]) -> None:
    bpy.ops.wm.open_mainfile(filepath=str(OUTPUT_BLEND_PATH))
    scene = resolve_scene(scene_name)
    nodes = scene_image_nodes(scene)

    for item in repathed_layers:
        layer_name, _, path_str = item.partition(" -> ")
        node = nodes.get(layer_name)
        if node is None:
            raise RuntimeError(f"Missing image node for repathed layer '{layer_name}'.")
        if node.image is None:
            raise RuntimeError(f"Image node '{node.name}' has no image after repath.")
        actual_path = Path(bpy.path.abspath(node.image.filepath))
        expected_path = Path(path_str)
        if actual_path != expected_path:
            raise RuntimeError(
                f"Node '{node.name}' points to '{actual_path}', expected '{expected_path}'."
            )

    if RUN_TEST:
        scene.render.filepath = str(TEST_RENDER_PATH)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        bpy.ops.render.render(write_still=True, scene=scene.name)
        if not TEST_RENDER_PATH.exists():
            raise RuntimeError(f"Test render did not write: {TEST_RENDER_PATH}")
        log(f"Test render: {TEST_RENDER_PATH}")


def main() -> None:
    require_file(INPUT_BLEND_PATH, "Input compositor blend")
    bpy.ops.wm.open_mainfile(filepath=str(INPUT_BLEND_PATH))

    scene = resolve_scene(TARGET_SCENE_NAME)
    repathed, untouched, missing = repath_scene_inputs(scene)
    if missing:
        raise FileNotFoundError(f"Missing explicit EXR overrides: {missing}")

    stamp_scene_metadata(scene, repathed, untouched)
    save_output_blend()

    log(f"Input blend: {INPUT_BLEND_PATH}")
    log(f"Output blend: {OUTPUT_BLEND_PATH}")
    log(f"EXR root: {EXR_ROOT or '[none]'}")
    log(f"Repathed: {', '.join(repathed) if repathed else 'none'}")
    log(f"Untouched: {', '.join(untouched) if untouched else 'none'}")

    verify(scene.name, repathed)


if __name__ == "__main__":
    main()
