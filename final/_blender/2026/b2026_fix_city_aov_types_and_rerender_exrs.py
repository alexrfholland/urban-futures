import importlib.util
from pathlib import Path

import bpy


TARGET_SCENE_NAME = "city"
REFERENCE_VIEW_LAYER_NAME = "pathway_state"
REPO_ROOT = Path(__file__).resolve().parents[3]
DISCARD_RENDER_PREFIX = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "_discard_city_exr_render"
EXR_SETUP_SCRIPT = Path(__file__).resolve().parent / "b2026_setup_view_layer_exr_outputs.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_scene() -> bpy.types.Scene:
    scene = bpy.data.scenes.get(TARGET_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{TARGET_SCENE_NAME}' was not found.")
    return scene


def require_reference_view_layer(scene: bpy.types.Scene):
    view_layer = scene.view_layers.get(REFERENCE_VIEW_LAYER_NAME)
    if view_layer is None:
        raise ValueError(
            f"Reference view layer '{REFERENCE_VIEW_LAYER_NAME}' was not found in scene '{scene.name}'."
        )
    return view_layer


def ensure_view_layer_aov(view_layer, name: str, aov_type: str):
    for existing in view_layer.aovs:
        if existing.name == name:
            existing.type = aov_type
            return existing
    aov = view_layer.aovs.add()
    aov.name = name
    aov.type = aov_type
    return aov


def harmonize_aov_types(scene: bpy.types.Scene):
    reference_layer = require_reference_view_layer(scene)
    reference_types = {aov.name: aov.type for aov in reference_layer.aovs}
    fixes = []

    for view_layer in scene.view_layers:
        for aov_name, expected_type in reference_types.items():
            existing = next((aov for aov in view_layer.aovs if aov.name == aov_name), None)
            if existing is None:
                ensure_view_layer_aov(view_layer, aov_name, expected_type)
                fixes.append((view_layer.name, aov_name, None, expected_type))
                continue
            if existing.type != expected_type:
                original = existing.type
                existing.type = expected_type
                fixes.append((view_layer.name, aov_name, original, expected_type))

    return fixes


def mute_non_exr_file_outputs(scene: bpy.types.Scene, exr_output_prefix: str):
    original = {}
    for node in scene.node_tree.nodes:
        if node.bl_idname != "CompositorNodeOutputFile":
            continue
        original[node.name] = node.mute
        node.mute = not node.name.startswith(exr_output_prefix)
    return original


def restore_file_output_mutes(scene: bpy.types.Scene, original_mutes: dict[str, bool]):
    for node_name, mute in original_mutes.items():
        node = scene.node_tree.nodes.get(node_name)
        if node is not None:
            node.mute = mute


def cleanup_discard_render_files():
    for path in DISCARD_RENDER_PREFIX.parent.glob(f"{DISCARD_RENDER_PREFIX.name}*"):
        if path.is_file():
            path.unlink()


def render_exrs(scene: bpy.types.Scene, exr_module):
    original_filepath = scene.render.filepath
    original_mutes = mute_non_exr_file_outputs(scene, exr_module.OUTPUT_NODE_PREFIX)
    try:
        scene.render.filepath = str(DISCARD_RENDER_PREFIX)
        bpy.ops.render.render(write_still=True, scene=scene.name)
    finally:
        scene.render.filepath = original_filepath
        restore_file_output_mutes(scene, original_mutes)
        cleanup_discard_render_files()


def main():
    scene = require_scene()
    fixes = harmonize_aov_types(scene)

    exr_module = load_module("b2026_setup_view_layer_exr_outputs_runtime", EXR_SETUP_SCRIPT)
    output_dir = exr_module.build_outputs(scene)

    bpy.ops.wm.save_mainfile()
    render_exrs(scene, exr_module)
    bpy.ops.wm.save_mainfile()

    print(f"Saved blend: {bpy.data.filepath}")
    if fixes:
        print("AOV type fixes:")
        for view_layer_name, aov_name, original, updated in fixes:
            print(f"- {view_layer_name}: {aov_name} {original} -> {updated}")
    else:
        print("AOV type fixes: none")

    for view_layer in scene.view_layers:
        print(f"EXR {exr_module.expected_final_path(output_dir, scene.name, view_layer.name)}")


if __name__ == "__main__":
    main()
