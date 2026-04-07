from __future__ import annotations

import os
import sys
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_setup_view_layer_exr_outputs as exr_setup


SCENE_NAME = os.environ["B2026_SCENE_NAME"]
CAMERA_NAME = os.environ["B2026_CAMERA_NAME"]
TARGET_LAYER = os.environ["B2026_TARGET_VIEW_LAYER"]
RES_X = int(os.environ.get("B2026_RES_X", "7680"))
RES_Y = int(os.environ.get("B2026_RES_Y", "4320"))
RES_PERCENT = int(os.environ.get("B2026_RES_PERCENT", "100"))
SAMPLES = int(os.environ.get("B2026_SAMPLES", "64"))


def mute_nonmanaged_file_outputs(scene: bpy.types.Scene) -> None:
    if not scene.use_nodes or scene.node_tree is None:
        return
    for node in scene.node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile" and not node.name.startswith(exr_setup.NODE_PREFIX):
            node.mute = True


def main() -> None:
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}")

    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}")

    if scene.view_layers.get(TARGET_LAYER) is None:
        raise ValueError(f"View layer '{TARGET_LAYER}' not found in scene '{SCENE_NAME}'")

    scene.camera = camera
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    scene.render.film_transparent = True
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = RES_PERCENT
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"
    scene.sequencer_colorspace_settings.name = "sRGB"
    scene.render.filepath = str(Path(os.environ["B2026_OUTPUT_DIR"]) / f"__discard_{TARGET_LAYER}")

    if hasattr(scene, "cycles"):
        scene.cycles.samples = SAMPLES
        scene.cycles.preview_samples = SAMPLES

    for view_layer in scene.view_layers:
        view_layer.use = view_layer.name == TARGET_LAYER

    mute_nonmanaged_file_outputs(scene)
    exr_setup.main()

    bpy.ops.render.render(write_still=False, scene=scene.name, use_viewport=False)
    print(f"Rendered single EXR layer={TARGET_LAYER} scene={SCENE_NAME}")


if __name__ == "__main__":
    main()
