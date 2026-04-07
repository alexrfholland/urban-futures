from __future__ import annotations

import os
from pathlib import Path

import bpy


SCENE_NAME = os.environ["B2026_SCENE_NAME"]
CAMERA_NAME = os.environ["B2026_CAMERA_NAME"]
VIEW_LAYER_NAME = os.environ.get("B2026_VIEW_LAYER_NAME", "debug_camera_framing")
OUTPUT_PATH = Path(os.environ["B2026_OUTPUT_PATH"])
SAMPLES = int(os.environ.get("B2026_SAMPLES", "8"))
RES_X = int(os.environ.get("B2026_RES_X", "1920"))
RES_Y = int(os.environ.get("B2026_RES_Y", "1080"))


def main() -> None:
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}.")

    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}.")

    view_layer = scene.view_layers.get(VIEW_LAYER_NAME)
    if view_layer is None:
        raise ValueError(f"View layer '{VIEW_LAYER_NAME}' not found in scene '{scene.name}'.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    original_use = {layer.name: layer.use for layer in scene.view_layers}
    original_camera = scene.camera

    try:
        for layer in scene.view_layers:
            layer.use = layer.name == VIEW_LAYER_NAME
        scene.camera = camera
        scene.render.use_compositing = False
        scene.render.use_sequencer = False
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.display_settings.display_device = "sRGB"
        scene.render.resolution_x = RES_X
        scene.render.resolution_y = RES_Y
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(OUTPUT_PATH)
        if hasattr(scene, "cycles"):
            scene.cycles.samples = SAMPLES
            scene.cycles.preview_samples = SAMPLES
        bpy.ops.render.render(write_still=True, scene=scene.name, layer=VIEW_LAYER_NAME, use_viewport=False)
    finally:
        scene.camera = original_camera
        for layer in scene.view_layers:
            layer.use = original_use.get(layer.name, True)

    print(f"RENDERED_DEBUG_CAMERA_FRAMING {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
