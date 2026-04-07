from __future__ import annotations

from pathlib import Path
import os

import bpy


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


SCENE_NAME = env_str("B2026_SCENE_NAME")
CAMERA_NAME = env_str("B2026_CAMERA_NAME")
OUTPUT_DIR_ENV = env_str("B2026_OUTPUT_DIR")
OUTPUT_PREFIX = env_str("B2026_OUTPUT_PREFIX")
TARGET_VIEW_LAYERS = tuple(
    name.strip()
    for name in env_str("B2026_TARGET_VIEW_LAYERS").split(",")
    if name.strip()
)
RES_X = int(env_str("B2026_RES_X", "0") or "0")
RES_Y = int(env_str("B2026_RES_Y", "0") or "0")
RES_PERCENT = int(env_str("B2026_RES_PERCENT", "100") or "100")


def resolve_scene() -> bpy.types.Scene:
    if SCENE_NAME:
        scene = bpy.data.scenes.get(SCENE_NAME)
        if scene is None:
            raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}")
        return scene
    return bpy.context.scene


def resolve_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    if CAMERA_NAME:
        camera = bpy.data.objects.get(CAMERA_NAME)
        if camera is None or camera.type != "CAMERA":
            raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}")
        return camera
    if scene.camera is None or scene.camera.type != "CAMERA":
        raise ValueError(f"Scene '{scene.name}' has no active camera")
    return scene.camera


def resolve_output_dir(scene: bpy.types.Scene) -> Path:
    if OUTPUT_DIR_ENV:
        return Path(OUTPUT_DIR_ENV)
    blend_path = Path(bpy.data.filepath)
    if blend_path.name:
        return blend_path.parent / "renders" / f"{scene.name}_workbench_view_layers"
    return Path.cwd() / "renders" / f"{scene.name}_workbench_view_layers"


def main() -> None:
    scene = resolve_scene()
    camera = resolve_camera(scene)
    output_dir = resolve_output_dir(scene)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_engine = scene.render.engine
    original_filepath = scene.render.filepath
    original_camera = scene.camera
    original_use_compositing = scene.render.use_compositing
    original_use_sequencer = scene.render.use_sequencer
    original_use_nodes = scene.use_nodes
    original_res_x = scene.render.resolution_x
    original_res_y = scene.render.resolution_y
    original_res_percent = scene.render.resolution_percentage
    original_transform = scene.view_settings.view_transform
    original_look = scene.view_settings.look
    original_device = scene.display_settings.display_device
    original_layer_use = {view_layer.name: view_layer.use for view_layer in scene.view_layers}

    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.use_nodes = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"

    if RES_X > 0:
        scene.render.resolution_x = RES_X
    if RES_Y > 0:
        scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = RES_PERCENT

    prefix = OUTPUT_PREFIX or scene.name
    target_layers = TARGET_VIEW_LAYERS or tuple(view_layer.name for view_layer in scene.view_layers)

    try:
        for layer_name in target_layers:
            if scene.view_layers.get(layer_name) is None:
                print(f"SKIP missing view layer: {layer_name}")
                continue

            for view_layer in scene.view_layers:
                view_layer.use = view_layer.name == layer_name

            output_path = output_dir / f"{prefix}_{layer_name}_workbench.png"
            scene.render.filepath = str(output_path)
            print(f"RENDERING {layer_name} -> {output_path}")
            bpy.ops.render.render(write_still=True, scene=scene.name, layer=layer_name, use_viewport=False)
    finally:
        scene.render.engine = original_engine
        scene.render.filepath = original_filepath
        scene.camera = original_camera
        scene.render.use_compositing = original_use_compositing
        scene.render.use_sequencer = original_use_sequencer
        scene.use_nodes = original_use_nodes
        scene.render.resolution_x = original_res_x
        scene.render.resolution_y = original_res_y
        scene.render.resolution_percentage = original_res_percent
        scene.view_settings.view_transform = original_transform
        scene.view_settings.look = original_look
        scene.display_settings.display_device = original_device
        for view_layer in scene.view_layers:
            view_layer.use = original_layer_use.get(view_layer.name, True)

    print(f"Rendered workbench view layers for {scene.name} to {output_dir}")


if __name__ == "__main__":
    main()
