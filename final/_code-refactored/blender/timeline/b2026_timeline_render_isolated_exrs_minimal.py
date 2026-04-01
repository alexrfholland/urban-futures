from __future__ import annotations

from pathlib import Path
import os

import bpy


SCENE_NAME = os.environ["B2026_SCENE_NAME"]
CAMERA_NAME = os.environ["B2026_CAMERA_NAME"]
OUTPUT_DIR = Path(os.environ["B2026_OUTPUT_DIR"])
OUTPUT_BASENAME = os.environ.get("B2026_OUTPUT_BASENAME", SCENE_NAME)
OUTPUT_TAG = os.environ.get("B2026_OUTPUT_TAG", "8k")
RES_X = int(os.environ.get("B2026_RES_X", "7680"))
RES_Y = int(os.environ.get("B2026_RES_Y", "4320"))
RES_PERCENT = int(os.environ.get("B2026_RES_PERCENT", "100"))
SAMPLES = int(os.environ.get("B2026_SAMPLES", "64"))
TARGET_VIEW_LAYERS = tuple(
    name.strip()
    for name in os.environ.get(
        "B2026_TARGET_VIEW_LAYERS",
        "existing_condition,pathway_state,priority_state,trending_state,bioenvelope_positive",
    ).split(",")
    if name.strip()
)


def ensure_render_passes(view_layer: bpy.types.ViewLayer) -> None:
    for attr in (
        "use_pass_combined",
        "use_pass_z",
        "use_pass_mist",
        "use_pass_normal",
        "use_pass_object_index",
        "use_pass_material_index",
        "use_pass_ambient_occlusion",
    ):
        if hasattr(view_layer, attr):
            setattr(view_layer, attr, True)


def clone_scene_for_layer(scene: bpy.types.Scene, view_layer_name: str) -> bpy.types.Scene:
    temp_scene = scene.copy()
    temp_scene.name = f"{scene.name}__{view_layer_name}__minimal_exr"
    temp_scene.camera = bpy.data.objects[CAMERA_NAME]
    temp_scene.render.use_compositing = False
    temp_scene.render.use_sequencer = False
    temp_scene.render.film_transparent = True
    temp_scene.render.resolution_x = RES_X
    temp_scene.render.resolution_y = RES_Y
    temp_scene.render.resolution_percentage = RES_PERCENT
    temp_scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    temp_scene.render.image_settings.color_mode = "RGBA"
    temp_scene.render.image_settings.color_depth = "16"
    temp_scene.render.image_settings.exr_codec = "ZIP"
    temp_scene.view_settings.view_transform = "Standard"
    temp_scene.view_settings.look = "None"
    temp_scene.display_settings.display_device = "sRGB"
    temp_scene.sequencer_colorspace_settings.name = "sRGB"
    if hasattr(temp_scene, "cycles"):
        temp_scene.cycles.samples = SAMPLES
        temp_scene.cycles.preview_samples = SAMPLES

    for layer in list(temp_scene.view_layers):
        if layer.name != view_layer_name:
            temp_scene.view_layers.remove(layer)

    target_layer = temp_scene.view_layers.get(view_layer_name)
    if target_layer is None:
        raise ValueError(f"Failed to isolate view layer '{view_layer_name}'")
    ensure_render_passes(target_layer)
    return temp_scene


def main() -> None:
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}")

    camera = bpy.data.objects.get(CAMERA_NAME)
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    for view_layer_name in TARGET_VIEW_LAYERS:
        if scene.view_layers.get(view_layer_name) is None:
            continue
        temp_scene = clone_scene_for_layer(scene, view_layer_name)
        output_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}_{view_layer_name}_{OUTPUT_TAG}.exr"
        temp_scene.render.filepath = str(output_path)
        try:
            bpy.ops.render.render(
                write_still=True,
                scene=temp_scene.name,
                layer=view_layer_name,
                use_viewport=False,
            )
        finally:
            bpy.data.scenes.remove(temp_scene)
        outputs.append(str(output_path))
        print(f"RENDER_DONE {output_path}")

    print(f"Rendered {len(outputs)} isolated EXRs for {SCENE_NAME}")


if __name__ == "__main__":
    main()
