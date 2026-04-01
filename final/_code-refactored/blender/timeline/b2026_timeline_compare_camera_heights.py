from __future__ import annotations

import math
import os
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


SCENE_NAME = os.environ["B2026_SCENE_NAME"]
CAMERA_NAME = os.environ["B2026_CAMERA_NAME"]
SITE_NAME = os.environ["B2026_SITE_NAME"]
TARGET_LAYER = os.environ.get("B2026_TARGET_VIEW_LAYER", "pathway_state")
LOW_Z = float(os.environ["B2026_LOW_Z"])
HIGH_Z = float(os.environ["B2026_HIGH_Z"])
FOCAL_X = float(os.environ["B2026_FOCAL_X"])
FOCAL_Y = float(os.environ["B2026_FOCAL_Y"])
FOCAL_Z = float(os.environ["B2026_FOCAL_Z"])
OUTPUT_DIR = Path(os.environ["B2026_OUTPUT_DIR"])
RES_X = int(os.environ.get("B2026_RES_X", "1920"))
RES_Y = int(os.environ.get("B2026_RES_Y", "1080"))
SAMPLES = int(os.environ.get("B2026_SAMPLES", "24"))


def build_camera_orientation(location: Vector, target: Vector, up_hint: Vector) -> Matrix:
    forward = (target - location).normalized()
    z_axis = (-forward).normalized()

    x_axis = up_hint.cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((0.0, 0.0, 1.0)).cross(z_axis)
    x_axis.normalize()

    y_axis = z_axis.cross(x_axis).normalized()
    return Matrix((x_axis, y_axis, z_axis)).transposed()


def ensure_camera_variant(base_camera: bpy.types.Object, suffix: str, z_value: float, target: Vector) -> bpy.types.Object:
    name = f"{base_camera.name}__{suffix}"
    camera = bpy.data.objects.get(name)
    if camera is None:
        camera = base_camera.copy()
        camera.data = base_camera.data.copy()
        camera.name = name
        camera.data.name = f"{base_camera.data.name}__{suffix}"
        for collection in base_camera.users_collection:
            collection.objects.link(camera)

    location = base_camera.location.copy()
    location.z = z_value
    camera.location = location

    up_hint = (base_camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))).normalized()
    camera.rotation_euler = build_camera_orientation(location, target, up_hint).to_euler()
    camera.data.sensor_fit = base_camera.data.sensor_fit
    camera.data.lens = base_camera.data.lens
    camera.data.angle = base_camera.data.angle
    camera.data.clip_start = base_camera.data.clip_start
    camera.data.clip_end = base_camera.data.clip_end
    return camera


def render_preview(scene: bpy.types.Scene, camera: bpy.types.Object, suffix: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene.camera = camera
    for view_layer in scene.view_layers:
        view_layer.use = view_layer.name == TARGET_LAYER
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
    scene.render.filepath = str(OUTPUT_DIR / f"{SITE_NAME}_{TARGET_LAYER}_{suffix}_preview_v1.png")

    if hasattr(scene, "cycles"):
        scene.cycles.samples = SAMPLES
        scene.cycles.preview_samples = SAMPLES

    bpy.ops.render.render(write_still=True, scene=scene.name, layer=TARGET_LAYER, use_viewport=False)


def main() -> None:
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found.")

    base_camera = bpy.data.objects.get(CAMERA_NAME)
    if base_camera is None or base_camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' not found.")

    target = Vector((FOCAL_X, FOCAL_Y, FOCAL_Z))
    low_camera = ensure_camera_variant(base_camera, "height_parade", LOW_Z, target)
    high_camera = ensure_camera_variant(base_camera, "height_city_street", HIGH_Z, target)

    render_preview(scene, low_camera, "height_parade")
    render_preview(scene, high_camera, "height_city_street")

    bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
