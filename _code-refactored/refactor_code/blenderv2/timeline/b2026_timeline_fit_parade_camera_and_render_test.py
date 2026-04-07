from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import bpy
import bpy_extras.object_utils
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import b2026_timeline_render_parade_cleaned_pack as render_pack


BLEND_SCENE_NAME = "parade"
CAMERA_NAME = "paraview_camera_parade"
TARGET_VIEW_LAYER = "pathway_state"
TEST_OUTPUT = Path(
    r"D:\2026 Arboreal Futures\data\renders\paraview\parade_lightweight_cleaned_camera_fit_test\parade_paraview_camera_parade_pathway_state_camera_fit_test.png"
)
TEST_RESOLUTION = (3840, 2160)
TEST_PERCENTAGE = 100
TEST_SAMPLES = 32
FIT_MARGIN = 0.02
FIT_ITERATIONS = 28


def set_collection_render_state(collection_names, hide_render):
    original_state = {}
    for collection_name in collection_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            continue
        original_state[collection_name] = collection.hide_render
        collection.hide_render = hide_render
    return original_state


def restore_collection_render_state(state_by_name):
    for collection_name, hide_render in state_by_name.items():
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            collection.hide_render = hide_render


def iter_visible_bound_points(scene: bpy.types.Scene):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in scene.objects:
        if obj.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT", "POINTCLOUD"}:
            continue
        if obj.hide_render:
            continue
        if not obj.visible_get():
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        try:
            bbox = [eval_obj.matrix_world @ Vector(corner) for corner in eval_obj.bound_box]
        except Exception:
            continue
        if not bbox:
            continue
        for point in bbox:
            yield point


def fits_camera(scene: bpy.types.Scene, camera_obj: bpy.types.Object, coords):
    for coord in coords:
        projected = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, coord)
        if projected.z <= 0.0:
            return False
        if projected.x < FIT_MARGIN or projected.x > 1.0 - FIT_MARGIN:
            return False
        if projected.y < FIT_MARGIN or projected.y > 1.0 - FIT_MARGIN:
            return False
    return True


def fit_camera_dolly(scene: bpy.types.Scene, camera_obj: bpy.types.Object, coords):
    if not coords:
        raise ValueError("No visible coordinates were found to fit the camera against.")

    original_location = camera_obj.location.copy()
    camera_forward = camera_obj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))

    if not fits_camera(scene, camera_obj, coords):
        backward = 1.0
        while not fits_camera(scene, camera_obj, coords):
            camera_obj.location = original_location - (camera_forward * backward)
            bpy.context.view_layer.update()
            backward *= 1.5
            if backward > 5000.0:
                break
        low = -backward
        high = 0.0
    else:
        low = 0.0
        high = 1.0
        while fits_camera(scene, camera_obj, coords):
            low = high
            high *= 1.5
            camera_obj.location = original_location + (camera_forward * high)
            bpy.context.view_layer.update()
            if high > 5000.0:
                break
        camera_obj.location = original_location
        bpy.context.view_layer.update()

    for _ in range(FIT_ITERATIONS):
        mid = (low + high) * 0.5
        camera_obj.location = original_location + (camera_forward * mid)
        bpy.context.view_layer.update()
        if fits_camera(scene, camera_obj, coords):
            low = mid
        else:
            high = mid

    camera_obj.location = original_location + (camera_forward * low)
    bpy.context.view_layer.update()
    return original_location, camera_obj.location.copy()


def render_test(scene: bpy.types.Scene):
    TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    original_filepath = scene.render.filepath
    original_percentage = scene.render.resolution_percentage
    original_format = scene.render.image_settings.file_format
    original_mode = scene.render.image_settings.color_mode
    original_depth = scene.render.image_settings.color_depth
    original_samples = getattr(scene.cycles, "samples", None) if hasattr(scene, "cycles") else None
    original_preview_samples = getattr(scene.cycles, "preview_samples", None) if hasattr(scene, "cycles") else None
    original_layer_use = {view_layer.name: view_layer.use for view_layer in scene.view_layers}

    show_names, hide_names = render_pack.build_scene_collection_toggles(TARGET_VIEW_LAYER)
    shown_state = set_collection_render_state(show_names, hide_render=False)
    hidden_state = set_collection_render_state(hide_names, hide_render=True)

    try:
        scene.render.filepath = str(TEST_OUTPUT)
        scene.render.resolution_x = TEST_RESOLUTION[0]
        scene.render.resolution_y = TEST_RESOLUTION[1]
        scene.render.resolution_percentage = TEST_PERCENTAGE
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        if hasattr(scene, "cycles"):
            scene.cycles.samples = TEST_SAMPLES
            scene.cycles.preview_samples = TEST_SAMPLES
        for view_layer in scene.view_layers:
            view_layer.use = view_layer.name == TARGET_VIEW_LAYER
        bpy.ops.render.render(write_still=True, scene=scene.name, layer=TARGET_VIEW_LAYER, use_viewport=False)
    finally:
        scene.render.filepath = original_filepath
        scene.render.resolution_percentage = original_percentage
        scene.render.image_settings.file_format = original_format
        scene.render.image_settings.color_mode = original_mode
        scene.render.image_settings.color_depth = original_depth
        if hasattr(scene, "cycles") and original_samples is not None:
            scene.cycles.samples = original_samples
            scene.cycles.preview_samples = original_preview_samples
        for view_layer in scene.view_layers:
            view_layer.use = original_layer_use.get(view_layer.name, True)
        restore_collection_render_state(shown_state)
        restore_collection_render_state(hidden_state)

    if not TEST_OUTPUT.exists():
        raise RuntimeError(f"Test render did not produce {TEST_OUTPUT}")
    digest = hashlib.sha256(TEST_OUTPUT.read_bytes()).hexdigest()
    print(f"TEST_RENDER {TEST_OUTPUT} sha256={digest}")


def main():
    scene = bpy.data.scenes.get(BLEND_SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{BLEND_SCENE_NAME}' not found in {bpy.data.filepath}")
    camera_obj = bpy.data.objects.get(CAMERA_NAME)
    if camera_obj is None:
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}")

    render_pack.configure_common_render_settings(scene)
    render_pack.restore_production_materials(scene)
    scene.camera = camera_obj

    show_names, hide_names = render_pack.build_scene_collection_toggles(TARGET_VIEW_LAYER)
    shown_state = set_collection_render_state(show_names, hide_render=False)
    hidden_state = set_collection_render_state(hide_names, hide_render=True)
    try:
        bpy.context.view_layer.update()
        coords = list(iter_visible_bound_points(scene))
        before, after = fit_camera_dolly(scene, camera_obj, coords)
        print(f"CAMERA_FIT before={tuple(round(v, 6) for v in before)} after={tuple(round(v, 6) for v in after)}")
    finally:
        restore_collection_render_state(shown_state)
        restore_collection_render_state(hidden_state)

    bpy.ops.wm.save_mainfile()
    print(f"SAVED_BLEND {bpy.data.filepath}")
    render_test(scene)


if __name__ == "__main__":
    main()
