from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view


SCENE_NAME = os.environ["B2026_SCENE_NAME"]
CAMERA_NAME = os.environ["B2026_CAMERA_NAME"]
VIEW_LAYER_NAME = os.environ["B2026_VIEW_LAYER_NAME"]
FOCAL_X = float(os.environ["B2026_FOCAL_X"])
FOCAL_Y = float(os.environ["B2026_FOCAL_Y"])
FOCAL_Z = float(os.environ["B2026_FOCAL_Z"])
AIM_MODE = os.environ.get("B2026_AIM_MODE", "FOCAL").strip().upper()
FIT_MODE = os.environ.get("B2026_FIT_MODE", "FULL").strip().upper()
TARGET_CAMERA_NAME = os.environ.get("B2026_TARGET_CAMERA_NAME", f"{CAMERA_NAME}__shared_fit")
SOLVE_MODE = os.environ.get("B2026_SOLVE_MODE", "RAY").strip().upper()
MARGIN = float(os.environ.get("B2026_MARGIN", "1.05"))
FIXED_Z = os.environ.get("B2026_FIXED_Z", "")
MIN_DISTANCE = float(os.environ.get("B2026_MIN_DISTANCE", "1.0"))
MAX_DISTANCE = float(os.environ.get("B2026_MAX_DISTANCE", "20000.0"))
FRAME_TARGET_Y = float(os.environ.get("B2026_FRAME_TARGET_Y", "0.5"))
FRAME_TARGET_X = float(os.environ.get("B2026_FRAME_TARGET_X", "0.5"))
RENDER_PREVIEW = os.environ.get("B2026_RENDER_PREVIEW", "").strip().lower() in {"1", "true", "yes"}
PREVIEW_PATH = os.environ.get("B2026_PREVIEW_PATH", "")
PREVIEW_SAMPLES = int(os.environ.get("B2026_PREVIEW_SAMPLES", "24"))
PREVIEW_RES_X = int(os.environ.get("B2026_PREVIEW_RES_X", "1920"))
PREVIEW_RES_Y = int(os.environ.get("B2026_PREVIEW_RES_Y", "1080"))
VIEW_DIRECTION_X = os.environ.get("B2026_VIEW_DIRECTION_X", "").strip()
VIEW_DIRECTION_Y = os.environ.get("B2026_VIEW_DIRECTION_Y", "").strip()
VIEW_DIRECTION_Z = os.environ.get("B2026_VIEW_DIRECTION_Z", "").strip()


def matrix_is_identity(matrix: Matrix, tolerance: float = 1e-6) -> bool:
    size = len(matrix)
    identity = Matrix.Identity(size)
    for row_index in range(size):
        for col_index in range(size):
            if abs(matrix[row_index][col_index] - identity[row_index][col_index]) > tolerance:
                return False
    return True


def object_transform_matrix(obj: bpy.types.Object) -> Matrix:
    if obj.parent is None:
        basis = obj.matrix_basis.copy()
        if not matrix_is_identity(basis):
            return basis
    return obj.matrix_world.copy()


def camera_forward(camera: bpy.types.Object) -> Vector:
    return -(object_transform_matrix(camera).to_quaternion() @ Vector((0.0, 0.0, 1.0))).normalized()


def camera_up(camera: bpy.types.Object) -> Vector:
    return (object_transform_matrix(camera).to_quaternion() @ Vector((0.0, 1.0, 0.0))).normalized()


def parse_view_direction() -> Vector | None:
    if not (VIEW_DIRECTION_X and VIEW_DIRECTION_Y):
        return None

    direction = Vector((
        float(VIEW_DIRECTION_X),
        float(VIEW_DIRECTION_Y),
        float(VIEW_DIRECTION_Z or "0.0"),
    ))
    if direction.length < 1e-6:
        raise ValueError("B2026_VIEW_DIRECTION_* cannot describe a zero-length vector.")
    return direction.normalized()


def build_camera_rotation(location: Vector, target: Vector, up_hint: Vector) -> Matrix:
    forward = (target - location).normalized()
    z_axis = (-forward).normalized()

    x_axis = up_hint.cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((0.0, 0.0, 1.0)).cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0))
    x_axis.normalize()

    y_axis = z_axis.cross(x_axis).normalized()
    return Matrix((x_axis, y_axis, z_axis)).transposed()


def walk_included_layer_collections(layer_collection: bpy.types.LayerCollection) -> Iterable[bpy.types.LayerCollection]:
    if layer_collection.exclude:
        return
    yield layer_collection
    for child in layer_collection.children:
        yield from walk_included_layer_collections(child)


def renderable_objects_for_view_layer(scene: bpy.types.Scene, view_layer_name: str) -> list[bpy.types.Object]:
    view_layer = scene.view_layers.get(view_layer_name)
    if view_layer is None:
        raise ValueError(f"View layer '{view_layer_name}' not found in scene '{scene.name}'.")

    objects_by_name: dict[str, bpy.types.Object] = {}
    for layer_coll in walk_included_layer_collections(view_layer.layer_collection):
        for obj in layer_coll.collection.objects:
            if obj.name in objects_by_name:
                continue
            if obj.hide_render:
                continue
            if obj.type in {"CAMERA", "LIGHT", "LIGHT_PROBE", "EMPTY", "ARMATURE"}:
                continue
            objects_by_name[obj.name] = obj
    return list(objects_by_name.values())


def object_bbox_points(obj: bpy.types.Object) -> list[Vector]:
    if not hasattr(obj, "bound_box") or obj.bound_box is None:
        return []
    corners = [Vector(corner) for corner in obj.bound_box]
    if not corners:
        return []
    # Blender uses (-1, -1, -1) sentinel for invalid bbox on some objects.
    if all(corner == Vector((-1.0, -1.0, -1.0)) for corner in corners):
        return []
    matrix = obj.matrix_world.copy()
    return [matrix @ corner for corner in corners]


def collect_fit_points(scene: bpy.types.Scene, view_layer_name: str) -> list[Vector]:
    points: list[Vector] = []
    for obj in renderable_objects_for_view_layer(scene, view_layer_name):
        points.extend(object_bbox_points(obj))
    if not points:
        raise ValueError(
            f"No renderable fit points found for scene '{scene.name}' view layer '{view_layer_name}'."
        )
    return points


def is_ground_timeline_object(obj: bpy.types.Object) -> bool:
    name = obj.name.lower()
    if "__timeline" not in name:
        return False
    if any(token in name for token in ("treepositions", "logpositions", "polepositions", "timelinestripbox", "envelope", "bioenvelope", "_cubes")):
        return False
    return any(token in name for token in ("road", "base", "building", "buildings"))


def collect_ground_target(scene: bpy.types.Scene, view_layer_name: str, fallback: Vector) -> Vector:
    points: list[Vector] = []
    for obj in renderable_objects_for_view_layer(scene, view_layer_name):
        if not is_ground_timeline_object(obj):
            continue
        points.extend(object_bbox_points(obj))
    if not points:
        return fallback

    xs = [point.x for point in points]
    ys = [point.y for point in points]
    zs = [point.z for point in points]
    return Vector((
        (min(xs) + max(xs)) * 0.5,
        (min(ys) + max(ys)) * 0.5,
        min(zs),
    ))


def collect_ground_xy_fit_points(scene: bpy.types.Scene, view_layer_name: str, fallback_target: Vector) -> list[Vector]:
    points: list[Vector] = []
    for obj in renderable_objects_for_view_layer(scene, view_layer_name):
        if not is_ground_timeline_object(obj):
            continue
        points.extend(object_bbox_points(obj))
    if not points:
        return [fallback_target]

    ground_z = min(point.z for point in points)
    return [Vector((point.x, point.y, ground_z)) for point in points]


def fit_points_in_camera(
    points: list[Vector],
    location: Vector,
    rotation: Matrix,
    tan_half_x: float,
    tan_half_y: float,
) -> bool:
    world_to_camera = rotation.transposed()
    for point in points:
        local = world_to_camera @ (point - location)
        depth = -local.z
        if depth <= 1e-6:
            return False
        if abs(local.x) > tan_half_x * depth:
            return False
        if abs(local.y) > tan_half_y * depth:
            return False
    return True


def solve_position(
    base_camera: bpy.types.Object,
    target: Vector,
    points: list[Vector],
) -> tuple[Vector, Matrix, float]:
    up_hint = camera_up(base_camera)
    tan_half_x = math.tan(base_camera.data.angle_x / 2.0) / MARGIN
    tan_half_y = math.tan(base_camera.data.angle_y / 2.0) / MARGIN
    explicit_view_direction = parse_view_direction()

    if SOLVE_MODE == "FIXED_Z":
        if not FIXED_Z:
            raise ValueError("B2026_FIXED_Z is required when B2026_SOLVE_MODE=FIXED_Z.")
        fixed_z = float(FIXED_Z)
        if explicit_view_direction is not None:
            offset_xy = Vector((-explicit_view_direction.x, -explicit_view_direction.y, 0.0))
        else:
            base_offset = base_camera.location - target
            offset_xy = Vector((base_offset.x, base_offset.y, 0.0))
        if offset_xy.length < 1e-6:
            raise ValueError("Cannot solve FIXED_Z view because the camera sits directly above the focal point.")
        direction_xy = offset_xy.normalized()

        def candidate(distance: float) -> tuple[Vector, Matrix]:
            position = Vector((
                target.x + direction_xy.x * distance,
                target.y + direction_xy.y * distance,
                fixed_z,
            ))
            rotation = build_camera_rotation(position, target, up_hint)
            return position, rotation

    else:
        direction = explicit_view_direction if explicit_view_direction is not None else camera_forward(base_camera)

        def candidate(distance: float) -> tuple[Vector, Matrix]:
            position = target - direction * distance
            rotation = build_camera_rotation(position, target, up_hint)
            return position, rotation

    low = MIN_DISTANCE
    high = max(low, MIN_DISTANCE)
    for _ in range(32):
        position, rotation = candidate(high)
        if fit_points_in_camera(points, position, rotation, tan_half_x, tan_half_y):
            break
        high *= 1.5
        if high > MAX_DISTANCE:
            raise ValueError(
                f"Unable to fit scene within max distance {MAX_DISTANCE} for camera '{base_camera.name}'."
            )

    for _ in range(48):
        mid = (low + high) * 0.5
        position, rotation = candidate(mid)
        if fit_points_in_camera(points, position, rotation, tan_half_x, tan_half_y):
            high = mid
        else:
            low = mid

    position, rotation = candidate(high)
    return position, rotation, high


def ensure_target_camera(base_camera: bpy.types.Object, target_name: str) -> bpy.types.Object:
    camera = bpy.data.objects.get(target_name)
    if camera is None:
        camera = base_camera.copy()
        camera.data = base_camera.data.copy()
        camera.name = target_name
        camera.data.name = f"{base_camera.data.name}__{target_name}"
        for collection in base_camera.users_collection:
            collection.objects.link(camera)
    return camera


def apply_camera_state(target_camera: bpy.types.Object, base_camera: bpy.types.Object, location: Vector, rotation: Matrix) -> None:
    target_camera.location = location
    target_camera.rotation_euler = rotation.to_euler()
    if target_camera.parent is None:
        target_camera.matrix_world = Matrix.Translation(location) @ rotation.to_4x4()
    target_camera.data.lens = base_camera.data.lens
    target_camera.data.sensor_fit = base_camera.data.sensor_fit
    target_camera.data.angle = base_camera.data.angle
    target_camera.data.clip_start = base_camera.data.clip_start
    target_camera.data.clip_end = base_camera.data.clip_end
    target_camera.data.shift_x = base_camera.data.shift_x
    target_camera.data.shift_y = base_camera.data.shift_y


def reframe_camera_to_target(scene: bpy.types.Scene, camera: bpy.types.Object, target: Vector) -> tuple[float, float]:
    original_shift_x = camera.data.shift_x
    original_shift_y = camera.data.shift_y

    def set_shift_y(value: float) -> float:
        camera.data.shift_y = value
        return world_to_camera_view(scene, camera, target).y

    def set_shift_x(value: float) -> float:
        camera.data.shift_x = value
        return world_to_camera_view(scene, camera, target).x

    base_y = set_shift_y(original_shift_y)
    base_x = set_shift_x(original_shift_x)

    # Empirically in Blender camera space, increasing shift_y moves the target downward in frame.
    delta_y = base_y - FRAME_TARGET_Y
    camera.data.shift_y = original_shift_y + delta_y

    # Increasing shift_x moves the target left in frame.
    delta_x = base_x - FRAME_TARGET_X
    camera.data.shift_x = original_shift_x + delta_x

    final_coords = world_to_camera_view(scene, camera, target)
    return final_coords.x, final_coords.y


def maybe_render_preview(scene: bpy.types.Scene, camera: bpy.types.Object) -> None:
    if not RENDER_PREVIEW or not PREVIEW_PATH:
        return

    preview_path = Path(PREVIEW_PATH)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    original_use = {view_layer.name: view_layer.use for view_layer in scene.view_layers}
    original_camera = scene.camera

    try:
        for view_layer in scene.view_layers:
            view_layer.use = view_layer.name == VIEW_LAYER_NAME
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
        scene.render.resolution_x = PREVIEW_RES_X
        scene.render.resolution_y = PREVIEW_RES_Y
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(preview_path)
        if hasattr(scene, "cycles"):
            scene.cycles.samples = PREVIEW_SAMPLES
            scene.cycles.preview_samples = PREVIEW_SAMPLES
        bpy.ops.render.render(write_still=True, scene=scene.name, layer=VIEW_LAYER_NAME, use_viewport=False)
    finally:
        scene.camera = original_camera
        for view_layer in scene.view_layers:
            view_layer.use = original_use.get(view_layer.name, True)


def main() -> None:
    scene = bpy.data.scenes.get(SCENE_NAME)
    if scene is None:
        raise ValueError(f"Scene '{SCENE_NAME}' not found in {bpy.data.filepath}.")

    base_camera = bpy.data.objects.get(CAMERA_NAME)
    if base_camera is None or base_camera.type != "CAMERA":
        raise ValueError(f"Camera '{CAMERA_NAME}' not found in {bpy.data.filepath}.")

    fallback_target = Vector((FOCAL_X, FOCAL_Y, FOCAL_Z))
    if AIM_MODE == "GROUND":
        target = collect_ground_target(scene, VIEW_LAYER_NAME, fallback_target)
    else:
        target = fallback_target
    if FIT_MODE == "GROUND_XY":
        points = collect_ground_xy_fit_points(scene, VIEW_LAYER_NAME, target)
    else:
        points = collect_fit_points(scene, VIEW_LAYER_NAME)
    position, rotation, solved_distance = solve_position(base_camera, target, points)

    target_camera = ensure_target_camera(base_camera, TARGET_CAMERA_NAME)
    apply_camera_state(target_camera, base_camera, position, rotation)
    final_frame_x, final_frame_y = reframe_camera_to_target(scene, target_camera, target)
    maybe_render_preview(scene, target_camera)

    print(f"FIT_CAMERA {target_camera.name}")
    print(f"  scene={scene.name}")
    print(f"  view_layer={VIEW_LAYER_NAME}")
    print(f"  solve_mode={SOLVE_MODE}")
    print(f"  aim_mode={AIM_MODE}")
    print(f"  fit_mode={FIT_MODE}")
    print(f"  target={tuple(round(v, 6) for v in target)}")
    print(f"  location={tuple(round(v, 6) for v in target_camera.location)}")
    print(f"  rotation_euler={tuple(round(v, 6) for v in target_camera.rotation_euler)}")
    print(f"  shift_x={round(target_camera.data.shift_x, 6)}")
    print(f"  shift_y={round(target_camera.data.shift_y, 6)}")
    print(f"  framed_target=({round(final_frame_x, 6)}, {round(final_frame_y, 6)})")
    print(f"  solved_distance={round(solved_distance, 6)}")
    print(f"  fit_point_count={len(points)}")

    if bpy.data.is_saved:
        bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
