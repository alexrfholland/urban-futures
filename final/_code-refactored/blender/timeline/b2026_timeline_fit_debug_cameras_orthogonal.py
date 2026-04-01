from __future__ import annotations

import math
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


# CAMERA ITERATION SCRIPT.
# Run this inside the prepared debug blend when you only want to update camera
# placement/composition and keep the existing scenes/objects intact.


VIEW_LAYER_NAME = "debug_camera_framing"
TARGET_FRAME_X = 0.5
# Raise this to place the ground target higher in frame.
# Lower it to push the composition down.
TARGET_FRAME_Y = 0.18
MARGIN = 1.05
MIN_DISTANCE = 1.0
MAX_DISTANCE = 20000.0
RENDER_PREVIEWS = False
PREVIEW_SAMPLES = 12
PREVIEW_RES_X = 1920
PREVIEW_RES_Y = 1080
PREVIEW_DIR = Path(r"D:\2026 Arboreal Futures\data\renders\timeslices\camera_tests")
WORLD_UP = Vector((0.0, 0.0, 1.0))
ARCHIVE_LABEL = "orthogonal_apr01"

SITE_SPECS = [
    {
        "scene_name": "parade",
        "current_camera_name": "parade_current_timeslice_cam",
        "archive_collection_name": "parade_timeslice_cam_archive",
        "archive_camera_name": f"parade_timeslice_cam_{ARCHIVE_LABEL}",
        "pitch_reference_camera_name": "paraview_camera_parade__debug_ground_xy_axes",
        "source_camera_candidates": [
            "parade_current_timeslice_cam",
            "parade_time_slice_camera",
            "parade_time_slice_camera_v01",
            "parade_time_slice_camera_v02",
            "parade_time_slice_camera_v03",
            "paraview_camera_parade__debug_ground_xy_orthogonal",
            "paraview_camera_parade__debug_ground_xy_axes",
            "paraview_camera_parade__shared_fit_ground_bottom_height_parade",
        ],
        "legacy_camera_names": [
            "parade_time_slice_camera",
            "parade_time_slice_camera_v01",
            "parade_time_slice_camera_v02",
            "parade_time_slice_camera_v03",
            "paraview_camera_parade__debug_ground_xy_orthogonal",
            "paraview_camera_parade__debug_ground_xy_axes",
            "paraview_camera_parade__shared_fit_ground_bottom_height_parade",
        ],
        "horizontal_direction": Vector((1.0, 0.0, 0.0)),
        "point_object_names": [
            "trimmed-parade_base__timeline",
            "trimmed-parade_highResRoad__timeline",
        ],
    },
    {
        "scene_name": "city",
        "current_camera_name": "city_current_timeslice_cam",
        "archive_collection_name": "city_timeslice_cam_archive",
        "archive_camera_name": f"city_timeslice_cam_{ARCHIVE_LABEL}",
        "source_camera_candidates": [
            "city_current_timeslice_cam",
            "city_time_slice_camera",
            "city_time_slice_camera_v01",
            "city_time_slice_camera_v02",
            "city_time_slice_camera_v03",
            "paraview_camera_city__debug_ground_xy_orthogonal",
            "paraview_camera_city__debug_ground_xy_axes",
            "paraview_camera_city__shared_fit_ground_bottom_height_parade",
        ],
        "legacy_camera_names": [
            "city_time_slice_camera",
            "city_time_slice_camera_v01",
            "city_time_slice_camera_v02",
            "city_time_slice_camera_v03",
            "paraview_camera_city__debug_ground_xy_orthogonal",
            "paraview_camera_city__debug_ground_xy_axes",
            "paraview_camera_city__shared_fit_ground_bottom_height_parade",
        ],
        "horizontal_direction": Vector((-1.0, 0.0, 0.0)),
        "point_object_names": [
            "city_buildings.001__timeline",
            "city_highResRoad.001__timeline",
        ],
    },
    {
        "scene_name": "street",
        "current_camera_name": "street_current_timeslice_cam",
        "archive_collection_name": "street_timeslice_cam_archive",
        "archive_camera_name": f"street_timeslice_cam_{ARCHIVE_LABEL}",
        "source_camera_candidates": [
            "street_current_timeslice_cam",
            "street_time_slice_camera",
            "street_time_slice_camera_v01",
            "street_time_slice_camera_v02",
            "street_time_slice_camera_v03",
            "paraview_camera_street__debug_ground_xy_orthogonal",
            "paraview_camera_street__debug_ground_xy_axes",
            "paraview_camera_street__shared_fit_ground_bottom_height_parade",
        ],
        "legacy_camera_names": [
            "street_time_slice_camera",
            "street_time_slice_camera_v01",
            "street_time_slice_camera_v02",
            "street_time_slice_camera_v03",
            "paraview_camera_street__debug_ground_xy_orthogonal",
            "paraview_camera_street__debug_ground_xy_axes",
            "paraview_camera_street__shared_fit_ground_bottom_height_parade",
        ],
        "horizontal_direction": Vector((0.0, 1.0, 0.0)),
        "point_object_names": [
            "uni_base__timeline",
            "uni_highResRoad__timeline",
        ],
    },
]


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


def pitch_degrees_from_camera(camera: bpy.types.Object) -> float:
    forward = camera_forward(camera)
    horizontal = Vector((forward.x, forward.y, 0.0))
    horizontal_length = max(horizontal.length, 1e-6)
    return math.degrees(math.atan2(-forward.z, horizontal_length))


def object_bbox_points(obj: bpy.types.Object) -> list[Vector]:
    if not hasattr(obj, "bound_box") or obj.bound_box is None:
        return []
    corners = [Vector(corner) for corner in obj.bound_box]
    if not corners:
        return []
    if all(corner == Vector((-1.0, -1.0, -1.0)) for corner in corners):
        return []
    matrix = obj.matrix_world.copy()
    return [matrix @ corner for corner in corners]


def collect_points_by_names(object_names: list[str]) -> list[Vector]:
    points: list[Vector] = []
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise ValueError(f"Missing required object '{name}' in {bpy.data.filepath}.")
        points.extend(object_bbox_points(obj))
    if not points:
        raise ValueError(f"No fit points collected from objects: {object_names}")
    return points


def collect_ground_target(points: list[Vector]) -> Vector:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    zs = [point.z for point in points]
    return Vector((
        (min(xs) + max(xs)) * 0.5,
        (min(ys) + max(ys)) * 0.5,
        min(zs),
    ))


def collect_ground_xy_fit_points(points: list[Vector]) -> list[Vector]:
    ground_z = min(point.z for point in points)
    return [Vector((point.x, point.y, ground_z)) for point in points]


def build_rotation_from_forward(forward: Vector) -> Matrix:
    z_axis = (-forward).normalized()
    x_axis = WORLD_UP.cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0))
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()
    return Matrix((x_axis, y_axis, z_axis)).transposed()


def build_exact_axis_rotation(horizontal_direction: Vector, pitch_degrees: float) -> Matrix:
    horizontal = horizontal_direction.normalized()
    pitch_radians = math.radians(pitch_degrees)
    horizontal_scale = math.cos(pitch_radians)
    vertical_scale = math.sin(pitch_radians)
    forward = Vector((
        horizontal.x * horizontal_scale,
        horizontal.y * horizontal_scale,
        -vertical_scale,
    )).normalized()
    return build_rotation_from_forward(forward)


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


def solve_distance(
    base_camera: bpy.types.Object,
    target: Vector,
    horizontal_direction: Vector,
    fixed_z: float,
    rotation: Matrix,
    points: list[Vector],
) -> tuple[Vector, float]:
    tan_half_x = math.tan(base_camera.data.angle_x / 2.0) / MARGIN
    tan_half_y = math.tan(base_camera.data.angle_y / 2.0) / MARGIN
    horizontal = horizontal_direction.normalized()

    def candidate(distance: float) -> Vector:
        return Vector((
            target.x - horizontal.x * distance,
            target.y - horizontal.y * distance,
            fixed_z,
        ))

    low = MIN_DISTANCE
    high = max(low, MIN_DISTANCE)
    for _ in range(32):
        position = candidate(high)
        if fit_points_in_camera(points, position, rotation, tan_half_x, tan_half_y):
            break
        high *= 1.5
        if high > MAX_DISTANCE:
            raise ValueError(
                f"Unable to fit scene within max distance {MAX_DISTANCE} for camera '{base_camera.name}'."
            )

    for _ in range(48):
        mid = (low + high) * 0.5
        position = candidate(mid)
        if fit_points_in_camera(points, position, rotation, tan_half_x, tan_half_y):
            high = mid
        else:
            low = mid

    return candidate(high), high


def ensure_child_collection(parent: bpy.types.Collection, child_name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(child_name)
    if collection is None:
        collection = bpy.data.collections.new(child_name)
    if child_name not in parent.children:
        parent.children.link(collection)
    return collection


def relink_object_to_collection(obj: bpy.types.Object, target_collection: bpy.types.Collection) -> None:
    for collection in list(obj.users_collection):
        if collection == target_collection:
            continue
        collection.objects.unlink(obj)
    if obj.name not in target_collection.objects:
        target_collection.objects.link(obj)


def move_legacy_cameras_to_archive(
    scene: bpy.types.Scene,
    spec: dict,
    archive_collection: bpy.types.Collection,
) -> list[str]:
    moved_names: list[str] = []
    for camera_name in spec["legacy_camera_names"]:
        camera = bpy.data.objects.get(camera_name)
        if camera is None or camera.type != "CAMERA":
            continue
        camera.hide_render = True
        camera.hide_viewport = True
        relink_object_to_collection(camera, archive_collection)
        moved_names.append(camera.name)
    return moved_names


def ensure_target_camera(base_camera: bpy.types.Object, target_name: str, target_collection: bpy.types.Collection) -> bpy.types.Object:
    camera = bpy.data.objects.get(target_name)
    if camera is None:
        camera = base_camera.copy()
        camera.data = base_camera.data.copy()
        camera.name = target_name
        camera.data.name = f"{base_camera.data.name}__{target_name}"
    relink_object_to_collection(camera, target_collection)
    camera.hide_render = False
    camera.hide_viewport = False
    return camera


def apply_camera_state(target_camera: bpy.types.Object, base_camera: bpy.types.Object, location: Vector, rotation: Matrix) -> None:
    target_camera.location = location
    target_camera.rotation_mode = "XYZ"
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

    delta_y = base_y - TARGET_FRAME_Y
    delta_x = base_x - TARGET_FRAME_X
    camera.data.shift_y = original_shift_y + delta_y
    camera.data.shift_x = original_shift_x + delta_x

    final_coords = world_to_camera_view(scene, camera, target)
    return final_coords.x, final_coords.y


def maybe_render_preview(scene: bpy.types.Scene, camera: bpy.types.Object, preview_name: str) -> None:
    if not RENDER_PREVIEWS:
        return

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
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
        scene.render.filepath = str(PREVIEW_DIR / preview_name)
        if hasattr(scene, "cycles"):
            scene.cycles.samples = PREVIEW_SAMPLES
            scene.cycles.preview_samples = PREVIEW_SAMPLES
        bpy.ops.render.render(write_still=True, scene=scene.name, layer=VIEW_LAYER_NAME, use_viewport=False)
    finally:
        scene.camera = original_camera
        for view_layer in scene.view_layers:
            view_layer.use = original_use.get(view_layer.name, True)


def resolve_shared_pitch_degrees() -> float:
    reference_name = SITE_SPECS[0]["pitch_reference_camera_name"]
    reference_camera = bpy.data.objects.get(reference_name)
    if reference_camera is None:
        for candidate_name in SITE_SPECS[0]["source_camera_candidates"]:
            reference_camera = bpy.data.objects.get(candidate_name)
            if reference_camera is not None:
                break
    if reference_camera is None:
        return 37.0
    return pitch_degrees_from_camera(reference_camera)


def resolve_fixed_z() -> float:
    reference_camera = None
    for candidate_name in SITE_SPECS[0]["source_camera_candidates"]:
        reference_camera = bpy.data.objects.get(candidate_name)
        if reference_camera is not None:
            break
    if reference_camera is None:
        raise ValueError(f"No reference camera found for scene '{SITE_SPECS[0]['scene_name']}' in {bpy.data.filepath}.")
    return reference_camera.location.z


def solve_site(spec: dict, fixed_z: float, shared_pitch_degrees: float) -> None:
    scene = bpy.data.scenes.get(spec["scene_name"])
    if scene is None:
        raise ValueError(f"Scene '{spec['scene_name']}' not found in {bpy.data.filepath}.")

    base_camera = None
    for candidate_name in spec["source_camera_candidates"]:
        base_camera = bpy.data.objects.get(candidate_name)
        if base_camera is not None and base_camera.type == "CAMERA":
            break
    if base_camera is None or base_camera.type != "CAMERA":
        raise ValueError(f"No source camera found for scene '{spec['scene_name']}' in {bpy.data.filepath}.")

    raw_points = collect_points_by_names(spec["point_object_names"])
    target = collect_ground_target(raw_points)
    fit_points = collect_ground_xy_fit_points(raw_points)
    rotation = build_exact_axis_rotation(spec["horizontal_direction"], shared_pitch_degrees)
    position, solved_distance = solve_distance(
        base_camera,
        target,
        spec["horizontal_direction"],
        fixed_z,
        rotation,
        fit_points,
    )

    active_collection = ensure_child_collection(scene.collection, f"{scene.name}_debug_camera_framing")
    archive_collection = ensure_child_collection(scene.collection, spec["archive_collection_name"])
    moved_legacy_names = move_legacy_cameras_to_archive(scene, spec, archive_collection)

    current_camera = ensure_target_camera(base_camera, spec["current_camera_name"], active_collection)
    apply_camera_state(current_camera, base_camera, position, rotation)
    final_frame_x, final_frame_y = reframe_camera_to_target(scene, current_camera, target)
    scene.camera = current_camera

    archive_camera = ensure_target_camera(base_camera, spec["archive_camera_name"], archive_collection)
    apply_camera_state(archive_camera, current_camera, position, rotation)
    archive_camera.data.shift_x = current_camera.data.shift_x
    archive_camera.data.shift_y = current_camera.data.shift_y
    archive_camera.hide_render = True
    archive_camera.hide_viewport = True

    maybe_render_preview(scene, current_camera, f"{spec['scene_name']}-debug-camera-ground-xy-orthogonal.png")

    print(f"FIT_CAMERA {current_camera.name}")
    print(f"  scene={scene.name}")
    print(f"  target={tuple(round(v, 6) for v in target)}")
    print(f"  location={tuple(round(v, 6) for v in current_camera.location)}")
    print(f"  rotation_euler={tuple(round(v, 6) for v in current_camera.rotation_euler)}")
    print(f"  shift_x={round(current_camera.data.shift_x, 6)}")
    print(f"  shift_y={round(current_camera.data.shift_y, 6)}")
    print(f"  framed_target=({round(final_frame_x, 6)}, {round(final_frame_y, 6)})")
    print(f"  solved_distance={round(solved_distance, 6)}")
    print(f"  shared_pitch_degrees={round(shared_pitch_degrees, 6)}")
    print(f"  fixed_z={round(fixed_z, 6)}")
    print(f"  archive_collection={archive_collection.name}")
    print(f"  archive_camera={archive_camera.name}")
    print(f"  moved_legacy={moved_legacy_names}")


def main() -> None:
    shared_pitch_degrees = resolve_shared_pitch_degrees()
    fixed_z = resolve_fixed_z()

    for spec in SITE_SPECS:
        solve_site(spec, fixed_z, shared_pitch_degrees)

    if bpy.data.is_saved:
        bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
