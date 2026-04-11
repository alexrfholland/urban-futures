from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


# Run this inside the prepared debug blend.
# It creates `good` and `good_zoomed` orthogonal cameras for all three scenes,
# using the city framing as the master for bottom offset.

VIEW_LAYER_NAME = "debug_camera_framing"
OUTPUT_DIR = Path(r"D:\2026 Arboreal Futures\data\renders\timeslices\camera_tests")
RENDER_SAMPLES = 8
RENDER_RES_X = 1920
RENDER_RES_Y = 1080
WORLD_UP = Vector((0.0, 0.0, 1.0))

SOLVE_PRESETS = [
    {"label": "good", "solve_margin": 1.05},
    {"label": "good_zoomed", "solve_margin": 1.0},
]

SITE_SPECS = [
    {
        "scene_name": "parade",
        "horizontal_direction": Vector((1.0, 0.0, 0.0)),
        "point_object_names": [
            "trimmed-parade_base__timeline",
            "trimmed-parade_highResRoad__timeline",
        ],
        "source_camera_candidates": [
            "parade_current_timeslice_cam",
            "parade_time_slice_camera",
            "parade_time_slice_camera_v01",
            "parade_time_slice_camera_v02",
            "parade_time_slice_camera_v03",
        ],
    },
    {
        "scene_name": "city",
        "horizontal_direction": Vector((-1.0, 0.0, 0.0)),
        "point_object_names": [
            "city_buildings.001__timeline",
            "city_highResRoad.001__timeline",
        ],
        "source_camera_candidates": [
            "city_current_timeslice_cam",
            "city_time_slice_camera",
            "city_time_slice_camera_v01",
            "city_time_slice_camera_v02",
            "city_time_slice_camera_v03",
        ],
    },
    {
        "scene_name": "street",
        "horizontal_direction": Vector((0.0, 1.0, 0.0)),
        "point_object_names": [
            "uni_base__timeline",
            "uni_highResRoad__timeline",
        ],
        "source_camera_candidates": [
            "street_current_timeslice_cam",
            "street_time_slice_camera",
            "street_time_slice_camera_v01",
            "street_time_slice_camera_v02",
            "street_time_slice_camera_v03",
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
    if not corners or all(corner == Vector((-1.0, -1.0, -1.0)) for corner in corners):
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


def projected_bounds(
    points: list[Vector],
    location: Vector,
    rotation: Matrix,
    tan_half_x: float,
    tan_half_y: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> tuple[float, float, float, float] | None:
    world_to_camera = rotation.transposed()
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    for point in points:
        local = world_to_camera @ (point - location)
        depth = -local.z
        if depth <= 1e-6:
            return None
        x = 0.5 + (local.x / (2.0 * tan_half_x * depth)) - shift_x
        y = 0.5 + (local.y / (2.0 * tan_half_y * depth)) - shift_y
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    return min_x, max_x, min_y, max_y


def fit_points_in_camera(
    points: list[Vector],
    location: Vector,
    rotation: Matrix,
    tan_half_x: float,
    tan_half_y: float,
) -> bool:
    bounds = projected_bounds(points, location, rotation, tan_half_x, tan_half_y)
    if bounds is None:
        return False
    min_x, max_x, min_y, max_y = bounds
    return (max_x - min_x) <= 1.0 and (max_y - min_y) <= 1.0


def resolve_source_camera(spec: dict) -> bpy.types.Object:
    for camera_name in spec["source_camera_candidates"]:
        camera = bpy.data.objects.get(camera_name)
        if camera is not None and camera.type == "CAMERA":
            return camera
    raise ValueError(f"No source camera found for scene '{spec['scene_name']}' in {bpy.data.filepath}.")


def resolve_shared_pitch_degrees() -> float:
    city_camera = resolve_source_camera(next(spec for spec in SITE_SPECS if spec["scene_name"] == "city"))
    return pitch_degrees_from_camera(city_camera)


def resolve_fixed_z() -> float:
    city_camera = resolve_source_camera(next(spec for spec in SITE_SPECS if spec["scene_name"] == "city"))
    return city_camera.location.z


def solve_distance(
    points: list[Vector],
    target: Vector,
    horizontal_direction: Vector,
    fixed_z: float,
    rotation: Matrix,
    base_camera: bpy.types.Object,
    solve_margin: float,
) -> tuple[Vector, float]:
    tan_half_x = math.tan(base_camera.data.angle_x / 2.0) / solve_margin
    tan_half_y = math.tan(base_camera.data.angle_y / 2.0) / solve_margin
    horizontal = horizontal_direction.normalized()

    def candidate(distance: float) -> Vector:
        return Vector((
            target.x - horizontal.x * distance,
            target.y - horizontal.y * distance,
            fixed_z,
        ))

    low = 1.0
    high = 1.0
    for _ in range(40):
        position = candidate(high)
        if fit_points_in_camera(points, position, rotation, tan_half_x, tan_half_y):
            break
        high *= 1.5
    else:
        raise ValueError(f"Could not fit scene within search range for camera '{base_camera.name}'.")

    for _ in range(60):
        mid = (low + high) * 0.5
        position = candidate(mid)
        if fit_points_in_camera(points, position, rotation, tan_half_x, tan_half_y):
            high = mid
        else:
            low = mid

    return candidate(high), high


def apply_camera_state(target_camera: bpy.types.Object, base_camera: bpy.types.Object, location: Vector, rotation: Matrix, shift_x: float, shift_y: float) -> None:
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
    target_camera.data.shift_x = shift_x
    target_camera.data.shift_y = shift_y
    target_camera.hide_render = False
    target_camera.hide_viewport = False


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


def render_preview(scene: bpy.types.Scene, camera: bpy.types.Object, output_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        scene.render.resolution_x = RENDER_RES_X
        scene.render.resolution_y = RENDER_RES_Y
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(OUTPUT_DIR / output_name)
        if hasattr(scene, "cycles"):
            scene.cycles.samples = RENDER_SAMPLES
            scene.cycles.preview_samples = RENDER_SAMPLES
        bpy.ops.render.render(write_still=True, scene=scene.name, layer=VIEW_LAYER_NAME, use_viewport=False)
    finally:
        scene.camera = original_camera
        for view_layer in scene.view_layers:
            view_layer.use = original_use.get(view_layer.name, True)


def city_master_metrics(
    city_spec: dict,
    shared_pitch_degrees: float,
    fixed_z: float,
    solve_margin: float,
) -> dict:
    base_camera = resolve_source_camera(city_spec)
    points = collect_points_by_names(city_spec["point_object_names"])
    target = collect_ground_target(points)
    rotation = build_exact_axis_rotation(city_spec["horizontal_direction"], shared_pitch_degrees)
    position, distance = solve_distance(
        points,
        target,
        city_spec["horizontal_direction"],
        fixed_z,
        rotation,
        base_camera,
        solve_margin,
    )

    tan_half_x = math.tan(base_camera.data.angle_x / 2.0)
    tan_half_y = math.tan(base_camera.data.angle_y / 2.0)
    bounds = projected_bounds(points, position, rotation, tan_half_x, tan_half_y)
    if bounds is None:
        raise ValueError("Could not compute city projected bounds.")
    min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y

    return {
        "position": position,
        "distance": distance,
        "width": width,
        "height": height,
        "target_bottom": (1.0 - height) * 0.5,
        "target_top": (1.0 - height) * 0.5,
        "target_center_x": 0.5,
        "rotation": rotation,
    }


def solve_site_camera(
    spec: dict,
    shared_pitch_degrees: float,
    fixed_z: float,
    preset: dict,
    target_bottom: float,
) -> tuple[bpy.types.Object, dict]:
    scene = bpy.data.scenes[spec["scene_name"]]
    base_camera = resolve_source_camera(spec)
    points = collect_points_by_names(spec["point_object_names"])
    target = collect_ground_target(points)
    rotation = build_exact_axis_rotation(spec["horizontal_direction"], shared_pitch_degrees)
    position, distance = solve_distance(
        points,
        target,
        spec["horizontal_direction"],
        fixed_z,
        rotation,
        base_camera,
        preset["solve_margin"],
    )

    actual_tan_half_x = math.tan(base_camera.data.angle_x / 2.0)
    actual_tan_half_y = math.tan(base_camera.data.angle_y / 2.0)
    bounds = projected_bounds(points, position, rotation, actual_tan_half_x, actual_tan_half_y)
    if bounds is None:
        raise ValueError(f"Could not compute projected bounds for scene '{scene.name}'.")
    min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y

    shift_x = (min_x + max_x) * 0.5 - 0.5
    shift_y = min_y - target_bottom

    shifted_bounds = projected_bounds(points, position, rotation, actual_tan_half_x, actual_tan_half_y, shift_x, shift_y)
    if shifted_bounds is None:
        raise ValueError(f"Shifted bounds failed for scene '{scene.name}'.")
    shifted_min_x, shifted_max_x, shifted_min_y, shifted_max_y = shifted_bounds
    if shifted_min_x < -1e-4 or shifted_max_x > 1.0001 or shifted_min_y < -1e-4 or shifted_max_y > 1.0001:
        raise ValueError(
            f"Scene '{scene.name}' with preset '{preset['label']}' would clip after bottom alignment: "
            f"{tuple(round(v, 6) for v in shifted_bounds)}"
        )

    camera_name = f"{scene.name}_{preset['label']}_timeslice_cam"
    target_camera = ensure_target_camera(base_camera, camera_name)
    apply_camera_state(target_camera, base_camera, position, rotation, shift_x, shift_y)

    metrics = {
        "distance": distance,
        "width": width,
        "height": height,
        "bottom": shifted_min_y,
        "top": 1.0 - shifted_max_y,
        "left": shifted_min_x,
        "right": 1.0 - shifted_max_x,
    }
    return target_camera, metrics


def main() -> None:
    city_spec = next(spec for spec in SITE_SPECS if spec["scene_name"] == "city")
    shared_pitch_degrees = resolve_shared_pitch_degrees()
    fixed_z = resolve_fixed_z()

    for preset in SOLVE_PRESETS:
        master = city_master_metrics(city_spec, shared_pitch_degrees, fixed_z, preset["solve_margin"])
        print(
            f"CITY_MASTER {preset['label']} "
            f"bottom={round(master['target_bottom'], 6)} "
            f"top={round(master['target_top'], 6)} "
            f"width={round(master['width'], 6)} "
            f"height={round(master['height'], 6)} "
            f"distance={round(master['distance'], 6)}"
        )

        for spec in SITE_SPECS:
            scene = bpy.data.scenes[spec["scene_name"]]
            camera, metrics = solve_site_camera(
                spec,
                shared_pitch_degrees,
                fixed_z,
                preset,
                master["target_bottom"],
            )
            render_preview(scene, camera, f"{scene.name}-{preset['label']}.png")
            print(
                f"SAVED_CAMERA {camera.name} "
                f"bottom={round(metrics['bottom'], 6)} "
                f"top={round(metrics['top'], 6)} "
                f"left={round(metrics['left'], 6)} "
                f"right={round(metrics['right'], 6)} "
                f"distance={round(metrics['distance'], 6)}"
            )

    if bpy.data.is_saved:
        bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
