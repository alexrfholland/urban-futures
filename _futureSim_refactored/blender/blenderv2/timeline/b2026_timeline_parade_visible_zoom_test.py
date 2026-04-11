from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Matrix, Vector


# PARADE-ONLY CAMERA ITERATION SCRIPT.
# Run this inside the prepared debug blend when you want a tighter parade zoom
# based on the rendered alpha silhouette.


OUTPUT_DIR = Path(r"D:\2026 Arboreal Futures\data\renders\timeslices\camera_tests")
OUTPUT_IMAGE_NAME = "parade-visible-zoom-test-v6.png"
SCENE_NAME = "parade"
VIEW_LAYER_NAME = "debug_camera_framing"
SOURCE_CAMERA_NAME = "parade_good_zoomed_timeslice_cam"
TARGET_CAMERA_NAME = "parade_visible_zoom_test_cam"
RENDER_SAMPLES = 8
RENDER_RES_X = 1920
RENDER_RES_Y = 1080
TARGET_CENTER_X = 0.5
TARGET_CENTER_Y = 0.5
TARGET_BUFFER_X_PX = 30
TARGET_BUFFER_Y_PX = 30
BUFFER_TOLERANCE_PX = 3
CENTER_TOLERANCE_PX = 2
# Match the exported PNG edge check: count any non-zero alpha as visible.
ALPHA_THRESHOLD = 1.0 / 255.0
MIN_OPAQUE_PIXELS_PER_COLUMN = 1
MIN_OPAQUE_PIXELS_PER_ROW = 1
MAX_ITERATIONS = 12
MAX_SCALE_PER_ITERATION = 1.2
ZOOM_SAFETY = 0.985

POINT_OBJECT_NAMES = [
    "trimmed-parade_base__timeline",
    "trimmed-parade_highResRoad__timeline",
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


def object_bbox_points(obj: bpy.types.Object) -> list[Vector]:
    corners = [Vector(corner) for corner in obj.bound_box]
    if not corners or all(corner == Vector((-1.0, -1.0, -1.0)) for corner in corners):
        return []
    matrix = obj.matrix_world.copy()
    return [matrix @ corner for corner in corners]


def collect_ground_target() -> Vector:
    points: list[Vector] = []
    for name in POINT_OBJECT_NAMES:
        obj = bpy.data.objects[name]
        points.extend(object_bbox_points(obj))
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    zs = [point.z for point in points]
    return Vector((
        (min(xs) + max(xs)) * 0.5,
        (min(ys) + max(ys)) * 0.5,
        min(zs),
    ))


def ensure_camera_copy(source_camera: bpy.types.Object, target_name: str) -> bpy.types.Object:
    camera = bpy.data.objects.get(target_name)
    if camera is None:
        camera = source_camera.copy()
        camera.data = source_camera.data.copy()
        camera.name = target_name
        camera.data.name = f"{source_camera.data.name}__{target_name}"
        for collection in source_camera.users_collection:
            collection.objects.link(camera)
    return camera


def copy_camera_state(source_camera: bpy.types.Object, target_camera: bpy.types.Object) -> None:
    target_camera.location = source_camera.location.copy()
    target_camera.rotation_mode = source_camera.rotation_mode
    target_camera.rotation_euler = source_camera.rotation_euler.copy()
    target_camera.data.lens = source_camera.data.lens
    target_camera.data.sensor_fit = source_camera.data.sensor_fit
    target_camera.data.angle = source_camera.data.angle
    target_camera.data.clip_start = source_camera.data.clip_start
    target_camera.data.clip_end = source_camera.data.clip_end
    target_camera.data.shift_x = source_camera.data.shift_x
    target_camera.data.shift_y = source_camera.data.shift_y
    target_camera.hide_render = False
    target_camera.hide_viewport = False


def render_preview(scene: bpy.types.Scene, camera: bpy.types.Object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        scene.render.filepath = str(output_path)
        if hasattr(scene, "cycles"):
            scene.cycles.samples = RENDER_SAMPLES
            scene.cycles.preview_samples = RENDER_SAMPLES
        bpy.ops.render.render(write_still=True, scene=scene.name, layer=VIEW_LAYER_NAME, use_viewport=False)
    finally:
        scene.camera = original_camera
        for view_layer in scene.view_layers:
            view_layer.use = original_use.get(view_layer.name, True)


def load_visible_bbox(output_path: Path) -> dict[str, float]:
    image = bpy.data.images.load(str(output_path), check_existing=False)
    try:
        width, height = image.size
        pixels = image.pixels[:]
    finally:
        bpy.data.images.remove(image)

    column_counts = [0] * width
    row_counts = [0] * height

    for y in range(height):
        base_index = y * width * 4 + 3
        for x in range(width):
            alpha = pixels[base_index + x * 4]
            if alpha >= ALPHA_THRESHOLD:
                column_counts[x] += 1
                row_counts[y] += 1

    occupied_columns = [
        index for index, count in enumerate(column_counts)
        if count >= MIN_OPAQUE_PIXELS_PER_COLUMN
    ]
    occupied_rows = [
        index for index, count in enumerate(row_counts)
        if count >= MIN_OPAQUE_PIXELS_PER_ROW
    ]

    if not occupied_columns or not occupied_rows:
        raise ValueError(f"No visible mask detected in {output_path}.")

    left = occupied_columns[0]
    right = occupied_columns[-1]
    bottom = occupied_rows[0]
    top = occupied_rows[-1]
    bbox_width = right - left + 1
    bbox_height = top - bottom + 1

    return {
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "left_gap": left,
        "right_gap": width - right - 1,
        "bottom_gap": bottom,
        "top_gap": height - top - 1,
        "center_x_norm": ((left + right + 1) * 0.5) / width,
        "center_y_norm": ((bottom + top + 1) * 0.5) / height,
    }


def camera_forward(camera: bpy.types.Object) -> Vector:
    return -(object_transform_matrix(camera).to_quaternion() @ Vector((0.0, 0.0, 1.0))).normalized()


def recenter_camera(camera: bpy.types.Object, visible_bbox: dict[str, float]) -> tuple[float, float]:
    delta_x = visible_bbox["center_x_norm"] - TARGET_CENTER_X
    delta_y = visible_bbox["center_y_norm"] - TARGET_CENTER_Y
    camera.data.shift_x += delta_x
    camera.data.shift_y += delta_y
    return delta_x * RENDER_RES_X, delta_y * RENDER_RES_Y


def zoom_camera(camera: bpy.types.Object, target: Vector, visible_bbox: dict[str, float]) -> tuple[bool, float, float, float]:
    target_width = RENDER_RES_X - 2 * TARGET_BUFFER_X_PX
    target_height = RENDER_RES_Y - 2 * TARGET_BUFFER_Y_PX
    width_scale = target_width / max(visible_bbox["bbox_width"], 1.0)
    height_scale = target_height / max(visible_bbox["bbox_height"], 1.0)
    target_scale = min(width_scale, height_scale)

    if abs(target_scale - 1.0) <= 0.001:
        return False, 0.0, 0.0, target_scale

    if target_scale > 1.0:
        applied_scale = min(target_scale, MAX_SCALE_PER_ITERATION)
    else:
        applied_scale = max(target_scale, 1.0 / MAX_SCALE_PER_ITERATION)

    forward = camera_forward(camera)
    current_depth = max((target - camera.location).dot(forward), 1e-6)
    new_depth = current_depth / applied_scale
    camera.location = camera.location + forward * (current_depth - new_depth)
    return True, current_depth, new_depth, applied_scale


def should_continue(visible_bbox: dict[str, float], shift_px_x: float, shift_px_y: float) -> bool:
    horizontal_spare = visible_bbox["left_gap"] + visible_bbox["right_gap"]
    vertical_spare = visible_bbox["top_gap"] + visible_bbox["bottom_gap"]
    horizontal_limit = 2 * TARGET_BUFFER_X_PX + BUFFER_TOLERANCE_PX
    vertical_limit = 2 * TARGET_BUFFER_Y_PX + BUFFER_TOLERANCE_PX
    needs_recentre = abs(shift_px_x) > CENTER_TOLERANCE_PX or abs(shift_px_y) > CENTER_TOLERANCE_PX
    needs_zoom = horizontal_spare > horizontal_limit or vertical_spare > vertical_limit
    return needs_recentre or needs_zoom


def main() -> None:
    output_path = OUTPUT_DIR / OUTPUT_IMAGE_NAME
    scene = bpy.data.scenes[SCENE_NAME]
    source_camera = bpy.data.objects[SOURCE_CAMERA_NAME]
    target_camera = ensure_camera_copy(source_camera, TARGET_CAMERA_NAME)
    copy_camera_state(source_camera, target_camera)
    scene.camera = target_camera

    target = collect_ground_target()

    for iteration in range(1, MAX_ITERATIONS + 1):
        render_preview(scene, target_camera, output_path)
        visible_bbox = load_visible_bbox(output_path)
        shift_px_x, shift_px_y = recenter_camera(target_camera, visible_bbox)
        did_zoom, current_depth, new_depth, applied_scale = zoom_camera(target_camera, target, visible_bbox)

        horizontal_spare = visible_bbox["left_gap"] + visible_bbox["right_gap"]
        vertical_spare = visible_bbox["top_gap"] + visible_bbox["bottom_gap"]

        print(f"PARADE_VISIBLE_ZOOM_TEST iteration={iteration}")
        print(
            "  bbox_px="
            f"({visible_bbox['left']}, {visible_bbox['bottom']}) -> "
            f"({visible_bbox['right']}, {visible_bbox['top']})"
        )
        print(
            "  gaps_px="
            f"left={visible_bbox['left_gap']} right={visible_bbox['right_gap']} "
            f"bottom={visible_bbox['bottom_gap']} top={visible_bbox['top_gap']}"
        )
        print(
            "  centre_shift_px="
            f"x={round(shift_px_x, 3)} y={round(shift_px_y, 3)}"
        )
        print(
            "  spare_px="
            f"horizontal={horizontal_spare} vertical={vertical_spare}"
        )
        print(
            "  zoom="
            f"did_zoom={did_zoom} scale={round(applied_scale, 6)} "
            f"depth={round(current_depth, 6)} -> {round(new_depth, 6)}"
        )

        if not should_continue(visible_bbox, shift_px_x, shift_px_y):
            break

    render_preview(scene, target_camera, output_path)
    final_bbox = load_visible_bbox(output_path)

    print(f"PARADE_VISIBLE_ZOOM_TEST_FINAL {target_camera.name}")
    print(f"  output={output_path}")
    print(
        "  final_gaps_px="
        f"left={final_bbox['left_gap']} right={final_bbox['right_gap']} "
        f"bottom={final_bbox['bottom_gap']} top={final_bbox['top_gap']}"
    )
    print(
        "  final_bbox_px="
        f"width={final_bbox['bbox_width']} height={final_bbox['bbox_height']}"
    )
    print(
        "  final_shift="
        f"x={round(target_camera.data.shift_x, 6)} y={round(target_camera.data.shift_y, 6)}"
    )
    print(f"  iterations={iteration}")

    if bpy.data.is_saved:
        bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
