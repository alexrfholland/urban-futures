from __future__ import annotations

from pathlib import Path

import bpy


OUTPUT_DIR = Path(r"D:\2026 Arboreal Futures\data\renders\timeslices\camera_tests")
VIEW_LAYER_NAME = "debug_camera_framing"
RENDER_SAMPLES = 8
RENDER_RES_X = 1920
RENDER_RES_Y = 1080
ALPHA_THRESHOLD = 1.0 / 255.0
TARGET_IMAGE_NAME = "city-visible-zoom-test-v7.png"
MAX_ITERATIONS = 8
PIXEL_TOLERANCE = 2

SITE_SPECS = [
    {
        "scene_name": "parade",
        "camera_name": "parade_visible_zoom_test_cam",
        "output_image_name": "parade-visible-zoom-test-v7.png",
    },
    {
        "scene_name": "street",
        "camera_name": "street_visible_zoom_test_cam",
        "output_image_name": "street-visible-zoom-test-v7.png",
    },
]


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


def load_visible_bbox(output_path: Path) -> dict[str, int]:
    image = bpy.data.images.load(str(output_path), check_existing=False)
    try:
        width, height = image.size
        pixels = image.pixels[:]
    finally:
        bpy.data.images.remove(image)

    min_x = width
    max_x = -1
    min_y = height
    max_y = -1

    for y in range(height):
        base_index = y * width * 4 + 3
        for x in range(width):
            if pixels[base_index + x * 4] >= ALPHA_THRESHOLD:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

    if max_x < 0 or max_y < 0:
        raise ValueError(f"No visible pixels found in {output_path}.")

    return {
        "left_gap": min_x,
        "right_gap": width - 1 - max_x,
        "bottom_gap": min_y,
        "top_gap": height - 1 - max_y,
        "bbox_width": max_x - min_x + 1,
        "bbox_height": max_y - min_y + 1,
    }


def match_bottom_gap(scene: bpy.types.Scene, camera: bpy.types.Object, target_bottom_gap: int, output_path: Path) -> dict[str, int]:
    latest_bbox: dict[str, int] | None = None
    for iteration in range(1, MAX_ITERATIONS + 1):
        render_preview(scene, camera, output_path)
        latest_bbox = load_visible_bbox(output_path)
        delta_px = latest_bbox["bottom_gap"] - target_bottom_gap
        print(f"MATCH_BOTTOM {scene.name} iteration={iteration} bottom_gap={latest_bbox['bottom_gap']} target={target_bottom_gap} delta={delta_px} shift_y={camera.data.shift_y}")
        if abs(delta_px) <= PIXEL_TOLERANCE:
            break
        camera.data.shift_y += delta_px / RENDER_RES_Y

    if latest_bbox is None:
        raise ValueError(f"No bbox measured for {scene.name}.")
    return latest_bbox


def main() -> None:
    target_bbox = load_visible_bbox(OUTPUT_DIR / TARGET_IMAGE_NAME)
    target_bottom_gap = target_bbox["bottom_gap"]
    print(f"CITY_TARGET bottom_gap={target_bottom_gap} top_gap={target_bbox['top_gap']} left_gap={target_bbox['left_gap']} right_gap={target_bbox['right_gap']}")

    for spec in SITE_SPECS:
        scene = bpy.data.scenes[spec["scene_name"]]
        camera = bpy.data.objects[spec["camera_name"]]
        output_path = OUTPUT_DIR / spec["output_image_name"]
        bbox = match_bottom_gap(scene, camera, target_bottom_gap, output_path)
        print(
            f"FINAL {spec['scene_name']} output={output_path} "
            f"left={bbox['left_gap']} right={bbox['right_gap']} "
            f"bottom={bbox['bottom_gap']} top={bbox['top_gap']} "
            f"shift_y={camera.data.shift_y}"
        )

    if bpy.data.is_saved:
        bpy.ops.wm.save_mainfile()


if __name__ == "__main__":
    main()
