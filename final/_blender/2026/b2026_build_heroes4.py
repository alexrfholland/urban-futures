import bpy
import importlib.util
import json
import sys
from pathlib import Path
from mathutils import Vector


SOURCE_BLEND = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes3.blend"
)
OUTPUT_BLEND = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/2026 futures heroes4.blend"
)
VERIFICATION_DIR = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/heroes4_verification"
)
CAMERA_CLIPBOXES_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/final/_blender/2026/b2026_camera_clipboxes.py"
)
TEST_RENDER_SAMPLES = 64
TEST_RENDER_RESOLUTION = (1920, 1080)


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sync_text_block(text_name: str, source_path: Path, register_module: bool = False):
    source_text = source_path.read_text(encoding="utf-8")
    text = bpy.data.texts.get(text_name)
    if text is None:
        text = bpy.data.texts.new(text_name)
    text.clear()
    text.write(source_text)
    text.use_module = register_module
    return text


def duplicate_camera(scene_name: str, source_camera_name: str, new_name: str):
    if bpy.data.objects.get(new_name) is not None:
        return bpy.data.objects[new_name]
    source = bpy.data.objects[source_camera_name]
    new_camera_data = source.data.copy()
    new_object = source.copy()
    new_object.data = new_camera_data
    new_object.name = new_name
    new_object.data.name = new_name
    target_collection_name = "City_Camera" if scene_name == "city" else "Parade_Camera"
    bpy.data.collections[target_collection_name].objects.link(new_object)
    return new_object


def configure_test_cameras():
    city_scene = bpy.data.scenes["city"]
    parade_scene = bpy.data.scenes["parade"]

    city_original = city_scene.camera
    parade_original = parade_scene.camera

    city_alt = duplicate_camera("city", city_original.name, "City Clip Test Camera")
    parade_alt = duplicate_camera(
        "parade", parade_original.name, "Parade Clip Test Camera"
    )

    return {
        "city": {"primary": city_original.name, "alternate": city_alt.name},
        "parade": {"primary": parade_original.name, "alternate": parade_alt.name},
    }


def adjust_proxy(mod, scene_name: str, camera_name: str, location_offset: Vector, scale_factor: float):
    spec = mod.SCENE_SPECS[scene_name]
    camera = bpy.data.objects[camera_name]
    proxy_name = camera.get("clip_proxy_object")
    if not proxy_name:
        return None
    proxy = bpy.data.objects[proxy_name]
    proxy.location = proxy.location + location_offset
    proxy.scale = proxy.scale * scale_factor
    return proxy.name


def render_scene(clipmod, scene_name: str, camera_name: str):
    scene = bpy.data.scenes[scene_name]
    original_camera = scene.camera
    original_engine = scene.render.engine
    original_samples = scene.cycles.samples
    original_res_x = scene.render.resolution_x
    original_res_y = scene.render.resolution_y
    original_res_pct = scene.render.resolution_percentage
    original_format = scene.render.image_settings.file_format
    original_color_mode = scene.render.image_settings.color_mode
    original_filepath = scene.render.filepath
    scene.camera = bpy.data.objects[camera_name]
    clipmod.sync_scene_clipbox(scene)
    scene.use_nodes = True
    scene.render.engine = "CYCLES"
    scene.cycles.samples = TEST_RENDER_SAMPLES
    scene.render.resolution_x = TEST_RENDER_RESOLUTION[0]
    scene.render.resolution_y = TEST_RENDER_RESOLUTION[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    output_path = VERIFICATION_DIR / f"{scene_name}_{camera_name.replace(' ', '_')}.png"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    render_result = bpy.data.images.get("Render Result")
    if render_result is not None:
        render_result.save_render(filepath=str(output_path), scene=scene)
    if not output_path.exists():
        raise RuntimeError(f"Expected render was not written: {output_path}")
    scene.camera = original_camera
    scene.render.engine = original_engine
    scene.cycles.samples = original_samples
    scene.render.resolution_x = original_res_x
    scene.render.resolution_y = original_res_y
    scene.render.resolution_percentage = original_res_pct
    scene.render.image_settings.file_format = original_format
    scene.render.image_settings.color_mode = original_color_mode
    scene.render.filepath = original_filepath
    return output_path


def main():
    if Path(bpy.data.filepath) != SOURCE_BLEND:
        bpy.ops.wm.open_mainfile(filepath=str(SOURCE_BLEND))

    sync_text_block("camera_clipboxes", CAMERA_CLIPBOXES_PATH, register_module=True)
    clipmod = load_module("b2026_camera_clipboxes_live", CAMERA_CLIPBOXES_PATH)
    clipmod.register()

    camera_map = configure_test_cameras()
    clipmod.ensure_camera_clipboxes()

    adjusted = {
        "city": adjust_proxy(
            clipmod,
            "city",
            camera_map["city"]["alternate"],
            Vector((70.0, 0.0, 0.0)),
            0.65,
        ),
        "parade": adjust_proxy(
            clipmod,
            "parade",
            camera_map["parade"]["alternate"],
            Vector((35.0, 0.0, 0.0)),
            0.65,
        ),
    }

    clipmod.sync_all_scene_clipboxes()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))

    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    renders = []
    for scene_name, cameras in camera_map.items():
        renders.append(str(render_scene(clipmod, scene_name, cameras["primary"])))
        renders.append(str(render_scene(clipmod, scene_name, cameras["alternate"])))
        bpy.data.scenes[scene_name].camera = bpy.data.objects[cameras["primary"]]

    clipmod.sync_all_scene_clipboxes()
    bpy.ops.wm.save_mainfile()

    summary_path = VERIFICATION_DIR / "heroes4_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "camera_map": camera_map,
                "adjusted_proxy_objects": adjusted,
                "renders": renders,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
