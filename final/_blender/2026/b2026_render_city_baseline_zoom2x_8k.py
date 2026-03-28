from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
BUILD_SCRIPT = REPO_ROOT / "final" / "_blender" / "2026" / "b2026_build_city_baseline.py"
PNG_RENDER_PATH = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "baseline_renders"
    / "city_baseline_pathway_8k_zoom2x.png"
)
EXR_VIEW_LAYERS = ("pathway_state", "existing_condition", "city_priority")


def load_module(module_name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_name(value: str) -> str:
    return "".join(char if char not in '\\/:*?"<>|' else "_" for char in value).strip()


def exr_output_dir_for_scene(scene: bpy.types.Scene) -> Path:
    blend_path = Path(bpy.data.filepath)
    return blend_path.parent / f"{blend_path.stem}-{safe_name(scene.name)}"


def archive_exr_set(scene: bpy.types.Scene, archive_dir: Path) -> None:
    source_dir = exr_output_dir_for_scene(scene)
    archive_dir.mkdir(parents=True, exist_ok=True)
    for existing in archive_dir.glob("*.exr"):
        existing.unlink()
    for view_layer_name in EXR_VIEW_LAYERS:
        filename = f"{safe_name(scene.name)}-{safe_name(view_layer_name)}.exr"
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Expected EXR was not rendered: {source_path}")
        shutil.copy2(source_path, archive_dir / filename)


def delete_object_if_present(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and getattr(data, "users", 0) == 0:
        bpy.data.cameras.remove(data)


def ensure_zoom_camera(scene: bpy.types.Scene, source_camera_name: str = "WorldCam") -> bpy.types.Object:
    source_camera = bpy.data.objects.get(source_camera_name)
    if source_camera is None or source_camera.type != "CAMERA":
        raise ValueError(f"Expected source camera '{source_camera_name}' was not found")

    delete_object_if_present("WorldCam_2x")

    zoom_camera = source_camera.copy()
    zoom_camera.data = source_camera.data.copy()
    zoom_camera.name = "WorldCam_2x"
    zoom_camera.data.name = "WorldCam_2x"
    if zoom_camera.data.type != "ORTHO":
        raise ValueError("Expected an orthographic camera for 2x zoom")
    zoom_camera.data.ortho_scale = source_camera.data.ortho_scale / 2.0

    linked = False
    for collection in source_camera.users_collection:
        collection.objects.link(zoom_camera)
        linked = True
    if not linked:
        scene.collection.objects.link(zoom_camera)

    return zoom_camera


def main() -> None:
    build_module = load_module("b2026_build_city_baseline_runtime_zoom2x", BUILD_SCRIPT)
    scene = bpy.data.scenes[build_module.BASELINE_SCENE_NAME]

    build_module.retarget_render_layers(scene)
    build_module.configure_baseline_exr_outputs(scene)
    build_module.enable_baseline_view_layers(scene)

    scene.render.resolution_x = 7680
    scene.render.resolution_y = 4320
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(PNG_RENDER_PATH)
    scene.render.image_settings.file_format = "PNG"
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    if hasattr(scene.view_settings, "look"):
        scene.view_settings.look = "None"
    PNG_RENDER_PATH.parent.mkdir(parents=True, exist_ok=True)

    original_camera = bpy.data.objects.get("WorldCam") or scene.camera
    if original_camera is None:
        raise ValueError("No baseline render camera was found")
    zoom_camera = ensure_zoom_camera(scene, original_camera.name)
    scene.camera = zoom_camera

    print("[city_baseline_zoom2x] Rendering 2x zoom baseline set", flush=True)
    bpy.ops.render.render(scene=scene.name)

    archive_dir = exr_output_dir_for_scene(scene).parent / f"{exr_output_dir_for_scene(scene).name}_zoom2x_8k"
    archive_exr_set(scene, archive_dir)
    scene.camera = original_camera
    bpy.ops.wm.save_mainfile()
    print(
        f"[city_baseline_zoom2x] Archived EXRs to {archive_dir} and saved blend with WorldCam_2x",
        flush=True,
    )


if __name__ == "__main__":
    main()
