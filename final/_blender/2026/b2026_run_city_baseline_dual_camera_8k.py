from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
BUILD_SCRIPT = REPO_ROOT / "final" / "_blender" / "2026" / "b2026_build_city_baseline.py"
WORLD_PNG_PATH = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "baseline_renders"
    / "city_baseline_pathway_8k_worldcam.png"
)
ZOOM_PNG_PATH = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "baseline_renders"
    / "city_baseline_pathway_8k_zoom3x.png"
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

    delete_object_if_present("WorldCam_3x")

    zoom_camera = source_camera.copy()
    zoom_camera.data = source_camera.data.copy()
    zoom_camera.name = "WorldCam_3x"
    zoom_camera.data.name = "WorldCam_3x"
    if zoom_camera.data.type != "ORTHO":
        raise ValueError("Expected an orthographic camera for 3x zoom")
    zoom_camera.data.ortho_scale = source_camera.data.ortho_scale / 3.0

    linked = False
    for collection in source_camera.users_collection:
        collection.objects.link(zoom_camera)
        linked = True
    if not linked:
        scene.collection.objects.link(zoom_camera)

    return zoom_camera


def render_and_archive(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    render_path: Path,
    archive_dir: Path,
) -> None:
    render_path.parent.mkdir(parents=True, exist_ok=True)
    scene.camera = camera
    scene.render.filepath = str(render_path)
    print(
        f"[city_baseline_dual] Rendering {camera.name} to {render_path.name}",
        flush=True,
    )
    bpy.ops.render.render(write_still=True, scene=scene.name)
    archive_exr_set(scene, archive_dir)
    print(
        f"[city_baseline_dual] Archived EXRs for {camera.name} to {archive_dir}",
        flush=True,
    )


def main() -> None:
    os.environ["B2026_BASELINE_SAVE_MAINFILE"] = "1"
    os.environ["B2026_BASELINE_RENDER"] = "0"
    os.environ["B2026_BASELINE_MUTE_FILE_OUTPUTS"] = "0"
    os.environ["B2026_BASELINE_RENDER_PATH"] = str(WORLD_PNG_PATH)

    module = load_module("b2026_build_city_baseline_runtime_dual", BUILD_SCRIPT)
    original_configure_render = module.configure_render

    def patched_configure_render(scene):
        original_configure_render(scene)
        scene.render.resolution_x = 7680
        scene.render.resolution_y = 4320
        scene.render.resolution_percentage = 100
        print("[city_baseline_dual] Forced render resolution 7680x4320", flush=True)

    module.configure_render = patched_configure_render
    module.main()

    scene = bpy.data.scenes[module.BASELINE_SCENE_NAME]
    original_camera = bpy.data.objects.get("WorldCam") or scene.camera
    if original_camera is None:
        raise ValueError("No baseline render camera was found")
    zoom_camera = ensure_zoom_camera(scene, original_camera.name)

    source_dir = exr_output_dir_for_scene(scene)
    world_archive_dir = source_dir.parent / f"{source_dir.name}_worldcam_8k"
    zoom_archive_dir = source_dir.parent / f"{source_dir.name}_zoom3x_8k"

    render_and_archive(scene, original_camera, WORLD_PNG_PATH, world_archive_dir)
    render_and_archive(scene, zoom_camera, ZOOM_PNG_PATH, zoom_archive_dir)

    scene.camera = original_camera
    bpy.ops.wm.save_mainfile()
    print("[city_baseline_dual] Saved baseline blend with WorldCam_3x", flush=True)


if __name__ == "__main__":
    main()
