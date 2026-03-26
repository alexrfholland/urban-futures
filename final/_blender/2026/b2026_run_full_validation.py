import importlib.util
import json
from pathlib import Path

import bpy


ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
SNAPSHOT_SCRIPT_PATH = ROOT / "final/_blender/2026/b2026_snapshot_compositor.py"
VALIDATION_DIR = ROOT / "data/blender/2026/snapshot_validation"
MANIFEST_PATH = VALIDATION_DIR / "full_validation_manifest.json"
SCENES = ("city", "parade")


def log(message: str) -> None:
    print(f"[full_validate] {message}")


def load_snapshot_module():
    spec = importlib.util.spec_from_file_location("b2026_snapshot_compositor_runtime", SNAPSHOT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def latest_archive(root: Path, before: set[str]) -> Path | None:
    after = {path.name for path in root.iterdir() if path.is_dir()}
    new_names = sorted(after - before)
    if new_names:
        return root / new_names[-1]
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    return candidates[-1] if candidates else None


def save_manifest(data: dict) -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(data, indent=2))


def render_live(scene_name: str, output_path: Path) -> None:
    scene = bpy.data.scenes[scene_name]
    render = scene.render
    image_settings = render.image_settings
    saved = {
        "filepath": render.filepath,
        "file_format": image_settings.file_format,
        "color_mode": image_settings.color_mode,
    }
    try:
        render.filepath = str(output_path)
        image_settings.file_format = "PNG"
        image_settings.color_mode = "RGBA"
        log(f"Rendering live compositor for {scene_name} to {output_path}")
        bpy.ops.render.render(write_still=True, scene=scene_name)
    finally:
        render.filepath = saved["filepath"]
        image_settings.file_format = saved["file_format"]
        image_settings.color_mode = saved["color_mode"]


def export_snapshot(scene_name: str) -> Path:
    module = load_snapshot_module()
    module.TARGET_SCENE_NAME = scene_name
    module.SNAPSHOT_SCENE_NAME = f"{scene_name} Snapshot"
    module.SNAPSHOT_BLEND_FILENAME = f"{scene_name}_snapshot.blend"
    module.OVERRIDE_RESOLUTION_PERCENTAGE = None
    module.OVERRIDE_RENDER_SAMPLES = None

    root = module.snapshot_root_folder()
    before = {path.name for path in root.iterdir() if path.is_dir()}
    log(f"Exporting full snapshot EXRs for {scene_name}")
    module.main()
    archive = latest_archive(root, before)
    if archive is None:
        raise RuntimeError(f"No snapshot archive folder found after exporting {scene_name}")
    return archive


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "blend": bpy.data.filepath,
        "archives": {},
        "renders": {},
    }
    save_manifest(manifest)

    for scene_name in SCENES:
        archive = export_snapshot(scene_name)
        manifest["archives"][scene_name] = str(archive)
        save_manifest(manifest)

    render_live("city", VALIDATION_DIR / "city_live_4k.png")
    manifest["renders"]["city_live"] = str(VALIDATION_DIR / "city_live_4k.png")
    save_manifest(manifest)

    render_live("parade", VALIDATION_DIR / "parade_live_4k.png")
    manifest["renders"]["parade_live"] = str(VALIDATION_DIR / "parade_live_4k.png")
    save_manifest(manifest)

    log(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
