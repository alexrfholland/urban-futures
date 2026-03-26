import json
import shutil
from datetime import datetime
from pathlib import Path

import bpy


ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
HERO_BLEND = ROOT / "data/blender/2026/2026 futures heroes.blend"
SNAPSHOT_ROOT = ROOT / "data/blender/2026/2026 futures heroes_snapshots"
VALIDATION_ROOT = ROOT / "data/blender/2026/snapshot_validation"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = VALIDATION_ROOT / f"full_snapshot_test_{TIMESTAMP}"
SCENES = ("city", "parade")
VIEW_LAYERS = ("pathway_state", "existing_condition")


def log(message: str) -> None:
    print(f"[snapshot_test] {message}")


def helper_output_path(scene_name: str, view_layer_name: str) -> Path:
    return SNAPSHOT_ROOT / f"{scene_name}_{view_layer_name}0000.exr"


def archived_output_path(scene_name: str, view_layer_name: str) -> Path:
    return RUN_DIR / f"{scene_name}_{view_layer_name}.exr"


def live_png_path(scene_name: str) -> Path:
    return RUN_DIR / f"{scene_name}_live.png"


def render_scene(scene_name: str) -> None:
    scene = bpy.data.scenes[scene_name]
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 512
    scene.cycles.preview_samples = 64
    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(live_png_path(scene_name))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    log(f"Rendering live scene {scene_name}")
    bpy.ops.render.render(write_still=True, scene=scene_name)


def move_exrs(scene_name: str) -> None:
    for view_layer_name in VIEW_LAYERS:
        source = helper_output_path(scene_name, view_layer_name)
        target = archived_output_path(scene_name, view_layer_name)
        if not source.exists():
            raise FileNotFoundError(f"Expected EXR not found after rendering {scene_name}: {source}")
        shutil.move(str(source), str(target))
        log(f"Archived {scene_name}:{view_layer_name} EXR to {target}")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "hero_blend": str(HERO_BLEND),
        "run_dir": str(RUN_DIR),
        "live_pngs": {},
        "exrs": {},
    }

    for scene_name in SCENES:
        render_scene(scene_name)
        move_exrs(scene_name)
        manifest["live_pngs"][scene_name] = str(live_png_path(scene_name))
        manifest["exrs"][scene_name] = {
            view_layer_name: str(archived_output_path(scene_name, view_layer_name))
            for view_layer_name in VIEW_LAYERS
        }

    manifest_path = RUN_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
