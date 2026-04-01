from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import bpy


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
BLEND_PATH = env_path(
    "EDGE_LAB_BLEND_PATH",
    str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "edge_lab_final_template.blend"),
)
OUTPUT_DIR = env_path(
    "EDGE_LAB_OUTPUT_DIR",
    str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "edge_lab_final_template_depth_outliner"),
)
PATHWAY_EXR = env_path(
    "EDGE_LAB_PATHWAY_EXR",
    str(REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city" / "city-pathway_state.exr"),
)
PRIORITY_EXR = env_path(
    "EDGE_LAB_PRIORITY_EXR",
    str(REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city" / "city-city_priority.exr"),
)
TRENDING_EXR = env_path(
    "EDGE_LAB_TRENDING_EXR",
    str(REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city" / "city-trending_state.exr"),
)
SCRATCH_SCENE_NAME = os.environ.get("EDGE_LAB_DEPTH_SCENE_NAME", "__CurrentDepthOutlinerScratch")


def log(message: str) -> None:
    print(f"[render_edge_lab_current_depth_outliner] {message}")


def load_legacy_depth_module():
    module_path = Path(__file__).with_name("render_exr_arboreal_depth_outliner_baseline_blender.py")
    spec = importlib.util.spec_from_file_location("edge_lab_legacy_depth_outliner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(BLEND_PATH)
    for path in (PATHWAY_EXR, PRIORITY_EXR, TRENDING_EXR):
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))

    legacy = load_legacy_depth_module()
    legacy.OUTPUT_DIR = OUTPUT_DIR
    legacy.BLEND_PATH = BLEND_PATH
    legacy.PATHWAY_EXR = PATHWAY_EXR
    legacy.PRIORITY_EXR = PRIORITY_EXR
    legacy.TRENDING_EXR = TRENDING_EXR

    existing = bpy.data.scenes.get(SCRATCH_SCENE_NAME)
    if existing is not None:
        bpy.data.scenes.remove(existing)
    scratch = bpy.data.scenes.new(SCRATCH_SCENE_NAME)
    log(f"Rendering depth outliner outputs from EXRs into {OUTPUT_DIR}")
    rendered_paths = legacy.build_scene(scratch)
    bpy.context.window.scene = scratch
    legacy.finalize_render(scratch, rendered_paths)
    for duplicate in OUTPUT_DIR.glob("*_0001.png"):
        duplicate.unlink()
    for prep in OUTPUT_DIR.glob("*_depth_normalized_visible_arboreal.png"):
        prep.unlink()
    if scratch.name in bpy.data.scenes:
        bpy.data.scenes.remove(scratch)

    base_outlines = OUTPUT_DIR.parent / "base" / "base_outlines.png"
    if base_outlines.exists():
        destination = OUTPUT_DIR / "base_depth_outliner.png"
        shutil.copy2(base_outlines, destination)
        log(f"Copied {base_outlines} -> {destination}")


if __name__ == "__main__":
    main()
