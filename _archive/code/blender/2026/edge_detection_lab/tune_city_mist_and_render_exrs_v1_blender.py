from __future__ import annotations

import os
import runpy
from pathlib import Path

import bpy


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
SETUP_EXR_OUTPUTS_SCRIPT = REPO_ROOT / "final" / "_blender" / "2026" / "b2026_setup_view_layer_exr_outputs.py"
TARGET_SCENE_NAME = os.environ.get("B2026_TARGET_SCENE_NAME", "city")
MIST_START = float(os.environ.get("B2026_MIST_START", "60.0"))
MIST_DEPTH = float(os.environ.get("B2026_MIST_DEPTH", "700.0"))
MIST_FALLOFF = os.environ.get("B2026_MIST_FALLOFF", "LINEAR").upper()
TUNED_BLEND_PATH = Path(
    os.environ.get(
        "B2026_TUNED_BLEND_PATH",
        str(REPO_ROOT / "data" / "blender" / "2026" / f"2026 futures heroes6_mist_s{int(MIST_START)}_d{int(MIST_DEPTH)}_{MIST_FALLOFF.lower()}.blend"),
    )
)


def log(message: str) -> None:
    print(f"[tune_city_mist_and_render_exrs_v1_blender] {message}")


def require_scene(scene_name: str) -> bpy.types.Scene:
    scene = bpy.data.scenes.get(scene_name)
    if scene is None:
        raise ValueError(f"Scene '{scene_name}' was not found.")
    return scene


def configure_mist(scene: bpy.types.Scene) -> None:
    if scene.world is None:
        raise ValueError(f"Scene '{scene.name}' has no world.")
    mist = scene.world.mist_settings
    try:
        mist.use_mist = True
    except Exception:
        pass
    mist.start = MIST_START
    mist.depth = MIST_DEPTH
    mist.falloff = MIST_FALLOFF
    for view_layer in scene.view_layers:
        view_layer.use_pass_mist = True
    log(
        f"Configured mist on scene '{scene.name}': start={mist.start:.2f}, depth={mist.depth:.2f}, "
        f"falloff={mist.falloff}, view_layers={len(scene.view_layers)}"
    )


def save_tuned_copy() -> None:
    TUNED_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(TUNED_BLEND_PATH))
    log(f"Saved tuned blend copy: {TUNED_BLEND_PATH}")


def render_exrs(scene: bpy.types.Scene) -> Path:
    os.environ["B2026_TARGET_SCENE_NAME"] = scene.name
    runpy.run_path(str(SETUP_EXR_OUTPUTS_SCRIPT), run_name="__main__")
    scene.frame_set(scene.frame_start)
    bpy.ops.render.render(scene=scene.name, write_still=False)
    output_dir = TUNED_BLEND_PATH.parent / f"{TUNED_BLEND_PATH.stem}-{scene.name}"
    log(f"Rendered EXRs to: {output_dir}")
    return output_dir


def main() -> None:
    if not bpy.data.filepath:
        raise ValueError("Open the source blend before running this script.")
    source_path = Path(bpy.data.filepath)
    scene = require_scene(TARGET_SCENE_NAME)
    log(f"Source blend: {source_path}")
    configure_mist(scene)
    save_tuned_copy()
    render_exrs(scene)


if __name__ == "__main__":
    main()
