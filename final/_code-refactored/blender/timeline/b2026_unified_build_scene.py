from __future__ import annotations

from pathlib import Path
import os
import sys

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from b2026_unified_runtime import env_flag, run_local_script
import b2026_timeline_generate_single_state as single_state_builder
import b2026_unified_scene_contract as unified_contract


def apply_post_build_view_layers() -> None:
    if unified_contract.get_build_mode() != "single_state":
        return

    scene_name = (
        os.environ.get("B2026_SCENE_NAME", "").strip()
        or os.environ.get("B2026_TARGET_SCENE_NAME", "").strip()
    )
    scene = bpy.data.scenes.get(scene_name) if scene_name else bpy.context.scene
    if scene is None:
        raise ValueError("Single-state post-build view-layer setup could not find a scene")

    site_name = os.environ.get("B2026_SITE_KEY", "").strip()
    if not site_name:
        inferred = unified_contract.infer_site_from_scene_name(scene.name)
        if inferred is None:
            raise ValueError(f"Could not infer site from scene '{scene.name}'")
        site_name = inferred

    year = int(os.environ.get("B2026_SINGLE_STATE_YEAR", "180"))
    single_state_builder.apply_single_state_view_layers(scene, site_name, year)
    if os.environ.get("B2026_SAVE_MAINFILE", "1") != "0":
        bpy.ops.wm.save_mainfile()


def main() -> None:
    if env_flag("B2026_UNIFIED_BUILD_TEMPLATE", default=False):
        run_local_script("b2026_unified_build_template.py")

    run_local_script("b2026_unified_build_instancers.py")
    run_local_script("b2026_unified_build_bioenvelopes.py")
    run_local_script("b2026_unified_build_world.py")
    apply_post_build_view_layers()

    if env_flag("B2026_UNIFIED_VALIDATE", default=True):
        run_local_script("b2026_unified_validate_scene.py")


if __name__ == "__main__":
    main()
