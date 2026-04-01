from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_local_module(module_name: str, filename: str):
    file_path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    scene = bpy.data.scenes.get("city")
    if scene is None:
        raise ValueError("Scene 'city' not found")
    if getattr(bpy.context, "window", None) is not None:
        bpy.context.window.scene = scene

    instancer = load_local_module(
        "b2026_timeline_instancer_refresh_city_runtime",
        "b2026_timeline_instancer.py",
    )
    instancer.TARGET_SCENE_NAME = "city"
    instancer.SITE = "city"
    instancer.main()
    bpy.ops.wm.save_mainfile()
    print("CITY_REFRESH instancer saved")

    clipbox_setup = load_local_module(
        "b2026_timeline_clipbox_setup_refresh_city_runtime",
        "b2026_timeline_clipbox_setup.py",
    )
    timeline_layout = load_local_module(
        "b2026_timeline_layout_refresh_city_runtime",
        "b2026_timeline_layout.py",
    )
    bio = load_local_module(
        "b2026_timeline_bioenvelopes_refresh_city_runtime",
        "b2026_timeline_bioenvelopes.py",
    )
    for scenario in ("positive", "trending"):
        bio.build_site_scenario_bioenvelopes("city", scenario, clipbox_setup, timeline_layout)
    bpy.ops.wm.save_mainfile()
    print("CITY_REFRESH bioenvelopes saved")

    os.environ["B2026_SITE_KEY"] = "city"
    os.environ["B2026_SCENE_NAME"] = "city"
    os.environ["B2026_WORLD_SCENARIO"] = "positive"
    os.environ["B2026_SAVE_MAINFILE"] = "1"
    world = load_local_module(
        "b2026_timeline_rebuild_world_year_attrs_refresh_city_runtime",
        "b2026_timeline_rebuild_world_year_attrs.py",
    )
    world.SITE_KEY = "city"
    world.SCENE_NAME = "city"
    world.WORLD_SCENARIO = "positive"
    world.main()
    print("CITY_REFRESH world attrs saved")


if __name__ == "__main__":
    main()
