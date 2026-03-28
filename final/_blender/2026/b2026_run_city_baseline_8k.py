from __future__ import annotations

import importlib.util
import os
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
BLEND_RENDER_PATH = (
    REPO_ROOT / "data" / "blender" / "2026" / "baseline_renders" / "city_baseline_pathway_8k_refresh.png"
)
BUILD_SCRIPT = REPO_ROOT / "final" / "_blender" / "2026" / "b2026_build_city_baseline.py"


def load_module(module_name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    os.environ["B2026_BASELINE_SAVE_MAINFILE"] = "1"
    os.environ["B2026_BASELINE_RENDER"] = "1"
    os.environ["B2026_BASELINE_MUTE_FILE_OUTPUTS"] = "0"
    os.environ["B2026_BASELINE_RENDER_PATH"] = str(BLEND_RENDER_PATH)

    module = load_module("b2026_build_city_baseline_runtime", BUILD_SCRIPT)
    original_configure_render = module.configure_render

    def patched_configure_render(scene):
        original_configure_render(scene)
        scene.render.resolution_x = 7680
        scene.render.resolution_y = 4320
        scene.render.resolution_percentage = 100
        print("[city_baseline] Forced render resolution 7680x4320", flush=True)

    module.configure_render = patched_configure_render
    module.main()


if __name__ == "__main__":
    main()
