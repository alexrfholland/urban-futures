from __future__ import annotations

import os
import runpy
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
SCRIPT_PATH = REPO_ROOT / "code" / "blender" / "2026" / "edge_detection_lab" / "render_exr_arboreal_mist_variants_v2_blender.py"

os.environ.setdefault(
    "EDGE_LAB_OUTPUT_DIR",
    str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_0000_arboreal_mist_v2_extrathin"),
)
os.environ.setdefault(
    "EDGE_LAB_BLEND_PATH",
    str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_0000_arboreal_mist_v2_extrathin.blend"),
)
os.environ.setdefault("EDGE_LAB_VARIANT_PRESET", "extrathin")
os.environ.setdefault("EDGE_LAB_WORKFLOW_ID", "blender_exr_arboreal_mist_v2_extrathin")
os.environ.setdefault(
    "EDGE_LAB_WORKFLOW_NOTES",
    "Reuses the screen-lift arboreal mist workflow but pushes thresholds and width shaping toward a finer extra-thin line.",
)

runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
