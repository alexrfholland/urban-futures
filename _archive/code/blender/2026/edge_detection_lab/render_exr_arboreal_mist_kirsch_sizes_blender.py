from __future__ import annotations

import os
import runpy
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
SCRIPT_PATH = REPO_ROOT / "code" / "blender" / "2026" / "edge_detection_lab" / "render_exr_arboreal_mist_variants_v2_blender.py"
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"

os.environ.setdefault(
    "EDGE_LAB_OUTPUT_DIR",
    str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_arboreal_mist_kirsch_sizes"),
)
os.environ.setdefault(
    "EDGE_LAB_BLEND_PATH",
    str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_arboreal_mist_kirsch_sizes.blend"),
)
os.environ.setdefault(
    "EDGE_LAB_PREP_OUTPUT_DIR",
    str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_arboreal_mist_kirsch_sizes" / "_prep"),
)
os.environ.setdefault("EDGE_LAB_VARIANT_PRESET", "kirschsizes")
os.environ.setdefault("EDGE_LAB_RENDER_MODE", "edges_only")
os.environ.setdefault("EDGE_LAB_WORKFLOW_ID", "blender_exr_arboreal_mist_kirschsizes_v1")
os.environ.setdefault(
    "EDGE_LAB_WORKFLOW_NOTES",
    "Exports only the Kirsch mist edge PNGs in three widths: thin, fine, and extra-thin.",
)
os.environ.setdefault("EDGE_LAB_PATHWAY_EXR", str(EXR_ROOT / "city-pathway_state.exr"))
os.environ.setdefault("EDGE_LAB_PRIORITY_EXR", str(EXR_ROOT / "city-city_priority.exr"))
os.environ.setdefault("EDGE_LAB_TRENDING_EXR", str(EXR_ROOT / "city-trending_state.exr"))

runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
