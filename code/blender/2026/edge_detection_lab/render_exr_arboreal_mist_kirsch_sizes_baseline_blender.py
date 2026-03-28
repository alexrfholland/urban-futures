from __future__ import annotations

import os
import runpy
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
SCRIPT_PATH = REPO_ROOT / "code" / "blender" / "2026" / "edge_detection_lab" / "render_exr_arboreal_mist_variants_v2_blender.py"
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6_baseline-city"

os.environ.setdefault(
    "EDGE_LAB_OUTPUT_DIR",
    str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city_blender_arboreal_mist_kirsch_sizes_baseline"),
)
os.environ.setdefault(
    "EDGE_LAB_BLEND_PATH",
    str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "exr_city_blender_arboreal_mist_kirsch_sizes_baseline.blend"),
)
os.environ.setdefault(
    "EDGE_LAB_PREP_OUTPUT_DIR",
    str(
        REPO_ROOT
        / "data"
        / "blender"
        / "2026"
        / "edge_detection_lab"
        / "outputs"
        / "exr_city_blender_arboreal_mist_kirsch_sizes_baseline"
        / "_prep"
    ),
)
os.environ.setdefault("EDGE_LAB_VARIANT_PRESET", "kirschsizes")
os.environ.setdefault("EDGE_LAB_RENDER_MODE", "edges_only")
os.environ.setdefault("EDGE_LAB_WORKFLOW_ID", "blender_exr_arboreal_mist_kirschsizes_baseline_v1")
os.environ.setdefault(
    "EDGE_LAB_WORKFLOW_NOTES",
    "Runs the Kirsch mist-size arboreal edge workflow on the baseline city EXRs.",
)
os.environ.setdefault("EDGE_LAB_PATHWAY_EXR", str(EXR_ROOT / "city-pathway_state.exr"))
os.environ.setdefault("EDGE_LAB_PRIORITY_EXR", str(EXR_ROOT / "city-city_priority.exr"))
os.environ.setdefault("EDGE_LAB_TRENDING_EXR", str(EXR_ROOT / "city-existing_condition.exr"))

runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
