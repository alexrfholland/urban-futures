from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
COMPOSITOR_ROOT = REPO_ROOT / "_futureSim_refactored" / "blender" / "compositor"
SCRIPT_ROOT = COMPOSITOR_ROOT / "scripts"
CANONICAL_ROOT = COMPOSITOR_ROOT / "canonical_templates"
DATA_ROOT = REPO_ROOT / "_data-refactored" / "compositor"
LEGACY_CODE_ROOT = REPO_ROOT / "code" / "blender" / "2026" / "edge_detection_lab"
LEGACY_DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
BLENDER_BIN = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


FINAL_TEMPLATE_BLEND = env_path(
    "EDGE_LAB_FINAL_TEMPLATE_BLEND",
    CANONICAL_ROOT / "edge_lab_final_template_safe_rebuild_20260405.blend",
)
CURRENT_SOURCE_BLEND = env_path(
    "EDGE_LAB_CURRENT_SOURCE_BLEND",
    LEGACY_DATA_ROOT / "edge_lab_output_suite_combined.blend",
)
LEGACY_SOURCE_BLEND = env_path(
    "EDGE_LAB_LEGACY_SOURCE_BLEND",
    LEGACY_DATA_ROOT / "city_exr_compositor_lightweight_city_final.blend",
)
OUTPUT_ROOT = env_path(
    "EDGE_LAB_OUTPUT_ROOT",
    DATA_ROOT / "outputs" / "edge_lab_final_template",
)
EXR_ROOT = env_path(
    "EDGE_LAB_EXR_ROOT",
    LEGACY_DATA_ROOT / "inputs" / "LATEST_REMOTE_EXRS" / "simv3-7_20260405_8k64s_simv3-7" / "city_timeline",
)

PATHWAY_EXR = env_path("EDGE_LAB_PATHWAY_EXR", EXR_ROOT / "city_timeline__positive_state__8k64s.exr")
PRIORITY_EXR = env_path("EDGE_LAB_PRIORITY_EXR", EXR_ROOT / "city_timeline__positive_priority_state__8k64s.exr")
EXISTING_EXR = env_path("EDGE_LAB_EXISTING_EXR", EXR_ROOT / "city_timeline__existing_condition_positive__8k64s.exr")
EXISTING_TRENDING_EXR = env_path(
    "EDGE_LAB_EXISTING_TRENDING_EXR",
    EXR_ROOT / "city_timeline__existing_condition_trending__8k64s.exr",
)
TRENDING_EXR = env_path("EDGE_LAB_TRENDING_EXR", EXR_ROOT / "city_timeline__trending_state__8k64s.exr")
BIOENVELOPE_EXR = env_path("EDGE_LAB_BIOENVELOPE_EXR", EXR_ROOT / "city_timeline__bioenvelope_positive__8k64s.exr")
BIOENVELOPE_TRENDING_EXR = env_path(
    "EDGE_LAB_BIOENVELOPE_TRENDING_EXR",
    EXR_ROOT / "city_timeline__bioenvelope_trending__8k64s.exr",
)


def log(message: str) -> None:
    print(f"[run_edge_lab_final_template] {message}")


def run_blender_python(script_path: Path, env: dict[str, str]) -> None:
    command = [
        str(BLENDER_BIN),
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
    ]
    subprocess.run(command, check=True, env={**os.environ, **env})


def build_template() -> None:
    run_blender_python(
        LEGACY_CODE_ROOT / "build_edge_lab_final_template_blend.py",
        {
            "EDGE_LAB_CURRENT_SOURCE_BLEND": str(LEGACY_DATA_ROOT / "edge_lab_output_suite_refined.blend"),
            "EDGE_LAB_LEGACY_SOURCE_BLEND": str(LEGACY_SOURCE_BLEND),
            "EDGE_LAB_FINAL_TEMPLATE_BLEND": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_PROPOSAL_PATHWAY_EXR": str(PATHWAY_EXR),
            "EDGE_LAB_PROPOSAL_PRIORITY_EXR": str(PRIORITY_EXR),
            "EDGE_LAB_PROPOSAL_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_current_outputs() -> None:
    current_root = OUTPUT_ROOT / "current"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_core_outputs.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_SCENE_NAME": "Current",
            "EDGE_LAB_OUTPUT_ROOT": str(current_root),
            "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
            "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
            "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
            "EDGE_LAB_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_legacy_shading() -> None:
    shading_root = OUTPUT_ROOT / "legacy_shading"
    run_blender_python(
        LEGACY_CODE_ROOT / "render_edge_lab_legacy_shading.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_SCENE_NAME": "Legacy",
            "EDGE_LAB_OUTPUT_DIR": str(shading_root),
            "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
            "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
            "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
        },
    )


def run_current_shading() -> None:
    shading_root = OUTPUT_ROOT / "current" / "shading"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_shading.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_SCENE_NAME": "Current",
            "EDGE_LAB_OUTPUT_DIR": str(shading_root),
            "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
            "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
            "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
            "EDGE_LAB_EXISTING_TRENDING_EXR": str(EXISTING_TRENDING_EXR),
            "EDGE_LAB_BIOENVELOPE_EXR": str(BIOENVELOPE_EXR),
            "EDGE_LAB_BIOENVELOPE_TRENDING_EXR": str(BIOENVELOPE_TRENDING_EXR),
        },
    )


def run_current_bioenvelopes() -> None:
    bio_root = OUTPUT_ROOT / "current" / "bioenvelope"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_bioenvelopes.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_SCENE_NAME": "Current",
            "EDGE_LAB_OUTPUT_DIR": str(bio_root),
            "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
            "EDGE_LAB_TRENDING_EXR": str(TRENDING_EXR),
            "EDGE_LAB_BIOENVELOPE_EXR": str(BIOENVELOPE_EXR),
            "EDGE_LAB_BIOENVELOPE_TRENDING_EXR": str(BIOENVELOPE_TRENDING_EXR),
        },
    )


def run_current_mist() -> None:
    mist_root = OUTPUT_ROOT / "current" / "outlines_mist"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_mist.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_SCENE_NAME": "Current",
            "EDGE_LAB_OUTPUT_DIR": str(mist_root),
            "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
            "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
            "EDGE_LAB_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_current_depth_outliner() -> None:
    depth_root = OUTPUT_ROOT / "current" / "depth_outliner"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_depth_outliner.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_OUTPUT_DIR": str(depth_root),
            "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
            "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
            "EDGE_LAB_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_current_base() -> None:
    base_root = OUTPUT_ROOT / "current" / "base"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_base.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_SCENE_NAME": "Current",
            "EDGE_LAB_OUTPUT_DIR": str(base_root),
            "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
        },
    )


def run_current_sizes() -> None:
    sizes_root = OUTPUT_ROOT / "current" / "sizes"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_sizes.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_SCENE_NAME": "Current",
            "EDGE_LAB_OUTPUT_DIR": str(sizes_root),
            "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
            "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
            "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
            "EDGE_LAB_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_current_proposals() -> None:
    proposals_root = OUTPUT_ROOT / "current" / "proposals"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_proposals.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_SCENE_NAME": "Current",
            "EDGE_LAB_OUTPUT_DIR": str(proposals_root),
            "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
            "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
            "EDGE_LAB_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def prune_current_outputs() -> None:
    current_root = OUTPUT_ROOT / "current"
    for path in list(current_root.rglob("*")):
        if path.is_file() and (path.name.endswith("_0001.png") or "_prep" in path.parts):
            path.unlink()
    prep_dir = current_root / "outlines_mist" / "_prep"
    if prep_dir.exists():
        shutil.rmtree(prep_dir, ignore_errors=True)


def prune_shading_outputs() -> None:
    for root in (OUTPUT_ROOT / "current" / "shading", OUTPUT_ROOT / "legacy_shading"):
        if not root.exists():
            continue
        for path in root.glob("*_0001.png"):
            path.unlink()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_template()
    run_current_outputs()
    run_current_depth_outliner()
    run_current_mist()
    prune_current_outputs()
    run_current_shading()
    run_current_base()
    run_current_sizes()
    run_current_proposals()
    run_current_bioenvelopes()
    run_legacy_shading()
    prune_shading_outputs()
    log(f"Template blend: {FINAL_TEMPLATE_BLEND}")
    log(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
