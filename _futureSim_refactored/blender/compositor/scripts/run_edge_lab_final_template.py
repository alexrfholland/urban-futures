from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from _futureSim_refactored.paths import (
    blenderv2_exr_family_dir,
    compositor_run_dir,
    compositor_run_name,
    exr_case_from_family,
)

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


def default_exr_root() -> Path:
    sim_root = os.environ.get("COMPOSITOR_SIM_ROOT", "").strip()
    exr_family = os.environ.get("COMPOSITOR_EXR_FAMILY", "").strip()
    if sim_root and exr_family:
        return blenderv2_exr_family_dir(sim_root, exr_family)
    return LEGACY_DATA_ROOT / "inputs" / "LATEST_REMOTE_EXRS" / "simv3-7_20260405_8k64s_simv3-7" / "city_timeline"


def default_output_root(default_family: str) -> Path:
    sim_root = os.environ.get("COMPOSITOR_SIM_ROOT", "").strip()
    exr_family = os.environ.get("COMPOSITOR_EXR_FAMILY", "").strip()
    compositor_family = os.environ.get("COMPOSITOR_FAMILY", default_family).strip()
    timestamp = os.environ.get("COMPOSITOR_RUN_TIMESTAMP", "").strip() or time.strftime("%Y%m%d_%H%M")
    note = os.environ.get("COMPOSITOR_RUN_NOTE", "").strip()
    if sim_root and exr_family and compositor_family:
        compositor_run = compositor_run_name(compositor_family, timestamp, note or None)
        return compositor_run_dir(sim_root, exr_family, compositor_run)
    return DATA_ROOT / "outputs" / "edge_lab_final_template"


def default_exr_case_name(exr_root: Path) -> str:
    exr_family = os.environ.get("COMPOSITOR_EXR_FAMILY", "").strip()
    if exr_family:
        return exr_case_from_family(exr_family)
    return exr_root.name


FINAL_TEMPLATE_BLEND = env_path(
    "COMPOSITOR_FINAL_TEMPLATE_BLEND",
    CANONICAL_ROOT / "edge_lab_final_template_safe_rebuild_20260405.blend",
)
CURRENT_SOURCE_BLEND = env_path(
    "COMPOSITOR_CURRENT_SOURCE_BLEND",
    LEGACY_DATA_ROOT / "edge_lab_output_suite_combined.blend",
)
LEGACY_SOURCE_BLEND = env_path(
    "COMPOSITOR_LEGACY_SOURCE_BLEND",
    LEGACY_DATA_ROOT / "city_exr_compositor_lightweight_city_final.blend",
)
OUTPUT_ROOT = default_output_root("edge-lab-final-template")
EXR_ROOT = default_exr_root()
EXR_CASE = default_exr_case_name(EXR_ROOT)

PATHWAY_EXR = env_path("COMPOSITOR_PATHWAY_EXR", EXR_ROOT / f"{EXR_CASE}__positive_state__8k64s.exr")
PRIORITY_EXR = env_path("COMPOSITOR_PRIORITY_EXR", EXR_ROOT / f"{EXR_CASE}__positive_priority_state__8k64s.exr")
EXISTING_EXR = env_path("COMPOSITOR_EXISTING_EXR", EXR_ROOT / f"{EXR_CASE}__existing_condition_positive__8k64s.exr")
EXISTING_TRENDING_EXR = env_path(
    "COMPOSITOR_EXISTING_TRENDING_EXR",
    EXR_ROOT / f"{EXR_CASE}__existing_condition_trending__8k64s.exr",
)
TRENDING_EXR = env_path("COMPOSITOR_TRENDING_EXR", EXR_ROOT / f"{EXR_CASE}__trending_state__8k64s.exr")
BIOENVELOPE_EXR = env_path("COMPOSITOR_BIOENVELOPE_EXR", EXR_ROOT / f"{EXR_CASE}__bioenvelope_positive__8k64s.exr")
BIOENVELOPE_TRENDING_EXR = env_path(
    "COMPOSITOR_BIOENVELOPE_TRENDING_EXR",
    EXR_ROOT / f"{EXR_CASE}__bioenvelope_trending__8k64s.exr",
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
            "COMPOSITOR_CURRENT_SOURCE_BLEND": str(LEGACY_DATA_ROOT / "edge_lab_output_suite_refined.blend"),
            "COMPOSITOR_LEGACY_SOURCE_BLEND": str(LEGACY_SOURCE_BLEND),
            "COMPOSITOR_FINAL_TEMPLATE_BLEND": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_PROPOSAL_PATHWAY_EXR": str(PATHWAY_EXR),
            "COMPOSITOR_PROPOSAL_PRIORITY_EXR": str(PRIORITY_EXR),
            "COMPOSITOR_PROPOSAL_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_current_outputs() -> None:
    current_root = OUTPUT_ROOT / "current"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_core_outputs.py",
        {
            "COMPOSITOR_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_SCENE_NAME": "Current",
            "COMPOSITOR_OUTPUT_DIR": str(current_root),
            "COMPOSITOR_PATHWAY_EXR": str(PATHWAY_EXR),
            "COMPOSITOR_PRIORITY_EXR": str(PRIORITY_EXR),
            "COMPOSITOR_EXISTING_EXR": str(EXISTING_EXR),
            "COMPOSITOR_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_legacy_shading() -> None:
    shading_root = OUTPUT_ROOT / "legacy_shading"
    run_blender_python(
        LEGACY_CODE_ROOT / "render_edge_lab_legacy_shading.py",
        {
            "COMPOSITOR_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_SCENE_NAME": "Legacy",
            "COMPOSITOR_OUTPUT_DIR": str(shading_root),
            "COMPOSITOR_PATHWAY_EXR": str(PATHWAY_EXR),
            "COMPOSITOR_PRIORITY_EXR": str(PRIORITY_EXR),
            "COMPOSITOR_EXISTING_EXR": str(EXISTING_EXR),
        },
    )


def run_current_shading() -> None:
    shading_root = OUTPUT_ROOT / "current" / "shading"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_shading.py",
        {
            "COMPOSITOR_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_SCENE_NAME": "Current",
            "COMPOSITOR_OUTPUT_DIR": str(shading_root),
            "COMPOSITOR_PATHWAY_EXR": str(PATHWAY_EXR),
            "COMPOSITOR_PRIORITY_EXR": str(PRIORITY_EXR),
            "COMPOSITOR_EXISTING_EXR": str(EXISTING_EXR),
            "COMPOSITOR_EXISTING_TRENDING_EXR": str(EXISTING_TRENDING_EXR),
            "COMPOSITOR_BIOENVELOPE_EXR": str(BIOENVELOPE_EXR),
            "COMPOSITOR_BIOENVELOPE_TRENDING_EXR": str(BIOENVELOPE_TRENDING_EXR),
        },
    )


def run_current_bioenvelopes() -> None:
    bio_root = OUTPUT_ROOT / "current" / "bioenvelope"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_bioenvelopes.py",
        {
            "COMPOSITOR_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_SCENE_NAME": "Current",
            "COMPOSITOR_OUTPUT_DIR": str(bio_root),
            "COMPOSITOR_EXISTING_EXR": str(EXISTING_EXR),
            "COMPOSITOR_TRENDING_EXR": str(TRENDING_EXR),
            "COMPOSITOR_BIOENVELOPE_EXR": str(BIOENVELOPE_EXR),
            "COMPOSITOR_BIOENVELOPE_TRENDING_EXR": str(BIOENVELOPE_TRENDING_EXR),
        },
    )


def run_current_mist() -> None:
    mist_root = OUTPUT_ROOT / "current" / "outlines_mist"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_mist.py",
        {
            "COMPOSITOR_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_SCENE_NAME": "Current",
            "COMPOSITOR_OUTPUT_DIR": str(mist_root),
            "COMPOSITOR_PATHWAY_EXR": str(PATHWAY_EXR),
            "COMPOSITOR_PRIORITY_EXR": str(PRIORITY_EXR),
            "COMPOSITOR_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_current_depth_outliner() -> None:
    depth_root = OUTPUT_ROOT / "current" / "depth_outliner"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_depth_outliner.py",
        {
            "COMPOSITOR_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_OUTPUT_DIR": str(depth_root),
            "COMPOSITOR_PATHWAY_EXR": str(PATHWAY_EXR),
            "COMPOSITOR_PRIORITY_EXR": str(PRIORITY_EXR),
            "COMPOSITOR_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_current_base() -> None:
    base_root = OUTPUT_ROOT / "current" / "base"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_base.py",
        {
            "COMPOSITOR_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_SCENE_NAME": "Current",
            "COMPOSITOR_OUTPUT_DIR": str(base_root),
            "COMPOSITOR_EXISTING_EXR": str(EXISTING_EXR),
        },
    )


def run_current_sizes() -> None:
    sizes_root = OUTPUT_ROOT / "current" / "sizes"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_sizes.py",
        {
            "COMPOSITOR_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_SCENE_NAME": "Current",
            "COMPOSITOR_OUTPUT_DIR": str(sizes_root),
            "COMPOSITOR_PATHWAY_EXR": str(PATHWAY_EXR),
            "COMPOSITOR_PRIORITY_EXR": str(PRIORITY_EXR),
            "COMPOSITOR_EXISTING_EXR": str(EXISTING_EXR),
            "COMPOSITOR_TRENDING_EXR": str(TRENDING_EXR),
        },
    )


def run_current_proposals() -> None:
    proposals_root = OUTPUT_ROOT / "current" / "proposals"
    run_blender_python(
        SCRIPT_ROOT / "render_edge_lab_current_proposals.py",
        {
            "COMPOSITOR_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "COMPOSITOR_SCENE_NAME": "Current",
            "COMPOSITOR_OUTPUT_DIR": str(proposals_root),
            "COMPOSITOR_PATHWAY_EXR": str(PATHWAY_EXR),
            "COMPOSITOR_PRIORITY_EXR": str(PRIORITY_EXR),
            "COMPOSITOR_TRENDING_EXR": str(TRENDING_EXR),
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
