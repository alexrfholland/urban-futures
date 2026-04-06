from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
CODE_ROOT = REPO_ROOT / "code" / "blender" / "2026" / "edge_detection_lab"
DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
BLENDER_BIN = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


FINAL_TEMPLATE_BLEND = env_path(
    "EDGE_LAB_FINAL_TEMPLATE_BLEND",
    DATA_ROOT / "edge_lab_final_template.blend",
)
CURRENT_SOURCE_BLEND = env_path(
    "EDGE_LAB_CURRENT_SOURCE_BLEND",
    DATA_ROOT / "edge_lab_output_suite_combined.blend",
)
LEGACY_SOURCE_BLEND = env_path(
    "EDGE_LAB_LEGACY_SOURCE_BLEND",
    DATA_ROOT / "city_exr_compositor_lightweight_city_final.blend",
)
OUTPUT_ROOT = env_path(
    "EDGE_LAB_OUTPUT_ROOT",
    DATA_ROOT / "outputs" / "edge_lab_final_template_city_20260329",
)
EXR_ROOT = env_path(
    "EDGE_LAB_EXR_ROOT",
    REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city",
)

PATHWAY_EXR = env_path("EDGE_LAB_PATHWAY_EXR", EXR_ROOT / "city-pathway_state.exr")
PRIORITY_EXR = env_path("EDGE_LAB_PRIORITY_EXR", EXR_ROOT / "city-city_priority.exr")
EXISTING_EXR = env_path("EDGE_LAB_EXISTING_EXR", EXR_ROOT / "city-existing_condition.exr")
EXISTING_TRENDING_EXR = env_path(
    "EDGE_LAB_EXISTING_TRENDING_EXR",
    EXISTING_EXR,
)
TRENDING_EXR = env_path("EDGE_LAB_TRENDING_EXR", EXR_ROOT / "city-trending_state.exr")
BIOENVELOPE_EXR = env_path("EDGE_LAB_BIOENVELOPE_EXR", EXR_ROOT / "city-city_bioenvelope.exr")
BIOENVELOPE_TRENDING_EXR = env_path(
    "EDGE_LAB_BIOENVELOPE_TRENDING_EXR",
    TRENDING_EXR,
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
        CODE_ROOT / "build_edge_lab_final_template_blend.py",
        {
            "EDGE_LAB_CURRENT_SOURCE_BLEND": str(DATA_ROOT / "edge_lab_output_suite_refined.blend"),
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
        CODE_ROOT / "render_edge_lab_current_core_outputs.py",
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
        CODE_ROOT / "render_edge_lab_legacy_shading.py",
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
        CODE_ROOT / "render_edge_lab_legacy_shading.py",
        {
            "EDGE_LAB_BLEND_PATH": str(FINAL_TEMPLATE_BLEND),
            "EDGE_LAB_SCENE_NAME": "Current",
            "EDGE_LAB_NODE_PREFIX": "Current Shading :: ",
            "EDGE_LAB_OUTPUT_DIR": str(shading_root),
            "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
            "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
            "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
            "EDGE_LAB_EXISTING_TRENDING_EXR": str(EXISTING_TRENDING_EXR),
            "EDGE_LAB_BIOENVELOPE_EXR": str(BIOENVELOPE_EXR),
            "EDGE_LAB_BIOENVELOPE_TRENDING_EXR": str(BIOENVELOPE_TRENDING_EXR),
            "EDGE_LAB_PATHWAY_NODE_CANDIDATES": "Current Shading :: EXR Pathway|AO::EXR Pathway",
            "EDGE_LAB_PRIORITY_NODE_CANDIDATES": "Current Shading :: EXR Priority|AO::EXR Priority",
            "EDGE_LAB_EXISTING_NODE_CANDIDATES": "Current Shading :: EXR Existing|AO::EXR Existing",
        },
    )


def run_current_bioenvelopes() -> None:
    bio_root = OUTPUT_ROOT / "current" / "bioenvelope"
    run_blender_python(
        CODE_ROOT / "render_edge_lab_current_bioenvelopes.py",
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
        CODE_ROOT / "render_edge_lab_current_mist.py",
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
        CODE_ROOT / "render_edge_lab_current_depth_outliner.py",
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
        CODE_ROOT / "render_edge_lab_current_base.py",
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
        CODE_ROOT / "render_edge_lab_current_sizes.py",
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
        CODE_ROOT / "render_edge_lab_current_proposals.py",
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
