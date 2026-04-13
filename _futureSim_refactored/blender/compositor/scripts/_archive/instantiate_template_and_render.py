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
LEGACY_INPUT_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
BLENDER_BIN = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def default_dataset_root() -> Path:
    sim_root = os.environ.get("COMPOSITOR_SIM_ROOT", "").strip()
    exr_family = os.environ.get("COMPOSITOR_EXR_FAMILY", "").strip()
    if sim_root and exr_family:
        return blenderv2_exr_family_dir(sim_root, exr_family)
    return (
        LEGACY_INPUT_ROOT
        / "inputs"
        / "LATEST_REMOTE_EXRS"
        / "simv3-7_20260405_8k64s_simv3-7"
        / "city_timeline"
    )


def default_output_root(default_family: str) -> Path:
    sim_root = os.environ.get("COMPOSITOR_SIM_ROOT", "").strip()
    exr_family = os.environ.get("COMPOSITOR_EXR_FAMILY", "").strip()
    compositor_family = os.environ.get("COMPOSITOR_FAMILY", default_family).strip()
    timestamp = os.environ.get("COMPOSITOR_RUN_TIMESTAMP", "").strip() or time.strftime("%Y%m%d_%H%M")
    note = os.environ.get("COMPOSITOR_RUN_NOTE", "").strip()
    if sim_root and exr_family and compositor_family:
        compositor_run = compositor_run_name(compositor_family, timestamp, note or None)
        return compositor_run_dir(sim_root, exr_family, compositor_run)
    return DATA_ROOT / "outputs" / "edge_lab_template_instantiation"


CANONICAL_BLEND = env_path(
    "COMPOSITOR_CANONICAL_BLEND",
    CANONICAL_ROOT / "edge_lab_final_template_safe_rebuild_20260405.blend",
)
DATASET_ROOT = default_dataset_root()
OUTPUT_ROOT = default_output_root("edge-lab-template-instantiation")
WORKING_BLEND = env_path(
    "COMPOSITOR_WORKING_BLEND",
    OUTPUT_ROOT / "_working" / CANONICAL_BLEND.name,
)
FAMILY_FILTER = {
    item.strip().lower()
    for item in os.environ.get("COMPOSITOR_RENDER_FAMILIES", "ao,bioenvelope,base").split(",")
    if item.strip()
}


def log(message: str) -> None:
    print(f"[instantiate_template_and_render] {message}")


def run_blender_python(script_path: Path, env: dict[str, str]) -> None:
    command = [
        str(BLENDER_BIN),
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
    ]
    subprocess.run(command, check=True, env={**os.environ, **env})


def dataset_name(dataset_root: Path) -> str:
    exr_family = os.environ.get("COMPOSITOR_EXR_FAMILY", "").strip()
    if exr_family:
        return exr_case_from_family(exr_family)
    return dataset_root.name


def resolve_dataset_paths(dataset_root: Path) -> dict[str, Path]:
    stem = dataset_name(dataset_root)
    return {
        "PATHWAY_EXR": dataset_root / f"{stem}__positive_state__8k64s.exr",
        "PRIORITY_EXR": dataset_root / f"{stem}__positive_priority_state__8k64s.exr",
        "EXISTING_EXR": dataset_root / f"{stem}__existing_condition_positive__8k64s.exr",
        "EXISTING_TRENDING_EXR": dataset_root / f"{stem}__existing_condition_trending__8k64s.exr",
        "TRENDING_EXR": dataset_root / f"{stem}__trending_state__8k64s.exr",
        "BIOENVELOPE_EXR": dataset_root / f"{stem}__bioenvelope_positive__8k64s.exr",
        "BIOENVELOPE_TRENDING_EXR": dataset_root / f"{stem}__bioenvelope_trending__8k64s.exr",
    }


def require_files(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")


def instantiate_working_blend() -> None:
    WORKING_BLEND.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CANONICAL_BLEND, WORKING_BLEND)
    blend1 = CANONICAL_BLEND.with_suffix(".blend1")
    if blend1.exists():
        shutil.copy2(blend1, WORKING_BLEND.with_suffix(".blend1"))
    log(f"Instantiated {CANONICAL_BLEND} -> {WORKING_BLEND}")


def family_env(base_env: dict[str, str], output_dir: Path) -> dict[str, str]:
    return {
        **base_env,
        "COMPOSITOR_BLEND_PATH": str(WORKING_BLEND),
        "COMPOSITOR_SCENE_NAME": "Current",
        "COMPOSITOR_OUTPUT_DIR": str(output_dir),
    }


def run_selected_families(paths: dict[str, Path]) -> None:
    current_root = OUTPUT_ROOT / "current"
    common_env = {
        "COMPOSITOR_PATHWAY_EXR": str(paths["PATHWAY_EXR"]),
        "COMPOSITOR_PRIORITY_EXR": str(paths["PRIORITY_EXR"]),
        "COMPOSITOR_EXISTING_EXR": str(paths["EXISTING_EXR"]),
        "COMPOSITOR_EXISTING_TRENDING_EXR": str(paths["EXISTING_TRENDING_EXR"]),
        "COMPOSITOR_TRENDING_EXR": str(paths["TRENDING_EXR"]),
        "COMPOSITOR_BIOENVELOPE_EXR": str(paths["BIOENVELOPE_EXR"]),
        "COMPOSITOR_BIOENVELOPE_TRENDING_EXR": str(paths["BIOENVELOPE_TRENDING_EXR"]),
    }

    core_requested = FAMILY_FILTER & {"ao", "normals", "resources"}
    if core_requested:
        env = family_env(common_env, current_root)
        env["COMPOSITOR_RENDER_FAMILIES"] = ",".join(sorted(core_requested))
        run_blender_python(SCRIPT_ROOT / "render_edge_lab_current_core_outputs.py", env)

    if "shading" in FAMILY_FILTER:
        run_blender_python(
            SCRIPT_ROOT / "render_edge_lab_current_shading.py",
            {
                **family_env(common_env, current_root / "shading"),
            },
        )

    if "bioenvelope" in FAMILY_FILTER:
        run_blender_python(
            SCRIPT_ROOT / "render_edge_lab_current_bioenvelopes.py",
            family_env(common_env, current_root / "bioenvelope"),
        )

    if "sizes" in FAMILY_FILTER:
        run_blender_python(
            SCRIPT_ROOT / "render_edge_lab_current_sizes.py",
            family_env(common_env, current_root / "sizes"),
        )

    if "base" in FAMILY_FILTER:
        run_blender_python(
            SCRIPT_ROOT / "render_edge_lab_current_base.py",
            family_env(common_env, current_root / "base"),
        )


def prune_runtime_artifacts(output_root: Path) -> None:
    for path in output_root.rglob("_discard_render.png"):
        path.unlink()
        log(f"Removed runtime artifact {path}")


def main() -> None:
    if not CANONICAL_BLEND.exists():
        raise FileNotFoundError(CANONICAL_BLEND)
    paths = resolve_dataset_paths(DATASET_ROOT)
    require_files(paths)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    instantiate_working_blend()
    run_selected_families(paths)
    prune_runtime_artifacts(OUTPUT_ROOT)
    log(f"Dataset root: {DATASET_ROOT}")
    log(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
