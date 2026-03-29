from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
CODE_ROOT = REPO_ROOT / "code" / "blender" / "2026" / "edge_detection_lab"
DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
DEFAULT_BLENDER_BIN = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


BLENDER_BIN = env_path("EDGE_LAB_BLENDER_BIN", DEFAULT_BLENDER_BIN)
EXR_ROOT = env_path("EDGE_LAB_EXR_ROOT", REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6_baseline-city")
OUTPUT_ROOT = env_path(
    "EDGE_LAB_OUTPUT_ROOT",
    DATA_ROOT / "outputs" / f"edge_lab_output_suite_{timestamp_slug()}",
)
TEMPLATE_BLEND = env_path(
    "EDGE_LAB_TEMPLATE_BLEND",
    DATA_ROOT / "city_exr_compositor_lightweight.blend",
)

PATHWAY_EXR = env_path("EDGE_LAB_PATHWAY_EXR", EXR_ROOT / "pathway_state.exr")
PRIORITY_EXR = env_path("EDGE_LAB_PRIORITY_EXR", EXR_ROOT / "priority.exr")
EXISTING_EXR = env_path("EDGE_LAB_EXISTING_EXR", EXR_ROOT / "existing_condition.exr")
BIOENVELOPE_EXR = env_path("EDGE_LAB_BIOENVELOPE_EXR", EXR_ROOT / "bioenvelope.exr")
TRENDING_EXR = env_path("EDGE_LAB_TRENDING_EXR", EXR_ROOT / "trending_state.exr")


def log(message: str) -> None:
    print(f"[run_edge_lab_output_suite] {message}")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def blender_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra)
    return env


def run_blender_python(script_path: Path, extra_env: dict[str, str]) -> None:
    command = [
        str(BLENDER_BIN),
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
    ]
    log(f"Running {script_path.name}")
    subprocess.run(command, check=True, env=blender_env(extra_env))


def maybe_env(path: Path) -> str | None:
    return str(path) if path.exists() else None


def build_suite_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUTPUT_ROOT,
        "blends": OUTPUT_ROOT / "blends",
        "resources": OUTPUT_ROOT / "resources",
        "ao": OUTPUT_ROOT / "ao",
        "normals": OUTPUT_ROOT / "normals",
        "shading": OUTPUT_ROOT / "shading",
        "outlines_mist": OUTPUT_ROOT / "outlines_mist",
        "depth_outliner": OUTPUT_ROOT / "depth_outliner",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def repath_lightweight_blend(dirs: dict[str, Path]) -> Path:
    output_blend = dirs["blends"] / "lightweight_classic_inputs.blend"
    extra_env = {
        "B2026_INPUT_BLEND_PATH": str(TEMPLATE_BLEND),
        "B2026_OUTPUT_BLEND_PATH": str(output_blend),
        "B2026_RUN_TEST": "0",
        "B2026_EXR_PATH__PATHWAY_STATE": str(PATHWAY_EXR),
        "B2026_EXR_PATH__PRIORITY": str(PRIORITY_EXR),
        "B2026_EXR_PATH__EXISTING_CONDITION": str(EXISTING_EXR),
    }
    bio = maybe_env(BIOENVELOPE_EXR)
    if bio:
        extra_env["B2026_EXR_PATH__BIOENVELOPE"] = bio
    trending = maybe_env(TRENDING_EXR)
    if trending:
        extra_env["B2026_EXR_PATH__TRENDING_STATE"] = trending
    run_blender_python(CODE_ROOT / "repath_city_exr_compositor_inputs.py", extra_env)
    return output_blend


def run_resources(dirs: dict[str, Path]) -> None:
    env = {
        "EDGE_LAB_EXR_ROOT": str(EXR_ROOT),
        "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
        "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
        "EDGE_LAB_OUTPUT_DIR": str(dirs["resources"]),
        "EDGE_LAB_BLEND_PATH": str(dirs["blends"] / "resources.blend"),
    }
    trending = maybe_env(TRENDING_EXR)
    if trending:
        env["EDGE_LAB_TRENDING_EXR"] = trending
    run_blender_python(CODE_ROOT / "render_exr_arboreal_resource_fills_v1_blender.py", env)


def run_ao(dirs: dict[str, Path]) -> None:
    env = {
        "EDGE_LAB_EXR_ROOT": str(EXR_ROOT),
        "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
        "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
        "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
        "EDGE_LAB_OUTPUT_DIR": str(dirs["ao"]),
        "EDGE_LAB_BLEND_PATH": str(dirs["blends"] / "ao.blend"),
    }
    run_blender_python(CODE_ROOT / "render_exr_ao_v2_blender.py", env)


def run_normals(dirs: dict[str, Path]) -> None:
    env = {
        "EDGE_LAB_EXR_ROOT": str(EXR_ROOT),
        "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
        "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
        "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
        "EDGE_LAB_OUTPUT_DIR": str(dirs["normals"]),
        "EDGE_LAB_BLEND_PATH": str(dirs["blends"] / "normals.blend"),
    }
    run_blender_python(CODE_ROOT / "render_exr_normals_v2_blender.py", env)


def run_shading(dirs: dict[str, Path], repathed_blend: Path) -> None:
    env = {
        "EDGE_LAB_BLEND_PATH": str(repathed_blend),
        "EDGE_LAB_OUTPUT_DIR": str(dirs["shading"]),
        "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
        "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
        "EDGE_LAB_EXISTING_EXR": str(EXISTING_EXR),
    }
    run_blender_python(CODE_ROOT / "add_named_masks_and_render_baseline_lightweight_ao.py", env)


def run_mist_outlines(dirs: dict[str, Path]) -> None:
    env = {
        "EDGE_LAB_EXR_ROOT": str(EXR_ROOT),
        "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
        "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
        "EDGE_LAB_OUTPUT_DIR": str(dirs["outlines_mist"]),
        "EDGE_LAB_PREP_OUTPUT_DIR": str(dirs["outlines_mist"] / "_prep"),
        "EDGE_LAB_BLEND_PATH": str(dirs["blends"] / "outlines_mist.blend"),
        "EDGE_LAB_VARIANT_PRESET": "kirschsizes",
        "EDGE_LAB_RENDER_MODE": "edges_only",
        "EDGE_LAB_WORKFLOW_ID": "blender_exr_arboreal_mist_kirschsizes_v1",
        "EDGE_LAB_WORKFLOW_NOTES": "Canonical mist-based arboreal outline export.",
    }
    trending = maybe_env(TRENDING_EXR)
    if trending:
        env["EDGE_LAB_TRENDING_EXR"] = trending
    run_blender_python(CODE_ROOT / "render_exr_arboreal_mist_kirsch_sizes_blender.py", env)


def run_depth_outliner(dirs: dict[str, Path]) -> None:
    env = {
        "EDGE_LAB_EXR_ROOT": str(EXR_ROOT),
        "EDGE_LAB_PATHWAY_EXR": str(PATHWAY_EXR),
        "EDGE_LAB_PRIORITY_EXR": str(PRIORITY_EXR),
        "EDGE_LAB_OUTPUT_DIR": str(dirs["depth_outliner"]),
        "EDGE_LAB_BLEND_PATH": str(dirs["blends"] / "depth_outliner.blend"),
    }
    run_blender_python(CODE_ROOT / "render_exr_arboreal_depth_outliner_baseline_blender.py", env)


def main() -> None:
    require_file(BLENDER_BIN, "Blender binary")
    require_file(TEMPLATE_BLEND, "Lightweight compositor template blend")
    require_file(PATHWAY_EXR, "Pathway EXR")
    require_file(PRIORITY_EXR, "Priority EXR")
    require_file(EXISTING_EXR, "Existing condition EXR")

    dirs = build_suite_dirs()
    repathed_blend = repath_lightweight_blend(dirs)

    run_resources(dirs)
    run_ao(dirs)
    run_normals(dirs)
    run_shading(dirs, repathed_blend)
    run_mist_outlines(dirs)
    run_depth_outliner(dirs)

    log(f"Output root: {dirs['root']}")
    log(f"Saved blends: {dirs['blends']}")
    log("Completed resource fills, AO, normals, shading, mist outlines, and depth outliner.")


if __name__ == "__main__":
    main()
