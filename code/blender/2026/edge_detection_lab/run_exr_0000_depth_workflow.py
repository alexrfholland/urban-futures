from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_depth_edge_variants import normalize_field, save_rgba


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"
DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
INPUT_ROOT = DATA_ROOT / "inputs" / "exr_0000_depth_workflow"
OUTPUT_ROOT = DATA_ROOT / "outputs" / "exr_0000_depth_workflow"
CACHE_ROOT = OUTPUT_ROOT / "_cache"
OIIO_TOOL = Path(os.environ.get("OIIO_TOOL", "/opt/local/bin/oiiotool"))
DEPTH_WORKFLOW_SCRIPT = SCRIPT_DIR / "render_depth_edge_variants.py"

TREE_OBJECT_ID = 3

EXR_FILES = {
    "pathway": EXR_ROOT / "city-pathway_state0000.exr",
    "priority": EXR_ROOT / "city-city_priority0000.exr",
    "trending": EXR_ROOT / "city-trending_state0000.exr",
    "existing": EXR_ROOT / "city-existing_condition0000.exr",
}


def log(message: str) -> None:
    print(f"[run_exr_0000_depth_workflow] {message}")


def require_oiio() -> None:
    if not OIIO_TOOL.exists():
        raise FileNotFoundError(f"Could not find oiiotool at {OIIO_TOOL}")


def safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def extract_channels_to_tiff(exr_path: Path, channel_spec: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    target = CACHE_ROOT / f"{exr_path.stem}__{safe_token(channel_spec)}.tif"
    if target.exists():
        return target

    command = [
        str(OIIO_TOOL),
        str(exr_path),
        "--ch",
        channel_spec,
        "-d",
        "float",
        "-o",
        str(target),
    ]
    subprocess.run(command, check=True)
    return target


def load_gray(exr_path: Path, channel_name: str) -> np.ndarray:
    tif_path = extract_channels_to_tiff(exr_path, channel_name)
    image = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {tif_path}")
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.float32)


def load_rgba(exr_path: Path, channel_prefix: str) -> np.ndarray:
    tif_path = extract_channels_to_tiff(
        exr_path,
        f"{channel_prefix}.R,{channel_prefix}.G,{channel_prefix}.B,{channel_prefix}.A",
    )
    image = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {tif_path}")
    rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return rgba.astype(np.float32)


def mask_for_object_id(index_values: np.ndarray, object_id: int) -> np.ndarray:
    return (np.abs(index_values - float(object_id)) < 0.25).astype(np.float32)


def write_depth_input(path: Path, depth_values: np.ndarray, mask: np.ndarray) -> None:
    rgba = np.zeros((depth_values.shape[0], depth_values.shape[1], 4), dtype=np.float32)
    normalized = normalize_field(depth_values, mask)
    rgba[..., 0] = normalized
    rgba[..., 1] = normalized
    rgba[..., 2] = normalized
    rgba[..., 3] = np.clip(mask, 0.0, 1.0)
    save_rgba(rgba, path)


def write_input_set(scene_name: str, base_rgba: np.ndarray, world_rgba: np.ndarray, depth_values: np.ndarray, mask: np.ndarray) -> dict[str, Path]:
    scene_dir = INPUT_ROOT / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    base_path = scene_dir / "base.png"
    world_path = scene_dir / "world.png"
    depth_path = scene_dir / "depth.png"

    save_rgba(base_rgba, base_path)
    save_rgba(world_rgba, world_path)
    write_depth_input(depth_path, depth_values, mask)
    log(f"Wrote prepared inputs for {scene_name} in {scene_dir}")

    return {
        "base": base_path,
        "world": world_path,
        "depth": depth_path,
    }


def run_depth_workflow(scene_name: str, input_paths: dict[str, Path]) -> Path:
    output_tag = f"exr_0000_depth_workflow/{scene_name}"
    env = os.environ.copy()
    env["EDGE_LAB_BASE_PATH"] = str(input_paths["base"])
    env["EDGE_LAB_WORLD_PATH"] = str(input_paths["world"])
    env["EDGE_LAB_DEPTH_PATH"] = str(input_paths["depth"])
    env["EDGE_LAB_OUTPUT_TAG"] = output_tag

    command = [sys.executable, str(DEPTH_WORKFLOW_SCRIPT)]
    subprocess.run(command, env=env, check=True, cwd=str(REPO_ROOT))
    return DATA_ROOT / "outputs" / output_tag


def main() -> None:
    require_oiio()
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    existing_rgba = load_rgba(EXR_FILES["existing"], "Image")

    pathway_rgba = load_rgba(EXR_FILES["pathway"], "Image")
    pathway_depth = load_gray(EXR_FILES["pathway"], "Depth.V")
    pathway_index = load_gray(EXR_FILES["pathway"], "IndexOB.V")
    pathway_alpha = load_gray(EXR_FILES["pathway"], "Alpha.V")
    pathway_visible_mask = np.clip(mask_for_object_id(pathway_index, TREE_OBJECT_ID) * pathway_alpha, 0.0, 1.0)

    priority_rgba = load_rgba(EXR_FILES["priority"], "Image")
    priority_depth = load_gray(EXR_FILES["priority"], "Depth.V")
    priority_index = load_gray(EXR_FILES["priority"], "IndexOB.V")
    priority_alpha = load_gray(EXR_FILES["priority"], "Alpha.V")
    priority_all_mask = np.clip(mask_for_object_id(priority_index, TREE_OBJECT_ID) * priority_alpha, 0.0, 1.0)
    priority_visible_mask = np.clip(priority_all_mask * pathway_visible_mask, 0.0, 1.0)

    trending_rgba = load_rgba(EXR_FILES["trending"], "Image")
    trending_depth = load_gray(EXR_FILES["trending"], "Depth.V")
    trending_index = load_gray(EXR_FILES["trending"], "IndexOB.V")
    trending_alpha = load_gray(EXR_FILES["trending"], "Alpha.V")
    trending_visible_mask = np.clip(mask_for_object_id(trending_index, TREE_OBJECT_ID) * trending_alpha, 0.0, 1.0)

    scene_specs = {
        "pathway": {
            "world": pathway_rgba,
            "depth": pathway_depth,
            "mask": pathway_visible_mask,
            "source_exr": str(EXR_FILES["pathway"]),
        },
        "priority": {
            "world": priority_rgba,
            "depth": priority_depth,
            "mask": priority_visible_mask,
            "source_exr": str(EXR_FILES["priority"]),
        },
        "trending": {
            "world": trending_rgba,
            "depth": trending_depth,
            "mask": trending_visible_mask,
            "source_exr": str(EXR_FILES["trending"]),
        },
    }

    manifest: dict[str, object] = {
        "workflow_id": "png_depth_tuned_v1",
        "canonical_workflow_script": str(DEPTH_WORKFLOW_SCRIPT),
        "adapter_script": str(Path(__file__).resolve()),
        "notes": "Prepared masked normalized depth PNGs from updated 0000 EXRs, then ran render_depth_edge_variants.py unchanged apart from env-path overrides.",
        "scenes": {},
    }

    for scene_name, spec in scene_specs.items():
        input_paths = write_input_set(
            scene_name,
            base_rgba=existing_rgba,
            world_rgba=spec["world"],
            depth_values=spec["depth"],
            mask=spec["mask"],
        )
        output_dir = run_depth_workflow(scene_name, input_paths)
        manifest["scenes"][scene_name] = {
            "source_exr": spec["source_exr"],
            "input_paths": {key: str(value) for key, value in input_paths.items()},
            "output_dir": str(output_dir),
            "mask_coverage_over_0_5": float((spec["mask"] > 0.5).mean()),
        }
        log(f"Rendered {scene_name} via render_depth_edge_variants.py -> {output_dir}")

    manifest_path = OUTPUT_ROOT / "workflow_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
