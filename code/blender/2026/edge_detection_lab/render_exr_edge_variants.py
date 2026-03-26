from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from render_depth_edge_variants import (
    BACKGROUND_FILL,
    VARIANTS,
    WIDTH_MODES,
    adapt_variant_for_mask_density,
    adapt_width_mode_for_mask_density,
    composite_variant,
    detect_edges,
    save_rgba,
    write_contact_sheet,
    apply_variable_width,
)


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
EXR_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "2026 futures heroes6-city"
OUTPUT_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "exr_city"
CACHE_ROOT = OUTPUT_ROOT / "_cache"
OIIO_TOOL = Path(os.environ.get("OIIO_TOOL", "/opt/local/bin/oiiotool"))

TREE_OBJECT_ID = 3
BASE_OBJECT_IDS = (1, 2)

EXR_FILES = {
    "pathway": EXR_ROOT / "city-pathway_state.exr",
    "priority": EXR_ROOT / "city-city_priority.exr",
    "trending": EXR_ROOT / "city-trending_state.exr",
    "existing": EXR_ROOT / "city-existing_condition.exr",
    "bioenvelope": EXR_ROOT / "city-city_bioenvelope.exr",
}


def log(message: str) -> None:
    print(f"[render_exr_edge_variants] {message}")


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


def load_rgb(exr_path: Path, channel_prefix: str) -> np.ndarray:
    tif_path = extract_channels_to_tiff(
        exr_path,
        f"{channel_prefix}.R,{channel_prefix}.G,{channel_prefix}.B",
    )
    image = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {tif_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)


def mask_for_object_ids(index_values: np.ndarray, object_ids: tuple[int, ...] | list[int] | set[int] | int) -> np.ndarray:
    if isinstance(object_ids, int):
        object_ids = (object_ids,)
    mask = np.zeros_like(index_values, dtype=np.float32)
    for object_id in object_ids:
        mask = np.maximum(mask, (np.abs(index_values - float(object_id)) < 0.25).astype(np.float32))
    return mask


def extract_masked_rgba(image_rgba: np.ndarray, mask: np.ndarray, blur_sigma: float = 1.1, alpha_gain: float = 1.0) -> np.ndarray:
    softened = ndimage.gaussian_filter(mask.astype(np.float32), sigma=blur_sigma, mode="nearest")
    alpha = np.clip(softened * alpha_gain, 0.0, 1.0)
    rgba = np.zeros_like(image_rgba)
    rgba[..., :3] = image_rgba[..., :3]
    rgba[..., 3] = alpha
    return rgba


def make_white_canvas(shape: tuple[int, int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.float32)
    canvas[..., :3] = BACKGROUND_FILL
    canvas[..., 3] = 1.0
    return canvas


def save_white_preview(rgb: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    mask_rgb = np.clip(mask[..., None], 0.0, 1.0)
    preview = np.ones((*mask.shape, 4), dtype=np.float32)
    preview[..., :3] = 1.0 - mask_rgb + rgb * mask_rgb
    preview[..., 3] = 1.0
    save_rgba(preview, output_path)


def run_edge_suite(
    signal_values: np.ndarray,
    mask: np.ndarray,
    base_rgba: np.ndarray,
    overlay_rgba: np.ndarray,
    output_dir: Path,
    prefix: str,
    signal_label: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    white = make_white_canvas(base_rgba.shape)
    zero_edge = np.zeros(mask.shape, dtype=np.float32)
    base_on_white = composite_variant(white, base_rgba, zero_edge)
    rendered_paths: list[Path] = []
    summary: dict[str, object] = {
        "prefix": prefix,
        "signal": signal_label,
        "mask_coverage_over_0_5": float((mask > 0.5).mean()),
        "outputs": {},
    }

    for variant in VARIANTS:
        tuned_variant = adapt_variant_for_mask_density(variant, summary["mask_coverage_over_0_5"])
        normalized_field, thresholded, edge_strength = detect_edges(signal_values, tuned_variant, mask)

        for width_mode in WIDTH_MODES:
            tuned_width = adapt_width_mode_for_mask_density(width_mode, summary["mask_coverage_over_0_5"])
            output_name = f"{prefix}_{variant['name']}_{width_mode['name']}"
            edge_alpha = apply_variable_width(
                normalized_field,
                thresholded,
                edge_strength,
                mask,
                tuned_width,
            )
            result_rgba = composite_variant(base_on_white, overlay_rgba, edge_alpha)

            edge_layer = np.zeros_like(base_rgba)
            edge_layer[..., :3] = np.array([34.0, 18.0, 59.0], dtype=np.float32) / 255.0
            edge_layer[..., 3] = edge_alpha

            edge_path = output_dir / f"{output_name}_edges.png"
            result_path = output_dir / f"{output_name}_composite.png"
            save_rgba(edge_layer, edge_path)
            save_rgba(result_rgba, result_path)
            rendered_paths.extend([edge_path, result_path])

            summary["outputs"][output_name] = {
                "detector": variant["detector"],
                "width_mode": width_mode["name"],
                "edge_coverage_over_0_05": float((edge_alpha > 0.05).mean()),
                "edge_alpha_mean": float(edge_alpha.mean()),
                "edge_layer": str(edge_path),
                "composite": str(result_path),
            }
            log(f"Wrote {result_path}")

    contact_sheet_path = output_dir / f"{prefix}_{signal_label}_contact_sheet.png"
    write_contact_sheet([path for path in rendered_paths if path.name.endswith("_composite.png")], contact_sheet_path)
    summary["contact_sheet"] = str(contact_sheet_path)

    summary_path = output_dir / f"{prefix}_{signal_label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Wrote {summary_path}")
    return summary


def write_ao_pngs() -> dict[str, str]:
    output_dir = OUTPUT_ROOT / "01_ao"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    specs = (
        ("pathway_tree_ao", EXR_FILES["pathway"]),
        ("priority_tree_ao", EXR_FILES["priority"]),
        ("trending_visible_tree_ao", EXR_FILES["trending"]),
    )

    for label, exr_path in specs:
        ao_rgb = load_rgb(exr_path, "AO")
        index_values = load_gray(exr_path, "IndexOB.V")
        tree_mask = mask_for_object_ids(index_values, TREE_OBJECT_ID)
        output_path = output_dir / f"{label}.png"
        save_white_preview(ao_rgb, tree_mask, output_path)
        results[label] = str(output_path)
        log(f"Wrote {output_path}")

    return results


def run_trending_tree_edges() -> dict:
    output_dir = OUTPUT_ROOT / "02_trending_trees"
    output_dir.mkdir(parents=True, exist_ok=True)

    trending_rgba = load_rgba(EXR_FILES["trending"], "Image")
    trending_depth = load_gray(EXR_FILES["trending"], "Depth.V")
    trending_index = load_gray(EXR_FILES["trending"], "IndexOB.V")
    existing_rgba = load_rgba(EXR_FILES["existing"], "Image")

    visible_tree_mask = mask_for_object_ids(trending_index, TREE_OBJECT_ID)
    trending_tree_rgba = extract_masked_rgba(trending_rgba, visible_tree_mask, blur_sigma=1.0, alpha_gain=0.95)
    tree_preview = output_dir / "trending_visible_trees.png"
    save_rgba(trending_tree_rgba, tree_preview)
    log(f"Wrote {tree_preview}")

    return run_edge_suite(
        signal_values=trending_depth,
        mask=visible_tree_mask,
        base_rgba=existing_rgba,
        overlay_rgba=trending_tree_rgba,
        output_dir=output_dir,
        prefix="trending_trees",
        signal_label="depth",
    )


def run_base_layer_edges() -> dict[str, dict]:
    output_dir = OUTPUT_ROOT / "03_base_layer"
    output_dir.mkdir(parents=True, exist_ok=True)

    pathway_rgba = load_rgba(EXR_FILES["pathway"], "Image")
    pathway_index = load_gray(EXR_FILES["pathway"], "IndexOB.V")
    pathway_depth = load_gray(EXR_FILES["pathway"], "Depth.V")
    pathway_mist = load_gray(EXR_FILES["pathway"], "Mist.V")

    base_mask = mask_for_object_ids(pathway_index, BASE_OBJECT_IDS)
    base_rgba = extract_masked_rgba(pathway_rgba, base_mask, blur_sigma=0.8, alpha_gain=1.0)
    base_preview = output_dir / "base_visible_layer.png"
    save_rgba(base_rgba, base_preview)
    log(f"Wrote {base_preview}")

    empty_overlay = np.zeros_like(base_rgba)
    depth_summary = run_edge_suite(
        signal_values=pathway_depth,
        mask=base_mask,
        base_rgba=base_rgba,
        overlay_rgba=empty_overlay,
        output_dir=output_dir / "depth",
        prefix="base_layer",
        signal_label="depth",
    )
    mist_summary = run_edge_suite(
        signal_values=pathway_mist,
        mask=base_mask,
        base_rgba=base_rgba,
        overlay_rgba=empty_overlay,
        output_dir=output_dir / "mist",
        prefix="base_layer",
        signal_label="mist",
    )
    return {
        "depth": depth_summary,
        "mist": mist_summary,
    }


def main() -> None:
    require_oiio()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("ao", "trending", "base", "all"),
        default="all",
    )
    args = parser.parse_args()

    if args.task in ("ao", "all"):
        write_ao_pngs()
    if args.task in ("trending", "all"):
        run_trending_tree_edges()
    if args.task in ("base", "all"):
        run_base_layer_edges()


if __name__ == "__main__":
    main()
