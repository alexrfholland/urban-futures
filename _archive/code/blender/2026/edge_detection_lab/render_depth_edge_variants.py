from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DATA_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab"
INPUT_DIR = DATA_ROOT / "inputs"
OUTPUT_DIR = DATA_ROOT / "outputs"

BASE_PATH = INPUT_DIR / "base.png"
WORLD_PATH = INPUT_DIR / "test-world.png"
DEPTH_PATH = INPUT_DIR / "test-depth.png"

TREE_SHADING_ALPHA_GAIN = 0.82
TREE_SHADING_BLUR_SIGMA = 1.2
EDGE_COLOR_HEX = "#22123B"
EDGE_COLOR = np.array([34.0, 18.0, 59.0], dtype=np.float32) / 255.0
BACKGROUND_FILL = np.array([1.0, 1.0, 1.0], dtype=np.float32)

VARIANTS = (
    {
        "name": "depth_kirsch",
        "detector": "kirsch",
        "blur_sigma": 0.9,
        "mask_sigma": 1.0,
        "threshold_low": 0.38,
        "threshold_high": 0.67,
        "edge_gain": 0.80,
        "notes": "Strong fine branch definition without filling the whole canopy.",
    },
    {
        "name": "depth_sobel",
        "detector": "sobel",
        "blur_sigma": 1.1,
        "mask_sigma": 1.1,
        "threshold_low": 0.34,
        "threshold_high": 0.64,
        "edge_gain": 0.72,
        "notes": "Clean and restrained general-purpose depth silhouette edges.",
    },
    {
        "name": "depth_laplace",
        "detector": "laplace",
        "blur_sigma": 1.3,
        "mask_sigma": 1.3,
        "threshold_low": 0.52,
        "threshold_high": 0.82,
        "edge_gain": 0.62,
        "notes": "Sparse structural outlines only; lowest clutter but least detail.",
    },
    {
        "name": "depth_hybrid",
        "detector": "hybrid",
        "blur_sigma": 1.0,
        "mask_sigma": 1.0,
        "threshold_low": 0.42,
        "threshold_high": 0.70,
        "edge_gain": 0.76,
        "notes": "Consensus of Kirsch and Sobel, tuned to keep readable edge density.",
    },
)

WIDTH_MODES = (
    {
        "name": "thin",
        "core_dilation": 2,
        "mid_dilation": 3,
        "wide_dilation": 5,
        "presence_low": 0.08,
        "presence_high": 0.32,
        "mid_weight": 0.44,
        "wide_weight": 0.20,
        "mid_low": 0.54,
        "mid_high": 0.84,
        "wide_low": 0.74,
        "wide_high": 0.96,
        "final_sigma": 0.40,
        "opacity": 0.86,
        "min_component_pixels": 10,
        "solid_threshold": 0.22,
        "coverage_notes": "Narrow variable-width outline.",
    },
    {
        "name": "regular",
        "core_dilation": 3,
        "mid_dilation": 5,
        "wide_dilation": 7,
        "presence_low": 0.08,
        "presence_high": 0.30,
        "mid_weight": 0.52,
        "wide_weight": 0.28,
        "mid_low": 0.48,
        "mid_high": 0.82,
        "wide_low": 0.66,
        "wide_high": 0.94,
        "final_sigma": 0.58,
        "opacity": 0.90,
        "min_component_pixels": 18,
        "solid_threshold": 0.18,
        "coverage_notes": "Broader variable-width outline.",
    },
)


def log(message: str) -> None:
    print(f"[edge_depth_variants] {message}")


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


def load_rgba(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgba(array: np.ndarray, path: Path) -> None:
    clipped = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(clipped, mode="RGBA").save(path)


def smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    span = max(high - low, 1e-6)
    x = np.clip((values - low) / span, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def normalize_field(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    active = field[mask > 0.03]
    if active.size == 0:
        return np.zeros_like(field)
    lower = np.percentile(active, 5.0)
    upper = np.percentile(active, 99.5)
    if upper <= lower:
        upper = lower + 1e-6
    return np.clip((field - lower) / (upper - lower), 0.0, 1.0)


def detect_sobel(values: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(values, axis=1, mode="nearest")
    gy = ndimage.sobel(values, axis=0, mode="nearest")
    return np.hypot(gx, gy)


def detect_laplace(values: np.ndarray) -> np.ndarray:
    return np.abs(ndimage.laplace(values, mode="nearest"))


def detect_kirsch(values: np.ndarray) -> np.ndarray:
    kernels = (
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),
    )
    responses = [np.abs(ndimage.convolve(values, kernel, mode="nearest")) for kernel in kernels]
    return np.max(np.stack(responses, axis=0), axis=0)


def detect_edges(depth_values: np.ndarray, variant: dict, mask: np.ndarray):
    blurred = ndimage.gaussian_filter(depth_values, sigma=variant["blur_sigma"], mode="nearest")

    if variant["detector"] == "sobel":
        field = detect_sobel(blurred)
    elif variant["detector"] == "laplace":
        field = detect_laplace(blurred)
    elif variant["detector"] == "kirsch":
        field = detect_kirsch(blurred)
    elif variant["detector"] == "hybrid":
        sobel = normalize_field(detect_sobel(blurred), mask)
        kirsch = normalize_field(detect_kirsch(blurred), mask)
        field = np.maximum(kirsch * 0.65, sobel * 0.85)
    else:
        raise ValueError(f"Unknown detector: {variant['detector']}")

    normalized = normalize_field(field, mask)
    thresholded = smoothstep(variant["threshold_low"], variant["threshold_high"], normalized)
    softened_mask = ndimage.gaussian_filter(mask, sigma=variant["mask_sigma"], mode="nearest")
    edge_strength = np.clip(thresholded * softened_mask * variant["edge_gain"], 0.0, 1.0)
    return normalized, thresholded, edge_strength


def adapt_variant_for_mask_density(variant: dict, mask_coverage: float) -> dict:
    tuned = dict(variant)
    if mask_coverage <= 0.04:
        return tuned

    density_factor = min(max((mask_coverage - 0.04) / 0.04, 0.0), 1.0)
    tuned["threshold_low"] = min(0.92, tuned["threshold_low"] + 0.06 * density_factor)
    tuned["threshold_high"] = min(0.98, tuned["threshold_high"] + 0.08 * density_factor)
    tuned["edge_gain"] = max(0.52, tuned["edge_gain"] - 0.08 * density_factor)
    tuned["mask_sigma"] = max(0.8, tuned["mask_sigma"] - 0.15 * density_factor)
    return tuned


def adapt_width_mode_for_mask_density(width_mode: dict, mask_coverage: float) -> dict:
    tuned = dict(width_mode)
    if mask_coverage <= 0.04:
        return tuned

    density_factor = min(max((mask_coverage - 0.04) / 0.04, 0.0), 1.0)
    tuned["presence_low"] = min(0.22, tuned["presence_low"] + 0.05 * density_factor)
    tuned["presence_high"] = min(0.42, tuned["presence_high"] + 0.06 * density_factor)
    tuned["mid_weight"] *= 1.0 - 0.16 * density_factor
    tuned["wide_weight"] *= 1.0 - 0.24 * density_factor
    tuned["opacity"] = min(0.92, tuned["opacity"] + 0.02 * density_factor)
    tuned["min_component_pixels"] = int(round(tuned["min_component_pixels"] * (1.0 + 0.7 * density_factor)))
    return tuned


def remove_small_components(coverage: np.ndarray, min_pixels: int) -> np.ndarray:
    binary = coverage > 0.14
    labeled, num_labels = ndimage.label(binary)
    if num_labels == 0:
        return coverage
    counts = np.bincount(labeled.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return coverage * keep[labeled]


def apply_variable_width(
    normalized_field: np.ndarray,
    thresholded: np.ndarray,
    edge_strength: np.ndarray,
    mask: np.ndarray,
    width_mode: dict,
) -> np.ndarray:
    hard_mask = mask > 0.5
    presence = smoothstep(
        width_mode["presence_low"],
        width_mode["presence_high"],
        thresholded,
    ) * hard_mask
    core = ndimage.grey_dilation(
        presence,
        size=(width_mode["core_dilation"], width_mode["core_dilation"]),
    )
    mid_source = presence * smoothstep(
        width_mode["mid_low"],
        width_mode["mid_high"],
        normalized_field,
    )
    wide_source = presence * smoothstep(
        width_mode["wide_low"],
        width_mode["wide_high"],
        normalized_field,
    )
    mid = ndimage.grey_dilation(
        mid_source,
        size=(width_mode["mid_dilation"], width_mode["mid_dilation"]),
    ) * width_mode["mid_weight"]
    wide = ndimage.grey_dilation(
        wide_source,
        size=(width_mode["wide_dilation"], width_mode["wide_dilation"]),
    ) * width_mode["wide_weight"]
    support = np.maximum.reduce((core, mid, wide))
    support = remove_small_components(support, width_mode["min_component_pixels"])
    solid_support = support > width_mode["solid_threshold"]
    support = ndimage.gaussian_filter(
        solid_support.astype(np.float32),
        sigma=width_mode["final_sigma"],
        mode="nearest",
    )
    support = np.clip(support * 1.15, 0.0, 1.0)
    support[solid_support] = 1.0
    variable_alpha = np.clip(support * width_mode["opacity"], 0.0, 1.0)
    variable_alpha *= hard_mask
    return np.clip(variable_alpha, 0.0, 1.0)


def alpha_over(bottom_rgb: np.ndarray, bottom_alpha: np.ndarray, top_rgb: np.ndarray, top_alpha: np.ndarray):
    out_alpha = top_alpha + bottom_alpha * (1.0 - top_alpha)
    safe = np.where(out_alpha > 1e-6, out_alpha, 1.0)
    out_rgb = (top_rgb * top_alpha[..., None] + bottom_rgb * bottom_alpha[..., None] * (1.0 - top_alpha[..., None])) / safe[..., None]
    return out_rgb, out_alpha


def extract_tree_shading(world_rgba: np.ndarray, depth_alpha: np.ndarray) -> np.ndarray:
    mask = ndimage.gaussian_filter(depth_alpha, sigma=TREE_SHADING_BLUR_SIGMA, mode="nearest")
    alpha = np.clip(mask * TREE_SHADING_ALPHA_GAIN, 0.0, 1.0)
    rgb = world_rgba[..., :3]
    rgba = np.zeros_like(world_rgba)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return rgba


def composite_variant(base_rgba: np.ndarray, tree_rgba: np.ndarray, edge_alpha: np.ndarray) -> np.ndarray:
    base_rgb = base_rgba[..., :3]
    base_alpha = base_rgba[..., 3]
    tree_rgb = tree_rgba[..., :3]
    tree_alpha = tree_rgba[..., 3]

    shaded_rgb, shaded_alpha = alpha_over(base_rgb, base_alpha, tree_rgb, tree_alpha)
    edge_rgb = np.broadcast_to(EDGE_COLOR, shaded_rgb.shape)
    final_rgb, final_alpha = alpha_over(shaded_rgb, shaded_alpha, edge_rgb, edge_alpha)

    result = np.zeros_like(base_rgba)
    result[..., :3] = np.clip(final_rgb, 0.0, 1.0)
    result[..., 3] = np.clip(final_alpha, 0.0, 1.0)
    return result


def write_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    thumbs = []
    for path in image_paths:
        image = Image.open(path).convert("RGBA")
        thumb = image.resize((960, 540), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (960, 600), (255, 255, 255, 255))
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 552), path.stem, fill=(20, 20, 20, 255))
        thumbs.append(canvas)

    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * 960, rows * 600), (248, 248, 248, 255))
    for index, image in enumerate(thumbs):
        x = (index % cols) * 960
        y = (index // cols) * 600
        sheet.paste(image, (x, y))
    sheet.save(output_path)


def main() -> None:
    base_path = env_path("EDGE_LAB_BASE_PATH", BASE_PATH)
    world_path = env_path("EDGE_LAB_WORLD_PATH", WORLD_PATH)
    depth_path = env_path("EDGE_LAB_DEPTH_PATH", DEPTH_PATH)
    output_tag = os.environ.get("EDGE_LAB_OUTPUT_TAG", "").strip()
    output_dir = OUTPUT_DIR / output_tag if output_tag else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rgba = load_rgba(base_path)
    world_rgba = load_rgba(world_path)
    depth_rgba = load_rgba(depth_path)

    depth_alpha = depth_rgba[..., 3]
    depth_values = luminance(depth_rgba[..., :3])
    mask_coverage = float((depth_alpha > 0.5).mean())
    tree_shading = extract_tree_shading(world_rgba, depth_alpha)

    tree_shading_path = output_dir / "tree_shading_extracted.png"
    save_rgba(tree_shading, tree_shading_path)
    log(f"Wrote {tree_shading_path}")

    overlay_base = np.zeros_like(base_rgba)
    overlay_base[..., :3] = BACKGROUND_FILL
    overlay_base[..., 3] = 1.0
    base_on_white = composite_variant(overlay_base, base_rgba, np.zeros_like(depth_alpha))

    rendered_paths: list[Path] = [tree_shading_path]
    summary = {
        "inputs": {
            "base": str(base_path),
            "world": str(world_path),
            "depth": str(depth_path),
        },
        "outputs": {},
    }

    summary["edge_color_hex"] = EDGE_COLOR_HEX
    summary["mask_coverage_over_0_5"] = mask_coverage

    for variant in VARIANTS:
        tuned_variant = adapt_variant_for_mask_density(variant, mask_coverage)
        normalized_field, thresholded, edge_strength = detect_edges(depth_values, tuned_variant, depth_alpha)

        for width_mode in WIDTH_MODES:
            tuned_width_mode = adapt_width_mode_for_mask_density(width_mode, mask_coverage)
            output_name = f"{variant['name']}_{width_mode['name']}"
            edge_alpha = apply_variable_width(
                normalized_field,
                thresholded,
                edge_strength,
                depth_alpha,
                tuned_width_mode,
            )
            result_rgba = composite_variant(base_on_white, tree_shading, edge_alpha)

            edge_layer = np.zeros_like(base_rgba)
            edge_layer[..., :3] = EDGE_COLOR
            edge_layer[..., 3] = edge_alpha

            edge_path = output_dir / f"{output_name}_edges.png"
            result_path = output_dir / f"{output_name}_composite.png"
            save_rgba(edge_layer, edge_path)
            save_rgba(result_rgba, result_path)
            rendered_paths.extend([edge_path, result_path])

            coverage = float((edge_alpha > 0.05).mean())
            alpha_mean = float(edge_alpha.mean())
            summary["outputs"][output_name] = {
                "detector": variant["detector"],
                "width_mode": width_mode["name"],
                "notes": f"{variant['notes']} {width_mode['coverage_notes']}",
                "tuned_variant": {
                    "threshold_low": tuned_variant["threshold_low"],
                    "threshold_high": tuned_variant["threshold_high"],
                    "edge_gain": tuned_variant["edge_gain"],
                    "mask_sigma": tuned_variant["mask_sigma"],
                },
                "tuned_width_mode": {
                    "presence_low": tuned_width_mode["presence_low"],
                    "presence_high": tuned_width_mode["presence_high"],
                    "mid_weight": tuned_width_mode["mid_weight"],
                    "wide_weight": tuned_width_mode["wide_weight"],
                    "opacity": tuned_width_mode["opacity"],
                    "min_component_pixels": tuned_width_mode["min_component_pixels"],
                },
                "edge_coverage_over_0_05": coverage,
                "edge_alpha_mean": alpha_mean,
                "edge_layer": str(edge_path),
                "composite": str(result_path),
            }
            log(f"Wrote {result_path}")

    contact_sheet_path = output_dir / "variant_contact_sheet.png"
    write_contact_sheet([path for path in rendered_paths if path.name.endswith("_composite.png")], contact_sheet_path)
    summary["contact_sheet"] = str(contact_sheet_path)

    summary_path = output_dir / "variant_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
