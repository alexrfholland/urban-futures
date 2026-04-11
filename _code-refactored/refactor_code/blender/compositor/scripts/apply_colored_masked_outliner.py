"""Post-processor: colorize depth outlines and mask them with a proposal fill.

Usage:
    python apply_colored_masked_outliner.py \
        --outliner path/to/depth_outliner.png \
        --mask path/to/proposal-only_release-control.png \
        --color 0.55,0.10,0.85 \
        --output path/to/output.png

The outliner PNG provides edge geometry (white on transparent).
The mask PNG provides the proposal area (any non-transparent pixel = inside).
The output is the outliner colorized and clipped to the mask region.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def apply_colored_masked_outliner(
    outliner_path: Path,
    mask_path: Path,
    color: tuple[float, float, float],
    output_path: Path,
) -> None:
    outliner = np.array(Image.open(outliner_path).convert("RGBA"), dtype=np.float32) / 255.0
    mask = np.array(Image.open(mask_path).convert("RGBA"), dtype=np.float32) / 255.0

    outline_alpha = outliner[:, :, 3]
    mask_alpha = mask[:, :, 3]

    combined_alpha = outline_alpha * mask_alpha

    result = np.zeros_like(outliner)
    result[:, :, 0] = color[0]
    result[:, :, 1] = color[1]
    result[:, :, 2] = color[2]
    result[:, :, 3] = combined_alpha

    out_img = Image.fromarray((result * 255).clip(0, 255).astype(np.uint8), "RGBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)
    print(f"[colored_outliner] Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outliner", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--color", required=True, help="R,G,B floats 0-1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    r, g, b = (float(x) for x in args.color.split(","))
    apply_colored_masked_outliner(args.outliner, args.mask, (r, g, b), args.output)


if __name__ == "__main__":
    main()
