from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter


PURPLE_SRGB = (34, 18, 59)
DEFAULT_MASK_THRESHOLD = 1
DEFAULT_OUTLINE_WIDTH = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build flat purple proposal outliners from proposal mask PNGs."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing proposal mask PNGs.")
    parser.add_argument("output_dir", type=Path, help="Folder to write proposal outliner PNGs.")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_OUTLINE_WIDTH,
        help="Odd pixel width used for the edge dilation/erosion pass.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_MASK_THRESHOLD,
        help="Alpha threshold above which the mask counts as filled.",
    )
    return parser.parse_args()


def proposal_pngs(input_dir: Path) -> list[Path]:
    pngs = sorted(
        path
        for path in input_dir.glob("proposal-*.png")
        if path.is_file() and not path.name.endswith("_outliner.png")
    )
    if not pngs:
        raise FileNotFoundError(f"No proposal PNGs found in {input_dir}")
    return pngs


def binary_alpha(mask_path: Path, threshold: int) -> Image.Image:
    rgba = Image.open(mask_path).convert("RGBA")
    alpha = rgba.getchannel("A")
    return alpha.point(lambda value: 255 if value >= threshold else 0, mode="L")


def edge_alpha(alpha: Image.Image, width: int) -> Image.Image:
    if width < 3:
        width = 3
    if width % 2 == 0:
        width += 1
    dilated = alpha.filter(ImageFilter.MaxFilter(width))
    eroded = alpha.filter(ImageFilter.MinFilter(width))
    return ImageChops.subtract(dilated, eroded)


def purple_outliner(alpha: Image.Image) -> Image.Image:
    image = Image.new("RGBA", alpha.size, PURPLE_SRGB + (0,))
    image.putalpha(alpha)
    return image


def render_mask_outliners(input_dir: Path, output_dir: Path, *, width: int, threshold: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for mask_path in proposal_pngs(input_dir):
        alpha = binary_alpha(mask_path, threshold)
        outline = edge_alpha(alpha, width)
        image = purple_outliner(outline)
        output_path = output_dir / f"{mask_path.stem}_outliner.png"
        image.save(output_path)
        written.append(output_path)
    return written


def main() -> None:
    args = parse_args()
    written = render_mask_outliners(
        args.input_dir,
        args.output_dir,
        width=args.width,
        threshold=args.threshold,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
