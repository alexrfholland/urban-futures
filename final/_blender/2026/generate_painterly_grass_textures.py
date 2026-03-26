from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


OUTPUT_DIR = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/textures"
)
SIZE = 2048


def wrapped_offsets(size: int) -> tuple[tuple[int, int], ...]:
    return (
        (-size, -size),
        (-size, 0),
        (-size, size),
        (0, -size),
        (0, 0),
        (0, size),
        (size, -size),
        (size, 0),
        (size, size),
    )


def draw_wrapped_ellipse(
    draw: ImageDraw.ImageDraw,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    fill: int,
    wrap_size: int,
) -> None:
    for dx, dy in wrapped_offsets(wrap_size):
        if (
            x1 + dx < -128
            or x0 + dx > wrap_size + 128
            or y1 + dy < -128
            or y0 + dy > wrap_size + 128
        ):
            continue
        draw.ellipse((x0 + dx, y0 + dy, x1 + dx, y1 + dy), fill=fill)


def draw_wrapped_polygon(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: int,
    wrap_size: int,
) -> None:
    for dx, dy in wrapped_offsets(wrap_size):
        shifted = [(px + dx, py + dy) for px, py in points]
        if all(
            px < -128 or px > wrap_size + 128 or py < -128 or py > wrap_size + 128
            for px, py in shifted
        ):
            continue
        draw.polygon(shifted, fill=fill)


def draw_tapered_stroke(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    y: float,
    length: float,
    base_width: float,
    tip_width: float,
    lean: float,
    bend: float,
    fill: int,
    wrap_size: int,
) -> None:
    x1 = x
    y1 = y
    x2 = x + lean * 0.45
    y2 = y - length * 0.58
    x3 = x + lean + bend
    y3 = y - length

    points = [
        (x1 - base_width, y1),
        (x1 + base_width, y1),
        (x2 + base_width * 0.35, y2),
        (x3 + tip_width, y3),
        (x3 - tip_width, y3),
        (x2 - base_width * 0.35, y2),
    ]
    draw_wrapped_polygon(draw, points, fill=fill, wrap_size=wrap_size)


def add_painterly_clump(
    base: Image.Image,
    fringe: Image.Image,
    *,
    rng: random.Random,
    x: float,
    y: float,
    radius_x: float,
    radius_y: float,
    base_fill_range: tuple[int, int],
    fringe_fill_range: tuple[int, int],
    lobe_count_range: tuple[int, int],
    fringe_density_range: tuple[int, int],
    fringe_length_scale: tuple[float, float],
) -> None:
    base_draw = ImageDraw.Draw(base)
    fringe_draw = ImageDraw.Draw(fringe)

    lobe_count = rng.randint(*lobe_count_range)
    for _ in range(lobe_count):
        offset_x = rng.uniform(-radius_x * 0.45, radius_x * 0.45)
        offset_y = rng.uniform(-radius_y * 0.28, radius_y * 0.28)
        lobe_x = radius_x * rng.uniform(0.40, 1.05)
        lobe_y = radius_y * rng.uniform(0.55, 1.10)
        draw_wrapped_ellipse(
            base_draw,
            x0=x + offset_x - lobe_x,
            y0=y + offset_y - lobe_y,
            x1=x + offset_x + lobe_x,
            y1=y + offset_y + lobe_y,
            fill=int(rng.uniform(*base_fill_range)),
            wrap_size=SIZE,
        )

    # Add a few internal sweeps so the clump reads like a painted mass, not a flat blob.
    internal_sweeps = rng.randint(2, 5)
    for _ in range(internal_sweeps):
        draw_tapered_stroke(
            base_draw,
            x=x + rng.uniform(-radius_x * 0.65, radius_x * 0.65),
            y=y + rng.uniform(-radius_y * 0.15, radius_y * 0.55),
            length=rng.uniform(radius_y * 0.35, radius_y * 0.90),
            base_width=rng.uniform(radius_x * 0.08, radius_x * 0.22),
            tip_width=rng.uniform(radius_x * 0.02, radius_x * 0.05),
            lean=rng.uniform(-radius_x * 0.18, radius_x * 0.18),
            bend=rng.uniform(-radius_x * 0.12, radius_x * 0.12),
            fill=int(rng.uniform(*base_fill_range)),
            wrap_size=SIZE,
        )

    fringe_count = max(5, int(radius_x / rng.uniform(*fringe_density_range)))
    for _ in range(fringe_count):
        start_x = x + rng.uniform(-radius_x * 0.92, radius_x * 0.92)
        start_y = y + rng.uniform(radius_y * 0.05, radius_y * 0.78)
        length = rng.uniform(
            radius_y * fringe_length_scale[0], radius_y * fringe_length_scale[1]
        )
        draw_tapered_stroke(
            fringe_draw,
            x=start_x,
            y=start_y,
            length=length,
            base_width=rng.uniform(radius_x * 0.04, radius_x * 0.12),
            tip_width=rng.uniform(radius_x * 0.01, radius_x * 0.03),
            lean=rng.uniform(-radius_x * 0.08, radius_x * 0.08),
            bend=rng.uniform(-radius_x * 0.05, radius_x * 0.05),
            fill=int(rng.uniform(*fringe_fill_range)),
            wrap_size=SIZE,
        )


def finalize_mask(
    base: Image.Image,
    fringe: Image.Image,
    *,
    base_blur: float,
    fringe_blur: float,
    wash_blur: float,
    lift_threshold: int,
    lift_gain: float,
) -> Image.Image:
    base = base.filter(ImageFilter.GaussianBlur(radius=base_blur))
    fringe = fringe.filter(ImageFilter.GaussianBlur(radius=fringe_blur))
    wash = base.filter(ImageFilter.GaussianBlur(radius=wash_blur))
    merged = ImageChops.screen(ImageChops.lighter(base, fringe), wash)
    return merged.point(
        lambda value: 0
        if value < lift_threshold
        else min(255, int((value - lift_threshold) * lift_gain))
    )


def build_field_mask() -> Image.Image:
    rng = random.Random(20260331)
    base = Image.new("L", (SIZE, SIZE), 0)
    fringe = Image.new("L", (SIZE, SIZE), 0)

    for _ in range(58):
        add_painterly_clump(
            base,
            fringe,
            rng=rng,
            x=rng.uniform(0, SIZE),
            y=rng.uniform(0, SIZE),
            radius_x=rng.uniform(120, 280),
            radius_y=rng.uniform(42, 110),
            base_fill_range=(70, 150),
            fringe_fill_range=(70, 135),
            lobe_count_range=(4, 7),
            fringe_density_range=(16, 24),
            fringe_length_scale=(0.28, 0.95),
        )

    return finalize_mask(
        base,
        fringe,
        base_blur=11.0,
        fringe_blur=3.8,
        wash_blur=22.0,
        lift_threshold=12,
        lift_gain=1.15,
    )


def build_dark_clump_mask() -> Image.Image:
    rng = random.Random(20260332)
    base = Image.new("L", (SIZE, SIZE), 0)
    fringe = Image.new("L", (SIZE, SIZE), 0)

    for _ in range(36):
        add_painterly_clump(
            base,
            fringe,
            rng=rng,
            x=rng.uniform(0, SIZE),
            y=rng.uniform(0, SIZE),
            radius_x=rng.uniform(96, 210),
            radius_y=rng.uniform(34, 84),
            base_fill_range=(110, 220),
            fringe_fill_range=(120, 255),
            lobe_count_range=(3, 5),
            fringe_density_range=(18, 28),
            fringe_length_scale=(0.40, 1.20),
        )

    return finalize_mask(
        base,
        fringe,
        base_blur=6.0,
        fringe_blur=1.6,
        wash_blur=11.0,
        lift_threshold=18,
        lift_gain=1.35,
    )


def build_light_clump_mask() -> Image.Image:
    rng = random.Random(20260333)
    base = Image.new("L", (SIZE, SIZE), 0)
    fringe = Image.new("L", (SIZE, SIZE), 0)

    for _ in range(42):
        add_painterly_clump(
            base,
            fringe,
            rng=rng,
            x=rng.uniform(0, SIZE),
            y=rng.uniform(0, SIZE),
            radius_x=rng.uniform(72, 180),
            radius_y=rng.uniform(26, 72),
            base_fill_range=(90, 200),
            fringe_fill_range=(130, 255),
            lobe_count_range=(3, 5),
            fringe_density_range=(20, 32),
            fringe_length_scale=(0.25, 0.90),
        )

    return finalize_mask(
        base,
        fringe,
        base_blur=5.2,
        fringe_blur=1.4,
        wash_blur=9.0,
        lift_threshold=20,
        lift_gain=1.25,
    )


def save_rgba(mask: Image.Image, path: Path) -> None:
    rgba = Image.merge("RGBA", (mask, mask, mask, mask))
    rgba.save(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    field_mask = build_field_mask()
    dark_clump_mask = build_dark_clump_mask()
    light_clump_mask = build_light_clump_mask()

    save_rgba(field_mask, OUTPUT_DIR / "ghibli_grass_field_mask.png")
    save_rgba(dark_clump_mask, OUTPUT_DIR / "ghibli_grass_stroke_mask.png")
    save_rgba(light_clump_mask, OUTPUT_DIR / "ghibli_grass_highlight_mask.png")

    print("Saved painterly grass textures to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
