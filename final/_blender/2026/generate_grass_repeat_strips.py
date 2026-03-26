from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


SOURCE_PATH = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/texture/grass/grass.png"
)
OUTPUT_DIR = SOURCE_PATH.parent / "repeat"

TARGET_SIZE = (3072, 640)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def make_crop_box(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    left = int(width * 0.02)
    right = int(width * 0.98)
    top = int(height * 0.20)
    bottom = int(height * 0.95)
    return (left, top, right, bottom)


def build_horizontal_tile(image: Image.Image) -> Image.Image:
    width, height = image.size
    overlap = max(96, int(width * 0.08))
    source = image.convert("RGB")
    tile = source.copy()

    left_band = source.crop((0, 0, overlap, height))
    right_band = source.crop((width - overlap, 0, width, height))
    seam_band = Image.blend(left_band, right_band, 0.5)
    seam_band = seam_band.filter(
        ImageFilter.UnsharpMask(radius=1, percent=90, threshold=1)
    )

    tile.paste(seam_band, (0, 0))
    tile.paste(seam_band, (width - overlap, 0))
    return tile


def build_panel_grayscale(color_strip: Image.Image) -> Image.Image:
    rgb = color_strip.convert("RGB")
    values = bytearray()

    for r, g, b in rgb.getdata():
        rf = r / 255.0
        gf = g / 255.0
        bf = b / 255.0

        luma = (0.2126 * rf) + (0.7152 * gf) + (0.0722 * bf)
        sat = max(rf, gf, bf) - min(rf, gf, bf)
        green_lift = clamp01((gf * 1.18) + (rf * 0.08) - (bf * 0.34))
        yellow_lift = clamp01((rf * 0.44) + (gf * 0.82) - (bf * 0.22))
        cool_drop = clamp01((bf * 1.10) + (gf * 0.22) - (rf * 0.54))

        mixed = (
            luma * 0.82
            + green_lift * 0.18
            + yellow_lift * 0.10
            - cool_drop * 0.30
            + sat * 0.06
        )
        values.append(round(clamp01(mixed) * 255.0))

    image = Image.frombytes("L", rgb.size, bytes(values))
    image = ImageOps.autocontrast(image, cutoff=(1, 1))
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))


def apply_levels(
    image: Image.Image,
    *,
    in_min: int,
    in_max: int,
    gamma: float = 1.0,
) -> Image.Image:
    span = max(1, in_max - in_min)
    table = []
    for value in range(256):
        normalized = clamp01((value - in_min) / float(span))
        adjusted = normalized ** gamma
        table.append(round(adjusted * 255.0))
    return image.point(table)


def derive_channel_masks(
    base_gray: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    shadow = ImageOps.invert(base_gray)
    shadow = apply_levels(shadow, in_min=78, in_max=232, gamma=0.92)
    shadow = ImageOps.autocontrast(shadow, cutoff=(1, 1))

    mid = apply_levels(base_gray, in_min=18, in_max=232, gamma=0.98)
    mid = ImageOps.autocontrast(mid, cutoff=(1, 1))

    light = apply_levels(base_gray, in_min=138, in_max=255, gamma=0.86)
    light = ImageOps.autocontrast(light, cutoff=(1, 1))

    soft_light = base_gray.filter(ImageFilter.GaussianBlur(radius=6))
    mid = ImageChops.blend(mid, soft_light, 0.18)
    return base_gray, shadow, mid, light


def derive_detail_mask(base_gray: Image.Image, mid_mask: Image.Image) -> Image.Image:
    blurred = base_gray.filter(ImageFilter.GaussianBlur(radius=10))
    high_pass = ImageChops.difference(base_gray, blurred)
    detailed = ImageChops.multiply(high_pass, mid_mask)
    detailed = ImageOps.autocontrast(detailed)
    return detailed.filter(ImageFilter.GaussianBlur(radius=0.8))


def build_tripled_preview(image: Image.Image, repeats: int = 3) -> Image.Image:
    preview = Image.new("RGB", (image.width * repeats, image.height))
    tile = image.convert("RGB")
    for idx in range(repeats):
        preview.paste(tile, (idx * image.width, 0))
    return preview


def add_label(image: Image.Image, label: str) -> Image.Image:
    labeled = image.convert("RGB").copy()
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, 180, 28), fill=(20, 20, 20))
    draw.text((8, 7), label, fill=(240, 240, 240))
    return labeled


def build_preview_sheet(rows: list[tuple[str, Image.Image]], output_path: Path) -> None:
    row_images = [add_label(build_tripled_preview(image), label) for label, image in rows]
    width = max(image.width for image in row_images)
    height = sum(image.height for image in row_images)
    sheet = Image.new("RGB", (width, height), color=(18, 18, 18))

    y = 0
    for row in row_images:
        sheet.paste(row, (0, y))
        y += row.height

    sheet.save(output_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = Image.open(SOURCE_PATH).convert("RGB")
    crop_box = make_crop_box(source)
    cropped = source.crop(crop_box).resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    color_strip = build_horizontal_tile(cropped)

    base_gray = build_panel_grayscale(color_strip)
    base_gray, shadow_mask, mid_mask, light_mask = derive_channel_masks(base_gray)
    detail_mask = derive_detail_mask(base_gray, mid_mask)

    color_path = OUTPUT_DIR / "grass_repeat_color.png"
    gray_path = OUTPUT_DIR / "grass_repeat_bw.png"
    shadow_path = OUTPUT_DIR / "grass_repeat_shadow.png"
    mid_path = OUTPUT_DIR / "grass_repeat_mid.png"
    light_path = OUTPUT_DIR / "grass_repeat_light.png"
    detail_path = OUTPUT_DIR / "grass_repeat_detail.png"
    crop_preview_path = OUTPUT_DIR / "grass_repeat_source_crop.png"
    color_tripled_path = OUTPUT_DIR / "grass_repeat_color_tripled.png"
    sheet_path = OUTPUT_DIR / "grass_repeat_preview_sheet.png"

    cropped.save(crop_preview_path)
    color_strip.save(color_path)
    base_gray.save(gray_path)
    shadow_mask.save(shadow_path)
    mid_mask.save(mid_path)
    light_mask.save(light_path)
    detail_mask.save(detail_path)
    build_tripled_preview(color_strip).save(color_tripled_path)
    build_preview_sheet(
        [
            ("color", color_strip),
            ("bw", base_gray),
            ("shadow", shadow_mask),
            ("mid", mid_mask),
            ("light", light_mask),
            ("detail", detail_mask),
        ],
        sheet_path,
    )

    print(f"Saved grass repeat strips to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
