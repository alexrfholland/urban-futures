from dataclasses import dataclass
from pathlib import Path
from collections import deque

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat


GRASS_DIR = Path(
    "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/blender/2026/texture/grass"
)
SOURCE_PATH = GRASS_DIR / "grass.png"
OUTPUT_DIR = GRASS_DIR / "extracted"


@dataclass(frozen=True)
class ExtractionSpec:
    name: str
    kind: str
    width: int
    height: int


@dataclass
class BandComponent:
    area: int
    bbox: tuple[int, int, int, int]
    mean_score: float
    pixels: list[int]


SPECS = [
    ExtractionSpec("dark_01", "dark", 780, 420),
    ExtractionSpec("light_01", "light", 780, 420),
    ExtractionSpec("dark_02", "dark", 560, 320),
    ExtractionSpec("light_02", "light", 560, 320),
    ExtractionSpec("dark_03", "dark", 420, 240),
    ExtractionSpec("light_03", "light", 420, 240),
]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def image_from_values(values: bytearray, size: tuple[int, int]) -> Image.Image:
    return Image.frombytes("L", size, bytes(values))


def build_masks(
    image: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    rgba = image.convert("RGBA")
    size = rgba.size
    bw_values = bytearray()
    dark_values = bytearray()
    light_values = bytearray()
    affinity_values = bytearray()
    dark_blue_values = bytearray()

    for r, g, b, a in rgba.getdata():
        rf = r / 255.0
        gf = g / 255.0
        bf = b / 255.0
        af = a / 255.0

        luma = (0.2126 * rf) + (0.7152 * gf) + (0.0722 * bf)
        sat = max(rf, gf, bf) - min(rf, gf, bf)
        affinity = clamp01((gf * 0.90) + (bf * 0.30) - (rf * 0.38))
        cool = clamp01((bf * 0.95) + (gf * 0.40) - (rf * 0.65))
        warm = clamp01((gf * 0.72) + (rf * 0.48) - (bf * 0.25))
        dark_blue = clamp01(
            (((1.0 - luma) ** 1.18) * (0.20 + (cool * 0.80)) * (0.28 + (sat * 0.72)) * affinity * af)
        )

        bw = clamp01(luma * af)
        dark = clamp01((((1.0 - luma) ** 1.08) * (0.35 + (cool * 0.65)) * (0.30 + (sat * 0.70)) * affinity * af))
        light = clamp01(((luma**1.05) * (0.30 + (warm * 0.70)) * (0.28 + (sat * 0.72)) * affinity * af))

        bw_values.append(round(bw * 255.0))
        dark_values.append(round(dark * 255.0))
        light_values.append(round(light * 255.0))
        affinity_values.append(round(affinity * af * 255.0))
        dark_blue_values.append(round(dark_blue * 255.0))

    bw_map = ImageOps.autocontrast(image_from_values(bw_values, size))
    dark_map = ImageOps.autocontrast(image_from_values(dark_values, size))
    light_map = ImageOps.autocontrast(image_from_values(light_values, size))
    affinity_map = ImageOps.autocontrast(image_from_values(affinity_values, size))
    dark_blue_map = ImageOps.autocontrast(image_from_values(dark_blue_values, size))
    return bw_map, dark_map, light_map, affinity_map, dark_blue_map


def image_quantile(image: Image.Image, quantile: float) -> int:
    histogram = image.histogram()
    total = sum(histogram)
    if total <= 0:
        return 0
    target = total * quantile
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return value
    return 255


def build_band_binary(
    band_map: Image.Image,
    quantile: float = 0.86,
    min_threshold: int = 112,
) -> tuple[Image.Image, Image.Image, int]:
    softened = band_map.filter(ImageFilter.GaussianBlur(radius=3.0))
    softened = softened.filter(ImageFilter.MaxFilter(9))
    softened = softened.filter(ImageFilter.MinFilter(7))
    threshold = max(min_threshold, image_quantile(softened, quantile))
    binary = softened.point(lambda value: 255 if value >= threshold else 0)
    binary = binary.filter(ImageFilter.MaxFilter(7))
    binary = binary.filter(ImageFilter.MinFilter(5))
    return softened, binary, threshold


def expand_rect(
    rect: tuple[int, int, int, int],
    margin: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    width, height = image_size
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def build_contrast_base(bw_map: Image.Image, affinity_map: Image.Image) -> Image.Image:
    base = ImageChops.multiply(bw_map, affinity_map)
    base = ImageOps.autocontrast(base)
    base = base.filter(ImageFilter.UnsharpMask(radius=10, percent=220, threshold=2))
    base = base.filter(ImageFilter.GaussianBlur(radius=0.8))
    return ImageOps.autocontrast(base)


def make_triangular_band(
    source_map: Image.Image,
    center: int,
    radius: int,
    exponent: float = 1.0,
) -> Image.Image:
    radius = max(1, radius)
    lut = []
    for value in range(256):
        distance = abs(value - center) / float(radius)
        band_value = clamp01(1.0 - distance) ** exponent
        lut.append(round(band_value * 255.0))
    return source_map.point(lut)


def build_contrast_bands(contrast_base: Image.Image) -> dict[str, Image.Image]:
    q10 = image_quantile(contrast_base, 0.10)
    q25 = image_quantile(contrast_base, 0.25)
    q50 = image_quantile(contrast_base, 0.50)
    q75 = image_quantile(contrast_base, 0.75)
    q90 = image_quantile(contrast_base, 0.90)

    shadow = make_triangular_band(
        contrast_base,
        center=q25,
        radius=max(24, q50 - q10),
        exponent=1.15,
    )
    mid = make_triangular_band(
        contrast_base,
        center=q50,
        radius=max(30, q75 - q25),
        exponent=1.05,
    )
    light = make_triangular_band(
        contrast_base,
        center=q75,
        radius=max(24, q90 - q50),
        exponent=1.15,
    )

    return {
        "shadow": ImageOps.autocontrast(shadow),
        "mid": ImageOps.autocontrast(mid),
        "light": ImageOps.autocontrast(light),
    }


def find_band_components(
    binary_mask: Image.Image,
    score_map: Image.Image,
    min_area: int = 2400,
    max_components: int = 6,
) -> list[BandComponent]:
    width, height = binary_mask.size
    binary_bytes = binary_mask.tobytes()
    score_bytes = score_map.tobytes()
    visited = bytearray(width * height)
    components: list[BandComponent] = []

    for idx, value in enumerate(binary_bytes):
        if value == 0 or visited[idx]:
            continue

        queue = deque([idx])
        visited[idx] = 1
        pixels: list[int] = []
        area = 0
        score_total = 0
        min_x = width
        min_y = height
        max_x = 0
        max_y = 0

        while queue:
            current = queue.pop()
            pixels.append(current)
            area += 1
            score_total += score_bytes[current]

            x = current % width
            y = current // width
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

            if x > 0:
                neighbor = current - 1
                if binary_bytes[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
            if x < width - 1:
                neighbor = current + 1
                if binary_bytes[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
            if y > 0:
                neighbor = current - width
                if binary_bytes[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)
            if y < height - 1:
                neighbor = current + width
                if binary_bytes[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)

        if area < min_area:
            continue

        components.append(
            BandComponent(
                area=area,
                bbox=(min_x, min_y, max_x + 1, max_y + 1),
                mean_score=score_total / float(area * 255.0),
                pixels=pixels,
            )
        )

    components.sort(key=lambda item: (item.mean_score * (item.area ** 0.42)), reverse=True)
    return components[:max_components]


def rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    intersection = iw * ih
    if intersection == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return intersection / float(area_a + area_b - intersection)


def rect_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    intersection = iw * ih
    if intersection == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return intersection / float(min(area_a, area_b))


def iter_positions(limit: int, step: int) -> list[int]:
    positions = list(range(0, limit + 1, max(1, step)))
    if positions[-1] != limit:
        positions.append(limit)
    return positions


def choose_window(
    score_map: Image.Image,
    bw_map: Image.Image,
    affinity_map: Image.Image,
    width: int,
    height: int,
    taken_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    image_w, image_h = score_map.size
    max_x = image_w - width
    max_y = image_h - height
    if max_x < 0 or max_y < 0:
        return None

    best_rect = None
    best_score = -1.0
    step_x = max(24, width // 7)
    step_y = max(24, height // 7)

    for y in iter_positions(max_y, step_y):
        for x in iter_positions(max_x, step_x):
            rect = (x, y, x + width, y + height)
            if any(
                rect_iou(rect, taken) > 0.18 or rect_overlap_ratio(rect, taken) > 0.34
                for taken in taken_rects
            ):
                continue

            score_crop = score_map.crop(rect)
            bw_crop = bw_map.crop(rect)
            affinity_crop = affinity_map.crop(rect)

            score_stat = ImageStat.Stat(score_crop)
            bw_stat = ImageStat.Stat(bw_crop)
            affinity_stat = ImageStat.Stat(affinity_crop)

            score_mean = score_stat.mean[0] / 255.0
            score_peak = score_crop.getextrema()[1] / 255.0
            texture_var = bw_stat.stddev[0] / 255.0
            affinity_mean = affinity_stat.mean[0] / 255.0
            weighted_score = (score_mean * 0.62) + (score_peak * 0.28) + (texture_var * 0.10)

            if affinity_mean < 0.18 or score_peak < 0.20:
                continue
            if weighted_score > best_score:
                best_score = weighted_score
                best_rect = rect

    return best_rect


def make_soft_alpha(mask_image: Image.Image) -> Image.Image:
    alpha = mask_image.point(
        lambda value: 0
        if value < 42
        else min(255, round(((value - 42) / 170.0) * 255.0))
    )
    return alpha.filter(ImageFilter.GaussianBlur(radius=4.0))


def rgba_preview(color_crop: Image.Image, alpha_image: Image.Image) -> Image.Image:
    rgba = color_crop.convert("RGBA")
    rgba.putalpha(alpha_image)
    background = Image.new("RGBA", rgba.size, (30, 30, 30, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def save_crop_set(
    spec: ExtractionSpec,
    source_image: Image.Image,
    bw_map: Image.Image,
    dark_map: Image.Image,
    light_map: Image.Image,
    rect: tuple[int, int, int, int],
) -> list[tuple[str, Image.Image]]:
    color_crop = source_image.crop(rect).convert("RGB")
    bw_crop = bw_map.crop(rect)
    dark_crop = dark_map.crop(rect)
    light_crop = light_map.crop(rect)
    main_mask = dark_crop if spec.kind == "dark" else light_crop
    alpha_image = make_soft_alpha(main_mask)

    color_path = OUTPUT_DIR / f"grass_cluster_{spec.name}.png"
    rgba_path = OUTPUT_DIR / f"grass_cluster_{spec.name}_rgba.png"
    bw_path = OUTPUT_DIR / f"grass_cluster_{spec.name}_bw.png"
    dark_path = OUTPUT_DIR / f"grass_cluster_{spec.name}_dark_mask.png"
    light_path = OUTPUT_DIR / f"grass_cluster_{spec.name}_light_mask.png"

    color_crop.save(color_path)
    rgba = color_crop.convert("RGBA")
    rgba.putalpha(alpha_image)
    rgba.save(rgba_path)
    bw_crop.save(bw_path)
    dark_crop.save(dark_path)
    light_crop.save(light_path)

    return [
        (f"{spec.name}_color", color_crop),
        (f"{spec.name}_rgba", rgba_preview(color_crop, alpha_image)),
        (f"{spec.name}_dark", dark_crop),
        (f"{spec.name}_light", light_crop),
    ]


def build_component_mask(
    component: BandComponent,
    score_map: Image.Image,
    rect: tuple[int, int, int, int],
) -> Image.Image:
    x0, y0, x1, y1 = rect
    local_width = x1 - x0
    local_height = y1 - y0
    values = bytearray(local_width * local_height)
    score_bytes = score_map.tobytes()
    image_width, _ = score_map.size

    for idx in component.pixels:
        x = idx % image_width
        y = idx // image_width
        if x < x0 or x >= x1 or y < y0 or y >= y1:
            continue
        local_idx = ((y - y0) * local_width) + (x - x0)
        values[local_idx] = score_bytes[idx]

    mask = image_from_values(values, (local_width, local_height))
    mask = mask.filter(ImageFilter.MaxFilter(5))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3.0))
    return mask


def save_band_component_set(
    index: int,
    band_name: str,
    component: BandComponent,
    source_image: Image.Image,
    bw_map: Image.Image,
    band_map: Image.Image,
) -> list[tuple[str, Image.Image]]:
    rect = expand_rect(component.bbox, margin=18, image_size=source_image.size)
    color_crop = source_image.crop(rect).convert("RGB")
    bw_crop = bw_map.crop(rect)
    band_crop = band_map.crop(rect)
    alpha_image = build_component_mask(component, band_map, rect)

    base_name = f"grass_{band_name}_clump_{index:02d}"
    color_path = OUTPUT_DIR / f"{base_name}.png"
    rgba_path = OUTPUT_DIR / f"{base_name}_rgba.png"
    bw_path = OUTPUT_DIR / f"{base_name}_bw.png"
    band_path = OUTPUT_DIR / f"{base_name}_band_mask.png"

    color_crop.save(color_path)
    bw_crop.save(bw_path)
    band_crop.save(band_path)

    rgba = color_crop.convert("RGBA")
    rgba.putalpha(alpha_image)
    rgba.save(rgba_path)

    return [
        (f"{base_name}_color", color_crop),
        (f"{base_name}_rgba", rgba_preview(color_crop, alpha_image)),
        (f"{base_name}_band", band_crop),
        (f"{base_name}_bw", bw_crop),
    ]


def build_contact_sheet(images: list[tuple[str, Image.Image]], output_path: Path) -> None:
    if not images:
        return
    tile_w = max(img.width for _, img in images)
    tile_h = max(img.height for _, img in images)
    columns = 4
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), color=(18, 18, 18))

    for index, (_, image) in enumerate(images):
        row = index // columns
        col = index % columns
        x = col * tile_w
        y = row * tile_h
        tile = image.convert("RGB")
        if tile.size != (tile_w, tile_h):
            tile = ImageOps.pad(tile, (tile_w, tile_h), color=(18, 18, 18))
        sheet.paste(tile, (x, y))

    sheet.save(output_path)


def save_extraction_preview(
    image: Image.Image,
    selections: list[tuple[ExtractionSpec, tuple[int, int, int, int]]],
    output_path: Path,
) -> None:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    colors = {"dark": (31, 63, 110), "light": (220, 190, 80)}

    for spec, rect in selections:
        x0, y0, _, _ = rect
        draw.rectangle(rect, outline=colors[spec.kind], width=5)
        draw.rectangle((x0, y0, x0 + 160, y0 + 34), fill=colors[spec.kind])
        draw.text((x0 + 8, y0 + 8), spec.name, fill=(245, 245, 245))

    preview.save(output_path)


def save_band_preview(
    image: Image.Image,
    component_sets: dict[str, list[BandComponent]],
    output_path: Path,
) -> None:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    colors = {
        "shadow": (31, 63, 110),
        "mid": (102, 122, 70),
        "light": (220, 190, 80),
    }

    for band_name, components in component_sets.items():
        color = colors.get(band_name, (200, 200, 200))
        for index, component in enumerate(components, start=1):
            rect = expand_rect(component.bbox, margin=18, image_size=image.size)
            x0, y0, _, _ = rect
            draw.rectangle(rect, outline=color, width=5)
            draw.rectangle((x0, y0, x0 + 206, y0 + 34), fill=color)
            draw.text((x0 + 8, y0 + 8), f"{band_name}_{index:02d}", fill=(245, 245, 245))

    preview.save(output_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_image = Image.open(SOURCE_PATH)
    bw_map, dark_map, light_map, affinity_map, dark_blue_map = build_masks(source_image)

    bw_map.save(OUTPUT_DIR / "grass_full_bw.png")
    dark_map.save(OUTPUT_DIR / "grass_full_dark_mask.png")
    light_map.save(OUTPUT_DIR / "grass_full_light_mask.png")
    dark_blue_map.save(OUTPUT_DIR / "grass_dark_blue_band.png")

    dark_blue_softened, dark_blue_binary, dark_blue_threshold = build_band_binary(dark_blue_map)
    dark_blue_softened.save(OUTPUT_DIR / "grass_dark_blue_band_softened.png")
    dark_blue_binary.save(OUTPUT_DIR / "grass_dark_blue_band_binary.png")

    contrast_base = build_contrast_base(bw_map, affinity_map)
    contrast_base.save(OUTPUT_DIR / "grass_contrast_base.png")
    contrast_bands = build_contrast_bands(contrast_base)
    contrast_band_thresholds: dict[str, int] = {}
    contrast_component_sets: dict[str, list[BandComponent]] = {}
    contrast_contact_tiles: list[tuple[str, Image.Image]] = []

    for band_name, band_map in contrast_bands.items():
        band_map.save(OUTPUT_DIR / f"grass_band_{band_name}.png")
        band_softened, band_binary, band_threshold = build_band_binary(
            band_map,
            quantile=0.85 if band_name == "mid" else 0.88,
            min_threshold=96 if band_name == "mid" else 104,
        )
        band_softened.save(OUTPUT_DIR / f"grass_band_{band_name}_softened.png")
        band_binary.save(OUTPUT_DIR / f"grass_band_{band_name}_binary.png")
        contrast_band_thresholds[band_name] = band_threshold

        components = find_band_components(
            band_binary,
            band_map,
            min_area=2000 if band_name == "mid" else 2400,
            max_components=3,
        )
        contrast_component_sets[band_name] = components
        for index, component in enumerate(components, start=1):
            contrast_contact_tiles.extend(
                save_band_component_set(index, band_name, component, source_image, bw_map, band_map)
            )

    selections: list[tuple[ExtractionSpec, tuple[int, int, int, int]]] = []
    taken_rects: list[tuple[int, int, int, int]] = []
    for spec in SPECS:
        score_map = dark_map if spec.kind == "dark" else light_map
        rect = choose_window(
            score_map=score_map,
            bw_map=bw_map,
            affinity_map=affinity_map,
            width=spec.width,
            height=spec.height,
            taken_rects=taken_rects,
        )
        if rect is None:
            continue
        selections.append((spec, rect))
        taken_rects.append(rect)

    contact_tiles: list[tuple[str, Image.Image]] = []
    for spec, rect in selections:
        contact_tiles.extend(save_crop_set(spec, source_image, bw_map, dark_map, light_map, rect))

    build_contact_sheet(contact_tiles, OUTPUT_DIR / "grass_cluster_contact_sheet.png")
    save_extraction_preview(source_image, selections, OUTPUT_DIR / "grass_extraction_preview.png")

    band_components = find_band_components(dark_blue_binary, dark_blue_map)
    band_tiles: list[tuple[str, Image.Image]] = []
    for index, component in enumerate(band_components, start=1):
        band_tiles.extend(save_band_component_set(index, "dark_blue", component, source_image, bw_map, dark_blue_map))

    build_contact_sheet(band_tiles, OUTPUT_DIR / "grass_dark_blue_clumps_contact_sheet.png")
    save_band_preview(source_image, {"dark_blue": band_components}, OUTPUT_DIR / "grass_dark_blue_components_preview.png")
    build_contact_sheet(contrast_contact_tiles, OUTPUT_DIR / "grass_contrast_clumps_contact_sheet.png")
    save_band_preview(source_image, contrast_component_sets, OUTPUT_DIR / "grass_contrast_components_preview.png")
    print(
        f"Saved extracted grass texture set to {OUTPUT_DIR} "
        f"(dark_blue_threshold={dark_blue_threshold}, blue_components={len(band_components)}, "
        f"contrast_thresholds={contrast_band_thresholds})"
    )


if __name__ == "__main__":
    main()
