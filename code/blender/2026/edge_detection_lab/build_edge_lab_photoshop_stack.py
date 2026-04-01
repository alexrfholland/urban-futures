from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "edge_detection_lab"
    / "outputs"
    / "edge_lab_final_template_timeslices_20260330"
)
OUTPUT_ROOT = Path(os.environ.get("EDGE_LAB_PHOTOSHOP_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).expanduser()
PSD_PATH = Path(
    os.environ.get(
        "EDGE_LAB_PHOTOSHOP_PSD_PATH",
        str(OUTPUT_ROOT / "edge_lab_timeslices_stack_20260330.psd"),
    )
).expanduser()
PHOTOSHOP_APP = os.environ.get("EDGE_LAB_PHOTOSHOP_APP", "Adobe Photoshop 2026")
PSD_MIST_VARIANT = os.environ.get("EDGE_LAB_PHOTOSHOP_MIST_VARIANT", "mist_kirsch_extra_thin").strip()


def require(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def file_layer(name: str, path: Path, blend_mode: str = "NORMAL", opacity: int = 100) -> dict[str, object]:
    return {
        "type": "file",
        "name": name,
        "path": require(path),
        "blend_mode": blend_mode,
        "opacity": opacity,
    }


def levels_layer(name: str, clipped: bool = False) -> dict[str, object]:
    return {
        "type": "levels",
        "name": name,
        "clipped": clipped,
    }


def group(name: str, children: list[dict[str, object]]) -> dict[str, object]:
    return {
        "type": "group",
        "name": name,
        "children": children,
    }


def write_outline_from_alpha(source: Path, destination: Path, width: int = 3) -> Path:
    source_rgba = Image.open(source).convert("RGBA")
    alpha = source_rgba.getchannel("A").point(lambda v: 255 if v > 0 else 0)
    dilated = alpha.filter(ImageFilter.MaxFilter(width))
    eroded = alpha.filter(ImageFilter.MinFilter(width))
    edge = ImageChops.subtract(dilated, eroded)

    outline = Image.new("RGBA", source_rgba.size, (0, 0, 0, 0))
    outline.putalpha(edge)
    destination.parent.mkdir(parents=True, exist_ok=True)
    outline.save(destination)
    return destination


def normalize_png_for_photoshop(source: Path, destination: Path) -> Path:
    image = Image.open(source).convert("RGBA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return destination


def normalized_file_spec_path(spec: dict[str, object], cache_root: Path, cache: dict[str, str], index: int) -> None:
    source = Path(str(spec["path"]))
    cache_key = str(source)
    if cache_key in cache:
        spec["path"] = cache[cache_key]
        return

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(spec["name"])).strip("._") or f"layer_{index:03d}"
    destination = cache_root / f"{index:03d}_{safe_name}.png"
    normalized = normalize_png_for_photoshop(source, destination)
    cache[cache_key] = str(normalized)
    spec["path"] = str(normalized)


def normalize_stack_file_paths(stack: list[dict[str, object]], cache_root: Path) -> None:
    cache: dict[str, str] = {}
    counter = 0

    def visit(node: dict[str, object]) -> None:
        nonlocal counter
        if node["type"] == "group":
            for child in node["children"]:
                visit(child)
            return
        if node["type"] == "file":
            normalized_file_spec_path(node, cache_root, cache, counter)
            counter += 1

    for item in stack:
        visit(item)


def resource_layers(resources: Path, phase: str) -> list[dict[str, object]]:
    # Manifest is built bottom-to-top because imported layers land at the top of the group.
    return [
        file_layer(f"{phase}_other", resources / f"{phase}_other.png"),
        file_layer(f"{phase}_perch", resources / f"{phase}_perch.png"),
        file_layer(f"{phase}_dead", resources / f"{phase}_dead.png"),
        file_layer(f"{phase}_peeling", resources / f"{phase}_peeling.png"),
        file_layer(f"{phase}_fallen", resources / f"{phase}_fallen.png"),
        file_layer(f"{phase}_epiphyte", resources / f"{phase}_epiphyte.png"),
        file_layer(f"{phase}_hollow", resources / f"{phase}_hollow.png"),
    ]


def size_layers(sizes: Path, phase: str) -> list[dict[str, object]]:
    return [
        file_layer("small", sizes / f"{phase}_small.png"),
        file_layer("medium", sizes / f"{phase}_medium.png"),
        file_layer("large", sizes / f"{phase}_large.png"),
        file_layer("senescing", sizes / f"{phase}_senescing.png"),
        file_layer("snag", sizes / f"{phase}_snag.png"),
        file_layer("fallen", sizes / f"{phase}_fallen.png"),
        file_layer("artificial", sizes / f"{phase}_artificial.png"),
    ]


def phase_group(
    title: str,
    phase: str,
    resources: Path,
    sizes: Path,
    mist: Path,
    mist_variant: str,
    depth: Path | None = None,
) -> dict[str, object]:
    resource_children = resource_layers(resources, phase)
    if depth is not None:
        resource_children.append(file_layer(f"{phase}_depth_outliner", depth))
    resource_children.append(file_layer(f"{phase}_{mist_variant}", mist / f"{phase}_{mist_variant}.png"))
    return group(
        title,
        [
            group("resources", resource_children),
            group("sizes", size_layers(sizes, phase)),
        ],
    )


def build_manifest(bioenvelope_outline_path: Path, base_rgb_path: Path) -> dict[str, object]:
    current = OUTPUT_ROOT / "current"
    base = current / "base"
    bio = current / "bioenvelope"
    resources = current / "resources"
    mist = current / "outlines_mist"
    depth = current / "depth_outliner"
    ao = current / "ao"
    sizes = current / "sizes"

    return {
        "psd_path": str(PSD_PATH),
        "canvas_source": require(base_rgb_path),
        "document_name": PSD_PATH.stem,
        # Build bottom-to-top so new groups/layers added at the top land in the intended order.
        "stack": [
            group(
                "BASE WORLD",
                [
                    file_layer("base_rgb", base_rgb_path, "COLOR"),
                    file_layer("base ambient occlusion", ao / "existing_condition_ao_full.png", "MULTIPLY"),
                    file_layer("base_depth_filter", base / "base_outlines.png", "MULTIPLY"),
                ],
            ),
            group(
                "1. POSITIVE BIOENVELOPES",
                [
                    group(
                        "1.1 POSITIVE BIOENVELOPES MASKS",
                        [
                            file_layer("bioenvelope_exoskeleton", bio / "bioenvelope_exoskeleton.png"),
                            file_layer("bioenvelope_brownroof", bio / "bioenvelope_brownroof.png"),
                            file_layer("bioenvelope_otherground", bio / "bioenvelope_otherground.png"),
                            file_layer("bioenvelope_rewilded", bio / "bioenvelope_rewilded.png"),
                            file_layer("bioenvelope_footprintdepaved", bio / "bioenvelope_footprintdepaved.png"),
                            file_layer("bioenvelope_livingfacade", bio / "bioenvelope_livingfacade.png"),
                            file_layer("bioenvelope_greenroof", bio / "bioenvelope_greenroof.png"),
                            levels_layer("Levels - POSITIVE BIOENVELOPES MASKS"),
                        ],
                    ),
                    file_layer(
                        "positive bioenvelope shading",
                        bio / "bioenvelope_full-image.png",
                        "MULTIPLY",
                    ),
                    levels_layer("Levels - positive bioenvelope shading", clipped=True),
                    file_layer(
                        "bioenvelope_full-image outline",
                        bioenvelope_outline_path,
                        "MULTIPLY",
                    ),
                ],
            ),
            phase_group(
                "2. TRENDING",
                "trending",
                resources,
                sizes,
                mist,
                PSD_MIST_VARIANT,
            ),
            phase_group(
                "3. PATHWAY",
                "pathway",
                resources,
                sizes,
                mist,
                PSD_MIST_VARIANT,
                depth / "pathway_depth_outliner.png",
            ),
            phase_group(
                "4. PRIORITY",
                "priority",
                resources,
                sizes,
                mist,
                PSD_MIST_VARIANT,
                depth / "priority_depth_outliner.png",
            ),
        ],
    }


def build_jsx(config: dict[str, object]) -> str:
    config_literal = json.dumps(config)
    return f"""#target photoshop
app.displayDialogs = DialogModes.NO;

function blendModeFromName(name) {{
    var modes = {{
        "NORMAL": BlendMode.NORMAL,
        "MULTIPLY": BlendMode.MULTIPLY,
        "COLOR": BlendMode.COLORBLEND
    }};
    return modes[name] || BlendMode.NORMAL;
}}

function openCanvasDocument(path, docName) {{
    var src = app.open(new File(path));
    var width = src.width;
    var height = src.height;
    var res = src.resolution;
    src.close(SaveOptions.DONOTSAVECHANGES);
    try {{
        return app.documents.add(
            width,
            height,
            res,
            docName,
            NewDocumentMode.RGB,
            DocumentFill.TRANSPARENT,
            1.0,
            BitsPerChannelType.EIGHT,
            "sRGB IEC61966-2.1"
        );
    }} catch (e) {{
        return app.documents.add(width, height, res, docName, NewDocumentMode.RGB, DocumentFill.TRANSPARENT);
    }}
}}

function importFileLayer(doc, parent, spec) {{
    var src = app.open(new File(spec.path));
    var imported = src.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
    src.close(SaveOptions.DONOTSAVECHANGES);
    imported.move(parent, ElementPlacement.INSIDE);
    imported.name = spec.name;
    imported.blendMode = blendModeFromName(spec.blend_mode);
    imported.opacity = spec.opacity;
    imported.visible = true;
    return imported;
}}

function makeLevelsLayer(parent, spec) {{
    var layer = parent.artLayers.add();
    layer.name = spec.name;
    if (spec.clipped) {{
        try {{
            layer.grouped = true;
        }} catch (e) {{}}
    }}
    return layer;
}}

function buildNode(doc, parent, spec) {{
    if (spec.type === "group") {{
        var set = parent.layerSets.add();
        set.name = spec.name;
        for (var i = 0; i < spec.children.length; i++) {{
            buildNode(doc, set, spec.children[i]);
        }}
        return set;
    }}
    if (spec.type === "file") {{
        return importFileLayer(doc, parent, spec);
    }}
    if (spec.type === "levels") {{
        return makeLevelsLayer(parent, spec);
    }}
    throw new Error("Unknown spec type: " + spec.type);
}}

function savePsd(doc, path) {{
    var file = new File(path);
    if (file.exists) {{
        file.remove();
    }}
    var lowerPath = path.toLowerCase();
    if (lowerPath.match(/\\.psb$/)) {{
        var desc1 = new ActionDescriptor();
        var desc2 = new ActionDescriptor();
        desc2.putBoolean(stringIDToTypeID("maximizeCompatibility"), true);
        desc1.putObject(charIDToTypeID("As  "), charIDToTypeID("Pht8"), desc2);
        desc1.putPath(charIDToTypeID("In  "), file);
        desc1.putBoolean(charIDToTypeID("LwCs"), true);
        executeAction(charIDToTypeID("save"), desc1, DialogModes.NO);
        return;
    }}
    var psdOpts = new PhotoshopSaveOptions();
    psdOpts.layers = true;
    psdOpts.embedColorProfile = true;
    psdOpts.maximizeCompatibility = true;
    doc.saveAs(file, psdOpts, true, Extension.LOWERCASE);
}}

var config = {config_literal};
var doc = openCanvasDocument(config.canvas_source, config.document_name);
for (var i = 0; i < config.stack.length; i++) {{
    buildNode(doc, doc, config.stack[i]);
}}
savePsd(doc, config.psd_path);
"""


def run() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PSD_PATH.parent.mkdir(parents=True, exist_ok=True)
    base_rgb_path = normalize_png_for_photoshop(
        OUTPUT_ROOT / "current" / "base" / "base_rgb.png",
        OUTPUT_ROOT / "_photoshop_generated" / "base_rgb_photoshop.png",
    )
    outline_path = write_outline_from_alpha(
        OUTPUT_ROOT / "current" / "bioenvelope" / "bioenvelope_full-image.png",
        OUTPUT_ROOT / "_photoshop_generated" / "bioenvelope_full-image_outline.png",
    )
    config = build_manifest(outline_path, base_rgb_path)
    normalize_stack_file_paths(config["stack"], OUTPUT_ROOT / "_photoshop_generated" / "import_cache")

    with tempfile.TemporaryDirectory(prefix="edge_lab_psd_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        jsx_path = tmpdir_path / "build_stack.jsx"
        jsx_path.write_text(build_jsx(config))
        subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "{PHOTOSHOP_APP}"',
                "-e",
                "with timeout of 7200 seconds",
                "-e",
                f'do javascript file POSIX file "{jsx_path}"',
                "-e",
                "end timeout",
                "-e",
                "end tell",
            ],
            check=True,
        )

    print(f"[build_edge_lab_photoshop_stack] Wrote {PSD_PATH}")


if __name__ == "__main__":
    run()
