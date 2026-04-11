from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
OUTPUTS_ROOT = REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs"
TIMESLICE_ROOT = REPO_ROOT / "_outputs-refactored" / "Blender" / "timeslices"
PHOTOSHOP_APP = os.environ.get("EDGE_LAB_PHOTOSHOP_APP", "Adobe Photoshop 2026")

SITES = ("city", "street")


def require(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def file_layer(name: str, path: Path, blend_mode: str = "NORMAL", opacity: int = 100, visible: bool = True) -> dict[str, object]:
    return {
        "type": "file",
        "name": name,
        "path": require(path),
        "blend_mode": blend_mode,
        "opacity": opacity,
        "visible": visible,
    }


def group(name: str, children: list[dict[str, object]], visible: bool = True, opacity: int = 100) -> dict[str, object]:
    return {
        "type": "group",
        "name": name,
        "visible": visible,
        "opacity": opacity,
        "children": children,
    }


def output_root(site: str) -> Path:
    return OUTPUTS_ROOT / f"edge_lab_final_template_{site}_8k_network_20260402" / "current"


def size_group(site: str, phase: str) -> dict[str, object]:
    root = output_root(site) / "sizes"
    return group(
        "tree sizes",
        [
            file_layer("artificial", root / f"{phase}_artificial.png"),
            file_layer("fallen", root / f"{phase}_fallen.png"),
            file_layer("snag", root / f"{phase}_snag.png"),
            file_layer("senescing", root / f"{phase}_senescing.png"),
            file_layer("large", root / f"{phase}_large.png"),
            file_layer("medium", root / f"{phase}_medium.png"),
            file_layer("small", root / f"{phase}_small.png"),
        ],
    )


def proposal_groups(site: str, phase: str, visible: bool) -> dict[str, object]:
    root = output_root(site) / "proposals" / phase
    return group(
        "proposals",
        [
            group(
                "proposal-deploy-structure",
                [
                    file_layer("adapt-utility-pole", root / "proposal-deploy-structure-adapt-utility-pole.png"),
                    file_layer("translocated-log", root / "proposal-deploy-structure-translocated-log.png"),
                    file_layer("upgrade-feature", root / "proposal-deploy-structure-upgrade-feature.png"),
                ],
            ),
            group(
                "proposal-decay",
                [
                    file_layer("buffer-feature", root / "proposal-decay-buffer-feature.png"),
                    file_layer("brace-feature", root / "proposal-decay-brace-feature.png"),
                ],
            ),
            group(
                "proposal-recruit",
                [
                    file_layer("buffer-feature", root / "proposal-recruit-buffer-feature.png"),
                    file_layer("rewild-ground", root / "proposal-recruit-rewild-ground.png"),
                ],
            ),
            group(
                "proposal-colonise",
                [
                    file_layer("rewild-ground", root / "proposal-colonise-rewild-ground.png"),
                    file_layer("enrich-envelope", root / "proposal-colonise-enrich-envelope.png"),
                    file_layer("roughen-envelope", root / "proposal-colonise-roughen-envelope.png"),
                ],
            ),
            group(
                "proposal-release-control",
                [
                    file_layer("rejected", root / "proposal-release-control-rejected.png"),
                    file_layer("reduce-pruning", root / "proposal-release-control-reduce-pruning.png"),
                    file_layer("eliminate-pruning", root / "proposal-release-control-eliminate-pruning.png"),
                ],
            ),
        ],
        visible=visible,
        opacity=80 if visible else 100,
    )


def build_site_manifest(site: str) -> dict[str, object]:
    current = output_root(site)
    outliners = current / "outliners"
    canvas_source = current / "proposals" / "pathway" / "proposal-deploy-structure-adapt-utility-pole.png"
    return {
        "site": site,
        "psd_path": str(TIMESLICE_ROOT / f"timeslice-{site}.psd"),
        "canvas_source": require(canvas_source),
        "stack": [
            group(
                "trending",
                [
                    size_group(site, "trending"),
                    proposal_groups(site, "trending", visible=False),
                    file_layer("trending_depth_outliner", outliners / "trending_depth_outliner.png", "NORMAL", 50),
                ],
                visible=False,
            ),
            group(
                "positive",
                [
                    proposal_groups(site, "pathway", visible=True),
                    size_group(site, "pathway"),
                    file_layer("base_sim-turns", current / "base" / "base_sim-turns.png", "SCREEN", 50),
                    file_layer("pathway_depth_outliner", outliners / "pathway_depth_outliner.png", "NORMAL", 50),
                    file_layer("priority_depth_outliner", outliners / "priority_depth_outliner.png", "NORMAL", 60),
                ],
            ),
        ],
    }


def build_jsx(configs: list[dict[str, object]]) -> str:
    config_literal = json.dumps(configs)
    return f"""#target photoshop
app.displayDialogs = DialogModes.NO;

function blendModeFromName(name) {{
    var modes = {{
        "NORMAL": BlendMode.NORMAL,
        "MULTIPLY": BlendMode.MULTIPLY,
        "COLOR": BlendMode.COLORBLEND,
        "SCREEN": BlendMode.SCREEN
    }};
    return modes[name] || BlendMode.NORMAL;
}}

function openCanvasDocument(path, docName) {{
    var src = app.open(new File(path));
    var width = src.width;
    var height = src.height;
    var res = src.resolution;
    src.close(SaveOptions.DONOTSAVECHANGES);
    var doc;
    try {{
        doc = app.documents.add(
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
        doc = app.documents.add(width, height, res, docName, NewDocumentMode.RGB, DocumentFill.TRANSPARENT);
    }}
    if (doc.layers.length === 1) {{
        try {{
            doc.activeLayer.remove();
        }} catch (e2) {{
        }}
    }}
    return doc;
}}

function importFileLayer(doc, parent, spec) {{
    var src = app.open(new File(spec.path));
    var imported = src.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
    src.close(SaveOptions.DONOTSAVECHANGES);
    imported.move(parent, ElementPlacement.INSIDE);
    imported.name = spec.name;
    imported.blendMode = blendModeFromName(spec.blend_mode);
    imported.opacity = spec.opacity;
    imported.visible = spec.visible;
    return imported;
}}

function buildNode(doc, parent, spec) {{
    if (spec.type === "group") {{
        var set = parent.layerSets.add();
        set.name = spec.name;
        set.visible = spec.visible;
        try {{
            set.opacity = spec.opacity;
        }} catch (e) {{
        }}
        for (var i = 0; i < spec.children.length; i++) {{
            buildNode(doc, set, spec.children[i]);
        }}
        return set;
    }}
    if (spec.type === "file") {{
        return importFileLayer(doc, parent, spec);
    }}
    throw new Error("Unknown spec type: " + spec.type);
}}

function savePsd(doc, path) {{
    var file = new File(path);
    if (file.exists) {{
        file.remove();
    }}
    var psdOpts = new PhotoshopSaveOptions();
    psdOpts.layers = true;
    psdOpts.embedColorProfile = true;
    psdOpts.maximizeCompatibility = true;
    doc.saveAs(file, psdOpts, true, Extension.LOWERCASE);
}}

function removeDefaultLayer(doc) {{
    for (var i = doc.layers.length - 1; i >= 0; i--) {{
        var layer = doc.layers[i];
        if (!layer.typename || layer.typename !== "ArtLayer") {{
            continue;
        }}
        if (layer.name === "Layer 1") {{
            try {{
                layer.remove();
            }} catch (e) {{
            }}
            return;
        }}
    }}
}}

var configs = {config_literal};
for (var c = 0; c < configs.length; c++) {{
    var config = configs[c];
    var doc = openCanvasDocument(config.canvas_source, "timeslice-" + config.site);
    for (var i = 0; i < config.stack.length; i++) {{
        buildNode(doc, doc, config.stack[i]);
    }}
    removeDefaultLayer(doc);
    savePsd(doc, config.psd_path);
    doc.close(SaveOptions.DONOTSAVECHANGES);
}}
"""


def run() -> None:
    TIMESLICE_ROOT.mkdir(parents=True, exist_ok=True)
    configs = [build_site_manifest(site) for site in SITES]
    with tempfile.TemporaryDirectory(prefix="edge_lab_timeslice_site_") as tmpdir:
        jsx_path = Path(tmpdir) / "build_timeslices.jsx"
        jsx_path.write_text(build_jsx(configs))
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
    for site in SITES:
        print(f"[build_timeslice_site_psds] Wrote {TIMESLICE_ROOT / f'timeslice-{site}.psd'}")


if __name__ == "__main__":
    run()
