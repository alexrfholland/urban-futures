from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
TIMESLICE_ROOT = REPO_ROOT / "_outputs-refactored" / "Blender" / "timeslices"
PHOTOSHOP_APP = os.environ.get("EDGE_LAB_PHOTOSHOP_APP", "Adobe Photoshop 2026")

SITE_OUTPUTS = {
    "city": REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "edge_detection_lab"
    / "outputs"
    / "edge_lab_final_template_city_8k_network_20260402"
    / "current",
    "street": REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "edge_detection_lab"
    / "outputs"
    / "edge_lab_final_template_street_8k_network_20260402"
    / "current",
}


def require(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def layer_spec(name: str, path: Path, blend_mode: str = "NORMAL", opacity: int = 100) -> dict[str, object]:
    return {
        "name": name,
        "path": require(path),
        "blend_mode": blend_mode,
        "opacity": opacity,
    }


def build_config() -> dict[str, object]:
    sites = []
    for site, current_root in SITE_OUTPUTS.items():
        base_root = current_root / "base"
        ao_root = current_root / "ao"
        canvas_source = ao_root / "existing_condition_ao_full.png"
        sites.append(
            {
                "site": site,
                "output_path": str(TIMESLICE_ROOT / f"{site}_timeline_base.psb"),
                "canvas_source": require(canvas_source),
                "layers": [
                    layer_spec("base ambient occlusion", ao_root / "existing_condition_ao_full.png", "MULTIPLY", 100),
                    layer_spec("base_rgb", base_root / "base_rgb.png", "COLOR", 100),
                    # Match the extracted PSB's effective 51/255 opacity.
                    layer_spec("base_rgb copy", base_root / "base_rgb.png", "MULTIPLY", 20),
                    layer_spec("base_depth_windowed_balanced_dense", base_root / "base_depth_windowed_balanced_dense.png", "NORMAL", 100),
                    layer_spec("base_depth_filter", base_root / "base_outlines.png", "MULTIPLY", 100),
                ],
            }
        )
    return {
        "sites": sites,
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

function makeDocFromSource(path, docName) {{
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
        return app.documents.add(
            width,
            height,
            res,
            docName,
            NewDocumentMode.RGB,
            DocumentFill.TRANSPARENT
        );
    }}
}}

function importFileLayer(doc, spec) {{
    var src = app.open(new File(spec.path));
    var imported = src.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
    src.close(SaveOptions.DONOTSAVECHANGES);
    imported.name = spec.name;
    imported.blendMode = blendModeFromName(spec.blend_mode);
    imported.opacity = spec.opacity;
    imported.visible = true;
    return imported;
}}

function saveDoc(doc, path) {{
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

for (var s = 0; s < config.sites.length; s++) {{
    var site = config.sites[s];
    var doc = makeDocFromSource(site.canvas_source, site.site + "_timeline_base");
    for (var i = 0; i < site.layers.length; i++) {{
        importFileLayer(doc, site.layers[i]);
    }}
    for (var l = doc.layers.length - 1; l >= 0; l--) {{
        if (doc.layers[l].name === "Layer 1") {{
            doc.layers[l].remove();
        }}
    }}
    saveDoc(doc, site.output_path);
    doc.close(SaveOptions.DONOTSAVECHANGES);
}}
"""


def run() -> None:
    config = build_config()
    TIMESLICE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edge_lab_timeline_base_") as tmpdir:
        jsx_path = Path(tmpdir) / "build_timeline_base.jsx"
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
    for site in SITE_OUTPUTS:
        print(f"[build_timeline_base_files] Wrote {TIMESLICE_ROOT / f'{site}_timeline_base.psb'}")


if __name__ == "__main__":
    run()
