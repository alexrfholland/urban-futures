from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia")
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data"
    / "blender"
    / "2026"
    / "edge_detection_lab"
    / "outputs"
    / "edge_lab_final_template_parade_8k_network_20260402"
)
OUTPUT_ROOT = Path(os.environ.get("EDGE_LAB_PHOTOSHOP_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).expanduser()
PROPOSAL_ROOT = Path(
    os.environ.get(
        "EDGE_LAB_PROPOSAL_OUTPUT_ROOT",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "proposal_release_control_pathway_debug_20260402"),
    )
).expanduser()
DEPTH_OUTLINER = Path(
    os.environ.get(
        "EDGE_LAB_PATHWAY_DEPTH_OUTLINER",
        str(REPO_ROOT / "data" / "blender" / "2026" / "edge_detection_lab" / "outputs" / "parade_8k_current_depth_and_mist_outliner_20260402" / "pathway_depth_outliner.png"),
    )
).expanduser()
PSD_PATH = Path(
    os.environ.get(
        "EDGE_LAB_PHOTOSHOP_PSD_PATH",
        str(OUTPUT_ROOT / "edge_lab_parade_8k_network_pathway_proposals_20260402.psd"),
    )
).expanduser()
PHOTOSHOP_APP = os.environ.get("EDGE_LAB_PHOTOSHOP_APP", "Adobe Photoshop 2026")


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


def group(name: str, children: list[dict[str, object]]) -> dict[str, object]:
    return {
        "type": "group",
        "name": name,
        "children": children,
    }


def base_world_group() -> dict[str, object]:
    current = OUTPUT_ROOT / "current"
    base = current / "base"
    ao = current / "ao"
    return group(
        "BASE WORLD",
        [
            file_layer("base ambient occlusion", ao / "existing_condition_ao_full.png", "MULTIPLY"),
            file_layer("base_rgb", base / "base_rgb.png", "COLOR"),
            file_layer("base_rgb copy", base / "base_rgb.png", "MULTIPLY", 40),
            file_layer("base_depth_filter", base / "base_outlines.png", "MULTIPLY"),
            file_layer("base_depth_windowed_balanced_dense", base / "base_depth_windowed_balanced_dense.png"),
        ],
    )


def proposal_group(title: str, prefix: str) -> dict[str, object]:
    members = sorted(PROPOSAL_ROOT.glob(f"{prefix}-*_0001.png"))
    if not members:
        raise FileNotFoundError(f"No proposal outputs found for {prefix} in {PROPOSAL_ROOT}")
    return group(
        title,
        [file_layer(member.stem.replace("_0001", ""), member) for member in members],
    )


def build_manifest() -> dict[str, object]:
    base_rgb_path = OUTPUT_ROOT / "current" / "base" / "base_rgb.png"
    stack = [
        base_world_group(),
        proposal_group("pathway - proposal-release-control", "proposal-release-control"),
        proposal_group("pathway - proposal-colonise", "proposal-colonise"),
        proposal_group("pathway - proposal-recruit", "proposal-recruit"),
        proposal_group("pathway - proposal-decay", "proposal-decay"),
        proposal_group("pathway - proposal-deploy-structure", "proposal-deploy-structure"),
        file_layer("pathway_depth_outliner", DEPTH_OUTLINER, "NORMAL"),
    ]
    return {
        "psd_path": str(PSD_PATH),
        "canvas_source": require(base_rgb_path),
        "document_name": PSD_PATH.stem,
        "stack": stack,
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
    config = build_manifest()

    with tempfile.TemporaryDirectory(prefix="edge_lab_pathway_psd_") as tmpdir:
        jsx_path = Path(tmpdir) / "build_pathway_stack.jsx"
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
    print(f"[build_pathway_proposals_photoshop_stack] Wrote {PSD_PATH}")


if __name__ == "__main__":
    run()
