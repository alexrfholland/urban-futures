"""Build base_parade_timeline_2.psb at canvas-fit (7680x4320).

The original base_parade_timeline.psb was authored at 7441x2221 — when
placed into a 7680x4320 doc its SO lands centered with a ~120/1050px
offset. This rebuild replays generate_base_template.jsx's stack against
parade_timeline's latest base__* run but saves as PSB at the canonical
7680x4320 canvas so it aligns to (0,0) when placed elsewhere.

Stack (top -> bottom):
    base_depth_windowed_balanced_dense    (NORMAL, visible)
    base_depth_windowed_internal_refined  (NORMAL, hidden)
    Hue/Saturation 1                      (master=(0,-73,0))
    base_rgb                              (COLOR blend)
    existing_condition_ao_full            (NORMAL, base_white_render.png)

Usage:
    uv run python -m _futureSim_refactored.photoshopparser.build_base_parade_timeline_2
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "Adobe Photoshop 2026"
SCENE_DIR = REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / "4.12" / "parade_timeline"
DEFAULT_OUT = REPO_ROOT / "_data-refactored" / "_psds" / "psd-live" / "base_parade_timeline_2.psb"

RUN_RE = re.compile(r"^base__(?P<ts>\d{8}_\d{4,6})$")


def latest_base_run(scene_dir: Path) -> Path:
    candidates = [c for c in scene_dir.iterdir() if c.is_dir() and RUN_RE.match(c.name)]
    if not candidates:
        sys.exit(f"ERROR: no base__* runs in {scene_dir}")
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


JSX_TEMPLATE = r"""
(function () {
    var SRC = __SRC__;
    var OUT = __OUT__;
    var STACK = [
        { file: "base_white_render.png",
          name: "existing_condition_ao_full",
          blend: "Nrml", visible: true },
        { file: "base_rgb.png",
          name: "base_rgb",
          blend: "Clr ", visible: true },
        { kind: "huesat" },
        { file: "base_depth_windowed_internal_refined.png",
          name: "base_depth_windowed_internal_refined",
          blend: "Nrml", visible: false },
        { file: "base_depth_windowed_balanced_dense.png",
          name: "base_depth_windowed_balanced_dense",
          blend: "Nrml", visible: true },
    ];

    for (var s = 0; s < STACK.length; s++) {
        if (STACK[s].file) {
            var f = new File(SRC + "/" + STACK[s].file);
            if (!f.exists) return "ERROR: missing input " + f.fsName;
        }
    }

    while (app.documents.length > 0) {
        app.documents[0].close(SaveOptions.DONOTSAVECHANGES);
    }

    var savedUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    var doc = app.documents.add(
        new UnitValue(7680, "px"),
        new UnitValue(4320, "px"),
        72, "base_parade_timeline_2",
        NewDocumentMode.RGB,
        DocumentFill.TRANSPARENT,
        1.0, BitsPerChannelType.EIGHT);

    var report = [];

    for (var i = 0; i < STACK.length; i++) {
        var entry = STACK[i];
        if (entry.kind === "huesat") {
            addHueSaturationAdjustment(0, -73, 0);
            report.push("added HueSat (master=-73)");
            continue;
        }
        var pngPath = new File(SRC + "/" + entry.file);
        placeLinkedFromFile(pngPath);
        var layer = doc.activeLayer;
        layer.name = entry.name;
        layer.blendMode = blendModeFromCode(entry.blend);
        layer.visible = entry.visible;
        report.push("placed " + entry.name);
    }

    // Remove any starter "Layer N" transparent layer.
    for (var j = doc.artLayers.length - 1; j >= 0; j--) {
        var L = doc.artLayers[j];
        var recognised = false;
        for (var k = 0; k < STACK.length; k++) {
            if (STACK[k].name && L.name === STACK[k].name) { recognised = true; break; }
        }
        if (!recognised && L.kind === LayerKind.NORMAL && L.name.indexOf("Layer") === 0) {
            L.remove();
            report.push("removed starter layer: " + L.name);
        }
    }

    // Save as PSB (largeDocumentFormat).
    var saveDesc = new ActionDescriptor();
    var saveOpts = new ActionDescriptor();
    saveOpts.putBoolean(stringIDToTypeID("maximizeCompatibility"), true);
    saveDesc.putObject(
        stringIDToTypeID("as"),
        stringIDToTypeID("largeDocumentFormat"),
        saveOpts
    );
    saveDesc.putPath(stringIDToTypeID("in"), new File(OUT));
    saveDesc.putBoolean(stringIDToTypeID("lowerCase"), true);
    executeAction(stringIDToTypeID("save"), saveDesc, DialogModes.NO);
    report.push("SAVED " + OUT);

    app.preferences.rulerUnits = savedUnits;
    return report.join("\n");

    function placeLinkedFromFile(file) {
        var d = new ActionDescriptor();
        d.putPath(charIDToTypeID("null"), file);
        d.putBoolean(stringIDToTypeID("linked"), true);
        executeAction(charIDToTypeID("Plc "), d, DialogModes.NO);
    }

    function blendModeFromCode(c) {
        if (c === "Clr ") return BlendMode.COLORBLEND;
        return BlendMode.NORMAL;
    }

    function addHueSaturationAdjustment(hue, sat, light) {
        var d = new ActionDescriptor();
        var ref = new ActionReference();
        ref.putClass(stringIDToTypeID("adjustmentLayer"));
        d.putReference(charIDToTypeID("null"), ref);
        var using = new ActionDescriptor();
        var adj = new ActionDescriptor();
        adj.putBoolean(stringIDToTypeID("colorize"), false);
        using.putObject(charIDToTypeID("Type"), stringIDToTypeID("hueSaturation"), adj);
        d.putObject(charIDToTypeID("Usng"), stringIDToTypeID("adjustmentLayer"), using);
        executeAction(charIDToTypeID("Mk  "), d, DialogModes.NO);

        var set = new ActionDescriptor();
        var setRef = new ActionReference();
        setRef.putEnumerated(charIDToTypeID("AdjL"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
        set.putReference(charIDToTypeID("null"), setRef);
        var p = new ActionDescriptor();
        p.putBoolean(stringIDToTypeID("colorize"), false);
        var list = new ActionList();
        var entry = new ActionDescriptor();
        entry.putInteger(charIDToTypeID("H   "), hue);
        entry.putInteger(charIDToTypeID("Strt"), sat);
        entry.putInteger(charIDToTypeID("Lght"), light);
        list.putObject(stringIDToTypeID("hueSatAdjustmentV2"), entry);
        p.putList(stringIDToTypeID("adjustment"), list);
        set.putObject(charIDToTypeID("T   "), stringIDToTypeID("hueSaturation"), p);
        executeAction(charIDToTypeID("setd"), set, DialogModes.NO);
    }
})();
"""

JXA_TEMPLATE = r"""
ObjC.import('Foundation');
var jsxSrc = {jsx_literal};
var ps = Application({app_literal});
ps.includeStandardAdditions = true;
ps.activate();
var result = ps.doJavascript(jsxSrc);
result === undefined ? "" : String(result);
"""


def _js_literal(s: str) -> str:
    escaped = (s.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r"))
    return '"' + escaped + '"'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=None,
                    help="parade_timeline base__<ts> dir (default: latest)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--app", default=APP_NAME)
    args = ap.parse_args()

    src = (args.src or latest_base_run(SCENE_DIR)).resolve()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"src: {src}")
    print(f"out: {out}")

    jsx_src = (
        JSX_TEMPLATE
        .replace("__SRC__", _js_literal(str(src)))
        .replace("__OUT__", _js_literal(str(out)))
    )
    script = JXA_TEMPLATE.format(
        jsx_literal=_js_literal(jsx_src),
        app_literal=_js_literal(args.app),
    )
    proc = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", script],
        capture_output=True, text=True,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
