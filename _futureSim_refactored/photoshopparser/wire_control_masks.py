"""Wire per-branch desaturation masks into parade_library.psb.

For each top-level branch group (positive / priority / trending) this inserts
a Hue/Sat adjustment layer (saturation -100) immediately above the branch's
`sizes` group, clips it to `sizes`, and loads the compositor `control.png`
into the adjustment layer's mask channel. The mask grey values are the
per-class desat strengths (0.8 street / 0.5 park / 0.0 reserve+improved).

Idempotent: a pre-existing `sizes_desat` adjustment layer in a branch is
removed and recreated so the mask always reflects the current PNG.

Usage:
    uv run python -m _futureSim_refactored.photoshopparser.wire_control_masks \\
        --psb     _data-refactored/_psds/psd-live/parade_library.psb \\
        --control-root _data-refactored/compositor/outputs/_library/parade_library \\
        --run-ts  20260415_2037
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "Adobe Photoshop 2026"

BRANCHES = [
    # (top-level PSB group name, control-run branch tail)
    ("positive", "positive"),
    ("priority", "positive-priority"),
    ("trending", "trending"),
]

ADJ_LAYER_NAME = "sizes_desat"

JSX_TEMPLATE = r"""
(function () {
    var BRANCHES = __BRANCHES__;
    var ADJ_NAME = __ADJ_NAME__;
    if (app.documents.length === 0) return "ERROR: no document open";
    var doc = app.activeDocument;
    var report = [];

    // Need a non-group top-level layer to park activeLayer on before
    // creating adjustment layers: if activeLayer is a LayerSet, PS puts
    // the new adj INSIDE that group. We use the bottom "Background" layer.
    var anchor = null;
    for (var ai = 0; ai < doc.layers.length; ai++) {
        if (doc.layers[ai].typename !== "LayerSet") { anchor = doc.layers[ai]; break; }
    }
    if (!anchor) return "ERROR: no non-LayerSet layer to anchor on";

    for (var b = 0; b < BRANCHES.length; b++) {
        var br = BRANCHES[b];
        var branchGroup = findChildSet(doc, br.group);
        if (!branchGroup) { report.push("MISS branch: " + br.group); continue; }
        var sizesGroup = findChildSet(branchGroup, "sizes");
        if (!sizesGroup) { report.push("MISS sizes in " + br.group); continue; }

        // Sweep strays from previous runs. Two places they can sit:
        //   - inside branchGroup itself (intended home)
        //   - inside sizesGroup (mis-landing when activeLayer was a group)
        // branchGroup's rightful children are only LayerSets (outlines, sizes).
        // sizesGroup's rightful children are only smart objects.
        for (var si = branchGroup.layers.length - 1; si >= 0; si--) {
            var c1 = branchGroup.layers[si];
            if (c1.typename === "LayerSet") continue;
            report.push("SWEEP " + br.group + "/" + c1.name + " [" + c1.typename + "]");
            c1.remove();
        }
        for (var sj = sizesGroup.layers.length - 1; sj >= 0; sj--) {
            var c2 = sizesGroup.layers[sj];
            if (c2.kind === LayerKind.SMARTOBJECT) continue;
            report.push("SWEEP " + br.group + "/sizes/" + c2.name
                        + " [" + c2.typename + "/" + c2.kind + "]");
            c2.remove();
        }

        // Park active on a non-group so the new adj lands at doc top level,
        // then we move it to its true home.
        doc.activeLayer = anchor;

        // Create Hue/Sat adjustment layer.
        var mk = new ActionDescriptor();
        var mkRef = new ActionReference();
        mkRef.putClass(stringIDToTypeID("adjustmentLayer"));
        mk.putReference(stringIDToTypeID("null"), mkRef);
        var using = new ActionDescriptor();
        var typ = new ActionDescriptor();
        typ.putEnumerated(
            stringIDToTypeID("presetKind"),
            stringIDToTypeID("presetKindType"),
            stringIDToTypeID("presetKindDefault")
        );
        typ.putBoolean(stringIDToTypeID("colorize"), false);
        using.putObject(
            stringIDToTypeID("type"),
            stringIDToTypeID("hueSaturation"),
            typ
        );
        mk.putObject(
            stringIDToTypeID("using"),
            stringIDToTypeID("adjustmentLayer"),
            using
        );
        executeAction(stringIDToTypeID("make"), mk, DialogModes.NO);

        var adj = doc.activeLayer;
        adj.name = ADJ_NAME;

        // Set master saturation = -100.
        var setD = new ActionDescriptor();
        var setR = new ActionReference();
        setR.putEnumerated(
            stringIDToTypeID("adjustmentLayer"),
            stringIDToTypeID("ordinal"),
            stringIDToTypeID("targetEnum")
        );
        setD.putReference(stringIDToTypeID("null"), setR);
        var hsl = new ActionDescriptor();
        hsl.putBoolean(stringIDToTypeID("colorize"), false);
        var adjList = new ActionList();
        var master = new ActionDescriptor();
        master.putInteger(stringIDToTypeID("hue"), 0);
        master.putInteger(stringIDToTypeID("saturation"), -100);
        master.putInteger(stringIDToTypeID("lightness"), 0);
        adjList.putObject(stringIDToTypeID("hueSatAdjustmentV2"), master);
        hsl.putList(stringIDToTypeID("adjustment"), adjList);
        setD.putObject(
            stringIDToTypeID("to"),
            stringIDToTypeID("hueSaturation"),
            hsl
        );
        executeAction(stringIDToTypeID("set"), setD, DialogModes.NO);

        // Move the adj into branchGroup, immediately above the sizes group.
        adj.move(sizesGroup, ElementPlacement.PLACEBEFORE);

        // Clip to sizes beneath (layer clipping mask on the group below).
        adj.grouped = true;

        // Load PNG into the mask channel.
        var maskDoc = app.open(new File(br.png));
        maskDoc.selection.selectAll();
        maskDoc.selection.copy();
        maskDoc.close(SaveOptions.DONOTSAVECHANGES);
        // app.activeDocument reverts to the PSB automatically after close().
        doc = app.activeDocument;
        doc.activeLayer = adj;

        // Target the layer mask channel. makeVisible must be true so the
        // raw "paste" action routes pixels into the mask instead of
        // creating a new image layer.
        var selMask = new ActionDescriptor();
        var selMaskRef = new ActionReference();
        selMaskRef.putEnumerated(
            stringIDToTypeID("channel"),
            stringIDToTypeID("channel"),
            stringIDToTypeID("mask")
        );
        selMask.putReference(stringIDToTypeID("null"), selMaskRef);
        selMask.putBoolean(stringIDToTypeID("makeVisible"), true);
        executeAction(stringIDToTypeID("select"), selMask, DialogModes.NO);

        // Raw paste action honours the current channel target; DOM
        // doc.paste() always creates a new image layer.
        executeAction(charIDToTypeID("past"), undefined, DialogModes.NO);

        // Deselect the pasted region.
        doc.selection.deselect();

        // Flip back to composite channel so subsequent ops aren't mask-scoped.
        var selRGB = new ActionDescriptor();
        var selRGBRef = new ActionReference();
        selRGBRef.putEnumerated(
            stringIDToTypeID("channel"),
            stringIDToTypeID("channel"),
            stringIDToTypeID("RGB")
        );
        selRGB.putReference(stringIDToTypeID("null"), selRGBRef);
        executeAction(stringIDToTypeID("select"), selRGB, DialogModes.NO);

        report.push("WIRED " + br.group + "/" + ADJ_NAME + " mask=" + br.png);
    }

    doc.save();
    report.push("SAVED");
    return report.join("\n");

    function findChildSet(parent, name) {
        for (var i = 0; i < parent.layers.length; i++) {
            var L = parent.layers[i];
            if (L.typename === "LayerSet" && L.name === name) return L;
        }
        return null;
    }
    function findChild(parent, name) {
        for (var i = 0; i < parent.layers.length; i++) {
            if (parent.layers[i].name === name) return parent.layers[i];
        }
        return null;
    }
})();
"""

JXA_TEMPLATE = r"""
ObjC.import('Foundation');
var psbPath = {psb_literal};
var jsxSrc  = {jsx_literal};
var ps = Application({app_literal});
ps.includeStandardAdditions = true;
ps.activate();
ps.open(Path(psbPath));
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
    ap.add_argument("--psb", required=True, type=Path)
    ap.add_argument("--control-root", required=True, type=Path,
                    dest="control_root",
                    help="compositor outputs dir containing "
                         "control__<ts>__<branch>/control.png")
    ap.add_argument("--run-ts", required=True, dest="run_ts",
                    help="timestamp tag of the control__* run dirs")
    ap.add_argument("--app", default=APP_NAME)
    args = ap.parse_args()

    args.psb = args.psb.resolve()
    args.control_root = args.control_root.resolve()
    if not args.psb.exists():
        sys.exit(f"ERROR: PSB not found: {args.psb}")

    branch_entries = []
    for group_name, run_branch in BRANCHES:
        png = args.control_root / f"control__{args.run_ts}__{run_branch}" / "control.png"
        if not png.exists():
            sys.exit(f"ERROR: control PNG missing: {png}")
        branch_entries.append({"group": group_name, "png": str(png)})

    branch_literal = "[" + ",".join(
        f'{{group:{_js_literal(e["group"])},png:{_js_literal(e["png"])}}}'
        for e in branch_entries
    ) + "]"
    jsx_src = (
        JSX_TEMPLATE
        .replace("__BRANCHES__", branch_literal)
        .replace("__ADJ_NAME__", _js_literal(ADJ_LAYER_NAME))
    )

    script = JXA_TEMPLATE.format(
        psb_literal=_js_literal(str(args.psb)),
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
