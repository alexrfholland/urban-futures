"""Relink the 4 smart objects inside a base PSB to compositor base PNGs.

The base PSB's 4 SOs keep their template links to whichever base folder
generated base_template.psd. This repoints them at the latest `base__<ts>/`
folder under `compositor/outputs/<sim_root>/<exr_family>/`.

`--psb` and `--exr-family` are independent on purpose: the PSB being edited
and the compositor family feeding it are conceptually different, and for
tests you may want to edit one PSB with another variant's compositor output.

Usage:
    uv run python -m _futureSim_refactored.photoshopparser.relink_base_psb \\
        --psb _data-refactored/_psds/psd-live/base_parade_single-state_yr180.psb \\
        --exr-family parade_single-state_yr180 \\
        [--sim-root 4.12]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PSD_LIVE = REPO_ROOT / "_data-refactored" / "_psds" / "psd-live"
COMPOSITOR_ROOT = REPO_ROOT / "_data-refactored" / "compositor" / "outputs"

# Layer name -> source PNG stem
SO_MAP = {
    "existing_condition_ao_full": "base_white_render.png",
    "base_rgb": "base_rgb.png",
    "base_depth_windowed_internal_refined": "base_depth_windowed_internal_refined.png",
    "base_depth_windowed_balanced_dense": "base_depth_windowed_balanced_dense.png",
}

BASE_RUN_RE = re.compile(r"^base__\d{8}_\d{6}(?:__.+)?$")

APP_NAME = "Adobe Photoshop 2026"

JSX_TEMPLATE = r"""
(function () {
    if (app.documents.length === 0) return "ERROR: no document open";
    var doc = app.activeDocument;
    var MAPPING = __MAPPING__;
    var report = [];

    for (var i = 0; i < MAPPING.length; i++) {
        var entry = MAPPING[i];
        var layer = findChild(doc, entry.name);
        if (!layer) { report.push("MISSING: " + entry.name); continue; }
        var f = new File(entry.file);
        if (!f.exists) { report.push("MISSING FILE: " + entry.file); continue; }
        try {
            doc.activeLayer = layer;
            relinkToFile(f);
            report.push("RELINKED: " + entry.name + " -> " + entry.file);
        } catch (e) {
            report.push("ERROR: " + entry.name + " -> " + e);
        }
    }

    doc.save();
    report.push("SAVED");
    return report.join("\n");

    function findChild(parent, name) {
        var layers = parent.layers;
        for (var k = 0; k < layers.length; k++) {
            if (layers[k].name === name) return layers[k];
        }
        return null;
    }

    function relinkToFile(file) {
        var desc = new ActionDescriptor();
        desc.putPath(stringIDToTypeID("null"), file);
        executeAction(stringIDToTypeID("placedLayerRelinkToFile"), desc, DialogModes.NO);
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


def find_latest_base_folder(exr_family: str, sim_root: str) -> Path:
    parent = COMPOSITOR_ROOT / sim_root / exr_family
    if not parent.is_dir():
        sys.exit(f"ERROR: compositor dir not found: {parent}")
    candidates = [p for p in parent.iterdir() if p.is_dir() and BASE_RUN_RE.match(p.name)]
    if not candidates:
        sys.exit(f"ERROR: no base__<ts>/ folder under {parent}")
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--psb", required=True, type=Path,
                    help="Path to the base PSB to edit "
                         "(e.g. _data-refactored/_psds/psd-live/base_<variant>.psb)")
    ap.add_argument("--exr-family", required=True, dest="exr_family",
                    help="Compositor family folder under "
                         "compositor/outputs/<sim_root>/ whose latest "
                         "base__<ts>/ provides the 4 PNGs "
                         "(e.g. parade_single-state_yr180, uni_timeline).")
    ap.add_argument("--sim-root", default="4.12")
    ap.add_argument("--app", default=APP_NAME)
    args = ap.parse_args()

    psb = args.psb.resolve()
    if not psb.exists():
        sys.exit(f"ERROR: base PSB not found: {psb}")

    base_dir = find_latest_base_folder(args.exr_family, args.sim_root)
    print(f"base folder: {base_dir.relative_to(REPO_ROOT)}")

    mapping = []
    for name, png in SO_MAP.items():
        f = base_dir / png
        if not f.exists():
            sys.exit(f"ERROR: missing PNG: {f}")
        mapping.append({"name": name, "file": str(f)})

    mapping_literal = (
        "[" + ",".join(
            f'{{name:{_js_literal(m["name"])},file:{_js_literal(m["file"])}}}'
            for m in mapping
        ) + "]"
    )
    jsx_src = JSX_TEMPLATE.replace("__MAPPING__", mapping_literal)

    script = JXA_TEMPLATE.format(
        psb_literal=_js_literal(str(psb)),
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
