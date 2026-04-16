"""Build interventions_bioenvelope_ref.psb from scratch.

A four-scene reference PSB with identical sub-structure per scene
(stack top -> bottom):
  <scene>/outlines/                  -> 1 PNG (positive_bioenvelope_outlines-depth)
  <scene>/proposals/                 -> 13 PNGs from proposal_and_interventions__*_positive
  <scene>/interventions_bioenvelope/ -> 8 PNGs from intervention_int__*_positive
                                        (minus the combined png)
  <scene>/base/                      -> single linked SO -> base_<exr_family>.psb

Scenes (group name -> compositor exr_family under 4.12):
    city_single_state    <- city_single-state_yr180
    city_timeline        <- city_timeline
    parade_single_state  <- parade_single-state_yr180
    parade_timeline      <- parade_timeline

One layer comp per top-level group, owner-gated so each comp shows only
its own scene.

Usage:
    uv run python -m _futureSim_refactored.photoshopparser.build_interventions_bioenvelope_ref
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "Adobe Photoshop 2026"
COMPOSITOR_ROOT = REPO_ROOT / "_data-refactored" / "compositor" / "outputs" / "4.12"
PSD_LIVE = REPO_ROOT / "_data-refactored" / "_psds" / "psd-live"
LINKED_ROOT = REPO_ROOT / "_data-refactored" / "_psds" / "linked_pngs" / "interventions_bioenvelope_ref"
DEFAULT_OUT = PSD_LIVE / "_interventions_bioenvelope_ref.psb"

SCENES = [
    # (group name in PSB, compositor exr_family)
    ("city_single_state",   "city_single-state_yr180"),
    ("city_timeline",       "city_timeline"),
    ("parade_single_state", "parade_single-state_yr180"),
    ("parade_timeline",     "parade_timeline"),
]

# Per-scene base template overrides. Default is base_<exr_family>.psb;
# parade_timeline's original was clipped at 7441x2221, so we point at the
# canvas-fit rebuild from build_base_parade_timeline_2.py.
BASE_TEMPLATE_OVERRIDES = {
    "parade_timeline": "base_parade_timeline_2.psb",
}

# Outlines slots we want from the bioenvelope run (flat layout).
OUTLINE_SLOTS = [
    "positive_bioenvelope_outlines-depth.png",
]

# intervention_int slots — all except the combined one.
INTERVENTION_SKIP = "interventions_bioenvelope.png"

RUN_RE = re.compile(r"^(?P<family>[a-z_]+)__(?P<ts>\d{8}_\d{4,6})(?:__(?P<tail>.+))?$")


def find_latest_run(scene_dir: Path, family: str, branch: str | None) -> Path:
    """Return the lexically-latest <family>__<ts>[__<branch>] dir in scene_dir."""
    candidates = []
    for child in scene_dir.iterdir():
        if not child.is_dir():
            continue
        m = RUN_RE.match(child.name)
        if not m or m.group("family") != family:
            continue
        tail = m.group("tail") or ""
        if branch is None:
            if tail:
                continue
        else:
            if not tail.endswith(branch):
                continue
        candidates.append(child)
    if not candidates:
        raise FileNotFoundError(
            f"no {family} run (branch={branch}) in {scene_dir}")
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def stage_linked_pngs() -> list[dict]:
    """Copy compositor PNGs into linked_pngs/interventions_bioenvelope_ref/...

    Returns a list of `{path, file, blend}` entries that the JSX will consume.
    `path` is the slash-joined PSB path; `file` is the absolute PNG path used
    for linked placement; `blend` is a BlendMode name (Photoshop enum).
    """
    LINKED_ROOT.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for group, exr_family in SCENES:
        scene_dir = COMPOSITOR_ROOT / exr_family
        if not scene_dir.is_dir():
            sys.exit(f"ERROR: compositor scene dir missing: {scene_dir}")

        bio_run = find_latest_run(scene_dir, "bioenvelope", None)
        int_run = find_latest_run(scene_dir, "intervention_int", "positive")
        prop_run = find_latest_run(scene_dir, "proposal_and_interventions", "positive")

        scene_linked = LINKED_ROOT / group
        (scene_linked / "outlines").mkdir(parents=True, exist_ok=True)
        (scene_linked / "proposals").mkdir(parents=True, exist_ok=True)
        (scene_linked / "interventions_bioenvelope").mkdir(parents=True, exist_ok=True)

        # Base template — single linked SO into the per-scene base PSB.
        base_name = BASE_TEMPLATE_OVERRIDES.get(exr_family, f"base_{exr_family}.psb")
        base_template = PSD_LIVE / base_name
        if not base_template.exists():
            sys.exit(f"ERROR: missing base template: {base_template}")
        entries.append({
            "path": f"{group}/base/{base_template.stem}",
            "file": str(base_template),
        })

        # Sub-groups are staged in stack order BOTTOM->TOP because each new
        # subgroup goes to PLACEATBEGINNING of its parent (newest on top).
        # Final panel order top->bottom: outlines, proposals, interventions, base.

        # Interventions (all except the combined one).
        for png in sorted(int_run.glob("*.png")):
            if png.name == INTERVENTION_SKIP:
                continue
            dst = scene_linked / "interventions_bioenvelope" / png.name
            if not (dst.exists() and dst.stat().st_size == png.stat().st_size):
                shutil.copy2(png, dst)
            entries.append({
                "path": f"{group}/interventions_bioenvelope/{png.stem}",
                "file": str(dst),
            })

        # Proposals (13 PNGs from latest proposal_and_interventions positive run).
        for png in sorted(prop_run.glob("*.png")):
            dst = scene_linked / "proposals" / png.name
            if not (dst.exists() and dst.stat().st_size == png.stat().st_size):
                shutil.copy2(png, dst)
            entries.append({
                "path": f"{group}/proposals/{png.stem}",
                "file": str(dst),
            })

        # Outlines.
        for slot in OUTLINE_SLOTS:
            src = bio_run / slot
            if not src.exists():
                sys.exit(f"ERROR: missing outline slot: {src}")
            dst = scene_linked / "outlines" / slot
            if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
                shutil.copy2(src, dst)
            entries.append({
                "path": f"{group}/outlines/{slot[:-4]}",
                "file": str(dst),
            })

    return entries


JSX_TEMPLATE = r"""
(function () {
    var OUT_PATH   = __OUT_PATH__;
    var MAPPING    = __MAPPING__;
    var GROUPS     = __GROUPS__;    // top-level group names in panel order
    var COMPS      = __COMPS__;     // comp names (= group names, owner-gated)

    var savedUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    while (app.documents.length > 0) {
        app.documents[0].close(SaveOptions.DONOTSAVECHANGES);
    }

    var doc = app.documents.add(
        new UnitValue(7680, "px"),
        new UnitValue(4320, "px"),
        72,
        "interventions_bioenvelope_ref",
        NewDocumentMode.RGB,
        DocumentFill.WHITE,
        1.0,
        BitsPerChannelType.EIGHT
    );

    // Top-level groups in panel order (first entry on top).
    var groupByName = {};
    for (var g = 0; g < GROUPS.length; g++) {
        var ls = doc.layerSets.add();
        ls.name = GROUPS[g];
        groupByName[GROUPS[g]] = ls;
    }

    // Place each mapping entry. Create parent LayerSets on demand.
    for (var m = 0; m < MAPPING.length; m++) {
        var entry = MAPPING[m];
        var segs = entry.path.split("/");
        var leafName = segs.pop();
        var topName = segs[0];
        var parent = groupByName[topName];
        if (!parent) {
            // Top group not pre-created; build it.
            parent = doc.layerSets.add();
            parent.name = topName;
            groupByName[topName] = parent;
        }
        for (var s = 1; s < segs.length; s++) {
            var next = findChildSet(parent, segs[s]);
            if (!next) {
                doc.activeLayer = parent;
                next = parent.layerSets.add();
                next.name = segs[s];
            }
            parent = next;
        }

        doc.activeLayer = parent;
        var pd = new ActionDescriptor();
        pd.putPath(charIDToTypeID("null"), new File(entry.file));
        pd.putBoolean(stringIDToTypeID("linked"), true);
        executeAction(stringIDToTypeID("placeEvent"), pd, DialogModes.NO);
        var placed = doc.activeLayer;
        placed.name = leafName;
        if (placed.parent !== parent) {
            placed.move(parent, ElementPlacement.PLACEATBEGINNING);
        }
    }

    // Remove any untouched transparent/bg starter layers at doc root.
    // The "white" doc-fill lives in a locked Background; keep it as a floor.
    // No-op otherwise.

    // Build one layer comp per top-level group (owner-gated visibility).
    for (var c = 0; c < COMPS.length; c++) {
        var owner = COMPS[c];
        // Apply target visibility: only `owner` top-level is on.
        for (var i = 0; i < doc.layers.length; i++) {
            var L = doc.layers[i];
            if (L.typename === "LayerSet") {
                L.visible = (L.name === owner);
            }
        }
        doc.layerComps.add(owner, "auto " + owner, true, true, true);
    }

    // Save as PSB. PhotoshopSaveOptions defaults to PSD; we must force PSB
    // via the largeDocumentFormat ActionDescriptor.
    var saveDesc = new ActionDescriptor();
    var saveOpts = new ActionDescriptor();
    saveOpts.putBoolean(stringIDToTypeID("maximizeCompatibility"), true);
    saveDesc.putObject(
        stringIDToTypeID("as"),
        stringIDToTypeID("largeDocumentFormat"),
        saveOpts
    );
    saveDesc.putPath(stringIDToTypeID("in"), new File(OUT_PATH));
    saveDesc.putBoolean(stringIDToTypeID("lowerCase"), true);
    executeAction(stringIDToTypeID("save"), saveDesc, DialogModes.NO);

    app.preferences.rulerUnits = savedUnits;
    return "SAVED " + OUT_PATH;

    function findChildSet(parent, name) {
        for (var i = 0; i < parent.layers.length; i++) {
            var L = parent.layers[i];
            if (L.typename === "LayerSet" && L.name === name) return L;
        }
        return null;
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


def _js_array(items: list[str]) -> str:
    return "[" + ",".join(_js_literal(i) for i in items) + "]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path,
                    help=f"Output PSB path (default {DEFAULT_OUT.name}).")
    ap.add_argument("--app", default=APP_NAME)
    args = ap.parse_args()

    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    entries = stage_linked_pngs()
    print(f"Staged {len(entries)} linked PNGs under {LINKED_ROOT}")

    groups = [g for g, _ in SCENES]

    mapping_literal = "[" + ",".join(
        f'{{path:{_js_literal(e["path"])},file:{_js_literal(e["file"])}}}'
        for e in entries
    ) + "]"

    jsx_src = (
        JSX_TEMPLATE
        .replace("__OUT_PATH__", _js_literal(str(args.out)))
        .replace("__MAPPING__", mapping_literal)
        .replace("__GROUPS__", _js_array(groups))
        .replace("__COMPS__", _js_array(groups))
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
