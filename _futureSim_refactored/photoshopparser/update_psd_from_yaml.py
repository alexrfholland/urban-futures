"""Reconcile a PSB's smart-object tree against the site manifest.

Given a `--psd-path` selector (leaf or group), resolve the yaml rule, find
the corresponding compositor PNG(s), copy into `linked_pngs/<site>/`, and
ensure the PSB has a matching linked smart object at that path. Creates any
missing parent groups. Idempotent: SOs already linked to the right file are
skipped; SOs pointing at the wrong file are relinked.

Selectors:
- Leaf path (`Positive/arboreal/positive/sizes/size_artificial`): one SO.
- Group path that exactly matches a `key_folder` key
  (`Positive/arboreal/positive/sizes`): one SO per PNG produced by that
  rule's compositor run (layer name = PNG stem).

Usage:
    # One leaf
    uv run python -m _futureSim_refactored.photoshopparser.update_psd_from_yaml \\
        --psb         _data-refactored/_psds/psd-live/uni_timeline_test.psb \\
        --psd-path    "Positive/arboreal/positive/sizes/size_artificial" \\
        --exr-family  uni_timeline

    # Whole folder rule
    uv run python -m _futureSim_refactored.photoshopparser.update_psd_from_yaml \\
        --psb         _data-refactored/_psds/psd-live/uni_timeline_test.psb \\
        --psd-path    "Positive/arboreal/positive/sizes" \\
        --exr-family  uni_timeline
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from _futureSim_refactored.photoshopparser.copy_pngs import (
    find_run_dir,
    resolve_rule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO_ROOT / "_data-refactored" / "_psds" / "sources" / "parade.yaml"
LINKED_PNGS = REPO_ROOT / "_data-refactored" / "_psds" / "linked_pngs"

APP_NAME = "Adobe Photoshop 2026"

JSX_TEMPLATE = r"""
(function () {
    var MAPPING = __MAPPING__;
    var UPDATE_COMPS = __UPDATE_COMPS__;
    if (app.documents.length === 0) return "ERROR: no document open";
    var doc = app.activeDocument;
    var report = [];

    for (var k = 0; k < MAPPING.length; k++) {
        var entry = MAPPING[k];
        var segments = entry.path.split("/");
        var leafName = segments.pop();

        var parent = doc;
        for (var s = 0; s < segments.length; s++) {
            var next = findChildSet(parent, segments[s]);
            if (!next) {
                next = parent.layerSets.add();
                next.name = segments[s];
                report.push("MKGROUP: " + segments.slice(0, s+1).join("/"));
            }
            parent = next;
        }

        var existing = findChild(parent, leafName);
        if (existing) {
            if (isLinkedTo(doc, existing, entry.file)) {
                report.push("SKIP (already linked): " + entry.path);
                continue;
            }
            doc.activeLayer = existing;
            var rd = new ActionDescriptor();
            rd.putPath(stringIDToTypeID("null"), new File(entry.file));
            executeAction(stringIDToTypeID("placedLayerRelinkToFile"), rd, DialogModes.NO);
            report.push("RELINK: " + entry.path + " -> " + entry.file);
            continue;
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
        report.push("PLACE: " + entry.path + " -> " + entry.file);
    }

    if (UPDATE_COMPS) {
        // Layer-comp convention: comp names are prefixed with a top-level
        // folder that "owns" the comp. Only Trending / Positive are
        // owner-gated — a new SO under them should be visible in its own
        // owner's comps and hidden elsewhere. Every other top-level folder
        // (BASE WORLD, shading, ...) is always-on across all comps, with
        // the single exception of top-levels prefixed "hidden_" which are
        // intentionally kept hidden everywhere.
        var OWNER_GATED = __OWNER_GATED__;
        function isOwnerGated(top) {
            for (var g = 0; g < OWNER_GATED.length; g++) {
                if (top === OWNER_GATED[g]) return true;
            }
            return false;
        }

        var topLevels = [];
        for (var i = 0; i < doc.layers.length; i++) {
            if (doc.layers[i].typename === "LayerSet") {
                topLevels.push(doc.layers[i].name);
            }
        }
        // Longest-prefix first so e.g. "Positive - shading" wins over "Positive".
        topLevels.sort(function (a, b) { return b.length - a.length; });

        var comps = doc.layerComps;
        for (var c = 0; c < comps.length; c++) {
            var comp = comps[c];
            var owner = null;
            for (var o = 0; o < topLevels.length; o++) {
                var t = topLevels[o];
                if (comp.name === t || comp.name.indexOf(t) === 0) {
                    owner = t; break;
                }
            }
            if (!owner) {
                report.push("COMP SKIP (no top-level match): " + comp.name);
                continue;
            }

            try {
                comp.apply();
                for (var m2 = 0; m2 < MAPPING.length; m2++) {
                    var entry2 = MAPPING[m2];
                    var segs = entry2.path.split("/");
                    var lyr = navigate(doc, segs);
                    if (!lyr) continue;
                    var top = segs[0];
                    if (top.indexOf("hidden_") === 0) {
                        lyr.visible = false;
                    } else if (isOwnerGated(top)) {
                        lyr.visible = (top === owner);
                    } else {
                        lyr.visible = true;
                    }
                }
                comp.recapture();
                report.push("RECAPTURE (" + owner + "): " + comp.name);
            } catch (e) {
                report.push("COMP FAIL (" + comp.name + "): " + e);
            }
        }
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
    function navigate(root, segments) {
        var current = root;
        for (var j = 0; j < segments.length; j++) {
            current = findChild(current, segments[j]);
            if (!current) return null;
        }
        return current;
    }
    function isLinkedTo(doc, layer, file) {
        try {
            var ref = new ActionReference();
            ref.putIdentifier(charIDToTypeID("Lyr "), layer.id);
            var d = executeActionGet(ref);
            var sid = stringIDToTypeID("smartObject");
            if (!d.hasKey(sid)) return false;
            var so = d.getObjectValue(sid);
            var linkedId = stringIDToTypeID("linked");
            if (!so.hasKey(linkedId) || !so.getBoolean(linkedId)) return false;
            return so.getString(stringIDToTypeID("fileReference")) === file;
        } catch (e) { return false; }
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


def derive_site(psb_stem: str) -> str:
    """Mirror relink_psb.jsx deriveSite: strip trailing ' test', special-case parade."""
    base = psb_stem.rstrip()
    if base.endswith(" test"):
        base = base[:-5].rstrip()
    if base == "parade_timeline":
        return "parade"
    return base


def resolve_targets(
    manifest: dict, psd_path: str,
    compositor_root: Path, sim_root: str, exr_family: str,
) -> list[tuple[str, Path]]:
    """Resolve the selector to a list of (layer_path, compositor_png_path)."""
    key_folder = manifest.get("key_folder", {}) or {}
    key_layer = manifest.get("key_layer", {}) or {}
    runs = manifest.get("runs", {})

    def resolve_one(leaf_path: str) -> tuple[str, Path]:
        rule = resolve_rule(leaf_path, key_folder, key_layer)
        if rule is None:
            sys.exit(f"ERROR: no yaml rule matches {leaf_path}")
        if rule.get("skip"):
            sys.exit(f"ERROR: rule for {leaf_path} is skip:true")
        family = rule["family"]
        branch = rule.get("branch")
        leaf_name = leaf_path.rsplit("/", 1)[-1]
        slot = rule.get("slot") or f"{leaf_name}.png"
        pin = runs.get(family, "latest")
        run_dir = find_run_dir(compositor_root, sim_root, exr_family,
                               family, branch, pin)
        if run_dir is None:
            sys.exit(f"ERROR: no compositor run for family={family} "
                     f"branch={branch} in {compositor_root}/{sim_root}/{exr_family}")
        src = run_dir / slot
        if not src.exists():
            sys.exit(f"ERROR: missing slot {slot} in {run_dir}")
        return leaf_path, src

    if psd_path in key_folder:
        rule = key_folder[psd_path]
        if rule.get("skip"):
            sys.exit(f"ERROR: folder rule for {psd_path} is skip:true")
        family = rule["family"]
        branch = rule.get("branch")
        pin = runs.get(family, "latest")
        run_dir = find_run_dir(compositor_root, sim_root, exr_family,
                               family, branch, pin)
        if run_dir is None:
            sys.exit(f"ERROR: no compositor run for family={family} "
                     f"branch={branch} in {compositor_root}/{sim_root}/{exr_family}")
        pngs = sorted(run_dir.glob("*.png"))
        if not pngs:
            sys.exit(f"ERROR: no PNGs in {run_dir}")
        results: list[tuple[str, Path]] = []
        for p in pngs:
            leaf = f"{psd_path}/{p.stem}"
            leaf_rule = key_layer.get(leaf)
            if leaf_rule and leaf_rule.get("skip"):
                print(f"SKIP (key_layer skip:true): {leaf}")
                continue
            results.append((leaf, p))
        return results

    return [resolve_one(psd_path)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--psb", required=True, type=Path,
                    help="Outer PSB to edit.")
    ap.add_argument("--psd-path", required=True, dest="psd_path",
                    help='Slash-joined path. Leaf or key_folder group key.')
    ap.add_argument("--exr-family", required=True, dest="exr_family",
                    help="compositor/outputs/<sim_root>/<exr_family>/ source folder.")
    ap.add_argument("--sim-root", default=None, dest="sim_root",
                    help="Overrides sim_root from the manifest.")
    ap.add_argument("--yaml", default=DEFAULT_YAML, type=Path, dest="yaml_path",
                    help=f"Manifest (default: {DEFAULT_YAML.name}).")
    ap.add_argument("--site", default=None,
                    help="linked_pngs/<site>/ subdir; defaults from --psb name.")
    ap.add_argument("--app", default=APP_NAME)
    ap.add_argument("--update-comps", action=argparse.BooleanOptionalAction,
                    default=True, dest="update_comps",
                    help="Recapture every layer comp so new SOs become part "
                         "of each comp's recorded state (owner-gated rule). "
                         "On by default; pass --no-update-comps to skip.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.psb = args.psb.resolve()
    args.yaml_path = args.yaml_path.resolve()

    if not args.psb.exists():
        sys.exit(f"ERROR: PSB not found: {args.psb}")
    if not args.yaml_path.exists():
        sys.exit(f"ERROR: yaml not found: {args.yaml_path}")

    manifest = yaml.safe_load(args.yaml_path.read_text()) or {}
    sim_root = args.sim_root or manifest["sim_root"]
    compositor_root = REPO_ROOT / manifest["compositor_root"]

    site = args.site or derive_site(args.psb.stem)
    linked_site = LINKED_PNGS / site

    targets = resolve_targets(
        manifest, args.psd_path, compositor_root, sim_root, args.exr_family,
    )

    mapping: list[dict] = []
    for layer_path, src in targets:
        dst = linked_site / (layer_path + ".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
            print(f"copy: {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")
            if not args.dry_run:
                shutil.copy2(src, dst)
        else:
            print(f"keep: {dst.relative_to(REPO_ROOT)} (size match)")
        mapping.append({"path": layer_path, "file": str(dst)})

    if args.dry_run:
        print("(dry-run — PSB not touched)")
        for m in mapping:
            print(f"  would place: {m['path']} -> {m['file']}")
        return 0

    mapping_literal = "[" + ",".join(
        f'{{path:{_js_literal(m["path"])},file:{_js_literal(m["file"])}}}'
        for m in mapping
    ) + "]"
    owner_gated = manifest.get("layer_comp_owner_gated",
                                ["Trending", "Positive"])
    owner_gated_literal = "[" + ",".join(_js_literal(s) for s in owner_gated) + "]"
    jsx_src = (
        JSX_TEMPLATE
        .replace("__MAPPING__", mapping_literal)
        .replace("__UPDATE_COMPS__", "true" if args.update_comps else "false")
        .replace("__OWNER_GATED__", owner_gated_literal)
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
