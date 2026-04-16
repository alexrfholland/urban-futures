"""Populate the linked_pngs/<site>/ folder from compositor outputs.

Reads a site manifest (sources/<site>.yaml) plus the PSB layer tree
(from read_psb.py). For each layer, resolves family/branch/slot,
finds the target compositor run, and copies the PNG into place mirroring
the PSB group hierarchy.

Usage:
    uv run python -m _futureSim_refactored.photoshopparser.copy_pngs \\
        --manifest _data-refactored/_psds/sources/parade.yaml \\
        --layers   _data-refactored/_psds/assembled/parade_timeline_test_layers.json \\
        --out-root _data-refactored/_psds/linked_pngs \\
        --site parade \\
        --exr-family parade_timeline \\
        [--only-group Trending] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_RE = re.compile(r"^(?P<family>[a-z_]+)__(?P<ts>\d{8}_\d{4,6})(?:__(?P<tail>.+))?$")


def resolve_rule(path: str, key_folder: dict, key_layer: dict) -> dict | None:
    """Return the effective rule for a layer path.

    key_layer (exact layer path) wins. Otherwise walk key_folder prefixes
    longest-first. Returns None if no rule applies.
    """
    if path in key_layer:
        return key_layer[path]
    parts = path.split("/")
    for i in range(len(parts) - 1, 0, -1):
        key = "/".join(parts[:i])
        if key in key_folder:
            return key_folder[key]
    return None


def find_run_dir(
    compositor_root: Path, sim_root: str, exr_family: str,
    family: str, branch: str | None, pin: str,
) -> Path | None:
    """Locate the compositor run folder.

    pin: "latest" or an explicit timestamp string.
    Handles both flat (<family>__<ts>[__<branch>]) and nested (<family>/<family>__...) layouts.
    """
    base = compositor_root / sim_root / exr_family
    candidates: list[Path] = []

    def match(dir_name: str) -> bool:
        m = RUN_RE.match(dir_name)
        if not m or m.group("family") != family:
            return False
        if pin != "latest" and m.group("ts") != pin:
            return False
        tail = m.group("tail") or ""
        if branch and not tail.endswith(branch):
            return False
        return True

    for p in [base, base / family]:
        if not p.is_dir():
            continue
        for child in p.iterdir():
            if child.is_dir() and match(child.name):
                candidates.append(child)

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)  # timestamps sort chronologically
    return candidates[-1]


def walk_layers(nodes, parent_path: str = ""):
    """Yield (path, node) for every layer in the tree."""
    for n in nodes:
        path = f"{parent_path}/{n['name']}" if parent_path else n["name"]
        yield path, n
        if "layers" in n:
            yield from walk_layers(n["layers"], path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--layers", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--site", required=True,
                    help="linked_pngs/<site>/ output folder name "
                         "(e.g. parade, uni_timeline, parade_single-state_yr180).")
    ap.add_argument("--exr-family", required=True, dest="exr_family",
                    help="compositor/outputs/<sim_root>/<exr_family>/ input "
                         "folder name (e.g. parade_timeline, uni_timeline).")
    ap.add_argument("--only-group", default=None,
                    help="Restrict to layers whose path starts with this prefix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.manifest = args.manifest.resolve()
    args.layers = args.layers.resolve()
    args.out_root = args.out_root.resolve()

    manifest = yaml.safe_load(args.manifest.read_text())
    layers = json.loads(args.layers.read_text())

    site = args.site
    sim_root = manifest["sim_root"]
    exr_family = args.exr_family
    compositor_root = REPO_ROOT / manifest["compositor_root"]
    runs = manifest.get("runs", {})
    key_folder = manifest.get("key_folder", {}) or {}
    key_layer = manifest.get("key_layer", {}) or {}

    out_site = args.out_root / site

    stats = {"populated": 0, "skipped_rule": 0, "skipped_no_rule": 0,
             "missing_run": 0, "missing_slot": 0, "not_a_png_layer": 0}

    for path, node in walk_layers(layers["layers"]):
        if args.only_group and not path.startswith(args.only_group):
            continue

        ntype = node["type"]

        # Groups themselves aren't populated (their children are), but we do
        # honour skip-at-group rules.
        if ntype == "group":
            rule = resolve_rule(path, key_folder, key_layer) or {}
            if rule.get("skip"):
                print(f"skip (group): {path}")
            continue

        # Adjustment layers etc. aren't PNG slots.
        if ntype != "smart_object" and ntype != "pixel":
            stats["not_a_png_layer"] += 1
            print(f"leave alone ({ntype}): {path}")
            continue

        rule = resolve_rule(path, key_folder, key_layer)
        if rule is None:
            stats["skipped_no_rule"] += 1
            print(f"NO RULE: {path}")
            continue
        if rule.get("skip"):
            stats["skipped_rule"] += 1
            print(f"leave alone (rule): {path}")
            continue

        family = rule["family"]
        branch = rule.get("branch")
        slot = rule.get("slot") or f"{node['name']}.png"
        pin = runs.get(family, "latest")

        run_dir = find_run_dir(compositor_root, sim_root, exr_family,
                               family, branch, pin)
        if run_dir is None:
            stats["missing_run"] += 1
            print(f"NO RUN for {family}/{branch} (pin={pin}): {path}")
            continue

        src = run_dir / slot
        if not src.exists():
            stats["missing_slot"] += 1
            print(f"NO SLOT {slot} in {run_dir.name}: {path}")
            continue

        dst = out_site / (path + ".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        action = "DRY-RUN" if args.dry_run else "copy"
        print(f"{action}: {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")
        if not args.dry_run:
            shutil.copy2(src, dst)
        stats["populated"] += 1

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
