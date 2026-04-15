"""Add a routing rule to a site manifest based on a PSD's structure.

Point at a group or single SO inside a PSD and emit the corresponding
`key_folder` (group) or `key_layer` (single layer) entry into the manifest.
Additive only — if the exact key already exists it is logged and skipped,
never overwritten.

Usage:
    # Group -> key_folder rule (applies to every PNG child by prefix)
    uv run python -m _futureSim_refactored.photoshopparser.update_yaml_from_psd \\
        --psd       _data-refactored/_psds/psd-live/uni_timeline.psb \\
        --psd-path  "Trending/arboreal/trending/sizes/new_subgroup" \\
        --family    sizes_single_input \\
        --branch    trending

    # Single SO -> key_layer pin (precise, overrides folder rule)
    uv run python -m _futureSim_refactored.photoshopparser.update_yaml_from_psd \\
        --psd       _data-refactored/_psds/psd-live/uni_timeline.psb \\
        --psd-path  "Positive/envelopes/outlines/some_custom_layer" \\
        --family    base \\
        --slot      base_custom.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from psd_tools import PSDImage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO_ROOT / "_data-refactored" / "_psds" / "sources" / "parade.yaml"


def navigate(psd: PSDImage, segments: list[str]):
    """Walk `segments` from the PSD root. Returns the matching layer or None."""
    current = list(reversed(list(psd)))  # panel-top-first
    target = None
    for seg in segments:
        found = None
        for layer in current:
            if layer.name == seg:
                found = layer
                break
        if found is None:
            return None
        target = found
        current = list(reversed(list(found))) if found.is_group() else []
    return target


def classify(layer) -> str:
    if layer.is_group():
        return "group"
    if getattr(layer, "kind", "") == "smartobject":
        return "smart_object"
    return getattr(layer, "kind", "") or "other"


def insert_entry(
    yaml_text: str, section: str, path: str, rule: dict
) -> str:
    """Append a new `"<path>": { rule }` entry at the end of `section`.

    Preserves surrounding comments and unrelated formatting by editing text
    lines rather than reserialising. The section header must already exist.
    """
    lines = yaml_text.splitlines()
    header = f"{section}:"
    header_idx = None
    for i, line in enumerate(lines):
        if line.rstrip() == header:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"section {section!r} header not found in yaml")

    # End of section = next line that starts at column 0 and isn't a comment.
    end_idx = len(lines)
    for i in range(header_idx + 1, len(lines)):
        line = lines[i]
        if not line:
            continue
        if line[0].isspace():
            continue
        if line.lstrip().startswith("#"):
            continue
        end_idx = i
        break

    # Walk back past trailing blanks so the insert sits right after the last
    # real entry in the section.
    insert_at = end_idx
    while insert_at > header_idx + 1 and lines[insert_at - 1].strip() == "":
        insert_at -= 1

    new_block = ["", f'  "{path}":']
    for k, v in rule.items():
        if v is None:
            new_block.append(f"    {k}: null")
        elif isinstance(v, bool):
            new_block.append(f"    {k}: {'true' if v else 'false'}")
        else:
            new_block.append(f"    {k}: {v}")

    updated = lines[:insert_at] + new_block + lines[insert_at:]
    return "\n".join(updated) + ("\n" if yaml_text.endswith("\n") else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--psd", required=True, type=Path,
                    help="Source PSD/PSB to read structure from.")
    ap.add_argument("--psd-path", required=True, dest="psd_path",
                    help="Slash-separated path to a group or SO inside the "
                         "PSD (panel order), e.g. "
                         "\"Trending/arboreal/trending/sizes/new_subgroup\".")
    ap.add_argument("--family", required=True,
                    help="Compositor family for the new rule "
                         "(e.g. sizes_single_input, base).")
    ap.add_argument("--branch", default=None,
                    help="Compositor branch, if any (e.g. trending, positive).")
    ap.add_argument("--slot", default=None,
                    help="Override slot filename for key_layer entries. "
                         "Only applies to single-SO targets.")
    ap.add_argument("--yaml", default=DEFAULT_YAML, type=Path, dest="yaml_path",
                    help=f"Manifest to edit (default: {DEFAULT_YAML.name}).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.psd = args.psd.resolve()
    args.yaml_path = args.yaml_path.resolve()

    if not args.psd.exists():
        sys.exit(f"ERROR: PSD not found: {args.psd}")
    if not args.yaml_path.exists():
        sys.exit(f"ERROR: yaml not found: {args.yaml_path}")

    segments = [s for s in args.psd_path.split("/") if s]
    if not segments:
        sys.exit("ERROR: --psd-path must not be empty")

    print(f"opening {args.psd.relative_to(REPO_ROOT)} ...", flush=True)
    psd = PSDImage.open(args.psd)
    target = navigate(psd, segments)
    if target is None:
        sys.exit(f"ERROR: path not found in PSD: {args.psd_path}")

    kind = classify(target)
    if kind == "group":
        section = "key_folder"
        if args.slot is not None:
            sys.exit("ERROR: --slot is only meaningful for single SO targets")
    elif kind == "smart_object":
        section = "key_layer"
    else:
        sys.exit(f"ERROR: target is a {kind!r} layer — must be a group or SO")

    manifest = yaml.safe_load(args.yaml_path.read_text()) or {}
    existing = manifest.get(section, {}) or {}
    if args.psd_path in existing:
        current = existing[args.psd_path]
        print(f"SKIP: {section}[{args.psd_path!r}] already present:")
        for k, v in current.items():
            print(f"    {k}: {v}")
        print("  (additive merge — no overwrite)")
        return 0

    rule: dict = {"family": args.family}
    if args.branch is not None:
        rule["branch"] = args.branch
    if args.slot is not None:
        rule["slot"] = args.slot

    yaml_text = args.yaml_path.read_text()
    updated = insert_entry(yaml_text, section, args.psd_path, rule)

    print(f"ADD: {section}[{args.psd_path!r}]:")
    for k, v in rule.items():
        print(f"    {k}: {v}")

    if args.dry_run:
        print("(dry-run — yaml not written)")
        return 0

    args.yaml_path.write_text(updated)
    print(f"wrote {args.yaml_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
