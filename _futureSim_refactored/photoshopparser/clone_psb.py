"""Clone the template PSB to a new named variant under _psds/psd-live/.

Uses APFS copy-on-write (`cp -c`) so clones are instant and share storage
until they diverge.

Usage:
    uv run python -m _futureSim_refactored.photoshopparser.clone_psb \\
        --name parade_single-state_yr180

Produces: _data-refactored/_psds/psd-live/<name>.psb
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PSD_LIVE = REPO_ROOT / "_data-refactored" / "_psds" / "psd-live"
TEMPLATE = PSD_LIVE / "template.psb"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True,
                    help="Variant name, e.g. parade_single-state_yr180")
    ap.add_argument("--source", type=Path, default=TEMPLATE,
                    help=f"Source PSB (default: {TEMPLATE.name})")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite if destination already exists")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        return 1

    dst = PSD_LIVE / f"{args.name}.psb"
    if dst.exists() and not args.force:
        print(f"ERROR: already exists (use --force to overwrite): {dst}",
              file=sys.stderr)
        return 1

    if dst.exists():
        dst.unlink()

    subprocess.run(["cp", "-c", str(args.source), str(dst)], check=True)
    print(f"cloned: {args.source.name} -> {dst.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
