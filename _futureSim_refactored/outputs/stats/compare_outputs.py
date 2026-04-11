from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
if str(CODE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT.parent))

from _futureSim_refactored.outputs.stats.validation import build_summary, compare_summaries, write_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Compare canonical and validation scenario outputs.")
    parser.add_argument("--sites", nargs="+", default=["trimmed-parade", "city", "uni"])
    parser.add_argument("--scenarios", nargs="+", default=["positive", "trending"])
    parser.add_argument("--years", nargs="+", type=int, default=[0, 10, 30, 60, 90, 120, 150, 180])
    parser.add_argument("--voxel-size", type=int, default=1)
    parser.add_argument("--left-mode", default="canonical", choices=["canonical", "validation"])
    parser.add_argument("--right-mode", default="validation", choices=["canonical", "validation"])
    parser.add_argument("--filename", default="comparison_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    left = build_summary(
        sites=args.sites,
        scenarios=args.scenarios,
        years=args.years,
        voxel_size=args.voxel_size,
        output_mode=args.left_mode,
    )
    right = build_summary(
        sites=args.sites,
        scenarios=args.scenarios,
        years=args.years,
        voxel_size=args.voxel_size,
        output_mode=args.right_mode,
    )
    comparison = {
        "left_mode": args.left_mode,
        "right_mode": args.right_mode,
        "diff": compare_summaries(left, right),
    }
    output_path = write_summary(comparison, args.filename, output_mode="validation")
    print(output_path)


if __name__ == "__main__":
    main()
