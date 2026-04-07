from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.outputs.stats.validation import build_summary, write_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Capture scenario-output summaries for regression comparison.")
    parser.add_argument("--sites", nargs="+", default=["trimmed-parade", "city", "uni"])
    parser.add_argument("--scenarios", nargs="+", default=["positive", "trending"])
    parser.add_argument("--years", nargs="+", type=int, default=[0, 10, 30, 60, 90, 120, 150, 180])
    parser.add_argument("--voxel-size", type=int, default=1)
    parser.add_argument("--output-mode", default="canonical", choices=["canonical", "validation"])
    parser.add_argument("--filename", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_summary(
        sites=args.sites,
        scenarios=args.scenarios,
        years=args.years,
        voxel_size=args.voxel_size,
        output_mode=args.output_mode,
    )
    filename = args.filename or f"{args.output_mode}_summary.json"
    output_path = write_summary(summary, filename, output_mode="validation")
    print(output_path)


if __name__ == "__main__":
    main()
