from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_code-refactored")
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from refactor_code.sim.baseline.baseline_v3 import (
    DEFAULT_DECAYED_SHARE,
    DEFAULT_FALLEN_SHARE,
    DEFAULT_SENESCING_SHARE,
    DEFAULT_SNAG_SHARE,
    DEFAULT_TOTAL_DEADWOOD_TARGET_M3_PER_HA,
    DEFAULT_WOOD_DENSITY_T_PER_M3,
    generate_baseline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate v3 baselines with base-template-volume-targeted fallen and decayed allocation."
    )
    parser.add_argument("site", help="Site key, for example trimmed-parade, city, or uni.")
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--template-root", type=str, default=None)
    parser.add_argument("--run-output-root", type=str, default=None)
    parser.add_argument(
        "--deadwood-target-m3-per-ha",
        type=float,
        default=DEFAULT_TOTAL_DEADWOOD_TARGET_M3_PER_HA,
    )
    parser.add_argument("--fallen-share", type=float, default=DEFAULT_FALLEN_SHARE)
    parser.add_argument("--decayed-share", type=float, default=DEFAULT_DECAYED_SHARE)
    parser.add_argument("--senescing-share", type=float, default=DEFAULT_SENESCING_SHARE)
    parser.add_argument("--snag-share", type=float, default=DEFAULT_SNAG_SHARE)
    parser.add_argument("--wood-density-t-per-m3", type=float, default=DEFAULT_WOOD_DENSITY_T_PER_M3)
    parser.add_argument("--visualize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = generate_baseline(
        site=args.site,
        voxel_size=args.voxel_size,
        template_root=args.template_root,
        run_output_root=args.run_output_root,
        total_deadwood_target_m3_per_ha=args.deadwood_target_m3_per_ha,
        fallen_share=args.fallen_share,
        decayed_share=args.decayed_share,
        senescing_share=args.senescing_share,
        snag_share=args.snag_share,
        wood_density_t_per_m3=args.wood_density_t_per_m3,
        visualize=args.visualize,
    )
    print(f"Baseline trees: {artifacts.trees_csv_path}")
    print(f"Baseline resources: {artifacts.resource_vtk_path}")
    print(f"Baseline terrain: {artifacts.terrain_vtk_path}")
    print(f"Baseline combined: {artifacts.combined_vtk_path}")
    print(f"Deadwood allocation: {artifacts.allocation_csv_path}")
    print(f"Metadata: {artifacts.metadata_json_path}")


if __name__ == "__main__":
    main()
