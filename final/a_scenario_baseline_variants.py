#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

import a_scenario_get_baselines


DEFAULT_SITES = ("trimmed-parade", "city", "uni")


def parse_sites(raw: str | None) -> list[str]:
    if not raw or raw.strip().lower() == "all":
        return list(DEFAULT_SITES)
    return [site.strip() for site in raw.split(",") if site.strip()]


def scenario_variant_root(variant_name: str) -> Path:
    return REPO_ROOT / "data" / "revised" / "baseline-variants" / variant_name


def engine_variant_root(variant_name: str) -> Path:
    return REPO_ROOT / "_data-refactored" / "baseline-variants" / variant_name


@contextmanager
def temporary_env(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def write_metadata(
    variant_name: str,
    template_root: Path,
    sites: Iterable[str],
    voxel_size: float,
    scenario_root: Path,
    engine_root: Path,
) -> Path:
    metadata = {
        "variant_name": variant_name,
        "template_root": str(template_root.resolve()),
        "sites": list(sites),
        "voxel_size": voxel_size,
        "scenario_root": str(scenario_root.resolve()),
        "engine_root": str(engine_root.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = engine_root / "variant_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path


def generate_variant_baselines(
    variant_name: str,
    template_root: Path,
    sites: list[str],
    voxel_size: float,
) -> tuple[Path, Path]:
    scenario_root = scenario_variant_root(variant_name)
    engine_root = engine_variant_root(variant_name)
    scenario_root.mkdir(parents=True, exist_ok=True)
    engine_root.mkdir(parents=True, exist_ok=True)

    env = {
        "TREE_TEMPLATE_ROOT": str(template_root.resolve()),
        "REFACTOR_SCENARIO_OUTPUT_ROOT": str(scenario_root.resolve()),
        "REFACTOR_ENGINE_OUTPUT_ROOT": str(engine_root.resolve()),
    }

    with temporary_env(env):
        for site in sites:
            print(f"\n===== Generating baseline variant: {variant_name} / {site} =====\n")
            a_scenario_get_baselines.generate_baseline(
                site=site,
                voxel_size=voxel_size,
                output_folder=str(scenario_root / "baselines"),
            )

    metadata_path = write_metadata(
        variant_name=variant_name,
        template_root=template_root,
        sites=sites,
        voxel_size=voxel_size,
        scenario_root=scenario_root,
        engine_root=engine_root,
    )
    print(f"\nWrote metadata: {metadata_path}")
    return scenario_root, engine_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate baseline variants against a chosen template library and keep each iteration on disk."
    )
    parser.add_argument(
        "--variant-name",
        required=True,
        help="Folder label for this baseline iteration, e.g. template-edits__fallens-nonpre-direct__snags-elm-snags-old",
    )
    parser.add_argument(
        "--template-root",
        required=True,
        type=Path,
        help="Directory containing the chosen template library, e.g. .../tree_variants/<variant>/trees",
    )
    parser.add_argument(
        "--sites",
        default="all",
        help="Comma-separated sites or 'all' (default).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1,
    )

    args = parser.parse_args()
    if not args.template_root.exists():
        raise FileNotFoundError(f"Template root not found: {args.template_root}")

    sites = parse_sites(args.sites)
    scenario_root, engine_root = generate_variant_baselines(
        variant_name=args.variant_name,
        template_root=args.template_root,
        sites=sites,
        voxel_size=args.voxel_size,
    )

    print("\nCompleted baseline variant generation.")
    print(f"Scenario root: {scenario_root}")
    print(f"Engine root: {engine_root}")


if __name__ == "__main__":
    main()
