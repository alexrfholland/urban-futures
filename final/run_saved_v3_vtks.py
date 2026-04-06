"""
Generate V3 VTK outputs from saved scenario CSVs only.

This script does not rerun the scenario engine. It rebuilds the site dataset,
loads saved tree/log/pole CSVs for each requested year, and runs the normal
VTK generation path.

Expected env vars:
- REFACTOR_RUN_OUTPUT_ROOT
- TREE_TEMPLATE_ROOT

Typical usage:
  uv run python final/run_saved_v3_vtks.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

import a_scenario_generateVTKs
import a_info_gather_capabilities
import a_scenario_initialiseDS
import a_scenario_urban_elements_count
from refactor_code.scenario import params_v3


SITES = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]
YEARS = params_v3.generate_timesteps(interval=30)
VOXEL_SIZE = 1


def _reject_legacy_root_envs() -> None:
    legacy_keys = [
        "REFACTOR_SCENARIO_OUTPUT_ROOT",
        "REFACTOR_ENGINE_OUTPUT_ROOT",
        "REFACTOR_STATISTICS_ROOT",
    ]
    configured = {key: os.environ.get(key) for key in legacy_keys if os.environ.get(key)}
    if configured:
        details = ", ".join(f"{key}={value}" for key, value in configured.items())
        raise RuntimeError(
            "Legacy root env vars are no longer supported. "
            f"Unset them and use REFACTOR_RUN_OUTPUT_ROOT only. Found: {details}"
        )


def _parse_csv_arg(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_years(raw: str | None) -> list[int]:
    if not raw:
        return YEARS
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VTK outputs from saved v3 scenario CSVs.")
    parser.add_argument("--sites", help="Comma-separated sites. Default: all canonical sites.")
    parser.add_argument("--scenarios", help="Comma-separated scenarios. Default: positive,trending.")
    parser.add_argument("--years", help="Comma-separated assessed years. Default: canonical v3 years.")
    parser.add_argument("--voxel-size", type=int, default=VOXEL_SIZE, help="Voxel size. Default: 1.")
    parser.add_argument(
        "--enable-visualization",
        action="store_true",
        help="Show PyVista visualization during VTK generation.",
    )
    parser.add_argument(
        "--save-raw-vtk",
        action="store_true",
        help="Also save the raw scenario state VTK file.",
    )
    return parser.parse_args()


def _prepare_subset_dataset(site: str, voxel_size: int):
    possibility_space_ds = a_scenario_initialiseDS.initialize_dataset(site, voxel_size, write_cache=False)
    tree_df, _pole_df, _log_df = a_scenario_initialiseDS.load_node_dataframes(site, voxel_size)
    tree_df, possibility_space_ds = a_scenario_initialiseDS.PreprocessData(tree_df, possibility_space_ds, None)
    possibility_space_ds, _ = a_scenario_initialiseDS.further_xarray_processing(possibility_space_ds)
    return possibility_space_ds


def run_saved_site_scenario(
    site: str,
    scenario: str,
    years: list[int],
    *,
    voxel_size: int = 1,
    enable_visualization: bool = False,
    save_raw_vtk: bool = False,
) -> None:
    print(f"\n===== Saved V3 VTKs: {site} / {scenario} =====\n")
    possibility_space_ds = _prepare_subset_dataset(site, voxel_size)

    for year in years:
        print(f"\n----- Saved V3 VTKs: {site} / {scenario} / yr{year} -----\n")
        tree_df, log_df, pole_df = a_scenario_generateVTKs.load_scenario_dataframes(site, scenario, year, voxel_size)
        if tree_df is None:
            print(f"Skipping {site} / {scenario} / yr{year}: saved scenario CSVs not found")
            continue

        vtk_result = a_scenario_generateVTKs.generate_vtk(
            site,
            scenario,
            year,
            voxel_size,
            possibility_space_ds.copy(deep=True),
            tree_df,
            log_df,
            pole_df,
            enable_visualization=enable_visualization,
            return_polydata=True,
            save_raw_vtk=save_raw_vtk,
        )
        vtk_file, state_polydata = vtk_result
        state_polydata = a_scenario_urban_elements_count.process_scenario_polydata(
            state_polydata,
            site=site,
            voxel_size=voxel_size,
            scenario=scenario,
            year=year,
            save_path=None,
            enable_visualization=enable_visualization,
        )
        a_info_gather_capabilities.process_polydata(
            state_polydata,
            site,
            scenario,
            year,
            voxel_size=voxel_size,
            save_vtk=True,
            save_stats=False,
            output_mode="validation",
        )
        if vtk_file:
            print(f"Saved raw VTK to {vtk_file}")


def main() -> None:
    _reject_legacy_root_envs()
    args = parse_args()
    sites = _parse_csv_arg(args.sites, SITES)
    scenarios = _parse_csv_arg(args.scenarios, SCENARIOS)
    years = _parse_years(args.years)

    env_summary = {
        "REFACTOR_RUN_OUTPUT_ROOT": os.environ.get("REFACTOR_RUN_OUTPUT_ROOT"),
        "TREE_TEMPLATE_ROOT": os.environ.get("TREE_TEMPLATE_ROOT"),
    }
    print("Environment:")
    for key, value in env_summary.items():
        print(f"  {key}={value}")

    for site in sites:
        for scenario in scenarios:
            run_saved_site_scenario(
                site,
                scenario,
                years,
                voxel_size=args.voxel_size,
                enable_visualization=args.enable_visualization,
                save_raw_vtk=args.save_raw_vtk,
            )


if __name__ == "__main__":
    main()
