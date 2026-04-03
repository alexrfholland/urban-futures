"""
Full V3 batch runner.

This script is intentionally non-interactive so validation can be repeated
exactly under explicit environment-variable roots.

Expected env vars (see _documentation-refactored/scenario_engine_v3_validation.md):
- REFACTOR_RUN_OUTPUT_ROOT
- TREE_TEMPLATE_ROOT
- EXPORT_ALL_POINTDATA_VARIABLES

Typical usage:
  PYTHONPATH=$REPO/final:$REPO/_code-refactored .venv/bin/python final/run_full_v3_batch.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

import a_info_gather_capabilities
import a_scenario_generateVTKs
import a_scenario_get_baselines
import a_scenario_initialiseDS
import a_scenario_runscenario
import a_scenario_urban_elements_count
from refactor_code.scenario import params_v3

from refactor_code.paths import (
    engine_output_validation_dir,
    refactor_statistics_root,
    scenario_baseline_dir,
    scenario_urban_features_vtk_path,
)


SITES = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]
YEARS = params_v3.generate_timesteps(interval=30)  # [0,10,30,60,90,120,150,180]
VOXEL_SIZE = 1


def _run_config() -> dict:
    keys = [
        "REFACTOR_RUN_OUTPUT_ROOT",
        "TREE_TEMPLATE_ROOT",
        "EXPORT_ALL_POINTDATA_VARIABLES",
        "REFACTOR_OUTPUT_MODE",
    ]
    return {key: os.environ.get(key) for key in keys}


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
    parser = argparse.ArgumentParser(description="Run the full v3 batch, a node-only CSV pass, or a saved-CSV VTK pass.")
    parser.add_argument("--sites", help="Comma-separated sites. Default: all canonical sites.")
    parser.add_argument("--scenarios", help="Comma-separated scenarios. Default: positive,trending.")
    parser.add_argument("--years", help="Comma-separated assessed years. Default: canonical v3 years.")
    parser.add_argument("--voxel-size", type=int, default=VOXEL_SIZE, help="Voxel size. Default: 1.")
    parser.add_argument(
        "--node-only",
        action="store_true",
        help="Run scenario CSV/dataframe outputs only and skip VTK/baseline/capability generation.",
    )
    parser.add_argument(
        "--vtk-only",
        action="store_true",
        help="Skip scenario CSV generation, load saved scenario CSVs, and run downstream VTK work only.",
    )
    parser.add_argument(
        "--capabilities-only",
        action="store_true",
        help=(
            "Skip scenario and VTK generation and run the capability/statistics pass only "
            "against existing urban_features outputs for the requested site/scenario/year slice."
        ),
    )
    parser.add_argument(
        "--regenerate-baselines",
        action="store_true",
        help="Regenerate baseline outputs after the site/scenario pass. Default: off.",
    )
    parser.add_argument(
        "--multiple-agent",
        action="store_true",
        help=(
            "Per-slice batch mode for parallel agent work. Runs the requested site/scenario/year slice "
            "and skips the cross-state capability pass so another agent can compile it later."
        ),
    )
    parser.add_argument(
        "--save-raw-vtk",
        action="store_true",
        help="Save raw scenario state VTKs during VTK generation.",
    )
    return parser.parse_args()


def _prepare_subset_dataset(site: str, voxel_size: int, *, write_cache: bool = True):
    subset_ds = a_scenario_initialiseDS.initialize_dataset(site, voxel_size, write_cache=write_cache)
    tree_df, pole_df, log_df = a_scenario_initialiseDS.load_node_dataframes(site, voxel_size)
    tree_df, subset_ds = a_scenario_initialiseDS.PreprocessData(tree_df, subset_ds, None)
    subset_ds, _initial_poly = a_scenario_initialiseDS.further_xarray_processing(subset_ds)

    if log_df is not None:
        log_df = a_scenario_initialiseDS.log_processing(log_df, subset_ds)
    if pole_df is not None:
        pole_df = a_scenario_initialiseDS.pole_processing(pole_df, None, subset_ds)
    return subset_ds, tree_df, pole_df, log_df


def run_site_scenario(
    site: str,
    scenario: str,
    years: list[int],
    *,
    voxel_size: int = 1,
    node_only: bool = False,
    vtk_only: bool = False,
    multiple_agent: bool = False,
    save_raw_vtk: bool = False,
) -> None:
    print(f"\n===== V3: {site} / {scenario} =====\n")

    subsetDS, treeDF, poleDF, logDF = _prepare_subset_dataset(
        site,
        voxel_size,
        write_cache=not (vtk_only or multiple_agent),
    )

    current_tree_df = treeDF.copy()
    previous_year = 0

    for year in years:
        print(f"\n----- V3: {site} / {scenario} / yr{year} -----\n")

        if vtk_only:
            treeDF_scenario, logDF_scenario, poleDF_scenario = a_scenario_generateVTKs.load_scenario_dataframes(
                site, scenario, year, voxel_size
            )
            if treeDF_scenario is None:
                print(f"Skipping {site} / {scenario} / yr{year}: saved scenario CSVs not found")
                previous_year = year
                continue
        else:
            treeDF_scenario, logDF_scenario, poleDF_scenario = a_scenario_runscenario.run_scenario(
                site,
                scenario,
                year,
                voxel_size,
                current_tree_df,
                subsetDS,
                logDF,
                poleDF,
                previous_year=previous_year,
            )

        if node_only:
            print(f"Node-only mode: skipping VTK generation for {site} / {scenario} / yr{year}")
        else:
            vtk_file, state_polydata = a_scenario_generateVTKs.generate_vtk(
                site,
                scenario,
                year,
                voxel_size,
                subsetDS.copy(deep=True),
                treeDF_scenario,
                logDF_scenario,
                poleDF_scenario,
                enable_visualization=False,
                return_polydata=True,
                save_raw_vtk=save_raw_vtk,
            )

            a_scenario_urban_elements_count.process_scenario_polydata(
                state_polydata,
                site=site,
                voxel_size=voxel_size,
                scenario=scenario,
                year=year,
                save_path=scenario_urban_features_vtk_path(site, scenario, year, voxel_size),
                enable_visualization=False,
            )

            if vtk_file:
                print(f"Saved scenario VTK (manager path): {vtk_file}")

        current_tree_df = treeDF_scenario.copy()
        previous_year = year


def run_baselines(sites: list[str], *, voxel_size: int = 1) -> None:
    output_folder = str(scenario_baseline_dir())
    print(f"\n===== V3: Baselines (output_folder={output_folder}) =====\n")
    for site in sites:
        print(f"\n----- V3: baseline / {site} -----\n")
        a_scenario_get_baselines.generate_baseline(site, voxel_size, output_folder, visualize=False)


def run_capabilities(sites: list[str], scenarios: list[str], years: list[int], *, voxel_size: int = 1) -> None:
    print("\n===== V3: Capability Pass (state_with_indicators + indicator CSVs) =====\n")
    csv_dir = refactor_statistics_root("validation") / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for site in sites:
        indicator_df, action_df = a_info_gather_capabilities.process_site(
            site,
            scenarios=scenarios,
            years=years,
            voxel_size=voxel_size,
            save_vtk=True,
            include_baseline=True,
            output_mode="validation",
        )
        if not indicator_df.empty:
            indicator_path = csv_dir / f"{site}_{voxel_size}_indicator_counts.csv"
            indicator_df.to_csv(indicator_path, index=False)
            print(f"Saved indicator CSV: {indicator_path}")
        if not action_df.empty:
            action_path = csv_dir / f"{site}_{voxel_size}_action_counts.csv"
            action_df.to_csv(action_path, index=False)
            print(f"Saved action CSV: {action_path}")
        print(
            f"Capability pass complete: {site} "
            f"(indicator_rows={len(indicator_df)}, action_rows={len(action_df)})"
        )


def main() -> None:
    _reject_legacy_root_envs()
    args = parse_args()
    sites = _parse_csv_arg(args.sites, SITES)
    scenarios = _parse_csv_arg(args.scenarios, SCENARIOS)
    years = _parse_years(args.years)

    validation_dir = engine_output_validation_dir("validation")
    validation_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta_path = validation_dir / f"v3_full_run_metadata_{stamp}.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "sites": sites,
                "scenarios": scenarios,
                "years": years,
                "voxel_size": args.voxel_size,
                "node_only": args.node_only,
                "vtk_only": args.vtk_only,
                "capabilities_only": args.capabilities_only,
                "regenerate_baselines": args.regenerate_baselines,
                "multiple_agent": args.multiple_agent,
                "save_raw_vtk": args.save_raw_vtk,
                "env": _run_config(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote run metadata: {run_meta_path}")

    if args.capabilities_only:
        print("\n===== V3: Capabilities-only mode =====\n")
        if args.regenerate_baselines:
            run_baselines(sites, voxel_size=args.voxel_size)
        else:
            print("\n===== V3: Baseline regeneration disabled =====\n")
        run_capabilities(sites, scenarios, years, voxel_size=args.voxel_size)
        return

    for site in sites:
        for scenario in scenarios:
            run_site_scenario(
                site,
                scenario,
                years,
                voxel_size=args.voxel_size,
                node_only=args.node_only,
                vtk_only=args.vtk_only,
                multiple_agent=args.multiple_agent,
                save_raw_vtk=args.save_raw_vtk,
            )

    if args.node_only:
        print("\n===== V3: Node-only mode, skipping baselines and capability pass =====\n")
        return

    if args.regenerate_baselines:
        run_baselines(sites, voxel_size=args.voxel_size)
    else:
        print("\n===== V3: Baseline regeneration disabled =====\n")

    if args.multiple_agent:
        print("\n===== V3: Multiple-agent mode, skipping capability pass for later aggregation =====\n")
        return

    run_capabilities(sites, scenarios, years, voxel_size=args.voxel_size)


if __name__ == "__main__":
    main()
