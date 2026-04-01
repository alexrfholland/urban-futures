"""
Full V3 candidate batch run (canonical manager path).

This script is intentionally non-interactive so validation can be repeated
exactly under explicit environment-variable roots.

Expected env vars (see _documentation-refactored/scenario_engine_v3_validation.md):
- TREE_TEMPLATE_ROOT
- REFACTOR_SCENARIO_OUTPUT_ROOT
- REFACTOR_ENGINE_OUTPUT_ROOT
- REFACTOR_STATISTICS_ROOT
- EXPORT_ALL_POINTDATA_VARIABLES

Typical usage:
  PYTHONPATH=$REPO/final .venv/bin/python final/run_full_v3_batch.py
"""

from __future__ import annotations

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
import a_scenario_params
import a_scenario_runscenario
import a_scenario_urban_elements_count

from refactor_code.paths import (
    engine_output_validation_dir,
    scenario_baseline_dir,
    scenario_urban_features_vtk_path,
)


SITES = ["trimmed-parade", "city", "uni"]
SCENARIOS = ["positive", "trending"]
YEARS = a_scenario_params.generate_timesteps(interval=30)  # [0,10,30,60,90,120,150,180]
VOXEL_SIZE = 1


def _run_config() -> dict:
    keys = [
        "TREE_TEMPLATE_ROOT",
        "REFACTOR_SCENARIO_OUTPUT_ROOT",
        "REFACTOR_ENGINE_OUTPUT_ROOT",
        "REFACTOR_STATISTICS_ROOT",
        "EXPORT_ALL_POINTDATA_VARIABLES",
        "REFACTOR_OUTPUT_MODE",
    ]
    return {key: os.environ.get(key) for key in keys}


def run_site_scenario(site: str, scenario: str, *, voxel_size: int = 1) -> None:
    print(f"\n===== V3: {site} / {scenario} =====\n")

    subsetDS = a_scenario_initialiseDS.initialize_dataset(site, voxel_size)
    treeDF, poleDF, logDF = a_scenario_initialiseDS.load_node_dataframes(site, voxel_size)

    treeDF, subsetDS = a_scenario_initialiseDS.PreprocessData(treeDF, subsetDS, None)
    subsetDS, _initial_poly = a_scenario_initialiseDS.further_xarray_processing(subsetDS)

    if logDF is not None:
        logDF = a_scenario_initialiseDS.log_processing(logDF, subsetDS)
    if poleDF is not None:
        poleDF = a_scenario_initialiseDS.pole_processing(poleDF, None, subsetDS)

    current_tree_df = treeDF.copy()
    previous_year = 0

    for year in YEARS:
        print(f"\n----- V3: {site} / {scenario} / yr{year} -----\n")

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
            save_raw_vtk=False,
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


def run_baselines(*, voxel_size: int = 1) -> None:
    output_folder = str(scenario_baseline_dir())
    print(f"\n===== V3: Baselines (output_folder={output_folder}) =====\n")
    for site in SITES:
        print(f"\n----- V3: baseline / {site} -----\n")
        a_scenario_get_baselines.generate_baseline(site, voxel_size, output_folder, visualize=False)


def run_capabilities(*, voxel_size: int = 1) -> None:
    print("\n===== V3: Capability Pass (state_with_indicators + indicator CSVs) =====\n")
    for site in SITES:
        # Keep this explicit so it cannot silently omit the 90/120/150 years.
        indicator_df, action_df = a_info_gather_capabilities.process_site(
            site,
            scenarios=SCENARIOS,
            years=YEARS,
            voxel_size=voxel_size,
            save_vtk=True,
            include_baseline=True,
            output_mode="validation",
        )
        print(
            f"Capability pass complete: {site} "
            f"(indicator_rows={len(indicator_df)}, action_rows={len(action_df)})"
        )


def main() -> None:
    validation_dir = engine_output_validation_dir("validation")
    validation_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta_path = validation_dir / f"v3_full_run_metadata_{stamp}.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "sites": SITES,
                "scenarios": SCENARIOS,
                "years": YEARS,
                "voxel_size": VOXEL_SIZE,
                "env": _run_config(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote run metadata: {run_meta_path}")

    for site in SITES:
        for scenario in SCENARIOS:
            run_site_scenario(site, scenario, voxel_size=VOXEL_SIZE)

    run_baselines(voxel_size=VOXEL_SIZE)
    run_capabilities(voxel_size=VOXEL_SIZE)


if __name__ == "__main__":
    main()

