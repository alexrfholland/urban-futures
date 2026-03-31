"""
Run all simulations for all sites, scenarios, and years.
Non-interactive version of a_scenario_manager.py
"""
import os
import sys

# Add the final directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_code-refactored"))

import a_scenario_initialiseDS
import a_scenario_runscenario
import a_scenario_generateVTKs
import a_scenario_urban_elements_count
import a_scenario_get_baselines
from refactor_code.paths import (
    scenario_baseline_dir,
    scenario_urban_features_vtk_path,
)

# Configuration
SITES = ['trimmed-parade', 'city', 'uni']
SCENARIOS = ['positive', 'trending']
YEARS = [0, 10, 30, 60, 180]
VOXEL_SIZE = 1
SKIP_SCENARIO = False
ENABLE_VISUALIZATION = False


def process_scenario(site, scenario, years, voxel_size, skip_scenario=False, enable_visualization=False):
    """Process a single site with the given parameters."""
    print(f"\n===== Processing {site} with {scenario} scenario =====\n")
    
    # Initialize site xarray
    print("Step 1: Initializing site xarray dataset")
    subsetDS = a_scenario_initialiseDS.initialize_dataset(site, voxel_size)
    
    # Load initial tree, log, and pole dataframes
    print("Loading initial dataframes...")
    treeDF, poleDF, logDF = a_scenario_initialiseDS.load_node_dataframes(site, voxel_size)
    
    # Preprocess the dataframes
    print("Preprocessing dataframes...")
    treeDF, subsetDS = a_scenario_initialiseDS.PreprocessData(treeDF, subsetDS, None)
    subsetDS, initialPoly = a_scenario_initialiseDS.further_xarray_processing(subsetDS)
    
    # Process logs and poles if needed
    if logDF is not None:
        print("Processing log dataframe...")
        logDF = a_scenario_initialiseDS.log_processing(logDF, subsetDS)
    
    if poleDF is not None:
        print("Processing pole dataframe...")
        poleDF = a_scenario_initialiseDS.pole_processing(poleDF, None, subsetDS)
    
    generated_vtk_files = []
    
    years = sorted(years)
    current_tree_df = treeDF.copy()
    previous_year = 0

    # Process each year
    for year in years:
        print(f"\n----- Processing year {year} -----\n")
        
        # Run scenario simulation
        if not skip_scenario:
            print(f"Step 2.1: Running scenario simulation for year {year}")
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
        else:
            print(f"Step 2.1: Loading existing scenario data for year {year}")
            treeDF_scenario, logDF_scenario, poleDF_scenario = a_scenario_generateVTKs.load_scenario_dataframes(
                site, scenario, year, voxel_size
            )
        
        # Generate VTKs
        print(f"Step 2.2: Generating VTKs for year {year}")
        vtk_result = a_scenario_generateVTKs.generate_vtk(
            site, scenario, year, voxel_size, subsetDS.copy(deep=True),
            treeDF_scenario, logDF_scenario, poleDF_scenario, enable_visualization,
            return_polydata=True,
        )

        vtk_file, state_polydata = vtk_result
        a_scenario_urban_elements_count.process_scenario_polydata(
            state_polydata,
            site=site,
            voxel_size=voxel_size,
            scenario=scenario,
            year=year,
            save_path=scenario_urban_features_vtk_path(site, scenario, year, voxel_size),
            enable_visualization=enable_visualization,
        )
        
        if vtk_file:
            generated_vtk_files.append(vtk_file)

        current_tree_df = treeDF_scenario.copy()
        previous_year = year
    
    return generated_vtk_files


def process_baseline(site, voxel_size=1):
    """Process baseline for a site."""
    synced_baseline = a_scenario_get_baselines.sync_existing_baseline(
        site,
        voxel_size=voxel_size,
        source_mode="canonical",
        target_mode="validation",
        overwrite=False,
    )
    if synced_baseline is not None:
        print(f"\n===== Reused canonical baseline for {site} =====\n")
        print(f"Copied baseline assets into validation roots: {synced_baseline}")
        return str(synced_baseline)

    output_folder = str(scenario_baseline_dir())
    
    print(f"\n===== Generating baseline for {site} =====\n")
    trees_csv, resource_vtk, terrain_vtk, combined_vtk = a_scenario_get_baselines.generate_baseline(
        site, voxel_size, output_folder
    )
    
    print(f"Baseline generation completed for {site}")
    return combined_vtk


def main():
    print("=" * 60)
    print("RUNNING ALL SIMULATIONS")
    print("=" * 60)
    print(f"Sites: {SITES}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Years: {YEARS}")
    print(f"Voxel Size: {VOXEL_SIZE}")
    print("=" * 60)
    
    all_vtk_files = {site: [] for site in SITES}
    
    # STEP 1: Process scenarios
    print("\n" + "=" * 60)
    print("STEP 1: PROCESSING SCENARIOS")
    print("=" * 60)
    
    for site in SITES:
        for scenario in SCENARIOS:
            scenario_vtk_files = process_scenario(
                site, scenario, YEARS, VOXEL_SIZE,
                SKIP_SCENARIO, ENABLE_VISUALIZATION
            )
            all_vtk_files[site].extend(scenario_vtk_files)
    
    # STEP 2: Process baselines
    print("\n" + "=" * 60)
    print("STEP 2: PROCESSING BASELINES")
    print("=" * 60)
    
    baseline_vtk_files = {}
    for site in SITES:
        baseline_vtk = process_baseline(site, VOXEL_SIZE)
        if baseline_vtk:
            baseline_vtk_files[site] = baseline_vtk
    
    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
