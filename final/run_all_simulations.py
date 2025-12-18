"""
Run all simulations for all sites, scenarios, and years.
Non-interactive version of a_scenario_manager.py
"""
import os
import sys

# Add the final directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a_scenario_initialiseDS
import a_scenario_runscenario
import a_scenario_generateVTKs
import a_scenario_urban_elements_count
import a_scenario_get_baselines

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
    
    # Process each year
    for year in years:
        print(f"\n----- Processing year {year} -----\n")
        
        # Run scenario simulation
        if not skip_scenario:
            print(f"Step 2.1: Running scenario simulation for year {year}")
            treeDF_scenario, logDF_scenario, poleDF_scenario = a_scenario_runscenario.run_scenario(
                site, scenario, year, voxel_size, treeDF, subsetDS, logDF, poleDF
            )
        else:
            print(f"Step 2.1: Loading existing scenario data for year {year}")
            treeDF_scenario, logDF_scenario, poleDF_scenario = a_scenario_generateVTKs.load_scenario_dataframes(
                site, scenario, year, voxel_size
            )
        
        # Generate VTKs
        print(f"Step 2.2: Generating VTKs for year {year}")
        vtk_file = a_scenario_generateVTKs.generate_vtk(
            site, scenario, year, voxel_size, subsetDS, 
            treeDF_scenario, logDF_scenario, poleDF_scenario, enable_visualization
        )
        
        if vtk_file:
            generated_vtk_files.append(vtk_file)
    
    return generated_vtk_files


def process_baseline(site, voxel_size=1):
    """Process baseline for a site."""
    output_folder = 'data/revised/final/baselines'
    baseline_vtk_path = f'{output_folder}/{site}_baseline_combined_{voxel_size}.vtk'
    
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
    
    # STEP 3: Process urban elements
    print("\n" + "=" * 60)
    print("STEP 3: PROCESSING URBAN ELEMENTS")
    print("=" * 60)
    
    for site in SITES:
        # Process baseline VTK
        if site in baseline_vtk_files and os.path.exists(baseline_vtk_files[site]):
            print(f"\nProcessing urban elements for {site} baseline")
            a_scenario_urban_elements_count.run_from_manager(
                site=site,
                voxel_size=VOXEL_SIZE,
                specific_files=[baseline_vtk_files[site]],
                should_process_baseline=True,
                enable_visualization=ENABLE_VISUALIZATION
            )
        
        # Process all scenarios
        all_scenario_files = []
        for scenario in SCENARIOS:
            for year in YEARS:
                vtk_path = f'data/revised/final/{site}/{site}_{scenario}_{VOXEL_SIZE}_scenarioYR{year}.vtk'
                if os.path.exists(vtk_path):
                    all_scenario_files.append(vtk_path)
        
        if all_scenario_files:
            print(f"\nProcessing urban elements for {site} scenarios ({len(all_scenario_files)} files)")
            a_scenario_urban_elements_count.run_from_manager(
                site=site,
                voxel_size=VOXEL_SIZE,
                specific_files=all_scenario_files,
                should_process_baseline=False,
                enable_visualization=ENABLE_VISUALIZATION
            )
    
    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

