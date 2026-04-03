import os
import importlib
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "_code-refactored"))

# Import required modules
import a_scenario_initialiseDS
import a_scenario_runscenario
import a_scenario_generateVTKs
import a_scenario_urban_elements_count
import a_scenario_get_baselines
from refactor_code.scenario import params_v3

from refactor_code.paths import (
    engine_output_validation_dir,
    scenario_baseline_combined_vtk_path,
    scenario_baseline_dir,
    scenario_log_df_path,
    scenario_pole_df_path,
    scenario_tree_df_path,
    scenario_urban_features_vtk_path,
)

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================

def check_data_files_exist(site, scenario, year, voxel_size):
    """
    Check if the scenario data files exist for the given parameters.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    year (int): Year to check
    voxel_size (int): Voxel size
    
    Returns:
    bool: True if all required files exist, False otherwise
    """
    # Handle site-specific conditions first
    check_logs = True
    check_poles = True
    
    if site == 'trimmed-parade':
        check_logs = False
        check_poles = False
    
    if site == 'uni':
        check_logs = False

    if site == 'city':
        check_poles = False
    
    # Check for the tree dataframe file
    tree_file = scenario_tree_df_path(site, scenario, year, voxel_size)
    if not os.path.exists(tree_file):
        print(f"Tree dataframe file not found: {tree_file}")
        return False
    
    # Check for log file if needed
    if check_logs:
        log_file = scenario_log_df_path(site, scenario, year, voxel_size)
        if not os.path.exists(log_file):
            print(f"Log dataframe file not found: {log_file}")
            return False
    
    # Check for pole file if needed
    if check_poles:
        pole_file = scenario_pole_df_path(site, scenario, year, voxel_size)
        if not os.path.exists(pole_file):
            print(f"Pole dataframe file not found: {pole_file}")
            return False
    
    return True

#==============================================================================
# CORE PROCESSING FUNCTIONS
#==============================================================================

def process_scenario(
    site,
    scenario,
    years,
    voxel_size,
    skip_scenario=False,
    enable_visualization=False,
    node_only=False,
):
    """
    Process a single site with the given parameters.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    years (list): List of years to process
    voxel_size (int): Voxel size
    skip_scenario (bool): Whether to skip the scenario simulation step
    enable_visualization (bool): Whether to enable visualization in VTK generation
    node_only (bool): Whether to stop after writing scenario CSV/dataframe outputs
    
    Returns:
    list: List of generated VTK file paths
    """
    print(f"\n===== Processing {site} with {scenario} scenario =====\n")
    
    #--------------------------------------------------------------------------
    # STEP 1: PREPROCESS SITE DATA
    #--------------------------------------------------------------------------
    print("Step 1: Initializing site xarray dataset")
    # Initialize site xarray
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
    
    # Collect the VTK files that are generated for later urban elements processing
    generated_vtk_files = []
    
    #--------------------------------------------------------------------------
    # STEP 2: PROCESS EACH YEAR
    #--------------------------------------------------------------------------
    years = sorted(years)
    current_tree_df = treeDF.copy()
    previous_year = 0
    telemetry_dir = engine_output_validation_dir("validation") / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    telemetry_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recruit_telemetry_path = telemetry_dir / f"{telemetry_stamp}_{site}_{scenario}_telemetry_recruit.csv"

    # Process each year
    for year in years:
        print(f"\n----- Processing year {year} -----\n")
        
        #----------------------------------------------------------------------
        # STEP 2.1: RUN SCENARIO SIMULATION
        #----------------------------------------------------------------------
        # Run scenario simulation (if not skipped)
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
                recruit_telemetry_path=recruit_telemetry_path,
            )
        else:
            # Check if scenario data files exist
            if check_data_files_exist(site, scenario, year, voxel_size):
                print(f"Step 2.1: Skipping scenario simulation, loading existing data for year {year}")
                # Load the scenario dataframes
                treeDF_scenario, logDF_scenario, poleDF_scenario = a_scenario_generateVTKs.load_scenario_dataframes(
                    site, scenario, year, voxel_size
                )
            else:
                print(f"Cannot skip scenario simulation for year {year} - required data files not found")
                print(f"Running scenario simulation instead")
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
                    recruit_telemetry_path=recruit_telemetry_path,
                )
        
        if node_only:
            print(f"Step 2.2: Node-only mode enabled, skipping VTK generation for year {year}")
        else:
            #----------------------------------------------------------------------
            # STEP 2.2: GENERATE VTKs
            #----------------------------------------------------------------------
            print(f"Step 2.2: Generating VTKs for year {year}")
            vtk_result = a_scenario_generateVTKs.generate_vtk(
                site, scenario, year, voxel_size, subsetDS.copy(deep=True),
                treeDF_scenario, logDF_scenario, poleDF_scenario, enable_visualization,
                return_polydata=True,
                save_raw_vtk=False,
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
            
            # Track the VTK file path
            if vtk_file:
                generated_vtk_files.append(vtk_file)

        current_tree_df = treeDF_scenario.copy()
        previous_year = year
    
    return generated_vtk_files

def process_baseline(site, voxel_size=1,check=False):
    """
    Process baseline for a site. Checks if baseline files exist first.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    voxel_size (int): Voxel size
    
    Returns:
    str: Path to the baseline resource VTK file
    """
    # Define output paths
    output_folder = str(scenario_baseline_dir())
    baseline_vtk_path = scenario_baseline_combined_vtk_path(site, voxel_size)
    
    # Check if baseline file already exists in validation baselines folder
    if check and os.path.exists(baseline_vtk_path):
        print(f"\n===== Using existing baseline for {site} =====")
        print(f"Found existing baseline file: {baseline_vtk_path}")
        return str(baseline_vtk_path)

    synced_baseline = a_scenario_get_baselines.sync_existing_baseline(
        site,
        voxel_size=voxel_size,
        source_mode="canonical",
        target_mode="validation",
        overwrite=False,
    )
    if synced_baseline is not None:
        print(f"\n===== Reused canonical baseline for {site} =====")
        print(f"Copied baseline assets into validation roots: {synced_baseline}")
        return str(synced_baseline)

    
    # If no existing baseline found, generate a new one
    print(f"\n===== Generating new baseline for {site} =====\n")
    
    # Generate baseline using the encapsulated function
    trees_csv, resource_vtk, terrain_vtk, combined_vtk = a_scenario_get_baselines.generate_baseline(
        site, voxel_size, output_folder
    )
    
    print(f"Baseline generation completed for {site}")
    print(f"Resource VTK: {combined_vtk}")
    
    return str(combined_vtk)

#==============================================================================
# MAIN FUNCTION
#==============================================================================

def main():
    """Main function to gather user input and process sites."""
    
    #--------------------------------------------------------------------------
    # STEP 1: GATHER USER INPUTS
    #--------------------------------------------------------------------------
    # Default values
    default_sites = ['trimmed-parade', 'city', 'uni']
    default_scenarios = ['positive', 'trending']
    default_base_years = [0, 10, 30, 60, 180]
    default_voxel_size = 1
    default_interval = 30  # Sub-timestep interval between 60 and 180
    
    # Ask for sites
    sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default {default_sites}: ")
    sites = sites_input.split(',') if sites_input else default_sites
    sites = [site.strip() for site in sites]
    
    # Ask for scenarios
    scenarios_input = input(f"Enter scenario(s) to process (comma-separated) or press Enter for default {default_scenarios}: ")
    scenarios = scenarios_input.split(',') if scenarios_input else default_scenarios
    scenarios = [scenario.strip() for scenario in scenarios]
    
    # Ask for sub-timestep interval
    interval_input = input(f"Enter sub-timestep interval between 60-180 years (default {default_interval}, 0 to disable): ")
    try:
        interval = int(interval_input) if interval_input else default_interval
        if interval <= 0:
            interval = None
    except ValueError:
        print("Invalid input for interval. Using default value.")
        interval = default_interval
    
    # Generate years with sub-timesteps
    years = params_v3.generate_timesteps(default_base_years, interval)
    print(f"Generated timesteps: {years}")
    
    # Allow user to override years
    years_input = input(f"Enter years to process (comma-separated) or press Enter to use generated {years}: ")
    try:
        if years_input:
            years = [int(year.strip()) for year in years_input.split(',')]
    except ValueError:
        print("Invalid input for years. Using generated timesteps.")
    
    # Ask for voxel size
    voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
    try:
        voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size
    except ValueError:
        print("Invalid input for voxel size. Using default value.")
        voxel_size = default_voxel_size
    
    # Ask whether to skip scenario simulation
    skip_scenario_input = input("Skip scenario simulation if data files exist? (yes/no, default no): ")
    skip_scenario = skip_scenario_input.lower() in ['yes', 'y', 'true', '1']
    
    # Ask whether to enable visualization
    vis_input = input("Enable visualization during VTK generation? (yes/no, default no): ")
    enable_visualization = vis_input.lower() in ['yes', 'y', 'true', '1']

    # Ask whether to stop after CSV/dataframe outputs
    node_only_input = input("Run node-only mode and skip all VTK generation? (yes/no, default no): ")
    node_only = node_only_input.lower() in ['yes', 'y', 'true', '1']
    
    # Print summary of selected options
    print("\n===== Processing with the following parameters =====")
    print(f"Sites: {sites}")
    print(f"Scenarios: {scenarios}")
    print(f"Sub-timestep interval: {interval if interval else 'disabled'}")
    print(f"Years/Timesteps: {years}")
    print(f"Voxel Size: {voxel_size}")
    print(f"Skip Scenario Simulation: {skip_scenario}")
    print(f"Enable Visualization: {enable_visualization}")
    print(f"Node-only Mode: {node_only}")
    
    # Confirm proceeding
    confirm = input("\nProceed with these settings? (yes/no, default yes): ")
    if confirm.lower() in ['no', 'n']:
        print("Operation cancelled.")
        return
    
    #--------------------------------------------------------------------------
    # STEP 2: PROCESS SCENARIOS
    #--------------------------------------------------------------------------
    print("\n===== STEP 2: PROCESSING SCENARIOS =====")
    # Dictionary to store all generated VTK files by site
    all_vtk_files = {site: [] for site in sites}
    
    # Process each site and scenario
    for site in sites:
        for scenario in scenarios:
            # Process site and get generated VTK files
            scenario_vtk_files = process_scenario(
                site, scenario, years, voxel_size,
                skip_scenario, enable_visualization, node_only
            )
            
            # Store the generated VTK files
            all_vtk_files[site].extend(scenario_vtk_files)
    
    #--------------------------------------------------------------------------
    # STEP 3: PROCESS BASELINES
    #--------------------------------------------------------------------------
    if node_only:
        print("\n===== STEP 3: SKIPPING BASELINES (node-only mode) =====")
    else:
        print("\n===== STEP 3: PROCESSING BASELINES =====")
        # Process baselines after scenario VTKs
        baseline_vtk_files = {}
        for site in sites:
            baseline_vtk = process_baseline(site, voxel_size)
            if baseline_vtk:
                baseline_vtk_files[site] = baseline_vtk
    
    print("\n===== All processing completed =====")

#==============================================================================
# SCRIPT ENTRY POINT
#==============================================================================

if __name__ == "__main__":
    # Run the main function
    main()
    print("Done")
