import os
import importlib

# Import required modules
import a_scenario_initialiseDS
# Import a_scenario_runscenario but don't use it directly in imports to avoid potential circular dependencies
import a_scenario_runscenario
import a_scenario_generateVTKs
import a_scenario_urban_elements_count
import a_scenario_get_baselines

# Note: a_scenario_generateVTKs now imports assign_rewilded_status directly from a_scenario_runscenario

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
    filepath = f'data/revised/final/{site}'
    
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
    tree_file = f'{filepath}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv'
    if not os.path.exists(tree_file):
        print(f"Tree dataframe file not found: {tree_file}")
        return False
    
    # Check for log file if needed
    if check_logs:
        log_file = f'{filepath}/{site}_{scenario}_{voxel_size}_logDF_{year}.csv'
        if not os.path.exists(log_file):
            print(f"Log dataframe file not found: {log_file}")
            return False
    
    # Check for pole file if needed
    if check_poles:
        pole_file = f'{filepath}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv'
        if not os.path.exists(pole_file):
            print(f"Pole dataframe file not found: {pole_file}")
            return False
    
    return True

#==============================================================================
# CORE PROCESSING FUNCTIONS
#==============================================================================

def process_scenario(site, scenario, years, voxel_size, skip_scenario=False, enable_visualization=False):
    """
    Process a single site with the given parameters.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    years (list): List of years to process
    voxel_size (int): Voxel size
    skip_scenario (bool): Whether to skip the scenario simulation step
    enable_visualization (bool): Whether to enable visualization in VTK generation
    
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
                site, scenario, year, voxel_size, treeDF, subsetDS, logDF, poleDF
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
                    site, scenario, year, voxel_size, treeDF, subsetDS, logDF, poleDF
                )
        
        #----------------------------------------------------------------------
        # STEP 2.2: GENERATE VTKs
        #----------------------------------------------------------------------
        print(f"Step 2.2: Generating VTKs for year {year}")
        vtk_file = a_scenario_generateVTKs.generate_vtk(
            site, scenario, year, voxel_size, subsetDS, 
            treeDF_scenario, logDF_scenario, poleDF_scenario, enable_visualization
        )
        
        # Track the VTK file path
        if vtk_file:
            generated_vtk_files.append(vtk_file)
    
    return generated_vtk_files

def process_baseline(site, voxel_size=1):
    """
    Process baseline for a site. Checks if baseline files exist first.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    voxel_size (int): Voxel size
    
    Returns:
    str: Path to the baseline resource VTK file
    """
    # Define output paths
    output_folder = 'data/revised/final/baselines'
    resource_vtk_path = f'{output_folder}/{site}_baseline_resources_{voxel_size}.vtk'
    
    # Check if baseline file already exists in baselines folder
    if os.path.exists(resource_vtk_path):
        print(f"\n===== Using existing baseline for {site} =====")
        print(f"Found existing baseline file: {resource_vtk_path}")
        return resource_vtk_path
    
    # Alternative path to check in site-specific folder
    alt_path = f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}.vtk'
    if os.path.exists(alt_path):
        print(f"\n===== Using existing baseline for {site} (alternative location) =====")
        print(f"Found existing baseline file: {alt_path}")
        
        # Copy to standardized location in baselines folder
        import shutil
        os.makedirs(output_folder, exist_ok=True)
        print(f"Copying baseline file to standardized location...")
        shutil.copy2(alt_path, resource_vtk_path)
        print(f"Copied {alt_path} to {resource_vtk_path}")
        
        return resource_vtk_path
    
    # If no existing baseline found, generate a new one
    print(f"\n===== Generating new baseline for {site} =====\n")
    
    # Generate baseline using the encapsulated function
    trees_csv, resource_vtk, terrain_vtk, combined_vtk = a_scenario_get_baselines.generate_baseline(
        site, voxel_size, output_folder
    )
    
    print(f"Baseline generation completed for {site}")
    print(f"Resource VTK: {resource_vtk}")
    
    return resource_vtk

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
    default_years = [0, 10, 30, 60, 180]
    default_voxel_size = 1
    
    # Ask for sites
    sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default {default_sites}: ")
    sites = sites_input.split(',') if sites_input else default_sites
    sites = [site.strip() for site in sites]
    
    # Ask for scenarios
    scenarios_input = input(f"Enter scenario(s) to process (comma-separated) or press Enter for default {default_scenarios}: ")
    scenarios = scenarios_input.split(',') if scenarios_input else default_scenarios
    scenarios = [scenario.strip() for scenario in scenarios]
    
    # Ask for years/trimesters
    years_input = input(f"Enter years to process (comma-separated) or press Enter for default {default_years}: ")
    try:
        years = [int(year.strip()) for year in years_input.split(',')] if years_input else default_years
    except ValueError:
        print("Invalid input for years. Using default values.")
        years = default_years
    
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
    
    # Print summary of selected options
    print("\n===== Processing with the following parameters =====")
    print(f"Sites: {sites}")
    print(f"Scenarios: {scenarios}")
    print(f"Years/Trimesters: {years}")
    print(f"Voxel Size: {voxel_size}")
    print(f"Skip Scenario Simulation: {skip_scenario}")
    print(f"Enable Visualization: {enable_visualization}")
    
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
                skip_scenario, enable_visualization
            )
            
            # Store the generated VTK files
            all_vtk_files[site].extend(scenario_vtk_files)
    
    #--------------------------------------------------------------------------
    # STEP 3: PROCESS BASELINES
    #--------------------------------------------------------------------------
    print("\n===== STEP 3: PROCESSING BASELINES =====")
    # Process baselines after scenario VTKs
    baseline_vtk_files = {}
    for site in sites:
        baseline_vtk = process_baseline(site, voxel_size)
        if baseline_vtk:
            baseline_vtk_files[site] = baseline_vtk
    
    #--------------------------------------------------------------------------
    # STEP 4: PROCESS URBAN ELEMENTS
    #--------------------------------------------------------------------------
    print("\n===== STEP 4: PROCESSING URBAN ELEMENTS =====")
    # Process urban elements for all sites
    for site in sites:
        # Process each scenario
        for scenario in scenarios:
            print(f"\nProcessing urban elements for {site} - {scenario}")
            
            # Construct the list of VTK files for this site and scenario
            scenario_files = []
            for year in years:
                vtk_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk'
                if os.path.exists(vtk_path):
                    scenario_files.append(vtk_path)
                    print(f"Found VTK file: {os.path.basename(vtk_path)}")
                else:
                    print(f"Warning: VTK file not found: {os.path.basename(vtk_path)}")
            
            # Process the urban elements
            processed_files = a_scenario_urban_elements_count.run_from_manager(
                site=site,
                scenario=scenario,
                years=years,
                voxel_size=voxel_size,
                specific_files=scenario_files,
                process_baseline=False
            )
            
            if processed_files:
                print(f"Successfully processed {len(processed_files)} urban element files for {site} - {scenario}")
            else:
                print(f"No urban element files were processed for {site} - {scenario}")
        
        # Process baseline VTK
        if site in baseline_vtk_files and os.path.exists(baseline_vtk_files[site]):
            print(f"\nProcessing urban elements for {site} baseline")
            baseline_file = baseline_vtk_files[site]
            print(f"Baseline file: {os.path.basename(baseline_file)}")
            
            # Process the urban elements for baseline
            processed_baseline = a_scenario_urban_elements_count.run_from_manager(
                site=site,
                voxel_size=voxel_size,
                specific_files=[baseline_file],
                process_baseline=True
            )
            
            if processed_baseline:
                print(f"Successfully processed baseline urban elements for {site}")
            else:
                print(f"Failed to process baseline urban elements for {site}")
    
    print("\n===== All processing completed =====")

#==============================================================================
# SCRIPT ENTRY POINT
#==============================================================================

if __name__ == "__main__":
    # Run the main function
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nThank you for using Scenario Manager.")