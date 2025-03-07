import os
import importlib

# Import required modules
import a_scenario_initialiseDS
import a_scenario_runscenario
import a_scenario_generateVTKs

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

def process_site(site, scenario, years, voxel_size, skip_scenario=False, enable_visualization=False):
    """
    Process a single site with the given parameters.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    years (list): List of years to process
    voxel_size (int): Voxel size
    skip_scenario (bool): Whether to skip the scenario simulation step
    enable_visualization (bool): Whether to enable visualization in VTK generation
    """
    print(f"\n===== Processing {site} with {scenario} scenario =====\n")
    
    # Step 1: Initialize site xarray
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
    
    # Process each year
    for year in years:
        print(f"\n----- Processing year {year} -----\n")
        
        # Step 2: Run scenario simulation (if not skipped)
        if not skip_scenario:
            print(f"Step 2: Running scenario simulation for year {year}")
            treeDF_scenario, logDF_scenario, poleDF_scenario = a_scenario_runscenario.run_scenario(
                site, scenario, year, voxel_size, treeDF, subsetDS, logDF, poleDF
            )
        else:
            # Check if scenario data files exist
            if check_data_files_exist(site, scenario, year, voxel_size):
                print(f"Step 2: Skipping scenario simulation, loading existing data for year {year}")
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
        
        # Step 3: Generate VTKs
        print(f"Step 3: Generating VTKs for year {year}")
        a_scenario_generateVTKs.generate_vtk(
            site, scenario, year, voxel_size, subsetDS, 
            treeDF_scenario, logDF_scenario, poleDF_scenario, enable_visualization
        )
    
    print(f"\nProcessing complete for {site} with {scenario} scenario")

def main():
    """Main function to gather user input and process sites."""
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
    years_input = input(f"Enter years/trimesters to process (comma-separated) or press Enter for default {default_years}: ")
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
    
    # Process each site and scenario
    for site in sites:
        for scenario in scenarios:
            process_site(
                site, scenario, years, voxel_size,
                skip_scenario, enable_visualization
            )
    
    print("\n===== All processing completed =====")

if __name__ == "__main__":
    # Display welcome message
    print("\n===== Scenario Manager =====")
    print("This tool manages the workflow for scenario simulation and VTK generation.")
    print("You will be prompted for various options to configure the process.\n")
    
    # Check if required modules are available
    required_modules = [
        'a_scenario_initialiseDS', 
        'a_scenario_runscenario', 
        'a_scenario_generateVTKs'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Error: The following required modules are missing: {', '.join(missing_modules)}")
        print("Please make sure all required scripts are in the same directory.")
        exit(1)
    
    # Run the main function
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nThank you for using Scenario Manager.")