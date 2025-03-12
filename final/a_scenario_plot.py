import os
import numpy as np
import pandas as pd
import pyvista as pv

def load_vtk(site, scenario, year, voxel_size):
    """Load a VTK file for the given parameters."""
    filepath = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk'
    if os.path.exists(filepath):
        print(f"Loading VTK file: {filepath}")
        return pv.read(filepath)
    else:
        print(f"VTK file not found: {filepath}")
        return None

def visualize_scenario(polydata, site, scenario, year, voxel_size, output_dir):
    """
    Visualize a scenario and save the image to the output directory.
    Uses exact settings from plot_scenario_rewilded in a_scenario_generateVTKs.py.
    
    Parameters:
    polydata (pyvista.PolyData): The polydata to visualize
    site (str): Site name
    scenario (str): Scenario type
    year (int): Year/timestep
    voxel_size (int): Voxel size
    output_dir (str): Output directory for images
    """
    if polydata is None:
        print(f"Skipping visualization for {site}, {scenario}, year {year} - polydata not found")
        return
    
    # Get masks from polydata (created in xarray)
    maskforTrees = polydata.point_data['maskforTrees']
    maskForRewilding = polydata.point_data['maskForRewilding']

    # Extract different polydata subsets based on the masks
    treePoly = polydata.extract_points(maskforTrees)  # Points where forest_tree_id is not NaN
    designActionPoly = polydata.extract_points(maskForRewilding)

    # Extract site voxels (neither trees nor rewilding)
    siteMask = ~(maskforTrees | maskForRewilding)
    sitePoly = polydata.extract_points(siteMask)

    # Print all point_data variables in polydata
    print(f'point_data variables in polydata: {sitePoly.point_data.keys()}')

    # Create the plotter with off-screen rendering
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    
    # Add title to the plotter
    plotter.add_text(f"Scenario at {site} after {year} years", position="upper_edge", font_size=16, color='black')

    # Add 'none' points as white cube glyphs
    # Add site points if they exist
    if sitePoly.n_points > 0:
        # Create cube glyphs for site points
        cubes = sitePoly.glyph(geom=pv.Cube(), scale=False, factor=1.0)
        plotter.add_mesh(cubes, color='white')
    else:
        print("No site points to visualize")
    
    # Add tree points if they exist
    if treePoly.n_points > 0:
        # Create cube glyphs for tree points
        tree_cubes = treePoly.glyph(geom=pv.Cube(), scale=False, factor=1.0)
        plotter.add_mesh(tree_cubes, scalars='forest_size', cmap='Set1')
    else:
        print("No tree points to visualize")
    
    # Add rewilding/design action points if they exist
    if designActionPoly.n_points > 0:
        # Create cube glyphs for rewilding points
        rewilding_cubes = designActionPoly.glyph(geom=pv.Cube(), scale=False, factor=1.0)
        plotter.add_mesh(rewilding_cubes, scalars='scenario_bioEnvelope', cmap='Set2', show_scalar_bar=True)
    else:
        print("No rewilding/design action points to visualize")
    
    plotter.enable_eye_dome_lighting()
    
    # Save the image
    output_file = f"{output_dir}/{site}_{scenario}_{voxel_size}_{year}.png"
    plotter.screenshot(output_file)
    print(f"Saved visualization to {output_file}")
    
    # Close the plotter to free resources
    plotter.close()

def batch_visualize():
    """Process all combinations of sites, scenarios, and years and generate visualizations."""
    # Default values from a_scenario_manager.py
    default_sites = ['trimmed-parade', 'city', 'uni']
    default_scenarios = ['positive', 'trending']
    default_years = [0, 10, 30, 60, 180]
    default_voxel_size = 1
    
    # Ask if user wants to use defaults or specify parameters
    use_defaults = input("Use default parameters? (yes/no, default yes): ")
    
    if use_defaults.lower() in ['no', 'n']:
        # Get user input for sites
        sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default {default_sites}: ")
        sites = sites_input.split(',') if sites_input else default_sites
        sites = [site.strip() for site in sites]
        
        # Get user input for scenarios
        scenarios_input = input(f"Enter scenario(s) to process (comma-separated) or press Enter for default {default_scenarios}: ")
        scenarios = scenarios_input.split(',') if scenarios_input else default_scenarios
        scenarios = [scenario.strip() for scenario in scenarios]
        
        # Get user input for years
        years_input = input(f"Enter years to process (comma-separated) or press Enter for default {default_years}: ")
        try:
            years = [int(year.strip()) for year in years_input.split(',')] if years_input else default_years
        except ValueError:
            print("Invalid input for years. Using default values.")
            years = default_years
        
        # Get user input for voxel size
        voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
        try:
            voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size
        except ValueError:
            print("Invalid input for voxel size. Using default value.")
            voxel_size = default_voxel_size
    else:
        sites = default_sites
        scenarios = default_scenarios
        years = default_years
        voxel_size = default_voxel_size
    
    # Print summary of selected options
    print("\n===== Processing with the following parameters =====")
    print(f"Sites: {sites}")
    print(f"Scenarios: {scenarios}")
    print(f"Years/Trimesters: {years}")
    print(f"Voxel Size: {voxel_size}")
    
    # Confirm proceeding
    confirm = input("\nProceed with these settings? (yes/no, default yes): ")
    if confirm.lower() in ['no', 'n']:
        print("Operation cancelled.")
        return
    
    # Process each site, scenario, and year
    for site in sites:
        # Create output directory for this site
        output_dir = f'data/revised/final/{site}/tempOutputs'
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n===== Processing {site} =====")
        
        for scenario in scenarios:
            print(f"\n--- Processing {scenario} scenario ---")
            
            for year in years:
                print(f"\n- Processing year {year} -")
                
                # Load the VTK file
                polydata = load_vtk(site, scenario, year, voxel_size)
                
                if polydata is not None:
                    # Visualize and save the image
                    visualize_scenario(polydata, site, scenario, year, voxel_size, output_dir)
                else:
                    print(f"Skipping visualization for {site}, {scenario}, year {year} - VTK file not found")
    
    print("\n===== All visualizations completed =====")
    print(f"Output images are saved in the 'tempOutputs' folder within each site directory")

if __name__ == "__main__":
    # Run the batch visualization
    try:
        batch_visualize()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nVisualization process completed.")