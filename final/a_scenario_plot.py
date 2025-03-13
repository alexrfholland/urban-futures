import os
import numpy as np
import pandas as pd
import pyvista as pv
import a_scenario_camera
from a_scenario_colors import (
    get_lifecycle_colormap,
    get_bioenvelope_colormap,
    map_to_lifecycle_indices,
    map_to_bioenvelope_indices,
    get_lifecycle_to_int_mapping,
    get_bioenvelope_to_int_mapping,
    get_int_to_lifecycle_mapping,
    get_int_to_bioenvelope_mapping,
    get_lifecycle_display_names,
    get_bioenvelope_display_names,
    get_lifecycle_colors,
    get_bioenvelope_colors,
    get_lifecycle_stages,
    get_bioenvelope_stages,
    normalize_rgb_colors
)

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
    """Visualize a scenario and save the image to the output directory."""
    if polydata is None:
        print(f"Skipping visualization for {site}, {scenario}, year {year} - polydata not found")
        return
    
    # Extract different polydata subsets
    maskforTrees = polydata.point_data['maskforTrees']
    maskForRewilding = polydata.point_data['maskForRewilding']
    treePoly = polydata.extract_points(maskforTrees)
    designActionPoly = polydata.extract_points(maskForRewilding)

    unique_bioenvelope_values = pd.Series(polydata.point_data['scenario_bioEnvelope']).unique()
    print(f"Unique scenario_bioEnvelope values: {unique_bioenvelope_values}")

    
    
    sitePoly = polydata.extract_points(~(maskforTrees | maskForRewilding))
    
    # Print data variables and point counts
    print(f'point_data variables in polydata: {polydata.point_data.keys()}')
    print(f'Tree points: {treePoly.n_points}, Design action points: {designActionPoly.n_points}, Site points: {sitePoly.n_points}')
    
    # Create plotter and setup camera
    plotter = pv.Plotter(off_screen=True, window_size=[3840, 2160])  
    a_scenario_camera.setup_plotter_with_lighting(plotter)
    a_scenario_camera.set_isometric_view(plotter)
    plotter.add_text(f"Scenario at {site} after {year} years", position="upper_edge", font_size=16, color='black')
    
    # Add site points
    if sitePoly.n_points > 0:
        plotter.add_mesh(sitePoly.glyph(geom=pv.Cube(), scale=False, factor=1.0), 
                         color='white', 
                         label="Site")
    else:
        print("No site points to visualize")
    
    # Get mapping dictionaries
    lifecycle_colors = normalize_rgb_colors(get_lifecycle_colors())
    int_to_lifecycle = get_int_to_lifecycle_mapping()
    lifecycle_display_names = get_lifecycle_display_names()
    
    # Add tree points with categorical legend
    has_trees = False
    if treePoly.n_points > 0 and 'forest_size' in treePoly.point_data:
        has_trees = True
        forest_size = treePoly.point_data['forest_size']
        print(f"Unique forest_size values: {np.unique(forest_size)}")
        
        # Map forest_size to lifecycle categories
        lifecycle_category = map_to_lifecycle_indices(forest_size)
        
        # Add lifecycle category to tree points
        treePoly.point_data['lifecycle_category'] = lifecycle_category
        
        # Create cube glyphs for all trees
        tree_cubes = treePoly.glyph(geom=pv.Cube(), scale=False, factor=1.0)
        
        # Add the main mesh with scalar coloring
        plotter.add_mesh(
            tree_cubes, 
            scalars='lifecycle_category',
            cmap=get_lifecycle_colormap(),
            clim=[0.5, max(int_to_lifecycle.keys()) + 0.5],
            show_scalar_bar=False
        )
            
    elif treePoly.n_points > 0:
        # Fallback for trees without forest_size
        has_trees = True
        tree_cubes = treePoly.glyph(geom=pv.Cube(), scale=False, factor=1.0)
        plotter.add_mesh(tree_cubes, 
                         scalars='forest_size', 
                         cmap='Set1', 
                         show_scalar_bar=False,
                         label="Trees")
    else:
        print("No tree points to visualize")
    
    # Get bioenvelope mappings
    bioenvelope_colors = normalize_rgb_colors(get_bioenvelope_colors())
    int_to_bioenvelope = get_int_to_bioenvelope_mapping()
    bioenvelope_display_names = get_bioenvelope_display_names()
    
    # Add rewilding points with categorical legend  
    has_bioenvelope = False
    if designActionPoly.n_points > 0 :
        bio_envelope = designActionPoly.point_data['scenario_bioEnvelope']
        print(f"Unique bioenvelope values: {np.unique(bio_envelope)}")
        
        bio_category = map_to_bioenvelope_indices(bio_envelope)
        print(f"Unique bio_category values: {np.unique(bio_category)}")
        designActionPoly.point_data['bio_category'] = bio_category
        
         # Create cube glyphs
        rewilding_cubes = designActionPoly.glyph(geom=pv.Cube(), scale=False, factor=1.0)
        
        # Add main mesh with scalar coloring
        plotter.add_mesh(
            rewilding_cubes, 
            scalars='bio_category',
            cmap=get_bioenvelope_colormap(),
            clim=[0.5, max(int_to_bioenvelope.keys()) + 0.5],
            show_scalar_bar=False
        )
  
    
    # Add legends for ALL categories
    legend_entries = []
    
    # Tree lifecycle legend (all categories)
    if has_trees:
        # Add all lifecycle stages to legend
        for stage in get_lifecycle_stages():
            display_name = lifecycle_display_names.get(stage, stage)
            # Convert RGB tuple to hex color string
            r, g, b = lifecycle_colors[stage]
            color_str = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            # Use dictionary format for legend entry
            legend_entries.append({
                "label": display_name, 
                "color": color_str,
                "face": "none"  # Use rectangle for tree lifecycle entries
            })
    
    # Bioenvelope legend (all categories)
    if has_bioenvelope:
        # For bioenvelope types with many categories, maybe want to select important ones
        important_categories = ['none', 'rewilded', 'exoskeleton', 'footprint-depaved', 
                            'livingFacade', 'greenRoof', 'brownRoof']
        
        for category in important_categories:
            if category in bioenvelope_colors:
                display_name = bioenvelope_display_names.get(category, category)
                # Convert RGB tuple to hex color string
                r, g, b = bioenvelope_colors[category]
                color_str = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                # Use dictionary format for legend entry
                legend_entries.append({
                    "label": display_name, 
                    "color": color_str,
                    "face": "circle"  # Use square for bioenvelope entries
                })
    
    # Add a legend if we have entries
    if legend_entries:
        plotter.add_legend(
            legend_entries,
            bcolor=[0.9, 0.9, 0.9],
            border=True,
            size=(0.2, 0.2),
            loc='lower right',
            face='w',
            background_opacity=0.2
        )
    
    # Finalize and save
    plotter.enable_eye_dome_lighting()
    output_file = f"{output_dir}/{site}_{scenario}_{voxel_size}_{year}.png"
    plotter.screenshot(output_file)
    print(f"Saved visualization to {output_file}")
    plotter.close()

def batch_visualize():
    """Process all combinations of sites, scenarios, and years and generate visualizations."""
    default_sites = ['trimmed-parade', 'city', 'uni']
    default_scenarios = ['positive', 'trending']
    default_years = [0, 10, 30, 60, 180]
    default_voxel_size = 1
    
    use_defaults = input("Use default parameters? (yes/no, default yes): ")
    
    if use_defaults.lower() in ['no', 'n']:
        sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default: ")
        sites = sites_input.split(',') if sites_input else default_sites
        sites = [site.strip() for site in sites]
        
        scenarios_input = input(f"Enter scenario(s) to process (comma-separated) or press Enter for default: ")
        scenarios = scenarios_input.split(',') if scenarios_input else default_scenarios
        scenarios = [scenario.strip() for scenario in scenarios]
        
        years_input = input(f"Enter years to process (comma-separated) or press Enter for default: ")
        try:
            years = [int(year.strip()) for year in years_input.split(',')] if years_input else default_years
        except ValueError:
            print("Invalid input for years. Using default values.")
            years = default_years
        
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
    
    confirm = input(f"\nProceed with these settings? (yes/no, default yes): ")
    if confirm.lower() in ['no', 'n']:
        print("Operation cancelled.")
        return
    
    for site in sites:
        output_dir = f'data/revised/final/{site}/tempOutputs'
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n===== Processing {site} =====")
        
        for scenario in scenarios:
            print(f"\n--- Processing {scenario} scenario ---")
            
            for year in years:
                print(f"\n- Processing year {year} -")
                polydata = load_vtk(site, scenario, year, voxel_size)
                if polydata:
                    visualize_scenario(polydata, site, scenario, year, voxel_size, output_dir)

if __name__ == "__main__":
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