import os
import numpy as np
import pandas as pd
import pyvista as pv
import a_scenario_camera as a_vis_camera
from a_vis_colors import (
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
    normalize_rgb_colors,
    get_urban_element_colormap, 
    get_urban_element_display_names,
    map_to_urban_element_indices,
    get_urban_element_to_int_mapping,
    get_int_to_urban_element_mapping, 
    get_urban_element_stages,
    get_urban_element_colors 
)

# Define base output directory
BASE_OUTPUT_DIR = 'data/revised/final/visualizations'

def load_vtk(site, scenario, year, voxel_size):
    """Load a VTK file for the given parameters."""
    filepath = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk'
    if os.path.exists(filepath):
        print(f"Loading VTK file: {filepath}")
        return pv.read(filepath)
    else:
        print(f"VTK file not found: {filepath}")
        return None

def plot_trees_and_bioenvelope(polydata, site, scenario, year, voxel_size):
    """
    Visualize trees and bioenvelope data from the scenario.
    
    Parameters:
    polydata (pyvista.PolyData): The scenario VTK data
    site (str): Site name
    scenario (str): Scenario type
    year (int): Year/timestep
    voxel_size (int): Voxel size
    """
    if polydata is None:
        print(f"Skipping visualization for {site}, {scenario}, year {year} - polydata not found")
        return
    
    # Construct site-specific path using global BASE_OUTPUT_DIR
    site_output_dir = os.path.join(BASE_OUTPUT_DIR, site)
    # Note: Directory is assumed to be created in process_scenario
    
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
    a_vis_camera.setup_plotter_with_lighting(plotter)
    a_vis_camera.set_isometric_view(plotter)
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
    # Save directly to the site directory
    output_file = f"{site_output_dir}/{site}_{scenario}_{voxel_size}_yr{year}_trees_bioenvelope.png"
    plotter.screenshot(output_file)
    print(f"Saved trees and bioenvelope visualization to {output_file}")
    plotter.close()

def plot_urban_features_design_actions(polydata, site, scenario, year, voxel_size):
    """
    Visualize the three search variables side by side.
    
    Parameters:
    polydata (pyvista.PolyData): The scenario VTK with search variables
    site (str): Site name
    scenario (str): Scenario type
    year (int): Year/timestep
    voxel_size (int): Voxel size
    """
    if polydata is None:
        print(f"Skipping urban features visualization for {site}, {scenario}, year {year} - polydata not found")
        return
        
    print(f"\nVisualizing urban features for {site}, {scenario}, year {year}...")
    
    # Construct site-specific path using global BASE_OUTPUT_DIR
    site_output_dir = os.path.join(BASE_OUTPUT_DIR, site)
    # Note: Directory is assumed to be created in process_scenario
    
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[3840, 2160])
    # Apply camera setup before setting views
    a_vis_camera.setup_plotter_with_lighting(plotter)
    a_vis_camera.set_isometric_view(plotter) 
    
    # Common glyph geometry
    glyph_geom = pv.Cube() 
    glyph_factor = 1.0

    # Plot 1: Bioavailable envelope
    plotter.subplot(0, 0)
    # Updated font size
    plotter.add_title(f"Bioavailable envelope", font_size=16) 
    bioavailable = polydata.point_data['search_bioavailable']
    bio_mask = bioavailable != 'none'
    if np.any(bio_mask):
        bio_poly = polydata.extract_points(bio_mask)
        bio_glyphs = bio_poly.glyph(geom=glyph_geom, scale=False, factor=glyph_factor)
        plotter.add_mesh(bio_glyphs, scalars='search_bioavailable', cmap='viridis', 
                         scalar_bar_args={'title': 'Bioavailable'})
    if np.any(~bio_mask):
        rest_poly = polydata.extract_points(~bio_mask)
        rest_glyphs = rest_poly.glyph(geom=glyph_geom, scale=False, factor=glyph_factor)
        plotter.add_mesh(rest_glyphs, color='white', opacity=0.5)
    
    # Plot 2: Design Actions
    plotter.subplot(0, 1)
    # Updated font size
    plotter.add_title(f"Design Actions", font_size=16)
    design_action = polydata.point_data['search_design_action']
    design_mask = design_action != 'none'
    if np.any(design_mask):
        design_poly = polydata.extract_points(design_mask)
        design_glyphs = design_poly.glyph(geom=glyph_geom, scale=False, factor=glyph_factor)
        plotter.add_mesh(design_glyphs, scalars='search_design_action', cmap='tab10',
                         scalar_bar_args={'title': 'Design Action'})
    if np.any(~design_mask):
        rest_poly = polydata.extract_points(~design_mask)
        rest_glyphs = rest_poly.glyph(geom=glyph_geom, scale=False, factor=glyph_factor)
        plotter.add_mesh(rest_glyphs, color='white', opacity=0.5)
    
    # Plot 3: Urban Elements
    plotter.subplot(0, 2)
    # Updated font size
    plotter.add_title(f"Urban Elements", font_size=16)
    urban_elements = polydata.point_data['search_urban_elements']
    urban_mask = urban_elements != 'none'
    if np.any(urban_mask):
        urban_poly = polydata.extract_points(urban_mask)
        urban_glyphs = urban_poly.glyph(geom=glyph_geom, scale=False, factor=glyph_factor)
        plotter.add_mesh(urban_glyphs, scalars='search_urban_elements', cmap='tab20', 
                         scalar_bar_args={'title': 'Urban Element'})
    if np.any(~urban_mask):
        rest_poly = polydata.extract_points(~urban_mask)
        rest_glyphs = rest_poly.glyph(geom=glyph_geom, scale=False, factor=glyph_factor)
        plotter.add_mesh(rest_glyphs, color='white', opacity=0.5)
    
    # Add overall title
    plotter.add_text(f"{site} - {scenario} - Year {year}", position='upper_edge', 
                    font_size=16, color='black')
    
    # Link views and enable lighting
    plotter.link_views()
    plotter.enable_eye_dome_lighting()
    
    # Save the image
    # Save directly to the site directory
    output_file = f"{site_output_dir}/{site}_{scenario}_{voxel_size}_yr{year}_urban_features.png"
    plotter.screenshot(output_file)
    print(f"Saved urban features visualization to {output_file}")
    
    plotter.close()

def plot_resistance(polydata, site, scenario, year, voxel_size):
    """Visualize the combined resistance scalar field using glyphs, excluding 'arboreal' bioavailable points."""
    if polydata is None:
        print(f"Skipping resistance plot for {site}, {scenario}, year {year} - polydata not found")
        return
    if 'analysis_combined_resistance' not in polydata.point_data:
        print(f"Skipping resistance plot for {site}, {scenario}, year {year} - 'analysis_combined_resistance' not found")
        return
        
    # Filter out 'arboreal' points from search_bioavailable if the field exists
    if 'search_bioavailable' in polydata.point_data:
        mask = polydata.point_data['search_bioavailable'] != 'arboreal'
        original_count = polydata.n_points
        polydata = polydata.extract_points(mask)
        print(f"  Filtered {original_count - polydata.n_points} 'arboreal' points for resistance plot.")
        if polydata.n_points == 0:
            print(f"  Skipping resistance plot for {site}, {scenario}, year {year} - no non-arboreal points remain.")
            return
    else:
        print(f"  Warning: 'search_bioavailable' field not found. Cannot filter 'arboreal' points for resistance plot.")

    print(f"Plotting resistance for {site}, {scenario}, year {year} (filtered)... DPoints = {polydata.n_points}")
    # Construct site-specific path using global BASE_OUTPUT_DIR
    site_output_dir = os.path.join(BASE_OUTPUT_DIR, site)
    # Note: Directory is assumed to be created in process_scenario
    
    # Updated window size
    plotter = pv.Plotter(off_screen=True, window_size=[3840, 2160]) 
    a_vis_camera.setup_plotter_with_lighting(plotter)
    a_vis_camera.set_isometric_view(plotter)
    plotter.add_text(f"Resistance (excluding arboreal) at {site} - {scenario} - Year {year}", position="upper_edge", font_size=16, color='black')

    # Create cube glyphs from the filtered polydata
    glyphs = polydata.glyph(geom=pv.Cube(), scale=False, factor=1.0)
    
    # Add the glyph mesh
    plotter.add_mesh(
        glyphs, 
        scalars='analysis_combined_resistance', 
        cmap='coolwarm', 
        scalar_bar_args={'title': 'Combined Resistance'}
    )
    
    plotter.enable_eye_dome_lighting()
    # Save directly to the site directory
    output_file = f"{site_output_dir}/{site}_{scenario}_{voxel_size}_yr{year}_resistance_filtered.png" # Added filtered suffix
    plotter.screenshot(output_file)
    print(f"Saved filtered resistance visualization to {output_file}")
    plotter.close()

def plot_urban_elements_single(polydata, site, scenario, year, voxel_size):
    """Visualize urban elements, excluding arboreal points, with categorical legend."""
    if polydata is None:
        print(f"Skipping urban elements plot for {site}, {scenario}, year {year} - polydata not found")
        return
    if 'search_urban_elements' not in polydata.point_data:
        print(f"Skipping urban elements plot for {site}, {scenario}, year {year} - 'search_urban_elements' not found")
        return
        
    # Filter out 'arboreal' points from search_bioavailable if the field exists
    if 'search_bioavailable' in polydata.point_data:
        mask = polydata.point_data['search_bioavailable'] != 'arboreal'
        original_count = polydata.n_points
        polydata = polydata.extract_points(mask)
        print(f"  Filtered {original_count - polydata.n_points} 'arboreal' points for urban elements plot.")
        if polydata.n_points == 0:
            print(f"  Skipping urban elements plot for {site}, {scenario}, year {year} - no non-arboreal points remain.")
            return
    else:
        print(f"  Warning: 'search_bioavailable' field not found. Cannot filter 'arboreal' points for urban elements plot.")

    print(f"Plotting urban elements for {site}, {scenario}, year {year} (filtered)... DPoints = {polydata.n_points}")
    site_output_dir = os.path.join(BASE_OUTPUT_DIR, site)

    plotter = pv.Plotter(off_screen=True, window_size=[3840, 2160]) 
    a_vis_camera.setup_plotter_with_lighting(plotter)
    a_vis_camera.set_isometric_view(plotter)
    plotter.add_text(f"Urban Elements (excl. arboreal) at {site} - {scenario} - Year {year}", position="upper_edge", font_size=16, color='black')

    urban_elements_data = polydata.point_data['search_urban_elements']
    
    # --- Use new mapping and colormap --- 
    try:
        # Map data to indices (0 for 'none', 1+ for others)
        urban_category_indices = map_to_urban_element_indices(urban_elements_data)
        polydata.point_data['urban_category_idx'] = urban_category_indices
        
        # Get necessary items from a_vis_colors
        urban_cmap = get_urban_element_colormap()
        urban_colors_dict_normalized = normalize_rgb_colors(get_urban_element_colors())
        urban_display_names = get_urban_element_display_names()
        urban_stages = get_urban_element_stages()
        int_to_urban_map = get_int_to_urban_element_mapping()

    except NameError as e:
        print(f"Error: Missing urban element definitions/functions in a_vis_colors: {e}")
        print("Falling back to default plot for urban elements.")
        plotter.add_mesh(polydata.glyph(geom=pv.Cube(), scale=False, factor=1.0), color='gray')
        output_file = f"{site_output_dir}/{site}_{scenario}_{voxel_size}_yr{year}_urban_elements_single_fallback.png"
        plotter.screenshot(output_file)
        plotter.close()
        return
    except Exception as e:
        print(f"An unexpected error occurred during urban element setup: {e}")
        # Potentially add fallback plot here too
        plotter.close()
        return

    # Plot elements using mapped indices and custom colormap
    # Extract non-'none' points for colored plotting (indices > 0)
    colored_mask = polydata.point_data['urban_category_idx'] > 0
    if np.any(colored_mask):
        colored_poly = polydata.extract_points(colored_mask)
        colored_glyphs = colored_poly.glyph(geom=pv.Cube(), scale=False, factor=1.0)
        
        # Adjust clim based on actual indices used (excluding 0)
        max_index = np.max(colored_poly.point_data['urban_category_idx'])
        plotter.add_mesh(
            colored_glyphs, 
            scalars='urban_category_idx', 
            cmap=urban_cmap,
            clim=[0.5, max_index + 0.5], # Range for indices 1 to max_index
            show_scalar_bar=False
        )

    # Plot 'none' elements (index 0) as white glyphs separately
    none_mask = polydata.point_data['urban_category_idx'] == 0
    if np.any(none_mask):
        none_poly = polydata.extract_points(none_mask)
        if none_poly.n_points > 0:
            none_glyphs = none_poly.glyph(geom=pv.Cube(), scale=False, factor=1.0)
            plotter.add_mesh(none_glyphs, color='white', opacity=1.0) # Keep white opaque
            
    # --- Add Categorical Legend --- 
    legend_entries = []
    # Iterate through stages defined in a_vis_colors, skipping 'none'
    for stage in urban_stages:
        if stage == 'none':
            continue
        display_name = urban_display_names.get(stage, stage)
        if stage in urban_colors_dict_normalized:
            r, g, b = urban_colors_dict_normalized[stage]
            color_str = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            legend_entries.append({
                "label": display_name, 
                "color": color_str,
                "face": "rectangle" # Changed from square
            })
        else:
            print(f"Warning: Color not found for urban element stage '{stage}' in legend generation.")

    if legend_entries:
        plotter.add_legend(
            legend_entries,
            bcolor=[0.9, 0.9, 0.9],
            border=True,
            size=(0.2, 0.2),
            loc='lower right',
            face=None, # Use faces defined in entries
            background_opacity=0.2
        )

    plotter.enable_eye_dome_lighting()
    output_file = f"{site_output_dir}/{site}_{scenario}_{voxel_size}_yr{year}_urban_elements_single_filtered.png" # Add filtered suffix
    plotter.screenshot(output_file)
    print(f"Saved filtered urban elements visualization to {output_file}")
    plotter.close()

def plot_canopy_resistance(polydata, site, scenario, year, voxel_size):
    """Visualize node_CanopyResistance, excluding arboreal points and showing values < 0 as white."""
    if polydata is None:
        print(f"Skipping canopy resistance plot for {site}, {scenario}, year {year} - polydata not found")
        return
    if 'node_CanopyResistance' not in polydata.point_data:
        print(f"Skipping canopy resistance plot for {site}, {scenario}, year {year} - 'node_CanopyResistance' not found")
        return
        
    # Filter out 'arboreal' points from search_bioavailable if the field exists
    if 'search_bioavailable' in polydata.point_data:
        mask = polydata.point_data['search_bioavailable'] != 'arboreal'
        original_count = polydata.n_points
        polydata = polydata.extract_points(mask)
        print(f"  Filtered {original_count - polydata.n_points} 'arboreal' points for canopy resistance plot.")
        if polydata.n_points == 0:
            print(f"  Skipping canopy resistance plot for {site}, {scenario}, year {year} - no non-arboreal points remain.")
            return
    else:
        print(f"  Warning: 'search_bioavailable' field not found. Cannot filter 'arboreal' points for canopy resistance plot.")

    print(f"Plotting canopy resistance for {site}, {scenario}, year {year} (filtered)... DPoints = {polydata.n_points}")
    site_output_dir = os.path.join(BASE_OUTPUT_DIR, site)
    
    plotter = pv.Plotter(off_screen=True, window_size=[3840, 2160]) 
    a_vis_camera.setup_plotter_with_lighting(plotter)
    a_vis_camera.set_isometric_view(plotter)
    plotter.add_text(f"Canopy Resistance (>=0, excl. arboreal) at {site} - {scenario} - Year {year}", position="upper_edge", font_size=16, color='black')

    # Create cube glyphs
    glyphs = polydata.glyph(geom=pv.Cube(), scale=False, factor=1.0)
    
    # Determine clim range, ensuring lower bound is 0
    resistance_data = polydata.point_data['node_CanopyResistance']
    valid_resistance = resistance_data[~np.isnan(resistance_data) & (resistance_data >= 0)]
    clim_min = 0
    clim_max = np.max(valid_resistance) if len(valid_resistance) > 0 else 1 # Set default max if no valid data >= 0
    if clim_max <= clim_min: # Handle case where max is 0
      clim_max = clim_min + 1

    # Add the glyph mesh, specifying below_color and clim
    plotter.add_mesh(
        glyphs, 
        scalars='node_CanopyResistance', 
        cmap='coolwarm',
        below_color='white',
        clim=[clim_min, clim_max],
        scalar_bar_args={'title': 'Canopy Resistance (>=0)'}
    )
    
    plotter.enable_eye_dome_lighting()
    output_file = f"{site_output_dir}/{site}_{scenario}_{voxel_size}_yr{year}_canopy_resistance_filtered.png" # Added filtered suffix
    plotter.screenshot(output_file)
    print(f"Saved filtered canopy resistance visualization to {output_file}")
    plotter.close()

def run_scenario_visualizations(polydata, site, scenario, year, voxel_size):
    """Runs the visualizations specific to scenario progression."""
    print(f"\n--- Running Scenario Visualizations for {site}, {scenario}, Year {year} ---")
    # Ensure site directory exists (needed here as process_scenario is removed)
    site_output_dir = os.path.join(BASE_OUTPUT_DIR, site)
    os.makedirs(site_output_dir, exist_ok=True)
    
    plot_trees_and_bioenvelope(polydata, site, scenario, year, voxel_size)
    plot_urban_features_design_actions(polydata, site, scenario, year, voxel_size)

def run_urban_analysis_visualizations(polydata, site, scenario, year, voxel_size):
    """Runs the visualizations specific to urban analysis (typically done once)."""
    print(f"\n--- Running Urban Analysis Visualizations for {site}, {scenario}, Year {year} ---")
    # Ensure site directory exists (needed here as process_scenario is removed)
    site_output_dir = os.path.join(BASE_OUTPUT_DIR, site)
    os.makedirs(site_output_dir, exist_ok=True)
    
    plot_resistance(polydata, site, scenario, year, voxel_size)
    plot_urban_elements_single(polydata, site, scenario, year, voxel_size)
    plot_canopy_resistance(polydata, site, scenario, year, voxel_size)

def batch_visualize():
    """Process all combinations of sites, scenarios, and years and generate visualizations."""
    default_sites = ['trimmed-parade', 'city', 'uni']
    default_scenarios = ['positive', 'trending']
    default_years = [0, 10, 30, 60, 180]
    default_voxel_size = 1
    
    # --- Parameter Input Handling --- 
    use_defaults = input("Use default parameters? (yes/no, default yes): ")
    if use_defaults.lower() in ['no', 'n']:
        sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default: {default_sites} ")
        sites = [site.strip() for site in sites_input.split(',')] if sites_input else default_sites
        
        scenarios_input = input(f"Enter scenario(s) to process (comma-separated) or press Enter for default: {default_scenarios} ")
        scenarios = [scenario.strip() for scenario in scenarios_input.split(',')] if scenarios_input else default_scenarios
        
        years_input = input(f"Enter years to process (comma-separated) or press Enter for default: {default_years} ")
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
    
    print("\n===== Processing with the following parameters =====")
    print(f"Sites: {sites}")
    print(f"Scenarios: {scenarios}")
    print(f"Years/Trimesters: {years}")
    print(f"Voxel Size: {voxel_size}")
    
    confirm = input(f"\nProceed with these settings? (yes/no, default yes): ")
    if confirm.lower() in ['no', 'n']:
        print("Operation cancelled.")
        return
    
    # Use the global BASE_OUTPUT_DIR and ensure it exists
    output_dir = BASE_OUTPUT_DIR 
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Revised Processing Loop ---
    for site in sites:
        print(f"\n===== Processing Site: {site} =====")

        # 1. Urban Analysis Visualizations (once per site, using trending, year 0)
        urban_analysis_scenario = 'trending'
        urban_analysis_year = 0
        print(f"\n=== Loading data for Urban Analysis ({urban_analysis_scenario}, Year {urban_analysis_year}) ===")
        polydata_ua = load_vtk(site, urban_analysis_scenario, urban_analysis_year, voxel_size)
        if polydata_ua:
            run_urban_analysis_visualizations(polydata_ua, site, urban_analysis_scenario, urban_analysis_year, voxel_size)
        else:
            print(f"Skipping Urban Analysis for {site} - VTK not found.")

        # 2. Scenario Visualizations (COMMENTED OUT FOR NOW)
        # print(f"\n=== Processing Scenario Visualizations for {site} ===")
        # for scenario in scenarios:
        #     print(f"\n--- Scenario: {scenario} ---")
        #     for year in years:
        #         print(f"\n-- Year: {year} --")
        #         polydata_scenario = load_vtk(site, scenario, year, voxel_size)
        #         if polydata_scenario:
        #             run_scenario_visualizations(polydata_scenario, site, scenario, year, voxel_size)
        #         else:
        #             print(f"Skipping Scenario Visualization for {site}, {scenario}, Year {year} - VTK not found.")

if __name__ == "__main__":
    batch_visualize()