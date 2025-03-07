import numpy as np
import pyvista as pv
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
import a_helper_functions
from scipy.spatial import cKDTree

# Configuration variables
SITES = ['trimmed-parade']
SCENARIOS = ['positive', 'trending']
VOXEL_SIZE = 1
PROCESS_BASELINE = True  # Set to True to process baseline file

#also do baseline: polyfilePath = f'{output_folder}/{site}_baseline_resources_{voxel_size}.vtk'
#search_bioavailable: 'traversable'
#search_design_action: 'rewilded'
#search_urban_elements: 'none'

def load_xarray(site, voxel_size):
    """Load xarray dataset for the site"""
    print(f"Loading xarray for site {site}...")
    
    # Construct path to xarray file - using the exact same path as in a_create_resistance_grid.py
    input_folder = f'data/revised/final/{site}'
    xarray_path = f'{input_folder}/{site}_{voxel_size}_voxelArray_withLogs.nc'
    
    try:
        ds = xr.open_dataset(xarray_path)
        print(f"Loaded xarray from {xarray_path}")
        return ds
    except Exception as e:
        print(f"Error loading xarray from {xarray_path}: {e}")
        return None

def preprocess_site_vtk(site_vtk):
    """Create analysis layers in the site VTK"""
    print("Preprocessing site VTK...")
    
    # Create analysis_busyRoadway
    if 'road_roadInfo_type' in site_vtk.point_data and 'road_terrainInfo_roadCorridors_str_type' in site_vtk.point_data:
        road_type = site_vtk.point_data['road_roadInfo_type']
        corridor_type = site_vtk.point_data['road_terrainInfo_roadCorridors_str_type']
        
        busy_roadway = (road_type == 'Carriageway') & (corridor_type == 'Council Major')
        site_vtk.point_data['analysis_busyRoadway'] = busy_roadway
        print(f"  Created analysis_busyRoadway with {np.sum(busy_roadway):,} positive values")
    
    # Create analysis_Roadway
    if 'road_roadInfo_type' in site_vtk.point_data:
        road_type = site_vtk.point_data['road_roadInfo_type']
        roadway = (road_type == 'Carriageway')
        site_vtk.point_data['analysis_Roadway'] = roadway
        print(f"  Created analysis_Roadway with {np.sum(roadway):,} positive values")
    
    # Create analysis_forestLane
    if 'road_terrainInfo_forest' in site_vtk.point_data:
        forest = site_vtk.point_data['road_terrainInfo_forest']
        if np.issubdtype(forest.dtype, np.number):
            forest_lane = np.zeros(site_vtk.n_points)
            valid_mask = forest > 0
            if np.any(valid_mask):
                forest_lane[valid_mask] = forest[valid_mask]
                site_vtk.point_data['analysis_forestLane'] = forest_lane
                print(f"  Created analysis_forestLane with {np.sum(valid_mask):,} positive values")
    
    # Create analysis_Canopies
    canopies = np.zeros(site_vtk.n_points, dtype=bool)
    if 'site_canopy_isCanopy' in site_vtk.point_data:
        canopies |= (site_vtk.point_data['site_canopy_isCanopy'] == 1)
    if 'road_canopy_isCanopy' in site_vtk.point_data:
        canopies |= (site_vtk.point_data['road_canopy_isCanopy'] == 1)
    site_vtk.point_data['analysis_Canopies'] = canopies
    print(f"  Created analysis_Canopies with {np.sum(canopies):,} positive values")
    
    return site_vtk

def transfer_site_features_to_scenario(site_vtk, scenario_vtk):
    """Transfer site features to scenario VTK using KDTree with vectorized operations"""
    # List of variables to transfer
    variables_to_transfer = [
        # Site variables
        'site_building_element',
        'site_canopy_isCanopy',
        
        # Road/Terrain variables
        'road_terrainInfo_roadCorridors_str_type',
        'road_roadInfo_type',
        'road_terrainInfo_forest',
        'road_terrainInfo_isOpenSpace',
        'road_terrainInfo_isParkingMedian3mBuffer',
        'road_terrainInfo_isLittleStreet',
        'road_terrainInfo_isParking',
        'road_canopy_isCanopy',
        
        #Canopy variables
        'site_canopy_isCanopy',
        
        # Pole variables
        'poles_pole_type',
        
        # Envelope variables
        'envelope_roofType',
        
        # Analysis variables
        'analysis_busyRoadway',
        'analysis_Roadway',
        'analysis_forestLane',
        'analysis_Canopies'
    ]
    
    print("Transferring site features to scenario VTK...")
    
    # Build KDTree from scenario points
    scenario_points = np.vstack([
        scenario_vtk.points[:, 0],
        scenario_vtk.points[:, 1],
        scenario_vtk.points[:, 2]
    ]).T
    
    # Build KDTree from site points
    site_points = np.vstack([
        site_vtk.points[:, 0],
        site_vtk.points[:, 1],
        site_vtk.points[:, 2]
    ]).T
    
    # Create KDTree
    tree = cKDTree(site_points)
    
    # Find nearest neighbors for all scenario points at once
    # Add a maximum distance of 1 meter
    distances, indices = tree.query(scenario_points, k=1, distance_upper_bound=1.0)
    
    # Create a mask for valid indices (those within the distance threshold)
    valid_mask = np.isfinite(distances)
    print(f"  {np.sum(valid_mask):,} of {len(valid_mask):,} points are within 1.0m distance")
    
    # Transfer variables
    for var in variables_to_transfer:
        if var in site_vtk.point_data:
            # Get data from site VTK
            site_data = site_vtk.point_data[var]
            
            # Create new data array for scenario
            if np.issubdtype(site_data.dtype, np.number):
                # For numeric data, initialize with zeros or NaN
                if np.issubdtype(site_data.dtype, np.integer):
                    new_data = np.zeros(scenario_vtk.n_points, dtype=site_data.dtype)
                else:
                    new_data = np.full(scenario_vtk.n_points, np.nan, dtype=site_data.dtype)
            else:
                # For string or other data, initialize with empty strings or 'none'
                if site_data.dtype.kind == 'U':
                    new_data = np.full(scenario_vtk.n_points, 'none', dtype=site_data.dtype)
                else:
                    new_data = np.zeros(scenario_vtk.n_points, dtype=site_data.dtype)
            
            # Only transfer data for valid indices (within distance threshold)
            new_data[valid_mask] = site_data[indices[valid_mask]]
            
            # Add to scenario with FEATURES- prefix
            feature_name = f'FEATURES-{var}'
            scenario_vtk.point_data[feature_name] = new_data
            print(f"  Transferred {var} to {feature_name}")
        else:
            print(f"  Variable {var} not found in site VTK")

    #scenario_vtk.plot(scalars='FEATURES-road_roadInfo_type', cmap='tab10')
    
    return scenario_vtk

def create_bioavailablity_layer(scenario_vtk):
    """Create the search_bioavailable layer with 'open space', 'low-vegetation', and 'arboreal' categories"""
    print("Creating bioavailable habitat type layer...")
    
    # Initialize with 'none'
    search_bioavailable = np.full(scenario_vtk.n_points, 'none', dtype='<U20')
    
    # Create bioavailable and low-vegetation masks using existing logic
    bioavailable_mask = np.zeros(scenario_vtk.n_points, dtype=bool)
    low_vegetation_mask = np.zeros(scenario_vtk.n_points, dtype=bool)
    
    # Check for rewilding plantings
    if 'scenario_rewildingPlantings' in scenario_vtk.point_data:
        bioavailable_mask |= (scenario_vtk.point_data['scenario_rewildingPlantings'] > 0)
        low_vegetation_mask |= (scenario_vtk.point_data['scenario_rewildingPlantings'] > 0)
    
    # Check for rewilded areas
    if 'scenario_rewilded' in scenario_vtk.point_data:
        bioavailable_mask |= (scenario_vtk.point_data['scenario_rewilded'] != 'none')
        low_vegetation_mask |= (scenario_vtk.point_data['scenario_rewilded'] != 'none')
    
    # Check for bio envelope
    if 'scenario_bioEnvelope' in scenario_vtk.point_data:
        bioavailable_mask |= (scenario_vtk.point_data['scenario_bioEnvelope'] != 'none')
        low_vegetation_mask |= (scenario_vtk.point_data['scenario_bioEnvelope'] != 'none')
    
    # Check forest_size - add all non-'nan' points (only for bioavailable, not low-vegetation)
    if 'forest_size' in scenario_vtk.point_data:
        forest_size = scenario_vtk.point_data['forest_size']
        
        # Add all points where forest_size is not 'nan'
        if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
            bioavailable_mask |= (forest_size != 'nan')
        else:  # Numeric types
            bioavailable_mask |= ~np.isnan(forest_size)
        
        # Add points where forest_size is 'fallen'
        if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
            fallen_mask = (forest_size == 'fallen')
            bioavailable_mask |= fallen_mask
            low_vegetation_mask |= fallen_mask
    
    # Check for fallen logs
    if 'resource_fallen log' in scenario_vtk.point_data:
        fallen_log_mask = (scenario_vtk.point_data['resource_fallen log'] > 0)
        bioavailable_mask |= fallen_log_mask
        low_vegetation_mask |= fallen_log_mask
    
    # Set 'open space' where road_terrainInfo_isOpenSpace == 1
    if 'FEATURES-road_terrainInfo_isOpenSpace' in scenario_vtk.point_data:
        open_space_mask = scenario_vtk.point_data['FEATURES-road_terrainInfo_isOpenSpace'] == 1
        search_bioavailable[open_space_mask] = 'open space'
        print(f"  Open space points: {np.sum(open_space_mask):,}")
    
    # Set 'low-vegetation' for low-vegetation areas (excluding open spaces)
    low_veg_not_open = low_vegetation_mask & (search_bioavailable == 'none')
    search_bioavailable[low_veg_not_open] = 'low-vegetation'
    print(f"  Low-vegetation points: {np.sum(low_veg_not_open):,}")
    
    # Set 'arboreal' for bioavailable areas (excluding open spaces and low-vegetation)
    arboreal_mask = bioavailable_mask & (search_bioavailable == 'none')
    search_bioavailable[arboreal_mask] = 'arboreal'
    print(f"  Arboreal points: {np.sum(arboreal_mask):,}")
    
    # Add to scenario
    scenario_vtk.point_data['search_bioavailable'] = search_bioavailable
    
    print(f"  Total bioavailable habitat points: {np.sum(search_bioavailable != 'none'):,}")
    
    return scenario_vtk

def create_design_action_layer(scenario_vtk):
    """Create the search_design_action layer
    design actions can be:
    [list of design actions]
    """
    print("Creating design action layer...")
    search_design_action = np.full(scenario_vtk.n_points, 'none', dtype='<U20')
    
    # Set rewilded areas
    if 'scenario_rewildingPlantings' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['scenario_rewildingPlantings'] > 0
        search_design_action[mask] = 'rewilded'
    
    # Override with specific rewilded types
    if 'scenario_rewilded' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['scenario_rewilded'] != 'none'
        search_design_action[mask] = scenario_vtk.point_data['scenario_rewilded'][mask]
    
    # Override with bio envelope types
    if 'scenario_bioEnvelope' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['scenario_bioEnvelope'] != 'none'
        search_design_action[mask] = scenario_vtk.point_data['scenario_bioEnvelope'][mask]
    
    # Add improved-tree design action
    if 'forest_control' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['forest_control'] == 'improved-tree'
        search_design_action[mask] = 'improved-tree'
        print(f"  Added {np.sum(mask):,} improved-tree points to design actions")
    
    scenario_vtk.point_data['search_design_action'] = search_design_action
    
    # Print non-none value counts for search_design_action
    actions, action_counts = np.unique(search_design_action[search_design_action != 'none'], return_counts=True)
    if len(actions) > 0:
        print("  Design actions found:")
        for action, count in zip(actions, action_counts):
            print(f"    {action}: {count:,}")
    else:
        print("  No design actions found")
    
    return scenario_vtk

def create_urban_elements_layer(scenario_vtk):
    """Create the search_urban_elements layer"""
    print("Creating urban elements layer...")

    #print all point data variables in scenario_vtk
    print(scenario_vtk.point_data.keys())
    
    search_urban_elements = np.full(scenario_vtk.n_points, 'none', dtype='<U20')
    
    # Apply urban element classifications in order
    
    #tree resources
    if 'resource_other' in scenario_vtk.point_data:
        mask = ~np.isnan(scenario_vtk.point_data['resource_other'])
        search_urban_elements[mask] = 'arboreal'
        print(f"  arboreal points: {np.sum(mask):,}")
    else:
        print("  'resource_other' not found in point data")

    # Add tree types based on forest_size
    if 'forest_size' in scenario_vtk.point_data:
        forest_size = scenario_vtk.point_data['forest_size']
        
        # Get points where forest_size is not 'nan'
        if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
            valid_forest = forest_size != 'nan'
        else:  # Numeric types
            valid_forest = ~np.isnan(forest_size)
        
        # For each unique forest_size, create a tree_X category
        unique_sizes = np.unique(forest_size[valid_forest])
        for size in unique_sizes:
            if size != 'nan' and size != 'none' and size != '':
                # Create mask for this size
                if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':
                    size_mask = (forest_size == size) & valid_forest
                else:
                    size_mask = (forest_size == size) & valid_forest
                
                # Set urban element to tree_X
                tree_type = f'tree_{size}'
                search_urban_elements[size_mask] = tree_type
                print(f"  {tree_type} points: {np.sum(size_mask):,}")
    else:
        print("  'forest_size' not found in point data")

    if 'FEATURES-road_terrainInfo_isOpenSpace' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['FEATURES-road_terrainInfo_isOpenSpace'] == 1
        search_urban_elements[mask] = 'open space'
        print(f"  Open space points: {np.sum(mask):,}")
    else:
        print("  'FEATURES-road_terrainInfo_isOpenSpace' not found in point data")
    
    if 'FEATURES-envelope_roofType' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['FEATURES-envelope_roofType'] == 'green roof'
        search_urban_elements[mask] = 'green roof'
        print(f"  Green roof points: {np.sum(mask):,}")
        
        mask = scenario_vtk.point_data['FEATURES-envelope_roofType'] == 'brown roof'
        search_urban_elements[mask] = 'brown roof'
        print(f"  Brown roof points: {np.sum(mask):,}")
    else:
        print("  'FEATURES-envelope_roofType' not found in point data")
    
    if 'FEATURES-site_building_element' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['FEATURES-site_building_element'] == 'facade'
        search_urban_elements[mask] = 'facade'
        print(f"  Facade points: {np.sum(mask):,}")
    else:
        print("  'FEATURES-site_building_element' not found in point data")
    
    # Use analysis layers for roadway classifications
    if 'FEATURES-analysis_Roadway' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['FEATURES-analysis_Roadway']
        search_urban_elements[mask] = 'roadway'
        print(f"  Roadway points: {np.sum(mask):,}")
    else:
        print("  'FEATURES-analysis_Roadway' not found in point data")
    
    if 'FEATURES-analysis_busyRoadway' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['FEATURES-analysis_busyRoadway']
        search_urban_elements[mask] = 'busy roadway'
        print(f"  Busy roadway points: {np.sum(mask):,}")
    else:
        print("  'FEATURES-analysis_busyRoadway' not found in point data")
    
    if 'FEATURES-analysis_forestLane' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['FEATURES-analysis_forestLane'] > 0
        search_urban_elements[mask] = 'existing conversion'
        print(f"  Existing conversion points: {np.sum(mask):,}")
    else:
        print("  'FEATURES-analysis_forestLane' not found in point data")
    
    if 'FEATURES-road_terrainInfo_isParkingMedian3mBuffer' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['FEATURES-road_terrainInfo_isParkingMedian3mBuffer'] == 1
        search_urban_elements[mask] = 'other street potential'
        print(f"  Other street potential points: {np.sum(mask):,}")
    else:
        print("  'FEATURES-road_terrainInfo_isParkingMedian3mBuffer' not found in point data")
    
    if 'FEATURES-road_terrainInfo_isParking' in scenario_vtk.point_data:
        mask = scenario_vtk.point_data['FEATURES-road_terrainInfo_isParking'] == 1
        search_urban_elements[mask] = 'parking'
        print(f"  Parking points: {np.sum(mask):,}")
    else:
        print("  'FEATURES-road_terrainInfo_isParking' not found in point data")

    

    
    scenario_vtk.point_data['search_urban_elements'] = search_urban_elements
    
    return scenario_vtk

def summarize_search_variables(scenario_vtk):
    """Summarize search variables"""
    print("\nSummary of search variables:")
    
    # Count bioavailable points by category
    if 'search_bioavailable' in scenario_vtk.point_data:
        bioavailable = scenario_vtk.point_data['search_bioavailable']
        # Count non-'none' values
        bioavailable_count = np.sum(bioavailable != 'none')
        print(f"  Total bioavailable points: {bioavailable_count:,}")
        
        # Count by category
        categories, counts = np.unique(bioavailable[bioavailable != 'none'], return_counts=True)
        for category, count in zip(categories, counts):
            print(f"    {category}: {count:,}")
    
    # Count design actions
    if 'search_design_action' in scenario_vtk.point_data:
        design_actions = scenario_vtk.point_data['search_design_action']
        # Count non-'none' values
        design_action_count = np.sum(design_actions != 'none')
        print(f"  Total design action points: {design_action_count:,}")
        
        # Count by action
        actions, counts = np.unique(design_actions[design_actions != 'none'], return_counts=True)
        for action, count in zip(actions, counts):
            print(f"    {action}: {count:,}")
    
    # Count urban elements
    if 'search_urban_elements' in scenario_vtk.point_data:
        urban_elements = scenario_vtk.point_data['search_urban_elements']
        # Count non-'none' values
        urban_element_count = np.sum(urban_elements != 'none')
        print(f"  Total urban element points: {urban_element_count:,}")
        
        # Count by element
        elements, counts = np.unique(urban_elements[urban_elements != 'none'], return_counts=True)
        for element, count in zip(elements, counts):
            print(f"    {element}: {count:,}")
    
    return scenario_vtk

def plot_search_variables(scenario_vtk):
    """Plot search variables in a 3-up layout"""
    print("Plotting search variables in a 3-up layout...")
    
    # Create a plotter with 3 subplots
    plotter = pv.Plotter(shape=(1, 3))
    
    # Plot 1: Bioavailable points
    plotter.subplot(0, 0)
    plotter.add_text("Bioavailable", position="upper_edge", font_size=10, color='black')
    
    if 'search_bioavailable' in scenario_vtk.point_data:
        # Create a copy of the scenario_vtk
        bioavailable_vtk = scenario_vtk.copy()
        
        # Get bioavailable data
        bioavailable = scenario_vtk.point_data['search_bioavailable']
        
        # Create a string array for display
        bioavailable_str = np.full(scenario_vtk.n_points, 'No', dtype='<U20')
        
        # Use boolean indexing to set values
        bioavailable_mask = (bioavailable != 'none')
        bioavailable_str[bioavailable_mask] = 'Yes'
        
        # Add the string array to the vtk data
        bioavailable_vtk.point_data['bioavailable_str'] = bioavailable_str
        
        # Add the mesh with direct string scalar coloring
        plotter.add_mesh(bioavailable_vtk, scalars='bioavailable_str', cmap='viridis',
                        scalar_bar_args={'title': 'Bioavailable'})
    else:
        plotter.add_text("No bioavailable data", position="center", font_size=10, color='red')
    
    # Plot 2: Design action
    plotter.subplot(0, 1)
    plotter.add_text("Design Action", position="upper_edge", font_size=10, color='black')
    
    if 'search_design_action' in scenario_vtk.point_data:
        # Create a copy of the scenario_vtk
        design_action_vtk = scenario_vtk.copy()
        
        # Add the mesh with direct string scalar coloring
        plotter.add_mesh(design_action_vtk, scalars='search_design_action', cmap='tab10',
                        scalar_bar_args={'title': 'Design Action'})
    else:
        plotter.add_text("No design action data", position="center", font_size=10, color='red')
    
    # Plot 3: Urban elements
    plotter.subplot(0, 2)
    plotter.add_text("Urban Elements", position="upper_edge", font_size=10, color='black')
    
    if 'search_urban_elements' in scenario_vtk.point_data:
        # Create a copy of the scenario_vtk
        urban_elements_vtk = scenario_vtk.copy()
        
        # Add the mesh with direct string scalar coloring
        plotter.add_mesh(urban_elements_vtk, scalars='search_urban_elements', cmap='tab20',
                        scalar_bar_args={'title': 'Urban Element'})
    else:
        plotter.add_text("No urban elements data", position="center", font_size=10, color='red')
    
    # Link all views so camera movements affect all subplots
    plotter.link_views()
    
    # Show the plot
    #plotter.show(screenshot='search_variables.png')
    plotter.close()
    
    print("Plotting complete. Image saved as search_variables.png")

def plot_urban_elements(scenario_vtk):
    """Plot only the urban elements with eye dome lighting"""
    print("Plotting urban elements with eye dome lighting...")
    
    if 'search_urban_elements' not in scenario_vtk.point_data:
        print("  No urban elements data found in the scenario")
        return
    
    # Create a plotter
    plotter = pv.Plotter()
    
    # Add title
    plotter.add_text("Urban Elements", position="upper_edge", font_size=16, color='black')
    
    # Get urban elements data
    urban_elements = scenario_vtk.point_data['search_urban_elements']
    
    # Create a copy of the scenario_vtk
    urban_elements_vtk = scenario_vtk.copy()
    
    # Add the mesh with direct string scalar coloring
    plotter.add_mesh(urban_elements_vtk, scalars='search_urban_elements', cmap='tab20',
                    scalar_bar_args={'title': 'Urban Element'})
    
    # Enable eye dome lighting for better depth perception
    plotter.enable_eye_dome_lighting()
    
    # Add a text with statistics
    elements, element_counts = np.unique(urban_elements[urban_elements != 'none'], return_counts=True)
    if len(elements) > 0:
        stats_text = "Urban Elements:\n"
        for element, count in zip(elements, element_counts):
            stats_text += f"{element}: {count:,}\n"
        plotter.add_text(stats_text, position=(0.02, 0.02), font_size=10, viewport=True, color='black')
    
    # Show the plot
    #plotter.show(screenshot='urban_elements_detailed.png')
    plotter.close()
    
    print("Urban elements plot complete. Image saved as urban_elements_detailed.png")

def add_search_variables_to_scenario(scenario_vtk):
    """Add search variables to scenario VTK and plot the results"""
    print("Adding search variables to scenario VTK...")
    
    # 1. Create bioavailability layer
    scenario_vtk = create_bioavailablity_layer(scenario_vtk)
    
    # 2. Create design action layer
    scenario_vtk = create_design_action_layer(scenario_vtk)
    
    # 3. Create urban elements layer
    scenario_vtk = create_urban_elements_layer(scenario_vtk)

    # 4. Summarize search variables
    scenario_vtk = summarize_search_variables(scenario_vtk)
    
    return scenario_vtk

def process_scenarios(site, scenario, voxel_size):
    """Process all scenario VTK files"""
    # Load xarray
    ds = load_xarray(site, voxel_size)
    if ds is None:
        return
    
    # Convert to polydata
    print("\nConverting xarray to polydata...")
    site_vtk = a_helper_functions.convert_xarray_into_polydata(ds)
    
    # Preprocess site VTK to create analysis variables
    site_vtk = preprocess_site_vtk(site_vtk)
    
    # Create temp directory for non-VTK outputs
    temp_dir = f'data/revised/final/{site}/temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    years = [0, 10, 30, 60, 180]
    
    for year in years:
        print(f"\nProcessing scenario for year {year}...")
        
        # Load scenario VTK - using the same path as in a_scenario_generatorB.py
        scenario_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk'
        try:
            scenario_vtk = pv.read(scenario_path)
            print(f"Loaded scenario VTK for year {year}")
        except Exception as e:
            print(f"Could not load scenario for year {year}: {e}")
            continue
        
        # Transfer site features to scenario
        updated_scenario = transfer_site_features_to_scenario(site_vtk, scenario_vtk)
        
        # Add search variables
        updated_scenario = add_search_variables_to_scenario(updated_scenario)
        
        # Save updated scenario
        output_path = f'data/revised/final/{site}/{site}_{scenario}_{voxel_size}_scenarioYR{year}_urban_features.vtk'
        updated_scenario.save(output_path)
        print(f"Saved updated scenario to {output_path}")
        
        # Save visualization outputs to temp directory
        # Plot search variables and save to temp directory
        plot_search_variables(updated_scenario)
        if os.path.exists('search_variables.png'):
            os.rename('search_variables.png', f'{temp_dir}/{site}_{scenario}_yr{year}_search_variables.png')
            print(f"Saved search variables plot to {temp_dir}/{site}_{scenario}_yr{year}_search_variables.png")
        
        # Plot urban elements and save to temp directory
        plot_urban_elements(updated_scenario)
        if os.path.exists('urban_elements_detailed.png'):
            os.rename('urban_elements_detailed.png', f'{temp_dir}/{site}_{scenario}_yr{year}_urban_elements_detailed.png')
            print(f"Saved urban elements plot to {temp_dir}/{site}_{scenario}_yr{year}_urban_elements_detailed.png")

def process_baseline(site, voxel_size):
    """Process baseline VTK file to add search variables"""
    print(f"\nProcessing baseline for site {site}...")
    
    # Load baseline VTK file
    baseline_path = f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}.vtk'
    baseline_vtk = pv.read(baseline_path)

    
    # Initialize search variables
    print("Adding search variables to baseline...")
    
    # 1. Initialize search_bioavailable based on forest_size
    search_bioavailable = np.full(baseline_vtk.n_points, 'none', dtype='<U20')
    
    # Set to 'arboreal' where forest_size is not 'nan'
    if 'forest_size' in baseline_vtk.point_data:
        forest_size = baseline_vtk.point_data['forest_size']
        
        if forest_size.dtype.kind == 'U' or forest_size.dtype.kind == 'S':  # String types
            arboreal_mask = (forest_size != 'nan') & (forest_size != 'none')
            search_bioavailable[arboreal_mask] = 'arboreal'
            print(f"  Set {np.sum(arboreal_mask):,} points to 'arboreal' based on forest_size")
        else:  # Numeric types
            arboreal_mask = ~np.isnan(forest_size)
            search_bioavailable[arboreal_mask] = 'arboreal'
            print(f"  Set {np.sum(arboreal_mask):,} points to 'arboreal' based on forest_size")
    
    # Set remaining points to 'low-vegetation' (renamed from 'traversable')
    low_veg_mask = search_bioavailable == 'none'
    search_bioavailable[low_veg_mask] = 'low-vegetation'
    print(f"  Set {np.sum(low_veg_mask):,} remaining points to 'low-vegetation'")
    
    baseline_vtk.point_data['search_bioavailable'] = search_bioavailable
    print(f"  Added search_bioavailable with {np.sum(search_bioavailable == 'arboreal'):,} 'arboreal' and {np.sum(search_bioavailable == 'low-vegetation'):,} 'low-vegetation' points")
    
    # 2. Initialize search_design_action as 'rewilded'
    search_design_action = np.full(baseline_vtk.n_points, 'rewilded', dtype='<U20')
    baseline_vtk.point_data['search_design_action'] = search_design_action
    print(f"  Added search_design_action with all points set to 'rewilded'")
    
    # 3. Initialize search_urban_elements as 'none'
    search_urban_elements = np.full(baseline_vtk.n_points, 'none', dtype='<U20')
    baseline_vtk.point_data['search_urban_elements'] = search_urban_elements
    print(f"  Added search_urban_elements with all points set to 'none'")
    
    # 4. Initialize forest_control with 'reserve-tree' for all points
    forest_control = np.full(baseline_vtk.n_points, 'reserve-tree', dtype='<U20')
    baseline_vtk.point_data['forest_control'] = forest_control
    print(f"  Added forest_control with all points set to 'reserve-tree'")
    
    # Save updated baseline
    updated_baseline_path = f'data/revised/final/{site}/{site}_baseline_resources_{voxel_size}_urban_features.vtk'
    baseline_vtk.save(updated_baseline_path)
    print(f"Saved updated baseline to {updated_baseline_path}")
    
    return baseline_vtk

def main():
    """Main function to process sites and scenarios"""
    # Process baseline if requested
    if PROCESS_BASELINE:
        for site in SITES:
            process_baseline(site, VOXEL_SIZE)
    
    # Process all sites and scenarios
    """for site in SITES:
        print(f"\n=== Processing site: {site} ===")
        
        for scenario in SCENARIOS:
            print(f"\n--- Processing scenario: {scenario} ---")
            
            # Process scenarios
            process_scenarios(site, scenario, VOXEL_SIZE)
    """
if __name__ == "__main__":
    main()