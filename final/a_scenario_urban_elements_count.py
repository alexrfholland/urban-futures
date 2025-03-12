import numpy as np
import pyvista as pv
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
import a_helper_functions
from scipy.spatial import cKDTree

# Configuration variables that will be overridden by the manager
SITES = ['trimmed-parade']
SCENARIOS = ['positive', 'trending']
VOXEL_SIZE = 1
PROCESS_BASELINE = True
YEARS = [0, 10, 30, 60, 180]
SPECIFIC_FILES = []  # Will be populated by the manager

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
    
    # Set 'low-vegetation' for all low-vegetation areas (including those marked as open space)
    search_bioavailable[low_vegetation_mask] = 'low-vegetation'
    print(f"  Low-vegetation points: {np.sum(low_vegetation_mask):,}")
    
    # Set 'arboreal' for bioavailable areas (excluding open spaces and low-vegetation)
    arboreal_mask = bioavailable_mask & (search_bioavailable == 'none')
    search_bioavailable[arboreal_mask] = 'arboreal'
    print(f"  Arboreal points: {np.sum(arboreal_mask):,}")
    
    # Add to scenario
    scenario_vtk.point_data['search_bioavailable'] = search_bioavailable
    
    print(f"  Total bioavailable habitat points: {np.sum(search_bioavailable != 'none'):,}")
    
    return scenario_vtk

def create_design_action_layer(scenario_vtk):
    """Create the search_design_action layer"""
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

def process_baseline(site, voxel_size):
    """Process baseline VTK file to add search variables"""
    print(f"\nProcessing baseline for site {site}...")
    
    # First check in the standardized baselines folder
    baseline_path = f'data/revised/final/baselines/{site}_baseline_combined_{voxel_size}.vtk'
        
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline file not found for {site} in either location")
        return None
    
    print(f"Loading baseline file from: {baseline_path}")
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
            # Use np.isnan directly instead of treating it as a function
            if np.issubdtype(forest_size.dtype, np.floating):
                arboreal_mask = ~np.isnan(forest_size)
            else:
                arboreal_mask = np.ones(len(forest_size), dtype=bool)
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
    
    # Save updated baseline - save to the same folder as the input file
    output_folder = os.path.dirname(baseline_path)
    filename = os.path.basename(baseline_path)
    updated_baseline_path = os.path.join(output_folder, filename.replace('.vtk', '_urban_features.vtk'))
    baseline_vtk.save(updated_baseline_path)
    print(f"Saved updated baseline to {updated_baseline_path}")
    
    return baseline_vtk

def visualize_search_variables(scenario_vtk, site, scenario, year, voxel_size, output_dir='data/revised/final/visualizations'):
    """
    Visualize the three search variables (bioavailable, design action, urban elements) side by side.
    
    Parameters:
    scenario_vtk (pyvista.PolyData): The scenario VTK with search variables
    site (str): Site name
    scenario (str): Scenario type
    year (int): Year/timestep
    voxel_size (int): Voxel size
    output_dir (str): Output directory for images
    """
    print(f"\nVisualizing search variables for {site}, {scenario}, year {year}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create a plotter with 1 row and 3 columns
        plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[2400, 800])
        
        # Plot 1: Bioavailable habitat
        plotter.subplot(0, 0)
        plotter.add_title(f"Bioavailable Habitat", font_size=12)
        
        # Filter out 'none' values
        bioavailable = scenario_vtk.point_data['search_bioavailable']
        bio_mask = bioavailable != 'none'
        if np.any(bio_mask):
            # Use boolean array directly with extract_points
            bio_poly = scenario_vtk.extract_points(bio_mask)
            plotter.add_points(bio_poly, scalars='search_bioavailable', cmap='viridis', 
                              point_size=5, render_points_as_spheres=True)
            plotter.add_scalar_bar(title='Habitat Type', n_labels=3, position_x=0.05, position_y=0.05, width=0.25)
        else:
            print("No bioavailable habitat points to visualize")
        
        # Plot 2: Design Actions
        plotter.subplot(0, 1)
        plotter.add_title(f"Design Actions", font_size=12)
        
        # Filter out 'none' values
        design_action = scenario_vtk.point_data['search_design_action']
        design_mask = design_action != 'none'
        if np.any(design_mask):
            design_poly = scenario_vtk.extract_points(design_mask)
            plotter.add_points(design_poly, scalars='search_design_action', cmap='tab10', 
                              point_size=5, render_points_as_spheres=True)
            plotter.add_scalar_bar(title='Design Action', n_labels=5, position_x=0.05, position_y=0.05, width=0.25)
        else:
            print("No design action points to visualize")
        
        # Plot 3: Urban Elements
        plotter.subplot(0, 2)
        plotter.add_title(f"Urban Elements", font_size=12)
        
        # Filter out 'none' values
        urban_elements = scenario_vtk.point_data['search_urban_elements']
        urban_mask = urban_elements != 'none'
        if np.any(urban_mask):
            urban_poly = scenario_vtk.extract_points(urban_mask)
            plotter.add_points(urban_poly, scalars='search_urban_elements', cmap='tab20', 
                              point_size=5, render_points_as_spheres=True)
            plotter.add_scalar_bar(title='Urban Element', n_labels=8, position_x=0.05, position_y=0.05, width=0.25)
        else:
            print("No urban element points to visualize")
        
        # Add overall title
        plotter.add_text(f"{site} - {scenario} - Year {year}", position='upper_edge', 
                        font_size=16, color='black')
        
        # Link all camera positions
        plotter.link_views()
        
        # Save the image
        output_file = f"{output_dir}/{site}_{scenario}_{voxel_size}_yr{year}_search_variables.png"
        plotter.screenshot(output_file)
        print(f"Saved visualization to {output_file}")
        
        # Close the plotter to free resources
        plotter.close()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

def process_scenario_vtks(site, scenario, years, voxel_size):
    """
    Process all scenario VTK files for a given site, scenario, and years.
    Uses the exact same filepath pattern as in a_scenario_generateVTKs.py.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    years (list): List of years to process
    voxel_size (int): Voxel size
    
    Returns:
    list: List of processed VTK files
    """
    processed_files = []
    
    # Base directory for VTK files
    base_dir = f'data/revised/final/{site}'
    
    print(f"\n===== Processing Urban Elements for {site} - {scenario} =====")
    
    # Process each year
    for year in years:
        # Construct the exact filepath as used in a_scenario_generateVTKs.py
        vtk_file = f'{base_dir}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk'
        
        print(f"\nLooking for VTK file: {vtk_file}")
        
        if os.path.exists(vtk_file):
            print(f"Found VTK file for {site} - {scenario} - Year {year}")
            
            # Process the VTK file
            result = process_single_file(
                vtk_file_path=vtk_file,
                site=site,
                voxel_size=voxel_size,
                scenario=scenario,
                year=year
            )
            
            if result:
                processed_files.append(vtk_file)
                print(f"Successfully processed: {os.path.basename(vtk_file)}")
            else:
                print(f"Failed to process: {os.path.basename(vtk_file)}")
        else:
            print(f"VTK file not found: {vtk_file}")
    
    print(f"\nProcessed {len(processed_files)} of {len(years)} VTK files for {site} - {scenario}")
    return processed_files

def run_from_manager(site=None, scenario=None, years=None, voxel_size=None, specific_files=None, should_process_baseline=False):
    """
    Run the urban elements processing from the manager script.
    
    Parameters:
    site (str): Site name
    scenario (str): Scenario type
    years (list): List of years to process
    voxel_size (int): Voxel size
    specific_files (list): List of specific VTK files to process
    process_baseline (bool): Whether to process baseline
    
    Returns:
    list: List of processed VTK files
    """
    processed_files = []
    
    # 1. Process baseline if requested
    if should_process_baseline:
        print(f"\nProcessing baseline for site: {site}")
        baseline_result = process_baseline(site, voxel_size)
        if baseline_result:
            print(f"Successfully processed baseline for {site}")
    
    # 2. Process specific files if provided, otherwise construct paths from site/scenario/years
    vtk_files_to_process = []
    
    if specific_files:
        vtk_files_to_process = specific_files
        print(f"\nProcessing {len(specific_files)} specific files")
    else:
        # Construct VTK filepaths using the pattern from a_scenario_generateVTKs.py
        base_dir = f'data/revised/final/{site}'
        for year in years:
            vtk_file = f'{base_dir}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk'
            vtk_files_to_process.append(vtk_file)
        print(f"\nProcessing {len(vtk_files_to_process)} VTK files for {site} - {scenario}")
    
    # Process each VTK file
    for vtk_file in vtk_files_to_process:
        if os.path.exists(vtk_file):
            print(f"Processing file: {os.path.basename(vtk_file)}")
            
            # Extract year from filename if possible
            year = None
            if 'YR' in vtk_file:
                try:
                    year_part = vtk_file.split('YR')[1].split('.')[0]
                    year = int(year_part)
                except (IndexError, ValueError):
                    year = None
            
            # Process the file
            result = process_single_file(
                vtk_file_path=vtk_file,
                site=site,
                voxel_size=voxel_size,
                scenario=scenario,
                year=year
            )
            
            if result:
                processed_files.append(vtk_file)
                print(f"Successfully processed: {os.path.basename(vtk_file)}")
        else:
            print(f"File not found: {vtk_file}")
    
    print(f"\nSuccessfully processed {len(processed_files)} files")
    return processed_files

def process_single_file(vtk_file_path, site=None, voxel_size=None, scenario=None, year=None):
    """Process a single VTK file specified by path"""
    print(f"Processing file: {os.path.basename(vtk_file_path)}")
    
    try:
        # Load the VTK file
        scenario_vtk = pv.read(vtk_file_path)
        print(f"Loaded VTK with {scenario_vtk.n_points:,} points")
        
        # If this is a baseline file, process it differently
        if 'baseline' in vtk_file_path.lower():
            if site is None:
                # Try to extract site from filename
                filename = os.path.basename(vtk_file_path)
                parts = filename.split('_')
                if len(parts) >= 1:
                    site = parts[0]
            
            if site:
                print(f"Processing as baseline for site: {site}")
                updated_vtk = process_baseline(site, voxel_size)
                
                # Generate visualization for baseline
                if updated_vtk and voxel_size:
                    visualize_search_variables(updated_vtk, site, "baseline", 0, voxel_size)
                
                return updated_vtk
            else:
                print("Cannot process baseline without site information")
                return None
        
        # For regular scenario files
        # Load xarray for site context
        ds = None
        if site and voxel_size:
            ds = load_xarray(site, voxel_size)
            if ds is None:
                print(f"Warning: Could not load xarray for site {site}")
        
        # Process the scenario VTK
        if ds is not None:
            # Convert xarray to polydata for site context
            site_vtk = a_helper_functions.convert_xarray_into_polydata(ds)
            site_vtk = preprocess_site_vtk(site_vtk)
            
            # Transfer site features to scenario
            scenario_vtk = transfer_site_features_to_scenario(site_vtk, scenario_vtk)
            
            # Add search variables
            updated_vtk = add_search_variables_to_scenario(scenario_vtk)
            print(f"Processed with site context from xarray")
        else:
            # Process without site context
            updated_vtk = add_search_variables_to_scenario(scenario_vtk)
            print(f"Processed without site context (no xarray available)")
        
        # Save updated VTK with "_urban_features" suffix
        output_path = vtk_file_path.replace('.vtk', '_urban_features.vtk')
        updated_vtk.save(output_path)
        print(f"Saved to: {os.path.basename(output_path)}")
        
        # Generate visualization
        if site and scenario and year is not None and voxel_size:
            visualize_search_variables(updated_vtk, site, scenario, year, voxel_size)
            print(f"Generated visualization for {site} - {scenario} - Year {year}")
        
        return updated_vtk
    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the main function if script is executed directly
    run_from_manager()