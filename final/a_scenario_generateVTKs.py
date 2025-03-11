import os
import pandas as pd
import xarray as xr
import numpy as np
import pyvista as pv
import a_helper_functions
import a_voxeliser

def update_rewilded_voxel_catagories(ds, df):
    """
    Updates the 'scenario_rewilded' variable in the xarray dataset based on the dataframe values.
    Matches are made based on NodeID. Non-matching NodeIDs are ignored.
    
    Parameters:
    ds (xarray.Dataset): The xarray dataset containing voxel and node information.
    df (pandas.DataFrame): The dataframe containing NodeID and rewilded scenarios.
    
    Returns:
    xarray.Dataset: The updated dataset with the 'scenario_rewilded' variable modified.
    """
    # Replace 'None' with 'none' in the DataFrame
    df['rewilded'] = df['rewilded'].replace('None', 'none')
    
    # Step 1: Initialize 'scenario_rewilded' if it doesn't exist
    if 'scenario_rewilded' not in ds.variables:
        # Use object dtype for variable-length strings
        ds = ds.assign(scenario_rewilded=('voxel', np.array(['none'] * ds.dims['voxel'], dtype='O')))
    
    # Step 2: Extract relevant arrays from xarray
    canopy_id = ds['node_CanopyID'].values
    sim_nodes = ds['sim_Nodes'].values
    
    # Step 3: Iterate over dataframe rows and update scenario_rewilded based on NodeID
    for idx, row in df.iterrows():
        if row['rewilded'] in ['exoskeleton', 'footprint-depaved']:
            # Match using 'node_CanopyID'
            mask = (canopy_id == row['NodeID'])
        elif row['rewilded'] == 'node-rewilded':
            # Match using 'sim_Nodes'
            mask = (sim_nodes == row['NodeID'])
        else:
            continue
        
        # Update 'scenario_rewilded' for matching voxels
        ds['scenario_rewilded'].values[mask] = row['rewilded']

    # Print all unique values and counts for df['rewilded'] using pandas
    print('Column rewilded values and counts in dataframe:')
    print(df['rewilded'].value_counts())
    
    # Print all unique variable values and counts for scenario_rewilded
    unique_values, counts = np.unique(ds['scenario_rewilded'], return_counts=True)
    print('Column scenario_rewilded values and counts in xarray dataset:')
    for value, count in zip(unique_values, counts):
        print(f'scenario_rewilded value: {value}, count: {count}')
    
    # Step 4: Return the updated dataset
    return ds

def update_bioEnvelope_voxel_catagories(ds, params):
    # Create another version of the depaved column that also considers the green envelopes
    year = params['years_passed']
    turnThreshold = params['sim_TurnsThreshold'][year]
    resistanceThreshold = params['sim_averageResistance'][year] if 'sim_averageResistance' in params else 0

    bioMask = (ds['sim_Turns'] <= turnThreshold) & (ds['sim_averageResistance'] <= resistanceThreshold) & (ds['sim_Turns'] >= 0)
    ds['bioMask'] = bioMask

    # Initialize scenario_bioEnvelope as a copy of ds['scenario_rewilded']
    ds['scenario_bioEnvelope'] = xr.DataArray(
        data=np.array(ds['scenario_rewilded'].values, dtype='O'),
        dims='voxel'
    )

    # Assign 'otherGround' to scenario_bioEnvelope where bioMask is true and scenario_bioEnvelope is 'none'
    ds['scenario_bioEnvelope'].loc[bioMask & (ds['scenario_bioEnvelope'] == 'none')] = 'otherGround'

    # Assign 'livingFacade' to scenario_bioEnvelope where site_building_element == 'facade' and bioMask is True
    ds['scenario_bioEnvelope'].loc[(ds['site_building_element'] == 'facade') & bioMask] = 'livingFacade'

    # Assign 'greenRoof' to scenario_bioEnvelope where envelope_roofType == 'green roof' and bioMask is True
    ds['scenario_bioEnvelope'].loc[(ds['envelope_roofType'] == 'green roof') & bioMask] = 'greenRoof'

    # Assign 'brownRoof' to scenario_bioEnvelope where envelope_roofType == 'brown roof' and bioMask is True
    ds['scenario_bioEnvelope'].loc[(ds['envelope_roofType'] == 'brown roof') & bioMask] = 'brownRoof'

    """poly = a_helper_functions.convert_xarray_into_polydata(ds)
    poly.plot(scalars='scenario_bioEnvelope', cmap='Set1')"""

    unique_values, counts = np.unique(ds['scenario_bioEnvelope'], return_counts=True)
    print('column scenario_bioEnvelope values and counts in xarray dataset:')
    for value, count in zip(unique_values, counts):
        print(f'scenario_bioEnvelope value: {value}, count: {count}')

    return ds

def finalDSprocessing(ds):
    # Create updated resource variables in xarray

    # Elevated dead branches
    ds['updatedResource_elevatedDeadBranches'] = ds['resource_dead branch'].copy()
    
    # Get mask for 'forest_size' == 'senescing'
    mask_senescing = ds['forest_size'] == 'senescing'
    
    # Update 'updatedResource_elevatedDeadBranches' for senescing trees
    ds['updatedResource_elevatedDeadBranches'].loc[mask_senescing] = ds['resource_dead branch'].loc[mask_senescing] + ds['resource_other'].loc[mask_senescing]

    # Ground dead branches
    ds['updatedResource_groundDeadBranches'] = ds['resource_fallen log'].copy()
    
    # Get mask for 'forest_size' == 'fallen'
    mask_fallen = ds['forest_size'] == 'fallen'
    
    # Update 'updatedResource_groundDeadBranches' for fallen trees
    ds['updatedResource_groundDeadBranches'].loc[mask_fallen] = (
        ds['resource_dead branch'].loc[mask_fallen] + 
        ds['resource_other'].loc[mask_fallen] + 
        ds['resource_peeling bark'].loc[mask_fallen] + 
        ds['resource_fallen log'].loc[mask_fallen] + 
        ds['resource_perch branch'].loc[mask_fallen]
    )

    return ds

def print_simulation_statistics(df, year, site):
    print(f"Simulation Summary for Year: {year}, Site: {site}")
    
    # Print total number of trees
    total_trees = len(df)
    print(f"Total number of trees: {total_trees}")
    
    # Print unique values and their counts for the 'size' column
    print("\nUnique values and their counts for 'size':")
    print(df['size'].value_counts())

    # Print unique values and their counts for the 'action' column
    print("\nUnique values and their counts for 'action':")
    print(df['action'].value_counts())

    # Print unique values and their counts for the 'rewilded' column
    print("\nUnique values and their counts for 'rewilded':")
    print(df['rewilded'].value_counts())

    print(f"Trees planted: {df[df['isNewTree'] == True].shape[0]}")
    
    print("\nEnd of simulation statistics.\n")

def process_polydata(polydata):
    # Create a mask that checks all polydata.point_data variables starting with resource,
    # except for resource_leaf_litter. If any of the resource variables are >0, mask is True, else False
    maskforTrees = np.zeros(polydata.n_points, dtype=bool)

    # Loop through point_data keys and update the mask
    for key in polydata.point_data.keys():
        if key.startswith('resource_') and key != 'resource_leaf litter':
            # Get the data as a NumPy array
            resource_data = polydata.point_data[key]
            
            # Create a boolean mask where the values are greater than 0
            resource_mask = resource_data > 0
            
            # Combine the mask with the current mask using logical OR
            maskforTrees = np.logical_or(maskforTrees, resource_mask)

    # Add the mask to point data
    polydata.point_data['maskforTrees'] = maskforTrees
    maskForRewilding = polydata['scenario_rewilded'] != 'none'
    polydata.point_data['maskForRewilding'] = maskForRewilding

    # Initialize with object dtype
    scenario_outputs = np.full(polydata.n_points, 'none', dtype='O')
    scenario_outputs[maskForRewilding] = polydata.point_data['scenario_rewilded'][maskForRewilding]
    scenario_outputs[maskforTrees] = polydata.point_data['forest_size'][maskforTrees]
    
    # Print unique values and counts for scenario_outputs
    print(f'unique values and counts for scenario_outputs: {pd.Series(scenario_outputs).value_counts()}')
    polydata.point_data['scenario_outputs'] = scenario_outputs
    
    print(f'unique values and counts for scenario_outputs in polydata: {pd.Series(polydata.point_data["scenario_outputs"]).value_counts()}')
    return polydata

def plot_scenario_rewilded(polydata, treeDF, years_passed, site):
    maskforTrees = polydata.point_data['maskforTrees']
    maskForRewilding = polydata.point_data['maskForRewilding']

    sitePoly = polydata.extract_points(~maskforTrees)  # Points where forest_tree_id is NaN
    treePoly = polydata.extract_points(maskforTrees)  # Points where forest_tree_id is not NaN

    # Extract two different polydata based on the masks
    rewildingVoxels = sitePoly.extract_points(maskForRewilding)
    siteVoxels = sitePoly.extract_points(~maskForRewilding)

    # Print all point_data variables in polydata
    print(f'point_data variables in polydata: {sitePoly.point_data.keys()}')

    # Create the plotter
    plotter = pv.Plotter()
    # Add title to the plotter
    plotter.add_text(f"Scenario at {site} after {years_passed} years", position="upper_edge", font_size=16, color='black')

    # Label trees
    label_trees(treeDF, plotter)

    # Add 'none' points as white
    plotter.add_mesh(siteVoxels, color='white')
    plotter.add_mesh(treePoly, scalars='forest_size', cmap='Set1')

    # Add other points with scalar-based coloring
    plotter.add_mesh(rewildingVoxels, scalars='scenario_rewilded', cmap='Set2', show_scalar_bar=True)
    plotter.enable_eye_dome_lighting()

    plotter.show()
    plotter.close()

def label_trees(df, plotter):
    # Prepare points and labels from the filtered subset
    TARGET_SIZES_FOR_LABELS = ['large', 'senescing', 'snag', 'fallen']
    label_df = df[df['size'].isin(TARGET_SIZES_FOR_LABELS)]
    label_points = label_df[['x', 'y', 'z']].values
    label_points[:,2] = label_points[:,2] + 10  # Give some elevation to the labels so they are easier to see
    labels = label_df['size'].astype(str).tolist()
    
    # Add the labels to the plotter
    plotter.add_point_scalar_labels(
        points=label_points,
        labels=labels,
        fmt='%s',              # Since 'size' is categorical, no formatting is needed
        preamble='Size: ',
        font_size=20,          # Adjust font size as needed
        text_color='black',    # Choose a contrasting color for visibility
        shadow=True,           # Add shadow for better readability
        render_points_as_spheres=False,  # Optional: Customize label rendering
        point_size=10          # Size of the points associated with labels if rendered
    )

def get_scenario_parameters():
    """Returns a dictionary of parameters for each site and scenario."""
    paramsPARADE_positive = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 70,
    'plantThreshold' : 50,
    'rewildThreshold' : 10,
    'senescingThreshold' : -25,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 0,
                            30: 3000,
                            60: 4000,
                            180: 4500},
    }

    paramsPARADE_trending = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 2, # Highest threshold
    'plantThreshold' : 1, # Middle threshold
    'rewildThreshold' : 0, # Lowest threshold
    'senescingThreshold' : -5, 
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 0,
                            30: 0,
                            60: 0,
                            180: 0},
    }

    paramsCITY_positive = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 70,
    'plantThreshold' : 50,
    'rewildThreshold' : 10,
    'senescingThreshold' : -25,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 300,
                            30: 1249.75,
                            60: 4999,
                            180: 5000},
    'sim_averageResistance' : {0: 0,
                            10: 50,
                            30: 50,
                            60: 67.90487670898438,
                            180: 96},
    }

    paramsCITY_trending = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 15,
    'plantThreshold' : 10,
    'rewildThreshold' : 5,
    'senescingThreshold' : -5,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 20,
                            30: 50,
                            60: 100,
                            180: 200},
    'sim_averageResistance' : {0: 0,
                            10: 10,
                            30: 20,
                            60: 30,
                            180: 50},
    }

    paramsUNI_positive = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 70,
    'plantThreshold' : 50,
    'rewildThreshold' : 10,
    'senescingThreshold' : -25,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 300,
                            30: 1249.75,
                            60: 4999,
                            180: 5000},
    'sim_averageResistance' : {0: 0,
                            10: 50,
                            30: 50,
                            60: 67.90487670898438,
                            180: 0},
    }

    paramsUNI_trending = {
    'growth_factor_range' : [0.37, 0.51], # Growth factor is a range
    'plantingDensity' : 50, # 10 per hectare
    'ageInPlaceThreshold' : 15,
    'plantThreshold' : 10,
    'rewildThreshold' : 5,
    'senescingThreshold' : -5,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'sim_TurnsThreshold' : {0: 0,
                            10: 20,
                            30: 50,
                            60: 100,
                            180: 200},
    'sim_averageResistance' : {0: 0,
                            10: 0,
                            30: 0,
                            60: 0,
                            180: 0},
    }

    paramsDic = {
        ('trimmed-parade', 'positive'): paramsPARADE_positive,
        ('trimmed-parade', 'trending'): paramsPARADE_trending,
        ('city', 'positive'): paramsCITY_positive,
        ('city', 'trending'): paramsCITY_trending,
        ('uni', 'positive'): paramsUNI_positive,
        ('uni', 'trending'): paramsUNI_trending,
    }
    
    return paramsDic

def generate_vtk(site, scenario, year, voxel_size, ds, treeDF, logDF=None, poleDF=None, enable_visualization=False):
    """
    Generates a VTK file for a given site, scenario, and year.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    year (int): Year to simulate
    voxel_size (int): Voxel size
    ds (xarray.Dataset): Xarray dataset
    treeDF (pd.DataFrame): Tree dataframe
    logDF (pd.DataFrame, optional): Log dataframe
    poleDF (pd.DataFrame, optional): Pole dataframe
    enable_visualization (bool, optional): Whether to show visualization
    
    Returns:
    xarray.Dataset: Updated dataset
    """
    # Get scenario parameters
    paramsDic = get_scenario_parameters()
    params = paramsDic[(site, scenario)]
    params['years_passed'] = year
    
    # Output file path
    output_path = f'data/revised/final/{site}'
    os.makedirs(output_path, exist_ok=True)
    
    print(f'Generating VTK for {site}, {scenario}, year {year}')
    
    # Update the xarray dataset
    print('Integrating results into xarray')
    ds = update_rewilded_voxel_catagories(ds, treeDF)
    
    print('Updating bioEnvelope voxel categories')
    if logDF is not None or poleDF is not None:
        ds = update_bioEnvelope_voxel_catagories(ds, params)

    unique_values, counts = np.unique(ds['scenario_rewilded'], return_counts=True)
    print(f'Unique values and counts for scenario_rewilded: {unique_values}, {counts}')

    # Prepare valid points for resource voxelization
    validpointsMask = ds['scenario_rewilded'].values != 'none'
    if logDF is not None:
        validpointsMask = ds['scenario_bioEnvelope'].values != 'none'

    if poleDF is not None:
        validpointsMask = ds['scenario_bioEnvelope'].values != 'none'

    # Extract valid points as a numpy array
    valid_points = np.array([
        ds['centroid_x'].values[validpointsMask],
        ds['centroid_y'].values[validpointsMask],
        ds['centroid_z'].values[validpointsMask]
    ]).T

    # Integrate resources into xarray
    
    templateResolution = 0.5
    ds, combinedDF_scenario = a_voxeliser.integrate_resources_into_xarray(ds, treeDF, templateResolution, logDF, poleDF, valid_points)
    
    # Final processing
    ds = finalDSprocessing(ds)

    # Save combinedDF_scenario to csv
    print(f'Saving {year} combinedDF_scenario to csv')
    combinedDF_scenario.to_csv(f'{output_path}/{site}_{scenario}_{voxel_size}_nodeDF_{year}.csv', index=False)
    
    # Print statistics
    print_simulation_statistics(combinedDF_scenario, year, site)

    # Convert to polydata and process
    polydata = a_helper_functions.convert_xarray_into_polydata(ds)
    polydata = process_polydata(polydata)
    
    # Save polydata
    vtk_file = f'{output_path}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk'
    polydata.save(vtk_file)
    print(f'Saved VTK file to {vtk_file}')

    # Optional visualization
    if enable_visualization:
        plot_scenario_rewilded(polydata, treeDF, year, site)
    
    return ds

def load_scenario_dataframes(site, scenario, year, voxel_size):
    """
    Loads scenario dataframes for a given site, scenario, and year.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    year (int): Year to load
    voxel_size (int): Voxel size
    
    Returns:
    tuple: (treeDF, logDF, poleDF) - Loaded dataframes
    """
    filepath = f'data/revised/final/{site}'
    
    # Load tree dataframe
    tree_file = f'{filepath}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv'
    if os.path.exists(tree_file):
        treeDF = pd.read_csv(tree_file)
        print(f'Loaded tree dataframe from {tree_file}')
    else:
        print(f'Tree dataframe file not found: {tree_file}')
        return None, None, None
    
    # Load log dataframe
    log_file = f'{filepath}/{site}_{scenario}_{voxel_size}_logDF_{year}.csv'
    if os.path.exists(log_file):
        logDF = pd.read_csv(log_file)
        print(f'Loaded log dataframe from {log_file}')
    else:
        logDF = None
        print(f'Log dataframe file not found: {log_file}')
    
    # Load pole dataframe
    pole_file = f'{filepath}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv'
    if os.path.exists(pole_file):
        poleDF = pd.read_csv(pole_file)
        print(f'Loaded pole dataframe from {pole_file}')
    else:
        poleDF = None
        print(f'Pole dataframe file not found: {pole_file}')
    
    return treeDF, logDF, poleDF

if __name__ == "__main__":
    import a_scenario_initialiseDS
    
    # Define default sites and scenarios
    default_sites = ['trimmed-parade', 'city', 'uni']
    default_scenarios = ['positive', 'trending']
    default_voxel_size = 1
    
    # Get user input for site
    sites_input = input(f"Enter site(s) to process (comma-separated) or press Enter for default {default_sites}: ")
    sites = sites_input.split(',') if sites_input else default_sites
    sites = [site.strip() for site in sites]
    
    # Get user input for scenarios
    scenarios_input = input(f"Enter scenario(s) to process (comma-separated) or press Enter for default {default_scenarios}: ")
    scenarios = scenarios_input.split(',') if scenarios_input else default_scenarios
    scenarios = [scenario.strip() for scenario in scenarios]
    
    # Get user input for voxel size
    voxel_size_input = input(f"Enter voxel size (default {default_voxel_size}): ")
    voxel_size = int(voxel_size_input) if voxel_size_input else default_voxel_size
    
    # Get user input for years
    all_years = [0, 10, 30, 60, 180]
    years_input = input(f"Enter years to process (comma-separated) or press Enter for default {all_years}: ")
    years = [int(year) for year in years_input.split(',')] if years_input else all_years
    
    # Get user input for visualization
    vis_input = input("Enable visualization? (yes/no, default no): ")
    enable_visualization = vis_input.lower() in ['yes', 'y', 'true', '1']
    
    # Generate VTKs for each site, scenario, and year
    for site in sites:
        print(f"\n===== Processing {site} =====\n")
        # Initialize dataset
        subsetDS = a_scenario_initialiseDS.initialize_dataset(site, voxel_size)
        
        for scenario in scenarios:
            print(f"\n--- Processing {scenario} scenario ---\n")
            
            for year in years:
                print(f"\n- Processing year {year} -\n")
                
                # Load scenario dataframes
                treeDF, logDF, poleDF = load_scenario_dataframes(site, scenario, year, voxel_size)
                
                if treeDF is not None:
                    # Generate VTK
                    generate_vtk(site, scenario, year, voxel_size, subsetDS, treeDF, logDF, poleDF, enable_visualization)
                else:
                    print(f"Skipping VTK generation for {site}, {scenario}, year {year} - dataframes not found")
                
    print("\nAll VTK generation completed.")
