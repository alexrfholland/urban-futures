import os
import pandas as pd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
import a_helper_functions
import a_voxeliser
import a_scenario_initialiseDS
import copy

def remap_values_xarray(values, old_min, old_max, new_min, new_max):
    return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

# Function to age trees based on passed years
def age_trees(df, params):
    years_passed = params['years_passed']
    # Only update DBH for trees sized 'small', 'medium', or 'large'
    mask = df['size'].isin(['small', 'medium', 'large'])
    growth_factor_per_year = (params['growth_factor_range'][0] + params['growth_factor_range'][1])/2 #get mean
    growth = growth_factor_per_year * years_passed
    df.loc[mask, 'diameter_breast_height'] = df.loc[mask, 'diameter_breast_height'] + growth

    print('df is now:')
    print(df.columns)

    print(f'useful_life_expectancy range: {df["useful_life_expectancy"].min()} to {df["useful_life_expectancy"].max()}')

    # Update size only for trees that are already sized 'small' or 'medium'
    mask = df['size'].isin(['small', 'medium'])

    # Update tree size classification based on DBH, but only for small/medium trees
    df.loc[mask, 'size'] = pd.cut(df.loc[mask, 'diameter_breast_height'], 
                                  bins=[-10, 30, 80, float('inf')], 
                                  labels=['small', 'medium', 'large']).astype(str)

    # Decrease useful_life_expectancy based on years passed
    df['useful_life_expectancy'] -= years_passed

    # Print useful_life_expectancy range after aging
    print(f'after aging {years_passed} years:')
    print(f'useful_life_expectancy range: {df["useful_life_expectancy"].min()} to {df["useful_life_expectancy"].max()}')
    print(f'breakdown of useful_life_expectancy at year {years_passed}:')

    print(df["useful_life_expectancy"])
    print(df["useful_life_expectancy"].value_counts())

    return df

# Function to senesce trees based on resistance score and life expectancy
def determine_ageinplace_or_replace(df, params, seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Set the threshold for senescence
    senesce_threshold = params['senescingThreshold']

    # Remap the `useful_life_expectancy` to calculate `senesceChance`
    df['senesceChance'] = remap_values_xarray(df['useful_life_expectancy'], old_min=senesce_threshold, old_max=0, new_min=100, new_max=0).clip(0, 100)
    
    # Assign a random roll between 0 and 100 to each row
    df['senesceRoll'] = np.random.uniform(0, 100, len(df))

    # Apply the senescing condition if senesceRoll is below senesceChance
    senesce_mask = df['size'].isin(['small', 'medium', 'large']) & (df['senesceRoll'] < df['senesceChance'])
    df.loc[senesce_mask, 'action'] = 'SENESCENT'

    # Apply probabilistic decision between AGE-IN-PLACE and REPLACE based on `CanopyResistance`
    age_in_place_mask = (df['action'] == 'SENESCENT') & (df['CanopyResistance'] < params['ageInPlaceThreshold'])
    replace_mask = (df['action'] == 'SENESCENT') & (df['CanopyResistance'] >= params['ageInPlaceThreshold'])

    df.loc[age_in_place_mask, 'action'] = 'AGE-IN-PLACE'
    df.loc[replace_mask, 'action'] = 'REPLACE'

    # Print number of trees senescing, aging in place, and being replaced
    print(f"Number of trees senescing: {df[df['action'] == 'SENESCENT'].shape[0]}")
    print(f"Number of trees aging in place: {df[df['action'] == 'AGE-IN-PLACE'].shape[0]}")
    print(f"Number of trees being replaced: {df[df['action'] == 'REPLACE'].shape[0]}")

    return df

def senesce_trees(df, params, seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Mark all AGE-IN-PLACE trees as 'senescing'
    senescing_mask = df['action'] == 'AGE-IN-PLACE'
    df.loc[senescing_mask, 'size'] = 'senescing'

    df.loc[senescing_mask, 'control'] = 'improved-tree' #TODO: for compatibility with current tree meshes

    # Parameters for snag and collapsed thresholds
    snag_threshold = params['snagThreshold']
    collapsed_threshold = params['collapsedThreshold']
    
    # Use the remap function to calculate snagChance and collapseChance
    df['snagChance'] = remap_values_xarray(df['useful_life_expectancy'], old_min=snag_threshold, old_max=0, new_min=100, new_max=0).clip(0, 100)
    df['collapseChance'] = remap_values_xarray(df['useful_life_expectancy'], old_min=collapsed_threshold, old_max=0, new_min=100, new_max=0).clip(0, 100)
    
    # Assign a random roll between 0 and 100 to each row
    df['snagRoll'] = np.random.uniform(0, 100, len(df))
    df['collapseRoll'] = np.random.uniform(0, 100, len(df))

    # Apply the snag condition if snagRoll is below snagChance
    snag_mask = (df['size'] == 'senescing') & (df['snagRoll'] < df['snagChance'])
    df.loc[snag_mask, 'size'] = 'snag'

    # Apply the collapsed condition if collapseRoll is below collapseChance
    collapse_mask = df['size'].isin(['senescing', 'snag']) & (df['collapseRoll'] < df['collapseChance'])
    df.loc[collapse_mask, 'size'] = 'fallen'

    # Print counts for verification
    num_senescing = df[df['size'] == 'senescing'].shape[0]
    num_collapsed = df[df['size'] == 'fallen'].shape[0]
    num_snag = df[df['size'] == 'snag'].shape[0]
    
    print(f'Number of senescing trees: {num_senescing}')
    print(f'Number of collapsed trees: {num_collapsed}')
    print(f'Number of snag trees: {num_snag}')
    
    return df

def assign_depaved_status(df, params):
    # Create a mask for rows where action is 'AGE-IN-PLACE'
    mask = df['action'] == 'AGE-IN-PLACE'

    print(f'Number of trees where action is AGE-IN-PLACE: {mask.sum()}')

    if mask.sum() > 0:
    
        # Use pd.cut to assign rewilded status based on CanopyResistance and the given thresholds
        df.loc[mask, 'rewilded'] = pd.cut(df.loc[mask, 'CanopyResistance'],
                                        bins=[-float('inf'), params['rewildThreshold'], params['plantThreshold'], params['ageInPlaceThreshold'], float('inf')],
                                        labels=['node-rewilded', 'footprint-depaved', 'exoskeleton', 'None'])

    # Print breakdown of rewilded column for verification
    print(f'Number of trees rewilded as "node-rewilded": {df[df["rewilded"] == "node-rewilded"].shape[0]}')
    print(f'Number of trees rewilded as "footprint-depaved": {df[df["rewilded"] == "footprint-depaved"].shape[0]}')
    print(f'Number of trees rewilded as "exoskeleton": {df[df["rewilded"] == "exoskeleton"].shape[0]}')
    
    return df

def assign_rewilded_status(df, ds, params):
    # Mask for sim_Nodes exceeding rewilding threshold (now called depavedMask)
    depaved_threshold = params['sim_TurnsThreshold'][params['years_passed']]
    depaved_mask = (ds['sim_Turns'] <= depaved_threshold) & (ds['sim_Turns'] >= 0)
    print(f'Number of voxels where depaved threshold of {depaved_threshold} is satisfied: {depaved_mask.sum().item()}')
    
    # Terrain mask to exclude 'facade' and 'roof' elements
    terrain_mask = (ds['site_building_element'] != 'facade') & (ds['site_building_element'] != 'roof')

    print(f'Number of voxels where terrain mask is satisfied: {terrain_mask.sum().item()}')
    
    # Combined mask to filter relevant points for proximity check
    combined_mask = depaved_mask & terrain_mask

    # Save combined mask to ds
    ds['scenario_rewildingEnabled'][combined_mask] = params['years_passed']

    print(f'Number of voxels where rewilding is enabled: {(ds["scenario_rewildingEnabled"] >= 0).sum().item()}')
    
    # Filter relevant voxel positions (centroid_x, centroid_y, centroid_z) using the combined mask
    voxel_positions = np.vstack([
        ds['centroid_x'].values[combined_mask], 
        ds['centroid_y'].values[combined_mask], 
        ds['centroid_z'].values[combined_mask]
    ]).T
    
    # Get the corresponding voxel IDs based on the same mask
    filtered_voxel_ids = np.arange(ds.dims['voxel'])[combined_mask]
    
    # Stack tree locations (x, y, z) from the dataframe
    tree_locations = np.vstack([df['x'], df['y'], df['z']]).T
    
    # Create a cKDTree for efficient proximity searching
    tree_kdtree = cKDTree(tree_locations)
    
    # Query distances to determine proximity to tree locations
    distance_threshold = 5  # meters
    distances, _ = tree_kdtree.query(voxel_positions, distance_upper_bound=distance_threshold)
    
    # Create a proximity mask for the relevant filtered points
    filtered_proximity_mask = distances > distance_threshold
    
    # Initialize a full-length proximity mask for ds, starting as all False
    proximity_mask = np.full(ds.dims['voxel'], False)
    
    # Map the true values of filtered_proximity_mask back to the full-length proximity_mask
    proximity_mask[filtered_voxel_ids] = filtered_proximity_mask
    
    # Final mask: Only include voxels that satisfy all three conditions
    final_mask = depaved_mask & terrain_mask & proximity_mask
    
    # Assign the rewilding plantings status where the final mask is satisfied
    ds['scenario_rewildingPlantings'] = xr.where(final_mask, params['years_passed'], -1)

    print(f'Number of voxels where rewilding plantings are enabled: {(ds["scenario_rewildingPlantings"] >= 0).sum().item()}')
    
    return df, ds

# Function to handle REWILD/FOOTPRINT-DEPAVED logic for other trees
def reduce_control_of_trees(df, params):
    print('Reduce control of trees')
    nonSenescentMask = df['useful_life_expectancy'] > 0

    print(f'Number of non-senescent trees: {nonSenescentMask.sum()}')

    # Decide whether to REWILD or not based on params
    df.loc[nonSenescentMask, 'rewilded'] = pd.cut(df.loc[nonSenescentMask, 'CanopyResistance'],
                                      bins=[-float('inf'), params['rewildThreshold'], params['plantThreshold'], float('inf')],
                                      labels=['node-rewilded', 'footprint-depaved', 'None'])
    
    # Print how many non-senescent trees were allocated to 'node-rewilded' and 'footprint-depaved'
    num_rewilded = df[(df['rewilded'] == 'node-rewilded') & nonSenescentMask].shape[0]
    num_footprint_depaved = df[(df['rewilded'] == 'footprint-depaved') & nonSenescentMask].shape[0]
    print(f"Number of non-senescent trees allocated to 'node-rewilded': {num_rewilded}")
    print(f"Number of non-senescent trees allocated to 'footprint-depaved': {num_footprint_depaved}")

    # Get mask for rows that are rewilded and non-senescent
    mask = (df['rewilded'] != 'None') & nonSenescentMask
    
    # Increase unmanagedCount by the years passed, not set it directly
    df.loc[mask, 'unmanagedCount'] = df.loc[mask, 'unmanagedCount'] + params['years_passed']
    
    # Store current control status for comparison
    previous_control = df['control'].copy()

    # Update control status based on unmanagedCount and control thresholds
    df.loc[mask, 'control'] = pd.cut(df.loc[mask, 'unmanagedCount'],
                                     bins=[-float('inf'), params['controlSteps'], 2*params['controlSteps'], float('inf')],
                                     labels=['street-tree', 'park-tree', 'reserve-tree'])
    
    # Reset unmanagedCount to 0 for rows where control status advanced
    advanced_mask = df['control'].isin(['park-tree', 'reserve-tree'])
    df.loc[advanced_mask, 'unmanagedCount'] = 0

    # Print the number of small, medium, and large trees that advanced from 'street-tree' to 'park-tree'
    advanced_street_to_park = df[(previous_control == 'street-tree') & (df['control'] == 'park-tree')]
    print(f"Number of small trees advanced from 'street-tree' to 'park-tree': {advanced_street_to_park[advanced_street_to_park['size'] == 'small'].shape[0]}")
    print(f"Number of medium trees advanced from 'street-tree' to 'park-tree': {advanced_street_to_park[advanced_street_to_park['size'] == 'medium'].shape[0]}")
    print(f"Number of large trees advanced from 'street-tree' to 'park-tree': {advanced_street_to_park[advanced_street_to_park['size'] == 'large'].shape[0]}")

    # Print the number of small, medium, and large trees that advanced from 'park-tree' to 'reserve-tree'
    advanced_park_to_reserve = df[(previous_control == 'park-tree') & (df['control'] == 'reserve-tree')]
    print(f"Number of small trees advanced from 'park-tree' to 'reserve-tree': {advanced_park_to_reserve[advanced_park_to_reserve['size'] == 'small'].shape[0]}")
    print(f"Number of medium trees advanced from 'park-tree' to 'reserve-tree': {advanced_park_to_reserve[advanced_park_to_reserve['size'] == 'medium'].shape[0]}")
    print(f"Number of large trees advanced from 'park-tree' to 'reserve-tree': {advanced_park_to_reserve[advanced_park_to_reserve['size'] == 'large'].shape[0]}")

    return df

# Function to handle REPLACE logic
def handle_replace_trees(df):
    # Vectorized approach for replacing collapsed trees
    df.loc[df['action'] == 'REPLACE', 
           ['size', 'diameter_breast_height', 'precolonial']] = ['small', 10, True]
    
    return df

def handle_plant_trees(df, ds, params, seed=42):
    # Option to use node-level or turn-based logic for rewilding plantings
    np.random.seed(seed)  # Set the random seed
    
    # Assign rows in sim_NodesArea that are NaN to their CanopyArea values
    df['sim_NodesArea'] = df['sim_NodesArea'].fillna(df['CanopyArea'])

    # Create temp area column
    df['temp_area'] = 0
    df.loc[df['rewilded'] == 'footprint-depaved', 'temp_area'] = df['CanopyArea']
    df.loc[df['rewilded'] == 'node-rewilded', 'temp_area'] = df['sim_NodesArea']
    df.loc[df['rewilded'] == 'exoskeleton', 'temp_area'] = 0

    # Only apply the planting logic to existing trees (not new trees) and nodes that have not been replanted
    mask = ~df['isNewTree'] & ~df['hasbeenReplanted']
    
    # Calculate planting density (convert to square meters)
    planting_density_sqm = params['plantingDensity'] / 10000
    
    # Calculate number of trees to plant, rounding up to nearest whole number
    df.loc[mask, 'number_of_trees_to_plant'] = np.ceil(df.loc[mask, 'temp_area'] * planting_density_sqm)
    
    # Set number_of_trees_to_plant to 0 for new trees
    df.loc[~mask, 'number_of_trees_to_plant'] = 0

    # Create a mask for rows where trees need to be planted
    to_plant_mask = df['number_of_trees_to_plant'] > 0

    print(f'Node-based area to plant trees: {df["sim_NodesArea"].sum()}')
    print(f'Node-based number of trees to plant: {df["number_of_trees_to_plant"].sum()}')
    
    ####END OF NODE BASED LOGIC#####

    #####DETERMINE TURN BASED PLANTING LOGIC#####
    # Print all variables in ds
    print("Variables in ds:")
    for var_name in ds.variables:
        print(f"- {var_name}")
    print("\n")

    # Determine voxels to plant trees based on ds['scenario_rewildingPlantings']
    plantingMask = ds['scenario_rewildingPlantings'] == params['years_passed']

    # Determine area to plant trees based on plantingMask and xarray attribute ds.attrs['voxel_size']
    area_to_plant = plantingMask.sum().item() * ds.attrs['voxel_size'] * ds.attrs['voxel_size']  # area in square meters
    print(f'Turn-based area to rewild trees: {area_to_plant} m²')

    # Determine number of trees to plant based on area_to_plant and planting_density
    noTreesToPlantTurnBased = np.round(area_to_plant * planting_density_sqm)
    print(f'Turn-based number of trees to plant: {noTreesToPlantTurnBased}')

    # Create new rows for turn-based logic
    turn_based_tree_data = {
        'size': ['small'] * int(noTreesToPlantTurnBased),
        'diameter_breast_height': [2] * int(noTreesToPlantTurnBased),
        'precolonial': [True] * int(noTreesToPlantTurnBased),
        'isNewTree': [True] * int(noTreesToPlantTurnBased),
        'control': ['reserve-tree'] * int(noTreesToPlantTurnBased),
        'useful_life_expectency': [120] * int(noTreesToPlantTurnBased),
        'tree_id': [-1] * int(noTreesToPlantTurnBased),
        'tree_number': [-1] * int(noTreesToPlantTurnBased),
        'nodeID': [-1] * int(noTreesToPlantTurnBased),
        'isRewildedTree': [True] * int(noTreesToPlantTurnBased)
    }

    # Extract x, y, z from ds for the plantingMask
    available_positions = np.vstack([
        ds['centroid_x'].values[plantingMask],
        ds['centroid_y'].values[plantingMask],
        ds['centroid_z'].values[plantingMask]
    ]).T

    # Shuffle the available positions
    np.random.shuffle(available_positions)

    # Select the first noTreesToPlantTurnBased positions from the shuffled array
    selected_positions = available_positions[:int(noTreesToPlantTurnBased)]

    # Assign the selected x, y, z values to the new trees
    turn_based_tree_data['x'] = selected_positions[:, 0] if len(selected_positions) > 0 else []
    turn_based_tree_data['y'] = selected_positions[:, 1] if len(selected_positions) > 0 else []
    turn_based_tree_data['z'] = selected_positions[:, 2] if len(selected_positions) > 0 else []

    # Create the new DataFrame for turn-based trees
    new_trees_turn_df = pd.DataFrame(turn_based_tree_data)

    # Print turn-based planting details
    print(f'Turn-based number of new trees added: {len(new_trees_turn_df)}')

    ##END OF TURN_BASED LOGIC#####

    # Repeat the rows based on 'number_of_trees_to_plant' for node-based logic and create a new DataFrame
    repeated_indices = np.repeat(df[to_plant_mask].index, df.loc[to_plant_mask, 'number_of_trees_to_plant'].astype(int))
    new_trees_node_df = df.loc[repeated_indices].copy() if len(repeated_indices) > 0 else pd.DataFrame()

    # Update node-based new trees' specific attributes
    if not new_trees_node_df.empty:
        # Randomly jitter the original x and y positions by up to 2.5 meters. #TODO: select node from available positions
        new_trees_node_df['x'] = new_trees_node_df['x'] + np.random.uniform(-2.5, 2.5, len(new_trees_node_df))
        new_trees_node_df['y'] = new_trees_node_df['y'] + np.random.uniform(-2.5, 2.5, len(new_trees_node_df))
        new_trees_node_df['size'] = 'small'
        new_trees_node_df['diameter_breast_height'] = 2
        new_trees_node_df['precolonial'] = True
        new_trees_node_df['isNewTree'] = True
        new_trees_node_df['control'] = 'reserve-tree'
        new_trees_node_df['useful_life_expectency'] = 120
        new_trees_node_df['tree_id'] = -1

    # Print statistics about the new node-based trees
    print(f'Node-based number of new trees planted: {len(new_trees_node_df)}')
    if not new_trees_node_df.empty:
        print(f'Average CanopyArea for node-based new trees: {new_trees_node_df["CanopyArea"].mean()}')

    # Combine the node-based and turn-based new trees DataFrames
    newTreesDF = pd.concat([new_trees_node_df, new_trees_turn_df], ignore_index=True)

    if not newTreesDF.empty:
        # Updated DBH and size for newTreesDF
        years_passed = params['years_passed']

        # Informative print statement for years passed
        print(f"Years passed since last planting: {years_passed}")

        # Define percentage fractions for growth stages (algorithmic binning), allowing fractions above 1
        growth_fractions = params.get('growth_fractions', [0, 0.25, 0.75, 1.0, 1.2])  # Flexible fraction list
        n_bins = len(growth_fractions) - 1  # Exclude the 0th fraction, since it represents no growth

        # Shuffle the newTreesDF to randomize the assignment to bins
        newTreesDF = newTreesDF.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Calculate the bin size (how many trees per bin)
        total_trees = len(newTreesDF)
        bin_size = total_trees // n_bins if n_bins > 0 else 1  # Integer division

        # Assign bin numbers to a new column 'temp_growthbin' based on bin size
        newTreesDF['temp_growthbin'] = (np.arange(total_trees) // bin_size).clip(0, n_bins - 1)

        # Assign years passed for each bin to a new column 'temp_yearspassed'
        newTreesDF['temp_yearspassed'] = newTreesDF['temp_growthbin'].map(lambda i: growth_fractions[i + 1] * years_passed)

        # Growth factor per year (using mean as before)
        growth_factor_per_year = (params['growth_factor_range'][0] + params['growth_factor_range'][1]) / 2
        print(f"Growth factor per year: {growth_factor_per_year}")

        # Assign DBH increase for each bin based on years passed to a new column 'temp_dbhIncrease'
        newTreesDF['temp_dbhIncrease'] = growth_factor_per_year * newTreesDF['temp_yearspassed']

        # Apply the DBH increase using the 'temp_dbhIncrease' column
        newTreesDF['diameter_breast_height'] += newTreesDF['temp_dbhIncrease']

        # Print the intermediate values of the temp columns for debugging
        print(f"Distribution across growth bins (temp_growthbin):")
        print(newTreesDF['temp_growthbin'].value_counts())
        print(f"Years passed per bin (temp_yearspassed):")
        print(newTreesDF[['temp_growthbin', 'temp_yearspassed']].drop_duplicates())
        print(f"DBH increase per bin (temp_dbhIncrease):")
        print(newTreesDF[['temp_growthbin', 'temp_dbhIncrease']].drop_duplicates())

        # Print values and counts of newTreesDF['diameter_breast_height'] after growth
        print(f"Values and counts of newTreesDF['diameter_breast_height'] after growth:")
        print(newTreesDF['diameter_breast_height'].value_counts())

        # Update tree size classification based on DBH
        newTreesDF['size'] = pd.cut(newTreesDF['diameter_breast_height'], 
                                    bins=[-10, 30, 80, float('inf')], 
                                    labels=['small', 'medium', 'large']).astype(str)
        print(f"Updated tree size classification based on DBH. Tree count per size category:")
        print(newTreesDF['size'].value_counts())

        # Decrease useful_life_expectency of each tree based on years passed
        if 'useful_life_expectency' in newTreesDF.columns:
            newTreesDF['useful_life_expectency'] -= years_passed
            print(f"Reduced useful life expectancy by {years_passed} years for all trees.")

            # Print useful_life_expectency range after aging
            print(f"After aging {years_passed} years:")
            print(f"Useful life expectancy range: {newTreesDF['useful_life_expectency'].min()} to {newTreesDF['useful_life_expectency'].max()}")
            print(f"Breakdown of useful life expectancy at year {years_passed}:")
            print(newTreesDF['useful_life_expectency'].value_counts())

    # Append both new node-based and turn-based trees DataFrames to the original
    df = pd.concat([df, newTreesDF], ignore_index=True)

    # Print diagnostics for the combined DataFrame
    print(f'Total area for tree planting: {df["sim_NodesArea"].sum()}')
    print(f'Total number of trees after planting: {len(df)}')

    return df

def assign_logs(logDF, params):
    if logDF is None:
        return None
        
    turnThreshold = params['sim_TurnsThreshold'][params['years_passed']]
    resistanceThreshold = params['sim_averageResistance'][params['years_passed']] if 'sim_averageResistance' in params else 0

    mask = (logDF['sim_averageResistance'] <= resistanceThreshold) & (logDF['sim_Turns'] <= turnThreshold)

    # Assign the mask to the logDF
    logDF['isEnabled'] = mask

    # Print the number of logs enabled this turn
    enabled_logs_count = logDF['isEnabled'].sum()
    print(f"Number of logs enabled this turn: {enabled_logs_count}")

    # Print the value counts of the enabled logDF['logSize']
    if enabled_logs_count > 0 and 'logSize' in logDF.columns:
        enabled_log_size_counts = logDF[logDF['isEnabled']]['logSize'].value_counts()
        print(f"Value counts of enabled log sizes:\n{enabled_log_size_counts}")

    # Extract the logDF that is enabled
    enabled_logDF = logDF[logDF['isEnabled']]

    return enabled_logDF

def assign_poles(poleDF, params):
    if poleDF is None:
        return None
        
    turnThreshold = params['sim_TurnsThreshold'][params['years_passed']]
    resistanceThreshold = params['sim_averageResistance'][params['years_passed']] if 'sim_averageResistance' in params else 0

    print(f'resistanceThreshold: {resistanceThreshold}')
    mask = poleDF['sim_averageResistance'] < resistanceThreshold

    print(f'value counts of poleDF["sim_averageResistance"]: {poleDF["sim_averageResistance"].value_counts()}')

    # Assign the mask to the poleDF
    poleDF['isEnabled'] = mask

    enabled_poleDF = poleDF[poleDF['isEnabled']]

    # Extract the poleDF that is enabled
    enabled_poleDF = enabled_poleDF[poleDF['isEnabled']]

    print(f'number of poles enabled this turn: {len(enabled_poleDF)}')  

    print(f'value counts of poleDF["sim_averageResistance"] that are enabled this turn: {enabled_poleDF["sim_averageResistance"].value_counts() if not enabled_poleDF.empty else "None"}')

    return enabled_poleDF

def run_simulation(df, ds, params, logDF=None, poleDF=None):    
    # Print all variable names in ds
    print("Variables in ds:")
    for var_name in ds.variables:
        print(f"- {var_name}")
    
    print(f'Total Trees: {len(df)}')
    df = df.copy()

    print(f'aging trees')
    df = age_trees(df, params)

    print(f'determine trees that AGE-IN-PLACE or REPLACE')
    df = determine_ageinplace_or_replace(df.copy(), params)
    
    print('senesce trees that AGE-IN-PLACE')
    df = senesce_trees(df.copy(), params)

    print('replace trees trees below AGE-IN-PLACE')
    df = handle_replace_trees(df.copy())

    print(f'handle REWILD/EXOSKELETON/FOOTPRINT-DEPAVED logic for AGE-IN-PLACE trees')
    df = assign_depaved_status(df.copy(), params)  # Assign the correct rewilding status

    print(f'Handle node-based rewilding')
    df, ds = assign_rewilded_status(df.copy(), ds, params)

    print(f'reduce control of non senescent trees')
    df = reduce_control_of_trees(df.copy(), params)

    print(f'handle PLANT logic')
    df = handle_plant_trees(df.copy(), ds, params)

    print(f'add structureID column')
    # This is just the index
    df['structureID'] = df.index

    if logDF is not None:
        print(f'assign logs')
        logDF = assign_logs(logDF, params)
        if logDF is not None and not logDF.empty:
            logDF['structureID'] = logDF.index + len(df)

    if poleDF is not None:
        print(f'assign poles')
        poleDF = assign_poles(poleDF, params)
        if poleDF is not None and not poleDF.empty:
            poleDF['structureID'] = poleDF.index + len(df)

            if logDF is not None and not logDF.empty:
                poleDF['structureID'] = poleDF['structureID'] + len(logDF)
    
    return df, logDF, poleDF

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

    poly = a_helper_functions.convert_xarray_into_polydata(ds)
    poly.plot(scalars='scenario_bioEnvelope', cmap='Set1')

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

def process_scenarios(site, scenario, voxel_size):
    # Query the user to enter a list of years passed or press return to run all years
    all_years = [0, 10, 30, 60, 180]
    years_passed = input(f"Please enter a list of years passed or press return to run all years: {all_years}")
    if years_passed == '':
        years_passed = all_years
        print(f'Running all years: {years_passed}')
    else:
        years_passed = [int(year) for year in years_passed.split(',')]
        print(f'Running years: {years_passed}')

    # Initialize dataset and load nodes
    filePATH = f'data/revised/final/{site}'
    
    # Get the data
    treeDF, poleDF, logDF, subsetDS = a_scenario_initialiseDS.initialize_scenario_data(site, voxel_size)
    
    # Get scenario parameters
    paramsDic = get_scenario_parameters()
    params = paramsDic[(site, scenario)]
    
    for year in years_passed:
        print(f'Running simulation for year {year}')
        
        ##############################
        ## SET UP
        ##############################
        
        params['years_passed'] = year
        
        # Set up site-specific conditions
        if site == 'trimmed-parade':
            logDF = None
            poleDF = None
        
        if site == 'uni':
            logDF = None

        if site == 'city':
            poleDF = None

        ##############################
        # SIMULATION
        ##############################

        print(f'treeDF is {treeDF}')
        print(f'logDF is {logDF}')
        print(f'poleDF is {poleDF}')
 
        treeDF_scenario, logDF_scenario, poleDF_scenario = run_simulation(treeDF.copy(), subsetDS, params, logDF, poleDF)
        print(f'Done running simulation for year {year}')
        
        ##############################
        # REWILDING CATEGORY UPDATES
        ##############################

        print('Integrating results into xarray')
        ds = update_rewilded_voxel_catagories(subsetDS, treeDF_scenario)
        print('Updating bioEnvelope voxel categories')
        if logDF is not None or poleDF is not None:
            ds = update_bioEnvelope_voxel_catagories(ds, params)

        unique_values, counts = np.unique(ds['scenario_rewilded'], return_counts=True)
        print(f'Unique values and counts for scenario_rewilded: {unique_values}, {counts}')

        ###################################################
        # ADDING RESOURCE VOXELS AND CALCULATING ROTATIONS
        ###################################################

        validpointsMask = ds['scenario_rewilded'].values != 'none'
        if logDF_scenario is not None:
             validpointsMask = ds['scenario_bioEnvelope'].values != 'none'

        if poleDF_scenario is not None:
             validpointsMask = ds['scenario_bioEnvelope'].values != 'none'

        # Extract valid points as a numpy array
        valid_points = np.array([
            ds['centroid_x'].values[validpointsMask],
            ds['centroid_y'].values[validpointsMask],
            ds['centroid_z'].values[validpointsMask]
        ]).T

        ds, combinedDF_scenario = a_voxeliser.integrate_resources_into_xarray(ds, treeDF_scenario, logDF_scenario, poleDF_scenario, valid_points)
        
        ##############################
        # FINAL PROCESSING
        ##############################

        ds = finalDSprocessing(ds)

        ##############################
        # SAVING
        ##############################

        # Save combinedDF_scenario to csv
        print(f'Saving {year} combinedDF_scenario to csv')
        combinedDF_scenario.to_csv(f'{filePATH}/{site}_{scenario}_{voxel_size}_nodeDF_{year}.csv', index=False)
        
        print_simulation_statistics(combinedDF_scenario, year, site)

        polydata = a_helper_functions.convert_xarray_into_polydata(ds)
        polydata = process_polydata(polydata)
        polydata.save(f'{filePATH}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk')

        ##############################
        # VALIDATION
        ##############################
        print('Validating results')
        # Print values and counts for scenario_rewilded
        unique_values, counts = np.unique(ds['scenario_rewilded'], return_counts=True)
        print(f'Unique values and counts for scenario_rewilded: {unique_values}, {counts}')
        
        if logDF is not None:
            # Print values and counts for scenario_bioEnvelope
            unique_values, counts = np.unique(ds['scenario_bioEnvelope'], return_counts=True)
            print(f'Unique values and counts for scenario_bioEnvelope: {unique_values}, {counts}')

        # Optional: Uncomment to enable plotting
        # plot_scenario_rewilded(polydata, treeDF_scenario, year, site)

if __name__ == "__main__":
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
    
    # Run the scenarios
    for site in sites:
        for scenario in scenarios:
            print(f"\n===== Processing {site} with {scenario} scenario =====\n")
            process_scenarios(site, scenario, voxel_size)
    
    print("\nAll scenario processing completed.") 