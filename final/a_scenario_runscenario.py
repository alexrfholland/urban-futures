import os
import pandas as pd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import a_helper_functions
import a_scenario_initialiseDS  # Import for preprocessing functions
import a_scenario_params  # Import the centralized parameters module

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
    ds['scenario_rewildingEnabled'] = params['years_passed'] * xr.where(combined_mask, 1, -1)

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
        'useful_life_expectancy': [120] * int(noTreesToPlantTurnBased),
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
        new_trees_node_df['useful_life_expectancy'] = 120
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

        # Decrease useful_life_expectancy of each tree based on years passed
        if 'useful_life_expectancy' in newTreesDF.columns:
            newTreesDF['useful_life_expectancy'] -= years_passed
            print(f"Reduced useful life expectancy by {years_passed} years for all trees.")

            # Print useful_life_expectancy range after aging
            print(f"After aging {years_passed} years:")
            print(f"Useful life expectancy range: {newTreesDF['useful_life_expectancy'].min()} to {newTreesDF['useful_life_expectancy'].max()}")
            print(f"Breakdown of useful life expectancy at year {years_passed}:")
            print(newTreesDF['useful_life_expectancy'].value_counts())

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
    
    # More robust check for sim_averageResistance
    resistanceThreshold = 0  # Default value
    if 'sim_averageResistance' in params:
        if params['years_passed'] in params['sim_averageResistance']:
            resistanceThreshold = params['sim_averageResistance'][params['years_passed']]
        else:
            print(f"Warning: Year {params['years_passed']} not found in sim_averageResistance. Using default value 0.")
    
    print(f"Turn threshold: {turnThreshold}, Resistance threshold: {resistanceThreshold}")

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
    
    # More robust check for sim_averageResistance
    resistanceThreshold = 0  # Default value
    if 'sim_averageResistance' in params:
        if params['years_passed'] in params['sim_averageResistance']:
            resistanceThreshold = params['sim_averageResistance'][params['years_passed']]
        else:
            print(f"Warning: Year {params['years_passed']} not found in sim_averageResistance. Using default value 0.")

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

def run_scenario(site, scenario, year, voxel_size, treeDF, subsetDS, logDF=None, poleDF=None):
    """
    Runs a scenario simulation for a given site, scenario, and year.
    
    Parameters:
    site (str): Site name ('trimmed-parade', 'city', 'uni')
    scenario (str): Scenario type ('positive', 'trending')
    year (int): Year to simulate
    voxel_size (int): Voxel size
    treeDF (pd.DataFrame): Tree dataframe
    subsetDS (xarray.Dataset): Xarray dataset
    logDF (pd.DataFrame, optional): Log dataframe
    poleDF (pd.DataFrame, optional): Pole dataframe
    
    Returns:
    tuple: (treeDF_scenario, logDF_scenario, poleDF_scenario) - Simulated dataframes
    """
    # Get scenario parameters from the centralized module
    paramsDic = a_scenario_params.get_scenario_parameters()
    params = paramsDic[(site, scenario)]
    params['years_passed'] = year
    
    # Set up site-specific conditions
    if site == 'trimmed-parade':
        logDF = None
        poleDF = None
    
    if site == 'uni':
        logDF = None

    if site == 'city':
        poleDF = None
    
    # Process log and pole dataframes using the functions from a_scenario_initialiseDS
    if logDF is not None:
        logDF = a_scenario_initialiseDS.log_processing(logDF, subsetDS)
    
    if poleDF is not None:
        poleDF = a_scenario_initialiseDS.pole_processing(poleDF, None, subsetDS)
        
    # Run the simulation
    print(f'Running simulation for {site}, {scenario}, year {year}')
    treeDF_scenario, logDF_scenario, poleDF_scenario = run_simulation(treeDF.copy(), subsetDS, params, logDF, poleDF)
    print(f'Done running simulation for year {year}')
    
    # Save the dataframes
    output_path = f'data/revised/final/{site}'
    os.makedirs(output_path, exist_ok=True)
    
    treeDF_scenario.to_csv(f'{output_path}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv', index=False)
    print(f'Saved tree dataframe to {output_path}/{site}_{scenario}_{voxel_size}_treeDF_{year}.csv')
    
    if logDF_scenario is not None:
        logDF_scenario.to_csv(f'{output_path}/{site}_{scenario}_{voxel_size}_logDF_{year}.csv', index=False)
        print(f'Saved log dataframe to {output_path}/{site}_{scenario}_{voxel_size}_logDF_{year}.csv')
    
    if poleDF_scenario is not None:
        poleDF_scenario.to_csv(f'{output_path}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv', index=False)
        print(f'Saved pole dataframe to {output_path}/{site}_{scenario}_{voxel_size}_poleDF_{year}.csv')
    
    return treeDF_scenario, logDF_scenario, poleDF_scenario

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
    years_input = input(f"Enter years to simulate (comma-separated) or press Enter for default {all_years}: ")
    years = [int(year) for year in years_input.split(',')] if years_input else all_years
    
    # Run scenarios for each site, scenario, and year
    for site in sites:
        for scenario in scenarios:
            print(f"\n===== Processing {site} with {scenario} scenario =====\n")
            
            # Initialize data
            treeDF, poleDF, logDF, subsetDS = a_scenario_initialiseDS.initialize_scenario_data(site, voxel_size)
            
            for year in years:
                print(f"\n--- Running simulation for year {year} ---\n")
                run_scenario(site, scenario, year, voxel_size, treeDF, subsetDS, logDF, poleDF)
                
    print("\nAll scenario simulations completed.")