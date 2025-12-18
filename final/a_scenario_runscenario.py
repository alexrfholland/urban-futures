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

def calculate_rewilded_status(df, ds, params):
    # Mask for sim_Nodes exceeding rewilding threshold (now called depavedMask)
    # Note: sim_TurnsThreshold is pre-interpolated for sub-timesteps
    depaved_threshold = params['sim_TurnsThreshold']
    depaved_mask = (ds['sim_Turns'] <= depaved_threshold) & (ds['sim_Turns'] >= 0)
    print(f'Number of voxels where depaved threshold of {depaved_threshold} is satisfied: {depaved_mask.sum().item()}')
    
    # Terrain mask to exclude 'facade' and 'roof' elements
    terrain_mask = (ds['site_building_element'] != 'facade') & (ds['site_building_element'] != 'roof')

    print(f'Number of voxels where terrain mask is satisfied: {terrain_mask.sum().item()}')
    
    # Combined mask to filter relevant points for proximity check
    combined_mask = depaved_mask & terrain_mask

    # Save combined mask to ds
    ds['scenario_rewildingEnabled'] = max(1, params['years_passed']) * xr.where(combined_mask, 1, -1)


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
    # Also modified to use max(1, years_passed)
    ds['scenario_rewildingPlantings'] = xr.where(final_mask, max(1, params['years_passed']), -1)

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

# Function to handle REPLACE logic with time-distributed growth
def handle_replace_trees(df, params, seed=42):
    """
    Replace trees with age-appropriate replacements based on when they died.
    
    BACKDATING LOGIC FOR REPLACED TREES:
    =====================================
    When a tree is marked for REPLACE, its useful_life_expectancy (ULE) tells us
    exactly when it died:
    
    - ULE is reduced by years_passed in age_trees()
    - If ULE becomes negative, the tree exceeded its useful life
    - The magnitude of negative ULE = how many years ago the tree "died"
    - The replacement tree was planted at that death time
    - The replacement has been growing since then
    
    Example:
        - Initial ULE = 10 years
        - years_passed = 60
        - Current ULE = 10 - 60 = -50
        - Tree died 50 years ago (when ULE hit 0)
        - Replacement was planted 50 years ago
        - Replacement growth years = 50
        - Replacement DBH = 2 + (0.44 × 50) = 24 cm (medium tree)
    
    Formula: replacement_growth_years = abs(min(0, current_ULE))
    """
    np.random.seed(seed)
    
    replace_mask = df['action'] == 'REPLACE'
    if not replace_mask.any():
        return df
    
    years_passed = params['years_passed']
    growth_factor = (params['growth_factor_range'][0] + params['growth_factor_range'][1]) / 2
    initial_dbh = 2  # Small sapling starting DBH
    
    # Get current ULE (already reduced by years_passed in age_trees)
    current_ule = df.loc[replace_mask, 'useful_life_expectancy'].values
    
    # Calculate how long the replacement has been growing
    # If ULE is negative, replacement has been growing for abs(ULE) years
    # Clamp to [0, years_passed] for safety
    replacement_growth_years = np.clip(-np.minimum(0, current_ule), 0, years_passed)
    
    # Calculate replacement DBH based on growth time
    replacement_dbh = initial_dbh + (growth_factor * replacement_growth_years)
    
    # Update the replaced trees
    df.loc[replace_mask, 'diameter_breast_height'] = replacement_dbh
    df.loc[replace_mask, 'precolonial'] = True
    df.loc[replace_mask, 'useful_life_expectancy'] = 120 - replacement_growth_years
    
    # Update size classification based on new DBH (vectorized)
    df.loc[replace_mask, 'size'] = pd.cut(
        df.loc[replace_mask, 'diameter_breast_height'],
        bins=[-10, 30, 80, float('inf')],
        labels=['small', 'medium', 'large']
    ).astype(str)
    
    # Print summary statistics
    print(f"\n--- REPLACED Trees with Time-Distributed Growth ---")
    print(f"Number of replaced trees: {replace_mask.sum()}")
    print(f"Replacement growth years - min: {replacement_growth_years.min():.1f}, "
          f"max: {replacement_growth_years.max():.1f}, mean: {replacement_growth_years.mean():.1f}")
    print(f"Replacement DBH range: {replacement_dbh.min():.1f} - {replacement_dbh.max():.1f} cm")
    print(f"Size distribution: {df.loc[replace_mask, 'size'].value_counts().to_dict()}")
    
    return df

def get_previous_timestep(years_passed):
    """Get the previous timestep for the current years_passed."""
    timesteps = [0, 10, 30, 60, 180]
    for i, ts in enumerate(timesteps):
        if ts == years_passed:
            return timesteps[i - 1] if i > 0 else 0
    return 0


def handle_plant_trees(df, ds, params, seed=42):
    """
    Plant new trees around rewilded areas with time-distributed growth.
    
    BACKDATING LOGIC FOR AGE-IN-PLACE TREE PLANTINGS:
    ==================================================
    When an AGE-IN-PLACE tree triggers rewilding, new trees are planted around it.
    The timing of when the parent tree became senescing/snag/fallen determines
    how long the new trees have been growing.
    
    We use the parent tree's snagChance (or senesceChance) to estimate when
    within the timestep window the event occurred:
    
    - Higher snagChance = event happened earlier in the window = more growth time
    - Lower snagChance = event happened later in the window = less growth time
    
    Formula: growth_years = (snagChance / 100) × time_window
    
    Example (timesteps 60 → 180, window = 120 years):
        - Parent tree snagChance = 75%
        - growth_years = 0.75 × 120 = 90 years
        - New tree DBH = 2 + (0.44 × 90) = 41.6 cm (medium tree)
    
    For turn-based plantings (not tied to specific parent trees), we distribute
    growth uniformly across the time window using random sampling.
    """
    np.random.seed(seed)
    
    years_passed = params['years_passed']
    previous_timestep = get_previous_timestep(years_passed)
    time_window = years_passed - previous_timestep
    growth_factor = (params['growth_factor_range'][0] + params['growth_factor_range'][1]) / 2
    initial_dbh = 2
    
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
    # Determine voxels to plant trees based on ds['scenario_rewildingPlantings']
    plantingMask = ds['scenario_rewildingPlantings'] == params['years_passed']

    # Determine area to plant trees based on plantingMask and xarray attribute ds.attrs['voxel_size']
    area_to_plant = plantingMask.sum().item() * ds.attrs['voxel_size'] * ds.attrs['voxel_size']
    print(f'Turn-based area to rewild trees: {area_to_plant} m²')

    # Determine number of trees to plant based on area_to_plant and planting_density
    noTreesToPlantTurnBased = int(np.round(area_to_plant * planting_density_sqm))
    print(f'Turn-based number of trees to plant: {noTreesToPlantTurnBased}')

    # Create new rows for turn-based logic
    turn_based_tree_data = {
        'size': ['small'] * noTreesToPlantTurnBased,
        'diameter_breast_height': [initial_dbh] * noTreesToPlantTurnBased,
        'precolonial': [True] * noTreesToPlantTurnBased,
        'isNewTree': [True] * noTreesToPlantTurnBased,
        'control': ['reserve-tree'] * noTreesToPlantTurnBased,
        'useful_life_expectancy': [120] * noTreesToPlantTurnBased,
        'tree_id': [-1] * noTreesToPlantTurnBased,
        'tree_number': [-1] * noTreesToPlantTurnBased,
        'nodeID': [-1] * noTreesToPlantTurnBased,
        'isRewildedTree': [True] * noTreesToPlantTurnBased
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
    selected_positions = available_positions[:noTreesToPlantTurnBased]

    # Assign the selected x, y, z values to the new trees
    turn_based_tree_data['x'] = selected_positions[:, 0] if len(selected_positions) > 0 else []
    turn_based_tree_data['y'] = selected_positions[:, 1] if len(selected_positions) > 0 else []
    turn_based_tree_data['z'] = selected_positions[:, 2] if len(selected_positions) > 0 else []

    # Create the new DataFrame for turn-based trees
    new_trees_turn_df = pd.DataFrame(turn_based_tree_data)

    print(f'Turn-based number of new trees added: {len(new_trees_turn_df)}')

    ##END OF TURN_BASED LOGIC#####

    # =========================================================================
    # NODE-BASED PLANTINGS: New trees around AGE-IN-PLACE parent trees
    # =========================================================================
    # Each new tree inherits timing from its parent tree's snagChance/senesceChance
    
    repeated_indices = np.repeat(df[to_plant_mask].index, df.loc[to_plant_mask, 'number_of_trees_to_plant'].astype(int))
    new_trees_node_df = df.loc[repeated_indices].copy() if len(repeated_indices) > 0 else pd.DataFrame()

    if not new_trees_node_df.empty:
        # Randomly jitter positions
        new_trees_node_df['x'] = new_trees_node_df['x'] + np.random.uniform(-2.5, 2.5, len(new_trees_node_df))
        new_trees_node_df['y'] = new_trees_node_df['y'] + np.random.uniform(-2.5, 2.5, len(new_trees_node_df))
        new_trees_node_df['precolonial'] = True
        new_trees_node_df['isNewTree'] = True
        new_trees_node_df['control'] = 'reserve-tree'
        new_trees_node_df['tree_id'] = -1
        
        # ---------------------------------------------------------------------
        # BACKDATING: Use parent tree's snagChance to determine planting timing
        # ---------------------------------------------------------------------
        # Higher snagChance = parent became snag earlier = new tree grew longer
        # snagChance ranges from 0 to 100, maps to fraction of time window
        
        # Use snagChance if available, otherwise senesceChance, otherwise uniform random
        if 'snagChance' in new_trees_node_df.columns:
            parent_chance = new_trees_node_df['snagChance'].fillna(50).values
        elif 'senesceChance' in new_trees_node_df.columns:
            parent_chance = new_trees_node_df['senesceChance'].fillna(50).values
        else:
            # Fallback: uniform random distribution across window
            parent_chance = np.random.uniform(0, 100, len(new_trees_node_df))
        
        # Clamp to valid range
        parent_chance = np.clip(parent_chance, 0, 100)
        
        # Calculate growth years based on parent tree timing
        # Higher parent_chance = earlier event = more growth time
        node_growth_years = (parent_chance / 100) * time_window
        
        # Calculate DBH and update
        new_trees_node_df['diameter_breast_height'] = initial_dbh + (growth_factor * node_growth_years)
        new_trees_node_df['useful_life_expectancy'] = 120 - node_growth_years
        
        # Update size classification
        new_trees_node_df['size'] = pd.cut(
            new_trees_node_df['diameter_breast_height'],
            bins=[-10, 30, 80, float('inf')],
            labels=['small', 'medium', 'large']
        ).astype(str)
        
        print(f"\n--- NODE-BASED Plantings with Time-Distributed Growth ---")
        print(f"Time window: {previous_timestep} → {years_passed} = {time_window} years")
        print(f"Number of node-based new trees: {len(new_trees_node_df)}")
        print(f"Growth years - min: {node_growth_years.min():.1f}, max: {node_growth_years.max():.1f}, "
              f"mean: {node_growth_years.mean():.1f}")
        print(f"Size distribution: {new_trees_node_df['size'].value_counts().to_dict()}")

    # =========================================================================
    # TURN-BASED PLANTINGS: Distribute uniformly across time window
    # =========================================================================
    if not new_trees_turn_df.empty:
        # Uniform random distribution of planting times across the window
        # Each tree planted at a random time, grows from then until now
        turn_growth_years = np.random.uniform(0, time_window, len(new_trees_turn_df))
        
        # Calculate DBH and update
        new_trees_turn_df['diameter_breast_height'] = initial_dbh + (growth_factor * turn_growth_years)
        new_trees_turn_df['useful_life_expectancy'] = 120 - turn_growth_years
        
        # Update size classification
        new_trees_turn_df['size'] = pd.cut(
            new_trees_turn_df['diameter_breast_height'],
            bins=[-10, 30, 80, float('inf')],
            labels=['small', 'medium', 'large']
        ).astype(str)
        
        print(f"\n--- TURN-BASED Plantings with Time-Distributed Growth ---")
        print(f"Time window: {previous_timestep} → {years_passed} = {time_window} years")
        print(f"Number of turn-based new trees: {len(new_trees_turn_df)}")
        print(f"Growth years - min: {turn_growth_years.min():.1f}, max: {turn_growth_years.max():.1f}, "
              f"mean: {turn_growth_years.mean():.1f}")
        print(f"Size distribution: {new_trees_turn_df['size'].value_counts().to_dict()}")

    # Combine the node-based and turn-based new trees DataFrames
    newTreesDF = pd.concat([new_trees_node_df, new_trees_turn_df], ignore_index=True)

    # Append new trees to original dataframe
    df = pd.concat([df, newTreesDF], ignore_index=True)

    print(f'\nTotal number of trees after planting: {len(df)}')

    return df

def assign_logs(logDF, params):
    if logDF is None:
        return None
    
    # Note: These values are pre-interpolated for sub-timesteps
    turnThreshold = params['sim_TurnsThreshold']
    resistanceThreshold = params.get('sim_averageResistance', 0)
    
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
    
    # Note: These values are pre-interpolated for sub-timesteps
    turnThreshold = params['sim_TurnsThreshold']
    resistanceThreshold = params.get('sim_averageResistance', 0)
    
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
    df = handle_replace_trees(df.copy(), params)

    print(f'handle REWILD/EXOSKELETON/FOOTPRINT-DEPAVED logic for AGE-IN-PLACE trees')
    df = assign_depaved_status(df.copy(), params)  # Assign the correct rewilding status

    print(f'Handle node-based rewilding')
    df, ds = calculate_rewilded_status(df.copy(), ds, params)

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
    # Get scenario parameters with interpolation for sub-timesteps
    params = a_scenario_params.get_params_for_year(site, scenario, year)
    
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