
####STEPS TO DO:


 




#DEFINE FUNCTIONS

#RUN SCENARIO FOR URBAN FOREST

#RUN SCENARIO FOR CITY


"""
Preprocesing:
#COMPLETE
In xarray:
#Create variable 'tree_CanopyID'. Get canopy footprints for all trees. (Trees include logs, trees, poles). This is the ID of the tree that is closest to the voxel. Get ID from the treeDF dataframe. Max distance 10m. If type of node is log or pole, then this is just all voxels within a radius of 5m.
#Create variable 'tree_CanopySize'. Size of tree that is the footprints. 
#Create variable 'tree_CanopyControl'. Control of tree that is the footprint. 
#Create variable 'tree_CanopyType'. Type that is the footprint. (logs, trees, pole)


#PREPROCESSING
In xarray:
#Create variable 'node_CanopyResistance'. Average resistance for canopy footprints.
#Create variable 'scenario_rewilded'. Initialise as false.

In treeDF:
#Create column canopyResistance for each tree. Get average resistance for canopy footprints.
#Create column 'nodeID'. This is the mapping of the nodeID to the treeID. TODO: do earlier in the pipeline (ie. in the assign logs)
#Create column 'nodeArea'. This is the area of the rewilded node. TODO: do earlier in the pipeline (ie. in the assign logs)

#Assign useful_life_expectancy for trees without one. 
    -if small, useful_life_expectancy = 80, 
    -if medium, useful_life_expectancy = 50,
    -if large, useful_life_expectancy = 10

#Create column 'Improvement'. Initialise as False.




#DEFINE FUNCTIONS

#RUN SCENARIO FOR URBAN FOREST

#RUN SCENARIO FOR CITY




GLOBAL VARIABLES
    *Year
    *resistanceThreshold
    *dieChance
    *collapseChance
    *PlantingDensity

ACTIONS

#AGE
    *DbhGrowthFactor
    - Increase 'DBH' by factor by how many years have passed
    - Update tree size (ie. DBH <30 is small, DBH 30-80 is medium, DBH >80 is large)
    - Update useful_life_expectancy

#SENESCE
    - Check useful_life_expectancy. If <=0:
        -if resistanceScore < resistanceThreshold:
            - Do AGEINPLACE
            - Do REMOVE_TARGET
        -else:
            - Do REPLACE

#AGEINPLACE
    - look up control
        if  street-tree:
            - Change tree 'control' to 'park-tree'
        if park-tree:
            - Change tree 'control' to 'reserve-tree' #TODO: Check this logic
    - look up size of tree in treeDF.
        if large
            Change tree 'Size' to 'senescing' 
        if Senesce. 
            Change tree 'Size' to 'fallen'
        
#REMOVE_TARGET
    *PlantThreshold
    - Look up treeID of tree. Convert voxels matching tree_CanopyID into Rewilded == TRUE
    - if tree_CanopyResistance is less than PlantThreshold, then do PLANT (if not, consider this an exoskeleton)

#REPLACE
    -Convert tree 'Size' to 'small', DBH to 10, isPrecolonial to 'True'

#IMPROVE:
    -set Improvement=True

#PLANT:
    -initilaise new row in treeDF
        -set DBH to 10
        -set Size to small
        -set isPrecolonial to True
        -set x,y,z. Use voxel centroid_x, centroid_y, centroid_z
        -set useful_life_expectancy to 100
        -set treeNumber (row index)
    -find closest voxel in xarray and add tree.

#REWILD(Tree, Pole, Log)
   -Convert all voxels with simResults matching the nodeID into Rewilded == TRUE
   if rewilded == Tree
        - Determine numberTreestoPlant. Use tree's nodeArea and the PlantingDensity variable to determine number of new trees to plant.
        - Get voxels where simResults == nodeID. Randomly choose numberTreestoPlant number of voxels. 
        - do PLANT for each new tree, using the x,y,z of the selectd voxels for the new trees location.


        
###### SCENARIO 1: Trimmed Parade

Year 1
•	Existing conditions. Humans manage elms intensively. Although they are old, humans remove senescing features such as dead and lateral branches.

#ACTION: NO modifications to site.

Year 5

Year 10
•	Elms senescing. Reduced managegment has allowed their dead, lateral branches to persist. 
•	Car spaces and pathways have been depabed. Plants now grow in such spaces. This removes the human targets under suchelms.
•	Most eucalypts are still young, with smooth bark on limbs, and vertical-oriented branches. 
•	Young eucalypts have also been planted alongside the dead elms.

#ACTION:
* set resistanceThreshold
# All trees AGE
# All trees SENESCE

Year 20
•	More elms are senscing. In other places, elms have now died. These are left standing. 
•	Increase in public transport means that more road space has been converted to habitat islands.
•	Some senscent elms are located in places where humans must be able to travel underneath. Exoskeletons support these elms. 
•	The urban forest now provides an abundant number of elevated, horizontal perches and dead branches.
•	Some elms have now collapsed. They are left in place to create coarse woody debris. 
•	More eucalypts planted alongside these collapsed trees.
•	Eucalypts still very young, with smooth bark on limbs, and vertical-oriented branches. 
•	The elms provide substrates for artificial bark and 3D printed hollows.

#ACTION:
# increase resistanceThreshold
# All trees AGE
# All trees SENESCE.

Year 50
•	Most elms have now died. Yet more elms have collapsed. These provide significant amounts of coarse woody debris. 
•	The eucalyptus are now middle aged. They still have smooth bark on limbs, vertically-oriented branches, and minimal natural hollows. 
•	Such trees now support many types of artificial bark, nesting platforms and hollows. 

ACTION:
# increase resistanceThreshold
# All trees AGE
# All trees SENESCE.
# Eucalypts IMPROVE TODO: define logic for this
# Select trees to REWILD: TODO:define logic for this

Year 175
•	The collapsed elms have all rotted away, enriching the soil
•	Lots of mature eucalypts. 

ACTION:
# All trees AGE
# All trees SENESCE.
# Select trees to REWILD: TODO:define logic for this
"""

#Scenario for the 'city' site
"""
Year 1
•	Existing conditions. Resource poor. Minimal locations for replantings. Building envelopes are kept hostile to life.
Year 10
•	Some laneways have been rewilded. The ground has been devpaved. Here, eucalypts have been planted. 

ACTION:
#Select nodes to REWILD: TODO:define logic for this

Year 20
•	More laneways rewilded. 
•	On roofs designed to bear heavier loads, trunks of fallen trees have been transported to create shelter. Here, a dense understory of plants thrives, offering nectar, seeds, and temporary ponds 
•	Conversely, roofs that accommodate only lighter loads host hot arid landscapes. Here, a mix of rocks, nurse logs (shown in orange), and dry vegetation (shown in yellow) creates diverse microclimates, some even simulating dry creek beds. Z square meters of such lightweight roofs have been converted. Here, some lizards bask in the sunlight, while others seek shade.
•	Porous building façades turn building volumes into walkable surfaces and lizard ladders. 
•	Epiphytes and other features such as rocks embedded in the walls, facilitating the ability of lizards to climb to the roofs above. 
•	Some car parking now also given over to habitat islands. 

ACTION:
#Select nodes to REWILD: TODO:define logic for this

Year 50
•	Although the eucs are still young, logs have been upcycled. 
•	Yet more roofs and envelopes. Free roaming lizards supported by abundance of log roofs, leaf piles, shrubs and water on other roofs 
•	More car parking, private yards have been converted to habitat islands. 
•	Young trees sprouted from the depaved ground

ACTION: 
#Select nodes to REWILD: TODO:define logic for this

Year 175
•	Many eucs old. Middle aged eucs also dot the area.
•	Almost all previous private vehicle space now rewilded. Tram tracks and small laneways support public transport and access. 
•	All roofs now green.
#ACTION:    
#Select nodes to REWILD: TODO:define logic for this
"""


#TODO: 
# - get average node resistance


from scipy.spatial import cKDTree
import pandas as pd
import xarray as xr
import numpy as np
import a_helper_functions, a_voxeliser
import pyvista as pv

######PREPROCESSING FUNCTIONS######

def getDataSubest(ds):
    attributes = ['voxel_size', 'bounds']
    variables = [
        'voxel_I',
        'voxel_J',
        'voxel_K',
        'centroid_x',
        'centroid_y',
        'centroid_z',
        'node_CanopyID',
        'node_CanopyResistance',
        'sim_Nodes',
        'sim_Turns',
        'analysis_combined_resistanceNOCANOPY',
        'analysis_potentialNOCANOPY',
        'analysis_combined_resistance',
        'analysis_busyRoadway',
        'analysis_Roadway',
        'site_building_element',
        'isTerrainUnderBuilding'
    ]

    subsetDS = a_helper_functions.create_subset_dataset(ds, variables, attributes)

    #subsetDS['analysis_forestLane'] = xr.full_like(subsetDS['node_CanopyID'], -1)

    print(subsetDS.attrs['bounds'])


    return subsetDS

def preprocess_logs(logDF, ds):
    """
    This function calculates the mean canopy resistance for logDF based on sim_Nodes clusters
    and assigns the results to the xarray dataset and the logDF DataFrame.

    Parameters:
    logDF (pandas.DataFrame): DataFrame containing log node information.
    ds (xarray.Dataset): The xarray dataset containing voxel data, including sim_Nodes and resistance.

    Returns:
    Updated ds and logDF with log-based resistance information.
    """

    ###THIS STUFF EVENTUALLY MOVED TO LOG DISTRIBUTOR
    logDF['x'] = ds['centroid_x'].values[logDF['voxelID'].astype(int)]
    logDF['y'] = ds['centroid_y'].values[logDF['voxelID'].astype(int)]
    logDF['z'] = ds['centroid_z'].values[logDF['voxelID'].astype(int)]

    ####
    
    
    # Initialize 'node_LogResistance' to -1 for all voxels in the dataset
    ds['node_LogResistance'] = xr.full_like(ds['sim_Nodes'], -1)

    logDF['voxel_Resistance'] = ds['analysis_combined_resistanceNOCANOPY'].values[logDF['voxelID'].astype(int)]
    logDF['voxel_sim_Turn'] = ds['sim_Turns'].values[logDF['voxelID'].astype(int)]

    # Get the roof mask: ds['site_building_element'] == 'roof'
    """roofMask = ds['site_building_element'] == 'roof'

    # Prepare the KDTree using voxel centroids from ds
    kdtree = cKDTree(np.vstack((ds['centroid_x'].values, ds['centroid_y'].values, ds['centroid_z'].values)).T)

    # Get the log node centroids as an array
    log_node_coords = np.vstack((logDF['x'], logDF['y'], logDF['z'])).T

    # Query the KDTree for all log node centroids, finding voxels within 10m
    distances, indices = kdtree.query(log_node_coords, distance_upper_bound=10)

    # Filter the indices for valid matches (within distance and roofMask is True)
    valid_indices = indices[distances < 10]
    valid_voxels = (roofMask.values[valid_indices]) & (ds['sim_Nodes'].values[valid_indices] == logDF['nodeID'].values[:, None])

    # Use vectorized grouping by sim_Nodes (log nodeID)
    mean_resistance = ds['analysis_combined_resistanceNOCANOPY'].values[valid_voxels].mean(axis=1)

    # Create a dictionary mapping nodeID to mean resistance
    log_resistance_dict = pd.Series(mean_resistance, index=logDF['nodeID']).to_dict()

    # Map the mean resistances to ds['node_LogResistance'] and logDF['node_CanopyResistance']
    ds['node_LogResistance'] = xr.apply_ufunc(
        lambda x: log_resistance_dict.get(x, -1),  # map or return -1 for unmapped values
        ds['sim_Nodes'],
        vectorize=True
    )
    
    logDF['node_CanopyResistance'] = logDF['nodeID'].map(log_resistance_dict)"""

    logDF['isEnabled'] = False

    return logDF, ds



# Initialize the 'size' and 'control' columns as strings at the beginning
def PreprocessData(treeDF, ds):
    # Convert 'size' and 'control' to strings
    print("Converting 'size' and 'control' columns to strings...")
    treeDF['size'] = treeDF['size'].astype(str)
    treeDF['control'] = treeDF['control'].astype(str)

    # Fill NaN values in 'diameter_breast_height' based on the 'size' column
    print("Checking for missing values in 'diameter_breast_height' and filling them based on 'size' column...")
    nan_before = treeDF['diameter_breast_height'].isna().sum()
    treeDF['diameter_breast_height'] = treeDF['diameter_breast_height'].fillna(
        treeDF['size'].map({'small': 10, 'medium': 40, 'large': 80})
    )
    nan_after = treeDF['diameter_breast_height'].isna().sum()

    # Print the number of NaNs filled
    print(f"Number of missing values before: {nan_before}")
    print(f"Number of missing values after: {nan_after}")
    print("Filling completed.")

    #initialse rewilding column as paved
    treeDF['rewilded'] = 'paved'
    treeDF['isNewTree'] = False
    treeDF['isRewildedTree'] = False
    treeDF['hasbeenReplanted'] = False
    treeDF['unmanagedCount'] = 0
    treeDF['action'] = 'None'

    ds['scenario_rewildingEnabled'] = xr.full_like(ds['node_CanopyID'], -1)
    ds['scenario_rewildingPlantings'] = xr.full_like(ds['node_CanopyID'], -1)
    underBuildingMask = ds['isTerrainUnderBuilding'] == True
    ds['sim_Turns'][underBuildingMask] = -1
    ds['sim_Nodes'][underBuildingMask] = -1
    
    return treeDF, ds


######SIMULATION FUNCTIONS######

# Function to age trees based on passed years
def age_trees(df, params):
    years_passed = params['years_passed']
    # Only update DBH for trees sized 'small', 'medium', or 'large'
    mask = df['size'].isin(['small', 'medium', 'large'])
    growth_factor_per_year = (params['growth_factor_range'][0] + params['growth_factor_range'][1])/2 #get mean
    growth = growth_factor_per_year * years_passed
    df.loc[mask, 'diameter_breast_height'] = df.loc[mask, 'diameter_breast_height'] + growth

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

def determine_ageinplace_or_replace2(df, params):
    # All trees where useful life expectancy is <= 0 need to perform actions
    senescent = df['useful_life_expectancy'] <= 0
    #print how many trees are senescing
    print(f'Number of trees marked as SENESCENT: {senescent.sum()}')
    
    # Decide whether to AGEINPLACE or REPLACE
    df.loc[senescent & (df['CanopyResistance'] < params['ageInPlaceThreshold']), 'action'] = 'AGE-IN-PLACE'
    df.loc[senescent & (df['CanopyResistance'] >= params['ageInPlaceThreshold']), 'action'] = 'REPLACE'
    #print number of trees aging in place and number that is replaced
    print(f'Number of trees aging in place: {df[df["action"] == "AGE-IN-PLACE"].shape[0]}')
    print(f'Number of trees replaced: {df[df["action"] == "REPLACE"].shape[0]}')
    
    return df

def remap_values_xarray(values, old_min, old_max, new_min, new_max):
    return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


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
    
    # Use pd.cut to assign rewilded status based on CanopyResistance and the given thresholds
    df.loc[mask, 'rewilded'] = pd.cut(df.loc[mask, 'CanopyResistance'],
                                      bins=[-float('inf'), params['rewildThreshold'], params['plantThreshold'], params['ageInPlaceThreshold'], float('inf')],
                                      labels=['node-rewilded', 'footprint-depaved', 'exoskeleton', 'None'])

    # Print breakdown of rewilded column for verification
    print(f'Number of trees rewilded as "node-rewilded": {df[df["rewilded"] == "node-rewilded"].shape[0]}')
    print(f'Number of trees rewilded as "footprint-depaved": {df[df["rewilded"] == "footprint-depaved"].shape[0]}')
    print(f'Number of trees rewilded as "exoskeleton": {df[df["rewilded"] == "exoskeleton"].shape[0]}')
    
    return df

from scipy.spatial import cKDTree


def assign_rewilded_status(df, ds, params):

    if site == 'trimmed-parade':
        # Mask for sim_Nodes exceeding rewilding threshold (now called depavedMask)
        depaved_threshold = params['rewildThresholdNEW'][params['years_passed']]['sim_Turns']

        #ie, get the dictionary of rewildThresholdNEW for this yeare, and choose the simTurns value
        depaved_mask = (ds['sim_Turns'] <= depaved_threshold) & (ds['sim_Turns'] >= 0)
        print(f'Number of voxels where depaved threshold of {depaved_threshold} is satisfied: {depaved_mask.sum().item()}')
        
        # Terrain mask to exclude 'facade' and 'roof' elements
        terrain_mask = (ds['site_building_element'] != 'facade') & (ds['site_building_element'] != 'roof')

        print(f'Number of voxels where terrain mask is satisfied: {terrain_mask.sum().item()}')
        
        # Combined mask to filter relevant points for proximity check
        combined_mask = depaved_mask & terrain_mask

    elif site == 'city':
        print(params)
        
        turn_threshold = params['rewildThresholdNEW'][params['years_passed']]['sim_Turns']
        resistanceThreshold = params['rewildThresholdNEW'][params['years_passed']]['sim_averageResistance']

        combined_mask = (ds['sim_Turns'] >= 0) & (ds['sim_Turns'] <= turn_threshold) & (ds['sim_averageResistance'] <= resistanceThreshold)

    #Save combined mask to ds
    ds['scenario_rewildingEnabled'][combined_mask] = params['years_passed']

    print(f'Number of voxels where rewilding is enabled: {ds["scenario_rewildingEnabled"].sum().item()}')
    
    # Filter relevant voxel positions (centroid_x, centroid_y, centroid_z) using the combined mask
    voxel_positions = np.vstack([
        ds['centroid_x'].values[combined_mask], 
        ds['centroid_y'].values[combined_mask], 
        ds['centroid_z'].values[combined_mask]
    ]).T
    
    # Get the corresponding voxel IDs based on the same mask
    filtered_voxel_ids = ds['voxel'].values[combined_mask]
    
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

    print(f'Number of voxels where rewilding plantings are enabled: {ds["scenario_rewildingPlantings"].sum().item()}')
    
    return df, ds

#function to handle REWILD/FOOTPRINT-DEPAVED logic for other trees
def reduce_control_of_trees(df, params):
    print(f'reduce control of trees')
    nonSenescentMask = df['useful_life_expectancy'] > 0

    # Decide whether to REWILD or not based on params
    df.loc[nonSenescentMask, 'rewilded'] = pd.cut(df.loc[nonSenescentMask, 'CanopyResistance'],
                                      bins=[-float('inf'), params['rewildThreshold'], params['plantThreshold'], float('inf')],
                                      labels=['node-rewilded', 'footprint-depaved', 'None'])
    
    # TODO: consider logic for replanting where there are small and medium sized trees, do we really want this?

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
    ##DETERMINE NODE BASED PLANTING LOGIC

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

    #only apply to non building elements
    terrain_mask = (ds['site_building_element'] != 'facade') & (ds['site_building_element'] != 'roof')
    mask = mask & terrain_mask

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
    turn_based_tree_data['x'] = selected_positions[:, 0]
    turn_based_tree_data['y'] = selected_positions[:, 1]
    turn_based_tree_data['z'] = selected_positions[:, 2]

    # Create the new DataFrame for turn-based trees
    new_trees_turn_df = pd.DataFrame(turn_based_tree_data)

    # Print turn-based planting details
    print(f'Turn-based number of new trees added: {len(new_trees_turn_df)}')

    ##END OF TURN_BASED LOGIC#####

    # Repeat the rows based on 'number_of_trees_to_plant' for node-based logic and create a new DataFrame
    repeated_indices = np.repeat(df[to_plant_mask].index, df.loc[to_plant_mask, 'number_of_trees_to_plant'].astype(int))
    new_trees_node_df = df.loc[repeated_indices].copy()

    # Update node-based new trees' specific attributes
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
    print(f'Average CanopyArea for node-based new trees: {new_trees_node_df["CanopyArea"].mean()}')


        # Combine the node-based and turn-based new trees DataFrames
    newTreesDF = pd.concat([new_trees_node_df, new_trees_turn_df], ignore_index=True)

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
    bin_size = total_trees // n_bins  # Integer division

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

def assign_logs(logDF, ds, params):
    # Assign logs to the trees in logDF
    resistanceThreshold = params['rewildThresholdNEW'][params['years_passed']]['sim_averageResistance']
    turnThreshold = params['rewildThresholdNEW'][params['years_passed']]['sim_Turns']

    # Print the thresholds for debugging
    print(f"Resistance Threshold: {resistanceThreshold}")
    print(f"Turn Threshold: {turnThreshold}")

    # Create a mask for the logs that are below the resistance and turn thresholds
    mask = (logDF['voxel_Resistance'] <= resistanceThreshold) & (logDF['voxel_sim_Turn'] <= turnThreshold)

    # Assign the mask to the logDF
    logDF['isEnabled'] = mask

    # Print the number of logs enabled this turn
    enabled_logs_count = logDF['isEnabled'].sum()
    print(f"Number of logs enabled this turn: {enabled_logs_count}")

    # Print the value counts of the enabled logDF['logSize']
    enabled_log_size_counts = logDF[logDF['isEnabled']]['logSize'].value_counts()
    print(f"Value counts of enabled log sizes:\n{enabled_log_size_counts}")

    return logDF


def check_for_missing_values(df, column_names):
    for column in column_names:
        missing_values = df[column].isna().sum()
        if missing_values > 0:
            print(f"Missing values in {column}: {missing_values}")
        else:
            print(f"No missing values in {column}")


# Simulate the scenario for Year 10
def run_simulation(df, ds, params, logDF=None):
    
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

    print(f'handle LOG logic')
    if logDF is not None:
        logDF = assign_logs(logDF.copy(), ds, params)

    print(f'add structureID column')
    #this is just the index
    df['structureID'] = df.index

    return df, logDF

###UPDATE FUNCTIONS #####


def update_scenario_rewilded(ds, df):
    """
    Updates the 'scenario_rewilded' variable in the xarray dataset based on the dataframe values.
    Matches are made based on NodeID. Non-matching NodeIDs are ignored.
    
    Parameters:
    ds (xarray.Dataset): The xarray dataset containing voxel and node information.
    df (pandas.DataFrame): The dataframe containing NodeID and rewilded scenarios.
    
    Returns:
    xarray.Dataset: The updated dataset with the 'scenario_rewilded' variable modified.
    """
    # Step 1: Initialize 'scenario_rewilded' if it doesn't exist
    if 'scenario_rewilded' not in ds.variables:
        ds = ds.assign(scenario_rewilded=('voxel', ['none'] * ds.dims['voxel']))
    
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
    
    # Step 4: Return the updated dataset
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


    #extract 


    #fallen dead branches






######PLOTTING FUNCTIONS######
import pyvista as pv

import pandas as pd
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Essential for color conversions

def label_trees(df, plotter):
    # Prepare points and labels from the filtered subset
        TARGET_SIZES_FOR_LABELS = ['large', 'senescing', 'snag', 'fallen']
        label_df = df[df['size'].isin(TARGET_SIZES_FOR_LABELS)]
        label_points = label_df[['x', 'y', 'z']].values
        label_points[:,2] = label_points[:,2] + 10 #give some elevation to the labels so they are easier to see
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


import copy

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
     #create a mask that checks all polydata.point_data variables starting with resource, execpt for resource_leaf_litter. 
    # If any of the resource variables are >0, mask is True, else False
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

    # Split the polydata based on the mask

    polydata.point_data['maskforTrees'] = maskforTrees
    maskForRewilding = polydata['scenario_rewilded'] != 'none'
    polydata.point_data['maskForRewilding'] = maskForRewilding

    scenario_outputs = np.full(polydata.n_points, 'none', dtype='<U20')
    scenario_outputs[maskForRewilding] = polydata.point_data['scenario_rewilded'][maskForRewilding]
    scenario_outputs[maskforTrees] = polydata.point_data['forest_size'][maskforTrees]
    #print unique values and counts for scenario_outputs
    print(f'unique values and counts for scenario_outputs: {pd.Series(scenario_outputs).value_counts()}')
    polydata.point_data['scenario_outputs'] = scenario_outputs



    
    print(f'unique values and counts for scenario_outputs in polydata: {pd.Series(polydata.point_data["scenario_outputs"]).value_counts()}')
    return polydata

def plot_scenario_details(ds, years):
    polydata = a_helper_functions.convert_xarray_into_polydata(ds)


    # Get mask where polydata['forest_tree_id'] is NaN
    siteMask = np.isnan(polydata['forest_tree_id'])
    sitePoly = polydata.extract_points(siteMask)
    
    # Create the mask for trees
    maskforTrees = np.zeros(polydata.n_points, dtype=bool)
    for key in polydata.point_data.keys():
        if key.startswith('resource_') and key != 'resource_leaf litter':
            resource_data = polydata.point_data[key]
            resource_mask = resource_data > 0
            maskforTrees = np.logical_or(maskforTrees, resource_mask)

    polydata.point_data['maskforTrees'] = maskforTrees
    treePoly = polydata.extract_points(maskforTrees)

    # Deep copy the treePoly for each scalar we want to plot
    treePoly_size = copy.deepcopy(treePoly)
    treePoly_diameter = copy.deepcopy(treePoly)
    treePoly_life_expectancy = copy.deepcopy(treePoly)
    treePoly_precolonial = copy.deepcopy(treePoly)

    #plot bounds of polydata
    bounds = polydata.bounds
    print(f'bounds of polydata: {bounds}')

    # Create the plotter with 2 rows and 3 columns
    plotter = pv.Plotter(shape=(2, 3))
    
    # Add a title based on the site and years passed
    plotter.add_text(f"Scenario at {site} after {years} years", position="upper_edge", font_size=14, color='black')

    # Plot 1: Tree size, with eye-dome lighting (EDL)
    plotter.subplot(0, 0)
    plotter.add_text("Size", position="upper_edge", font_size=10, color='black')
    plotter.add_mesh(treePoly_size, scalars='forest_size', cmap='Set1', show_scalar_bar=True)
    plotter.enable_eye_dome_lighting()

    # Plot 2: Diameter at breast height
    plotter.subplot(0, 1)
    plotter.add_text("Diameter Breast Height", position="upper_edge", font_size=10, color='black')
    plotter.add_mesh(treePoly_diameter, scalars='forest_diameter_breast_height', cmap='tab20b', show_scalar_bar=True)

    # Plot 3: Useful life expectancy
    plotter.subplot(1, 0)
    plotter.add_text("Useful Life Expectancy", position="upper_edge", font_size=10, color='black')
    plotter.add_mesh(treePoly_life_expectancy, scalars='forest_useful_life_expectancy', cmap='tab20c', show_scalar_bar=True)

    # Plot 4: Precolonial forest status
    plotter.subplot(1, 1)
    plotter.add_text("Precolonial Status", position="upper_edge", font_size=10, color='black')
    plotter.add_mesh(treePoly_precolonial, scalars='forest_precolonial', cmap='viridis', show_scalar_bar=True)
    plotter.link_views() 

    # Plot 5: new Plantings
    plotter.subplot(0, 2)
    treePoly_newPlantings = copy.deepcopy(treePoly)
    plotter.add_text("New Plantings Status", position="upper_edge", font_size=10, color='black')
    plotter.add_mesh(treePoly_newPlantings, scalars='forest_precolonial', cmap='viridis', show_scalar_bar=True)
    plotter.add_mesh(sitePoly, scalars = 'scenario_rewildingPlantings', clim = [0,5000], below_color = 'white', cmap = 'Set2', show_scalar_bar = True)

    # Plot 6: Rotations
    plotter.subplot(1, 2)
    treePoly_Rotations = copy.deepcopy(treePoly)
    plotter.add_text("Rotations", position="upper_edge", font_size=10, color='black')
    plotter.add_mesh(treePoly_newPlantings, scalars='forest_rotateZ', cmap='viridis', show_scalar_bar=True)
    plotter.add_mesh(sitePoly, scalars = 'scenario_rewildingPlantings', clim = [0,5000], below_color = 'white', cmap = 'Set2', show_scalar_bar = True)




    plotter.link_views() 



    # Show all subplots
    plotter.show()
    plotter.close()


def plot_scenario_rewilded(polydata, treeDF, years_passed):
    #print unique values of forest_tree_id

    #split polydata into 2 based on point_data['forest_tree_id']. nan values are sitePoly, others are treePoly

    maskforTrees = polydata.point_data['maskforTrees']
    maskForRewilding = polydata.point_data['maskForRewilding']


    


    sitePoly = polydata.extract_points(~maskforTrees)  # Points where forest_tree_id is NaN
    treePoly = polydata.extract_points(maskforTrees)  # Points where forest_tree_id is not NaN

    # Extract two different polydata based on the masks
    rewildingVoxels = sitePoly.extract_points(maskForRewilding)
    siteVoxels = sitePoly.extract_points(~maskForRewilding)

    #print all point_data variables in polydata
    print(f'point_data variables in polydata: {sitePoly.point_data.keys()}')

    # Extract indices for 'none' and non-'none' values


 

    

    # Create the plotter
    plotter = pv.Plotter()
    # Add title to the plotter
    plotter.add_text(f"Scenario at {site} after {years_passed} years", position="upper_edge", font_size=16, color='black')

    label_trees(treeDF, plotter)

    # Add 'none' points as white
    plotter.add_mesh(siteVoxels, color='white')
    plotter.add_mesh(treePoly, scalars='forest_size', cmap = 'Set1')

    # Add other points with scalar-based coloring
    plotter.add_mesh(rewildingVoxels, scalars='scenario_rewilded', cmap='Set2', show_scalar_bar=True)
    plotter.enable_eye_dome_lighting()


    plotter.show()
    plotter.close()


# Call the function
#plot_scenario_rewilded(ds,treeDF_final)

import pandas as pd
import pyvista as pv
import numpy as np


def process_scenarios(site, voxel_size, isLog = False):

    params_trimmed_parade = {
    'growth_factor_range' : [0.37, 0.51], #growth factor is a range
    'plantingDensity' : 50, #10 per hectare
    'ageInPlaceThreshold' : 70,
    'plantThreshold' : 50,
    'rewildThreshold' : 10,
    'senescingThreshold' : -25,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'rewildThresholdNEW' : {
                            0: {'sim_averageResistance': 0, 'sim_Turns': 0},
                            10: {'sim_averageResistance': 0, 'sim_Turns': 0},
                            30: {'sim_averageResistance': 0, 'sim_Turns': 3000},
                            60: {'sim_averageResistance': 0, 'sim_Turns': 4000},
                            180: {'sim_averageResistance': 0, 'sim_Turns': 4500}
                           }
    }

    params_city = {
    'growth_factor_range' : [0.37, 0.51], #growth factor is a range
    'plantingDensity' : 50, #10 per hectare
    'ageInPlaceThreshold' : 70,
    'plantThreshold' : 50,
    'rewildThreshold' : 10,
    'senescingThreshold' : -25,
    'snagThreshold' : -200,
    'collapsedThreshold' : -250,
    'controlSteps' : 20,
    'rewildThresholdNEW' : {
                            0: {'sim_averageResistance': 0, 'sim_Turns': 0},
                            10: {'sim_averageResistance': 50, 'sim_Turns': 300},
                            30: {'sim_averageResistance': 50, 'sim_Turns': 1249.75},
                            60: {'sim_averageResistance': 67.90487670898438, 'sim_Turns': 4999},
                            180: {'sim_averageResistance': 96, 'sim_Turns': 5000}
                           }
    }

    paramDic = {'trimmed-parade' : params_trimmed_parade,
                'city' : params_city}
    
    if site == 'city':
        isLog = True
    else:
        isLog = False   

    print(f'isLog: {isLog}')

    #query the user to enter a list of years passed or press return to run all years
    all_years = [0, 10, 30, 60, 180]
    years_passed = input(f"Please enter a list of years passed or press return to run all years: {all_years}")
    if years_passed == '':
        years_passed = all_years
        print(f'running all years: {years_passed}')
    else:
        years_passed = [int(year) for year in years_passed.split(',')]
        print(f'running years: {years_passed}')

    #treeDf, poleDF, logDF
    input_folder = f'data/revised/final/{site}'
    filepath = f'{input_folder}/{site}_{voxel_size}_voxelArray_RewildingNodes.nc'
    xarray_dataset = xr.open_dataset(filepath)

    #print all attributes in xarray_dataset
    print(xarray_dataset.attrs)
    subsetDS = getDataSubest(xarray_dataset)
    subsetDS.to_netcdf(f'{input_folder}/{site}_{voxel_size}_subsetForScenarios.nc')
    print('loaded and subsetted xarray data')

    #loading nodes
    filePATH = f'data/revised/final/{site}'
    treeDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_treeDF.csv')
    poleDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_poleDF.csv')
    logDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_logDF.csv')

    print('preprocessing treeDF')
    # Apply the initialization at the start of the script
    treeDF, subsetDS = PreprocessData(treeDF, subsetDS)

    if isLog:
        #preprocess logs
        logDF, ds = preprocess_logs(logDF, subsetDS)
        #poly = a_helper_functions.convert_xarray_into_polydata(ds)
        #poly.plot(scalars = 'node_LogResistance', cmap = 'viridis', show_scalar_bar = True)


    for year in years_passed:
        print(f'running simulation for year {year}')
        scenarioParams = paramDic[site]
        scenarioParams['years_passed'] = year

        if isLog:
            treeDF_scenario = run_simulation(treeDF.copy(), subsetDS, scenarioParams, logDF)
        else:
            treeDF_scenario = run_simulation(treeDF.copy(), subsetDS, scenarioParams)

        print(f'done running simulation for year {year}')
        
        print('integrating results into xarray')
        ds = update_scenario_rewilded(subsetDS, treeDF_scenario)
        # Create a mask for valid points
        validpointsMask = ds['scenario_rewilded'] != 'none'

        # Extract valid points as a numpy array
        valid_points = np.array([
            ds['centroid_x'].values[validpointsMask],
            ds['centroid_y'].values[validpointsMask],
            ds['centroid_z'].values[validpointsMask]
        ]).T

        ds, treeDF_scenario = a_voxeliser.integrate_resources_into_xarray(ds, treeDF_scenario, valid_points)
        
        ds = finalDSprocessing(ds)

        #save treeDF_scenario to csv
        print(f'saving {year} treeDF_scenario to csv')
        treeDF_scenario.to_csv(f'{filePATH}/{site}_{voxel_size}_treeDF_{year}.csv', index=False)
        print_simulation_statistics(treeDF_scenario, year, site)

        polydata = a_helper_functions.convert_xarray_into_polydata(ds)
        polydata = process_polydata(polydata)
        polydata.save(f'{filePATH}/{site}_{voxel_size}_scenarioYR{year}.vtk')

        #plot_scenario_rewilded(polydata, treeDF_scenario, year)
        #plot_scenario_details(ds, year)

        # Display the updated dataframe and summary stats
        #print(summary_stats)

if __name__ == "__main__":
    #sites = ['trimmed-parade']
    sites = ['city']
    voxel_size = 1
    for site in sites:
        process_scenarios(site, voxel_size)


