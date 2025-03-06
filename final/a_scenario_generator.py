import os
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
        'analysis_combined_resistance',
        'analysis_potentialNOCANOPY',
        'analysis_busyRoadway',
        'analysis_Roadway',
        'site_building_element',
        'isTerrainUnderBuilding',
        'analysis_nodeID',
        'analysis_nodeType',
        'envelope_roofType'
    ]

    subsetDS = a_helper_functions.create_subset_dataset(ds, variables, attributes)

    print(subsetDS.variables)

    subsetDS['envelopeIsBrownRoof'] = xr.full_like(subsetDS['node_CanopyID'], -1)
    brownRoofMask = subsetDS['envelope_roofType'] == 'brown roof'
    subsetDS['envelopeIsBrownRoof'][brownRoofMask] = 1
    print(subsetDS.attrs['bounds'])


    return subsetDS

def further_xarray_processing(ds):
    """
    This function creates new variables in the xarray: 'sim_nodeType' and 'sim_averageResistance'.
    
    'sim_nodeType' is assigned based on 'analysis_nodeID' and 'analysis_nodeType'.
    'sim_averageResistance' calculates the mean of 'analysis_combined_resistanceNOCANOPY' 
    for nodes that satisfy specific conditions.
    
    Parameters:
    ds (xarray.Dataset): The xarray dataset containing 'sim_node', 'analysis_nodeID', 'analysis_nodeType', and other relevant fields.
    
    Returns:
    ds (xarray.Dataset): Updated dataset with the new 'sim_nodeType' and 'sim_averageResistance' variables.
    """
    # Create a mapping from 'analysis_nodeID' to 'analysis_nodeType'
    node_type_mapping = dict(zip(ds['analysis_nodeID'].values, ds['analysis_nodeType'].values))

    # Assign 'sim_nodeType' by mapping 'sim_Nodes' to the corresponding 'analysis_nodeType'
    ds['sim_nodeType'] = xr.apply_ufunc(
        lambda x: node_type_mapping.get(x, -1),  # Use -1 for unmapped nodes
        ds['sim_Nodes'],
        vectorize=True
    )

    # Print unique values of 'sim_Nodes'
    unique_sim_nodes = ds['sim_Nodes'].values
    print("Unique values in 'sim_Nodes':", np.unique(unique_sim_nodes))

    # Create a new variable in the xarray called 'sim_averageResistance'
    ds['sim_averageResistance'] = xr.full_like(ds['sim_Nodes'], fill_value=-1, dtype=np.float32)

    # Get the mask for voxels where 'sim_Turns' < 50 and 'isTerrainUnderBuilding' is False
    valid_voxel_mask = (ds['sim_Turns'] < 50) & (ds['isTerrainUnderBuilding'] == False)

    # Remove invalid 'sim_Nodes' (-1 and NaN)
    valid_nodes_mask = (ds['sim_Nodes'] != -1) & (~ds['sim_Nodes'].isnull())

    # Apply both masks
    combined_mask = valid_voxel_mask & valid_nodes_mask

    # Convert necessary variables to a pandas DataFrame for easy manipulation
    df = ds[['sim_Nodes', 'analysis_combined_resistanceNOCANOPY']].where(combined_mask, drop=True).to_dataframe()

    # Group by 'sim_Nodes' and calculate the mean of 'analysis_combined_resistanceNOCANOPY'
    df_avg_resistance = df.groupby('sim_Nodes')['analysis_combined_resistanceNOCANOPY'].mean()

    # Debugging step: check the result of the mean calculation
    print("Node Average Resistance DataFrame:")
    print(df_avg_resistance)

    # Assign the calculated mean back to the xarray dataset for each 'sim_Nodes'
    ds['sim_averageResistance'] = xr.DataArray(
        ds['sim_Nodes'].to_series().map(df_avg_resistance).fillna(-1).values, 
        dims=ds['sim_Nodes'].dims, 
        coords=ds['sim_Nodes'].coords
    )

    # Print unique values and counts for 'sim_averageResistance'
    unique_resistances, counts = np.unique(ds['sim_averageResistance'].values, return_counts=True)
    print("Unique values in 'sim_averageResistance':")
    for value, count in zip(unique_resistances, counts):
        print(f"Value: {value}, Count: {count}")


    # Visualization and plotting
    poly = a_helper_functions.convert_xarray_into_polydata(ds)


    # Create masks for the conditions
    """turns_mask = poly.point_data['sim_Turns'] < 50
    resistance_mask = poly.point_data['sim_averageResistance'] < 10
    node_mask = poly.point_data['sim_Nodes'] != -1
    
    # Combine the masks
    combined_mask = np.logical_and(turns_mask, resistance_mask, node_mask)
    
    # Create a subPoly using the combined mask
    subPoly = poly.extract_points(combined_mask)
    rest = poly.extract_points(~combined_mask)
    
    plotter = pv.Plotter()
    plotter.add_mesh(poly, scalars='sim_Turns', clim=[0, 5000], below_color='white', cmap='viridis', show_scalar_bar=True)

    plotter.add_mesh(subPoly, scalars='analysis_combined_resistanceNOCANOPY', clim=[0, 100], below_color='white', cmap='viridis', show_scalar_bar=True)
    plotter.add_mesh(rest, color='white')
    plotter.enable_eye_dome_lighting()
    plotter.show()"""

    return ds, poly






def further_xarray_processing2(ds):
    """
    This function creates a new variable in the xarray called 'sim_nodeType'.
    It assigns the 'analysis_nodeType' to each 'sim_node' by matching 'analysis_nodeID'.
    
    Parameters:
    ds (xarray.Dataset): The xarray dataset containing 'sim_node', 'analysis_nodeID', and 'analysis_nodeType'.

    Returns:
    ds (xarray.Dataset): Updated dataset with the new 'sim_nodeType' variable.
    """
     #create a new variable in the xarray called 'sim_nodeType'.
    #to do so, use the 'analysis_nodeID' and 'analysis_nodeType' variables.
    #the logic is as follows: for each sim_node, find the corresponding analysis_nodeID. 
    #then, assign the analysis_nodeType to sim_nodeType.

    # Create a mapping from 'analysis_nodeID' to 'analysis_nodeType'
    node_type_mapping = dict(zip(ds['analysis_nodeID'].values, ds['analysis_nodeType'].values))

    # Assign 'sim_nodeType' by mapping 'sim_node' to the corresponding 'analysis_nodeType'
    ds['sim_nodeType'] = xr.apply_ufunc(
        lambda x: node_type_mapping.get(x, -1),  # Use -1 for unmapped nodes
        ds['sim_Nodes'],
        vectorize=True
    )

    #create a new variable in the xarray called 'sim_averageResistance'
    ds['sim_averageResistance'] = xr.full_like(ds['sim_Nodes'], fill_value=-1, dtype=np.float32)

    #get the mask of the voxels that have sim_Turns < 50
    #get the mask of voxels that are not under buildings 'isTerrainUnderBuilding' is False

    #apply both masks
    #group by sim_Nodes
    #calculate the mean of analysis_combined_resistanceNOCANOPY for each sim_Nodes group that satisfies the masks
    #assign the mean to sim_averageResistance for each sim_Nodes, including those that do not satisfy the masks

    return ds

import numpy as np
import pandas as pd
import xarray as xr

def xarray_value_counts(da):
    """
    Performs value counts on an xarray DataArray.
    
    Parameters:
    da (xarray.DataArray): The DataArray to perform value counts on.
    
    Returns:
    pandas.Series: A series with unique values and their counts.
    """
    unique, counts = np.unique(da.values, return_counts=True)
    return pd.Series(counts, index=unique).sort_values(ascending=False)



def log_processing(logDF, ds):
    #get the centroid_x, centroid_y, centroid_z from the xarray dataset and make a ckdtree
    

    
    logDF['x'] = ds['centroid_x'].values[logDF['voxelID']]
    logDF['y'] = ds['centroid_y'].values[logDF['voxelID']]
    logDF['z'] = ds['centroid_z'].values[logDF['voxelID']]

    logDF['sim_averageResistance'] = ds['sim_averageResistance'].values[logDF['voxelID']]
    logDF['sim_Turns'] = ds['sim_Turns'].values[logDF['voxelID']]

    

    return logDF

def pole_processing(poleDF, extraPoleDF, ds):
    
    if extraPoleDF is not None:
        print(f'columns in poleDF: {poleDF.columns}')
        print(f'columns in extraPoleDF: {extraPoleDF.columns}')

        poleDF['sim_averageResistance'] = ds['sim_averageResistance'].values[poleDF['voxel_index']]
        print(f'adding {extraPoleDF.shape[0]} extra poles')
        # Add missing columns to extraPoleDF
        extraPoleDF['eastings'] = -1
        extraPoleDF['northings'] = -1
        extraPoleDF['poletype'] = 'artificial'
        extraPoleDF['isArtificial'] = True
        extraPoleDF['pole_number'] = extraPoleDF.index + len(poleDF)
        extraPoleDF['Improvement'] = False
        extraPoleDF['NodeID'] = extraPoleDF['pole_number']
        extraPoleDF['debugNodeID'] = extraPoleDF['pole_number']
        extraPoleDF['sim_NodesVoxels'] = 1
        extraPoleDF['sim_NodesArea'] = 1

        # Get voxel indices for extraPoleDF
        all_points = np.vstack((ds['centroid_x'].values, ds['centroid_y'].values, ds['centroid_z'].values)).T
        tree = cKDTree(all_points)
        extraPoleDF['voxel_index'] = tree.query(np.vstack((extraPoleDF['x'].values, extraPoleDF['y'].values, extraPoleDF['z'].values)).T)[1]
        extraPoleDF['sim_averageResistance'] = ds['analysis_combined_resistance'].values[extraPoleDF['voxel_index']]

        print(f'value counts of extraPoleDF["sim_averageResistance"]: {extraPoleDF["sim_averageResistance"].value_counts()}')

        # Combine with poleDF
        poleDF = pd.concat([poleDF, extraPoleDF], ignore_index=True)


    poleDF['sim_Turns'] = ds['sim_Turns'].values[poleDF['voxel_index']]
    poleDF['control'] = 'reserve-tree'
    poleDF['size'] = 'artificial'
    poleDF['precolonial'] = False
    poleDF['tree_id'] = -1
    poleDF['diameter_breast_height'] = 80
    poleDF['tree_number'] = poleDF.index
    poleDF['useful_life_expectancy'] = -1




    #if sim_averageResistance is -1, set to poleDF['sim_averageResistance'] = ds['analysis_combined_resistance'].values[poleDF['voxel_index']]
    #check if sim_averageResistance is in poleDF, if not, set to -1
    if 'sim_averageResistance' not in poleDF.columns:
        poleDF['sim_averageResistance'] = -1

    mask = poleDF['sim_averageResistance'] == -1
    poleDF.loc[mask, 'sim_averageResistance'] = ds['analysis_combined_resistance'].values[poleDF.loc[mask, 'voxel_index']]

    print(f'value counts of combined poleDF["sim_averageResistance"]: {poleDF["sim_averageResistance"].value_counts()}')
    
    return poleDF

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree



# Initialize the 'size' and 'control' columns as strings at the beginning
def PreprocessData(treeDF, ds, extraTreeDF):
    if extraTreeDF is not None:
        print(f'n number of extra trees: {extraTreeDF.shape[0]}')
        extraTreeDF['diameter_breast_height'] = np.nan
        extraTreeDF['age_description'] = np.nan
        extraTreeDF['eastings'] = -1
        extraTreeDF['northings'] = -1

        extraTreeDF['tree_id'] = -1
        extraTreeDF['tree_number'] = extraTreeDF.index + treeDF.shape[0]
        extraTreeDF['Improvement'] = False
        extraTreeDF['NodeID'] = extraTreeDF['tree_number']
        extraTreeDF['debugNodeID'] = extraTreeDF['tree_number']

        #get the voxelIndex
        #extract the centroid_x, centroid_y, centroid_z from the xarray dataset and construct a ckdtree
        all_points = np.vstack((ds['centroid_x'].values, ds['centroid_y'].values, ds['centroid_z'].values)).T
        tree = cKDTree(all_points)

        #query the tree to get the nearest point and assign the index to extraTreeDF['voxelID']
        extraTreeDF['voxel_index'] = tree.query(np.vstack((extraTreeDF['x'].values, extraTreeDF['y'].values, extraTreeDF['z'].values)).T)[1]


        extraTreeDF['CanopyResistance'] = ds['analysis_combined_resistance'].values[extraTreeDF['voxel_index']]


        #get the sim_averageResistance from the xarray dataset and assign to extraTreeDF['CanopyResistance']
        extraTreeDF['CanopyResistance'] = ds['analysis_combined_resistance'].values[extraTreeDF['voxel_index']]

        extraTreeDF['CanopyResistance'] = np.nan
        extraTreeDF['CanopyArea'] = 1
        extraTreeDF['sim_NodesVoxels'] = 1
        extraTreeDF['sim_NodesArea'] = 1

         # Debug voxel indices
        print("Voxel index range:", extraTreeDF['voxel_index'].min(), "to", extraTreeDF['voxel_index'].max())
        print("Dataset size:", len(ds['analysis_combined_resistance']))
        
        # Debug resistance values
        resistance_values = ds['analysis_combined_resistance'].values[extraTreeDF['voxel_index']]
        print("Resistance values shape:", resistance_values.shape)
        print("Number of NaN resistance values:", np.isnan(resistance_values).sum())
        print("Resistance value range:", np.nanmin(resistance_values), "to", np.nanmax(resistance_values))
        
        extraTreeDF['CanopyResistance'] = resistance_values
        
        # Set useful_life_expectancy based on size
        extraTreeDF['useful_life_expectency'] = extraTreeDF['size'].map({
            'small': 80, 
            'medium': 50, 
            'large': 10
        })
        
        # After assignment, verify no NaNs
        print("Final NaN count in CanopyResistance:", extraTreeDF['CanopyResistance'].isna().sum())

        print('first 5 rows of extraTreeDF:')
        print(extraTreeDF.head())

        #combine treeDF and extraTreeDF
        treeDF = pd.concat([treeDF, extraTreeDF], ignore_index=True)
        
    #check if useful_life_expectency is in treeDF and rename to useful_life_expectancy
    if 'useful_life_expectency' in treeDF.columns:
        treeDF.rename(columns={'useful_life_expectency': 'useful_life_expectancy'}, inplace=True)
    
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

from scipy.spatial import cKDTree


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

    print(f'Number of non-senescent trees: {nonSenescentMask.sum()}')

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


def assign_logs(logDF, params):
    turnThreshold = params['sim_TurnsThreshold'][params['years_passed']]
    resistanceThreshold = params['sim_averageResistance'][params['years_passed']]

    mask = (logDF['sim_averageResistance'] <= resistanceThreshold) & (logDF['sim_Turns'] <= turnThreshold)

    # Assign the mask to the logDF
    logDF['isEnabled'] = mask

    # Print the number of logs enabled this turn
    enabled_logs_count = logDF['isEnabled'].sum()
    print(f"Number of logs enabled this turn: {enabled_logs_count}")

    # Print the value counts of the enabled logDF['logSize']
    enabled_log_size_counts = logDF[logDF['isEnabled']]['logSize'].value_counts()
    print(f"Value counts of enabled log sizes:\n{enabled_log_size_counts}")

    #extract the logDF that is enabled
    enabled_logDF = logDF[logDF['isEnabled']]

    return enabled_logDF


def assign_poles(poleDF, params):
    turnThreshold = params['sim_TurnsThreshold'][params['years_passed']]
    resistanceThreshold = params['sim_averageResistance'][params['years_passed']]

    
    print(f'resistanceThreshold: {resistanceThreshold}')
    #mask = (poleDF['sim_averageResistance'] <= resistanceThreshold) & (poleDF['sim_Turns'] <= turnThreshold)

    mask = poleDF['sim_averageResistance'] < resistanceThreshold

    print(f'value counts of poleDF["sim_averageResistance"]: {poleDF["sim_averageResistance"].value_counts()}'
          )

    # Assign the mask to the logDF
    poleDF['isEnabled'] = mask

    enabled_poleDF = poleDF[poleDF['isEnabled']]

    #extract the poleDF that is enabled
    enabled_poleDF = enabled_poleDF[poleDF['isEnabled']]

    print(f'number of poles enabled this turn: {len(enabled_poleDF)}')  

    print(f'value counts of poleDF["sim_averageResistance"] that are enabled this turn: {enabled_poleDF["sim_averageResistance"].value_counts()}')


    return enabled_poleDF


def check_for_missing_values(df, column_names):
    for column in column_names:
        missing_values = df[column].isna().sum()
        if missing_values > 0:
            print(f"Missing values in {column}: {missing_values}")
        else:
            print(f"No missing values in {column}")


# Simulate the scenario for Year 10
def run_simulation(df, ds, params, logDF= None, poleDF=None):    
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
    #this is just the index
    df['structureID'] = df.index


    if logDF is not None:
        print(f'assign logs')
        logDF = assign_logs(logDF, params)    
        logDF['structureID'] = logDF.index + len(df)

    if poleDF is not None:
        print(f'assign poles')
        poleDF = assign_poles(poleDF, params)    
        poleDF['structureID'] = poleDF.index + len(df)

        if logDF is not None:
            poleDF['structureID'] = poleDF['structureID'] + len(logDF)
    
    return df, logDF, poleDF

###UPDATE FUNCTIONS #####


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

    print(df)
    
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

    #print all unique values and counts for df['rewilded'] using pandas
    print('column rewilded values and counts in dataframe:')
    print(df['rewilded'].value_counts())
    
    # Print all unique variable values and counts for scenario_rewilded
    unique_values, counts = np.unique(ds['scenario_rewilded'], return_counts=True)
    print('column scenario_rewilded values and counts in xarray dataset:')
    for value, count in zip(unique_values, counts):
        print(f'scenario_rewilded value: {value}, count: {count}')
    
    # Step 4: Return the updated dataset
    return ds

def update_bioEnvelope_voxel_catagories(ds, params):
    #create another version of the depaved column that also considers the green envelopes
    year = params['years_passed']
    turnThreshold = params['sim_TurnsThreshold'][year]
    resistanceThreshold = params['sim_averageResistance'][year]

    bioMask = (ds['sim_Turns'] <= turnThreshold) & (ds['sim_averageResistance'] <= resistanceThreshold) &  (ds['sim_Turns'] >= 0)

    #initialise scenario_bioEnvelope as a copy of ds['scenario_rewilded']
    ds['scenario_bioEnvelope'] = xr.DataArray(
        data=np.array(ds['scenario_rewilded'].values, dtype='O'),
        dims='voxel'
    )

    #assign 'otherGround' to scenario_bioEnvelope where bioMask is true and scenario_bioEnvelope is 'none'
    ds['scenario_bioEnvelope'].loc[bioMask & (ds['scenario_bioEnvelope'] == 'none')] = 'otherGround'

    #assign 'livingFacade' to scenario_bioEnvelope where site_building_element == 'facade' and bioMask is True
    ds['scenario_bioEnvelope'].loc[(ds['site_building_element'] == 'facade') & bioMask] = 'livingFacade'

    #assign 'greenRoof' to scenario_bioEnvelope where envelope_roofType == 'green roof' and bioMask is True
    ds['scenario_bioEnvelope'].loc[(ds['envelope_roofType'] == 'green roof') & bioMask] = 'greenRoof'

    #assign 'brownRoof' to scenario_bioEnvelope where envelope_roofType == 'brown roof' and bioMask is True
    ds['scenario_bioEnvelope'].loc[(ds['envelope_roofType'] == 'brown roof') & bioMask] = 'brownRoof'

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


    #extract 


    #fallen dead branches





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

    # Initialize with object dtype
    scenario_outputs = np.full(polydata.n_points, 'none', dtype='O')
    scenario_outputs[maskForRewilding] = polydata.point_data['scenario_rewilded'][maskForRewilding]
    scenario_outputs[maskforTrees] = polydata.point_data['forest_size'][maskforTrees]
    #print unique values and counts for scenario_outputs
    print(f'unique values and counts for scenario_outputs: {pd.Series(scenario_outputs).value_counts()}')
    polydata.point_data['scenario_outputs'] = scenario_outputs



    
    print(f'unique values and counts for scenario_outputs in polydata: {pd.Series(polydata.point_data["scenario_outputs"]).value_counts()}')
    return polydata







# Call the function
#plot_scenario_rewilded(ds,treeDF_final)

import pandas as pd
import pyvista as pv
import numpy as np


def process_scenarios(site, scenario, voxel_size):
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
    print(xarray_dataset.variables)
    subsetDS = getDataSubest(xarray_dataset)
    subsetDS.to_netcdf(f'{input_folder}/{site}_{voxel_size}_subsetForScenarios.nc')
    print('loaded and subsetted xarray data')

    #loading nodes
    filePATH = f'data/revised/final/{site}'
    treeDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_treeDF.csv')
    poleDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_poleDF.csv')
    logDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_logDF.csv')


    #get extra nodes
    #first, check if extraPoleDF and extraTreeDF exist      
    if os.path.exists(f'{filePATH}/{site}-extraPoleDF.csv'):
        extraPoleDF = pd.read_csv(f'{filePATH}/{site}-extraPoleDF.csv')
    else:
        extraPoleDF = None
    if os.path.exists(f'{filePATH}/{site}-extraTreeDF.csv'):
        extraTreeDF = pd.read_csv(f'{filePATH}/{site}-extraTreeDF.csv')
    else:
        extraTreeDF = None

    print('preprocessing treeDF')

    # Apply the initialization at the start of the script
    treeDF, subsetDS = PreprocessData(treeDF, subsetDS, extraTreeDF)
    subsetDS, initialPoly = further_xarray_processing(subsetDS)
    initialPoly.save(f'{filePATH}/{site}_{voxel_size}_scenarioInitialPoly.vtk')

    #treeDF = tree_processing(treeDF, extraTreeDF, subsetDS)
    logDF = log_processing(logDF, subsetDS)
    poleDF = pole_processing(poleDF, extraPoleDF, subsetDS)

    print(f'poleDF has headings: {poleDF.columns}')

    ####
    paramsPARADE_positive = {
    'growth_factor_range' : [0.37, 0.51], #growth factor is a range
    'plantingDensity' : 50, #10 per hectare
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
    'growth_factor_range' : [0.37, 0.51], #growth factor is a range
    'plantingDensity' : 50, #10 per hectare
    'ageInPlaceThreshold' : 2, #highest threshold
    'plantThreshold' : 1, #middle threshold
    'rewildThreshold' : 0, #lowest threshold
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
    'growth_factor_range' : [0.37, 0.51], #growth factor is a range
    'plantingDensity' : 50, #10 per hectare
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
    'growth_factor_range' : [0.37, 0.51], #growth factor is a range
    'plantingDensity' : 50, #10 per hectare
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
    'growth_factor_range' : [0.37, 0.51], #growth factor is a range
    'plantingDensity' : 50, #10 per hectare
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
    'growth_factor_range' : [0.37, 0.51], #growth factor is a range
    'plantingDensity' : 50, #10 per hectare
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
    
    params = paramsDic[(site, scenario)]

    for year in years_passed:
        print(f'running simulation for year {year}')
        
        ##############################
        ##SET UP
        ##############################
        
        params['years_passed'] = year
        if site == 'trimmed-parade':
            logDF = None
            poleDF = None
        
        if site == 'uni':
            logDF = None

        if site == 'city':
            poleDF = None

        ##############################
        #SIMULTION
        ##############################

        print(f'treeDF is {treeDF}')
        print(f'logDF is {logDF}')
        print(f'poleDF is {poleDF}')
 
        treeDF_scenario, logDF_scenario, poleDF_scenario = run_simulation(treeDF.copy(), subsetDS, params, logDF, poleDF)
        print(f'done running simulation for year {year}')
        

        ##############################
        #REWILDING CATAGORY UPDATES
        ##############################

        print('integrating results into xarray')
        ds = update_rewilded_voxel_catagories(subsetDS, treeDF_scenario)
        print('updating bioEnvelope voxel catagories')
        if logDF is not None or poleDF is not None:
            ds = update_bioEnvelope_voxel_catagories(ds, params)

        """poly = a_helper_functions.convert_xarray_into_polydata(ds)
        poly.plot(scalars = 'scenario_rewilded')"""

        unique_values, counts = np.unique(ds['scenario_rewilded'], return_counts=True)
        print(f'unique values and counts for scenario_rewilded: {unique_values}, {counts}')

        ###################################################
        #ADDING RESOURCE VOXELS AND CALCULATING ROTATIONS
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

        """valid_points_polydata = pv.PolyData(valid_points)
        valid_points_polydata.plot()"""

        ds, combinedDF_scenario = a_voxeliser.integrate_resources_into_xarray(ds, treeDF_scenario, logDF_scenario, poleDF_scenario, valid_points)
        
        ##############################
        #FINAL PROCESSING
        ##############################

        ds = finalDSprocessing(ds)

        
        ##############################
        #SAVING
        ##############################

        #save treeDF_scenario to csv
        print(f'saving {year} treeDF_scenario to csv')
        #treeDF_scenario.to_csv(f'{filePATH}/{site}_{voxel_size}_treeDF_{year}.csv', index=False)
        combinedDF_scenario.to_csv(f'{filePATH}/{site}_{scenario}_{voxel_size}_nodeDF_{year}.csv', index=False)
        
        print_simulation_statistics(combinedDF_scenario, year, site)

        polydata = a_helper_functions.convert_xarray_into_polydata(ds)
        polydata = process_polydata(polydata)
        polydata.save(f'{filePATH}/{site}_{scenario}_{voxel_size}_scenarioYR{year}.vtk')

        ##############################
        #VALIDATION
        ##############################
        print('validating results')
        #print values and counts for scenario_rewilded
        unique_values, counts = np.unique(ds['scenario_rewilded'], return_counts=True)
        print(f'unique values and counts for scenario_rewilded: {unique_values}, {counts}')
        
        if logDF is not None:
            #print values and counts for scenario_bioEnvelope
            unique_values, counts = np.unique(ds['scenario_bioEnvelope'], return_counts=True)
            print(f'unique values and counts for scenario_bioEnvelope: {unique_values}, {counts}')


        ##############################
        #PLOTTING
        ##############################

        #plot_scenario_rewilded(polydata, treeDF_scenario, year)
        #plot_scenario_details(ds, year)

        # Display the updated dataframe and summary stats
        #print(summary_stats)

if __name__ == "__main__":
    sites = ['trimmed-parade']
    #sites = ['uni']
    scenarios = ['positive', 'trending']
    #scenarios = ['trending']
    #sites = ['trimmed-parade']
    
    voxel_size = 1
    for site in sites:
        for scenario in scenarios:
            process_scenarios(site, scenario, voxel_size)
    










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
