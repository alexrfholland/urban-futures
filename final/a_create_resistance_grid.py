import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import xarray as xr
import a_helper_functions
from scipy.spatial import cKDTree

#precompute neighbours for first shell and second shell
#voxels should be able to see neighbours but also second shell neighbours (ie. i j k + 2)


#resistance negative

#resistance neutral (so they can be legible)

#road_roadInfo_type    


    #Carriageway


###########

#preprocessing
    #analysis_forestLane: subset where road_terrainInfo_forest != nan. Normalise subset to 0-1.
    #analysis_greenRoof: subset where site_greenRoof_ratingInt > 1. Normalise subset to 0-1.
    #analysis_brownRoof: subset where site_brownRoof_ratingInt > 1. Normalise subset to 0-1.

    #analysis_busyRoadway: subset where (road_roadInfo_type = Carriageway && road_terrainInfo_roadCorridors_str_type == Council Major)
    #analysis_Roadway: subset where road_roadInfo_type = Carriageway.

#movement
    #site_building_element == 'facade': voxels should travel in lines to connect roofs to ground rather than pool into areas on facades

#Cannot travel (remove these voxels from all analyses)
    #subset where isTerrainUnderBuilding == True


##RESISTANCE: (from range 0-100). (do these first and then overwrite with subsequent values)
    #analysis_busyRoadway: very negative. RESISTANCE = 100
    #analysis_Roadway: negative. RESISTANCE = 80.

#NUETRAL: 0

#POTENTIAL: (from range 0-100)
    #subset where road_terrainInfo_isOpenSpace == 1.0. POTENTIAL = 100
    #analysis_forestLane > 0.5. (analysis_forestLane == 0.5 -> POTENTIAL = 50, analysis_forestLane == 1 -> POTENTIAL = 90)
    #analysis_greenRoof > 0.5. (analysis_greenRoof == 0.5 -> POTENTIAL = 40, analysis_greenRoof == 1 -> POTENTIAL = 75)
    #analysis_brownRoof > 0.5. (analysis_brownRoof == 0.5 -> POTENTIAL = 40, analysis_brownRoof == 1 -> POTENTIAL = 60)

    #MEDIUM
    #analysis_forestLane < 0.5. POTENTIAL = 50
    #subset where road_terrainInfo_isParking == 1.0. POTENTIAL = 30
    #subset where roadCorridors_str_type == 'Private'. POTENTIAL = 25
    #subset where road_terrainInfo_isLittleStreet == 1.0. POTENTIAL = 25

    #LOW
    #susbet where road_terrainInfo_isParkingMedian3mBuffer == 1.0. POTENTIAL = 10

#nodes
    #HIGH PRIORITY
    #subset where trees_tree_id != -1.
        #trees_size ('small','medium','large'). Priority goes large, medium, small

    #MEDIUM PRIORITY
    #susbet where envelope_logSize != 'unassigned'riorit
        #envelope_logSize ('small','medium','large'). Priority goes large, medium, small

    #LOW PRIORITY
    #subset where poles_pole_number != -1



###

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Define function to remap values to a new range
def remap_values_xarray(values, old_min, old_max, new_min, new_max):
    return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

# 1. Preprocessing Analysis Layers
def preprocess_analysis_layers(ds):
    # Initialize analysis_forestLane, analysis_greenRoof, and analysis_brownRoof with -1
    ds['analysis_forestLane'] = xr.full_like(ds['road_terrainInfo_forest'], -1)
    ds['analysis_greenRoof'] = xr.full_like(ds['site_greenRoof_ratingInt'], -1)
    ds['analysis_brownRoof'] = xr.full_like(ds['site_brownRoof_ratingInt'], -1)
    ds['analysis_Canopies'] = xr.full_like(ds['site_canopy_isCanopy'], 0)

    # Remap non-NaN values of forest lane to 0-1
    forest_lane_mask = ~np.isnan(ds['road_terrainInfo_forest'])
    #print how many non Nans are in analysis_forestLane
    remapped_forest_lane = remap_values_xarray(ds['road_terrainInfo_forest'], old_min=ds['road_terrainInfo_forest'].min(), old_max=ds['road_terrainInfo_forest'].max(), new_min=0, new_max=1)
    ds['analysis_forestLane'] = xr.where(forest_lane_mask, remapped_forest_lane, ds['analysis_forestLane'])

    # Remap green roof values where >1 to 0-1
    green_roof_mask = ds['site_greenRoof_ratingInt'] > 1
    remapped_green_roof = remap_values_xarray(ds['site_greenRoof_ratingInt'], old_min=ds['site_greenRoof_ratingInt'].min(), old_max=ds['site_greenRoof_ratingInt'].max(), new_min=0, new_max=1)
    ds['analysis_greenRoof'] = xr.where(green_roof_mask, remapped_green_roof, ds['analysis_greenRoof'])

    # Remap brown roof values where >1 to 0-1
    brown_roof_mask = ds['site_brownRoof_ratingInt'] > 1
    remapped_brown_roof = remap_values_xarray(ds['site_brownRoof_ratingInt'], old_min=ds['site_brownRoof_ratingInt'].min(), old_max=ds['site_brownRoof_ratingInt'].max(), new_min=0, new_max=1)
    ds['analysis_brownRoof'] = xr.where(brown_roof_mask, remapped_brown_roof, ds['analysis_brownRoof'])

    # Create binary masks for roadways
    ds['analysis_busyRoadway'] = xr.where(
        (ds['road_roadInfo_type'] == 'Carriageway') & 
        (ds['road_terrainInfo_roadCorridors_str_type'] == 'Council Major'),
        1, 0
    )
    
    ds['analysis_Roadway'] = xr.where(
        ds['road_roadInfo_type'] == 'Carriageway',
        1, 0
    )

    #create an analysis_Canopies layer initialsied as False. Set True if any of the following are True
    ds['analysis_Canopies'] = xr.where(ds['site_canopy_isCanopy'] == 1.0, 1, ds['analysis_Canopies'])
    ds['analysis_Canopies'] = xr.where(ds['road_canopy_isCanopy'] == 1.0, 1, ds['analysis_Canopies'])

    polydata = a_helper_functions.convert_xarray_into_polydata(ds)

    #polydata.plot(scalars='analysis_combined_resistance', cmap='viridis')
    #polydata.plot(scalars='analysis_Canopies', cmap='viridis')

    
    return ds

# Define function to calculate potential values using masks
def define_potential(ds, site):
    # Initialize potential array with -1
    ds['analysis_potential']  = xr.full_like(ds['analysis_forestLane'], -1)
    
    # Create masks for valid data (non -1) for brown roof, green roof, and forest lane
    brown_roof_mask = ds['analysis_brownRoof'] != -1
    green_roof_mask = ds['analysis_greenRoof'] != -1
    forest_lane_mask = ds['analysis_forestLane'] != -1

    # Remap based on masks
    ds['analysis_potential']  = xr.where(brown_roof_mask, remap_values_xarray(ds['analysis_brownRoof'], old_min=0, old_max=1, new_min=20, new_max=60), ds['analysis_potential'])
    ds['analysis_potential']  = xr.where(green_roof_mask, remap_values_xarray(ds['analysis_greenRoof'], old_min=0, old_max=1, new_min=30, new_max=75), ds['analysis_potential'])

    if site == 'city':
        ds['analysis_potential']  = xr.where(
                forest_lane_mask, 
                remap_values_xarray(ds['analysis_forestLane'], old_min=0, old_max=1, new_min=80, new_max=100), 
                ds['analysis_potential'])
        
    

    
    # Apply fixed potentials for open space, parking, private roads, little streets, and parking median buffer
    ds['analysis_potential']  = xr.where(ds['road_terrainInfo_isOpenSpace'] == 1.0, 80, ds['analysis_potential'])
    ds['analysis_potential']  = xr.where(ds['road_terrainInfo_isParkingMedian3mBuffer'] == 1.0, 50, ds['analysis_potential'])
    ds['analysis_potential']  = xr.where(ds['road_terrainInfo_isLittleStreet'] == 1.0, 60, ds['analysis_potential'])
    ds['analysis_potential']  = xr.where(ds['road_terrainInfo_roadCorridors_str_type'] == 'Private', 50, ds['analysis_potential'])
    ds['analysis_potential']  = xr.where(ds['road_terrainInfo_isParking'] == 1.0, 100, ds['analysis_potential'])

    ds['analysis_potentialNOCANOPY'] = ds['analysis_potential']

    # Increase potential by +50 for canopy areas, regardless of the current value
    ds['analysis_potential']  = xr.where(ds['analysis_Canopies'] == 1, 
                                         np.minimum(ds['analysis_potential'] + 30, 100), 
                                         ds['analysis_potential'] )

    
    
    print(f"Potential - Min: {ds['analysis_potential'] .min().item()}, Max: {ds['analysis_potential'] .max().item()}")

    """polydata = a_helper_functions.convert_xarray_into_polydata(ds)

    polydata.plot(scalars='analysis_potential', cmap='turbo')
    """
    return ds

# Define function to calculate resistance values and store them in the xarray
def define_resistance(ds):
    # Initialize resistance array with -1
    resistance = xr.full_like(ds['analysis_forestLane'], -1)
    
    # Apply resistance values for different roadways
    resistance = resistance.where(ds['analysis_busyRoadway'] != 1, 100)
    resistance = resistance.where(ds['analysis_Roadway'] != 1, 80)
    
    # Store resistance in the dataset
    ds['analysis_resistance'] = resistance

    # Display the min and max values of resistance
    print(f"Resistance - Min: {resistance.min().item()}, Max: {resistance.max().item()}")
    
    return ds


# Define function to calculate combined resistance surface and store it in the xarray
def define_combined_resistance_surface(ds, potentialName, resistanceName, combinedName):
    # Extract potential and resistance
    potential = ds[potentialName]
    resistance = ds[resistanceName]
    
    # Initialize combined values with neutral value (60)
    combined_resistance = xr.full_like(ds['analysis_forestLane'], 60.0)
    
    # Remap resistance values to 80-100 and apply to combined resistance
    resistance_mask = resistance > -1
    remapped_resistance = remap_values_xarray(resistance, old_min=0, old_max=100, new_min=80, new_max=100)
    combined_resistance = xr.where(resistance_mask, remapped_resistance, combined_resistance)
    
    # Remap potential values to 50-0 and replace the combined resistance in the potential mask
    potential_mask = potential > -1
    remapped_potential = remap_values_xarray(potential, old_min=0, old_max=100, new_min=50, new_max=0)
    combined_resistance = xr.where(potential_mask, remapped_potential, combined_resistance)


    # Store combined resistance in the dataset
    ds[combinedName] = combined_resistance

    # Display the min and max values of combined resistance
    print(f"{combinedName}: - Min: {combined_resistance.min().item()}, Max: {combined_resistance.max().item()}")
    
    return ds

def define_starting_nodes(ds):
    # Initialize variables in ds
    ds['analysis_nodeType'] = xr.full_like(ds['analysis_forestLane'], 'unassigned', dtype='U10')
    ds['analysis_nodeSize'] = xr.full_like(ds['analysis_forestLane'], 'unassigned', dtype='U10')
    ds['analysis_originalID'] = xr.full_like(ds['analysis_forestLane'], -1, dtype=int)
    ds['analysis_nodeID'] = xr.full_like(ds['analysis_forestLane'], -1, dtype=int)

    # Create masks
    tree_mask = ds['trees_tree_id'] != -1
    log_mask = ds['envelope_logSize'] != 'unassigned'

    # Create a mask for poles
    pole_mask = ds['poles_pole_number'] != -1

    poles = ds['poles_pole_number'][pole_mask]
    print(f"Pole IDs range from {poles.min().item()} to {poles.max().item()}")


    # Apply masks for trees
    ds['analysis_nodeType'] = xr.where(tree_mask, 'tree', ds['analysis_nodeType'])
    ds['analysis_nodeSize'] = xr.where(tree_mask, ds['trees_size'], ds['analysis_nodeSize'])
    ds['analysis_originalID'] = xr.where(tree_mask, ds['trees_tree_number'], ds['analysis_originalID'])
    ds['analysis_nodeID'] = xr.where(tree_mask, ds['trees_tree_number'], ds['analysis_nodeID'])

    # Apply masks for logs
    ds['analysis_nodeType'] = xr.where(log_mask, 'log', ds['analysis_nodeType'])
    ds['analysis_nodeSize'] = xr.where(log_mask, ds['envelope_logSize'], ds['analysis_nodeSize'])
    ds['analysis_originalID'] = xr.where(log_mask, ds['envelope_logNo'], ds['analysis_originalID'])
    max_tree_id = ds['trees_tree_number'].max().item()
    ds['analysis_nodeID'] = xr.where(log_mask, ds['envelope_logNo'] + max_tree_id, ds['analysis_nodeID'])

    # Apply masks for poles
    ds['analysis_nodeType'] = xr.where(pole_mask, 'pole', ds['analysis_nodeType'])
    ds['analysis_nodeSize'] = xr.where(pole_mask, 'medium', ds['analysis_nodeSize'])
    ds['analysis_originalID'] = xr.where(pole_mask, ds['poles_pole_number'], ds['analysis_originalID'])
    max_log_id = ds['envelope_logNo'].max().item()
    ds['analysis_nodeID'] = xr.where(pole_mask, ds['poles_pole_number'] + max_tree_id + max_log_id, ds['analysis_nodeID'])

    # Filter out unassigned nodes (where analysis_nodeID is -1)
    node_mask = ds['analysis_nodeID'] != -1

    # Create a DataFrame from the filtered xarray variables
    node_mappingsDF = pd.DataFrame({
        'analysis_nodeID': ds['analysis_nodeID'].values[node_mask],
        'analysis_originalID': ds['analysis_originalID'].values[node_mask],
        'analysis_nodeType': ds['analysis_nodeType'].values[node_mask]
    })

    # Reset the index
    node_mappingsDF = node_mappingsDF.reset_index(drop=True)

    print(f"Created node DataFrame with {len(node_mappingsDF)} rows")
    print(node_mappingsDF.head())


    return ds, node_mappingsDF


# 5. Visualization
def plot_combined_surface(ds, combined_resistance):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the coordinates and combined resistance values for plotting
    x = ds['centroid_x'].values
    y = ds['centroid_y'].values
    z = ds['centroid_z'].values
    combined_resistance_values = combined_resistance.values
    
    # Use a diverging colormap ('coolwarm') to show combined resistance values
    sc = ax.scatter(x, y, z, c=combined_resistance_values, cmap='coolwarm', alpha=0.6, marker='o', s=2)
    
    # Add color bar to indicate combined resistance values
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Combined Resistance Value')
    
    # Set labels and title
    ax.set_xlabel('Centroid X')
    ax.set_ylabel('Centroid Y')
    ax.set_zlabel('Centroid Z')
    ax.set_title('3D Combined Resistance Surface (Diverging Colormap)')
    
    plt.tight_layout()
    plt.show()

def updateNodeDfs(treeDF, poleDF, logDF):

    # Create column 'Improvement' in treeDF, poleDF, logDF. Initialise as False.
    treeDF['Improvement'] = False
    poleDF['Improvement'] = False
    logDF['Improvement'] = False

    #Convert Size and Control to catagories
    size_categories = ['small', 'medium', 'large', 'snag', 'senescing', 'collapsed']
    control_categories = ['street-tree', 'park-tree', 'reserve-tree']

    treeDF['size'] = pd.Categorical(treeDF['size'], categories=size_categories, ordered=True)
    treeDF['control'] = pd.Categorical(treeDF['control'], categories=control_categories, ordered=True)

    return treeDF, poleDF, logDF


def update_node_mappings_inDFs(ds, treeDF, poleDF, logDF, node_mappings):
    
    #ASSIGN NODE IDS TO DFS
    # Create the 'voxelIndex' for each dataframe
    tree_voxelIndex = treeDF['voxel_index'].values
    log_voxelIndex = logDF['voxelID'].values
    pole_voxelIndex = poleDF['voxel_index'].values

    # Create the 'debugNodeID' by looking up ds['analysis_nodeID'][voxelIndex] for each dataframe
    treeDF['NodeID'] = ds['analysis_nodeID'].sel(voxel=tree_voxelIndex).values
    logDF['NodeID'] = ds['analysis_nodeID'].sel(voxel=log_voxelIndex).values
    poleDF['NodeID'] = ds['analysis_nodeID'].sel(voxel=pole_voxelIndex).values
    
    #ERROR CHECK THE NODE IDS (WE COULD DELETE ALL THIS AS THEY MACTH!)
    # Extract the relevant portions of node_mappings for each node type
    tree_mappings = node_mappings[node_mappings['analysis_nodeType'] == 'tree']
    log_mappings = node_mappings[node_mappings['analysis_nodeType'] == 'log']
    pole_mappings = node_mappings[node_mappings['analysis_nodeType'] == 'pole']

    # Merge the dataframes with their respective mappings
    treeDF = pd.merge(treeDF, tree_mappings[['analysis_originalID', 'analysis_nodeID']], 
                      left_on='tree_number', right_on='analysis_originalID', how='left')

    logDF = pd.merge(logDF, log_mappings[['analysis_originalID', 'analysis_nodeID']], 
                     left_on='logNo', right_on='analysis_originalID', how='left')

    poleDF = pd.merge(poleDF, pole_mappings[['analysis_originalID', 'analysis_nodeID']], 
                      left_on='pole_number', right_on='analysis_originalID', how='left')

    # Rename the 'analysis_nodeID' to 'NodeID' in each dataframe and convert to int
    treeDF['debugNodeID'] = treeDF['analysis_nodeID']
    logDF['debugNodeID'] = logDF['analysis_nodeID']
    poleDF['debugNodeID'] = poleDF['analysis_nodeID']

    # Drop the 'analysis_originalID' and 'analysis_nodeID' columns since they are no longer needed
    treeDF.drop(columns=['analysis_originalID', 'analysis_nodeID'], inplace=True)
    logDF.drop(columns=['analysis_originalID', 'analysis_nodeID'], inplace=True)
    poleDF.drop(columns=['analysis_originalID', 'analysis_nodeID'], inplace=True)
    return treeDF, logDF, poleDF

def AssignCanopyFootprintsToNodes(ds, treeDF, poleDF, logDF):
    from scipy.spatial import cKDTree

    # Construct a kdtree of the treeDF points 'x' and 'y'
    tree_points = treeDF[['x', 'y']].values
    kdtree = cKDTree(tree_points)

    # Initialize new variable in xarray called 'node_CanopyID'. Set all values to -1
    ds['node_CanopyID'] = xr.full_like(ds['site_canopy_isCanopy'], -1)

    # Create the canopy mask
    canopy_mask = (ds['site_canopy_isCanopyCorrected'] == 1.0) | (ds['road_canopy_isCanopy'] == 1.0)

    # Find nearest index of tree_points for each point in ds within 10 meters
    distances, nearest_indices = kdtree.query(np.column_stack((ds['centroid_x'].values, ds['centroid_y'].values)), distance_upper_bound=10)

    # Create a mask of valid distances (within 10 meters)
    valid_mask = distances <= 10

    # Create an array to store nearest NodeIDs, initializing it with -1
    nearest_NodeIDs = np.full_like(ds['centroid_x'].values, -1, dtype=int)

    # Assign NodeID values from the nearest treeDF based on the valid indices within 10 meters
    nearest_NodeIDs[valid_mask] = treeDF.loc[nearest_indices[valid_mask], 'NodeID'].values

    # Assign the NodeIDs to ds['node_CanopyID'] where the canopy mask is True and within 10 meters
    ds['node_CanopyID'] = xr.where(canopy_mask & valid_mask, nearest_NodeIDs, ds['node_CanopyID'])

    return ds

def Calculate_average_canopy_resistance(ds, treeDF, poleDF, logDF):
    """
    This function calculates the average canopy resistance for each group of voxels with node_CanopyID > 0.
    It also updates treeDF with the corresponding CanopyResistance and CanopyArea for each tree node.

    Parameters:
    ds (xarray.Dataset): The xarray dataset containing canopy data.
    treeDF (pandas.DataFrame): DataFrame containing tree node information.
    poleDF (pandas.DataFrame): DataFrame containing pole node information.
    logDF (pandas.DataFrame): DataFrame containing log node information.

    Returns:
    Updated ds, treeDF, poleDF, logDF with canopy resistance information.
    """

    # Initialize 'node_CanopyResistance' to -1 for all voxels
    ds['node_CanopyResistance'] = xr.full_like(ds['node_CanopyID'], -1)

    # Filter the voxels with node_CanopyID > 0
    canopy_voxels = ds.where(ds['node_CanopyID'] > 0, drop=True)

    # Group the filtered voxels by node_CanopyID and calculate the average of analysis_combined_resistanceNOCANOPY per group
    avg_canopy_resistance = canopy_voxels.groupby('node_CanopyID').mean()['analysis_combined_resistanceNOCANOPY']

    # Create a dictionary mapping each node_CanopyID to its average resistance
    canopy_resistance_dict = avg_canopy_resistance.to_pandas().to_dict()

    # Use xarray's apply_ufunc to map the values from the dictionary back to the xarray dataset
    ds['node_CanopyResistance'] = xr.apply_ufunc(
        lambda x: canopy_resistance_dict.get(x, -1),  # map or return -1 for unmapped values
        ds['node_CanopyID'],
        vectorize=True
    )

    # Update treeDF by mapping CanopyResistance using NodeID
    treeDF['CanopyResistance'] = treeDF['NodeID'].map(canopy_resistance_dict)

    # Calculate the number of voxels per node_CanopyID group
    voxel_counts = canopy_voxels.groupby('node_CanopyID').count()['centroid_x'].to_pandas()

    # Calculate voxel area (voxel_size^2) from the 'voxel_size' attribute in the dataset
    voxel_area = ds.attrs['voxel_size'] ** 2

    # Create CanopyArea column in treeDF: number of voxels per group * voxel area
    treeDF['CanopyArea'] = treeDF['NodeID'].map(voxel_counts) * voxel_area

    return ds, treeDF, poleDF, logDF



def Calculate_average_canopy_resistance2(ds, treeDF, poleDF, logDF):
    """
    This function calculates the average canopy resistance for each group of voxels with node_CanopyID > 0
    
    Parameters:
    ds (xarray.Dataset): The xarray dataset containing canopy data.
    """

    # Initialize 'node_CanopyResistance' to -1 for all voxels
    ds['node_CanopyResistance'] = xr.full_like(ds['node_CanopyID'], -1)

    # Filter the voxels with node_CanopyID > 0
    canopy_voxels = ds.where(ds['node_CanopyID'] > 0, drop=True)

    # Group the filtered voxels by node_CanopyID and calculate the average of analysis_combined_resistanceNOCANOPY per group
    avg_canopy_resistance = canopy_voxels.groupby('node_CanopyID').mean()['analysis_combined_resistanceNOCANOPY']

    # Create a dictionary mapping each node_CanopyID to its average resistance
    canopy_resistance_dict = avg_canopy_resistance.to_pandas().to_dict()

    # Use xarray's apply_ufunc to map the values from the dictionary back to the xarray dataset
    ds['node_CanopyResistance'] = xr.apply_ufunc(
        lambda x: canopy_resistance_dict.get(x, -1),  # map or return -1 for unmapped values
        ds['node_CanopyID'],
        vectorize=True
    )

    # Now update the treeDF by mapping the averageCanopyResistance using NodeID
    treeDF['CanopyResistance'] = treeDF['NodeID'].map(canopy_resistance_dict)

    #TODO: resistance for pole and log nodes

    return ds, treeDF, poleDF, logDF




# 6. Main Execution
def get_resistance(ds, treeDF, poleDF, logDF, site):
    # Load the xarray dataset
    #ds = xr.open_dataset('/mnt/data/city_5_voxelArray_withLogs.nc')

    print(f"TEST: Before preprocessing, Unique values in xarray_dataset['poles_pole_number']: {np.unique(ds['poles_pole_number'].values)}")
 
    
    
    print("Starting preprocessing of analysis layers...")
    ds = preprocess_analysis_layers(ds)
    print("Preprocessing complete. Dataset shape:", ds.dims)

    print("Defining potential...")
    ds = define_potential(ds, site)
    print("Potential defined. Range:", ds['analysis_potential'].min().item(), "to", ds['analysis_potential'].max().item())

    print("Defining resistance...")
    ds = define_resistance(ds)
    print("Resistance defined. Range:", ds['analysis_resistance'].min().item(), "to", ds['analysis_resistance'].max().item())

    print("Calculating combined resistance surface...")

    
    ds = define_combined_resistance_surface(ds,'analysis_potential','analysis_resistance','analysis_combined_resistance')
    print("Combined resistance surface calculated. Range:", ds['analysis_combined_resistance'].min().item(), "to", ds['analysis_combined_resistance'].max().item())
    
    ds = define_combined_resistance_surface(ds,'analysis_potentialNOCANOPY','analysis_resistance','analysis_combined_resistanceNOCANOPY')
    print("Combined resistance surface calculated. Range:", ds['analysis_combined_resistance'].min().item(), "to", ds['analysis_combined_resistance'].max().item())
    

    print('Defining starting nodes...')
    ds, node_mappingsDF = define_starting_nodes(ds)

    print('Preprocessing node dataframes...')
    treeDF, poleDF, logDF = updateNodeDfs(treeDF, poleDF, logDF)

    print('Updating node mappings in dataframes...')
    treeDF, logDF, poleDF = update_node_mappings_inDFs(ds, treeDF, poleDF, logDF, node_mappingsDF)

    print('Assigning canopies to nodes...')
    AssignCanopyFootprintsToNodes(ds, treeDF, poleDF, logDF)
    print('Calculating average canopy resistance...')
    ds, treeDF, poleDF, logDF = Calculate_average_canopy_resistance(ds, treeDF, poleDF, logDF)


    required_variables = ['voxel_I', 'voxel_J', 'voxel_K', 'centroid_x', 'centroid_y', 'centroid_z', 'analysis_combined_resistance','analysis_potential','analysis_resistance', 'analysis_nodeType', 'analysis_nodeSize','analysis_originalID', 'analysis_nodeID', 'node_CanopyID','node_CanopyResistance']
    required_attributes = ['bounds', 'voxel_size']


    subsetDS = a_helper_functions.create_subset_dataset(ds, required_variables, required_attributes)

    voxel_size = subsetDS.attrs['voxel_size']

    dataframes = {'treeDF': treeDF, 'poleDF': poleDF, 'logDF': logDF, 'node_mappingsDF': node_mappingsDF}

    ds = a_helper_functions.update_xarray_with_subset(ds, subsetDS)

    return ds, subsetDS, dataframes

    
    # Plot the combined resistance surface
    
# Run the main function
if __name__ == "__main__":
    site = 'trimmed-parade'
    site = 'city'
    input_folder = f'data/revised/final/{site}'
    voxel_size = 1
    xarray_path = f'{input_folder}/{site}_{voxel_size}_voxelArray_withLogs.nc'
    ds = xr.open_dataset(xarray_path)
    
    treeDF = pd.read_csv(f'{input_folder}/{site}_{voxel_size}_treeDF.csv')
    poleDF = pd.read_csv(f'{input_folder}/{site}_{voxel_size}_poleDF.csv')
    logDF = pd.read_csv(f'{input_folder}/{site}_{voxel_size}_logDF.csv')    
    
    ds, subsetDS, dataframes = get_resistance(ds, treeDF, poleDF, logDF, site)
    
    #save subsetDS
    subsetDS.to_netcdf(f'{input_folder}/{site}_{voxel_size}_voxelArray_withResistance.nc')
    
    #save updated dataframes
    for df in dataframes:
        print(f'Saving {df} to {input_folder}/{site}_{voxel_size}_{df}.csv')
        dataframes[df].to_csv(f'{input_folder}/{site}_{voxel_size}_{df}.csv', index=False)

    print(f"Node mappings saved to {input_folder}/{site}_{voxel_size}_node_mappings.csv")

    polydata = a_helper_functions.convert_xarray_into_polydata(subsetDS)

    import pyvista as pv
    # Plot analysis_potential
    plotter = pv.Plotter()
    plotter.add_mesh(polydata, scalars='analysis_potential', clim=[0, 100],below_color='white', show_scalar_bar=True, cmap='BrBG')
    plotter.enable_eye_dome_lighting()
    plotter.show()
    plotter.close()

    plotter = pv.Plotter()
    plotter.add_mesh(polydata, scalars='analysis_resistance', clim=[0, 100],below_color='white', show_scalar_bar=True, cmap='PuRd')
    plotter.enable_eye_dome_lighting()
    plotter.show()
    plotter.close()

    # Plot node_CanopyResistance
    plotter = pv.Plotter()
    plotter.add_mesh(polydata, scalars='node_CanopyResistance', clim=[0, 100],below_color='white', show_scalar_bar=True, cmap='viridis')
    plotter.enable_eye_dome_lighting()
    plotter.show()
    plotter.close()

    # Plot analysis_combined_resistance
    plotter = pv.Plotter()
    plotter.add_mesh(polydata, scalars='analysis_combined_resistance', cmap='coolwarm')
    plotter.enable_eye_dome_lighting()
    plotter.show()
    plotter.close()

  



