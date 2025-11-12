import os
import pandas as pd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
import a_helper_functions

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

    #poly = a_helper_functions.convert_xarray_into_polydata(subsetDS)
    #poly.plot(scalars='site_building_element', cmap='Set1')

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

    return ds, poly

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
    # Get positions from the xarray dataset
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

    # Check if sim_averageResistance is in poleDF, if not, set to -1
    if 'sim_averageResistance' not in poleDF.columns:
        poleDF['sim_averageResistance'] = -1

    mask = poleDF['sim_averageResistance'] == -1
    poleDF.loc[mask, 'sim_averageResistance'] = ds['analysis_combined_resistance'].values[poleDF.loc[mask, 'voxel_index']]

    print(f'value counts of combined poleDF["sim_averageResistance"]: {poleDF["sim_averageResistance"].value_counts()}')
    
    return poleDF

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

        # Get the voxelIndex
        all_points = np.vstack((ds['centroid_x'].values, ds['centroid_y'].values, ds['centroid_z'].values)).T
        tree = cKDTree(all_points)

        # Query the tree to get the nearest point and assign the index to extraTreeDF['voxelID']
        extraTreeDF['voxel_index'] = tree.query(np.vstack((extraTreeDF['x'].values, extraTreeDF['y'].values, extraTreeDF['z'].values)).T)[1]

        extraTreeDF['CanopyResistance'] = ds['analysis_combined_resistance'].values[extraTreeDF['voxel_index']]

        # Get the sim_averageResistance from the xarray dataset and assign to extraTreeDF['CanopyResistance']
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

        # Combine treeDF and extraTreeDF
        treeDF = pd.concat([treeDF, extraTreeDF], ignore_index=True)
        
    # Check if useful_life_expectency is in treeDF and rename to useful_life_expectancy
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

    # Initialize rewilding column as paved
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

def initialize_dataset(site, voxel_size):
    input_folder = f'data/revised/final/{site}'
    filepath = f'{input_folder}/{site}_{voxel_size}_voxelArray_RewildingNodes.nc'
    xarray_dataset = xr.open_dataset(filepath)

    print("Variables in xarray_dataset:")
    print(xarray_dataset.variables)
    
    subsetDS = getDataSubest(xarray_dataset)
    subsetDS.to_netcdf(f'{input_folder}/{site}_{voxel_size}_subsetForScenarios.nc')
    print('Loaded and subsetted xarray data')
    
    return subsetDS

def load_node_dataframes(site, voxel_size):
    filePATH = f'data/revised/final/{site}'
    treeDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_treeDF.csv')
    poleDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_poleDF.csv')
    logDF = pd.read_csv(f'{filePATH}/{site}_{voxel_size}_logDF.csv')
    
    return treeDF, poleDF, logDF

def load_extra_node_dataframes(site):
    filePATH = f'data/revised/final/{site}'
    
    extraPoleDF = None
    if os.path.exists(f'{filePATH}/{site}-extraPoleDF.csv'):
        extraPoleDF = pd.read_csv(f'{filePATH}/{site}-extraPoleDF.csv')
        print(f'Loaded {extraPoleDF.shape[0]} extra poles')
    
    extraTreeDF = None
    if os.path.exists(f'{filePATH}/{site}-extraTreeDF.csv'):
        extraTreeDF = pd.read_csv(f'{filePATH}/{site}-extraTreeDF.csv')
        print(f'Loaded {extraTreeDF.shape[0]} extra trees')
    
    return extraTreeDF, extraPoleDF

def initialize_scenario_data(site, voxel_size):
    """Main function to initialize all data needed for scenarios"""
    # Load and initialize xarray dataset
    subsetDS = initialize_dataset(site, voxel_size)
    
    # Load tree, pole, and log dataframes
    treeDF, poleDF, logDF = load_node_dataframes(site, voxel_size)
    
    # Load extra dataframes if they exist
    extraTreeDF, extraPoleDF = load_extra_node_dataframes(site)
    
    # Preprocess the data
    print('Preprocessing tree dataframe')
    treeDF, subsetDS = PreprocessData(treeDF, subsetDS, extraTreeDF)
    
    # Further process the xarray dataset
    subsetDS, initialPoly = further_xarray_processing(subsetDS)
    
    # Save initial polydata for visualization
    filePATH = f'data/revised/final/{site}'
    initialPoly.save(f'{filePATH}/{site}_{voxel_size}_scenarioInitialPoly.vtk')
    
    # Process log and pole dataframes
    logDF = log_processing(logDF, subsetDS)
    poleDF = pole_processing(poleDF, extraPoleDF, subsetDS)
    
    print(f'poleDF has headings: {poleDF.columns}')
    
    return treeDF, poleDF, logDF, subsetDS

if __name__ == "__main__":
    # Get site and voxel size from user
    site = input("Please enter the site name (e.g., trimmed-parade, city, uni): ")
    voxel_size = int(input("Please enter the voxel size (default 1): ") or "1")
    
    # Initialize all data needed for scenarios
    treeDF, poleDF, logDF, subsetDS = initialize_scenario_data(site, voxel_size)
    
    print("Initialization complete. Data is ready for scenario generation.") 