import os
import sys
from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv

CODE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "_futureSim_refactored")
REPO_ROOT = CODE_ROOT.parent

if str(CODE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT.parent))

from _futureSim_refactored.sim.voxel import voxel_a_helper_functions as a_helper_functions
from _futureSim_refactored.paths import (
    site_extra_pole_locations_path,
    site_extra_tree_locations_path,
    site_inputs_dir,
    site_log_locations_path,
    site_pole_locations_path,
    site_rewilding_voxel_array_path,
    site_subset_dataset_path,
    site_tree_locations_path,
)

LEAN_SUBSET_DROP_VARIABLES = {
    'analysis_busyRoadway',
    'analysis_Roadway',
    'analysis_potentialNOCANOPY',
    'node_CanopyResistance',
    'analysis_nodeType',
}


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

    if not a_helper_functions.export_all_pointdata_variables():
        variables = [name for name in variables if name not in LEAN_SUBSET_DROP_VARIABLES]

    possibility_space_ds = a_helper_functions.create_subset_dataset(ds, variables, attributes)

    print(possibility_space_ds.variables)

    if a_helper_functions.export_all_pointdata_variables():
        possibility_space_ds['envelopeIsBrownRoof'] = xr.full_like(possibility_space_ds['node_CanopyID'], -1)
        brownRoofMask = possibility_space_ds['envelope_roofType'] == 'brown roof'
        possibility_space_ds['envelopeIsBrownRoof'][brownRoofMask] = 1
    print(possibility_space_ds.attrs['bounds'])

    #poly = a_helper_functions.convert_xarray_into_polydata(possibility_space_ds)
    #poly.plot(scalars='site_building_element', cmap='Set1')

    return possibility_space_ds

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
    if a_helper_functions.export_all_pointdata_variables():
        node_type_mapping = dict(zip(ds['analysis_nodeID'].values, ds['analysis_nodeType'].values))
        ds['sim_nodeType'] = xr.apply_ufunc(
            lambda x: node_type_mapping.get(x, -1),
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

    # Fill NaN values in 'diameter_breast_height' using triangular distributions
    print("Checking for missing values in 'diameter_breast_height' and filling them based on 'size' column...")
    nan_before = treeDF['diameter_breast_height'].isna().sum()
    dbh_distributions = {
        'small':  {'left': 0,  'mode': 15, 'right': 40},
        'medium': {'left': 40, 'mode': 55, 'right': 80},
        'large':  {'left': 80, 'mode': 95, 'right': 110},
    }
    rng = np.random.default_rng(seed=42)
    nan_mask = treeDF['diameter_breast_height'].isna()
    for size_class, params in dbh_distributions.items():
        size_nan_mask = nan_mask & treeDF['size'].eq(size_class)
        n = size_nan_mask.sum()
        if n > 0:
            values = rng.triangular(params['left'], params['mode'], params['right'], size=n)
            treeDF.loc[size_nan_mask, 'diameter_breast_height'] = np.round(values, 1)
            print(f"  Filled {n} {size_class} trees: range {values.min():.1f}-{values.max():.1f}cm, mean {values.mean():.1f}cm")
    nan_after = treeDF['diameter_breast_height'].isna().sum()

    # Print the number of NaNs filled
    print(f"Number of missing values before: {nan_before}")
    print(f"Number of missing values after: {nan_after}")
    print("Filling completed.")

    # Distribute trees sharing identical DBH values using Fischer growth spread.
    # For each duplicate DBH, compute ±5 years of growth to get a natural range,
    # then spread trees uniformly across it. Wider for small (fast-growing) trees,
    # narrower for large (slow-growing) trees.
    fischer_k = 0.0197135 * np.pi / 4.0  # Fischer SI constant
    spread_years = 5
    min_group = 2  # only spread groups of 2+
    living_mask = treeDF['size'].isin(['small', 'medium', 'large'])
    spread_rng = np.random.default_rng(seed=123)
    total_spread = 0
    for dbh_val, group in treeDF.loc[living_mask].groupby('diameter_breast_height'):
        if len(group) < min_group:
            continue
        virtual_age = fischer_k * dbh_val ** 2
        age_lo = max(0, virtual_age - spread_years)
        age_hi = virtual_age + spread_years
        dbh_lo = np.sqrt(age_lo / fischer_k)
        dbh_hi = np.sqrt(age_hi / fischer_k)
        spread_values = spread_rng.uniform(dbh_lo, dbh_hi, size=len(group))
        treeDF.loc[group.index, 'diameter_breast_height'] = np.round(spread_values, 1)
        total_spread += len(group)
    if total_spread > 0:
        print(f"Spread {total_spread} trees with duplicate DBH values using ±{spread_years}yr Fischer growth range")

    # Base trees without canopy footprints can still inherit support values from their own voxel.
    voxel_index = pd.to_numeric(treeDF["voxel_index"], errors="coerce")
    valid_voxel_mask = voxel_index.notna()
    valid_voxel_index = voxel_index.loc[valid_voxel_mask].astype(int)

    if "CanopyResistance" not in treeDF.columns:
        treeDF["CanopyResistance"] = np.nan
    missing_canopy_resistance = valid_voxel_mask & treeDF["CanopyResistance"].isna()
    if missing_canopy_resistance.any():
        treeDF.loc[missing_canopy_resistance, "CanopyResistance"] = (
            ds["analysis_combined_resistance"].values[
                voxel_index.loc[missing_canopy_resistance].astype(int)
            ]
        )

    if "sim_averageResistance" not in treeDF.columns:
        treeDF["sim_averageResistance"] = np.nan
    missing_sim_average_resistance = valid_voxel_mask & treeDF["sim_averageResistance"].isna()
    if missing_sim_average_resistance.any():
        resistance_source = (
            ds["sim_averageResistance"]
            if "sim_averageResistance" in ds.variables
            else ds["analysis_combined_resistance"]
        )
        treeDF.loc[missing_sim_average_resistance, "sim_averageResistance"] = (
            resistance_source.values[
                voxel_index.loc[missing_sim_average_resistance].astype(int)
            ]
        )

    if "sim_Turns" not in treeDF.columns:
        treeDF["sim_Turns"] = np.nan
    missing_sim_turns = valid_voxel_mask & treeDF["sim_Turns"].isna()
    if missing_sim_turns.any():
        treeDF.loc[missing_sim_turns, "sim_Turns"] = (
            ds["sim_Turns"].values[
                voxel_index.loc[missing_sim_turns].astype(int)
            ]
        )

    print(f"Final NaN count in treeDF['CanopyResistance']: {treeDF['CanopyResistance'].isna().sum()}")
    print(f"Final NaN count in treeDF['sim_averageResistance']: {treeDF['sim_averageResistance'].isna().sum()}")
    print(f"Final NaN count in treeDF['sim_Turns']: {treeDF['sim_Turns'].isna().sum()}")

    # Initialize rewilding column as paved
    treeDF['under-node-treatment'] = 'paved'
    treeDF['isNewTree'] = False
    treeDF['isRewildedTree'] = False
    treeDF['hasbeenReplanted'] = False
    treeDF['unmanagedCount'] = 0
    treeDF['action'] = 'None'

    ds['scenario_rewildingEnabled'] = xr.full_like(ds['node_CanopyID'], -1)
    ds['scenario_rewildGroundRecruitZone'] = xr.full_like(ds['node_CanopyID'], -1)
    ds['scenario_nodeRewildRecruitZone'] = xr.full_like(ds['node_CanopyID'], -1)
    ds['scenario_underCanopyRecruitZone'] = xr.full_like(ds['node_CanopyID'], -1)
    ds['scenario_underCanopyLinkedRecruitZone'] = xr.full_like(ds['node_CanopyID'], -1)
    underBuildingMask = ds['isTerrainUnderBuilding'] == True
    ds['sim_Turns'][underBuildingMask] = -1
    ds['sim_Nodes'][underBuildingMask] = -1
    
    return treeDF, ds

def initialize_dataset(site, voxel_size, *, write_cache: bool = True):
    filepath = site_rewilding_voxel_array_path(site, voxel_size)
    subset_path = site_subset_dataset_path(site, voxel_size)

    if subset_path.exists() and subset_path.stat().st_size > 0:
        try:
            possibility_space_ds = xr.open_dataset(subset_path)
            print(f'Loaded existing subset dataset from {subset_path}')
            return possibility_space_ds
        except Exception as exc:
            print(f'Failed to load existing subset dataset from {subset_path}: {exc}')
            print('Rebuilding subset dataset.')

    xarray_dataset = xr.open_dataset(filepath)

    print("Variables in xarray_dataset:")
    print(xarray_dataset.variables)

    possibility_space_ds = getDataSubest(xarray_dataset)
    if write_cache:
        possibility_space_ds.to_netcdf(subset_path)
    print('Loaded and subsetted xarray data')

    return possibility_space_ds

def load_node_dataframes(site, voxel_size):
    treeDF = pd.read_csv(site_tree_locations_path(site, voxel_size))
    poleDF = pd.read_csv(site_pole_locations_path(site, voxel_size))
    logDF = pd.read_csv(site_log_locations_path(site, voxel_size))
    
    return treeDF, poleDF, logDF

def load_extra_node_dataframes(site):
    extraPoleDF = None
    extra_pole_path = site_extra_pole_locations_path(site)
    if extra_pole_path.exists():
        extraPoleDF = pd.read_csv(extra_pole_path)
        print(f'Loaded {extraPoleDF.shape[0]} extra poles')
    
    extraTreeDF = None
    extra_tree_path = site_extra_tree_locations_path(site)
    if extra_tree_path.exists():
        extraTreeDF = pd.read_csv(extra_tree_path)
        print(f'Loaded {extraTreeDF.shape[0]} extra trees')
    
    return extraTreeDF, extraPoleDF

def initialize_scenario_data(site, voxel_size):
    """Main function to initialize all data needed for scenarios"""
    # Load and initialize xarray dataset
    possibility_space_ds = initialize_dataset(site, voxel_size)
    
    # Load tree, pole, and log dataframes
    treeDF, poleDF, logDF = load_node_dataframes(site, voxel_size)
    
    # Load extra dataframes if they exist
    extraTreeDF, extraPoleDF = load_extra_node_dataframes(site)
    
    # Preprocess the data
    print('Preprocessing tree dataframe')
    treeDF, possibility_space_ds = PreprocessData(treeDF, possibility_space_ds, extraTreeDF)
    
    # Further process the xarray dataset
    possibility_space_ds, initialPoly = further_xarray_processing(possibility_space_ds)
    
    # Save initial polydata for visualization
    initial_poly_path = site_inputs_dir(site) / f"{site}_{voxel_size}_scenarioInitialPoly.vtk"
    initialPoly.save(initial_poly_path)
    
    # Process log and pole dataframes
    logDF = log_processing(logDF, possibility_space_ds)
    poleDF = pole_processing(poleDF, extraPoleDF, possibility_space_ds)
    
    print(f'poleDF has headings: {poleDF.columns}')
    
    return treeDF, poleDF, logDF, possibility_space_ds

if __name__ == "__main__":
    # Get site and voxel size from user
    site = input("Please enter the site name (e.g., trimmed-parade, city, uni): ")
    voxel_size = int(input("Please enter the voxel size (default 1): ") or "1")
    
    # Initialize all data needed for scenarios
    treeDF, poleDF, logDF, possibility_space_ds = initialize_scenario_data(site, voxel_size)
    
    print("Initialization complete. Data is ready for scenario generation.") 
