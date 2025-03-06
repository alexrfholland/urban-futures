import a_resource_distributor_dataframes, a_rotate_resource_structures, f_SiteCoordinates
import pyvista as pv
import xarray as xr
import numpy as np
from pathlib import Path
import logging
import pandas as pd
import sparse  # Ensure 'sparse' is installed
import os
import a_helper_functions

# ================================
# Setup Logging
# ================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




# ================================
# INITITAL VOXELISATION
# ================================

def validate_voxel_properties_df(voxel_properties_df):
    """
    Validates that each (voxel_I, voxel_J, voxel_K) combination is unique.

    Parameters:
        voxel_properties_df (pd.DataFrame): DataFrame containing voxel properties.

    Raises:
        ValueError: If duplicate voxel indices are found.
    """
    duplicates = voxel_properties_df[voxel_properties_df.duplicated(subset=['voxel_I', 'voxel_J', 'voxel_K'], keep=False)]
    if not duplicates.empty:
        logger.error("Duplicated I, J, K combinations found in voxel_properties_df.")
        print(duplicates)
        raise ValueError("Duplicated I, J, K combinations detected in voxel_properties_df.")
    logger.info("voxel_properties_df has unique I, J, K combinations after grouping.")

def print_bounds(ds, df):
    """
    Prints the spatial bounds of the xarray Dataset and the resource DataFrame.

    Parameters:
        ds (xr.Dataset): The voxelized Dataset.
        df (pd.DataFrame): The resource DataFrame.
    """
    ds_bounds = (
        ds['centroid_x'].min().item(), ds['centroid_x'].max().item(),
        ds['centroid_y'].min().item(), ds['centroid_y'].max().item(),
        ds['centroid_z'].min().item(), ds['centroid_z'].max().item()
    )

    df_bounds = (
        df['x'].min(), df['x'].max(),
        df['y'].min(), df['y'].max(),
        df['z'].min(), df['z'].max()
    )

    print(f"Dataset (xarray) bounds: xmin={ds_bounds[0]}, xmax={ds_bounds[1]}, "
          f"ymin={ds_bounds[2]}, ymax={ds_bounds[3]}, zmin={ds_bounds[4]}, zmax={ds_bounds[5]}")
    print(f"Resource DataFrame bounds: xmin={df_bounds[0]}, xmax={df_bounds[1]}, "
          f"ymin={df_bounds[2]}, ymax={df_bounds[3]}, zmin={df_bounds[4]}, zmax={df_bounds[5]}")

def polydata_to_flat_dict(point_data, prefix):
    """
    Flattens multi-dimensional point data by splitting into separate keys.

    Parameters:
        point_data (pyvista.core.pointset.PointSet.point_data): Point data from PolyData.
        prefix (str): Prefix to add to each key.

    Returns:
        dict: Flattened dictionary with prefixed keys.
    """
    flat_dict = {}
    for key in point_data.keys():
        data = point_data[key]
        if data.ndim == 1:
            new_key = f"{prefix}_{key}"
            flat_dict[new_key] = data
        elif data.ndim == 2:
            for dim in range(data.shape[1]):
                new_key = f"{prefix}_{key}_{dim}"
                flat_dict[new_key] = data[:, dim]
        else:
            logger.error(f"Unsupported data dimensionality for key '{key}': {data.ndim}")
            raise ValueError(f"Unsupported data dimensionality for key '{key}': {data.ndim}")
    return flat_dict

def load_polydata(site):
    """
    Loads 'site' and 'road' PolyData files and extracts point coordinates and point data.

    Parameters:
        site (str): Identifier for the site.

    Returns:
        dict: Dictionary containing 'site' and 'road' PolyData information.
    """
    sitePolyPath = f"data/revised/final/{site}-siteVoxels-masked.vtk"
    roadPolyPath = f"data/revised/final/{site}-roadVoxels-coloured.vtk"

    # Load PolyData
    site_pd = pv.read(sitePolyPath)
    logger.info(f"Loaded 'site' PolyData from: {sitePolyPath}")
    site_pd.point_data['isSite'] = True

    road_pd = pv.read(roadPolyPath)
    road_pd.point_data['isRoad'] = True
    logger.info(f"Loaded 'road' PolyData from: {roadPolyPath}")

    # Extract points
    site_points = site_pd.points  # Shape: (N_site, 3)
    road_points = road_pd.points  # Shape: (N_road, 3)

    # Extract point data and handle multi-dimensional data
    site_data_raw = polydata_to_flat_dict(site_pd.point_data, prefix='site')
    road_data_raw = polydata_to_flat_dict(road_pd.point_data, prefix='road')

    # Combine into a dictionary
    polydata = {
        'site': {
            'points': site_points,
            'point_data': site_data_raw
        },
        'road': {
            'points': road_points,
            'point_data': road_data_raw
        }
    }

    logger.info(f"Extracted {len(site_points)} 'site' points and {len(road_points)} 'road' points.")

    return polydata

def get_overall_bounds(poly_dict):
    """
    Computes the overall spatial bounds for a dictionary of PolyData objects.

    Parameters:
        poly_dict (dict): Dictionary containing PolyData information.

    Returns:
        tuple: Overall bounds as (xmin, xmax, ymin, ymax, zmin, zmax).
    """
    bounds = np.array([[
        np.inf, -np.inf,
        np.inf, -np.inf,
        np.inf, -np.inf
    ]])

    for key, pdata in poly_dict.items():
        poly = pdata['points']
        current_bounds = np.array([
            poly[:,0].min(), poly[:,0].max(),
            poly[:,1].min(), poly[:,1].max(),
            poly[:,2].min(), poly[:,2].max()
        ])
        bounds = np.vstack([bounds, current_bounds])

    overall_bounds = np.array([
        bounds[:,0].min(),
        bounds[:,1].max(),
        bounds[:,2].min(),
        bounds[:,3].max(),
        bounds[:,4].min(),
        bounds[:,5].max()
    ])

    logger.info(f"Overall bounds: {overall_bounds}")
    return tuple(overall_bounds)

def assign_voxel_indices(points, bounds, voxel_size=1.0):
    """
    Assigns voxel grid indices (voxel_I, voxel_J, voxel_K) to each point.

    Parameters:
        points (np.ndarray): Array of point coordinates.
        bounds (tuple): Spatial bounds as (xmin, xmax, ymin, ymax, zmin, zmax).
        voxel_size (float): Size of each voxel in meters.

    Returns:
        np.ndarray: Array of voxel indices with shape (N_points, 3).
    """
    xmin, ymin, zmin = bounds[0], bounds[2], bounds[4]
    voxel_indices = np.floor((points - [xmin, ymin, zmin]) / voxel_size).astype(int)
    logger.info(f"Assigned voxel indices to all points.")
    return voxel_indices

def calculate_voxel_centroids(voxel_indices, bounds, voxel_size=1.0):
    """
    Calculates voxel centroids based on voxel indices.

    Parameters:
        voxel_indices (np.ndarray): Array of voxel indices with shape (N_voxels, 3).
        bounds (tuple): Spatial bounds as (xmin, xmax, ymin, ymax, zmin, zmax).
        voxel_size (float): Size of each voxel in meters.

    Returns:
        np.ndarray: Array of voxel centroids with shape (N_voxels, 3).
    """
    xmin, ymin, zmin = bounds[0], bounds[2], bounds[4]
    centroids = (voxel_indices + 0.5) * voxel_size + np.array([xmin, ymin, zmin])
    logger.info(f"Calculated voxel centroids.")
    return centroids

def create_xarray_dataset(centroids, data_dict, bounds, voxel_size):
    """
    Creates an xarray Dataset from voxel centroids and associated data.
    Stores bounds and voxel size as attributes for future use.

    Parameters:
        centroids (np.ndarray): Array of voxel centroids with shape (N_voxels, 3).
        data_dict (dict): Dictionary containing voxel properties.
        bounds (tuple): Overall bounds of the voxel grid.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        xr.Dataset: The created xarray Dataset with bounds and voxel size as attributes.
    """
    num_voxels = centroids.shape[0]
    voxel_indices = np.arange(num_voxels)

    ds = xr.Dataset({
        'voxel': ('voxel', voxel_indices),
        'centroid_x': ('voxel', centroids[:, 0]),
        'centroid_y': ('voxel', centroids[:, 1]),
        'centroid_z': ('voxel', centroids[:, 2]),
        'voxel_I': ('voxel', data_dict['voxel_I']),
        'voxel_J': ('voxel', data_dict['voxel_J']),
        'voxel_K': ('voxel', data_dict['voxel_K']),
    })

    # Add additional data variables
    for key, values in data_dict.items():
        if key in ['voxel_I', 'voxel_J', 'voxel_K']:
            continue  # Already added
        ds[key] = ('voxel', values)

    # Store bounds and voxel size as attributes
    ds.attrs['bounds'] = bounds
    ds.attrs['voxel_size'] = voxel_size

    logger.info(f"Created xarray Dataset with {len(ds.data_vars)} variables, voxel bounds, and voxel size.")
    return ds

def assign_has_voxels_below(ds):
    """
    Assigns 'hasVoxelsBelow' and 'isTerrainUnderBuilding' flags to each voxel in the Dataset.
    
    Parameters:
        ds (xr.Dataset): The voxelized xarray Dataset.
    
    Returns:
        xr.Dataset: Updated Dataset with 'hasVoxelsBelow' and 'isTerrainUnderBuilding' variables.
    """
    df = ds.to_dataframe().reset_index()
    grouped = df.groupby(['voxel_I', 'voxel_J'])
    
    # Assign 'hasVoxelsBelow': True for max K in groups with size >1
    df['hasVoxelsBelow'] = (df['voxel_K'] == grouped['voxel_K'].transform('max')) & (grouped['voxel_K'].transform('size') > 1)
    
    # Assign 'isTerrainUnderBuilding': True for road voxels with any voxel_K > current K +1
    max_k = grouped['voxel_K'].transform('max')
    df['isTerrainUnderBuilding'] = (max_k > (df['voxel_K'] + 1)) & (df['road_isRoad'])
    
    # Assign the new variables back to the Dataset
    ds['hasVoxelsBelow'] = ('voxel', df['hasVoxelsBelow'].values)
    ds['isTerrainUnderBuilding'] = ('voxel', df['isTerrainUnderBuilding'].values)
    
    logger.info("Assigned 'hasVoxelsBelow' and 'isTerrainUnderBuilding' flags.")
    return ds

def voxelize_polydata_and_create_xarray(site, voxel_size=1.0):
    """
    Generates the xarray.Dataset from the polydata files with enhanced voxel assignment based on priorities.

    Parameters:
        site (str): Identifier for the site.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        xr.Dataset: The voxelized Dataset.
    """
    # Load PolyData
    poly_dict = load_polydata(site)

    # Get overall bounds
    bounds = get_overall_bounds(poly_dict)

    # Combine points from both 'site' and 'road'
    site_points = poly_dict['site']['points']
    road_points = poly_dict['road']['points']
    points_combined = np.vstack([site_points, road_points])  # Shape: (N_total, 3)

    # Combine point data with prefixed keys
    all_data_keys = list(poly_dict['site']['point_data'].keys()) + list(poly_dict['road']['point_data'].keys())
    all_data_keys = list(set(all_data_keys))  # Ensure uniqueness
    print(f'all_data_keys: {all_data_keys}')

    # Initialize data dictionary with default values (NaN or object)
    data_combined = {}
    N_total = points_combined.shape[0]
    for key in all_data_keys:
        # Determine data type based on availability in 'site' or 'road'
        if key in poly_dict['site']['point_data']:
            dtype = poly_dict['site']['point_data'][key].dtype
        elif key in poly_dict['road']['point_data']:
            dtype = poly_dict['road']['point_data'][key].dtype
        else:
            dtype = float  # Default data type

        # Assign appropriate dtype: use object for non-numeric data
        if np.issubdtype(dtype, np.number):
            data_combined[key] = np.full(N_total, np.nan, dtype=np.float32)
        else:
            data_combined[key] = np.full(N_total, None, dtype=object)

    # Assign 'site' data
    site_num_points = site_points.shape[0]
    for key in poly_dict['site']['point_data'].keys():
        data_combined[key][:site_num_points] = poly_dict['site']['point_data'][key]

    # Assign 'road' data
    road_num_points = road_points.shape[0]
    for key in poly_dict['road']['point_data'].keys():
        data_combined[key][site_num_points:] = poly_dict['road']['point_data'][key]

    logger.info(f"Combined point data from 'site' and 'road'.")

    # Assign voxel indices
    voxel_indices = assign_voxel_indices(points_combined, bounds, voxel_size=voxel_size)

    # Create DataFrame with voxel indices
    voxel_properties_df = pd.DataFrame(voxel_indices, columns=['voxel_I', 'voxel_J', 'voxel_K'])

    # Add other data columns
    for key, array in data_combined.items():
        voxel_properties_df[key] = array

    # ===============================
    # Define priority mappings once
    # ===============================
    priority_mappings = {
        'road_terrainInfo_forest': [1],  # High priority for forest terrain
        'site_greenRoof_ratingInt': [5, 4, 3, 2, 1],  # Green roof ratings, highest first
        'site_brownRoof_ratingInt': [5, 4, 3, 2, 1],  # Brown roof ratings, highest first
        'road_roadInfo_type': ['Footway','Tramway','Median'],  # Priority for road type Carriageway
        'road_terrainInfo_roadCorridors_str_type': ['Council Minor','Private'],  # Priority for Council Major road corridors
        'road_terrainInfo_isOpenSpace': [1.0],  # Open space has the highest priority
        'road_terrainInfo_isParking': [1.0],  # Parking areas
        'road_terrainInfo_isLittleStreet': [1.0],  # Little streets
        'road_terrainInfo_isParkingMedian3mBuffer': [1.0],  # Parking median buffer
        'site_canopy_isCanopy': [1.0],  # Canopy has the highest priority
        'road_canopy_isCanopy': [1.0],  # Canopy has the highest priority
    }

    # ===============================
    # Check for missing columns and initialize them
    # ===============================
    for key in priority_mappings.keys():
        if key not in voxel_properties_df.columns:
            # Check dtype of values in priority mappings and initialize accordingly
            sample_value = priority_mappings[key][0]
            if isinstance(sample_value, float):
                voxel_properties_df[key] = np.full(voxel_properties_df.shape[0], np.nan)
            elif isinstance(sample_value, int):
                voxel_properties_df[key] = np.full(voxel_properties_df.shape[0], -1)
            elif isinstance(sample_value, bool):
                voxel_properties_df[key] = np.full(voxel_properties_df.shape[0], False)
            elif isinstance(sample_value, str):
                voxel_properties_df[key] = np.full(voxel_properties_df.shape[0], 'unassigned')

    # ===============================
    # Begin: Priority-Based Selection
    # ===============================
    # Assign priority values for the columns in priority_mappings
    for col, priorities in priority_mappings.items():
        priority_col = f"{col}_priority"
        voxel_properties_df[priority_col] = voxel_properties_df[col].map({val: i for i, val in enumerate(priorities)}).fillna(len(priorities))
    
    # Sort by voxel indices and priority columns
    sort_columns = ['voxel_I', 'voxel_J', 'voxel_K'] + [f"{col}_priority" for col in priority_mappings]
    voxel_properties_df = voxel_properties_df.sort_values(by=sort_columns)
    
    # Group by voxel indices to ensure one row per voxel, taking the first (highest priority)
    voxel_properties_df_grouped = voxel_properties_df.groupby(['voxel_I', 'voxel_J', 'voxel_K'], as_index=False).first()
    
    # Drop priority columns after grouping to clean up
    for col in priority_mappings.keys():
        priority_col = f"{col}_priority"
        if priority_col in voxel_properties_df_grouped.columns:
            voxel_properties_df_grouped.drop(columns=[priority_col], inplace=True)
    # ===============================
    # End: Priority-Based Selection
    # ===============================

    # Validate uniqueness after grouping
    validate_voxel_properties_df(voxel_properties_df_grouped)

    # Calculate voxel centroids
    centroids = calculate_voxel_centroids(voxel_properties_df_grouped[['voxel_I', 'voxel_J', 'voxel_K']].values, bounds, voxel_size=voxel_size)

    # Create xarray Dataset
    data_dict = {
        'voxel_I': voxel_properties_df_grouped['voxel_I'].values,
        'voxel_J': voxel_properties_df_grouped['voxel_J'].values,
        'voxel_K': voxel_properties_df_grouped['voxel_K'].values
    }

    # Add all other columns to data_dict
    for column in voxel_properties_df_grouped.columns:
        if column not in ['voxel_I', 'voxel_J', 'voxel_K']:
            data_dict[column] = voxel_properties_df_grouped[column].values

    ds = create_xarray_dataset(centroids, data_dict, bounds, voxel_size)

    ds = assign_has_voxels_below(ds)

    return ds




# ================================
# ADD RESOURCE DATAFRAMES
# ================================


import logging
import pandas as pd
import numpy as np
import xarray as xr

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def filter_resources_within_bounds(ds, resource_df):
    """
    Filters the resource DataFrame to include only points within the voxel grid bounds.

    Parameters:
        ds (xr.Dataset): The voxelized xarray Dataset containing voxel grid bounds.
        resource_df (pd.DataFrame): DataFrame containing resource data with 'x', 'y', 'z' columns.

    Returns:
        pd.DataFrame: Filtered resource DataFrame within voxel grid bounds.
    """
    overall_bounds = ds.attrs['bounds']
    xmin, xmax, ymin, ymax, zmin, zmax = overall_bounds

    resource_df_within = resource_df[
        (resource_df['x'] >= xmin) & (resource_df['x'] < xmax) &
        (resource_df['y'] >= ymin) & (resource_df['y'] < ymax) &
        (resource_df['z'] >= zmin) & (resource_df['z'] < zmax)
    ].copy()
    logger.info(f"Filtered resource DataFrame to {len(resource_df_within)} points within Dataset voxel grid bounds.")
    return resource_df_within

def assign_voxel_indices_to_resources(resource_df_within, ds, voxel_size):
    """
    Assigns voxel grid indices to each resource point.

    Parameters:
        resource_df_within (pd.DataFrame): Filtered resource DataFrame within voxel grid bounds.
        ds (xr.Dataset): The voxelized xarray Dataset containing voxel grid bounds.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        pd.DataFrame: Resource DataFrame with assigned voxel indices.
    """
    overall_bounds = ds.attrs['bounds']
    voxel_indices = assign_voxel_indices(resource_df_within[['x', 'y', 'z']].values, overall_bounds, voxel_size=voxel_size)
    resource_df_within['voxel_I'] = voxel_indices[:, 0]
    resource_df_within['voxel_J'] = voxel_indices[:, 1]
    resource_df_within['voxel_K'] = voxel_indices[:, 2]
    logger.info("Assigned voxel indices to resource points.")
    return resource_df_within

def assign_voxel_indices(points, bounds, voxel_size=1.0):
    """
    Assigns voxel grid indices (voxel_I, voxel_J, voxel_K) to each point.

    Parameters:
        points (np.ndarray): Array of point coordinates.
        bounds (tuple): Spatial bounds as (xmin, xmax, ymin, ymax, zmin, zmax).
        voxel_size (float): Size of each voxel in meters.

    Returns:
        np.ndarray: Array of voxel indices with shape (N_points, 3).
    """
    xmin, ymin, zmin = bounds[0], bounds[2], bounds[4]
    voxel_indices = np.floor((points - [xmin, ymin, zmin]) / voxel_size).astype(int)
    logger.info(f"Assigned voxel indices to all points.")
    return voxel_indices

def aggregate_resource_counts(resource_df_within):
    """
    Aggregates resource counts per voxel.

    Parameters:
        resource_df_within (pd.DataFrame): DataFrame containing resources within voxel grid bounds.

    Returns:
        pd.DataFrame: Aggregated resource DataFrame per voxel.
    """
    resource_prefix = 'resource_'
    resource_cols = [col for col in resource_df_within.columns if col.startswith(resource_prefix)]

    # Specify which columns to keep with 'first' aggregation
    columns_to_keep = ['nodeType'] + [col for col in resource_df_within.columns 
                                    if col not in ['x', 'y', 'z'] + resource_cols]

    voxelised_resource_df = resource_df_within.groupby(['voxel_I', 'voxel_J', 'voxel_K'], as_index=False).agg(
        {**{col: 'sum' for col in resource_cols},
         **{col: 'first' for col in columns_to_keep}}
    )
    logger.info("Aggregated resource counts per voxel.")
    return voxelised_resource_df

def rename_non_resource_columns(voxelised_resource_df):
    """
    Renames non-resource columns with 'forest_' prefix, except for columns in exception list.

    Parameters:
        voxelised_resource_df (pd.DataFrame): Aggregated resource DataFrame per voxel.

    Returns:
        pd.DataFrame: Renamed resource DataFrame.
    """
    resource_prefix = 'resource_'
    exception_list = ['nodeType', 'nodeTypeInt']
    non_resource_cols = [col for col in voxelised_resource_df.columns 
                        if col not in [col for col in voxelised_resource_df.columns if col.startswith(resource_prefix)] + 
                        ['voxel_I', 'voxel_J', 'voxel_K'] + 
                        exception_list]
    voxelised_resource_df.rename(columns={col: f'forest_{col}' for col in non_resource_cols}, inplace=True)
    logger.info(f"Renamed {len(non_resource_cols)} non-resource columns with 'forest_' prefix.")
    return voxelised_resource_df

def check_for_duplicate_voxels(voxelised_resource_df):
    """
    Checks for duplicate voxel indices.

    Parameters:
        voxelised_resource_df (pd.DataFrame): Aggregated and renamed resource DataFrame.

    Raises:
        ValueError: If duplicate voxel indices are found.
    """
    if voxelised_resource_df.duplicated(subset=['voxel_I', 'voxel_J', 'voxel_K']).any():
        duplicates = voxelised_resource_df[voxelised_resource_df.duplicated(subset=['voxel_I', 'voxel_J', 'voxel_K'], keep=False)]
        logger.error("Duplicated I, J, K combinations found in voxelised_resource_df.")
        print(duplicates)
        raise ValueError("Duplicated I, J, K combinations detected in voxelised_resource_df.")
    else:
        logger.info("No duplicate voxel indices found in voxelised_resource_df.")

def split_existing_and_new_voxels(voxelised_resource_df, ds):
    """
    Splits voxelised_resource_df into existing voxels (dfExisting) and new voxels (dfSparse).

    Parameters:
        voxelised_resource_df (pd.DataFrame): Aggregated and renamed resource DataFrame.
        ds (xr.Dataset): Existing xarray Dataset.

    Returns:
        tuple: (dfExisting, dfSparse) DataFrames.
    """
    existing_voxel_tuples = set(zip(ds['voxel_I'].values, ds['voxel_J'].values, ds['voxel_K'].values))
    voxel_tuples = list(zip(voxelised_resource_df['voxel_I'], voxelised_resource_df['voxel_J'], voxelised_resource_df['voxel_K']))
    mask_existing = np.array([voxel in existing_voxel_tuples for voxel in voxel_tuples])

    dfExisting = voxelised_resource_df[mask_existing].copy()
    dfSparse = voxelised_resource_df[~mask_existing].copy()

    logger.info(f"Identified {len(dfExisting)} existing voxels to update and {len(dfSparse)} new voxels to append.")
    return dfExisting, dfSparse

def update_existing_voxels(ds, dfExisting):
    """
    Updates data variables for existing voxels in the Dataset.

    Parameters:
        ds (xr.Dataset): The existing xarray Dataset.
        dfExisting (pd.DataFrame): DataFrame containing updated data for existing voxels.

    Returns:
        xr.Dataset: Updated xarray Dataset.
    """
    if dfExisting.empty:
        logger.info("No existing voxels to update.")
        return ds  # No changes

    # Convert Dataset to DataFrame for merging
    ds_df = ds.to_dataframe().reset_index()

    # Merge existing Dataset DataFrame with dfExisting
    ds_df = pd.merge(
        ds_df,
        dfExisting,
        on=['voxel_I', 'voxel_J', 'voxel_K'],
        how='left',
        suffixes=('', '_new')
    )

    # Overwrite existing data with new data where available
    for column in dfExisting.columns:
        if column in ['voxel_I', 'voxel_J', 'voxel_K']:
            continue  # Skip voxel indices
        new_column = f"{column}_new"
        if new_column in ds_df.columns:
            # If the column is numeric, prefer new values; otherwise, handle accordingly
            if pd.api.types.is_numeric_dtype(ds_df[column].dtype):
                ds_df[column] = ds_df[new_column].combine_first(ds_df[column])
            else:
                ds_df[column] = ds_df[new_column].fillna(ds_df[column])
            ds_df.drop(columns=[new_column], inplace=True)

    # Convert back to xarray Dataset
    ds_final = xr.Dataset.from_dataframe(ds_df.set_index(['voxel']))
    logger.info("Updated existing voxels in the Dataset.")
    return ds_final

def prepare_new_voxel_dataframe(dfSparse, ds, voxel_size):
    """
    Prepares a DataFrame for new voxels with appropriate data variables.
    """
    # Extract the min bounds from the voxel grid
    xmin = ds['centroid_x'].min().item() - 0.5 * voxel_size
    ymin = ds['centroid_y'].min().item() - 0.5 * voxel_size
    zmin = ds['centroid_z'].min().item() - 0.5 * voxel_size

    # Calculate centroids for new voxels
    centroids = (dfSparse[['voxel_I', 'voxel_J', 'voxel_K']].values + 0.5) * voxel_size + np.array([xmin, ymin, zmin])

    # Create a new DataFrame for new voxels
    new_voxel_df = pd.DataFrame({
        'voxel_I': dfSparse['voxel_I'],
        'voxel_J': dfSparse['voxel_J'],
        'voxel_K': dfSparse['voxel_K'],
        'centroid_x': centroids[:, 0],
        'centroid_y': centroids[:, 1],
        'centroid_z': centroids[:, 2],
    })

    # Initialize existing variables with appropriate default values
    existing_vars = [var for var in ds.data_vars if var not in ['centroid_x', 'centroid_y', 'centroid_z', 'voxel_I', 'voxel_J', 'voxel_K']]

    for var in existing_vars:
        if var in dfSparse.columns:
            new_voxel_df[var] = dfSparse[var].values
        else:
            # Get the dtype from the existing dataset
            dtype = ds[var].dtype
            
            # Assign default values based on variable type
            if var.startswith('resource_'):
                new_voxel_df[var] = 0  # Initialize resource counts to 0
            elif np.issubdtype(dtype, np.floating):
                new_voxel_df[var] = np.nan  # Initialize float variables to NaN
            elif np.issubdtype(dtype, np.integer):
                new_voxel_df[var] = -1  # Initialize integer variables to -1
            elif dtype == bool:
                new_voxel_df[var] = False  # Initialize boolean variables to False
            elif dtype == 'O' or dtype.kind in ['U', 'S']:  # Object or string dtype
                new_voxel_df[var] = 'none'  # Initialize string variables to 'none'
            else:
                logger.warning(f"Unknown dtype {dtype} for variable {var}, initializing with 'none'")
                new_voxel_df[var] = 'none'  # Default to 'none' for unknown types

    logger.info(f"Prepared new voxels DataFrame with {len(new_voxel_df)} voxels.")
    return new_voxel_df


def verify_new_voxel_dataframe(new_voxel_df, ds, prefix=''):
    """
    Verifies that all required variables are present in new_voxel_df and have correct data types.
    If variables are missing or have incorrect data types, they are initialized or corrected 
    using the existing helper functions.
    
    Parameters:
        new_voxel_df (pd.DataFrame): DataFrame containing new voxels with data variables.
        ds (xr.Dataset): Existing xarray Dataset.
        prefix (str): Optional prefix to use for variable names in the xarray dataset.
    
    Raises:
        ValueError: If any required variable is missing and cannot be initialized.
    """
    # Identify the required variables from ds, excluding specified columns
    required_vars = [var for var in ds.data_vars if var not in ['voxel', 'voxel_I', 'voxel_J', 'voxel_K', 'centroid_x', 'centroid_y', 'centroid_z']]
    missing_vars = [var for var in required_vars if var not in new_voxel_df.columns]
    
    # If there are missing variables, initialize them using the helper function
    if missing_vars:
        logger.warning(f"Initializing missing variables in new_voxel_df: {missing_vars}")
        ds = a_helper_functions.initialize_xarray_variables_generic_auto(ds, new_voxel_df, missing_vars, prefix)
    
    # Ensure data types match and handle missing values
    for var in required_vars:
        expected_dtype = a_helper_functions.infer_dtype(ds[var])  # Use infer_dtype from a_helper_functions
        actual_dtype = a_helper_functions.infer_dtype(new_voxel_df[var])  # Infer dtype from new_voxel_df
        
        # If data types do not match, handle conversion or initialization
        if expected_dtype != actual_dtype:
            logger.warning(f"Type mismatch for '{var}': expected '{expected_dtype}', got '{actual_dtype}'")

            # Handle missing values for integer types by filling with defaults
            if expected_dtype == int and new_voxel_df[var].isnull().any():
                logger.info(f"Filling missing values in '{var}' with -1.")
                new_voxel_df[var].fillna(-1, inplace=True)
            
            # Attempt to convert the column to the expected dtype
            try:
                new_voxel_df[var] = new_voxel_df[var].astype(expected_dtype)
                logger.info(f"Converted '{var}' to dtype '{expected_dtype}'.")
            except Exception as e:
                logger.error(f"Error converting '{var}' to dtype '{expected_dtype}': {e}")
                raise
    
    return new_voxel_df

def append_new_voxels_to_dataset(new_voxel_df, ds_final, voxel_size):
    """
    Appends new voxels to the existing xarray.Dataset.

    Parameters:
        new_voxel_df (pd.DataFrame): DataFrame containing new voxels with data variables.
        ds_final (xr.Dataset): The existing xarray Dataset to append new voxels to.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        xr.Dataset: Updated xarray Dataset with new voxels appended.
    """
    # Assign unique 'voxel' indices by extending the existing ones
    last_voxel = ds_final['voxel'].max().item() if 'voxel' in ds_final.coords else -1
    new_voxel_indices = np.arange(last_voxel + 1, last_voxel + 1 + len(new_voxel_df))
    new_voxel_df['voxel'] = new_voxel_indices

    # Rearrange columns to have 'voxel' first
    cols = ['voxel'] + [col for col in new_voxel_df.columns if col != 'voxel']
    new_voxel_df = new_voxel_df[cols]

    # Convert new_voxel_df to xarray Dataset
    new_ds = xr.Dataset.from_dataframe(new_voxel_df.set_index(['voxel']))
    logger.info(f"Prepared new voxels Dataset with {len(new_ds['voxel'])} voxels.")

    # Verify that new_ds has all required variables
    missing_vars_new_ds = [var for var in ds_final.data_vars if var not in new_ds.data_vars]
    if missing_vars_new_ds:
        logger.error(f"Missing variables in new_ds: {missing_vars_new_ds}")
        raise ValueError(f"Missing variables in new_ds: {missing_vars_new_ds}")
    else:
        logger.info("All required variables are present in new_ds.")

    # Concatenate the new voxels to the existing Dataset
    ds_final = xr.concat([ds_final, new_ds], dim='voxel')
    logger.info(f"Appended {len(new_ds['voxel'])} new voxels to the Dataset.")

    # Log sample new voxels
    sample_new_voxel_df = new_voxel_df.head(5)
    logger.info(f"Sample of new voxels appended:\n{sample_new_voxel_df}")

    return ds_final

def integrate_resource_df_into_voxels(ds, resource_df, voxel_size=1.0):
    """
    Integrates the resource DataFrame into the existing voxelized xarray.Dataset.

    Parameters:
        ds (xr.Dataset): The voxelized xarray Dataset.
        resource_df (pd.DataFrame): DataFrame containing resource data with 'x', 'y', 'z' columns.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        tuple: Updated xarray.Dataset and voxelised_resource_df DataFrame.
    """
    # Step 1: Filter resources within voxel grid bounds
    resource_df_within = filter_resources_within_bounds(ds, resource_df)

    # Step 2: Assign voxel indices to resources
    resource_df_within = assign_voxel_indices_to_resources(resource_df_within, ds, voxel_size)

    # Step 3: Aggregate resource counts per voxel
    voxelised_resource_df = aggregate_resource_counts(resource_df_within)

    # Step 4: Rename non-resource columns
    voxelised_resource_df = rename_non_resource_columns(voxelised_resource_df)

    # Step 5: Check for duplicate voxels
    check_for_duplicate_voxels(voxelised_resource_df)

    # Step 6: Split into existing and new voxels
    dfExisting, dfSparse = split_existing_and_new_voxels(voxelised_resource_df, ds)

    # Step 7: Update existing voxels in the Dataset
    ds_final = update_existing_voxels(ds, dfExisting)

    # Step 8: Prepare new voxels DataFrame
    if not dfSparse.empty:
        new_voxel_df = prepare_new_voxel_dataframe(dfSparse, ds_final, voxel_size)
        # Step 9: Append new voxels to the Dataset
        ds_final = append_new_voxels_to_dataset(new_voxel_df, ds_final, voxel_size)
    else:
        logger.info("No new voxels to append.")

    return ds_final, voxelised_resource_df


# ================================
# VALIDATION AND VISUALISATION
# ================================


def check_unique_voxel_indices(ds):
    """
    Checks that each (voxel_I, voxel_J, voxel_K) combination is unique in the Dataset.

    Parameters:
        ds (xr.Dataset): The xarray Dataset to check.

    Raises:
        ValueError: If duplicate voxel indices are found.
    """
    voxel_tuples = list(zip(ds['voxel_I'].values, ds['voxel_J'].values, ds['voxel_K'].values))
    if len(voxel_tuples) != len(set(voxel_tuples)):
        logger.error("Duplicate (voxel_I, voxel_J, voxel_K) combinations found in the Dataset.")
        raise ValueError("Duplicate voxel indices detected.")
    logger.info("All (voxel_I, voxel_J, voxel_K) combinations are unique in the Dataset.")

def validate_resource_counts(ds_final, voxelised_resource_df, resource_count_vars):
    """
    Validates that resource counts are correctly assigned to their respective voxels.

    Parameters:
        ds_final (xr.Dataset): The final xarray Dataset after updating and appending voxels.
        voxelised_resource_df (pd.DataFrame): Voxelised Resource DataFrame with aggregated counts.
        resource_count_vars (list): List of resource count variable names in the Dataset.
    """
    # Convert ds_final to DataFrame for easy comparison
    ds_final_df = ds_final.to_dataframe().reset_index()

    # Merge voxelised_resource_df with ds_final_df on voxel_I, voxel_J, voxel_K
    merged_df = pd.merge(
        voxelised_resource_df,
        ds_final_df,
        on=['voxel_I', 'voxel_J', 'voxel_K'],
        how='left',
        suffixes=('_expected', '_actual')
    )

    # Iterate through each resource count variable to validate
    for resource_count_var in resource_count_vars:
        resource_base = resource_count_var.replace('_count', '')
        expected_col = resource_base
        actual_col = resource_count_var

        if expected_col not in merged_df.columns:
            logger.error(f"Expected column '{expected_col}' not found in the merged DataFrame.")
            raise ValueError(f"Expected column '{expected_col}' missing.")

        if actual_col not in merged_df.columns:
            logger.error(f"Actual column '{actual_col}' not found in the merged DataFrame.")
            raise ValueError(f"Actual column '{actual_col}' missing.")

        # Compare expected and actual counts
        mismatch = merged_df[actual_col] != merged_df[expected_col]
        if mismatch.any():
            num_mismatches = mismatch.sum()
            logger.error(f"{num_mismatches} mismatches found in {resource_count_var}.")
            # Optionally, print mismatched rows for debugging
            print(merged_df[mismatch][['voxel_I', 'voxel_J', 'voxel_K', expected_col, actual_col]])
        else:
            logger.info(f"All counts match for {resource_count_var}.")

def visualize_voxel_dataset(ds, site, voxel_size):
    """
    Visualizes the voxel dataset using PyVista and xarray integration.

    Parameters:
        ds (xr.Dataset): The voxelized Dataset to visualize.
        site (str): Identifier for the site.
        voxel_size (float): Size of each voxel in meters.
    """
    # Extract centroids
    centroids = np.vstack([ds['centroid_x'].values, ds['centroid_y'].values, ds['centroid_z'].values]).T

    # Create PyVista PolyData for visualization
    cloud = pv.PolyData(centroids)

    # Loop through all data variables in the xarray dataset and add them as point data
    for var_name in ds.data_vars:
        logger.info(f"Adding variable '{var_name}' to visualization.")
        values = ds[var_name].values

        # Check if the length matches
        if len(values) != len(cloud.points):
            logger.error(f"Variable '{var_name}' has length {len(values)} but expected {len(cloud.points)}.")
            raise ValueError(f"Variable '{var_name}' has length {len(values)} but expected {len(cloud.points)}.")

        # Add the variable as point data
        cloud.point_data[var_name] = values

    # Save PyVista PolyData
    output_path = f'data/revised/final/xarray_voxels_{site}_{voxel_size}.vtk'
    cloud.save(output_path)
    print(f"Dataset saved to: {output_path}")

    # Initialize PyVista Plotter
    plotter = pv.Plotter()

    plotter.add_points(
            cloud,
            scalars="isTerrainUnderBuilding",
            point_size=8,
            cmap="viridis",
            render_points_as_spheres=True,
        )
    plotter.show()


# Process logs
    

def integrate_resources_into_xarray(ds, treeLocationsDF, logLocationsDF=None, poleLocationsDF=None, valid_points=None):
    # Save original attributes
        voxel_size = ds.attrs['voxel_size']
        original_attrs = ds.attrs.copy()

        
        #process trees
        treeLocationsDF['nodeType'] = 'tree'
        treeLocationsDF, treeResource_df = a_resource_distributor_dataframes.process_all_trees(treeLocationsDF)
        print('rotating resource structures...')
        treeLocationsDF, treeResource_df = a_rotate_resource_structures.process_rotations(treeLocationsDF, treeResource_df, valid_points)
        print(f'resource_df columns are {treeResource_df.columns}')
        treeLocationsDF['nodeType'] = 'tree'
        treeResource_df['nodeType'] = 'tree'

        treeLocationsDF['nodeTypeInt'] = 1
        treeResource_df['nodeTypeInt'] = 1
        #set as default
        resource_DF = treeResource_df
        location_DF = treeLocationsDF

        if poleLocationsDF is not None and len(poleLocationsDF) > 0:
            print('processing poles')
            poleLocationsDF['nodeType'] = 'pole'
            poleLocationsDF, poleResourceDF = a_resource_distributor_dataframes.process_all_trees(poleLocationsDF)
            poleLocationsDF, poleResourceDF = a_rotate_resource_structures.process_rotations(poleLocationsDF, poleResourceDF, valid_points)

            poleLocationsDF['nodeTypeInt'] = 2
            poleResourceDF['nodeTypeInt'] = 2

            print(f'poleResourceDF columns are {poleResourceDF.columns}')
            print('poles')


        if logLocationsDF is not None and len(logLocationsDF) > 0:
            #process logs
            logLibrary = a_resource_distributor_dataframes.preprocess_logLibrary()
            logLocationsDF = a_resource_distributor_dataframes.preprocess_logLocationsDF(logLocationsDF, logLibrary)
            logResourceDF = a_resource_distributor_dataframes.create_log_resource_df(logLocationsDF, logLibrary)
            logLocationsDF, logResourceDF = a_rotate_resource_structures.process_rotations(logLocationsDF, logResourceDF, valid_points)
               
            logLocationsDF['nodeTypeInt'] = 3
            logResourceDF['nodeTypeInt'] = 3

            logLocationsDF['nodeType'] = 'log'
            logResourceDF['nodeType'] = 'log'


        
        # Prepare lists of dataframes to concatenate
        resource_DF = [treeResource_df]
        location_DF = [treeLocationsDF]

        if logLocationsDF is not None and len(logLocationsDF) > 0:
            resource_DF.append(logResourceDF)
            location_DF.append(logLocationsDF)

        if poleLocationsDF is not None and len(poleLocationsDF) > 0:
            resource_DF.append(poleResourceDF)
            location_DF.append(poleLocationsDF)

        # Concatenate the dataframes
        resource_DF = pd.concat(resource_DF, ignore_index=True)
        location_DF = pd.concat(location_DF, ignore_index=True)
        
        print_bounds(ds, resource_DF)

        # Integrate resource DataFrame into voxel Dataset
        ds, voxelised_resource_df = integrate_resource_df_into_voxels(ds, resource_DF, voxel_size=voxel_size)

        print('Integration complete!')

        # Restore the original attributes (bounds, voxel_size)
        ds.attrs.update(original_attrs)

        # Inspect the merged Dataset
        logger.info(f"Merged Dataset variables: {list(ds.data_vars.keys())}")
        logger.info(f"Merged Dataset dimensions: {ds.dims}")

        # Ensure that all variables have the same length as 'voxel' dimension
        num_voxels = len(ds['voxel'])
        for var in ds.data_vars:
            var_length = ds[var].sizes['voxel']
            if var_length != num_voxels:
                logger.error(f"Variable '{var}' has length {var_length}, expected {num_voxels}.")
            else:
                logger.info(f"Variable '{var}' has correct length {var_length}.")

        # Validate uniqueness of voxel indices
        check_unique_voxel_indices(ds)

        # Identify 'resource_' prefixed columns ending with '_count'
        resource_prefix = 'resource_'
        resource_count_vars = [var for var in ds.data_vars if var.startswith(resource_prefix) and var.endswith('_count')]

        # Validate resource counts
        validate_resource_counts(ds, voxelised_resource_df, resource_count_vars)

        # Final validation
        validate_integrated_dataset(ds, voxelised_resource_df, resource_count_vars)

        print(f'ds attributes are {ds.attrs}')  # Should display bounds and voxel_size before saving

        print(f'value counts of nodeType in location_DF: {location_DF["nodeType"].value_counts()}')

        return ds, location_DF



# ================================
# Entry Point
# ================================

def validate_integrated_dataset(ds_final, voxelised_resource_df, resource_count_vars):
    """
    Performs final validation on the integrated Dataset.

    Parameters:
        ds_final (xr.Dataset): The final xarray Dataset after integration.
        voxelised_resource_df (pd.DataFrame): Voxelised Resource DataFrame with aggregated counts.
        resource_count_vars (list): List of resource count variable names in the Dataset.
    """
    # Check total voxels
    total_voxels = len(ds_final['voxel'])
    logger.info(f"Total voxels after integration: {total_voxels}")

    # Ensure uniqueness
    check_unique_voxel_indices(ds_final)

    # Validate resource counts
    validate_resource_counts(ds_final, voxelised_resource_df, resource_count_vars)

def main():
    # Define your sites and parameters
    sites = ['city','trimmed-parade','uni']  # Replace with actual site identifiers
    sites = ['city']
    sites = ['trimmed-parade']
    voxel_size = 1

    for site in sites:
        filePATH = f'data/revised/final/{site}'
        os.makedirs(filePATH, exist_ok=True)

        print('Voxelizing site and road data...')
        ds = voxelize_polydata_and_create_xarray(site, voxel_size=voxel_size)
        ds.to_netcdf(f'{filePATH}/{site}_{voxel_size}_voxelArray_initial.nc')
        print(f'xarray saved to {filePATH}/{site}_{voxel_size}_voxelArray_initial.nc')

        """# Integrate resources into xarray
        from a_urban_forest_parser import get_resource_dataframe
        ds = get_resource_dataframe(site, ds)

        # Visualization and further processing...
        visualize_voxel_dataset(ds, site, voxel_size)

        logger.info(f"Voxelization and resource integration complete for site: {site}")

        print('Printing the voxel indices...')
        print(f"{ds['voxel_I']}")


        # Save ds to a NetCDF file
        output_path = f'data/revised/final/xarray_voxels{scenarioName}_{site}_{voxel_size}.nc'
        ds.to_netcdf(output_path)
        print(f"Dataset saved to: {output_path}")"""

if __name__ == "__main__":
    main()
