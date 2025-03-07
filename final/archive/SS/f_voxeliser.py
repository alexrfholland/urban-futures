import f_resource_distributor_dataframes
import pyvista as pv
import xarray as xr
import numpy as np
from pathlib import Path
import logging
import pandas as pd
import sparse  # Ensure 'sparse' is installed

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
# Helper Functions
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
    sitePolyPath = f"data/revised/{site}-siteVoxels-masked.vtk"
    roadPolyPath = f"data/revised/{site}-roadVoxels-coloured.vtk"

    # Load PolyData
    site_pd = pv.read(sitePolyPath)
    logger.info(f"Loaded 'site' PolyData from: {sitePolyPath}")

    road_pd = pv.read(roadPolyPath)
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

def voxelize_polydata(site, voxel_size=1.0):
    """
    Voxelizes 'site' and 'road' PolyData files and returns an xarray Dataset.

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

    # Group by voxel indices to ensure one row per voxel
    voxel_properties_df_grouped = voxel_properties_df.groupby(['voxel_I', 'voxel_J', 'voxel_K'], as_index=False).first()

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

    return ds

def ijk_to_centroid(i, j, k, ds):
    """
    Converts voxel grid indices (I, J, K) to the corresponding voxel centroid (x, y, z) coordinates
    using the bounds and voxel size stored in the xarray Dataset.

    Parameters:
        i (int): Voxel index along the I-axis.
        j (int): Voxel index along the J-axis.
        k (int): Voxel index along the K-axis.
        ds (xr.Dataset): xarray Dataset containing the voxel grid information, bounds, and voxel size.

    Returns:
        tuple: The corresponding centroid coordinates (x, y, z) for the voxel (i, j, k).
    """
    # Extract the bounds and voxel size from the dataset attributes
    bounds = ds.attrs['bounds']
    voxel_size = ds.attrs['voxel_size']
    
    # Unpack the bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    # Calculate the centroid by adding the half voxel size offset to the voxel indices
    x_centroid = xmin + (i + 0.5) * voxel_size
    y_centroid = ymin + (j + 0.5) * voxel_size
    z_centroid = zmin + (k + 0.5) * voxel_size
    
    return x_centroid, y_centroid, z_centroid


def map_xyz_to_ijk(resource_df, ds, voxel_size=1.0):
    """
    Maps spatial coordinates (x, y, z) in resource_df to voxel grid indices (I, J, K)
    based on the xarray Dataset's bounds and voxel size.

    Parameters:
        resource_df (pd.DataFrame): DataFrame with 'x', 'y', 'z' columns.
        ds (xr.Dataset): Existing xarray Dataset with voxel information.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        pd.DataFrame: Resource DataFrame with added 'voxel_I', 'voxel_J', 'voxel_K' columns.
    """
    # Extract voxel grid bounds from the Dataset
    xmin = ds['centroid_x'].min().item() - 0.5 * voxel_size
    ymin = ds['centroid_y'].min().item() - 0.5 * voxel_size
    zmin = ds['centroid_z'].min().item() - 0.5 * voxel_size

    # Assign voxel indices based on the voxel grid bounds
    resource_df['voxel_I'] = np.floor((resource_df['x'].values - xmin) / voxel_size).astype(int)
    resource_df['voxel_J'] = np.floor((resource_df['y'].values - ymin) / voxel_size).astype(int)
    resource_df['voxel_K'] = np.floor((resource_df['z'].values - zmin) / voxel_size).astype(int)

    logger.info(f"Mapped spatial coordinates to voxel indices using Dataset bounds.")

    return resource_df

def filter_resources_within_bounds(resource_df, ds, voxel_size=1.0):
    """
    Filters the resource DataFrame to include only points within the xarray Dataset's voxel grid bounds.

    Parameters:
        resource_df (pd.DataFrame): Resource DataFrame with 'x', 'y', 'z' columns.
        ds (xr.Dataset): The existing xarray Dataset with voxel information.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        pd.DataFrame: Filtered resource DataFrame within voxel grid bounds.
    """
    # Calculate voxel grid bounds based on voxel indices and voxel size
    xmin = ds['centroid_x'].min().item() - 0.5 * voxel_size
    xmax = ds['centroid_x'].max().item() + 0.5 * voxel_size
    ymin = ds['centroid_y'].min().item() - 0.5 * voxel_size
    ymax = ds['centroid_y'].max().item() + 0.5 * voxel_size
    zmin = ds['centroid_z'].min().item() - 0.5 * voxel_size
    zmax = ds['centroid_z'].max().item() + 0.5 * voxel_size

    # Filter resource_df to include only points within ds bounds
    within_bounds_mask = (
        (resource_df['x'] >= xmin) & (resource_df['x'] < xmax) &
        (resource_df['y'] >= ymin) & (resource_df['y'] < ymax) &
        (resource_df['z'] >= zmin) & (resource_df['z'] < zmax)
    )
    resource_df_within = resource_df[within_bounds_mask].copy()
    logger.info(f"Filtered resource DataFrame to {len(resource_df_within)} points within Dataset voxel grid bounds.")

    # Ensure that the DataFrame has 'x', 'y', 'z' columns
    required_columns = {'x', 'y', 'z'}
    if not required_columns.issubset(resource_df_within.columns):
        missing = required_columns - set(resource_df_within.columns)
        logger.error(f"Resource DataFrame is missing required columns: {missing}")
        raise ValueError(f"Resource DataFrame must contain columns: {required_columns}")

    return resource_df_within

def voxelise_resource_df(resource_df):
    """
    Groups the resource DataFrame by (voxel_I, voxel_J, voxel_K), sums resource_ columns,
    and takes the first row for other columns.

    Parameters:
        resource_df (pd.DataFrame): Resource DataFrame with 'voxel_I', 'voxel_J', 'voxel_K' and resource columns.

    Returns:
        pd.DataFrame: Voxelised Resource DataFrame with unique (voxel_I, voxel_J, voxel_K) combinations.
    """
    # Identify 'resource_' prefixed columns
    resource_prefix = 'resource_'
    resource_cols = [col for col in resource_df.columns if col.startswith(resource_prefix)]

    # Group by voxel indices
    voxelised_df = resource_df.groupby(['voxel_I', 'voxel_J', 'voxel_K'], as_index=False).agg(
        {**{col: 'sum' for col in resource_cols},
         **{col: 'first' for col in resource_df.columns if col not in ['x', 'y', 'z'] + resource_cols}}
    )
    
    # Rename columns that are not resource columns or voxel indices
    columns_to_rename = [col for col in voxelised_df.columns 
                         if col not in resource_cols + ['voxel_I', 'voxel_J', 'voxel_K']]
    
    rename_dict = {col: f'forest_{col}' for col in columns_to_rename}
    voxelised_df.rename(columns=rename_dict, inplace=True)
    
    logger.info(f"Renamed {len(rename_dict)} columns with 'forest_' prefix.")

    # Validate no duplicates
    if voxelised_df.duplicated(subset=['voxel_I', 'voxel_J', 'voxel_K']).any():
        duplicates = voxelised_df[voxelised_df.duplicated(subset=['voxel_I', 'voxel_J', 'voxel_K'], keep=False)]
        logger.error("Duplicated I, J, K combinations found in voxelised_resource_df.")
        print(duplicates)
        raise ValueError("Duplicated I, J, K combinations detected in voxelised_resource_df.")
    else:
        logger.info("voxelised_resource_df has unique I, J, K combinations after grouping.")

    return voxelised_df

def split_voxel_df(voxelised_df, ds):
    """
    Splits voxelised_df into existing voxels (dfExisting) and new voxels (dfSparse)
    based on whether their I, J, K coordinates exist in the existing Dataset 'ds'.

    Parameters:
        voxelised_df (pd.DataFrame): Voxelised Resource DataFrame with unique I, J, K.
        ds (xr.Dataset): Existing xarray Dataset.

    Returns:
        tuple: (dfExisting, dfSparse) DataFrames.
    """
    # Extract existing voxel indices as a set of tuples
    existing_voxel_tuples = set(zip(ds['voxel_I'].values, ds['voxel_J'].values, ds['voxel_K'].values))

    # Define a function to check if a voxel exists
    def is_existing(row):
        return (row['voxel_I'], row['voxel_J'], row['voxel_K']) in existing_voxel_tuples

    # Apply the function to split the DataFrame
    mask_existing = voxelised_df.apply(is_existing, axis=1)

    dfExisting = voxelised_df[mask_existing].copy()
    dfSparse = voxelised_df[~mask_existing].copy()

    logger.info(f"Identified {len(dfExisting)} existing voxels and {len(dfSparse)} new voxels.")

    # Log sample voxels
    if not dfSparse.empty:
        logger.info(f"Sample new voxels to append:\n{dfSparse.head()}")
    else:
        logger.warning("No new voxels identified to append.")

    return dfExisting, dfSparse

def update_existing_voxels(ds, dfExisting):
    """
    Updates existing voxels in 'ds' with data from 'dfExisting'.

    Parameters:
        ds (xr.Dataset): Existing xarray Dataset.
        dfExisting (pd.DataFrame): DataFrame containing updates for existing voxels.

    Returns:
        xr.Dataset: Updated xarray Dataset.
    """
    if dfExisting.empty:
        logger.info("No existing voxels to update.")
        return ds

    # Convert xarray Dataset to DataFrame
    ds_df = ds.to_dataframe().reset_index()

    # Merge with dfExisting on voxel_I, voxel_J, voxel_K to update existing voxels
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

    logger.info("Updated existing voxels in Dataset.")

    # Convert back to xarray Dataset with single 'voxel' dimension
    ds_updated = xr.Dataset.from_dataframe(ds_df.set_index(['voxel']))

    logger.info("Converted updated DataFrame back to xarray Dataset with single 'voxel' dimension.")
    return ds_updated

def append_new_voxels(ds, dfSparse, voxel_size=1.0):
    """
    Appends new voxels from 'dfSparse' to the Dataset.

    Parameters:
        ds (xr.Dataset): Existing xarray Dataset.
        dfSparse (pd.DataFrame): DataFrame containing new voxels to append.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        xr.Dataset: Updated xarray Dataset with new voxels appended.
    """
    if dfSparse.empty:
        logger.info("No new voxels to append.")
        return ds

    # Extract the min bounds from the voxel grid (assuming all voxels align)
    xmin = ds['centroid_x'].min().item() - 0.5 * voxel_size
    ymin = ds['centroid_y'].min().item() - 0.5 * voxel_size
    zmin = ds['centroid_z'].min().item() - 0.5 * voxel_size

    # Calculate centroids for new voxels based on their I, J, K indices
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

    # Identify 'resource_' prefixed columns
    resource_prefix = 'resource_'
    resource_cols = [col for col in dfSparse.columns if col.startswith(resource_prefix)]

    # Initialize existing variables with NaN or appropriate default values
    existing_vars = list(ds.data_vars.keys())
    # Exclude voxel indices and centroids
    existing_vars = [var for var in existing_vars if var not in ['centroid_x', 'centroid_y', 'centroid_z', 'voxel_I', 'voxel_J', 'voxel_K']]

    for var in existing_vars:
        if var in dfSparse.columns:
            new_voxel_df[var] = dfSparse[var].values
        else:
            # Initialize with 0 for resource counts, NaN for numeric vars, or None for object vars
            if var.startswith('resource_'):
                new_voxel_df[var] = 0
            elif np.issubdtype(ds[var].dtype, np.number):
                new_voxel_df[var] = np.nan
            else:
                new_voxel_df[var] = None

    logger.info(f"Prepared new voxels DataFrame with {len(new_voxel_df)} voxels.")

    # Assign unique 'voxel' indices by extending the existing ones
    last_voxel = ds['voxel'].max().item()
    new_voxel_indices = np.arange(last_voxel + 1, last_voxel + 1 + len(new_voxel_df))
    new_voxel_df['voxel'] = new_voxel_indices

    # Rearrange columns to have 'voxel' first
    cols = ['voxel'] + [col for col in new_voxel_df.columns if col != 'voxel']
    new_voxel_df = new_voxel_df[cols]

    # Convert new_voxel_df to xarray Dataset
    new_ds = xr.Dataset.from_dataframe(new_voxel_df.set_index(['voxel']))

    logger.info(f"Prepared new voxels Dataset with {len(new_ds['voxel'])} voxels.")

    # Concatenate the new voxels to the existing Dataset
    ds_final = xr.concat([ds, new_ds], dim='voxel')

    logger.info(f"Appended {len(new_ds['voxel'])} new voxels to the Dataset.")

    return ds_final


def validate_voxel_counts(ds_final, voxelised_resource_df, resource_count_vars):
    """
    Validates that the resource counts in the voxelGrid match the voxelised_resource_df.

    Parameters:
        ds_final (xr.Dataset): The final xarray Dataset after updating and appending voxels.
        voxelised_resource_df (pd.DataFrame): Voxelised Resource DataFrame with aggregated counts.
        resource_count_vars (list): List of resource count variable names in the Dataset.

    Raises:
        ValueError: If mismatches are found.
    """
    # Calculate total resource counts in voxelised_resource_df
    total_expected = voxelised_resource_df[[col.replace('_count', '') for col in resource_count_vars]].sum()

    # Calculate total resource counts in ds_final
    total_actual = ds_final[resource_count_vars].sum().values

    # Compare totals
    mismatches = total_expected.values != total_actual
    if mismatches.any():
        for idx, var in enumerate(resource_count_vars):
            if mismatches[idx]:
                logger.error(f"Total count mismatch for {var}: Expected {total_expected[idx]}, Got {total_actual[idx]}")
        raise ValueError("Total resource counts mismatch between Dataset and voxelised Resource DataFrame.")
    else:
        logger.info("Total resource counts match between Dataset and voxelised Resource DataFrame.")

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
            
def visualize_voxel_dataset(ds, site, voxel_size, plotter=None):
    """
    Visualizes the voxel dataset using PyVista and xarray integration.

    Parameters:
        ds (xr.Dataset): The voxelized Dataset to visualize.
        plotter (pv.Plotter, optional): Existing PyVista Plotter instance. Defaults to None.
    """
    # Extract centroids
    centroids = np.vstack([ds['centroid_x'].values, ds['centroid_y'].values, ds['centroid_z'].values]).T

    # Verify shape
    if centroids.shape[1] != 3:
        logger.error(f"Centroids array has incorrect shape: {centroids.shape}. Expected (N, 3).")
        raise ValueError(f"Centroids array has incorrect shape: {centroids.shape}. Expected (N, 3).")

    # Create PyVista PolyData for visualization
    cloud = pv.PolyData(centroids)

    # Loop through all data variables in the xarray dataset and add them as point data
    for var_name in ds.data_vars:
        if var_name in ['voxel', 'voxel_I', 'voxel_J', 'voxel_K']:
            continue  # Skip these variables
        logger.info(f"Adding variable '{var_name}' to visualization.")
        values = ds[var_name].values

        # Check if the length matches
        if len(values) != len(cloud.points):
            logger.error(f"Variable '{var_name}' has length {len(values)} but expected {len(cloud.points)}.")
            raise ValueError(f"Variable '{var_name}' has length {len(values)} but expected {len(cloud.points)}.")

        # Add the variable as point data
        cloud.point_data[var_name] = values

    # Save pyvista
    output_path = f'data/revised/xarray_voxels_{site}_{voxel_size}.vtk'
    cloud.save(output_path)
    print(f"Dataset saved to: {output_path}")

    if plotter is None:
        plotter = pv.Plotter()

    # Add scalar data (prefer resource counts)
    resource_scalar = None
    for var in ds.data_vars:
        if var.startswith('resource_') and var.endswith('_count'):
            resource_scalar = var
            break

    if resource_scalar:
        plotter.add_points(
            cloud,
            scalars=resource_scalar,
            point_size=8,
            cmap="viridis",
            render_points_as_spheres=True,
        )
    else:
        # If no resource counts, default to centroid_z
        plotter.add_points(
            cloud,
            scalars='resource_peeling bark',
            point_size=8,
            cmap="viridis",
            render_points_as_spheres=True,
        )

    # Add axes and show plot
    plotter.add_axes()
    plotter.show()

def automated_resource_count_check(ds_final, voxelised_resource_df, resource_count_vars):
    """
    Checks that the total resource counts in the Dataset match those in the voxelised Resource DataFrame.

    Parameters:
        ds_final (xr.Dataset): The final xarray Dataset after updating and appending voxels.
        voxelised_resource_df (pd.DataFrame): Voxelised Resource DataFrame with aggregated counts.
        resource_count_vars (list): List of resource count variable names in the Dataset.

    Raises:
        ValueError: If mismatches are found in total resource counts.
    """
    # Calculate total resource counts in voxelised_resource_df
    total_expected = voxelised_resource_df[[col.replace('_count', '') for col in resource_count_vars]].sum()

    # Calculate total resource counts in ds_final
    total_actual = ds_final[resource_count_vars].sum().values

    # Compare totals
    mismatches = total_expected.values != total_actual
    if mismatches.any():
        for idx, var in enumerate(resource_count_vars):
            if mismatches[idx]:
                logger.error(f"Total count mismatch for {var}: Expected {total_expected[idx]}, Got {total_actual[idx]}")
        raise ValueError("Total resource counts mismatch between Dataset and voxelised Resource DataFrame.")
    else:
        logger.info("Total resource counts match between Dataset and voxelised Resource DataFrame.")

def convert_to_ijk_grid(ds_final, voxel_size=1.0):
    """
    Converts a sparse voxel dataset with a single 'voxel' dimension into a 3D sparse grid with dimensions 'I', 'J', 'K'.

    Parameters:
        ds_final (xr.Dataset): xarray Dataset with single 'voxel' dimension and 'voxel_I', 'voxel_J', 'voxel_K' variables.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        xr.Dataset: xarray Dataset with dimensions 'I', 'J', 'K' and data variables as sparse arrays.
    """
    # Extract I, J, K indices
    I = ds_final['voxel_I'].values
    J = ds_final['voxel_J'].values
    K = ds_final['voxel_K'].values

    # Determine grid shape
    I_max = I.max()
    J_max = J.max()
    K_max = K.max()

    # Define grid dimensions
    dims = {'I': I_max + 1, 'J': J_max + 1, 'K': K_max + 1}

    # Define coordinates
    coords = {
        'I': np.arange(dims['I']),
        'J': np.arange(dims['J']),
        'K': np.arange(dims['K'])
    }

    # Initialize new Dataset
    voxelGrid = xr.Dataset(coords=coords)

    # Prepare indices for sparse arrays
    # **Corrected Here:** Shape should be (3, N)
    indices = np.vstack([I, J, K])  # Shape: (3, N)
    logger.info(f"Indices shape after stacking: {indices.shape}")  # Should be (3, N)

    # List of data variables to convert
    # Exclude 'voxel_I', 'voxel_J', 'voxel_K', 'centroid_x', 'centroid_y', 'centroid_z' as they are now dimensions or not needed
    data_vars = [var for var in ds_final.data_vars if var not in ['voxel_I', 'voxel_J', 'voxel_K', 
                                                                  'centroid_x', 'centroid_y', 'centroid_z']]

    # Separate numeric and object variables
    numeric_vars = [var for var in data_vars if np.issubdtype(ds_final[var].dtype, np.number)]
    object_vars = [var for var in data_vars if not np.issubdtype(ds_final[var].dtype, np.number)]

    # Process numeric variables
    if numeric_vars:
        for var in numeric_vars:
            data = ds_final[var].values

            # **Ensure that 'data' aligns with 'indices'**
            if len(data) != indices.shape[1]:
                logger.error(f"Data length for variable '{var}' ({len(data)}) does not match number of indices ({indices.shape[1]}).")
                raise ValueError(f"Data length for variable '{var}' does not match number of indices.")

            # Create sparse COO array with corrected indices
            sparse_array = sparse.COO(indices, data, shape=(dims['I'], dims['J'], dims['K']))

            # Assign to voxelGrid
            voxelGrid[var] = (('I', 'J', 'K'), sparse_array)

            logger.info(f"Converted numeric variable '{var}' to 3D sparse grid.")

    # Process object variables
    if object_vars:
        for var in object_vars:
            data = ds_final[var].values

            # **Replace NaNs and None with 'Unknown'**
            # Convert to pandas Series to handle replacement
            data_series = pd.Series(data).replace({None: 'Unknown'}).fillna('Unknown')
            data_clean = data_series.astype(str).values  # Ensure all data is string

            unique_vals, inverse = np.unique(data_clean, return_inverse=True)

            if len(unique_vals) <= 256:
                # Convert to categorical integers
                category_mapping = {val: idx for idx, val in enumerate(unique_vals)}
                categorical_data = inverse.astype(np.uint8)

                # Create sparse COO array
                sparse_array = sparse.COO(indices, categorical_data, shape=(dims['I'], dims['J'], dims['K']))

                # Assign to voxelGrid with categorical attribute
                voxelGrid[var] = (('I', 'J', 'K'), sparse_array)
                voxelGrid[var].attrs['categories'] = unique_vals.tolist()

                logger.info(f"Converted object variable '{var}' to 3D sparse grid with categorical encoding.")
            else:
                logger.warning(f"Object variable '{var}' has too many unique values ({len(unique_vals)}) to convert to categorical. Skipping.")

    logger.info("Conversion to 3D voxel grid complete.")
    return voxelGrid


# ================================
# Voxelization Process Functions
# ================================
def main():
    # Define your sites and parameters
    sites = ['uni']  # Replace with actual site identifiers
    voxel_size = 2.5
    scenarioName = 'original' 

    for site in sites:
        print('Voxelizing site and road data...')
        ds = voxelize_polydata(site, voxel_size=voxel_size)

        # Save original attributes
        original_attrs = ds.attrs.copy()

        print(f'Loading and processing resource DataFrame for site: {site}')
        filepath = f'data/revised/{site}-{scenarioName}-tree-locations.csv'
        treeLocationsDF = pd.read_csv(filepath)

        # Replace 'f_resource_distributor_dataframes.process_all_trees' with your actual processing function
        resource_df = f_resource_distributor_dataframes.process_all_trees(treeLocationsDF)

        print_bounds(ds, resource_df)

        # Filter resources within voxel grid bounds
        resource_df = filter_resources_within_bounds(resource_df, ds, voxel_size=voxel_size)

        # Assign voxel indices using the new mapping function
        resource_df = map_xyz_to_ijk(resource_df, ds, voxel_size=voxel_size)

        # Voxelise the resource DataFrame
        voxelised_resource_df = voxelise_resource_df(resource_df)

        # Check for duplicates in voxelised_resource_df
        if voxelised_resource_df.duplicated(subset=['voxel_I', 'voxel_J', 'voxel_K']).any():
            duplicates = voxelised_resource_df[voxelised_resource_df.duplicated(subset=['voxel_I', 'voxel_J', 'voxel_K'], keep=False)]
            logger.error("Duplicated I, J, K combinations found in voxelised_resource_df.")
            print(duplicates)
            raise ValueError("Duplicated I, J, K combinations detected in voxelised_resource_df.")
        else:
            logger.info("voxelised_resource_df has unique I, J, K combinations after grouping.")

        # Step 1: Split voxelised_resource_df into existing and new voxels based on ds
        print('Splitting voxelised Resource DataFrame into existing and new voxels...')
        dfExisting, dfSparse = split_voxel_df(voxelised_resource_df, ds)

        # Step 2: Update existing voxels in the Dataset
        print('Updating existing voxels in the Dataset...')
        ds = update_existing_voxels(ds, dfExisting)

        # Step 3: Append new voxels to the Dataset
        print('Appending new voxels to the Dataset...')
        ds = append_new_voxels(ds, dfSparse, voxel_size=voxel_size)

        print('Integration complete!')

        # **Restore the original attributes (bounds, voxel_size)**
        ds.attrs.update(original_attrs)

        # **Inspect the merged Dataset**
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

        print(f'ds attributes are {ds.attrs}')  # Should display bounds and voxel_size before saving

        # Visualization and further processing...
        visualize_voxel_dataset(ds, site, voxel_size)

        logger.info(f"Voxelization and resource integration complete for site: {site}")

        print('printing the voxel indicies...')
        print(f"{ds['voxel_I']}")

        # Save ds to a NetCDF file
        output_path = f'data/revised/xarray_voxels_{site}_{voxel_size}.nc'
        ds.to_netcdf(output_path)
        print(f"Dataset saved to: {output_path}")



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

# ================================
# Entry Point
# ================================

if __name__ == "__main__":
    main()
