import pyvista as pv
import pvxarray
import xarray as xr
import numpy as np
from pathlib import Path
import logging
import f_resource_distributor_dataframes
import pandas as pd
# ================================
# Setup Logging (Optional)
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# Step 1: Load PolyData Objects
# ================================

def print_bounds(ds, df):
    """
    Prints the bounds of the xarray dataset and the resource DataFrame.
    
    Parameters:
        ds (xr.Dataset): The xarray Dataset containing voxel data.
        df (pd.DataFrame): The resource DataFrame with 'x', 'y', 'z' columns.
    """
    # Get bounds from the xarray Dataset
    ds_bounds = (
        ds['centroid_x'].min().item(), ds['centroid_x'].max().item(),
        ds['centroid_y'].min().item(), ds['centroid_y'].max().item(),
        ds['centroid_z'].min().item(), ds['centroid_z'].max().item()
    )

    # Calculate bounds for the resource DataFrame
    df_bounds = (
        df['x'].min(), df['x'].max(),
        df['y'].min(), df['y'].max(),
        df['z'].min(), df['z'].max()
    )

    print(f"Dataset (xarray) bounds: xmin={ds_bounds[0]}, xmax={ds_bounds[1]}, "
          f"ymin={ds_bounds[2]}, ymax={ds_bounds[3]}, zmin={ds_bounds[4]}, zmax={ds_bounds[5]}")
    print(f"Resource DataFrame bounds: xmin={df_bounds[0]}, xmax={df_bounds[1]}, "
          f"ymin={df_bounds[2]}, ymax={df_bounds[3]}, zmin={df_bounds[4]}, zmax={df_bounds[5]}")

def load_polydata(site):
    """
    Loads 'site' and 'road' PolyData files and extracts point coordinates and point data.
    
    Parameters:
        site (str): Site identifier.
    
    Returns:
        dict: Dictionary containing 'site' and 'road' data with point coordinates and point data.
    """
    sitePolyPath = f"data/revised/{site}-siteVoxels-masked.vtk"
    roadPolyPath = f"data/revised/{site}-roadVoxels-coloured.vtk"

    # Load PolyData
    try:
        site_pd = pv.read(sitePolyPath)
        logger.info(f"Loaded 'site' PolyData from: {sitePolyPath}")
    except Exception as e:
        logger.error(f"Failed to read {sitePolyPath}: {e}")
        raise e

    try:
        road_pd = pv.read(roadPolyPath)
        logger.info(f"Loaded 'road' PolyData from: {roadPolyPath}")
    except Exception as e:
        logger.error(f"Failed to read {roadPolyPath}: {e}")
        raise e

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

def polydata_to_flat_dict(point_data, prefix):
    """
    Flattens multi-dimensional point data by splitting into separate keys.
    
    Parameters:
        point_data (pyvista.core.pointset.PointData): PointData object from PolyData.
        prefix (str): Prefix to add to each key to indicate source ('site' or 'road').
    
    Returns:
        dict: Flattened point data with prefixed keys.
    """
    flat_dict = {}
    for key in point_data.keys():
        data = point_data[key]
        if data.ndim == 1:
            # 1D data, simply prefix the key
            new_key = f"{prefix}_{key}"
            flat_dict[new_key] = data
        elif data.ndim == 2:
            # Multi-dimensional data, split into separate components
            for dim in range(data.shape[1]):
                new_key = f"{prefix}_{key}_{dim}"
                flat_dict[new_key] = data[:, dim]
        else:
            logger.error(f"Unsupported data dimensionality for key '{key}': {data.ndim}")
            raise ValueError(f"Unsupported data dimensionality for key '{key}': {data.ndim}")
    return flat_dict

# ================================
# Step 2: Get Overall Bounds
# ================================

def get_overall_bounds(poly_dict):
    """
    Computes the overall spatial bounds for a dictionary of PolyData objects.
    
    Parameters:
        poly_dict (dict): Dictionary with keys as identifiers ('site', 'road') and values as PolyData dicts.
    
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

# ================================
# Step 3: Assign Voxel Indices
# ================================

def assign_voxel_indices(points, bounds, voxel_size=1.0):
    """
    Assigns voxel grid indices (I, J, K) to each point.
    
    Parameters:
        points (np.ndarray): Array of point coordinates with shape (N, 3).
        bounds (tuple): Overall bounds as (xmin, xmax, ymin, ymax, zmin, zmax).
        voxel_size (float): Size of each voxel in meters.
    
    Returns:
        np.ndarray: Array of voxel indices with shape (N, 3).
    """
    xmin, ymin, zmin = bounds[0], bounds[2], bounds[4]
    voxel_indices = np.floor((points - [xmin, ymin, zmin]) / voxel_size).astype(int)
    logger.info(f"Assigned voxel indices to all points.")
    return voxel_indices

# ================================
# Step 4: Calculate Voxel Centroids
# ================================

def calculate_voxel_centroids(voxel_indices, bounds, voxel_size=1.0):
    """
    Calculates voxel centroids based on voxel indices.
    
    Parameters:
        voxel_indices (np.ndarray): Array of voxel indices with shape (N, 3).
        bounds (tuple): Overall bounds as (xmin, xmax, ymin, ymax, zmin, zmax).
        voxel_size (float): Size of each voxel in meters.
    
    Returns:
        np.ndarray: Array of voxel centroids with shape (N_unique, 3).
    """
    xmin, ymin, zmin = bounds[0], bounds[2], bounds[4]
    centroids = (voxel_indices + 0.5) * voxel_size + [xmin, ymin, zmin]
    logger.info(f"Calculated voxel centroids.")
    return centroids

# ================================
# Step 5: Select First Occurrence per Voxel
# ================================

def select_first_per_voxel(voxel_indices, data_dict):
    """
    Selects the first occurrence of data for each unique voxel.
    
    Parameters:
        voxel_indices (np.ndarray): Array of voxel indices with shape (N, 3).
        data_dict (dict): Dictionary of data arrays corresponding to each point.
    
    Returns:
        tuple: Unique voxel indices, corresponding data dictionaries, and unique keys.
    """
    # Create a unique key for each voxel by combining I, J, K
    # Assuming voxel grid dimensions do not exceed 1,000,000 per axis
    voxel_keys = voxel_indices[:,0] * 1_000_000 + voxel_indices[:,1] * 1_000 + voxel_indices[:,2]

    # Find unique voxels and their first occurrence indices
    unique_keys, unique_indices = np.unique(voxel_keys, return_index=True)

    # Sort unique_indices to preserve original ordering
    sorted_order = np.argsort(unique_indices)
    unique_indices = unique_indices[sorted_order]
    unique_keys = unique_keys[sorted_order]

    # Select unique voxel indices
    unique_voxel_indices = voxel_indices[unique_indices]

    # Select corresponding data for each unique voxel
    unique_data = {}
    for key, array in data_dict.items():
        # Ensure array is 1D or 2D
        if array.ndim == 1:
            unique_data[key] = array[unique_indices]
        elif array.ndim == 2:
            unique_data[key] = array[unique_indices, :]
        else:
            logger.error(f"Unsupported data dimensionality for key '{key}': {array.ndim}")
            raise ValueError(f"Unsupported data dimensionality for key '{key}': {array.ndim}")

    logger.info(f"Selected first occurrence for {len(unique_keys)} unique voxels.")

    return unique_voxel_indices, unique_data, unique_keys

# ================================
# Step 6: Create xarray Dataset
# ================================

def create_xarray_dataset(centroids, data_dict, voxel_size=1.0):
    """
    Creates an xarray Dataset from voxel centroids and associated data.

    Parameters:
        centroids (np.ndarray): Array of voxel centroids with shape (N, 3).
        data_dict (dict): Dictionary of data arrays to include in the Dataset.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        xr.Dataset: xarray Dataset containing voxelized data.
    """
    # Create DataArrays for centroids
    voxel = np.arange(len(centroids))
    centroid_x = xr.DataArray(centroids[:,0], dims="voxel", name="centroid_x")
    centroid_y = xr.DataArray(centroids[:,1], dims="voxel", name="centroid_y")
    centroid_z = xr.DataArray(centroids[:,2], dims="voxel", name="centroid_z")

    # Create a unique voxel key
    voxel_IJK = np.floor((centroids - [centroids[:,0].min(), centroids[:,1].min(), centroids[:,2].min()]) / voxel_size).astype(int)
    voxel_key = voxel_IJK[:,0] * 1_000_000 + voxel_IJK[:,1] * 1_000 + voxel_IJK[:,2]
    voxel_key = xr.DataArray(voxel_key, dims="voxel", name="voxel_key")

    # Initialize Dataset with centroids and voxel_key
    ds = xr.Dataset({
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'centroid_z': centroid_z,
        'voxel_key': voxel_key.values  # Extract raw data
    })

    # Add additional data variables
    for key, array in data_dict.items():
        ds[key] = xr.DataArray(array, dims="voxel", name=key)

    logger.info(f"Created xarray Dataset with {len(ds.data_vars)} variables.")
    return ds

# ================================
# Step 7: Voxelization Process
# ================================

def voxelize_polydata(site, voxel_size=1.0):
    """
    Voxelizes 'site' and 'road' PolyData files and returns an xarray Dataset.
    
    Parameters:
        site (str): Site identifier.
        voxel_size (float): Size of each voxel in meters.
    
    Returns:
        xr.Dataset: Voxelized Dataset containing centroids and point data.
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
    # Collect all unique data keys from both 'site' and 'road'
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

    # Select first occurrence per voxel
    unique_voxel_indices, unique_data, unique_keys = select_first_per_voxel(voxel_indices, data_combined)

    # Calculate voxel centroids
    centroids = calculate_voxel_centroids(unique_voxel_indices, bounds, voxel_size=voxel_size)

    # Create xarray Dataset
    ds = create_xarray_dataset(centroids, unique_data)

    return ds

def create_xarray_dataset(centroids, data_dict, voxel_size=1.0):
    """
    Creates an xarray Dataset from voxel centroids and associated data.

    Parameters:
        centroids (np.ndarray): Array of voxel centroids with shape (N, 3).
        data_dict (dict): Dictionary of data arrays to include in the Dataset.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        xr.Dataset: xarray Dataset containing voxelized data.
    """
    # Create DataArrays for centroids
    voxel = np.arange(len(centroids))
    centroid_x = xr.DataArray(centroids[:,0], dims="voxel", name="centroid_x")
    centroid_y = xr.DataArray(centroids[:,1], dims="voxel", name="centroid_y")
    centroid_z = xr.DataArray(centroids[:,2], dims="voxel", name="centroid_z")

    # Create a unique voxel key
    voxel_IJK = np.floor((centroids - [centroids[:,0].min(), centroids[:,1].min(), centroids[:,2].min()]) / voxel_size).astype(int)
    voxel_key = voxel_IJK[:,0] * 1_000_000 + voxel_IJK[:,1] * 1_000 + voxel_IJK[:,2]

    # Initialize Dataset with centroids and voxel_key
    ds = xr.Dataset({
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'centroid_z': centroid_z,
        'voxel_key': ('voxel', voxel_key)  # Assign as a coordinate
    })

    # Add additional data variables
    for key, array in data_dict.items():
        ds[key] = xr.DataArray(array, dims="voxel", name=key)

    logger.info(f"Created xarray Dataset with {len(ds.data_vars)} variables.")
    return ds

def voxelize_resources_by_calculating_ijk(df, ds, voxel_size=1.0):
    """
    Assigns (I, J, K) voxel indices for each point in the resource DataFrame based on the existing voxel grid.

    Parameters:
        df (pd.DataFrame): DataFrame containing resource data with 'x', 'y', 'z' columns.
        ds (xr.Dataset): The existing xarray Dataset with voxel information.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        pd.DataFrame: Resource data with assigned (I, J, K) indices for voxel alignment.
    """
    xmin = ds['centroid_x'].min().item()
    ymin = ds['centroid_y'].min().item()
    zmin = ds['centroid_z'].min().item()

    df['voxel_I'] = np.floor((df['x'].values - xmin) / voxel_size).astype(int)
    df['voxel_J'] = np.floor((df['y'].values - ymin) / voxel_size).astype(int)
    df['voxel_K'] = np.floor((df['z'].values - zmin) / voxel_size).astype(int)
    df['voxel_key'] = df['voxel_I'] * 1_000_000 + df['voxel_J'] * 1_000 + df['voxel_K']
    return df

def calculate_voxel_properties(voxelized_df):
    """
    Groups the DataFrame by voxel indices (I, J, K) and calculates the voxel properties.
    
    Parameters:
        voxelized_df (pd.DataFrame): DataFrame containing voxelized resource data with (I, J, K) indices.

    Returns:
        pd.DataFrame: DataFrame with aggregated voxel properties per (I, J, K) group.
    """
    # Separate resource and non-resource columns
    resource_columns = [col for col in voxelized_df.columns if col.startswith('resource_')]
    non_resource_columns = [col for col in voxelized_df.columns if col not in resource_columns + ['voxel_I', 'voxel_J', 'voxel_K', 'voxel_key']]

    # Group by voxel indices and aggregate resource columns using sum
    grouped_df = voxelized_df.groupby(['voxel_I', 'voxel_J', 'voxel_K']).agg({
        col: 'sum' for col in resource_columns
    }).reset_index()

    # For non-resource columns, take the first value in each group and add 'resource_' prefix
    first_values = voxelized_df.groupby(['voxel_I', 'voxel_J', 'voxel_K'])[non_resource_columns].first().reset_index()

    # Add the 'resource_' prefix to non-resource columns
    first_values = first_values.rename(columns={col: f'resource_{col}' for col in non_resource_columns})

    # Merge the aggregated resource columns and non-resource columns
    grouped_df = pd.merge(grouped_df, first_values, on=['voxel_I', 'voxel_J', 'voxel_K'])

    # Create the voxel_key again after grouping
    grouped_df['voxel_key'] = grouped_df['voxel_I'] * 1_000_000 + grouped_df['voxel_J'] * 1_000 + grouped_df['voxel_K']

    return grouped_df

# Modified transfer_to_xarray function to handle non-numeric columns
def transfer_to_xarray(ds, voxel_properties_df, voxel_size=1.0):
    """
    Transfers the voxel properties from the DataFrame to the xarray dataset, appending new voxels if needed.
    Optimized with vectorized operations to avoid iterating over rows. Handles both numeric and non-numeric data.
    
    Parameters:
        ds (xr.Dataset): The existing xarray Dataset.
        voxel_properties_df (pd.DataFrame): DataFrame with aggregated voxel properties.
        voxel_size (float): Size of each voxel in meters.

    Returns:
        xr.Dataset: Updated xarray Dataset with integrated voxel properties and new voxels (I, J, K).
    """
    for var_name in ds.data_vars:
        print(var_name)

    # Get current voxel keys in the dataset
    ds_voxel_keys = ds['voxel_key'].values

    # Identify new voxels not already in the dataset
    new_voxels = ~voxel_properties_df['voxel_key'].isin(ds_voxel_keys)

    if new_voxels.any():
        new_df = voxel_properties_df[new_voxels]

        # Create new entries for centroid_x, centroid_y, centroid_z based on voxel indices
        new_centroid_x = (new_df['voxel_I'] * voxel_size).values + ds['centroid_x'].min().item()
        new_centroid_y = (new_df['voxel_J'] * voxel_size).values + ds['centroid_y'].min().item()
        new_centroid_z = (new_df['voxel_K'] * voxel_size).values + ds['centroid_z'].min().item()

        # Manually concatenate the new voxel data
        ds = xr.Dataset({
            'centroid_x': xr.DataArray(np.concatenate([ds['centroid_x'].values, new_centroid_x]), dims="voxel"),
            'centroid_y': xr.DataArray(np.concatenate([ds['centroid_y'].values, new_centroid_y]), dims="voxel"),
            'centroid_z': xr.DataArray(np.concatenate([ds['centroid_z'].values, new_centroid_z]), dims="voxel"),
            'voxel_key': xr.DataArray(np.concatenate([ds_voxel_keys, new_df['voxel_key'].values]), dims="voxel")
        })

    # Update voxel keys
    ds_voxel_keys = ds['voxel_key'].values

    # Use vectorized indexing to match voxel keys in `voxel_properties_df` with `ds_voxel_keys`
    voxel_key_to_index = pd.Series(np.arange(len(ds_voxel_keys)), index=ds_voxel_keys)

    # Separate numeric and non-numeric resource columns
    resource_columns = [col for col in voxel_properties_df.columns if col.startswith('resource_')]
    numeric_columns = voxel_properties_df.select_dtypes(include=[np.number]).columns.intersection(resource_columns)
    non_numeric_columns = voxel_properties_df.select_dtypes(exclude=[np.number]).columns.intersection(resource_columns)

    # Integrate numeric resource columns into the dataset
    for col in numeric_columns:
        # Create an array of zeros for the full length of `ds_voxel_keys`
        data_array = np.zeros(len(ds_voxel_keys))

        # Use vectorized indexing to map voxel keys to their respective indices
        matching_indices = voxel_key_to_index[voxel_properties_df['voxel_key']].values

        # Populate the data_array at the correct indices
        data_array[matching_indices] = voxel_properties_df[col].values

        # Add the resource data to the dataset
        ds[col] = xr.DataArray(
            data_array,
            dims="voxel",
            coords={"voxel": np.arange(len(data_array))},
            name=col
        )

    # Integrate non-numeric resource columns into the dataset
    for col in non_numeric_columns:
        # Create an array of empty strings (or None if needed)
        data_array = np.full(len(ds_voxel_keys), '', dtype=object)

        # Use vectorized indexing to map voxel keys to their respective indices
        matching_indices = voxel_key_to_index[voxel_properties_df['voxel_key']].values

        # Populate the data_array at the correct indices
        data_array[matching_indices] = voxel_properties_df[col].values

        # Add the non-numeric resource data to the dataset
        ds[col] = xr.DataArray(
            data_array,
            dims="voxel",
            coords={"voxel": np.arange(len(data_array))},
            name=col
        )

    logger.info("Voxel properties and new voxels successfully transferred to the xarray dataset with vectorized operations.")
    for var_name in ds.data_vars:
        print(var_name)
    
    return ds





def visualize_voxel_dataset(ds, plotter=None):
    """
    Visualizes the voxel dataset using PyVista and xarray integration.

    Parameters:
        ds (xr.Dataset): xarray Dataset containing voxel data.
        plotter (pv.Plotter): The PyVista plotter used for visualization.
    """
    # Extract centroids
    centroids = np.vstack([ds['centroid_x'].values, ds['centroid_y'].values, ds['centroid_z'].values]).T

    # Create PyVista PolyData for visualization
    cloud = pv.PolyData(centroids)

    # Loop through all data variables in the xarray dataset and add them as point data
    for var_name in ds.data_vars:
        print(var_name)
        values = ds[var_name].values
        cloud.point_data[var_name] = values  # Add each variable to the PolyData as point data

    if plotter is None:
        plotter = pv.Plotter()
    # Add scalar data (you can choose one variable to be visualized as a scalar, e.g., 'resource_size')
    plotter.add_points(cloud, scalars='resource_useful_life_expectency', point_size=8, cmap="viridis", render_points_as_spheres=True)
    
    # Add axes and show plot
    plotter.add_axes()
    plotter.show()

def main():
    # Define your sites and parameters
    sites = ['trimmed-parade']  # Replace with actual site identifiers
    voxel_size = 5.0
    scenarioName = 'original' 

    for site in sites:
        print('Voxelizing site and road data...')
        ds = voxelize_polydata(site, voxel_size=voxel_size)

        print(f'Loading and processing resource DataFrame for site: {site}')
        filepath = f'data/revised/{site}-{scenarioName}-tree-locations.csv'
        treeLocationsDF = pd.read_csv(filepath)
        resource_df = f_resource_distributor_dataframes.process_all_trees(treeLocationsDF)

        # Save the dataset and resource DataFrame to specified paths
        voxel_test_output_path = Path(f"data/revised/_voxeltest_voxels.nc")
        resource_df_output_path = Path(f"data/revised/_voxeltest_resourcrDF.csv")

        # Save the dataset to NetCDF
        ds.to_netcdf(voxel_test_output_path)
        logger.info(f"Dataset saved to {voxel_test_output_path}")

        # Save the resource DataFrame to CSV
        resource_df.to_csv(resource_df_output_path, index=False)
        logger.info(f"Resource DataFrame saved to {resource_df_output_path}")
        
        print_bounds(ds, resource_df)

        # Step 1: Calculate voxel properties by grouping (I, J, K)
        print('Find i j k indices of df...')
        resource_df = voxelize_resources_by_calculating_ijk(resource_df, ds, voxel_size)

        print('Calculating voxel properties by (I, J, K) grouping...')
        voxel_properties_df = calculate_voxel_properties(resource_df)

        # Step 2: Transfer voxel properties to the xarray dataset and append missing voxels
        print('Transferring voxel properties to the xarray dataset and appending new voxels...')
        ds = transfer_to_xarray(ds, voxel_properties_df, voxel_size=voxel_size)

        print('Integration complete!')
        # Visualization and further processing...
        visualize_voxel_dataset(ds)  # Example with resource

        # Optionally, save to NetCDF or any other format
        # output_path = Path(f"data/revised/{site}-voxelized-with-resources.nc")
        # ds.to_netcdf(output_path)

        logger.info(f"Voxelization and resource integration complete for site: {site}")

if __name__ == "__main__":
    main()