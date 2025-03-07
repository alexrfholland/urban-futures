import pyvista as pv
import pvxarray
import xarray as xr
import numpy as np
from pathlib import Path
import logging

# ================================
# Setup Logging (Optional)
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# Step 1: Load PolyData Objects
# ================================

def visualize_voxel_dataset(ds, variable=None):
    """
    Visualizes the voxel dataset using PyVista and xarray integration.
    
    Parameters:
        ds (xr.Dataset): xarray Dataset containing voxel data.
        variable (str): The variable to visualize. If None, the first data variable is used.
    """
    # Select the variable to visualize
    if variable is None:
        variable = list(ds.data_vars)[0]  # Use the first variable if none is specified
    
    # Extract centroids and the selected variable data
    centroids = np.vstack([ds['centroid_x'].values, ds['centroid_y'].values, ds['centroid_z'].values]).T
    values = ds[variable].values
    
    # Create PyVista PolyData for visualization
    cloud = pv.PolyData(centroids)
    
    # Add scalar data (the selected variable) for coloring
    cloud[variable] = values
    
    # Create a plotter and visualize the voxel data
    plotter = pv.Plotter()
    plotter.add_points(cloud, scalars=variable, point_size=8, cmap="viridis", render_points_as_spheres=True)
    plotter.add_axes()
    plotter.show()


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

def create_xarray_dataset(centroids, data_dict):
    """
    Creates an xarray Dataset from voxel centroids and associated data.
    
    Parameters:
        centroids (np.ndarray): Array of voxel centroids with shape (N, 3).
        data_dict (dict): Dictionary of data arrays to include in the Dataset.
    
    Returns:
        xr.Dataset: xarray Dataset containing voxelized data.
    """
    # Create DataArrays for centroids
    voxel = np.arange(len(centroids))
    centroid_x = xr.DataArray(centroids[:,0], dims="voxel", name="centroid_x")
    centroid_y = xr.DataArray(centroids[:,1], dims="voxel", name="centroid_y")
    centroid_z = xr.DataArray(centroids[:,2], dims="voxel", name="centroid_z")

    # Initialize Dataset with centroids
    ds = xr.Dataset({
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'centroid_z': centroid_z
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

# ================================
# Step 8: Main Function
# ================================

def main():
    """
    Main function to perform voxelization on specified sites.
    """
    # Define sites to process
    sites = ['trimmed-parade']  # Replace with your actual site identifiers

    # Define voxel size (1 meter)
    voxel_size = 1.0

    # Process each site
    for site in sites:
        logger.info(f"Starting voxelization for site: {site}")
        ds = voxelize_polydata(site, voxel_size=voxel_size)
        logger.info(f"Voxelization complete for site: {site}")

        # Print all the variables in the dataset
        print("Variables in the dataset:")
        for var in ds.data_vars:
            print(var)

        #ds.pyvista.plot(x="centroid_x", y="centroid_y",z="centroid_z", show_edges=True, cpos='xy')

        visualize_voxel_dataset(ds, variable='site_building_normalZ')  # Replace 'site_source' with the actual variable


        
        # Define output path
        #output_path = Path(f"data/revised/{site}-voxelized.nc")

        # Save Dataset to NetCDF file
        #ds.to_netcdf(output_path)
        #logger.info(f"Saved voxelized data to: {output_path}\n")

        # Optional: Visualize a subset using xarray's plotting capabilities or other visualization tools
        # Example:
        # visualize_voxel_dataset(ds, variable='site_source')

# ================================
# Execute the Script
# ================================

if __name__ == "__main__":
    main()
