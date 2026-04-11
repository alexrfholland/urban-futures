# Encapsulating all the steps into functions

import xarray as xr
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
import colorcet as cc
from scipy.spatial import cKDTree

# Function to concatenate datasets through pandas
def concatenate_datasets_through_pandas(ds1, ds2):
    df1 = ds1.to_dataframe().reset_index()
    df2 = ds2.to_dataframe().reset_index()

    all_columns = set(df1.columns).union(set(df2.columns))
    for col in all_columns:
        if col not in df1.columns:
            df1[col] = np.nan
        if col not in df2.columns:
            df2[col] = np.nan

    concatenated_df = pd.concat([df1, df2], ignore_index=True)
    final_xr = concatenated_df.set_index('point_index').to_xarray()
    final_xr = final_xr.assign_coords(point_index=concatenated_df.index)

    return final_xr

# Function to apply the updated weighting logic
def GetAborealResourceWeights(data):
    # Initialize the weighted sum to zero
    weighted_sum = np.zeros(data.dims['point_index'])

    # List of resources with their respective multipliers
    resources = [
        ('resource_dead branch', 2),
        ('resource_peeling bark', 2),
        ('resource_perchable branch', 2),
        ('resource_fallen log', 2),
        ('resource_leaf litter', 2),
        ('resource_other', 1)
    ]
    
    # Add the weights of each resource if the column exists
    for resource_name, multiplier in resources:
        if resource_name in data:
            resource_values = np.nan_to_num(data[resource_name].values, nan=0)
            weighted_sum += resource_values * multiplier

    # Apply negative weighting if 'road_types-type' is 'Carriageway'
    if 'road_types-type' in data:
        road_types_type = data['road_types-type'].fillna(False).values
        weighted_sum -= (road_types_type == 'Carriageway') * 100
    
    return weighted_sum

def GetAborealResourceWeightsOLD(data):
    
    
    # Treat NaNs in numeric columns as 0
    resource_dead_branch = np.nan_to_num(data['resource_dead branch'].values, nan=0)
    resource_peeling_bark = np.nan_to_num(data['resource_peeling bark'].values, nan=0)
    resource_perchable_branch = np.nan_to_num(data['resource_perchable branch'].values, nan=0)
    resource_fallen_log = np.nan_to_num(data['resource_fallen log'].values, nan=0)
    resource_leaf_liftter = np.nan_to_num(data['resource_leaf litter'].values, nan=0)
    resource_other = np.nan_to_num(data['resource_other'].values, nan=0)

    # Treat NaNs in boolean columns as False
    road_types_type = data['road_types-type'].fillna(False).values if 'road_types-type' in data else np.zeros_like(resource_dead_branch)

    # Calculate the weighted sum
    weighted_sum = (
        (resource_dead_branch + resource_peeling_bark + resource_perchable_branch + resource_fallen_log + resource_leaf_liftter) * 2 +
        resource_other
    )
    
    # Apply negative weighting if 'road_types-type' is 'Carriageway'
    weighted_sum -= (road_types_type == 'Carriageway') * 100
    
    return weighted_sum


def GetPylonWeights(data):
    # Ensure 'ispylon' column is treated as a boolean and convert to integer weights
    ispylon = data['ispylons'].fillna(False).astype(int).values
    return ispylon

# Gaussian kernel function
def gaussian_kernel_3d(size_x, size_y, size_z, sigma_x, sigma_y, sigma_z):
    ax_x = np.linspace(-size_x // 2, size_x // 2, size_x)
    ax_y = np.linspace(-size_y // 2, size_y // 2, size_y)
    ax_z = np.linspace(-size_z // 2, size_z // 2, size_z)
    xx, yy, zz = np.meshgrid(ax_x, ax_y, ax_z, indexing='ij')
    kernel = np.exp(-((xx**2 / (2 * sigma_x**2)) + (yy**2 / (2 * sigma_y**2)) + (zz**2 / (2 * sigma_z**2))))
    return kernel / np.sum(kernel)

# Function to calculate detail loss
def calculate_detail_loss(original_density, smoothed_density):
    return np.sum(np.abs(original_density - smoothed_density))

# Visualization function
def plot_kde_result_3d(kde_result, x_edges, y_edges, z_edges, iteration):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x_pos, y_pos, z_pos = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2, (z_edges[:-1] + z_edges[1:]) / 2, indexing="ij")
    x_pos, y_pos, z_pos, densities = x_pos.flatten(), y_pos.flatten(), z_pos.flatten(), kde_result.flatten()
    scatter = ax.scatter(x_pos, y_pos, z_pos, c=densities, cmap='viridis', marker='o')
    fig.colorbar(scatter, ax=ax, label='Density')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Z Coordinate (m)')
    plt.title(f'KDE Result - Iteration {iteration}')
    plt.show()

# Function to run KDE iterations automatically
def run_kde_iterations_auto(density_grid, grid_edges_x, grid_edges_y, grid_edges_z, x_resolution, y_resolution, z_resolution):
    sigma_x = sigma_y = 10
    sigma_z = 5
    kernel_size_x = kernel_size_y = 21
    kernel_size_z = 11

    iteration_count = 5
    for i in range(iteration_count):  # Perform 5 iterations or more if needed
        kernel = gaussian_kernel_3d(kernel_size_x, kernel_size_y, kernel_size_z, sigma_x, sigma_y, sigma_z)
        kde_result = fftconvolve(density_grid, kernel, mode='same')
        detail_loss = calculate_detail_loss(density_grid, kde_result)
        
        print(f"Iteration {i+1}: Sigma_x = {sigma_x}, Sigma_y = {sigma_y}, Sigma_z = {sigma_z}")
        print(f"Kernel Size: ({kernel_size_x}, {kernel_size_y}, {kernel_size_z}), Detail Loss: {detail_loss}")
        
        plot_kde_result_3d(kde_result, grid_edges_x, grid_edges_y, grid_edges_z, i+1)
        
        sigma_x -= 2
        sigma_y -= 2
        sigma_z -= 1
        kernel_size_x = int(5 * sigma_x / x_resolution * 2 + 1)
        kernel_size_y = int(5 * sigma_y / y_resolution * 2 + 1)
        kernel_size_z = int(5 * sigma_z / z_resolution * 2 + 1)

# Function to normalize KDE result
def normalize_kde_result(kde_result):
    kde_result_flat = kde_result.flatten()
    normalized_densities = {}

    normalization_types = ['minmax', 'quartile', 'zscore', 'log', 'robust']

    for normalization_type in normalization_types:
        print(f"Calculating {normalization_type} weights")
        if normalization_type == 'minmax':
            normalized_weights = (kde_result_flat - np.min(kde_result_flat)) / (np.max(kde_result_flat) - np.min(kde_result_flat))
        elif normalization_type == 'quartile':
            Q1, Q3 = np.percentile(kde_result_flat, [25, 75])
            IQR = Q3 - Q1
            normalized_weights = (kde_result_flat - Q1) / IQR
        elif normalization_type == 'zscore':
            normalized_weights = (kde_result_flat - np.mean(kde_result_flat)) / np.std(kde_result_flat)
        elif normalization_type == 'log':
            log_weights = np.log(kde_result_flat + 1e-5)
            normalized_weights = (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights))
        elif normalization_type == 'robust':
            median = np.median(kde_result_flat)
            Q1, Q3 = np.percentile(kde_result_flat, [25, 75])
            IQR = Q3 - Q1
            normalized_weights = (kde_result_flat - median) / IQR
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
        
        # Reshape to match the original kde_result shape and add to normalized_densities
        normalized_densities[normalization_type] = normalized_weights.reshape(kde_result.shape)
        
        print(f"{normalization_type} weights range: [{normalized_weights.min():.4f}, {normalized_weights.max():.4f}]")

    # Extend kde_result with normalized densities
    extended_kde_result = {'original': kde_result}
    extended_kde_result.update(normalized_densities)

    return extended_kde_result

# Visualization function for point cloud using PyVista
def plot_kde_result_point_cloud(kde_result, x_edges, y_edges, z_edges, result_type='minmax'):
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    x_pos, y_pos, z_pos = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    points = np.c_[x_pos.flatten(), y_pos.flatten(), z_pos.flatten()]
    densities = kde_result[result_type].flatten()

    #min_density_threshold = 0.001  # Set a threshold to reduce noise
    min_density_threshold = 0.6  # Set a threshold to reduce noise
    valid_points = densities > min_density_threshold
    points = points[valid_points]
    densities = densities[valid_points]
    
    point_cloud = pv.PolyData(points)
    point_cloud["Density"] = densities
    
    cmapCol=cc.cm['CET_CBTD1']

    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars="Density", opacity="Density", cmap=cmapCol, point_size=50, render_points_as_spheres=True)
    plotter.add_axes(labels_off=False)
    plotter.show()


# Function to plot the site data with KDE using PyVista
def plot(site_data, name, cmap_name='CET_CBTD1'):
    """
    Plots the site data using PyVista with the specified scalar field representing the normalized KDE result.
    
    Parameters:
    - site_data: xarray Dataset containing the site data and normalized KDE results.
    - scalar_field: String name of the scalar field (e.g., 'field_kde-minmax') to plot.
    - cmap_name: Name of the colormap to use from the Colorcet library (default: 'CET_CBTD1').
    """
    # Extract the coordinates and scalar field data
    x_coords = site_data['x'].values
    y_coords = site_data['y'].values
    z_coords = site_data['z'].values
    
    scalar_field = f'kde-{name}-minmax'
    
    scalars = site_data[scalar_field].values

    # Create a PyVista PolyData object
    points = np.c_[x_coords, y_coords, z_coords]
    point_cloud = pv.PolyData(points)
    point_cloud[scalar_field] = scalars

    # Create a PyVista plotter object
    plotter = pv.Plotter()

    # Add the point cloud to the plotter with the specified scalar field and colormap
    cmap = cc.cm[cmap_name]
    plotter.add_mesh(point_cloud, scalars=scalar_field, cmap=cmap, point_size=8, render_points_as_spheres=True)

    # Add axes and display the plot
    plotter.add_axes()
    plotter.show()


# Function to transfer KDE values back to the site data
def transfer_kde_to_site(site_data, normalized_kde, x_edges, y_edges, z_edges, name):
    # Extract coordinates from the site data
    site_coords = np.array([site_data['x'].values, site_data['y'].values, site_data['z'].values]).T

    # Create coordinates for the KDE grid
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    x_grid, y_grid, z_grid = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    kde_coords = np.array([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(kde_coords)

    # Find nearest KDE point for each site point
    _, indices = tree.query(site_coords)

    # Transfer KDE attributes to site data
    for norm_type, kde_values in normalized_kde.items():
        if norm_type != 'original':
            field_name = f'kde-{name}-{norm_type}'
            site_data[field_name] = xr.DataArray(
                kde_values.ravel()[indices],
                dims=site_data.x.dims,
                coords=site_data.x.coords
            )

    return site_data

def get_kde(downsampled_xarray, weightings, name, full_xarray = None, x_resolution=5, y_resolution=5, z_resolution=5, normalization_type='minmax', use_debug=False):
    """
    Runs the KDE process, normalizes the results, and transfers them to the site data.

    Parameters:
    - xarray: The combined xarray dataset.
    - weightings: A function to calculate the weights.
    - name: Base name for the fields in the site data.
    - x_resolution, y_resolution, z_resolution: Resolutions in each axis.
    - normalization_type: The type of normalization to apply to the KDE results.
    - use_debug: If True, runs the KDE iterations in debug mode.

    Returns:
    - site_data_with_kde: xarray Dataset with the KDE results added.
    - normalized_kde: Dictionary of normalized KDE results.
    """
    # Apply the weighting function
    weights_v2 = weightings(downsampled_xarray)

    # Extract coordinates and weighted densities
    x_coords = downsampled_xarray['x'].values
    y_coords = downsampled_xarray['y'].values
    z_coords = downsampled_xarray['z'].values

    # Define grid edges based on resolution
    grid_edges_x = np.linspace(x_coords.min(), x_coords.max(), int((x_coords.max() - x_coords.min()) / x_resolution) + 1)
    grid_edges_y = np.linspace(y_coords.min(), y_coords.max(), int((y_coords.max() - y_coords.min()) / y_resolution) + 1)
    grid_edges_z = np.linspace(z_coords.min(), z_coords.max(), int((z_coords.max() - z_coords.min()) / z_resolution) + 1)

    # Calculate the density grid with the improved weights
    density_grid_v2, _ = np.histogramdd(
        (x_coords, y_coords, z_coords), 
        bins=(grid_edges_x, grid_edges_y, grid_edges_z), 
        weights=weights_v2
    )

    # Optionally run KDE iterations in debug mode
    if use_debug:
        run_kde_iterations_auto(density_grid_v2, grid_edges_x, grid_edges_y, grid_edges_z, x_resolution, y_resolution, z_resolution, use_debug=True)

    # Process KDE and get the normalized KDE results
    sigma_x = sigma_y = 3
    sigma_z = 2
    kernel_size_x = int(5 * sigma_x / x_resolution * 2 + 1)
    kernel_size_y = int(5 * sigma_y / y_resolution * 2 + 1)
    kernel_size_z = int(5 * sigma_z / z_resolution * 2 + 1)

    kernel = gaussian_kernel_3d(kernel_size_x, kernel_size_y, kernel_size_z, sigma_x, sigma_y, sigma_z)
    kde_result = fftconvolve(density_grid_v2, kernel, mode='same')
    detail_loss = calculate_detail_loss(density_grid_v2, kde_result)

    # Normalize KDE results
    normalized_kde = normalize_kde_result(kde_result)

    print(f"Final Iteration: Sigma_x = {sigma_x}, Sigma_y = {sigma_y}, Sigma_z = {sigma_z}")
    print(f"Kernel Size: ({kernel_size_x}, {kernel_size_y}, {kernel_size_z}), Detail Loss: {detail_loss}")

    if full_xarray is not None:
        site_xarray = full_xarray
    else:
        site_xarray = downsampled_xarray
    # Transfer the KDE values back to the site data
    site_data_with_kde = transfer_kde_to_site(site_xarray, normalized_kde, grid_edges_x, grid_edges_y, grid_edges_z, name=name)

    return site_data_with_kde, normalized_kde

def print_column_names(data):
    # Get the column names from the xarray Dataset
    column_names = list(data.data_vars.keys())
    
    # Format the column names (strip whitespace, handle any other formatting needs)
    formatted_column_names = [name.strip() for name in column_names]

    # Print each column name
    for name in formatted_column_names:
        print(name)

def xarray_to_pyvista_polydata(xarray):
    """
    Converts an xarray Dataset with 'point_index', 'x', 'y', 'z' coordinates
    into a PyVista PolyData object.
    
    Parameters:
    - xarray: The xarray Dataset containing 'x', 'y', 'z' coordinates and other data variables.
    
    Returns:
    - point_cloud: PyVista PolyData object containing the points and associated scalar fields.
    """
    # Extract the coordinates
    x_coords = xarray['x'].values
    y_coords = xarray['y'].values
    z_coords = xarray['z'].values
    
    # Create a numpy array of the points
    points = np.c_[x_coords, y_coords, z_coords]
    
    # Initialize the PyVista PolyData object
    point_cloud = pv.PolyData(points)
    
    # Add each data variable in the xarray as a scalar field to the PolyData
    for var_name in xarray.data_vars:
        point_cloud[var_name] = xarray[var_name].values
    
    return point_cloud


def main():
    # Load datasets
    #site_path = 'data/revised/trimmed-parade-processed_downsampled.nc'
    #urban_forest_site_path = 'data/revised/trimmed-parade-forestVoxels_downsampled.nc'

    #CHOOSE SITE

    sites = ['street', 'city', 'trimmed-parade']
    #sites = ['street', 'city']

    for site in sites:

        #FILE PATHS
        print(f'running site {site}')
        site_path = f'data/revised/{site}-processed.nc'
        urban_forest_site_path = f'data/revised/{site}-forestVoxels.nc'
        downsampled_site_path = f'data/revised/{site}-processed_downsampled.nc'
        downsample_urban_forest_site_path = f'data/revised/{site}-forestVoxels_downsampled.nc'
    
        #LOAD DATASETS
        site_data = xr.open_dataset(site_path, engine='h5netcdf')
        urban_forest_site_data = xr.open_dataset(urban_forest_site_path, engine='h5netcdf')
        downsampled_site_data = xr.open_dataset(downsampled_site_path, engine='h5netcdf')
        downsampled_urban_forest_site_data = xr.open_dataset(downsample_urban_forest_site_path, engine='h5netcdf')
        
        # Merge the datasets
        site = concatenate_datasets_through_pandas(site_data, urban_forest_site_data)
        downsampled_site = concatenate_datasets_through_pandas(downsampled_site_data, downsampled_urban_forest_site_data)

        # Run the KDE process with the combined xarray, weights function, and a name
        site_data_with_kde, normalized_kde = get_kde(downsampled_site, GetAborealResourceWeights, full_xarray=site, name='resource')

        site_data_with_kde, normalized_kde = get_kde(downsampled_site, GetPylonWeights, full_xarray=site_data_with_kde, name='pylons')

        

        # Optionally save or further process the site_data_with_kde
        #site_data_with_kde.to_netcdf('output_with_kde.nc')

        #plot(site_data_with_kde, 'resource')
        #plot(site_data_with_kde, 'pylons')

        polydata = xarray_to_pyvista_polydata(site_data_with_kde)
        print('converted to polydata')
        import os
        site_name = downsampled_site_path.split('/')[-1].split('-')[0]
        if not os.path.exists('data/revised/processed'):
            os.makedirs('data/revised/processed')
        polydata.save(f'data/revised/processed/{site_name}-processed.vtk') 
        print(f'saved vtk for {site}')




# Run the main function
main()
