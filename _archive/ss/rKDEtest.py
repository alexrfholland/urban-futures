import xarray as xr
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pyvista as pv



###MERGING

import xarray as xr
import numpy as np

def concatenate_datasets_through_pandas(ds1, ds2):
    # Convert Xarray datasets to Pandas DataFrames
    df1 = ds1.to_dataframe().reset_index()
    df2 = ds2.to_dataframe().reset_index()

    # Print info about the DataFrames before concatenation
    print(f"DataFrame 1: {df1.shape[0]} rows, columns: {df1.columns.tolist()}")
    print(f"DataFrame 2: {df2.shape[0]} rows, columns: {df2.columns.tolist()}")

    # Get shared and unique columns
    shared_columns = set(df1.columns).intersection(set(df2.columns))
    print(f"Shared columns: {list(shared_columns)}")

    # Align columns by adding missing columns with NaN values
    all_columns = set(df1.columns).union(set(df2.columns))
    for col in all_columns:
        if col not in df1.columns:
            df1[col] = np.nan
        if col not in df2.columns:
            df2[col] = np.nan

    # Concatenate the DataFrames
    concatenated_df = pd.concat([df1, df2], ignore_index=True)

    # Check for NaN values in 'x', 'y', and 'z' columns
    nan_x = concatenated_df['x'].isna().sum() if 'x' in concatenated_df.columns else 'Column x not present'
    nan_y = concatenated_df['y'].isna().sum() if 'y' in concatenated_df.columns else 'Column y not present'
    nan_z = concatenated_df['z'].isna().sum() if 'z' in concatenated_df.columns else 'Column z not present'

    # Print the results
    print(f"NaN values in 'x' column: {nan_x}")
    print(f"NaN values in 'y' column: {nan_y}")
    print(f"NaN values in 'z' column: {nan_z}")
    
    print(f"Final concatenated DataFrame: {concatenated_df.shape[0]} rows, columns: {concatenated_df.columns.tolist()}")

    # Convert the final concatenated DataFrame back to an Xarray Dataset
    final_xr = concatenated_df.set_index('point_index').to_xarray()

    # Adjust the coordinates to reflect the row index from the Pandas DataFrame
    final_xr = final_xr.assign_coords(point_index=concatenated_df.index)

    print(f'Converting back into xarray dataset')

    # Check for NaN values in x, y, z columns of the final xarray dataset
    nan_x_xr = final_xr['x'].isnull().sum().item()
    nan_y_xr = final_xr['y'].isnull().sum().item()
    nan_z_xr = final_xr['z'].isnull().sum().item()

    print(f"NaN values in 'x' column of final xarray: {nan_x_xr}")
    print(f"NaN values in 'y' column of final xarray: {nan_y_xr}")
    print(f"NaN values in 'z' column of final xarray: {nan_z_xr}")

    print(f"Number of rows in final xarray: {final_xr.sizes['point_index']}")

    return final_xr



###
# Load datasets and extract coordinates
urban_forest_site_path = 'data/revised/trimmed-parade-forestVoxels.nc'
site_path = 'data/revised/trimmed-parade-processed_downsampled.nc'

site_data = xr.open_dataset(site_path, engine='h5netcdf')
urban_forest_site_data = xr.open_dataset(urban_forest_site_path, engine='h5netcdf')

# Print the length of site_data and urban_forest_site_data
print(f"Length of site_data: {len(site_data.x)}")
print(f"Length of urban_forest_site_data: {len(urban_forest_site_data.x)}")


# Print properties of both datasets, i.e., columns and number of rows
print(f"Columns in site_data: {list(site_data.data_vars.keys())}, Number of rows: {len(site_data.x)}")
print(f"Columns in urban_forest_site_data: {list(urban_forest_site_data.data_vars.keys())}, Number of rows: {len(urban_forest_site_data.x)}")



#Merge the datasets
data = concatenate_datasets_through_pandas(site_data, urban_forest_site_data)


#SUM: 'resource_dead branch' 'resource_peeling bark', 'resource_perchable branch') * 2 + resource_other
#CHECK if ['road_types-type'] == 'Carriageway' if TRUE - 100

def apply_weightings(data):
    # Calculate the weighted sum of the resources
    weighted_sum = (
        (data['resource_dead branch'] + 
         data['resource_peeling bark'] + 
         data['resource_perchable branch']) * 2 +
        data['resource_other']
    )
    
    # Apply negative weighting if 'road_types-type' is 'Carriageway'
    if 'road_types-type' in data:
        weighted_sum -= (data['road_types-type'] == 'Carriageway') * 1
    
    return weighted_sum

# Apply the weightings to the merged dataset
data['weighted_density'] = apply_weightings(data)

# Extract coordinates and weighted densities
x_coords = data['x'].values
y_coords = data['y'].values
z_coords = data['z'].values
weights = data['weighted_density'].values

# Define grid edges based on resolution
x_resolution, y_resolution, z_resolution = 5, 5, 5  # meters
grid_edges_x = np.linspace(x_coords.min(), x_coords.max(), int((x_coords.max() - x_coords.min()) / x_resolution) + 1)
grid_edges_y = np.linspace(y_coords.min(), y_coords.max(), int((y_coords.max() - y_coords.min()) / y_resolution) + 1)
grid_edges_z = np.linspace(z_coords.min(), z_coords.max(), int((z_coords.max() - z_coords.min()) / z_resolution) + 1)

# Calculate the density grid with the weighted values
density_grid, _ = np.histogramdd(
    (x_coords, y_coords, z_coords), 
    bins=(grid_edges_x, grid_edges_y, grid_edges_z), 
    weights=weights
)

"""coords_mask = data['urban systems'].values == 'Existing Canopies'
x_coords, y_coords, z_coords = data['x'].values[coords_mask], data['y'].values[coords_mask], data['z'].values[coords_mask]

# Grid resolutions
x_resolution, y_resolution = 5, 5  # meters
z_resolution = 5  # meters for vertical resolution

# Initial sigma and kernel size
sigma_x = sigma_y = 10
sigma_z = 5
kernel_size_x = kernel_size_y = 21
kernel_size_z = 11

# Original density grid
grid_edges_x = np.linspace(data['x'].values.min(), data['x'].values.max(), int((data['x'].values.max() - data['x'].values.min()) / x_resolution) + 1)
grid_edges_y = np.linspace(data['y'].values.min(), data['y'].values.max(), int((data['y'].values.max() - data['y'].values.min()) / y_resolution) + 1)
grid_edges_z = np.linspace(data['z'].values.min(), data['z'].values.max(), int((data['z'].values.max() - data['z'].values.min()) / z_resolution) + 1)
density_grid, _ = np.histogramdd((x_coords, y_coords, z_coords), bins=(grid_edges_x, grid_edges_y, grid_edges_z))"""


# Function to calculate detail loss
def calculate_detail_loss(original_density, smoothed_density):
    return np.sum(np.abs(original_density - smoothed_density))

# Gaussian kernel function
def gaussian_kernel_3d(size_x, size_y, size_z, sigma_x, sigma_y, sigma_z):
    ax_x = np.linspace(-size_x // 2, size_x // 2, size_x)
    ax_y = np.linspace(-size_y // 2, size_y // 2, size_y)
    ax_z = np.linspace(-size_z // 2, size_z // 2, size_z)
    xx, yy, zz = np.meshgrid(ax_x, ax_y, ax_z, indexing='ij')
    kernel = np.exp(-((xx**2 / (2 * sigma_x**2)) + (yy**2 / (2 * sigma_y**2)) + (zz**2 / (2 * sigma_z**2))))
    return kernel / np.sum(kernel)



def plot_kde_result_point_cloud(kde_results, x_edges, y_edges, z_edges, result_type='original'):
    # Select the appropriate KDE result based on the provided result_type
    kde_result = kde_results.get(result_type)
    if kde_result is None:
        raise ValueError(f"No KDE result found for type '{result_type}'")

    # Calculate the center points of the grid cells
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    # Create a meshgrid of the center points
    x_pos, y_pos, z_pos = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    
    # Flatten the positions and densities
    points = np.c_[x_pos.flatten(), y_pos.flatten(), z_pos.flatten()]
    densities = kde_result.flatten()

    # Filter out points with very low density (optional)
    min_density_threshold = 0.001  # Set a threshold to reduce noise
    valid_points = densities > min_density_threshold
    points = points[valid_points]
    densities = densities[valid_points]
    
    # Create the point cloud
    point_cloud = pv.PolyData(points)
    point_cloud["Density"] = densities
    
    # Set up the plotter
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars="Density", opacity = "Density", cmap="viridis", point_size=50, render_points_as_spheres=True)
    
    # Add labels
    plotter.add_axes(labels_off=False)

    # Show plot
    plotter.show()


def plot_kde_result_point_cloudOLD(kde_result, x_edges, y_edges, z_edges):
    # Calculate the center points of the grid cells
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    # Create a meshgrid of the center points
    x_pos, y_pos, z_pos = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    
    # Flatten the positions and densities
    points = np.c_[x_pos.flatten(), y_pos.flatten(), z_pos.flatten()]
    densities = kde_result.flatten()

    # Filter out points with very low density (optional)
    min_density_threshold = 0.001  # Set a threshold to reduce noise
    valid_points = densities > min_density_threshold
    points = points[valid_points]
    densities = densities[valid_points]
    
    # Create the point cloud
    point_cloud = pv.PolyData(points)
    point_cloud["Density"] = densities
    
    # Set up the plotter
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars="Density", opacity = "Density", cmap="viridis", point_size=50, render_points_as_spheres=True)
    
    # Add labels
    plotter.add_axes(labels_off=False)

    #show plot
    plotter.show()


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



def run_kde_iterations(density_grid, grid_edges_x, grid_edges_y, grid_edges_z, x_resolution, y_resolution, z_resolution, use_debug=True):
    if use_debug:
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
            
            # Plot using Matplotlib
            plot_kde_result_3d(kde_result, grid_edges_x, grid_edges_y, grid_edges_z, i+1)
            
            # Adjust sigma and kernel size with larger steps to reduce smoothing
            sigma_x -= 2
            sigma_y -= 2
            sigma_z -= 1
            kernel_size_x = int(5 * sigma_x / x_resolution * 2 + 1)
            kernel_size_y = int(5 * sigma_y / y_resolution * 2 + 1)
            kernel_size_z = int(5 * sigma_z / z_resolution * 2 + 1)
            
            user_input = input("Continue with the next iteration? (y/n): ")
            if user_input.lower() != 'y':
                break
    else:
        # Values at iteration 4
        sigma_x = sigma_y = 3
        sigma_z = 2
        kernel_size_x = int(5 * sigma_x / x_resolution * 2 + 1)
        kernel_size_y = int(5 * sigma_y / y_resolution * 2 + 1)
        kernel_size_z = int(5 * sigma_z / z_resolution * 2 + 1)

        # Calculate KDE
        kernel = gaussian_kernel_3d(kernel_size_x, kernel_size_y, kernel_size_z, sigma_x, sigma_y, sigma_z)
        kde_result = fftconvolve(density_grid, kernel, mode='same')
        detail_loss = calculate_detail_loss(density_grid, kde_result)

        #HERE ADD ONE FUNCTIO that adds the normalised density values'
        normalised_kde = normalize_kde_result(kde_result)

        
        print(f"Final Iteration: Sigma_x = {sigma_x}, Sigma_y = {sigma_y}, Sigma_z = {sigma_z}")
        print(f"Kernel Size: ({kernel_size_x}, {kernel_size_y}, {kernel_size_z}), Detail Loss: {detail_loss}")
        
        # Plot using PyVista
        plot_kde_result_point_cloud(normalised_kde, grid_edges_x, grid_edges_y, grid_edges_z, result_type='minmax')


# Example usage
run_kde_iterations(density_grid, grid_edges_x, grid_edges_y, grid_edges_z, x_resolution, y_resolution, z_resolution, use_debug=False)
