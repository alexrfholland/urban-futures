"""
Visualize a 3D distribution of weighted points using PyVista.

This script takes 3D point data with associated weights representing various attributes
and processes them into a visual representation. It employs a sparse data structure to efficiently
handle large datasets, typically encountered in environmental modeling and urban planning scenarios.

Key Steps:
1. Define the spatial extents and resolution of a 3D grid to encompass the points.
2. Assign weights to the 3D points, reflecting their attributes or importance.
3. Utilize a sparse matrix to create a 3D histogram, placing points into volumetric bins.
4. Apply a threshold to the weighted bins to filter out points below a certain significance level.
5. Generate glyphs for each significant bin in the form of cubes, colored based on their weights.
6. Render the resulting glyph field in a 3D space using PyVista to illustrate the spatial structure.

This approach is particularly well-suited for modeling complex environments where the
significance of each point can vary and needs to be represented in a three-dimensional context.
"""



import numpy as np
from scipy.sparse import coo_matrix
import pyvista as pv
from typing import List, Tuple, Dict, Union, Any
import glyphs

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from scipy.spatial import cKDTree
from numba import njit
from numba.typed import List

from scipy.stats import gaussian_kde
import cameraSetUpRevised



from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import pandas as pd
import pyvista as pv


import pandas as pd

import json
import os

# File path for stakeholder.json
stakeholder_file = "data/stakeholder.json"

# Initialize or load the stakeholder dictionary
if os.path.exists(stakeholder_file):
    with open(stakeholder_file, "r") as file:
        stakeholder_data = json.load(file)
else:
    stakeholder_data = {}



def kde_evaluate_chunk(args):
    kde, centers_chunk = args
    return kde.evaluate(centers_chunk.T)

def compute_bandwidth_factor(data, weights, method='scott', factor=.1):
    # Instantiate a temporary KDE to compute the bandwidth
    kde_temp = gaussian_kde(data, weights=weights, bw_method=method)
    # Apply the scaling factor directly to the bandwidth factor of the temporary KDE
    bandwidth = kde_temp.factor * factor
    return bandwidth

def create_evaluation_grid(bounds, cell_size):
    # Unpack the bounds
    min_x, max_x, min_y, max_y, min_z, max_z = bounds

    # Create arrays of points along each axis
    x_grid = np.arange(min_x, max_x, cell_size)
    y_grid = np.arange(min_y, max_y, cell_size)
    z_grid = np.arange(min_z, max_z, cell_size)

    # Create a meshgrid from the axes arrays
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    # Combine the grid points into a single array of 3D points
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    print(f'evaluating grid total points: {len(grid_points)}')


    return grid_points


def weighted_3d_kde(centers_1d, weights, cell_size, bandwidth_factor=0.5, useGrid=False):
    print("Reshaping the centers array...")
    centers = centers_1d.reshape(kdvalues-1, 3) if centers_1d.ndim == 1 else centers_1d

    print("Computing the bandwidth factor...")
    bandwidth = compute_bandwidth_factor(centers.T, weights, method='scott', factor=bandwidth_factor)

    print("Creating the gaussian_kde object...")
    kde = gaussian_kde(centers.T, weights=weights, bw_method=bandwidth)

    if useGrid:
        print(f"Creating a regular grid with voxel size {cell_size}...")
        # Calculate the 3D bounds of the points
        min_bounds = np.min(centers, axis=0)
        max_bounds = np.max(centers, axis=0)
        bounds = (min_bounds[0], max_bounds[0], min_bounds[1], max_bounds[1], min_bounds[2], max_bounds[2])
        grid_points = create_evaluation_grid(bounds, cell_size)
        evaluation_points = grid_points
    else:
        evaluation_points = centers

    print("Setting up multiprocessing for KDE evaluation...")
    num_processes = cpu_count()
    centers_chunks = np.array_split(evaluation_points, num_processes)

    with Pool(processes=num_processes) as pool:
        args = [(kde, chunk) for chunk in centers_chunks]
        results = pool.map(kde_evaluate_chunk, args)

    print("Combining the results...")
    kdvalues = np.concatenate(results)

    print(f'kd values: {len(kdvalues)}, grid points: {len(evaluation_points)}')

    print("Processing complete.")
    return kde, evaluation_points, 


def weighted_3d_kdeOLD(centers_1d, weights, bandwidth_factor=0.5):
    print("Reshaping the centers array...")
    # Ensure centers is a 2D array with shape (number_of_points, 3)
    centers = centers_1d.reshape(-1, 3) if centers_1d.ndim == 1 else centers_1d

    print("Transposing the centers array for KDE...")
    # Use centers with shape (3, number_of_points) for KDE
    kde_centers = centers.T  # This is for the KDE which needs (variables, samples)

    print("Computing the bandwidth factor...")
    # Compute the bandwidth factor and scale it down to make the KDE less smooth
    bandwidth = compute_bandwidth_factor(kde_centers, weights, method='scott', factor=bandwidth_factor)

    print("Creating the gaussian_kde object...")
    # Create a gaussian_kde object with weights and specified bandwidth method
    kde = gaussian_kde(kde_centers, weights=weights, bw_method=bandwidth)

    # Determine the number of processes to use
    num_processes = cpu_count()

    print("Splitting the centers array into chunks for multiprocessing...")
    # Split the centers array into chunks for multiprocessing
    chunk_size = len(centers) // num_processes
    centers_chunks = [centers[i:i + chunk_size] for i in range(0, len(centers), chunk_size)]

    print("Setting up multiprocessing for KDE evaluation...")
    with Pool(processes=num_processes) as pool:
        # Create a list of arguments for each chunk
        args = [(kde, chunk) for chunk in centers_chunks]

        # Map the kde_evaluate_chunk function to the arguments
        results = pool.map(kde_evaluate_chunk, args)
        

    print("Combining the results...")
    # Combine the results from each process
    kdvalues = np.concatenate(results)

    print("Processing complete.")
    # Return centers with shape (number_of_points, 3) for PyVista, along with KDE values
    return kde, centers, kdvalues  # centers is not transposed here, as PyVista needs (samples, variables)


"""def create_evaluation_grid(centers, cell_size):
    x_min, y_min, z_min = np.min(centers, axis=0)
    x_max, y_max, z_max = np.max(centers, axis=0)
    x_grid = np.arange(x_min, x_max, cell_size)
    y_grid = np.arange(y_min, y_max, cell_size)
    z_grid = np.arange(z_min, z_max, cell_size)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return grid_points, x_grid, y_grid, z_grid

def kde_evaluate_chunk(args):
    kde, chunk = args
    return kde(chunk.T)



def weighted_3d_kde(centers_1d, weights, bw_method=None, useGrid=None):
    print("Reshaping the centers array...")
    if centers_1d.ndim == 1:
        centers = centers_1d.reshape(-1, 3)
    else:
        centers = centers_1d
    
    print("Creating the gaussian_kde object...")
    kde = gaussian_kde(centers.T, weights=weights, bw_method=bw_method)
    
    if useGrid is not None:
        print(f"Creating a regular grid with cell size {useGrid}...")
        grid_points, x_grid, y_grid, z_grid = create_evaluation_grid(centers, useGrid)
        evaluation_points = grid_points
    else:
        evaluation_points = centers

    print("Setting up multiprocessing for KDE evaluation...")
    num_processes = cpu_count()
    centers_chunks = np.array_split(evaluation_points, num_processes)

    with Pool(processes=num_processes) as pool:
        args = [(kde, chunk) for chunk in centers_chunks]
        results = pool.starmap(kde_evaluate_chunk, args)
        

    print("Combining the results...")
    kdvalues = np.concatenate(results)

    print("Processing complete.")
    if useGrid is not None:
        # Reshape the results to match the grid
        kdvalues = kdvalues.reshape((len(x_grid), len(y_grid), len(z_grid)))
        return kde, grid_points, kdvalues
    else:
        return kde, centers, kdvalues"""



def construct_kde_weighting(points, weights, bounds, cell_size, threshold=0, usesGrid=False):
    """
    Visualize a 3D grid of weighted points.

    Parameters:
    - points: A numpy array of point coordinates.
    - weights: A numpy array of weights corresponding to the points.
    - bounds: A tuple of (min_x, max_x, min_y, max_y, min_z, max_z) defining the bounds of the site.
    - cmap_name: The name of the colormap to be used.
    - cell_size: The size of each cell in the grid.
    - threshold: The threshold value for filtering weights (default is 0).
    """

    print(bounds)

    # Unpack the bounds
    min_x, max_x, min_y, max_y, min_z, max_z = bounds

    # Calculate the number of bins in each dimension
    bins_x = int(np.ceil((max_x - min_x) / cell_size))
    bins_y = int(np.ceil((max_y - min_y) / cell_size))
    bins_z = int(np.ceil((max_z - min_z) / cell_size))

    # Create the bin edges
    bin_edges_x = np.linspace(min_x, max_x, bins_x + 1)
    bin_edges_y = np.linspace(min_y, max_y, bins_y + 1)
    bin_edges_z = np.linspace(min_z, max_z, bins_z + 1)

    # Digitize points into bins
    bin_index_x = np.digitize(points[:, 0], bin_edges_x) - 1
    bin_index_y = np.digitize(points[:, 1], bin_edges_y) - 1
    bin_index_z = np.digitize(points[:, 2], bin_edges_z) - 1

    # Clip indices to handle points that are exactly at the max bound
    bin_index_x = np.clip(bin_index_x, 0, bins_x - 1)
    bin_index_y = np.clip(bin_index_y, 0, bins_y - 1)
    bin_index_z = np.clip(bin_index_z, 0, bins_z - 1)

    # Filter out weights below the threshold
    valid_indices = weights > threshold
    filtered_weights = weights[valid_indices]
    filtered_bin_index_x = bin_index_x[valid_indices]
    filtered_bin_index_y = bin_index_y[valid_indices]
    filtered_bin_index_z = bin_index_z[valid_indices]

    # Create the linear indices for the sparse matrix
    linear_bin_indices = np.ravel_multi_index(
        (filtered_bin_index_x, filtered_bin_index_y, filtered_bin_index_z),
        (bins_x, bins_y, bins_z)
    )

    # Construct the sparse histogram
    sparse_hist = coo_matrix(
        (filtered_weights, (linear_bin_indices, np.zeros_like(linear_bin_indices))),
        shape=(bins_x * bins_y * bins_z, 1)
    )

    # Unravel the linear indices to 3D indices for visualization
    unraveled_indices = np.column_stack(np.unravel_index(sparse_hist.row, (bins_x, bins_y, bins_z)))

    # Calculate bin centers
    centers_x = (unraveled_indices[:, 0] + 0.5) * cell_size + min_x
    centers_y = (unraveled_indices[:, 1] + 0.5) * cell_size + min_y
    centers_z = (unraveled_indices[:, 2] + 0.5) * cell_size + min_z
    centers = np.vstack((centers_x, centers_y, centers_z)).T


    # Min-Max Normalization: Good for scaling the data to a fixed range [0, 1].

    # Quartile Normalization: Useful for dealing with outliers and non-normal distributions.

    # Z-Score Normalization: Ideal for understanding the relative standing in a normal distribution.

    # Logarithmic Scaling: Best for data spanning several orders of magnitude.

    # Robust Scaling: Effective for reducing the influence of outliers, more robust than Min-Max scaling.

    # Opacity Mapping using Normalized Weights: Standard for visual clarity in plots, scaling original weights between 0 and 1.

    weights = sparse_hist.data

    #kde, centers, weights = weighted_3d_kde(centers, weights, bw_method='scott', useGrid = 5)
    #kde, centers, weights = weighted_3d_kde(centers, weights, bw_method='scott')
    
    useKDE = False
    if useKDE:
    
        factor = 0.25
        kde, centers, weights = weighted_3d_kdeOLD(centers, weights,bandwidth_factor=2)





    def plot_weight_distribution(weights, plot_type='histogram', bins=30, **kwargs):
        """
        Plots the distribution of weights.

        Parameters:
        - weights: Array of weights to plot.
        - plot_type: Type of plot to generate. Options are 'histogram' or 'kde' for kernel density estimate.
        - bins: Number of bins to use for the histogram. Ignored for KDE plot.
        - kwargs: Additional keyword arguments to pass to the plotting function.
        """
        if plot_type == 'histogram':
            plt.hist(weights, bins=bins, **kwargs)
            plt.title('Histogram of Weights')
        elif plot_type == 'kde':
            density = gaussian_kde(weights)
            xs = np.linspace(min(weights), max(weights), 1000)
            plt.plot(xs, density(xs), **kwargs)
            plt.title('Kernel Density Estimate of Weights')
        else:
            raise ValueError("plot_type must be either 'histogram' or 'kde'")
        
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    plot_weight_distribution(weights)



    # Create a PyVista grid
    grid = pv.PolyData(centers)

    # Assign the weights to the grid
    grid['intensity'] = weights
    #grid['unmodified'] = sparse_hist.data

    # Min-Max Normalization
    weights_minmax = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    grid['weights_minmax'] = weights_minmax

    # Quartile Normalization
    Q1 = np.percentile(weights, 25)
    Q3 = np.percentile(weights, 75)
    IQR = Q3 - Q1
    weights_quartile = (weights - Q1) / IQR
    grid['weights_quartile'] = weights_quartile

    # Z-Score Normalization
    mean_weight = np.mean(weights)
    std_weight = np.std(weights)
    weights_zscore = (weights - mean_weight) / std_weight
    grid['weights_zscore'] = weights_zscore

    # Logarithmic Scaling
    log_weights = np.log(weights + 1e-5)  # To avoid log(0)
    weights_log = (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights))
    grid['weights_log'] = weights_log

    # Robust Scaling
    median_weight = np.median(weights)
    weights_robust = (weights - median_weight) / IQR
    grid['weights_robust'] = weights_robust

    # Normalize the original weights for opacity mapping
    normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())
    grid['opacity'] = normalized_weights

    return grid









import numpy as np

def process_agent_resources(multiblock: pv.MultiBlock, agent_name: str = 'survey'):
    # Predefined dictionary of agents with their resources and associated weights
    
    
    agents = {
        'woodland bird': {
            'resources': {
                'dead branch': 100,
                'peeling bark': .1,
                'perchable branch': 5,
                'epiphyte': 10,
                'fallen log': 10,
                'leaf litter': 1
            },
            'cmap': 'YlGn'
        },

        'tree': {
            'resources': {
                'dead branch': 1,
                'peeling bark': 1,
                'perchable branch': 1,
                'epiphyte': 1,
                'fallen log': 0,
                'leaf litter': 0,
                'other' : 1
            },
            'cmap': 'YlGn'
        },
        'hollow nesting bird': {
            'resources': {
                'hollow' : 100000,
                'dead branch': 10,
                'peeling bark': .1,
                'perchable branch': .1,
                'epiphyte': 100,
                'fallen log': 1
            },
            'cmap': 'PuRd'
        },
        'bird': {
            'resources': {
                'hollow' : 1,
                'dead branch': 1,
                'peeling bark': 1
            },
            'cmap': 'PuRd'
        },
        'reptile': {
            'resources': {
                'leaf litter': 1,
                'fallen log': 10,
                'perchable branch': .01,
            },
            'cmap': 'Oranges'
        },
        'survey': {
            'resources': {
                'dead branch': 1000,
                'peeling bark': 1000,
                'perchable branch': 1,
                'epiphyte': 1000,
                'hollow' : 1000,
                'fallen log': 10,
                'leaf litter': 10
            },
            'cmap': 'YlGn'
        }
    }



    if agent_name not in agents:
        print(f"Agent '{agent_name}' not found, breaking out of function.")
        return None

    # Extract points and resource data from the multiblock dataset
    all_points = np.empty((0, 3)) 
    all_resources = np.empty((0,))
    #required_blocks = ['branches', 'canopy resources', 'ground resources']
    required_blocks = ['branches', 'canopy resources', 'ground resources']

    print('done')



    
    print(f"Extracting points and resources for {agent_name}...")
    for block in required_blocks:
        block_data = multiblock.get(block)
        if block_data is not None and block_data.number_of_points > 0:  # Check if block_data has points
            points = block_data.points
            resources = block_data.point_data['resource']
            
            all_points = np.vstack((all_points, points))
            all_resources = np.concatenate((all_resources, resources))
        else:
            print(f"Block '{block}' not found in multiblock.")
    # Create a zero weights array
    weights = np.zeros(len(all_points), dtype=float)
    
    # Assign weights as per the agent dictionary
    print(f"Assigning weights for {agent_name}...")
    for resource, weight in agents[agent_name]['resources'].items():
        # Create masks for resources and assign weights
        mask = all_resources == resource
        weights[mask] = weight
    

    minx = np.min(all_points[:,0])
    maxx = np.max(all_points[:,0])
    miny = np.min(all_points[:,1])
    maxy = np.max(all_points[:,1])
    minz = np.min(all_points[:,2])
    maxz = np.max(all_points[:,2])

    bounds = (minx - 10, maxx + 10, miny - 10, maxy + 10, minz - 10, maxz + 10)

    # Get the colormap for the agent
    cmap = agents[agent_name]['cmap']

    print(np.unique(all_resources))

    return all_points, weights, all_resources, bounds, cmap




def distribute_resource_points(centers_1d, weights, cell_size):
    # Check if centers need to be reshaped
    if centers_1d.ndim == 1:
        # Assuming 'kdvalues' is known beforehand and is correct. 
        # You might need to adjust this to directly compute the correct size based on centers_1d and weights
        centers = centers_1d.reshape(-1, 3)
    else:
        centers = centers_1d

    # Create a PolyData object
    grid = pv.PolyData(centers)
    
    # Add the weights as a scalar data to the grid
    grid["intensity"] = weights

    print(f'max weight is {np.max(weights)}')

    return grid



import numpy as np

def sum_weights_in_voxels_vectorized(points, weights, bounds, cell_size):
    """
    Sum weights of points within each voxel in a vectorized manner.
    
    Parameters:
    - points: (N, 3) array of point coordinates.
    - weights: (N,) array of weights corresponding to the points.
    - bounds: (min_x, max_x, min_y, max_y, min_z, max_z) defining the bounds of the grid.
    - cell_size: The edge length of each voxel in the grid.
    
    Returns:
    - voxel_weights_sum: A dictionary with voxel indices as keys and summed weights as values.
    """
    # Calculate grid dimensions
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    num_voxels_x = int(np.ceil((max_x - min_x) / cell_size))
    num_voxels_y = int(np.ceil((max_y - min_y) / cell_size))
    num_voxels_z = int(np.ceil((max_z - min_z) / cell_size))
    
    # Assign points to voxels
    voxel_indices_x = np.floor((points[:, 0] - min_x) / cell_size).astype(int)
    voxel_indices_y = np.floor((points[:, 1] - min_y) / cell_size).astype(int)
    voxel_indices_z = np.floor((points[:, 2] - min_z) / cell_size).astype(int)
    
    # Clip indices to be within the grid
    voxel_indices_x = np.clip(voxel_indices_x, 0, num_voxels_x - 1)
    voxel_indices_y = np.clip(voxel_indices_y, 0, num_voxels_y - 1)
    voxel_indices_z = np.clip(voxel_indices_z, 0, num_voxels_z - 1)
    
    # Flatten voxel indices to create unique keys for each voxel
    voxel_keys = (voxel_indices_z * num_voxels_y * num_voxels_x) + (voxel_indices_y * num_voxels_x) + voxel_indices_x
    
    # Use np.bincount to sum weights for each voxel key
    summed_weights = np.bincount(voxel_keys, weights=weights, minlength=num_voxels_x*num_voxels_y*num_voxels_z)
    
    # Optionally, reshape summed_weights back into a 3D grid shape if needed
    voxel_weights_sum_grid = summed_weights.reshape((num_voxels_z, num_voxels_y, num_voxels_x))
    
    return voxel_weights_sum_grid





if __name__ == "__main__":
    

    site = 'trimmed-parade'
    site = 'street'
    
    site = 'parade'
    site = 'city'
    #site = 'street'
    #state = 'trending'
    #state = 'now'
    state = 'trending'
    #agents = ['woodland bird', 'hollow nesting bird', 'reptile']
    agents= ['tree']


    agents= ['reptile']

    site = 'street'
    #tempstate = 'preferred'
    tempstate = 'now'
    #tempstate = 'trending'
    #state = 'trending'
    agents = ['reptile']
    #agents = ['hollow nesting bird']
    #agents = ['reptile']

    #

    #siteMultiBlock = pv.read(f'data/{site}/structures-{site}-{state}.vtm')

    

    if tempstate == 'preferred':
        state = 'now'
        siteMultiBlock = pv.read(f'data/{site}/all_resources-{site}-{state}.vtm')
        treeblock = siteMultiBlock
    else:
        state = tempstate
        siteMultiBlock = siteMultiBlock = pv.read(f'data/{site}/combined-{site}-{state}.vtm')
        treeblock = siteMultiBlock.get('trees')

    

    

    # Assume 'multiblock' is your MultiBlock dataset
    

    print(siteMultiBlock.keys())



    cellSize = 10



    for agent in agents:

        print(f'agent is {agent}')
    
        points, weights, allResources, bounds, cmap = process_agent_resources(treeblock, agent_name=agent)
        print(f'new bounds are {bounds}')

        #points, weights, bounds = getAttributes2(siteMultiBlock)
        #print(f'old bounds are {bounds}')




        rest = siteMultiBlock.get('rest of points')

        print(f'points are length {len(points)}')
        print(f'weights are length {len(weights)}')
        print(f'bounds are {bounds}')

        
        grid = construct_kde_weighting(points, weights, bounds, cellSize, usesGrid=True)

        #grid = distribute_resource_points(points, weights, cellSize)



        # Find the maximum KDE value
        """max_kde_value = np.max(grid['intensity'])  # Replace 'weights' with the array of KDE values

        # Update the dictionary
        if agent not in stakeholder_data:
            stakeholder_data[agent] = {}
        if site not in stakeholder_data[agent]:
            stakeholder_data[agent][site] = {}
        stakeholder_data[agent][site][tempstate] = max_kde_value

        with open(stakeholder_file, "w") as file:
            json.dump(stakeholder_data, file, indent=4)"""



        # Create glyphs from grid with cubes scaled by cell_size
        glyphsObject = grid.glyph(orient=False, scale=False, factor=cellSize, geom=pv.Cube())

        #for name in ['intensity', 'weights_minmax', 'weights_quartile', 'weights_zscore', 'weights_log', 'weights_robust']:

        # Create and configure the plotter
        plotter = pv.Plotter()
        plotter.enable_depth_peeling(number_of_peels=0)
        cameraSetUpRevised.setup_camera(plotter, 300, 600)
                        
        #plotter.add_mesh(glyphsObject, scalars = 'weights_minmax', cmap = cmap, opacity = 'opacity', show_edges=True, edge_color="grey")

        
        """climmax = max_kde_value
        if 'preferred' in stakeholder_data[agent][site]:
            print(f"{agent}/{site}/{tempstate} kde max is {max_kde_value}")
            print(f"{agent}/{site}/preferred kde max is {stakeholder_data[agent][site]['preferred']}")
            climmax = stakeholder_data[agent][site]['preferred']

        plotter.add_mesh(glyphsObject, scalars = 'intensity', cmap = cmap, opacity = 'opacity', show_edges=True, edge_color="grey", clim = [0, climmax])"""
        plotter.add_mesh(glyphsObject, scalars = 'intensity', cmap = cmap, opacity = 'intensity', show_edges=True, edge_color="grey")

        #plotter.add_mesh(glyphsObject, scalars = 'unmodified', cmap = cmap_name, opacity = 'opacity', show_edges=True, edge_color="grey")



        #glyphOrig = origs.glyph(geom=pv.Cube(), factor=cell_size, scale = False)
        #plotter.add_mesh(glyphOrig, color = 'white')



        plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.4))
        light2 = pv.Light(light_type='cameralight', intensity=.4)
        light2.specular = 0.3  # Reduced specular reflection
        plotter.add_light(light2)
        plotter.enable_eye_dome_lighting()




        glyphs.add_mesh_rgba(plotter, rest.points, 1, rest.point_data["RGBdesat"], rotation=70, isCol=False)
    


        # Visualize




        plotter.show()
        



#old
