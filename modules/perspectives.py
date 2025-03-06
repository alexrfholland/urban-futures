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


#tempstate = 'trending'
#tempstate = 'now'
tempstate = 'preferred'

site = 'street'
#agents = ['reptile']
agents = ['bird']

#site = 'city'
#agents = ['reptile']
#agents = ['bird']

#site = 'trimmed-parade'
#agents = ['tree']


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


###stakeholders

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
                'perchable branch': 10,
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



###VOXELISE

def plot_weight_distribution(weights, plot_type='histogram', bins=30, name=None, **kwargs):
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
        plt.title(f'{name} - Histogram of Weights')
    elif plot_type == 'kde':
        density = gaussian_kde(weights)
        xs = np.linspace(min(weights), max(weights), 1000)
        plt.plot(xs, density(xs), **kwargs)
        plt.title(f'{name} - Kernel Density Estimate of Weights')
    else:
        raise ValueError("plot_type must be either 'histogram' or 'kde'")
    
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def calculate_voxel_centroids_and_weights(points, weights, bounds, cell_size):
    """
    Calculate centroids and sum weights of points within each voxel.
    
    Parameters:
    - points: (N, 3) numpy array of point coordinates.
    - weights: (N,) numpy array of weights corresponding to the points.
    - bounds: Tuple of (min_x, max_x, min_y, max_y, min_z, max_z) defining the bounds of the grid.
    - cell_size: The edge length of each voxel in the grid.
    
    Returns:
    - centroids: (M, 3) numpy array of voxel centroids.
    - summed_weights: (M,) numpy array of summed weights for each voxel.
    """
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
    voxel_keys = voxel_indices_z * (num_voxels_y * num_voxels_x) + voxel_indices_y * num_voxels_x + voxel_indices_x
    
    # Find unique voxel keys and their reverse indices
    unique_keys, reverse_indices = np.unique(voxel_keys, return_inverse=True)
    
    # Sum weights for each unique voxel
    summed_weights = np.bincount(reverse_indices, weights=weights)
    
    # Calculate centroids for unique voxels
    centroids_x = (unique_keys % num_voxels_x) * cell_size + min_x + cell_size / 2
    centroids_y = ((unique_keys // num_voxels_x) % num_voxels_y) * cell_size + min_y + cell_size / 2
    centroids_z = (unique_keys // (num_voxels_x * num_voxels_y)) * cell_size + min_z + cell_size / 2
    centroids = np.vstack((centroids_x, centroids_y, centroids_z)).T


    
    return centroids, summed_weights


def normalize_weights(weights, new_min=0, new_max=1, existingMin=None, existingMax=None):
    """
    Normalize the weights array to a new given range [new_min, new_max], optionally using existing min and max values.
    
    Parameters:
    - weights: numpy array of original weights.
    - new_min: the minimum value in the normalized range.
    - new_max: the maximum value in the normalized range.
    - existingMin: optional, the existing minimum value of the weights. If not provided, min(weights) will be used.
    - existingMax: optional, the existing maximum value of the weights. If not provided, max(weights) will be used.
    
    Returns:
    - normalized_weights: numpy array of normalized weights.
    """
    # Ensure weights is a numpy array to handle the operations correctly
    weights = np.array(weights)
    
    # Use provided existingMin and existingMax if given, otherwise calculate from weights
    orig_min = existingMin if existingMin is not None else weights.min()
    orig_max = existingMax if existingMax is not None else weights.max()
    
    # Normalize the weights to the new range
    normalized_weights = new_min + ((weights - orig_min) * (new_max - new_min)) / (orig_max - orig_min)
    
    return normalized_weights



def distribute_resource_points(centers, weights, maxKDE=None):

    # Create a PolyData object
    grid = pv.PolyData(centers)

    print(f'centers is {centers}')
    print(f'weights is {weights}')
    print(f'maxKDE is {maxKDE}')

    if maxKDE is not None:
        weights = normalize_weights(weights, 0,1,0,maxKDE)

    else:
        weights = normalize_weights(weights, 0,1)

    # Add the weights as a scalar data to the grid
    grid["intensity"] = weights

    #normalized_weights = normalize_weights(weights, new_min = alphaMin, new_max=1)

    #grid['opacity'] = normalized_weights

    #print(f'max weight is {np.max(weights)}')

    return grid


#######KDE

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


def weighted_3d_kde(centers_1d, weights, bandwidth_factor=0.5):
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





if __name__ == "__main__":
    
    

    
    if tempstate == 'preferred':
        state = 'now'
        siteMultiBlock = pv.read(f'data/{site}/all_resources-{site}-{state}.vtm')
        treeblock = siteMultiBlock
    else:
        state = tempstate
        siteMultiBlock = siteMultiBlock = pv.read(f'data/{site}/combined-{site}-{state}.vtm')
        treeblock = siteMultiBlock.get('trees')


    cellSize = 2.5



    for agent in agents:

        print(f'agent is {agent}')
    
        points, weights, allResources, bounds, cmap = process_agent_resources(treeblock, agent_name=agent)
        print(f'new bounds are {bounds}')

 
        rest = siteMultiBlock.get('rest of points')

        print(f'points are length {len(points)}')
        print(f'weights are length {len(weights)}')
        print(f'bounds are {bounds}')
        


        centroids, summed_weights = calculate_voxel_centroids_and_weights(points, weights, bounds, cellSize)
        total_voxel_weight = np.sum(summed_weights)
        maxVoxelWeight = np.max(summed_weights)

        plot_weight_distribution(summed_weights, name=f'{site}-{agent}-{tempstate}')


        kde, kdecenters, kdeweights = weighted_3d_kde(centroids, summed_weights, bandwidth_factor=.2)


        #for trees
        #kde, kdecenters, kdeweights = weighted_3d_kde(centroids, summed_weights, bandwidth_factor=.2)


        # Find the maximum KDE value
        max_kde_value = np.max(kdeweights)  # Replace 'weights' with the array of KDE values


        ######

        # Scale KDE values by total weight, moderated by adjustment factor
        scaled_kdeweights = kdeweights * (maxVoxelWeight * 0.1)




        scaled_max_kde = np.max(scaled_kdeweights)

        # Update the dictionary
        if agent not in stakeholder_data:
            stakeholder_data[agent] = {}
        if site not in stakeholder_data[agent]:
            stakeholder_data[agent][site] = {}
        stakeholder_data[agent][site][tempstate] = max_kde_value


        maxResKey = f'{tempstate}-maxresources'
        stakeholder_data[agent][site][maxResKey] = maxVoxelWeight
        stakeholder_data[agent][site][f'{tempstate}-summedVoxels'] = total_voxel_weight
        stakeholder_data[agent][site][f"{tempstate}-scaledMaxKDE"] = scaled_max_kde

        print(f"{agent}/{site}/{tempstate} kde max is {max_kde_value}")
        print(f"{agent}/{site}/{maxResKey} max resources in a voxel is {np.max(summed_weights)}")
        print(f"{agent}/{site}/{f'{tempstate}-summedVoxels'} summed voxel weights are {total_voxel_weight}")
        print(f"{agent}/{site}/{f'{tempstate}-scaledMaxKDE'} scaled max KDE Value by Total Voxel Summed Weight are {scaled_max_kde}")

        with open(stakeholder_file, "w") as file:
            json.dump(stakeholder_data, file, indent=4)

        if 'preferred' in stakeholder_data[agent][site]:
            print(f"{agent}/{site}/preferred kde max is {stakeholder_data[agent][site]['preferred']}")
            print(f"{agent}/{site}/preferred summed voxels weights is {stakeholder_data[agent][site]['preferred-summedVoxels']}")
            print(f"{agent}/{site}/preferred max resources in a voxel is {stakeholder_data[agent][site]['preferred-maxresources']}")
            print(f"{agent}/{site}/preferred scaled max kde is {stakeholder_data[agent][site]['preferred-scaledMaxKDE']}")

            max_kde_value = stakeholder_data[agent][site]['preferred']
            scaled_max_kde = stakeholder_data[agent][site]['preferred-scaledMaxKDE']
            maxVoxelWeight = stakeholder_data[agent][site]['preferred-maxresources']

        useScaled = True

        if useScaled:
            kdeweights = scaled_kdeweights
            max_kde_value = scaled_max_kde

        

        grid = distribute_resource_points(kdecenters, kdeweights, max_kde_value)
                        
        #grid = distribute_resource_points(centroids, summed_weights, maxVoxelWeight)

                #for name in ['intensity', 'weights_minmax', 'weights_quartile', 'weights_zscore', 'weights_log', 'weights_robust']:

        # Create and configure the plotter
        plotter = pv.Plotter()
        plotter.enable_depth_peeling(number_of_peels=5)
        cameraSetUpRevised.setup_camera(plotter, 300, 600, name='stakeholders')

        #display outlines for near 0 values
        threshold_value = 0.2
        above_threshold = grid.threshold(value=threshold_value, scalars="intensity")
        below_threshold = grid.threshold(value=threshold_value, scalars="intensity", invert=True)



                        
        #plotter.add_mesh(glyphsObject, scalars = 'intensity', cmap = cmap, opacity = 'opacity', show_edges=True, edge_color="grey", clim=[0,25]
        if above_threshold.n_points > 0:
            glyphsObject = above_threshold.glyph(orient=False, scale=False, factor=cellSize, geom=pv.Cube())
            plotter.add_mesh(glyphsObject, scalars = 'intensity', cmap = cmap, opacity = 'intensity', show_edges=True, edge_color="grey", clim = [0,.5])

        if below_threshold.n_points > 0:
            glyphsBelow = below_threshold.glyph(orient=False, scale=False, factor=cellSize, geom=pv.Cube())
            plotter.add_mesh(glyphsBelow, scalars = 'intensity', cmap = cmap, opacity = 'intensity', style='wireframe', clim = [0,.5])








        plotter.enable_eye_dome_lighting()

        glyphs.add_mesh_rgba(plotter, rest.points, 1, rest.point_data["RGBdesat"], rotation=70, isCol=False)
    
        plotter.show()
        


