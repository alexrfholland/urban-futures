import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import a_helper_functions

import pyvista as pv
import numpy as np
import random
import matplotlib.pyplot as plt
import xarray as xr

import numpy as np
import random
import matplotlib.pyplot as plt
import xarray as xr

from scipy.ndimage import label
import copy

def advanced_cull_small_groups(rewilded, voxel_coords, threshold):
    """
    Cull small clusters of growth in the rewilded grid based on the binary mask of rewilded voxels.
    Clusters are identified by treating the grid as a binary mask (not subdivided by nodeID).
    
    Args:
    - rewilded: The array containing growth IDs for each voxel.
    - voxel_coords: The voxel coordinates (I, J, K).
    - threshold: Minimum number of voxels in a cluster for it to remain unculted.
    
    Returns:
    - rewilded: The updated array with small clusters culled.
    """
    # Create binary mask (1 for rewilded voxels, 0 for non-rewilded voxels)
    binary_rewilded = np.where(rewilded != -1, 1, 0)

    # Identify clusters using connected components in the binary mask
    # `label` finds connected regions of 1's in the binary mask
    structure = np.ones((3, 3, 3))  # 3x3x3 connectivity (26-connected neighborhood)
    labeled_clusters, num_clusters = label(binary_rewilded, structure=structure)

    print(f"Number of end clusters identified: {num_clusters}")

    # Cull clusters smaller than the threshold
    for cluster_id in range(1, num_clusters + 1):
        cluster_size = np.sum(labeled_clusters == cluster_id)
        if cluster_size < threshold:
            # Set the small cluster's voxels back to -1 (culled)
            rewilded[labeled_clusters == cluster_id] = -1

    return rewilded

def precompute_neighbors(voxel_coords, coord_to_idx):
    """
    Precompute the neighboring voxels for each voxel, including first-shell and second-shell neighbors.
    First shell includes direct neighbors, and second shell includes diagonals at a distance of 2.
    """
    directions_1st_shell = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    directions_2nd_shell = [
        (-2, 0, 0), (2, 0, 0), (0, -2, 0), (0, 2, 0), (0, 0, -2), (0, 0, 2),
        (-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0), 
        (0, -1, -1), (0, 1, 1), (0, -1, 1), (0, 1, -1),
        (-1, 0, -1), (1, 0, 1), (-1, 0, 1), (1, 0, -1),
        (-2, -2, -2), (2, 2, 2), (2, -2, -2), (-2, 2, 2)
    ]
    
    neighbors_first_shell = [[] for _ in range(len(voxel_coords))]
    neighbors_second_shell = [[] for _ in range(len(voxel_coords))]

    for idx, (i, j, k) in enumerate(voxel_coords):
        # First-shell neighbors
        for di, dj, dk in directions_1st_shell:
            neighbor_idx = coord_to_idx.get((i + di, j + dj, k + dk), -1)
            if neighbor_idx != -1:
                neighbors_first_shell[idx].append(neighbor_idx)

        # Second-shell neighbors
        for di, dj, dk in directions_2nd_shell:
            neighbor_idx = coord_to_idx.get((i + di, j + dj, k + dk), -1)
            if neighbor_idx != -1:
                neighbors_second_shell[idx].append(neighbor_idx)

    return neighbors_first_shell, neighbors_second_shell

def grow_plants(grid_data, params, max_turns=100, seed=None):
    """
    Simulate growth from starting nodes with an energy budget and resistance.
    Includes both first-shell and second-shell neighbors for growth.

    Args:
    - grid_data: Precomputed grid information including neighbors and resistance.
    - params: Dictionary of simulation parameters.
    - max_turns: Maximum number of turns for the simulation (default=100).
    - seed: Seed for the random number generator to ensure reproducibility (default=None).

    Returns:
    - growth_origin: An array tracking the origin nodeID for each voxel.
    - growth_turn: An array tracking the turn number when each voxel was rewilded.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Starting energy is determined from the `initial_energy` dictionary in params.
    growth_origin = np.full(grid_data['num_voxels'], -1, dtype=int)
    growth_turn = np.full(grid_data['num_voxels'], -1, dtype=int)  # New array to track turns
    neighbors_1st_shell = grid_data['neighbors_1st_shell']
    neighbors_2nd_shell = grid_data['neighbors_2nd_shell']
    node_ids = grid_data['analysis_nodeID']
    resistances = grid_data['analysis_combined_resistance']
    node_sizes = grid_data['analysis_nodeSize']
    node_types = grid_data['analysis_nodeType']

    # Retrieve parameters
    resistance_factor = params.get('resistance_factor', 10)
    resistance_threshold = params.get('resistance_threshold', 50)
    high_resistance_cutoff = params.get('high_resistance_cutoff', 80)
    high_termination_chance = params.get('high_termination_chance', 0.8)
    fast_growth_split_chance = params.get('fast_growth_split_chance', 0.65)

    # Initialize energy for each voxel based on nodeSize and nodeType
    energy = np.full(grid_data['num_voxels'], -1, dtype=float)

    for start_voxel in grid_data['start_voxels']:
        node_size = node_sizes[start_voxel]
        node_type = node_types[start_voxel]
        starting_energy = params['initial_energy'].get((node_size, node_type), 1000)
        
        growth_stack = [(start_voxel, starting_energy, 0)]  # Added turn number (0) to the stack

        while growth_stack:
            current_idx, current_energy, current_turn = growth_stack.pop()
            if growth_origin[current_idx] != -1 or current_energy <= 0 or current_turn >= max_turns:
                continue

            growth_origin[current_idx] = node_ids[start_voxel]
            growth_turn[current_idx] = current_turn  # Record the turn number

            if resistances[current_idx] <= resistance_threshold:
                energy_loss = 0
                split_chance = fast_growth_split_chance
                termination_chance = 0
            elif resistances[current_idx] > high_resistance_cutoff:
                energy_loss = resistances[current_idx] / resistance_factor
                termination_chance = high_termination_chance
                split_chance = params['split_chance']
            else:
                energy_loss = resistances[current_idx] / resistance_factor
                termination_chance = params['termination_chance']
                split_chance = params['split_chance']

            new_energy = current_energy - energy_loss

            if new_energy <= 0 or random.random() < termination_chance:
                continue

            available_neighbors = [n for n in neighbors_1st_shell[current_idx] if growth_origin[n] == -1]
            if random.random() < 0.5:
                available_neighbors += [n for n in neighbors_2nd_shell[current_idx] if growth_origin[n] == -1]

            if available_neighbors:
                num_new = random.randint(1, min(3, len(available_neighbors))) if random.random() < split_chance else 1
                for neighbor in random.sample(available_neighbors, num_new):
                    growth_stack.append((neighbor, new_energy, current_turn + 1))

    return growth_origin, growth_turn

def grow_plants2(grid_data, params, max_turns=100, seed=None):
    """
    Simulate growth from starting nodes with an energy budget and resistance.
    Includes both first-shell and second-shell neighbors for growth.

    Args:
    - grid_data: Precomputed grid information including neighbors and resistance.
    - params: Dictionary of simulation parameters.
    - max_turns: Maximum number of turns for the simulation (default=100).
    - seed: Seed for the random number generator to ensure reproducibility (default=None).

    Returns:
    - growth_origin: An array tracking the origin nodeID for each voxel.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Starting energy is determined from the `initial_energy` dictionary in params.
    growth_origin = np.full(grid_data['num_voxels'], -1, dtype=int)
    neighbors_1st_shell = grid_data['neighbors_1st_shell']
    neighbors_2nd_shell = grid_data['neighbors_2nd_shell']
    node_ids = grid_data['analysis_nodeID']
    resistances = grid_data['analysis_combined_resistance']
    node_sizes = grid_data['analysis_nodeSize']
    node_types = grid_data['analysis_nodeType']

    # Retrieve parameters
    resistance_factor = params.get('resistance_factor', 10)
    resistance_threshold = params.get('resistance_threshold', 50)
    high_resistance_cutoff = params.get('high_resistance_cutoff', 80)
    high_termination_chance = params.get('high_termination_chance', 0.8)
    fast_growth_split_chance = params.get('fast_growth_split_chance', 0.65)

    # Initialize energy for each voxel based on nodeSize and nodeType
    energy = np.full(grid_data['num_voxels'], -1, dtype=float)

    for start_voxel in grid_data['start_voxels']:
        node_size = node_sizes[start_voxel]
        node_type = node_types[start_voxel]
        starting_energy = params['initial_energy'].get((node_size, node_type), 1000)
        
        growth_stack = [(start_voxel, starting_energy)]

        for _ in range(max_turns):
            if not growth_stack:
                break

            current_idx, current_energy = growth_stack.pop()
            if growth_origin[current_idx] != -1 or current_energy <= 0:
                continue

            growth_origin[current_idx] = node_ids[start_voxel]

            if resistances[current_idx] <= resistance_threshold:
                energy_loss = 0
                split_chance = fast_growth_split_chance
                termination_chance = 0
            elif resistances[current_idx] > high_resistance_cutoff:
                energy_loss = resistances[current_idx] / resistance_factor
                termination_chance = high_termination_chance
                split_chance = params['split_chance']
            else:
                energy_loss = resistances[current_idx] / resistance_factor
                termination_chance = params['termination_chance']
                split_chance = params['split_chance']

            new_energy = current_energy - energy_loss

            if new_energy <= 0 or random.random() < termination_chance:
                continue

            available_neighbors = [n for n in neighbors_1st_shell[current_idx] if growth_origin[n] == -1]
            if random.random() < 0.5:
                available_neighbors += [n for n in neighbors_2nd_shell[current_idx] if growth_origin[n] == -1]

            if available_neighbors:
                num_new = random.randint(1, min(3, len(available_neighbors))) if random.random() < split_chance else 1
                for neighbor in random.sample(available_neighbors, num_new):
                    growth_stack.append((neighbor, new_energy))

    return growth_origin, growth_origin



def plot_growth_by_nodeID(growth_origin, voxel_coords):
    """
    Plot the growth tracking each voxel's origin nodeID in 3D.
    """
    I, J, K = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
    unique_node_ids = np.unique(growth_origin[growth_origin != -1])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for node_id in unique_node_ids:
        node_voxels = np.where(growth_origin == node_id)[0]
        ax.scatter(I[node_voxels], J[node_voxels], K[node_voxels], label=f'NodeID {node_id}', s=5, alpha=0.6)

    ax.set_xlabel('I')
    ax.set_ylabel('J')
    ax.set_zlabel('K')
    ax.legend()
    plt.title("3D Voxel Growth Simulation by NodeID")
    plt.show()

def initialize_grid(xarray_dataset):
    """
    Extracts grid data from the xarray dataset.
    Includes first-shell and second-shell neighbors.
    """
    voxel_coords = np.vstack([xarray_dataset['voxel_I'].values, 
                              xarray_dataset['voxel_J'].values, 
                              xarray_dataset['voxel_K'].values]).T
    coord_to_idx = {tuple(voxel): idx for idx, voxel in enumerate(voxel_coords)}
    start_voxels = np.where(xarray_dataset['analysis_nodeID'].values != -1)[0]
    node_ids = xarray_dataset['analysis_nodeID'].values
    resistances = xarray_dataset['analysis_combined_resistance'].values

    # Extract nodeSize and nodeType from dataset
    node_sizes = xarray_dataset['analysis_nodeSize'].values
    node_types = xarray_dataset['analysis_nodeType'].values

    neighbors_1st_shell, neighbors_2nd_shell = precompute_neighbors(voxel_coords, coord_to_idx)

    return {
        'num_voxels': len(voxel_coords),
        'voxel_coords': voxel_coords,
        'coord_to_idx': coord_to_idx,
        'start_voxels': start_voxels,
        'neighbors_1st_shell': neighbors_1st_shell,
        'neighbors_2nd_shell': neighbors_2nd_shell,
        'analysis_nodeID': node_ids,
        'analysis_combined_resistance': resistances,
        'analysis_nodeSize': node_sizes,
        'analysis_nodeType': node_types
    }


def load_xarray(file_path):
    """
    Load the xarray dataset.
    """
    return xr.open_dataset(file_path)

def save_results_to_xarray(xarray_dataset, growth_origin, starting_nodes, output_path):
    """
    Add the simResults (growth_origin) and startingNodes to the xarray dataset and save the updated dataset.
    """
    # Add the growth_origin array as 'simResults' to the dataset
    xarray_dataset['simResults'] = (['voxel'], growth_origin)

    # Add the starting_nodes array as 'startingNodes' to the dataset
    xarray_dataset['startingNodes'] = (['voxel'], starting_nodes)

    # Save the updated dataset to a new NetCDF file
    xarray_dataset.to_netcdf(output_path)
    print(f"Results saved to {output_path}")

def create_starting_nodes_array(grid_data):
    """
    Create an array that marks the starting nodes for each voxel.
    1 for starting nodes, 0 for others.
    """
    starting_nodes = np.zeros(grid_data['num_voxels'], dtype=int)
    starting_nodes[grid_data['start_voxels']] = 1  # Mark starting nodes as 1
    return starting_nodes

def GetNodeAreas(ds, treeDF,poleDF,logDF):
    canopy_voxels = ds.where(ds['sim_Nodes'] > 0, drop=True)

    # Print unique values and occurrences in sim_nodes
    unique_sim_nodes, counts = np.unique(canopy_voxels['sim_Nodes'].values, return_counts=True)
    for value, count in zip(unique_sim_nodes, counts):
        print(f'Value {value} occurs {count} times in sim_nodes')
    
    # Calculate the number of voxels per sim group
    voxel_counts = canopy_voxels.groupby('sim_Nodes').count()['centroid_x'].to_pandas()

    # Calculate voxel area (voxel_size^2) from the 'voxel_size' attribute in the dataset

    # Create simNodes column in treeDF: number of voxels per group * voxel area
    treeDF['sim_NodesVoxels'] = treeDF['NodeID'].map(voxel_counts)
    poleDF['sim_NodesVoxels'] = poleDF['NodeID'].map(voxel_counts)
    logDF['sim_NodesVoxels'] = logDF['NodeID'].map(voxel_counts)

    #add areas columns to treeDF, poleDF, logDF
    treeDF['sim_NodesArea'] = treeDF['sim_NodesVoxels'] * ds.attrs['voxel_size']**2
    poleDF['sim_NodesArea'] = poleDF['sim_NodesVoxels'] * ds.attrs['voxel_size']**2
    logDF['sim_NodesArea'] = logDF['sim_NodesVoxels'] * ds.attrs['voxel_size']**2

    dataframes = {'treeDF': treeDF, 'poleDF': poleDF, 'logDF': logDF}


    return dataframes

def GetRewildingNodes(xarray_dataset, treeDF, poleDF, logDF):
    grid_data = initialize_grid(xarray_dataset)


    startingEnergy = {
        ('large', 'tree'): 2500,
        ('medium', 'tree'): 2000,
        ('small', 'tree'): 1000,
        ('large', 'log'): 750,
        ('medium', 'log'): 500,
        ('small', 'log'): 250,
        ('medium', 'pole'): 125
    }

    # Define simulation parameters including energy, resistance, and threshold
    params = {
    'initial_energy': startingEnergy,             # Initial energy for each growth
    'resistance_factor': 50,            # Factor for energy loss due to resistance
    'resistance_threshold': 50,         # Resistance below this means no energy loss
    'high_resistance_cutoff': 80,       # Above this resistance, termination chance is high
    'high_termination_chance': 0.8,     # Termination chance when resistance > 80
    'fast_growth_split_chance': 0.65,   # Split chance when resistance is low
    'split_chance': 0.35,               # Default split chance
    'termination_chance': 0.2           # Default termination chance
    }

    max_turns = 5000

    # Run simulation with resistance and energy budget
    seed = 0
    growth_origin, growth_turn = grow_plants(grid_data, params, max_turns=max_turns, seed=0)


    # Ensure starting nodes array is of integer type
    xarray_dataset['sim_startingNodes'] = (['voxel'], create_starting_nodes_array(grid_data).astype(int))

    # Ensure growth origin array is of integer type
    xarray_dataset['sim_Nodes'] = (['voxel'], growth_origin.astype(int))
    xarray_dataset['sim_Turns'] = (['voxel'], growth_turn.astype(int))

    dataframes = GetNodeAreas(xarray_dataset, treeDF, poleDF, logDF)

    return xarray_dataset, dataframes


# Main Execution
if __name__ == "__main__":
    sites = ['city', 'trimmed-parade', 'uni']
    voxel_size = 1
    xarrays = {}
    polydatas = {}
    sites = ['trimmed-parade']

  
    for site in sites:
        print(f'processing rewilding for {site}')
        input_folder = f'data/revised/final/{site}'
        filepath = f'{input_folder}/{site}_{voxel_size}_voxelArray_withResistance.nc'

        xarray_dataset = xr.open_dataset(filepath)
        
        treeDF = pd.read_csv(f'{input_folder}/{site}_{voxel_size}_treeDF.csv')
        poleDF = pd.read_csv(f'{input_folder}/{site}_{voxel_size}_poleDF.csv')
        logDF = pd.read_csv(f'{input_folder}/{site}_{voxel_size}_logDF.csv')  
        
        xarray_dataset, dataframes = GetRewildingNodes(xarray_dataset,treeDF, poleDF, logDF)

        #save updated dataframes
        for df in dataframes:
            print(f'Saving {df} to {input_folder}/{site}_{voxel_size}_{df}.csv')
            dataframes[df].to_csv(f'{input_folder}/{site}_{voxel_size}_{df}.csv', index=False)


        polydata = a_helper_functions.convert_xarray_into_polydata(xarray_dataset)

        polydata.save(f'{input_folder}/{site}_{voxel_size}_voxelGrowResults.vtk')

        

        xarray_dataset.to_netcdf(f'{input_folder}/{site}_{voxel_size}_voxelArray_Nodes.nc')

        xarrays[site] = xarray_dataset
        polydatas[site] = polydata



        #Create a plotter object
        """plotter = pv.Plotter(shape=(1, 2))

        #print bounds of simTurns and other stats of it
        print(f'simTurns bounds: {np.min(growth_turn)} to {np.max(growth_turn)}')
        print(f'simTurns mean: {np.mean(growth_turn)}')
        print(f'simTurns median: {np.median(growth_turn)}')
        print(f'simTurns std: {np.std(growth_turn)}')

        # Plot simResults
        plotter.subplot(0, 0)
        # Clip data to range [0, 1000] and color out of bounds in white
        plotter.add_mesh(polydata, scalars='simTurns', cmap='turbo', clim=[0, max_turns],below_color='white', show_scalar_bar=True)
        plotter.add_title('Simulation Results')

        # Create a new polydata for nodes
        node_polydata = polydata.extract_points(polydata.point_data['analysis_nodeType'] != 'unassigned')
        plotter.add_mesh(node_polydata, scalars='analysis_nodeType', render_points_as_spheres=True, point_size=20, cmap='Set1')

        # Add EDL (Eye-Dome Lighting) to the plotter
        plotter.enable_eye_dome_lighting()

        # Plot analysis_combined_resistance
        plotter.subplot(0, 1)
        polydata_copy = polydata.copy(deep=True)

        plotter.add_mesh(polydata_copy, scalars='analysis_combined_resistance', cmap='coolwarm', show_scalar_bar=True)
        plotter.add_title('Combined Resistance')

        # Show the plot""
        plotter.show()"""
    
        
