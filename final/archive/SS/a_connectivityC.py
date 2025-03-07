import xarray as xr
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
from itertools import product
import pyvista as pv
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
from numba import njit
from numba.typed import List
import multiprocessing as mp

import final.SS.a_get_agent_nodes as a_get_agent_nodes  # Ensure this module is accessible and properly implemented


# --------------------- CONFIGURATION ---------------------

# Define default bias and maximum total bias
DEFAULT_BIAS = 0.3  # Adjust as needed
MAX_TOTAL_BIAS = 0.99

# Define resistance values
LOW_RESISTANCE = 1
NEUTRAL = 10
HIGH_RESISTANCE = 100
NOTTRAVEL = 1e6  # A very high resistance value for non-traversable voxels

# Define agents with their specific parameters
agents = {
    'bird': {
        'energy': 100000,                # Total energy available
        'speed': 1000,                   # Steps per iteration
        'autocorrelation': 0.3,          # Tendency to continue in the same direction
        'resistance_neighborhood_size': 5,  # Neighborhood size for resistance
        'resistance_focal_function': 'min', # Focal function for resistance ('mean', 'max', 'min')
        'risk': 0.1,                     # Default risk (overridden by risk surface)
        'name': 'bird',
        'travelOnGround': False          # New parameter
    },
    'lizard': {
        'energy': 5000,                  # Total energy available
        'speed': 1000,                   # Steps per iteration
        'autocorrelation': 0.3,          # Tendency to continue in the same direction
        'resistance_neighborhood_size': 2,  # Neighborhood size for resistance
        'resistance_focal_function': 'max', # Focal function for resistance ('mean', 'max', 'min')
        'risk': 0.2,                     # Default risk (overridden by risk surface)
        'name': 'lizard',
        'travelOnGround': True           # New parameter
    }
}

# Define number of simulation runs per agent per source node
NUM_RUNS = 500


# --------------------- FUNCTION DEFINITIONS ---------------------

def define_resistance_surface(ds, agents):
    """
    Define resistance surfaces for each agent based on the specified rules.
    Returns a dictionary of 3D numpy arrays: resistance_bird, resistance_lizard, etc.
    Also returns the grid shape.
    
    Parameters:
        ds (xr.Dataset): The loaded xarray Dataset containing voxel data.
        agents (dict): Dictionary of agents with their configurations.
        
    Returns:
        dict: Dictionary with agent names as keys and their respective resistance surfaces as values.
        tuple: Shape of the 3D grid.
    """    
    
    # Initialize dictionary to hold resistance surfaces for each agent
    resistance_surfaces = {}
    
    # Identify resource variables
    resource_vars = [var for var in ds.data_vars if var.startswith('resource_')]

    # Convert resource variables into a new 'resource' dimension
    if resource_vars:
        resource_data = ds[resource_vars].to_array(dim='resource')
        # Create a mask where any of the resource variables are not null
        low_res_mask = resource_data.notnull().any(dim='resource').values
    else:
        # If no resource variables, all voxels have high resistance
        low_res_mask = np.zeros(ds.dims['voxel'], dtype=bool)

    # Initialize resistance surface with neutral resistance
    resistance = np.full(ds.dims['voxel'], NEUTRAL)

    # Assign low resistance to voxels with any resource_ variables
    resistance[low_res_mask] = LOW_RESISTANCE

    # Apply high resistance rules
    # Rule 1: Buildings with height > 10
    if 'site_building_ID' in ds and 'site_LAS_HeightAboveGround' in ds:
        high_res_buildings = (ds['site_building_ID'].notnull()) & (ds['site_LAS_HeightAboveGround'] > 10)
        resistance[high_res_buildings.values] = HIGH_RESISTANCE

    # Rule 2: Roads with specific widths
    if 'road_roadInfo_width' in ds:
        road_widths = ds['road_roadInfo_width'].values
        high_res_roads = np.isin(road_widths, ['200 / 100 / 100', '300 / 100 / 100'])
        resistance[high_res_roads] = HIGH_RESISTANCE

    # Extract voxel indices
    if {'voxel_I', 'voxel_J', 'voxel_K'}.issubset(ds.data_vars):
        I = ds['voxel_I'].values
        J = ds['voxel_J'].values
        K = ds['voxel_K'].values
    else:
        raise KeyError("Dataset must contain 'voxel_I', 'voxel_J', and 'voxel_K' variables.")

    # Determine grid shape based on maximum indices
    grid_shape = (
        int(np.max(I)) + 1,
        int(np.max(J)) + 1,
        int(np.max(K)) + 1
    )

    # Initialize 3D resistance grid with neutral resistance
    resistance_3d = np.full(grid_shape, NEUTRAL)

    # Assign resistance values based on voxel indices
    resistance_3d[I, J, K] = resistance

    # Assign NOTTRAVEL to voxels below the lowest filled voxel in each (i, j) column
    print("Assigning 'NOTTRAVEL' resistance to appropriate voxels...")
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            filled_k = np.where(resistance_3d[i, j, :] != NEUTRAL)[0]
            if filled_k.size > 0:
                lowest_k = filled_k.min()
                # Assign NOTTRAVEL to all voxels below the lowest filled voxel
                resistance_3d[i, j, :lowest_k] = NOTTRAVEL

    # If agents have 'travelOnGround' == True, assign NOTTRAVEL based on additional criteria
    for agent_key, agent in agents.items():
        if agent.get('travelOnGround', False):
            print(f"Processing 'travelOnGround' for agent '{agent_key}'...")
            # Iterate through all voxels
            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    for k in range(grid_shape[2]):
                        if resistance_3d[i, j, k] == NEUTRAL:
                            # Check within +2 in all directions
                            i_min = max(i - 2, 0)
                            i_max = min(i + 3, grid_shape[0])
                            j_min = max(j - 2, 0)
                            j_max = min(j + 3, grid_shape[1])
                            k_min = max(k - 2, 0)
                            k_max = min(k + 3, grid_shape[2])
                            neighborhood = resistance_3d[i_min:i_max, j_min:j_max, k_min:k_max]
                            if not np.any(neighborhood != NEUTRAL):
                                resistance_3d[i, j, k] = NOTTRAVEL

    # Create separate resistance surfaces for each agent (copy of base resistance)
    for agent_key in agents.keys():
        resistance_surfaces[agent_key] = resistance_3d.copy()

    print("'NOTTRAVEL' resistance assignment completed.")

    return resistance_surfaces, grid_shape



def load_movement_nodes(site, voxel_size, agents, movementNodesOptions, scenarioName, base_dir='data/revised'):
    """
    Load or generate movement nodes based on the specified options.
    
    Parameters:
        site (str): Name of the site.
        voxel_size (int): Size of the voxel.
        agents (dict): Dictionary of agents.
        movementNodesOptions (str): Option for movement nodes ('load', 'choose', 'random').
        scenarioName (str): Name of the scenario.
        base_dir (str): Base directory for movement nodes files.
    
    Returns:
        dict: Dictionary with agent names as keys and their respective movement nodes as values.
              Each value is a dictionary containing 'site', 'agent', 'state', 'source_nodes', and 'target_nodes'.
    """
    movement_nodes = {}
    
    for agent_key in agents.keys():
        agent = agents[agent_key]
        if movementNodesOptions == 'load':
            # Load from JSON file: "data/revised/{site}-{agent}-{voxel_size}-movementnodes.json"
            filename = f"{site}-{agent_key}-{voxel_size}-movementnodes.json"
            filepath = os.path.join(base_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as file:
                    data = json.load(file)
                source_voxels = [tuple(node['voxel_indices']) for node in data.get('source_nodes', [])]
                target_voxels = []
                for node in data.get('target_nodes', []):
                    voxel = tuple(node['voxel_indices'])
                    bias = node.get('bias', DEFAULT_BIAS)
                    target_voxels.append({'coordinates': voxel, 'bias': bias})
                movement_nodes[agent_key] = {
                    "site": site,
                    "agent": agent_key,
                    "state": "loaded",
                    "source_nodes": source_voxels,
                    "target_nodes": target_voxels
                }
                print(f"Loaded movement nodes for agent '{agent_key}' from '{filepath}'.")
            else:
                print(f"Movement nodes file '{filepath}' not found for agent '{agent_key}'. Using default or empty nodes.")
                movement_nodes[agent_key] = {
                    "site": site,
                    "agent": agent_key,
                    "state": "file_not_found",
                    "source_nodes": [],
                    "target_nodes": []
                }
        elif movementNodesOptions == 'choose':
            # Use a_get_agent_nodes.getNodes
            nodes = a_get_agent_nodes.getNodes(site, [agent_key], voxel_size)
            movement_nodes[agent_key] = nodes.get(agent_key, {
                "site": site,
                "agent": agent_key,
                "state": "no_nodes_returned",
                "source_nodes": [],
                "target_nodes": []
            })
            print(f"Chosen movement nodes for agent '{agent_key}': {movement_nodes[agent_key]}")
        elif movementNodesOptions == 'random':
            # Initialize with empty lists; will assign randomly later
            movement_nodes[agent_key] = {
                "site": site,
                "agent": agent_key,
                "state": "randomly_initialized",
                "source_nodes": [],
                "target_nodes": []
            }
            print(f"Random movement nodes initialized for agent '{agent_key}'. Will assign after grid shape is known.")
        else:
            print(f"Invalid movementNodesOptions '{movementNodesOptions}' for agent '{agent_key}'. Using empty nodes.")
            movement_nodes[agent_key] = {
                "site": site,
                "agent": agent_key,
                "state": "invalid_option",
                "source_nodes": [],
                "target_nodes": []
            }
    
    return movement_nodes


def assign_random_movement_nodes(movement_nodes, grid_shape, num_sources=3, num_targets=2):
    """
    Assign random source and target nodes for agents.
    
    Parameters:
        movement_nodes (dict): Dictionary with agent names as keys and their movement nodes as values.
        grid_shape (tuple): Shape of the 3D grid.
        num_sources (int): Number of source nodes to assign.
        num_targets (int): Number of target nodes to assign.
    
    Returns:
        dict: Updated movement_nodes with randomly assigned source and target nodes.
    """
    for agent_key, nodes in movement_nodes.items():
        if nodes['state'] == 'randomly_initialized':
            sources = []
            targets = []
            for _ in range(num_sources):
                i = random.randint(0, grid_shape[0] - 1)
                j = random.randint(0, grid_shape[1] - 1)
                k = random.randint(0, grid_shape[2] - 1)
                sources.append((i, j, k))
            for _ in range(num_targets):
                i = random.randint(0, grid_shape[0] - 1)
                j = random.randint(0, grid_shape[1] - 1)
                k = random.randint(0, grid_shape[2] - 1)
                bias = random.uniform(0.05, 0.2)  # Assign random bias between 0.05 and 0.2
                targets.append({'coordinates': (i, j, k), 'bias': bias})
            movement_nodes[agent_key]['source_nodes'] = sources
            movement_nodes[agent_key]['target_nodes'] = targets
            movement_nodes[agent_key]['state'] = 'randomly_assigned'
            print(f"Randomly assigned {len(sources)} source nodes and {len(targets)} target nodes for agent '{agent_key}'.")
    return movement_nodes


def window_3d(scale):
    """
    Generate a list of 3D window offsets based on the given scale.

    Parameters:
        scale (int): Determines the size of the neighborhood. The window size will be (2*scale + 1)^3.

    Returns:
        list: List of (dx, dy, dz) tuples representing the neighborhood offsets.
    """
    w = []
    for dx in range(-scale, scale + 1):
        for dy in range(-scale, scale + 1):
            for dz in range(-scale, scale + 1):
                w.append((dx, dy, dz))
    return w



def focal_function_3d(ij, surf, scale, agent):
    """
    Compute the focal resistance value for a given voxel using the agent's focal function.

    Parameters:
        ij (tuple): Current voxel coordinates (i, j, k).
        surf (np.ndarray): 3D array of resistance values.
        scale (int): Neighborhood scale.
        agent (dict): Agent's parameters.

    Returns:
        float: Focal resistance value.
    """
    # Define focal functions: mean, max, min
    def fmean(ij, surf, scale):
        w = window_3d(scale)
        Aw = [(ij[0] + dx, ij[1] + dy, ij[2] + dz) for dx, dy, dz in w]
        Sw = [surf[x] for x in Aw if (0 <= x[0] < surf.shape[0] and
                                      0 <= x[1] < surf.shape[1] and
                                      0 <= x[2] < surf.shape[2])]
        return np.mean(Sw) if Sw else NEUTRAL

    def fmax(ij, surf, scale):
        w = window_3d(scale)
        Aw = [(ij[0] + dx, ij[1] + dy, ij[2] + dz) for dx, dy, dz in w]
        Sw = [surf[x] for x in Aw if (0 <= x[0] < surf.shape[0] and
                                      0 <= x[1] < surf.shape[1] and
                                      0 <= x[2] < surf.shape[2])]
        return np.max(Sw) if Sw else NEUTRAL

    def fmin(ij, surf, scale):
        w = window_3d(scale)
        Aw = [(ij[0] + dx, ij[1] + dy, ij[2] + dz) for dx, dy, dz in w]
        Sw = [surf[x] for x in Aw if (0 <= x[0] < surf.shape[0] and
                                      0 <= x[1] < surf.shape[1] and
                                      0 <= x[2] < surf.shape[2])]
        return np.min(Sw) if Sw else NEUTRAL

    # Map function names to actual functions
    fm = {
        'mean': fmean,
        'max': fmax,
        'min': fmin
    }

    # Retrieve the focal function based on agent parameters
    focal_func = fm.get(agent['resistance_focal_function'], fmean)  # Default to mean if not specified

    return focal_func(ij, surf, scale)


def get_neighbors(coord, grid_shape):
    """
    Get all valid neighboring coordinates in 3D.

    Parameters:
        coord (tuple): Current voxel coordinates (i, j, k).
        grid_shape (tuple): Shape of the 3D grid.

    Returns:
        list: List of neighboring voxel coordinates.
    """
    directions = list(product([-1, 0, 1], repeat=3))
    directions.remove((0, 0, 0))  # Exclude the current voxel

    neighbors = []
    for dx, dy, dz in directions:
        neighbor = (coord[0] + dx, coord[1] + dy, coord[2] + dz)
        if (0 <= neighbor[0] < grid_shape[0] and
            0 <= neighbor[1] < grid_shape[1] and
            0 <= neighbor[2] < grid_shape[2]):
            neighbors.append(neighbor)
    return neighbors


@njit
def simulate_movement_numba(start_i, start_j, start_k, resistance, risk, energy, speed, autocorrelation, targets_coords, targets_bias, grid_shape):
    # Initialize separate typed lists for i, j, k coordinates
    path_i = List()
    path_j = List()
    path_k = List()
    
    # Append the starting node
    path_i.append(start_i)
    path_j.append(start_j)
    path_k.append(start_k)
    
    current = (start_i, start_j, start_k)
    
    # Initialize previous direction as a float64 array
    prev_direction = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Calculate total target bias
    total_target_bias = 0.0
    for bias in targets_bias:
        total_target_bias += bias
    # Cap the total bias to 0.99 - autocorrelation
    if autocorrelation + total_target_bias > 0.99:
        scaling_factor = 0.99 - autocorrelation
        if scaling_factor <= 0.0:
            # No room for target bias
            scaled_target_bias = np.zeros(len(targets_bias))
        else:
            scaled_target_bias = np.array(targets_bias) * (scaling_factor / total_target_bias)
    else:
        scaled_target_bias = np.array(targets_bias)
    
    for _ in range(speed):
        if energy <= 0:
            break

        neighbors = []
        # Generate neighbors within the grid boundaries
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni = current[0] + di
                    nj = current[1] + dj
                    nk = current[2] + dk
                    if 0 <= ni < grid_shape[0] and 0 <= nj < grid_shape[1] and 0 <= nk < grid_shape[2]:
                        neighbors.append((ni, nj, nk))
        num_neighbors = len(neighbors)
        if num_neighbors == 0:
            break

        movement_probabilities = np.zeros(num_neighbors, dtype=np.float64)
        for idx in range(num_neighbors):
            neighbor = neighbors[idx]
            focal_res = resistance[neighbor[0], neighbor[1], neighbor[2]]
            movement_probabilities[idx] = 1.0 / focal_res if focal_res > 0 else 0.0

        total = movement_probabilities.sum()
        if total == 0:
            break
        movement_probabilities /= total

        # Apply autocorrelation
        if prev_direction[0] != 0.0 or prev_direction[1] != 0.0 or prev_direction[2] != 0.0:
            directions = np.zeros((num_neighbors, 3), dtype=np.float64)
            for idx in range(num_neighbors):
                neighbor = neighbors[idx]
                directions[idx, 0] = neighbor[0] - current[0]
                directions[idx, 1] = neighbor[1] - current[1]
                directions[idx, 2] = neighbor[2] - current[2]
            
            # Normalize directions
            for idx in range(num_neighbors):
                norm = np.sqrt(directions[idx, 0]**2 + directions[idx, 1]**2 + directions[idx, 2]**2)
                if norm != 0.0:
                    directions[idx, 0] /= norm
                    directions[idx, 1] /= norm
                    directions[idx, 2] /= norm

            prev_norm = np.sqrt(prev_direction[0]**2 + prev_direction[1]**2 + prev_direction[2]**2)
            if prev_norm != 0.0:
                prev_dir_normalized = prev_direction / prev_norm
            else:
                prev_dir_normalized = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            dot_products = np.zeros(num_neighbors, dtype=np.float64)
            for idx in range(num_neighbors):
                dot = (directions[idx, 0] * prev_dir_normalized[0] +
                       directions[idx, 1] * prev_dir_normalized[1] +
                       directions[idx, 2] * prev_dir_normalized[2])
                dot_products[idx] = (dot + 1.0) / 2.0  # Normalize to [0,1]

            # Normalize autocorr_bias
            autocorr_bias = dot_products.sum()
            if autocorr_bias > 0.0:
                autocorr_bias = dot_products / autocorr_bias
            else:
                autocorr_bias = np.ones(num_neighbors, dtype=np.float64) / num_neighbors

            # Combine with movement probabilities
            movement_probabilities = (1.0 - autocorrelation) * movement_probabilities + autocorrelation * autocorr_bias
            movement_probabilities /= movement_probabilities.sum()

        # Apply destination biases
        if len(targets_coords) > 0:
            bias_probs = np.zeros(num_neighbors, dtype=np.float64)
            for t in range(len(targets_coords)):
                target = targets_coords[t]
                bias = scaled_target_bias[t]
                # Calculate direction to target
                direction_to_target = np.array([
                    target[0] - current[0],
                    target[1] - current[1],
                    target[2] - current[2]
                ], dtype=np.float64)
                norm = np.sqrt(direction_to_target[0]**2 + direction_to_target[1]**2 + direction_to_target[2]**2)
                if norm != 0.0:
                    direction_to_target /= norm
                else:
                    direction_to_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                
                for idx in range(num_neighbors):
                    neighbor = neighbors[idx]
                    neighbor_dir = np.array([
                        neighbor[0] - current[0],
                        neighbor[1] - current[1],
                        neighbor[2] - current[2]
                    ], dtype=np.float64)
                    neighbor_norm = np.sqrt(neighbor_dir[0]**2 + neighbor_dir[1]**2 + neighbor_dir[2]**2)
                    if neighbor_norm != 0.0:
                        neighbor_dir /= neighbor_norm
                    else:
                        neighbor_dir = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    
                    dot = (neighbor_dir[0] * direction_to_target[0] +
                           neighbor_dir[1] * direction_to_target[1] +
                           neighbor_dir[2] * direction_to_target[2])
                    bias_probs[idx] += bias * ((dot + 1.0) / 2.0)  # Normalize to [0,1]

            total_bias = bias_probs.sum()
            if total_bias > 0.0:
                bias_probs /= total_bias
                movement_probabilities = movement_probabilities + bias_probs
                movement_probabilities /= movement_probabilities.sum()

        # Choose the next step based on probabilities
        cumulative = np.cumsum(movement_probabilities)
        rand = np.random.random()
        next_idx = 0
        while next_idx < len(cumulative) and rand > cumulative[next_idx]:
            next_idx += 1
        if next_idx >= len(cumulative):
            next_idx = len(cumulative) - 1
        next_step = neighbors[next_idx]
        
        # Append the next step to the path
        path_i.append(next_step[0])
        path_j.append(next_step[1])
        path_k.append(next_step[2])

        # Update energy
        energy -= resistance[next_step[0], next_step[1], next_step[2]]
        if energy <= 0:
            break

        # Update previous direction with float differences
        prev_direction[0] = float(next_step[0] - current[0])
        prev_direction[1] = float(next_step[1] - current[1])
        prev_direction[2] = float(next_step[2] - current[2])
        current = next_step

        # Apply risk of stopping
        current_risk = risk[next_step[0], next_step[1], next_step[2]]
        if np.random.random() < current_risk:
            break  # Agent stops moving due to risk

    # Combine the separate lists into a list of tuples
    path = List()
    for idx in range(len(path_i)):
        path.append((path_i[idx], path_j[idx], path_k[idx]))
    
    return path

def visualize_travelable_voxels(resistance_surfaces, ds, agents, save_dir='pathwalker_3d_results'):
    """
    Visualize travelable voxels for each agent by plotting voxels with resistance less than NOTTRAVEL.

    Parameters:
        resistance_surfaces (dict): Dictionary with agent names as keys and their respective resistance surfaces as values.
        ds (xr.Dataset): xarray Dataset containing voxel grid information.
        agents (dict): Dictionary of agents.
        save_dir (str): Directory where the plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    for agent_key, resistance in resistance_surfaces.items():
        # Identify travelable voxels
        travelable_mask = resistance < NOTTRAVEL
        travelable_voxels = np.argwhere(travelable_mask)

        if travelable_voxels.size == 0:
            print(f"No travelable voxels found for agent '{agent_key}'.")
            continue

        # Convert voxel indices to centroid coordinates
        centroids = np.array([ijk_to_centroid(i, j, k, ds) for i, j, k in travelable_voxels])

        # Create PyVista PolyData
        polydata = pv.PolyData(centroids)

        # Initialize the plotter
        plotter = pv.Plotter()
        plotter.add_points(polydata, color='green', point_size=5, render_points_as_spheres=True, name='Travelable Voxels')

        # Add axes and title
        plotter.add_axes()
        plotter.add_title(f"Travelable Voxels for '{agent_key}'")

        # Save the plot as an image
        image_path = os.path.join(save_dir, f"{agent_key}_travelable_voxels.png")
        plotter.show(screenshot=image_path, auto_close=True)
        print(f"Travelable voxels for '{agent_key}' saved as '{image_path}'.")


def worker_simulate(agent_key, agent_params, source, resistance, risk, grid_shape, target_nodes):
    """
    Simulate movement for a single agent starting from a source node.
    
    Parameters:
        agent_key (str): Name of the agent.
        agent_params (dict): Parameters for the agent.
        source (tuple): Starting voxel indices (i, j, k).
        resistance (np.ndarray): 3D resistance surface.
        risk (np.ndarray): 3D risk surface.
        grid_shape (tuple): Shape of the 3D grid.
        target_nodes (list of dict): List of target nodes with 'coordinates' and 'bias'.
    
    Returns:
        tuple: (agent_key, path) where path is a list of (i, j, k) tuples.
    """
    # Extract targets coordinates and biases
    targets_coords = [tuple(target['coordinates']) for target in target_nodes]
    targets_bias = [target['bias'] for target in target_nodes]
    
    path = simulate_movement_numba(
        start_i=source[0],
        start_j=source[1],
        start_k=source[2],
        resistance=resistance,
        risk=risk,
        energy=agent_params['energy'],
        speed=agent_params['speed'],
        autocorrelation=agent_params['autocorrelation'],
        targets_coords=targets_coords,
        targets_bias=targets_bias,
        grid_shape=grid_shape
    )
    
    # Convert Numba List of tuples to a standard Python list of tuples
    python_path = []
    for voxel in path:
        python_path.append((voxel[0], voxel[1], voxel[2]))

    # Debug: Print the path length
    if len(python_path) > 0:
        print(f"Agent '{agent_key}' started at {python_path[0]}")
        print(f"Path length: {len(python_path)}")
    
    return (agent_key, python_path)


def aggregate_paths(paths, grid_shape):
    """
    Aggregate multiple paths into a density surface using NumPy's advanced indexing.

    Parameters:
        paths (list): List of paths, each path is a list of (i, j, k) tuples.
        grid_shape (tuple): Shape of the 3D grid.

    Returns:
        np.ndarray: 3D numpy array representing movement density.
    """
    density = np.zeros(grid_shape, dtype=np.int32)
    for path in paths:
        for voxel in path:
            # Ensure voxel indices are within bounds
            if (0 <= voxel[0] < grid_shape[0] and
                0 <= voxel[1] < grid_shape[1] and
                0 <= voxel[2] < grid_shape[2]):
                density[voxel[0], voxel[1], voxel[2]] += 1
    return density


def ijk_to_centroid(i, j, k, ds):
    """
    Converts voxel grid indices (I, J, K) to the corresponding voxel centroid (x, y, z) coordinates
    using the bounds and voxel size stored in the xarray Dataset.

    Parameters:
        i (int): Voxel index along the I-axis.
        j (int): Voxel index along the J-axis.
        k (int): Voxel index along the K-axis.
        ds (xr.Dataset): xarray Dataset containing voxel grid information, bounds, and voxel size.

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


def create_pyvista_polydata(density_surface, ds):
    """
    Convert the movement density surface into a PyVista PolyData object using centroid coordinates.

    Parameters:
        density_surface (np.ndarray): 3D numpy array of movement density.
        ds (xr.Dataset): xarray Dataset containing voxel grid information.

    Returns:
        pv.PolyData: PyVista PolyData object.
    """
    # Get voxel indices where density >=1
    voxel_indices = np.argwhere(density_surface >= 1)
    if voxel_indices.size == 0:
        raise ValueError("No voxels with density >=1 found.")

    # Initialize lists to store centroid coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    movement_frequency = []

    for idx in voxel_indices:
        i, j, k = idx
        x, y, z = ijk_to_centroid(i, j, k, ds)
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        movement_frequency.append(density_surface[i, j, k])

    # Convert lists to numpy arrays
    points = np.column_stack((x_coords, y_coords, z_coords))
    movement_frequency = np.array(movement_frequency)

    # Create PyVista points
    polydata = pv.PolyData(points)

    # Add movement frequency as point data
    polydata.point_data["movement_frequency"] = movement_frequency

    return polydata


def visualize_pyvista_polydata(polydata, agent_name='agent', save_dir='pathwalker_3d_results'):
    """
    Visualize the PyVista PolyData object and save it.

    Parameters:
        polydata (pv.PolyData): PyVista PolyData object.
        agent_name (str): Name of the agent for plot title and filename.
        save_dir (str): Directory where the plot and PolyData will be saved.
    """
    # Initialize the plotter
    plotter = pv.Plotter()

    # Add movement density
    plotter.add_mesh(polydata, 
                     scalars="movement_frequency", 
                     cmap="hot", 
                     point_size=5, 
                     render_points_as_spheres=True, 
                     opacity=0.8, 
                     name='Movement Density')

    # Add color bar
    plotter.add_scalar_bar(title="Movement Frequency", label_font_size=16, title_font_size=18)

    # Set camera position
    plotter.view_isometric()

    # Add axes
    plotter.add_axes()

    # Save the PolyData as a VTK file
    vtk_filename = os.path.join(save_dir, f'{agent_name}_movement_density.vtk')
    polydata.save(vtk_filename)
    print(f"PyVista PolyData for '{agent_name}' movement density saved as VTK file at '{vtk_filename}'.")

def sanity_check_resistance(resistance_surfaces, grid_shape, agents):
    """
    Perform sanity checks on resistance surfaces to ensure 'NOTTRAVEL' voxels are correctly assigned.

    Parameters:
        resistance_surfaces (dict): Dictionary with agent names as keys and their respective resistance surfaces as values.
        grid_shape (tuple): Shape of the 3D grid.
        agents (dict): Dictionary of agents.

    Returns:
        None
    """
    for agent_key, resistance in resistance_surfaces.items():
        num_not_travel = np.sum(resistance == NOTTRAVEL)
        num_travel = np.sum(resistance < NOTTRAVEL)
        print(f"Agent '{agent_key}':")
        print(f"  Travelable voxels: {num_travel}")
        print(f"  NOTTRAVEL voxels: {num_not_travel}")
        print(f"  Total voxels: {grid_shape[0] * grid_shape[1] * grid_shape[2]}")


def visualize_resistance_surface(resistance_surface, ds, save_dir='pathwalker_3d_results'):
    """
    Visualize the 3D resistance surface using PyVista.

    Parameters:
        resistance_surface (np.ndarray): 3D numpy array of resistance values.
        ds (xr.Dataset): xarray Dataset containing voxel grid information.
        save_dir (str): Directory where the plot and PolyData will be saved.
    """
    # Get indices where resistance <= threshold and != NOTTRAVEL
    threshold = 200  # Adjust as needed
    indices = np.argwhere((resistance_surface <= threshold) & (resistance_surface != NOTTRAVEL))
    if indices.size == 0:
        print("No voxels with resistance <= threshold found.")
        return

    # Initialize lists to store centroid coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    resistance_values = []

    for idx in indices:
        i, j, k = idx
        x, y, z = ijk_to_centroid(i, j, k, ds)
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        resistance_values.append(resistance_surface[i, j, k])

    # Convert lists to numpy arrays
    points = np.column_stack((x_coords, y_coords, z_coords))
    resistance_values = np.array(resistance_values)

    # Create PyVista points
    polydata = pv.PolyData(points)

    # Add resistance values as point data
    polydata.point_data["resistance"] = resistance_values

    # Initialize the plotter
    plotter = pv.Plotter()

    # Add resistance surface
    plotter.add_mesh(polydata, 
                     scalars="resistance", 
                     cmap="viridis", 
                     point_size=3, 
                     render_points_as_spheres=True, 
                     opacity=0.6, 
                     name='Resistance Surface')

    # Add color bar
    plotter.add_scalar_bar(title="Resistance Value", label_font_size=16, title_font_size=18)

    # Set camera position
    plotter.view_isometric()

    # Add axes
    plotter.add_axes()

    # Save the resistance PolyData as a VTK file
    vtk_filename = os.path.join(save_dir, 'resistance_surface.vtk')
    polydata.save(vtk_filename)
    print(f"PyVista PolyData for resistance surface saved as VTK file at '{vtk_filename}'.")


def visualize_density_slice(density_surface, resistance_surface, slice_index, axis='z', agent_name='agent', sources=[], targets=[], save_dir='pathwalker_3d_results'):
    """
    Visualize a 2D slice of the movement density surface with resistance overlay and source/target nodes.

    Parameters:
        density_surface (np.ndarray): 3D numpy array of movement density.
        resistance_surface (np.ndarray): 3D numpy array of resistance values.
        slice_index (int): Index of the slice to visualize.
        axis (str): Axis along which to take the slice ('x', 'y', or 'z').
        agent_name (str): Name of the agent for plot title and filename.
        sources (list): List of source voxel indices as tuples (i, j, k).
        targets (list): List of target voxel indices as tuples (i, j, k).
        save_dir (str): Directory where the plot will be saved.
    """
    
    print(f'Source nodes: {sources}')
    print(f'Target nodes: {targets}')
    if axis == 'x':
        density_slice = density_surface[slice_index, :, :]
        resistance_slice = resistance_surface[slice_index, :, :]
        # Project sources and targets onto Y-Z plane
        source_coords = [(j, k) for (i, j, k) in sources if i == slice_index]
        target_coords = [(j, k) for (i, j, k) in targets if i == slice_index]
        xlabel, ylabel = 'Y-axis', 'Z-axis'
    elif axis == 'y':
        density_slice = density_surface[:, slice_index, :]
        resistance_slice = resistance_surface[:, slice_index, :]
        # Project sources and targets onto X-Z plane
        source_coords = [(i, k) for (i, j, k) in sources if j == slice_index]
        target_coords = [(i, k) for (i, j, k) in targets if j == slice_index]
        xlabel, ylabel = 'X-axis', 'Z-axis'
    else:
        density_slice = density_surface[:, :, slice_index]
        resistance_slice = resistance_surface[:, :, slice_index]
        # Project sources and targets onto X-Y plane
        source_coords = [(i, j) for (i, j, k) in sources if k == slice_index]
        target_coords = [(i, j) for (i, j, k) in targets if k == slice_index]
        xlabel, ylabel = 'X-axis', 'Y-axis'

    # Create masks
    populated_mask = density_slice >= 1
    high_density_mask = density_slice >= 10  # Adjust threshold as needed

    # Plot using matplotlib
    plt.figure(figsize=(10, 8))

    # Plot resistance as background
    plt.imshow(resistance_slice, cmap='Greys', origin='lower', alpha=0.3)

    # Overlay movement density
    plt.imshow(np.where(populated_mask, density_slice, np.nan), cmap='hot', origin='lower', alpha=0.7)

    # Plot high-density areas
    plt.imshow(np.where(high_density_mask, density_slice, np.nan), cmap='autumn', origin='lower', alpha=0.5)

    # Project and plot source nodes
    if source_coords:
        x_sources, y_sources = zip(*source_coords)
        plt.scatter(x_sources, y_sources, marker='*', color='blue', s=150, label='Source Nodes', edgecolors='black')

    # Project and plot target nodes
    if target_coords:
        x_targets, y_targets = zip(*target_coords)
        plt.scatter(x_targets, y_targets, marker='X', color='green', s=100, label='Target Nodes', edgecolors='black')

    plt.title(f'{agent_name.capitalize()} Movement Density - Slice {axis.upper()}={slice_index}')
    plt.colorbar(label='Movement Frequency')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{agent_name}_density_slice_{axis}_{slice_index}.png'))
    plt.close()
    print(f"{agent_name.capitalize()} movement density slice ({axis}={slice_index}) saved as PNG.")


def save_density_surface(density, agent_name='agent', results_dir='pathwalker_3d_results'):
    """
    Save the entire density surface as a NumPy binary file.

    Parameters:
        density (np.ndarray): 3D numpy array of movement density.
        agent_name (str): Name of the agent for filename.
        results_dir (str): Directory to save the density file.
    """
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'{agent_name}_density.npy'), density)
    print(f"'{agent_name.capitalize()}' movement density surface saved as .npy file at '{os.path.join(results_dir, f'{agent_name}_density.npy')}'.")


def load_custom_risk_surface(json_filepath, grid_shape, agents):
    """
    Load a custom risk surface from a JSON file.

    Parameters:
        json_filepath (str): Path to the JSON file containing custom risk data.
        grid_shape (tuple): Shape of the 3D grid.
        agents (dict): Dictionary of agents.

    Returns:
        dict: Dictionary with agent names as keys and their respective risk surfaces as values.
    """
    if not os.path.isfile(json_filepath):
        print(f"Custom risk surface file '{json_filepath}' not found. Using default risk surfaces.")
        return None

    with open(json_filepath, 'r') as file:
        data = json.load(file)
    
    # Extract voxel indices and risk values
    if 'risk_surface' not in data:
        print("No 'risk_surface' key found in JSON. Using default risk surfaces.")
        return None

    risk_data = data['risk_surface']  # Assuming the JSON has a 'risk_surface' key
    # Initialize dictionary to hold custom risk surfaces for each agent
    custom_risk_surfaces = {}

    for agent_key in agents.keys():
        # Initialize a zeroed risk surface
        risk_surface = np.zeros(grid_shape, dtype=float)

        for voxel in risk_data:
            i, j, k = voxel['voxel_indices']
            risk = voxel['risk']
            if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1] and 0 <= k < grid_shape[2]:
                risk_surface[i, j, k] = risk

        custom_risk_surfaces[agent_key] = risk_surface
        print(f"Custom risk surface loaded for agent '{agent_key}'.")

    return custom_risk_surfaces


def save_paths_and_nodes_as_vtk(paths, movement_nodes, ds, results_dir, sphere_radius=0.5):
    """
    Save movement paths and nodes as VTK files using PyVista.

    Parameters:
        paths (dict): Dictionary of paths for each agent.
        movement_nodes (dict): Dictionary of movement nodes for each agent.
        ds (xr.Dataset): xarray Dataset containing voxel grid information.
        results_dir (str): Directory to save the VTK files.
        sphere_radius (float): Radius of the spheres representing nodes.
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f'Paths: {paths}')
    print(f'Movement nodes: {movement_nodes}')

    for agent, agent_paths in paths.items():
        path_file = os.path.join(results_dir, f'{agent}_movement_paths.vtk')
        nodes_file = os.path.join(results_dir, f'{agent}_nodes.vtk')

        # Convert all paths to spatial coordinates
        paths_spatial = []
        for path in agent_paths:
            spatial_path = [ijk_to_centroid(i, j, k, ds) for voxel in path for i, j, k in [voxel]]
            paths_spatial.append(spatial_path)

        if not paths_spatial:
            print(f"No paths to save for agent '{agent}'. Skipping.")
            continue

        # Flatten the list of paths into a single numpy array for all points
        all_points = np.vstack(paths_spatial)

        # Create a lines array for PyVista
        lines = []
        offset = 0
        for path in paths_spatial:
            n_points = len(path)
            lines.append([n_points] + list(range(offset, offset + n_points)))
            offset += n_points

        # Convert lines to a flat numpy array
        lines = np.hstack(lines)

        # Create PyVista PolyData object for paths
        polypaths = pv.PolyData(all_points, lines)
        polypaths.save(path_file)
        print(f"Saved movement paths for '{agent}' to '{path_file}'.")

        # Convert source voxels to spatial coordinates
        source_voxels = movement_nodes[agent]['source_nodes']
        if source_voxels:
            start_nodes = np.array([ijk_to_centroid(*voxel, ds) for voxel in source_voxels])
            start_pv = pv.PolyData(start_nodes)
            start_spheres = start_pv.glyph(scale=False, geom=pv.Sphere(radius=sphere_radius))
        else:
            start_spheres = pv.PolyData()

        # Convert target voxels to spatial coordinates
        target_voxels = movement_nodes[agent]['target_nodes']
        if target_voxels:
            end_nodes = np.array([ijk_to_centroid(*voxel['coordinates'], ds) for voxel in target_voxels])
            end_pv = pv.PolyData(end_nodes)
            end_spheres = end_pv.glyph(scale=False, geom=pv.Sphere(radius=sphere_radius))
        else:
            end_spheres = pv.PolyData()

        # Combine start and end spheres into one PolyData object
        if start_spheres.n_points > 0 and end_spheres.n_points > 0:
            nodes = start_spheres.merge(end_spheres)
        elif start_spheres.n_points > 0:
            nodes = start_spheres
        elif end_spheres.n_points > 0:
            nodes = end_spheres
        else:
            nodes = pv.PolyData()

        if nodes.n_points > 0:
            # Save the combined nodes
            nodes.save(nodes_file)
            print(f"Saved start and end nodes for '{agent}' to '{nodes_file}'.")
        else:
            print(f"No nodes to save for agent '{agent}'. Skipping node saving.")


def runSimulations(siteName, voxelSize=5, scenarioName='original', movementNodesOptions='load'):
    """
    Run movement simulations based on the provided configuration.

    Parameters:
        siteName (str): Name of the site.
        voxelSize (int, optional): Size of the voxel. Defaults to 5.
        scenarioName (str, optional): Name of the scenario. Defaults to 'default_scenario'.
        movementNodesOptions (str, optional): Option for movement nodes ('load', 'choose', 'random'). Defaults to 'load'.
    """
    # Define paths based on input arguments
    base_dir = 'data/revised'
    RESULTS_DIR = os.path.join(base_dir, f'pathwalker_3d_results_{scenarioName}')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Define path to the dataset
    DATASET_PATH = os.path.join(base_dir, f'xarray_voxels{scenarioName}_{siteName}_{voxelSize}.nc')

    if not os.path.isfile(DATASET_PATH):
        print(f"Dataset file '{DATASET_PATH}' not found. Exiting simulation.")
        return

    # --------------------- LOAD DATA ---------------------
    print("Loading dataset...")
    ds = xr.open_dataset(DATASET_PATH)
    # Print all variable names in ds
    print(f"Variable names in dataset: {ds.data_vars}")

    # --------------------- TIDY DATA ---------------------
    print("Tidying data by removing voxels with 'isTerrainUnderBuilding' == True...")
    if 'isTerrainUnderBuilding' in ds:
        terrain_under_building = ds['isTerrainUnderBuilding'].values
        num_voxels_before = terrain_under_building.size
        voxels_to_remove = terrain_under_building == True
        num_voxels_removed = np.sum(voxels_to_remove)
        print(f"Number of voxels to remove: {num_voxels_removed} out of {num_voxels_before}")

        # Extract voxel indices where isTerrainUnderBuilding == True
        I = ds['voxel_I'].values[voxels_to_remove]
        J = ds['voxel_J'].values[voxels_to_remove]
        K = ds['voxel_K'].values[voxels_to_remove]
        removed_voxels = list(zip(I, J, K))
        print(f"Removed voxel indices: {removed_voxels}")

        # Note: The actual removal is handled later in the resistance surface
    else:
        print("'isTerrainUnderBuilding' variable not found in dataset. No voxels removed.")
        removed_voxels = []

    # --------------------- DEFINE RESISTANCE ---------------------
    print("Defining resistance surfaces for each agent...")
    resistance_surfaces, grid_shape = define_resistance_surface(ds, agents)
    print("Resistance surfaces defined.")

    # --------------------- SANITY CHECK ---------------------
    print("Performing sanity checks on resistance surfaces...")
    sanity_check_resistance(resistance_surfaces, grid_shape, agents)
    print("Sanity checks completed.")

    # --------------------- VISUALIZE TRAVELABLE VOXELS ---------------------
    print("Visualizing travelable voxels for each agent...")
    visualize_travelable_voxels(
        resistance_surfaces=resistance_surfaces,
        ds=ds,
        agents=agents,
        save_dir=RESULTS_DIR
    )
    print("Travelable voxels visualization completed.")


    # --------------------- LOAD MOVEMENT NODES ---------------------
    print("Loading movement nodes...")
    movement_nodes = load_movement_nodes(
        site=siteName,
        voxel_size=voxelSize,
        agents=agents,
        movementNodesOptions=movementNodesOptions,
        scenarioName=scenarioName,
        base_dir=base_dir
    )
    
    # If movementNodesOptions is 'random', assign random movement nodes now that grid_shape is known
    if movementNodesOptions == 'random':
        movement_nodes = assign_random_movement_nodes(movement_nodes, grid_shape)

    # --------------------- APPLY VOXEL REMOVAL ---------------------
    if removed_voxels:
        print("Applying 'NOTTRAVEL' resistance to removed voxels...")
        for agent_key in agents.keys():
            for voxel in removed_voxels:
                i, j, k = voxel
                if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1] and 0 <= k < grid_shape[2]:
                    resistance_surfaces[agent_key][i, j, k] = NOTTRAVEL
        print(f"Applied 'NOTTRAVEL' to {len(removed_voxels)} voxels for all agents.")
    else:
        print("No voxels to remove. Skipping 'NOTTRAVEL' assignment.")

    # --------------------- LOAD RISK SURFACE ---------------------
    print("Loading risk surfaces...")
    # Check if a custom risk surface JSON file exists
    # For simplicity, assume 'custom_risk_surface.json' contains risk data
    custom_risk_surface_file = os.path.join(base_dir, 'custom_risk_surface.json')  # Modify as needed
    if os.path.isfile(custom_risk_surface_file):
        custom_risk_surfaces = load_custom_risk_surface(custom_risk_surface_file, grid_shape, agents)
        if custom_risk_surfaces:
            risk_surfaces = custom_risk_surfaces
        else:
            # Use default risk surfaces
            min_res = min(resist.min() for resist in resistance_surfaces.values())
            max_res = max(resist.max() for resist in resistance_surfaces.values())
            risk_surfaces = {}
            for agent_key, resistance in resistance_surfaces.items():
                rescaled = 1 + 99 * (resistance - min_res) / (max_res - min_res)  # Rescale to [1, 100]
                risk = rescaled / 1000  # Divide by 1000
                risk_surfaces[agent_key] = risk
                print(f"Default risk surface created for agent '{agent_key}'.")
    else:
        # Use default risk surfaces
        min_res = min(resist.min() for resist in resistance_surfaces.values())
        max_res = max(resist.max() for resist in resistance_surfaces.values())
        risk_surfaces = {}
        for agent_key, resistance in resistance_surfaces.items():
            rescaled = 1 + 99 * (resistance - min_res) / (max_res - min_res)  # Rescale to [1, 100]
            risk = rescaled / 1000  # Divide by 1000
            risk_surfaces[agent_key] = risk
            print(f"Default risk surface created for agent '{agent_key}'.")

    print("Risk surfaces loaded.")

    # --------------------- SIMULATE MOVEMENT ---------------------
    print("Starting movement simulations...")
    simulated_paths = {agent_key: [] for agent_key in agents.keys()}

    # Prepare tasks for multiprocessing
    tasks = []
    for agent_key, agent_params in agents.items():
        agent_movement = movement_nodes.get(agent_key, {})
        source_voxels = agent_movement.get('source_nodes', [])
        target_voxels = agent_movement.get('target_nodes', [])
        for source in source_voxels:
            for run in range(NUM_RUNS):
                tasks.append((agent_key, agent_params, source, resistance_surfaces[agent_key],
                              risk_surfaces[agent_key], grid_shape, target_voxels))

    # Use multiprocessing Pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(worker_simulate, tasks)

    # Aggregate results
    for agent_key, path in results:
        simulated_paths[agent_key].append(path)

    print("Movement simulations completed.")

    # --------------------- AGGREGATE PATHS ---------------------
    print("Aggregating movement paths into density surfaces...")
    density_surfaces = {}
    for agent_key, paths in simulated_paths.items():
        density = aggregate_paths(paths, grid_shape)
        density_surfaces[agent_key] = density
        save_density_surface(density, agent_name=agent_key, results_dir=RESULTS_DIR)
    print("Aggregation completed.")

    # --------------------- VISUALIZE RESULTS ---------------------
    print("Visualizing movement density and resistance surfaces...")
    for agent_key, density in density_surfaces.items():
        print(f"Visualizing results for agent '{agent_key}'...")
        # Visualize 2D slice (middle slice along z-axis)
        slice_index = grid_shape[2] // 2
        movement_nodes_agent = movement_nodes.get(agent_key, {})
        sources = movement_nodes_agent.get('source_nodes', [])
        targets = [target['coordinates'] for target in movement_nodes_agent.get('target_nodes', [])]
        visualize_density_slice(
            density_surface=density,
            resistance_surface=resistance_surfaces[agent_key],
            slice_index=slice_index,
            axis='z',
            agent_name=agent_key,
            sources=sources,
            targets=targets,
            save_dir=RESULTS_DIR
        )

        # Visualize 3D movement density with PyVista
        try:
            polydata = create_pyvista_polydata(density, ds)
            visualize_pyvista_polydata(
                polydata=polydata,
                agent_name=agent_key,
                save_dir=RESULTS_DIR
            )
        except ValueError as e:
            print(f"Visualization skipped for agent '{agent_key}': {e}")
        print(f"Visualization for agent '{agent_key}' completed.")

    # Save agent paths and nodes as VTK files
    print("Saving movement paths and nodes as VTK files...")
    save_paths_and_nodes_as_vtk(simulated_paths, movement_nodes, ds, RESULTS_DIR)
    
    # Visualize resistance surface (shared among agents)
    print("Visualizing resistance surface...")
    # Assuming all agents share the same resistance initially, otherwise choose appropriately
    # Here, we'll visualize for the first agent
    first_agent = list(agents.keys())[0]
    visualize_resistance_surface(
        resistance_surface=resistance_surfaces[first_agent],
        ds=ds,
        save_dir=RESULTS_DIR
    )
    print("Resistance surface visualization completed.")

    print("All simulations and visualizations are complete. Check the results directory.")


# --------------------- MAIN EXECUTION ---------------------

def main():
    # Define simulation parameters
    site = 'city'  # Replace with your site name
    scenarioName = 'original'  # Replace with your scenario name
    movementNodesOptions = 'load'  # Options: 'load', 'choose', 'random'
    voxelSize = 5  # Replace with your voxel size

    # Run the simulations
    runSimulations(
        siteName=site,
        voxelSize=voxelSize,
        scenarioName=scenarioName,
        movementNodesOptions=movementNodesOptions
    )

if __name__ == "__main__":
    main()
