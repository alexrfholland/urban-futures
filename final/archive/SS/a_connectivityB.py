import xarray as xr
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
from itertools import product
import pyvista as pv
from numba import njit
from numba.typed import List as NumbaList
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
NOTTRAVEL = -1  # New resistance value for non-traversable voxels

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

def initialize_flat_grid(xarray_dataset):
    """
    Initializes the flattened (1D) voxel grid and related variables from the xarray dataset.

    Parameters:
        xarray_dataset (xarray.Dataset): Loaded xarray dataset.

    Returns:
        tuple: (voxel_grid, grid_shape)
            voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
            grid_shape (tuple): Shape of the original 3D grid (I, J, K).
    """
    # Extract voxel indices and counts
    voxel_ids = xarray_dataset['voxel'].values
    I = xarray_dataset['voxel_I'].values
    J = xarray_dataset['voxel_J'].values
    K = xarray_dataset['voxel_K'].values
    num_voxels = len(voxel_ids)

    # Initialize voxel grid: columns - i, j, k, resistance, risk
    voxel_grid = np.zeros((num_voxels, 5), dtype=np.float32)
    voxel_grid[:, 0] = I  # i
    voxel_grid[:, 1] = J  # j
    voxel_grid[:, 2] = K  # k
    voxel_grid[:, 3] = NEUTRAL  # resistance (initially neutral)
    voxel_grid[:, 4] = 0.0  # risk (to be assigned later)

    # Determine grid shape based on maximum indices
    grid_shape = (int(np.max(I)) + 1, int(np.max(J)) + 1, int(np.max(K)) + 1)

    return voxel_grid, grid_shape

def precompute_neighbors(voxel_grid, grid_shape):
    """
    Precomputes the neighbor indices for each voxel in the flattened grid.

    Parameters:
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        grid_shape (tuple): Shape of the original 3D grid (I, J, K).

    Returns:
        np.ndarray: 2D array of neighbor indices with shape (num_voxels, 26).
                    Unused neighbor slots are filled with -1.
    """
    num_voxels = voxel_grid.shape[0]
    neighbors = np.full((num_voxels, 26), -1, dtype=np.int32)

    # Create a mapping from (i, j, k) to voxel index for quick lookup
    coord_to_idx = {}
    for idx in range(num_voxels):
        i, j, k = int(voxel_grid[idx, 0]), int(voxel_grid[idx, 1]), int(voxel_grid[idx, 2])
        coord_to_idx[(i, j, k)] = idx

    # Define all possible neighbor directions (26 neighbors in 3D)
    directions = list(product([-1, 0, 1], repeat=3))
    directions.remove((0, 0, 0))  # Exclude the voxel itself

    for idx in range(num_voxels):
        i, j, k = int(voxel_grid[idx, 0]), int(voxel_grid[idx, 1]), int(voxel_grid[idx, 2])
        neighbor_count = 0
        for dx, dy, dz in directions:
            ni, nj, nk = i + dx, j + dy, k + dz
            if (0 <= ni < grid_shape[0] and
                0 <= nj < grid_shape[1] and
                0 <= nk < grid_shape[2]):
                neighbor_idx = coord_to_idx.get((ni, nj, nk), -1)
                if neighbor_idx != -1:
                    neighbors[idx, neighbor_count] = neighbor_idx
                    neighbor_count += 1
        # Remaining slots are already -1

    return neighbors

def define_resistance_and_risk_surfaces(ds, voxel_grid, grid_shape, neighbors):
    """
    Define resistance and risk surfaces for each agent based on the specified rules using a flattened grid.

    Parameters:
        ds (xarray.Dataset): The loaded xarray Dataset containing voxel data.
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        grid_shape (tuple): Shape of the original 3D grid (I, J, K).
        neighbors (np.ndarray): 2D array of neighbor indices with shape (num_voxels, 26).

    Returns:
        tuple: (resistance_surfaces, risk_surfaces)
            resistance_surfaces (dict): Dictionary with agent names as keys and their respective resistance arrays as values.
            risk_surfaces (dict): Dictionary with agent names as keys and their respective risk arrays as values.
    """
    num_voxels = voxel_grid.shape[0]
    resistance_surfaces = {agent: np.full(num_voxels, NEUTRAL, dtype=np.int32) for agent in agents.keys()}
    risk_surfaces = {agent: np.full(num_voxels, agents[agent]['risk'], dtype=np.float32) for agent in agents.keys()}

    # Identify resource variables
    resource_vars = [var for var in ds.data_vars if var.startswith('resource_')]

    if resource_vars:
        # Create a mask where any of the resource variables are not null
        resource_data = np.any(np.stack([ds[var].values for var in resource_vars], axis=0), axis=0).flatten()
        for agent in agents.keys():
            resistance_surfaces[agent] = np.where(resource_data, LOW_RESISTANCE, resistance_surfaces[agent])
    else:
        print("No resource variables found in dataset.")

    # Apply high resistance rules
    # Rule 1: Buildings with height > 10
    if {'site_building_ID', 'site_LAS_HeightAboveGround'}.issubset(ds.data_vars):
        building_mask = (ds['site_building_ID'].notnull()) & (ds['site_LAS_HeightAboveGround'] > 10)
        building_mask = building_mask.values.flatten()
        for agent in agents.keys():
            resistance_surfaces[agent] = np.where(building_mask, HIGH_RESISTANCE, resistance_surfaces[agent])
    else:
        print("Building-related variables not found in dataset.")

    # Rule 2: Roads with specific widths
    if 'road_roadInfo_width' in ds:
        road_widths = ds['road_roadInfo_width'].values.flatten()
        high_res_roads = np.isin(road_widths, ['200 / 100 / 100', '300 / 100 / 100'])
        for agent in agents.keys():
            resistance_surfaces[agent] = np.where(high_res_roads, HIGH_RESISTANCE, resistance_surfaces[agent])
    else:
        print("'road_roadInfo_width' variable not found in dataset.")

    # Assign NOTTRAVEL to voxels below the lowest filled voxel in each (i, j) column
    print("Assigning 'NOTTRAVEL' resistance to voxels below the lowest filled voxel in each (i, j) column...")
    for agent in agents.keys():
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                # Get all voxel indices for this (i, j)
                voxel_indices = np.where((voxel_grid[:, 0] == i) & (voxel_grid[:, 1] == j))[0]
                if voxel_indices.size == 0:
                    continue
                # Find the lowest k with non-NEUTRAL resistance
                filled = voxel_indices[resistance_surfaces[agent][voxel_indices] != NEUTRAL]
                if filled.size == 0:
                    continue
                lowest_k = int(voxel_grid[filled, 2].min())
                # Assign NOTTRAVEL to all voxels below lowest_k
                below_mask = (voxel_grid[:, 0] == i) & (voxel_grid[:, 1] == j) & (voxel_grid[:, 2] < lowest_k)
                below_indices = np.where(below_mask)[0]
                resistance_surfaces[agent][below_indices] = NOTTRAVEL

    # Handle 'travelOnGround' parameter
    for agent_key, agent in agents.items():
        if agent.get('travelOnGround', False):
            print(f"Processing 'travelOnGround' for agent '{agent_key}'...")
            # Assign NOTTRAVEL to voxels that are not on the ground surface (k > 0)
            non_ground = voxel_grid[:, 2] > 0
            resistance_surfaces[agent_key] = np.where(non_ground, NOTTRAVEL, resistance_surfaces[agent_key])

    print("'NOTTRAVEL' resistance assignment completed.")

    return resistance_surfaces, risk_surfaces

def load_movement_nodes_flat(site, voxel_size, agents, movementNodesOptions, scenarioName, base_dir='data/revised', voxel_grid=None, grid_shape=None):
    """
    Load or generate movement nodes based on the specified options using a flattened grid.

    Parameters:
        site (str): Name of the site.
        voxel_size (int): Size of the voxel.
        agents (dict): Dictionary of agents.
        movementNodesOptions (str): Option for movement nodes ('load', 'choose', 'random').
        scenarioName (str): Name of the scenario.
        base_dir (str): Base directory for movement nodes files.
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        grid_shape (tuple): Shape of the original 3D grid (I, J, K).

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
                # Convert voxel coordinates to voxel indices
                source_voxels = [coord_to_index(voxel_grid, tuple(node['voxel_indices'])) for node in data.get('source_nodes', [])]
                target_voxels = []
                for node in data.get('target_nodes', []):
                    voxel_idx = coord_to_index(voxel_grid, tuple(node['voxel_indices']))
                    bias = node.get('bias', DEFAULT_BIAS)
                    if voxel_idx is not None:
                        target_voxels.append({'index': voxel_idx, 'bias': bias})
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

    # If movementNodesOptions is 'random', assign random movement nodes now that grid_shape is known
    if movementNodesOptions == 'random':
        movement_nodes = assign_random_movement_nodes_flat(movement_nodes, voxel_grid, grid_shape)

    return movement_nodes

def coord_to_index(voxel_grid, coord):
    """
    Convert voxel coordinates to voxel index in the flattened grid.

    Parameters:
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        coord (tuple): Voxel coordinates (i, j, k).

    Returns:
        int or None: Voxel index if found, else None.
    """
    matches = np.where((voxel_grid[:, 0] == coord[0]) &
                       (voxel_grid[:, 1] == coord[1]) &
                       (voxel_grid[:, 2] == coord[2]))[0]
    if matches.size > 0:
        return int(matches[0])
    else:
        return None

def assign_random_movement_nodes_flat(movement_nodes, voxel_grid, grid_shape, num_sources=3, num_targets=2):
    """
    Assign random source and target nodes for agents using a flattened grid.

    Parameters:
        movement_nodes (dict): Dictionary with agent names as keys and their movement nodes as values.
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        grid_shape (tuple): Shape of the original 3D grid (I, J, K).
        num_sources (int): Number of source nodes to assign.
        num_targets (int): Number of target nodes to assign.

    Returns:
        dict: Updated movement_nodes with randomly assigned source and target nodes.
    """
    num_voxels = voxel_grid.shape[0]
    for agent_key, nodes in movement_nodes.items():
        if nodes['state'] == 'randomly_initialized':
            sources = []
            targets = []
            for _ in range(num_sources):
                voxel_idx = random.randint(0, num_voxels - 1)
                sources.append(voxel_idx)
            for _ in range(num_targets):
                voxel_idx = random.randint(0, num_voxels - 1)
                bias = random.uniform(0.05, 0.2)  # Assign random bias between 0.05 and 0.2
                targets.append({'index': voxel_idx, 'bias': bias})
            movement_nodes[agent_key]['source_nodes'] = sources
            movement_nodes[agent_key]['target_nodes'] = targets
            movement_nodes[agent_key]['state'] = 'randomly_assigned'
            print(f"Randomly assigned {len(sources)} source nodes and {len(targets)} target nodes for agent '{agent_key}'.")
    return movement_nodes

@njit
def simulate_movement_numba(voxel_grid, resistance, risk, energy, speed, autocorrelation, targets_coords, targets_bias, neighbors, source_idx):
    """
    Simulate movement for an agent starting from a source voxel using a flattened grid.

    Parameters:
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        resistance (np.ndarray): 1D array of resistance values aligned with voxel_grid.
        risk (np.ndarray): 1D array of risk values aligned with voxel_grid.
        energy (float): Total energy available.
        speed (int): Number of steps per iteration.
        autocorrelation (float): Tendency to continue in the same direction.
        targets_coords (NumbaList(int)): Target voxel indices.
        targets_bias (NumbaList(float)): Bias values for targets.
        neighbors (np.ndarray): 2D array of neighbor indices with shape (num_voxels, 26).
        source_idx (int): Starting voxel index.

    Returns:
        NumbaList(int): List of voxel indices representing the path.
    """
    path = NumbaList()
    if voxel_grid.shape[0] == 0:
        return path
    path.append(source_idx)

    current = source_idx
    prev_direction = np.zeros(3, dtype=np.float64)

    # Calculate total target bias
    total_target_bias = 0.0
    for bias in targets_bias:
        total_target_bias += bias

    # Cap the total bias to MAX_TOTAL_BIAS - autocorrelation
    if autocorrelation + total_target_bias > MAX_TOTAL_BIAS:
        scaling_factor = MAX_TOTAL_BIAS - autocorrelation
        if scaling_factor <= 0.0:
            scaled_target_bias = NumbaList()
            for _ in range(len(targets_bias)):
                scaled_target_bias.append(0.0)
        else:
            scaled_target_bias = NumbaList()
            for bias in targets_bias:
                scaled_target_bias.append(bias * (scaling_factor / total_target_bias))
    else:
        scaled_target_bias = NumbaList()
        for bias in targets_bias:
            scaled_target_bias.append(bias)

    for _ in range(speed):
        if energy <= 0:
            break

        current_neighbors = neighbors[current]
        # movement_probabilities: array of 26 elements
        movement_probabilities = 0.0
        valid_neighbors = 0

        # Initialize movement_probabilities as an array
        movement_probabilities_array = np.zeros(26, dtype=np.float64)

        # Calculate movement probabilities based on resistance
        for idx in range(26):
            neighbor = current_neighbors[idx]
            if neighbor == -1:
                movement_probabilities_array[idx] = 0.0
            else:
                if resistance[neighbor] > 0:
                    movement_probabilities_array[idx] = 1.0 / resistance[neighbor]
                else:
                    movement_probabilities_array[idx] = 0.0
                valid_neighbors += 1

        if valid_neighbors == 0:
            break

        total = 0.0
        for idx in range(26):
            total += movement_probabilities_array[idx]
        if total == 0.0:
            break
        for idx in range(26):
            movement_probabilities_array[idx] /= total

        # Apply autocorrelation
        if (prev_direction[0] != 0.0 or prev_direction[1] != 0.0 or prev_direction[2] != 0.0):
            directions = np.zeros((26, 3), dtype=np.float64)
            for idx in range(26):
                neighbor = current_neighbors[idx]
                if neighbor == -1:
                    directions[idx, 0] = 0.0
                    directions[idx, 1] = 0.0
                    directions[idx, 2] = 0.0
                else:
                    di = voxel_grid[neighbor, 0] - voxel_grid[current, 0]
                    dj = voxel_grid[neighbor, 1] - voxel_grid[current, 1]
                    dk = voxel_grid[neighbor, 2] - voxel_grid[current, 2]
                    norm = np.sqrt(di**2 + dj**2 + dk**2)
                    if norm != 0.0:
                        directions[idx, 0] = di / norm
                        directions[idx, 1] = dj / norm
                        directions[idx, 2] = dk / norm
                    else:
                        directions[idx, 0] = 0.0
                        directions[idx, 1] = 0.0
                        directions[idx, 2] = 0.0

            # Normalize directions
            prev_norm = np.sqrt(prev_direction[0]**2 + prev_direction[1]**2 + prev_direction[2]**2)
            if prev_norm != 0.0:
                prev_dir_normalized = prev_direction / prev_norm
            else:
                prev_dir_normalized = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            dot_products = np.zeros(26, dtype=np.float64)
            for idx in range(26):
                dot = (directions[idx, 0] * prev_dir_normalized[0] +
                       directions[idx, 1] * prev_dir_normalized[1] +
                       directions[idx, 2] * prev_dir_normalized[2])
                dot_products[idx] = (dot + 1.0) / 2.0  # Normalize to [0,1]

            # Normalize autocorr_bias
            autocorr_bias_sum = 0.0
            for idx in range(26):
                autocorr_bias_sum += dot_products[idx]

            autocorr_bias = np.zeros(26, dtype=np.float64)
            if autocorr_bias_sum > 0.0:
                for idx in range(26):
                    autocorr_bias[idx] = dot_products[idx] / autocorr_bias_sum
            else:
                for idx in range(26):
                    autocorr_bias[idx] = 1.0 / 26.0

            # Combine with movement probabilities
            for idx in range(26):
                movement_probabilities_array[idx] = (1.0 - autocorrelation) * movement_probabilities_array[idx] + autocorrelation * autocorr_bias[idx]

            # Normalize movement_probabilities_array
            total_prob = 0.0
            for idx in range(26):
                total_prob += movement_probabilities_array[idx]
            if total_prob > 0.0:
                for idx in range(26):
                    movement_probabilities_array[idx] /= total_prob

        # Apply destination biases
        if len(targets_coords) > 0:
            bias_probs = np.zeros(26, dtype=np.float64)
            for t in range(len(targets_coords)):
                target = targets_coords[t]
                bias = scaled_target_bias[t]
                # Calculate distance-based bias (Manhattan distance)
                distance = 0.0
                for dim in range(3):
                    distance += abs(voxel_grid[current, dim] - voxel_grid[target, dim])
                # Avoid division by zero by ensuring distance >= 0
                bias_value = bias / (1.0 + distance)
                for idx in range(26):
                    neighbor = current_neighbors[idx]
                    if neighbor == -1:
                        continue
                    bias_probs[idx] += bias_value
            total_bias = 0.0
            for idx in range(26):
                total_bias += bias_probs[idx]
            if total_bias > 0.0:
                for idx in range(26):
                    bias_probs[idx] /= total_bias
                for idx in range(26):
                    movement_probabilities_array[idx] += bias_probs[idx]
                # Normalize again
                total_movement = 0.0
                for idx in range(26):
                    total_movement += movement_probabilities_array[idx]
                if total_movement > 0.0:
                    for idx in range(26):
                        movement_probabilities_array[idx] /= total_movement

        # Choose the next step based on probabilities
        cumulative = 0.0
        rand = random.random()
        next_idx = 0
        while next_idx < 26 and rand > (cumulative + movement_probabilities_array[next_idx]):
            cumulative += movement_probabilities_array[next_idx]
            next_idx += 1
        if next_idx >= 26:
            next_idx = 25
        next_step = current_neighbors[next_idx]

        if next_step == -1:
            break  # Invalid step

        # Append the next step to the path
        path.append(next_step)

        # Update energy
        energy -= resistance[next_step]
        if energy <= 0:
            break

        # Update previous direction
        prev_direction[0] = float(voxel_grid[next_step, 0] - voxel_grid[current, 0])
        prev_direction[1] = float(voxel_grid[next_step, 1] - voxel_grid[current, 1])
        prev_direction[2] = float(voxel_grid[next_step, 2] - voxel_grid[current, 2])
        current = next_step

        # Apply risk of stopping
        if random.random() < risk[next_step]:
            break  # Agent stops moving due to risk

    return path

def worker_simulate(agent_key, agent_params, source, resistance, risk, grid_shape, neighbors, voxel_grid, target_nodes):
    """
    Simulate movement for a single agent starting from a source node using a flattened grid.

    Parameters:
        agent_key (str): Name of the agent.
        agent_params (dict): Parameters for the agent.
        source (int): Starting voxel index.
        resistance (np.ndarray): 1D array of resistance values aligned with voxel_grid.
        risk (np.ndarray): 1D array of risk values aligned with voxel_grid.
        grid_shape (tuple): Shape of the original 3D grid (I, J, K).
        neighbors (np.ndarray): 2D array of neighbor indices with shape (num_voxels, 26).
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        target_nodes (list of dict): List of target nodes with 'index' and 'bias'.

    Returns:
        tuple: (agent_key, path) where path is a list of voxel indices.
    """
    # Extract targets coordinates and biases
    targets_indices = [target['index'] for target in target_nodes]
    targets_bias = [target['bias'] for target in target_nodes]

    # Convert to Numba-compatible lists
    num_targets = len(targets_indices)
    targets_coords = NumbaList()
    for idx in range(num_targets):
        targets_coords.append(targets_indices[idx])
    targets_bias_nb = NumbaList()
    for bias in targets_bias:
        targets_bias_nb.append(bias)

    # Simulate movement
    path = simulate_movement_numba(
        voxel_grid=voxel_grid,
        resistance=resistance,
        risk=risk,
        energy=agent_params['energy'],
        speed=agent_params['speed'],
        autocorrelation=agent_params['autocorrelation'],
        targets_coords=targets_coords,
        targets_bias=targets_bias_nb,
        neighbors=neighbors,
        source_idx=source
    )

    # Convert Numba List to standard Python list
    python_path = list(path)

    # Debug: Print the path length
    if len(python_path) > 0:
        print(f"Agent '{agent_key}' started at voxel index {source}")
        print(f"Path length: {len(python_path)}")

    return (agent_key, python_path)

def aggregate_paths(paths, grid_shape, voxel_grid):
    """
    Aggregate multiple paths into a density surface using NumPy's advanced indexing.

    Parameters:
        paths (list): List of paths, each path is a list of voxel indices.
        grid_shape (tuple): Shape of the original 3D grid (I, J, K).
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.

    Returns:
        np.ndarray: 3D numpy array representing movement density.
    """
    # Flatten all paths into a single list
    all_voxel_indices = np.concatenate(paths)
    
    # Get i, j, k coordinates for all voxel indices
    i = voxel_grid[all_voxel_indices, 0].astype(np.int32)
    j = voxel_grid[all_voxel_indices, 1].astype(np.int32)
    k = voxel_grid[all_voxel_indices, 2].astype(np.int32)
    
    # Initialize density array
    density = np.zeros(grid_shape, dtype=np.int32)
    
    # Use numpy's add.at for efficient accumulation
    np.add.at(density, (i, j, k), 1)
    
    return density

def create_pyvista_polydata(density_surface, voxel_grid, ds):
    """
    Convert the movement density surface into a PyVista PolyData object using voxel indices.

    Parameters:
        density_surface (np.ndarray): 3D numpy array of movement density.
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        ds (xarray.Dataset): xarray Dataset containing voxel grid information.

    Returns:
        pv.PolyData: PyVista PolyData object.
    """
    # Get voxel indices where density >=1
    voxel_indices = np.argwhere(density_surface >= 1)
    if voxel_indices.size == 0:
        raise ValueError("No voxels with density >=1 found.")

    # Initialize lists to store coordinates and frequency
    points = []
    movement_frequency = []

    for idx in voxel_indices:
        i, j, k = idx
        points.append([i, j, k])  # Storing grid indices directly
        movement_frequency.append(density_surface[i, j, k])

    # Convert to numpy arrays
    points = np.array(points)
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

def visualize_density_slice(density_surface, resistance_surface, slice_index, axis='z', agent_name='agent', sources=[], targets=[], save_dir='pathwalker_3d_results'):
    """
    Visualize a 2D slice of the movement density surface with resistance overlay and source/target nodes.

    Parameters:
        density_surface (np.ndarray): 3D numpy array of movement density.
        resistance_surface (np.ndarray): 3D numpy array of resistance values.
        slice_index (int): Index of the slice to visualize.
        axis (str): Axis along which to take the slice ('x', 'y', or 'z').
        agent_name (str): Name of the agent for plot title and filename.
        sources (list): List of source voxel indices as integers.
        targets (list): List of target voxel indices as integers.
        save_dir (str): Directory where the plot will be saved.
    """
    print(f'Source nodes: {sources}')
    print(f'Target nodes: {targets}')

    if axis == 'x':
        density_slice = density_surface[slice_index, :, :]
        resistance_slice = resistance_surface[slice_index, :, :]
        # Project sources and targets onto Y-Z plane
        source_coords = [(int(voxel_grid[source,1]), int(voxel_grid[source,2])) for source in sources if int(voxel_grid[source,0]) == slice_index]
        target_coords = [(int(voxel_grid[target,1]), int(voxel_grid[target,2])) for target in targets if int(voxel_grid[target,0]) == slice_index]
        xlabel, ylabel = 'Y-axis', 'Z-axis'
    elif axis == 'y':
        density_slice = density_surface[:, slice_index, :]
        resistance_slice = resistance_surface[:, slice_index, :]
        # Project sources and targets onto X-Z plane
        source_coords = [(int(voxel_grid[source,0]), int(voxel_grid[source,2])) for source in sources if int(voxel_grid[source,1]) == slice_index]
        target_coords = [(int(voxel_grid[target,0]), int(voxel_grid[target,2])) for target in targets if int(voxel_grid[target,1]) == slice_index]
        xlabel, ylabel = 'X-axis', 'Z-axis'
    else:
        density_slice = density_surface[:, :, slice_index]
        resistance_slice = resistance_surface[:, :, slice_index]
        # Project sources and targets onto X-Y plane
        source_coords = [(int(voxel_grid[source,0]), int(voxel_grid[source,1])) for source in sources if int(voxel_grid[source,2]) == slice_index]
        target_coords = [(int(voxel_grid[target,0]), int(voxel_grid[target,1])) for target in targets if int(voxel_grid[target,2]) == slice_index]
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

    # Plot source nodes
    if source_coords:
        x_sources, y_sources = zip(*source_coords)
        plt.scatter(x_sources, y_sources, marker='*', color='blue', s=150, label='Source Nodes', edgecolors='black')

    # Plot target nodes
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

def load_custom_risk_surface_flat(json_filepath, grid_shape, voxel_grid, agents):
    """
    Load a custom risk surface from a JSON file using a flattened grid.

    Parameters:
        json_filepath (str): Path to the JSON file containing custom risk data.
        grid_shape (tuple): Shape of the original 3D grid (I, J, K).
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        agents (dict): Dictionary of agents.

    Returns:
        dict: Dictionary with agent names as keys and their respective risk arrays as values.
    """
    if not os.path.isfile(json_filepath):
        print(f"Custom risk surface file '{json_filepath}' not found. Using default risk surfaces.")
        return None

    with open(json_filepath, 'r') as file:
        data = json.load(file)

    if 'risk_surface' not in data:
        print("No 'risk_surface' key found in JSON. Using default risk surfaces.")
        return None

    risk_data = data['risk_surface']  # Assuming the JSON has a 'risk_surface' key
    custom_risk_surfaces = {agent: np.full(voxel_grid.shape[0], agents[agent]['risk'], dtype=np.float32) for agent in agents.keys()}

    for voxel in risk_data:
        i, j, k = voxel['voxel_indices']
        risk = voxel['risk']
        # Find the voxel index
        matches = np.where((voxel_grid[:, 0] == i) &
                           (voxel_grid[:, 1] == j) &
                           (voxel_grid[:, 2] == k))[0]
        if matches.size > 0:
            voxel_idx = matches[0]
            for agent in agents.keys():
                custom_risk_surfaces[agent][voxel_idx] = risk
        else:
            print(f"Voxel ({i}, {j}, {k}) not found in voxel grid. Skipping.")

    print("Custom risk surfaces loaded.")
    return custom_risk_surfaces

def save_paths_and_nodes_as_vtk(paths, movement_nodes, voxel_grid, ds, results_dir, sphere_radius=0.5):
    """
    Save movement paths and nodes as VTK files using PyVista.

    Parameters:
        paths (dict): Dictionary of paths for each agent.
        movement_nodes (dict): Dictionary of movement nodes for each agent.
        voxel_grid (np.ndarray): 2D array where each row represents a voxel with its properties.
        ds (xarray.Dataset): xarray Dataset containing voxel grid information.
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
            spatial_path = []
            for voxel_idx in path:
                i, j, k = int(voxel_grid[voxel_idx, 0]), int(voxel_grid[voxel_idx, 1]), int(voxel_grid[voxel_idx, 2])
                spatial_path.append([i, j, k])  # Storing grid indices directly
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
            if n_points == 0:
                continue
            lines.append([n_points] + list(range(offset, offset + n_points)))
            offset += n_points

        if len(lines) == 0:
            print(f"No valid paths to save for agent '{agent}'. Skipping.")
            continue

        # Convert lines to a flat numpy array
        lines = np.hstack(lines)

        # Create PyVista PolyData object for paths
        polypaths = pv.PolyData(all_points, lines)
        polypaths.save(path_file)
        print(f"Saved movement paths for '{agent}' to '{path_file}'.")

        # Convert source voxels to spatial coordinates
        source_voxels = movement_nodes[agent]['source_nodes']
        if source_voxels:
            start_nodes = np.array([voxel_grid[source, :3] for source in source_voxels])
            start_pv = pv.PolyData(start_nodes)
            start_spheres = start_pv.glyph(scale=False, geom=pv.Sphere(radius=sphere_radius))
        else:
            start_spheres = pv.PolyData()

        # Convert target voxels to spatial coordinates
        target_voxels = movement_nodes[agent]['target_nodes']
        if target_voxels:
            end_nodes = np.array([voxel_grid[target['index'], :3] for target in target_voxels])
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

def run_simulations_flat(siteName, voxelSize=5, scenarioName='original', movementNodesOptions='load'):
    """
    Run movement simulations based on the provided configuration using a flattened grid.

    Parameters:
        siteName (str): Name of the site.
        voxelSize (int, optional): Size of the voxel. Defaults to 5.
        scenarioName (str, optional): Name of the scenario. Defaults to 'original'.
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

    # --------------------- INITIALIZE FLAT GRID ---------------------
    print("Initializing flattened grid...")
    voxel_grid, grid_shape = initialize_flat_grid(ds)
    print(f"Flattened grid initialized with shape {voxel_grid.shape}.")

    # --------------------- PRECOMPUTE NEIGHBORS ---------------------
    print("Precomputing neighbors...")
    neighbors = precompute_neighbors(voxel_grid, grid_shape)
    print("Neighbors precomputed.")

    # --------------------- DEFINE RESISTANCE AND RISK SURFACES ---------------------
    print("Defining resistance and risk surfaces for each agent...")
    resistance_surfaces, risk_surfaces = define_resistance_and_risk_surfaces(ds, voxel_grid, grid_shape, neighbors)
    print("Resistance and risk surfaces defined.")

    # --------------------- LOAD MOVEMENT NODES ---------------------
    print("Loading movement nodes...")
    movement_nodes = load_movement_nodes_flat(
        site=siteName,
        voxel_size=voxelSize,
        agents=agents,
        movementNodesOptions=movementNodesOptions,
        scenarioName=scenarioName,
        base_dir=base_dir,
        voxel_grid=voxel_grid,
        grid_shape=grid_shape
    )

    # --------------------- LOAD CUSTOM RISK SURFACE ---------------------
    print("Loading custom risk surfaces if available...")
    custom_risk_surface_file = os.path.join(base_dir, 'custom_risk_surface.json')  # Modify as needed
    custom_risk_surfaces = load_custom_risk_surface_flat(custom_risk_surface_file, grid_shape, voxel_grid, agents)
    
    if custom_risk_surfaces:
        risk_surfaces = custom_risk_surfaces
    else:
        print("Using default risk surfaces.")

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
                              risk_surfaces[agent_key], grid_shape, neighbors, voxel_grid, target_voxels))

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
        density = aggregate_paths(paths, grid_shape, voxel_grid)
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
        targets = [target['index'] for target in movement_nodes_agent.get('target_nodes', [])]
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
            polydata = create_pyvista_polydata(density, voxel_grid, ds)
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
    save_paths_and_nodes_as_vtk(simulated_paths, movement_nodes, voxel_grid, ds, RESULTS_DIR)
    print("Movement paths and nodes saved.")

    # --------------------- VISUALIZE RESISTANCE SURFACE ---------------------
    print("Visualizing resistance surface for the first agent...")
    first_agent = list(agents.keys())[0]
    try:
        # Create a density array where resistance is not NOTTRAVEL
        resistance_density = (resistance_surfaces[first_agent] != NOTTRAVEL).astype(np.int32).reshape(grid_shape)
        polydata_resistance = create_pyvista_polydata(resistance_density, voxel_grid, ds)
        visualize_pyvista_polydata(
            polydata=polydata_resistance,
            agent_name=f"{first_agent}_resistance",
            save_dir=RESULTS_DIR
        )
    except ValueError as e:
        print(f"Resistance surface visualization skipped: {e}")
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
    run_simulations_flat(
        siteName=site,
        voxelSize=voxelSize,
        scenarioName=scenarioName,
        movementNodesOptions=movementNodesOptions
    )

if __name__ == "__main__":
    main()
