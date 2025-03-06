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

def initialize_flat_grid(xarray_dataset, grouped_roof_info):
    """
    Initializes the flattened (1D) voxel grid and related variables from the xarray dataset.

    Parameters:
        xarray_dataset (xarray.Dataset): Loaded xarray dataset.
        grouped_roof_info (pd.DataFrame): Loaded grouped roof information DataFrame.

    Returns:
        dict: Dictionary containing initialized variables for the simulation.
    """
    # Extract voxel indices and counts
    voxel_ids = xarray_dataset['voxel'].values
    I = xarray_dataset['voxel_I'].values
    J = xarray_dataset['voxel_J'].values
    K = xarray_dataset['voxel_K'].values
    num_voxels = len(voxel_ids)

    # Initialize plant assignment arrays as 1D boolean arrays
    greenPlants = np.zeros(num_voxels, dtype=bool)
    brownPlants = np.zeros(num_voxels, dtype=bool)
    greenFacades = np.zeros(num_voxels, dtype=bool)

    # Extract other relevant 1D arrays
    site_building_element = xarray_dataset['site_building_element'].values  # 1D array
    roofID_grid = xarray_dataset['roofID'].values  # 1D array
    height_above_ground = xarray_dataset['site_Contours_HeightAboveGround'].values  # 1D array
    assigned_logs = xarray_dataset['assigned_logs'].values  # 1D array

    # Create a lookup for remaining roof load from grouped_roof_info
    roof_load_info = grouped_roof_info.set_index('roofID')[['Roof load', 'Remaining roof load']].to_dict('index')

    # Check and Update Negative Remaining Roof Loads
    negative_load_roofs = [roof_id for roof_id, info in roof_load_info.items() if info['Remaining roof load'] < 0]
    if negative_load_roofs:
        print(f"Found {len(negative_load_roofs)} roofs with negative remaining load. Setting them to 1000.")
        for roof_id in negative_load_roofs:
            roof_load_info[roof_id]['Remaining roof load'] = 1000

    # Create a mapping from voxel index to (I, J, K) coordinates
    voxel_coords = np.vstack((I, J, K)).T  # Shape: (num_voxels, 3)

    # Create a coordinate to index mapping for quick neighbor lookups
    coord_to_idx = {(i, j, k): idx for idx, (i, j, k) in enumerate(voxel_coords)}

    return {
        'num_voxels': num_voxels,
        'greenPlants': greenPlants,
        'brownPlants': brownPlants,
        'greenFacades': greenFacades,
        'site_building_element': site_building_element,  # 1D array
        'roofID_grid': roofID_grid,  # 1D array
        'height_above_ground': height_above_ground,  # 1D array
        'assigned_logs': assigned_logs,  # 1D array
        'roof_load_info': roof_load_info,
        'voxel_coords': voxel_coords,  # 2D array: (num_voxels, 3)
        'coord_to_idx': coord_to_idx
    }


def precompute_neighbors(flat_grid_data):
    """
    Precomputes the neighbor indices for each voxel in the flattened grid.

    Parameters:
        flat_grid_data (dict): Dictionary containing flattened grid data.

    Returns:
        list: List where each element is a list of neighbor indices for the corresponding voxel.
    """
    num_voxels = flat_grid_data['num_voxels']
    voxel_coords = flat_grid_data['voxel_coords']
    coord_to_idx = flat_grid_data['coord_to_idx']

    neighbors = [[] for _ in range(num_voxels)]

    for idx, (i, j, k) in enumerate(voxel_coords):
        # Define the 26 possible neighbor directions in 3D
        directions = list(product([-1, 0, 1], repeat=3))
        directions.remove((0, 0, 0))  # Exclude the current voxel

        for di, dj, dk in directions:
            ni, nj, nk = i + di, j + dj, k + dk
            neighbor_coord = (ni, nj, nk)
            neighbor_idx = coord_to_idx.get(neighbor_coord, -1)
            if neighbor_idx != -1:
                neighbors[idx].append(neighbor_idx)

    return neighbors


def define_resistance_surface_flat(ds, flat_grid_data, neighbors):
    """
    Define resistance surfaces for each agent based on the specified rules using a flattened grid.
    Returns a dictionary of 1D numpy arrays: resistance_bird, resistance_lizard, etc.

    Parameters:
        ds (xr.Dataset): The loaded xarray Dataset containing voxel data.
        flat_grid_data (dict): Dictionary containing flattened grid data.
        neighbors (list): Precomputed neighbor indices for each voxel.

    Returns:
        dict: Dictionary with agent names as keys and their respective resistance surfaces as values.
    """
    num_voxels = flat_grid_data['num_voxels']
    resistance_surfaces = {agent: np.full(num_voxels, NEUTRAL, dtype=np.int32) for agent in agents.keys()}

    # Identify resource variables
    resource_vars = [var for var in ds.data_vars if var.startswith('resource_')]

    if resource_vars:
        # Create a mask where any of the resource variables are not null
        low_res_mask = np.any(np.stack([ds[var].values for var in resource_vars], axis=0), axis=0).flatten()
        for agent in agents.values():
            resistance_surfaces[agent['name']][low_res_mask] = LOW_RESISTANCE

    # Apply high resistance rules
    if 'site_building_ID' in ds and 'site_LAS_HeightAboveGround' in ds:
        high_res_buildings = (ds['site_building_ID'].notnull()) & (ds['site_LAS_HeightAboveGround'] > 10)
        high_res_buildings_flat = high_res_buildings.values.flatten()
        for agent in agents.values():
            resistance_surfaces[agent['name']][high_res_buildings_flat] = HIGH_RESISTANCE

    if 'road_roadInfo_width' in ds:
        road_widths = ds['road_roadInfo_width'].values.flatten()
        high_res_roads = np.isin(road_widths, ['200 / 100 / 100', '300 / 100 / 100'])
        for agent in agents.values():
            resistance_surfaces[agent['name']][high_res_roads] = HIGH_RESISTANCE

    # Assign NOTTRAVEL to voxels below the lowest filled voxel in each (i, j) column
    print("Assigning 'NOTTRAVEL' resistance to appropriate voxels...")
    voxel_coords = flat_grid_data['voxel_coords']
    for idx in range(flat_grid_data['num_voxels']):
        i, j, k = voxel_coords[idx]
        # Find all voxels with the same (i, j)
        same_column = np.where((voxel_coords[:, 0] == i) & (voxel_coords[:, 1] == j))[0]
        if same_column.size > 0:
            lowest_k = voxel_coords[same_column, 2].min()
            below_lowest = same_column[voxel_coords[same_column, 2] < lowest_k]
            for agent in agents.values():
                resistance_surfaces[agent['name']][below_lowest] = NOTTRAVEL

    # Handle 'travelOnGround' parameter
    for agent_key, agent in agents.items():
        if agent.get('travelOnGround', False):
            print(f"Processing 'travelOnGround' for agent '{agent_key}'...")
            # Assign NOTTRAVEL to voxels that are not on the ground surface
            # Assuming ground is k=0
            non_ground = voxel_coords[:, 2] > 0
            resistance_surfaces[agent_key][non_ground] = NOTTRAVEL

    print("'NOTTRAVEL' resistance assignment completed.")

    return resistance_surfaces


def load_movement_nodes_flat(site, voxel_size, agents, movementNodesOptions, scenarioName, base_dir='data/revised', flat_grid_data=None):
    """
    Load or generate movement nodes based on the specified options using a flattened grid.

    Parameters:
        site (str): Name of the site.
        voxel_size (int): Size of the voxel.
        agents (dict): Dictionary of agents.
        movementNodesOptions (str): Option for movement nodes ('load', 'choose', 'random').
        scenarioName (str): Name of the scenario.
        base_dir (str): Base directory for movement nodes files.
        flat_grid_data (dict): Flattened grid data.

    Returns:
        dict: Dictionary with agent names as keys and their respective movement nodes as values.
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
                source_voxels = [node['voxel_index'] for node in data.get('source_nodes', [])]
                target_voxels = []
                for node in data.get('target_nodes', []):
                    voxel = node['voxel_index']
                    bias = node.get('bias', DEFAULT_BIAS)
                    target_voxels.append({'voxel_index': voxel, 'bias': bias})
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


def assign_random_movement_nodes_flat(movement_nodes, flat_grid_data, num_sources=3, num_targets=2):
    """
    Assign random source and target nodes for agents using a flattened grid.

    Parameters:
        movement_nodes (dict): Dictionary with agent names as keys and their movement nodes as values.
        flat_grid_data (dict): Flattened grid data.
        num_sources (int): Number of source nodes to assign.
        num_targets (int): Number of target nodes to assign.

    Returns:
        dict: Updated movement_nodes with randomly assigned source and target nodes.
    """
    num_voxels = flat_grid_data['num_voxels']
    for agent_key, nodes in movement_nodes.items():
        if nodes['state'] == 'randomly_initialized':
            sources = random.sample(range(num_voxels), num_sources)
            targets = []
            for _ in range(num_targets):
                voxel_index = random.randint(0, num_voxels - 1)
                bias = random.uniform(0.05, 0.2)  # Assign random bias between 0.05 and 0.2
                targets.append({'voxel_index': voxel_index, 'bias': bias})
            movement_nodes[agent_key]['source_nodes'] = sources
            movement_nodes[agent_key]['target_nodes'] = targets
            movement_nodes[agent_key]['state'] = 'randomly_assigned'
            print(f"Randomly assigned {len(sources)} source nodes and {len(targets)} target nodes for agent '{agent_key}'.")
    return movement_nodes


@njit
def simulate_movement_numba_flat(start_idx, neighbors, resistance, risk, energy, speed, autocorrelation, targets_indices, targets_bias):
    """
    Simulate movement for an agent starting from a source voxel using a flattened grid.

    Parameters:
        start_idx (int): Starting voxel index.
        neighbors (list): Precomputed neighbors for each voxel.
        resistance (1D np.ndarray): Resistance values for each voxel.
        risk (1D np.ndarray): Risk values for each voxel.
        energy (float): Total energy available.
        speed (int): Number of steps per iteration.
        autocorrelation (float): Tendency to continue in the same direction.
        targets_indices (1D List(int)): Target voxel indices.
        targets_bias (1D List(float)): Bias values for targets.

    Returns:
        List(int): Path as a list of voxel indices.
    """
    path = List()
    path.append(start_idx)

    current = start_idx
    prev_direction = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Calculate total target bias
    total_target_bias = 0.0
    for bias in targets_bias:
        total_target_bias += bias

    # Cap the total bias to 0.99 - autocorrelation
    if autocorrelation + total_target_bias > MAX_TOTAL_BIAS:
        scaling_factor = MAX_TOTAL_BIAS - autocorrelation
        if scaling_factor <= 0.0:
            # No room for target bias
            scaled_target_bias = np.zeros(len(targets_bias))
        else:
            scaling_factor = scaling_factor / total_target_bias
            scaled_target_bias = np.array(targets_bias) * scaling_factor
    else:
        scaled_target_bias = np.array(targets_bias)

    for _ in range(speed):
        if energy <= 0:
            break

        current_neighbors = neighbors[current]
        num_neighbors = len(current_neighbors)
        if num_neighbors == 0:
            break

        movement_probabilities = np.zeros(num_neighbors, dtype=np.float64)
        for idx in range(num_neighbors):
            neighbor = current_neighbors[idx]
            focal_res = resistance[neighbor]
            movement_probabilities[idx] = 1.0 / focal_res if focal_res > 0 else 0.0

        total = movement_probabilities.sum()
        if total == 0:
            break
        movement_probabilities /= total

        # Apply autocorrelation
        if prev_direction[0] != 0.0 or prev_direction[1] != 0.0 or prev_direction[2] != 0.0:
            directions = np.zeros((num_neighbors, 3), dtype=np.float64)
            for idx in range(num_neighbors):
                neighbor = current_neighbors[idx]
                directions[idx, 0] = neighbor % 1000 - current % 1000  # Example calculation
                directions[idx, 1] = (neighbor // 1000) % 1000 - (current // 1000) % 1000
                directions[idx, 2] = neighbor // 1000000 - current // 1000000

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
        if len(targets_indices) > 0:
            bias_probs = np.zeros(num_neighbors, dtype=np.float64)
            for t in range(len(targets_indices)):
                target = targets_indices[t]
                bias = scaled_target_bias[t]
                # Simple distance-based bias
                distance = np.abs(target - current)
                bias_probs += bias / (1.0 + distance)
            total_bias = bias_probs.sum()
            if total_bias > 0.0:
                bias_probs /= total_bias
                movement_probabilities += bias_probs
                movement_probabilities /= movement_probabilities.sum()

        # Choose the next step based on probabilities
        cumulative = np.cumsum(movement_probabilities)
        rand = random.random()
        next_idx = 0
        while next_idx < len(cumulative) and rand > cumulative[next_idx]:
            next_idx += 1
        if next_idx >= len(cumulative):
            next_idx = len(cumulative) - 1
        next_step = current_neighbors[next_idx]

        # Append the next step to the path
        path.append(next_step)

        # Update energy
        energy -= resistance[next_step]
        if energy <= 0:
            break

        # Update previous direction
        prev_direction = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # Reset for simplicity
        current = next_step

        # Apply risk of stopping
        if random.random() < risk[next_step]:
            break  # Agent stops moving due to risk

    return path


def worker_simulate_flat(agent_key, agent_params, source, resistance, risk, neighbors, target_nodes):
    """
    Simulate movement for a single agent starting from a source node using a flattened grid.

    Parameters:
        agent_key (str): Name of the agent.
        agent_params (dict): Parameters for the agent.
        source (int): Starting voxel index.
        resistance (1D np.ndarray): Resistance values for the agent.
        risk (1D np.ndarray): Risk values for the agent.
        neighbors (list): Precomputed neighbors for each voxel.
        target_nodes (list of dict): List of target nodes with 'voxel_index' and 'bias'.

    Returns:
        tuple: (agent_key, path) where path is a list of voxel indices.
    """
    targets_coords = [target['voxel_index'] for target in target_nodes]
    targets_bias = [target['bias'] for target in target_nodes]

    path = simulate_movement_numba_flat(
        start_idx=source,
        neighbors=neighbors,
        resistance=resistance,
        risk=risk,
        energy=agent_params['energy'],
        speed=agent_params['speed'],
        autocorrelation=agent_params['autocorrelation'],
        targets_indices=targets_coords,
        targets_bias=targets_bias
    )

    # Debug: Print the path length
    if len(path) > 0:
        print(f"Agent '{agent_key}' started at voxel {path[0]}")
        print(f"Path length: {len(path)}")

    return (agent_key, path)


def aggregate_paths_flat(paths, num_voxels):
    """
    Aggregate multiple paths into a density surface using NumPy's advanced indexing.

    Parameters:
        paths (list): List of paths, each path is a list of voxel indices.
        num_voxels (int): Total number of voxels.

    Returns:
        np.ndarray: 1D numpy array representing movement density.
    """
    density = np.zeros(num_voxels, dtype=np.int32)
    for path in paths:
        for voxel in path:
            if 0 <= voxel < num_voxels:
                density[voxel] += 1
    return density


def ijk_to_centroid_flat(i, j, k, ds):
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


def create_pyvista_polydata_flat(density_surface, flat_grid_data, ds):
    """
    Convert the movement density surface into a PyVista PolyData object using centroid coordinates.

    Parameters:
        density_surface (np.ndarray): 1D numpy array of movement density.
        flat_grid_data (dict): Flattened grid data.
        ds (xr.Dataset): xarray Dataset containing voxel grid information.

    Returns:
        pv.PolyData: PyVista PolyData object.
    """
    voxel_indices = np.where(density_surface >= 1)[0]
    if voxel_indices.size == 0:
        raise ValueError("No voxels with density >=1 found.")

    voxel_coords = flat_grid_data['voxel_coords'][voxel_indices]
    x_coords, y_coords, z_coords = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
    movement_frequency = density_surface[voxel_indices]

    # Convert voxel indices to centroid coordinates
    centroids = np.array([ijk_to_centroid_flat(i, j, k, ds) for i, j, k in voxel_coords])

    # Create PyVista points
    polydata = pv.PolyData(centroids)
    polydata.point_data["movement_frequency"] = movement_frequency

    return polydata


def visualize_pyvista_polydata_flat(polydata, agent_name='agent', save_dir='pathwalker_3d_results'):
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


def visualize_density_slice_flat(density_surface, resistance_surface, slice_index, axis='z', agent_name='agent', sources=[], targets=[], save_dir='pathwalker_3d_results', flat_grid_data=None):
    """
    Visualize a 2D slice of the movement density surface with resistance overlay and source/target nodes.

    Parameters:
        density_surface (np.ndarray): 1D numpy array of movement density.
        resistance_surface (np.ndarray): 1D numpy array of resistance values.
        slice_index (int): Index of the slice to visualize.
        axis (str): Axis along which to take the slice ('x', 'y', or 'z').
        agent_name (str): Name of the agent for plot title and filename.
        sources (list): List of source voxel indices.
        targets (list): List of target voxel indices.
        save_dir (str): Directory where the plot will be saved.
        flat_grid_data (dict): Flattened grid data.
    """
    print(f'Source nodes: {sources}')
    print(f'Target nodes: {targets}')

    voxel_coords = flat_grid_data['voxel_coords']
    if axis == 'x':
        mask = voxel_coords[:, 0] == slice_index
        density_slice = density_surface[mask].reshape(-1, 1)
        resistance_slice = resistance_surface[mask].reshape(-1, 1)
        source_coords = [voxel_coords[idx][1:3] for idx in sources if voxel_coords[idx][0] == slice_index]
        target_coords = [voxel_coords[idx][1:3] for idx in targets if voxel_coords[idx][0] == slice_index]
        xlabel, ylabel = 'Y-axis', 'Z-axis'
    elif axis == 'y':
        mask = voxel_coords[:, 1] == slice_index
        density_slice = density_surface[mask].reshape(-1, 1)
        resistance_slice = resistance_surface[mask].reshape(-1, 1)
        source_coords = [voxel_coords[idx][0:1] + voxel_coords[idx][2:3] for idx in sources if voxel_coords[idx][1] == slice_index]
        target_coords = [voxel_coords[idx][0:1] + voxel_coords[idx][2:3] for idx in targets if voxel_coords[idx][1] == slice_index]
        xlabel, ylabel = 'X-axis', 'Z-axis'
    else:
        mask = voxel_coords[:, 2] == slice_index
        density_slice = density_surface[mask].reshape(-1, 1)
        resistance_slice = resistance_surface[mask].reshape(-1, 1)
        source_coords = [voxel_coords[idx][0:2] for idx in sources if voxel_coords[idx][2] == slice_index]
        target_coords = [voxel_coords[idx][0:2] for idx in targets if voxel_coords[idx][2] == slice_index]
        xlabel, ylabel = 'X-axis', 'Y-axis'

    if density_slice.size == 0 or resistance_slice.size == 0:
        print(f"No data found for slice {slice_index} along axis '{axis}'.")
        return

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


def save_density_surface_flat(density, agent_name='agent', results_dir='pathwalker_3d_results'):
    """
    Save the entire density surface as a NumPy binary file.

    Parameters:
        density (np.ndarray): 1D numpy array of movement density.
        agent_name (str): Name of the agent for filename.
        results_dir (str): Directory to save the density file.
    """
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'{agent_name}_density.npy'), density)
    print(f"'{agent_name.capitalize()}' movement density surface saved as .npy file at '{os.path.join(results_dir, f'{agent_name}_density.npy')}'.")


def load_custom_risk_surface_flat(json_filepath, num_voxels, agents):
    """
    Load a custom risk surface from a JSON file using a flattened grid.

    Parameters:
        json_filepath (str): Path to the JSON file containing custom risk data.
        num_voxels (int): Total number of voxels.
        agents (dict): Dictionary of agents.

    Returns:
        dict: Dictionary with agent names as keys and their respective risk surfaces as values.
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
    custom_risk_surfaces = {}

    for agent_key in agents.keys():
        risk_surface = np.zeros(num_voxels, dtype=float)
        for voxel in risk_data:
            idx = voxel['voxel_index']
            risk = voxel['risk']
            if 0 <= idx < num_voxels:
                risk_surface[idx] = risk
        custom_risk_surfaces[agent_key] = risk_surface
        print(f"Custom risk surface loaded for agent '{agent_key}'.")

    return custom_risk_surfaces


def save_paths_and_nodes_as_vtk_flat(paths, movement_nodes, flat_grid_data, ds, results_dir, sphere_radius=0.5):
    """
    Save movement paths and nodes as VTK files using PyVista.

    Parameters:
        paths (dict): Dictionary of paths for each agent.
        movement_nodes (dict): Dictionary of movement nodes for each agent.
        flat_grid_data (dict): Flattened grid data.
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
            spatial_path = [ijk_to_centroid_flat(*voxel_coords, ds) for voxel_coords in [flat_grid_data['voxel_coords'][voxel] for voxel in path]]
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
            start_nodes = np.array([ijk_to_centroid_flat(*flat_grid_data['voxel_coords'][voxel], ds) for voxel in source_voxels])
            start_pv = pv.PolyData(start_nodes)
            start_spheres = start_pv.glyph(scale=False, geom=pv.Sphere(radius=sphere_radius))
        else:
            start_spheres = pv.PolyData()

        # Convert target voxels to spatial coordinates
        target_voxels = movement_nodes[agent]['target_nodes']
        if target_voxels:
            end_nodes = np.array([ijk_to_centroid_flat(*flat_grid_data['voxel_coords'][voxel['voxel_index']], ds) for voxel in target_voxels])
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


def runSimulations_flat(siteName, voxelSize=5, scenarioName='original', movementNodesOptions='load'):
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
    print(f"Variable names in dataset: {ds.data_vars}")

    # Load grouped_roof_info if necessary
    # Assuming grouped_roof_info is available; replace with actual loading mechanism
    # grouped_roof_info = pd.read_csv('path_to_grouped_roof_info.csv')  # Example
    # For this example, we'll create an empty DataFrame
    import pandas as pd
    grouped_roof_info = pd.DataFrame(columns=['roofID', 'Roof load', 'Remaining roof load'])

    # --------------------- INITIALIZE FLAT GRID ---------------------
    print("Initializing flattened grid...")
    flat_grid_data = initialize_flat_grid(ds, grouped_roof_info)
    print("Flattened grid initialized.")

    # --------------------- PRECOMPUTE NEIGHBORS ---------------------
    print("Precomputing neighbors...")
    neighbors = precompute_neighbors(flat_grid_data)
    print("Neighbors precomputed.")

    # --------------------- DEFINE RESISTANCE ---------------------
    print("Defining resistance surfaces for each agent...")
    resistance_surfaces = define_resistance_surface_flat(ds, flat_grid_data, neighbors)
    print("Resistance surfaces defined.")

    # --------------------- LOAD MOVEMENT NODES ---------------------
    print("Loading movement nodes...")
    movement_nodes = load_movement_nodes_flat(
        site=siteName,
        voxel_size=voxelSize,
        agents=agents,
        movementNodesOptions=movementNodesOptions,
        scenarioName=scenarioName,
        base_dir=base_dir,
        flat_grid_data=flat_grid_data
    )
    
    # If movementNodesOptions is 'random', assign random movement nodes now that grid_shape is known
    if movementNodesOptions == 'random':
        movement_nodes = assign_random_movement_nodes_flat(movement_nodes, flat_grid_data)

    # --------------------- LOAD RISK SURFACE ---------------------
    print("Loading risk surfaces...")
    custom_risk_surface_file = os.path.join(base_dir, 'custom_risk_surface.json')  # Modify as needed
    custom_risk_surfaces = load_custom_risk_surface_flat(custom_risk_surface_file, flat_grid_data['num_voxels'], agents)
    
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
                              risk_surfaces[agent_key], neighbors, target_voxels))

    # Use multiprocessing Pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(worker_simulate_flat, tasks)

    # Aggregate results
    for agent_key, path in results:
        simulated_paths[agent_key].append(path)

    print("Movement simulations completed.")

    # --------------------- AGGREGATE PATHS ---------------------
    print("Aggregating movement paths into density surfaces...")
    density_surfaces = {}
    for agent_key, paths in simulated_paths.items():
        density = aggregate_paths_flat(paths, flat_grid_data['num_voxels'])
        density_surfaces[agent_key] = density
        save_density_surface_flat(density, agent_name=agent_key, results_dir=RESULTS_DIR)
    print("Aggregation completed.")

    # --------------------- VISUALIZE RESULTS ---------------------
    print("Visualizing movement density and resistance surfaces...")
    for agent_key, density in density_surfaces.items():
        print(f"Visualizing results for agent '{agent_key}'...")
        # Visualize 2D slice (middle slice along z-axis)
        slice_index = int(np.mean(flat_grid_data['voxel_coords'][:, 2]))
        agent_movement = movement_nodes.get(agent_key, {})
        sources = agent_movement.get('source_nodes', [])
        targets = [target['voxel_index'] for target in agent_movement.get('target_nodes', [])]
        visualize_density_slice_flat(
            density_surface=density,
            resistance_surface=resistance_surfaces[agent_key],
            slice_index=slice_index,
            axis='z',
            agent_name=agent_key,
            sources=sources,
            targets=targets,
            save_dir=RESULTS_DIR,
            flat_grid_data=flat_grid_data
        )

        # Visualize 3D movement density with PyVista
        try:
            polydata = create_pyvista_polydata_flat(density, flat_grid_data, ds)
            visualize_pyvista_polydata_flat(
                polydata=polydata,
                agent_name=agent_key,
                save_dir=RESULTS_DIR
            )
        except ValueError as e:
            print(f"Visualization skipped for agent '{agent_key}': {e}")
        print(f"Visualization for agent '{agent_key}' completed.")

    # Save agent paths and nodes as VTK files
    print("Saving movement paths and nodes as VTK files...")
    save_paths_and_nodes_as_vtk_flat(simulated_paths, movement_nodes, flat_grid_data, ds, RESULTS_DIR)

    # Visualize resistance surface (shared among agents)
    print("Visualizing resistance surface...")
    # Assuming all agents share the same resistance initially, otherwise choose appropriately
    first_agent = list(agents.keys())[0]
    # Create a 3D resistance surface if needed for visualization
    # For simplicity, we skip detailed resistance visualization in the flat grid
    print("Resistance surface visualization skipped for flattened grid.")
    print("All simulations and visualizations are complete. Check the results directory.")


# --------------------- MAIN EXECUTION ---------------------

def main():
    # Define simulation parameters
    site = 'city'  # Replace with your site name
    scenarioName = 'original'  # Replace with your scenario name
    movementNodesOptions = 'load'  # Options: 'load', 'choose', 'random'
    voxelSize = 5  # Replace with your voxel size

    # Run the simulations
    runSimulations_flat(
        siteName=site,
        voxelSize=voxelSize,
        scenarioName=scenarioName,
        movementNodesOptions=movementNodesOptions
    )

if __name__ == "__main__":
    main()
