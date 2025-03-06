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
import numpy as np
import random

from numba import njit
from numba.typed import List
import numpy as np



# --------------------- CONFIGURATION ---------------------

# Define default bias and maximum total bias
DEFAULT_BIAS = 0.3  # Adjust as needed
MAX_TOTAL_BIAS = 0.99


# Define resistance values
LOW_RESISTANCE = 1
NUETRAL = 10
HIGH_RESISTANCE = 100

# Define risk surface input
# Set to 0 to use the default risk surface
# Or provide the filename of the custom risk surface (e.g., 'custom_risk_surface.nc')
RISK_SURFACE_INPUT = 0  # Change this to the filename if using a custom risk surface

# Define path to movement nodes JSON file
MOVEMENT_NODES_FILE = 'data/revised/trimmed-parade-5-movementnodes.json'

# Define agents with their specific parameters
agents = {
    'bird': {
        'energy': 100000,                # Total energy available
        'speed': 1000,                    # Steps per iteration
        'autocorrelation': 0.3,        # Tendency to continue in the same direction
        'resistance_neighborhood_size': 5,  # Neighborhood size for resistance
        'resistance_focal_function': 'min', # Focal function for resistance ('mean', 'max', 'min')
        'risk': 0.1,                    # Default risk (overridden by risk surface)
        'name': 'bird'
    },
    'lizard': {
        'energy': 5000,                # Total energy available
        'speed': 1000,                    # Steps per iteration
        'autocorrelation': 0.3,        # Tendency to continue in the same direction
        'resistance_neighborhood_size': 2,  # Neighborhood size for resistance
        'resistance_focal_function': 'max',  # Focal function for resistance ('mean', 'max', 'min')
        'risk': 0.2,                    # Default risk (overridden by risk surface)
        'name': 'lizard'
    }
}

# Define number of simulation runs per agent per source node
NUM_RUNS = 500

# Path to save results
RESULTS_DIR = 'pathwalker_3d_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to the dataset
DATASET_PATH = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/xarray_voxels_trimmed-parade_5.nc'


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
    resistance = np.full(ds.dims['voxel'], NUETRAL)

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

    # Initialize 3D resistance grid with high resistance
    resistance_3d = np.full(grid_shape, NUETRAL)

    # Assign resistance values based on voxel indices
    resistance_3d[I, J, K] = resistance

    #CHATGPT: We need a 'NOTTRAVEL' resistance, perhaps -1? Agents cannot move into these voxels
    #To implement, for every i j voxel coordinate,find the lowest k coordinate. Any 'empty' voxel below this is assigned not travel
    #also add a parameter to agents called 'travelOnGround'. If true, any other empty voxel that doesnt have a filled voxel within +2 i k l directions is also assigned not travel

    # Create separate resistance surfaces for each agent (copy of base resistance)
    for agent_key in agents.keys():
        resistance_surfaces[agent_key] = resistance_3d.copy()

    return resistance_surfaces, grid_shape

def load_movement_nodes(json_filepath, default_bias=0.1):
    """
    Load source and target nodes from a JSON file.
    Assign a default bias to target nodes if not specified.
    
    Parameters:
        json_filepath (str): Path to the JSON file containing movement nodes.
        default_bias (float): Default bias to assign if not specified.
    
    Returns:
        list: List of source node voxel indices.
        list: List of target node voxel indices with biases.
    """
    if not os.path.isfile(json_filepath):
        raise FileNotFoundError(f"Movement nodes file '{json_filepath}' not found.")
    
    with open(json_filepath, 'r') as file:
        data = json.load(file)
    
    # Extract voxel indices for sources and targets
    source_voxels = [tuple(node['voxel_indices']) for node in data.get('source_nodes', [])]
    target_voxels = []
    for node in data.get('target_nodes', []):
        voxel = tuple(node['voxel_indices'])
        bias = node.get('bias', default_bias)
        target_voxels.append({'coordinates': voxel, 'bias': bias})
    
    print(f'source nodes from json are {source_voxels}')
    print(f'target_voxels from json are {target_voxels}')
    
    return source_voxels, target_voxels


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
        return np.mean(Sw) if Sw else NUETRAL

    def fmax(ij, surf, scale):
        w = window_3d(scale)
        Aw = [(ij[0] + dx, ij[1] + dy, ij[2] + dz) for dx, dy, dz in w]
        Sw = [surf[x] for x in Aw if (0 <= x[0] < surf.shape[0] and
                                      0 <= x[1] < surf.shape[1] and
                                      0 <= x[2] < surf.shape[2])]
        return np.max(Sw) if Sw else NUETRAL

    def fmin(ij, surf, scale):
        w = window_3d(scale)
        Aw = [(ij[0] + dx, ij[1] + dy, ij[2] + dz) for dx, dy, dz in w]
        Sw = [surf[x] for x in Aw if (0 <= x[0] < surf.shape[0] and
                                      0 <= x[1] < surf.shape[1] and
                                      0 <= x[2] < surf.shape[2])]
        return np.min(Sw) if Sw else NUETRAL

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

from numba import njit
import numpy as np

from numba import njit
from numba.typed import List
import numpy as np

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
        ds (xr.Dataset): xarray Dataset containing the voxel grid information, bounds, and voxel size.

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
    print(f"PyVista PolyData for {agent_name} movement density saved as VTK file at {vtk_filename}.")

def visualize_resistance_surface(resistance_surface, ds, save_dir='pathwalker_3d_results'):
    """
    Visualize the 3D resistance surface using PyVista.

    Parameters:
        resistance_surface (np.ndarray): 3D numpy array of resistance values.
        ds (xr.Dataset): xarray Dataset containing voxel grid information.
        save_dir (str): Directory where the plot and PolyData will be saved.
    """
    # Get indices where resistance <= threshold
    threshold = 200  # Adjust as needed
    indices = np.argwhere(resistance_surface <= threshold)
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
    print(f"PyVista PolyData for resistance surface saved as VTK file at {vtk_filename}.")

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
    
    print(f'source nodes are {sources}')
    print(f'target nodes are {targets}')
    if axis == 'x':
        density_slice = density_surface[slice_index, :, :]
        resistance_slice = resistance_surface[slice_index, :, :]
        # Project sources and targets onto Y-Z plane
        source_coords = [(j, k) for (i, j, k) in sources]
        target_coords = [(j, k) for (i, j, k) in targets]
        xlabel, ylabel = 'Y-axis', 'Z-axis'
    elif axis == 'y':
        density_slice = density_surface[:, slice_index, :]
        resistance_slice = resistance_surface[:, slice_index, :]
        # Project sources and targets onto X-Z plane
        source_coords = [(i, k) for (i, j, k) in sources]
        target_coords = [(i, k) for (i, j, k) in targets]
        xlabel, ylabel = 'X-axis', 'Z-axis'
    else:
        density_slice = density_surface[:, :, slice_index]
        resistance_slice = resistance_surface[:, :, slice_index]
        # Project sources and targets onto X-Y plane
        source_coords = [(i, j) for (i, j, k) in sources]
        target_coords = [(i, j) for (i, j, k) in targets]
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

def save_density_surface(density, agent_name='agent'):
    """
    Save the entire density surface as a NumPy binary file.

    Parameters:
        density (np.ndarray): 3D numpy array of movement density.
        agent_name (str): Name of the agent for filename.
    """
    np.save(os.path.join(RESULTS_DIR, f'{agent_name}_density.npy'), density)
    print(f"{agent_name.capitalize()} movement density surface saved as .npy file.")

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

# --------------------- MAIN FUNCTION ---------------------

import multiprocessing as mp

def save_paths_and_nodes_as_vtk(paths, source_voxels, target_voxels, ds, results_dir, sphere_radius=0.5):
    import os
    import pyvista as pv
    import numpy as np

    os.makedirs(results_dir, exist_ok=True)

    print(f'paths are {paths}')
    print(f'source voxels are {source_voxels}')
    print(f'target voxels are {target_voxels}')

    for animal, animal_paths in paths.items():
        path_file = os.path.join(results_dir, f'{animal}_movement_paths.vtk')
        nodes_file = os.path.join(results_dir, f'{animal}_nodes.vtk')

        # Convert all paths to spatial coordinates
        paths_spatial = []
        for path in animal_paths:
            spatial_path = [ijk_to_centroid(i, j, k, ds) for voxel in path for i, j, k in [voxel]]
            paths_spatial.append(spatial_path)

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
        print(f"Saved movement paths for '{animal}' to '{path_file}'.")

        # Convert source voxels to spatial coordinates
        start_nodes = np.array([ijk_to_centroid(*voxel, ds) for voxel in source_voxels])

        # Convert target voxels to spatial coordinates
        end_nodes = np.array([ijk_to_centroid(*voxel['coordinates'], ds) for voxel in target_voxels])

        # Create sphere markers for start nodes
        start_pv = pv.PolyData(start_nodes)
        start_spheres = start_pv.glyph(scale=False, geom=pv.Sphere(radius=sphere_radius))

        # Create sphere markers for end nodes
        end_pv = pv.PolyData(end_nodes)
        end_spheres = end_pv.glyph(scale=False, geom=pv.Sphere(radius=sphere_radius))

        # Combine start and end spheres into one PolyData object
        nodes = start_spheres.merge(end_spheres)

        # Save the combined nodes
        nodes.save(nodes_file)
        print(f"Saved start and end nodes for '{animal}' to '{nodes_file}'.")



def main():
    # --------------------- LOAD DATA ---------------------
    print("Loading dataset...")
    ds = xr.open_dataset(DATASET_PATH)
    #print all variable names in ds
    print(f"Variable names in dataset: {ds.data_vars}")


    # --------------------- TIDY DATA ---------------------
    #CHATGPT: add this!
    #remove al voxels that has isTerrainUnderBuilding == True 
    #log how many voxels removed

    # --------------------- DEFINE RESISTANCE ---------------------
    print("Defining resistance surfaces for each agent...")
    resistance_surfaces, grid_shape = define_resistance_surface(ds, agents)
    print("Resistance surfaces defined.")

    # --------------------- LOAD MOVEMENT NODES ---------------------
    print("Loading movement nodes...")
    try:
        source_voxels, target_voxels = load_movement_nodes(MOVEMENT_NODES_FILE)
        print("Movement nodes loaded successfully.")
    except FileNotFoundError:
        print(f"Movement nodes file '{MOVEMENT_NODES_FILE}' not found. Using default nodes.")
        # Define default source and target nodes if JSON file is not found
        source_voxels = [(25, 49, 0), (48, 83, 1), (5, 72, 4)]
        target_voxels = [{'coordinates': (47, 16, 0), 'bias': 0.2},
                        {'coordinates': (11, 13, 0), 'bias': 0.1}]
        print("Default movement nodes defined.")

    # --------------------- LOAD RISK SURFACE ---------------------
    print("Loading risk surfaces...")
    # Check if a custom risk surface JSON file exists
    # For simplicity, assume 'custom_risk_surface.json' contains risk data
    custom_risk_surface_file = 'custom_risk_surface.json'  # Modify as needed
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
        save_density_surface(density, agent_name=agent_key)
    print("Aggregation completed.")

    # --------------------- VISUALIZE RESULTS ---------------------
    print("Visualizing movement density and resistance surfaces...")
    for agent_key, density in density_surfaces.items():
        print(f"Visualizing results for agent '{agent_key}'...")
        # Visualize 2D slice (middle slice along z-axis)
        slice_index = grid_shape[2] // 2
        visualize_density_slice(
            density_surface=density,
            resistance_surface=resistance_surfaces[agent_key],
            slice_index=slice_index,
            axis='z',
            agent_name=agent_key,
            sources=source_voxels,
            targets=[target['coordinates'] for target in target_voxels],
            save_dir=RESULTS_DIR
        )

        # Visualize 3D movement density with PyVista
        polydata = create_pyvista_polydata(density, ds)
        visualize_pyvista_polydata(
            polydata=polydata,
            agent_name=agent_key,
            save_dir=RESULTS_DIR
        )
        print(f"Visualization for agent '{agent_key}' completed.")

    print(f'source_voxels are {source_voxels}')
    # Save agent paths and nodes as VTK files
    save_paths_and_nodes_as_vtk(simulated_paths, source_voxels, target_voxels, ds, RESULTS_DIR)

    # Visualize resistance surface (shared among agents)
    print("Visualizing resistance surface...")
    visualize_resistance_surface(
        resistance_surface=resistance_surfaces['bird'],  # Assuming all agents share the same resistance initially
        ds=ds,
        save_dir=RESULTS_DIR
    )
    print("Resistance surface visualization completed.")

    print("All simulations and visualizations are complete. Check the results directory.")


if __name__ == "__main__":
    main()
