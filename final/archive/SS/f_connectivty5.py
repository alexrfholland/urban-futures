import xarray as xr
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import json
from itertools import product
import pyvista as pv
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

# --------------------- CONFIGURATION ---------------------

# Define resistance values
LOW_RESISTANCE = 1
NUETRAL = 1
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
        'energy': 5000,                # Total energy available
        'speed': 20,                    # Steps per iteration
        'autocorrelation': 0.4,        # Tendency to continue in the same direction
        'resistance_neighborhood_size': 3,  # Neighborhood size for resistance
        'resistance_focal_function': 'mean', # Focal function for resistance ('mean', 'max', 'min')
        'risk': 0.1,                    # Default risk (overridden by risk surface)
        'targets': [                    # Destination biases
            {'coordinates': (47, 16, 0), 'bias': 0.2},
            {'coordinates': (11, 13, 0), 'bias': 0.1}
        ],
        'name': 'bird'
    },
    'lizard': {
        'energy': 5000,                # Total energy available
        'speed': 1,                    # Steps per iteration
        'autocorrelation': 0.3,        # Tendency to continue in the same direction
        'resistance_neighborhood_size': 2,  # Neighborhood size for resistance
        'resistance_focal_function': 'max',  # Focal function for resistance ('mean', 'max', 'min')
        'risk': 0.2,                    # Default risk (overridden by risk surface)
        'targets': [                    # Destination biases
            {'coordinates': (47, 16, 0), 'bias': 0.15},
            {'coordinates': (11, 13, 0), 'bias': 0.05}
        ],
        'name': 'lizard'
    }
}

# Validate agent parameters
for agent_key, agent_params in agents.items():
    total_bias = sum(target['bias'] for target in agent_params['targets'])
    if agent_params['autocorrelation'] + total_bias > 1:
        raise ValueError(f"For agent '{agent_key}', the sum of autocorrelation and destination biases exceeds 1.")

# Define number of simulation runs per agent per source node
NUM_RUNS = 100

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

    # Create separate resistance surfaces for each agent (copy of base resistance)
    for agent_key in agents.keys():
        resistance_surfaces[agent_key] = resistance_3d.copy()

    return resistance_surfaces, grid_shape

def load_movement_nodes(json_filepath):
    """
    Load source and target nodes from a JSON file.
    If the file does not exist, raise an error.
    
    Parameters:
        json_filepath (str): Path to the JSON file containing movement nodes.
    
    Returns:
        list: List of source node voxel indices.
        list: List of target node voxel indices.
    """
    if not os.path.isfile(json_filepath):
        raise FileNotFoundError(f"Movement nodes file '{json_filepath}' not found.")

    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # Extract voxel indices for sources and targets
    source_voxels = [tuple(node['voxel_indices']) for node in data.get('source_nodes', [])]
    target_voxels = [tuple(node['voxel_indices']) for node in data.get('target_nodes', [])]

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

def simulate_movement(start, resistance, risk, agent, grid_shape, targets):
    """
    Simulate movement for an agent in a 3D grid with speed, autocorrelation, and destination biases.

    Parameters:
        start (tuple): Starting voxel coordinates (i, j, k).
        resistance (np.ndarray): 3D array of resistance values for the agent.
        risk (np.ndarray): 3D array of risk values for the agent.
        agent (dict): Agent's parameters.
        grid_shape (tuple): Shape of the 3D grid.
        targets (list): List of target node coordinates with biases.

    Returns:
        list: List of voxel coordinates representing the path.
    """
    path = [start]
    current = start
    energy = agent['energy']
    speed = agent['speed']
    autocorrelation = agent['autocorrelation']

    # Initialize previous direction as None
    prev_direction = None

    for _ in range(speed):
        if energy <= 0:
            break

        neighbors = get_neighbors(current, grid_shape)
        if not neighbors:
            break  # No available moves

        movement_probabilities = []
        for neighbor in neighbors:
            # Compute focal resistance for the neighbor voxel
            focal_res = focal_function_3d(neighbor, resistance, agent['resistance_neighborhood_size'], agent)
            # Avoid division by zero
            movement_probabilities.append(1 / focal_res if focal_res > 0 else 0)

        movement_probabilities = np.array(movement_probabilities)
        if movement_probabilities.sum() == 0:
            break  # No viable moves

        # Normalize probabilities
        movement_probabilities /= movement_probabilities.sum()

        # Apply autocorrelation if applicable
        if prev_direction is not None and autocorrelation > 0:
            directions = np.array([np.array(n) - np.array(current) for n in neighbors])
            norms = np.linalg.norm(directions, axis=1)
            # Avoid division by zero
            directions_normalized = np.where(norms[:, np.newaxis] == 0, 0, directions / norms[:, np.newaxis])
            prev_dir_normalized = prev_direction / np.linalg.norm(prev_direction) if np.linalg.norm(prev_direction) != 0 else np.zeros(3)
            dot_products = directions_normalized.dot(prev_dir_normalized)
            autocorr_bias = (dot_products + 1) / 2  # Normalize to [0,1]
            # Normalize autocorr_bias
            if autocorr_bias.sum() > 0:
                autocorr_bias /= autocorr_bias.sum()
            else:
                autocorr_bias = np.ones_like(autocorr_bias) / len(autocorr_bias)
            # Combine with movement probabilities
            movement_probabilities = (1 - autocorrelation) * movement_probabilities + autocorrelation * autocorr_bias
            movement_probabilities /= movement_probabilities.sum()

        # Apply destination biases
        if targets:
            bias_probs = np.zeros(len(neighbors))
            for target in targets:
                target_coord = np.array(target['coordinates'])
                direction_to_target = target_coord - np.array(current)
                norm = np.linalg.norm(direction_to_target)
                if norm == 0:
                    continue
                direction_to_target = direction_to_target / norm
                neighbor_dirs = np.array([np.array(n) - np.array(current) for n in neighbors])
                neighbor_norms = np.linalg.norm(neighbor_dirs, axis=1)
                neighbor_dirs_normalized = np.where(neighbor_norms[:, np.newaxis] == 0, 0, neighbor_dirs / neighbor_norms[:, np.newaxis])
                dot_products = neighbor_dirs_normalized.dot(direction_to_target)
                bias = (dot_products + 1) / 2  # Normalize to [0,1]
                bias_probs += target['bias'] * bias
            # Normalize bias_probs
            if bias_probs.sum() > 0:
                bias_probs /= bias_probs.sum()
                # Combine with movement probabilities
                total_bias = sum(target['bias'] for target in targets)
                movement_probabilities = (1 - total_bias) * movement_probabilities + bias_probs
                movement_probabilities /= movement_probabilities.sum()

        # Choose the next step based on probabilities
        next_step = random.choices(neighbors, weights=movement_probabilities, k=1)[0]
        path.append(next_step)

        # Update energy
        energy -= resistance[next_step]
        if energy <= 0:
            break

        # Update previous direction
        prev_direction = np.array(next_step) - np.array(current)
        current = next_step

        # Apply risk of stopping based on the current voxel's risk value
        current_risk = risk[next_step]
        if random.random() < current_risk:
            break  # Agent stops moving due to risk

    return path

def aggregate_paths(paths, grid_shape):
    """
    Aggregate multiple paths into a density surface.

    Parameters:
        paths (list): List of paths, each path is a list of (i, j, k) tuples.
        grid_shape (tuple): Shape of the 3D grid.

    Returns:
        np.ndarray: 3D numpy array representing movement density.
    """
    density = np.zeros(grid_shape, dtype=int)
    for path in paths:
        for voxel in path:
            density[voxel] += 1
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

import numpy as np
import pyvista as pv

def save_paths_and_nodes_as_vtk(paths, source_voxels, target_voxels, ds, results_dir, sphere_radius=1):
    """
    Save agent paths and nodes as VTK files in the specified results directory, 
    converting voxel indices to spatial coordinates.

    Args:
        paths (list of list of tuples): Paths where each path is a list of (i, j, k) voxel indices.
        source_voxels (list of tuples): List of starting (i, j, k) voxel indices for sources.
        target_voxels (list of tuples): List of ending (i, j, k) voxel indices for targets.
        ds (xr.Dataset): xarray Dataset containing the voxel grid information, bounds, and voxel size.
        results_dir (str): Directory to save the VTK files.
        sphere_radius (float, optional): Radius of the sphere markers. Default is 0.5.
    """
    print(f'paths are: {paths}')
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # File paths for saving
    path_file = os.path.join(results_dir, 'movement_paths.vtk')
    nodes_file = os.path.join(results_dir, 'nodes.vtk')

    # Convert all paths to spatial coordinates
    paths_spatial = [[ijk_to_centroid(i, j, k, ds) for i, j, k in path] for path in paths]
    
    # Flatten the list of paths into a single numpy array for all points
    all_points = np.vstack(paths_spatial)
    
    # Create a lines array for PyVista
    lines = []
    offset = 0
    for path in paths_spatial:
        n_points = len(path)
        lines.append([n_points] + list(range(offset, offset + n_points)))
        offset += n_points

    # Convert to PyVista format
    lines = np.hstack(lines)
    polypaths = pv.PolyData(all_points, lines)
    polypaths.save(path_file)

    # Convert source and target voxels to spatial coordinates
    start_nodes = np.array([ijk_to_centroid(i, j, k, ds) for i, j, k in source_voxels])
    end_nodes = np.array([ijk_to_centroid(i, j, k, ds) for i, j, k in target_voxels])

    # Create start nodes with red spheres
    start_pv = pv.PolyData(start_nodes)
    start_spheres = start_pv.glyph(scale=False, geom=pv.Sphere(radius=sphere_radius))
    start_spheres.point_data['colors'] = np.array([[1, 0, 0]] * len(start_nodes))  # Red color

    # Create end nodes with blue spheres
    end_pv = pv.PolyData(end_nodes)
    end_spheres = end_pv.glyph(scale=False, geom=pv.Sphere(radius=sphere_radius))
    end_spheres.point_data['colors'] = np.array([[0, 0, 1]] * len(end_nodes))  # Blue color

    # Combine start and end spheres into one file
    nodes = start_spheres.merge(end_spheres)
    nodes.save(nodes_file)


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

def visualize_density_slice(density_surface, resistance_surface, slice_index, axis='z', agent_name='agent', sources=[], save_dir='pathwalker_3d_results'):
    """
    Visualize a 2D slice of the movement density surface with resistance overlay.

    Parameters:
        density_surface (np.ndarray): 3D numpy array of movement density.
        resistance_surface (np.ndarray): 3D numpy array of resistance values.
        slice_index (int): Index of the slice to visualize.
        axis (str): Axis along which to take the slice ('x', 'y', or 'z').
        agent_name (str): Name of the agent for plot title and filename.
        sources (list): List of source voxel indices.
        save_dir (str): Directory where the plot will be saved.
    """
    if axis == 'x':
        density_slice = density_surface[slice_index, :, :]
        resistance_slice = resistance_surface[slice_index, :, :]
        source_coords = [(j, k) for (i, j, k) in sources if i == slice_index]
        xlabel, ylabel = 'Y-axis', 'Z-axis'
    elif axis == 'y':
        density_slice = density_surface[:, slice_index, :]
        resistance_slice = resistance_surface[:, slice_index, :]
        source_coords = [(i, k) for (i, j, k) in sources if j == slice_index]
        xlabel, ylabel = 'X-axis', 'Z-axis'
    else:
        density_slice = density_surface[:, :, slice_index]
        resistance_slice = resistance_surface[:, :, slice_index]
        source_coords = [(i, j) for (i, j, k) in sources if k == slice_index]
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

    # Plot starting nodes
    if source_coords:
        x_sources, y_sources = zip(*source_coords)
        plt.scatter(x_sources, y_sources, marker='*', color='blue', s=100, label='Starting Nodes')

    plt.title(f'{agent_name.capitalize()} Movement Density - {axis.upper()}={slice_index}')
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

def main():
    # --------------------- LOAD DATA ---------------------
    print("Loading dataset...")
    try:
        ds = xr.open_dataset(DATASET_PATH)
        print("Dataset loaded successfully.")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load dataset: {e}")

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
        target_voxels = [(47, 16, 0), (11, 13, 0)]
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

    for agent_key, agent_params in agents.items():
        print(f"Simulating movements for agent '{agent_key}'...")
        for source in source_voxels:
            for run in range(NUM_RUNS):
                path = simulate_movement(
                    start=source,
                    resistance=resistance_surfaces[agent_key],
                    risk=risk_surfaces[agent_key],
                    agent=agent_params,
                    grid_shape=grid_shape,
                    targets=agents[agent_key]['targets']
                )
                simulated_paths[agent_key].append(path)
        print(f"Completed simulations for agent '{agent_key}'.")

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
