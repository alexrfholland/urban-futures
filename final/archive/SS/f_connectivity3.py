import xarray as xr
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product
import os
import pyvista as pv  # New import for PyVista
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

"precomputing focal resistance surfaces for each agent"
##
"Mimics how different species or entities might perceive and respond to environmental challenges based on their sensory and cognitive capabilities."
##

"Ensure that the neighborhood_size is appropriate for the scale of your grid to avoid excessive smoothing or highlighting of features."

"""Choice of Focal Function:
The selection between 'mean', 'max', and 'min' should align with the intended behavior of the agent. For instance:
Mean: Smooth movement, less sensitivity to local extremes.
Max: Heightened sensitivity to obstacles, promoting cautious navigation.
Min: Encouraging movement through the easiest paths, possibly neglecting nearby challenges."""
# --------------------- CONFIGURATION ---------------------

# Define resistance values
LOW_RESISTANCE = 1
NUETRAL = 30
HIGH_RESISTANCE = 100

# Define risk surface input
# Set to 0 to use the default risk surface
# Or provide the filename of the custom risk surface (e.g., 'custom_risk_surface.nc')
RISK_SURFACE_INPUT = 0  # Change this to the filename if using a custom risk surface

# Define agents with their specific parameters
agents = {
    'bird': {
        'energy': 5000,                # Total energy available
        'speed': 10,                     # Steps per iteration
        'autocorrelation': 0.4,        # Tendency to continue in the same direction
        'resistance_neighborhood_size': 3,  # Neighborhood size for resistance
        'resistance_focal_function': 'mean', # Focal function for resistance
        'risk': 0.1,                    # Default risk (will be overridden by risk surface)
        'targets': [                    # Destination biases
            {'coordinates': (50, 50, 10), 'bias': 0.2},
            {'coordinates': (60, 60, 10), 'bias': 0.1}
        ],
        'name': 'bird'
    },
    'lizard': {
        'energy': 1600,                # Total energy available
        'speed': 1,                     # Steps per iteration
        'autocorrelation': 0.3,        # Tendency to continue in the same direction
        'resistance_neighborhood_size': 1,  # Neighborhood size for resistance
        'resistance_focal_function': 'max',  # Focal function for resistance
        'risk': 0.2,                    # Default risk (will be overridden by risk surface)
        'targets': [                    # Destination biases
            {'coordinates': (50, 50, 10), 'bias': 0.15},
            {'coordinates': (60, 60, 10), 'bias': 0.05}
        ],
        'name': 'lizard'
    }
}

# Validate agent parameters
for agent_key, agent_params in agents.items():
    total_bias = sum(target['bias'] for target in agent_params['targets'])
    if agent_params['autocorrelation'] + total_bias > 1:
        raise ValueError(f"For agent '{agent_key}', the sum of autocorrelation and destination biases exceeds 1.")

# Define source and target nodes (example coordinates)
# Replace these with actual coordinates relevant to your dataset
source_nodes = {
    'bird': [(10, 10, 5), (20, 20, 5)],
    'lizard': [(10, 10, 5), (20, 20, 5)]
}

# Number of simulation runs per agent per target
NUM_RUNS = 500

# Path to save results
RESULTS_DIR = 'pathwalker_3d_results_updated_pyvista'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to the dataset
DATASET_PATH = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/xarray_voxels_trimmed-parade_5.nc'

# --------------------- FUNCTION DEFINITIONS ---------------------

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

def load_dataset(filepath):
    """
    Load the xarray dataset.
    """
    ds = xr.open_dataset(filepath)
    return ds

def apply_focal_function(resistance, neighborhood_size, focal_function):
    """
    Apply a focal function to the resistance surface.

    Parameters:
        resistance (np.ndarray): 3D array of resistance values.
        neighborhood_size (int): Size of the neighborhood (must be odd).
        focal_function (str): One of 'mean', 'max', or 'min'.

    Returns:
        np.ndarray: The focal resistance surface.
    """
    if neighborhood_size < 1 or neighborhood_size % 2 == 0:
        raise ValueError("Neighborhood size must be a positive odd integer.")
    
    if focal_function == 'mean':
        return uniform_filter(resistance, size=neighborhood_size, mode='nearest')
    elif focal_function == 'max':
        return maximum_filter(resistance, size=neighborhood_size, mode='nearest')
    elif focal_function == 'min':
        return minimum_filter(resistance, size=neighborhood_size, mode='nearest')
    else:
        raise ValueError("Invalid focal function. Choose from 'mean', 'max', or 'min'.")

def define_resistance_surfaces(ds, agents):
    """
    Define resistance surfaces for each agent based on their neighborhood and focal function.

    Returns:
        dict: Dictionary of resistance surfaces per agent.
        tuple: Grid shape.
    """
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
    resistance_3d = np.full(grid_shape, HIGH_RESISTANCE)

    # Assign resistance values based on voxel indices
    resistance_3d[I, J, K] = resistance

    # Compute focal resistance surfaces for each agent
    resistance_surfaces = {}
    for agent_key, agent_params in agents.items():
        neighborhood_size = agent_params['resistance_neighborhood_size']
        focal_function = agent_params['resistance_focal_function']
        focal_resistance = apply_focal_function(resistance_3d, neighborhood_size, focal_function)
        resistance_surfaces[agent_key] = focal_resistance
        print(f"Resistance surface for agent '{agent_key}' computed using '{focal_function}' over neighborhood size {neighborhood_size}.")

    return resistance_surfaces, grid_shape

def load_risk_surface(resistance_surfaces, ds):
    """
    Load the risk surface based on user input.

    Parameters:
        resistance_surfaces (dict): Dictionary of resistance surfaces per agent.
        ds (xr.Dataset): The loaded xarray Dataset.

    Returns:
        dict: Dictionary of risk surfaces per agent.
    """
    risk_surfaces = {}
    if RISK_SURFACE_INPUT == 0:
        # Use default risk surface: rescale resistance to [1, 100] and divide by 1000
        for agent_key, resistance in resistance_surfaces.items():
            min_res = resistance.min()
            max_res = resistance.max()
            rescaled = 1 + 99 * (resistance - min_res) / (max_res - min_res)  # Rescale to [1, 100]
            risk = rescaled / 1000  # Divide by 1000
            risk_surfaces[agent_key] = risk
            print(f"Default risk surface created for agent '{agent_key}'.")
    else:
        # Load custom risk surface from file
        risk_filepath = os.path.join(os.getcwd(), RISK_SURFACE_INPUT)
        if not os.path.isfile(risk_filepath):
            raise FileNotFoundError(f"Risk surface file '{RISK_SURFACE_INPUT}' not found in the session folder.")
        
        # Load the risk surface dataset
        risk_ds = xr.open_dataset(risk_filepath)
        
        # Extract risk data (assuming similar structure to resistance)
        if {'voxel_I', 'voxel_J', 'voxel_K', 'risk'}.issubset(risk_ds.data_vars):
            I = risk_ds['voxel_I'].values
            J = risk_ds['voxel_J'].values
            K = risk_ds['voxel_K'].values
            risk_values = risk_ds['risk'].values
            grid_shape = (
                int(np.max(I)) + 1,
                int(np.max(J)) + 1,
                int(np.max(K)) + 1
            )
            # Initialize risk grid
            for agent_key in resistance_surfaces.keys():
                risk_grid = np.full(grid_shape, 0.0)  # Initialize with 0
                risk_grid[I, J, K] = risk_values
                risk_surfaces[agent_key] = risk_grid
                print(f"Custom risk surface loaded for agent '{agent_key}'.")
        else:
            raise KeyError("Custom risk surface file must contain 'voxel_I', 'voxel_J', 'voxel_K', and 'risk' variables.")
    
    return risk_surfaces

def get_neighbors(coord, grid_shape):
    """
    Get all valid neighboring coordinates in 3D.
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

def simulate_movement(start, resistance, risk, agent_params, grid_shape, targets=None):
    """
    Simulate movement for an agent in a 3D grid with speed, autocorrelation, and destination biases.

    :param start: Tuple (i, j, k) starting coordinates.
    :param resistance: 3D numpy array of resistance values for the agent.
    :param risk: 3D numpy array of risk values for the agent.
    :param agent_params: Dictionary containing agent-specific parameters.
    :param grid_shape: Shape of the 3D grid.
    :param targets: List of target dictionaries with 'coordinates' and 'bias'.
    :return: List of coordinates representing the path.
    """
    path = [start]
    current = start
    energy = agent_params['energy']
    speed = agent_params.get('speed', 1)
    autocorrelation = agent_params.get('autocorrelation', 0.0)
    targets = targets or []
    
    # Initialize previous direction as None
    prev_direction = None

    while energy > 0:
        for _ in range(speed):
            neighbors = get_neighbors(current, grid_shape)
            if not neighbors:
                break  # Nowhere to move

            # Calculate adjusted resistance (already handled via focal function)
            resistances = np.array([resistance[n] for n in neighbors])

            # Avoid division by zero
            resistances = np.where(resistances == 0, 0.1, resistances)
            resistance_prob = 1 / resistances  # Lower resistance -> higher probability
            resistance_prob = resistance_prob / resistance_prob.sum()

            # Initialize movement probabilities with resistance-based probabilities
            movement_prob = resistance_prob.copy()

            # Apply autocorrelation if previous direction exists
            if prev_direction is not None and autocorrelation > 0:
                # Calculate direction vectors for neighbors
                directions = np.array([np.array(n) - np.array(current) for n in neighbors])
                norms = np.linalg.norm(directions, axis=1)
                # Avoid division by zero
                directions = np.where(norms[:, np.newaxis] == 0, 0, directions / norms[:, np.newaxis])
                if np.linalg.norm(prev_direction) != 0:
                    prev_dir_normalized = prev_direction / np.linalg.norm(prev_direction)
                else:
                    prev_dir_normalized = np.zeros(3)
                dot_products = directions.dot(prev_dir_normalized)
                autocorr_bias = (dot_products + 1) / 2  # Normalize to [0,1]
                if autocorr_bias.sum() > 0:
                    autocorr_bias /= autocorr_bias.sum()
                else:
                    autocorr_bias = np.ones_like(autocorr_bias) / len(autocorr_bias)
                movement_prob = (1 - autocorrelation) * movement_prob + autocorrelation * autocorr_bias
                movement_prob = movement_prob / movement_prob.sum()

            # Apply destination biases
            total_bias = sum(target['bias'] for target in targets)
            if total_bias > 0:
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
                # Ensure total_bias does not exceed 1 - autocorrelation
                movement_prob = (1 - total_bias - autocorrelation) * movement_prob + bias_probs
                movement_prob = movement_prob / movement_prob.sum()

            # Choose next step
            next_step = random.choices(neighbors, weights=movement_prob, k=1)[0]
            path.append(next_step)

            # Update energy
            energy -= resistance[next_step]
            current = next_step

            # Update previous direction
            prev_direction = np.array(next_step) - np.array(path[-2])

            # Apply risk of stopping based on the current voxel's risk value
            current_risk = risk[current]
            if random.random() < current_risk:
                energy = 0
                break

            if energy <= 0:
                break

    return path

def aggregate_paths(paths, grid_shape):
    """
    Aggregate multiple paths into a density surface.

    :param paths: List of paths, each path is a list of (i, j, k) tuples.
    :param grid_shape: Shape of the 3D grid.
    :return: 3D numpy array representing movement density.
    """
    density = np.zeros(grid_shape, dtype=int)
    for path in paths:
        for voxel in path:
            density[voxel] += 1
    return density

def create_pyvista_polydata(density_surface, ds):
    """
    Convert the movement density surface into a PyVista PolyData object using centroid coordinates.

    :param density_surface: 3D numpy array of movement density.
    :param ds: xarray Dataset containing voxel grid information.
    :return: PyVista PolyData object.
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

def visualize_pyvista_polydata(polydata, agent_name='agent', save_dir='pathwalker_3d_results_updated_pyvista'):
    """
    Visualize the PyVista PolyData object and save it.

    :param polydata: PyVista PolyData object.
    :param agent_name: Name of the agent for plot title and filename.
    :param save_dir: Directory where the plot and PolyData will be saved.
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

    # Show the plot
    # plotter.show()

    # Save the PolyData as a VTK file
    vtk_filename = os.path.join(save_dir, f'{agent_name}_movement_density.vtk')
    polydata.save(vtk_filename)
    print(f"PyVista PolyData for {agent_name} movement density saved as VTK file at {vtk_filename}.")

def visualize_resistance_surface(resistance_surface, ds, save_dir='pathwalker_3d_results_updated_pyvista'):
    """
    Visualize the 3D resistance surface using PyVista.

    :param resistance_surface: 3D numpy array of resistance values.
    :param ds: xarray Dataset containing voxel grid information.
    :param save_dir: Directory where the plot and PolyData will be saved.
    """
    # Get indices where resistance <= threshold
    threshold = 20000  # Adjust as needed
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

    # Show the plot
    # plotter.show()

    # Save the resistance PolyData as a VTK file
    vtk_filename = os.path.join(save_dir, 'resistance_surface.vtk')
    polydata.save(vtk_filename)
    print(f"PyVista PolyData for resistance surface saved as VTK file at {vtk_filename}.")

def visualize_density_slice(density_surface, resistance_surface, slice_index, axis='z', agent_name='agent', sources=[], save_dir='pathwalker_3d_results_updated_pyvista'):
    """
    Visualize a 2D slice of the movement density surface with basemap features.

    :param density_surface: 3D numpy array of movement density.
    :param resistance_surface: 3D numpy array of resistance values.
    :param slice_index: Index of the slice to visualize.
    :param axis: Axis along which to take the slice ('x', 'y', or 'z').
    :param agent_name: Name of the agent for plot title and filename.
    :param sources: List of source coordinates to plot as starting nodes.
    :param save_dir: Directory where the plot will be saved.
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
    print(f"{agent_name.capitalize()} movement density slice ({axis}={slice_index}) with basemap saved as PNG.")

def save_density_surface(density, agent_name='agent'):
    """
    Save the entire density surface as a NumPy binary file.

    :param density: 3D numpy array of movement density.
    :param agent_name: Name of the agent for filename.
    """
    np.save(os.path.join(RESULTS_DIR, f'{agent_name}_density.npy'), density)
    print(f"{agent_name.capitalize()} movement density surface saved as .npy file.")

def run_simulation(agent, resistance, risk, grid_shape, sources, targets=None):
    """
    Run movement simulations for a given agent.

    :param agent: Dictionary containing agent parameters.
    :param resistance: 3D numpy array of resistance values for the agent.
    :param risk: 3D numpy array of risk values for the agent.
    :param grid_shape: Shape of the 3D grid.
    :param sources: List of source coordinates.
    :param targets: List of target dictionaries with 'coordinates' and 'bias'.
    :return: List of all simulated paths.
    """
    all_paths = []
    for source in sources:
        for _ in range(NUM_RUNS):
            path = simulate_movement(source, resistance, risk, agent, grid_shape, targets=targets)
            all_paths.append(path)
    return all_paths

# --------------------- MAIN FUNCTION ---------------------

def main():
    # --------------------- LOAD DATA ---------------------
    ds = load_dataset(DATASET_PATH)
    print("Dataset loaded successfully.")

    # --------------------- DEFINE RESISTANCE ---------------------
    resistance_surfaces, grid_shape = define_resistance_surfaces(ds, agents)
    print("Resistance surfaces defined for all agents.")

    # --------------------- LOAD RISK SURFACE ---------------------
    risk_surfaces = load_risk_surface(resistance_surfaces, ds)
    print("Risk surfaces loaded successfully.")

    # --------------------- SIMULATE MOVEMENT ---------------------
    simulated_paths = {}

    for agent_key, agent_params in agents.items():
        print(f"Simulating movements for '{agent_key}'...")
        sources = source_nodes[agent_key]
        targets = agent_params.get('targets', [])
        resistance = resistance_surfaces[agent_key]
        risk = risk_surfaces[agent_key]
        paths = run_simulation(agent_params, resistance, risk, grid_shape, sources, targets=targets)
        simulated_paths[agent_key] = paths
        print(f"Simulations for '{agent_key}' completed.")

    # --------------------- AGGREGATE PATHS ---------------------
    density = {}
    for agent_key, paths in simulated_paths.items():
        print(f"Aggregating paths for '{agent_key}'...")
        density_surface = aggregate_paths(paths, grid_shape)
        density[agent_key] = density_surface
        save_density_surface(density_surface, agent_name=agent_key)
        print(f"Density surface for '{agent_key}' saved.")

    # --------------------- VISUALIZE RESULTS ---------------------
    for agent_key, density_surface in density.items():
        print(f"Visualizing results for '{agent_key}'...")
        # Visualize 2D slice with basemap
        slice_idx = grid_shape[2] // 2  # Middle slice along z-axis
        visualize_density_slice(
            density_surface,
            resistance_surfaces[agent_key],
            slice_index=slice_idx,
            axis='z',
            agent_name=agent_key,
            sources=source_nodes[agent_key],
            save_dir=RESULTS_DIR
        )

        # Visualize 3D movement density with PyVista
        polydata = create_pyvista_polydata(density_surface, ds)
        visualize_pyvista_polydata(
            polydata,
            agent_name=agent_key,
            save_dir=RESULTS_DIR
        )

        print(f"Visualization for '{agent_key}' completed.")

    # --------------------- VISUALIZE RESISTANCE SURFACE ---------------------
    print("Visualizing resistance surface for 'bird' (as an example)...")
    visualize_resistance_surface(
        resistance_surfaces['bird'],
        ds,
        save_dir=RESULTS_DIR
    )

    print("All simulations and visualizations are complete. Check the results directory.")

if __name__ == "__main__":
    main()
