import xarray as xr
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product
import os
import pyvista as pv  # Import for PyVista

# --------------------- CONFIGURATION ---------------------

# Define resistance values
LOW_RESISTANCE = 1
NEUTRAL = 30
HIGH_RESISTANCE = 100

# Define agents with their specific parameters
agents = {
    'bird': {
        'energy': 2000,       # Total energy available
        'speed': 2,           # Steps per iteration
        'attraction': 0.3,    # Bias towards target
        'risk': 0.1,          # Risk of stopping
        'name': 'bird'
    },
    'lizard': {
        'energy': 1600,       # Total energy available
        'speed': 1,           # Steps per iteration
        'attraction': 0.2,    # Bias towards target
        'risk': 0.3,          # Risk of stopping
        'name': 'lizard'
    }
}

# Define source and target nodes (example voxel coordinates)
# Replace these with actual voxel indices relevant to your dataset
source_nodes = {
    'bird': [(10, 10, 5), (20, 20, 5)],
    'lizard': [(10, 10, 5), (20, 20, 5)]
}

target_nodes = {
    'bird': [(50, 50, 10), (60, 60, 10)],
    'lizard': [(50, 50, 10), (60, 60, 10)]
}

# Number of simulation runs per agent
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

    Parameters:
        filepath (str): Path to the NetCDF dataset.

    Returns:
        xr.Dataset: Loaded xarray Dataset.
    """
    ds = xr.open_dataset(filepath)
    return ds

def get_neighbors(coord, grid_shape):
    """
    Get all valid neighboring coordinates in 3D.

    Parameters:
        coord (tuple): Current voxel coordinates (i, j, k).
        grid_shape (tuple): Shape of the 3D grid (I, J, K).

    Returns:
        list: List of neighboring voxel coordinates as tuples.
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

def define_resistance_surface(ds):
    """
    Define resistance surfaces for bird and lizard based on the specified rules.
    Returns two 3D numpy arrays: resistance_bird and resistance_lizard.
    Also returns the grid shape.

    Parameters:
        ds (xr.Dataset): xarray Dataset containing the voxel grid information and attributes.

    Returns:
        tuple: (resistance_bird, resistance_lizard, grid_shape)
    """
    # Extract bounds and voxel_size
    bounds = ds.attrs['bounds']
    voxel_size = ds.attrs['voxel_size']
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # Calculate number of voxels along each axis
    num_i = int(np.ceil((xmax - xmin) / voxel_size))
    num_j = int(np.ceil((ymax - ymin) / voxel_size))
    num_k = int(np.ceil((zmax - zmin) / voxel_size))
    grid_shape = (num_i, num_j, num_k)
    print(f"Grid shape determined as: {grid_shape}")

    # Initialize voxel_I, voxel_J, voxel_K via meshgrid
    i = np.arange(num_i)
    j = np.arange(num_j)
    k = np.arange(num_k)
    voxel_I, voxel_J, voxel_K = np.meshgrid(i, j, k, indexing='ij')  # Shape: (I,J,K)

    # Initialize resistance fields with NEUTRAL resistance
    resistance_bird = np.full(grid_shape, NEUTRAL)
    resistance_lizard = np.full(grid_shape, NEUTRAL)
    print("Initialized resistance surfaces for Bird and Lizard with neutral resistance.")

    # Identify resource variables
    resource_vars = [var for var in ds.data_vars if var.startswith('resource_')]

    # Process resource variables
    if resource_vars:
        # Stack resource variables along a new dimension
        resource_data = ds[resource_vars].to_array(dim='resource')  # Shape: (resource, I,J,K)
        print(f"Resource data shape: {resource_data.shape}")

        # Create a mask where any of the resource variables are not null
        low_res_mask = resource_data.notnull().any(dim='resource').values  # Shape: (I,J,K)
        print(f"Low resistance mask created with {np.sum(low_res_mask)} voxels.")
    else:
        # If no resource variables, all voxels have high resistance
        low_res_mask = np.zeros(grid_shape, dtype=bool)
        print("No resource variables found. All voxels have high resistance.")

    # Assign low resistance to voxels with any resource variables
    resistance_bird[low_res_mask] = LOW_RESISTANCE
    resistance_lizard[low_res_mask] = LOW_RESISTANCE
    print("Assigned low resistance to resource-occupied voxels for both Bird and Lizard.")

    # Apply high resistance rules
    # Rule 1: Buildings with height > 10
    if 'site_building_ID' in ds and 'site_LAS_HeightAboveGround' in ds:
        high_res_buildings = (ds['site_building_ID'].notnull()) & (ds['site_LAS_HeightAboveGround'] > 10)
        high_res_buildings_mask = high_res_buildings.values  # Shape: (I,J,K)
        print(f"High resistance buildings mask created with {np.sum(high_res_buildings_mask)} voxels.")

        # Assign high resistance
        resistance_bird[high_res_buildings_mask] = HIGH_RESISTANCE * 2  # More restrictive for birds
        resistance_lizard[high_res_buildings_mask] = HIGH_RESISTANCE  # Standard high resistance for lizards
        print("Assigned high resistance to tall buildings: Double for Bird, standard for Lizard.")

    # Rule 2: Roads with specific widths (Optional)
    # Uncomment and modify as needed based on your dataset
    """
    if 'road_roadInfo_width' in ds:
        road_widths = ds['road_roadInfo_width'].values
        # Define road width categories
        high_res_roads_bird = np.isin(road_widths, ['300 / 100 / 100'])
        high_res_roads_lizard = np.isin(road_widths, ['200 / 100 / 100', '300 / 100 / 100'])
        # Assign high resistance based on road width
        resistance_bird[high_res_roads_bird] = HIGH_RESISTANCE
        resistance_lizard[high_res_roads_lizard] = HIGH_RESISTANCE
        print("Assigned high resistance to roads based on width for Bird and Lizard.")
    """

    return resistance_bird, resistance_lizard, grid_shape

def simulate_movement(start, resistance, agent_params, grid_shape, target=None):
    """
    Simulate movement for an agent in a 3D grid with speed, attraction, and risk parameters.

    Parameters:
        start (tuple): Starting voxel coordinates (i, j, k).
        resistance (np.ndarray): 3D array of resistance values.
        agent_params (dict): Agent-specific parameters (energy, speed, attraction, risk).
        grid_shape (tuple): Shape of the 3D grid (I, J, K).
        target (tuple, optional): Target voxel coordinates (i, j, k).

    Returns:
        list: List of voxel coordinates representing the path.
    """
    path = [start]
    current = start
    energy = agent_params['energy']
    speed = agent_params.get('speed', 1)
    attraction = agent_params.get('attraction', 0.0)
    risk = agent_params.get('risk', 0.0)

    while energy > 0:
        for _ in range(speed):
            neighbors = get_neighbors(current, grid_shape)
            if not neighbors:
                break  # Nowhere to move

            # Calculate movement probabilities based on resistance and attraction
            resistances = np.array([resistance[n] for n in neighbors])
            resistances = np.where(resistances == 0, 0.1, resistances)  # Avoid division by zero
            probs = 1 / resistances
            probs = probs / probs.sum()

            # Incorporate attraction towards the target
            if target is not None:
                direction = np.array(target) - np.array(current)
                norm = np.linalg.norm(direction)
                if norm != 0:
                    direction = direction / norm
                    # Compute dot product with each neighbor direction
                    neighbor_dirs = np.array([np.array(n) - np.array(current) for n in neighbors])
                    # Handle zero vectors to avoid division by zero
                    neighbor_norms = np.linalg.norm(neighbor_dirs, axis=1)
                    neighbor_dirs = np.where(neighbor_norms[:, np.newaxis] == 0, 0, neighbor_dirs / neighbor_norms[:, np.newaxis])
                    dot_products = neighbor_dirs.dot(direction)
                    # Bias probabilities towards higher dot products
                    bias = (dot_products + 1) / 2  # Normalize to [0,1]
                    bias_sum = bias.sum()
                    if bias_sum > 0:
                        bias /= bias_sum
                        # Combine with resistance-based probabilities
                        combined_prob = (1 - attraction) * probs + attraction * bias
                        combined_prob /= combined_prob.sum()
                        probs = combined_prob

            # Choose next step
            next_step = random.choices(neighbors, weights=probs, k=1)[0]
            path.append(next_step)

            # Update energy
            energy -= resistance[next_step]

            # Apply risk: possibly stop the walk
            if random.random() < risk:
                energy = 0  # Stop movement due to risk

            current = next_step

            # Optional: Stop if target is reached
            if target is not None and current == target:
                break

            if energy <= 0:
                break

    return path

def aggregate_paths(paths, grid_shape):
    """
    Aggregate multiple paths into a density surface.

    Parameters:
        paths (list): List of paths, each path is a list of (i, j, k) tuples.
        grid_shape (tuple): Shape of the 3D grid (I, J, K).

    Returns:
        np.ndarray: 3D array representing movement density.
    """
    density = np.zeros(grid_shape, dtype=int)
    for path in paths:
        for voxel in path:
            density[voxel] += 1
    return density

def create_pyvista_polydata(density_surface, ds):
    """
    Convert the movement density surface into a PyVista PolyData object using centroid coordinates.

    Parameters:
        density_surface (np.ndarray): 3D array of movement density.
        ds (xr.Dataset): xarray Dataset containing voxel grid information.

    Returns:
        pv.PolyData: PyVista PolyData object.
    """
    voxel_I = ds['voxel_I'].values
    voxel_J = ds['voxel_J'].values
    voxel_K = ds['voxel_K'].values

    # Get voxel indices where density >=1
    populated_mask = density_surface >= 1
    indices = np.argwhere(populated_mask)
    if indices.size == 0:
        raise ValueError("No voxels with density >=1 found.")

    # Extract centroid coordinates
    centroids = np.array([ijk_to_centroid(i, j, k, ds) for i, j, k in indices])

    # Extract movement frequencies
    movement_frequency = density_surface[populated_mask]

    # Create PyVista PolyData
    polydata = pv.PolyData(centroids)
    polydata.point_data["movement_frequency"] = movement_frequency

    return polydata

def create_pyvista_lines(paths, ds, agent_name='agent'):
    """
    Create a PyVista PolyData object representing movement paths as lines.

    Parameters:
        paths (list): List of paths, each path is a list of (i, j, k) tuples.
        ds (xr.Dataset): xarray Dataset containing voxel grid information.
        agent_name (str): Name of the agent for labeling.

    Returns:
        pv.PolyData: PyVista PolyData object.
    """
    lines = []
    points = []
    point_id = 0
    for path in paths:
        if len(path) < 2:
            continue  # Need at least two points to form a line
        for idx in range(len(path) - 1):
            current_voxel = path[idx]
            next_voxel = path[idx + 1]
            current_centroid = ijk_to_centroid(*current_voxel, ds)
            next_centroid = ijk_to_centroid(*next_voxel, ds)
            points.extend([current_centroid, next_centroid])
            lines.append([2, point_id, point_id + 1])
            point_id += 2

    if not lines:
        print(f"No valid paths for agent {agent_name}.")
        return None

    lines_flat = [item for sublist in lines for item in sublist]
    polydata = pv.PolyData()
    polydata.points = np.array(points)
    polydata.lines = np.array(lines_flat)

    return polydata

def visualize_pyvista_polydata(polydata, agent_name='agent', save_dir='pathwalker_3d_results_updated_pyvista'):
    """
    Visualize the PyVista PolyData object and save it.

    Parameters:
        polydata (pv.PolyData): PyVista PolyData object.
        agent_name (str): Name of the agent for plot title and filename.
        save_dir (str): Directory where the plot and PolyData will be saved.
    """
    if polydata is None:
        print(f"No polydata to visualize for agent {agent_name}.")
        return

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

    # Add legend
    plotter.add_legend()

    # Show the plot (optional)
    # plotter.show()

    # Save the PolyData as a VTK file
    vtk_filename = os.path.join(save_dir, f'{agent_name}_movement_density.vtk')
    polydata.save(vtk_filename)
    print(f"PyVista PolyData for {agent_name} movement density saved as VTK file at {vtk_filename}.")

def visualize_movement_paths(polydata, agent_name='agent', save_dir='pathwalker_3d_results_updated_pyvista'):
    """
    Visualize movement paths using PyVista and save the visualization.

    Parameters:
        polydata (pv.PolyData): PyVista PolyData object representing movement paths.
        agent_name (str): Name of the agent for plot title and filename.
        save_dir (str): Directory where the visualization will be saved.
    """
    if polydata is None:
        print(f"No movement paths to visualize for agent {agent_name}.")
        return

    # Initialize the plotter
    plotter = pv.Plotter()

    # Add movement paths
    plotter.add_mesh(polydata, color='blue', line_width=1, label='Movement Paths')

    # Add color bar (if any scalars are present)
    if 'movement_frequency' in polydata.point_data:
        plotter.add_scalar_bar(title="Movement Frequency", label_font_size=16, title_font_size=18)

    # Add axes
    plotter.add_axes()

    # Add legend
    plotter.add_legend()

    # Set camera position
    plotter.view_isometric()

    # Show the plot (optional)
    # plotter.show()

    # Save the PolyData as a VTK file
    vtk_filename = os.path.join(save_dir, f'{agent_name}_movement_paths.vtk')
    polydata.save(vtk_filename)
    print(f"PyVista PolyData for {agent_name} movement paths saved as VTK file at {vtk_filename}.")

def visualize_resistance_surface(resistance_surface, ds, agent_name='agent', save_dir='pathwalker_3d_results_updated_pyvista'):
    """
    Visualize the 3D resistance surface using PyVista.

    Parameters:
        resistance_surface (np.ndarray): 3D array of resistance values.
        ds (xr.Dataset): xarray Dataset containing voxel grid information.
        agent_name (str): Name of the agent for labeling.
        save_dir (str): Directory where the visualization will be saved.
    """
    # Get voxel indices
    voxel_I = ds['voxel_I'].values
    voxel_J = ds['voxel_J'].values
    voxel_K = ds['voxel_K'].values

    # Create a mask for resistance <= threshold
    threshold = HIGH_RESISTANCE  # Adjust as needed
    mask = resistance_surface <= threshold
    indices = np.argwhere(mask)
    if indices.size == 0:
        print("No voxels with resistance <= threshold found.")
        return

    # Extract centroid coordinates
    centroids = np.array([ijk_to_centroid(i, j, k, ds) for i, j, k in indices])

    # Extract resistance values
    resistance_values = resistance_surface[mask]

    # Create PyVista PolyData
    polydata = pv.PolyData(centroids)
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

    # Add legend
    plotter.add_legend()

    # Show the plot (optional)
    # plotter.show()

    # Save the resistance PolyData as a VTK file
    vtk_filename = os.path.join(save_dir, f'{agent_name}_resistance_surface.vtk')
    polydata.save(vtk_filename)
    print(f"PyVista PolyData for {agent_name} resistance surface saved as VTK file at {vtk_filename}.")

def save_density_surface(density, agent_name='agent'):
    """
    Save the entire density surface as a NumPy binary file.

    Parameters:
        density (np.ndarray): 3D array of movement density.
        agent_name (str): Name of the agent for filename.
    """
    np.save(os.path.join(RESULTS_DIR, f'{agent_name}_density.npy'), density)
    print(f"{agent_name.capitalize()} movement density surface saved as .npy file.")

def run_simulation(agent, resistance, grid_shape, sources, targets=None):
    """
    Run movement simulations for a given agent.

    Parameters:
        agent (dict): Dictionary containing agent parameters.
        resistance (np.ndarray): 3D array of resistance values.
        grid_shape (tuple): Shape of the 3D grid (I, J, K).
        sources (list): List of source voxel coordinates [(i, j, k), ...].
        targets (list, optional): List of target voxel coordinates [(i, j, k), ...].

    Returns:
        list: List of all simulated paths.
    """
    all_paths = []
    for source, target in zip(sources, targets):
        for run in range(NUM_RUNS):
            path = simulate_movement(source, resistance, agent, grid_shape, target=target)
            all_paths.append(path)
            if (run + 1) % 100 == 0:
                print(f"  {agent['name'].capitalize()} - Completed {run + 1} runs for source {source} to target {target}.")
    return all_paths

# --------------------- MAIN FUNCTION ---------------------

def main():
    # --------------------- LOAD DATA ---------------------
    ds = load_dataset(DATASET_PATH)
    print("Dataset loaded successfully.")

    # --------------------- DEFINE RESISTANCE ---------------------
    try:
        resistance_bird, resistance_lizard, grid_shape = define_resistance_surface(ds)
    except KeyError as e:
        print(f"KeyError: {e}. Please ensure the dataset contains 'voxel_I', 'voxel_J', 'voxel_K' variables.")
        return
    except Exception as e:
        print(f"An error occurred while defining resistance surfaces: {e}")
        return
    print("Resistance surfaces defined for Bird and Lizard.")

    # --------------------- SIMULATE MOVEMENT ---------------------
    simulated_paths = {
        'bird': [],
        'lizard': []
    }

    for agent_key, agent_params in agents.items():
        print(f"Simulating movements for {agent_key.capitalize()}...")
        resistance = resistance_bird if agent_key == 'bird' else resistance_lizard
        sources = source_nodes[agent_key]
        targets = target_nodes.get(agent_key, [None]*len(sources))  # Handle cases without targets
        paths = run_simulation(agent_params, resistance, grid_shape, sources, targets)
        simulated_paths[agent_key] = paths
        print(f"Completed simulations for {agent_key.capitalize()}.")

    # --------------------- AGGREGATE PATHS ---------------------
    density = {}
    for agent_key, paths in simulated_paths.items():
        print(f"Aggregating paths for {agent_key.capitalize()}...")
        density_surface = aggregate_paths(paths, grid_shape)
        density[agent_key] = density_surface
        save_density_surface(density_surface, agent_name=agent_key)
        print(f"Density surface for {agent_key.capitalize()} saved.")

    # --------------------- VISUALIZE RESULTS ---------------------
    for agent_key, density_surface in density.items():
        print(f"Visualizing results for {agent_key.capitalize()}...")
        # Visualize movement density as points
        try:
            polydata_density = create_pyvista_polydata(density_surface, ds)
            visualize_pyvista_polydata(
                polydata_density,
                agent_name=agent_key,
                save_dir=RESULTS_DIR
            )
        except ValueError as e:
            print(f"ValueError: {e} for {agent_key.capitalize()}. Skipping density visualization.")
            continue
        except Exception as e:
            print(f"An error occurred while visualizing density for {agent_key.capitalize()}: {e}")
            continue

        # Visualize movement paths as lines
        polydata_lines = create_pyvista_lines(simulated_paths[agent_key], ds, agent_name=agent_key)
        if polydata_lines is not None:
            visualize_movement_paths(
                polydata_lines,
                agent_name=agent_key,
                save_dir=RESULTS_DIR
            )
        else:
            print(f"No movement paths to visualize for {agent_key.capitalize()}.")

        # Visualize resistance surface
        visualize_resistance_surface(
            resistance_bird if agent_key == 'bird' else resistance_lizard,
            ds,
            agent_name=agent_key,
            save_dir=RESULTS_DIR
        )

        print(f"Visualization for {agent_key.capitalize()} completed.")

    print("All simulations and visualizations are complete. Check the results directory.")

if __name__ == "__main__":
    main()
