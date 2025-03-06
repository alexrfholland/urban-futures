import xarray as xr
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product
import os
import pyvista as pv  # New import for PyVista

# --------------------- CONFIGURATION ---------------------

# Define resistance values
LOW_RESISTANCE = 1
NUETRAL = 30
HIGH_RESISTANCE = 100

# Define agents with their specific parameters
agents = {
    'bird': {
        'energy': 2000,       # Increased energy for greater dispersal
        'speed': 20,           # Moves two steps per iteration
        'shyness': 0.1,       # Reduces resistance by 50%
        'name': 'bird'
    },
    'lizard': {
        'energy': 1600,       # Increased energy for greater dispersal
        'speed': 1,           # Moves one step per iteration
        'shyness': 0.3,       # Reduces resistance by 30%
        'name': 'lizard'
    }
}

# Define source and target nodes (example coordinates)
# Replace these with actual coordinates relevant to your dataset
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
    """
    ds = xr.open_dataset(filepath)
    return ds

def define_resistance_surface(ds):
    """
    Define resistance surfaces for bird and lizard based on the specified rules.
    Returns two 3D numpy arrays: resistance_bird and resistance_lizard.
    Also returns the grid shape.
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

    # Initialize resistance surface with high resistance
    resistance = np.full(ds.dims['voxel'], NUETRAL)

    # Assign low resistance to voxels with any resource_ variables
    resistance[low_res_mask] = LOW_RESISTANCE

    # Apply high resistance rules
    # Rule 1: Buildings with height > 10
    if 'site_building_ID' in ds and 'site_LAS_HeightAboveGround' in ds:
        high_res_buildings = (ds['site_building_ID'].notnull()) & (ds['site_LAS_HeightAboveGround'] > 10)
        resistance[high_res_buildings.values] = HIGH_RESISTANCE

    # Rule 2: Roads with specific widths
    """if 'road_roadInfo_width' in ds:
        road_widths = ds['road_roadInfo_width'].values
        high_res_roads = np.isin(road_widths, ['200 / 100 / 100', '300 / 100 / 100'])
        resistance[high_res_roads] = HIGH_RESISTANCE
    """
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

    # Create separate resistance surfaces for bird and lizard if needed
    # For simplicity, assuming same resistance for both
    resistance_bird = resistance_3d.copy()
    resistance_lizard = resistance_3d.copy()

    # Return resistance surfaces and grid shape
    return resistance_bird, resistance_lizard, grid_shape

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

def simulate_movement(start, resistance, agent_params, grid_shape, target=None):
    """
    Simulate movement for an agent in a 3D grid with speed and shyness parameters.

    :param start: Tuple (i, j, k) starting coordinates.
    :param resistance: 3D numpy array of resistance values.
    :param agent_params: Dictionary containing agent-specific parameters (energy, speed, shyness).
    :param grid_shape: Shape of the 3D grid.
    :param target: Tuple (i, j, k) target coordinates (optional).
    :return: List of coordinates representing the path.
    """
    path = [start]
    current = start
    energy = agent_params['energy']
    speed = agent_params.get('speed', 1)
    shyness = agent_params.get('shyness', 0.0)  # Default shyness is 0 (no effect)

    while energy > 0:
        for _ in range(speed):
            neighbors = get_neighbors(current, grid_shape)
            if not neighbors:
                break  # Nowhere to move

            # Adjust resistance based on shyness
            adjusted_resistance = resistance.copy()
            adjusted_resistance = adjusted_resistance * (1 - shyness)  # Reduce resistance
            resistances = np.array([adjusted_resistance[n] for n in neighbors])

            # Avoid division by zero
            resistances = np.where(resistances == 0, 0.1, resistances)
            probabilities = 1 / resistances  # Lower resistance -> higher probability
            probabilities /= probabilities.sum()

            # If target is specified, bias the movement towards the target
            if target:
                # Calculate direction to target
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
                        combined_prob = 0.7 * probabilities + 0.3 * bias
                        combined_prob /= combined_prob.sum()
                        probabilities = combined_prob

            # Choose next step
            next_step = random.choices(neighbors, weights=probabilities, k=1)[0]
            path.append(next_step)

            # Update energy
            energy -= resistance[next_step]
            current = next_step

            # Optional: Stop if target is reached
            if target and current == target:
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

def create_pyvista_polydata(density_surface, centroid_x, centroid_y, centroid_z, ds):
    """
    Convert the movement density surface into a PyVista PolyData object using centroid coordinates.

    :param density_surface: 3D numpy array of movement density.
    :param centroid_x: 1D numpy array of centroid x-coordinates.
    :param centroid_y: 1D numpy array of centroid y-coordinates.
    :param centroid_z: 1D numpy array of centroid z-coordinates.
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
    #plotter.show()

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

    # Show the plot
    #plotter.show()

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

    # Create resistance masks
    low_res_mask = resistance_slice <= (LOW_RESISTANCE * 1.5)  # Adjust threshold as needed
    high_res_mask = resistance_slice >= (HIGH_RESISTANCE * 0.7)  # Adjust threshold as needed

    plt.figure(figsize=(10, 8))

    # Plot low and high resistance voxels in light grey
    combined_resistance_mask = low_res_mask | high_res_mask
    plt.imshow(combined_resistance_mask, cmap='Greys', origin='lower', alpha=0.3)

    # Plot all populated voxels in semi-transparent grey
    plt.imshow(populated_mask, cmap='Greys', origin='lower', alpha=0.2)

    # Overlay high-density voxels with 'hot' colormap
    plt.imshow(np.where(high_density_mask, density_slice, np.nan), cmap='hot', origin='lower', alpha=0.7)

    # Plot starting nodes
    if source_coords:
        x_sources, y_sources = zip(*source_coords)
        plt.scatter(x_sources, y_sources, marker='*', color='red', s=100, label='Starting Nodes')

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

def run_simulation(agent, resistance, grid_shape, sources, targets=None):
    """
    Run movement simulations for a given agent.

    :param agent: Dictionary containing agent parameters.
    :param resistance: 3D numpy array of resistance values.
    :param grid_shape: Shape of the 3D grid.
    :param sources: List of source coordinates.
    :param targets: List of target coordinates (optional).
    :return: List of all simulated paths.
    """
    all_paths = []
    for source, target in zip(sources, targets):
        for _ in range(NUM_RUNS):
            path = simulate_movement(source, resistance, agent, grid_shape, target=target)
            all_paths.append(path)
    return all_paths

# --------------------- MAIN FUNCTION ---------------------

def main():
    # --------------------- LOAD DATA ---------------------
    ds = load_dataset(DATASET_PATH)
    print("Dataset loaded successfully.")

    # --------------------- DEFINE RESISTANCE ---------------------
    resistance_bird, resistance_lizard, grid_shape = define_resistance_surface(ds)
    print("Resistance surfaces defined.")

    # --------------------- SIMULATE MOVEMENT ---------------------
    simulated_paths = {}

    for agent_key, agent_params in agents.items():
        print(f"Simulating movements for {agent_key}...")
        sources = source_nodes[agent_key]
        targets = target_nodes.get(agent_key, [None]*len(sources))  # Handle cases without targets
        resistance = resistance_bird if agent_key == 'bird' else resistance_lizard
        paths = run_simulation(agent_params, resistance, grid_shape, sources, targets)
        simulated_paths[agent_key] = paths
        print(f"Simulations for {agent_key} completed.")

    # --------------------- AGGREGATE PATHS ---------------------
    density = {}
    for agent_key, paths in simulated_paths.items():
        print(f"Aggregating paths for {agent_key}...")
        density_surface = aggregate_paths(paths, grid_shape)
        density[agent_key] = density_surface
        save_density_surface(density_surface, agent_name=agent_key)
        print(f"Density surface for {agent_key} saved.")

    # --------------------- VISUALIZE RESULTS ---------------------
    for agent_key, density_surface in density.items():
        print(f"Visualizing results for {agent_key}...")
        # Visualize 2D slice with basemap
        slice_idx = grid_shape[2] // 2  # Middle slice along z-axis
        visualize_density_slice(
            density_surface,
            resistance_bird if agent_key == 'bird' else resistance_lizard,
            slice_index=slice_idx,
            axis='z',
            agent_name=agent_key,
            sources=source_nodes[agent_key],
            save_dir=RESULTS_DIR
        )

        # Visualize 3D movement density with basemap using PyVista
        polydata = create_pyvista_polydata(density_surface, ds['voxel_I'].values, ds['voxel_J'].values, ds['voxel_K'].values, ds)
        visualize_pyvista_polydata(
            polydata,
            agent_name=agent_key,
            save_dir=RESULTS_DIR
        )

        print(f"Visualization for {agent_key} completed.")

    # --------------------- VISUALIZE RESISTANCE SURFACE ---------------------
    print("Visualizing resistance surface...")
    visualize_resistance_surface(
        resistance_bird,
        ds,
        save_dir=RESULTS_DIR
    )

    print("All simulations and visualizations are complete. Check the results directory.")

if __name__ == "__main__":
    main()
