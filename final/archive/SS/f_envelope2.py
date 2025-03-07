import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import xarray as xr
import pickle
import os

def load_inputs(site, input_folder):
    """
    Load the xarray dataset and grouped roof information CSV.

    Parameters:
        site (str): Site identifier (e.g., 'uni').
        input_folder (str): Path to the folder containing input files.

    Returns:
        xarray.Dataset: Loaded xarray dataset.
        pd.DataFrame: Loaded grouped roof information DataFrame.
    """
    # Load xarray dataset
    xarray_path = os.path.join(input_folder, f"{site}xarray_voxels.pkl")
    if not os.path.exists(xarray_path):
        raise FileNotFoundError(f"xarray dataset not found at {xarray_path}")

    print(f"Loading xarray dataset from {xarray_path}...")
    with open(xarray_path, 'rb') as f:
        xarray_dataset = pickle.load(f)

    # Load grouped roof information
    roof_info_path = os.path.join(input_folder, f"{site}_roofInfo.csv")
    if not os.path.exists(roof_info_path):
        raise FileNotFoundError(f"Grouped roof info CSV not found at {roof_info_path}")

    print(f"Loading grouped roof info from {roof_info_path}...")
    grouped_roof_info = pd.read_csv(roof_info_path)

    return xarray_dataset, grouped_roof_info

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
        # Define the 6 possible neighbor directions (up, down, left, right, front, back)
        directions = [(-1, 0, 0), (1, 0, 0),
                      (0, -1, 0), (0, 1, 0),
                      (0, 0, -1), (0, 0, 1)]
        
        for di, dj, dk in directions:
            ni, nj, nk = i + di, j + dj, k + dk
            neighbor_coord = (ni, nj, nk)
            neighbor_idx = coord_to_idx.get(neighbor_coord, -1)
            if neighbor_idx != -1:
                neighbors[idx].append(neighbor_idx)
    
    return neighbors

def initialize_growth_parameters():
    """
    Initializes simulation parameters.

    Returns:
        dict: Dictionary containing simulation parameters.
    """
    params = {
        'green_weight_per_voxel': 50,    # Weight for green plants
        'brown_weight_per_voxel': 50,    # Weight for brown plants
        'facade_weight_per_voxel': 50,   # Weight when growing down the facade
        'growth_split_prob': 0.5,         # Probability of growing in multiple directions
        'termination_prob': 0.1,          # Chance of terminating growth early
        'voxel_area': 6.25                # Area of each voxel in square meters
    }
    return params

def grow_on_roof_flat(current_idx, flat_grid_data, params, greenPlants, brownPlants, remaining_roof_load, neighbors, growth_stack):
    """
    Grow green or brown plants on the roof based on the remaining roof load.

    Parameters:
        current_idx (int): Current voxel index.
        flat_grid_data (dict): Dictionary containing flattened grid data.
        params (dict): Simulation parameters.
        greenPlants (np.ndarray): 1D boolean array for green plants.
        brownPlants (np.ndarray): 1D boolean array for brown plants.
        remaining_roof_load (float): Remaining load capacity for the roof.
        neighbors (list): Precomputed neighbor indices.
        growth_stack (list): List managing the growth stack.

    Returns:
        tuple: Updated remaining_roof_load and a boolean indicating whether to continue growth.
    """
    if remaining_roof_load > params['green_weight_per_voxel']:
        greenPlants[current_idx] = True
        remaining_roof_load -= params['green_weight_per_voxel']
    elif remaining_roof_load > params['brown_weight_per_voxel']:
        brownPlants[current_idx] = True
        remaining_roof_load -= params['brown_weight_per_voxel']
    else:
        return remaining_roof_load, False  # No load left, terminate growth

    # Decide whether to terminate early based on the probability
    if random.random() < params['termination_prob']:
        return remaining_roof_load, False

    # Decide the number of directions to grow based on growth_split_prob
    if random.random() < params['growth_split_prob']:
        # Randomly select a number of neighbors to grow into (1 to 3)
        available_neighbors = [n for n in neighbors[current_idx] if not greenPlants[n] and not brownPlants[n]]
        num_new = random.randint(1, min(3, len(available_neighbors)))
        selected_neighbors = random.sample(available_neighbors, num_new) if available_neighbors else []
        for neighbor_idx in selected_neighbors:
            growth_stack.append(neighbor_idx)  # growth_stack is modified here

    return remaining_roof_load, True

def grow_on_facade_flat(current_idx, flat_grid_data, params, greenFacades, remaining_roof_load, height_above_ground):
    """
    Grow down the facade until reaching ground level or exhausting the load.

    Parameters:
        current_idx (int): Current voxel index.
        flat_grid_data (dict): Dictionary containing flattened grid data.
        params (dict): Simulation parameters.
        greenFacades (np.ndarray): 1D boolean array for green facades.
        remaining_roof_load (float): Remaining load capacity for the roof.
        height_above_ground (np.ndarray): 1D array for heights above ground.

    Returns:
        float: Updated remaining_roof_load.
    """
    while height_above_ground[current_idx] > 1 and remaining_roof_load > params['facade_weight_per_voxel']:
        greenFacades[current_idx] = True
        remaining_roof_load -= params['facade_weight_per_voxel']
        
        # Move down along the Z-axis (decreasing K)
        i, j, k = flat_grid_data['voxel_coords'][current_idx]
        new_k = k - 1
        new_coord = (i, j, new_k)
        new_idx = flat_grid_data['coord_to_idx'].get(new_coord, -1)
        
        if new_idx == -1:
            break  # Reached boundary or no voxel below
        
        current_idx = new_idx  # Continue growing downwards

    return remaining_roof_load

def run_simulation_flat(grid_data):
    """
    Runs the Cellular Automaton growth simulation using a flattened (1D) grid.

    Parameters:
        grid_data (dict): Dictionary containing flattened grid data.

    Returns:
        tuple: greenPlants, brownPlants, greenFacades (1D boolean arrays)
    """
    # Unpack grid data
    num_voxels = grid_data['num_voxels']
    greenPlants = grid_data['greenPlants']
    brownPlants = grid_data['brownPlants']
    greenFacades = grid_data['greenFacades']
    site_building_element = grid_data['site_building_element']
    roofID_grid = grid_data['roofID_grid']
    height_above_ground = grid_data['height_above_ground']
    assigned_logs = grid_data['assigned_logs']
    roof_load_info = grid_data['roof_load_info']
    voxel_coords = grid_data['voxel_coords']
    coord_to_idx = grid_data['coord_to_idx']

    # Initialize simulation parameters
    params = initialize_growth_parameters()

    # Precompute neighbors
    print("Precomputing neighbors...")
    neighbors = precompute_neighbors(grid_data)
    print("Neighbor precomputation completed.")

    steps = 0
    num_green, num_brown, num_facades = 0, 0, 0

    # Start Cellular Automaton growth
    unique_roof_ids = np.unique(roofID_grid)
    for roof_id in unique_roof_ids:
        if roof_id == -1:
            continue  # Skip voxels that are not associated with a roof

        # Retrieve initial conditions for this roof
        roof_info = roof_load_info.get(roof_id, None)
        if roof_info is None:
            print(f"RoofID {roof_id} info missing. Skipping.")
            continue  # Skip if roof info is missing

        remaining_roof_load = roof_info.get('Remaining roof load', 0)
        if remaining_roof_load <= 0:
            print(f"RoofID {roof_id} has non-positive remaining load ({remaining_roof_load}). Skipping.")
            continue  # Skip roofs with non-positive load

        print(f"Starting simulation for RoofID {roof_id} with remaining roof load: {remaining_roof_load}")

        # Identify starting nodes (voxels with assigned logs)
        start_voxels = np.where((roofID_grid == roof_id) & (assigned_logs != ''))[0]

        if len(start_voxels) == 0:
            print(f"No starting voxels found for RoofID {roof_id}. Skipping.")
            continue  # Skip roofs without starting voxels

        # Spread growth from the starting nodes
        for voxel_idx in start_voxels:
            growth_stack = [voxel_idx]  # Initialize with starting voxel

            while growth_stack and remaining_roof_load > 0:
                current_idx = growth_stack.pop()

                # Check if already assigned
                if greenPlants[current_idx] or brownPlants[current_idx] or greenFacades[current_idx]:
                    continue  # Skip already processed voxels

                element = site_building_element[current_idx]

                if element == 'roof':
                    remaining_roof_load, continue_growth = grow_on_roof_flat(
                        current_idx, grid_data, params, greenPlants, brownPlants,
                        remaining_roof_load, neighbors, growth_stack
                    )
                    if not continue_growth:
                        break
                    num_green += greenPlants[current_idx]
                    num_brown += brownPlants[current_idx]

                elif element == 'facade':
                    print(f"Growing on facade voxel index {current_idx} at coordinates {voxel_coords[current_idx]}")
                    remaining_roof_load = grow_on_facade_flat(
                        current_idx, grid_data, params, greenFacades,
                        remaining_roof_load, height_above_ground
                    )
                    num_facades += greenFacades[current_idx]

                steps += 1

                # Print roof load at every 20 steps
                if steps % 20 == 0:
                    print(f"Step {steps}: Remaining load for RoofID {roof_id}: {remaining_roof_load}")

        # Print final remaining load at the end of the simulation for this roof
        print(f"RoofID {roof_id} completed. Final remaining load: {remaining_roof_load}")

    print(f"Total steps: {steps}")
    print(f"Total Green Plants: {np.sum(greenPlants)}, Brown Plants: {np.sum(brownPlants)}, Green Facades: {np.sum(greenFacades)}")

    return greenPlants, brownPlants, greenFacades

def plot_simulation_results_flat(greenPlants, brownPlants, greenFacades, grid_data, start_voxels_indices):
    """
    Plots the results of the Cellular Automaton growth in 3D using flattened indices.

    Parameters:
        greenPlants (np.ndarray): 1D boolean array for green plants.
        brownPlants (np.ndarray): 1D boolean array for brown plants.
        greenFacades (np.ndarray): 1D boolean array for green facades.
        grid_data (dict): Dictionary containing flattened grid data.
        start_voxels_indices (np.ndarray): Array of starting voxel indices.
    """
    I = grid_data['voxel_coords'][:, 0]
    J = grid_data['voxel_coords'][:, 1]
    K = grid_data['voxel_coords'][:, 2]

    # Extract coordinates for each plant type
    green_idx = np.where(greenPlants)[0]
    brown_idx = np.where(brownPlants)[0]
    facade_idx = np.where(greenFacades)[0]
    start_idx = start_voxels_indices

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot green plants
    ax.scatter(I[green_idx], J[green_idx], K[green_idx], c='g', label='Green Plants', s=5, alpha=0.6)

    # Plot brown plants
    ax.scatter(I[brown_idx], J[brown_idx], K[brown_idx], c='brown', label='Brown Plants', s=5, alpha=0.6)

    # Plot green facades
    ax.scatter(I[facade_idx], J[facade_idx], K[facade_idx], c='b', label='Green Facades', s=5, alpha=0.6)

    # Plot starting nodes
    ax.scatter(I[start_idx], J[start_idx], K[start_idx], c='r', label='Starting Nodes', s=15)

    ax.set_xlabel('I')
    ax.set_ylabel('J')
    ax.set_zlabel('K')
    ax.legend()
    plt.title("3D Voxel Growth Simulation (Flattened Grid)")
    plt.show()

def validate_initialization(grid_data):
    """
    Validates the initialization of the flattened grid.

    Parameters:
        grid_data (dict): Dictionary containing flattened grid data.
    """
    num_voxels = grid_data['num_voxels']
    voxel_coords = grid_data['voxel_coords']
    coord_to_idx = grid_data['coord_to_idx']

    # Example validation: Check the first 5 voxels
    print("Validating first 5 voxels:")
    for idx in range(5):
        coord = tuple(voxel_coords[idx])
        mapped_idx = coord_to_idx.get(coord, -1)
        assert mapped_idx == idx, f"Mismatch at index {idx}: {mapped_idx} != {idx}"
        print(f"Voxel {idx}: Coord={coord}, Mapped Index={mapped_idx}")

    # Check for duplicate coordinates
    if len(coord_to_idx) != num_voxels:
        print("Warning: Duplicate voxel coordinates detected.")
    else:
        print("No duplicate voxel coordinates detected.")

def save_results(greenPlants, brownPlants, greenFacades, output_path):
    """
    Saves the simulation results to a file.

    Parameters:
        greenPlants (np.ndarray): 1D boolean array for green plants.
        brownPlants (np.ndarray): 1D boolean array for brown plants.
        greenFacades (np.ndarray): 1D boolean array for green facades.
        output_path (str): Path to save the results.
    """
    with open(output_path, 'wb') as f:
        pickle.dump({
            'greenPlants': greenPlants,
            'brownPlants': brownPlants,
            'greenFacades': greenFacades
        }, f)
    print(f"Results saved to {output_path}")

def inspect_assignments(greenPlants, brownPlants, greenFacades, grid_data):
    """
    Inspects and prints summary statistics of plant assignments.

    Parameters:
        greenPlants (np.ndarray): 1D boolean array for green plants.
        brownPlants (np.ndarray): 1D boolean array for brown plants.
        greenFacades (np.ndarray): 1D boolean array for green facades.
        grid_data (dict): Dictionary containing flattened grid data.
    """
    total_green = np.sum(greenPlants)
    total_brown = np.sum(brownPlants)
    total_facades = np.sum(greenFacades)

    print(f"Total Green Plants Assigned: {total_green}")
    print(f"Total Brown Plants Assigned: {total_brown}")
    print(f"Total Green Facades Assigned: {total_facades}")

    # Optionally, inspect specific voxels
    sample_indices = np.where(greenPlants | brownPlants | greenFacades)[0][:5]
    for idx in sample_indices:
        coord = grid_data['voxel_coords'][idx]
        print(f"Voxel Index {idx}: Coord={coord}, Green={greenPlants[idx]}, Brown={brownPlants[idx]}, Facade={greenFacades[idx]}")

def main():
    # Configuration
    site = 'uni'  # Example site identifier
    input_folder = '/path/to/input_folder'  # Update with your actual path
    output_folder = '/path/to/output_folder'  # Update with your actual path
    output_path = os.path.join(output_folder, 'simulation_results.pkl')

    # Load inputs
    try:
        xarray_dataset, grouped_roof_info = load_inputs(site, input_folder)
    except FileNotFoundError as e:
        print(e)
        return

    # Initialize flattened grid
    grid_data = initialize_flat_grid(xarray_dataset, grouped_roof_info)

    # Validate initialization
    validate_initialization(grid_data)

    # Run simulation
    greenPlants, brownPlants, greenFacades = run_simulation_flat(grid_data)

    # Inspect assignments
    inspect_assignments(greenPlants, brownPlants, greenFacades, grid_data)

    # Identify starting voxels for visualization
    start_voxels_indices = np.where((grid_data['roofID_grid'] >= 0) & (grid_data['assigned_logs'] != ''))[0]

    # Plot results
    plot_simulation_results_flat(greenPlants, brownPlants, greenFacades, grid_data, start_voxels_indices)

    # Save results
    save_results(greenPlants, brownPlants, greenFacades, output_path)

if __name__ == "__main__":
    main()
