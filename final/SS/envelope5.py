import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import xarray as xr


def precompute_neighbors(grid_data):
    """
    Precomputes the neighbor indices for each voxel in the grid.

    Parameters:
        grid_data (dict): Dictionary containing grid data.

    Returns:
        dict: Contains first_shell and second_shell neighbors.
    """
    num_voxels = grid_data['num_voxels']
    voxel_coords = grid_data['voxel_coords']
    coord_to_idx = grid_data['coord_to_idx']

    neighbors_first_shell = [[] for _ in range(num_voxels)]
    neighbors_second_shell = [[] for _ in range(num_voxels)]

    for idx, (i, j, k) in enumerate(voxel_coords):
        # Define first shell neighbor directions (6 directions: up, down, left, right, front, back)
        first_shell_directions = [(-1, 0, 0), (1, 0, 0),
                                  (0, -1, 0), (0, 1, 0),
                                  (0, 0, -1), (0, 0, 1)]

        # Define second shell neighbor directions (one voxel away)
        second_shell_directions = [(di * 2, dj * 2, dk * 2) for di, dj, dk in first_shell_directions]

        for direction_set, neighbors in zip([first_shell_directions, second_shell_directions],
                                            [neighbors_first_shell, neighbors_second_shell]):
            for di, dj, dk in direction_set:
                ni, nj, nk = i + di, j + dj, k + dk
                neighbor_coord = (ni, nj, nk)
                neighbor_idx = coord_to_idx.get(neighbor_coord, -1)
                if neighbor_idx != -1:
                    neighbors[idx].append(neighbor_idx)

    return {'first_shell': neighbors_first_shell, 'second_shell': neighbors_second_shell}


def is_downward(neighbor_idx, current_idx, grid_data):
    """
    Determines if the neighbor voxel is directly below the current voxel.

    Parameters:
        neighbor_idx (int): Index of the neighbor voxel.
        current_idx (int): Index of the current voxel.
        grid_data (dict): Dictionary containing grid data.

    Returns:
        bool: True if the neighbor is directly below the current voxel, False otherwise.
    """
    current_k = grid_data['voxel_coords'][current_idx][2]
    neighbor_k = grid_data['voxel_coords'][neighbor_idx][2]
    return neighbor_k < current_k


def identify_edge_voxels(grid_data, greenPlants, brownPlants, neighbors):
    """
    Identifies voxels on roof edges that have facade neighbors.

    Parameters:
        grid_data (dict): Dictionary containing grid data.
        greenPlants (np.ndarray): 1D boolean array for green plants.
        brownPlants (np.ndarray): 1D boolean array for brown plants.
        neighbors (list): List of neighbors for each voxel.

    Returns:
        np.ndarray: Boolean array indicating edge voxels.
    """
    num_voxels = grid_data['num_voxels']
    site_building_element = grid_data['site_building_element']
    edgeVoxels = np.zeros(num_voxels, dtype=bool)

    for idx in range(num_voxels):
        if (site_building_element[idx] == 'roof') and (greenPlants[idx] or brownPlants[idx]):
            for neighbor_idx in neighbors[idx]:
                if site_building_element[neighbor_idx] == 'facade':
                    edgeVoxels[idx] = True
                    break

    return edgeVoxels


def run_simulation_green_and_brown_roof(grid_data, grouped_roof_info):
    """
    Runs the simulation for assigning green and brown plants on roofs, considering weight constraints for each type of plant.

    Parameters:
        grid_data (dict): Dictionary containing grid data.
        grouped_roof_info (pd.DataFrame): Loaded grouped roof information DataFrame.

    Returns:
        tuple: greenPlants, brownPlants (1D boolean arrays)
    """
    # Unpack grid data
    num_voxels = grid_data['num_voxels']
    greenPlants = np.zeros(num_voxels, dtype=bool)
    brownPlants = np.zeros(num_voxels, dtype=bool)
    roofID_grid = grid_data['roofID_grid']
    assigned_logs = grid_data['assigned_logs']
    site_building_element = grid_data['site_building_element']  # Extract building element types

    # Initialize simulation parameters
    params = {
        'green_weight_per_voxel': 1250,   # Weight for green plants
        'brown_weight_per_voxel': 10,     # Weight for brown plants
        'termination_prob_roof': 0.01,    # Chance of terminating roof growth early
        'choice': 0.25                    # Probability modifier for choosing brown plants
    }

    # Precompute neighbors
    neighbors = precompute_neighbors(grid_data)['first_shell']

    # Create a lookup for remaining roof load from grouped_roof_info
    roof_load_info = grouped_roof_info.set_index('roofID')[['Roof load', 'Remaining roof load']].to_dict('index')

    # Correct roof load values if they are negative
    negative_load_roofs = [roof_id for roof_id, info in roof_load_info.items() if info['Remaining roof load'] < 0]
    if negative_load_roofs:
        for roof_id in negative_load_roofs:
            roof_load_info[roof_id]['Remaining roof load'] = 1000  # Setting negative values to 1000 as per original code

    # Start the growth process
    unique_roof_ids = np.unique(roofID_grid)
    for roof_id in unique_roof_ids:
        if roof_id == -1:
            continue  # Skip voxels that are not associated with a roof

        # Retrieve initial conditions for this roof
        roof_info = roof_load_info.get(roof_id, None)
        if roof_info is None:
            continue  # Skip if roof info is missing

        initial_load = roof_info.get('Roof load', 0)
        remaining_roof_load = roof_info.get('Remaining roof load', 0)
        if remaining_roof_load <= 0:
            continue  # Skip roofs with non-positive load

        # Identify starting nodes (voxels with assigned logs)
        start_voxels = np.where((roofID_grid == roof_id) & (assigned_logs != '') & (site_building_element == 'roof'))[0]

        # If no starting voxels, skip this roof
        if len(start_voxels) == 0:
            continue

        # Initialize a list to keep track of active growth nodes for this roof
        growth_stack = list(start_voxels)

        while growth_stack:
            current_idx = growth_stack.pop()

            # Check if already assigned
            if greenPlants[current_idx] or brownPlants[current_idx]:
                continue  # Skip already processed voxels

            # Ensure we're only growing on roof elements
            if site_building_element[current_idx] != 'roof':
                continue

            # Grow on roof considering weight constraints
            if remaining_roof_load > params['green_weight_per_voxel'] and remaining_roof_load > params['brown_weight_per_voxel']:
                # Calculate the probability of choosing a brown plant
                load_fraction = remaining_roof_load / initial_load  # Fraction of load remaining
                p_brown = params['choice'] * (1 - load_fraction)  # Increase p_brown as load_fraction decreases
                p_brown = min(max(p_brown, 0), 1)  # Clamp between 0 and 1

                # Decide plant type based on probabilities
                if random.random() < p_brown:
                    brownPlants[current_idx] = True
                    remaining_roof_load -= params['brown_weight_per_voxel']
                else:
                    greenPlants[current_idx] = True
                    remaining_roof_load -= params['green_weight_per_voxel']
            elif remaining_roof_load > params['brown_weight_per_voxel']:
                # Only brown plant can be assigned
                brownPlants[current_idx] = True
                remaining_roof_load -= params['brown_weight_per_voxel']
            else:
                # No load left, terminate growth for this branch
                continue

            # Decide whether to terminate early based on the roof termination probability
            if random.random() < params['termination_prob_roof']:
                continue

            # Identify available neighbors that are not yet assigned
            available_neighbors = [n for n in neighbors[current_idx] if not greenPlants[n] and not brownPlants[n]]

            if len(available_neighbors) == 0:
                # No available neighbors to grow into; terminate this branch of growth
                continue

            # Determine the number of new directions to grow into (1 to 3, capped by available neighbors)
            num_new = random.randint(1, min(3, len(available_neighbors)))

            # Randomly select neighbors to grow into
            selected_neighbors = random.sample(available_neighbors, num_new)

            for neighbor_idx in selected_neighbors:
                growth_stack.append(neighbor_idx)  # Add selected neighbor to the growth stack

        # Update the remaining roof load in the roof_load_info
        roof_load_info[roof_id]['Remaining roof load'] = remaining_roof_load

    return greenPlants, brownPlants





def run_simulation_facade_growth(grid_data, greenPlants, brownPlants):
    """
    Runs the simulation for growing green facades on building facades.

    Parameters:
        grid_data (dict): Dictionary containing grid data.
        greenPlants (np.ndarray): 1D boolean array for green plants on roofs.
        brownPlants (np.ndarray): 1D boolean array for brown plants on roofs.

    Returns:
        np.ndarray: Boolean array indicating green facades.
    """
    #TODO: add an array that is the step count when each voxel was assigned a plant
    num_voxels = grid_data['num_voxels']
    greenFacades = np.zeros(num_voxels, dtype=bool)
    site_building_element = grid_data['site_building_element']
    neighbors = precompute_neighbors(grid_data)['first_shell']
    downbias = 0.5  # Bias towards downward growth
    termination_prob_facade = 0.05  # Chance of terminating facade growth early

    # Identify edge voxels to start facade growth
    edgeVoxels = identify_edge_voxels(grid_data, greenPlants, brownPlants, neighbors)
    growth_stack = list(np.where(edgeVoxels)[0])

    while growth_stack:
        print(f'Growth stack size: {len(growth_stack)}')
        current_idx = growth_stack.pop()

        # Check if already assigned
        if greenFacades[current_idx]:
            print(f'Voxel {current_idx} already assigned as facade. Skipping.')
            continue  # Skip already processed voxels

        # Ensure we're only growing on facade elements
        if site_building_element[current_idx] != 'facade':
            print(f'Voxel {current_idx} is not a facade. Skipping.')
            continue

        # Grow facade plant
        greenFacades[current_idx] = True
        print(f'Grew facade plant at voxel {current_idx}.')

        # Decide whether to terminate early based on the facade termination probability
        if random.random() < termination_prob_facade:
            print(f'Terminating growth at voxel {current_idx} early due to termination probability.')
            continue

        # Determine possible directions: downward and lateral
        possible_directions = []

        # Downward neighbors
        available_down = [n for n in neighbors[current_idx] if is_downward(n, current_idx, grid_data) and not greenFacades[n]]
        if available_down:
            possible_directions.append('down')

        # Lateral neighbors (not downward)
        available_sideways = [n for n in neighbors[current_idx] if not is_downward(n, current_idx, grid_data) and not greenFacades[n]]
        if available_sideways:
            possible_directions.append('sideways')

        # Decide direction based on downbias
        if available_down and available_sideways:
            direction_choice = random.random()
            if direction_choice < downbias:
                # Grow downward
                selected_direction = 'down'
            else:
                # Grow sideways
                selected_direction = 'sideways'
        elif available_down:
            selected_direction = 'down'
        elif available_sideways:
            selected_direction = 'sideways'
        else:
            # No available directions to grow
            print(f'No available directions to grow from voxel {current_idx}.')
            continue

        if selected_direction == 'down':
            # Grow downward
            selected_down = random.choice(available_down)
            growth_stack.append(selected_down)
            print(f'Added voxel {selected_down} to growth stack for downward growth.')
        elif selected_direction == 'sideways':
            num_sideways = random.randint(1, min(3, len(available_sideways)))
            selected_sideways = random.sample(available_sideways, num_sideways)
            growth_stack.extend(selected_sideways)
            print(f'Added voxels {selected_sideways} to growth stack for sideways growth.')

    return greenFacades



def main(grid_data, grouped_roof_info):
    """
    Main function to run the complete growth simulation.

    Parameters:
        grid_data (dict): Dictionary containing grid data.
        grouped_roof_info (pd.DataFrame): Loaded grouped roof information DataFrame.

    Returns:
        dict: Contains greenPlants, brownPlants, and greenFacades.
    """
    # Run green and brown plant growth on roofs
    greenPlants, brownPlants = run_simulation_green_and_brown_roof(grid_data, grouped_roof_info)

    # Run green facade growth
    greenFacades = run_simulation_facade_growth(grid_data, greenPlants, brownPlants)

    # Plot results
    plot_green_brown_facades(greenPlants, brownPlants, greenFacades, grid_data)

    return {
        'greenPlants': greenPlants,
        'brownPlants': brownPlants,
        'greenFacades': greenFacades
    }


def plot_green_brown_facades(greenPlants, brownPlants, greenFacades, grid_data):
    """
    Plots the results of the growth simulation for green and brown plants and facades.

    Parameters:
        greenPlants (np.ndarray): 1D boolean array for green plants.
        brownPlants (np.ndarray): 1D boolean array for brown plants.
        greenFacades (np.ndarray): 1D boolean array for green facades.
        grid_data (dict): Dictionary containing grid data.
    """
    I = grid_data['voxel_coords'][:, 0]
    J = grid_data['voxel_coords'][:, 1]
    K = grid_data['voxel_coords'][:, 2]

    # Extract coordinates for each plant type
    green_idx = np.where(greenPlants)[0]
    brown_idx = np.where(brownPlants)[0]
    facade_idx = np.where(greenFacades)[0]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot green plants
    ax.scatter(I[green_idx], J[green_idx], K[green_idx], c='g', label='Green Plants', s=5, alpha=0.6)

    # Plot brown plants
    ax.scatter(I[brown_idx], J[brown_idx], K[brown_idx], c='brown', label='Brown Plants', s=5, alpha=0.6)

    # Plot green facades
    ax.scatter(I[facade_idx], J[facade_idx], K[facade_idx], c='b', label='Green Facades', s=5, alpha=0.6)

    ax.set_xlabel('I')
    ax.set_ylabel('J')
    ax.set_zlabel('K')
    ax.legend()
    plt.title("3D Voxel Growth Simulation")
    plt.show()

def load_inputs(site, input_folder):
    """
    Load the xarray dataset and grouped roof information CSV.

    Parameters:
        site (str): Site identifier (e.g., 'city').
        input_folder (str): Path to the folder containing input files.

    Returns:
        xarray.Dataset: Loaded xarray dataset.
        pd.DataFrame: Loaded grouped roof information DataFrame.
    """
    voxel_size = 3

    # Load xarray dataset
    xarray_path = f'{input_folder}/{site}_{voxel_size}_voxelArray_withLogs.nc'
    xarray_dataset = xr.open_dataset(xarray_path)

    # Load grouped roof information
    roof_info_path = f'{input_folder}/{site}_{voxel_size}_grouped_roof_info.csv'
    grouped_roof_info = pd.read_csv(roof_info_path)

    return xarray_dataset, grouped_roof_info


def initialize_grid(xarray_dataset, grouped_roof_info):
    """
    Initializes the voxel grid and related variables from the xarray dataset.

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
    roofID_grid = xarray_dataset['envelope_roofID'].values  # 1D array
    height_above_ground = xarray_dataset['site_Contours_HeightAboveGround'].values  # 1D array
    assigned_logs = xarray_dataset['assigned_logs'].values  # 1D array

    # Create a lookup for remaining roof load from grouped_roof_info
    roof_load_info = grouped_roof_info.set_index('roofID')[['Roof load', 'Remaining roof load']].to_dict('index')

    # Correct negative remaining roof loads
    negative_load_roofs = [roof_id for roof_id, info in roof_load_info.items() if info['Remaining roof load'] < 0]
    if negative_load_roofs:
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


if __name__ == "__main__":
    site = 'city'  # Example site identifier
    input_folder = f'data/revised/final/{site}'  # Example input folder

    # Load inputs
    xarray_dataset, grouped_roof_info = load_inputs(site, input_folder)

    # Initialize grid
    grid_data = initialize_grid(xarray_dataset, grouped_roof_info)

    # Run the main simulation
    main(grid_data, grouped_roof_info)
