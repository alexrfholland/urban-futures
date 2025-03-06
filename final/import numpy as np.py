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


def run_simulation_green_brown_weighted(grid_data, grouped_roof_info):
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

            if len(available_neighbors)