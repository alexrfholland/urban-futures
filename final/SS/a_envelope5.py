import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import xarray as xr


def precompute_neighbors(grid_data):
    """
    Precompute the first shell neighbors for each voxel in the grid.
    """
    num_voxels = grid_data['num_voxels']
    voxel_coords = grid_data['voxel_coords']
    coord_to_idx = grid_data['coord_to_idx']

    neighbors_first_shell = [[] for _ in range(num_voxels)]
    directions = [(-1, 0, 0), (1, 0, 0),
                  (0, -1, 0), (0, 1, 0),
                  (0, 0, -1), (0, 0, 1)]

    for idx, (i, j, k) in enumerate(voxel_coords):
        for di, dj, dk in directions:
            ni, nj, nk = i + di, j + dj, k + dk
            neighbor_coord = (ni, nj, nk)
            neighbor_idx = coord_to_idx.get(neighbor_coord, -1)
            if neighbor_idx != -1:
                neighbors_first_shell[idx].append(neighbor_idx)

    return {'first_shell': neighbors_first_shell}


def run_simulation_green_brown_weighted_with_roof_check(grid_data, grouped_roof_info, voxel_size):
    """
    Runs the simulation for assigning green and brown plants on roofs, considering weight constraints for each type of plant.
    Ensures that growth only occurs where `site_building_element` is 'roof'.
    """
    num_voxels = grid_data['num_voxels']
    greenPlants = np.zeros(num_voxels, dtype=bool)
    brownPlants = np.zeros(num_voxels, dtype=bool)
    roofID_grid = grid_data['roofID_grid']
    assigned_logs = grid_data['assigned_logs']
    site_building_element = grid_data['site_building_element']
    roof_load_info = grid_data['roof_load_info']

    # Adjust weights based on voxel size
    volume_factor = voxel_size ** 3

    params = {
        'green_weight_per_voxel': 1500 * volume_factor,
        'brown_weight_per_voxel': 400 * volume_factor,
        'termination_prob_roof': 0.2,
        'choice': 0.0,
        'split_chance': 0.35
    }

    neighbors = precompute_neighbors(grid_data)['first_shell']
    unique_roof_ids = np.unique(roofID_grid)
    for roof_id in unique_roof_ids:
        if roof_id == -1:
            continue  # Skip voxels that are not associated with a roof

        roof_info = roof_load_info.get(roof_id, None)
        if roof_info is None or roof_info['Remaining roof load'] <= 0:
            continue  # Skip if no roof info or load is non-positive

        initial_load = roof_info['Roof load']
        remaining_roof_load = roof_info['Remaining roof load']
        start_voxels = np.where(
            (roofID_grid == roof_id) & 
            (assigned_logs != -1) & 
            (site_building_element == 'roof')
        )[0]
        if len(start_voxels) == 0:
            continue

        growth_stack = list(start_voxels)

        while growth_stack:
            current_idx = growth_stack.pop()
            if greenPlants[current_idx] or brownPlants[current_idx] or site_building_element[current_idx] != 'roof':
                continue

            if remaining_roof_load <= 2000 * volume_factor:
                brownPlants[current_idx] = True
                remaining_roof_load -= params['brown_weight_per_voxel']
            else:
                load_fraction = remaining_roof_load / initial_load
                p_brown = params['choice'] * (1 - load_fraction)
                p_brown = min(max(p_brown, 0), 1)

                if random.random() < p_brown:
                    brownPlants[current_idx] = True
                    remaining_roof_load -= params['brown_weight_per_voxel']
                else:
                    greenPlants[current_idx] = True
                    remaining_roof_load -= params['green_weight_per_voxel']

            if random.random() < params['termination_prob_roof']:
                continue

            available_neighbors = [
                n for n in neighbors[current_idx]
                if site_building_element[n] == 'roof' and not greenPlants[n] and not brownPlants[n]
            ]
            if not available_neighbors:
                continue

            num_new = random.randint(1, min(3, len(available_neighbors))) if random.random() < params['split_chance'] else 1
            selected_neighbors = random.sample(available_neighbors, num_new)
            growth_stack.extend(selected_neighbors)

        roof_load_info[roof_id]['Remaining roof load'] = remaining_roof_load

    return greenPlants, brownPlants


def plot_green_brown_voxels(greenPlants, brownPlants, grid_data):
    """
    Plots the green and brown plant voxels in a 3D scatter plot.

    Parameters:
        greenPlants (np.ndarray): Boolean array indicating green plants.
        brownPlants (np.ndarray): Boolean array indicating brown plants.
        grid_data (dict): Dictionary containing voxel coordinates.
    """
    I = grid_data['voxel_coords'][:, 0]
    J = grid_data['voxel_coords'][:, 1]
    K = grid_data['voxel_coords'][:, 2]

    green_idx = np.where(greenPlants)[0]
    brown_idx = np.where(brownPlants)[0]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot green plants
    ax.scatter(I[green_idx], J[green_idx], K[green_idx], c='g', label='Green Plants', s=5, alpha=0.6)

    # Plot brown plants
    ax.scatter(I[brown_idx], J[brown_idx], K[brown_idx], c='brown', label='Brown Plants', s=5, alpha=0.6)

    ax.set_xlabel('I')
    ax.set_ylabel('J')
    ax.set_zlabel('K')
    ax.legend()
    plt.title("3D Voxel Growth Simulation: Green and Brown Plants")
    plt.show()


def initialize_grid(xarray_dataset, grouped_roof_info):
    """
    Initializes the voxel grid and related variables from the xarray dataset.
    """
    
    #print all variables in xar
    # ray_dataset
    print(xarray_dataset.variables)
    voxel_ids = xarray_dataset['voxel'].values
    I = xarray_dataset['voxel_I'].values
    J = xarray_dataset['voxel_J'].values
    K = xarray_dataset['voxel_K'].values
    num_voxels = len(voxel_ids)
    site_building_element = xarray_dataset['site_building_element'].values
    roofID_grid = xarray_dataset['envelope_roofID'].values
    assigned_logs = xarray_dataset['envelope_logNo'].values

    # Calculate remaining roof load based on the roof info
    grouped_roof_info['Remaining roof load'] = grouped_roof_info['Roof load'] - grouped_roof_info['Log load assigned']
    roof_load_info = grouped_roof_info.set_index('roofID')[['Roof load', 'Remaining roof load']].to_dict('index')

    # Correct negative remaining roof loads
    for roof_id, info in roof_load_info.items():
        if info['Remaining roof load'] < 0:
            roof_load_info[roof_id]['Remaining roof load'] = 1000

    voxel_coords = np.vstack((I, J, K)).T
    coord_to_idx = {(i, j, k): idx for idx, (i, j, k) in enumerate(voxel_coords)}

    return {
        'num_voxels': num_voxels,
        'site_building_element': site_building_element,
        'roofID_grid': roofID_grid,
        'assigned_logs': assigned_logs,
        'roof_load_info': roof_load_info,
        'voxel_coords': voxel_coords,
        'coord_to_idx': coord_to_idx
    }


def load_inputs(site, input_folder):
    """
    Load the xarray dataset and grouped roof information CSV.
    """
    voxel_size = 1
    xarray_path = f'{input_folder}/{site}_{voxel_size}_voxelArray_withLogs.nc'
    xarray_dataset = xr.open_dataset(xarray_path)
    roof_info_path = f'{input_folder}/{site}_{voxel_size}_grouped_roof_info.csv'
    grouped_roof_info = pd.read_csv(roof_info_path)

    return xarray_dataset, grouped_roof_info


# Main execution
if __name__ == "__main__":
    site = 'city'
    input_folder = f'data/revised/final/{site}'
    xarray_dataset, grouped_roof_info = load_inputs(site, input_folder)
    voxel_size = xarray_dataset.attrs.get('voxel_size')  # Default to 1 if not provided
    print(f'voxel size: {voxel_size}')
    grid_data = initialize_grid(xarray_dataset, grouped_roof_info)
    greenPlants, brownPlants = run_simulation_green_brown_weighted_with_roof_check(grid_data, grouped_roof_info, voxel_size)
    plot_green_brown_voxels(greenPlants, brownPlants, grid_data)

