import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import xarray as xr
import pickle
import os

INPUTFOLDER = "data/revised/envelopes" 

def initialize_growth_parameters():
    """
    Initializes simulation parameters.

    Returns:
        dict: Dictionary containing simulation parameters.
    """
    params = {
        'green_weight_per_voxel': 1250,    # Weight for green plants
        'brown_weight_per_voxel': 10,      # Weight for brown plants
        'facade_weight_per_voxel': 10,     # Weight when growing down the facade
        'growth_split_prob': 1.0,          # Ensure splitting occurs every turn
        'termination_prob': 0.01,          # Chance of terminating growth early
        'downbias': 0.8,                    # Bias towards downward growth (0: no bias, 1: always downward)
        'choice': 0,                      # Weight for brown plant selection (0: no bias towards brown, 1: full bias)
        'voxel_area': 6.25                  # Area of each voxel in square meters
    }
    return params


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

        # **Add Step: Check and Update Negative Remaining Roof Loads**
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


def is_downward(n, current_idx, flat_grid_data):
    """
    Determines if voxel 'n' is directly below 'current_idx'.

    Parameters:
        n (int): Neighbor voxel index.
        current_idx (int): Current voxel index.
        flat_grid_data (dict): Dictionary containing flattened grid data.

    Returns:
        bool: True if 'n' is directly below 'current_idx', False otherwise.
    """
    current_k = flat_grid_data['voxel_coords'][current_idx][2]
    neighbor_k = flat_grid_data['voxel_coords'][n][2]
    return neighbor_k < current_k



def grow_on_roof_flat(current_idx, flat_grid_data, params, greenPlants, brownPlants, remaining_roof_load, neighbors, growth_stack, initial_load):
    """
    Grow green or brown plants on the roof based on the remaining roof load and choice parameter.

    Parameters:
        current_idx (int): Current voxel index.
        flat_grid_data (dict): Dictionary containing flattened grid data.
        params (dict): Simulation parameters, including 'choice'.
        greenPlants (np.ndarray): 1D boolean array for green plants.
        brownPlants (np.ndarray): 1D boolean array for brown plants.
        remaining_roof_load (float): Remaining load capacity for the roof.
        neighbors (list): Precomputed neighbor indices.
        growth_stack (list): List managing the growth stack.
        initial_load (float): Initial load capacity of the roof.

    Returns:
        tuple: Updated remaining_roof_load and a boolean indicating whether to continue growth.
    """
    if remaining_roof_load > params['green_weight_per_voxel'] and remaining_roof_load > params['brown_weight_per_voxel']:
        # Calculate the probability of choosing a brown plant
        load_fraction = remaining_roof_load / initial_load  # Fraction of load remaining
        p_brown = params['choice'] * (1 - load_fraction)  # Increase p_brown as load_fraction decreases
        p_brown = min(max(p_brown, 0), 1)  # Clamp between 0 and 1
        p_green = 1 - p_brown

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
        return remaining_roof_load, False  # No load left, terminate growth

    # Decide whether to terminate early based on the probability
    if random.random() < params['termination_prob']:
        return remaining_roof_load, False

    # Always attempt to split (growth_split_prob = 1.0)
    # Identify available neighbors that are not yet assigned
    available_neighbors = [n for n in neighbors[current_idx] if not greenPlants[n] and not brownPlants[n]]

    if len(available_neighbors) == 0:
        # No available neighbors to grow into; terminate this branch of growth
        return remaining_roof_load, True

    # Determine the number of new directions to grow into (1 to 3, capped by available neighbors)
    num_new = random.randint(1, min(3, len(available_neighbors)))

    # Randomly select neighbors to grow into
    selected_neighbors = random.sample(available_neighbors, num_new)

    for neighbor_idx in selected_neighbors:
        growth_stack.append(neighbor_idx)  # Add selected neighbor to the growth stack

    return remaining_roof_load, True

def grow_on_facade_flat(current_idx, flat_grid_data, params, greenFacades, remaining_roof_load, height_above_ground, neighbors, growth_stack):
    """
    Grow facades downwards and possibly sideways based on downbias.

    Parameters:
        current_idx (int): Current voxel index.
        flat_grid_data (dict): Dictionary containing flattened grid data.
        params (dict): Simulation parameters, including 'downbias'.
        greenFacades (np.ndarray): 1D boolean array for green facades.
        remaining_roof_load (float): Remaining load capacity for the roof.
        height_above_ground (np.ndarray): 1D array for heights above ground.
        neighbors (list): Precomputed neighbor indices.
        growth_stack (list): List managing the growth stack.

    Returns:
        tuple: Updated remaining_roof_load and a boolean indicating whether to continue growth.
    """
    if remaining_roof_load > params['facade_weight_per_voxel']:
        greenFacades[current_idx] = True
        remaining_roof_load -= params['facade_weight_per_voxel']
    else:
        return remaining_roof_load, False  # No load left, terminate growth

    # Decide whether to terminate early based on the probability
    if random.random() < params['termination_prob']:
        return remaining_roof_load, False

    # Always attempt to split (growth_split_prob = 1.0)
    # Determine direction based on downbias
    if random.random() < params['downbias']:
        # Grow downward
        available_down = [n for n in neighbors[current_idx] if is_downward(n, current_idx, flat_grid_data)]
        if available_down:
            # Assuming only one downward neighbor
            growth_stack.append(available_down[0])
    else:
        # Grow sideways: choose available facade neighbors excluding downward
        available_sideways = [n for n in neighbors[current_idx] if not is_downward(n, current_idx, flat_grid_data) and not greenFacades[n]]
        if available_sideways:
            # Randomly select one sideways direction to grow
            selected_neighbor = random.choice(available_sideways)
            growth_stack.append(selected_neighbor)

    return remaining_roof_load, True


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

        initial_load = roof_info.get('Roof load', 0)
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
                        remaining_roof_load, neighbors, growth_stack, initial_load
                    )
                    if not continue_growth:
                        break
                    num_green += greenPlants[current_idx]
                    num_brown += brownPlants[current_idx]

                elif element == 'facade':
                    print(f"Growing on facade voxel index {current_idx} at coordinates {voxel_coords[current_idx]}")
                    remaining_roof_load, continue_growth = grow_on_facade_flat(
                        current_idx, grid_data, params, greenFacades,
                        remaining_roof_load, height_above_ground,
                        neighbors, growth_stack
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

def export_results(xarray_dataset, greenPlants, brownPlants, greenFacades, grid_data, output_folder, site):
    """
    Maps simulation results to 'envelopeBioavailability', saves the updated xarray dataset,
    and creates a PyVista PolyData object with point data.

    Parameters:
        xarray_dataset (xarray.Dataset): The original xarray dataset.
        greenPlants (np.ndarray): 1D boolean array indicating green plants.
        brownPlants (np.ndarray): 1D boolean array indicating brown plants.
        greenFacades (np.ndarray): 1D boolean array indicating green facades.
        grid_data (dict): Dictionary containing flattened grid data.
        output_pickle_path (str): Path to save the updated xarray pickle file.
        output_vtk_path (str): Path to save the PyVista PolyData VTK file.
    """
    import numpy as np
    import pickle
    import pyvista as pv
    import os

    # Map simulation results to 'envelopeBioavailability'
    print("Mapping simulation results to 'envelopeBioavailability'...")
    bioavailability = np.array(['none'] * grid_data['num_voxels'], dtype='<U12')  # Initialize with 'none'

    # Assign 'greenPlant' where greenPlants is True
    bioavailability[greenPlants] = 'greenPlant'

    # Assign 'brownPlant' where brownPlants is True
    bioavailability[brownPlants] = 'brownPlant'

    # Assign 'greenFacade' where greenFacades is True
    bioavailability[greenFacades] = 'greenFacade'

    # Add 'envelopeBioavailability' to the xarray dataset
    xarray_dataset = xarray_dataset.assign(envelopeBioavailability=(('voxel'), bioavailability))

    xpath  = os.path.join(output_folder, f'{site}_envelope_sim.pkl')
    ppath  = os.path.join(output_folder, f'{site}_envelope_sim.vtk')

    # Save the updated xarray dataset as a pickle
    print(f"Saving updated xarray dataset with 'envelopeBioavailability' to {xpath}...")
    with open(xpath, 'wb') as f:
        pickle.dump(xarray_dataset, f)
    print("xarray dataset saved successfully.")

    # Create PyVista PolyData
    print("Creating PyVista PolyData object...")

    # Extract points from 'centroid_x', 'centroid_y', 'centroid_z'
    try:
        points = np.vstack((
            xarray_dataset['centroid_x'].values,
            xarray_dataset['centroid_y'].values,
            xarray_dataset['centroid_z'].values
        )).T  # Shape: (num_voxels, 3)
    except KeyError as e:
        print(f"Missing required coordinate data in xarray dataset: {e}")
        return

    # Create the PolyData object
    polydata = pv.PolyData(points)

    # Add all other variables as point data
    # Exclude 'centroid_x', 'centroid_y', 'centroid_z' as they are used as coordinates
    print("Adding point data to PolyData...")
    for var in xarray_dataset.data_vars:
        if var not in ['centroid_x', 'centroid_y', 'centroid_z']:
            data = xarray_dataset[var].values
            if data.dtype.kind in {'S', 'U'}:
                # Convert bytes or unicode to string
                data = data.astype(str)
            polydata.point_data[var] = data

    # Save the PolyData as VTK file
    print(f"Saving PolyData to {ppath}...")
    polydata.save(ppath)
    print("PolyData saved successfully.")




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

def get_bioenvelopes(site, xarray_dataset, grouped_roof_info):
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
    export_results(xarray_dataset, greenPlants, brownPlants, greenFacades, grid_data, INPUTFOLDER, site)

    return xarray_dataset



if __name__ == "__main__":
    
    site = 'uni'  # Example site identifier # Update with your actual path
    xarray_dataset, grouped_roof_info = load_inputs(site, INPUTFOLDER)
    get_bioenvelopes(site, xarray_dataset, grouped_roof_info)
