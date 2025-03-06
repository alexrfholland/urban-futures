import pandas as pd
import numpy as np
from scipy.spatial import KDTree

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import warnings

def plot_rotation_steps(structure_id, sampled_points, rotated_points, valid_subset, optimal_angle, angles, pivot):
    """
    Plot the rotation steps for a single tree.

    Parameters:
    - structure_id (Any): Identifier for the tree.
    - sampled_points (np.ndarray): Array of shape (N, 2) representing sampled x and y coordinates.
    - rotated_points (np.ndarray): Array of shape (rotation_steps, N, 2) representing rotated points.
    - valid_subset (np.ndarray): Array of valid points within a specified range around the tree.
    - optimal_angle (float): The chosen optimal rotation angle.
    - angles (np.ndarray): Array of rotation angles evaluated.
    - pivot (np.ndarray): Array of shape (2,) representing the pivot point (x, y).
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(valid_subset[:, 0], valid_subset[:, 1], c='black', alpha=0.3, label='Valid Points')
    plt.scatter(sampled_points[:, 0], sampled_points[:, 1], c='blue', alpha=0.6, label='Sampled Points')

    # Plot rotated points with colormap
    num_angles = len(angles)
    cmap = plt.get_cmap('viridis')
    for idx, angle in enumerate(angles):
        color = cmap(idx / num_angles)
        plt.scatter(rotated_points[idx, :, 0], rotated_points[idx, :, 1],
                    c=[color], alpha=0.3, label=f'Rotation {angle:.1f}°' if idx == 0 else "")

    # Highlight the optimal rotation in red
    optimal_rotated = rotate_points(sampled_points, optimal_angle, pivot)
    plt.scatter(optimal_rotated[:, 0], optimal_rotated[:, 1], c='red', alpha=0.9, label=f'Optimal Rotation {optimal_angle:.1f}°')

    # Plot pivot
    plt.scatter(pivot[0], pivot[1], c='green', marker='x', s=100, label='Pivot')

    plt.title(f'StructureID: {structure_id} - Rotation Optimization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def rotate_points(points, angle_deg, pivot):
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_angle, -sin_angle],
                                [sin_angle, cos_angle]])
    shifted_points = points - pivot
    rotated_shifted = shifted_points @ rotation_matrix.T
    rotated = rotated_shifted + pivot
    return rotated


def process_rotations(
    treeDF,
    resourceDF,
    valid_points,
    sample_size=100,
    rotateThrehold = 0.9,
    valid_threshold=1.0,
    rotation_steps=10,  # For example, 10 steps (36-degree increments)
    random_state=42,
    enable_plotting=False,
    trees_to_plot=None  # List of structureIDs to plot; if None, plot all if enabled
):
    """
    Optimize and apply rotations to tree point data to align with valid footprints using Pandas and vectorization.

    Parameters:
    - resourceDF (pd.DataFrame): DataFrame containing point data with at least 'structureID', 'x', 'y', 'z' columns.
    - treeDF (pd.DataFrame): DataFrame containing tree positions with at least 'structureID', 'x', 'y', 'z' columns.
    - valid_points (np.ndarray): Array of shape (N, 3) containing valid x, y, z positions.
    - sample_size (int, optional): Number of points to sample per tree for footprint. Default is 100.
    - rotateThrehold (float, optional): Randomly choose one of the top rotations when multiple options are within a certain threshold
    - valid_threshold (float, optional): Distance threshold in meters for point validation. Default is 1.0.
    - rotation_steps (int, optional): Number of discrete rotation angles to evaluate (e.g., 10 for 36-degree steps). Default is 10.
    - random_state (int, optional): Seed for random number generator to ensure reproducibility. Default is 42.
    - enable_plotting (bool, optional): Flag to enable plotting of rotation steps. Default is False.
    - trees_to_plot (list, optional): List of 'structureID's to plot. If None and plotting is enabled, plot all trees.

    Returns:
    - updated_treeDF (pd.DataFrame): Updated treeDF with a new column 'rotateZ' indicating the optimal rotation angle.
    - updated_resourceDF (pd.DataFrame): Updated resourceDF with rotated 'x' and 'y' coordinates.
    """
    # Suppress SettingWithCopyWarning
    warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

    # Step 1: Data Preparation

    # Ensure 'structureID' is unique in treeDF
    if treeDF['structureID'].duplicated().any():
        print("Duplicate 'structureID' found in treeDF. Aggregating by mean.")
        treeDF = treeDF.groupby('structureID').agg({'x': 'mean', 'y': 'mean', 'z': 'mean'}).reset_index()
        print(f"After aggregation, treeDF has {treeDF.shape[0]} unique 'structureID's.")

    # Build KDTree from valid_points (using only x and y)
    valid_xy = valid_points[:, :2]
    valid_kdtree = KDTree(valid_xy)

    # Group resourceDF by 'structureID' and sample points
    grouped = resourceDF.groupby('structureID')

    # Function to sample points
    def sample_group(group, sample_size=100):
        if len(group) <= sample_size:
            return group[['x', 'y']].values
        else:
            return group[['x', 'y']].sample(n=sample_size, random_state=random_state).values

    # Apply sampling
    sampled_dict = grouped.apply(sample_group, sample_size=sample_size).to_dict()

    # Prepare tree positions as a dictionary for quick access
    tree_positions = treeDF.set_index('structureID')[['x', 'y']].to_dict('index')
    tree_positions = {k: np.array([v['x'], v['y']]) for k, v in tree_positions.items()}

    # Initialize a list to store optimal rotations
    optimal_rotations = []

    # Create a seeded random number generator
    rng = np.random.RandomState(random_state)

    # Step 2: Determine Optimal Rotation for Each Tree
    for structure_id, sampled_points in sampled_dict.items():
        if sampled_points is None or len(sampled_points) == 0:
            optimal_rotations.append({'structureID': structure_id, 'rotateZ': 0.0})
            continue

        pivot = tree_positions.get(structure_id, np.array([0.0, 0.0]))
        angles = np.linspace(0, 360, num=rotation_steps, endpoint=False)
        angles_rad = np.deg2rad(angles)
        cos_angles = np.cos(angles_rad)
        sin_angles = np.sin(angles_rad)

        shifted = sampled_points - pivot  # Shape: (sample_size, 2)
        rotated_x = shifted[:, 0][np.newaxis, :] * cos_angles[:, np.newaxis] - shifted[:, 1][np.newaxis, :] * sin_angles[:, np.newaxis]
        rotated_y = shifted[:, 0][np.newaxis, :] * sin_angles[:, np.newaxis] + shifted[:, 1][np.newaxis] * cos_angles[:, np.newaxis]
        rotated = np.stack((rotated_x, rotated_y), axis=2) + pivot  # Shape: (rotation_steps, sample_size, 2)

        rotated_reshaped = rotated.reshape(-1, 2)
        distances, _ = valid_kdtree.query(rotated_reshaped, distance_upper_bound=valid_threshold)
        distances = distances.reshape(rotation_steps, -1)
        valid_counts = np.sum(distances < valid_threshold, axis=1)

        # Find maximum count
        max_valid_count = np.max(valid_counts)

        # Define a threshold, e.g., within 5% of the maximum count
        threshold = max_valid_count * rotateThrehold
        top_indices = np.where(valid_counts >= threshold)[0]

        # Randomly choose one of the top rotations using the seeded random number generator
        optimal_idx = rng.choice(top_indices)

        optimal_angle = angles[optimal_idx]

        optimal_rotations.append({'structureID': structure_id, 'rotateZ': optimal_angle})


        # Plotting if enabled
        if enable_plotting:
            # Determine if this tree should be plotted
            if trees_to_plot is not None:
                if structure_id not in trees_to_plot:
                    continue  # Skip plotting for this tree
            # Get rotated points for all angles
            # For plotting purposes, we'll plot each rotated set separately
            # To visualize, select the rotated points for each angle
            # For efficiency, we'll limit the number of rotation_steps when plotting
            # Here, rotation_steps is already set to a manageable number (e.g., 10)

            # Select valid points within ±25 units (assuming meters) around the tree
            x_min, x_max = pivot[0] - 25, pivot[0] + 25
            y_min, y_max = pivot[1] - 25, pivot[1] + 25
            valid_subset = valid_points[
                (valid_points[:, 0] >= x_min) & (valid_points[:, 0] <= x_max) &
                (valid_points[:, 1] >= y_min) & (valid_points[:, 1] <= y_max)
            ]

            # Extract rotated points per angle
            rotated_angles = []
            rotated_data = []
            for angle_idx, angle in enumerate(angles):
                rotated_subset = rotated[angle_idx, :, :]
                rotated_angles.append(angle)
                rotated_data.append(rotated_subset)

            rotated_array = np.array(rotated_data)  # Shape: (rotation_steps, sample_size, 2)

            # Call the plotting function
            plot_rotation_steps(
                structure_id=structure_id,
                sampled_points=sampled_points,
                rotated_points=rotated_array,
                valid_subset=valid_subset[:, :2],
                optimal_angle=optimal_angle,
                angles=angles,
                pivot=pivot
            )

    # Convert optimal_rotations to DataFrame
    rotation_df = pd.DataFrame(optimal_rotations)

    # Merge rotation angles into treeDF
    updated_treeDF = treeDF.merge(rotation_df, on='structureID', how='left')
    updated_treeDF['rotateZ'] = updated_treeDF['rotateZ'].fillna(0.0)  # Fill NaNs with 0 if any

    # Step 3: Apply the Optimal Rotation to All Points in resourceDF

    # Merge rotation angles into resourceDF
    resourceDF = resourceDF.merge(rotation_df, on='structureID', how='left')
    resourceDF['rotateZ'] = resourceDF['rotateZ'].fillna(-1)  # Fill NaNs with 0 if any

    # Function to apply rotation
    def apply_rotation(df, angle, pivot):
        points = df[['x', 'y']].values
        rotated = rotate_points(points, angle, pivot)
        df = df.copy()
        df['x'] = rotated[:, 0]
        df['y'] = rotated[:, 1]
        return df

    # Function to rotate group
    def rotate_group(group):
        angle = group['rotateZ'].iloc[0]
        pivot = tree_positions.get(group.name, np.array([0.0, 0.0]))
        if angle == 0.0:
            return group
        return apply_rotation(group, angle, pivot)

    # Group by 'structureID' and apply rotation
    updated_resourceDF = resourceDF.groupby('structureID').apply(rotate_group).reset_index(drop=True)

    # Optionally, drop the 'rotateZ' column from resourceDF if not needed
    #updated_resourceDF = updated_resourceDF.drop(columns=['rotateZ'])

    return updated_treeDF, updated_resourceDF



def process_rotationsB(
    treeDF,
    resourceDF,
    valid_points,
    sample_size=100,
    valid_threshold=1.0,
    rotation_steps=10,  # For example, 10 steps (36-degree increments)
    random_state=42,
    enable_plotting=False,
    trees_to_plot=None  # List of structureIDs to plot; if None, plot all if enabled
):
    """
    Optimize and apply rotations to tree point data to align with valid footprints using Pandas and vectorization.

    Parameters:
    - resourceDF (pd.DataFrame): DataFrame containing point data with at least 'structureID', 'x', 'y', 'z' columns.
    - treeDF (pd.DataFrame): DataFrame containing tree positions with at least 'structureID', 'x', 'y', 'z' columns.
    - valid_points (np.ndarray): Array of shape (N, 3) containing valid x, y, z positions.
    - sample_size (int, optional): Number of points to sample per tree for footprint. Default is 100.
    - valid_threshold (float, optional): Distance threshold in meters for point validation. Default is 1.0.
    - rotation_steps (int, optional): Number of discrete rotation angles to evaluate (e.g., 10 for 36-degree steps). Default is 10.
    - random_state (int, optional): Seed for random number generator to ensure reproducibility. Default is 42.
    - enable_plotting (bool, optional): Flag to enable plotting of rotation steps. Default is False.
    - trees_to_plot (list, optional): List of 'structureID's to plot. If None and plotting is enabled, plot all trees.

    Returns:
    - updated_treeDF (pd.DataFrame): Updated treeDF with a new column 'rotateZ' indicating the optimal rotation angle.
    - updated_resourceDF (pd.DataFrame): Updated resourceDF with rotated 'x' and 'y' coordinates.
    """
    # Suppress SettingWithCopyWarning
    warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

    # Step 1: Data Preparation

    # Ensure 'structureID' is unique in treeDF
    if treeDF['structureID'].duplicated().any():
        print("Duplicate 'structureID' found in treeDF. Aggregating by mean.")
        treeDF = treeDF.groupby('structureID').agg({'x': 'mean', 'y': 'mean', 'z': 'mean'}).reset_index()
        print(f"After aggregation, treeDF has {treeDF.shape[0]} unique 'structureID's.")

    # Build KDTree from valid_points (using only x and y)
    valid_xy = valid_points[:, :2]
    valid_kdtree = KDTree(valid_xy)

    # Group resourceDF by 'structureID' and sample points
    grouped = resourceDF.groupby('structureID')

    # Function to sample points
    def sample_group(group, sample_size=100):
        if len(group) <= sample_size:
            return group[['x', 'y']].values
        else:
            return group[['x', 'y']].sample(n=sample_size, random_state=random_state).values

    # Apply sampling
    sampled_dict = grouped.apply(sample_group, sample_size=sample_size).to_dict()

    # Prepare tree positions as a dictionary for quick access
    tree_positions = treeDF.set_index('structureID')[['x', 'y']].to_dict('index')
    tree_positions = {k: np.array([v['x'], v['y']]) for k, v in tree_positions.items()}

    # Initialize a list to store optimal rotations
    optimal_rotations = []

    # Step 2: Determine Optimal Rotation for Each Tree
    for structure_id, sampled_points in sampled_dict.items():
        if sampled_points is None or len(sampled_points) == 0:
            # If no points, default rotation
            optimal_rotations.append({'structureID': structure_id, 'rotateZ': 0.0})
            continue

        # Get tree position
        pivot = tree_positions.get(structure_id, np.array([0.0, 0.0]))

        # Define rotation angles
        angles = np.linspace(0, 360, num=rotation_steps, endpoint=False)

        # Precompute sine and cosine for all angles
        angles_rad = np.deg2rad(angles)
        cos_angles = np.cos(angles_rad)
        sin_angles = np.sin(angles_rad)

        # Shift sampled points relative to pivot
        shifted = sampled_points - pivot  # Shape: (sample_size, 2)

        # Rotate sampled points for all angles
        # rotated_x: Shape (rotation_steps, sample_size)
        rotated_x = shifted[:, 0][np.newaxis, :] * cos_angles[:, np.newaxis] - shifted[:, 1][np.newaxis, :] * sin_angles[:, np.newaxis]
        rotated_y = shifted[:, 0][np.newaxis, :] * sin_angles[:, np.newaxis] + shifted[:, 1][np.newaxis, :] * cos_angles[:, np.newaxis]
        rotated = np.stack((rotated_x, rotated_y), axis=2) + pivot  # Shape: (rotation_steps, sample_size, 2)

        # Reshape for KDTree query: (rotation_steps * sample_size, 2)
        rotated_reshaped = rotated.reshape(-1, 2)

        # Query KDTree for all rotated points
        distances, _ = valid_kdtree.query(rotated_reshaped, distance_upper_bound=valid_threshold)

        # Reshape distances back to (rotation_steps, sample_size)
        distances = distances.reshape(rotation_steps, -1)

        # Count how many points are within the threshold for each rotation
        valid_counts = np.sum(distances < valid_threshold, axis=1)

        # Find the angle with the maximum count
        optimal_idx = np.argmax(valid_counts)
        optimal_angle = angles[optimal_idx]

        # Append to the list
        optimal_rotations.append({'structureID': structure_id, 'rotateZ': optimal_angle})

        # Plotting if enabled
        if enable_plotting:
            # Determine if this tree should be plotted
            if trees_to_plot is not None:
                if structure_id not in trees_to_plot:
                    continue  # Skip plotting for this tree
            # Get rotated points for all angles
            # For plotting purposes, we'll plot each rotated set separately
            # To visualize, select the rotated points for each angle
            # For efficiency, we'll limit the number of rotation_steps when plotting
            # Here, rotation_steps is already set to a manageable number (e.g., 10)

            # Select valid points within ±25 units (assuming meters) around the tree
            x_min, x_max = pivot[0] - 25, pivot[0] + 25
            y_min, y_max = pivot[1] - 25, pivot[1] + 25
            valid_subset = valid_points[
                (valid_points[:, 0] >= x_min) & (valid_points[:, 0] <= x_max) &
                (valid_points[:, 1] >= y_min) & (valid_points[:, 1] <= y_max)
            ]

            # Extract rotated points per angle
            rotated_angles = []
            rotated_data = []
            for angle_idx, angle in enumerate(angles):
                rotated_subset = rotated[angle_idx, :, :]
                rotated_angles.append(angle)
                rotated_data.append(rotated_subset)

            rotated_array = np.array(rotated_data)  # Shape: (rotation_steps, sample_size, 2)

            # Call the plotting function
            plot_rotation_steps(
                structure_id=structure_id,
                sampled_points=sampled_points,
                rotated_points=rotated_array,
                valid_subset=valid_subset[:, :2],
                optimal_angle=optimal_angle,
                angles=angles,
                pivot=pivot
            )

    # Convert optimal_rotations to DataFrame
    rotation_df = pd.DataFrame(optimal_rotations)

    # Merge rotation angles into treeDF
    updated_treeDF = treeDF.merge(rotation_df, on='structureID', how='left')
    updated_treeDF['rotateZ'] = updated_treeDF['rotateZ'].fillna(0.0)  # Fill NaNs with 0 if any

    # Step 3: Apply the Optimal Rotation to All Points in resourceDF

    # Merge rotation angles into resourceDF
    resourceDF = resourceDF.merge(rotation_df, on='structureID', how='left')
    resourceDF['rotateZ'] = resourceDF['rotateZ'].fillna(-1)  # Fill NaNs with 0 if any

    # Function to apply rotation
    def apply_rotation(df, angle, pivot):
        points = df[['x', 'y']].values
        rotated = rotate_points(points, angle, pivot)
        df = df.copy()
        df['x'] = rotated[:, 0]
        df['y'] = rotated[:, 1]
        return df

    # Function to rotate group
    def rotate_group(group):
        angle = group['rotateZ'].iloc[0]
        pivot = tree_positions.get(group.name, np.array([0.0, 0.0]))
        if angle == 0.0:
            return group
        return apply_rotation(group, angle, pivot)

    # Group by 'structureID' and apply rotation
    updated_resourceDF = resourceDF.groupby('structureID').apply(rotate_group).reset_index(drop=True)

    # Optionally, drop the 'rotateZ' column from resourceDF if not needed
    #updated_resourceDF = updated_resourceDF.drop(columns=['rotateZ'])

    return updated_treeDF, updated_resourceDF

