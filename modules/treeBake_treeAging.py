import os
import pickle
import json
import pandas as pd
import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from skimage import measure
from scipy.spatial import ConvexHull

def senescingTree(voxel_properties_df):
    if voxel_properties_df.iloc[0]['isPrecolonial']:
        reduced_df = voxel_properties_df[
            voxel_properties_df['_inverse_branch_order'] >= 3
        ]
    else:
        reduced_df = voxel_properties_df[
            (voxel_properties_df['_volume'] > 40) & 
            (voxel_properties_df['branchOrder'] <= 2)
        ]

    return reduced_df

def snagTree(voxel_properties_df):
    if voxel_properties_df.iloc[0]['isPrecolonial']:
        reduced_df = voxel_properties_df[
            (voxel_properties_df['_growth_volume'] > 0.2) |
            (voxel_properties_df['_inverse_branch_order'] < 2)
        ]

        large_volume_branches = voxel_properties_df[voxel_properties_df['_growth_volume'] > .2].index
        reduced_df = voxel_properties_df.loc[large_volume_branches]
        
    else:
        reduced_df = voxel_properties_df[
            (voxel_properties_df['clusterOrder'] <= 1) & 
            (voxel_properties_df['_volume'] > 40)
        ]

    return reduced_df

def identify_clusters(voxel_properties_df, voxel_size=0.1, eps=0.3, min_samples=10):
    coords = voxel_properties_df[['X', 'Y', 'Z']].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    voxel_properties_df['log_id'] = db.labels_
    
    num_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    # Print unique values and their counts
    unique_clusters = voxel_properties_df['log_id'].value_counts()
    print(f"Unique log_id values and their counts:\n{unique_clusters}")
    
    
    return voxel_properties_df, num_clusters

def mesh_voxels(voxel_properties_df, voxel_size=0.1):
    points = voxel_properties_df[['X', 'Y', 'Z']].values
    min_coords = points.min(axis=0)

    grid_shape = ((points.max(axis=0) - min_coords) / voxel_size).astype(int) + 1
    if np.any(grid_shape < 2):
        raise ValueError("Cluster is too small to create a valid mesh with Marching Cubes.")
    
    voxel_grid = np.zeros(grid_shape, dtype=bool)

    indices = ((points - min_coords) / voxel_size).astype(int)
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    voxel_grid = gaussian_filter(voxel_grid.astype(float), sigma=0.5) > 0.5

    verts, faces, _, _ = measure.marching_cubes(voxel_grid, level=0)

    verts = verts * voxel_size + min_coords

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    return mesh

def align_mesh_using_largest_facet(mesh):
    hull = ConvexHull(mesh.vertices)

    facet_areas = []
    hull_normals = []
    for simplex in hull.simplices:
        v0, v1, v2 = mesh.vertices[simplex]
        edge1 = v1 - v0
        edge2 = v2 - v0
        area_vector = np.cross(edge1, edge2)
        area = np.linalg.norm(area_vector) / 2.0
        facet_areas.append(area)
        normal = area_vector / np.linalg.norm(area_vector)
        hull_normals.append(normal)

    largest_facet_index = np.argmax(facet_areas)
    selected_facet_normal = hull_normals[largest_facet_index]

    z_axis = np.array([0, 0, 1])
    if np.abs(selected_facet_normal[2]) < 1 - 1e-3:
        rotation_matrix = trimesh.geometry.align_vectors(selected_facet_normal, z_axis)
        mesh.apply_transform(rotation_matrix)

    return mesh

def align_mesh_using_longest_edge_and_rotated_plane(mesh, angle=0):
    hull = ConvexHull(mesh.vertices)

    hull_edges = []
    for simplex in hull.simplices:
        edge = mesh.vertices[simplex[1]] - mesh.vertices[simplex[0]]
        length = np.linalg.norm(edge)
        hull_edges.append((length, simplex))

    longest_edge = max(hull_edges, key=lambda x: x[0])[1]
    
    edge_vector = mesh.vertices[longest_edge[1]] - mesh.vertices[longest_edge[0]]
    edge_vector /= np.linalg.norm(edge_vector)
    
    angle_rad = np.radians(angle)
    rotation_axis = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
    rotation_matrix = trimesh.geometry.align_vectors(edge_vector, rotation_axis)
    mesh.apply_transform(rotation_matrix)
    
    hull = ConvexHull(mesh.vertices)
    hull_normals = np.cross(
        mesh.vertices[hull.simplices[:, 1]] - mesh.vertices[hull.simplices[:, 0]],
        mesh.vertices[hull.simplices[:, 2]] - mesh.vertices[hull.simplices[:, 0]]
    )
    hull_normals /= np.linalg.norm(hull_normals, axis=1)[:, np.newaxis]

    z_axis = np.array([0, 0, 1])
    z_alignment = np.dot(hull_normals, z_axis)
    best_facet = np.argmax(z_alignment)
    selected_facet_normal = hull_normals[best_facet]

    if np.abs(selected_facet_normal[2]) < 1 - 1e-3:
        final_rotation = trimesh.geometry.align_vectors(selected_facet_normal, z_axis)
        mesh.apply_transform(final_rotation)

    return mesh

def rotate_and_translate_mesh(mesh, angle, ground_z=0):
    if angle is None:
        print('Align fallen logs largest facet (by area) to global XY plane')
        mesh = align_mesh_using_largest_facet(mesh)
    else:
        print('Align the longest edge to a rotated plane, then adjusting with best-fitting facet to the global plane')
        mesh = align_mesh_using_longest_edge_and_rotated_plane(mesh, angle=angle)
    
    min_z = mesh.vertices[:, 2].min()
    translation_vector = np.array([0, 0, ground_z - min_z])
    mesh.apply_translation(translation_vector)
    
    return mesh

def revoxelize_mesh(mesh, voxel_size=0.1):
    voxelized = mesh.voxelized(pitch=voxel_size)
    return voxelized.points

def simulate_fallen_branches(voxel_properties_df, ground_z=0, voxel_size=0.1, eps=0.3, min_samples=10, angle=None):
    voxel_properties_df = voxel_properties_df[
        ~voxel_properties_df['resource'].isin(['fallen log', 'leaf litter'])
    ]
    
    voxel_properties_df, num_clusters = identify_clusters(voxel_properties_df, voxel_size=voxel_size, eps=eps, min_samples=min_samples)

    # Print unique values and their counts
    unique_clusters = voxel_properties_df['log_id'].value_counts()
    print(f"AFTER calling identify_cluster: Unique log_id values and their counts:\n{unique_clusters}")
    
    all_fallen_voxels = []

    for log_id in range(num_clusters):
        cluster_df = voxel_properties_df[voxel_properties_df['log_id'] == log_id]

        if len(cluster_df) < 3:
            print(f"Cluster {log_id} is too small for PCA. Applying direct translation.")
            small_fallen_df = translate_small_cluster(cluster_df, ground_z=ground_z)
            all_fallen_voxels.append(small_fallen_df)
            continue

        try:
            mesh = mesh_voxels(cluster_df, voxel_size=voxel_size)
        except ValueError as e:
            print(f"Skipping cluster {log_id}: {e}")
            continue
        
        modified_mesh = rotate_and_translate_mesh(mesh, angle, ground_z=ground_z)
        
        new_voxel_points = revoxelize_mesh(modified_mesh, voxel_size=voxel_size)
        
        fallen_df = pd.DataFrame(new_voxel_points, columns=['X', 'Y', 'Z'])
        
        for col in cluster_df.columns:
            if col not in ['X', 'Y', 'Z']:
                fallen_df[col] = cluster_df[col].iloc[0]
        
        all_fallen_voxels.append(fallen_df)
    
    result_df = pd.concat(all_fallen_voxels, ignore_index=True)

    # Print unique values and their counts
    unique_clusters = result_df['log_id'].value_counts()
    print(f"END OF simulate_fallen_branches(): Unique log_id values and their counts:\n{unique_clusters}")
    
    return result_df

def translate_small_cluster(voxel_properties_df, ground_z=0):
    min_z = voxel_properties_df['Z'].min()
    translation_vector = ground_z - min_z
    voxel_properties_df['Z'] += translation_vector
    return voxel_properties_df[['X', 'Y', 'Z']]

import pickle

import random
from sklearn.cluster import DBSCAN

import pyvista as pv

def plot_detected_clusters(debug_df):
    # Convert the DataFrame to a PyVista PolyData object
    debug_df = debug_df[debug_df['detected_cluster'] != 'none']

    
    points = debug_df[['X', 'Y', 'Z']].values
    polydata = pv.PolyData(points)
    polydata.point_data['cluster'] = debug_df['detected_cluster']

    
    
    # Create a PyVista plotter object
    plotter = pv.Plotter()
    plotter.add_mesh(polydata, scalars='cluster', cmap='Set1', point_size=10)
    plotter.add_axes()
    plotter.show()


def recalculateResource(key, df, resource_dict):
    # Work on a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    is_precolonial = key[0]
    state = key[2]

    # Determine the tree label based on is_precolonial
    tree_label = 'euc' if is_precolonial else 'elm'

    # Print the count of unique resource column values and their names
    resource_counts = df['resource'].value_counts().to_dict()
    print(f"\nInitial resource counts for {tree_label}:")
    for resource, count in resource_counts.items():
        print(f"{resource}: {count}")

    # Print the initial processing statement with tree label
    print(f"Processing {tree_label} with State = {state}...")

    # Separate out the rows with 'fallen log' and 'leaf litter'
    excluded_df = df[df['resource'].isin(['fallen log', 'leaf litter'])]

    # Continue with the remaining DataFrame
    df = df[~df['resource'].isin(['fallen log', 'leaf litter'])]

    goldStandardKey = (True, 'large', 'reserve-tree', False)
    gold_standard_resource_dict = resource_dict.get(str(goldStandardKey), {})

    if state == 'snag':
        # Find all rows with resource = 'other' and change to 'dead branch'
        df.loc[df['resource'] == 'other', 'resource'] = 'dead branch'

    if state in ['fallen', 'propped']:
        # Initialize 'resource' to 'fallen branch'
        # df['resource'] = 'fallen branch'
        pass

    for resource in ['hollow', 'epiphyte']:
        resourceToAssign = round(gold_standard_resource_dict.get(resource, 0))
        print(f"Resource: {resource}")
        print(f"Number to assign: {resourceToAssign}")

        ######################################################
        if state in ['senescing', 'snag']:
            # Filter branches where the resource is not already assigned, and all Z values are >= 10
            eligible_branches = df.groupby('branch').filter(
                lambda x: (resource not in x['resource'].values) and (x['Z'].min() >= 10)
            )

            # Get a list of unique branches
            unique_branches = eligible_branches['branch'].unique()

            print(f"Number of eligible branches for {tree_label}, Resource = {resource}: {len(unique_branches)}")

            # Shuffle and select the required number of branches
            selected_branches = random.sample(list(unique_branches), min(resourceToAssign, len(unique_branches)))

            # Get the indexes of the first voxel in each selected branch
            selected_voxel_indexes = df[df['branch'].isin(selected_branches)].groupby('branch').head(1).index

            # Call the clustering function with the selected voxel indexes
            df = assign_resource_cluster(df, resource, selected_voxel_indexes)

            print(f"Number of branches selected for {tree_label}, Resource = {resource}: {len(selected_branches)}")
                    
        if state == 'propped':
            # Find all rows with Z > 3
            eligible_rows = df[df['Z'] > 3]

            print(f"Number of eligible rows for {tree_label}, Resource = {resource}: {len(eligible_rows)}")

            # Shuffle and select the required number of voxels
            selected_voxel_indexes = eligible_rows.sample(n=min(resourceToAssign, len(eligible_rows))).index

            # Call the clustering function with the selected voxel indexes
            df = assign_resource_cluster(df, resource, selected_voxel_indexes)

            print(f"Number of voxels selected for {tree_label}, Resource = {resource}: {len(selected_voxel_indexes)}")

        if state == 'fallen':
            if resource == 'epiphyte':
                # Filter rows where the resource is not 'fallen branch'
                eligible_rows = df[df['resource'] != 'fallen branch']

                print(f"Number of eligible rows for {tree_label}, Resource = {resource}: {len(eligible_rows)}")

                # Shuffle and select the required number of voxels
                selected_voxel_indexes = eligible_rows.sample(n=min(resourceToAssign, len(eligible_rows))).index

                # Call the clustering function with the selected voxel indexes
                df = assign_resource_cluster(df, resource, selected_voxel_indexes)

                print(f"Number of voxels selected for {tree_label}, Resource = {resource}: {len(selected_voxel_indexes)}")

    # Add back the excluded rows
    df = pd.concat([df, excluded_df], ignore_index=True)

    # Print the count of unique resource column values and their names
    resource_counts = df['resource'].value_counts().to_dict()
    print(f"\nFinal resource counts for {tree_label}:")
    for resource, count in resource_counts.items():
        print(f"{resource}: {count}")

    return df

def assign_resource_cluster(df, resource, selected_voxel_indexes, max_voxels=500, radius=0.5):
    # Initialize the debug DataFrame
    debug_df = df.copy()
    debug_df['detected_cluster'] = 'none'  # Initialize all as 'none'

    for idx in selected_voxel_indexes:
        # Get the coordinates of the selected voxel by its index
        seed_point = df.loc[idx, ['X', 'Y', 'Z']].values.reshape(1, -1)
        
        # Ensure the coordinates are float64 to avoid type errors
        all_points = df[['X', 'Y', 'Z']].values.astype(np.float64)
        seed_point = seed_point.astype(np.float64)
        
        # Compute the Euclidean distance from the seed voxel to all other voxels
        distances = np.sqrt(((all_points - seed_point) ** 2).sum(axis=1))
        
        # Select voxels within the specified radius
        nearby_voxels = df[distances <= radius]
        
        # Limit to max_voxels
        if len(nearby_voxels) > max_voxels:
            nearby_voxels = nearby_voxels.sample(n=max_voxels)
        
        # Assign the resource to the selected cluster points
        df.loc[nearby_voxels.index, 'resource'] = resource

        # Mark the detected clusters in the debug DataFrame
        debug_df.loc[nearby_voxels.index, 'detected_cluster'] = 'cluster'  # Mark assigned cluster voxels

        # Mark the selected seed voxel
        debug_df.loc[idx, 'detected_cluster'] = 'seed'

        # Print the assigned resource for this cluster
        print(f"Total voxels assigned for {resource} in this cluster: {len(nearby_voxels)}")

    # Call the PyVista plot function after all clusters have been processed
    #plot_detected_clusters(debug_df)
    
    return df



def explore_keys(tree_templates, fixed_size, fixed_control, fixed_improvement):
    # Iterate over the keys to find matching ones based on the fixed components
    matching_keys = []
    for key in tree_templates.keys():
        # Unpack the key
        is_precolonial, size, control, improvement, tree_id = key

        # Check if the key matches the fixed components (ignoring is_precolonial)
        if (size == fixed_size and
            control == fixed_control and
            improvement == fixed_improvement):
            matching_keys.append(key)

    return matching_keys
    
    
def main():
    # Load the tree templates dictionary from the pickle file
    with open('data/treeOutputs/tree_templates.pkl', 'rb') as file:
        tree_templates = pickle.load(file)

    with open('data/treeOutputs/tree_resources.json', 'r') as file:
        all_tree_resources_dict = json.load(file)


    # Initialize the dictionary to store the processed trees
    fallen_trees_dict = {}

    # Define the fixed components for exploration
    fixed_size = 'large'
    fixed_control = 'reserve-tree'  # or 'street-tree', 'park-tree'
    fixed_improvement = True  # or True

    # Get the matching keys based on fixed components
    matching_keys = explore_keys(tree_templates, fixed_size, fixed_control, fixed_improvement)

    #DEBUG
    """matching_keys = [
        (False, 'large', 'reserve-tree', True, 10),
        (True, 'large', 'reserve-tree', True, 11)
    ]"""


    # Iterate over each matching key and process the corresponding DataFrame
    for key in matching_keys:
        is_precolonial, _, _, _, tree_id = key  # Extract the necessary components from the key
        voxel_properties_df = tree_templates[key]

        # Initialize the 'log_id' column to -1
        voxel_properties_df['log_id'] = -1

        print(f"\nProcessing Tree ID: {tree_id}, Key: {key}")

        
        # Perform aging functions and store each result with its corresponding key
        states = {
            'senescing': senescingTree(voxel_properties_df),
            'snag': snagTree(voxel_properties_df),
            'fallen': simulate_fallen_branches(snagTree(voxel_properties_df)),
            'propped': simulate_fallen_branches(snagTree(voxel_properties_df), angle=30)
        }
        

        for state, df in states.items():
            # Ensure 'log_id' is present before any further processing
            if 'log_id' not in df.columns:
                print(f"Warning: 'log_id' is missing in state {state}. Adding 'log_id' initialized to -1.")
                df['log_id'] = -1

            
            # Construct the new dictionary key for each state
            new_key = (is_precolonial, tree_id, state)

            print(f'aging key is {new_key}')

            #Recalculate resources
            df = recalculateResource(new_key, df, all_tree_resources_dict)

             # Ensure 'log_id' is present after recalculation
            if 'log_id' not in df.columns:
                print(f"Error: 'log_id' is missing after recalculation for state {state}.")
            
            # Store the result in the dictionary
            fallen_trees_dict[new_key] = df

            # Print the key as a debug log
            print(f"Stored key: {new_key}")

            # Print the count of unique resource column values and their names
            resource_counts = fallen_trees_dict[new_key]['resource'].value_counts().to_dict()
            print(f"\n#######FINAL CHECK: Final resource counts for {new_key}:")
            for resource, count in resource_counts.items():
                print(f"{resource}: {count}")

            # Print unique values of 'log_id' to verify its presence
            unique_log_ids = df['log_id'].unique()
            print(f"Unique log_id values for {new_key}: {unique_log_ids}")

            print(df.columns)

            

    # Save the processed trees dictionary as a new pickle file
    output_path = 'data/treeOutputs/fallen_trees_dict.pkl'
    with open(output_path, 'wb') as file:
        pickle.dump(fallen_trees_dict, file)

    print(f"Fallen trees data has been saved to {output_path}")

if __name__ == "__main__":
    main()
