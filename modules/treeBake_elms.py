import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
import os

filename_dict = {
    "smallmed_QSM1_Tree1_cylinder.csv": ("medium", 1),
    "smallmed_QSM2_Tree2_cylinder.csv": ("medium", 2),
    "smallmed_QSM3_Tree3_cylinder.csv": ("medium", 3),
    "smallmed_QSM4_Tree4_cylinder.csv": ("small", 4),
    "smallmed_QSM5_Tree5_cylinder.csv": ("small", 5),
    "smallmed_QSM6_Tree6_cylinder.csv": ("small", 6),
    "large_QSM1_Tree1_cylinder.csv": ("large", 7),
    "large_QSM2_Tree2_cylinder.csv": ("large", 8),
    "large_QSM3_Tree3_cylinder.csv": ("large", 9),
    "large_QSM4_Tree4_cylinder.csv": ("large", 10),
    "large_QSM5_Tree5_cylinder.csv": ("large", 11),
    "large2_QSM1_Tree1_cylinder.csv": ("large", 12),
    "large2_QSM2_Tree2_cylinder.csv": ("large", 13),
    "large2_QSM3_Tree3_cylinder.csv": ("large", 14)
}

point_cloud_filenames = {
    4: "Small A.txt",
    5: "Small B.txt",
    6: "Small C.txt",
    1: "Med A 1 mil.txt",
    2: "Med B 1 mil.txt",
    3: "Med C 1 mil.txt",
    7: "ElmL1.txt",
    8: "Elm L2.txt",
    9: "Elm L3.txt",
    10: "Elm L4.txt",
    11: "Elm L5.txt",
    12: "Large Elm A 1 mil.txt",
    13: "Large Elm B - 1 mil.txt",
    14: "Large Elm C 1 mil.txt"
}


file_name_dict = 
{}

def load_csv_files(folder_path):
    """Load all CSV files from a given folder into a list of dataframes."""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = [(file, pd.read_csv(os.path.join(folder_path, file))) for file in csv_files]
    return dataframes

def build_children_dict(branch_data):
    """Create a dictionary mapping each branch to its children."""
    children = defaultdict(list)
    for index, row in branch_data.iterrows():
        if row['parent'] != 0:  # Assuming '0' as the main trunk with no parent
            children[row['parent']].append(row.name + 1)  # Use row index + 1 as branch ID
    return children

def collect_descendants(branch_id, group_id, branch_to_group, children):
    """Recursively collect all descendants of a given branch and assign them to a group."""
    stack = [branch_id]
    while stack:
        current_branch = stack.pop()
        branch_to_group[current_branch] = group_id
        stack.extend(children[current_branch])

def process_branch_data(branch_data, root_order):
    """Cluster branches based on the root node order."""
    children = build_children_dict(branch_data)
    branch_to_group = {}

    root_branches = branch_data[branch_data['order'] == root_order].index + 1  # branch ID is index + 1

    group_id = 1
    for branch_id in root_branches:
        collect_descendants(branch_id, group_id, branch_to_group, children)
        group_id += 1

    branch_data['segment'] = branch_data.index.map(lambda x: branch_to_group.get(x + 1, 0))
    return branch_data

def cluster_branches_from_graph_csvs(input_folder, output_folder, root_order=1):
    #NOTE: Could do a check and if the cluster segment has too many voxels go down the root order
    """Process all CSV files in a folder and cluster branches, saving results in a new folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    dataframes = load_csv_files(input_folder)
    
    processed_dfs = []
    for filename, df in dataframes:
        processed_df = process_branch_data(df, root_order)
        output_path = os.path.join(output_folder, filename)
        processed_df.to_csv(output_path, index=False)
        processed_dfs.append((filename, processed_df))
    
    return processed_dfs

def transfer_graph_to_cylinders(graph_dfs, cylinder_folder):
    """Transfer graph information to cylinder dataframes based on branch ID."""

     # Create a dictionary to map graph file prefixes to their corresponding dataframes
    graph_dict = {}
    for graph_filename, graph_df in graph_dfs:
        # Extract the common part of the filename (before the last underscore)
        prefix = '_'.join(graph_filename.split('_')[:-1])
        graph_dict[prefix] = (graph_filename, graph_df)
    
    # Load cylinder dataframes
    cylinder_dataframes = load_csv_files(cylinder_folder)
    
    for cylinder_filename, cylinder_df in cylinder_dataframes:
        # Extract the common part of the filename (before the last underscore)
        prefix = '_'.join(cylinder_filename.split('_')[:-1])
        
        if prefix in graph_dict:
            graph_filename, graph_df = graph_dict[prefix]
            print(f"Processing graph file: {graph_filename}")
            print(f"Processing cylinder file: {cylinder_filename}")
            
            max_index = graph_df.index.max()
            
            # Check and print the range of indices in the graph dataframe
            print(f"Graph dataframe '{graph_filename}' has indices ranging from {graph_df.index.min()} to {max_index}")
            
            # Check if branch IDs are within valid range
            invalid_branch_ids = cylinder_df['branch'] > (max_index + 1)
            if invalid_branch_ids.any():
                raise ValueError(f"Invalid branch IDs found in {cylinder_filename}: {cylinder_df['branch'][invalid_branch_ids].values}")
            
            # Map graph data to each cylinder row based on the branch ID
            for column in ['diameter', 'volume', 'area', 'length', 'angle', 'height', 'segment']:
                cylinder_df[column] = cylinder_df['branch'].apply(lambda x: graph_df.at[x - 1, column])
        else:
            print(f"No matching graph file found for cylinder file: {cylinder_filename}")
    
    return cylinder_dataframes


def format_cylinder_dataframes(cylinder_dataframes):
    """Add cylinder order columns to each cylinder dataframe in the list."""
    for cylinder_filename, cylinder_df in cylinder_dataframes:
        # Initialize the new columns
        cylinder_df['cylinder_order_in_segment'] = 0
        cylinder_df['inverse_cylinder_order_in_segment'] = 0

        # Group by the segment to order cylinders within each segment
        for segment_id, group in cylinder_df.groupby('segment'):
            # Sort by branch and then by PositionInBranch (order within the branch)
            group = group.sort_values(by=['branch', 'PositionInBranch'])

            # Assign the cylinder_order_in_segment
            group['cylinder_order_in_segment'] = range(len(group))

            # Assign the inverse_cylinder_order_in_segment
            group['inverse_cylinder_order_in_segment'] = group['cylinder_order_in_segment'].max() - group['cylinder_order_in_segment']

            # Update the original dataframe with these new values
            cylinder_df.loc[group.index, 'cylinder_order_in_segment'] = group['cylinder_order_in_segment']
            cylinder_df.loc[group.index, 'inverse_cylinder_order_in_segment'] = group['inverse_cylinder_order_in_segment']
    return cylinder_dataframes


def save_cylinder_dfs(cylinder_dataframes, output_folder_cylinders):
    if not os.path.exists(output_folder_cylinders):
        os.makedirs(output_folder_cylinders)
    for cylinder_filename, cylinder_df in cylinder_dataframes:
        # Save the updated cylinder dataframe
        output_path = os.path.join(output_folder_cylinders, cylinder_filename)
        cylinder_df.to_csv(output_path, index=False)


def calculate_and_add_inclination_angle(df): 
    #NOT USED!
    # Determine the new start and end points based on the lowest Z value
    new_startX = np.where(df['startZ'] <= df['endZ'], df['startX'], df['endX'])
    new_startY = np.where(df['startZ'] <= df['endZ'], df['startY'], df['endY'])
    new_startZ = np.where(df['startZ'] <= df['endZ'], df['startZ'], df['endZ'])

    new_endX = np.where(df['startZ'] <= df['endZ'], df['endX'], df['startX'])
    new_endY = np.where(df['startZ'] <= df['endZ'], df['endY'], df['startY'])
    new_endZ = np.where(df['startZ'] <= df['endZ'], df['endZ'], df['startZ'])

    # Calculate the vector components
    deltaX = new_endX - new_startX
    deltaY = new_endY - new_startY
    deltaZ = new_endZ - new_startZ

    # Calculate the horizontal magnitude
    horizontal_magnitude = np.sqrt(deltaX**2 + deltaY**2)

    # Calculate the inclination angle in degrees
    angle = np.degrees(np.arctan2(deltaZ, horizontal_magnitude))

    # Ensure the angle is within 0 to 90 degrees
    df['angle'] = np.clip(angle, 0, 90)

    return df

def format_elm_templates(cylinder_dataframes, output_folder_templates):
    """Create ELM templates from cylinder dataframes."""


    elm_templates = []

    for cylinder_filename, cylinder_df in cylinder_dataframes:
        tree_size, tree_id = filename_dict[cylinder_filename]
        
        # Initialize the new template dataframe
        elm_df = pd.DataFrame({
            'Tree.ID': [tree_id] * len(cylinder_df),
            'Size': [tree_size] * len(cylinder_df),
            'Filename' : [cylinder_filename] * len(cylinder_df),
            'isPrecolonial': [False] * len(cylinder_df),
            'startX': cylinder_df['start_1'],
            'startY': cylinder_df['start_2'],
            'startZ': cylinder_df['start_3'],
            'endX': cylinder_df['start_1'] + cylinder_df['axis_1'],
            'endY': cylinder_df['start_2'] + cylinder_df['axis_2'],
            'endZ': cylinder_df['start_3'] + cylinder_df['axis_3'],
            'axisX' : cylinder_df['axis_1'],
            'axisY' : cylinder_df['axis_2'],
            'axisZ' : cylinder_df['axis_3'],
            'radius': cylinder_df['radius'],
            'length': cylinder_df['length'],
            'angle': cylinder_df['angle'],
            'diameter': cylinder_df['diameter'],
            'volume': cylinder_df['volume'],
            'area': cylinder_df['area'],
            'cylinder_ID': cylinder_df.index + 1,  # Assuming branch ID is just the row number
            'branch': cylinder_df['branch'],
            'segment': cylinder_df['segment'],
            'cylinder_order_in_segment': cylinder_df['cylinder_order_in_segment'],
            'inverse_cylinder_order_in_segment': cylinder_df['inverse_cylinder_order_in_segment'],
            'BranchOrder': cylinder_df['BranchOrder'],  # Corrected column name
            'PositionInBranch': cylinder_df['PositionInBranch'],
            'resource' : ['other'] * len(cylinder_df)
        })

        elm_templates.append((cylinder_filename, elm_df))
    
    return elm_templates

import trimesh

def additional_qsm_points(elm_df, resolution=0.1):
    # Step 1: Filter based on radius > 0.25 or length > 0.25
    filtered_df = elm_df[(elm_df['radius'] > 0.25) | (elm_df['length'] > 0.25)].copy()

    # Initialize the debug logs
    tree_id = elm_df['Tree.ID'].iloc[0]
    original_filename = elm_df['Filename'].iloc[0]
    original_point_count = len(elm_df)
    additional_points = []

    print(f"Processing Tree ID: {tree_id}, Original Filename: {original_filename}")
    print(f"Original number of points: {original_point_count}")
    print(f"Number of points detected for additional sampling: {len(filtered_df)}")

    # Step 2: Iterate through each filtered point to create cylinder meshes and sample them
    for _, row in filtered_df.iterrows():
        start_point = np.array([row['startX'], row['startY'], row['startZ']])
        axis_vector = np.array([row['axisX'], row['axisY'], row['axisZ']])
        radius = row['radius']

        # Calculate length of the cylinder and normalize the direction vector
        length = np.linalg.norm(axis_vector)
        direction = axis_vector / length
        
        # Create cylinder mesh without caps
        cylinder = trimesh.creation.cylinder(radius=radius, height=length, cap_ends=False)
        
        # Align and position the cylinder
        cylinder.apply_translation([0, 0, length / 2])
        rotation_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
        cylinder.apply_transform(rotation_matrix)
        cylinder.apply_translation(start_point)
        
        # Calculate the number of samples based on the cylinder's surface area and the desired resolution
        surface_area = cylinder.area
        num_samples = int(surface_area / (resolution ** 2))
        
        # Step 3: Sample the surface of the cylinder at the specified resolution
        sampled_points = cylinder.sample(num_samples)

        # Step 4: Create DataFrame for sampled points and replicate all original columns
        additional_df = pd.DataFrame(sampled_points, columns=['startX', 'startY', 'startZ'])
        
        # Copy all other columns from the original row
        for col in row.index:
            if col not in ['startX', 'startY', 'startZ', 'axis_1', 'axis_2', 'axis_3']:
                additional_df[col] = row[col]

        additional_points.append(additional_df)

    # Step 5: Concatenate additional points with the original dataframe
    if additional_points:
        additional_df = pd.concat(additional_points, ignore_index=True)
        added_point_count = len(additional_df)
        elm_df = pd.concat([elm_df, additional_df], ignore_index=True)
        print(f"Number of additional points added: {added_point_count}")
        print(f"Total number of points after addition: {len(elm_df)}\n")
    else:
        print("No additional points added.\n")

    return elm_df



def voxelize_point_cloud(point_cloud, voxel_size=0.1):
    """Voxelize the point cloud by rounding each point to the nearest voxel center."""
    voxel_grid = np.floor(point_cloud / voxel_size).astype(int)
    unique_voxels, indices = np.unique(voxel_grid, axis=0, return_index=True)
    voxel_centers = unique_voxels * voxel_size + voxel_size / 2  # Get back to the center of the voxel
    return voxel_centers, indices

def load_point_cloud(file_path):
    """Load the point cloud from a text file."""
    point_cloud = np.loadtxt(file_path, delimiter=' ')
    print(f"Loaded point cloud from {os.path.basename(file_path)} with shape {point_cloud.shape}")
    return point_cloud

def match_voxels_to_template(voxel_centers, elm_df):
    """Match voxel centers to the nearest points in the ELM template using KDTree."""
    kdtree = KDTree(elm_df[['startX', 'startY', 'startZ']].values)
    distances, nearest_indices = kdtree.query(voxel_centers)

    distThreshold = 1
    
    # Print distances over 1 unit
    over_threshold = distances > distThreshold
    if np.any(over_threshold):
        print(f"Distances over {distThreshold} unit found:")
        for i, distance in enumerate(distances):
            if distance > 10:
                print(f"Voxel {i}: Distance = {distance}, Nearest Index = {nearest_indices[i]}")
    
    return nearest_indices


def create_elm_point_template(voxel_centers, elm_df, nearest_indices):
    """Create a DataFrame for the ELM point template with matched data."""
    elm_point_template = pd.DataFrame(voxel_centers, columns=['X', 'Y', 'Z'])
    for column in elm_df.columns:
        if column not in ['startX', 'startY', 'startZ', 'endX', 'endY', 'endZ']:
            elm_point_template[column] = elm_df.iloc[nearest_indices][column].values
    return elm_point_template

def save_elm_point_template(elm_point_template, tree_id, output_folder):
    """Save the ELM point template to a CSV file."""
    output_path = os.path.join(output_folder, f'elm_point_template_{tree_id}.csv')
    elm_point_template.to_csv(output_path, sep=';', index=False)
    print(f"Saved ELM point template to {output_path}")


def save_elm_templates(elmTemplate_dfs, output_folder_templates, output_folder_debug):
    """Save the ELM template dataframes and create debug point clouds with RGB initialized to 0,0,0."""
    if not os.path.exists(output_folder_templates):
        os.makedirs(output_folder_templates)
    
    if not os.path.exists(output_folder_debug):
        os.makedirs(output_folder_debug)

    for cylinder_filename, elm_df in elmTemplate_dfs:
        tree_id = elm_df['Tree.ID'].iloc[0]  # Get the Tree ID to use in the filename
        
        # Save the full ELM template CSV
        output_filename = f'elm-{tree_id}.csv'
        output_path = os.path.join(output_folder_templates, output_filename)
        elm_df.to_csv(output_path, sep=';', index=False)  # Use ';' as the delimiter

        # Create the debug point cloud DataFrame
        debug_point_cloud = elm_df[['startX', 'startY', 'startZ']].copy()
        debug_point_cloud.columns = ['X', 'Y', 'Z']
        
        # Initialize RGB columns to 0, 0, 0
        debug_point_cloud['R'] = 0
        debug_point_cloud['G'] = 0
        debug_point_cloud['B'] = 0
        
        # Save the debug point cloud as a CSV
        output_filename = f'elm_debug_point_cloud_{tree_id}.csv'
        output_path = os.path.join(output_folder_debug, output_filename)
        debug_point_cloud.to_csv(output_path, index=False)
        print(f"Saved debug point cloud to {output_path}")


        
def plot_segments(graph_dfs, cylinder_dfs):
    """Plot branches and cylinders X and Z coordinates according to their segment iteratively."""
    
    # Iterate through each graph-cylinder pair and plot them separately
    for (graph_filename, graph_df), (cylinder_filename, cylinder_df) in zip(graph_dfs, cylinder_dfs):
        plt.figure(figsize=(10, 8))
        
        # Plot cylinders
        plt.scatter(cylinder_df['start_1'], cylinder_df['start_3'], c=cylinder_df['segment'], cmap='tab20', s=10, alpha=0.7)
        plt.xlabel('Start X')
        plt.ylabel('Start Z')
        plt.title(f'Cylinders (File: {cylinder_filename})')
        plt.colorbar(label='Segment (Cylinder)')
        plt.show()

def voxelize_point_cloud(point_cloud, voxel_size=0.1):
    """Voxelize the point cloud by rounding each point to the nearest voxel center."""
    voxel_grid = np.floor(point_cloud[:, :3] / voxel_size).astype(int)  # Only consider the first three columns
    unique_voxels, indices = np.unique(voxel_grid, axis=0, return_index=True)
    voxel_centers = unique_voxels * voxel_size + voxel_size / 2  # Get back to the center of the voxel
    return voxel_centers, indices

def visualize_point_cloud(elm_point_template):
    """Visualize the point cloud using PyVista."""
    plotter = pv.Plotter()
    point_cloud_polydata = pv.PolyData(elm_point_template[['X', 'Y', 'Z']].values)
    plotter.add_mesh(point_cloud_polydata, color='cyan', point_size=5, render_points_as_spheres=True)
    plotter.show()

def pointclouds(elmTemplates_dfs, input_folder, output_folder, debug_voxel_folder, voxel_size=0.1):

    # Check centers for both elmTemplate and pointcloud
    for cylinder_filename, elm_df in elmTemplates_dfs:
        tree_id = elm_df['Tree.ID'].iloc[0]
        filename = elm_df['Filename'].iloc[0]

        point_cloud_filename = point_cloud_filenames[tree_id]
        point_cloud_path = os.path.join(input_folder, point_cloud_filename)
        
        # Load and voxelize the point cloud
        point_cloud = load_point_cloud(point_cloud_path)

        print(f'Checking tree ID {tree_id}...')
        print(f'QSM file name is {filename}')
        print(f'Point cloud filename is {point_cloud_filename}')

        # Calculate center point of point_cloud using only the first 3 columns (X, Y, Z)
        cloud_centre = np.mean(point_cloud[:, :3], axis=0)
        
        # Calculate center point of elm_df
        qsm_centre = np.mean(elm_df[['startX', 'startY', 'startZ']].values, axis=0)
        
        print(f'Centre of point cloud: {cloud_centre}')
        print(f'Centre of QSM: {qsm_centre}')
        print(f'Difference is: {cloud_centre - qsm_centre}\n')

    # The rest of the function remains unchanged...

    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(debug_voxel_folder):
        os.makedirs(debug_voxel_folder)

    for cylinder_filename, elm_df in elmTemplates_dfs:
        tree_id = elm_df['Tree.ID'].iloc[0]
        point_cloud_filename = point_cloud_filenames[tree_id]
        point_cloud_path = os.path.join(input_folder, point_cloud_filename)
        
        # Load and voxelize the point cloud
        point_cloud = load_point_cloud(point_cloud_path)
        voxel_centers, _ = voxelize_point_cloud(point_cloud, voxel_size)
        print(f"Voxelized point cloud to {len(voxel_centers)} unique voxels")

        # Save the voxel as a debug point cloud
        debug_voxel_cloud = pd.DataFrame(voxel_centers, columns=['X', 'Y', 'Z'])
        
        # Initialize RGB columns to 0, 0, 0
        debug_voxel_cloud['R'] = 0
        debug_voxel_cloud['G'] = 0
        debug_voxel_cloud['B'] = 0

        # Save the debug voxel cloud to the debug_voxel_folder
        debug_filename = f'voxel-debug-{tree_id}.csv'
        debug_path = os.path.join(debug_voxel_folder, debug_filename)
        debug_voxel_cloud.to_csv(debug_path, sep=',', index=False)
        print(f"Saved debug voxel cloud to {debug_path}")


        # Before matching voxels to the template
        elm_df = additional_qsm_points(elm_df)

        # Match voxels to the template
        nearest_indices = match_voxels_to_template(voxel_centers, elm_df)
        
        # Create and save the ELM point template
        elm_point_template = create_elm_point_template(voxel_centers, elm_df, nearest_indices)
        save_elm_point_template(elm_point_template, tree_id, output_folder)
        
        # Visualize the point cloud
        # visualize_point_cloud(elm_point_template)




def save_debug_point_cloud(elmTemplates_dfs, input_folder, output_folder, voxel_size=0.1):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for cylinder_filename, elm_df in elmTemplates_dfs:
        tree_id = elm_df['Tree.ID'].iloc[0]
        point_cloud_filename = point_cloud_filenames[tree_id]
        point_cloud_path = os.path.join(input_folder, point_cloud_filename)
        
        # Load and voxelize the point cloud
        point_cloud = load_point_cloud(point_cloud_path)
        voxel_centers, _ = voxelize_point_cloud(point_cloud, voxel_size)
        print(f"Voxelized point cloud to {len(voxel_centers)} unique voxels")

        # Create and save the debug point cloud with the specified naming convention
        debug_point_cloud = pd.DataFrame(voxel_centers, columns=['X', 'Y', 'Z'])
        debug_filename = f'debug-elmscan-{tree_id}.csv'
        debug_path = os.path.join(output_folder, debug_filename)
        debug_point_cloud.to_csv(debug_path, sep=',', index=False)
        print(f"Saved debug point cloud to {debug_path}")



# GRAPH BASED ANALYSIS
input_folder = 'data/treeInputs/trunks-elms/orig/csvs-orig/branch'
output_folder ='data/treeInputs/trunks-elms/orig/csvs-clustered/branch'
root_order = 2  # Change this to the desired order level for clustering
graphs = cluster_branches_from_graph_csvs(input_folder, output_folder, root_order)

# CYLINDER BASED ANALYSIS
input_folder_cylinders = 'data/treeInputs/trunks-elms/orig/csvs-orig/cylinder'
output_folder_cylinders ='data/treeInputs/trunks-elms/orig/csvs-clustered/cylinder'

# Transfer the graph-based info to the cylinder dataframes
cylinder_dfs = transfer_graph_to_cylinders(graphs, input_folder_cylinders)
cylinder_dfs = format_cylinder_dataframes(cylinder_dfs)
save_cylinder_dfs(cylinder_dfs, output_folder_cylinders)

# Plot branches and cylinders
#plot_segments(graphs, cylinder_dfs)

# ELM TEMPLATES
output_folder_templates = 'data/treeInputs/trunks-elms/orig/initial_templates'
output_folder_debug_templates = 'data/treeInputs/trunks-elms/orig/debug/df'
debug_voxel_folder = 'data/treeInputs/trunks-elms/debug/voxel'
elmTemplate_dfs = format_elm_templates(cylinder_dfs, output_folder_templates)
save_elm_templates(elmTemplate_dfs, output_folder_templates,output_folder_debug_templates)

# POINT TEMPLATES
input_point_folder = 'data/treeInputs/trunks-elms/orig/trunk_point_clouds'
output_point_folder ='data/treeInputs/trunks-elms/initial templates'
pointclouds(elmTemplate_dfs, input_point_folder, output_point_folder, debug_voxel_folder)

# DEBUG POINT CLOUDS FROM ELM TEMPLATES
output_folder_debug_points = 'data/treeInputs/trunks-elms/orig/debug/cloud'
#save_debug_template_cloud(elmTemplate_dfs, output_folder_debug_templates, delimiter=',')


