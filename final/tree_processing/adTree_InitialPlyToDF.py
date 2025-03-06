
"""
This PLY file describes the structure of a tree skeleton, represented by a series of connected points (vertices) and edges. Here’s a summary of its key elements based on what we've extracted so far:

Structure of the PLY file:
Vertices:
Each vertex in the file has four properties:
x, y, z coordinates: These define the position of the point in 3D space.
radius: This represents the radius of the branch or cylinder at that vertex.
The vertices are a sequence of connected points that define the structure of the tree, where each point represents a location along a branch or the tree's trunk.

Edges:
Each edge is prefaced by a uint32 value (typically 2) indicating the number of vertices it connects.
These edges represent the connections between the vertices, forming the "branches" of the tree.
The connectivity between vertices forms a topological description of the tree, where some vertices are connected to multiple others (branching points), and others terminate (leaf or branch tips).
Key Information:
Branching structure: The edges connect vertices in a way that defines the branching structure of the tree. Some edges represent branches off the main trunk, while others might represent sub-branches or smaller twigs.
Connectivity: The start_idx and end_idx from the edge data define the parent-child relationships between vertices, allowing us to build a hierarchy of branches.
Root identification: The logic of the script identifies the root (main trunk) by detecting branches without a parent (parentID = -1).
Radius data: The radius property describes the thickness of the branch at each vertex, which can vary along the tree's structure.
File Contents Summary:
Vertices: Each vertex is described by 4 floats (x, y, z, radius).
Edges: Each edge connects two vertices (via start_idx and end_idx), representing the skeletal structure of the tree.
This PLY file essentially encodes a 3D tree model as a graph, where the vertices represent points on the tree's structure and the edges represent the connections (branches) between them.

# Binary data structure
binary_data = [
    # Vertex data (671580 vertices)
    [
        float32(x),
        float32(y),
        float32(z),
        float32(radius)
    ] * 671580,

    # Edge data (572054 edges)
    [
        uint32(2),  # Number of vertex indices (always 2 for edges)
        int32(vertex_index_1),
        int32(vertex_index_2)
    ] * 572054
]

"""

import numpy as np
import pandas as pd
import pyvista as pv
import os

def parse_ply_header(ply_file_path):
    vertex_count = 0
    edge_count = 0
    header_end = 0
    with open(ply_file_path, 'rb') as f:
        for line in f:
            header_end += len(line)
            line = line.decode('ascii').strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('element edge'):
                edge_count = int(line.split()[-1])
            elif line == 'end_header':
                break
    return vertex_count, edge_count, header_end

def process_ply_file(ply_file_path):
    vertex_count, edge_count, header_end = parse_ply_header(ply_file_path)
    
    print(f"File contains {vertex_count} vertices and {edge_count} edges.")
    print(f"Header ends at byte {header_end}")

    with open(ply_file_path, 'rb') as f:
        # Skip the header
        f.seek(header_end)
        
        # Read vertices
        vertex_data = np.fromfile(f, dtype=np.float32, count=vertex_count*4)
        if len(vertex_data) != vertex_count * 4:
            print(f"Warning: Expected {vertex_count*4} floats for vertices, but read {len(vertex_data)}")
        vertices = vertex_data.reshape(-1, 4)

        # Read edges
        edges = []
        for _ in range(edge_count):
            num_vertices = np.fromfile(f, dtype=np.uint32, count=1)[0]
            if num_vertices != 2:
                print(f"Warning: Expected 2 vertices per edge, but got {num_vertices}")
            edge = np.fromfile(f, dtype=np.int32, count=2)
            edges.append(tuple(edge))

    edges = np.array(edges)

    # Additional checks on edge data
    print("\nEdge data statistics:")
    print(f"Min edge index: {edges.min()}")
    print(f"Max edge index: {edges.max()}")
    print(f"Number of unique vertices in edges: {np.unique(edges).size}")
    
    # Check for self-loops
    self_loops = np.sum(edges[:, 0] == edges[:, 1])
    print(f"Number of self-loops (edges connecting a vertex to itself): {self_loops}")

    # Check for frequency of each vertex in edges
    unique, counts = np.unique(edges, return_counts=True)
    most_common = unique[np.argsort(-counts)][:10]
    print("\nTop 10 most frequent vertices in edges:")
    for vertex, count in zip(most_common, counts[np.argsort(-counts)][:10]):
        print(f"Vertex {vertex}: {count} occurrences")

    # Print the first 50 edges for inspection
    print("\nFirst 50 edges:")
    print(edges[:50])

    return vertices, edges

def calculate_angles_vectorized(start_points, end_points):
    """
    Calculate the angles of lines relative to the horizontal plane using vectorized operations.
    
    Args:
    - start_points (numpy array): Array of start point coordinates [x, y, z]
    - end_points (numpy array): Array of end point coordinates [x, y, z]
    
    Returns:
    - numpy array: Angles in degrees
    """
    # Calculate the vectors from start to end
    vectors = end_points - start_points
    
    # Calculate the projections of the vectors onto the horizontal plane
    horizontal_projections = vectors.copy()
    horizontal_projections[:, 2] = 0  # Set z-component to zero
    
    # Calculate the dot products
    dot_products = np.sum(vectors * horizontal_projections, axis=1)
    
    # Calculate the magnitudes
    vector_magnitudes = np.linalg.norm(vectors, axis=1)
    projection_magnitudes = np.linalg.norm(horizontal_projections, axis=1)
    
    # Calculate the cosine of the angles
    cos_angles = dot_products / (vector_magnitudes * projection_magnitudes)
    
    # Handle potential numerical instability
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    
    # Calculate the angles in radians and convert to degrees
    angles_deg = np.degrees(np.arccos(cos_angles))
    
    # Handle vertical lines (where projection_magnitudes is zero)
    vertical_mask = projection_magnitudes == 0
    angles_deg[vertical_mask] = 90.0
    
    return angles_deg

def assign_branch_ids_and_parents_vectorized(df):
    """
    Assign branch IDs and parent branch IDs to each edge in the DataFrame using simple vectorized operations.
    
    Args:
    - df (pandas.DataFrame): DataFrame containing 'start_idx' and 'end_idx' columns.
    
    Returns:
    - pandas.DataFrame: Updated DataFrame with 'branch_id' and 'parent_branch_id' columns.
    """
    # Assign branch IDs as the start_idx of each edge
    df['branch_id'] = df['start_idx']
    
    # Create a mapping from end_idx to start_idx
    end_to_start = dict(zip(df['end_idx'], df['start_idx']))
    
    # Assign parent branch IDs
    df['parent_branch_id'] = df['branch_id'].map(end_to_start).fillna(-1).astype(int)
    
    return df

def create_edge_dataframe(vertices, edges):
    """
    Create a pandas DataFrame with start/end coordinates, radii, and angle.
    
    Args:
    - vertices (numpy array): Array of vertex data.
    - edges (numpy array): Array of edge data.
    
    Returns:
    - DataFrame: Pandas DataFrame with edge data.
    """
    df = pd.DataFrame({
        'start_idx': edges[:, 0],
        'end_idx': edges[:, 1],
        'startx': vertices[edges[:, 0], 0],
        'starty': vertices[edges[:, 0], 1],
        'startz': vertices[edges[:, 0], 2],
        'start_radius': vertices[edges[:, 0], 3],
        'endx': vertices[edges[:, 1], 0],
        'endy': vertices[edges[:, 1], 1],
        'endz': vertices[edges[:, 1], 2],
        'end_radius': vertices[edges[:, 1], 3]
    })
    
    # Calculate the angles using vectorized operations
    start_points = df[['startx', 'starty', 'startz']].values
    end_points = df[['endx', 'endy', 'endz']].values
    df['angle'] = calculate_angles_vectorized(start_points, end_points)
    df = assign_branch_ids_and_parents_vectorized(df)

    #export_csv = df.to_csv('/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/elm point clouds/tree_data.csv', index=False)
    
    return df

def create_polydata_from_df(df):
    """
    Create PyVista PolyData from start and end positions in a DataFrame, 
    with thicker lines colored based on their branch_id.

    Args:
    - df (pandas.DataFrame): DataFrame containing 'startx', 'starty', 'startz', 'endx', 'endy', 'endz', and 'branch_id' columns.

    Returns:
    - pv.PolyData: PolyData object ready for plotting.
    """
    # Extract start and end points
    starts = df[['startx', 'starty', 'startz']].values
    ends = df[['endx', 'endy', 'endz']].values

    # Create a single array of points
    points = np.vstack((starts, ends))

    # Create lines array
    n_lines = len(df)
    lines = np.column_stack((
        np.full(n_lines, 2),  # Each line has 2 points
        np.arange(0, n_lines),  # Start indices
        np.arange(n_lines, 2*n_lines)  # End indices
    )).ravel()

    # Create PyVista PolyData object
    poly = pv.PolyData(points, lines=lines)

    # Add branch_id as scalar data to the PolyData object
    poly['branch_id'] = np.repeat(df['branch_id'].values, 2)
    
    return poly


def plot_all_trees(treeDFs, tree_sizes):
    polyList = []
    
    for tree_id, df in treeDFs.items():
        poly = create_polydata_from_df(df)
        polyList.append(poly)

    n_plots = len(polyList)
    n_cols = min(n_plots, 4)
    n_rows = (n_plots - 1) // 4 + 1
    
    p = pv.Plotter(shape=(n_rows, n_cols))
    p.link_views()
    
    for i, poly in enumerate(polyList):
        row = i // 4
        col = i % 4
        p.subplot(row, col)
        p.add_mesh(poly, scalars='branch_id', cmap='turbo', line_width=4)
        p.camera_position = 'xy'
        p.reset_camera()
    p.show()

def plot_connected_lines(df):
    """
    Create and plot connected lines in PyVista from start and end positions in a DataFrame,
    with thicker lines colored based on their branch_id using the 'turbo' color palette.

    Args:
    - df (pandas.DataFrame): DataFrame containing 'startx', 'starty', 'startz', 'endx', 'endy', 'endz', and 'branch_id' columns.

    Returns:
    - None: Displays the plot.
    """
    # Extract start and end points
    starts = df[['startx', 'starty', 'startz']].values
    ends = df[['endx', 'endy', 'endz']].values

    # Create a single array of points
    points = np.vstack((starts, ends))

    # Create lines array
    n_lines = len(df)
    lines = np.column_stack((
        np.full(n_lines, 2),  # Each line has 2 points
        np.arange(0, n_lines),  # Start indices
        np.arange(n_lines, 2*n_lines)  # End indices
    )).ravel()

    # Create PyVista PolyData object
    poly = pv.PolyData(points, lines=lines)

    # Add branch_id as scalar data to the PolyData object
    poly['branch_id'] = np.repeat(df['branch_id'].values, 2)

    # Create a plotter
    plotter = pv.Plotter()

    # Add the lines to the plotter with a color map based on branch_id
    plotter.add_mesh(poly, scalars='branch_id', cmap='turbo', line_width=4, 
                     scalar_bar_args={'title': 'Branch ID'})

    # Set a black background for better visibility
    plotter.set_background('black')

    # Enable anti-aliasing for smoother lines
    plotter.enable_anti_aliasing()

    # Show the plot
    plotter.show()

def print_file_names(directory):
    # List all files and directories in the specified path
    for filename in os.listdir(directory):
        # Check if the current object is a file
        if os.path.isfile(os.path.join(directory, filename)):
            print(filename)


def create_tree_id_filename_dict(point_cloud_files):
    """
    Creates a new dictionary with tree ID as key and processed filename as value.
    
    :param point_cloud_files: Dictionary with filename as key and tree ID as value
    :return: New dictionary with tree ID as key and processed filename as value
    """
    tree_id_filename_dict = {}
    for filename, tree_id in point_cloud_files.items():
        processed_filename = filename.split('_')[0]
        print(f"Tree ID: {tree_id}, Filename: {processed_filename}")
        tree_id_filename_dict[tree_id] = processed_filename
    return tree_id_filename_dict

#if __name__ == "__main__":
# Prompt the user for the processing stage
processing_stage = input("Enter the processing stage (0: generate QSM, 1: process QSM): ")

# Convert input to integer and validate
try:
    processing_stage = int(processing_stage)
    if processing_stage not in [0, 1]:
        raise ValueError
except ValueError:
    print("Invalid input. Please enter either 0 or 1.")
    exit()

# Set the appropriate processing mode based on user input
if processing_stage == 0:
    print("Generating QSM...")
else:
    print("Processing QSM...")

folderPath = 'data/revised/lidar scans/elm/adtree'
point_cloud_files = {
    "Small A_skeleton.ply": 4,
    "Small B_skeleton.ply": 5,
    "Small C_skeleton.ply": 6,
    "Med A 1 mil_skeleton.ply": 1,
    "Med B 1 mil_skeleton.ply": 2,
    "Med C 1 mil_skeleton.ply": 3,
    "ElmL1_skeleton.ply": 7,
    #"Elm L2_skeleton.ply": 8,
    "Elm L3_skeleton.ply": 9,
    "Elm L4_skeleton.ply": 10,
    "Elm L5_skeleton.ply": 11,
    "Large Elm A 1 mil_skeleton.ply": 12,
    "Large Elm B - 1 mil_skeleton.ply": 13,
    "Large Elm C 1 mil_skeleton.ply": 14
}

fileNameDic = create_tree_id_filename_dict(point_cloud_files)




tree_sizes = {
    4: "small",
    5: "small",
    6: "small",
    1: "medium",
    2: "medium",
    3: "medium",
    7: "large",
    #8: "large",
    9: "large",
    10: "large",
    11: "large",
    12: "large",
    13: "large",
    14: "large"
}


selected_tree_ids = [12,13,14]
#get a subset dictory of point_cloud_files matching selected_tree_ids
selected_files = {k: v for k, v in point_cloud_files.items() if v in selected_tree_ids}

print(f'selected files are: {selected_files}')

treeDFs = {}

filePath = 'data/revised/lidar scans/elm/adtree/QSMs'


if processing_stage == 0:
    print("Generating QSM...")

    for filename, tree_id in point_cloud_files.items():
        print(f"Processing {filename}")
        vertices, edges = process_ply_file(os.path.join(folderPath, filename))
        df = create_edge_dataframe(vertices, edges)
        print(f'DataFrame shape: {df.shape}')
        print(df.head())
        #plot_connected_lines(df)
        treeDFs[tree_id] = df

    #save treeDFs to csvs
    for tree_id, df in treeDFs.items():
        df.to_csv(f'{filePath}/{fileNameDic[tree_id]}_treeDF.csv', index=False)

else:
    print("Processing QSM...")
    for tree_id, filename in fileNameDic.items():
        df = pd.read_csv(f'{filePath}/{filename}_treeDF.csv')
        treeDFs[tree_id] = df
        
    plot_all_trees(treeDFs, tree_sizes)
