import numpy as np
import pandas as pd
import pyvista as pv

# Function to extract vertices and edges from a PLY file
def extract_ply_data(ply_file_path):
    """
    Extract vertex and edge data from a PLY file.
    
    Args:
    - ply_file_path (str): Path to the PLY file.

    Returns:
    - vertices (numpy array): Array of vertex data (x, y, z, radius).
    - edges (numpy array): Array of edge data (start_idx, end_idx).
    """
    # Open the file and read its content
    with open(ply_file_path, 'rb') as f:
        content = f.read()
    
    # Locate the end of the header
    header_end = content.index(b'end_header') + len('end_header')
    binary_data = content[header_end:].strip()

    # Define vertex and edge data types
    vertices_dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('radius', np.float32)]
    edges_dtype = [('start_idx', np.int32), ('end_idx', np.int32)]

    # Read vertex count and edge count from the header
    vertex_count = 671580  # Retrieved from the header
    edge_count = 572054  # Retrieved from the header

    # Extract vertex data
    vertex_data = np.frombuffer(binary_data[:vertex_count * np.dtype(vertices_dtype).itemsize], dtype=vertices_dtype)

    # Extract edge data
    edge_data_start = vertex_count * np.dtype(vertices_dtype).itemsize
    edge_data = np.frombuffer(binary_data[edge_data_start:], dtype=edges_dtype)

    return vertex_data, edge_data

# Function to calculate lengths and angles in a vectorized manner
def calculate_lengths_and_angles(vertices, edges):
    """
    Calculate lengths and angles between edges in a vectorized manner.
    
    Args:
    - vertices (numpy array): Array of vertex data.
    - edges (numpy array): Array of edge data.

    Returns:
    - lengths (numpy array): Array of edge lengths.
    - angles (numpy array): Array of edge angles in degrees.
    """
    # Extract start and end vertices using the edges indices
    start_vertices = vertices[edges['start_idx']]
    end_vertices = vertices[edges['end_idx']]
    
    # Calculate lengths (Euclidean distance between start and end points)
    lengths = np.sqrt((end_vertices['x'] - start_vertices['x'])**2 +
                      (end_vertices['y'] - start_vertices['y'])**2 +
                      (end_vertices['z'] - start_vertices['z'])**2)
    
    # Calculate horizontal distances (ignoring z-axis)
    horizontal_dists = np.sqrt((end_vertices['x'] - start_vertices['x'])**2 +
                               (end_vertices['y'] - start_vertices['y'])**2)
    
    # Calculate vertical distances (z-axis difference)
    vertical_dists = np.abs(end_vertices['z'] - start_vertices['z'])
    
    # Calculate angles using arctan2 (convert from radians to degrees)
    angles = np.degrees(np.arctan2(vertical_dists, horizontal_dists))

    return lengths, angles

# Function to create a DataFrame with all necessary information
def create_edge_dataframe(vertices, edges, lengths, angles):
    """
    Create a pandas DataFrame with start/end coordinates, lengths, and angles.
    
    Args:
    - vertices (numpy array): Array of vertex data.
    - edges (numpy array): Array of edge data.
    - lengths (numpy array): Array of edge lengths.
    - angles (numpy array): Array of edge angles.

    Returns:
    - DataFrame: Pandas DataFrame with edge data, lengths, and angles.
    """
    df = pd.DataFrame({
        'start_idx': edges['start_idx'],
        'end_idx': edges['end_idx'],
        'startx': vertices[edges['start_idx']]['x'],
        'starty': vertices[edges['start_idx']]['y'],
        'startz': vertices[edges['start_idx']]['z'],
        'endx': vertices[edges['end_idx']]['x'],
        'endy': vertices[edges['end_idx']]['y'],
        'endz': vertices[edges['end_idx']]['z'],
        'length': lengths,
        'angle': angles
    })
    return df

# Main function to encapsulate the entire process
def process_ply_file(ply_file_path):
    """
    Process the PLY file to extract vertex/edge data and calculate lengths and angles.
    
    Args:
    - ply_file_path (str): Path to the PLY file.
    
    Returns:
    - DataFrame: Pandas DataFrame with start/end points, lengths, and angles.
    """
    # Step 1: Extract vertex and edge data from the PLY file
    vertices, edges = extract_ply_data(ply_file_path)
    
    # Step 2: Calculate lengths and angles
    lengths, angles = calculate_lengths_and_angles(vertices, edges)
    
    # Step 3: Create a DataFrame with all information
    df = create_edge_dataframe(vertices, edges, lengths, angles)

    
    
    # Step 4: Display the DataFrame to the user

    df.to_csv('data/revised/elm point clouds/Tree A again_skeleton_df.csv', index=False)
    
    return df

def plot_connected_lines(df):
    """
    Create and plot connected lines in PyVista from start and end positions in a DataFrame,
    optimized for large datasets.
    
    Args:
    - df (pandas.DataFrame): DataFrame containing 'startx', 'starty', 'startz', 'endx', 'endy', 'endz' columns.
    
    Returns:
    - None: Displays the plot.
    """
    # Extract start and end points
    starts = df[['startx', 'starty', 'startz']].values
    ends = df[['endx', 'endy', 'endz']].values
    
    # Create a single array of points
    points = np.vstack((starts, ends))
    
    # Create lines array
    lines = np.column_stack((
        np.full(len(df), 2),  # Each line has 2 points
        np.arange(0, len(df)),  # Start indices
        np.arange(len(df), 2*len(df))  # End indices
    )).ravel()
    
    # Create PyVista PolyData object
    poly = pv.PolyData(points, lines=lines)
    
    # Create a plotter
    plotter = pv.Plotter()
    
    # Add the lines to the plotter
    plotter.add_mesh(poly, color='white', line_width=1)
    
    # Set a black background for better visibility
    plotter.set_background('black')
    
    # Enable anti-aliasing for smoother lines
    plotter.enable_anti_aliasing()
    
    # Show the plot
    plotter.show()


# Path to the uploaded PLY file
ply_file_path = ply_file_path = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/elm point clouds/Tree A again_skeleton.ply'

# Run the process on the PLY file and get the DataFrame
edge_df = process_ply_file(ply_file_path)

short_edges = edge_df[edge_df['length'] < 1]


plot_connected_lines(short_edges)

# Preview the first few rows
print(edge_df.head())
