import numpy as np
import pandas as pd
import pyvista as pv

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

#angles not calculating correctly
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
    
    return df


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


if __name__ == "__main__":
    ply_file_path = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/elm point clouds/Tree A again_skeleton.ply'
    
    vertices, edges = process_ply_file(ply_file_path)
    print(f'Vertices shape: {vertices.shape}')
    print(f'Edges shape: {edges.shape}')
    df = create_edge_dataframe(vertices, edges)
    print(f'DataFrame shape: {df.shape}')
    print(df.head())
    plot_connected_lines(df)