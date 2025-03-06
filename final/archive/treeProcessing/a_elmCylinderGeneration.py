import os
import trimesh
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import pyvista as pv

def trimesh_to_pyvista(trimesh_mesh):
    """Converts a trimesh mesh to pyvista PolyData."""
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces
    
    # Convert trimesh faces to a format compatible with pyvista (each face preceded by the number of vertices, here 3)
    faces_pyvista = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
    
    # Create a PyVista PolyData object
    polydata = pv.PolyData(vertices, faces_pyvista)
    
    return polydata

def plot_trimesh_list_with_pyvista(trimesh_list, max_columns=3):
    """Plots a list of trimesh meshes in a grid using pyvista."""
    
    # Calculate the number of rows needed based on the number of meshes and max_columns
    num_meshes = len(trimesh_list)
    num_rows = (num_meshes + max_columns - 1) // max_columns  # ceiling division
    
    # Create a PyVista plotter with the specified grid layout
    plotter = pv.Plotter(shape=(num_rows, max_columns))

    # Iterate through the trimesh list and plot each mesh
    for i, trimesh_mesh in enumerate(trimesh_list):
        # Convert trimesh to pyvista PolyData
        polydata = trimesh_to_pyvista(trimesh_mesh)
        
        # Determine row and column in the grid
        row = i // max_columns
        col = i % max_columns
        
        # Add the mesh to the specific subplot in the grid
        plotter.subplot(row, col)
        plotter.add_mesh(polydata, color="lightblue", show_edges=True)
        plotter.add_text(f"Mesh {i+1}", font_size=10)
    
    # Display the plot with all the meshes in the grid
    plotter.show()


def calculate_and_add_inclination_angle(df):
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

    # Calculate the horizontal magnitude (distance in the XY plane)
    horizontal_magnitude = np.sqrt(deltaX**2 + deltaY**2)

    # Calculate the inclination angle in degrees using arctan2
    angle = np.degrees(np.arctan2(deltaZ, horizontal_magnitude))

    # Restrict angle to 0-90 degrees
    df['angle'] = np.clip(angle, 0, 90)

    return df

def create_cylinder_mesh_no_caps(start_point, end_point, radius, resolution=16):
    # Calculate the vector from the start to the endpoint
    vector = end_point - start_point
    length = np.linalg.norm(vector)
    direction = vector / length
    
    # Create a cylinder mesh without caps
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=resolution, cap_ends=False)
    cylinder.apply_translation([0, 0, length / 2])
    rotation_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cylinder.apply_transform(rotation_matrix)
    cylinder.apply_translation(start_point)
    
    return cylinder

def create_mesh_from_csv_no_caps(csv_file):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file, delimiter=';')
    cylinders = []

    # Debug: Print column names and number of rows
    print("Columns in the CSV:", df.columns)
    print(f"Number of rows in the CSV: {len(df)}")

    calculate_and_add_inclination_angle(df)
    print('added angles')

    # Iterate over each row to create a cylinder mesh
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f"Processing row {index}")

        start_point = np.array([row['startX'], row['startY'], row['startZ']])
        end_point = np.array([row['endX'], row['endY'], row['endZ']])
        radius = row['radius']
        
        try:
            cylinder = create_cylinder_mesh_no_caps(start_point, end_point, radius)
        except Exception as e:
            print(f"Error creating cylinder for row {index}: {e}")
            continue
        
        cylinders.append(cylinder)

    # Combine all cylinder meshes into one
    combined_mesh = trimesh.util.concatenate(cylinders)
    
    return combined_mesh, df

def voxelize_mesh(mesh, voxel_size=0.5):
    # Voxelize the mesh with the given voxel size
    voxels = mesh.voxelized(pitch=voxel_size)
    return voxels

def extract_voxel_centroids(voxelized_mesh):
    # Extract the centroids of the voxels
    centroids = voxelized_mesh.points
    return centroids

def match_properties_to_voxels(centroids, original_df):
    # Match the properties from the original DataFrame to the voxel centroids
    original_df.columns = original_df.columns.str.strip()
    original_points = original_df[['startX', 'startY', 'startZ']].values
    tree = KDTree(original_points)
    
    distances, indices = tree.query(centroids)
    
    voxel_properties = pd.DataFrame(centroids, columns=['X', 'Y', 'Z'])

    voxel_properties['resource'] = 'other'

    # Add additional columns from the original DataFrame
    for column in ['Tree.ID', 'Size', 'Filename', 'isPrecolonial', 'startX', 'startY', 'startZ',
                   'endX', 'endY', 'endZ', 'axisX', 'axisY', 'axisZ', 'radius', 'length', 
                   'angle', 'diameter', 'volume', 'area', 'cylinder_ID', 'branch', 'segment', 
                   'cylinder_order_in_segment', 'inverse_cylinder_order_in_segment', 
                   'BranchOrder', 'PositionInBranch', 'resource']:
        if column in original_df.columns:
            voxel_properties[column] = original_df[column].values[indices]
        else:
            print(f"Warning: Column '{column}' not found in the original DataFrame.")
    
    return voxel_properties

def format_euc_template(voxel_properties_df, tree_id):
    # Initialize the Tree.ID column with the tree_id
    voxel_properties_df['Tree.ID'] = tree_id
    
    # Assign treeSize based on Tree.ID
    voxel_properties_df['treeSize'] = pd.cut(voxel_properties_df['Tree.ID'], 
                                             bins=[0, 4, 10, 16], 
                                             labels=['small', 'medium', 'large'], 
                                             right=True)

    # Set isPrecolonial to True for all rows
    voxel_properties_df['isPrecolonial'] = True
    return voxel_properties_df

def process_csv_files(input_folder, output_folder_template, voxel_size=0.1):
    # Ensure the output folder exists
    if not os.path.exists(output_folder_template):
        os.makedirs(output_folder_template)

    trimeshes = []
    
    # Iterate over all CSV files in the input folder
    for csv_file in os.listdir(input_folder):
        if csv_file.endswith(".csv"):
            csv_file_path = os.path.join(input_folder, csv_file)
            print(f"Processing file: {csv_file_path}")
            
            # Extract Tree.ID from the filename
            #tree_id = int(os.path.splitext(csv_file)[0])
            tree_id = -1 #TODO: fix treeID assignment
            print(f"Extracted Tree.ID: {tree_id}")
            
            # Create the mesh from the CSV file
            mesh, original_df = create_mesh_from_csv_no_caps(csv_file_path)
            trimeshes.append(mesh)


            
            """print(f"Original number of points: {len(original_df)}")
            
            # Voxelize the mesh
            voxelized_mesh = voxelize_mesh(mesh, voxel_size=voxel_size)
            centroids = extract_voxel_centroids(voxelized_mesh)
            
            print(f"Voxelized number of points: {len(centroids)}")
            print(f"Voxel resolution: {voxel_size}")
            
            # Match properties to voxel centroids
            voxel_properties_df = match_properties_to_voxels(centroids, original_df)
            
            # Format the voxel properties with treeSize and isPrecolonial columns
            voxel_properties_df = format_euc_template(voxel_properties_df, tree_id)
            
            # Save the formatted DataFrame to the output folder
            output_csv_path = os.path.join(output_folder_template, csv_file)
            voxel_properties_df.to_csv(output_csv_path, sep=';', index=False)
            print(f"Saved formatted Euc template to: {output_csv_path}\n")
            """

    plot_trimesh_list_with_pyvista(trimeshes)

if __name__ == "__main__":
    #input_folder = 'data/treeInputs/trunks/orig'
    #output_folder_template = 'data/treeInputs/trunks/initial_templates'
    
    input_folder = 'data/revised/lidar scans/elm/treeQSM/large/4_templates'
    output_folder_template = 'data/revised/lidar scans/elm/treeQSM/large/5_cylinders'
    
    voxel_size = 0.1
    
    process_csv_files(input_folder, output_folder_template, voxel_size)
