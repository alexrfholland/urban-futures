import pandas as pd
from scipy.spatial import cKDTree
import os
import trimesh
import point_cloud_utils as pcu
import pyvista as pv

def sample_mesh_blue_noise(vertices, faces, radius):
    fid, bc = pcu.sample_mesh_poisson_disk(vertices, faces, num_samples=-1, radius=radius)
    blue_noise_samples = pcu.interpolate_barycentric_coords(faces, fid, bc, vertices)
    return blue_noise_samples


def load_obj_vertices_to_df(obj_file):
    """
    Load an OBJ file using trimesh and extract vertex positions into a DataFrame.
    
    :param obj_file: Path to the OBJ file
    :return: DataFrame with vertex positions (x, y, z)
    """
    # Load the OBJ file using trimesh
    mesh = trimesh.load(obj_file)
    
    # Extract vertices
    vertices = mesh.vertices  # Numpy array of shape (n, 3)
    
    # Create a DataFrame with vertex positions
    vertices_df = pd.DataFrame(vertices, columns=['x', 'y', 'z'])
    
    return mesh, vertices_df

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

def load_csv_to_df(filepath):
    """
    Load a CSV file into a Pandas DataFrame.
    
    :param filepath: Path to the CSV file
    :return: Pandas DataFrame
    """
    return pd.read_csv(filepath)

def build_ckdtree(df, x_col='startx', y_col='starty', z_col='startz'):
    """
    Build a cKDTree from a DataFrame using specified coordinate columns.
    
    :param df: DataFrame containing coordinates
    :param x_col, y_col, z_col: Columns for coordinates
    :return: cKDTree object
    """
    print(df)
    points = df[[x_col, y_col, z_col]].values
    return cKDTree(points)

def find_nearest_indices(tree, points):
    """
    Find nearest neighbor indices for each point in the given KDTree.
    
    :param tree: cKDTree object
    :param points: Array of points to query
    :return: List of nearest neighbor indices
    """
    _, indices = tree.query(points)
    return indices

def transfer_columns(source_df, target_df, indices, cols):
    """
    Transfer specified columns from source_df to target_df using nearest neighbor indices.
    
    :param source_df: DataFrame with source data
    :param target_df: DataFrame to receive the data
    :param indices: Indices of nearest rows in source_df
    :param cols: List of columns to transfer
    :return: Modified target_df
    """
    for col in cols:
        target_df[col] = source_df.iloc[indices][col].values
    return target_df

#if __name__ == "__main__":
folderPath = 'data/revised/lidar scans/elm/adtree'
point_cloud_files = {
    "Small A_skeleton.ply": 4,
    "Small B_skeleton.ply": 5,
    "Small C_skeleton.ply": 6,
    "Med A 1 mil_skeleton.ply": 1,
    "Med B 1 mil_skeleton.ply": 2,
    "Med C 1 mil_skeleton.ply": 3,
    "ElmL1_skeleton.ply": 7,
    "Elm L3_skeleton.ply": 9,
    "Elm L4_skeleton.ply": 10,
    "Elm L5_skeleton.ply": 11,
    "Large Elm A 1 mil_skeleton.ply": 12,
    "Large Elm B - 1 mil_skeleton.ply": 13,
    "Large Elm C 1 mil_skeleton.ply": 14
}

fileNameDic = create_tree_id_filename_dict(point_cloud_files)

selectedTreeIDs = [12]
selected_fileNameDic = {tree_id: filename for tree_id, filename in fileNameDic.items() if tree_id in selectedTreeIDs}
#fileNameDic = selected_fileNameDic

for tree_id, filename in fileNameDic.items():
    print(f"Processing tree ID: {tree_id}, filename: {filename}")
    
    #qsmFileName = f'{folderPath}/QSMs/{filename}_treeDF.csv'
    qsmFileName = f'{folderPath}/processedQSMs/{filename}_clusteredQSM.csv'
    invalidQSMFileName = f'{folderPath}/invalidQSMs/{filename}_invalidQSM.csv'

    objFileName = f'{folderPath}/{filename}_branches.obj'

    print(f"Loading OBJ file: {objFileName}")
    mesh, verticesDF = load_obj_vertices_to_df(objFileName)
    print(f"Vertices DataFrame created. Shape: {verticesDF.shape}")

    print("Sampling mesh for blue noise...")
    blue_noise_samples = sample_mesh_blue_noise(mesh.vertices, mesh.faces, .01)
    print(f"Blue noise samples created. Shape: {blue_noise_samples.shape}")

    blue_noise_df = pd.DataFrame(blue_noise_samples, columns=['x', 'y', 'z'])
    print(f"Blue noise DataFrame created. Shape: {blue_noise_df.shape}")

    output_csv_path = f'{folderPath}/elmBlueNoiseSamples/{filename}_blueNoiseDF.csv'
    print(f"Saving blue noise DataFrame to: {output_csv_path}")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    blue_noise_df.to_csv(output_csv_path, index=False)
    print(f"Saved {filename} blue noise DataFrame successfully")
    print(f"Processed {filename} and saved successfully")
    print("--------------------")

