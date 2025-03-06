import pickle
import pandas as pd
import pyvista as pv
import numpy as np
import os
import f_resource_meshdecimator
from scipy.spatial import cKDTree

# Function to load the tree templates from a pickle file
def load_tree_templates(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

# Resource specifications for isosurface and point data properties
isoSize = [0.15, 0.15, 0.15]
resource_specs = {
    'perch branch': {'voxelSize': isoSize},
    'peeling bark': {'voxelSize': isoSize},
    'dead branch': {'voxelSize': isoSize},
    'other': {'voxelSize': isoSize},
    'fallen log': {'voxelSize': isoSize},
    'leaf litter': {'voxelSize': [0.25, 0.25, 0.25]},
    'epiphyte': {'voxelSize': [0.3, 0.3, 0.3]},
    'hollow': {'voxelSize': [3, .3, .3]},
    'leaf cluster': {'voxelSize': [.5, .5, .5]}
}

# Create a mapping from resource names to unique integer IDs
resource_mapping = {name: idx for idx, name in enumerate(resource_specs.keys())}
# Create a reverse mapping for reference if needed
reverse_resource_mapping = {idx: name for name, idx in resource_mapping.items()}

# Function to extract isosurface from PolyData
def extract_isosurface_from_polydata(polydata: pv.PolyData, spacing: tuple[float, float, float], resource_name, isovalue: float = 1.0) -> pv.PolyData:
    print(f'{resource_name} polydata has {polydata.n_points}')
    
    if polydata is not None and polydata.n_points > 0:
        points = polydata.points
        x, y, z = points.T
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        dims = (
            int((x_max - x_min) / spacing[0]) + 1,
            int((y_max - y_min) / spacing[1]) + 1,
            int((z_max - z_min) / spacing[2]) + 1
        )

        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=(x_min, y_min, z_min))
        scalars = np.zeros(grid.n_points)

        for px, py, pz in points:
            ix = int((px - x_min) / spacing[0])
            iy = int((py - y_min) / spacing[1])
            iz = int((pz - z_min) / spacing[2])
            grid_idx = ix + iy * dims[0] + iz * dims[0] * dims[1]
            scalars[grid_idx] = 2

        grid.point_data['values'] = scalars
        # Extract the surface from the structured grid at the end
        
        isosurface = grid.contour(isosurfaces=[isovalue], scalars='values', method='flying_edges', compute_normals=True)
        polydata = isosurface.extract_surface()

        #polydata.plot()

        return polydata


    else:
        return None
    

def transfer_point_data(original_df: pd.DataFrame, combined_isosurface: pv.PolyData):
    if original_df is None or original_df.empty:
        print("WARNING: No valid original DataFrame to transfer from.")
        return combined_isosurface

    if combined_isosurface is None or combined_isosurface.n_points == 0:
        print("WARNING: No valid isosurface to transfer onto.")
        return combined_isosurface

    # Ensure the DataFrame contains the 'X', 'Y', 'Z' columns for point coordinates
    if not all(col in original_df.columns for col in ['X', 'Y', 'Z']):
        raise ValueError("The input DataFrame must contain 'X', 'Y', and 'Z' columns.")

    # Build a cKDTree on the original DataFrame's points (X, Y, Z)
    original_points = original_df[['X', 'Y', 'Z']].to_numpy()
    kd_tree = cKDTree(original_points)

    # Find the nearest neighbor indices for each isosurface point
    distances, indices = kd_tree.query(combined_isosurface.points)

    # Transfer the attributes using the nearest neighbor indices
    for data_key in original_df.columns:
        if data_key not in ['X', 'Y', 'Z', 'resource']:  # Skip coordinates
            print(f"Transferring attribute '{data_key}'...")
            # Get the nearest data from the DataFrame based on the nearest neighbor indices
            nearest_data = original_df[data_key].to_numpy()[indices]
            
            # Assign the nearest neighbor data to the isosurface
            combined_isosurface.point_data[data_key] = nearest_data
    
    return combined_isosurface, kd_tree

# Function to process tree templates and create a single VTK file for each tree
def process_tree_and_save_vtk(tree_key, tree_df, output_dir):
    print(f"\nProcessing tree {tree_key}...")

    tree_attributes = {}

    isPreColonial = tree_key[0]
    size = tree_key[1]
    control = tree_key[2]
    treeID = tree_key[3]

    print(f'tree key: isColonial: {isPreColonial}, size: {size}, control: {control}, treeID: {treeID}')
    filename = f'{isPreColonial}_{size}_{control}_{treeID}'



    # Get the unique resources in the current tree's DataFrame
    resources_in_tree = tree_df['resource'].dropna().unique()
    print(f"Resources present in tree {tree_key}: {resources_in_tree}")

    combined_polydata = None

    # Process each unique resource type in the tree
    for resource_name in resources_in_tree:
        # Ensure the resource name exists in the mapping
        if resource_name not in resource_mapping:
            print(f"WARNING: Resource '{resource_name}' not recognized. Skipping.")
            continue

        # Filter the DataFrame for this resource type
        resource_df = tree_df[tree_df['resource'] == resource_name]
        points = resource_df[['X', 'Y', 'Z']].dropna().values
        if points.size == 0:
            print(f"No points found for resource '{resource_name}' in tree {tree_key}. Skipping.")
            continue

        polydata = pv.PolyData(points)

        # Create isosurface using the resource's voxel size
        specs = resource_specs[resource_name]
        isosurface = extract_isosurface_from_polydata(polydata, specs['voxelSize'], resource_name)

        if isosurface is not None:
            # Assign resource ID to point data
            resource_id = resource_mapping[resource_name]
            isosurface.point_data['resource'] = np.full(isosurface.n_points, resource_id, dtype=np.int32)

            # Assign tree attributes to point data
            for attr_name, attr_value in tree_attributes.items():
                # If the attribute is categorical or needs encoding, handle it here
                # For simplicity, assuming all attributes are numerical or can be directly assigned
                if attr_name != 'resource':
                    print(f'transferring resource {attr_name}')
                    isosurface.point_data[attr_name] = np.full(isosurface.n_points, attr_value)

            # Merge the current isosurface into the combined_polydata
            if combined_polydata is None:
                combined_polydata = isosurface.copy()
            else:
                combined_polydata = combined_polydata.merge(isosurface)

            

    print(f'Cleaning the combined mesh...')
    combined_polydata = combined_polydata.clean()
    #combined_polydata.plot()
    print(f'Mesh cleaned.')

    combined_polydata, kdtree = transfer_point_data(tree_df, combined_polydata)

    # Print resource counts in the original DataFrame
    print(f"\nResource counts for tree {tree_key} (original DataFrame):")
    print(tree_df['resource'].value_counts())

    
    # Print resource counts in the point data
    print(f"\nResource counts in tree df for tree {tree_key}:")

    # Convert resource IDs back to names in the point data
    resource_ids = combined_polydata.point_data['resource']
    resource_names = np.array([reverse_resource_mapping.get(resource_id, "Unknown") for resource_id in resource_ids])

    # Replace the integer resource IDs with the corresponding resource names
    combined_polydata.point_data['resource'] = resource_names

    # Now, print the value counts directly on the 'resource' column
    resource_counts = pd.Series(combined_polydata.point_data['resource']).value_counts()

    # Print the resource counts
    print("\nResource counts in point data for combined isomesh:")
    for resource_name, count in resource_counts.items():
        print(f"{resource_name}: {count}")
            
            
    """point_resource_counts = pd.Series(combined_polydata.point_data['resource']).value_counts().sort_index()
    for resource_id, count in point_resource_counts.items():
        resource_name = reverse_resource_mapping.get(resource_id, "Unknown")
        print(f"{resource_name}: {count}")

    # Check for mismatches between original and point data
    original_counts = tree_df['resource'].value_counts()
    for resource_name, original_count in original_counts.items():
        resource_id = resource_mapping.get(resource_name)
        if resource_id is not None:
            point_count = point_resource_counts.get(resource_id, 0)
            if original_count != point_count:
                print(f"WARNING: Resource '{resource_name}' has {original_count} in original data but {point_count} in point data.")"""
    
    # Saving and decimation
    #decimation_levels = [0.5, 0.9]
    #print(f'Decimating meshes to {decimation_levels}')
    #output_meshes = f_resource_meshdecimator.decimate_mesh_levels(combined_polydata, decimation_levels=decimation_levels)

    output_meshes = {'original' : combined_polydata}

    for level, mesh in output_meshes.items():
        os.makedirs(f'{output_dir}/resolution_{level}', exist_ok=True)
        filepath = f"{output_dir}/resolution_{level}/{filename}.vtk"
        mesh.save(filepath)
        print(f"checking point_data resource is {mesh.point_data['resource']}")
        print(f'Mesh saved to {filepath}')
    else:
        print(f"No valid polydata generated for tree {tree_key}. Skipping saving.")

# Main execution
if __name__ == "__main__":
    tree_templates = load_tree_templates('data/revised/revised_tree_dict.pkl')

    # Output directory for VTK files
    output_dir = 'data/revised/treeMeshes'

    # Create a single VTK for each tree in the templates
    print("Processing tree templates...")
    for tree_key, tree_df in tree_templates.items():
        process_tree_and_save_vtk(tree_key, tree_df, output_dir)

    print("Processing completed.")
    
    """tree_templates = load_tree_templates('data/treeOutputs/adjusted_tree_templates.pkl')
    senescing_templates = load_tree_templates('data/treeOutputs/fallen_trees_dict.pkl')

    # Output directory for VTK files
    output_dir = 'data/revised/treeMeshes'

    # Create a single VTK for each tree in the templates
    print("Processing tree templates...")
    for tree_key, tree_df in tree_templates.items():xw
        process_tree_and_save_vtk(tree_key, tree_df, output_dir)

    print("Processing senescing tree templates...")
    for senescing_key, senescing_df in senescing_templates.items():
        process_tree_and_save_vtk(senescing_key, senescing_df, output_dir)

    print("Processing completed.")"""
