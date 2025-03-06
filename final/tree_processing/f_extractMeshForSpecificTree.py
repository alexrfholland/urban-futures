import pickle
import pandas as pd
import pyvista as pv
import numpy as np
import os
import f_resource_meshdecimator

# Function to load the tree templates from a pickle file
def load_tree_templates(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

# Single isoSize for all resources
isoSize = (0.15, 0.15, 0.15)

# Function to extract isosurface from PolyData
def extract_isosurface_from_polydata(polydata: pv.PolyData, spacing: tuple[float, float, float], isovalue: float = 1.0) -> pv.PolyData:
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
        return grid.contour(isosurfaces=[isovalue], scalars='values')
    else:
        return None

# Function to process tree templates and save the output as PLY
def process_tree_and_save_ply(tree_key, tree_df, output_dir):
    print(f"\nProcessing tree {tree_key}...")

    # Filter out resources 'leaf cluster', 'fallen log', and 'leaf litter'
    filtered_tree_df = tree_df[~tree_df['resource'].isin(['leaf cluster', 'fallen log', 'leaf litter'])]

    if filtered_tree_df.empty:
        print(f"No valid resources for tree {tree_key}. Skipping.")
        return

    if len(tree_key) == 5:
        filename = f'{tree_key[0]}_{tree_key[1]}_{tree_key[2]}_{tree_key[3]}_{tree_key[4]}'
    else:
        filename = f'{tree_key[0]}_{tree_key[1]}_{tree_key[2]}'

    polydata = pv.PolyData(filtered_tree_df[['X', 'Y', 'Z']].dropna().values)

    # Create isosurface using the single isoSize
    isosurface = extract_isosurface_from_polydata(polydata, isoSize)

    if isosurface is not None:
        print(f'cleaning the mesh')
        cleaned_mesh = isosurface.clean() 
        print(f'mesh cleaned')

        # Apply Laplacian smoothing
        print(f'smoothing the mesh')
        smoothed_mesh = cleaned_mesh.smooth(n_iter=30, relaxation_factor=0.01)  # Adjust n_iter and relaxation_factor as needed
        print(f'mesh smoothed')

        # Estimate normals
        print(f'estimating normals')
        cleaned_mesh = cleaned_mesh.compute_normals()

        cleaned_mesh.plot()

        filepath = f"{output_dir}/{filename}.ply"
        cleaned_mesh.save(filepath) 
        print(f'mesh saved to {filepath}')

# Main execution
if __name__ == "__main__":
    tree_templates = load_tree_templates('data/treeOutputs/adjusted_tree_templates.pkl')

    # Output directory for PLY files
    output_dir = 'data/revised/stanislav'

    # List of tree keys to process
    tree_keys_to_process = [
        (True, 'medium', 'reserve-tree', False, 10),
        (True, 'large', 'reserve-tree', False, 11),
        (True, 'large', 'reserve-tree', False, 12),
        (True, 'large', 'reserve-tree', False, 13),
        (True, 'large', 'reserve-tree', False, 14),
        (True, 'large', 'reserve-tree', False, 15),
        (True, 'large', 'reserve-tree', False, 16),
        ]  # Add your tree keys here

    # Create a single PLY for each tree in the list
    for tree_key in tree_keys_to_process:
        process_tree_and_save_ply(tree_key, tree_templates[tree_key], output_dir)

    print("Processing completed.")
