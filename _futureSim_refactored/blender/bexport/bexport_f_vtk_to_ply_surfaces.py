import pyvista as pv
import numpy as np
from plyfile import PlyData, PlyElement
import os
import numpy as np
import json

import numpy as np
import json
import os
from plyfile import PlyData, PlyElement

import numpy as np
import json
import os
from plyfile import PlyData, PlyElement

import numpy as np
import json
import os
from plyfile import PlyData, PlyElement

import numpy as np
import json
import os
from plyfile import PlyData, PlyElement

def export_polydata_to_ply(mesh, filename, attributesToTransfer=None):
    """
    Export a PyVista PolyData object to a PLY file as a point cloud using plyfile and structured NumPy arrays.

    Parameters:
    - mesh: pyvista.PolyData
        The PolyData object to export.
    - filename: str
        The output PLY filename.
    - attributesToTransfer: list of str, optional
        List of additional point data attributes to transfer.
    """
    print("Starting export of PolyData to PLY using plyfile with structured NumPy arrays.")

    # Step 1: Determine which attributes to transfer
    keys = determine_attributes(mesh, attributesToTransfer)

    # Step 2: Initialize point coordinates
    points = mesh.points
    num_points = points.shape[0]
    print(f"Number of points to export: {num_points}")

    # Step 3: Prepare dtype for structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    # Temporary storage for data assignment
    data = {}
    data['x'] = points[:, 0].astype(np.float32)
    data['y'] = points[:, 1].astype(np.float32)
    data['z'] = points[:, 2].astype(np.float32)

    # Step 4: Process each attribute and update dtype and data
    for key in keys:
        if key.lower() == 'colors':
            print("Processing 'Colors' attribute.")
            colors = mesh.point_data[key]

            # Ensure colors are in uint8 format
            if colors.dtype != np.uint8:
                if np.issubdtype(colors.dtype, np.floating):
                    colors = (colors * 255).clip(0, 255).astype(np.uint8)
                    print("Converted floating colors to uint8.")
                else:
                    colors = colors.astype(np.uint8)
                    print("Converted colors to uint8.")

            # Add 'red', 'green', 'blue' to dtype and data
            for i, color_component in enumerate(['red', 'green', 'blue']):
                dtype.append((color_component, 'u1'))
                data[color_component] = colors[:, i]
                print(f"Added '{color_component}' color component with dtype {data[color_component].dtype}.")
            continue

        array = mesh.point_data[key]

        if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            if array.ndim > 1:
                # Multi-component numeric attribute (e.g., Normals)
                for i in range(array.shape[1]):
                    component_name = f"{key}_{i}"  # e.g., Normals_0, Normals_1, Normals_2
                    dtype.append((component_name, 'f4'))
                    data[component_name] = array[:, i].astype(np.float32)
                    print(f"Processed multi-component numeric attribute: {component_name} with dtype {data[component_name].dtype}")
            else:
                if np.issubdtype(array.dtype, np.floating):
                    dtype.append((key, 'f4'))
                    data[key] = array.astype(np.float32)
                    print(f"Processed numeric (float) attribute: {key} with dtype {data[key].dtype}")
                elif np.issubdtype(array.dtype, np.integer):
                    dtype.append((key, 'i4'))
                    data[key] = array.astype(np.int32)
                    print(f"Processed integer attribute: {key} with dtype {data[key].dtype}")
                elif np.issubdtype(array.dtype, np.bool_):
                    dtype.append((key, 'u1'))  # Store bool as unsigned byte
                    data[key] = array.astype(np.uint8)
                    print(f"Processed boolean attribute: {key} with dtype {data[key].dtype}")
                else:
                    dtype.append((key, 'f4'))  # Default to float
                    data[key] = array.astype(np.float32)
                    print(f"Processed attribute with unknown numeric type as float: {key} with dtype {data[key].dtype}")
            continue

        if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.object_):
            # Handle string attributes
            print(f"Processing string attribute: {key}")
            mapping_filename = f"{filename}_{key}_mapping.json"

            # Load existing mapping if available
            if os.path.exists(mapping_filename):
                with open(mapping_filename, 'r') as f:
                    mapping = json.load(f)
                print(f"Loaded existing mapping for '{key}' from '{mapping_filename}'.")
            else:
                mapping = {}
                print(f"No existing mapping found for '{key}'. Creating new mapping.")

            unique_strings = np.unique(array)
            new_strings = [s for s in unique_strings if s not in mapping]

            if new_strings:
                max_id = max(mapping.values(), default=-1)
                new_ids = range(max_id + 1, max_id + 1 + len(new_strings))
                mapping.update({s: id_ for s, id_ in zip(new_strings, new_ids)})
                with open(mapping_filename, 'w') as f:
                    json.dump(mapping, f)
                print(f"Added {len(new_strings)} new entries to mapping for attribute '{key}'.")
            else:
                print(f"No new entries to add to mapping for attribute '{key}'.")

            # Vectorized mapping of strings to integers
            # Using list comprehension for efficiency
            mapped_ids = np.array([mapping.get(s, -1) for s in array], dtype=np.int32)
            if np.any(mapped_ids == -1):
                print(f"Warning: Some strings in attribute '{key}' were not found in mapping and set to -1.")

            dtype.append((key, 'i4'))
            data[key] = mapped_ids
            print(f"Mapped string attribute '{key}' to integer IDs with dtype {data[key].dtype}.")
            continue

        print(f"Skipped unsupported attribute type for key: {key}")

    # Step 5: Create structured array
    print("Creating structured NumPy array.")
    structured_array = np.empty(num_points, dtype=dtype)
    for field_name in structured_array.dtype.names:
        if field_name in data:
            structured_array[field_name] = data[field_name]
        else:
            # This should not happen, but just in case
            print(f"Warning: '{field_name}' not found in data dictionary. Filling with zeros.")
            structured_array[field_name] = 0

    # Verify data types before writing
    print("\nStructured Array Dtypes:")
    for name in structured_array.dtype.names:
        print(f"  {name}: {structured_array[name].dtype}")

    # Step 6: Create PlyElement and write PLY file
    print("\nCreating PlyElement.")
    ply_element = PlyElement.describe(structured_array, 'vertex')

    print(f"Writing PLY file to '{filename}'.")
    PlyData([ply_element], text=False).write(filename)

    print(f"Successfully wrote PLY file '{filename}' with {num_points} points.")

def determine_attributes(mesh, attributesToTransfer):
    """
    Determine which point data attributes to transfer based on user input.

    Parameters:
    - mesh: pyvista.PolyData
        The PolyData object containing point data.
    - attributesToTransfer: list of str or None
        User-specified attributes to transfer.

    Returns:
    - keys: list of str
        The list of attribute keys to transfer.
    """
    # First, rename any attributes containing spaces
    for old_key in list(mesh.point_data.keys()):
        if ' ' in old_key:
            new_key = old_key.replace(' ', '_')
            mesh.point_data[new_key] = mesh.point_data.pop(old_key)
            print(f"Renamed attribute '{old_key}' to '{new_key}'")

    if attributesToTransfer is None:
        keys = list(mesh.point_data.keys())
        print(f"No specific attributesToTransfer provided. Transferring all point data keys: {keys}")
    else:
        possible_keys = ['colors', 'normals']
        keys = []
        for key in possible_keys:
            cap_key = key.capitalize()
            if cap_key in mesh.point_data:
                keys.append(cap_key)
            elif key in mesh.point_data:
                keys.append(key)
                print(f"Confirmed key: {key}")
        keys.extend(attributesToTransfer)
        print(f"Confirmed keys to transfer: {keys}")
    return keys

# Example Usage

def process_site_voxels(site):
    # Load the VTK files containing the voxel data
    #siteVoxels = pv.read(f"data/revised/{site}-siteVoxels-masked.vtk")
    #roadVoxels = pv.read(f"data/revised/{site}-roadVoxels-coloured.vtk")
    treeVoxels = pv.read(f"data/revised/{site}-treeVoxels-coloured.vtk")
    

    # Save the siteVoxels with all point_data properties
    #save_ply(siteVoxels, f"data/revised/{site}-siteVoxels.ply")
    #save_ply(roadVoxels, f"data/revised/{site}-roadVoxels.ply")


    print(f'Point Data: {treeVoxels.point_data.keys()}')
    print(f'Cell Data: {treeVoxels.cell_data.keys()}')


    
    # Extract surface for treeVoxels and save
    """print('Extracting surfaces')

    # Extract surface
    treeVoxels = treeVoxels.extract_surface()

    # Print point and cell data attribute names after extraction
    print(f'After extraction - Point Data: {treeVoxels.point_data.keys()}')
    print(f'After extraction - Cell Data: {treeVoxels.cell_data.keys()}')

    print('Surfaces extracted')"""

    attributes = ['resource','radius','tree_number']

    export_polydata_to_ply(treeVoxels, f"data/revised/{site}-treeVoxels2.ply", attributes)

def main():
    sites = ['city', 'uni', 'trimmed-parade']
    sites = ['trimmed-parade']
    for site in sites:
        print(f"Processing site: {site}")
        process_site_voxels(site)

if __name__ == "__main__":
    main()
