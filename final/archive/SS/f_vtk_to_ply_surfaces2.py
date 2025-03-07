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

def export_polydata_to_ply(mesh, filename, attributesToTransfer=None):
    """
    Export a PyVista PolyData object to a PLY file as a point cloud using plyfile.

    Parameters:
    - mesh: pyvista.PolyData
        The PolyData object to export.
    - filename: str
        The output PLY filename.
    - attributesToTransfer: list of str, optional
        List of additional point data attributes to transfer.
    """
    print("Starting export of PolyData to PLY using plyfile.")

    # Step 1: Determine which attributes to transfer
    keys = determine_attributes(mesh, attributesToTransfer)

    # Step 2: Initialize point coordinates
    points = mesh.points
    num_points = points.shape[0]
    print(f"Number of points to export: {num_points}")

    # Prepare the data dictionary with point coordinates
    vertex_dict = {
        'x': points[:, 0].astype(np.float32),
        'y': points[:, 1].astype(np.float32),
        'z': points[:, 2].astype(np.float32)
    }

    # Step 3: Process each attribute
    vertex_dict = process_attributes(mesh, keys, vertex_dict, filename)

    # Step 4: Assemble structured array and write PLY
    write_ply(vertex_dict, mesh, filename, num_points)

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

def process_attributes(mesh, keys, vertex_dict, filename):
    """
    Process different types of attributes and add them to the vertex dictionary.

    Parameters:
    - mesh: pyvista.PolyData
        The PolyData object containing point data.
    - keys: list of str
        The list of attribute keys to process.
    - vertex_dict: dict
        The dictionary to store vertex data.
    - filename: str
        The base filename to create mapping JSON files for string attributes.

    Returns:
    - vertex_dict: dict
        Updated dictionary with processed attributes.
    """
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

            # Assign 'red', 'green', 'blue' components
            vertex_dict['red'] = colors[:, 0]
            vertex_dict['green'] = colors[:, 1]
            vertex_dict['blue'] = colors[:, 2]
            print("Added 'red', 'green', 'blue' color components.")
            continue

        array = mesh.point_data[key]

        if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            # Handle numeric and boolean attributes
            if array.ndim > 1:
                # Multi-component numeric attribute (e.g., Normals)
                for i in range(array.shape[1]):
                    component_name = f"{key}_{i}"  # e.g., Normals_0, Normals_1, Normals_2
                    vertex_dict[component_name] = array[:, i].astype(np.float32)
                    print(f"Processed multi-component numeric attribute: {component_name}")
            else:
                if np.issubdtype(array.dtype, np.floating):
                    vertex_dict[key] = array.astype(np.float32)
                    print(f"Processed numeric (float) attribute: {key}")
                elif np.issubdtype(array.dtype, np.integer):
                    vertex_dict[key] = array.astype(np.int32)
                    print(f"Processed integer attribute: {key}")
                elif np.issubdtype(array.dtype, np.bool_):
                    vertex_dict[key] = array.astype(np.uint8)  # Store bool as unsigned byte
                    print(f"Processed boolean attribute: {key}")
                else:
                    vertex_dict[key] = array.astype(np.float32)  # Default to float
                    print(f"Processed attribute with unknown numeric type as float: {key}")
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
            vectorized_map = np.vectorize(mapping.get)
            mapped_ids = vectorized_map(array).astype(np.int32)

            vertex_dict[key] = mapped_ids
            print(f"Mapped string attribute '{key}' to integer IDs.")
            continue

        print(f"Skipped unsupported attribute type for key: {key}")

    return vertex_dict

def write_ply(vertex_dict, mesh, filename, num_points):
    """
    Assemble the structured array and write the PLY file using plyfile.

    Parameters:
    - vertex_dict: dict
        The dictionary containing all vertex data.
    - mesh: pyvista.PolyData
        The PolyData object containing point data.
    - filename: str
        The output PLY filename.
    - num_points: int
        The number of points to export.
    """
    print("Assembling data for PlyElement.")
    vertex_dtype = []

    for key in vertex_dict:
        if key in ['x', 'y', 'z']:
            vertex_dtype.append((key, 'f4'))
        elif key in ['red', 'green', 'blue']:
            vertex_dtype.append((key, 'u1'))
        else:
            # Determine the dtype based on the data type
            array = vertex_dict[key]
            if np.issubdtype(array.dtype, np.floating):
                vertex_dtype.append((key, 'f4'))
            elif np.issubdtype(array.dtype, np.integer):
                vertex_dtype.append((key, 'i4'))
            elif np.issubdtype(array.dtype, np.bool_):
                vertex_dtype.append((key, 'u1'))
            else:
                vertex_dtype.append((key, 'f4'))  # Default to float

    structured_array = np.empty(num_points, dtype=vertex_dtype)
    for field in vertex_dtype:
        structured_array[field[0]] = vertex_dict[field[0]]

    # Create PlyElement
    ply_element = PlyElement.describe(structured_array, 'vertex')

    # Write the PLY file
    print(f"Writing PLY file to '{filename}'.")
    PlyData([ply_element], text=False).write(filename)



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

    export_polydata_to_ply(treeVoxels, f"data/revised/{site}-treeVoxels.ply", attributes)

def main():
    sites = ['city', 'uni', 'trimmed-parade']
    sites = ['trimmed-parade']
    for site in sites:
        print(f"Processing site: {site}")
        process_site_voxels(site)

if __name__ == "__main__":
    main()
