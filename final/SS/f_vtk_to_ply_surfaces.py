import pyvista as pv
import numpy as np
from plyfile import PlyData, PlyElement

import pyvista as pv
import numpy as np
from plyfile import PlyData, PlyElement

"""
def save_ply(mesh, filename, attributesToTransfer=None):
    # Get the number of points and cells
    n_points = mesh.number_of_points
    n_cells = mesh.number_of_cells

    # Initialize the list of tuples for the vertex data type
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    # Initialize the dictionary to hold the vertex data
    vertex_data = {
        'x': mesh.points[:, 0],
        'y': mesh.points[:, 1],
        'z': mesh.points[:, 2],
    }

    # Keep track of existing field names to avoid duplicates
    existing_field_names = set(['x', 'y', 'z'])

    # Define acceptable data types mapping
    numpy_to_ply_dtype = {
        np.dtype('int8'): 'int8',
        np.dtype('uint8'): 'uint8',
        np.dtype('int16'): 'int16',
        np.dtype('uint16'): 'uint16',
        np.dtype('int32'): 'int32',
        np.dtype('uint32'): 'uint32',
        np.dtype('float32'): 'float32',
        np.dtype('float64'): 'float64',
    }

    # Iterate over each point_data array in the mesh
    0...0.

    for key in keys:
        if key in ['x', 'y', 'z']:
            continue  # Skip 'x', 'y', 'z' fields in point_data

        array = mesh.point_data[key]

        # Handle colors separately
        if key == 'colors':
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating):
                    array = (array * 255).clip(0, 255).astype(np.uint8)
                else:
                    array = array.astype(np.uint8)

            for i, color_component in enumerate(['red', 'green', 'blue']):
                field_name = color_component
                if field_name not in existing_field_names:
                    vertex_dtype.append((field_name, 'uint8'))
                    vertex_data[field_name] = array[:, i]
                    existing_field_names.add(field_name)
            continue

        # Handle normals separately
        elif key.lower() == 'normals':
            if array.dtype != np.float32:
                array = array.astype(np.float32)

            for i, normal_component in enumerate(['nx', 'ny', 'nz']):
                field_name = normal_component
                if field_name not in existing_field_names:
                    vertex_dtype.append((field_name, 'float32'))
                    vertex_data[field_name] = array[:, i]
                    existing_field_names.add(field_name)
            continue

        # Handle scalar or vector data
        if array.dtype in numpy_to_ply_dtype:
            dtype = numpy_to_ply_dtype[array.dtype]
        elif array.dtype in [np.int64, np.uint64]:
            if np.all(array >= np.iinfo(np.int32).min) and np.all(array <= np.iinfo(np.int32).max):
                array = array.astype(np.int32)
                dtype = 'int32'
            else:
                continue
        else:
            continue

        if array.ndim == 1:
            field_name = key
            if field_name not in existing_field_names:
                vertex_dtype.append((field_name, dtype))
                vertex_data[field_name] = array
                existing_field_names.add(field_name)
        elif array.ndim == 2 and array.shape[1] == 3:
            for i, component in enumerate(['x', 'y', 'z']):
                field_name = f"{key}_{component}"
                if field_name not in existing_field_names:
                    vertex_dtype.append((field_name, dtype))
                    vertex_data[field_name] = array[:, i]
                    existing_field_names.add(field_name)

    # Now handle the cell data in a similar manner
    cell_dtype = []
    cell_data = {}

    for key in mesh.cell_data.keys():
        array = mesh.cell_data[key]
        array.dtype = array.dtype

        # Handle scalar or vector data
        if array.dtype in numpy_to_ply_dtype:
            dtype = numpy_to_ply_dtype[array.dtype]
        elif array.dtype in [np.int64, np.uint64]:
            if np.all(array >= np.iinfo(np.int32).min) and np.all(array <= np.iinfo(np.int32).max):
                array = array.astype(np.int32)
                dtype = 'int32'
            else:
                continue
        else:
            continue

        if array.ndim == 1:
            field_name = key
            if field_name not in existing_field_names:
                cell_dtype.append((field_name, dtype))
                cell_data[field_name] = array
                existing_field_names.add(field_name)
        elif array.ndim == 2 and array.shape[1] == 3:
            for i, component in enumerate(['x', 'y', 'z']):
                field_name = f"{key}_{component}"
                if field_name not in existing_field_names:
                    cell_dtype.append((field_name, dtype))
                    cell_data[field_name] = array[:, i]
                    existing_field_names.add(field_name)

    # Create a structured array for vertices
    vertex_all = np.empty(n_points, dtype=vertex_dtype)
    for name in vertex_dtype:
        field_name = name[0]
        vertex_all[field_name] = vertex_data[field_name]

    # Create the PlyElement for vertices
    vertex_element = PlyElement.describe(vertex_all, 'vertex')

    # Create a structured array for cells, if any cell data exists
    if cell_data:
        cell_all = np.empty(n_cells, dtype=cell_dtype)
        for name in cell_dtype:
            field_name = name[0]
            cell_all[field_name] = cell_data[field_name]

        # Create the PlyElement for cells
        cell_element = PlyElement.describe(cell_all, 'face')

        # Write the PLY file with both vertex and cell elements
        PlyData([vertex_element, cell_element], text=False).write(filename)
    else:
        # Write the PLY file with only vertex elements
        PlyData([vertex_element], text=False).write(filename)

    print(f"Saved {filename} with all point_data and cell_data properties.")
"""

import numpy as np
from plyfile import PlyData, PlyElement

import numpy as np
import h5py
import json
import pyvista as pv

def export_polydata_to_hdf5(mesh, filename, attributesToTransfer=None):
    """
    Export a PyVista PolyData mesh to an HDF5 file, preserving all vertex attributes,
    including strings and booleans.

    Parameters:
    - mesh (pv.PolyData): The PyVista mesh to export.
    - filename (str): The destination HDF5 filename.
    """
    
    if attributesToTransfer is None:
        keys = mesh.point_data.keys()
    else:
        possible_keys = ['colors', 'normals']
        # Create the actual list, capitalizing the item if the capitalized version exists in mesh.point_data
        keys = []
        for key in possible_keys:
            if key.capitalize() in mesh.point_data:
                keys.append(key.capitalize())
            elif key in mesh.point_data:
                keys.append(key)
                print(f'confirmed keys are: {keys}')
                keys.extend(attributesToTransfer)

    print(f'attributes to transfer are: {keys}')

    with h5py.File(filename, 'w') as h5f:
        # Store vertex coordinates
        h5f.create_dataset('points', data=mesh.points)

        # Initialize groups for attributes
        h5f.create_group('attributes')

        for attr_name in keys:
            array = mesh.point_data[attr_name]
            dtype = array.dtype

            if np.issubdtype(dtype, np.integer):
                h5f['attributes'].create_dataset(attr_name, data=array, dtype='i4')
            elif np.issubdtype(dtype, np.floating):
                h5f['attributes'].create_dataset(attr_name, data=array, dtype='f4')
            elif np.issubdtype(dtype, np.bool_):
                # Store booleans as integers (0 and 1)
                h5f['attributes'].create_dataset(attr_name, data=array.astype('i1'), dtype='i1')
            elif np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_):
                # Store strings as variable-length UTF-8
                dt = h5py.string_dtype(encoding='utf-8')
                # Convert to a list of Python strings
                data = array.astype(str).tolist()
                h5f['attributes'].create_dataset(attr_name, data=data, dtype=dt)
            else:
                print(f"Warning: Unsupported data type for attribute '{attr_name}'. Skipping.")

    print(f"Exported mesh to '{filename}' successfully.")


def save_ply(mesh, filename, attributesToTransfer=None):
    """
    Save a PyVista mesh as a PLY file containing only vertex (point cloud) data.
    
    Parameters:
    - mesh: PyVista mesh object.
    - filename: Destination filename for the PLY file.
    - attributesToTransfer: List of specific point data attributes to include. If None, all attributes are included.
    """
    # Initialize the vertex data with coordinates
    vertex_data = {
        'x': mesh.points[:, 0],
        'y': mesh.points[:, 1],
        'z': mesh.points[:, 2],
    }

    # Define the initial vertex dtype
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    # Keep track of existing field names to avoid duplicates
    existing_field_names = set(['x', 'y', 'z'])

    # Define acceptable NumPy to PLY dtype mappings
    numpy_to_ply_dtype = {
        np.dtype('int8'): 'int8',
        np.dtype('uint8'): 'uint8',
        np.dtype('int16'): 'int16',
        np.dtype('uint16'): 'uint16',
        np.dtype('int32'): 'int32',
        np.dtype('uint32'): 'uint32',
        np.dtype('float32'): 'float32',
        np.dtype('float64'): 'float64',
    }

    # Determine which attributes to transfer
    if attributesToTransfer is None:
        # Transfer all point_data keys except 'x', 'y', 'z'
        keys = [key for key in mesh.point_data.keys() if key not in existing_field_names]
    else:
        # Transfer only specified attributes
        keys = attributesToTransfer.copy()

    # Optionally include colors and normals if present
    possible_keys = ['colors', 'normals']
    for key in possible_keys:
        if key.capitalize() in mesh.point_data and key.capitalize() not in keys:
            keys.append(key.capitalize())
        elif key in mesh.point_data and key not in keys:
            keys.append(key)

    print(f'Attributes to transfer: {keys}')

    for key in keys:
        if key in ['x', 'y', 'z']:
            continue  # Skip coordinate fields

        if key not in mesh.point_data:
            print(f"Warning: '{key}' not found in point_data. Skipping.")
            continue  # Skip if key not present

        array = mesh.point_data[key]

        # Handle colors
        if key.lower() == 'colors':
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating):
                    array = (array * 255).clip(0, 255).astype(np.uint8)
                else:
                    array = array.astype(np.uint8)
            for i, color_component in enumerate(['red', 'green', 'blue']):
                if color_component not in existing_field_names:
                    vertex_dtype.append((color_component, 'uint8'))
                    vertex_data[color_component] = array[:, i]
                    existing_field_names.add(color_component)
            continue

        # Handle normals
        elif key.lower() == 'normals':
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            for i, normal_component in enumerate(['nx', 'ny', 'nz']):
                if normal_component not in existing_field_names:
                    vertex_dtype.append((normal_component, 'float32'))
                    vertex_data[normal_component] = array[:, i]
                    existing_field_names.add(normal_component)
            continue

        # Handle other scalar or vector attributes
        if array.dtype in numpy_to_ply_dtype:
            dtype = numpy_to_ply_dtype[array.dtype]
        elif array.dtype in [np.int64, np.uint64]:
            # Convert to int32 if within range
            if np.all(array >= np.iinfo(np.int32).min) and np.all(array <= np.iinfo(np.int32).max):
                array = array.astype(np.int32)
                dtype = 'int32'
            else:
                print(f"Skipping '{key}' due to unsupported int64/uint64 range.")
                continue
        elif array.dtype.kind in {'U', 'S'}:
            # Handle string attributes by determining max length
            max_length = max(len(s) for s in array) if array.size > 0 else 1
            dtype = f'S{max_length}'
            array = array.astype(dtype)
        else:
            print(f"Skipping '{key}' due to unsupported data type: {array.dtype}")
            continue

        if array.ndim == 1:
            # Scalar attribute
            field_name = key
            if field_name not in existing_field_names:
                vertex_dtype.append((field_name, dtype))
                vertex_data[field_name] = array
                existing_field_names.add(field_name)
        elif array.ndim == 2 and array.shape[1] == 3:
            # Vector attribute (e.g., RGB colors already handled)
            for i, component in enumerate(['x', 'y', 'z']):
                field_name = f"{key}_{component}"
                if field_name not in existing_field_names:
                    vertex_dtype.append((field_name, dtype))
                    vertex_data[field_name] = array[:, i]
                    existing_field_names.add(field_name)
        else:
            print(f"Skipping '{key}' due to unsupported array shape: {array.shape}")
            continue

    # Create a structured NumPy array for vertices
    vertex_all = np.empty(mesh.number_of_points, dtype=vertex_dtype)
    for name in vertex_dtype:
        field_name = name[0]
        vertex_all[field_name] = vertex_data[field_name]

    # Create the PlyElement for vertices
    vertex_element = PlyElement.describe(vertex_all, 'vertex')

    # Write the PLY file with only vertex elements
    PlyData([vertex_element], text=False).write(filename)

    print(f"Saved point cloud to '{filename}' with all specified point_data attributes.")

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

    export_polydata_to_hdf5(treeVoxels, f"data/revised/{site}-treeVoxels.h5", attributes)

    #save_ply(treeVoxels, f"data/revised/{site}-treeVoxels.ply", attributes)

def main():
    sites = ['city', 'uni', 'trimmed-parade']
    sites = ['trimmed-parade']
    for site in sites:
        print(f"Processing site: {site}")
        process_site_voxels(site)

if __name__ == "__main__":
    main()

