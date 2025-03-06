import numpy as np
import pyvista as pv
from plyfile import PlyData, PlyElement
import os
import json
import warnings
import trimesh
import sys
import glob
from pathlib import Path
import pandas as pd



resource_cols = [
    'resource_hollow',
    'resource_epiphyte', 
    'resource_dead branch',
    'resource_perch branch',
    'resource_peeling bark',
    'resource_fallen log',
    'resource_other'
]

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
        # Validate the requested attributes exist
        valid_keys = []
        for key in attributesToTransfer:
            if key in mesh.point_data:
                valid_keys.append(key)
            else:
                warnings.warn(f"Requested attribute '{key}' not found in point data")
        keys = valid_keys
        print(f"Transferring specified valid keys: {keys}")
    return keys

def convert_string_attributes(mesh, filename):
    """
    Convert string attributes to integers using a simple mapping.
    Returns a new mesh with only numeric attributes.
    """
    working_mesh = mesh.copy()
    
    # Process both point and cell data
    for data_dict in [working_mesh.point_data, working_mesh.cell_data]:
        keys_to_remove = []
        for key in data_dict.keys():
            array = data_dict[key]
            
            if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.object_):
                print(f"Processing string attribute: {key}")
                
                # Create simple mapping from unique strings to integers
                unique_strings = np.unique(array)
                mapping = {s: i for i, s in enumerate(unique_strings)}
                
                # Vectorized mapping of strings to integers
                mapped_ids = np.array([mapping[s] for s in array], dtype=np.int32)
                
                # Create new integer attribute
                new_key = f"{key}_int"
                data_dict[new_key] = mapped_ids
                keys_to_remove.append(key)
                print(f"Created new integer attribute '{new_key}' from '{key}' with {len(mapping)} unique values")

        # Remove original string attributes
        for key in keys_to_remove:
            del data_dict[key]
            print(f"Removed original string attribute '{key}'")
    
    return working_mesh

def convert_string_attributesWITHJSON(mesh, filename):
    """
    Convert string attributes to integers using our mapping system.
    Returns a new mesh with only numeric attributes.
    """
    working_mesh = mesh.copy()
    
    # Process both point and cell data
    for data_dict in [working_mesh.point_data, working_mesh.cell_data]:
        keys_to_remove = []
        for key in data_dict.keys():
            array = data_dict[key]
            
            if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.object_):
                print(f"Processing string attribute: {key}")
                # Create mapping filename in same directory as input file
                mapping_filename = os.path.join(
                    os.path.dirname(filename),
                    f"{os.path.basename(filename)}_{key}_mapping.json"
                )

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
                mapped_ids = np.array([mapping.get(s, -1) for s in array], dtype=np.int32)
                if np.any(mapped_ids == -1):
                    print(f"Warning: Some strings in attribute '{key}' were not found in mapping and set to -1.")

                # Create new integer attribute
                new_key = f"{key}_int"
                data_dict[new_key] = mapped_ids
                keys_to_remove.append(key)
                print(f"Created new integer attribute '{new_key}' from '{key}'")

        # Remove original string attributes
        for key in keys_to_remove:
            del data_dict[key]
            print(f"Removed original string attribute '{key}'")
    
    return working_mesh


def map_columns(template_df):
    resource_map = {
        'other': 0,
        'perch branch': 1,
        'dead branch': 2,
        'peeling bark': 3,
        'epiphyte': 4,
        'fallen log': 5,
        'hollow': 6
    }
    return template_df


def map_columns_mesh(mesh):
    resource_map = {
        'other': 0,
        'perch branch': 1,
        'dead branch': 2,
        'peeling bark': 3,
        'epiphyte': 4,
        'fallen log': 5,
        'hollow': 6
    }


    # Map 'resource' column
    mesh.point_data['int_resource'] = np.array(
        [resource_map.get(val, -1) for val in mesh.point_data['resource']], dtype=int
    )

    return mesh

def export_polydata_to_ply(mesh, filename, attributesToTransfer=None):
    """
    Export a PyVista PolyData object to a PLY file using trimesh.
    Only exports requested vertex attributes that exist in the mesh.
    """
    print("Starting export of PolyData to PLY")

    if 'resource' in mesh.point_data:
        mesh = map_columns_mesh(mesh)
        
    # Default attributes to look for
    default_attributes = ['int_resource', 'isSenescent', 'isTerminal', 'cluster_id']
    default_attributes.extend(resource_cols)
    if attributesToTransfer is None:
        attributesToTransfer = default_attributes
    else:
        attributesToTransfer.extend(resource_cols) # make unique incase repeating
        attributesToTransfer = list(set(attributesToTransfer))

    # Check which attributes actually exist in the mesh
    available_attributes = []
    for attr in attributesToTransfer:
        if attr in mesh.point_data:
            available_attributes.append(attr)
            print(f"Found attribute: {attr}")
        else:
            print(f"Attribute not found in mesh: {attr}")

    # Continue with existing export logic, but use available_attributes
    surface_mesh = mesh.extract_surface()
    print(f"\nExtracted surface mesh:")
    print(f"Number of vertices: {len(surface_mesh.points)}")
    print(f"Number of faces: {surface_mesh.n_cells}")

    # Rest of the function remains the same, but use available_attributes instead of attributesToTransfer
    new_mesh = surface_mesh.copy()
    for key in list(new_mesh.point_data.keys()):
        del new_mesh.point_data[key]
    
    for key in available_attributes:
        if key in surface_mesh.point_data:
            new_mesh.point_data[key] = surface_mesh.point_data[key]
            print(f"Copied attribute: {key}")

    new_mesh = convert_string_attributes(new_mesh, filename)
    
    vertices = new_mesh.points
    faces = np.array(new_mesh.faces).reshape(-1, 4)[:, 1:4]
    
    vertex_attributes = {}
    for key in available_attributes:
        actual_key = f"{key}_int" if f"{key}_int" in new_mesh.point_data else key
        
        if actual_key in new_mesh.point_data:
            array = new_mesh.point_data[actual_key]
            if np.issubdtype(array.dtype, np.number):
                vertex_attributes[key] = array.astype(np.float32)
                print(f"Added vertex attribute '{key}' (shape: {array.shape})")
    
    mesh_trimesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_attributes=vertex_attributes
    )

    
    mesh_trimesh.export(filename, file_type='ply', encoding='ascii')
    print(f"\nSuccessfully exported PLY file with vertex attributes:")
    print("Vertex attributes:", list(vertex_attributes.keys()))

# Example usage
if __name__ == "__main__":
    # Define input and output folders

    #ask user if processing trees, (2) logs or both (3)
    user_input = int(input("Enter 1 for trees, 2 for logs, 3 for both, 4 for just edits, 5 fora mask "))

    if user_input in [1, 3]:
        print('processing trees')
        input_folder = "data/revised/final/treeMeshes"
        output_folder = "data/revised/final/treeMeshesPly"
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all VTK files in input folder
        vtk_files = glob.glob(os.path.join(input_folder, "*.vtk"))
        
        if not vtk_files:
            print(f"No VTK files found in {input_folder}")
            sys.exit(1)
        
        print(f"Found {len(vtk_files)} VTK files to process")
        
        # Process each VTK file
        for vtk_file in vtk_files:
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(vtk_file))[0]
            output_path = os.path.join(output_folder, f"{base_name}.ply")
            
            print(f"\nProcessing: {vtk_file}")
            print(f"Output to: {output_path}")
            
            try:
                mesh = pv.read(vtk_file)
                export_polydata_to_ply(mesh, output_path)
            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")

    elif user_input == 2:
        print('processing logs')
        input_folder = "data/revised/final/logMeshes"
        output_folder = "data/revised/final/logMeshesPLY"

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all VTK files in input folder
        vtk_files = glob.glob(os.path.join(input_folder, "*.vtk"))

        print(f"Found {len(vtk_files)} VTK files to process")

        #process each file      
        for vtk_file in vtk_files:
            #get the base filename without extension
            base_name = os.path.splitext(os.path.basename(vtk_file))[0]
            output_path = os.path.join(output_folder, f"{base_name}.ply")

            print(f"\nProcessing: {vtk_file}")
            print(f"Output to: {output_path}")
            
            try:
                mesh = pv.read(vtk_file)

                for resource in resource_cols:
                    mesh.point_data[resource] = 0

                mesh.point_data['resource'] = 'fallen log'
                mesh.point_data['resource_fallen log'] = 1
                mesh.point_data['isSenescent'] = True
                
                export_polydata_to_ply(mesh, output_path,attributesToTransfer=['resource'])
            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")
    elif user_input == 4:
        print('processing just edits')
        
        # Convert paths to Path objects while maintaining consistent folder structure
        base_dir = Path('data/revised')
        input_folder = base_dir / 'final/treeMeshes'
        output_folder = base_dir / 'final/treeMeshesPly'  # Updated to match other sections
        
        justEditsDF = pd.read_pickle(base_dir / 'trees/just_edits_templateDF.pkl')

        # Create filenames for each row in justEditsDF
        justEditsDF['filename'] = justEditsDF.apply(
            lambda row: f"precolonial.{row['precolonial']}_size.{row['size']}_control.{row['control']}_id.{row['tree_id']}.vtk",
            axis=1
        )

        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(justEditsDF)} VTK files to process")

        # Process each file in justEditsDF
        for _, row in justEditsDF.iterrows():
            vtk_file = input_folder / row['filename']
            base_name = Path(row['filename']).stem
            output_path = output_folder / f"{base_name}.ply"

            print(f"\nProcessing: {vtk_file}")
            print(f"Output to: {output_path}")
            
            try:
                if not vtk_file.exists():
                    print(f"Warning: File not found: {vtk_file}")
                    continue
                
                mesh = pv.read(str(vtk_file))  # pyvista needs string paths
                export_polydata_to_ply(mesh, str(output_path))
            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")

    elif user_input == 5:
        print('processing only mask')
        # Convert paths to Path objects while maintaining consistent folder structure
        base_dir = Path('data/revised')
        input_folder = base_dir / 'final/treeMeshes'
        output_folder = base_dir / 'final/treeMeshesPly'  # Updated to match other sections
        
        # Load the full templates dataframe
        templatesDF = pd.read_pickle(base_dir / 'trees/edited_combined_templateDF.pkl')

        # Create mask for snags and precolonial trees
        mask = (templatesDF['size'] == 'snag') & (templatesDF['precolonial'] == False)
        #mask = (templatesDF['size'] == 'snag') | (templatesDF['precolonial'] == True)
        filteredDF = templatesDF[mask]
        print(filteredDF)

        # Create filenames for each row in filteredDF
        filteredDF['filename'] = filteredDF.apply(
            lambda row: f"precolonial.{row['precolonial']}_size.{row['size']}_control.{row['control']}_id.{row['tree_id']}.vtk",
            axis=1
        )

        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(filteredDF)} VTK files to process")

        # Process each file in filteredDF
        for _, row in filteredDF.iterrows():
            vtk_file = input_folder / row['filename']
            base_name = Path(row['filename']).stem
            output_path = output_folder / f"{base_name}.ply"

            print(f"\nProcessing: {vtk_file}")
            print(f"Output to: {output_path}")
            
            try:
                if not vtk_file.exists():
                    print(f"Warning: File not found: {vtk_file}")
                    continue
                
                mesh = pv.read(str(vtk_file))  # pyvista needs string paths
                export_polydata_to_ply(mesh, str(output_path))
            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")

        
        output_folder = "data/revised/treesPLY"
