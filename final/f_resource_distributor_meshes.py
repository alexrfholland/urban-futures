import numpy as np
import pandas as pd
import pyvista as pv
import os
import ast

def distribute_meshes(processed_df):
    print('Starting distribute_meshes function...')
    
    meshFolder = "data/revised/treeMeshes/resolution_original"

    # Extract positions and meshIDs
    positions = processed_df[['x', 'y', 'z']].values
    

    # Function to safely evaluate string representation of tuple
    def safe_eval(s):
        try:
            return ast.literal_eval(s)
        except:
            print(f"Failed to evaluate: {s}")
            return None

    # Function to generate filepath for each mesh
    def get_filepath(key):
        try:
            filename = f"{key}.vtk"
            return os.path.join(meshFolder, filename)
        except Exception as e:
            print(f"Error processing key {key}: {str(e)}")
            return None

    for col in ['precolonial', 'size', 'control', 'tree_id']:
        print(f"{col}: {processed_df[col].dtype}")


    #Create key
    processed_df['key'] = processed_df['precolonial'].astype(str) + '_' + \
            processed_df['size'].astype(str) + '_' + \
            processed_df['control'].astype(str) + '_' + \
            processed_df['tree_id'].astype(str)

    
    
    mesh_ids = processed_df['key'].values
    # Create a mapping of unique meshIDs to integers
    unique_mesh_ids = processed_df['key'].unique()
    id_to_int = {id: i for i, id in enumerate(unique_mesh_ids)}

    # Convert meshIDs to integers
    mesh_id_ints = np.array([id_to_int[id] for id in mesh_ids])

    # Create a dictionary of meshes
    mesh_dict = {}
    for id in unique_mesh_ids:
        filepath = get_filepath(id)
        if filepath:
            try:
                mesh_dict[id_to_int[id]] = pv.read(filepath)
            except FileNotFoundError:
                print(f"Mesh file not found: {filepath}")
            except Exception as e:
                print(f"Error loading mesh {id}: {str(e)}")

    # Create a PyVista MultiBlock object to hold all meshes
    multiblock = pv.MultiBlock()

    # Process each unique mesh
    for int_id in range(len(unique_mesh_ids)):
        # Get indices for this mesh ID
        indices = np.where(mesh_id_ints == int_id)[0]

        # Get the original mesh
        if int_id not in mesh_dict:
            print(f"Skipping mesh with ID {int_id} as it was not loaded successfully")
            continue

        original_mesh = mesh_dict[int_id]

        # Create transformation matrices for all instances of this mesh
        transforms = np.tile(np.eye(4), (len(indices), 1, 1))
        transforms[:, :3, 3] = positions[indices]

        # Debug: Print shape and sample of transforms
        print(f"Shape of transforms: {transforms.shape}")
        print(f"Sample transform:\n{transforms[0]}")

        # Apply transformations to the mesh and add to the MultiBlock
        for i, transform in enumerate(transforms):
            new_mesh = original_mesh.copy()

            # get the row of the relevant tree from processeddf. the row should match i
            row = processed_df.iloc[indices[i]]
            print(f'tree-level data is {row}')

            # Print the data types of each element in the row
            print('Data types of each element in the row:')
            for col, val in row.items():
                print(f'{col}: {type(val)}')

            attributesToTransfer = ['tree_number','useful_life_expectency']

            for attribute in attributesToTransfer:
                print(f'\nTransferring attribute {attribute}')
                num_points = new_mesh.n_points  # Get the number of points in the mesh
                value = row[attribute]          # Get the value from the row

                # Create an array filled with the value, of the correct size and data type
                attribute_array = np.full(num_points, value, dtype=type(value))
                new_mesh.point_data[attribute] = attribute_array

                print(f'Assigned {attribute} with value {value} to all {num_points} points.')


            new_mesh.transform(transform)
            multiblock.append(new_mesh)  # Add transformed mesh to the MultiBlock

    # Print the number of blocks in the MultiBlock
    num_blocks = multiblock.GetNumberOfBlocks()
    print(f"Number of individual blocks in the MultiBlock: {num_blocks}")

    print('Combining the meshes...')
    combined_mesh = multiblock.combine()
    print('Meshes combined!')

    # Extract surface for treeVoxels and save
    print('Extracting surfaces')
    combined_mesh = combined_mesh.extract_surface()
    print('Surfaces extracted')

    return combined_mesh

if __name__ == "__main__":
    df = pd.read_csv("/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/processedDF.csv")
    distribute_meshes(df)


