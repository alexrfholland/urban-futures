import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import pyvista as pv
import os

def convertToPoly(voxelDF, key, folderPath):


    precolonial, size, control, tree_id = key
    points = voxelDF[['x', 'y', 'z']].values
    poly = pv.PolyData(points)

    # Add all columns as point data attributes
    for col in voxelDF.columns:
        if col not in ['x', 'y', 'z']:  # Skip coordinate columns
            poly.point_data[col] = voxelDF[col].values

    #poly.plot(scalars=treeResource_peeling bark', render_points_as_spheres=True)

    print(key)
    name = f"{precolonial}_{size}_{control}_{tree_id}"
    
    #export polydata as a vtk file
    poly.save(f'{folderPath}/{name}.vtk')

    print(f'exported poly to {folderPath}/{name}.vtk')


def aggregate_data(df):
    """
    Splits the resource column into boolean columns for each resource type.

    Parameters:
        df (pd.DataFrame): The input dataframe with resource column.

    Returns:
        pd.DataFrame: DataFrame with boolean resource columns.
    """
    # Define a hardcoded list of resources
    resource_names = [
        'perch branch', 'peeling bark', 'dead branch', 'other',
        'fallen log', 'epiphyte', 'hollow', 'leaf cluster'
    ]

    # Make a copy of the input dataframe
    df = df.copy()

    # Initialize resource columns with 0s
    for resource in resource_names:
        df[f'resource_{resource}'] = 0

    # Set resource columns to 1 where resource matches
    for resource in resource_names:
        mask = df['resource'] == resource
        df.loc[mask, f'resource_{resource}'] = 1

    return df


# ================================
# Main Function
# ================================

def main():
    input_dir = Path('data/revised') 
    input_name = 'revised_tree_dict.pkl'
    input_path = input_dir / input_name

    # Load pickled input dictionary
    with open(input_path, 'rb') as f:
        tree_dict = pickle.load(f)

    updated_tree_dict = {}

    # Iterate through dictionary keys
    for tree_key, tree_df in tree_dict.items():
        #Extract valid df, ie. rows where tree_df['resource'] != 'leaf litter'. Reset index
        tree_df = tree_df[tree_df['resource'] != 'leaf litter'].reset_index(drop=True)
         
        # Ensure the DataFrame contains 'X', 'Y', 'Z' columns
        if not all(col in tree_df.columns for col in ['X', 'Y', 'Z']):
            raise KeyError("The DataFrame is missing 'X', 'Y', or 'Z' columns")
        
        tree_df = tree_df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'})

        tree_df = aggregate_data(tree_df)

        outputVTKpath =  'data/revised/lidar scans/euc/VTKs'
        #check for path existance and if not make it
        if not os.path.exists(outputVTKpath):
            os.makedirs(outputVTKpath)

        convertToPoly(tree_df, tree_key, outputVTKpath)

        # Add voxelized tree df to output voxel dictionary under same key
        updated_tree_dict[tree_key] = tree_df

    # Check that output directory exists, create if not
    output_dir = Path('data/revised/trees') 
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the tree dictionary as a pickle file
    output_name = f'updated_tree_dict.pkl'
    output_path = output_dir / output_name

    with open(output_path, 'wb') as f:
        pickle.dump(updated_tree_dict, f)

    print(f'Updated tree dictionary saved at {output_path}')

if __name__ == "__main__":
    main()
