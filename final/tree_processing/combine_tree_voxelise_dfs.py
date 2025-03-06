import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import pyvista as pv

# ================================
# Step 1: Assign Voxel Coordinates
# ================================


# ================================
# Step 1: Assign Voxel Coordinates
# ================================

def assign_voxel_coordinates(df, voxel_size):
    """
    Assigns voxel coordinates to each point in the dataframe based on a uniform voxel size.

    Parameters:
        df (pd.DataFrame): The input dataframe with 'X', 'Y', 'Z' columns.
        voxel_size (float): A single value representing uniform voxel size for all axes.

    Returns:
        pd.DataFrame: The dataframe with additional 'voxel_X', 'voxel_Y', 'voxel_Z' columns.
    """
    print(df.head())
    for axis in ['x', 'y', 'z']:
        #change axis to upper case
        voxel_col = f'voxel_{axis.upper()}'
        df[voxel_col] = np.floor(df[axis] / voxel_size) * voxel_size
    return df

# ================================
# Step 2: Aggregate Counts and Existence Flags
# ================================

def count_resources_by_voxel(df, resetCount=False):
    """
    Counts occurrences of all resource columns grouped by voxel coordinates and renames coordinate columns.
    If resetCount is True, sets all non-zero counts to 1 (presence/absence).

    Parameters:
        df (pd.DataFrame): DataFrame containing voxel coordinates and resource columns
        resetCount (bool): If True, converts counts to binary presence (1) or absence (0)
        
    Returns:
        pd.DataFrame: Aggregated counts of resources by voxel with renamed coordinates
    """
    # Define expected resource types
    resource_names = [
        'perch branch', 'peeling bark', 'dead branch', 'other',
        'fallen log', 'leaf litter', 'epiphyte', 'hollow', 'leaf cluster'
    ]
    
    # Find all columns that start with 'resource_'
    resource_cols = [col for col in df.columns if col.startswith('resource_')]
    
    # Initialize any missing resource columns with 0s
    for resource in resource_names:
        col_name = f'resource_{resource}'
        if col_name not in df.columns:
            df[col_name] = 0
    
    # Define grouping columns
    group_cols = ['voxel_X', 'voxel_Y', 'voxel_Z']
    
    # Group by voxel coordinates and sum all resource columns
    voxelised_df = df.groupby(group_cols)[resource_cols].sum().reset_index()
    
    # If resetCount is True, convert all non-zero values to 1
    if resetCount:
        for col in resource_cols:
            voxelised_df[col] = (voxelised_df[col] > 0).astype(int)
    
    # Rename coordinate columns
    voxelised_df = voxelised_df.rename(columns={
        'voxel_X': 'x',
        'voxel_Y': 'y',
        'voxel_Z': 'z'
    })
    
    return voxelised_df

#redudant, but keeping for now
def aggregate_voxel_data(df):
    """
    Aggregates voxel data by counting unique occurrences of resource types, ensuring all resources are initialized.

    Parameters:
        df (pd.DataFrame): The input dataframe with voxel coordinates and resource types.

    Returns:
        pd.DataFrame: Aggregated voxel data with all resources initialized.
    """
    # Define a hardcoded list of resources
    resource_names = [
        'perch branch', 'peeling bark', 'dead branch', 'other',
        'fallen log', 'leaf litter', 'epiphyte', 'hollow', 'leaf cluster'
    ]

    # One-hot encode 'resource' column
    count_dummies = pd.get_dummies(df['resource'], prefix='resource')

    # Initialize missing resources with 0s
    for resource in resource_names:
        col_name = f'resource_{resource}'
        if col_name not in count_dummies.columns:
            count_dummies[col_name] = 0

    # Define grouping columns (voxel_X, voxel_Y, voxel_Z)
    group_cols = ['voxel_X', 'voxel_Y', 'voxel_Z']

    # Group by voxel coordinates and sum the one-hot encoded count columns
    voxelised_df = pd.concat([df[group_cols], count_dummies], axis=1).groupby(group_cols).sum().reset_index()

    #rename ['voxel_X', 'voxel_Y', 'voxel_Z'] to 'x','y' and 'z'
    voxelised_df = voxelised_df.rename(columns={'voxel_X': 'x', 'voxel_Y': 'y', 'voxel_Z': 'z'})


    return voxelised_df



# ================================
# Main Function
# ================================



def process_trees(voxel_size = 0.25, resetCount = False):
    #Eucalyptus
    input_dir = Path('data/revised/trees') 
    #input_name = 'revised_tree_dict.pkl' #old one
    eucName = 'updated_tree_dict.pkl'
    elmName = 'elm_tree_dict.pkl'

    ##PROCESS EUCS
    #the euc trees are in the old dictionary format rather than the new dataframe format
    euc_path = input_dir / eucName
    print(f'loading euc tree dictionary from {euc_path}')
    with open(euc_path, 'rb') as f:
        euc_template_dic = pickle.load(f)

    voxelized_euc_tree_dict = {}

    # Iterate through dictionary keys
    for tree_key, tree_df in euc_template_dic.items():            
        # Voxelize the tree
        print(f'voxelizing tree key: {tree_key}')
        print(f'tree df is {tree_df}')
        voxelized_tree_df = assign_voxel_coordinates(tree_df, voxel_size)
        voxelized_tree_df = count_resources_by_voxel(voxelized_tree_df, resetCount)

        # Add voxelized tree df to output voxel dictionary under same key
        voxelized_euc_tree_dict[tree_key] = voxelized_tree_df

        # PyVista plot preview
        """coords = voxelized_tree_df[['x', 'y', 'z']]
        print(voxelized_tree_df)
        poly_tree = pv.PolyData(coords.values)
        poly_tree.point_data['resource_perch branch'] = voxelized_tree_df['resource_perch branch']
        plotter = pv.Plotter()
        plotter.add_mesh(poly_tree, scalars = 'resource_perch branch', render_points_as_spheres=True, point_size=5)
        plotter.show()"""

   ##PROCESS ELMS
   #elms are in the new dataframe format
    ##PROCESS ELMS
    elm_path = input_dir / elmName
    elm_template_DF = pd.read_pickle(elm_path)
    print(f'loaded elm template df from {elm_path}')

    # Create new DataFrame instead of modifying in place
    processed_templates = []
    
    # For each row, voxelize the template dataframe
    for _, row in elm_template_DF.iterrows():
        originalTemplate = row['template']
        voxelized_tree_df = assign_voxel_coordinates(originalTemplate, voxel_size)
        voxelized_tree_df = count_resources_by_voxel(voxelized_tree_df, resetCount)
        
        # Create a new row with the processed template
        new_row = row.copy()
        new_row['template'] = voxelized_tree_df
        processed_templates.append(new_row)

        """coords = voxelized_tree_df[['x', 'y', 'z']]
        print(voxelized_tree_df)
        poly_tree = pv.PolyData(coords.values)
        poly_tree.point_data['resource_perch branch'] = voxelized_tree_df['resource_perch branch']
        plotter = pv.Plotter()
        plotter.add_mesh(poly_tree, scalars = 'resource_perch branch', render_points_as_spheres=True, point_size=5)
        plotter.show()
        """

    voxelised_elm_templateDF = pd.DataFrame(processed_templates)
    
    ##SAVE OUTPUTS
    # Check that output directory exists, create if not
    output_dir = Path('data/revised/trees') 
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the voxelized euc tree dictionary as a pickle file
    euc_output_name = f'{voxel_size}_euc_voxel_tree_dict.pkl'
    euc_output_path = output_dir / euc_output_name
    with open(euc_output_path, 'wb') as f:
        pickle.dump(voxelized_euc_tree_dict, f)
    print(f'Voxelized euc tree dictionary saved at {euc_output_path}')

    # Save the voxelized elm template dataframe as a pickle file
    elm_output_name = f'{voxel_size}_elm_voxel_templateDF.pkl'
    elm_output_path = output_dir / elm_output_name
    voxelised_elm_templateDF.to_pickle(elm_output_path)
    print(voxelised_elm_templateDF.head())
    print(f'Voxelized elm template dataframe saved at {elm_output_path}')

    print(f'done')

if __name__ == "__main__":
    process_trees(voxel_size=0.25, resetCount=True)