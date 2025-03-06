import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import pyvista as pv
import aa_tree_helper_functions

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
    #print(df.head())
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
    """# Define expected resource types
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
            df[col_name] = 0"""
    
    resource_cols = aa_tree_helper_functions.resource_names()
    
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


# ================================
# Main Function
# ================================

def process_trees(tree_templates_DF, voxel_size = 0.25, resetCount = False):

    # Create new DataFrame instead of modifying in place
    processed_templates = []
    
    # For each row, voxelize the template dataframe
    for _, row in tree_templates_DF.iterrows():
        originalTemplate = row['template']
        originalTemplate = aa_tree_helper_functions.verify_resources_columns(originalTemplate)
        voxelized_tree_df = assign_voxel_coordinates(originalTemplate, voxel_size)
        voxelized_tree_df = count_resources_by_voxel(voxelized_tree_df, resetCount)
        
        # Create a new row with the processed template
        new_row = row.copy()
        extractedVoxelDF = voxelized_tree_df[['x', 'y', 'z'] + aa_tree_helper_functions.resource_names()]
        extractedVoxelDF = aa_tree_helper_functions.create_resource_column(extractedVoxelDF)
        print(extractedVoxelDF.head())
        new_row['template'] = extractedVoxelDF
        processed_templates.append(new_row)

    
    voxelised_templates_DF = pd.DataFrame(processed_templates)


    return voxelised_templates_DF
    
   

if __name__ == "__main__":
    voxel_size = 0.25

    # Load existing voxelised templates
    #combined_templates = pd.read_pickle('data/revised/trees/combined_templateDF.pkl')
    combined_templates = pd.read_pickle('data/revised/trees/edited_combined_templateDF.pkl')

    voxelised_templates_DF = process_trees(combined_templates, voxel_size=voxel_size, resetCount=True)

    
    
    ##SAVE OUTPUTS
    """# Check that output directory exists, create if not
    output_dir = Path('data/revised/trees') 
    output_dir.mkdir(parents=True, exist_ok=True)

    #elm_output_name = f'{voxel_size}_elm_voxel_templateDF.pkl'
    outputName = f'{voxel_size}_combined_voxel_templateDF.pkl'
    output_path = output_dir / outputName
    voxelised_templates_DF.to_pickle(output_path)
    print(voxelised_templates_DF.head())
    print(f'Voxelized templates dataframe saved at {output_path}')"""

    print(f'done')