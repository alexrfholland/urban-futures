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

##stats adjustment

def adjust_resource_quantities(voxelised_templates, resource_dic_path='data/revised/trees/resource_dicDF.csv'):
    """
    Adjust resource quantities in voxelized templates to match target percentages.
    
    Args:
        voxelised_templates (pd.DataFrame or str): Either a DataFrame of voxelized templates 
                                                 or a path to the pickle file
        resource_dic_path (str): Path to the resource dictionary CSV file
    
    Returns:
        pd.DataFrame: Updated voxelized templates DataFrame with adjusted resource quantities
    """
    # Handle input which could be either a DataFrame or a file path
    if isinstance(voxelised_templates, str):
        print(f"Loading voxelized templates from {voxelised_templates}")
        voxelised_templates = pd.read_pickle(voxelised_templates)
    else:
        print("Using provided voxelized templates DataFrame")
    
    print(f"Loading resource dictionary from {resource_dic_path}")
    resource_df = pd.read_csv(resource_dic_path)
    
    # Process each template
    for idx, row in voxelised_templates.iterrows():
        print(f"\nProcessing template {idx+1}/{len(voxelised_templates)}: precolonial={row['precolonial']}, "
              f"size={row['size']}, control={row['control']}, tree_id={row['tree_id']}")
        
        template = row['template']
        
        # Skip if template is empty
        if template.empty:
            print(f"Skipping empty template")
            continue
        
        # Get resource target percentages for this template configuration using efficient filtering
        mask = ((resource_df['precolonial'] == row['precolonial']) & 
                (resource_df['size'] == row['size']) & 
                (resource_df['control'] == row['control']))
        
        target_resources = resource_df[mask]
        
        # Skip if no matching resource targets found
        if target_resources.empty:
            print(f"No matching resource targets found for this configuration, skipping")
            continue
        
        # Identify resource columns in template - do this once
        resource_columns = [col for col in template.columns if col.startswith('resource_')]
        
        # Create stat columns vectorized if they don't exist
        stat_columns = [col.replace('resource_', 'stat_') for col in resource_columns]
        for resource_col, stat_col in zip(resource_columns, stat_columns):
            if stat_col not in template.columns:
                template[stat_col] = template[resource_col].copy()
        
        # Get canopy voxels (excluding fallen logs) - do this once
        fallen_log_col = 'resource_fallen log'
        if fallen_log_col in template.columns:
            canopy_mask = template[fallen_log_col] == 0
            canopy_indices = template.index[canopy_mask]
            total_canopy_voxels = len(canopy_indices)
        else:
            canopy_mask = pd.Series(True, index=template.index)
            canopy_indices = template.index
            total_canopy_voxels = len(template)
        
        if total_canopy_voxels == 0:
            print(f"No canopy voxels in template, skipping")
            continue
        
        # Process each resource type
        for resource_col in resource_columns:
            # Skip fallen log for canopy calculations
            if resource_col == 'resource_fallen log':
                continue
            
            # Extract resource name without prefix
            resource_name = resource_col.replace('resource_', '')
            stat_col = resource_col.replace('resource_', 'stat_')
            
            # Skip certain resources or configurations based on validation logic
            """if resource_name in ['other', 'peeling bark', 'perch branch', 'hollow', 'epiphyte', 'fallen log'] or \
               row['size'] in ['senescing'] or \
               row['control'] in ['improved-tree']:
                continue"""
            
            # Check if resource exists in target dictionary
            if resource_name not in target_resources.columns:
                continue
            
            # Get current stats - use boolean indexing more efficiently
            current_count = template.loc[canopy_indices, stat_col].sum()
            current_percentage = (current_count / total_canopy_voxels) * 100
            target_percentage = target_resources[resource_name].iloc[0]
            percentage_difference = current_percentage - target_percentage
            
            print(f"\nResource: {resource_name}")
            print(f"Current count: {current_count}, Current percentage: {current_percentage:.2f}%")
            print(f"Target percentage: {target_percentage:.2f}%")
            print(f"Difference: {percentage_difference:.2f}%")
            
            # Skip if close enough (within 1%)
            if abs(percentage_difference) <= 1.0:
                print(f"Current percentage is within 1% of target, no adjustment needed")
                continue
            
            # Calculate target count
            target_count = int(round((target_percentage / 100) * total_canopy_voxels))
            delta = target_count - current_count
            
            print(f"Need to {'add' if delta > 0 else 'remove'} {abs(delta)} voxels to reach target")
            
            # Create a mask for candidates
            if delta > 0:
                # Need to add more of this resource
                # Select candidates: voxels that don't already have this resource
                candidates_mask = (template.loc[canopy_indices, stat_col] == 0)
                candidates = canopy_indices[candidates_mask]
                
                if len(candidates) >= delta:
                    # Randomly select voxels to convert (more efficient method)
                    voxels_to_convert = np.random.choice(candidates, delta, replace=False)
                    template.loc[voxels_to_convert, stat_col] = 1
                    print(f"Added {delta} voxels with {resource_name}")
                else:
                    # Not enough candidates
                    template.loc[candidates, stat_col] = 1
                    print(f"WARNING: Could only add {len(candidates)} voxels, needed {delta}")
            else:
                # Need to remove some of this resource
                # Select candidates: voxels that currently have this resource
                candidates_mask = (template.loc[canopy_indices, stat_col] == 1)
                candidates = canopy_indices[candidates_mask]
                
                delta_abs = abs(delta)
                if len(candidates) >= delta_abs:
                    # Randomly select voxels to convert (more efficient method)
                    voxels_to_convert = np.random.choice(candidates, delta_abs, replace=False)
                    template.loc[voxels_to_convert, stat_col] = 0
                    print(f"Removed {delta_abs} voxels with {resource_name}")
                else:
                    # Not enough candidates
                    template.loc[candidates, stat_col] = 0
                    print(f"WARNING: Could only remove {len(candidates)} voxels, needed {delta_abs}")
            
            # Verify adjustment
            new_count = template.loc[canopy_indices, stat_col].sum()
            new_percentage = (new_count / total_canopy_voxels) * 100
            
            print(f"After adjustment: count = {new_count}, percentage = {new_percentage:.2f}%")
        
        # Update the template in the DataFrame
        voxelised_templates.at[idx, 'template'] = template
    
    return voxelised_templates


def generate_resource_stats(voxelised_templates, resource_dic_path='data/revised/trees/resource_dicDF.csv', 
                            output_dir='data/revised/trees'):
    """
    Generate statistics directly from the voxelized templates by comparing with target resources.
    
    Args:
        voxelised_templates (pd.DataFrame or str): Either a DataFrame of voxelized templates 
                                                 or a path to the pickle file
        resource_dic_path (str): Path to the resource dictionary CSV file
        output_dir (str): Directory to save summary files
    
    Returns:
        pd.DataFrame: Formatted statistics summary DataFrame
    """
    print("\nGenerating resource statistics...")
    
    # Handle input which could be either a DataFrame or a file path
    if isinstance(voxelised_templates, str):
        print(f"Loading voxelized templates from {voxelised_templates}")
        voxelised_templates = pd.read_pickle(voxelised_templates)
    else:
        print("Using provided voxelized templates DataFrame")
    
    # Load resource dictionary
    print(f"Loading resource dictionary from {resource_dic_path}")
    resource_df = pd.read_csv(resource_dic_path)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data structure for statistics
    stats_data = []
    
    # Process each template
    for idx, row in voxelised_templates.iterrows():
        template = row['template']
        
        # Skip if template is empty
        if template.empty:
            continue
        
        # Get resource target percentages for this template
        mask = ((resource_df['precolonial'] == row['precolonial']) & 
                (resource_df['size'] == row['size']) & 
                (resource_df['control'] == row['control']))
        
        target_resources = resource_df[mask]
        
        # Skip if no matching resource targets found
        if target_resources.empty:
            continue
        
        # Get resource columns
        resource_columns = [col for col in template.columns if col.startswith('resource_')]
        stat_columns = [col for col in template.columns if col.startswith('stat_')]
        
        # Get canopy voxels (excluding fallen logs)
        fallen_log_col = 'resource_fallen log'
        if fallen_log_col in template.columns:
            canopy_mask = template[fallen_log_col] == 0
            canopy_indices = template.index[canopy_mask]
            total_canopy_voxels = len(canopy_indices)
        else:
            canopy_indices = template.index
            total_canopy_voxels = len(template)
        
        if total_canopy_voxels == 0:
            continue
        
        # Process each resource type
        for resource_col in resource_columns:
            # Skip fallen log for canopy calculations
            if resource_col == 'resource_fallen log':
                continue
            
            # Extract resource name without prefix
            resource_name = resource_col.replace('resource_', '')
            
            # Skip certain resources or configurations based on validation logic
            """if (resource_name in ['other', 'peeling bark', 'perch branch', 'hollow', 'epiphyte', 'fallen log'] or
                row['size'] in ['senescing'] or 
                row['control'] in ['improved-tree']):
                continue"""
            
            # Check if resource exists in target dictionary and stat columns
            if resource_name not in target_resources.columns:
                continue
            
            stat_col = f"stat_{resource_name}"
            if stat_col not in template.columns:
                continue
            
            # Calculate resource counts and percentages
            resource_count = template.loc[canopy_indices, resource_col].sum()
            resource_pct = (resource_count / total_canopy_voxels) * 100
            
            stat_count = template.loc[canopy_indices, stat_col].sum()
            stat_pct = (stat_count / total_canopy_voxels) * 100
            
            target_pct = target_resources[resource_name].iloc[0]
            
            # Calculate differences
            resource_diff = abs(resource_pct - target_pct)
            stat_diff = abs(stat_pct - target_pct)
            
            # Determine if adjustment was needed and effective
            needed_adjustment = resource_diff > 1.0
            is_adjusted = abs(resource_count - stat_count) > 0
            
            # Add to statistics data
            stats_data.append({
                'precolonial': row['precolonial'],
                'size': row['size'],
                'control': row['control'],
                'tree_id': row['tree_id'],
                'resource': resource_name,
                'total_canopy_voxels': total_canopy_voxels,
                'original_count': int(resource_count),
                'original_percentage': resource_pct,
                'target_percentage': target_pct,
                'final_count': int(stat_count),
                'final_percentage': stat_pct,
                'needed_adjustment': needed_adjustment,
                'was_adjusted': is_adjusted,
                'improvement': stat_diff < resource_diff,
                'final_difference': stat_diff
            })
    
    # Create DataFrame from collected data
    stats_df = pd.DataFrame(stats_data)
    
    # Skip further processing if DataFrame is empty
    if stats_df.empty:
        print("No statistics data to summarize")
        return stats_df
    
    # Sort the DataFrame
    sort_columns = ['precolonial', 'size', 'control', 'tree_id', 'resource']
    stats_df = stats_df.sort_values(by=sort_columns)
    
    # Create formatted columns for display
    stats_df['original_count_fmt'] = stats_df['original_count'].apply(lambda x: f"{x:,}")
    stats_df['final_count_fmt'] = stats_df['final_count'].apply(lambda x: f"{x:,}")
    stats_df['original_percentage_fmt'] = stats_df['original_percentage'].apply(lambda x: f"{x:.2f}%")
    stats_df['target_percentage_fmt'] = stats_df['target_percentage'].apply(lambda x: f"{x:.2f}%")
    stats_df['final_percentage_fmt'] = stats_df['final_percentage'].apply(lambda x: f"{x:.2f}%")
    stats_df['final_difference_fmt'] = stats_df['final_difference'].apply(lambda x: f"{x:.2f}%")
    
    # Add status column
    def get_status(row):
        if not row['needed_adjustment']:
            return "No Adjustment Needed"
        elif row['was_adjusted'] and row['improvement']:
            return "Successfully Adjusted"
        elif row['was_adjusted'] and not row['improvement']:
            return "Adjusted (Partial)"
        else:
            return "Adjustment Failed"
    
    stats_df['status'] = stats_df.apply(get_status, axis=1)
    
    # Create display version with renamed columns
    display_columns = {
        'precolonial': 'Precolonial',
        'size': 'Size',
        'control': 'Control',
        'tree_id': 'Tree ID',
        'resource': 'Resource',
        'original_count_fmt': 'Original Count',
        'original_percentage_fmt': 'Original %',
        'target_percentage_fmt': 'Target %',
        'final_count_fmt': 'Final Count', 
        'final_percentage_fmt': 'Final %',
        'final_difference_fmt': 'Difference',
        'status': 'Status'
    }
    
    display_df = stats_df.rename(columns=display_columns)[list(display_columns.values())]

    
    # Print summary statistics
    print("\n=== RESOURCE STATISTICS SUMMARY ===")
    
    total_resources = len(stats_df)
    needed_adjustment = stats_df['needed_adjustment'].sum()
    actually_adjusted = stats_df['was_adjusted'].sum()
    improved = stats_df[stats_df['was_adjusted'] & stats_df['improvement']].shape[0]
    
    print(f"Total resources analyzed: {total_resources}")
    print(f"Resources needing adjustment: {needed_adjustment} ({needed_adjustment/total_resources*100:.1f}%)")
    print(f"Resources actually adjusted: {actually_adjusted} ({actually_adjusted/needed_adjustment*100:.1f}% of needed)")
    print(f"Successfully improved: {improved} ({improved/actually_adjusted*100:.1f}% of adjusted)")
    
    # Resource-specific statistics
    print("\nStatistics by resource type:")
    resource_stats = stats_df.groupby('resource').agg({
        'needed_adjustment': 'sum',
        'was_adjusted': 'sum',
        'improvement': lambda x: (x & stats_df.loc[x.index, 'was_adjusted']).sum(),
        'final_difference': 'mean',
        'resource': 'count'
    })
    
    resource_stats.columns = ['needed_adjustment', 'actually_adjusted', 'improved', 'avg_difference', 'total']
    
    for resource, row in resource_stats.iterrows():
        print(f"  {resource}:")
        print(f"    Total: {int(row['total'])}, Needed Adjustment: {int(row['needed_adjustment'])}")
        print(f"    Adjusted: {int(row['actually_adjusted'])}, Improved: {int(row['improved'])}")
        print(f"    Average difference from target: {row['avg_difference']:.2f}%")
    
    # Create a pivot table for resource percentages by tree type
    pivot_table = stats_df.pivot_table(
        index=['precolonial', 'size', 'control'],
        columns='resource',
        values=['original_percentage', 'target_percentage', 'final_percentage'],
        aggfunc='mean'
    ).round(2)
    
    
    return display_df



# ================================
# Main Function
# ================================

def process_trees(tree_templates_DF, voxel_size = 0.25, resetCount = False):

    # Create new DataFrame instead of modifying in place
    processed_templates = []
    
    # For each row, voxelize the template dataframe
    for _, row in tree_templates_DF.iterrows():
        # Print row information (excluding template column)
        row_info = row.drop('template').to_dict()
        print(f"Processing tree: {row_info}")
        
        originalTemplate = row['template']
        originalTemplate = aa_tree_helper_functions.verify_resources_columns(originalTemplate)
        print(f"Original template length: {len(originalTemplate)} points")
        
        voxelized_tree_df = assign_voxel_coordinates(originalTemplate, voxel_size)
        voxelized_tree_df = count_resources_by_voxel(voxelized_tree_df, resetCount)
        print(f"Voxelized template length: {len(voxelized_tree_df)} voxels")
        
        # Create a new row with the processed template
        new_row = row.copy()
        extractedVoxelDF = voxelized_tree_df[['x', 'y', 'z'] + aa_tree_helper_functions.resource_names()]
        extractedVoxelDF = aa_tree_helper_functions.create_resource_column(extractedVoxelDF)
        print(f"Extracted voxel DF sample:")
        print(extractedVoxelDF.head())
        new_row['template'] = extractedVoxelDF
        processed_templates.append(new_row)

    
    voxelised_templates_DF = pd.DataFrame(processed_templates)

    voxelised_templates_DF = adjust_resource_quantities(voxelised_templates_DF)

    adjustment_summary = generate_resource_stats(voxelised_templates_DF)


    return voxelised_templates_DF, adjustment_summary
    
###
#MAIN
###

if __name__ == "__main__":
    voxel_size = 1

    # Load existing voxelised templates
    #combined_templates = pd.read_pickle('data/revised/trees/combined_templateDF.pkl')
    combined_templates = pd.read_pickle('data/revised/trees/edited_combined_templateDF.pkl')

    voxelised_templates_DF, adjustment_summary = process_trees(combined_templates, voxel_size=voxel_size, resetCount=True)


    # Check that output directory exists, create if not
    output_dir = Path('data/revised/trees') 
    
    output_dir.mkdir(parents=True, exist_ok=True)

    #elm_output_name = f'{voxel_size}_elm_voxel_templateDF.pkl'
    outputName = f'{voxel_size}_combined_voxel_templateDF.pkl'
    output_path = output_dir / outputName
    voxelised_templates_DF.to_pickle(output_path)
    print(voxelised_templates_DF.head())
    print(f'Voxelized templates dataframe saved at {output_path}')

    summary_name = f'{voxel_size}_combined_voxel_adjustment_summary.csv'
    adjustment_summary.to_csv(output_dir / summary_name, index=False)
    print(f"\nAdjustment summary saved to {output_dir / summary_name}")

    print(f'done')