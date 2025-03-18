import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import pyvista as pv
import aa_tree_helper_functions


RESOURCE_TYPES = {
        'other': 'object',
        'dead branch': 'percent', 
        'peeling bark': 'percent',
        'perch branch': 'object',
        'epiphyte': 'count',
        'fallen log': 'object',
        'hollow': 'count'
    }
    

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


########################################################
#PRECOLONIAL TEMPLATE REPLACEMENT
########################################################

def create_treeid_mapping(tree_templates_DF, target_sizes=None):
    """
    Creates a mapping from precolonial tree_ids to non-precolonial tree_ids
    based on the size of their template dataframes.
    
    Parameters:
        tree_templates_DF (pd.DataFrame): DataFrame containing tree templates
        target_sizes (list): List of sizes to filter by (e.g., ['snag', 'senescing'])
                           If None, all sizes are considered
        
    Returns:
        dict: Mapping from precolonial tree_ids to non-precolonial tree_ids
    """
    print(f"\n=== Creating Tree ID Mapping ===")
    
    # Filter by target sizes if specified
    if target_sizes:
        filtered_df = tree_templates_DF[tree_templates_DF['size'].isin(target_sizes)].copy()
        print(f"Filtered to {len(filtered_df)} rows with sizes {target_sizes}")
    else:
        filtered_df = tree_templates_DF.copy()
    
    # Split into precolonial and non-precolonial dataframes
    precolonial_df = filtered_df[filtered_df['precolonial'] == True]
    non_precolonial_df = filtered_df[filtered_df['precolonial'] == False]
    
    print(f"Found {len(precolonial_df)} precolonial and {len(non_precolonial_df)} non-precolonial templates")
    
    # Create a function to process each dataframe
    def process_df(df):
        # Add template size column
        df = df.copy()
        df['template_size'] = df['template'].apply(len)
        
        # Group by tree_id and take the first entry for each group
        grouped = df.groupby('tree_id').first().reset_index()
        
        # Sort by template size in descending order
        return grouped.sort_values('template_size', ascending=False)
    
    # Process both dataframes
    precolonial_processed = process_df(precolonial_df)
    non_precolonial_processed = process_df(non_precolonial_df)
    
    print(f"After grouping: {len(precolonial_processed)} unique precolonial tree_ids and "
          f"{len(non_precolonial_processed)} unique non-precolonial tree_ids")
    
    # Create mapping based on sorted index
    mapping = {}
    min_length = min(len(precolonial_processed), len(non_precolonial_processed))
    
    for i in range(min_length):
        pre_id = precolonial_processed.iloc[i]['tree_id']
        non_pre_id = non_precolonial_processed.iloc[i]['tree_id']
        mapping[pre_id] = non_pre_id
    
    print(f"Created mapping for {len(mapping)} tree_ids")
    
    # Print mapping for debugging
    print("\nTree ID mapping (precolonial -> non-precolonial):")
    for pre_id, non_pre_id in mapping.items():
        pre_size = precolonial_processed[precolonial_processed['tree_id'] == pre_id]['template_size'].iloc[0]
        non_pre_size = non_precolonial_processed[non_precolonial_processed['tree_id'] == non_pre_id]['template_size'].iloc[0]
        print(f"  {pre_id} (size: {pre_size}) -> {non_pre_id} (size: {non_pre_size})")
    
    return mapping

def replace_precolonial_templates(tree_templates_DF, target_sizes=None):
    """
    Replaces templates in precolonial trees with corresponding templates from 
    non-precolonial trees based on a mapping of tree_ids.
    
    Parameters:
        tree_templates_DF (pd.DataFrame): DataFrame containing tree templates
        target_sizes (list): List of sizes to consider for replacement (e.g., ['snag', 'senescing'])
                           If None, all sizes are considered
        
    Returns:
        pd.DataFrame: Updated DataFrame with replaced templates
    """
    print("\n=== Starting Precolonial Template Replacement ===")
    print(f"Input DataFrame contains {len(tree_templates_DF)} templates")
    
    # Create a copy to avoid modifying the original DataFrame
    df = tree_templates_DF.copy()
    
    # Create tree_id mapping
    tree_id_mapping = create_treeid_mapping(df, target_sizes)
    
    if not tree_id_mapping:
        print("No mappings created. No replacements will be performed.")
        return df
    
    # Filter to target sizes if specified
    if target_sizes:
        size_mask = df['size'].isin(target_sizes)
        replace_mask = df['precolonial'] == True
        full_mask = size_mask & replace_mask
        filtered_df = df[full_mask]
        print(f"Found {len(filtered_df)} precolonial templates to potentially replace")
    else:
        replace_mask = df['precolonial'] == True
        filtered_df = df[replace_mask]
        print(f"Found {len(filtered_df)} precolonial templates to potentially replace")
    
    # Count replacements
    replacement_count = 0
    
    # For each precolonial template that matches our criteria
    for idx, row in filtered_df.iterrows():
        pre_tree_id = row['tree_id']
        
        # Check if we have a mapping for this tree_id
        if pre_tree_id in tree_id_mapping:
            non_pre_tree_id = tree_id_mapping[pre_tree_id]
            
            # Find matching non-precolonial templates
            matching_templates = df[(df['precolonial'] == False) & 
                                    (df['tree_id'] == non_pre_tree_id) &
                                    (df['size'] == row['size'])]
            
            if not matching_templates.empty:
                # Get the first matching template
                template_to_use = matching_templates.iloc[0]['template']
                
                # Replace the template
                df.at[idx, 'template'] = template_to_use
                replacement_count += 1
                
                print(f"Replaced template at index {idx}: tree_id {pre_tree_id} -> {non_pre_tree_id}, "
                      f"size: {row['size']}, control: {row['control']}")
    
    print(f"\n=== Precolonial Template Replacement Complete ===")
    print(f"Replaced {replacement_count} templates")
    
    return df

########################################################
#RESOURCE STATS ADJUSTMENT
########################################################

def adjust_count_resource(template, resource_name, stat_col, canopy_indices, target_resources):
    """Handle resources that are pure counts"""
    current_count = template.loc[canopy_indices, stat_col].sum()
    # Round the target count to the nearest integer
    target_count = round(target_resources[resource_name].iloc[0])
    delta = target_count - current_count
    
    print(f"\nResource: {resource_name} (count)")
    print(f"Current count: {current_count}")
    print(f"Target count: {target_count}")
    print(f"Need to {'add' if delta > 0 else 'remove'} {abs(delta)} voxels")
    
    return delta

def adjust_percent_resource(template, resource_name, stat_col, canopy_indices, target_resources, total_canopy_voxels):
    """Handle resources that are percentages"""
    current_count = template.loc[canopy_indices, stat_col].sum()
    current_percentage = (current_count / total_canopy_voxels) * 100
    target_percentage = target_resources[resource_name].iloc[0]
    percentage_difference = current_percentage - target_percentage
    
    print(f"\nResource: {resource_name} (percent)")
    print(f"Current count: {current_count}, Current percentage: {current_percentage:.2f}%")
    print(f"Target percentage: {target_percentage:.2f}%")
    print(f"Difference: {percentage_difference:.2f}%")
    
    # Skip if close enough (within 1%)
    if abs(percentage_difference) <= 1.0:
        print(f"Current percentage is within 1% of target, no adjustment needed")
        return 0
    
    target_count = int(round((target_percentage / 100) * total_canopy_voxels))
    delta = target_count - current_count
    print(f"Need to {'add' if delta > 0 else 'remove'} {abs(delta)} voxels to reach target")
    
    return delta

def make_adjustment(template, delta, stat_col, canopy_indices):
    """Make the actual adjustment to the template"""
    if delta == 0:
        return
        
    if delta > 0:
        # Need to add more of this resource
        candidates_mask = (template.loc[canopy_indices, stat_col] == 0)
        candidates = canopy_indices[candidates_mask]
        
        if len(candidates) >= delta:
            voxels_to_convert = np.random.choice(candidates, delta, replace=False)
            template.loc[voxels_to_convert, stat_col] = 1
            print(f"Added {delta} voxels")
        else:
            template.loc[candidates, stat_col] = 1
            print(f"WARNING: Could only add {len(candidates)} voxels, needed {delta}")
    else:
        # Need to remove some of this resource
        candidates_mask = (template.loc[canopy_indices, stat_col] == 1)
        candidates = canopy_indices[candidates_mask]
        
        delta_abs = abs(delta)
        if len(candidates) >= delta_abs:
            voxels_to_convert = np.random.choice(candidates, delta_abs, replace=False)
            template.loc[voxels_to_convert, stat_col] = 0
            print(f"Removed {delta_abs} voxels")
        else:
            template.loc[candidates, stat_col] = 0
            print(f"WARNING: Could only remove {len(candidates)} voxels, needed {delta_abs}")

def adjust_resource_quantities(voxelised_templates, resource_dic_path='data/revised/trees/resource_dicDF.csv'):
    """Adjust resource quantities in voxelized templates to match target percentages."""
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
        
        # Ensure all stat columns exist by creating them from resource columns
        for resource_col in resource_columns:
            stat_col = resource_col.replace('resource_', 'stat_')
            if stat_col not in template.columns:
                print(f"Creating missing column: {stat_col}")
                template[stat_col] = template[resource_col].copy()
        
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
            stat_col = resource_col.replace('resource_', 'stat_')
            
            # Skip if resource doesn't exist in target dictionary
            if resource_name not in target_resources.columns:
                continue
            
            # Get resource type
            resource_type = RESOURCE_TYPES.get(resource_name, 'unknown')
            
            # Calculate delta based on resource type
            delta = 0
            if resource_type == 'count':
                delta = adjust_count_resource(template, resource_name, stat_col, canopy_indices, target_resources)
            elif resource_type == 'percent':
                delta = adjust_percent_resource(template, resource_name, stat_col, canopy_indices, target_resources, total_canopy_voxels)
            elif resource_type == 'object':
                print(f"\nSkipping {resource_name} (object type)")
                continue
            else:
                print(f"\nWarning: Unknown resource type for {resource_name}")
                continue
            
            # Make the adjustment
            make_adjustment(template, delta, stat_col, canopy_indices)
            
            # Verify adjustment
            new_count = template.loc[canopy_indices, stat_col].sum()
            if resource_type == 'percent':
                new_percentage = (new_count / total_canopy_voxels) * 100
                print(f"After adjustment: count = {new_count}, percentage = {new_percentage:.2f}%")
            else:
                print(f"After adjustment: count = {new_count}")
        
        # Update the template in the DataFrame
        voxelised_templates.at[idx, 'template'] = template
    
    return voxelised_templates


def generate_resource_stats(voxelised_templates, resource_dic_path='data/revised/trees/resource_dicDF.csv', 
                            output_dir='data/revised/trees'):
    """
    Generate statistics from the voxelized templates:
    1. Compares with target resources to evaluate adjustments
    2. Collects statistics for all 'stat_' resources regardless of targets
    
    Returns two DataFrames: adjustment summary and complete resource statistics
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
    
    ###########################################################
    ### PREPARE DATA STRUCTURES FOR BOTH SETS OF STATISTICS ###
    ###########################################################
    
    # Prepare data structures for statistics
    adjustment_stats_data = []
    all_resource_stats_data = []
    
    ###########################################################
    ### PROCESS EACH TEMPLATE TO COLLECT STATISTICS         ###
    ###########################################################
    
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
        
        # Get resource and stat columns
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
            
        ###########################################################
        ### COLLECT STATISTICS FOR ALL STAT RESOURCES          ###
        ###########################################################
        
        # Process each stat resource to collect complete statistics
        for stat_col in stat_columns:
            # Extract resource name without prefix
            resource_name = stat_col.replace('stat_', '')
            
            # Get resource type from RESOURCE_TYPES dictionary
            count_type = RESOURCE_TYPES.get(resource_name, 'unknown')
            
            # Calculate resource counts and percentages
            stat_count = template.loc[canopy_indices, stat_col].sum()
            stat_pct = (stat_count / total_canopy_voxels) * 100
            
            # Add to all resources statistics data
            all_resource_stats_data.append({
                'precolonial': row['precolonial'],
                'size': row['size'],
                'control': row['control'],
                'tree_id': row['tree_id'],
                'resource': resource_name,
                'countType': count_type,
                'count': int(stat_count),
                'percentage': stat_pct,
                'total_canopy_voxels': total_canopy_voxels
            })
        
        ###########################################################
        ### COLLECT ADJUSTMENT STATISTICS FOR TARGET RESOURCES  ###
        ###########################################################
        
        # Skip if no matching resource targets found
        if target_resources.empty:
            continue
        
        # Process each resource type for adjustment statistics
        for resource_col in resource_columns:
            # Skip fallen log for canopy calculations
            if resource_col == 'resource_fallen log':
                continue
            
            # Extract resource name without prefix
            resource_name = resource_col.replace('resource_', '')
            
            # Skip if resource doesn't exist in target dictionary
            if resource_name not in target_resources.columns:
                continue
            
            stat_col = f"stat_{resource_name}"
            if stat_col not in template.columns:
                continue
            
            # Get resource type
            resource_type = RESOURCE_TYPES.get(resource_name, 'unknown')
            
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
            
            # Add to adjustment statistics data
            adjustment_stats_data.append({
                'precolonial': row['precolonial'],
                'size': row['size'],
                'control': row['control'],
                'tree_id': row['tree_id'],
                'resource': resource_name,
                'countType': resource_type,
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
    
    ###########################################################
    ### CREATE AND FORMAT THE ADJUSTMENT STATISTICS         ###
    ###########################################################
    
    # Create DataFrame from collected adjustment data
    adjustment_stats_df = pd.DataFrame(adjustment_stats_data)
    
    # Skip further processing if DataFrame is empty
    if adjustment_stats_df.empty:
        print("No adjustment statistics data to summarize")
        adjustment_display_df = pd.DataFrame()
    else:
        # Sort the DataFrame
        sort_columns = ['precolonial', 'size', 'control', 'tree_id', 'resource']
        adjustment_stats_df = adjustment_stats_df.sort_values(by=sort_columns)
        
        # Create formatted columns for display
        adjustment_stats_df['original_count_fmt'] = adjustment_stats_df['original_count'].apply(lambda x: f"{x:,}")
        adjustment_stats_df['final_count_fmt'] = adjustment_stats_df['final_count'].apply(lambda x: f"{x:,}")
        adjustment_stats_df['original_percentage_fmt'] = adjustment_stats_df['original_percentage'].apply(lambda x: f"{x:.2f}%")
        adjustment_stats_df['target_percentage_fmt'] = adjustment_stats_df['target_percentage'].apply(lambda x: f"{x:.2f}%")
        adjustment_stats_df['final_percentage_fmt'] = adjustment_stats_df['final_percentage'].apply(lambda x: f"{x:.2f}%")
        adjustment_stats_df['final_difference_fmt'] = adjustment_stats_df['final_difference'].apply(lambda x: f"{x:.2f}%")
        
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
        
        adjustment_stats_df['status'] = adjustment_stats_df.apply(get_status, axis=1)
        
        # Create display version with renamed columns
        display_columns = {
            'precolonial': 'Precolonial',
            'size': 'Size',
            'control': 'Control',
            'tree_id': 'Tree ID',
            'resource': 'Resource',
            'countType': 'Count Type',
            'original_count_fmt': 'Original Count',
            'original_percentage_fmt': 'Original %',
            'target_percentage_fmt': 'Target %',
            'final_count_fmt': 'Final Count', 
            'final_percentage_fmt': 'Final %',
            'final_difference_fmt': 'Difference',
            'status': 'Status'
        }
        
        adjustment_display_df = adjustment_stats_df.rename(columns=display_columns)[list(display_columns.values())]
        
        # Explicitly list all columns to ensure proper ordering and inclusion
        column_order = [
            'Precolonial', 'Size', 'Control', 'Tree ID', 'Resource', 'Count Type',
            'Original Count', 'Original %', 'Target %', 'Final Count', 'Final %', 
            'Difference', 'Status'
        ]
        # Ensure all columns are included in the correct order
        adjustment_display_df = adjustment_display_df[column_order]
        
        print("Adjustment statistics sample:")
        print(adjustment_display_df.head())
    
    ###########################################################
    ### CREATE AND FORMAT THE COMPLETE RESOURCE STATISTICS  ###
    ###########################################################
    
    # Create DataFrame from all resource stats data
    all_stats_df = pd.DataFrame(all_resource_stats_data)
    
    # Skip further processing if DataFrame is empty
    if all_stats_df.empty:
        print("No resource statistics data to summarize")
    else:
        # Sort the DataFrame
        sort_columns = ['precolonial', 'size', 'control', 'tree_id', 'resource']
        all_stats_df = all_stats_df.sort_values(by=sort_columns)
        
        print("Complete resource statistics sample:")
        print(all_stats_df.head())
    
    return adjustment_display_df, all_stats_df

def generate_all_resource_stats(voxelised_templates, output_dir='data/revised/trees'):
    """
    Generate statistics for all 'stat_' resources in the voxelized templates.
    
    Parameters:
        voxelised_templates: DataFrame or path to pickle file with voxelized templates
        output_dir: Directory to save output statistics
        
    Returns:
        DataFrame with resource statistics for all stat resources
    """
    print("\nGenerating complete resource statistics...")
    
    # Handle input which could be either a DataFrame or a file path
    if isinstance(voxelised_templates, str):
        print(f"Loading voxelized templates from {voxelised_templates}")
        voxelised_templates = pd.read_pickle(voxelised_templates)
    else:
        print("Using provided voxelized templates DataFrame")
    
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
        
        # Get stat columns
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
        
        # Process each stat resource
        for stat_col in stat_columns:
            # Extract resource name without prefix
            resource_name = stat_col.replace('stat_', '')
            
            # Get resource type from RESOURCE_TYPES dictionary
            count_type = RESOURCE_TYPES.get(resource_name, 'unknown')
            
            # Calculate resource counts and percentages
            stat_count = template.loc[canopy_indices, stat_col].sum()
            stat_pct = (stat_count / total_canopy_voxels) * 100
            
            # Add to statistics data
            stats_data.append({
                'precolonial': row['precolonial'],
                'size': row['size'],
                'control': row['control'],
                'tree_id': row['tree_id'],
                'resource': resource_name,
                'countType': count_type,
                'count': int(stat_count),
                'percentage': stat_pct,
                'total_canopy_voxels': total_canopy_voxels
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
    
    return stats_df

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

    # PRECOLONIAL TEMPLATE REPLACEMENT
    print("\nReplacing precolonial snag and senescing templates with non-precolonial versions...")
    voxelised_templates_DF = replace_precolonial_templates(voxelised_templates_DF, target_sizes=['snag', 'senescing'])

    voxelised_templates_DF = adjust_resource_quantities(voxelised_templates_DF)

    adjustment_summary, all_resource_stats = generate_resource_stats(voxelised_templates_DF)

    return voxelised_templates_DF, adjustment_summary, all_resource_stats
    
###
#MAIN
###

if __name__ == "__main__":
    voxel_size = 1

    # Load existing voxelised templates
    #combined_templates = pd.read_pickle('data/revised/trees/combined_templateDF.pkl')
    combined_templates = pd.read_pickle('data/revised/trees/edited_combined_templateDF.pkl')

    voxelised_templates_DF, adjustment_summary, all_resource_stats = process_trees(combined_templates, voxel_size=voxel_size, resetCount=True)


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

    # Save the complete resource statistics
    all_stats_name = f'{voxel_size}_combined_voxel_all_resource_stats.csv'
    all_resource_stats.to_csv(output_dir / all_stats_name, index=False)
    print(f"\nComplete resource statistics saved to {output_dir / all_stats_name}")

    print(f'done')

