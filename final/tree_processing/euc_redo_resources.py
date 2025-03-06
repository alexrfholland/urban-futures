import pandas as pd
import pickle
from pathlib import Path
import json
import adTree_AssignResources
import pyvista as pv
import numpy as np
import aa_tree_helper_functions

def convertToPoly(voxelDF):
    """Convert DataFrame to PolyData for visualization."""
    points = voxelDF[['x', 'y', 'z']].values
    poly = pv.PolyData(points)

    # Add all columns as point data attributes
    for col in voxelDF.columns:
        if col not in ['x', 'y', 'z']:  # Skip coordinate columns
            poly.point_data[col] = voxelDF[col].values

    return poly

def load_files():
    """Load eucalypt templates and resource DataFrame."""
    print('Loading templates and resource DataFrame')
    
    # Define paths
    template_dir = Path('data/revised/trees')
    euc_path = template_dir / 'updated_tree_dict.pkl'
    
    # Load templates
    print(f'Loading euc templates from {euc_path}')
    euc_tree_templates = pickle.load(open(euc_path, 'rb'))
    
    # Load resource DataFrame
    resourceDFPath = 'data/revised/trees/resource_dicDF.csv'

    resourceDF = pd.read_csv(resourceDFPath)
    
    return euc_tree_templates, resourceDF


def assign_dead_branches(template, target_percentage=30, seed=42):
    """
    Assign dead branches to tree template based on cluster grouping.
    
    Args:
        template (pd.DataFrame): Tree template DataFrame
        target_percentage (float): Target percentage of voxels to be dead branches
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Updated template with dead branches assigned
    """
    np.random.seed(seed)
    print(f"\nAssigning dead branches (target: {target_percentage}%, seed: {seed})")
    
    # Exclude fallen log voxels from consideration
    eligible_mask = template['resource_fallen log'] != 1
    eligible_voxels = template[eligible_mask]
    
    # Calculate target number of voxels based on eligible voxels only
    total_voxels = len(eligible_voxels)
    target_voxels = int(np.ceil((target_percentage / 100) * total_voxels))
    converted_voxels = 0
    
    print(f"\nExcluded {len(template) - total_voxels} fallen log voxels")
    print(f"Working with {total_voxels} eligible voxels")
    
    def get_eligible_clusters(df, height_threshold, radius_threshold, branch_order_threshold):
        eligible_clusters = []
        cluster_groups = df.groupby('cluster')
        for cluster_id, cluster in cluster_groups:
            is_eligible = (
                (cluster['clusterOrder'].mean() > branch_order_threshold) and
                (cluster['z'].mean() > height_threshold) and
                (cluster['start_radius'].mean() < radius_threshold) and
                (cluster['resource_fallen log'] == 0).all()  # Extra check to be safe
            )
            if is_eligible:
                eligible_clusters.append(cluster_id)
        return eligible_clusters
    
    # Initial criteria
    min_height = 5
    max_radius = 0.3
    min_branch_order = 2
    
    print(f"\nInitial eligibility criteria:")
    print(f"- Minimum height: {min_height}")
    print(f"- Maximum radius: {max_radius}")
    print(f"- Minimum branch order: {min_branch_order}")
    
    # First round with strict criteria
    eligible_clusters = get_eligible_clusters(eligible_voxels, min_height, max_radius, min_branch_order)
    print(f"\nFound {len(eligible_clusters)} eligible clusters with initial criteria")
    
    # Process eligible clusters
    eligible_clusters = np.random.permutation(eligible_clusters)
    for cluster_id in eligible_clusters:
        cluster = eligible_voxels[eligible_voxels['cluster'] == cluster_id]
        template.loc[cluster.index, 'resource_dead branch'] = 1
        converted_voxels += len(cluster)
        print(f"Converted cluster {cluster_id}: {len(cluster)} voxels")
    
    # If under target, try relaxing criteria
    if converted_voxels < target_voxels:
        print(f"\nOnly converted {converted_voxels}/{target_voxels} voxels. Relaxing criteria...")
        
        # First relax height and radius
        min_height = 3
        max_radius = 0.4
        print(f"\nRelaxed criteria - Round 1:")
        print(f"- Minimum height lowered to: {min_height}")
        print(f"- Maximum radius increased to: {max_radius}")
        
        new_eligible = get_eligible_clusters(eligible_voxels, min_height, max_radius, min_branch_order)
        new_eligible = [c for c in new_eligible if c not in eligible_clusters]
        print(f"Found {len(new_eligible)} additional eligible clusters")
        
        new_eligible = np.random.permutation(new_eligible)
        for cluster_id in new_eligible:
            cluster = eligible_voxels[eligible_voxels['cluster'] == cluster_id]
            template.loc[cluster.index, 'resource_dead branch'] = 1
            converted_voxels += len(cluster)
            print(f"Converted cluster {cluster_id}: {len(cluster)} voxels")
        
        # If still under target, relax ALL criteria further
        if converted_voxels < target_voxels:
            min_height = 1
            max_radius = 10
            min_branch_order = 0
            print(f"\nRelaxed criteria - Round 2 (all criteria):")
            print(f"- Minimum height further lowered to: {min_height}")
            print(f"- Maximum radius further increased to: {max_radius}")
            print(f"- Minimum branch order lowered to: {min_branch_order}")
            
            final_eligible = get_eligible_clusters(eligible_voxels, min_height, max_radius, min_branch_order)
            final_eligible = [c for c in final_eligible if c not in eligible_clusters and c not in new_eligible]
            print(f"Found {len(final_eligible)} additional eligible clusters")
            
            final_eligible = np.random.permutation(final_eligible)
            for cluster_id in final_eligible:
                cluster = eligible_voxels[eligible_voxels['cluster'] == cluster_id]
                template.loc[cluster.index, 'resource_dead branch'] = 1
                converted_voxels += len(cluster)
                print(f"Converted cluster {cluster_id}: {len(cluster)} voxels")
    
    # Check if we're over target and adjust at the very end
    final_dead_branches = template['resource_dead branch'].sum()
    if final_dead_branches > target_voxels:
        excess = final_dead_branches - target_voxels
        print(f"\nNeed to remove {excess} excess voxels")
        
        # Get all clusters that were converted, in reverse order
        converted_mask = template['resource_dead branch'] == 1
        clusters_to_adjust = template[converted_mask]['cluster'].unique()
        
        # Start removing from the last clusters until we meet target
        for cluster_id in reversed(clusters_to_adjust):
            if excess <= 0:
                break
                
            cluster_mask = (template['cluster'] == cluster_id) & (template['resource_dead branch'] == 1)
            cluster_size = cluster_mask.sum()
            
            if cluster_size <= excess:
                # Remove entire cluster
                template.loc[cluster_mask, 'resource_dead branch'] = 0
                excess -= cluster_size
                print(f"Removed entire cluster {cluster_id}: {cluster_size} voxels")
            else:
                # Remove partial cluster
                cluster_df = template[cluster_mask]
                to_reset = cluster_df.sort_index(ascending=False).head(excess)
                template.loc[to_reset.index, 'resource_dead branch'] = 0
                print(f"Partially removed cluster {cluster_id}: {excess} voxels")
                excess = 0
    
    # Calculate final statistics
    final_dead_branches = template['resource_dead branch'].sum()
    actual_percentage = (final_dead_branches / total_voxels) * 100
    percentage_difference = abs(actual_percentage - target_percentage)
    
    print(f"\nFinal results:")
    print(f"- Target: {target_voxels} voxels ({target_percentage}%)")
    print(f"- Achieved: {final_dead_branches} voxels ({actual_percentage:.1f}%)")
    print(f"- Difference from target: {final_dead_branches - target_voxels} voxels")
    print(f"- Percentage difference: {percentage_difference:.1f}%")
    
    if percentage_difference > 5:
        raise ValueError(f"Failed to assign dead branches within 5% of target. "
                        f"Target was {target_percentage}%, achieved {actual_percentage:.1f}%")
    else:
        print(f"SUCCESS!!! assigned dead branches within 5% of target. "
              f"Target was {target_percentage}%, achieved {actual_percentage:.1f}%")
        return template
    

    
def process_template(template, key, resourceDic, seed=42):
    """Process individual template to add resource columns."""
    print(f'\n######Processing template for key: {key}######')
    
    # Create a copy to avoid modifying original
    template = template.copy()

    template = template.rename(columns={'_segment_ID': 'cluster_id'})

    template['branch_id'] = template.index

    print(f'number of rows in template: {len(template)}')
    print(f'columns are:')
    print(template.columns)
    print(f'number of clusters in template: {template["cluster_id"].nunique()}')

    template = template.rename(columns={'radius': 'start_radius'})
    #change any values <=0 to 0.001
    template.loc[template['start_radius'] <= 0, 'start_radius'] = 0.001

    print(f'min of start_radius is {template["start_radius"].min()}')

    #remove column 'resource_leaf cluster'
    # Create isTerminal column
    template['isTerminal'] = 0
    
    # Set isTerminal to True where _length_to_leave < 1
    template.loc[template['_length_to_leave'] < 1, 'isTerminal'] = 1

    # Cluster df by branch_id
    #find the average angle for each branch_id
    
    print("\nAssigning resources based on template type:")
    print(f"Resource values: {resourceDic}")
    
    # Add isSenescent column (required by some resource assignment functions)
    template['isSenescent'] = False

    # Add isValid column (required by some resource assignment functions)
    template['isValid'] = True

    #Add terminal branches where _length_to_leave < 1
    template['isTerminal'] = False
    template.loc[template['_length_to_leave'] < 1, 'isTerminal'] = True
    
    template['resource_fallen log'] = 0
    template.loc[template['resource'] == 'fallen log', 'resource_fallen log'] = 1

    template['resource_dead branch'] = 0

    if resourceDic['dead branch'] == 100:
        template['resource_dead branch'] = 0
    else:
        template = assign_dead_branches(template, resourceDic['dead branch'], seed)

    template.loc[template['resource_dead branch'] == 1, 'isSenescent'] = True

    template['resource_peeling bark'] = 0
    template = adTree_AssignResources.assign_peeling_bark(template, resourceDic['peeling bark'], 'resource_peeling bark', seed + 1)

    template['resource_perch branch'] = 0
    template.loc[template['resource'] == 'perch branch', 'resource_perch branch'] = 1

    #Further refine assignment based on angle IQR and radius
    angle_IQR = template.groupby('cluster_id')['angle'].quantile([0.25, 0.75]).unstack()
    clusters_to_assign = angle_IQR[
        (angle_IQR[0.75] < 20) & 
        (template.groupby('cluster_id')['start_radius'].mean() < 0.15)
    ].index

    template.loc[template['cluster_id'].isin(clusters_to_assign), 'resource_perch branch'] = 1
    print(f"Assigned perch branch based on angle IQR filtering and radius < 0.15")

    #assign all terminal branches to resource_perch branch
    template.loc[template['isTerminal'], 'resource_perch branch'] = 1

    #assign epiphytes and hollows
    template['resource_hollow'] = 0
    template['resource_epiphyte'] = 0

    print(f'assigning hollows and epiphytes')

    template = adTree_AssignResources.assign_hollows_and_epiphytes(
        template,
        resourceDic['hollow'],
        resourceDic['epiphyte'],
        'resource_hollow',
        'resource_epiphyte',
        seed=seed+1
    )


    template = template.rename(columns={'resource': 'old_resource_assignment'})

    template = aa_tree_helper_functions.verify_resources_columns(template)

    template = adTree_AssignResources.create_resource_column(template)

    template = aa_tree_helper_functions.verify_resources_columns(template)


    
    # Print resource statistics
    print("\nResource Statistics:")
    print(template['resource'].value_counts())
    
    # Add cluster-level statistics
    cluster_df = template.groupby('cluster_id').first().reset_index()
    print(f'\nCluster-level statistics:')
    print(f'Total number of clusters: {len(cluster_df)}')
    
    print('\nResource counts at cluster level:')
    print(cluster_df['resource'].value_counts())
    
    print('\nResource type counts at cluster level:')
    print(f'Hollows: {cluster_df["resource_hollow"].sum()}')
    print(f'Epiphytes: {cluster_df["resource_epiphyte"].sum()}')
    print(f'Peeling bark: {cluster_df["resource_peeling bark"].sum()}')
    print(f'Perch branches: {cluster_df["resource_perch branch"].sum()}')
    
    return template

def visualize_template(template, title):
    """Visualize template with resources."""
    if len(template) == 0:
        print(f"Warning: Empty template for {title}")
        return
        
    poly = convertToPoly(template)
    
    p = pv.Plotter()
    p.add_mesh(poly, scalars='resource', point_size=5, render_points_as_spheres=True)
    p.add_title(title)
    p.show()
    

def debug_keys(euc_templates, resourceDic):
    filepath = Path('data/revised/final/debug')
    # Create empty dataframe for resourceDic keys
    resource_keys_df = pd.DataFrame(columns=[
        'is_precolonial', 'size', 'control', 'improvement'
    ])

    # Iterate through resourceDic and break down keys
    resource_keys_list = []
    for key in resourceDic.keys():
        is_precolonial, size, control, improvement = key
        resource_keys_list.append({
            'is_precolonial': is_precolonial,
            'size': size,
            'control': control, 
            'improvement': improvement
        })
    resource_keys_df = pd.concat([resource_keys_df, pd.DataFrame(resource_keys_list)], ignore_index=True)

    # Save resource keys to CSV
    resource_keys_df.to_csv(filepath / 'resource_dict_keys.csv', index=False)
    print("Saved resource dictionary keys to resource_dict_keys.csv")

    # Create empty dataframe for euc_templates keys
    euc_keys_df = pd.DataFrame(columns=[
        'precolonial', 'size', 'control', 'tree_id'
    ])

    # Iterate through euc_templates and break down keys
    euc_keys_list = []
    for key in euc_templates.keys():
        precolonial, size, control, tree_id = key
        euc_keys_list.append({
            'precolonial': precolonial,
            'size': size,
            'control': control,
            'tree_id': tree_id
        })
    euc_keys_df = pd.concat([euc_keys_df, pd.DataFrame(euc_keys_list)], ignore_index=True)

    # Save eucalypt template keys to CSV  
    euc_keys_df.to_csv(filepath / 'eucalypt_template_keys.csv', index=False)
    print("Saved eucalypt template keys to eucalypt_template_keys.csv")




def redoResources(euc_templates, resourceDF):
    print('Redoing resources...')

    # Filter for precolonial only
    resourceDF = resourceDF[resourceDF['precolonial'] == True]
    print(f"\nWorking with {len(resourceDF)} precolonial resource combinations")

    # Create dictionary for updated templates and lists for logging
    updated_templates = {}
    skipped_keys = []
    used_resources = set()

    # Process each template
    for key, template in euc_templates.items():
        precolonial, size, control, tree_id = key
        
        # Skip non-precolonial templates
        if not precolonial:
            reason = "Template is not precolonial"
            skipped_keys.append({'key': key, 'reason': reason})
            updated_templates[key] = template
            continue

        # Skip propped templates
        if size in ['propped']:
            reason = f"Template is {size}"
            skipped_keys.append({'key': key, 'reason': reason})
            updated_templates[key] = template
            continue
        
        # Get the resource row for this template
        mask = (resourceDF['size'] == size) & \
               (resourceDF['control'] == control)
        
        resource_row = resourceDF[mask]
        
        if len(resource_row) == 0:
            reason = f"No matching resource found for precolonial {size} {control}"
            skipped_keys.append({'key': key, 'reason': reason})
            updated_templates[key] = template
            continue
        
        # Process template with the found resource row
        print(f'\nProcessing template for: {key}')
        updated_template = process_template(template, key, resource_row.iloc[0])
        updated_templates[key] = updated_template
        print(f'added template for {key}')

        # Track which resource combination was used
        used_resources.add((size, control))

    # Find unused resource combinations from the precolonial-only resourceDF
    unused_resources = []
    for _, row in resourceDF.iterrows():
        combo = (row['size'], row['control'])
        if combo not in used_resources:
            unused_resources.append({
                'size': combo[0],
                'control': combo[1],
                'reason': 'No matching template found'
            })
    
    # Create DataFrames from our logging lists
    skipped_df = pd.DataFrame(skipped_keys)
    if len(skipped_keys) > 0:  # Only process if we have skipped keys
        skipped_df['precolonial'] = [k[0] for k in skipped_df['key']]
        skipped_df['size'] = [k[1] for k in skipped_df['key']]
        skipped_df['control'] = [k[2] for k in skipped_df['key']]
        skipped_df['tree_id'] = [k[3] for k in skipped_df['key']]
    
    unused_df = pd.DataFrame(unused_resources)
    
    # Create DataFrame of successfully converted combinations
    converted_resources = [
        {'size': size, 'control': control}
        for size, control in used_resources
    ]
    converted_df = pd.DataFrame(converted_resources)
    
    # Save all logs
    filepath = Path('data/revised/final/debug')
    filepath.mkdir(parents=True, exist_ok=True)
    
    skipped_df.to_csv(filepath / 'skipped_templates.csv', index=False)
    unused_df.to_csv(filepath / 'unused_resources.csv', index=False)
    converted_df.to_csv(filepath / 'converted_resources.csv', index=False)
    
    print('\nSkipped Templates Summary:')
    print(f'Total skipped: {len(skipped_keys)}')
    print('\nReasons for skipping:')
    print(skipped_df['reason'].value_counts())
    
    print('\nUnused Resources Summary:')
    print(f'Total unused: {len(unused_resources)}')
    if len(unused_resources) > 0:
        print('\nUnused combinations:')
        print(unused_df.groupby(['size', 'control']).size())
    
    print('\nConverted Resources Summary:')
    print(f'Total converted: {len(converted_resources)}')
    print('\nConverted combinations:')
    print(converted_df.groupby(['size', 'control']).size())
    
    print('\nDone!')
    return updated_templates

def check_and_add_keys(updatedEucTemplates):
    treeIds = [11,12,13,14,15,16]
    for treeId in treeIds:
        keySenescing = (True, 'senescing', 'reserve-tree', treeId)
        keySnag = (True, 'snag', 'reserve-tree', treeId)
        keyFallen = (True, 'fallen', 'reserve-tree', treeId)

        if keyFallen not in updatedEucTemplates.keys():
            print(f'keyFallen {keyFallen} not found')
            dupKey = (True, 'fallen', 'improved-tree', treeId)
            template = updatedEucTemplates[dupKey].copy()
            updatedEucTemplates[keyFallen] = template
            print(f'added fallen to {keyFallen}')

        if keySenescing not in updatedEucTemplates.keys():
            dupKey = (True, 'senescing', 'improved-tree', treeId)
            template = updatedEucTemplates[dupKey].copy()
            updatedEucTemplates[keySenescing] = template

        if keySnag not in updatedEucTemplates.keys():
            dupKey = (True, 'snag', 'improved-tree', treeId)
            template = updatedEucTemplates[dupKey].copy()
            updatedEucTemplates[keySnag] = template

    return updatedEucTemplates

if __name__ == "__main__":
        # Load files
    euc_templates, resoureceDF = load_files()

    

    #check if these keys exist:
    updatedEucTemplates = check_and_add_keys(euc_templates)


    updatedEucTemplates = redoResources(euc_templates, resoureceDF)
