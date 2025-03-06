import pandas as pd
import pyvista as pv
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import networkx as nx
import pickle
import json

###UTILITY FUNCTIONS###


def convertToPoly(row, folderPath):

    voxelDF = row['template']

    points = voxelDF[['x', 'y', 'z']].values
    poly = pv.PolyData(points)

    # Add all columns as point data attributes
    for col in voxelDF.columns:
        if col not in ['x', 'y', 'z']:  # Skip coordinate columns
            poly.point_data[col] = voxelDF[col].values

    #poly.plot(scalars=treeResource_peeling bark', render_points_as_spheres=True)

    name = f"{row['precolonial']}_{row['size']}_{row['control']}_{row['tree_id']}"

    #make folderpath f{folderPath}/vtks/
    path = f"{folderPath}/vtks"
    if not os.path.exists(path):
        os.makedirs(path)
    
    #export polydata as a vtk file
    poly.save(f'{path}/{name}.vtk')

    print(f'exported poly to {path}/{name}.vtk')


def create_tree_id_filename_dict(point_cloud_files):
    """
    Creates a new dictionary with tree ID as key and processed filename as value.
    
    :param point_cloud_files: Dictionary with filename as key and tree ID as value
    :return: New dictionary with tree ID as key and processed filename as value
    """
    tree_id_filename_dict = {}
    for filename, tree_id in point_cloud_files.items():
        processed_filename = filename.split('_')[0]
        print(f"Tree ID: {tree_id}, Filename: {processed_filename}")
        tree_id_filename_dict[tree_id] = processed_filename
    return tree_id_filename_dict

def preprocessing_tree_resources(voxelDF, clusterGraph):
    # Print all attributes of first node in clusterGraph
    first_node = list(clusterGraph.nodes())[0]
    print("\nAttributes of first node in cluster graph:")
    for attr, value in clusterGraph.nodes[first_node].items():
        print(f"{attr}: {value}")
    
    # Count number of voxels per cluster_id without grouping the main dataframe
    print("Counting voxels per cluster...")
    voxelCount = voxelDF['cluster_id'].value_counts().reset_index()
    voxelCount.columns = ['cluster_id', 'voxelCount']
    
    # Add voxelCount to graph nodes by mapping cluster_ids
    print("Adding voxel counts to cluster graph nodes...")
    voxel_count_dict = {str(cluster_id): count for cluster_id, count in zip(voxelCount['cluster_id'], voxelCount['voxelCount'])}
    nx.set_node_attributes(clusterGraph, voxel_count_dict, 'voxelCount')
    
    # Get community attributes from graph
    first_node = list(clusterGraph.nodes())[0]
    community_attrs = [attr for attr in clusterGraph.nodes[first_node].keys() if attr.startswith('community')]
    print(f"Found {len(community_attrs)} community attributes: {', '.join(community_attrs)}")

    # Find all leaf nodes (nodes with no successors)
    leaf_nodes = [n for n in clusterGraph.nodes() if clusterGraph.out_degree(n) == 0]
    terminal_branch_list = [int(node) for node in leaf_nodes]  # Convert node IDs to integers
    print(f"Found {len(terminal_branch_list)} terminal branches")
    
    # Initialize terminal_branch column as False
    voxelDF['isTerminalBranch'] = False
    
    # Mark voxels belonging to terminal branches as True
    voxelDF.loc[voxelDF['cluster_id'].isin(terminal_branch_list), 'isTerminalBranch'] = True
    print(f"Marked {voxelDF['isTerminalBranch'].sum()} voxels as terminal branches")
    
    # Add voxelCount back to the main dataframe
    print("Adding voxel counts to main dataframe...")
    voxelDF = voxelDF.merge(voxelCount, on='cluster_id', how='left')
    
    # Transfer community attributes from clusterGraph to voxelDF
    print("Transferring community attributes from cluster graph to voxel dataframe...")
    for attr in community_attrs:
        # Convert cluster_id to string to match graph node ids
        attr_dict = {str(node): data[attr] for node, data in clusterGraph.nodes(data=True)}
        # Create new column in voxelDF for this attribute
        voxelDF[attr] = voxelDF['cluster_id'].astype(str).map(attr_dict)
    print(f"- Transferred {attr} attribute")
    return voxelDF, clusterGraph


###TEMPLATE FUNCTIONS###
def make_senscent_version(voxelDF, clusterGraph, seed=42):
    # Set only numpy seed
    np.random.seed(seed)
    
    # Initialize senescence columns as False
    voxelDF['isSenescent'] = False
    voxelDF['isPruned'] = False
    
    # Get unique community values and convert to long
    community_values = voxelDF['community_ancestors_threshold0'].astype('int64').unique()
    print(f"\nCommunity values found: {community_values}")

    # Randomly select 4/5 of communities to be senescent
    non_zero_communities = [c for c in community_values if c != 0]
    num_senescent = int(len(non_zero_communities) * 4/5)
    senescent_communities = np.random.choice(non_zero_communities, size=num_senescent, replace=False)
    print(f"\nSelected senescent communities: {senescent_communities}")

    # Create senescence mask based on community membership
    senescence_mask = voxelDF['community_ancestors_threshold0'].astype('int64').isin(senescent_communities)
    
    # Create pruning mask for small branches by traversing graph
    prune_list = set()  # Use a set for faster lookups
    visited = set()
    
    def should_prune_branch(node):
        """Check if a branch should be pruned based on radius"""
        radius = float(clusterGraph.nodes[node].get('startRadius', .5))
        should_prune = radius < 0.05
        return radius, should_prune
    
    def is_in_senescent_community(node):
        """Check if node is in a senescent community"""
        community = int(clusterGraph.nodes[node].get('community_ancestors_threshold0', -1))
        return community in senescent_communities
    
    # Get leaf nodes (nodes with no successors) that are in senescent communities
    leaf_nodes = [
        n for n in clusterGraph.nodes() 
        if clusterGraph.out_degree(n) == 0 and is_in_senescent_community(n)
    ]
    
    print(f"\nFound {len(leaf_nodes)} leaf nodes in senescent communities")
    
    # Process each leaf node and work upwards
    for leaf in leaf_nodes:
        if leaf in visited:
            continue
            
        #print(f"\nStarting new branch from leaf node: {leaf}")
        current_node = leaf
        continue_branch = True
        
        # Work up from leaf until we hit a node with radius >= threshold
        while continue_branch and current_node is not None:
            if current_node in visited:
                #print(f"Node {current_node} already visited, skipping branch")
                break
                
            visited.add(current_node)
            
            radius, should_prune = should_prune_branch(current_node)
            #print(f"Node {current_node}: radius = {radius:.6f} - {'FAIL' if should_prune else 'PASS'}")
            
            if should_prune:
                prune_list.add(current_node)
                # Get parent node if it exists and is in a senescent community
                parents = list(clusterGraph.predecessors(current_node))
                if parents and is_in_senescent_community(parents[0]):
                    current_node = parents[0]
                    #print(f"Moving to parent node: {current_node}")
                else:
                   # print("No valid parent node found, stopping branch")
                    current_node = None
            else:
               # print(f"Node {current_node} passed radius check, stopping branch")
                continue_branch = False

    print(f"\nNumber of clusters to prune: {len(prune_list)}")
    #print("Prune list:", sorted(list(prune_list)))

    # Create pruning mask based on cluster IDs in prune list
    pruning_mask = voxelDF['cluster_id'].astype(str).isin(prune_list)
    
    # Mark rows as senescent if they meet the conditions
    voxelDF.loc[pruning_mask, 'isPruned'] = True
    voxelDF.loc[senescence_mask, 'isSenescent'] = True
    voxelDF.loc[senescence_mask, 'resource_dead branch'] = True


    # Update clusterGraph with new node attributes
    # Group voxelDF by cluster_id and get isPruned and isSenescent values
    cluster_attributes = voxelDF.groupby('cluster_id').agg({
        'isPruned': 'first',  # Take first value since all voxels in cluster will have same value
        'isSenescent': 'first'
    }).to_dict()

    # Add attributes to clusterGraph nodes
    for node in clusterGraph.nodes():
        cluster_id = int(node)  # Convert node ID to int to match voxelDF cluster_id
        clusterGraph.nodes[node]['isPruned'] = cluster_attributes['isPruned'].get(cluster_id, False)
        clusterGraph.nodes[node]['isSenescent'] = cluster_attributes['isSenescent'].get(cluster_id, False)

    # Find nodes that meet perch branch conditions:
    # A) Senescent leaf nodes or B) Senescent nodes with pruned children
    perch_nodes = set()
    
    for node in clusterGraph.nodes():
        if clusterGraph.nodes[node]['isSenescent']:
            # Check if it's a leaf node (condition A)
            if clusterGraph.out_degree(node) == 0:
                perch_nodes.add(node)
            else:
                # Check if any children are pruned (condition B)
                children = list(clusterGraph.successors(node))
                if any(clusterGraph.nodes[child]['isPruned'] for child in children):
                    perch_nodes.add(node)

    # Mark perch branches in voxelDF
    voxelDF.loc[voxelDF['cluster_id'].astype(str).isin(perch_nodes), 'resource_perch branch'] = True

    # Create subset df where pruning_mask is False AND isValid is True
    valid_voxelDF = voxelDF[(voxelDF['isValid'] == True) & (voxelDF['isPruned'] == False)]

    # Count how many voxels are in valid_voxelDF
    num_valid_voxels = len(valid_voxelDF)
    print(f"\nNumber of valid voxels: {num_valid_voxels}")

    # Create pyvista polydata from voxelDF coordinates
    points = valid_voxelDF[['x', 'y', 'z']].values
    poly = pv.PolyData(points)

    # Add all columns as point data attributes
    for col in valid_voxelDF.columns:
        if col not in ['x', 'y', 'z']:  # Skip coordinate columns
            poly.point_data[col] = valid_voxelDF[col].values

    # Plot the polydata using isPruned as scalars
    """plotter = pv.Plotter()
    plotter.add_mesh(poly, scalars='isSenescent', point_size=10, render_points_as_spheres=True)
    plotter.show()
    plotter.close()
    """
    return voxelDF, valid_voxelDF


def get_tree_template(resourceDic, treeID, voxelDF, clusterGraph):

    # ------------------------------------------------------------------------------------------------------------
    # Structure and Keys of `tree_resources_dict`:
    # ------------------------------------------------------------------------------------------------------------
    # The `tree_resources_dict` is a dictionary where:
    # - The **keys** are tuples containing four elements:
    #   1. **is_precolonial**: A boolean value (`True` or `False`) indicating whether the tree is precolonial.
    #   2. **size**: A string representing the size of the tree (`'small'`, `'medium'`, or `'large'`).
    #   3. **control**: A string representing the control category (`'street-tree'`, `'park-tree'`, or `'reserve-tree'`).
    #   4. **improvement**: A boolean value (`True` or `False`) indicating whether the improvement logic has been applied.
    # 
    # - The **values** are dictionaries where:
    #   - The keys are resource names (`'peeling bark'`, `'dead branch'`, `'fallen log'`, `'leaf litter'`, `'hollow'`, `'epiphyte'`).
    #   - The values are the computed resource counts for that specific tree configuration.
    # ------------------------------------------------------------------------------------------------------------


    
    treeSizes = {
        4 : 'small',
        5 : 'small',
        6 : 'small',
        1 : 'medium',
        2 : 'medium',
        3 : 'medium',
        7 : 'large',
        9 : 'large',
        10 : 'large',
        11 : 'large',
        12 : 'large',
        13 : 'large',
        14 : 'large',
        -1 : 'senescent',
    }

    precolonial = False

    controls = ['street-tree', 'park-tree', 'reserve-tree']

    improvement = False

    rows = []

    print('resource dic is')
    print(resourceDic)

    for control in controls:
        key = (precolonial, treeSizes[treeID], control, improvement)
        
        treeResourceDic = resourceDic[key]

        template = assign_resources(voxelDF, treeResourceDic, key, seed=treeID)
        
        rows.append({'precolonial': precolonial, 'size': treeSizes[treeID], 'control': control, 'tree_id': treeID, 'template': template})

        
        if control == 'reserve-tree':
            #make senescent version from reserve tree
            voxelDF_versions = {}
            seed_offset = [1]
            for offset in seed_offset:
                voxelDF_versions[f'senescent_v{offset}'] = make_senscent_version(template, clusterGraph, seed=treeID+offset)[1]
            rows.append({'precolonial': False, 'size': 'senescing', 'control': control, 'tree_id': treeID, 'template': voxelDF_versions[f'senescent_v{offset}']})
            
            #for now, copy senescent version to make 'improved' version
            rows.append({'precolonial': False, 'size': 'senescing', 'control': 'improved-tree', 'tree_id': treeID, 'template': voxelDF_versions[f'senescent_v{offset}']})
            
            #make improved version from reserve tree
            rows.append({'precolonial': False, 'size': treeSizes[treeID], 'control': 'improved-tree', 'tree_id': treeID, 'template': voxelDF_versions[f'senescent_v{offset}']})

    return rows


###RESOURCE ASSIGNMENT FUNCTIONS###
def assign_dead_branches(voxel_df, deadValue, columnName, seed):
    # Set only numpy seed
    np.random.seed(seed)

    deadPercentage = deadValue/100

    voxel_df[columnName] = False

    if deadPercentage < 0.01:
        print(f'dead percentage value is {deadPercentage}, skipping assignment')
        return voxel_df
    
    voxel_df = voxel_df.copy()

    # Step 3: Find the total number of rows (valid voxels)
    total_valid_voxels = voxel_df.shape[0]
    print(f"Total valid voxels: {total_valid_voxels}")

    # Step 4: Calculate the number of voxels to convert (percentage of valid voxels)
    voxels_to_convert = total_valid_voxels * deadPercentage
    print(f"Voxels to convert {deadPercentage}%: {int(voxels_to_convert)}")

    # Step 5: Group the subset by community_ancestors_threshold0 and count rows per group
    threshold0_groups = voxel_df.groupby('community_ancestors_threshold0').size()

    # Step 6: Exclude group 0 and identify the group with the closest number of rows to the needed number to convert
    threshold0_groups_excluding_0 = threshold0_groups[threshold0_groups.index != 0]
    closest_group_name = threshold0_groups_excluding_0.sub(voxels_to_convert).abs().idxmin()
    closest_group_count = threshold0_groups_excluding_0[closest_group_name]
    print(f"Community_ancestors_threshold0 group selected: {closest_group_name} with {closest_group_count} voxels")

    # Assign rows from the closest group to the target column
    voxel_df.loc[voxel_df['community_ancestors_threshold0'] == closest_group_name, columnName] = True

    # Step 7: Recalculate the number of assigned voxels
    total_voxels_converted = voxel_df[columnName].sum()
    print(f"Total voxels converted so far: {total_voxels_converted}")

    # Step 8: If more voxels are needed, continue to assign from Leiden groups, but stop once the target is reached or exceeded
    remaining_voxels_needed = voxels_to_convert - total_voxels_converted
    print(f"Remaining voxels needed: {remaining_voxels_needed}")

    if remaining_voxels_needed > 0:
        leiden_groups = voxel_df.groupby('community-leiden').size()

        # Step 9: Identify valid Leiden groups with isTerminalBranch == True and not part of the community_ancestors_threshold0 group
        valid_leiden_groups = voxel_df[
            (voxel_df['isTerminalBranch'] == True) & 
            (voxel_df['community_ancestors_threshold0'] != closest_group_name)
        ].groupby('community-leiden').size()

        # Use the seed for random sampling
        for leiden_group_name, leiden_group_count in valid_leiden_groups.items():
            if remaining_voxels_needed <= 0:
                break

            if leiden_group_count <= remaining_voxels_needed:
                voxel_df.loc[voxel_df['community-leiden'] == leiden_group_name, columnName] = True
                total_voxels_converted = voxel_df[columnName].sum()
                remaining_voxels_needed -= leiden_group_count
            else:
                # Use the seed for random sampling
                group_to_add = voxel_df[
                    (voxel_df['community-leiden'] == leiden_group_name) & 
                    (voxel_df[columnName] == False)
                ].sample(n=int(remaining_voxels_needed), random_state=seed)
                voxel_df.loc[group_to_add.index, columnName] = True
                total_voxels_converted = voxel_df[columnName].sum()
                remaining_voxels_needed = 0
                break

    # Final count of assigned voxels
    total_voxels_converted = voxel_df[columnName].sum()
    print(f"Total voxels converted: {total_voxels_converted}")

    # Print unique values and counts of the column
    column_counts = voxel_df[columnName].value_counts()
    print(f"\n{columnName} unique elements and counts:")
    print(column_counts)

    # Final checks
    print(f"\nNumber of voxels that were meant to be converted: {int(voxels_to_convert)}")
    print(f"Difference: {int(voxels_to_convert) - int(total_voxels_converted)}")

    # Step 10: If over-converted, trim excess rows
    excess_voxels = total_voxels_converted - voxels_to_convert

    if excess_voxels > 0:
        rows_assigned_as_dead = voxel_df[voxel_df[columnName] == True]
        rows_to_unassign = rows_assigned_as_dead.sort_values(by='branch_id').index[:int(excess_voxels)]
        voxel_df.loc[rows_to_unassign, columnName] = False
        print(f"Trimmed {excess_voxels} excess voxels")

    # Final count of assigned voxels after trimming
    total_voxels_converted = voxel_df[columnName].sum()
    print(f"Total voxels converted after trimming: {total_voxels_converted}")

    # Print unique values and counts after trimming
    column_counts = voxel_df[columnName].value_counts()
    print(f"\n{columnName} unique elements and counts after trimming:")
    print(column_counts)

    # Final checks
    print(f"\nNumber of voxels that were meant to be converted: {int(voxels_to_convert)}")
    print(f"Difference after trimming: {int(voxels_to_convert) - int(total_voxels_converted)}")

    return voxel_df

def assign_peeling_bark(voxelDF, peelingBarkValue, columnName, seed):
    # Set only numpy seed
    np.random.seed(seed)

    voxelDF[columnName] = False
    print(f"Initialized {columnName} column with False values")
    peelingBarkPercentage = peelingBarkValue/100

    if peelingBarkPercentage < 0.01:
        print(f'peeling bark value is {peelingBarkPercentage}, skipping assignment')
        return voxelDF
    
    total_valid_voxels = voxelDF.shape[0]
    print(f"\nAssigning peeling bark...")
    print(f"Total valid voxels: {total_valid_voxels}")

    
    
    voxels_to_convert = total_valid_voxels * peelingBarkPercentage
    print(f"Target voxels to convert ({peelingBarkPercentage*100}%): {int(voxels_to_convert)}")


    # Get cluster stats with mean radius
    cluster_stats = voxelDF.groupby('cluster_id').agg({
        'start_radius': 'mean',
        'cluster_id': 'size'
    })
    cluster_stats.columns = ['mean_radius', 'voxel_count']
    
    # Use seed for random factors
    random_factors = np.random.uniform(0.8, 1.2, size=len(cluster_stats))
    weights = (cluster_stats['mean_radius'] * random_factors) / (cluster_stats['mean_radius'] * random_factors).sum()
    
    # Randomly select clusters with probability proportional to radius
    clusters_to_convert = []
    voxels_assigned = 0
    
    while voxels_assigned < voxels_to_convert and not cluster_stats.empty:
        # Select a cluster based on weights
        selected_cluster = cluster_stats.sample(n=1, weights=weights)
        cluster_id = selected_cluster.index[0]
        cluster_voxels = selected_cluster['voxel_count'].iloc[0]
        
        clusters_to_convert.append(cluster_id)
        voxels_assigned += cluster_voxels
        
        # Remove selected cluster and recalculate weights
        cluster_stats = cluster_stats.drop(cluster_id)
        if not cluster_stats.empty:
            random_factors = np.random.uniform(0.8, 1.2, size=len(cluster_stats))
            weights = (cluster_stats['mean_radius'] * random_factors) / (cluster_stats['mean_radius'] * random_factors).sum()
    
    print(f"Selected {len(clusters_to_convert)} clusters for conversion")

    # Assign True to all rows in selected clusters
    voxelDF.loc[voxelDF['cluster_id'].isin(clusters_to_convert), columnName] = True
    
    # Calculate how many voxels were converted
    total_converted = voxelDF[columnName].sum()
    print(f"\nInitial conversion results:")
    print(f"Total voxels converted: {total_converted}")
    print(f"Target was: {int(voxels_to_convert)}")
    print(f"Initial difference: {int(total_converted - voxels_to_convert)}")

    # If we converted too many, trim the excess
    excess_voxels = total_converted - voxels_to_convert
    if excess_voxels > 0:
        print(f"\nTrimming excess voxels...")
        # Get rows that were assigned as True
        converted_rows = voxelDF[voxelDF[columnName] == True]
        
        # Sort by branch_id descending and select rows to unassign
        rows_to_unassign = converted_rows.sort_values(by='branch_id', ascending=False).index[:int(excess_voxels)]
        
        # Set selected rows back to False
        voxelDF.loc[rows_to_unassign, columnName] = False
        print(f"Trimmed {excess_voxels} excess voxels")

    # Final count and statistics
    print(f"\nFinal results:")
    print(f"Final voxels converted: {voxelDF[columnName].sum()}")
    print(f"Target was: {int(voxels_to_convert)}")
    print(f"Final difference: {int(voxels_to_convert) -  voxelDF[columnName].sum()}")

    # Print distribution of True/False values
    value_counts = voxelDF[columnName].value_counts()
    print(f"\nDistribution of {columnName}:")
    print(value_counts)

    return voxelDF

def assign_perch_branch(voxelDF, columnName, seed):
    # Set only numpy seed
    np.random.seed(seed)
    
    voxelDF[columnName] = False
    
    # Group by cluster_id and find groups with isTerminalBranch == True
    terminal_groups = voxelDF[(voxelDF['isTerminalBranch'] == True) & 
                             (voxelDF['resource_dead branch'] == True)]['cluster_id'].unique()
    voxelDF.loc[voxelDF['cluster_id'].isin(terminal_groups), columnName] = True
    #print unique values and counts of column
    print(f"\n{columnName} unique elements and counts:")
    print(voxelDF[columnName].value_counts())
    #Further refine assignment based on angle IQR and radius
    angle_IQR = voxelDF.groupby('cluster_id')['angle'].quantile([0.25, 0.75]).unstack()
    clusters_to_assign = angle_IQR[
        (angle_IQR[0.75] < 20) & 
        (voxelDF.groupby('cluster_id')['start_radius'].mean() < 0.15)
    ].index
    
    voxelDF.loc[voxelDF['cluster_id'].isin(clusters_to_assign), columnName] = True
    print(f"Assigned perch branch based on angle IQR filtering and radius < 0.15")
    print(f"Final count for perch branch")
    print(voxelDF[columnName].value_counts())

    return voxelDF

def assign_resources(voxelDF,resource_dict, key, seed=42):
    validDF = voxelDF[voxelDF['isValid'] == True] 
    # Pass the seed + an offset to each function to ensure different sequences
    print(f'assigning resources for precolonial: {key[0]}, size: {key[1]}, control: {key[2]}')
    print(f'resource dict is {resource_dict}')
    validDF = assign_dead_branches(validDF, resource_dict['dead branch'], "resource_dead branch", seed=seed)
    validDF = assign_peeling_bark(validDF, resource_dict['peeling bark'], "resource_peeling bark", seed=seed+1)
    validDF = assign_perch_branch(validDF, "resource_perch branch", seed=seed+2)

    #convert all columns starting with resource to int
    """for col in validDF.columns:
        if col.startswith('resource'):
            validDF[col] = validDF[col].astype(int)"""

    print(f"Final resource df is {validDF.columns}")
    return validDF

###MAIN FUNCTION###

def main():
    resourceDicFilePath = 'data/treeOutputs/tree_resources.json'
    resourceDic = {eval(k): v for k, v in json.load(open(resourceDicFilePath)).items()}

    print('resource dic is')
    print(resourceDic)

    folderPath = 'data/revised/lidar scans/elm/adtree'
    point_cloud_files = {
        "Small A_skeleton.ply": 4,
        "Small B_skeleton.ply": 5,
        "Small C_skeleton.ply": 6,
        "Med A 1 mil_skeleton.ply": 1,
        "Med B 1 mil_skeleton.ply": 2,
        "Med C 1 mil_skeleton.ply": 3,
        "ElmL1_skeleton.ply": 7,
        "Elm L3_skeleton.ply": 9,
        "Elm L4_skeleton.ply": 10,
        "Elm L5_skeleton.ply": 11,
        "Large Elm A 1 mil_skeleton.ply": 12,
        "Large Elm B - 1 mil_skeleton.ply": 13,
        "Large Elm C 1 mil_skeleton.ply": 14
    }

    fileNameDic = create_tree_id_filename_dict(point_cloud_files)

    
    selectedTreeIDs = [12]
    selected_fileNameDic = {tree_id: filename for tree_id, filename in fileNameDic.items() if tree_id in selectedTreeIDs}
    #fileNameDic = selected_fileNameDic    

    
    all_rows = []
    for idx, (tree_id, filename) in enumerate(fileNameDic.items()):
        print(f"\nProcessing tree ID: {tree_id}, filename: {filename}")
        
        # Create a unique seed for this tree by combining a base seed with the loop index or tree_id
        tree_seed = 42 + tree_id  # or use idx if you prefer

        print(f"Loading voxel data from {folderPath}/elmVoxelDFs/{filename}_voxelDF.csv")
        voxelDF = pd.read_csv(f'{folderPath}/elmVoxelDFs/{filename}_voxelDF.csv')
        clusterGraph = nx.read_graphml(f'{folderPath}/processedGraph/{filename}_processedGraph.graphml')
        
        print(f'preprocessing {filename} and clustering the graph')
        voxelDF, clusterGraph = preprocessing_tree_resources(voxelDF, clusterGraph)

        print(f'getting all versions of the tree template for {filename}')
        rows = get_tree_template(resourceDic, tree_id, voxelDF, clusterGraph)

        for row in rows:
            print(f"converting tree id: {row['tree_id']}, size: {row['size']}, control: {row['control']}, to polydata")
            convertToPoly(row, folderPath)

        all_rows.extend(rows)

        #nx.write_graphml(clusterGraph, f'{folderPath}/resources/{filename}_resourceGraph.graphml')
        #voxelDF.to_csv(f'{folderPath}/resources/{filename}_resourceVoxels.csv')
        #print(f'exported {filename} dataframe and resource graph')
        #senescentDF = make_senscent_version(voxelDF, clusterGraph)

    df = pd.DataFrame(all_rows)
    print("Saving DataFrame with columns:", df.columns)
    print("First row structure:", df.iloc[0])
    path = f"{folderPath}/elm_dataframes"
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_pickle(f'{path}/elm_templates.pkl')
    print(f'exported elm templates')


main()
