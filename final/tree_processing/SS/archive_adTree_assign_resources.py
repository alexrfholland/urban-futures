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
from scipy.spatial import cKDTree
import aa_tree_helper_functions
###UTILITY FUNCTIONS###

def transform_elm_df(df):
    """
    Apply transformation to move the minimum Z point to (0,0,0) exactly,
    without centering based on mean X and Y.
    """
    # Rename original coordinates
    df.rename(columns={'x': 'origX', 'y': 'origY', 'z': 'origZ'}, inplace=True)

    # Find the coordinates of the point with the minimum Z value
    lowest_z_point = df.loc[df['origZ'].idxmin()]

    # Calculate the transformation to move this point to (0,0,0)
    transformX = -lowest_z_point['origX']
    transformY = -lowest_z_point['origY']
    transformZ = -lowest_z_point['origZ']

    # Save the transform columns to the DataFrame
    df['transformX'] = transformX
    df['transformY'] = transformY
    df['transformZ'] = transformZ


    # Apply the transformation to the DataFrame
    df['x'] = df['origX'] + transformX
    df['y'] = df['origY'] + transformY
    df['z'] = df['origZ'] + transformZ

    print(f'transformed elm df')

    return df



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

    voxelDF = transform_elm_df(voxelDF)
   
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
    print(f"counts: {voxelDF['resource_dead branch'].value_counts()}")
    print(f"types: {voxelDF['resource_dead branch'].dtype}")

    # Check if column contains object dtype
    if voxelDF['resource_dead branch'].dtype == 'object':
        print(f"ValueError: Column 'resource_dead branch' contains object dtype which is not supported")
        raise ValueError(f"Column 'resource_dead branch' contains object dtype which is not supported")

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
    voxelDF.loc[senescence_mask, 'resource_dead branch'] = 1


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
    voxelDF.loc[voxelDF['cluster_id'].astype(str).isin(perch_nodes), 'resource_perch branch'] = 1

    # Create subset df where pruning_mask is False AND isValid is True
    valid_voxelDF = voxelDF[(voxelDF['isValid'] == True) & (voxelDF['isPruned'] == False)]

    # Count how many voxels are in valid_voxelDF
    num_valid_voxels = len(valid_voxelDF)
    print(f"\nNumber of valid voxels: {num_valid_voxels}")

    # Create pyvista polydata from voxelDF coordinates
    """points = valid_voxelDF[['x', 'y', 'z']].values
    poly = pv.PolyData(points)

    # Add all columns as point data attributes
    for col in valid_voxelDF.columns:
        if col not in ['x', 'y', 'z']:  # Skip coordinate columns
            poly.point_data[col] = valid_voxelDF[col].values

            print(f'adding resource column {col}')
            #print values, counts and types of values per column
            print(f'counts: {valid_voxelDF[col].value_counts()}')
            print(f'types: {valid_voxelDF[col].dtype}')

            # Check if column contains object dtype
            if valid_voxelDF[col].dtype == 'object':
                print(f"ValueError: Column '{col}' contains object dtype which is not supported")
                raise ValueError(f"Column '{col}' contains object dtype which is not supported")

            print(f"Added column '{col}' as point data attribute to PolyData.")


    # Plot the polydata using isPruned as scalars
    plotter = pv.Plotter()
    plotter.add_mesh(poly, scalars='isSenescent', point_size=10, render_points_as_spheres=True)
    plotter.show()
    plotter.close()"""
    
    return voxelDF, valid_voxelDF


def get_tree_template(allTemplateResourceDic, treeID, voxelDF, clusterGraph):

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
    rows = []

    # Regular versions for each control type
    for control in controls:
        key = (precolonial, treeSizes[treeID], control, False)
        individualTreeResourceDic = allTemplateResourceDic[key]
        template = assign_resources(voxelDF, individualTreeResourceDic, key, seed=treeID)
        template = create_resource_column(template)
        rows.append({'precolonial': precolonial, 'size': treeSizes[treeID], 
                    'control': control, 'tree_id': treeID, 'template': template})

        # Senescing versions only for large reserve trees
        if treeSizes[treeID] == 'large' and control == 'reserve-tree':
            senescingKey = (False, 'senescing', 'reserve-tree', True)
            individualTreeResourceDic = allTemplateResourceDic[senescingKey]
            
            senescingTemplate = make_senscent_version(template, clusterGraph, seed=treeID+1)[1]
            senescingTemplate = assign_hollows_and_epiphytes(senescingTemplate, 
                individualTreeResourceDic['hollow'], 
                individualTreeResourceDic['epiphyte'], 
                "resource_hollow", "resource_epiphyte", 
                seed=treeID+3)
            senescingTemplate = assign_perch_branch(senescingTemplate, "resource_perch branch", seed=treeID+6)
            senescingTemplate = create_resource_column(senescingTemplate)
            rows.append({'precolonial': False, 'size': 'senescing', 
                        'control': control, 'tree_id': treeID, 'template': senescingTemplate})
            rows.append({'precolonial': False, 'size': 'senescing', 
                        'control': 'improved-tree', 'tree_id': treeID, 'template': senescingTemplate})

    # Improved version (using park-tree as base)
    improveKey = (precolonial, treeSizes[treeID], 'park-tree', True)
    individualTreeResourceDic = allTemplateResourceDic.get(improveKey)
    
    # Check if park-tree improved version exists
    if individualTreeResourceDic is None:
        # Check if reserve-tree or street-tree improved versions exist
        reserve_key = (precolonial, treeSizes[treeID], 'reserve-tree', True)
        street_key = (precolonial, treeSizes[treeID], 'street-tree', True)
        if reserve_key in allTemplateResourceDic or street_key in allTemplateResourceDic:
            raise ValueError(f"Improved version exists for reserve-tree or street-tree but not for park-tree. Tree ID: {treeID}")
    else:
        improved_template = assign_resources(voxelDF, individualTreeResourceDic, improveKey, seed=treeID)
        improved_template = create_resource_column(improved_template)
        rows.append({'precolonial': precolonial, 'size': treeSizes[treeID], 
                    'control': 'improved-tree', 'tree_id': treeID, 'template': improved_template})

    return rows


###RESOURCE ASSIGNMENT FUNCTIONS###
def assign_logs(voxelDF, logValue, logColumnName, control, size, seed=42):

    # Set numpy random seed for reproducibility
    np.random.seed(seed) 

    #Step 0: get log library
    logLibrary = pd.read_pickle('data/treeOutputs/logLibrary.pkl')

    print(f'first bit of log library is {logLibrary.head()}')
    # Group by logNo and count rows
    log_counts = logLibrary.groupby('logNo').size()
    
    # Filter to keep only logs with >= 200 points
    valid_logs = log_counts[log_counts >= 200].index
    
    # Filter logLibrary to only include valid logs
    logLibrary = logLibrary[logLibrary['logNo'].isin(valid_logs)]
    
    print(f'After filtering, {len(valid_logs)} logs remain')

    rng = np.random.RandomState(seed)


    # Initialize log column with 0s
    voxelDF[logColumnName] = 0

    # Step 1: Calculate the number of logs to generate
    noLogs = round(logValue)
    if noLogs > 0:
        logs = []
        new_rows = []
        print(f'noLogs is {noLogs}')

        for _ in range(noLogs):
            # Step 2: Create start points (x, y) coords randomly in a circle of radius 10
            radius = 10
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            start_point = np.array([x, y, 0])

            # Step 3: Create a direction vector, a random 2D vector in the x, y plane
            direction_angle = np.random.uniform(0, 2 * np.pi)
            direction_vector = np.array([np.cos(direction_angle), np.sin(direction_angle), 0.1])

            # Normalize the direction vector
            direction_vector /= np.linalg.norm(direction_vector)


            if control == 'reserve-tree' or control == 'improved-tree' or size == 'senescing': #either reserve tree or improved tree
                logSize = 'medium'
            if size in ['small','medium']:
                logSize = 'small'
            else:
                logSize = 'small'

            # Filter the log library by the selected log size
            included_logs = logLibrary[logLibrary['logSize'] == logSize]
            selected_logNo = rng.choice(included_logs['logNo'].unique())
            selected_log = included_logs[included_logs['logNo'] == selected_logNo]

            print(f'Selected logNo: {selected_logNo} with size: {logSize}')

            # Extract X, Y, Z coordinates
            log_points = selected_log[['X', 'Y', 'Z']].values

            print(f'log {selected_logNo} chosen with {selected_log.shape[0]} points')

            # Step 5: Apply transformation to the log points (translation and rotation)
            rotation_matrix = np.eye(3)
            rotation_matrix[:2, :2] = [[np.cos(direction_angle), -np.sin(direction_angle)],
                                    [np.sin(direction_angle),  np.cos(direction_angle)]]
            
            transformed_points = (log_points @ rotation_matrix.T) + start_point

            #print the centroid of the transformed fallen log points
            print(f'Centroid of transformed fallen log points: {np.mean(transformed_points, axis=0)}')

            logs.append(transformed_points)

            # Step 6: Add the transformed points to new rows with log column value of 1
            log_values = np.ones(transformed_points.shape[0])
            new_rows.append(np.column_stack((transformed_points, log_values)))

        if logs:
            all_points = np.vstack(logs)
            print(f"Fallen Log: Number of Logs: {noLogs}, Points Generated: {len(all_points)}")
        else:
            all_points = np.array([])

        if new_rows:
            new_rows = np.vstack(new_rows)

            #create a polydata from new_rows xyz columns
            #fallen_logs = pv.PolyData(new_rows[:, :3])
            #fallen_logs.plot()

            new_df = pd.DataFrame(new_rows, columns=['x', 'y', 'z', logColumnName])

            # Ensure X, Y, Z are numeric
            new_df[['x', 'y', 'z']] = new_df[['x', 'y', 'z']].apply(pd.to_numeric)

            # Copy the remaining columns from the template DataFrame
            for col in voxelDF.columns:
                if col not in new_df.columns:
                    new_df[col] = voxelDF[col].iloc[0]

            print(new_df)

            """voxelDF_polydata = pv.PolyData(new_df[['x', 'y', 'z']].values)
            voxelDF_polydata.plot()"""

        
            # Concatenate the new rows with the original DataFrame
            voxelDF = pd.concat([voxelDF, new_df], ignore_index=True)

            #convert voxelDF to polydata
            #voxelDF_polydata = pv.PolyData(voxelDF[['x', 'y', 'z']].values)
            #voxelDF_polydata.plot()

    else:
        print("No logs to generate.")

    print(f'final tree_template being returned is {voxelDF}')
    return voxelDF


def assign_hollows_and_epiphytes(voxelDF, hollowValue, epiphyteValue, columnNameHollow, columnNameEpiphyte, seed=42):
    """
    Assigns hollows and epiphytes to tree clusters based on various criteria.
    
    Args:
        voxelDF (pd.DataFrame): Voxel data with cluster information
        hollowValue (float): Number of hollow clusters to create
        epiphyteValue (float): Number of epiphyte clusters to create
        columnNameHollow (str): Name for hollow assignment column
        columnNameEpiphyte (str): Name for epiphyte assignment column
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Step 1: Initialize columns and check if assignment needed
    voxelDF[columnNameHollow] = 0
    voxelDF[columnNameEpiphyte] = 0
    
    hollowValue = round(hollowValue)
    epiphyteValue = round(epiphyteValue)
    
    if hollowValue == 0 and epiphyteValue == 0:
        print("No hollows or epiphytes to assign")
        return voxelDF
        
    # Step 2: Calculate cluster statistics
    cluster_stats = voxelDF.groupby('cluster_id').agg({
        'start_radius': ['mean', 'min'],
        'x': 'mean',
        'y': 'mean',
        'z': 'mean',
        'isSenescent': 'any',
        'resource_dead branch': 'any',
        'cluster_id': 'size'
    })
    
    # Flatten column names
    cluster_stats.columns = [
        'radius_mean', 'radius_min', 'centroid_x', 'centroid_y', 
        'centroid_z', 'is_senescent', 'is_dead', 'voxel_count'
    ]
    
    # Step 3: Filter clusters based on criteria
    height_threshold = cluster_stats['centroid_z'].max() * 0.3
    min_radius = cluster_stats['radius_mean'].mean() * 0.2
    min_size = cluster_stats['voxel_count'].mean() * 0.2
    
    base_mask = (
        (cluster_stats['centroid_z'] >= height_threshold) &
        (cluster_stats['radius_mean'] >= min_radius) &
        (cluster_stats['voxel_count'] >= min_size)
    )
    
    # Print filtering statistics
    print("\nCluster Filtering Statistics:")
    print(f"Total clusters: {len(cluster_stats)}")
    print(f"Clusters above height threshold ({height_threshold:.2f}): {(cluster_stats['centroid_z'] >= height_threshold).sum()}")
    print(f"Clusters above radius threshold ({min_radius:.2f}): {(cluster_stats['radius_mean'] >= min_radius).sum()}")
    print(f"Clusters above size threshold ({min_size:.0f}): {(cluster_stats['voxel_count'] >= min_size).sum()}")
    print(f"Clusters passing all filters: {base_mask.sum()}")
    
    # Step 4: Calculate weights for cluster selection
    def calculate_weights(stats_df, mask):
        """Calculate weights based on multiple factors"""
        if not mask.any():
            print("Warning: No clusters passed the filtering mask")
            return pd.Series(0, index=stats_df.index)
            
        weights = pd.Series(0.0, index=stats_df.index)
        filtered_stats = stats_df[mask]
        
        # Print diagnostic information
        print("\nWeight Calculation Diagnostics:")
        print(f"Total clusters: {len(stats_df)}")
        print(f"Clusters passing mask: {len(filtered_stats)}")
        
        # Calculate and check each factor separately
        radius_range = filtered_stats['radius_mean'].max() - filtered_stats['radius_mean'].min()
        height_range = filtered_stats['centroid_z'].max() - filtered_stats['centroid_z'].min()
        
        print("\nFactor ranges:")
        print(f"Radius range: {radius_range:.2f}")
        print(f"Height range: {height_range:.2f}")
        
        # Normalize each factor to 0-1 range with safety checks
        if radius_range > 1e-6:
            radius_factor = (filtered_stats['radius_mean'] - filtered_stats['radius_mean'].min()) / radius_range
        else:
            print("Warning: All clusters have same radius")
            radius_factor = pd.Series(1.0, index=filtered_stats.index)
            
        if height_range > 1e-6:
            height_factor = (filtered_stats['centroid_z'] - filtered_stats['centroid_z'].min()) / height_range
        else:
            print("Warning: All clusters at same height")
            height_factor = pd.Series(1.0, index=filtered_stats.index)
        
        # Weight factors (adjust these to change priorities)
        weights[mask] = (
            radius_factor * 0.4 +      # 40% weight for radius
            height_factor * 0.3 +      # 30% weight for height
            filtered_stats['is_senescent'] * 0.2 +  # 20% boost for senescent
            filtered_stats['is_dead'] * 0.1         # 10% boost for dead branches
        )
        
        # Print weight statistics
        print("\nWeight statistics:")
        print(f"Zero weights: {(weights == 0).sum()}")
        print(f"Non-zero weights: {(weights > 0).sum()}")
        print(f"Mean weight: {weights[weights > 0].mean():.4f}")
        print(f"Max weight: {weights.max():.4f}")
        
        # Print details of clusters with zero weights
        zero_weight_clusters = weights[weights == 0].index.tolist()
        if zero_weight_clusters:
            print("\nClusters with zero weights:")
            print(stats_df.loc[zero_weight_clusters])
        
        return weights


    
    # Step 5: Select clusters for hollows and epiphytes
    def select_clusters(weights, n_select, centroids, min_distance=2.0):
        """Select clusters with spatial spacing consideration using cKDTree."""
        if n_select == 0 or not weights.any():
            return []
        
        print("\nCluster Selection Diagnostics:")
        print(f"Number of clusters to select: {n_select}")
        print(f"Total clusters available: {len(weights)}")
        print(f"Clusters with non-zero weights: {(weights > 0).sum()}")
        
        points = centroids[['centroid_x', 'centroid_y', 'centroid_z']].values
        tree = cKDTree(points)
        curr_weights = weights.values
        
        # Try with progressively relaxed distance constraints
        original_distance = min_distance
        distance_reduction_factor = 0.75
        min_allowed_distance = 0.5  # Minimum acceptable distance between clusters
        
        while min_distance >= min_allowed_distance:
            print(f"\nTrying with minimum distance: {min_distance:.2f}")
            
            selected = []
            available_mask = np.ones(len(weights), dtype=bool)
            available_indices = np.arange(len(weights))
            
            while len(selected) < n_select and available_mask.any():
                curr_weights_masked = curr_weights[available_mask]
                valid_weights = curr_weights_masked > 0
                
                if not np.any(valid_weights):
                    print("No valid weights remaining with current constraints")
                    break
                    
                valid_indices = available_indices[available_mask][valid_weights]
                valid_weights_values = curr_weights_masked[valid_weights]
                normalized_weights = valid_weights_values / valid_weights_values.sum()
                
                selected_idx = np.random.choice(
                    valid_indices,
                    p=normalized_weights
                )
                
                if not selected:
                    selected.append(selected_idx)
                else:
                    nearby_points = tree.query_ball_point(
                        points[selected_idx], 
                        min_distance
                    )
                    
                    if not any(s in nearby_points for s in selected):
                        selected.append(selected_idx)
                
                available_mask[selected_idx] = False
            
            if len(selected) >= n_select:
                print(f"Successfully selected {len(selected)} clusters with minimum distance {min_distance:.2f}")
                selected_cluster_ids = centroids.index[selected].tolist()
                return selected_cluster_ids
                
            # If we couldn't select enough clusters, reduce the minimum distance
            min_distance *= distance_reduction_factor
            print(f"Could only select {len(selected)} clusters. Reducing minimum distance to {min_distance:.2f}")
        
        # If we still couldn't select enough clusters with minimum distance
        print("\nWarning: Could not select enough clusters even with reduced distance constraints")
        print("Attempting final selection with minimal spacing requirements...")
        
        # Final attempt with minimal spacing
        selected = []
        available_mask = weights > 0
        available_indices = np.arange(len(weights))
        
        while len(selected) < n_select and available_mask.any():
            valid_indices = available_indices[available_mask]
            valid_weights = weights[available_mask].values
            normalized_weights = valid_weights / valid_weights.sum()
            
            selected_idx = np.random.choice(
                valid_indices,
                p=normalized_weights
            )
            selected.append(selected_idx)
            available_mask[selected_idx] = False
        
        selected_cluster_ids = centroids.index[selected].tolist()
        print(f"Final selection: {len(selected_cluster_ids)} clusters: {selected_cluster_ids}")
        return selected_cluster_ids
    
    # Calculate centroids DataFrame
    centroids = cluster_stats[['centroid_x', 'centroid_y', 'centroid_z']]
    
    # Select clusters for hollows
    if hollowValue > 0:
        hollow_weights = calculate_weights(cluster_stats, base_mask)
        hollow_clusters = select_clusters(
            hollow_weights, 
            hollowValue, 
            centroids
        )
        
        # Assign hollows
        if hollow_clusters:
            voxelDF.loc[voxelDF['cluster_id'].isin(hollow_clusters), columnNameHollow] = 1
    
    # Select clusters for epiphytes
    if epiphyteValue > 0:
        epiphyte_weights = calculate_weights(cluster_stats, base_mask)
        epiphyte_clusters = select_clusters(
            epiphyte_weights, 
            epiphyteValue, 
            centroids
        )
        
        # Assign epiphytes
        if epiphyte_clusters:
            voxelDF.loc[voxelDF['cluster_id'].isin(epiphyte_clusters), columnNameEpiphyte] = 1
    
    # Step 6: Print final assignment statistics
    print("\nFinal Assignment Statistics:")
    print(f"Hollows requested: {hollowValue}")
    print(f"Hollows assigned: {len(voxelDF[voxelDF[columnNameHollow] == 1]['cluster_id'].unique())}")
    print(f"Epiphytes requested: {epiphyteValue}")
    print(f"Epiphytes assigned: {len(voxelDF[voxelDF[columnNameEpiphyte] == 1]['cluster_id'].unique())}")
    
    # If not enough assignments, try with relaxed criteria
    if (hollowValue > 0 and voxelDF[columnNameHollow].sum() == 0) or \
       (epiphyteValue > 0 and voxelDF[columnNameEpiphyte].sum() == 0):
        print("\nWarning: Not enough suitable clusters found with primary criteria")
        print("Attempting assignment with relaxed criteria...")
        
        # Relax criteria by 50%
        relaxed_mask = (
            (cluster_stats['centroid_z'] >= height_threshold * 0.5) &
            (cluster_stats['radius_mean'] >= min_radius * 0.5) &
            (cluster_stats['voxel_count'] >= min_size * 0.5)
        )
        
        # Repeat selection process with relaxed criteria
        # (Only for features that weren't fully assigned)
        if hollowValue > 0 and voxelDF[columnNameHollow].sum() == 0:
            hollow_weights = calculate_weights(cluster_stats, relaxed_mask)
            hollow_clusters = select_clusters(
                hollow_weights, 
                hollowValue, 
                centroids
            )
            if hollow_clusters:
                voxelDF.loc[voxelDF['cluster_id'].isin(hollow_clusters), columnNameHollow] = 1
        
        if epiphyteValue > 0 and voxelDF[columnNameEpiphyte].sum() == 0:
            epiphyte_weights = calculate_weights(cluster_stats, relaxed_mask)
            epiphyte_clusters = select_clusters(
                epiphyte_weights, 
                epiphyteValue, 
                centroids
            )
            if epiphyte_clusters:
                voxelDF.loc[voxelDF['cluster_id'].isin(epiphyte_clusters), columnNameEpiphyte] = 1
        
        print("\nFinal Assignment Statistics (after relaxation):")
        print(f"Hollows assigned: {len(voxelDF[voxelDF[columnNameHollow] == 1]['cluster_id'].unique())}")
        print(f"Epiphytes assigned: {len(voxelDF[voxelDF[columnNameEpiphyte] == 1]['cluster_id'].unique())}")

        
    
    # Add validation checks before return
    actual_hollows = len(voxelDF[voxelDF[columnNameHollow] == 1]['cluster_id'].unique())
    actual_epiphytes = len(voxelDF[voxelDF[columnNameEpiphyte] == 1]['cluster_id'].unique())
    
    if hollowValue > 0 and actual_hollows != hollowValue:
        raise ValueError(
            f"Final hollow count ({actual_hollows}) differs from target "
            f"({hollowValue}). Unable to assign requested number of hollows."
        )
    
    if epiphyteValue > 0 and actual_epiphytes != epiphyteValue:
        raise ValueError(
            f"Final epiphyte count ({actual_epiphytes}) differs from target "
            f"({epiphyteValue}). Unable to assign requested number of epiphytes."
        )
    
    return voxelDF

def select_spaced_clusters(n_select, weights, centroids, cluster_ids, min_distance=2.0):
    """Vectorized selection of spaced clusters with handling for identical weights"""
    if n_select == 0:
        return []
        
    selected_indices = []
    remaining_mask = np.ones(len(weights), dtype=bool)
    
    # Check if all weights are identical
    if np.allclose(weights, weights[0]):
        print("Warning: All weights are identical, using uniform probabilities")
        weights = np.ones_like(weights) / len(weights)
    
    while len(selected_indices) < n_select and remaining_mask.any():
        curr_weights = weights[remaining_mask]
        if len(curr_weights) == 0:
            print("No more valid clusters available")
            break
            
        curr_weights = curr_weights / curr_weights.sum()
        curr_centroids = centroids[remaining_mask]
        
        selected_idx = np.random.choice(len(curr_weights), p=curr_weights)
        actual_idx = np.where(remaining_mask)[0][selected_idx]
        
        if not selected_indices:
            selected_indices.append(actual_idx)
        else:
            selected_centroids = centroids[selected_indices]
            distances = np.linalg.norm(
                curr_centroids[selected_idx][None, :] - selected_centroids, 
                axis=1
            )
            
            if np.all(distances >= min_distance):
                selected_indices.append(actual_idx)
        
        remaining_mask[actual_idx] = False
        
    return [cluster_ids[idx] for idx in selected_indices]

def assign_dead_branches(voxel_df, deadValue, columnName, seed):
    # Initialize with 0 instead of False
    voxel_df[columnName] = 0

    # Set only numpy seed
    np.random.seed(seed)

    deadPercentage = deadValue/100

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
    voxel_df.loc[voxel_df['community_ancestors_threshold0'] == closest_group_name, columnName] = 1

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
                voxel_df.loc[voxel_df['community-leiden'] == leiden_group_name, columnName] = 1
                total_voxels_converted = voxel_df[columnName].sum()
                remaining_voxels_needed -= leiden_group_count
            else:
                # Use the seed for random sampling
                group_to_add = voxel_df[
                    (voxel_df['community-leiden'] == leiden_group_name) & 
                    (voxel_df[columnName] == 0)
                ].sample(n=int(remaining_voxels_needed), random_state=seed)
                voxel_df.loc[group_to_add.index, columnName] = 1
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
        rows_assigned_as_dead = voxel_df[voxel_df[columnName] == 1]
        rows_to_unassign = rows_assigned_as_dead.sort_values(by='branch_id').index[:int(excess_voxels)]
        voxel_df.loc[rows_to_unassign, columnName] = 0
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
    # Initialize with 0 instead of False
    voxelDF[columnName] = 0
    
    # Set only numpy seed
    np.random.seed(seed)

    peelingBarkPercentage = peelingBarkValue/100

    if peelingBarkPercentage < 0.01:
        print(f'peeling bark value is {peelingBarkPercentage}, skipping assignment')
        return voxelDF
    
    total_valid_voxels = voxelDF.shape[0]
    print(f"\nAssigning peeling bark...")
    print(f"Total valid voxels: {total_valid_voxels}")


    
    
    voxels_to_convert = total_valid_voxels * peelingBarkPercentage
    print(f"Target voxels to convert ({peelingBarkPercentage*100}%): {int(voxels_to_convert)}")


    # Get cluster stats with mean radius and senescence information
    cluster_stats = voxelDF.groupby('cluster_id').agg({
        'start_radius': 'mean',
        'isSenescent': 'any',  # True if any voxel in cluster is senescent
        'cluster_id': 'size'
    })
    cluster_stats.columns = ['mean_radius', 'is_senescent', 'voxel_count']
    
    # Create senescence multiplier (2x weight for senescent clusters)
    senescence_multiplier = np.where(cluster_stats['is_senescent'], 2.0, 1.0)
    
    # Use seed for random factors
    random_factors = np.random.uniform(0.8, 1.2, size=len(cluster_stats))
    
    # Combine radius, senescence, and random factors for final weights

    print(f'min of mean_radius is {cluster_stats["mean_radius"].min()}')


    weights = (cluster_stats['mean_radius'] * random_factors * senescence_multiplier)
    weights = weights / weights.sum()  # Normalize weights
    
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
    voxelDF.loc[voxelDF['cluster_id'].isin(clusters_to_convert), columnName] = 1
    
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
        converted_rows = voxelDF[voxelDF[columnName] == 1]
        
        # Sort by branch_id descending and select rows to unassign
        rows_to_unassign = converted_rows.sort_values(by='branch_id', ascending=False).index[:int(excess_voxels)]
        
        # Set selected rows back to False
        voxelDF.loc[rows_to_unassign, columnName] = 0
        print(f"Trimmed {excess_voxels} excess voxels")

    # Final count and statistics
    final_converted = voxelDF[columnName].sum()
    actual_percentage = (final_converted / total_valid_voxels) * 100
    target_percentage = peelingBarkValue
    percentage_difference = abs(actual_percentage - target_percentage)

    print(f"\nFinal results:")
    print(f"Final voxels converted: {final_converted}")
    print(f"Target was: {int(voxels_to_convert)}")
    print(f"Actual percentage: {actual_percentage:.2f}%")
    print(f"Target percentage: {target_percentage:.2f}%")
    print(f"Percentage difference: {percentage_difference:.2f}%")

    # Print distribution of True/False values
    value_counts = voxelDF[columnName].value_counts()
    print(f"\nDistribution of {columnName}:")
    print(value_counts)

    # Validate final percentage is within 5% of target
    if percentage_difference > 5:
        raise ValueError(
            f"Final percentage ({actual_percentage:.2f}%) differs from target "
            f"({target_percentage:.2f}%) by more than 5% "
            f"(difference: {percentage_difference:.2f}%)"
        )

    return voxelDF

def assign_perch_branch(voxelDF, columnName, seed):
    # Initialize with 0 instead of False
    voxelDF[columnName] = 0
    
    # Set only numpy seed
    np.random.seed(seed)
    
    # Group by cluster_id and find groups with isTerminalBranch == True
    terminal_groups = voxelDF[(voxelDF['isTerminalBranch'] == True) & 
                             (voxelDF['resource_dead branch'] == True)]['cluster_id'].unique()
    voxelDF.loc[voxelDF['cluster_id'].isin(terminal_groups), columnName] = 1
    #print unique values and counts of column
    print(f"\n{columnName} unique elements and counts:")
    print(voxelDF[columnName].value_counts())
    #Further refine assignment based on angle IQR and radius
    angle_IQR = voxelDF.groupby('cluster_id')['angle'].quantile([0.25, 0.75]).unstack()
    clusters_to_assign = angle_IQR[
        (angle_IQR[0.75] < 20) & 
        (voxelDF.groupby('cluster_id')['start_radius'].mean() < 0.15)
    ].index
    
    voxelDF.loc[voxelDF['cluster_id'].isin(clusters_to_assign), columnName] = 1
    print(f"Assigned perch branch based on angle IQR filtering and radius < 0.15")
    print(f"Final count for perch branch")
    print(voxelDF[columnName].value_counts())

    return voxelDF

def create_resource_column(voxelDF):
    # Define resource priorities (higher number = higher priority)
    resource_priorities = {
        'other': 0,
        'dead branch': 1, 
        'peeling bark': 2,
        'perch branch': 3,
        'epiphyte': 4,
        'fallen log': 5,
        'hollow': 6
    }
    
    # Initialize the 'resource' column with 'other'
    voxelDF['resource'] = 'other'
    
    # Get resource columns and map them to resource names
    resource_cols = [col for col in voxelDF.columns if col.startswith('resource_') and col != 'resource']
    col_to_resource = {col: col.split('resource_')[1] for col in resource_cols}
    
    # Verify the mappings before processing
    print("\nColumn to Resource Mapping:")
    print(col_to_resource)
    
    # Process columns in order of descending priority
    for col in sorted(resource_cols, key=lambda col: -resource_priorities[col_to_resource[col]]):
        resource_name = col_to_resource[col]
        # Assign the resource where the column is 1 and the current resource is still 'other'
        mask = (voxelDF[col] == 1) & (voxelDF['resource'] == 'other')
        voxelDF.loc[mask, 'resource'] = resource_name
        print(f"\nResource '{resource_name}' assigned to {mask.sum()} rows.")
    
    # Verify the result for specific critical resources
    for resource_name, col in col_to_resource.items():
        if resource_name in ['epiphyte', 'hollow']:
            mismatches = voxelDF[(voxelDF[col] == 1) & (voxelDF['resource'] != resource_name)]
            print(f"\nVerification for '{resource_name}':")
            print(f"Rows with {col}=1 but not assigned '{resource_name}': {len(mismatches)}")
    
    return voxelDF


def assign_resources(voxelDF,resource_dict, key, seed=42):
    validDF = voxelDF[voxelDF['isValid'] == True] 
    validDF['isSenescent'] = False
    # Pass the seed + an offset to each function to ensure different sequences
    print(f'assigning resources for precolonial: {key[0]}, size: {key[1]}, control: {key[2]}')

    control = key[2]
    size = key[1]
    print(f'resource dict is {resource_dict}')
    validDF = assign_dead_branches(validDF, resource_dict['dead branch'], "resource_dead branch", seed=seed)
    validDF = assign_peeling_bark(validDF, resource_dict['peeling bark'], "resource_peeling bark", seed=seed+1)
    validDF = assign_perch_branch(validDF, "resource_perch branch", seed=seed+2)
    validDF = assign_hollows_and_epiphytes(validDF, resource_dict['hollow'], resource_dict['epiphyte'], "resource_hollow", "resource_epiphyte", seed=seed+3)
    validDF = assign_logs(validDF, resource_dict['fallen log'], "resource_fallen log", control, size, seed=seed+4)

    #print values and counts for 'resource_fallen log'
    print(f"Final values and counts for 'resource_fallen log':")
    print(validDF['resource_fallen log'].value_counts())

    #do an if statement if resource_fallen log has any values == 1
    if validDF['resource_fallen log'].any() == 1:
        print(f"df has fallen logs")
        #make a pyvista polydata out of validDF xyz columns
        """fallen_logs = pv.PolyData(validDF[['x', 'y', 'z']].values)
        #create a fallen logs point_data column with the values of resource_fallen log
        fallen_logs['resource_fallen log'] = validDF['resource_fallen log']
        fallen_logs.plot(scalars='resource_fallen log') """


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

        """for row in rows:
            print(f"converting tree id: {row['tree_id']}, size: {row['size']}, control: {row['control']}, to polydata")
            convertToPoly(row, folderPath)"""

        all_rows.extend(rows)

        #nx.write_graphml(clusterGraph, f'{folderPath}/resources/{filename}_resourceGraph.graphml')
        #voxelDF.to_csv(f'{folderPath}/resources/{filename}_resourceVoxels.csv')
        #print(f'exported {filename} dataframe and resource graph')
        #senescentDF = make_senscent_version(voxelsDF, clusterGraph)

    df = pd.DataFrame(all_rows)
    aa_tree_helper_functions.check_for_duplicates(df)

    print("Saving DataFrame with columns:", df.columns)
    print("First row structure:", df.iloc[0])
    
    input_dir = 'data/revised/trees'
    name = 'elm_tree_dict.pkl'
    path = f"{input_dir}/{name}"

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    df.to_pickle(path)
    print(f'exported elm templates to {path}')

if __name__ == "__main__":
    main()

