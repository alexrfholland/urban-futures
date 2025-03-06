
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import pyvista as pv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import os




def showRandomPolyCols(tree_df, attributeID):
    # Step 1: Create the PolyData object
    treePoly = pv.PolyData(tree_df[['startx', 'starty', 'startz']].values)

    # Step 2: Transfer other columns as point data attributes
    columns_to_transfer = tree_df.columns.drop(['startx', 'starty', 'startz']).tolist()
    for col in columns_to_transfer:
        treePoly.point_data[col] = tree_df[col].values

    # Step 3: Generate random colors and assign directly using cluster_ids
    n_clusters = tree_df[attributeID].max() + 1  # Assuming cluster_ids start from 0
    cluster_colors = np.random.rand(n_clusters, 3)  # Random RGB colors
    color_array = cluster_colors[tree_df[attributeID]]

    # Step 4: Assign colors to the mesh and plot
    treePoly.point_data['colors'] = color_array
    plotter = pv.Plotter()
    plotter.add_mesh(treePoly, scalars='colors', rgb=True)
    plotter.add_title(f"{attributeID}")
    plotter.show()

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

# Function to propagate segment IDs based on starting node cluster IDs
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def update_tree_df(original_df, subset_df, unique_id='id', spatial_cols=['startx', 'starty', 'startz']):
    """
    Updates the original DataFrame with new columns from the subset DataFrame.
    For rows not in subset_df (isValid=False), initializes new columns by copying
    values from the nearest isValid=True row based on spatial coordinates.

    Parameters:
    - original_df (pd.DataFrame): The original DataFrame to be updated.
    - subset_df (pd.DataFrame): The subset DataFrame containing updated rows and new columns.
    - unique_id (str or list): The column name(s) used as the unique identifier.
    - spatial_cols (list): List of column names used for spatial comparison.

    Returns:
    - pd.DataFrame: The updated original DataFrame.
    """
    
    print("=== Starting DataFrame Update Process ===\n")
    
    # Step 1: Identify New Columns
    new_columns = [col for col in subset_df.columns if col not in original_df.columns and col not in unique_id]
    
    if new_columns:
        print(f"New columns identified from subset_df: {new_columns}")
        print("Initializing these new columns in original_df with default value -1.\n")
        for col in new_columns:
            original_df[col] = -1  # Initialize with -1
    else:
        print("No new columns to add from subset_df.\n")
    
    # Step 2: Ensure All Rows in subset_df Exist in original_df
    # Assuming unique_id is a single column for simplicity
    if isinstance(unique_id, list):
        merge_on = unique_id
    else:
        merge_on = [unique_id]
    
    merged = subset_df[merge_on].merge(original_df[merge_on], on=merge_on, how='left', indicator=True)
    missing_rows = merged[merged['_merge'] == 'left_only']
    
    if not missing_rows.empty:
        raise ValueError(f"Error: The following rows in subset_df are not present in original_df based on {unique_id}:")
        print(missing_rows)
    else:
        print("All rows in subset_df are present in original_df.\n")
    
    # Step 3: Update original_df with subset_df values for isValid=True rows
    print(f"Updating {len(subset_df)} rows in original_df with values from subset_df.\n")
    original_df.set_index(unique_id, inplace=True)
    subset_df.set_index(unique_id, inplace=True)
    
    # Update new_columns with subset_df values
    if new_columns:
        original_df.update(subset_df[new_columns])
    
    # Reset index to default
    original_df.reset_index(inplace=True)
    subset_df.reset_index(inplace=True)
    
    # Step 4: Handle isValid=False Rows Using cKDTree
    print("Handling rows with isValid=False by finding nearest isValid=True rows using cKDTree.\n")
    
    # Separate valid and invalid rows
    valid_rows = original_df[original_df['isValid'] == True].copy()
    invalid_rows = original_df[original_df['isValid'] == False].copy()
    
    if valid_rows.empty:
        print("Warning: No valid (isValid=True) rows found. All new columns for invalid rows remain as -1.\n")
    else:
        # Build cKDTree with valid rows
        print(f"Building cKDTree with {len(valid_rows)} valid rows based on spatial columns {spatial_cols}.")
        tree = cKDTree(valid_rows[spatial_cols].values)
        
        # Query nearest valid row for each invalid row
        print(f"Querying nearest neighbors for {len(invalid_rows)} invalid rows.")
        distances, indices = tree.query(invalid_rows[spatial_cols].values, k=1)
        
        # Assign values from nearest valid rows to invalid rows
        for col in new_columns:
            nearest_values = valid_rows.iloc[indices][col].values
            original_df.loc[original_df['isValid'] == False, col] = nearest_values
            print(f"Updated column '{col}' for invalid rows with values from nearest valid rows.")
        
        print("\nAll invalid rows have been updated with nearest valid row values.\n")
    
    print("=== DataFrame Update Process Completed ===\n")
    return original_df

def update_tree_df2(original_df, subset_df, unique_id='id'):
    """
    Updates the original DataFrame with new columns from the subset DataFrame.
    
    Parameters:
    - original_df (pd.DataFrame): The original DataFrame to be updated.
    - subset_df (pd.DataFrame): The subset DataFrame containing updated rows and new columns.
    - unique_id (str or list): The column name(s) used as the unique identifier.
    
    Returns:
    - pd.DataFrame: The updated original DataFrame.
    """
    
    # Ensure unique_id is a list for consistency
    if isinstance(unique_id, str):
        unique_id = [unique_id]
    
    # Identify new columns in subset_df that are not in original_df
    new_columns = [col for col in subset_df.columns if col not in original_df.columns and col not in unique_id]
    
    if new_columns:
        print(f"New columns to initialize in original_df with default value -1: {new_columns}")
        # Initialize new columns in original_df with -1
        for col in new_columns:
            original_df[col] = -1
    else:
        print("No new columns to initialize.")
    
    # Check if all rows in subset_df exist in original_df
    missing_ids = subset_df[unique_id].merge(original_df[unique_id], on=unique_id, how='left', indicator=True)
    missing_ids = missing_ids[missing_ids['_merge'] == 'left_only']
    if not missing_ids.empty:
        raise ValueError("Some rows in subset_df do not exist in original_df. Please check your data.")
    else:
        print("All rows in subset_df are present in original_df.")
    
    # Set unique_id as index for both DataFrames for alignment
    original_df.set_index(unique_id, inplace=True)
    subset_df.set_index(unique_id, inplace=True)
    
    # Update original_df with subset_df's new column values
    print(f"Updating {len(subset_df)} rows with new column values from subset_df.")
    original_df.update(subset_df[new_columns])
    
    # Reset the index to restore unique_id as a column
    original_df.reset_index(inplace=True)
    
    print("Update complete.")
    
    return original_df



#if __name__ == "__main__":
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
# Create a subset of fileNameDic that only includes the selected TreeIDs
selected_fileNameDic = {tree_id: filename for tree_id, filename in fileNameDic.items() if tree_id in selectedTreeIDs}
#fileNameDic = selected_fileNameDic

print(f"Selected trees: {selected_fileNameDic}")

processedTreeDFs = {}


for tree_id, filename in fileNameDic.items():
    print(f"Processing tree ID: {tree_id}, filename: {filename}")

    # Load data from the QSM file
    qsmFileName = f'{folderPath}/QSMs/{filename}_treeDF.csv'
    tree_df = pd.read_csv(qsmFileName, delimiter=',') 

    #Add other attributes
    tree_df['rowLength'] = np.linalg.norm(
        tree_df[['startx', 'starty', 'startz']].values - 
        tree_df[['endx', 'endy', 'endz']].values, axis=1
    )

    # Step 1: ## INITIAL CLUSTER LOGIC ###
    # Sort by branch_id and calculate the difference in startX
    tree_df = tree_df.sort_values(by=['branch_id', 'startx'])
    tree_df['startx_diff'] = tree_df['startx'].diff().fillna(0)

    # Create clusters where there's a jump greater than 0.2 in startX
    tree_df['original_cluster_id'] = (tree_df['startx_diff'].abs() > 0.2).cumsum()
    initial_num_groups = tree_df['original_cluster_id'].nunique()

    # Step 2: ### EXTRACT VALID CLUSTERS ###
    # Calculate the average start_radius per group and filter valid clusters
    average_start_radius_per_group = tree_df.groupby('original_cluster_id')['start_radius'].transform('mean')
    max_radius_per_group = tree_df.groupby('original_cluster_id')['start_radius'].transform('max')
    valid_groups_mask = (average_start_radius_per_group > 0.005) | (max_radius_per_group > 0.1)
    tree_df['isValid'] = valid_groups_mask
    subset_df = tree_df[tree_df['isValid']].copy()
    valid_num_groups = subset_df['original_cluster_id'].nunique()

    # Get the invalid original_cluster_id's as a list, order by ascending, and print
    invalid_clusters = tree_df[~tree_df['isValid']]['original_cluster_id'].unique()
    invalid_clusters_sorted = sorted(invalid_clusters)
    print(f"Number of invalid clusters: {len(invalid_clusters_sorted)}")

    # Step 3: ## FURTHER CLUSTER LOGIC ###
    # Break up large clusters into smaller clusters
    max_cluster_length = 1
    # Step 3: ## FURTHER CLUSTER LOGIC ###
    # Calculate total length for each cluster and identify only those exceeding max_cluster_length
    """subset_df['rowLength'] = np.linalg.norm(
        subset_df[['startx', 'starty', 'startz']].values - 
        subset_df[['endx', 'endy', 'endz']].values, axis=1
    )"""
    cluster_total_lengths = subset_df.groupby('original_cluster_id')['rowLength'].transform('sum')

    # Set segment count only for clusters where the total length exceeds the threshold
    num_segments_per_cluster = (cluster_total_lengths // max_cluster_length).where(cluster_total_lengths > max_cluster_length, 1).astype(int)

    # Apply segmentation logic conditionally based on the calculated segments
    subset_df['cluster_segment_no'] = (
        subset_df.groupby('original_cluster_id').cumcount() // 
        (subset_df.groupby('original_cluster_id')['branch_id'].transform('size') // num_segments_per_cluster + 1)
    ) + 1

    # Re-assign unique cluster_id based on original_cluster_id and cluster_segment_no
    subset_df['cluster_id'] = subset_df.groupby(['original_cluster_id', 'cluster_segment_no']).ngroup()

    # Verify uniqueness of cluster_id and confirm there are no NaNs
    nan_check_subset_df = subset_df.isna().sum().sum()  # Check for any NaN values
    total_original_clusters = subset_df['original_cluster_id'].nunique()
    total_final_clusters = subset_df['cluster_id'].nunique()
    new_clusters_added = total_final_clusters - total_original_clusters

    print(f'nan_check_subset_df: {nan_check_subset_df}')

    # Print clustering results
    print(f"Initial Clustering - Total Number of Groups (original_cluster_id): {initial_num_groups}")
    print(f"Number of Valid Groups (after filtering): {valid_num_groups}")
    print(f"Further Clustering - Total clusters originally: {total_original_clusters}")
    print(f"Total clusters after processing with max length {max_cluster_length}: {total_final_clusters}")
    print(f"Number of new clusters added: {new_clusters_added}")

    # Step 4: ### CLUSTER CONNECTIVITY ###
    # Ensure unique cluster_id in clusterDF and create connectivity
    # Update the code to drop duplicates by grouping on 'cluster_id' and then proceed with assigning required columns.

    # Group by 'cluster_id' and select rows with the minimum 'branch_id' to ensure one row per cluster_id

    # First sort subset_df by branch_id within each cluster_id group
    subset_df_sorted = subset_df.sort_values(['cluster_id', 'branch_id'])

    print(f'subset_df_sorted: {subset_df_sorted}')

    # Then take the first row of each cluster_id group to create clusterDF
    clusterDF = subset_df_sorted.groupby('cluster_id').first().reset_index()

    # Keep only the columns we want, with desired names
    clusterDF = clusterDF[['cluster_id', 'branch_id', 'startx', 'starty', 'startz', 
                        'start_radius', 'original_cluster_id', 'cluster_segment_no']]
    # TODO: avoid renaming columns
    clusterDF = clusterDF.rename(columns={
        'startx': 'startX',
        'starty': 'startY',
        'startz': 'startZ',
        'start_radius': 'startRadius'
    })

    # Verify the columns
    print("\nClusterDF columns:")
    print(clusterDF.columns.tolist())

    

    """clusterDF = subset_df.loc[subset_df.groupby('cluster_id')['branch_id'].idxmin()].copy()

    # Assign relevant columns to clusterDF
    clusterDF['startX'] = subset_df['startx']
    clusterDF['startY'] = subset_df['starty']
    clusterDF['startZ'] = subset_df['startz']
    clusterDF['startRadius'] = subset_df['start_radius']
    clusterDF['original_cluster_id'] = subset_df['original_cluster_id']
    clusterDF['cluster_segment_no'] = subset_df['cluster_segment_no']"""



    # Drop any duplicate entries to ensure uniqueness in clusterDF
    clusterDF = clusterDF.drop_duplicates(subset='cluster_id')

    # Confirm no duplicates in 'cluster_id' within clusterDF
    is_unique_cluster_id = clusterDF['cluster_id'].is_unique
    print(f"ClusterDF has unique cluster_id values: {is_unique_cluster_id}")

    # Implement KDTree with exclusion conditions
    max_distance = 1
    all_branch_positions = subset_df[['startx', 'starty', 'startz']].values
    kdtree_all = cKDTree(all_branch_positions)
    lowest_branch_positions = clusterDF[['startX', 'startY', 'startZ']].values

    print(f"\nStarting parent assignment process for {len(clusterDF)} clusters...")

    # Get all nearest neighbors at once
    distances_all, indices_all = kdtree_all.query(lowest_branch_positions, k=1000)

    # Initialize arrays for storing results with -1 (no parent)
    parent_cluster_ids = np.full(len(clusterDF), -1)
    parent_branch_ids = np.full(len(clusterDF), -1)

    # Vectorized operations for each cluster
    for i in range(len(clusterDF)):
        current_cluster_id = clusterDF.iloc[i]['cluster_id']
        
        # Skip assigning parent if this is the origin cluster (cluster_id == 0)
        if current_cluster_id == 0:
            print(f"Cluster {current_cluster_id}: Origin node - no parent assigned")
            continue
            
        # Get neighbor cluster IDs and their info
        neighbor_cluster_ids = subset_df.iloc[indices_all[i]]['cluster_id'].values
        neighbor_branch_ids = subset_df.iloc[indices_all[i]]['branch_id'].values
        neighbor_original_clusters = subset_df.iloc[indices_all[i]]['original_cluster_id'].values
        neighbor_segments = subset_df.iloc[indices_all[i]]['cluster_segment_no'].values
        
        # Get current cluster's info
        current_original_cluster = clusterDF.iloc[i]['original_cluster_id']
        current_segment = clusterDF.iloc[i]['cluster_segment_no']
        
        # Create masks for valid neighbors
        distance_mask = distances_all[i] <= max_distance
        different_id_mask = neighbor_cluster_ids != current_cluster_id
        same_original_cluster_mask = neighbor_original_clusters == current_original_cluster
        lower_segment_mask = neighbor_segments < current_segment
        lower_id_mask = neighbor_cluster_ids < current_cluster_id
        
        # Priority 1: Try to find parent from same original cluster with lower segment number
        valid_indices = np.where(different_id_mask & same_original_cluster_mask & lower_segment_mask)[0]
        if len(valid_indices) > 0:
            first_valid_idx = valid_indices[0]
            parent_cluster_ids[i] = neighbor_cluster_ids[first_valid_idx]
            parent_branch_ids[i] = neighbor_branch_ids[first_valid_idx]
            #print(f"Cluster {current_cluster_id}: Parent found from same original cluster (segment {neighbor_segments[first_valid_idx]}) - Cluster {parent_cluster_ids[i]} (distance: {distances_all[i][first_valid_idx]:.3f})")
        
        # Priority 2: Try to find parent with lower cluster_id within distance
        else:
            valid_indices = np.where(distance_mask & lower_id_mask)[0]
            if len(valid_indices) > 0:
                first_valid_idx = valid_indices[0]
                parent_cluster_ids[i] = neighbor_cluster_ids[first_valid_idx]
                parent_branch_ids[i] = neighbor_branch_ids[first_valid_idx]
                #print(f"Cluster {current_cluster_id}: Parent found with lower ID - Cluster {parent_cluster_ids[i]} (distance: {distances_all[i][first_valid_idx]:.3f})")
            
            # Priority 3: Backup - find any different cluster_id within distance
            else:
                valid_indices = np.where(distance_mask & different_id_mask)[0]
                if len(valid_indices) > 0:
                    first_valid_idx = valid_indices[0]
                    parent_cluster_ids[i] = neighbor_cluster_ids[first_valid_idx]
                    parent_branch_ids[i] = neighbor_branch_ids[first_valid_idx]
                    #print(f"Cluster {current_cluster_id}: Backup parent found - Cluster {parent_cluster_ids[i]} (distance: {distances_all[i][first_valid_idx]:.3f})")
                else:
                    print(f'WARNING!!: Cluster {current_cluster_id}: No parent found')
                
    # Update clusterDF with the parent assignments
    clusterDF['ParentClusterID'] = parent_cluster_ids
    clusterDF['ClusterParentID'] = parent_branch_ids

    # Create a mapping Series for fast lookup
    parent_cluster_mapping = pd.Series(parent_cluster_ids, index=clusterDF['cluster_id'])
    parent_branch_mapping = pd.Series(parent_branch_ids, index=clusterDF['cluster_id'])

    # Update subset_df using the same mappings
    subset_df['ParentClusterID'] = subset_df['cluster_id'].map(parent_cluster_mapping).fillna(-1).astype(int)
    subset_df['ClusterParentID'] = subset_df['cluster_id'].map(parent_branch_mapping).fillna(-1).astype(int)

    # Verification
    print("\nVerification:")
    print("ClusterDF assignments:")
    print("Origin node (cluster_id 0) parent:", clusterDF[clusterDF['cluster_id'] == 0]['ParentClusterID'].values[0])
    print("\nParent assignments for all clusters:")
    print(clusterDF[['cluster_id', 'ParentClusterID']].sort_values('cluster_id'))

    print("\nSubset_df statistics:")
    print(f"Total rows: {len(subset_df)}")
    print(f"Rows with assigned parents: {(subset_df['ParentClusterID'] != -1).sum()}")
    print(f"Rows without parents (including origin cluster): {(subset_df['ParentClusterID'] == -1).sum()}")
    print(f"Any NaN values: {subset_df['ParentClusterID'].isna().any() or clusterDF['ParentClusterID'].isna().any()}")


    # Build connectivity graph
    cluster_graph = nx.DiGraph()

    # First add all nodes from clusterDF
    cluster_graph.add_nodes_from(clusterDF['cluster_id'])

    # Then add edges, including those connecting to node 0 but excluding -1
    edges = list(zip(clusterDF['ParentClusterID'], clusterDF['cluster_id']))
    cluster_graph.add_edges_from([(parent, child) for parent, child in edges 
                                if parent != -1])  # Only exclude -1, keep 0
    cluster_graph.remove_edges_from(nx.selfloop_edges(cluster_graph))

    # Add attributes to each node
    for _, row in clusterDF.iterrows():
        if row['cluster_id'] in cluster_graph:  # Check if node exists
            # Add all attributes from clusterDF for this cluster_id
            nx.set_node_attributes(cluster_graph, {row['cluster_id']: row.to_dict()})
        else:
            print(f"Cluster {row['cluster_id']} not found in cluster_graph")

    # Verify the graph structure
    print("\nGraph Verification:")
    print(f"Number of nodes: {cluster_graph.number_of_nodes()}")
    print(f"Number of edges: {cluster_graph.number_of_edges()}")
    print(f"Node 0 in graph: {0 in cluster_graph.nodes()}")
    print(f"Node -1 in graph: {-1 in cluster_graph.nodes()}")
    print(f"Edges connecting to node 0: {list(cluster_graph.in_edges(0)) + list(cluster_graph.out_edges(0))}")

    # Check for disconnected components (ie. disconnected subgraphs)
    weakly_connected_components = list(nx.weakly_connected_components(cluster_graph))
    print(f"\nNumber of disconnected subgraphs (weakly connected components): {len(weakly_connected_components)}")
    for idx, component in enumerate(weakly_connected_components):
        print(f"Component {idx} size: {len(component)}")
        print(f"Component {idx} root nodes: {[node for node in component if cluster_graph.in_degree(node) == 0]}")

    # Print sample of node attributes
    print("\nSample node attributes:")
    sample_node = list(cluster_graph.nodes())[0]
    print(f"Attributes for node {sample_node}:")
    for key, value in cluster_graph.nodes[sample_node].items():
        print(f"{key}: {value}")

    # Check for cyclic loops
    cycles = list(nx.simple_cycles(cluster_graph))
    num_cycles = len(cycles)

    # Check for nodes with multiple parents
    multiple_parents = [node for node, degree in cluster_graph.in_degree() if degree > 1]

    # Check for isolated nodes
    isolated_nodes = list(nx.isolates(cluster_graph))

    


    # Print final validity checks
    print(f"Number of Cycles Detected: {num_cycles}")
    print(f"Sample Cycles (if any exist): {cycles[:5]}")
    print(f"Number of Nodes with Multiple Parents: {len(multiple_parents)}")
    print(f"Sample Nodes with Multiple Parents (if any): {multiple_parents[:10]}")
    print(f"Number of Isolated Nodes: {len(isolated_nodes)}")
    print(f"Sample Isolated Nodes (if any): {isolated_nodes[:10]}")


    #showRandomPolyCols(tree_df, 'original_cluster_id')
    #showRandomPolyCols(tree_df, 'cluster_segment_no')
    

    processedTreeDFs[tree_id] = tree_df

    filePath = 'data/revised/lidar scans/elm/adtree/processedQSMs'

    mergedDF = update_tree_df(tree_df, subset_df, 'branch_id')

    #make sure folder exists
    os.makedirs(filePath, exist_ok=True)
    mergedDF.to_csv(f'{filePath}/{fileNameDic[tree_id]}_clusteredQSM.csv', index=False)
    print(f"Saved to {filePath}/{fileNameDic[tree_id]}_clusteredQSM.csv")

    # Save the graph in a GraphML format (which supports node attributes)
    nx.write_graphml(cluster_graph, f'{filePath}/{fileNameDic[tree_id]}_clusterGraph.graphml')
    print(f"Saved to {filePath}/{fileNameDic[tree_id]}_clusterGraph.graphml")

    #showRandomPolyCols(subset_df, 'cluster_id')





