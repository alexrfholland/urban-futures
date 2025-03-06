import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


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

if __name__ == "__main__":
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
    selectedTreeIDs = [12,13]
    # Create a subset of fileNameDic that only includes the selected TreeIDs
    selected_fileNameDic = {tree_id: filename for tree_id, filename in fileNameDic.items() if tree_id in selectedTreeIDs}
    print(f"Selected trees: {selected_fileNameDic}")

for tree_id, filename in selected_fileNameDic.items():
    print(f"Processing tree ID: {tree_id}, filename: {filename}")
    
    qsmFileName = f'{folderPath}/QSMs/{filename}_treeDF.csv'


    # Load your dataset (replace this with your actual data loading method)
    tree_df = pd.read_csv(qsmFileName)

    # Extract the first 1/5th of the branchIDs
    unique_branch_ids = tree_df['branch_id'].unique()
    subset_branch_ids = unique_branch_ids[:len(unique_branch_ids) // 5]
    subset_df = tree_df[tree_df['branch_id'].isin(subset_branch_ids)].copy()

    # Sort the subset by branch_id to ensure correct order
    subset_df = subset_df.sort_values(by=['branch_id', 'startx'])

    # Calculate the difference in startX between consecutive rows within the same branch_id
    subset_df['startx_diff'] = subset_df['startx'].diff().fillna(0)

    # Create a cluster where there's a jump greater than 1 in startX
    subset_df['cluster_id'] = (subset_df['startx_diff'].abs() > 1).cumsum()

    # Step 1: Group by cluster_id and find the lowest branch_id per cluster
    clusterDF = subset_df.groupby('cluster_id').agg(
        lowestBranchIDInCluster=('branch_id', 'min')
    ).reset_index()

    # Step 2: Merge to get the parent_branch_id of the lowest branch in each cluster
    clusterDF = clusterDF.merge(
        subset_df[['branch_id', 'parent_branch_id']], 
        left_on='lowestBranchIDInCluster', 
        right_on='branch_id', 
        how='left'
    ).drop(columns='branch_id')

    # Step 3: Build a KDTree for all branch positions (startx, starty, startz)
    all_branch_positions = subset_df[['startx', 'starty', 'startz']].values
    kdtree_all = cKDTree(all_branch_positions)

    # Get positions for the lowest branch in each cluster
    lowest_branch_positions = subset_df[subset_df['branch_id'].isin(
        clusterDF['lowestBranchIDInCluster']
    )][['startx', 'starty', 'startz']].values

    # Step 4: Query the KDTree to find the nearest neighbor for each lowest branch (excluding itself)
    distances_all, nearest_indices_all = kdtree_all.query(lowest_branch_positions, k=3)

    # Step 5: Assign ClusterParentID and ClusterParentClusterID, ensuring nearest neighbor is not within the same cluster
    ClusterParentID = []
    ClusterParentClusterID = []

    for i, (cluster_id, nearest_index_1, nearest_index_2) in enumerate(zip(clusterDF['cluster_id'], nearest_indices_all[:, 1], nearest_indices_all[:, 2])):
        # Get the nearest and second nearest branch_ids
        nearest_branch_id_1 = subset_df.iloc[nearest_index_1]['branch_id']
        nearest_branch_id_2 = subset_df.iloc[nearest_index_2]['branch_id']
        
        # Determine the cluster of the nearest neighbor
        nearest_cluster_id_1 = subset_df.loc[subset_df['branch_id'] == nearest_branch_id_1, 'cluster_id'].values[0]
        
        # If the nearest neighbor belongs to the same cluster, use the second nearest neighbor
        if nearest_cluster_id_1 != cluster_id:
            ClusterParentID.append(nearest_branch_id_1)
            ClusterParentClusterID.append(nearest_cluster_id_1)
        else:
            nearest_branch_id_2 = subset_df.iloc[nearest_index_2]['branch_id']
            nearest_cluster_id_2 = subset_df.loc[subset_df['branch_id'] == nearest_branch_id_2, 'cluster_id'].values[0]
            ClusterParentID.append(nearest_branch_id_2)
            ClusterParentClusterID.append(nearest_cluster_id_2)

    # Step 6: Add ClusterParentID and ClusterParentClusterID to cluster_with_parent dataframe
    clusterDF['ClusterParentID'] = ClusterParentID
    clusterDF['ClusterParentClusterID'] = ClusterParentClusterID

    # Display the resulting dataframe
    clusterDF[['cluster_id', 'lowestBranchIDInCluster', 'parent_branch_id', 'ClusterParentID', 'ClusterParentClusterID']]

    ### GET ORDER
    import pandas as pd
    import matplotlib.pyplot as plt

    #STEPS FOR GETTING ORDER:
    #1. INITIALISE ALL ORDERS TO -2
    #2. ASSIGN ClusterParentClusterID -1 TO MAIN STEM (ie. mainStemMask = cluster_ID == 0. cluster_with_parent.loc[mainStemMask, 'ClusterParentClusterID'] = -1)
    #3. Group df by ClusterParentClusterID. 
    #4. Order ClusterParentClusterID groups sequentially assending
    #5. The order column for a ClusterParentClusterID group is its position in the sorted assending ClusterParentClusterID group list

    # Step 1: Initialize the 'order' column in 'cluster_with_parent'
    clusterDF['order'] = -1  # Reset all orders to unassigned

    # Step 2: Explicitly assign order 0 to the main stem (cluster_id == 0)
    clusterDF.loc[clusterDF['cluster_id'] == 0, 'order'] = 0

    # Step 3: Assign orders incrementally based on parent-child relationships
    for current_order in range(1, clusterDF['cluster_id'].nunique()):
        # Find clusters where the parent cluster has already been assigned an order
        parent_clusters_with_order = clusterDF[clusterDF['order'] == current_order - 1]['cluster_id']
        clusters_to_assign = clusterDF[clusterDF['ClusterParentClusterID'].isin(parent_clusters_with_order)]
        
        # Assign the next order to these clusters
        clusterDF.loc[clusters_to_assign.index, 'order'] = current_order

    # Step 4: Use `map()` to directly map 'order' and 'ClusterParentClusterID' to 'subset_df'
    subset_df['order'] = subset_df['cluster_id'].map(clusterDF.set_index('cluster_id')['order'])
    subset_df['ClusterParentClusterID'] = subset_df['cluster_id'].map(clusterDF.set_index('cluster_id')['ClusterParentClusterID'])

    # Step 5: Verify by checking the number of rows where cluster_id == 0 and order == 0
    cluster_id_0_count = subset_df[subset_df['cluster_id'] == 0].shape[0]
    order_0_count = subset_df[subset_df['order'] == 0].shape[0]

    # Output the verification
    print(cluster_id_0_count, order_0_count)

    # Step 6: Plot the updated scatter plot with the correct order
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot colored by the 'order' column using a qualitative colormap
    sc = ax.scatter(subset_df['startx'], subset_df['starty'], subset_df['startz'], 
                    c=subset_df['order'], cmap='Set1', s=50)

    # Adding colorbar and labels
    plt.colorbar(sc, ax=ax, label='Order')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the updated plot
    plt.show()







    """# Generate a random distinct color for each cluster
    num_clusters = subset_df['cluster_id'].nunique()
    random_colors = np.random.rand(num_clusters, 3)  # Generate an array of random RGB values

    # Create a dictionary to map cluster_id to random colors
    color_map = {cluster_id: random_colors[i] for i, cluster_id in enumerate(subset_df['cluster_id'].unique())}

    # Assign colors based on cluster_id
    subset_df['color'] = subset_df['cluster_id'].map(color_map)

    # Plotting the subset with custom colors
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the subset with different clusters using random colors
    ax.scatter(subset_df['startx'], subset_df['starty'], subset_df['startz'], 
            c=subset_df['color'].tolist(), s=50)

    # Adding labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.show()"""
