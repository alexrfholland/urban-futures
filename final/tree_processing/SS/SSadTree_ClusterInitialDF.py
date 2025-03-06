import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import pyvista as pv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import os

# Function to propagate segment IDs based on starting node cluster IDs
def assign_segments(clusterDF, cluster_graph, segment_conditions):
    """
    Function to assign segment IDs based on the starting node cluster IDs.
    
    :param clusterDF: DataFrame containing cluster information
    :param cluster_graph: Connectivity graph of clusters
    :param segment_conditions: Dictionary where the keys are segment names and the values are lists of starting cluster IDs
    """
    for segment_name, starting_clusters in segment_conditions.items():
        # Create the column for this segment and initialize as -1
        column_name = f'{segment_name}_id'
        clusterDF[column_name] = -1
        
        # Assign unique search IDs to starting clusters
        segment_id_map = {cluster_id: idx + 1 for idx, cluster_id in enumerate(starting_clusters)}
        clusterDF.loc[clusterDF['cluster_id'].isin(starting_clusters), column_name] = \
            clusterDF['cluster_id'].map(segment_id_map)
        
        # Propagate the search ID from each starting cluster to all its descendants
        propagate_search_id(cluster_graph, clusterDF, column_name)

# Function to propagate search ID through the connectivity graph
def propagate_search_id(graph, df, segment_column):
    for start_cluster in df.loc[df[segment_column] > 0, 'cluster_id'].unique():
        segment_id = df.loc[df['cluster_id'] == start_cluster, segment_column].values[0]
        descendants = nx.descendants(graph, start_cluster)
        df.loc[df['cluster_id'].isin(descendants), segment_column] = segment_id

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

def create_polydata_from_df(df):
    """
    Create PyVista PolyData from start and end positions in a DataFrame, 
    with thicker lines colored based on their branch_id.

    Args:
    - df (pandas.DataFrame): DataFrame containing 'startx', 'starty', 'startz', 'endx', 'endy', 'endz', and 'branch_id' columns.

    Returns:
    - pv.PolyData: PolyData object ready for plotting.
    """
    # Extract start and end points
    starts = df[['startx', 'starty', 'startz']].values
    ends = df[['endx', 'endy', 'endz']].values

    # Create a single array of points
    points = np.vstack((starts, ends))

    # Create lines array
    n_lines = len(df)
    lines = np.column_stack((
        np.full(n_lines, 2),  # Each line has 2 points
        np.arange(0, n_lines),  # Start indices
        np.arange(n_lines, 2*n_lines)  # End indices
    )).ravel()

    # Create PyVista PolyData object
    poly = pv.PolyData(points, lines=lines)

    # Add all columns from df as point_data attributes
    for column in df.columns:
        if column not in ['startx', 'starty', 'startz', 'endx', 'endy', 'endz']:
            poly.point_data[column] = np.repeat(df[column].values, 2)
    
    return poly



def plot_all_trees(treeDFs):
    polyList = []
    
    for tree_id, df in treeDFs.items():
        poly = create_polydata_from_df(df)
        polyList.append(poly)

    n_plots = len(polyList)
    n_cols = min(n_plots, 4)
    n_rows = (n_plots - 1) // 4 + 1
    
    p = pv.Plotter(shape=(n_rows, n_cols))
    p.link_views()
    
    for i, poly in enumerate(polyList):
        row = i // 4
        col = i % 4
        p.subplot(row, col)
        p.add_mesh(poly, scalars='order', cmap='turbo', line_width=8)
        p.camera_position = 'xy'
        p.reset_camera()
    
    p.show()


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
    #selectedTreeIDs = [12,13]
    # Create a subset of fileNameDic that only includes the selected TreeIDs
    #selected_fileNameDic = {tree_id: filename for tree_id, filename in fileNameDic.items() if tree_id in selectedTreeIDs}
    selected_fileNameDic = fileNameDic


    print(f"Selected trees: {selected_fileNameDic}")

    processedTreeDFs = {}

    for tree_id, filename in selected_fileNameDic.items():
        print(f"Processing tree ID: {tree_id}, filename: {filename}")
        
        qsmFileName = f'{folderPath}/QSMs/{filename}_treeDF.csv'

        tree_df = pd.read_csv(qsmFileName, delimiter=',')  # Assuming a comma delimiter

        ##CLUSTER LOGIC###
        # Step 1: Extract a subset (1/5th of branch IDs)
        unique_branch_ids = tree_df['branch_id'].unique()
        subset_branch_ids = unique_branch_ids[:len(unique_branch_ids) // 2]
        subset_df = tree_df[tree_df['branch_id'].isin(subset_branch_ids)].copy()

        # Step 2: Sort by branch_id and calculate the difference in startX
        subset_df = subset_df.sort_values(by=['branch_id', 'startx'])
        subset_df['startx_diff'] = subset_df['startx'].diff().fillna(0)

        # Step 3: Create clusters where there's a jump greater than 1 in startX
        subset_df['cluster_id'] = (subset_df['startx_diff'].abs() > 1).cumsum()

        ### CLUSTER CONNECTIVITY LOGIC ###

        # Step 4: Group by cluster_id to find the lowest branch_id per cluster
        clusterDF = subset_df.groupby('cluster_id').agg(
            lowestBranchIDInCluster=('branch_id', 'min')
        ).reset_index()

        # Step 5: Get parent_branch_id for the lowest branch in each cluster
        clusterDF['parent_branch_id'] = clusterDF['lowestBranchIDInCluster'].map(
            subset_df.set_index('branch_id')['parent_branch_id']
        )

        # Step 6: Build a KDTree for all branch positions (startx, starty, startz)
        all_branch_positions = subset_df[['startx', 'starty', 'startz']].values
        kdtree_all = cKDTree(all_branch_positions)

        # Get positions for the lowest branch in each cluster
        lowest_branch_positions = subset_df[subset_df['branch_id'].isin(
            clusterDF['lowestBranchIDInCluster']
        )][['startx', 'starty', 'startz']].values

        # Step 7: Query the KDTree to find the nearest neighbor for each lowest branch
        distances_all, nearest_indices_all = kdtree_all.query(lowest_branch_positions, k=3)

        # Step 8: Assign ClusterParentID and ClusterParentClusterID without a for loop
        nearest_branch_ids = subset_df.iloc[nearest_indices_all[:, 1]]['branch_id'].values
        nearest_cluster_ids = subset_df.iloc[nearest_indices_all[:, 1]]['cluster_id'].values

        # Assign ClusterParentID and ClusterParentClusterID using vectorized operations
        clusterDF['ClusterParentID'] = nearest_branch_ids
        clusterDF['ClusterParentClusterID'] = nearest_cluster_ids
        
        # Step 11: Update the 'order' column in clusterDF using this dictionary
        clusterDF['order'] = clusterDF['cluster_id'].map(cluster_order_dict)

       ### ORDER LOGIC ###
       
        # Step 9: Create the ClusterIDGroupArray by getting unique ClusterParentClusterID and sorting ascending
        ClusterIDGroupArray = sorted(clusterDF['ClusterParentClusterID'].unique())

        # Step 10: Create a dictionary that maps ClusterParentClusterID to their position in ClusterIDGroupArray
        cluster_order_dict = {0: 0}
        for cluster_id in ClusterIDGroupArray:
            if cluster_id not in cluster_order_dict:
                parent_cluster_id = clusterDF.loc[clusterDF['cluster_id'] == cluster_id, 'ClusterParentClusterID'].values[0]
                cluster_order_dict[cluster_id] = cluster_order_dict.get(parent_cluster_id, 0) + 1


        ### SEGMENT CONNECTIVITY LOGIC ###
                # Rebuilding steps to prepare the data and graph
        unique_branch_ids = tree_df['branch_id'].unique()
        subset_branch_ids = unique_branch_ids[:len(unique_branch_ids) // 2]
        subset_df = tree_df[tree_df['branch_id'].isin(subset_branch_ids)].copy()

        # Create clusters and parent-child relationships
        subset_df = subset_df.sort_values(by=['branch_id', 'startx'])
        subset_df['startx_diff'] = subset_df['startx'].diff().fillna(0)
        subset_df['cluster_id'] = (subset_df['startx_diff'].abs() > 1).cumsum()

        clusterDF = subset_df.groupby('cluster_id').agg(lowestBranchIDInCluster=('branch_id', 'min')).reset_index()
        all_branch_positions = subset_df[['startx', 'starty', 'startz']].values
        kdtree_all = cKDTree(all_branch_positions)
        lowest_branch_positions = subset_df[subset_df['branch_id'].isin(clusterDF['lowestBranchIDInCluster'])][['startx', 'starty', 'startz']].values
        distances_all, nearest_indices_all = kdtree_all.query(lowest_branch_positions, k=3)
        nearest_branch_ids = subset_df.iloc[nearest_indices_all[:, 1]]['branch_id'].values
        nearest_cluster_ids = subset_df.iloc[nearest_indices_all[:, 1]]['cluster_id'].values

        clusterDF['ClusterParentID'] = nearest_branch_ids
        clusterDF['ClusterParentClusterID'] = nearest_cluster_ids

        # Create the connectivity graph
        cluster_graph = nx.DiGraph()
        edges = list(zip(clusterDF['ClusterParentClusterID'], clusterDF['cluster_id']))
        cluster_graph.add_edges_from(edges)

        # Define search conditions (changeable)
        segment_conditions = {
            'largeSegment': clusterDF.loc[subset_df['startz'] < 1, 'cluster_id'].unique(),
            'smallSegment': clusterDF.loc[subset_df['start_radius'] > 0.1, 'cluster_id'].unique()
        }

        # Assign segments based on search conditions
        assign_segments(clusterDF, cluster_graph, segment_conditions)

        # Map the segment IDs back to the subset DataFrame for visualization
        for segment_name in segment_conditions.keys():
            column_name = f'{segment_name}_id'
            subset_df[column_name] = subset_df['cluster_id'].map(clusterDF.set_index('cluster_id')[column_name])

        # Plot the scatter plot for largeSegment
        plt.figure(figsize=(10, 6))
        plt.scatter(subset_df['startx'], subset_df['starty'], c=subset_df['largeSegment_id'], cmap='tab20c', s=10)
        plt.title('Scatter Plot of x, y Colored by largeSegment_id')
        plt.xlabel('startX')
        plt.ylabel('startY')
        plt.colorbar(label='largeSegment_id')
        plt.show()

        # Plot the scatter plot for smallSegment
        plt.figure(figsize=(10, 6))
        plt.scatter(subset_df['startx'], subset_df['starty'], c=subset_df['smallSegment_id'], cmap='tab20', s=10)
        plt.title('Scatter Plot of x, y Colored by smallSegment_id')
        plt.xlabel('startX')
        plt.ylabel('startY')
        plt.colorbar(label='smallSegment_id')
        plt.show()
        
        ###CHAT GPT PLEASE DO: SEGMENT LOGIC operating on clusterDF ###
        #determine segmentLarge_id and segmentSmall_id columns. Initialise both as -1 
        
        #First, group consideredBranchesDF by cluster_id and sort by branch_id
        
        #For both, we will select some cluster_ids based on some criteria. For now, I will call both 'segmentSearchID'. Get the cluster_id indexes matching:
            #for segmentSmall_id, we will select cluster_ids that have a start_radius > 0.1
            #for segmentLarge_id, we will select cluster_ids that have an order == 1 

        #assign each valid cluster_id a 'segmentSearchID'. This will just be 1, 2, 3, etc.

        #then, we need to assign all child cluster_ids of each segment their parents 'segmentSearchID'. 
        # To do so, find all ClusterId's where the ClusterParentClusterID equals this segments ClusterId. Assign these child clusters their parents segmentSearchID
        # Repeat until we get to the terminal clusters (ie. clusters that have no clusters )

        ###MAP CLUSTERDF TO SUBSET_DF###
        #columns to map: segmentSmall_id, segmentLarge_id, order, ClusterParentID, ClusterParentClusterID

        # Step 12: Apply the 'order' column directly to subset_df using a map function (TODO: extend this to all columns to map)
        subset_df['order'] = subset_df['cluster_id'].map(cluster_order_dict)

        # Step 13: Verify by checking rows where cluster_id == 0 and order == 0
        cluster_id_0_count = subset_df[subset_df['cluster_id'] == 0].shape[0]
        order_0_count = subset_df[subset_df['order'] == 0].shape[0]

        # Final verification
        print(f"cluster_id_0_count: {cluster_id_0_count}, order_0_count: {order_0_count}")

        processedTreeDFs[tree_id] = subset_df

        """# Plot: 3D Scatter plot for all data points, colored by 'order', using the qualitative colormap 'Tab20c'
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(subset_df['startx'], subset_df['starty'], subset_df['startz'], 
                        c=subset_df['order'], cmap='tab20c', s=10)
        ax.set_title('3D Scatter Plot of x, y, z Colored by Order (Tab20c Colormap)')
        ax.set_xlabel('startX')
        ax.set_ylabel('startY')
        ax.set_zlabel('startZ')
        fig.colorbar(sc, ax=ax, label='Order')
        plt.tight_layout()
        plt.show()
        """

        filePath = 'data/revised/lidar scans/elm/adtree/processedQSMs'
        #make sure folder exists
        os.makedirs(filePath, exist_ok=True)
        subset_df.to_csv(f'{filePath}/{fileNameDic[tree_id]}_clusteredQSM.csv', index=False)


    plot_all_trees(processedTreeDFs)