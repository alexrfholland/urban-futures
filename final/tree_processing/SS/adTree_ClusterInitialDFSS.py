
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
# Function to propagate segment IDs based on starting node cluster IDs



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
selected_fileNameDic = fileNameDic

print(f"Selected trees: {selected_fileNameDic}")

processedTreeDFs = {}


for tree_id, filename in selected_fileNameDic.items():
    print(f"Processing tree ID: {tree_id}, filename: {filename}")

    # Load data from the QSM file
    qsmFileName = f'{folderPath}/QSMs/{filename}_treeDF.csv'
    tree_df = pd.read_csv(qsmFileName, delimiter=',') 

    # Step 1: Calculate rowLength using NumPy's vectorized operations for efficiency
    tree_df['rowLength'] = np.linalg.norm(
        tree_df[['startx', 'starty', 'startz']].values - 
        tree_df[['endx', 'endy', 'endz']].values,
        axis=1
    )

    ## INITIAL CLUSTER LOGIC ###
    # Step 2: Sort by branch_id and calculate the difference in startX
    tree_df = tree_df.sort_values(by=['branch_id', 'startx'])
    tree_df['startx_diff'] = tree_df['startx'].diff().fillna(0)

    # Step 3: Create clusters where there's a jump greater than 0.2 in startX
    tree_df['original_cluster_id'] = (tree_df['startx_diff'].abs() > 0.2).cumsum()

    # Step 4: Initialize cluster_segment_no and cluster_id
    tree_df['cluster_segment_no'] = 1
    tree_df['cluster_id'] = -1

    ## FURTHER CLUSTER LOGIC ###
    # Step 5: Define the max cluster length
    max_cluster_length = 2

    # Calculate the total length for each cluster based on rowLength
    cluster_total_lengths = tree_df.groupby('original_cluster_id')['rowLength'].transform('sum')

    # Identify how many segments each cluster needs, but only if it exceeds the max length
    num_segments_per_cluster = (cluster_total_lengths // max_cluster_length).astype(int) + 1

    # Calculate the segment assignment for each row in a vectorized manner
    tree_df['cluster_segment_no'] = (tree_df.groupby('original_cluster_id').cumcount() // 
                                    (tree_df.groupby('original_cluster_id')['branch_id'].transform('size') // num_segments_per_cluster + 1)) + 1

    # Reassign cluster_id based on original_cluster_id and cluster_segment_no
    tree_df['cluster_id'] = tree_df.groupby(['original_cluster_id', 'cluster_segment_no']).ngroup()

    # Recalculate the total number of clusters before and after processing
    total_original_clusters = tree_df['original_cluster_id'].nunique()
    total_final_clusters = tree_df['cluster_id'].nunique()
    new_clusters_added = total_final_clusters - total_original_clusters

    # Group by original_cluster_id and calculate the total length of each cluster using rowLength
    cluster_total_lengths = tree_df.groupby('original_cluster_id')['rowLength'].sum()

    # Count how many original clusters exceed the max length (cluster-wise, not row-wise)
    clusters_exceeding_length = (cluster_total_lengths > max_cluster_length).sum()

    # Print the correct informative statement
    print(f"Total clusters originally: {total_original_clusters}")
    print(f"Total clusters after processing with max length 2: {total_final_clusters}")
    print(f"Number of original clusters exceeding length > {max_cluster_length}: {clusters_exceeding_length}")
    print(f"Number of new clusters added: {new_clusters_added}")

    ### CLUSTER CONNECTIVITY ###
    # Step 4: Group by cluster_id to find the lowest branch_id per cluster
    clusterDF = tree_df.loc[tree_df.groupby('cluster_id')['branch_id'].idxmin()].copy()

    # Add startX, startY, startZ, startRadius, original_cluster_id, and cluster_segment_no to clusterDF
    clusterDF['startX'] = tree_df['startx']
    clusterDF['startY'] = tree_df['starty']
    clusterDF['startZ'] = tree_df['startz']
    clusterDF['startRadius'] = tree_df['start_radius']
    clusterDF['original_cluster_id'] = tree_df['original_cluster_id']
    clusterDF['cluster_segment_no'] = tree_df['cluster_segment_no']

    # Step 6: Build a KDTree for all branch positions (startx, starty, startz)
    all_branch_positions = tree_df[['startx', 'starty', 'startz']].values
    kdtree_all = cKDTree(all_branch_positions)

    # Get positions for the lowest branch in each cluster
    lowest_branch_positions = tree_df[tree_df['branch_id'].isin(
        clusterDF['branch_id']
    )][['startx', 'starty', 'startz']].values

    # Step 7: Query the KDTree to find the nearest 5 neighbors for each lowest branch
    distances_all, nearest_indices_all = kdtree_all.query(lowest_branch_positions, k=5)

    # Get the current cluster_ids for the lowest branches
    current_cluster_ids = clusterDF['cluster_id'].values

    # Use np.take to select the 'cluster_id' of the nearest neighbors using the nearest_indices_all array
    nearest_cluster_ids_matrix = np.take(tree_df['cluster_id'].values, nearest_indices_all)

    # Create a mask to filter neighbors that are from the same cluster
    # This mask will be True for neighbors from a different cluster
    different_cluster_mask = nearest_cluster_ids_matrix != current_cluster_ids[:, np.newaxis]

    # Use argmax to find the first True in each row of the mask (i.e., the first different cluster)
    first_valid_neighbor_indices = np.argmax(different_cluster_mask, axis=1)

    # Now, use the valid neighbor indices to select the corresponding branch and cluster IDs
    nearest_valid_indices = nearest_indices_all[np.arange(len(nearest_indices_all)), first_valid_neighbor_indices]

    # Assign the ClusterParentID and ClusterParentClusterID using the valid nearest neighbors
    clusterDF['cluster'] = tree_df.iloc[nearest_valid_indices]['branch_id'].values
    clusterDF['ClusterParentClusterID'] = tree_df.iloc[nearest_valid_indices]['cluster_id'].values

    # Create the connectivity graph
    cluster_graph = nx.DiGraph()
    edges = list(zip(clusterDF['ClusterParentClusterID'], clusterDF['cluster_id']))
    cluster_graph.add_edges_from(edges)


    # Plot the graph to visualize connections
    """plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(cluster_graph)  # Layout for plotting

    # Draw the nodes and edges with arrows to indicate direction
    nx.draw(cluster_graph, pos, with_labels=True, node_size=300, font_size=8, arrows=True, arrowstyle='->')

    # Show the plot
    plt.title("Visualization of Cluster Graph Connections")
    plt.show()"""

    # Check for isolated nodes
    isolated_nodes = list(nx.isolates(cluster_graph))

    isolated_nodes

    # Convert the DataFrame rows into a dictionary format suitable for node attributes
    attributes = clusterDF.set_index('cluster_id')[['startX', 'startY', 'startZ', 'startRadius', 'original_cluster_id', 'cluster_segment_no']].to_dict('index')

    # Set the attributes (including startX, startY, startZ, original_cluster_id, and cluster_segment_no) as node attributes in the graph
    nx.set_node_attributes(cluster_graph, attributes)

    # Map cluster info back to tree_df
    tree_df['ClusterParentClusterID'] = tree_df['cluster_id'].map(clusterDF.set_index('cluster_id')['ClusterParentClusterID'])

    #showRandomPolyCols(tree_df, 'original_cluster_id')
    #showRandomPolyCols(tree_df, 'cluster_segment_no')
    #showRandomPolyCols(tree_df, 'cluster_id')




    """#PLOTTING

    # 1. Extract node attributes as separate dictionaries
    x = nx.get_node_attributes(cluster_graph, 'x')
    y = nx.get_node_attributes(cluster_graph, 'y')
    z = nx.get_node_attributes(cluster_graph, 'z')

    # 2. Find nodes that have all three coordinates
    valid_nodes = list(set(x.keys()) & set(y.keys()) & set(z.keys()))
    num_nodes = len(valid_nodes)

    # 3. Create a NumPy array of positions
    positions = np.array([ [float(x[node]), float(y[node]), float(z[node])] for node in valid_nodes ])

    # 4. Create a mapping from node to index using NumPy's vectorized operations
    node_indices = np.arange(num_nodes)
    node_to_index = dict(zip(valid_nodes, node_indices))

    # 5. Extract edges where both nodes are valid
    edges = np.array([
        (node_to_index[node1], node_to_index[node2])
        for node1, node2 in cluster_graph.edges()
        if node1 in node_to_index and node2 in node_to_index
    ])

    # 6. Prepare edge points by stacking start and end positions
    edge_start = positions[edges[:, 0]]
    edge_end = positions[edges[:, 1]]
    all_edge_points = np.vstack((edge_start, edge_end))

    # 7. Create lines array for PyVista
    num_edges = len(edges)
    lines = np.column_stack((np.full(num_edges, 2, dtype=np.int32), 
                            np.arange(num_edges, dtype=np.int32), 
                            np.arange(num_edges, dtype=np.int32) + num_edges)).flatten()

    # 8. Create PyVista PolyData objects
    nodes_cloud = pv.PolyData(positions)
    edges_mesh = pv.PolyData(all_edge_points)
    edges_mesh.lines = lines

    # 9. Plot using PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(nodes_cloud, color='blue', point_size=10, render_points_as_spheres=True)
    plotter.add_mesh(edges_mesh, color='green', line_width=2)
    plotter.add_axes()
    plotter.show()

"""


    processedTreeDFs[tree_id] = tree_df

    filePath = 'data/revised/lidar scans/elm/adtree/processedQSMs'
    #make sure folder exists
    os.makedirs(filePath, exist_ok=True)
    tree_df.to_csv(f'{filePath}/{fileNameDic[tree_id]}_clusteredQSM.csv', index=False)
    print(f"Saved to {filePath}/{fileNameDic[tree_id]}_clusteredQSM.csv")

    # Save the graph in a GraphML format (which supports node attributes)
    nx.write_graphml(cluster_graph, f'{filePath}/{fileNameDic[tree_id]}_clusterGraph.graphml')
    print(f"Saved to {filePath}/{fileNameDic[tree_id]}_clusterGraph.graphml")





