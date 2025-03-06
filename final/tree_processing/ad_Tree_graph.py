
import pandas as pd
import pyvista as pv
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pyvista as pv




def plot_connected_lines_from_graph(G, point_size=10, line_width=2, 
                         point_color='blue', line_color='gray',
                         background_color='white',
                         window_size=(1024, 768)):
    """
    Plot a NetworkX graph efficiently using a single PolyData object.
    
    Args:
        G (nx.Graph): NetworkX graph where nodes have startx, starty, startz attributes
    """
    # Create points array from node positions
    points = []
    for node in G.nodes():
        try:
            points.append([G.nodes[node]['x'], 
                        G.nodes[node]['y'], 
                        G.nodes[node]['z']])
        except KeyError as e:
            print(f"Node {node} is missing attribute: {e}")
    points = np.array(points)

        
    
    
    # Print attributes of the first node in clusterGraph
    first_node = next(iter(G.nodes()))
    print("Attributes of the first node in clusterGraph:")
    for attr, value in G.nodes[first_node].items():
        print(f"{attr}: {value}")


    
    plotter = pv.Plotter(window_size=window_size)
    plotter.background_color = background_color
    
    # Create points array from node positions
    points = np.array([[G.nodes[node]['x'], 
                       G.nodes[node]['y'], 
                       G.nodes[node]['z']] for node in G.nodes()])
    
    # Create lines array for connections
    lines = []
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    
    for edge in G.edges():
        # Format: [number of points in line, start_idx, end_idx]
        lines.append([2, 
                     node_to_idx[edge[0]], 
                     node_to_idx[edge[1]]])
    
    lines = np.hstack(lines)
    
    # Create single PolyData object with points and lines
    poly = pv.PolyData(points, lines=lines)
    
    # Add to plotter
    plotter.add_mesh(poly, render_lines_as_tubes=True, line_width=line_width,
                    color=line_color)
    plotter.add_mesh(poly, render_points_as_spheres=True, point_size=point_size,
                    color=point_color)
    
    plotter.show_axes()
    plotter.show_grid()
    plotter.show()
    return plotter

def prune_tree(graph, nodes_to_remove):
    """
    Prune nodes from the graph, removing them and their descendants without breaking connectivity.
    :param graph: The NetworkX graph to prune
    :param nodes_to_remove: List of nodes to remove based on criteria
    :return: The pruned graph
    """
    for node in nodes_to_remove:
        if graph.has_node(node):
            # Remove node and all its descendants
            descendants = nx.descendants(graph, node)
            graph.remove_nodes_from(descendants)
            graph.remove_node(node)
    return graph


# Now you can use filtered_clusterGraph for further plotting


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
fileNameDic = selected_fileNameDic = fileNameDic

for tree_id, filename in fileNameDic.items():
    clusterGraph = nx.read_graphml(f'{folderPath}/processedQSMs/{filename}_clusterGraph.graphml')

      
    # Load voxel data
    voxelDF = pd.read_csv(f'{folderPath}/elmVoxelDFs/{filename}_voxelDF.csv')
    
    # Load cluster graph

    # Count voxels per cluster
    voxelCount = voxelDF.groupby('cluster_id').size().reset_index(name='voxelCount')

    # Convert cluster_id to string to match node IDs in the graph
    voxelCount['cluster_id'] = voxelCount['cluster_id'].astype(str)

    # Add voxel count to cluster graph
    voxelCountDict = voxelCount.set_index('cluster_id')['voxelCount'].to_dict()
    nx.set_node_attributes(clusterGraph, voxelCountDict, 'voxelCount')
    
    original_node_count = clusterGraph.number_of_nodes()
    print(f"Original number of nodes: {original_node_count}")

    ##PRUNE GRAPH
    # Iteratively prune leaf nodes with start_radius < 0.01 and check for isolated nodes
    isolated_nodes_counts = []
    iteration = 0
    total_pruned_nodes = 0

    while True:
        # Find leaf nodes (nodes with no outgoing edges)
        leaf_nodes = [node for node in clusterGraph.nodes if clusterGraph.out_degree(node) == 0]

        # Prune leaf nodes that have 'start_radius' < 0.01
        nodes_to_prune = [node for node in leaf_nodes if float(clusterGraph.nodes[node].get('start_radius', 0)) < 0.005]

        # If no more nodes to prune, stop the iteration
        if not nodes_to_prune:
            break

        # Remove the nodes
        clusterGraph.remove_nodes_from(nodes_to_prune)
        total_pruned_nodes += len(nodes_to_prune)

        # Check for isolated nodes after pruning (nodes with no edges)
        isolated_nodes_after_prune = [node for node in clusterGraph.nodes if clusterGraph.degree(node) == 0]
        
        # Append the number of isolated nodes for this iteration
        isolated_nodes_counts.append(len(isolated_nodes_after_prune))

        print(f'Iteration {iteration}: {len(isolated_nodes_after_prune)} isolated nodes')

        iteration += 1

    final_node_count = clusterGraph.number_of_nodes()
    print(f"Number of nodes pruned: {total_pruned_nodes}")
    print(f"Final number of nodes: {final_node_count}")

    #save pruned graph
    #check if file path exists
    if not os.path.exists(f'{folderPath}/prunedGraphs'):
        os.makedirs(f'{folderPath}/prunedGraphs')
    #save pruned graph
    nx.write_graphml(clusterGraph, f'{folderPath}/prunedGraphs/{filename}_prunedGraph.graphml')



    





for tree_id, filename in fileNameDic.items():
    test = 1
    """print(f"Processing tree ID: {tree_id}, filename: {filename}")
    
    # Load voxel data
    voxelDF = pd.read_csv(f'{folderPath}/elmVoxelDFs/{filename}_voxelDF.csv')
    
    # Load cluster graph
    clusterGraph = nx.read_graphml(f'{folderPath}/processedQSMs/{filename}_clusterGraph.graphml')

    # Count voxels per cluster
    voxelCount = voxelDF.groupby('cluster_id').size().reset_index(name='voxelCount')

    # Convert cluster_id to string to match node IDs in the graph
    voxelCount['cluster_id'] = voxelCount['cluster_id'].astype(str)

    # Add voxel count to cluster graph
    voxelCountDict = voxelCount.set_index('cluster_id')['voxelCount'].to_dict()
    nx.set_node_attributes(clusterGraph, voxelCountDict, 'voxelCount')
    
    
    
    # Add x, y, z information (first row per cluster) to the cluster graph
    cluster_attribute_info = voxelDF.groupby('cluster_id').first()[['x', 'y', 'z', 'start_radius']].reset_index()
    cluster_attribute_info['cluster_id'] = cluster_attribute_info['cluster_id'].astype(str)
    cluster_attribute_dic = cluster_attribute_info.set_index('cluster_id')[['x', 'y', 'z', 'start_radius']].to_dict(orient='index')
    nx.set_node_attributes(clusterGraph, cluster_attribute_dic)

    print("Added voxel counts and x, y, z coordinates to cluster graph")"""
    



    # Save updated cluster graph
    #output_path = f'{folderPath}/processedQSMs/{filename}_clusterGraph.graphml'
    #nx.write_graphml(clusterGraph, output_path)
    #print(f"Saved updated cluster graph to {output_path}")
    
    """print(f"Completed processing for tree ID: {tree_id}\n")

    # Identify nodes to prune based on start_radius < 0.01, handling missing attributes
    nodes_to_prune = [node for node, data in clusterGraph.nodes(data=True)
                    if 'start_radius' in data and float(data['start_radius']) < 0.0001]

    # Prune the tree, ensuring connectivity is preserved
    pruned_graph = prune_tree(clusterGraph.copy(), nodes_to_prune)

    # Extract remaining node positions from the pruned graph
    remaining_node_positions = [(float(data['x']), float(data['y']), float(data['z']))
                                for node, data in pruned_graph.nodes(data=True) if 'x' in data and 'y' in data and 'z' in data]



    # Print summary information
    print(f"Original nodes: {clusterGraph.number_of_nodes()}")
    print(f"Pruned nodes: {len(nodes_to_prune)}")
    print(f"Remaining nodes: {pruned_graph.number_of_nodes()}")"""

    # Extract the remaining edges and their positions for plotting
    """remaining_edges = [(node1, node2) for node1, node2 in pruned_graph.edges()]
    remaining_node_positions = np.array(remaining_node_positions)

    # Prepare lists to store start and end positions of each edge
    edge_start_positions = []
    edge_end_positions = []

    for node1, node2 in remaining_edges:
        if 'x' in pruned_graph.nodes[node1] and 'x' in pruned_graph.nodes[node2]:
            x1, y1, z1 = float(pruned_graph.nodes[node1]['x']), float(pruned_graph.nodes[node1]['y']), float(pruned_graph.nodes[node1]['z'])
            x2, y2, z2 = float(pruned_graph.nodes[node2]['x']), float(pruned_graph.nodes[node2]['y']), float(pruned_graph.nodes[node2]['z'])
            edge_start_positions.append([x1, y1, z1])
            edge_end_positions.append([x2, y2, z2])

    edge_start_positions = np.array(edge_start_positions)
    edge_end_positions = np.array(edge_end_positions)


    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add nodes as spheres to the plotter
    point_cloud = pv.PolyData(remaining_node_positions)
    plotter.add_mesh(point_cloud, color='blue', point_size=10, render_points_as_spheres=True)

    # Add edges as lines
    for i in range(len(edge_start_positions)):
        line = pv.Line(edge_start_positions[i], edge_end_positions[i])
        plotter.add_mesh(line, color='green')

    # Show the plot with axes
    plotter.show_axes()
    plotter.show()
    """

    # 1. Extract node attributes as separate dictionaries
    x = nx.get_node_attributes(pruned_graph, 'x')
    y = nx.get_node_attributes(pruned_graph, 'y')
    z = nx.get_node_attributes(pruned_graph, 'z')

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
        for node1, node2 in pruned_graph.edges()
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
