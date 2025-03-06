import networkx as nx
import igraph as ig
import leidenalg as la
import community as community_louvain
from networkx.algorithms import community as nx_community
from infomap import Infomap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt  # Added import
import pyvista as pv
import numpy as np
import pandas as pd  # Assuming pandas is used for reading CSVs
import os
from collections import defaultdict
from typing import Tuple, Dict
import adTree_CommunityGraphClustering

def showRandomPolyCols(treePoly, attributeID):

    n_clusters = treePoly[attributeID].max() + 1  # Assuming cluster_ids start from 0
    cluster_colors = np.random.rand(n_clusters, 3)  # Random RGB colors
    color_array = cluster_colors[treePoly[attributeID]]

    # Step 4: Assign colors to the mesh and plot
    treePoly.point_data['colors'] = color_array
    return treePoly


# -------------------- Community Detection Functions --------------------
import networkx as nx
from collections import deque

import networkx as nx

def apply_custom_hierarchical_community_detection(nx_graph, origin_cluster_id=0, level=0):
    """
    Applies a hierarchical community detection algorithm based on existing Leiden communities.
    
    Parameters:
    - nx_graph (networkx.Graph): The input NetworkX graph with 'community-leiden' attribute.
    - origin_cluster_id (int or str): The 'original_cluster_id' indicating the origin node's community.
    - level (int): The hierarchical level for community refinement (0, 1, 2, ...).
    
    Returns:
    - networkx.Graph: The original NetworkX graph with 'community-custom' labels assigned to each node.
      - 'community-custom' = 0 for unallocated nodes.
      - 'community-custom' >= 1 for allocated hierarchical communities.
    """
    # Step 1: Initialize 'community-custom' to 0 for all nodes (unallocated)
    for node in nx_graph.nodes():
        nx_graph.nodes[node][f'community-leiden-level{level}'] = 0

    # Step 2: Identify origin node(s): 'original_cluster_id' == origin_cluster_id
    origin_nodes = [node for node, data in nx_graph.nodes(data=True) 
                    if data.get('original_cluster_id') == origin_cluster_id]
    
    if not origin_nodes:
        print(f"No origin node found with 'original_cluster_id' = {origin_cluster_id}.")
        return nx_graph
    
    # Assuming single origin node; adjust if multiple
    origin_node = origin_nodes[0]
    print(f"Origin node identified: {origin_node}")
    
    # Step 3: Identify the Leiden community of the origin node
    origin_leiden_community = nx_graph.nodes[origin_node].get('community-leiden', None)
    
    if origin_leiden_community is None:
        print(f"Origin node '{origin_node}' does not belong to any Leiden community.")
        return nx_graph
    
    print(f"Origin node '{origin_node}' belongs to Leiden community {origin_leiden_community}.")
    
    # Step 4: Gather all nodes in the origin Leiden community
    origin_comm_nodes = [node for node, data in nx_graph.nodes(data=True) 
                         if data.get('community-leiden') == origin_leiden_community]
    print(f"Origin Leiden community contains {len(origin_comm_nodes)} nodes.")
    
    # Step 5: Initialize community-custom ID counter
    comm_id_counter = 1  # Start from 1 since 0 is reserved for unallocated
    
    # Step 6: Initialize a queue for BFS traversal
    # Each item in the queue is a tuple: (list of nodes in the community, current hierarchical level)
    queue = deque()
    queue.append((origin_comm_nodes, 0))  # Start with origin community at level 0
    
    while queue:
        current_nodes, current_level = queue.popleft()
        
        if current_level >= level:
            continue  # Do not process beyond the desired level
        
        # Identify child Leiden communities connected to the current_nodes
        child_comm_leiden_ids = set()
        for node in current_nodes:
            neighbors = list(nx_graph.neighbors(node))
            for neighbor in neighbors:
                neighbor_comm = nx_graph.nodes[neighbor].get('community-leiden')
                if neighbor_comm != origin_leiden_community:
                    child_comm_leiden_ids.add(neighbor_comm)
        
        # Collect child communities
        child_communities = []
        for leiden_id in child_comm_leiden_ids:
            nodes_in_comm = [node for node, data in nx_graph.nodes(data=True) 
                            if data.get('community-leiden') == leiden_id]
            # Ensure the child community is connected to the parent community
            connected = False
            for parent_node in current_nodes:
                for node in nodes_in_comm:
                    if nx_graph.has_edge(parent_node, node):
                        connected = True
                        break
                if connected:
                    break
            if connected:
                child_communities.append({'leiden_id': leiden_id, 'nodes': nodes_in_comm})
        
        for child_comm in child_communities:
            if not child_comm['nodes']:
                continue  # Skip empty communities
            
            # Assign a new community-custom ID
            for node in child_comm['nodes']:
                nx_graph.nodes[node][f'community-leiden-level{level}'] = comm_id_counter
            print(f"Assigned community-leiden-level{level}' {comm_id_counter} to Leiden community {child_comm['leiden_id']} with {len(child_comm['nodes'])} nodes.")
            comm_id_counter += 1
            
            # Enqueue child communities for the next level
            queue.append((child_comm['nodes'], current_level + 1))
    
    print("Hierarchical community detection completed.")
    return nx_graph


def apply_custom_community_detection(nx_graph, main_stem_cluster_id=0, upper_threshold=1/3, lower_threshold=1/10):
    """
    Applies a custom community detection algorithm to a NetworkX graph.
    
    Parameters:
    - nx_graph (nx.Graph): The input NetworkX graph.
    - main_stem_cluster_id (int): The 'original_cluster_id' value indicating the main stem.
    - upper_threshold (float): Fraction of total nodes above which a branch should be split.
    - lower_threshold (float): Fraction of total nodes below which a branch is assigned to main stem.
    
    Returns:
    - nx.Graph: The original NetworkX graph with 'community-custom' labels assigned to each node.
    """
    # Step 1: Initialize 'community-custom' to 0 for all nodes (main stem community)
    for node in nx_graph.nodes():
        nx_graph.nodes[node]['community-custom'] = 0

    # Step 2: Identify main stem nodes: 'original_cluster_id' == main_stem_cluster_id
    main_stem_nodes = [node for node, data in nx_graph.nodes(data=True) if data.get('original_cluster_id') == main_stem_cluster_id]
    
    if not main_stem_nodes:
        print("No main stem nodes found with 'original_cluster_id' = 0.")
        return nx_graph

    # Step 3: Identify the origin node (root of the main stem)
    if nx.is_directed(nx_graph):
        # For directed graphs, origin node has in-degree 0
        origin_nodes = [node for node, data in nx_graph.nodes(data=True) 
                       if data.get('original_cluster_id') == main_stem_cluster_id and nx_graph.in_degree(node) == 0]
        if not origin_nodes:
            print("No origin node found for the main stem in a directed graph.")
            return nx_graph
        origin_node = origin_nodes[0]  # Assuming single origin
    else:
        # For undirected graphs, select a main stem node with the highest degree as the origin
        origin_node = max(main_stem_nodes, key=lambda x: nx_graph.degree(x))
    
    # Step 4: Traverse the main stem in order using BFS
    def traverse_main_stem_bfs(graph, origin, stem_nodes):
        visited = set()
        queue = deque([origin])
        ordered_nodes = []
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            ordered_nodes.append(current)
            
            # Get neighbors that are also main stem nodes and not visited
            neighbors = list(graph.neighbors(current)) if not nx.is_directed(graph) else list(graph.successors(current))
            for neighbor in neighbors:
                if neighbor in stem_nodes and neighbor not in visited:
                    queue.append(neighbor)
        return ordered_nodes

    main_stem_ordered_nodes = traverse_main_stem_bfs(nx_graph, origin_node, main_stem_nodes)
    print(f"Main stem ordered nodes: {main_stem_ordered_nodes}")

    total_nodes = nx_graph.number_of_nodes()
    
    # Step 5: Process each main stem node's branch
    for main_node in main_stem_ordered_nodes:
        # a. Get branch nodes: 'original_cluster_id' == main_node
        branch_nodes = [node for node, data in nx_graph.nodes(data=True) if data.get('original_cluster_id') == main_node]
        branch_size = len(branch_nodes)
        print(f"\nProcessing main stem node '{main_node}' with branch size {branch_size}")

        if branch_size > upper_threshold * total_nodes:
            print(f"Branch size {branch_size} exceeds upper threshold {upper_threshold * total_nodes:.2f}. Splitting into two communities.")

            # b. Identify sub-branch roots within this branch
            sub_branch_roots = [node for node in branch_nodes if any(
                (child for child in nx_graph.neighbors(node) if child in branch_nodes and 
                 nx_graph.nodes[child].get('original_cluster_id') == node))]
            print(f"Identified sub-branch roots: {sub_branch_roots}")

            # c. Collect all sub-branches starting from sub_branch_roots
            sub_branches = []
            visited_sub_branches = set()

            for sub_root in sub_branch_roots:
                if sub_root in visited_sub_branches:
                    continue
                queue = deque([sub_root])
                current_sub_branch = set()

                while queue:
                    node = queue.popleft()
                    if node in visited_sub_branches:
                        continue
                    if node not in branch_nodes:
                        continue
                    visited_sub_branches.add(node)
                    current_sub_branch.add(node)

                    # Add neighbors that belong to this branch and have 'original_cluster_id' == node
                    neighbors = list(nx_graph.neighbors(node)) if not nx.is_directed(nx_graph) else list(nx_graph.successors(node))
                    for neighbor in neighbors:
                        if neighbor in branch_nodes and nx_graph.nodes[neighbor].get('original_cluster_id') == node:
                            queue.append(neighbor)
                if current_sub_branch:
                    sub_branches.append(current_sub_branch)

            print(f"Identified {len(sub_branches)} sub-branches within branch '{main_node}'.")

            # d. Assign sub-branches to two communities to balance node counts
            sub_branches_sorted = sorted(sub_branches, key=lambda x: len(x), reverse=True)
            community1 = set()
            community2 = set()

            for sub_branch in sub_branches_sorted:
                if len(community1) <= len(community2):
                    community1.update(sub_branch)
                else:
                    community2.update(sub_branch)

            # e. Assign new community IDs
            existing_communities = set(nx_graph.nodes[n].get('community-custom', 0) for n in nx_graph.nodes())
            new_comm_id1 = max(existing_communities) + 1
            new_comm_id2 = max(existing_communities) + 2

            print(f"Assigning sub-branches to communities {new_comm_id1} and {new_comm_id2}.")

            for node in community1:
                nx_graph.nodes[node]['community-custom'] = new_comm_id1

            for node in community2:
                nx_graph.nodes[node]['community-custom'] = new_comm_id2

        elif branch_size < lower_threshold * total_nodes:
            print(f"Branch size {branch_size} is below lower threshold {lower_threshold * total_nodes:.2f}. Assigning to main stem community (0).")
            # Already assigned to community 0
            continue

        else:
            print(f"Branch size {branch_size} is within thresholds. Assigning to a new community.")
            # Assign entire branch to a new community
            existing_communities = set(nx_graph.nodes[n].get('community-custom', 0) for n in nx_graph.nodes())
            new_comm_id = max(existing_communities) + 1

            for node in branch_nodes:
                nx_graph.nodes[node]['community-custom'] = new_comm_id

    return nx_graph

def apply_leiden_algorithm(nx_graph, weight_attr='startRadius', prioritize_main_stem=True):
    """
    Applies the Leiden algorithm to a NetworkX graph, assigns each node a 'community-leiden' attribute,
    and returns the original graph with the community labels and the number of communities detected.

    Parameters:
    nx_graph (nx.DiGraph): A directed NetworkX graph
    weight_attr (str): Node attribute to influence edge weights (default is 'startRadius')
    prioritize_main_stem (bool): Whether to prioritize the main stem (original_branch_id == 0)

    Returns:
    nx.DiGraph: The original NetworkX graph with 'community-leiden' labels for each node
    int: Number of communities detected
    """
    # Convert NetworkX graph to igraph object
    G_igraph = ig.Graph.TupleList(nx_graph.edges(data=True), directed=True, edge_attrs=['weight'])

    # Prepare edge weights based on 'startRadius'
    weights = []
    for edge in G_igraph.es:
        u = edge.source
        v = edge.target
        # Retrieve the corresponding node names from igraph
        node_u = G_igraph.vs[u]['name']
        node_v = G_igraph.vs[v]['name']
        
        start_radius_u = nx_graph.nodes[node_u].get(weight_attr, 1.0)
        start_radius_v = nx_graph.nodes[node_v].get(weight_attr, 1.0)
        edge_weight = (start_radius_u + start_radius_v) / 2.0

        # Prioritize edges connected to main stem
        if prioritize_main_stem:
            original_branch_id_u = nx_graph.nodes[node_u].get('original_branch_id', -1)
            original_branch_id_v = nx_graph.nodes[node_v].get('original_branch_id', -1)
            if original_branch_id_u == 0 or original_branch_id_v == 0:
                edge_weight += 5.0  # Adjusted boost to prevent over-merging

        weights.append(edge_weight)

    G_igraph.es['weight'] = weights

    # Apply Leiden algorithm with modularity optimization
    partition = la.find_partition(
        G_igraph,
        la.ModularityVertexPartition,
        weights='weight'
    )

    # Create a mapping from igraph vertex names to community IDs
    node_to_community = {}
    for community_id, community in enumerate(partition):
        for node in community:
            node_name = G_igraph.vs[node]['name']
            node_to_community[node_name] = community_id

    # Assign community as a node attribute in the original NetworkX graph
    nx.set_node_attributes(nx_graph, node_to_community, 'community-leiden')

    return nx_graph, len(partition)

def apply_louvain_algorithm(nx_graph, weight_attr='startRadius', resolution=1.0, prioritize_main_stem=True):
    """
    Applies the Louvain method to a NetworkX graph, assigns each node a 'community-louvain' attribute,
    and returns the original graph with the community labels and the number of communities detected.

    Parameters:
    nx_graph (nx.DiGraph): A directed NetworkX graph
    weight_attr (str): Node attribute to influence edge weights (default is 'startRadius')
    resolution (float): Resolution parameter for the Louvain method (default is 1.0)
    prioritize_main_stem (bool): Whether to prioritize the main stem (original_branch_id == 0)

    Returns:
    nx.DiGraph: The original NetworkX graph with 'community-louvain' labels for each node
    int: Number of communities detected
    """
    # Create an undirected copy for Louvain
    graph_copy = nx_graph.to_undirected()

    # Initialize edge weights based on node attributes
    for u, v, data in graph_copy.edges(data=True):
        start_radius_u = graph_copy.nodes[u].get(weight_attr, 1.0)
        start_radius_v = graph_copy.nodes[v].get(weight_attr, 1.0)
        edge_weight = (start_radius_u + start_radius_v) / 2.0

        # Prioritize edges connected to main stem
        if prioritize_main_stem:
            original_branch_id_u = graph_copy.nodes[u].get('original_branch_id', -1)
            original_branch_id_v = graph_copy.nodes[v].get('original_branch_id', -1)
            if original_branch_id_u == 0 or original_branch_id_v == 0:
                edge_weight += 5.0  # Adjusted boost

        data['weight'] = edge_weight

    # Apply Louvain method
    partition = community_louvain.best_partition(graph_copy, weight='weight', resolution=resolution)

    # Assign community as a node attribute in the original NetworkX graph
    nx.set_node_attributes(nx_graph, partition, 'community-louvain')

    num_communities = len(set(partition.values()))
    return nx_graph, num_communities



def apply_infomap_algorithm(nx_graph, weight_attr='startRadius', prioritize_main_stem=True):
    """
    Applies the Infomap algorithm to a NetworkX graph, assigns each node a 'community-infomap' attribute,
    and returns the original graph with the community labels and the number of communities detected.

    Parameters:
    nx_graph (nx.Graph): A NetworkX graph
    weight_attr (str): Edge attribute to use as weight (default is 'startRadius')
    prioritize_main_stem (bool): Whether to prioritize the main stem (original_branch_id == 0)

    Returns:
    nx.Graph: The original NetworkX graph with 'community-infomap' labels for each node
    int: Number of communities detected
    """
    im = Infomap()
    
    # Add edges with weights to Infomap
    for u, v, data in nx_graph.edges(data=True):
        weight = data.get(weight_attr, 1.0)
        im.add_link(u, v, weight)
    
    # Optionally prioritize main stem by adjusting weights
    if prioritize_main_stem:
        for u, v, data in nx_graph.edges(data=True):
            original_branch_id_u = nx_graph.nodes[u].get('original_branch_id', -1)
            original_branch_id_v = nx_graph.nodes[v].get('original_branch_id', -1)
            if original_branch_id_u == 0 or original_branch_id_v == 0:
                # Increase weight to prioritize these edges
                new_weight = data.get(weight_attr, 1.0) + 5.0
                im.set_link(u, v, new_weight)
    
    # Run Infomap
    im.run()
    
    # Assign communities
    communities = {}
    for node in im.nodes:
        communities[node.node_id] = node.module
    
    # Check if multiple communities are detected
    num_communities = len(set(communities.values()))
    print(f"Infomap detected {num_communities} communities.")
    
    # Assign community labels back to NetworkX graph
    nx.set_node_attributes(nx_graph, communities, 'community-infomap')
    
    return nx_graph, num_communities

# -------------------- Visualization Function --------------------
def create_pyvista_polydata(G):
    """
    Converts a directed graph into a PyVista PolyData object, representing nodes as points and
    edges as lines, with node attributes added as point_data. Also adds two columns: 'subgraph' 
    and 'isRootNode'.
    
    Parameters:
    - G (nx.DiGraph): A directed NetworkX graph
    
    Returns:
    - pv.PolyData: PolyData object with nodes as points, edges as lines, and attributes as point_data.
    """
    if not G.is_directed():
        raise ValueError("The graph must be directed to detect root nodes.")
    
    # Extract node attributes for spatial positions (startX, startY, startZ)
    node_positions = {
        node: (float(data['x']), float(data['y']), float(data['z'])) 
        for node, data in G.nodes(data=True) 
        if 'x' in data and 'y' in data and 'z' in data
    }
    
    # Extract positions as a numpy array
    nodes = list(node_positions.keys())
    points = np.array([node_positions[node] for node in nodes])
    
    # Get weakly connected components and assign them as a 'subgraph' attribute
    weakly_connected_components = list(nx.weakly_connected_components(G))
    subgraph_data = np.full(len(nodes), -1, dtype=int)
    for idx, component in enumerate(weakly_connected_components):
        for node in component:
            node_index = nodes.index(node)
            subgraph_data[node_index] = idx

    # Identify root nodes (nodes with in-degree 0) and mark them in 'isRootNode'
    is_root_node_data = np.zeros(len(nodes), dtype=bool)  # Initialize with False
    root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    for root_node in root_nodes:
        if root_node in nodes:
            root_index = nodes.index(root_node)
            is_root_node_data[root_index] = True

    # Prepare the edges for the line cells in PyVista
    lines = []
    for u, v in G.edges():
        if u in node_positions and v in node_positions:
            u_index = nodes.index(u)
            v_index = nodes.index(v)
            lines.append([2, u_index, v_index])  # Each line connects 2 points: u and v

    # Convert lines to a numpy array
    if lines:
        lines = np.hstack(lines)
    else:
        lines = np.array([])

    # Create the PyVista PolyData object
    polydata = pv.PolyData(points)

    # Add edges as lines to the PolyData object
    if len(lines) > 0:
        polydata.lines = lines

    # Add attributes (node data) to the PolyData point_data
    for key in G.nodes[nodes[0]].keys():
        print(f'adding attribute: {key}')
        attribute_data = np.array([G.nodes[node].get(key, np.nan) for node in nodes])
        polydata.point_data[key] = attribute_data

    # Add 'subgraph' and 'isRootNode' columns to point_data
    polydata.point_data['subgraph'] = subgraph_data
    polydata.point_data['isRootNode'] = is_root_node_data

    return polydata


# -------------------- Utility Function --------------------
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


import networkx as nx
def create_community_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Creates a community-level directed graph from a node-level directed graph,
    excluding self-loops by ignoring edges within the same community.
    Captures original node IDs within each community as node attributes.
    Also computes average x,y,z coordinates for each community.

    Args:
        G (nx.DiGraph): Input directed graph with 'community-leiden' attribute for each node.

    Returns:
        nx.DiGraph: Community-level directed graph.
    """
    if not all('community-leiden' in data for _, data in G.nodes(data=True)):
        raise ValueError("All nodes must have 'community-leiden' attribute")

    edge_counts = defaultdict(int)
    community_nodes = defaultdict(list)
    community_coords = defaultdict(lambda: {'x': [], 'y': [], 'z': []})

    # Map nodes to their communities and collect coordinates
    for node, data in G.nodes(data=True):
        comm = data['community-leiden']
        community_nodes[comm].append(node)
        community_coords[comm]['x'].append(float(data.get('x', 0)))
        community_coords[comm]['y'].append(float(data.get('y', 0))) 
        community_coords[comm]['z'].append(float(data.get('z', 0)))

    # Count edges between different communities
    for u, v in G.edges():
        comm_u = G.nodes[u]['community-leiden']
        comm_v = G.nodes[v]['community-leiden']
        if comm_u != comm_v:
            edge_counts[(comm_u, comm_v)] += 1

    # Create the community graph
    C = nx.DiGraph()

    # Add community nodes with original node IDs and average coordinates
    for comm, nodes in community_nodes.items():
        # Calculate average coordinates
        avg_x = sum(community_coords[comm]['x']) / len(nodes)
        avg_y = sum(community_coords[comm]['y']) / len(nodes)
        avg_z = sum(community_coords[comm]['z']) / len(nodes)
        
        # Add node with all attributes
        C.add_node(comm, 
                  original_nodes=','.join(map(str, nodes)),
                  x=avg_x,
                  y=avg_y, 
                  z=avg_z)

    # Add edges with weights
    for (comm_u, comm_v), count in edge_counts.items():
        C.add_edge(comm_u, comm_v, weight=count)

    return C


def validate_graph(original_graph: nx.DiGraph, community_graph: nx.DiGraph) -> None:
    """
    Validates the community graph and prints origin nodes.

    Args:
        original_graph (nx.DiGraph): The original node-level directed graph.
        community_graph (nx.DiGraph): The community-level directed graph.
    """
    issues = []

    # Check for isolated nodes in community graph
    isolated = list(nx.isolates(community_graph))  # Fixed: Pass the graph object directly
    if isolated:
        issues.append(f"Found {len(isolated)} isolated communities: {isolated}")

    # Check for self-loops in community graph
    self_loops = list(nx.selfloop_edges(community_graph))
    if self_loops:
        issues.append(f"Found {len(self_loops)} self-loops in community graph")

    # Check for single connected component
    undirected = community_graph.to_undirected()
    num_components = nx.number_connected_components(undirected)
    if num_components > 1:
        issues.append(f"Community graph has {num_components} connected components")

    # Identify origin nodes in original graph (in-degree 0)
    origin_original = [node for node, degree in original_graph.in_degree() if degree == 0]
    print(f"Origin nodes in the original graph: {origin_original}")

    # Identify origin communities in community graph (in-degree 0)
    origin_community = [node for node, degree in community_graph.in_degree() if degree == 0]
    print(f"Origin communities in the community graph: {origin_community}")

    # Print validation results
    if issues:
        print("Graph validation failed:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Graph is valid")




# Add this after the validate_graph function and before the main execution section

def assign_community_clusters(original_graph: nx.DiGraph, community_graph: nx.DiGraph, thresholds: list) -> nx.DiGraph:
    """
    Assigns community clusters from the community graph back to the nodes in the original graph.
    
    Parameters:
    - original_graph (nx.DiGraph): The original node-level graph
    - community_graph (nx.DiGraph): The community-level graph with cluster attributes
    - thresholds (list): List of threshold levels used for clustering
    
    Returns:
    - nx.DiGraph: Original graph with community cluster assignments added as node attributes
    """
    print(f"\nAssigning community clusters to original graph nodes...")
    print(f"Processing {len(thresholds)} threshold levels: {thresholds}")
    
    # Create a mapping of nodes to their communities
    node_to_community = {}
    for comm_id, comm_data in community_graph.nodes(data=True):
        # Get the list of original nodes in this community
        original_nodes = comm_data['original_nodes'].split(',')
        for node in original_nodes:
            node_to_community[node] = comm_id
    
    print(f"Found {len(set(node_to_community.values()))} unique communities")
    
    # For each threshold level, assign community clusters to original nodes
    for threshold in thresholds:
        print(f"\nProcessing threshold level {threshold}:")
        attr_name = f'community_ancestors_threshold{threshold}'
        
        # Create mapping of community IDs to their cluster assignments
        community_clusters = nx.get_node_attributes(community_graph, attr_name)
        unique_clusters = set(community_clusters.values())
        print(f"- Found {len(unique_clusters)} unique clusters at threshold {threshold}")
        
        # Track nodes per cluster
        cluster_node_counts = {cluster: 0 for cluster in unique_clusters}
        unassigned_count = 0
        
        # Assign clusters to original nodes based on their community membership
        for node in original_graph.nodes():
            comm_id = node_to_community.get(str(node))  # Convert node to string as GraphML stores IDs as strings
            if comm_id is not None:
                cluster = community_clusters.get(comm_id, -1)  # Default to -1 if no cluster assigned
                original_graph.nodes[node][attr_name] = cluster
                if cluster != -1:
                    cluster_node_counts[cluster] += 1
            else:
                print(f"Warning: Node {node} not found in any community")
                original_graph.nodes[node][attr_name] = -1
                unassigned_count += 1
        
        # Print cluster statistics
        print(f"- Node distribution across clusters:")
        for cluster, count in sorted(cluster_node_counts.items()):
            print(f"  Cluster {cluster}: {count} nodes")
        if unassigned_count > 0:
            print(f"  Unassigned nodes: {unassigned_count}")
    
    print("\nCommunity cluster assignment complete!")

    # Print all attributes of first node before returning
    """first_node = list(original_graph.nodes())[0]  # Get first node ID
    print("\nFinal attributes of first node:")
    for attr, value in original_graph.nodes[first_node].items():
        print(f"{attr}: {value}")"""
    return original_graph

# -------------------- Main Execution --------------------
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

for tree_id, filename in fileNameDic.items():
    graphPath = f'{folderPath}/processedQSMs/{filename}_clusterGraph.graphml'
    voxelDF = pd.read_csv(f'{folderPath}/elmVoxelDFs/{filename}_voxelDF.csv')

    # Read the graph
    G = nx.read_graphml(graphPath)

    # Rename start coordinates to x,y,z
    mapping = {
        'startX': 'x',
        'startY': 'y', 
        'startZ': 'z'
    }
    for old_attr, new_attr in mapping.items():
        nx.set_node_attributes(G, 
                                {n: G.nodes[n][old_attr] for n in G.nodes() if old_attr in G.nodes[n]},
                                new_attr)
        # Remove old attributes
        for n in G.nodes():
            if old_attr in G.nodes[n]:
                del G.nodes[n][old_attr]

    ##CONVERT GRAPH TO COMMUNITY GRAPH

    # Ensure the graph is directed
    if not G.is_directed():
        print(f"Warning: Graph {graphPath} is not directed. Converting to directed.")
        G = G.to_directed()

    # Apply Community Detection Algorithms
    print("Applying Leiden Algorithm...")
    G, leiden_count = apply_leiden_algorithm(
        G,
        weight_attr='startRadius',
        prioritize_main_stem=True
    )
    print(f"Leiden detected {leiden_count} communities.")

    community_graph = create_community_graph(G)
    validate_graph(G, community_graph)

    ##CLUSTER COMMUNITY GRAPH
    thresholds = [0, 1, 2]
    community_graph = adTree_CommunityGraphClustering.processCommunityGraph(community_graph, thresholds, visualize=False)

    ###ASSIGN COMMUNITY CLUSTERS TO NODES IN ORIGINAL GRAPH
    G = assign_community_clusters(G, community_graph, thresholds)

    updatedGraphPath = f'{folderPath}/processedGraph/'
    #check if path exists if not make folder
    if not os.path.exists(updatedGraphPath):
        os.makedirs(updatedGraphPath)
    
        # Print all attributes of first node before returning
    first_node = list(G.nodes())[0]  # Get first node ID
    print("\nFinal attributes of first node:")
    for attr, value in G.nodes[first_node].items():
        print(f"{attr}: {value}")

    path = f'{updatedGraphPath}/{filename}_processedGraph.graphml'
    nx.write_graphml(G, path)
    print(f'saving updated graph')

    path = f'{updatedGraphPath}/{filename}_communityGraph.graphml'
    nx.write_graphml(community_graph, path)
    print(f'saving community graph')

    # Prepare a list of graphs where each graph has all community attributes
    # Since all attributes are in the same graph, pass the same graph multiple times
    
    """points = np.array(voxelDF[['x', 'y', 'z']])
    voxelPoly = pv.PolyData(points)
    graphPoly = create_pyvista_polydata(G)

    community_detection_method_names = ['community-leiden']
    for threshold in thresholds:
        attr_name = f'community_ancestors_threshold{threshold}'
        community_detection_method_names.append(attr_name)
    # Plot all community detection results in subplots using PyVista
        # Initialize the plotter with multiple subplots
    num_methods = len(community_detection_method_names)
    plotter = pv.Plotter(shape=(1, num_methods))  # 1 row, num_methods columns

    for i, method_name in enumerate(community_detection_method_names):
        # Make a copy of the polydata
        poly_copy = graphPoly.copy()
        # Print all attributes in poly_copy
        print(f"\nAttributes for {method_name}:")
        print("Point data attributes:", poly_copy.point_data.keys())
        print("Cell data attributes:", poly_copy.cell_data.keys())
        print("Field data attributes:", poly_copy.field_data.keys())
        # Set the active subplot
        plotter.subplot(0, i)

        # Add the voxel mesh to the subplot
        plotter.add_mesh(voxelPoly, color='grey', point_size=10, opacity=0.25, render_points_as_spheres=True)
        
        # Add the poly mesh to the subplot with the current scalar field
        poly_copy = showRandomPolyCols(poly_copy, method_name)
        #plotter.add_mesh(poly_copy, scalars=method_name, cmap='Set2', line_width=10)
        plotter.add_mesh(poly_copy, scalars='colors', rgb=True, line_width=10)
        
        # Add title to each subplot
        plotter.add_title(method_name, font_size=10)


        # Show the plot with subplots
    plotter.show()
    plotter.close()"""

        





