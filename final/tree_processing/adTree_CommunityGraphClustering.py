import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import numpy as np

def find_origin_node(G):
    """Find the origin node (root) in a directed graph by looking for the node with no incoming edges."""
    for node in G.nodes():
        if G.in_degree(node) == 0:
            return node
    return None

def get_all_descendants(G, node):
    """Get all descendants of a node (no depth limit)."""
    descendants = set()
    for successor in G.successors(node):
        descendants.add(successor)
        descendants.update(get_all_descendants(G, successor))
    return descendants

def create_clusters_multi_threshold(G, thresholds=[0, 1, 2]):
    """
    Create clusters for multiple threshold depths and add cluster assignments as node attributes.
    Each threshold gets its own node attribute in format: community_ancestors_threshold{threshold}
    """
    clusters_by_threshold = {}
    
    for threshold in thresholds:
        # Find the origin node (only needs to be done once, but keeping in loop for clarity)
        origin = find_origin_node(G)
        if origin is None:
            raise ValueError("No origin node found in the graph")
        
        def get_descendants_with_depth(node, max_depth, current_depth=0):
            if current_depth >= max_depth:
                return set()
            descendants = set()
            for successor in G.successors(node):
                descendants.add(successor)
                descendants.update(get_descendants_with_depth(successor, max_depth, current_depth + 1))
            return descendants
        
        # Create Cluster 0 for this threshold
        cluster_0_nodes = {origin}
        cluster_0_nodes.update(get_descendants_with_depth(origin, threshold))
        
        # Find boundary nodes for this threshold
        boundary_nodes = set()
        for node in cluster_0_nodes:
            for child in G.successors(node):
                if child not in cluster_0_nodes:
                    boundary_nodes.add(child)
        
        # Create clusters for this threshold
        clusters = {0: cluster_0_nodes}
        
        # Create new clusters for each boundary node
        for cluster_id, boundary_node in enumerate(boundary_nodes, 1):
            cluster_nodes = {boundary_node}
            cluster_nodes.update(get_all_descendants(G, boundary_node))
            clusters[cluster_id] = cluster_nodes
        
        # Store clusters for this threshold
        clusters_by_threshold[threshold] = clusters
        
        # Add cluster assignments as node attributes for this threshold
        attr_name = f'community_ancestors_threshold{threshold}'
        nx.set_node_attributes(G, -1, attr_name)  # Default value for unclustered nodes
        for cluster_id, nodes in clusters.items():
            for node in nodes:
                G.nodes[node][attr_name] = cluster_id
    
    return clusters_by_threshold, G

def processCommunityGraph(communityGraph, thresholds=[0, 1, 2], visualize=True):
    """Process community graph with multiple threshold levels."""
    clusters_by_threshold, communityGraph = create_clusters_multi_threshold(communityGraph, thresholds)
    
    # Analyze and visualize for each threshold
    for threshold in thresholds:
        print(f'\nAnalyzing threshold {threshold}:')
        print('=' * 30)
        clusters = clusters_by_threshold[threshold]
        print(f'clusters: {clusters}')
        analyze_clusters(communityGraph, clusters)
        if visualize:
            plt.figure()
            plt.title(f"Community Clusters (Threshold={threshold})")
            visualize_clusters(communityGraph, clusters)
    
    return communityGraph


def generate_distinct_colors(n):
    """Generate n distinct colors using HSV color space."""
    import colorsys
    
    colors = []
    for i in range(n):
        # Use golden ratio to spread hues evenly
        hue = i * 0.618033988749895  # golden ratio conjugate
        hue %= 1.0
        # Vary saturation and value for more distinct colors
        saturation = 0.5 + (i % 3) * 0.2  # 0.5, 0.7, 0.9
        value = 0.95 - (i % 4) * 0.1      # 0.95, 0.85, 0.75, 0.65
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    
    return colors

def visualize_clusters(G, clusters):
    """Visualize the graph with nodes colored by cluster using existing x and y attributes."""
    plt.figure(figsize=(15, 10))
    
    # Create position dictionary using x and y attributes
    pos = {node: (float(G.nodes[node]['x']), float(G.nodes[node]['z'])) for node in G.nodes()}
    
    # Generate distinct colors for all clusters
    num_clusters = len(clusters)
    colors = generate_distinct_colors(num_clusters)
    
    # Draw nodes for each cluster separately
    for cluster_id in clusters:
        nodes = clusters[cluster_id]
        node_color = 'red' if cluster_id == 0 else colors[cluster_id]
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=list(nodes),
                             node_color=[node_color] * len(nodes),
                             node_size=100,
                             label=f'Cluster {cluster_id}')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True, arrowsize=10)
    
    # Draw labels with smaller font
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    
    plt.title("Community Clusters")
    plt.axis('on')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def analyze_clusters(G, clusters):
    """Analyze the clusters and print statistics."""
    print("\nCluster Analysis:")
    print("-----------------")
    
    for cluster_id, nodes in clusters.items():
        print(f"\nCluster {cluster_id}:")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Nodes: {sorted(nodes)}")
        
        if cluster_id == 0:
            # For Cluster 0, show which nodes have children in other clusters
            boundary_parents = {node for node in nodes 
                              if any(child not in nodes for child in G.successors(node))}
            print(f"Nodes with children outside cluster: {sorted(boundary_parents)}")
        else:
            # For other clusters, show the parent nodes from Cluster 0
            parents_in_cluster0 = {node for node in G.predecessors(min(nodes)) 
                                 if node in clusters[0]}
            print(f"Parents in Cluster 0: {sorted(parents_in_cluster0)}")
        
        # Calculate internal edges
        internal_edges = G.subgraph(nodes).number_of_edges()
        print(f"Internal edges: {internal_edges}")
        
        # Calculate external edges
        external_edges = sum(1 for u, v in G.edges() 
                           if (u in nodes and v not in nodes) or 
                              (u not in nodes and v in nodes))
        print(f"External edges: {external_edges}")

    
    
def main():
    # Set up file paths
    folderPath = 'data/revised/lidar scans/elm/adtree'
    updatedGraphPath = f'{folderPath}/processedGraph/'
    
    # Load the graph
    communityGraph = nx.read_graphml(f"{updatedGraphPath}/Large Elm A 1 mil_communityGraph.graphml")
    processCommunityGraph(communityGraph, visualize=True)

if __name__ == "__main__":
    main()