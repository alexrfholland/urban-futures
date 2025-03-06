import pandas as pd
import pickle
from pathlib import Path
import pyvista as pv
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import json
import aa_tree_helper_functions

###HELPER FUNCTIONS###
def convertToPoly(voxelDF):

    points = voxelDF[['x', 'y', 'z']].values
    poly = pv.PolyData(points)

    # Add all columns as point data attributes
    for col in voxelDF.columns:
        if col not in ['x', 'y', 'z']:  # Skip coordinate columns
            poly.point_data[col] = voxelDF[col].values

    return poly


def create_color_mapping(poly, attribute_name):
    # Ensure the attribute exists in point or cell data
    if attribute_name not in poly.point_data and attribute_name not in poly.cell_data:
        raise ValueError(f"Attribute '{attribute_name}' not found in point or cell data.")

    # Decide whether the attribute is point or cell-based
    if attribute_name in poly.point_data:
        data_array = poly.point_data[attribute_name]
    else:
        data_array = poly.cell_data[attribute_name]

    # Generate random colors
    n_clusters = data_array.max() + 1  # Assuming cluster IDs start from 0
    cluster_colors = np.random.rand(n_clusters, 3)  # Random RGB colors

    # Map colors based on the scalar field
    color_array = cluster_colors[data_array]

    # Assign the colors to the appropriate data set
    if attribute_name in poly.point_data:
        poly.point_data['colors'] = color_array
    else:
        poly.cell_data['colors'] = color_array

    print(f'poly has {poly.n_points} points')
    print(f'color array has shape: {color_array.shape}')

    return poly


### SECTION 1: REGNERATE SNAGS ####

def should_prune_branch(node, graph, maxRadius=0.2):
    """Check if a branch should be pruned based on radius and community"""
    radius = float(graph.nodes[node].get('startRadius', .5))
    community = int(graph.nodes[node].get('community_ancestors_threshold0', -1))
    
    # Get parent info
    parents = list(graph.predecessors(node))
    parent = parents[0] if parents else None
    parent_radius = float(graph.nodes[parent].get('startRadius', .5)) if parent else 0
    parent_community = int(graph.nodes[parent].get('community_ancestors_threshold0', -1)) if parent else -1
    
    trunk_radius_threshold = .15
    # Special handling for trunk (community 0)
    if community == 0:
        # For trunk, use radius threshold of 1.0
        if radius < trunk_radius_threshold:
            # Don't prune if parent has radius > 1.0
            if parent and parent_radius >= trunk_radius_threshold:
                return radius, False
            return radius, True
        return radius, False
    
    # For non-trunk nodes
    # Don't prune if parent is trunk (community 0) AND meets radius threshold
    if parent and parent_community == 0 and parent_radius >= maxRadius:
        return radius, False
        
    # Don't prune if parent meets radius threshold
    if parent and parent_radius >= maxRadius:
        return radius, False
    
    # Otherwise, prune if below threshold
    should_prune = radius < maxRadius
    return radius, should_prune

def find_terminal_branches(graph):
    """
    Find terminal branches in the snag (enabled branches with no enabled children).
    
    Args:
        graph: NetworkX graph with isPruned attributes
    """
    for node in graph.nodes():
        # Skip pruned branches
        if graph.nodes[node].get('isPruned', False):
            continue
            
        # Check if this is a terminal branch (no enabled children)
        children = list(graph.successors(node))
        is_terminal = all(graph.nodes[child].get('isPruned', False) for child in children)
        
        graph.nodes[node]['isTerminal'] = is_terminal

def compare_snag_templates():
    """
    Compare original snag templates with corresponding elm templates.
    """
    print('Loading templates for comparison')
    
    # Define template directory and file paths
    template_dir = Path('data/revised/trees')
    euc_path = template_dir / 'updated_tree_dict.pkl'
    elm_path = template_dir / 'elm_tree_dict.pkl'
    
    # Load templates
    euc_tree_templates = pickle.load(open(euc_path, 'rb'))
    elm_tree_templates = pd.read_pickle(elm_path)
    
    # Tree IDs to compare
    tree_ids = [8, 9, 10, 11, 12, 13, 14]
    
    for tree_id in tree_ids:
        print(f'\nComparing templates for tree_id: {tree_id}')
        
        # Get original snag template
        orig_key = (False, 'snag', 'improved-tree', tree_id)
        orig_template = euc_tree_templates.get(orig_key)
        
        if orig_template is None:
            print(f'No original template found for {orig_key}')
            continue
            
        # Get corresponding elm template
        elm_match = elm_tree_templates[
            (elm_tree_templates['precolonial'] == False) &
            (elm_tree_templates['size'] == 'large') &
            (elm_tree_templates['control'] == 'reserve-tree') &
            (elm_tree_templates['tree_id'] == tree_id)
        ]
        
        if len(elm_match) == 0:
            print(f'No matching elm template found for tree_id {tree_id}')
            continue
            
        elm_template = elm_match.iloc[0]['template']
        
        # Create PolyData objects
        orig_points = orig_template[['x', 'y', 'z']].values
        elm_points = elm_template[['x', 'y', 'z']].values
        
        orig_poly = pv.PolyData(orig_points)
        elm_poly = pv.PolyData(elm_points)
        
        # Create a plotter
        p = pv.Plotter()
        
        # Add both point clouds with different colors
        p.add_mesh(orig_poly, color='red', point_size=5, render_points_as_spheres=True, label='Original Snag')
        p.add_mesh(elm_poly, color='blue', point_size=5, render_points_as_spheres=True, label='Elm Template')
        
        p.add_legend()
        p.show(window_size=[1024, 768])
        
        # Print some statistics
        print(f'Original template points: {len(orig_points)}')
        print(f'Elm template points: {len(elm_points)}')
        
        # Calculate centroid difference
        orig_centroid = np.mean(orig_points, axis=0)
        elm_centroid = np.mean(elm_points, axis=0)
        centroid_diff = np.linalg.norm(orig_centroid - elm_centroid)
        print(f'Centroid difference: {centroid_diff:.2f} units')


def regenerate_snags(elm_tree_templates, graph_dict, maxRadius=0.2):
    """Create new snag versions using elm templates and graphs."""
    print('Creating new snags')

    print(f'elm templates are {elm_tree_templates.head()}')
    
    snag_dict = {}

    print(f'Graph dictionary keys: {list(graph_dict.keys())}')
    
    for tree_id, graph in graph_dict.items():
        print(f'\nProcessing snag for tree_id: {tree_id}')
        
        # Get elm template
        elm_match = elm_tree_templates[
            (elm_tree_templates['precolonial'] == False) &
            (elm_tree_templates['size'] == 'large') &
            (elm_tree_templates['control'] == 'reserve-tree') &
            (elm_tree_templates['tree_id'] == tree_id)
        ]
        
        if len(elm_match) == 0:
            print(f'No matching elm template found for tree_id {tree_id}')
            continue
            
        template = elm_match.iloc[0]['template'].copy()
        
        # Initialize tracking sets
        visited = set()
        
        # Initialize graph attributes
        for node in graph.nodes():
            graph.nodes[node]['isPruned'] = False
            graph.nodes[node]['isTerminal'] = False
        
        # Get leaf nodes (nodes with no successors)
        leaf_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        print(f"Found {len(leaf_nodes)} leaf nodes")
        
        # Process each leaf node and work upwards
        for leaf in leaf_nodes:
            if leaf in visited:
                continue
                
            current_node = leaf
            continue_branch = True
            
            while continue_branch and current_node is not None:
                if current_node in visited:
                    break
                    
                visited.add(current_node)
                
                radius, should_prune = should_prune_branch(current_node, graph, maxRadius)
                print(f"Node {current_node}: radius = {radius:.6f} - {'PRUNE' if should_prune else 'KEEP'}")
                
                # Update isPruned attribute in graph
                graph.nodes[current_node]['isPruned'] = should_prune
                
                if should_prune:
                    parents = list(graph.predecessors(current_node))
                    if parents:
                        current_node = parents[0]
                    else:
                        current_node = None
                else:
                    continue_branch = False
        
        # Find terminal branches
        find_terminal_branches(graph)
        
        # Count pruned and terminal branches
        pruned_count = sum(1 for n in graph.nodes() if graph.nodes[n]['isPruned'])
        terminal_count = sum(1 for n in graph.nodes() if graph.nodes[n]['isTerminal'])
        print(f"Number of pruned branches: {pruned_count}")
        print(f"Number of terminal branches: {terminal_count}")
        
        # Update template based on graph attributes
        template['isPruned'] = template['cluster_id'].astype(str).map(
            {str(n): graph.nodes[n]['isPruned'] for n in graph.nodes()}
        ).fillna(False)
        
        template['isTerminal'] = template['cluster_id'].astype(str).map(
            {str(n): graph.nodes[n]['isTerminal'] for n in graph.nodes()}
        ).fillna(False)

        # update resource_perch branch where isTerimanl = True
        template.loc[template['isTerminal'] == True, 'resource_perch branch'] = 1

        # Create snag template
        snag_template = template[template['isPruned'] == False].copy()
        snag_dict[tree_id] = snag_template

        # update resource to 'perch branch' where resource_perch branch is 1
        snag_template.loc[snag_template['resource_perch branch'] == 1, 'resource'] = 'perch branch'
        

        print(f'Created snag template for tree {tree_id} with {len(snag_template)} points')
        
        # Visualize the result
        """points = snag_template[['x', 'y', 'z']].values
        poly = pv.PolyData(points)
        poly.point_data['isTerminal'] = snag_template['isTerminal'].values
        
        p = pv.Plotter()
        p.add_mesh(poly, scalars='isTerminal', point_size=5, render_points_as_spheres=True)
        p.show()"""
    
    return snag_dict

### SECTION 2: UPDATE ORIGINAL SNAG TEMPLATES ###

def find_best_matches_vectorized(orig_points, regenerated_template):
    """
    Vectorized function to find best matches using optimized cKDTree queries.
    Maximum distance cap of 2.0 meters for any match.
    """
    print(f"\nMatching {len(orig_points)} original points to {len(regenerated_template)} template points")
    
    # Get new points from template
    new_points = regenerated_template[['x', 'y', 'z']].values
    
    # Build KD-tree once with optimized leaf size
    print("Building KD-tree...")
    tree = cKDTree(new_points, leafsize=16)
    
    # Query only k nearest neighbors for each point
    k = 10  # Adjust k based on your needs
    print(f"Finding {k} nearest neighbors for each point...")
    distances, indices = tree.query(orig_points, k=k, workers=-1)
    
    # Get candidate data directly from new_template
    print("Calculating priorities...")
    candidate_radii = regenerated_template['start_radius'].values[indices]
    candidate_terminals = regenerated_template['isTerminal'].values[indices]
    
    # Calculate priorities using numpy (no DataFrame needed)
    priorities = np.ones_like(distances)
    priorities[candidate_radii > 0.2] = 4
    priorities[candidate_terminals] = 3
    priorities[(priorities == 1) & (candidate_radii > 0.1)] = 2
    
    # Set priority to -inf for points beyond max distance
    priorities[distances > 2.0] = -np.inf
    
    # Calculate scores
    scores = -priorities + (distances * 0.0001)
    
    # Find best match for each point
    best_indices = np.argmin(scores, axis=1)
    row_indices = np.arange(len(orig_points))
    
    # Get final results
    matched_indices = indices[row_indices, best_indices]
    matched_distances = distances[row_indices, best_indices]
    
    # Create mask for valid matches (within distance cap)
    valid_matches = matched_distances <= 2.0
    
    # Set invalid matches to -1
    matched_indices[~valid_matches] = -1
    matched_distances[~valid_matches] = np.inf
    
    # Print matching statistics
    print("\nMatching Statistics:")
    print(f"Total points: {len(orig_points)}")
    close_matches = matched_distances <= 0.25
    medium_matches = (matched_distances > 0.25) & (matched_distances <= 2.0)
    unmatched = matched_distances > 2.0
    
    print(f"Points matched within 0.25 radius: {close_matches.sum()} ({close_matches.sum()/len(orig_points)*100:.1f}%)")
    print(f"Points matched between 0.25 and 2.0 radius: {medium_matches.sum()} ({medium_matches.sum()/len(orig_points)*100:.1f}%)")
    print(f"Unmatched points (>2.0): {unmatched.sum()} ({unmatched.sum()/len(orig_points)*100:.1f}%)")
    
    if valid_matches.any():
        print(f"\nAverage match distance: {matched_distances[valid_matches].mean():.3f}")
        
        # Print priority statistics for valid matches only
        matched_priorities = priorities[row_indices, best_indices][valid_matches]
        print("\nPriority Statistics for Matched Points:")
        print(f"Priority 4 (radius > 0.2): {(matched_priorities == 4).sum()} ({(matched_priorities == 4).sum()/valid_matches.sum()*100:.1f}%)")
        print(f"Priority 3 (terminal): {(matched_priorities == 3).sum()} ({(matched_priorities == 3).sum()/valid_matches.sum()*100:.1f}%)")
        print(f"Priority 2 (radius > 0.1): {(matched_priorities == 2).sum()} ({(matched_priorities == 2).sum()/valid_matches.sum()*100:.1f}%)")
        print(f"Priority 1 (other): {(matched_priorities == 1).sum()} ({(matched_priorities == 1).sum()/valid_matches.sum()*100:.1f}%)")
    
    return matched_indices, matched_distances

def update_original(euc_tree_templates, regenerated_snag_dict):
    """
    Update original templates with attributes from regenerated templates,
    maintaining original point positions but removing unmatched points.
    """
    print('Updating original templates')
    updated_templates = {}

    #iterate through euc_tree_templates and print all keys where key[0] == False and key[2] == 'snag'
    for key in euc_tree_templates.keys():
        if key[0] == False and key[1] == 'snag':
            print('snag key found:')
            print(key)
    
    for tree_id, regenerated_template in regenerated_snag_dict.items():
        print(f'\nProcessing updates for tree_id: {tree_id}')
        
        # Get original snag template
        orig_key = (False, 'snag', 'improved-tree', tree_id)
        orig_template = euc_tree_templates.get(orig_key)
        
        if orig_template is None:
            print(f'No original template found for {orig_key}')
            continue
        
        # Find best matches
        matched_indices, matched_distances = find_best_matches_vectorized(
            orig_template[['x', 'y', 'z']].values,
            regenerated_template
        )
        
        # Create mask for valid matches
        valid_matches = matched_indices != -1
        
        # Create updated template starting with original template, keeping only matched points
        updated_template = orig_template[valid_matches][['x', 'y', 'z']].copy()
        
        # Get the matched indices for valid matches only
        valid_matched_indices = matched_indices[valid_matches]
        
        # Transfer attributes from regenerated template for valid matches
        attribute_columns = [col for col in regenerated_template.columns if col not in ['x', 'y', 'z']]
        for col in attribute_columns:
            # Use the valid_matched_indices to get the correct rows from regenerated_template
            updated_template[col] = regenerated_template.iloc[valid_matched_indices][col].values
        
        # Reset index to make it sequential
        updated_template = updated_template.reset_index(drop=True)
        
        # Store updated template
        updated_templates[tree_id] = updated_template
        
        # Print statistics
        print(f"\nTemplate Update Statistics for tree {tree_id}:")
        print(f"Original points: {len(orig_template)}")
        print(f"Points matched and kept: {valid_matches.sum()} ({valid_matches.sum()/len(orig_template)*100:.1f}%)")
        print(f"Points removed (unmatched): {(~valid_matches).sum()} ({(~valid_matches).sum()/len(orig_template)*100:.1f}%)")
        print(f"Attributes transferred: {len(attribute_columns)}")
        print(f"Average match distance: {matched_distances[valid_matches].mean():.3f}")
        
    
    return updated_templates

### SECTION 3: UPDATE RESOURCE VALUES ###
def update_resource_values(regenerated_snags, updated_snags, resourceDF):
    import adTree_AssignResources    
    
        # Get senescing resources from DataFrame
    senescing_mask = (resourceDF['precolonial'] == False) & \
                     (resourceDF['size'] == 'senescing') & \
                     (resourceDF['control'] == 'improved-tree')
    
    senescing_resources = resourceDF[senescing_mask].iloc[0]

    def process_template(template, seed):
        """Process a single template with resource assignments"""
        template = template.copy()  # Create a copy to make modifications explicit
        
        # Initialize resource columns
        template['resource_hollow'] = 0
        template['resource_epiphyte'] = 0
        template['resource_peeling bark'] = 0
        template['isSenescent'] = True
        
        # Assign resources
        template = adTree_AssignResources.assign_peeling_bark(
            template, 
            senescing_resources['peeling bark'], 
            "resource_peeling bark", 
            seed=seed+1
        )
        template['resource_dead branch'] = 1
        
        template = adTree_AssignResources.assign_hollows_and_epiphytes(
            template, 
            senescing_resources['hollow'], 
            senescing_resources['epiphyte'],
            'resource_hollow', 
            'resource_epiphyte', 
            seed
        )
        
        template = adTree_AssignResources.create_resource_column(template)
        template = aa_tree_helper_functions.verify_resources_columns(template)
        return template

    def print_template_stats(template, name):
        """Print statistics for a template"""
        print(f'\nStatistics for {name} template:')
        print(f'Total points: {len(template)}')
        
        # Get cluster-level statistics
        cluster_df = template.groupby('cluster_id').first().reset_index()
        
        # Save cluster data
        cluster_df.to_csv(f'data/revised/trees/cluster_df_{name}.csv', index=False)
        
        # Print resource statistics at cluster level
        resource_counts = {
            'hollows': len(template[template['resource_hollow'] == 1]['cluster_id'].unique()),
            'epiphytes': len(template[template['resource_epiphyte'] == 1]['cluster_id'].unique()),
            'peeling_bark': len(template[template['resource_peeling bark'] == 1]['cluster_id'].unique()),
            'perch_branches': len(template[template['resource_perch branch'] == 1]['cluster_id'].unique())
        }
        
        for resource, count in resource_counts.items():
            print(f'Number of {resource}: {count}')
            
        print('\nResource distribution at cluster level:')
        print(cluster_df['resource'].value_counts())
        
        return cluster_df

    # Process all templates
    for tree_id in regenerated_snags:
        print(f'\nProcessing tree {tree_id}')
        
        # Process both template types
        regenerated_snags[tree_id] = process_template(regenerated_snags[tree_id], seed=tree_id)
        updated_snags[tree_id] = process_template(updated_snags[tree_id], seed=tree_id)
        
        # Print statistics
        print_template_stats(regenerated_snags[tree_id], f'regenerated_tree_{tree_id}')
        print_template_stats(updated_snags[tree_id], f'updated_tree_{tree_id}')

    return updated_snags, regenerated_snags

def load_files():
    """Load all necessary files and prepare graph dictionary."""
    print('Loading templates and graphs')
    
    # Define paths
    template_dir = Path('data/revised/trees')
    elm_path = template_dir / 'elm_tree_dict.pkl'
    euc_path = template_dir / 'updated_tree_dict.pkl'
    graph_dir = Path('data/revised/lidar scans/elm/adtree/processedGraph')
    
    # File mapping dictionary
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
    
    # Create filename dictionary
    filename_dict = {tree_id: filename.split('_')[0] for filename, tree_id in point_cloud_files.items()}
    
    # Load templates
    print(f'Loading elm templates from {elm_path}')
    elm_tree_templates = pd.read_pickle(elm_path)
    print(f'Loading euc templates from {euc_path}')
    euc_tree_templates = pickle.load(open(euc_path, 'rb'))
    
    # Create graph dictionary
    graph_dict = {}
    print(f'\nLooking for graphs in {graph_dir}')
    
    for tree_id in [7, 8, 9, 10, 11, 12, 13, 14]:
        if tree_id not in filename_dict:
            print(f'Warning: No filename mapping for tree {tree_id}')
            continue
            
        filename = filename_dict[tree_id]
        graph_path = graph_dir / f'{filename}_processedGraph.graphml'
        print(f'Loading graph from: {graph_path}')
        
        if graph_path.exists():
            graph = nx.read_graphml(graph_path)
            graph_dict[tree_id] = graph
            print(f'Loaded graph for tree {tree_id}')
        else:
            print(f'Warning: No graph found at {graph_path}')
    
    print(f'\nLoaded {len(graph_dict)} graphs')
    print(f'Graph dictionary keys: {list(graph_dict.keys())}')
    
    # Load resource DataFrame
    resourceDFPath = 'data/revised/trees/resource_dicDF.csv'
    resourceDF = pd.read_csv(resourceDFPath)
    
    return elm_tree_templates, euc_tree_templates, graph_dict, resourceDF




def process_snags(euc_templates, elm_templates, graph_dict, resourceDF):    
    print('\n1. Create new snags')
    new_snags = regenerate_snags(elm_templates, graph_dict, maxRadius=0.1)

    print('\n2. Update original templates')
    updated_templates = update_original(euc_templates, new_snags)

    print('\n3. Update resource values')
    updated_snags, regenerated_snags = update_resource_values(new_snags, updated_templates, resourceDF)



  

    return updated_snags, regenerated_snags

if __name__ == '__main__':
    elm_templates, euc_templates, graph_dict, resourceDF = load_files()
    updated_snags, regenerated_snags = process_snags(euc_templates, elm_templates, graph_dict, resourceDF)

    # Convert templates to pyvista PolyData
    updated_snag = aa_tree_helper_functions.convertToPoly(updated_snags[7])
    regen_snag = aa_tree_helper_functions.convertToPoly(regenerated_snags[7])

    print(updated_snags.keys())

    # Create pyvista plotter with 2 subplots
    plotter = pv.Plotter(shape=(1,2))
    
    # Plot updated snag colored by resource values
    plotter.subplot(0,0)
    plotter.add_mesh(updated_snag, scalars='resource', cmap='viridis')
    plotter.add_text("Updated Snag")
    
    """# Plot regenerated snag colored by resource values  
    plotter.subplot(0,1)
    plotter.add_mesh(regen_snag, scalars='resource', cmap='viridis')
    plotter.add_text("Regenerated Snag")"""
    
    # Show the plots
    plotter.show()
