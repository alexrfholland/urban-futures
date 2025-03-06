import pickle
import pandas as pd
import pyvista as pv
import numpy as np
import os
from pathlib import Path
from scipy.spatial import cKDTree
import aa_tree_helper_functions

def get_resource_selection(resource_names):
    """Prompt user to select a resource from the list."""
    print("\nAvailable resources:")
    print("0. Finish and save")
    for i, resource in enumerate(resource_names, 1):
        print(f"{i}. {resource}")
    
    while True:
        try:
            selection = int(input("\nSelect resource number (0-7): "))
            if selection == 0:
                return None
            if 1 <= selection <= len(resource_names):
                return resource_names[selection - 1]
            print("Invalid selection. Please choose a number between 0 and 7.")
        except ValueError:
            print("Please enter a valid number.")

def setup_plotter_text(plotter, current_resource):
    """Setup the instruction text in the plotter."""
    print(f"\nProcessing {current_resource}")
    print("\nInstructions:")
    print("1. Hover over points and press:")
    print("   - 'p' to assign (red)")
    print("   - 'x' to exclude (grey)")
    print("2. Press 'r' to toggle radius mode (affects clusters within 1m)")
    print("3. Press 'q' when done with this resource")
    
    plotter.add_text(
        f"Resource: {current_resource}\n"
        "Hover and press:\n"
        "'p' to assign (red)\n"
        "'x' to exclude (grey)\n"
        "'r' to toggle radius mode\n"
        "'q' to finish",
        position='upper_left',
        font_size=12
    )

def setup_picking_interface(plotter, poly, cluster_df, resource_names):
    """Setup and run the picking interface."""
    points = poly.points
    point_tree = cKDTree(points)
    radius_mode = False
    current_mode = 'assign'
    current_resource_idx = 0
    current_resource = resource_names[current_resource_idx]
    original_colors = poly.point_data['colors'].copy()
    
    def process_cluster_selection(cluster_id, action='assign'):
        value = 1 if action == 'assign' else 0  # Changed: 1 = assigned, 0 = excluded
        color = [1, 0, 0] if action == 'assign' else [0.5, 0.5, 0.5]  # Red or Grey
        
        if radius_mode:
            # Find all clusters within 1m
            cluster_points = points[poly.point_data['cluster_id'] == cluster_id]
            if len(cluster_points) == 0:
                return
            
            nearby_indices = point_tree.query_ball_point(cluster_points[0], r=1.0)
            nearby_clusters = set(poly.point_data['cluster_id'][nearby_indices])
            
            print(f"Selected cluster {cluster_id} and nearby clusters: {nearby_clusters}")
            
            # Update all nearby clusters
            for nearby_cluster in nearby_clusters:
                cluster_df.loc[cluster_df['cluster_id'] == nearby_cluster, current_resource] = value
                mask = poly.point_data['cluster_id'] == nearby_cluster
                poly.point_data['colors'][mask] = color
        else:
            # Update single cluster
            print(f"Selected cluster {cluster_id}")
            cluster_df.loc[cluster_df['cluster_id'] == cluster_id, current_resource] = value
            mask = poly.point_data['cluster_id'] == cluster_id
            poly.point_data['colors'][mask] = color
    
    def process_picked_point():
        if plotter.picked_point is None:
            return
            
        picked_position = plotter.picked_point
        _, index = point_tree.query(picked_position)
        cluster_id = poly.point_data['cluster_id'][index]
        
        process_cluster_selection(cluster_id, current_mode)
        plotter.render()
    
    def toggle_mode():
        nonlocal current_mode
        current_mode = 'exclude' if current_mode == 'assign' else 'assign'
        mode_text = 'EXCLUDE (grey)' if current_mode == 'exclude' else 'ASSIGN (red)'
        print(f"Mode: {mode_text}")
    
    def cycle_resource():
        nonlocal current_resource_idx, current_resource
        current_resource_idx = (current_resource_idx + 1) % len(resource_names)
        current_resource = resource_names[current_resource_idx]
        print(f"\nSwitched to: {current_resource}")
        
        # Reset colors to original
        poly.point_data['colors'] = original_colors.copy()
        
        # Apply current resource's colors
        for idx, row in cluster_df.iterrows():
            if row[current_resource] == 1:  # Assigned
                mask = poly.point_data['cluster_id'] == row['cluster_id']
                poly.point_data['colors'][mask] = [1, 0, 0]
            elif row[current_resource] == 0:  # Excluded (changed from -1)
                mask = poly.point_data['cluster_id'] == row['cluster_id']
                poly.point_data['colors'][mask] = [0.5, 0.5, 0.5]
            # -1 means unassigned, keep original color
        
        plotter.render()
    
    def toggle_radius_mode():
        nonlocal radius_mode
        radius_mode = not radius_mode
        status = "ON" if radius_mode else "OFF"
        print(f"Radius mode: {status}")
    
    plotter.enable_point_picking(
        callback=lambda point, picker: process_picked_point(),
        show_message=True,
        tolerance=0.01,
        use_picker=True,
        pickable_window=False,
        show_point=True,
        point_size=20,
        picker='point',
        font_size=20
    )
    
    # Add key events
    plotter.add_key_event('r', toggle_radius_mode)
    plotter.add_key_event('x', toggle_mode)
    plotter.add_key_event('n', cycle_resource)  # Add resource cycling
    plotter.add_key_event('q', lambda: plotter.close())
    
    setup_plotter_text(plotter, current_resource)
    
    try:
        plotter.show()
    except AttributeError:
        print(f"\nFinished with {current_resource}")

def edit_individual_treeVTK(poly):
    """Main function to edit individual trees using VTK polydata input."""
    
    poly = aa_tree_helper_functions.create_color_mapping(poly, 'cluster_id')
    
    # Create cluster_df from polydata's cluster_id array
    cluster_ids = poly.point_data['cluster_id']
    unique_clusters = np.unique(cluster_ids)
    cluster_df = pd.DataFrame({'cluster_id': unique_clusters})
    
    # Initialize resource columns with -1 (unassigned)
    resource_names = aa_tree_helper_functions.resource_names()
    for resource in resource_names:
        cluster_df[resource] = -1
    
    # Create plotter instance
    plotter = pv.Plotter()
    plotter.add_mesh(poly, rgb=True, scalars='colors', show_edges=False, point_size=10)
    
    # Set up picker and choose resources
    setup_picking_interface(plotter, poly, cluster_df, resource_names)

    # Create template DataFrame from poly with all existing resource columns
    template_data = {'cluster_id': poly.point_data['cluster_id']}
    for key in poly.point_data.keys():
        if key.startswith('resource'):
            template_data[key] = poly.point_data[key]
    template = pd.DataFrame(template_data)
    
    # Update resource columns
    choice = {
        'resource_hollow': 'replace',
        'resource_epiphyte': 'replace',
        'resource_dead branch': 'adjust',
        'resource_perch branch': 'adjust',
        'resource_peeling bark': 'adjust',
        'resource_fallen log': 'adjust',
        'resource_other': 'adjust'
    }

    # Apply edits to template
    for resource, action in choice.items():
        if resource not in template.columns:
            template[resource] = 0  # Initialize if doesn't exist
        if action == 'replace':
            template[resource] = 0
            continue
            
        print(f'{resource} edits:')
        print(cluster_df[resource].value_counts())
        valid_edits = cluster_df[cluster_df[resource] != -1]
        
        if not valid_edits.empty:
            resource_map = valid_edits.set_index('cluster_id')[resource]
            template.loc[template['cluster_id'].isin(resource_map.index), resource] = \
                template.loc[template['cluster_id'].isin(resource_map.index), 'cluster_id'].map(resource_map)

    # Create final resource column
    template = aa_tree_helper_functions.create_resource_column(template)
    
    # Update poly's point_data with template values
    for col in template.columns:
        if col.startswith('resource'):
            poly.point_data[col] = template[col].values
    
    return poly

def edit_individual_trees(templatesDF):
    """Main function to edit individual trees."""

    editsDF = pd.DataFrame(columns = templatesDF.columns.drop('template'))
    editsDF['edits'] = None

    for _, row in templatesDF.iterrows():
        # Create a copy of the template
        template_df = row['template'].copy()

        
        
        # Group by cluster_id and keep only cluster_id
        cluster_df = (template_df.groupby('cluster_id')
                     .first()
                     .reset_index()[['cluster_id']])
        
        # Initialize resource columns with -1 (unassigned)
        resource_names = aa_tree_helper_functions.resource_names()
        for resource in resource_names:
            cluster_df[resource] = -1  # -1 = unassigned (changed from 0)
        
        # Create poly object for visualization
        poly = aa_tree_helper_functions.convertToPoly(template_df)
        poly = aa_tree_helper_functions.create_color_mapping(poly, 'cluster_id')
        
        # Create plotter instance
        plotter = pv.Plotter()
        plotter.add_mesh(poly, rgb=True, scalars='colors', show_edges=False, point_size=10)
        
        # Set up picker and choose resources
        setup_picking_interface(plotter, poly, cluster_df, resource_names)

        print(f'edits are \n{cluster_df.head()}')
        for resource in resource_names:
            #create masks for excluded and assigned
            excluded_mask = cluster_df[resource] == 0
            assigned_mask = cluster_df[resource] == 1
            print(f'number of {resource} excluded: {excluded_mask.sum()}')
            print(f'number of {resource} assigned: {assigned_mask.sum()}')

        #create a copy of row and drop template
        row_copy = row.drop(columns = ['template'])
        #add row to editsDF with precolonial, size, control, tree_id from row and cluster_df to edits
        row_copy['edits'] = cluster_df
        editsDF = pd.concat([editsDF, pd.DataFrame([row_copy])], ignore_index=True)
        
    return editsDF


if __name__ == "__main__":
    print("Loading templates...")
    
    templateDir = Path('data/revised/trees')
    templateName = 'combined_templateDF.pkl'
    
    template_input_path = templateDir / templateName
    
    print(f"Loading templates from: {template_input_path}")
    templatesDF = pd.read_pickle(template_input_path)
  
    # Do single template
    precolonial = False
    size = 'snag'
    control = 'improved-tree'
    tree_id = 10

    mask = (templatesDF['precolonial'] == precolonial) & (templatesDF['size'] == size) & (templatesDF['control'] == control) & (templatesDF['tree_id'] == tree_id)
    templatesDF = templatesDF[mask]

    edits = edit_individual_trees(templatesDF)
    edits_output_path = templateDir / 'combined_editsDF.pkl'
    edits.to_pickle(edits_output_path)
    print(f"Saved edits to: {edits_output_path}")