import pyvista as pv
import numpy as np
from pathlib import Path

# Resource color specifications matching treeBake_glyphs.py
RESOURCE_SPECS = {
    'perch branch': {'colour': 'green', 'opacity': 1.0},
    'peeling bark': {'colour': 'orange', 'opacity': 1.0},
    'dead branch': {'colour': 'purple', 'opacity': 1.0},
    'other': {'colour': '#C5C5C5', 'opacity': 1.0},
    'fallen log': {'colour': 'plum', 'opacity': 1.0},
    'leaf litter': {'colour': 'peachpuff', 'opacity': 1.0},
    'epiphyte': {'colour': 'cyan', 'opacity': 1.0},
    'hollow': {'colour': 'magenta', 'opacity': 1.0},
    'leaf cluster': {'colour': '#d5ffd1', 'opacity': 0.1}
}

# Standardized silhouette parameters
SILHOUETTE_CONFIG = {
    'line_width': 10,
    'feature_angle': 90
}

def get_silhouette_params(resource_name):
    """Get silhouette parameters for a given resource type."""
    # Determine silhouette color
    if resource_name in ['hollow', 'epiphyte']:
        color = '#525252'  # Dark gray for these special resources
    else:
        color = RESOURCE_SPECS[resource_name]['colour']
    
    return {
        'color': color,
        'line_width': SILHOUETTE_CONFIG['line_width'],
        'feature_angle': SILHOUETTE_CONFIG['feature_angle']
    }

def load_tree_mesh(precolonial, size, control, tree_id, mesh_dir='data/revised/final/treeMeshes'):
    """Load a tree mesh from VTK file."""
    filename = f"precolonial.{precolonial}_size.{size}_control.{control}_id.{tree_id}.vtk"
    filepath = Path(mesh_dir) / filename
    
    mesh = pv.read(filepath)
    print(f"Loaded mesh with {mesh.n_points} points, {mesh.n_cells} cells")
    return mesh

def add_tree_to_plotter(plotter, mesh, shift=(0, 0, 0), use_silhouette=True, smooth_mesh=False):
    """Add tree mesh to plotter with resource-specific rendering."""
    # Apply shift if provided
    if shift != (0, 0, 0):
        mesh.points += np.array(shift)
    
    # Get unique resources in the mesh
    resources = np.unique(mesh.point_data['resource'])
    print(f"Found resources: {resources}")
    
    for resource in resources:
        # Extract points for this resource
        mask = mesh.point_data['resource'] == resource
        resource_mesh = mesh.extract_points(mask)
        
        if resource_mesh.n_points == 0:
            continue
        
        # Optional mesh smoothing to reduce edge artifacts
        if smooth_mesh and resource not in ['hollow', 'epiphyte']:
            # Convert to PolyData for smoothing
            resource_mesh = resource_mesh.extract_surface()
            resource_mesh = resource_mesh.smooth(n_iter=100, relaxation_factor=0.01, 
                                                feature_smoothing=False, 
                                                boundary_smoothing=True)
        
        # Get rendering parameters
        specs = RESOURCE_SPECS[resource]
        
        # Special handling for leaf litter (random z-shift)
        if resource == 'leaf litter':
            z_shift = np.random.uniform(0, 0.3, size=resource_mesh.points.shape[0])
            resource_mesh.points[:, 2] += z_shift
        
        # Prepare rendering parameters
        render_params = {
            'opacity': specs['opacity'],
            'show_edges': False
        }
        
        # Handle resource-specific coloring
        if resource in ['other', 'leaf litter']:
            # These resources use gray/neutral mesh color in original
            render_params['color'] = '#525252' if resource == 'other' else specs['colour']
        else:
            render_params['color'] = specs['colour']
        
        # Add silhouette if enabled
        if use_silhouette:
            render_params['silhouette'] = get_silhouette_params(resource)
        
        # Add mesh with appropriate styling
        plotter.add_mesh(resource_mesh, **render_params)

def clean_and_merge_mesh(mesh, min_component_size=100):
    """Clean and merge disconnected mesh components while being conservative about removal."""
    print(f"Original mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    
    # Extract surface if needed
    if hasattr(mesh, 'extract_surface'):
        mesh = mesh.extract_surface()
    
    # Clean the mesh to remove duplicate points and degenerate cells
    mesh = mesh.clean(tolerance=1e-6)
    print(f"After cleaning: {mesh.n_points} points, {mesh.n_cells} cells")
    
    # Compute normals to ensure consistent orientation
    mesh = mesh.compute_normals(cell_normals=True, point_normals=True, 
                                split_vertices=False, flip_normals=False)
    
    # Extract all connected components
    connectivity = mesh.connectivity(extraction_mode='all')
    region_ids = connectivity['RegionId']
    unique_regions, counts = np.unique(region_ids, return_counts=True)
    
    print(f"Found {len(unique_regions)} connected components")
    print(f"Component sizes: {sorted(counts, reverse=True)[:10]}...")  # Show top 10
    
    # Keep all components above threshold instead of just the largest
    large_components_mask = np.zeros(connectivity.n_cells, dtype=bool)
    for region_id, count in zip(unique_regions, counts):
        if count >= min_component_size:
            large_components_mask |= (region_ids == region_id)
    
    # Extract all significant components
    significant_mesh = connectivity.extract_cells(large_components_mask)
    removed_components = len(unique_regions) - len(np.unique(significant_mesh['RegionId']))
    print(f"Removed {removed_components} small components (< {min_component_size} cells)")
    print(f"Kept mesh: {significant_mesh.n_points} points, {significant_mesh.n_cells} cells")
    
    # Convert to PolyData for decimation
    if hasattr(significant_mesh, 'extract_surface'):
        poly_mesh = significant_mesh.extract_surface()
    else:
        poly_mesh = significant_mesh
    
    # Very gentle decimation - only 10% reduction to clean up while preserving detail
    decimated = poly_mesh.decimate(target_reduction=0.1, volume_preservation=True)
    print(f"After gentle decimation: {decimated.n_points} points, {decimated.n_cells} cells")
    
    # Gentle smoothing to blend faces
    final_mesh = decimated.smooth(n_iter=30, relaxation_factor=0.01,
                                  feature_smoothing=False,
                                  boundary_smoothing=True,
                                  edge_angle=SILHOUETTE_CONFIG['feature_angle'])
    print(f"Final mesh: {final_mesh.n_points} points, {final_mesh.n_cells} cells")
    
    return final_mesh

def add_whole_tree_to_plotter(plotter, mesh, shift=(0, 0, 0), use_silhouette=True, clean_mesh=True):
    """Add entire tree mesh without splitting by resource - for testing silhouette issues."""
    # Apply shift if provided
    if shift != (0, 0, 0):
        mesh.points += np.array(shift)
    
    # Clean and merge mesh if requested
    if clean_mesh:
        print("Cleaning and merging mesh components...")
        mesh = clean_and_merge_mesh(mesh)
    
    # Prepare rendering parameters
    render_params = {
        'opacity': 1.0,
        'show_edges': False,
        'color': 'lightgray',  # Neutral color for whole mesh
        'scalars': 'resource',  # Color by resource attribute
        'categories': True  # Treat scalars as categories
    }
    
    # Add silhouette if enabled
    if use_silhouette:
        render_params['silhouette'] = {
            'color': 'black',
            'line_width': SILHOUETTE_CONFIG['line_width'],
            'feature_angle': SILHOUETTE_CONFIG['feature_angle']
        }
    
    # Add the entire mesh as one object
    plotter.add_mesh(mesh, **render_params)
    print(f"Added whole mesh as single object")

def setup_lighting(plotter):
    """Setup lighting for better visibility."""
    # Add a light positioned above and to the side
    plotter.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=0.7))
    
    # Add camera light for additional illumination
    light2 = pv.Light(light_type='cameralight', intensity=0.5)
    light2.specular = 0.5
    plotter.add_light(light2)
    
    # Enable eye dome lighting for better depth perception
    plotter.enable_eye_dome_lighting()

def setup_camera(plotter, elevation=50, distance=600):
    """Set up camera position and view."""
    plotter.camera.elevation = elevation
    plotter.camera.azimuth = 0
    plotter.camera.zoom(distance / 100)
    plotter.camera.focal_point = (0, 0, 0)

def view_single_tree(precolonial, size, control, tree_id, mesh_dir='data/revised/final/treeMeshes', 
                     use_silhouette=True, smooth_mesh=False):
    """View a single tree with specified parameters."""
    plotter = pv.Plotter()
    setup_lighting(plotter)
    
    mesh = load_tree_mesh(precolonial, size, control, tree_id, mesh_dir)
    add_tree_to_plotter(plotter, mesh, use_silhouette=use_silhouette, smooth_mesh=smooth_mesh)
    
    setup_camera(plotter)
    plotter.add_title(f"{size} {control} (ID: {tree_id}, Precolonial: {precolonial})")
    plotter.show()

def view_whole_tree(precolonial, size, control, tree_id, mesh_dir='data/revised/final/treeMeshes', 
                    use_silhouette=True, clean_mesh=True):
    """View entire tree mesh without splitting - for testing silhouette rendering."""
    plotter = pv.Plotter()
    setup_lighting(plotter)
    
    mesh = load_tree_mesh(precolonial, size, control, tree_id, mesh_dir)
    add_whole_tree_to_plotter(plotter, mesh, use_silhouette=use_silhouette, clean_mesh=clean_mesh)
    
    setup_camera(plotter)
    title = f"WHOLE MESH: {size} {control} (ID: {tree_id})"
    if clean_mesh:
        title += " - CLEANED"
    plotter.add_title(title)
    plotter.show()

def view_tree_comparison(tree_id, size='large', precolonial=False, mesh_dir='data/revised/final/treeMeshes',
                        use_silhouette=True, smooth_mesh=False):
    """View comparison of tree across different controls (reserve, park, street)."""
    controls = ['reserve-tree', 'park-tree', 'street-tree']
    plotter = pv.Plotter(shape=(1, 3))
    setup_lighting(plotter)
    
    for i, control in enumerate(controls):
        plotter.subplot(0, i)
        
        mesh = load_tree_mesh(precolonial, size, control, tree_id, mesh_dir)
        add_tree_to_plotter(plotter, mesh, use_silhouette=use_silhouette, smooth_mesh=smooth_mesh)
        
        setup_camera(plotter)
        plotter.add_title(f"{control}")
    
    plotter.link_views()
    plotter.show()

def view_multiple_trees(tree_specs, layout=(2, 2), mesh_dir='data/revised/final/treeMeshes',
                       use_silhouette=True, smooth_mesh=False):
    """
    View multiple trees in a grid layout.
    
    Args:
        tree_specs: List of tuples (precolonial, size, control, tree_id)
        layout: Tuple of (rows, cols) for subplot layout
        use_silhouette: Whether to render with silhouette edges
        smooth_mesh: Whether to apply mesh smoothing
    """
    rows, cols = layout
    plotter = pv.Plotter(shape=layout)
    setup_lighting(plotter)
    
    for idx, (precolonial, size, control, tree_id) in enumerate(tree_specs):
        if idx >= rows * cols:
            break
            
        row = idx // cols
        col = idx % cols
        plotter.subplot(row, col)
        
        mesh = load_tree_mesh(precolonial, size, control, tree_id, mesh_dir)
        add_tree_to_plotter(plotter, mesh, use_silhouette=use_silhouette, smooth_mesh=smooth_mesh)
        
        setup_camera(plotter)
        plotter.add_title(f"{size} {control} (ID: {tree_id})")
    
    plotter.link_views()
    plotter.show()

if __name__ == "__main__":
    # Example usage
    print("Tree Mesh Viewer")
    print("-" * 50)
    
    # Test whole mesh with cleaning
    print("\n1. Testing WHOLE MESH with cleaning and merging")
    view_whole_tree(precolonial=True, size='large', control='reserve-tree', tree_id=13, 
                    use_silhouette=True, clean_mesh=True)
    
    # Test whole mesh without cleaning for comparison
    print("\n2. Testing WHOLE MESH without cleaning")
    view_whole_tree(precolonial=True, size='large', control='reserve-tree', tree_id=13, 
                    use_silhouette=True, clean_mesh=False)
    
    # Then show split version for comparison
    print("\n3. Split by resource (original method)")
    view_single_tree(precolonial=True, size='large', control='reserve-tree', tree_id=13, 
                     use_silhouette=True, smooth_mesh=False)
    
    """# Additional examples
    # Try without silhouettes
    print("\n2. Same tree without silhouettes")
    view_single_tree(precolonial=True, size='large', control='reserve-tree', tree_id=13, 
                     use_silhouette=False)
    
    # Try with mesh smoothing
    print("\n3. Same tree with mesh smoothing")
    view_single_tree(precolonial=True, size='large', control='reserve-tree', tree_id=13, 
                     use_silhouette=True, smooth_mesh=True)
    
    # View comparison across controls
    print("\n4. Comparing tree across different controls")
    view_tree_comparison(tree_id=13, size='large', precolonial=False)
    
    # View multiple trees
    print("\n5. Viewing multiple trees in grid")
    tree_specs = [
        (False, 'large', 'park-tree', 13),
        (False, 'medium', 'street-tree', 2),
        (False, 'small', 'reserve-tree', 4),
        (True, 'large', 'park-tree', 12)
    ]
    view_multiple_trees(tree_specs, layout=(2, 2))"""