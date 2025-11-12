import pickle
import pandas as pd
import pyvista as pv
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import sys
sys.path.append('../')
import aa_tree_helper_functions

# Resource specifications for isosurface and point data properties
isoSizeElm = [0.05, 0.05, 0.05]
resource_specs_elm = {
    'perch branch': {'voxelSize': isoSizeElm},
    'peeling bark': {'voxelSize': isoSizeElm},
    'dead branch': {'voxelSize': isoSizeElm},
    'other': {'voxelSize': isoSizeElm},
    'fallen log': {'voxelSize': [0.15, 0.15, 0.15]},
    'leaf litter': {'voxelSize': [0.25, 0.25, 0.25]},
    'epiphyte': {'voxelSize': [0.15, 0.15, 0.15]},
    'hollow': {'voxelSize': [0.15, 0.15, 0.15]},
    'leaf cluster': {'voxelSize': [0.5, 0.5, 0.5]}
}

isoSizeEuc = [0.15, 0.15, 0.15]
resource_specs_euc = {
    'perch branch': {'voxelSize': isoSizeEuc},
    'peeling bark': {'voxelSize': isoSizeEuc},
    'dead branch': {'voxelSize': isoSizeEuc},
    'other': {'voxelSize': isoSizeEuc},
    'fallen log': {'voxelSize': [0.15, 0.15, 0.15]},
    'leaf litter': {'voxelSize': [0.25, 0.25, 0.25]},
    'epiphyte': {'voxelSize': [0.3, 0.3, 0.3]},
    'hollow': {'voxelSize': [.3, .3, .3]},
    'leaf cluster': {'voxelSize': [.5, .5, .5]}
}

def load_point_cloud_vtk(filepath):
    """Load point cloud VTK file."""
    polydata = pv.read(filepath)
    print(f"Loaded point cloud with {polydata.n_points} points")
    return polydata

def extract_isosurface_with_connectivity(points, spacing, resource_name, isovalue=1.0):
    """
    Extract isosurface from points ensuring better connectivity within the mesh.
    """
    print(f'{resource_name}: Processing {len(points)} points')
    
    if len(points) == 0:
        return None
    
    x, y, z = points.T
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    
    # Add padding to avoid edge artifacts
    padding = max(spacing) * 2
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    z_min -= padding
    z_max += padding
    
    dims = (
        int((x_max - x_min) / spacing[0]) + 1,
        int((y_max - y_min) / spacing[1]) + 1,
        int((z_max - z_min) / spacing[2]) + 1
    )
    
    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=(x_min, y_min, z_min))
    scalars = np.zeros(grid.n_points)
    
    # Create a more continuous scalar field with better connectivity
    for px, py, pz in points:
        ix = int((px - x_min) / spacing[0])
        iy = int((py - y_min) / spacing[1])
        iz = int((pz - z_min) / spacing[2])
        
        if 0 <= ix < dims[0] and 0 <= iy < dims[1] and 0 <= iz < dims[2]:
            # Set primary voxel
            grid_idx = ix + iy * dims[0] + iz * dims[0] * dims[1]
            if grid_idx < len(scalars):
                scalars[grid_idx] = 2
                
                # Add influence to all 26 neighboring cells for better connectivity
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nx, ny, nz = ix+dx, iy+dy, iz+dz
                            if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
                                neighbor_idx = nx + ny * dims[0] + nz * dims[0] * dims[1]
                                if neighbor_idx < len(scalars):
                                    scalars[neighbor_idx] = max(scalars[neighbor_idx], 1)
    
    grid.point_data['values'] = scalars
    
    # Apply Gaussian smoothing to create more continuous field
    grid = grid.gaussian_smooth(radius_factor=1.5, std_dev=0.5)
    
    # Extract isosurface using marching cubes
    isosurface = grid.contour(isosurfaces=[isovalue], scalars='values', 
                             method='flying_edges', compute_normals=True,
                             generate_triangles=False)  # Keep quads
    
    surface = isosurface.extract_surface()
    
    # Clean up the mesh
    if surface.n_cells > 0:
        surface = surface.clean(tolerance=1e-6)
        surface = surface.compute_normals(cell_normals=True, point_normals=True, 
                                         split_vertices=False, flip_normals=False)
    
    return surface

def merge_close_components(mesh, merge_distance):
    """
    Merge mesh components that are within merge_distance of each other.
    This helps connect parts that should be continuous.
    """
    if mesh.n_cells == 0:
        return mesh
    
    # Extract all connected components
    connectivity = mesh.connectivity(extraction_mode='all')
    region_ids = connectivity['RegionId']
    unique_regions = np.unique(region_ids)
    
    if len(unique_regions) <= 1:
        return mesh
    
    print(f"Found {len(unique_regions)} disconnected components")
    
    # For each component, find its bounding box center
    component_centers = []
    component_meshes = []
    
    for region_id in unique_regions:
        mask = region_ids == region_id
        component = connectivity.extract_cells(mask)
        bounds = component.bounds
        center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
        component_centers.append(center)
        component_meshes.append(component)
    
    # Build KDTree to find nearby components
    tree = cKDTree(component_centers)
    
    # Find components that should be merged
    merge_groups = []
    merged = set()
    
    for i, center in enumerate(component_centers):
        if i in merged:
            continue
        
        # Find all components within merge_distance
        indices = tree.query_ball_point(center, r=merge_distance)
        if len(indices) > 1:
            merge_groups.append(indices)
            merged.update(indices)
    
    # Add ungrouped components as single-element groups
    for i in range(len(unique_regions)):
        if i not in merged:
            merge_groups.append([i])
    
    print(f"Merging into {len(merge_groups)} groups")
    
    # Merge components in each group
    final_meshes = []
    for group in merge_groups:
        if len(group) == 1:
            final_meshes.append(component_meshes[group[0]])
        else:
            # Merge multiple components
            merged_mesh = component_meshes[group[0]]
            for idx in group[1:]:
                merged_mesh = merged_mesh.merge(component_meshes[idx])
            final_meshes.append(merged_mesh)
    
    # Combine all groups
    if len(final_meshes) == 1:
        return final_meshes[0]
    else:
        result = final_meshes[0]
        for mesh in final_meshes[1:]:
            result = result.merge(mesh)
        return result

def process_tree_from_points(point_cloud_path, output_path, tree_type='elm'):
    """Process a tree from point cloud VTK file and save as mesh VTK."""
    
    # Load point cloud
    polydata = load_point_cloud_vtk(point_cloud_path)
    
    # Extract points as numpy array
    points = polydata.points
    
    # Check if we have resource data
    if 'resource' not in polydata.point_data:
        print("Warning: No resource data found in point cloud")
        return None
    
    # Select resource specs based on tree type
    resource_specs = resource_specs_elm if tree_type == 'elm' else resource_specs_euc
    print(f"Using {tree_type} resource specifications")
    
    # Get point data as DataFrame for easier processing
    point_df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'resource': polydata.point_data['resource']
    })
    
    # Add any other point data attributes
    for key in polydata.point_data.keys():
        if key != 'resource':
            point_df[key] = polydata.point_data[key]
    
    # Process each resource separately
    combined_polydata = None
    resources = point_df['resource'].unique()
    print(f"Found resources: {resources}")
    
    for resource in resources:
        if resource not in resource_specs:
            print(f"Warning: Resource '{resource}' not in specifications, skipping...")
            continue
        
        # Extract points for this resource
        resource_df = point_df[point_df['resource'] == resource]
        resource_points = resource_df[['x', 'y', 'z']].values
        
        if len(resource_points) == 0:
            continue
        
        # Get voxel size for this resource
        specs = resource_specs[resource]
        voxel_size = specs['voxelSize']
        
        # Create isosurface with better connectivity
        isosurface = extract_isosurface_with_connectivity(
            resource_points, 
            voxel_size,
            resource
        )
        
        if isosurface is not None and isosurface.n_cells > 0:
            # Merge close components within this resource
            # Use 2x voxel size as merge distance
            merge_distance = max(voxel_size) * 2
            isosurface = merge_close_components(isosurface, merge_distance)
            
            # Add resource identifier
            isosurface.point_data['resource'] = np.full(isosurface.n_points, resource)
            
            # Transfer other attributes from original points to mesh vertices
            if isosurface.n_points > 0:
                # Create KDTree for nearest neighbor lookup
                tree = cKDTree(resource_points)
                distances, indices = tree.query(isosurface.points)
                
                # Transfer attributes
                for col in resource_df.columns:
                    if col not in ['x', 'y', 'z', 'resource']:
                        isosurface.point_data[col] = resource_df[col].values[indices]
            
            # Combine with other resources
            if combined_polydata is None:
                combined_polydata = isosurface
            else:
                combined_polydata = combined_polydata.merge(isosurface)
            
            print(f"  {resource}: {isosurface.n_cells} cells")
    
    if combined_polydata is not None:
        # Final cleaning
        combined_polydata = combined_polydata.clean(tolerance=1e-6)
        
        # Save mesh
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_polydata.save(output_path)
        print(f"Saved mesh to {output_path}")
        print(f"Total: {combined_polydata.n_points} points, {combined_polydata.n_cells} cells")
        
        return combined_polydata
    else:
        print("No valid mesh generated")
        return None

def process_all_point_clouds(input_dir='data/revised/final/tree_VTKpts/', 
                           output_dir='data/revised/final/treeMeshes_conversation/'):
    """Process all point cloud VTK files in the input directory."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all VTK files
    vtk_files = list(input_path.glob('*.vtk'))
    print(f"Found {len(vtk_files)} VTK files to process")
    
    for vtk_file in vtk_files:
        print(f"\nProcessing {vtk_file.name}")
        
        # Parse filename to determine tree type
        parts = vtk_file.stem.split('_')
        precolonial = 'True' in parts[0]
        size = parts[1].split('.')[-1]
        
        # Determine tree type
        if precolonial or size == 'snag':
            tree_type = 'euc'
        else:
            tree_type = 'elm'
        
        # Process the tree
        output_file = output_path / vtk_file.name
        process_tree_from_points(vtk_file, output_file, tree_type)

if __name__ == "__main__":
    print("Tree Mesh Generator - Conversation Version")
    print("-" * 50)
    
    # Process a single file as example
    example_file = Path('data/revised/final/tree_VTKpts/precolonial.False_size.medium_control.park-tree_id.2.vtk')
    
    if example_file.exists():
        print(f"\nProcessing example file: {example_file}")
        output_file = Path('data/revised/final/treeMeshes_conversation') / example_file.name
        process_tree_from_points(example_file, output_file, tree_type='elm')
    else:
        print("\nExample file not found, processing all files...")
        process_all_point_clouds() 