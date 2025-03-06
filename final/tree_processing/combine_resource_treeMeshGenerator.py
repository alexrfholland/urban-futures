import pickle
import pandas as pd
import pyvista as pv
import numpy as np
import os
from pathlib import Path
from scipy.spatial import cKDTree
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
    #'epiphyte': {'voxelSize': isoSizeElm},
    #'hollow': {'voxelSize': isoSizeElm},
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
    #'epiphyte': {'voxelSize': isoSizeEuc},
    #'hollow': {'voxelSize': isoSizeEuc},
    'epiphyte': {'voxelSize': [0.3, 0.3, 0.3]},
    'hollow': {'voxelSize': [.3, .3, .3]},
    'leaf cluster': {'voxelSize': [.5, .5, .5]}
}


def transfer_point_data(mesh: pv.PolyData, source_df: pd.DataFrame, columns_to_transfer: list) -> pv.PolyData:
    """
    Transfer point data from source dataframe to mesh vertices using nearest neighbor.
    
    Args:
        mesh: PyVista PolyData mesh to transfer data to
        source_df: Source DataFrame containing point coordinates and data
        columns_to_transfer: List of column names to transfer
        
    Returns:
        PyVista PolyData mesh with transferred point data
    """
    
    # Create KD-tree from source points
    source_points = source_df[['x', 'y', 'z']].values
    tree = cKDTree(source_points)
    
    # Find nearest neighbors for each vertex in the mesh
    distances, indices = tree.query(mesh.points)
    
    # Transfer available columns
    for col in columns_to_transfer:
        if col in source_df.columns:
            print(f"Transferring {col} to mesh point_data")
            mesh.point_data[col] = source_df[col].values[indices]
        else:
            print(f"Column {col} not found in source dataframe")
    
    return mesh


def clean_meshABSOLUTE(surface: pv.PolyData, min_cells: int = 10) -> pv.PolyData:
    """
    Clean a mesh by removing small, isolated clusters.
    
    Args:
        surface: PyVista PolyData mesh to clean
        min_cells: Minimum number of cells for a cluster to be kept (default: 10)
        
    Returns:
        Cleaned PyVista PolyData mesh
    """
    print(f"\nInput mesh has {surface.n_points} points and {surface.n_cells} cells")
    
    # Apply connectivity filter to label each connected region
    conn = surface.connectivity(extraction_mode='all')
    
    # Access the 'RegionId' assigned to each cell
    region_ids = conn.cell_data['RegionId']
    
    # Count the number of cells in each region
    unique_region_ids, counts = np.unique(region_ids, return_counts=True)
    largest_size = counts.max()
    
    print(f"Found {len(unique_region_ids)} clusters")
    print(f"Largest cluster has {largest_size} cells")
    print(f"Keeping clusters with more than {min_cells} cells")
    
    # Identify regions that meet the minimum cell requirement
    keep_ids = unique_region_ids[counts >= min_cells]
    discard_ids = unique_region_ids[counts < min_cells]
    
    print(f"Keeping {len(keep_ids)} clusters")
    print(f"Discarding {len(discard_ids)} small clusters")
    
    if len(keep_ids) > 0:
        # Create a mask to select cells from the regions to keep
        mask = np.isin(region_ids, keep_ids)
        # Extract the desired cells to form the cleaned mesh
        clean_surface = conn.extract_cells(mask)
        return clean_surface
    else:
        print("Warning: No clusters meet the minimum cell requirement")
        clean_surface = surface.connectivity(extraction_mode='largest')
        print(f"Returning largest cluster with {clean_surface.n_cells} cells")
        return clean_surface


def clean_meshRELATIVE(surface: pv.PolyData, min_size_percentage: float = 0.05) -> pv.PolyData:
    """
    Clean a mesh by removing small, isolated clusters.
    
    Args:
        surface: PyVista PolyData mesh to clean
        min_size_percentage: Minimum size as a percentage of largest cluster (default: 0.05)
        
    Returns:
        Cleaned PyVista PolyData mesh
    """
    print(f"\nInput mesh has {surface.n_points} points and {surface.n_cells} cells")
    
    # Apply connectivity filter to label each connected region
    conn = surface.connectivity(extraction_mode='all')
    
    # Access the 'RegionId' assigned to each cell
    region_ids = conn.cell_data['RegionId']
    
    # Count the number of cells in each region
    unique_region_ids, counts = np.unique(region_ids, return_counts=True)
    largest_size = counts.max()
    min_cells = int(largest_size * min_size_percentage)
    
    print(f"Found {len(unique_region_ids)} clusters")
    print(f"Largest cluster has {largest_size} cells")
    print(f"Keeping clusters with more than {min_cells} cells ({min_size_percentage*100}% of largest)")
    
    # Identify regions that meet the minimum cell requirement
    keep_ids = unique_region_ids[counts >= min_cells]
    discard_ids = unique_region_ids[counts < min_cells]
    
    print(f"Keeping {len(keep_ids)} clusters")
    print(f"Discarding {len(discard_ids)} small clusters")
    
    if len(keep_ids) > 0:
        # Create a mask to select cells from the regions to keep
        mask = np.isin(region_ids, keep_ids)
        # Extract the desired cells to form the cleaned mesh
        clean_surface = conn.extract_cells(mask)
        return clean_surface
    else:
        print("Warning: No clusters meet the minimum size requirement")
        clean_surface = surface.connectivity(extraction_mode='largest')
        print(f"Returning largest cluster with {clean_surface.n_cells} cells")
        return clean_surface


def extract_isosurface_from_polydata(polydata: pv.PolyData, spacing: tuple[float, float, float], resource_name, isovalue: float = 1.0) -> pv.PolyData:
    """Extract isosurface from point cloud data"""
    print(f'{resource_name} polydata has {polydata.n_points} points')
    
    if polydata is not None and polydata.n_points > 0:
        points = polydata.points
        x, y, z = points.T
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        dims = (
            int((x_max - x_min) / spacing[0]) + 1,
            int((y_max - y_min) / spacing[1]) + 1,
            int((z_max - z_min) / spacing[2]) + 1
        )

        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=(x_min, y_min, z_min))
        scalars = np.zeros(grid.n_points)

        # Create a more continuous scalar field
        for px, py, pz in points:
            ix = int((px - x_min) / spacing[0])
            iy = int((py - y_min) / spacing[1])
            iz = int((pz - z_min) / spacing[2])
            if 0 <= ix < dims[0] and 0 <= iy < dims[1] and 0 <= iz < dims[2]:
                grid_idx = ix + iy * dims[0] + iz * dims[0] * dims[1]
                if grid_idx < len(scalars):
                    scalars[grid_idx] = 2
                    # Add some influence to neighboring cells
                    for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                        nx, ny, nz = ix+dx, iy+dy, iz+dz
                        if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
                            neighbor_idx = nx + ny * dims[0] + nz * dims[0] * dims[1]
                            if neighbor_idx < len(scalars):
                                scalars[neighbor_idx] = max(scalars[neighbor_idx], 1)

        grid.point_data['values'] = scalars
        isosurface = grid.contour(isosurfaces=[isovalue], scalars='values', method='flying_edges', compute_normals=True)
        surface = isosurface.extract_surface()
        return surface
    else:
        return None

def fill_holes_in_mesh(surface: pv.PolyData, hole_size: float = 100.0) -> pv.PolyData:
    """
    Fill holes in a mesh using PyVista's fill_holes method.
    
    Args:
        surface: PyVista PolyData mesh to fill
        hole_size: Maximum size of holes to fill (default: 100.0)
        
    Returns:
        PyVista PolyData mesh with holes filled
    """
    # Clean the mesh and ensure it stays as PolyData
    cleaned = surface.clean().extract_surface()
    
    # Fill the holes
    try:
        filled = cleaned.fill_holes(hole_size=hole_size)
        return filled
    except Exception as e:
        print(f"Warning: Could not fill holes in mesh: {str(e)}")
        return cleaned

def process_template(row, output_path, overide=False):
    """Process a single tree template and save as VTK"""
    template_df = row['template']

    
    if template_df is None or len(template_df) == 0:
        print("Empty template, skipping...")
        return
    
    # Select resource specs based on template source
    resource_specs = resource_specs_euc if row['precolonial'] or row['size'] == 'snag' or overide == False else resource_specs_elm
    print(f"Using {'elm' if row['precolonial'] else 'eucalyptus'} resource specifications")
    
    combined_polydata = None
    resources = template_df['resource'].unique()
    print(f"Found resources: {resources}")
    
    for resource in resources:
        if resource not in resource_specs:
            print(f"Warning: Resource '{resource}' not in specifications, skipping...")
            continue
            
        resource_points = template_df[template_df['resource'] == resource][['x', 'y', 'z']].values
        if len(resource_points) == 0:
            continue
            
        resource_polydata = pv.PolyData(resource_points)
        specs = resource_specs[resource]
        isosurface = extract_isosurface_from_polydata(
            resource_polydata, 
            specs['voxelSize'],
            resource
        )

        if isosurface is not None:
            if resource in ['fallen log', 'hollow', 'epiphyte']:
                print(f"Filled holes in {resource}")
                isosurface = fill_holes_in_mesh(isosurface)


            isosurface.point_data['resource'] = np.full(isosurface.n_points, resource)
            """for attr, value in tree_info.items():
                isosurface.point_data[attr] = np.full(isosurface.n_points, value)"""
            
            if combined_polydata is None:
                combined_polydata = isosurface
            else:
                combined_polydata = combined_polydata.merge(isosurface)
    
    if combined_polydata is not None:
        # Clean the combined mesh
        print("\nCleaning combined mesh...")
        combined_polydata = clean_meshABSOLUTE(combined_polydata, min_cells=20)
        
        # Fill holes in the mesh
        #print("Filling holes in mesh...")
        #combined_polydata = fill_holes_in_mesh(combined_polydata)

        # Transfer point data from template_df
        
        resourceCols = aa_tree_helper_functions.resource_names()
        
        columns_to_transfer = [
            'isSenescent',
            'isTerminal',
            'cluster_id',
            'community-leiden',
            'community_ancestors_threshold0',
            'community_ancestors_threshold1',
            'community_ancestors_threshold2',
        ]
        columns_to_transfer.extend(resourceCols)

        combined_polydata = transfer_point_data(combined_polydata, template_df, columns_to_transfer)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_polydata.save(output_path)
        print(f"Saved mesh to {output_path}")
        return combined_polydata
    else:
        print("No valid polydata generated for this tree")

def process_template_row(row, output_dir):
    print(f"\nProcessing tree: precolonial={row['precolonial']}, size={row['size']}, control={row['control']}, tree_id={row['tree_id']}")
    output_path = Path(output_dir) / f"precolonial.{row['precolonial']}_size.{row['size']}_control.{row['control']}_id.{row['tree_id']}.vtk"
    mesh = process_template(row, output_path)
    """plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='resource_hollow')
    plotter.show()"""
    return mesh
def process_all_templates(templatesDF, output_dir):
    for _, row in templatesDF.iterrows():
        """Process all templates"""
        process_template_row(row, output_dir)


if __name__ == "__main__":
    print("Loading templates...")

    # Choose whether to regenerate all meshes, just edited ones, or apply mask
    print("\n1. Regenerate all tree meshes")
    print("2. Regenerate only edited tree meshes") 
    print("3. Regenerate only masked meshes")
    choice = input("Enter choice (1, 2 or 3): ")

    if choice == "1":
        templateName = 'edited_combined_templateDF.pkl'
    elif choice == "2":
        templateName = 'just_edits_templateDF.pkl'
    elif choice == "3":
        templateName = 'edited_combined_templateDF.pkl'
    else:
        raise ValueError("Invalid choice. Please enter 1, 2 or 3.")
    
    templateDir = Path('data/revised/trees')    
    template_input_path = templateDir / templateName
    
    print(f"Loading templates from: {template_input_path}")
    templatesDF = pd.read_pickle(template_input_path)

    if choice == "3":
        mask = (templatesDF['size'] == 'snag')
        #mask = (templatesDF['size'] == 'snag') | (templatesDF['precolonial'] == True)
        templatesDF = templatesDF[mask]

    print(templatesDF.head())

    output_dir = 'data/revised/final/treeMeshes'
    process_all_templates(templatesDF,output_dir)
        
    print("Processing completed.")