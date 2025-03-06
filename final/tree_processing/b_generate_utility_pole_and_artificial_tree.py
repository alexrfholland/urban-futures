import trimesh
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import os
import combine_edit_individual_trees
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Update import to use _blender directory
from _blender import a_vtk_to_ply

def get_snag(filename):
    """Load and return the snag mesh."""
    mesh_folder_directory = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/final/treeMeshes'
    
    # Load the mesh with pyvista
    filename = f'{filename}.vtk'
    mesh = pv.read(os.path.join(mesh_folder_directory, filename))
    
    # Print original position
    print(f"Snag before translation: Center = {mesh.center}")
    
    # Translate mesh to specified coordinates
    translation = np.array([1.11371, -0.421528, -8.28461])
    mesh.points += translation  # Use points property instead of translate method
    
    # Print new position
    print(f"Snag after translation: Center = {mesh.center}")
    
    return mesh

def generate_utility_pole(voxel_size=0.25):
    """
    Generate a voxelized utility pole point cloud.
    
    Args:
        voxel_size (float): Size of each voxel in units (default: 0.25)
    
    Returns:
        pv.PolyData: Voxelized point cloud
    """
    # Load the utility pole point cloud using absolute path
    path = '/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/utlity poles/utility_pole.ply'
    try:
        point_cloud = trimesh.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file at {path}. Please check the path is correct relative to the script location.")
    
    points = point_cloud.vertices
    colors = point_cloud.visual.vertex_colors
    
    # Find the point with minimum z coordinate
    min_z_index = np.argmin(points[:, 2])
    base_point = points[min_z_index]  # This is the point we'll move to origin
    print(f"Base point found at: x={base_point[0]:.2f}, y={base_point[1]:.2f}, z={base_point[2]:.2f}")
    
    # Translate entire object so this point becomes (0,0,0)
    points = points - base_point
    print(f"Translated model by: x={base_point[0]:.2f}, y={base_point[1]:.2f}, z={base_point[2]:.2f}")
    
    # Create voxel grid using specified voxel size
    voxel_coords = (points / voxel_size).astype(int)
    unique_voxels, _ = np.unique(voxel_coords, axis=0, return_inverse=True)
    voxel_centers = (unique_voxels + 0.5) * voxel_size
    
    # Use KDTree to find nearest neighbors
    tree = cKDTree(points)
    _, indices = tree.query(voxel_centers)
    voxel_colors = colors[indices]
    
    # Create PyVista PolyData with voxelized points
    cloud = pv.PolyData(voxel_centers)
    cloud.point_data['colors'] = voxel_colors
    
    # Rotate -24 degrees about z axis
    angle = -24  # degrees
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
        [0, 0, 1]
    ])
    
    # Apply rotation to points
    rotated_points = voxel_centers @ rotation_matrix.T
    cloud.points = rotated_points
    
    return cloud

def clip_snag_to_pole_bounds(pole, snag, xy_extension=3.0, z_extension=5.0):
    """
    Clip the snag mesh to an extended bounding box of the utility pole.
    
    Args:
        pole (pv.PolyData): The utility pole mesh
        snag (pv.PolyData): The snag mesh to be clipped
        xy_extension (float): How much to extend the bounds in X and Y directions
        z_extension (float): How much to extend the bounds in Z direction (upward)
    
    Returns:
        pv.PolyData: Clipped snag mesh
    """
    # Get pole bounds
    bounds = pole.bounds  # returns (xmin, xmax, ymin, ymax, zmin, zmax)
    print(f"Original pole bounds: {bounds}")
    
    # Extend bounds
    extended_bounds = [
        bounds[0] - xy_extension,  # xmin
        bounds[1] + xy_extension,  # xmax
        bounds[2] - xy_extension,  # ymin
        bounds[3] + xy_extension,  # ymax
        bounds[4] - 0,            # zmin (no extension below)
        bounds[5] + z_extension,  # zmax
    ]
    print(f"Extended bounds: {extended_bounds}")
    
    # Clip snag with box
    clipped_snag = snag.clip_box(bounds=extended_bounds, invert=False)
    return clipped_snag

if __name__ == '__main__':
    # Generate and visualize
    pole = generate_utility_pole(voxel_size=0.1)
    snagName = 'precolonial.False_size.snag_control.improved-tree_id.10'
    snag = get_snag(filename = snagName)
    snag.plot(scalars='resource')
    # Clip snag to pole bounds with custom extensions
    clipped_snag = clip_snag_to_pole_bounds(pole, snag, xy_extension=3.0, z_extension=5.0)

    # Visualize
    """plotter = pv.Plotter()
    plotter.add_mesh(clipped_snag, scalars='resource')
    plotter.add_mesh(pole, color='blue')
    plotter.show()"""

    #poly = combine_edit_individual_trees.edit_individual_treeVTK(clipped_snag)

    clipped_snag.plot(scalars='resource')


    
    # Export with proper filename
    export_folder = 'data/revised/utlity poles'
    export_path = os.path.join(export_folder, f'utility_pole_voxelised_{0.1}.ply')
    pole.save(export_path, texture='colors')

    #export clipped snag
    artficialTreeName = f'artificial_{snagName}'

    #export vtk file
    vtk_path = 'data/revised/final/treeMeshes'
    ply_path = 'data/revised/final/treeMeshesPly'

    #save clipped snag as artifical tree vtk
    export_path_vtk = os.path.join(vtk_path, f'{artficialTreeName}.vtk')
    clipped_snag.save(export_path_vtk)

    ply_filepath = os.path.join(ply_path, f'{artficialTreeName}.ply')

    a_vtk_to_ply.export_polydata_to_ply(clipped_snag, ply_filepath)

    print(f"Voxelized point cloud saved to {export_path}")
    print(f"artifical tree saved to {ply_filepath}")