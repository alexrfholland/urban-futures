import trimesh
import numpy as np
from scipy.spatial import cKDTree
import xarray as xr
import time
import pyvista as pv

import trimesh
import pyvista as pv
def extract_and_voxelize_layer(obj_file_path, voxel_size=1.0, preview=False):
    """Loads a 3D model file (OBJ), corrects its orientation, voxelizes it, and optionally previews it using PyVista."""
    print(f"Reading 3D model file from {obj_file_path}")
    
    # Load the 3D model
    mesh = trimesh.load(obj_file_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded object is not a valid mesh.")
    
    # Correct the orientation: rotate 90 degrees around the X-axis
    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    mesh.apply_transform(rotation_matrix)
    
    print(f"Voxelizing mesh with voxel size {voxel_size}")
    voxelized_mesh = mesh.voxelized(voxel_size)
    voxelized_mesh = voxelized_mesh.fill()  # Fill the voxelized mesh
    print(f"Voxelization complete")
    
    if preview:
        # Convert voxelized mesh to PyVista PolyData
        voxel_points = voxelized_mesh.points
        voxel_pv = pv.PolyData(voxel_points)
        
        # Create a PyVista plotter
        plotter = pv.Plotter()
        plotter.add_mesh(voxel_pv, color="blue", point_size=voxel_size*100, render_points_as_spheres=True, label="Voxels")
        

        
        # Add axes
        plotter.add_axes()
        
        # Add a legend
        plotter.add_legend()
        
        # Show the plot
        plotter.show()
    return voxelized_mesh



from scipy.spatial import cKDTree

def update_xarray_with_voxel_layer(xarr, voxelized_mesh, data_var_name="default", use3D=False):
    """
    Updates the xarray dataset by marking points near the voxelized mesh using KD-tree.
    Allows for both 2D and 3D comparisons based on the use3D parameter.
    Uses different thresholds for x, y, and z dimensions, defined within the function.
    """
    start_time = time.time()

    # Define thresholds internally
    x_threshold = 1.0
    y_threshold = 1.0
    z_threshold = 2.0

    name = f'{data_var_name}-is{data_var_name}'

    # Initialize new data variable with all False, matching the length of 'point_index'
    xarr[name] = ('point_index', np.full(xarr.sizes['point_index'], False))
    print(f"Initialized xarray variable '{data_var_name}' with False values.")

    # Extract coordinates from the site data
    if use3D:
        site_coords = np.array([xarr['x'].values, xarr['y'].values, xarr['z'].values]).T
        voxel_coords = voxelized_mesh.points
        print("Using 3D coordinates for comparison.")
    else:
        site_coords = np.array([xarr['x'].values, xarr['y'].values]).T
        voxel_coords = voxelized_mesh.points[:, :2]  # Only use x and y coordinates
        print("Using 2D coordinates (x and y) for comparison.")

    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(voxel_coords)

    # Find nearest voxel point for each site point
    distances, indices = tree.query(site_coords)

    # Mark points as True if they are within the specified thresholds
    if use3D:
        dx = np.abs(site_coords[:, 0] - voxel_coords[indices, 0])
        dy = np.abs(site_coords[:, 1] - voxel_coords[indices, 1])
        dz = np.abs(site_coords[:, 2] - voxel_coords[indices, 2])
        xarr[name].values = (dx <= x_threshold) & (dy <= y_threshold) & (dz <= z_threshold)
    else:
        dx = np.abs(site_coords[:, 0] - voxel_coords[indices, 0])
        dy = np.abs(site_coords[:, 1] - voxel_coords[indices, 1])
        xarr[name].values = (dx <= x_threshold) & (dy <= y_threshold)

    print(f"Updated xarray variable '{name}' with True values for points near the voxelized mesh.")
    print(f"Number of points marked as True: {np.sum(xarr[name].values)}")
    print(f"Thresholds used - x: {x_threshold}, y: {y_threshold}, z: {z_threshold}")

    end_time = time.time()
    print(f"Time taken to update xarray: {end_time - start_time:.2f} seconds")

    return xarr

def plot_xarray_with_pyvista(xarray, voxelized_mesh, scalar_name, point_size=5, show_edges=True, cpos='xy'):
    """
    Converts an xarray Dataset with 'point_index', 'x', 'y', 'z' coordinates
    into a PyVista PolyData object and plots it along with the voxelized mesh.
    
    Parameters:
    - xarray: The xarray Dataset containing 'x', 'y', 'z' coordinates and other data variables.
    - voxelized_mesh: The voxelized mesh from trimesh.
    - scalar_name: The name of the scalar field to visualize.
    - point_size: Size of the points in the plot.
    - show_edges: Whether to show edges of the points.
    - cpos: The camera position for the plot.
    """
    # Extract the coordinates from the xarray
    x_coords = xarray['x'].values
    y_coords = xarray['y'].values
    z_coords = xarray['z'].values
    
    # Create a numpy array of the points
    points = np.c_[x_coords, y_coords, z_coords]
    
    # Initialize the PyVista PolyData object with the points
    point_cloud = pv.PolyData(points)
    
    # Add the selected data variable in the xarray as a scalar field to the PolyData
    if scalar_name in xarray:
        point_cloud[scalar_name] = xarray[scalar_name].values
    else:
        raise ValueError(f"Scalar field '{scalar_name}' not found in the xarray dataset.")
    
    # Create a PyVista Plotter
    plotter = pv.Plotter()
    
    # Add the PolyData to the plotter with the scalar field
    plotter.add_mesh(point_cloud, scalars=scalar_name, point_size=point_size, render_points_as_spheres=True, show_edges=show_edges)
    
    # Convert voxelized mesh to PyVista PolyData and add to the plot
    #voxel_points = voxelized_mesh.points
    #voxel_pv = pv.PolyData(voxel_points)
    #plotter.add_mesh(voxel_pv, color="red", point_size=voxel_size*100, render_points_as_spheres=True, label="Voxels")

    
    # Display the plot
    plotter.show(cpos=cpos)


def get_obj_properties(site, site_data, data_var_name="default", voxel_size=0.5, use3D=True):
    """
    Main function to process a specific site, load its data, voxelize the 3D model,
    and update the xarray dataset.
    """
    print(f"Starting processing for site: {site}")

    # Load 3D model file
    obj_file = f'data/revised/rhino/meshes/{site}-additions_{data_var_name}.obj'

    print(f"Starting mesh extraction and voxelization")

    # Step 1: Extract and voxelize the specific layer
    voxelized_mesh = extract_and_voxelize_layer(obj_file, voxel_size, preview=False)
    
    # Step 2: Update the xarray with the voxelized layer
    updated_xarray = update_xarray_with_voxel_layer(site_data, voxelized_mesh, data_var_name, use3D)
    
    print(f"Completed mesh extraction and voxelization")
    return updated_xarray

# Example usage
if __name__ == "__main__":
    print("Starting the script...")
    
    
    site = 'trimmed-parade'    # Load xarray dataset
    site_path = f'data/revised/{site}-processed.nc'
    site_data = xr.open_dataset(site_path, engine='h5netcdf')

    data_var_name = "parking"
    
    print("Beginning process...")
    process_start_time = time.time()
    
    # Process site, extract mesh, voxelize, and update xarray
    xarray_result, voxelized_mesh = get_obj_properties(site, site_data, data_var_name, use3D=True)

    # Plot both the xarray data and voxelized mesh
    plot_xarray_with_pyvista(xarray_result, voxelized_mesh, scalar_name='parking-isparking', point_size=5, show_edges=True, cpos='xy')

    process_duration = time.time() - process_start_time
    print(f"Process complete (took {process_duration:.2f} seconds). Here is the resulting xarray dataset:")
    print(xarray_result)