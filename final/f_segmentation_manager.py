import laspy
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import f_SiteCoordinates
import pyacvd

#segmentFunction. Arguments: las_filepaths, site_polydata_list (this is point data, buildingMeshPolydata (these are meshes)
#this function should:
#call a lasreader function that readsthe las files and combine them into a single numpy array
#read in the vtk files and combine them into a single vtk
#transfer the las attributes to the vtk
#transfer the building data to the vtk
#use pyvista to plot the site vtk by Building_ID
#use the below functions and have informative print statements

def get_las_tiles(easting, northing, x_dim_meters, y_dim_meters):
    las_folder = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/processedLAS"
    kml_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/LAS/Tile_Index.KML"

    tiles = f_SiteCoordinates.parse_kml(kml_path)
    print(f"Parsed KML file. Found {len(tiles)} tiles.")

    x_dim_meters = 800  # Example width in meters
    y_dim_meters = 800  # Example height in meters
    selected_tiles = f_SiteCoordinates.find_tiles_within_bbox(easting, northing, tiles, x_dim_meters, y_dim_meters)
    existing_filePaths, missing_files = f_SiteCoordinates.check_tile_existence(las_folder, selected_tiles)
    #print the names of the files for both these:
    print(f"Existing files: {existing_filePaths}")
    print(f"Missing files: {missing_files}")
    return existing_filePaths


# Function to read LAS files and combine them into a single numpy array
import laspy
import numpy as np
def transfer_las_numpy_to_vtk(combined_las, site_vtk):
    print("Transferring LAS attributes to VTK using KD-tree...")

    # Extract LAS coordinates (x, y, z) from the structured NumPy array
    las_coords = np.vstack((combined_las['x'], combined_las['y'], combined_las['z'])).T

    # Extract site_vtk point coordinates
    site_coords = np.array(site_vtk.points)  # Extract points from site_vtk

    # Create a KD-tree for the LAS coordinates
    kdtree = cKDTree(las_coords)

    # Query the KD-tree to find the closest LAS point for each point in site_vtk
    distances, indices = kdtree.query(site_coords, distance_upper_bound=2)

    print(f"Transferred LAS attributes for {np.sum(distances < 2)} points within 2m.")

    # Iterate through the attributes we want to transfer
    attributes = ['Classification', 'HeightAboveGround', 'ClusterID']
    
    for attr in attributes:
        attr_data = combined_las[attr][indices]  # Get the corresponding data using the KD-tree indices
        
        # Set NaN values for points where the distance is greater than 2 (no match found)
        attr_data[distances >= 2] = np.nan

        # Add the data to the point_data in site_vtk with prefix 'LAS_'
        site_vtk.point_data[f'LAS_{attr}'] = attr_data

    # Store distances as well
    site_vtk.point_data['LAS_Distance'] = distances

    print("LAS attributes successfully transferred to site VTK.")

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def preview_las_in_open3d(las_structured_numpy):
    # Create a PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    
    # Extract coordinates (x, y, z) from the structured array
    xyz = np.vstack((las_structured_numpy['x'], las_structured_numpy['y'], las_structured_numpy['z'])).T
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    
    # Extract Classification and create a color map
    classifications = las_structured_numpy['Classification']
    unique_classes = np.unique(classifications)
    color_map = plt.cm.get_cmap('tab20')  # You can change this to any colormap you prefer
    
    # Create colors based on Classification
    colors = color_map(classifications / np.max(classifications))[:, :3]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

def preview_las_with_open3d(las):
    # Open the LAS file
    # Extract x, y, z coordinates
    xyz = np.vstack((las.x, las.y, las.z)).transpose()

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Get classification data
    classification = las['Classification']

    unique_classes, counts = np.unique(classification, return_counts=True)
    for class_id, count in zip(unique_classes, counts):
        print(f"Class ID: {class_id}, Count: {count}")

    # Color the point cloud by classification
    pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap('tab20')(classification / classification.max())[:, :3])

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def read_las_files(las_filepaths):
    las_data = []
    
    for las_file in las_filepaths:
        print(f"Reading LAS file: {las_file}")
        
        # Read the LAS file using laspy
        las = laspy.read(las_file)

        """print("Available LAS file attributes:")
        for dimension in las.point_format.dimension_names:
            print(f"- {dimension}")
            # Optionally, you can print a sample value for each attribute
            # print(f"  Sample value: {las[dimension][0]}")


        preview_las_with_open3d(las)"""

        # Get x, y, z coordinates using laspy's xyz property
        xyz = las.xyz

        # Extract extra dimensions using laspy's extra_dimension_names
        extra_dims = list(las.point_format.extra_dimension_names)
        
        classification = las['Classification']
        height_above_ground = las['HeightAboveGround']
        cluster_id = las['ClusterID']
      

        # Create structured array with the attributes
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('Classification', 'i4'),
                 ('HeightAboveGround', 'f4'), ('ClusterID', 'i4')]
        structured_points = np.zeros(len(xyz), dtype=dtype)

        # Populate structured array
        structured_points['x'] = xyz[:, 0]
        structured_points['y'] = xyz[:, 1]
        structured_points['z'] = xyz[:, 2]
        structured_points['Classification'] = classification
        structured_points['HeightAboveGround'] = height_above_ground
        structured_points['ClusterID'] = cluster_id

        las_data.append(structured_points)

    # Combine all LAS data into a single structured numpy array
    las_structured_numpy = np.concatenate(las_data)
    print(f"Combined LAS data into a single structured array with shape: {las_structured_numpy.shape}")

    print(las_structured_numpy.dtype.names)

    #preview_las_in_open3d(las_structured_numpy)
    
    return las_structured_numpy


# Function to combine LAS tiles (numpy arrays)
def combine_las(las_data):
    print("Combining LAS tiles into a single array...")
    combined_las = np.vstack(las_data)
    print(f"Combined LAS data shape: {combined_las.shape}")
    return combined_las

def combine_vtk(vtk_data):
    print(f"combining {len(vtk_data)} number of polydata tiles")

    # Use PyVista's append_polydata to combine multiple PolyData objects
    combined_vtk = pv.PolyDataFilters.append_polydata(*vtk_data, inplace=False)

    print("Combined VTK created.")
    return combined_vtk

def transfer_las_numpy_to_vtk(combined_las, site_vtk):
    print("Transferring LAS attributes to VTK using KD-tree...")

    # Extract LAS coordinates (x, y, z) from the structured NumPy array
    las_coords = np.vstack((combined_las['x'], combined_las['y'], combined_las['z'])).T

    # Extract site_vtk point coordinates
    site_coords = np.array(site_vtk.points)

    # Create a KD-tree for the LAS coordinates
    kdtree = cKDTree(las_coords)

    # Query the KD-tree to find the closest LAS point for each point in site_vtk
    distances, indices = kdtree.query(site_coords, distance_upper_bound=2)

    print(f"Transferred LAS attributes for {np.sum(distances < 2)} points within 2m.")

    # List of attributes to transfer
    attributes = ['Classification', 'HeightAboveGround', 'ClusterID']

    # Iterate through the attributes
    for attr in attributes:
        # Create a NumPy array for the attribute data of site_vtk.n_points length, initialized to NaN
        attr_data = np.full(site_vtk.n_points, np.nan)

        # Only set data for points within the 2m threshold
        valid_points = distances < 2
        attr_data[valid_points] = combined_las[attr][indices[valid_points]]

        # Add the data to the point_data in site_vtk with prefix 'LAS_'
        site_vtk.point_data[f'LAS_{attr}'] = attr_data

    # Store distances as well
    site_vtk.point_data['LAS_Distance'] = distances

    print("LAS attributes successfully transferred to site VTK.")

import open3d as o3d
import numpy as np
import pyvista as pv
import time

def transfer_building_data(building_mesh_polydata, site_points_polydata):
    print("Starting building data transfer process...")
    start_time = time.time()

    print(f"Building mesh: {building_mesh_polydata.n_cells} cells, {building_mesh_polydata.n_points} points")
    print(f"Site points: {site_points_polydata.n_points} points")

    print("Converting PyVista mesh to Open3D mesh...")
    building_vertices = np.array(building_mesh_polydata.points)
    building_faces = np.array(building_mesh_polydata.faces).reshape(-1, 4)[:, 1:4]
    building_mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(building_vertices),
        triangles=o3d.utility.Vector3iVector(building_faces)
    )
    print(f"Open3D mesh created with {len(building_mesh_o3d.vertices)} vertices and {len(building_mesh_o3d.triangles)} triangles")

    print("Creating RaycastingScene...")
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(building_mesh_o3d))
    print("RaycastingScene created and triangles added")

    print("Extracting site point coordinates...")
    site_coords = np.array(site_points_polydata.points)
    print(f"Extracted {len(site_coords)} site coordinates")

    print("Computing closest points...")
    query_points = o3d.core.Tensor(site_coords, dtype=o3d.core.Dtype.Float32)
    closest_points_start = time.time()
    closest_points_info = scene.compute_closest_points(query_points)
    closest_points_end = time.time()
    print(f"Closest points computed in {closest_points_end - closest_points_start:.2f} seconds")

    print("Computing distances...")
    distances_start = time.time()
    distances = scene.compute_distance(query_points)
    distances_end = time.time()
    print(f"Distances computed in {distances_end - distances_start:.2f} seconds")

    print("Extracting primitive IDs...")
    primitive_ids = closest_points_info['primitive_ids'].numpy()
    print(f"Extracted {len(primitive_ids)} primitive IDs")

    print("Getting building IDs and normals...")
    building_ids = np.array(building_mesh_polydata.cell_data['building_ID'])[primitive_ids]
    
    if 'Normals' in building_mesh_polydata.cell_data:
        print("Using existing normals from building mesh")
        building_normals = np.array(building_mesh_polydata.cell_data['Normals'])[primitive_ids]
    else:
        print("Computing normals for building mesh")
        building_mesh_polydata.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        building_normals = np.array(building_mesh_polydata.cell_data['Normals'])[primitive_ids]
    print(f"Retrieved {len(building_ids)} building IDs and normals")

    print("Adding data to site_points_polydata point_data...")
    site_points_polydata.point_data['building_ID'] = building_ids
    site_points_polydata.point_data['building_normalX'] = building_normals[:, 0]
    site_points_polydata.point_data['building_normalY'] = building_normals[:, 1]
    site_points_polydata.point_data['building_normalZ'] = building_normals[:, 2]
    site_points_polydata.point_data['building_distance'] = distances.numpy()
    print("Data successfully added to site_points_polydata")

    end_time = time.time()
    print(f"Building data transfer completed in {end_time - start_time:.2f} seconds")

    print("Summary:")
    print(f"  Processed {building_mesh_polydata.n_cells} building cells")
    print(f"  Transferred data to {site_points_polydata.n_points} site points")
    print(f"  Closest point computation took {closest_points_end - closest_points_start:.2f} seconds")
    print(f"  Distance computation took {distances_end - distances_start:.2f} seconds")
    print(f"  Total processing time: {end_time - start_time:.2f} seconds")

    # Create a plotter
    """plotter = pv.Plotter()
    
    # Add points to the plotter, coloring by building_ID
    plotter.add_mesh(site_points_polydata, 
                     scalars='building_distance', 
                     render_points_as_spheres=True,
                     point_size=5,
                     cmap='viridis')
    
    # Display the plot
    plotter.show()"""



def transfer_building_data2(building_vtk, site_vtk):
    print("Transferring building data (ID, normals, distance)...")


    # Extract centroids of building_vtk cells (vectorized)
    building_cell_centroids = building_vtk.cell_centers().points

    print(f"Building cell centroids shape: {building_cell_centroids.shape}")
    
    # Extract or compute building normals
    if 'Normals' in building_vtk.cell_data:
        building_normals = np.array(building_vtk.cell_data['Normals'])
        print("Normals already exist in building VTK.")
    else:
        building_vtk.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        building_normals = np.array(building_vtk.cell_data['Normals'])
        print("Normals computed for building VTK.")

    # Extract building IDs
    building_ids_array = np.array(building_vtk.cell_data['building_ID'])

    # Extract site_vtk point coordinates
    site_coords = np.array(site_vtk.points)

    # Create a KD-tree for the building centroids
    kdtree = cKDTree(building_cell_centroids)

    # Query the KD-tree to find the closest building cell for each point in site_vtk
    distances, indices = kdtree.query(site_coords)

    # Create arrays to store the closest building attributes
    building_ids = building_ids_array[indices]
    normalX = building_normals[indices][:, 0]
    normalY = building_normals[indices][:, 1]
    normalZ = building_normals[indices][:, 2]
    building_distances = distances

    # Add data to site_vtk point_data
    site_vtk.point_data['building_ID'] = building_ids
    site_vtk.point_data['building_normalX'] = normalX
    site_vtk.point_data['building_normalY'] = normalY
    site_vtk.point_data['building_normalZ'] = normalZ
    site_vtk.point_data['building_distance'] = building_distances

    print("Building data successfully transferred to site VTK.")




def transfer_building_data2(building_vtk, site_vtk):
    print("Transferring building data (ID, normals, distance)...")


    # Extract centroids of building_vtk cells (vectorized)
    building_cell_centroids = building_vtk.cell_centers().points

    print(f"Building cell centroids shape: {building_cell_centroids.shape}")
    
    # Extract or compute building normals
    if 'Normals' in building_vtk.cell_data:
        building_normals = np.array(building_vtk.cell_data['Normals'])
        print("Normals already exist in building VTK.")
    else:
        building_vtk.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        building_normals = np.array(building_vtk.cell_data['Normals'])
        print("Normals computed for building VTK.")

    # Extract building IDs
    building_ids_array = np.array(building_vtk.cell_data['building_ID'])

    # Extract site_vtk point coordinates
    site_coords = np.array(site_vtk.points)

    # Create a KD-tree for the building centroids
    kdtree = cKDTree(building_cell_centroids)

    # Query the KD-tree to find the closest building cell for each point in site_vtk
    distances, indices = kdtree.query(site_coords)

    # Create arrays to store the closest building attributes
    building_ids = building_ids_array[indices]
    normalX = building_normals[indices][:, 0]
    normalY = building_normals[indices][:, 1]
    normalZ = building_normals[indices][:, 2]
    building_distances = distances

    # Add data to site_vtk point_data
    site_vtk.point_data['building_ID'] = building_ids
    site_vtk.point_data['building_normalX'] = normalX
    site_vtk.point_data['building_normalY'] = normalY
    site_vtk.point_data['building_normalZ'] = normalZ
    site_vtk.point_data['building_distance'] = building_distances

    print("Building data successfully transferred to site VTK.")


# Main segmentFunction
def segmentFunction(easting, northing, eastings_dim, northings_dim, site_polydata_list, buildingMeshPolydata, buffer=None):
    print("Starting segmentation function...")

      # Step 2: Combine VTK files into a single VTK (site VTK)
    if isinstance(site_polydata_list, list):
        print("site_polydata_list is a list. Combining VTK files...")
        site_vtk = combine_vtk(site_polydata_list)
    else:
        print("site_polydata_list is a single polydata. No combination needed.")
        site_vtk = site_polydata_list

    print("Type of site_vtk:", type(site_vtk))

    if buffer is not None:
        eastings_dim = eastings_dim + buffer
        northings_dim = northings_dim + buffer
        print(f"Applying buffer of {buffer} meters, eastings dim: {eastings_dim}, northings dim: {northings_dim}")

    #Step 0: Get the las filepaths
    las_filepaths = get_las_tiles(easting, northing, eastings_dim, northings_dim)
    
    # Step 1: Read and combine LAS files
    combined_las = read_las_files(las_filepaths)

    # Step 3: Transfer LAS attributes to the VTK
    transfer_las_numpy_to_vtk(combined_las, site_vtk)

   # Step 0: Transfer building data to the VTK
    transfer_building_data(buildingMeshPolydata, site_vtk)

    print("Segmentation process complete.")

    

    # Step 5: Plot the site VTK by Building_ID
    #print("Plotting the site VTK colored by Building_ID...")
    #site_vtk.plot(rgb=True)

    return site_vtk



if __name__ == "__main__":
    eastings_dim = 2000  # 2000 meters in eastings direction
    northings_dim = 2000  # 2000 meters in northings direction
    easting = 320757.79029528715
    northing = 5814438.136253171

    las_filepaths = get_las_tiles(easting, northing, eastings_dim, northings_dim)
    
    # Step 1: Read and combine LAS files
    combined_las = read_las_files(las_filepaths)
    
    # Step 3: Transfer LAS attributes to the VTK
    #transfer_las_numpy_to_vtk(combined_las, site_vtk)