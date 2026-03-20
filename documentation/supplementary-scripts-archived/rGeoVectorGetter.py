import geopandas as gpd
import numpy as np
import pyvista as pv
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString, shape
import requests
from io import StringIO
import csv
import pandas as pd
from pyproj import CRS, Transformer
import shapely.ops
from shapely.ops import linemerge
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from f_SiteCoordinates import get_site_coordinates
import math
from scipy.spatial import cKDTree


###
import numpy as np
from rasterio import features
from affine import Affine
from shapely.geometry import box, shape, mapping
import pandas as pd
from fastkml import kml
from f_SiteCoordinates import get_site_coordinates
import open3d as o3d

def raster_to_points(raster, transform, resolution):
    """
    Convert raster to 2D points, keeping only filled areas and respecting pixel resolution.
    
    :param raster: The input raster array
    :param transform: The affine transform from the rasterization step
    :param resolution: The pixel resolution in meters
    """
    print(f"Converting raster to points with resolution {resolution}m...")
    rows, cols = np.indices(raster.shape)
    ids = raster[rows, cols]
    
    # Filter out the -1 values (areas outside the features)
    valid_mask = ids != -1
    rows, cols = rows[valid_mask], cols[valid_mask]
    ids = ids[valid_mask]
    
    # Convert pixel coordinates to real-world coordinates
    # Add 0.5 to get the center of each pixel
    x = transform.c + (cols + 0.5) * resolution
    y = transform.f + (rows + 0.5) * resolution
    
    coordinates = np.column_stack((x, y))
    
    print(f"Extracted {len(coordinates)} points from the raster")
    print(f"Unique IDs: {np.unique(ids)}")
    
    return coordinates, ids

def rasterize_gdf(gdf, bbox, resolution=1):
    """Rasterize the GeoDataFrame."""
    print(f"Rasterizing GeoDataFrame with resolution: {resolution}m")
    bounds = bbox.bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = Affine.translation(bounds[0], bounds[1]) * Affine.scale(resolution, resolution)
    
    shapes = [(geom, i) for i, geom in enumerate(gdf.geometry)]
    raster = features.rasterize(shapes, out_shape=(height, width), transform=transform, fill=-1, all_touched=True)
    print(f"Rasterization complete. Raster shape: {raster.shape}")
    return raster, transform


def assign_attributes(points, ids, gdf):
    """Assign attributes from the GeoDataFrame to the points."""
    print("Assigning attributes to points...")
    points_df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'id': ids})
    merged_df = points_df.merge(gdf.reset_index(), left_on='id', right_index=True, how='left')
    print(f"Attributes assigned. DataFrame shape: {merged_df.shape}")
    return merged_df.drop(['id', 'geometry'], axis=1, errors='ignore')

def create_pyvista_polydata_with_terrain_and_roads(terrain_grid, road_points, attributes_df, distance_threshold=.25):
    """
    Create a PyVista PolyData object with terrain and road points, adding terrain points that are 
    beyond a distance threshold from roads, and initialize empty attributes for those terrain points.
    
    :param terrain_grid: pyvista.StructuredGrid representing the terrain mesh
    :param road_points: numpy 2D array of (x, y) coordinates representing road points
    :param attributes_df: pandas DataFrame containing attributes for road points
    :param distance_threshold: distance threshold beyond which terrain points are included
    :return: pyvista.PolyData object with points and attributes
    """
    # Terrain grid points
    terrain_points = terrain_grid.points
    print(f"Terrain grid contains {len(terrain_points)} points.")

    # KD-tree for road points
    road_kdtree = cKDTree(road_points)
    print(f"KD-tree created for {len(road_points)} road points.")

    # Find terrain points beyond the distance threshold from any road point
    distances, _ = road_kdtree.query(terrain_points[:, :2], k=1)
    terrain_far_points = terrain_points[distances > distance_threshold]
    print(f"Identified {len(terrain_far_points)} terrain points more than {distance_threshold}m from any road point.")

    # Create an empty attributes DataFrame for terrain points, including 'roadInfo_type' and 'scale'
    empty_attrs_df = pd.DataFrame({col: pd.Series(dtype=attributes_df[col].dtype) for col in attributes_df.columns}, 
                                   index=np.arange(len(terrain_far_points)))
    
    # Handle 'roadInfo_type' column
    if 'roadInfo_type' in attributes_df.columns:
        empty_attrs_df['roadInfo_type'] = 'other'
        print(f"Initialized 'roadInfo_type' for terrain points as 'other'.")

    # Add the 'scale' attribute: 1 for terrain points, 0.5 for road points
    attributes_df['scale'] = 0.5
    empty_attrs_df['scale'] = 1.0
    print(f"Initialized 'scale' attribute: 0.5 for road points and 1.0 for terrain points.")

    print(f"Initialized empty attributes DataFrame with {empty_attrs_df.shape[0]} rows and {empty_attrs_df.shape[1]} columns.")

    # Merge road points with terrain points
    all_points = np.vstack((road_points, terrain_far_points[:, :2]))
    z_road = terrain_grid.points[cKDTree(terrain_grid.points[:, :2]).query(road_points)[1], 2]
    all_points_3d = np.vstack((np.column_stack((road_points, z_road)), terrain_far_points))
    print(f"Combined road points and terrain points into 3D array with shape {all_points_3d.shape}.")

    # Combine the attributes
    combined_attrs_df = pd.concat([attributes_df, empty_attrs_df], ignore_index=True)
    print(f"Combined attributes into DataFrame with {combined_attrs_df.shape[0]} rows and {combined_attrs_df.shape[1]} columns.")

    # Create a PyVista PolyData object
    polydata = pv.PolyData(all_points_3d)
    print(f"Created PyVista PolyData with {polydata.n_points} points.")

    # Add attributes
    for column in combined_attrs_df.columns:
        polydata.point_data[column] = combined_attrs_df[column].values
        print(f"Added attribute '{column}' to PolyData.")

    return polydata


def adjust_height(mesh):
    # Define the height adjustments for each type
    height_adjustments = {
        'Road Channel': -0.25,
        'Median': 0.25,
        'Footway': 0,  # No adjustment specified, so keeping it at 0
        'Road Kerb': -1,
        'Carriageway': -0.5,
        'Tramway': 0,
        'Other': -1
    }

    # Get the points and 'type' array
    points = mesh.points
    type_array = mesh['roadInfo_type']

    # Initialize the displacement array
    displacement = np.zeros(len(points))

    # Apply displacements based on type
    for type_name, adjustment in height_adjustments.items():
        mask = (type_array == type_name)
        displacement[mask] = adjustment
    # Apply the displacement to the z-coordinate
    points[:, 2] += displacement

    # Update the mesh with the new point positions
    mesh.points = points

    return mesh

###
def create_pyvista_polydata_with_terrain_and_roads(terrain_grid, road_points, attributes_df, distance_threshold=.25):
    """
    Create a PyVista PolyData object with terrain and road points, adding terrain points that are 
    beyond a distance threshold from roads, and initialize empty attributes for those terrain points.
    
    :param terrain_grid: pyvista.StructuredGrid representing the terrain mesh
    :param road_points: numpy 2D array of (x, y) coordinates representing road points
    :param attributes_df: pandas DataFrame containing attributes for road points
    :param distance_threshold: distance threshold beyond which terrain points are included
    :return: pyvista.PolyData object with points and attributes
    """
    # Terrain grid points
    terrain_points = terrain_grid.points
    print(f"Terrain grid contains {len(terrain_points)} points.")

    # KD-tree for road points
    road_kdtree = cKDTree(road_points)
    print(f"KD-tree created for {len(road_points)} road points.")

    # Find terrain points beyond the distance threshold from any road point
    distances, _ = road_kdtree.query(terrain_points[:, :2], k=1)
    terrain_far_points = terrain_points[distances > distance_threshold]
    print(f"Identified {len(terrain_far_points)} terrain points more than {distance_threshold}m from any road point.")

    # Create an empty attributes DataFrame for terrain points, including 'roadInfo_type' and 'scale'
    empty_attrs_df = pd.DataFrame({col: pd.Series(dtype=attributes_df[col].dtype) for col in attributes_df.columns}, 
                                   index=np.arange(len(terrain_far_points)))
    
    # Handle 'roadInfo_type' column
    if 'roadInfo_type' in attributes_df.columns:
        empty_attrs_df['roadInfo_type'] = 'other'
        print(f"Initialized 'roadInfo_type' for terrain points as 'other'.")

    # Add the 'scale' attribute: 1 for terrain points, 0.5 for road points
    attributes_df['scale'] = 0.5
    empty_attrs_df['scale'] = 1.0
    print(f"Initialized 'scale' attribute: 0.5 for road points and 1.0 for terrain points.")

    print(f"Initialized empty attributes DataFrame with {empty_attrs_df.shape[0]} rows and {empty_attrs_df.shape[1]} columns.")

    # Merge road points with terrain points
    all_points = np.vstack((road_points, terrain_far_points[:, :2]))
    z_road = terrain_grid.points[cKDTree(terrain_grid.points[:, :2]).query(road_points)[1], 2]
    all_points_3d = np.vstack((np.column_stack((road_points, z_road)), terrain_far_points))
    print(f"Combined road points and terrain points into 3D array with shape {all_points_3d.shape}.")

    # Combine the attributes
    combined_attrs_df = pd.concat([attributes_df, empty_attrs_df], ignore_index=True)
    print(f"Combined attributes into DataFrame with {combined_attrs_df.shape[0]} rows and {combined_attrs_df.shape[1]} columns.")

    # Create a PyVista PolyData object
    polydata = pv.PolyData(all_points_3d)
    print(f"Created PyVista PolyData with {polydata.n_points} points.")

    # Add attributes
    for column in combined_attrs_df.columns:
        polydata.point_data[column] = combined_attrs_df[column].values
        print(f"Added attribute '{column}' to PolyData.")

    return polydata

def plot_polydata_with_scaled_glyphs_by_size(polydata):
    """
    Plot PyVista PolyData object using scaled box glyphs based on unique sizes, coloring by 'roadInfo_type'.
    
    :param polydata: pyvista.PolyData object containing the points and attributes.
    """
    # Ensure the required attributes exist
    if 'scale' not in polydata.point_data or 'roadInfo_type' not in polydata.point_data:
        raise ValueError("'scale' and 'roadInfo_type' attributes are required in the PolyData object.")

    # Find unique sizes from the 'scale' attribute
    unique_sizes = np.unique(polydata.point_data['scale'])
    print(f"Unique sizes found: {unique_sizes}")

    # Create a plotter
    plotter = pv.Plotter()

    # Loop through each unique size
    for size in unique_sizes:
        # Extract the PolyData where the 'scale' matches the current size
        mask = polydata.point_data['scale'] == size
        extracted_polydata = polydata.extract_points(mask)

        # Create a box glyph, scaled by x = size, y = size, z = 5
        box_glyph = pv.Box(bounds=(-size/2, size/2, -size/2, size/2, -2.5, 2.5))

        # Create glyphs using the extracted PolyData
        glyphs = extracted_polydata.glyph(orient=False, scale=False, geom=box_glyph)

        # Add the glyphs to the plot, coloring by 'roadInfo_type'
        plotter.add_mesh(glyphs, scalars='roadInfo_type', show_scalar_bar=True, cmap = 'Set1')

    # Show the plot
    plotter.show_grid()
    plotter.show()


def getRoadVoxels(site_easting, site_northing, eastingDim, northingDim, roadGdf, landscapeMesh, resolution=.25):
    print(f"Processing site at Easting: {site_easting}, Northing: {site_northing}")
    print(f"Bounding box size: {eastingDim}m x {northingDim}m")
    print(f"Using resolution: {resolution}m")
    
    bbox = box(site_easting - eastingDim/2, site_northing - northingDim/2,
               site_easting + eastingDim/2, site_northing + northingDim/2)
        
    raster, transform = rasterize_gdf(roadGdf, bbox, resolution)
    points, ids = raster_to_points(raster, transform, resolution)
    attributes = assign_attributes(points, ids, roadGdf)

    
    polydata = create_pyvista_polydata_with_terrain_and_roads(landscapeMesh, points, attributes)

    polydata = adjust_height(polydata)

    #plot_polydata_with_scaled_glyphs_by_size(polydata)


    #centroids, sizes, depths = create_octree_grid(polydata.points, max_depth=10)

    #create_polyOct_and_transfer_attributes(polydata, centroids, sizes, depths)



    return polydata


#OCTREE STUFF ACTING STRANGE


#acting strange
def traverse_octree_with_sibling_check(octree):
    # Initialize lists to store centroids, sizes, and depths
    centroids = []
    sizes = []
    depths = []

    # Recursive function to check parent nodes for siblings
    def check_parent_for_siblings(node, node_info, parent=None, parent_info=None):
        node_centroid = node_info.origin + 0.5 * node_info.size

        # For internal nodes, check if the parent has siblings
        if parent and isinstance(parent, o3d.geometry.OctreeInternalNode):
            non_null_children = [child for child in parent.children if child is not None]
            
            # If parent has multiple children (i.e., siblings), collect centroids and sizes
            if len(non_null_children) > 1:
                centroids.append(node_centroid)
                sizes.append(node_info.size)
                depths.append(node_info.depth)
            else:
                # If parent has no siblings, move up the hierarchy and check the next parent
                check_parent_for_siblings(parent, parent_info, parent_info.parent, parent_info.parent_info)
        else:
            # For leaf nodes, directly store centroid, size, and depth
            centroids.append(node_centroid)
            sizes.append(node_info.size)
            depths.append(node_info.depth)

    # Callback function to traverse the octree and start sibling check from leaf nodes
    def node_callback(node, node_info, parent=None, parent_info=None):
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            # Start the sibling check from leaf nodes
            check_parent_for_siblings(node, node_info, parent, parent_info)

    # Traverse the octree using the callback function
    octree.traverse(node_callback)

    # Output the results
    print(f"Number of valid nodes: {len(centroids)}")
    return centroids, sizes, depths

#acting strange
def create_octree_grid(points, max_depth=10):
    """
    Create an octree from points and return centroids of internal nodes with exactly one child 
    and all leaf nodes. Print the depth (level) of these nodes.
    
    :param points: numpy array of 3D points
    :param max_depth: maximum depth of the octree
    :return: numpy array of centroids and scales
    """
                        
    print(f"Creating point cloud from {len(points)} points.")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Create the octree
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    print(f"Octree created with maximum depth {max_depth}.")

    # Extract centroids of nodes (internal nodes with exactly one child and all leaf nodes), their scale, and depth
    centroids, sizes, depths = traverse_octree_with_sibling_check(octree)
    
    centroids = np.array(centroids)
    sizes = np.array(sizes)
    depths = np.array(depths)
    
    print(f"Extracted {len(centroids)} nodes (leaf nodes and nodes with exactly one child).")
    print(f"Unique levels (depth) at the nodes: {np.unique(depths)}")
    print(f"Unique node sizes: {np.unique(sizes)}")

    # Return the node information
    return centroids, sizes, depths


#acting strange
def create_polyOct_and_transfer_attributes(polydata, centroids, sizes, depths):
    """
    Create a new PyVista PolyData (polyOct) with points at the centroids and transfer the attributes from polydata.
    
    :param polydata: PyVista PolyData object containing the original points and attributes
    :param centroids: numpy array of 3D points representing octree centroids
    :param scales: numpy array of scale values corresponding to each centroid
    :return: polyOct PyVista PolyData object with centroids and transferred attributes
    """
    print(f"Creating PolyData from {len(centroids)} centroids.")

    # Step 1: Create polyOct with centroids as points
    polyOct = pv.PolyData(centroids)
    print(f"PolyData created with {polyOct.n_points} centroids.")

    # Step 2: Add the cellScale as a point data attribute
    polyOct.point_data['cellScale'] = sizes * 1.5
    polyOct.point_data['cellDepth'] = depths
    print(f"Added 'cellScale' attribute to PolyData with {len(sizes)} values.")

    # Step 3: Transfer attributes from original polydata to polyOct using KD-tree
    original_points = polydata.points
    kdtree = cKDTree(original_points)
    print(f"KD-tree created for {len(original_points)} original points.")

    # Find the nearest points in original polydata for each centroid in polyOct
    _, nearest_indices = kdtree.query(centroids)
    print(f"Found nearest points for each centroid using KD-tree.")

    # Transfer attributes
    for column in polydata.point_data.keys():
        nearest_attr_values = polydata.point_data[column][nearest_indices]
        polyOct.point_data[column] = nearest_attr_values
        print(f"Transferred attribute '{column}' to PolyData.")

        # Initialize the plotter
    
    print('plotting octree')
    plotter = pv.Plotter()
    
    # Create box glyphs based on the 'cellScale' attribute
    box = pv.Cube()  # Unit box glyph to be scaled
    glyphs = polyOct.glyph(orient=False, scale='cellScale', factor=1.0, geom=box)
    
    #print all unique scales
    print(f'unique scales in octree are {np.unique(polyOct.point_data["cellScale"])}')

    plotter.add_mesh(polyOct, scalars='cellScale', cmap='viridis', point_size=5, render_points_as_spheres=True, show_scalar_bar=True)

    # Add glyphs to the plotter
    plotter.add_mesh(glyphs, scalars="roadInfo_type", cmap="viridis")
    
    plotter.show()



    return polyOct

