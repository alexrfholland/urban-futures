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
from shapely.geometry import box


def retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url, dataset_name):
    # Calculate the radius that encompasses the entire rectangular area
    radius_meters = math.sqrt((eastings_dim/2)**2 + (northings_dim/2)**2)

    print(f"Retrieving {dataset_name} data for point ({easting}, {northing}) with dimensions {eastings_dim}x{northings_dim}m...")
    print(f"Using calculated radius of {radius_meters:.2f}m for the query")

    transformer = Transformer.from_crs("EPSG:28355", "EPSG:4326", always_xy=True)
    center_lon, center_lat = transformer.transform(easting, northing)
    
    print(f"Center point (EPSG:4326): Lon: {center_lon}, Lat: {center_lat}")

    all_results = []
    offset = 0
    limit = 100

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "select": "*",
        }
        if dataset_name == "trees":
            params["where"] = f"distance(geolocation, geom'POINT({center_lon} {center_lat})', {radius_meters}m)"
        else:
            params["where"] = f"distance(geo_point_2d, geom'POINT({center_lon} {center_lat})', {radius_meters}m)"
        
        print(f"Making API request with offset: {offset}, limit: {limit}")
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            all_results.extend(results)
            
            print(f"Retrieved {len(results)} new results. Total: {len(all_results)} {dataset_name}")
            
            if len(results) < limit:
                print(f"Received {len(results)} results, which is less than the limit of {limit}. Finished retrieving data.")
                break
            
            offset += limit
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    print(f"Total {dataset_name} retrieved: {len(all_results)}")

    # Print the keys of the first result to inspect the structure
    if all_results:
        print("Keys in the first result:", all_results[0].keys())

    # Create geometry from the 'geo_point_2d' field and build a debug GeoDataFrame
    geometry = [Point(item['geo_point_2d']['lon'], item['geo_point_2d']['lat']) for item in all_results if 'geo_point_2d' in item]
    gdf = gpd.GeoDataFrame(all_results, geometry=geometry, crs="EPSG:4326")

    # Reproject the GeoDataFrame to EPSG:28355
    gdf_epsg28355 = gdf.to_crs(epsg=28355)

    # Calculate the bounding box of the retrieved data in EPSG:28355
    retrieved_extent = gdf_epsg28355.total_bounds
    print(f"Retrieved data bounding box (EPSG:28355): {retrieved_extent}")

    # Calculate the input bounding box based on the easting/northing and dimensions
    min_easting, min_northing = easting - eastings_dim / 2, northing - northings_dim / 2
    max_easting, max_northing = easting + eastings_dim / 2, northing + northings_dim / 2
    input_extent = [min_easting, min_northing, max_easting, max_northing]
    print(f"Input bounding box (EPSG:28355): {input_extent}")

    # Calculate the dimensions of the input bounding box
    input_x_dim = max_easting - min_easting
    input_y_dim = max_northing - min_northing

    # Calculate the dimensions of the retrieved bounding box
    retrieved_x_dim = retrieved_extent[2] - retrieved_extent[0]
    retrieved_y_dim = retrieved_extent[3] - retrieved_extent[1]

    # Print the dimensions of both the input and retrieved bounding boxes
    print(f"\nBounding Box Dimensions:")
    print(f"Input bounding box dimensions: X: {input_x_dim:.2f}m, Y: {input_y_dim:.2f}m")
    print(f"Retrieved bounding box dimensions: X: {retrieved_x_dim:.2f}m, Y: {retrieved_y_dim:.2f}m")

    # Compare the bounding boxes
    print("\nBounding Box Comparison:")
    print(f"Retrieved bounding box: {retrieved_extent}")
    print(f"Input bounding box: {input_extent}")

    if (retrieved_extent[0] < input_extent[0] or retrieved_extent[1] < input_extent[1] or
        retrieved_extent[2] > input_extent[2] or retrieved_extent[3] > input_extent[3]):
        print("The retrieved data extends beyond the input bounding box.")
    else:
        print("The retrieved data is within the input bounding box.")

    return all_results



def retrieve_geospatial_data2(easting, northing, eastings_dim, northings_dim, base_url, dataset_name):
    # Calculate the radius that encompasses the entire rectangular area
    radius_meters = math.sqrt((eastings_dim/2)**2 + (northings_dim/2)**2)

    print(f"Retrieving {dataset_name} data for point ({easting}, {northing}) with dimensions {eastings_dim}x{northings_dim}m...")
    print(f"Using calculated radius of {radius_meters:.2f}m for the query")

    transformer = Transformer.from_crs("EPSG:28355", "EPSG:4326", always_xy=True)
    center_lon, center_lat = transformer.transform(easting, northing)
    
    print(f"Center point (EPSG:4326): Lon: {center_lon}, Lat: {center_lat}")

    all_results = []
    offset = 0
    limit = 100

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "select": "*",
        }
        if dataset_name == "trees":
            params["where"] = f"distance(geolocation, geom'POINT({center_lon} {center_lat})', {radius_meters}m)"
        else:
            params["where"] = f"distance(geo_point_2d, geom'POINT({center_lon} {center_lat})', {radius_meters}m)"
        
        print(f"Making API request with offset: {offset}, limit: {limit}")
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            all_results.extend(results)
            
            print(f"Retrieved {len(results)} new results. Total: {len(all_results)} {dataset_name}")
            
            if len(results) < limit:
                print(f"Received {len(results)} results, which is less than the limit of {limit}. Finished retrieving data.")
                break
            
            offset += limit
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    print(f"Total {dataset_name} retrieved: {len(all_results)}")
    return all_results

def plot_trimmed_and_untrimmed_gdfs(gdf, trimmed_gdf, title="Comparison of Untrimmed and Trimmed GeoDataFrames"):
    """
    Plot the untrimmed and trimmed GeoDataFrames for comparison.
    
    Parameters:
    - gdf: The original untrimmed GeoDataFrame.
    - trimmed_gdf: The GeoDataFrame after trimming to the bounding box.
    - title: The title of the plot.
    """
    # Create a plot with Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the original untrimmed GeoDataFrame
    gdf.plot(ax=ax, color='lightblue', edgecolor='blue', label='Original (Untrimmed)', alpha=0.5)
    
    # Plot the trimmed GeoDataFrame
    trimmed_gdf.plot(ax=ax, color='salmon', edgecolor='red', label='Trimmed', alpha=0.7)
    
    # Add plot title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    
    # Add legend
    ax.legend()
    
    # Show the plot
    plt.show()

def trim_gdf_to_bounding_box(gdf, easting, northing, eastings_dim, northings_dim, target_crs=28355):
    # Ensure the GeoDataFrame is in the target CRS (EPSG:28355)
    if gdf.crs != f"EPSG:{target_crs}":
        gdf = gdf.to_crs(epsg=target_crs)
    
    # Calculate the bounding box from the center point and dimensions
    min_easting = easting - eastings_dim / 2
    max_easting = easting + eastings_dim / 2
    min_northing = northing - northings_dim / 2
    max_northing = northing + northings_dim / 2

    # Create the bounding box geometry in the target CRS
    bounding_box = box(min_easting, min_northing, max_easting, max_northing)

    # Print statements with details before and after the trimming
    original_bounds = gdf.total_bounds
    print(f"Original bounding box dimensions: Easting_dim = {original_bounds[2] - original_bounds[0]:.1f}m, Northing_dim = {original_bounds[3] - original_bounds[1]:.1f}m")
    print(f"Desired bounding box dimensions: Easting_dim = {eastings_dim:.1f}m, Northing_dim = {northings_dim:.1f}m")
    print(f" - Calculated bounding box: MinEasting = {min_easting:.1f}, MaxEasting = {max_easting:.1f}, MinNorthing = {min_northing:.1f}, MaxNorthing = {max_northing:.1f}")
    
    # Clip the GeoDataFrame to the bounding box
    trimmed_gdf = gdf.clip(bounding_box)

    # Get the bounds of the trimmed GeoDataFrame
    trimmed_bounds = trimmed_gdf.total_bounds
    print(f"Trimmed bounding box dimensions: Easting_dim = {trimmed_bounds[2] - trimmed_bounds[0]:.1f}m, Northing_dim = {trimmed_bounds[3] - trimmed_bounds[1]:.1f}m")

    plot_trimmed_and_untrimmed_gdfs(gdf, trimmed_gdf)

    return trimmed_gdf


def handle_building_dataOLD(easting, northing, eastings_dim, northings_dim):
    print("Processing building data...")
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/2023-building-footprints/records"
    buildings = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url, "2023-building-footprints")

    gdf = gpd.GeoDataFrame.from_features([
        {
            "type": "Feature",
            "geometry": building["geo_shape"]["geometry"],
            "properties": {k: v for k, v in building.items() if k != "geo_shape"}
        }
        for building in buildings
    ])
    gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)

    gdf = trim_gdf_to_bounding_box(gdf, easting, northing, eastings_dim, northings_dim)
    print(f"Created GeoDataFrame with {len(gdf)} buildings")
    return create_3d_building_visualization(gdf)

    
def handle_road_segments(easting, northing, eastings_dim, northings_dim):
    dataset_name = "road-segments-with-surface-type"
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/road-segments-with-surface-type/records"
    road_segments = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url, dataset_name)
    

    gdf_data = []
    for segment in road_segments:
        attributes = segment.copy()  # Copy all attributes
        geo = attributes.pop("geo_shape")["geometry"]  # Remove and get the geometry
        attributes["geometry"] = shape(geo)  # Add geometry as a separate column
        gdf_data.append(attributes)
    
    gdf = gpd.GeoDataFrame(gdf_data)
    gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)
    print(f"Created GeoDataFrame with {len(gdf)} road segments")

    gdf = trim_gdf_to_bounding_box(gdf, easting, northing, eastings_dim, northings_dim)
    
    # Print all column names (attribute headings)
    print("Attribute headings:")
    print(gdf.columns.tolist())

    if 'type' in gdf.columns:
        unique_types = gdf['type'].unique()
        print("\nUnique values in the 'type' attribute:")
        for type_value in unique_types:
            print(f"- {type_value}")
    else:
        print("\n'type' attribute not found in the data.")
    
    return gdf

    
    
"""def handle_road_segments(easting, northing, eastings_dim, northings_dim):
    dataset_name = "road-segments-with-surface-type"
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/road-segments-with-surface-type/records"
    
    road_segments = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url, dataset_name)
    
    if road_segments:
        geometries = []
        for segment in road_segments:
            geo = segment["geo_shape"]["geometry"]
            geom = shape(geo)
            geometries.append(geom)
        
        gdf = gpd.GeoDataFrame(geometry=geometries)
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)
        
        print(f"Created GeoDataFrame with {len(gdf)} road segments")
        return gdf
    else:
        print("Failed to retrieve or process road segment data.")
        return None
"""
def handle_road_corridors(easting, northing, eastings_dim, northings_dim):
    dataset_name = "road-corridors"
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/road-corridors/records"
    road_corridors = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url, dataset_name)
    
    if road_corridors:
        geometries = []
        for corridor in road_corridors:
            geo = corridor["geo_shape"]["geometry"]
            geom = shape(geo)
            geometries.append(geom)
        
        gdf = gpd.GeoDataFrame(geometry=geometries)
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)
        
        print(f"Created GeoDataFrame with {len(gdf)} road corridors")
        return gdf
    else:
        print("Failed to retrieve or process road corridor data.")
        return None

def handle_trees(easting, northing, eastings_dim, northings_dim):
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/trees-with-species-and-dimensions-urban-forest/records"
    trees = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url, "trees")
    
    if trees:
        gdf = gpd.GeoDataFrame([
            {
                **tree,
                'geometry': Point(tree['longitude'], tree['latitude'])
            }
            for tree in trees
        ])
        
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)
        
        print(f"Number of trees: {len(gdf)}")
        if 'diameter_breast_height' in gdf.columns:
            # Filter out null values and convert to numeric
            dbh_values = pd.to_numeric(gdf['diameter_breast_height'], errors='coerce').dropna()
            if not dbh_values.empty:
                print(f"Average diameter at breast height: {dbh_values.mean():.2f} cm")
            else:
                print("No valid diameter at breast height data available.")
        
        # Additional statistics
        print("\nTop 5 most common tree species:")
        print(gdf['common_name'].value_counts().head())
        
        print("\nAge distribution:")
        print(gdf['age_description'].value_counts())
        
        print("\nLocation distribution:")
        print(gdf['located_in'].value_counts())
        
        return gdf
    else:
        print("Failed to retrieve or process tree data.")
        return None

def handle_tree_canopies(easting, northing, eastings_dim, northings_dim):
    dataset_name = "tree-canopies-2021-urban-forest"
    base_url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/tree-canopies-2021-urban-forest/records"
    
    tree_canopies = retrieve_geospatial_data(easting, northing, eastings_dim, northings_dim, base_url, dataset_name)
    
    if tree_canopies:
        geometries = []
        for canopy in tree_canopies:
            geo = canopy["geo_shape"]["geometry"]
            geom = shape(geo)
            geometries.append(geom)
        
        gdf = gpd.GeoDataFrame(geometry=geometries)
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=28355)
        
        print(f"Created GeoDataFrame with {len(gdf)} tree canopies")
        return gdf
    else:
        print("Failed to retrieve or process tree canopy data.")
        return None

def retrieve_local_shapefile(file_path, easting, northing, eastings_dim, northings_dim):
    print(f"Retrieving local shapefile data from {file_path}")
    gdf = gpd.read_file(file_path)
    
    # Ensure the CRS is set to EPSG:28355
    if gdf.crs is None or gdf.crs.to_epsg() != 28355:
        gdf = gdf.to_crs(epsg=28355)
    
    # Create a rectangle from easting, northing, eastings_dim, and northings_dim
    min_easting, min_northing = easting - eastings_dim / 2, northing - northings_dim / 2
    max_easting, max_northing = easting + eastings_dim / 2, northing + northings_dim / 2
    rectangle = Polygon([(min_easting, min_northing), (max_easting, min_northing), (max_easting, max_northing), (min_easting, max_northing)])
    
    # Filter features within the specified rectangle
    gdf_filtered = gdf[gdf.geometry.intersects(rectangle)]
    
    print(f"Retrieved {len(gdf_filtered)} features within the rectangle")
    return gdf_filtered

def handle_contours(easting, northing, eastings_dim, northings_dim):
    file_path = "data/revised/shapefiles/contours/EL_CONTOUR_1TO5M.shp"
    print(f"Retrieving contour data from {file_path}")
    gdf = gpd.read_file(file_path)
    
    # Ensure the CRS is set to EPSG:28355
    if gdf.crs is None or gdf.crs.to_epsg() != 28355:
        gdf = gdf.to_crs(epsg=28355)
    
    # Create a rectangle from easting, northing, eastings_dim, and northings_dim
    min_easting, min_northing = easting - eastings_dim / 2, northing - northings_dim / 2
    max_easting, max_northing = easting + eastings_dim / 2, northing + northings_dim / 2
    rectangle = Polygon([(min_easting, min_northing), (max_easting, min_northing), (max_easting, max_northing), (min_easting, max_northing)])
    
    # Filter features that intersect with the rectangle
    gdf_filtered = gdf[gdf.geometry.intersects(rectangle)]
    
    print(f"Retrieved {len(gdf_filtered)} contour lines that intersect with the rectangle")
    
    # Check if ALTITUDE column exists
    if 'ALTITUDE' not in gdf_filtered.columns:
        raise ValueError("ALTITUDE column not found. Available columns: " + ", ".join(gdf_filtered.columns))
    
    # Extract points and elevations from the contour lines
    points = []
    elevations = []
    for _, row in gdf_filtered.iterrows():
        if isinstance(row.geometry, LineString):
            points.extend(list(row.geometry.coords))
            elevations.extend([float(row['ALTITUDE'])] * len(row.geometry.coords))
        elif isinstance(row.geometry, MultiLineString):
            for line in row.geometry.geoms:
                points.extend(list(line.coords))
                elevations.extend([float(row['ALTITUDE'])] * len(line.coords))
    
    points = np.array(points)
    elevations = np.array(elevations)
    
    # Create a grid for interpolation with 10m spacing
    x_min, y_min = easting - eastings_dim / 2, northing - northings_dim / 2
    x_max, y_max = easting + eastings_dim / 2, northing + northings_dim / 2
    resolution = 100  # This will create a 100x100 grid instead of 1000x1000
    grid_x, grid_y = np.mgrid[x_min:x_max:resolution*1j, y_min:y_max:resolution*1j]
    
    # Interpolate the elevations
    grid_z = griddata(points[:,:2], elevations, (grid_x, grid_y), method='linear')
    
    # Replace NaN values with the minimum elevation
    min_elevation = np.min(elevations)
    grid_z = np.nan_to_num(grid_z, nan=min_elevation)
    
    # Create a PyVista structured grid
    grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
    
    # Add the elevation data as a scalar field
    grid.point_data["Elevation"] = grid_z.ravel()
    
    # Create a boundary box
    boundary_box = pv.Box(bounds=(x_min, x_max, y_min, y_max, grid_z.min(), grid_z.max()))
    
    # Clip the mesh with the boundary box
    clipped_grid = grid.clip_box(boundary_box, invert=False)
    
    print("Created 3D terrain mesh from contour data")
    return clipped_grid, gdf_filtered


import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from rtree import index



#########

import geopandas as gpd
import numpy as np
from rasterio import features
from affine import Affine
from shapely.geometry import box
import pandas as pd
from scipy.spatial import cKDTree

def rasterize_shapefile(gdf, transform, out_shape):
    """Rasterize the shapefile with unique IDs for each geometry."""
    shapes = [(geom, i) for i, geom in enumerate(gdf.geometry)]
    raster = features.rasterize(shapes, out_shape=out_shape, transform=transform, fill=-1, all_touched=True)
    return raster

def raster_to_points(raster, affine):
    """Convert raster to 2D points, keeping only filled areas."""
    rows, cols = np.where(raster != -1)
    points = np.vstack((cols, rows)).T
    # Convert pixel coordinates to real-world coordinates
    real_coords = affine * points
    return real_coords, raster[rows, cols]

def assign_attributes(points, ids, gdf):
    """Assign attributes from the GeoDataFrame to the points."""
    # Create a DataFrame for the points with their IDs
    points_df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'id': ids})
    
    # Merge attributes based on ID
    merged_df = points_df.merge(gdf.reset_index(), left_on='id', right_index=True, how='left')
    
    # Drop unnecessary columns and return
    return merged_df.drop(['geometry', 'id'], axis=1)

def process_shapefile(shapefile_path, bbox, resolution=0.25):
    """Process a shapefile: read, crop, rasterize, extract points, and assign attributes."""
    # Read and crop shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf_cropped = gdf[gdf.geometry.intersects(bbox)]
    
    if gdf_cropped.empty:
        print(f"No geometries from {shapefile_path} intersect with the bounding box.")
        return None, None

    # Prepare for rasterization
    bounds = bbox.bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = Affine.translation(bounds[0], bounds[1]) * Affine.scale(resolution, resolution)
    
    # Rasterize
    raster = rasterize_shapefile(gdf_cropped, transform, (height, width))
    
    # Extract points
    points, ids = raster_to_points(raster, ~transform)
    
    # Assign attributes
    if len(points) > 0:
        attributes = assign_attributes(points, ids, gdf_cropped)
        return points, attributes
    else:
        print(f"No points extracted from {shapefile_path} after rasterization.")
        return None, None

def main(site_easting, site_northing, shapefile_paths, bounding_box_size):
    bbox = box(site_easting - bounding_box_size/2, site_northing - bounding_box_size/2,
               site_easting + bounding_box_size/2, site_northing + bounding_box_size/2)
    
    all_points = []
    all_attributes = []

    for shapefile_path in shapefile_paths:
        points, attributes = process_shapefile(shapefile_path, bbox)
        if points is not None:
            all_points.append(points)
            all_attributes.append(attributes)
    
    if all_points:
        # Combine all points and attributes
        combined_points = np.vstack(all_points)
        combined_attributes = pd.concat(all_attributes, ignore_index=True)
        
        return combined_points, combined_attributes
    else:
        print("No points were extracted from any shapefiles.")
        return None, None

# Example usage
if __name__ == "__main__":
    site_easting = 0  # replace with actual easting
    site_northing = 0  # replace with actual northing
    bounding_box_size = 1000  # replace with actual size
    shapefile_paths = ['path/to/shapefile1.shp', 'path/to/shapefile2.shp']  # replace with actual paths
    
    points, attributes = main(site_easting, site_northing, shapefile_paths, bounding_box_size)
    
    if points is not None:
        print(f"Extracted {len(points)} points with attributes:")
        print(attributes.head())
    else:
        print("No points were extracted.")

########

"""def efficient_point_in_polygon(polygon_gdf, points):
    # Create spatial index for the polygons
    spatial_index = index.Index()
    for idx, geometry in enumerate(polygon_gdf.geometry):
        spatial_index.insert(idx, geometry.bounds)

    # Function to check if a point is in any polygon
    def check_point(point):
        potential_matches_idx = list(spatial_index.intersection(point.coords[0]))
        return any(polygon_gdf.iloc[idx].geometry.contains(point) for idx in potential_matches_idx)

    # Apply the check to all points
    results = np.array([check_point(Point(x, y)) for x, y in points])
    
    return results

def plot_points_and_polygons(polygon_gdf, points, results):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot polygons
    polygon_gdf.plot(ax=ax, alpha=0.5, edgecolor='black')
    
    # Separate points inside and outside
    points_inside = points[results]
    points_outside = points[~results]
    
    # Plot points
    ax.scatter(points_inside[:, 0], points_inside[:, 1], c='green', label='Inside', alpha=0.6, s=10)
    ax.scatter(points_outside[:, 0], points_outside[:, 1], c='red', label='Outside', alpha=0.6, s=10)
    
    ax.set_title('Points Inside and Outside Polygons')
    ax.legend()
    plt.tight_layout()
    plt.show()




###
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from rtree import index
import time
import psutil

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def create_polygon_index(gdf):
    print("Starting to create spatial index for polygons...")
    start_time = time.time()
    idx = index.Index()
    for i, geometry in enumerate(gdf.geometry):
        idx.insert(i, geometry.bounds)
        if (i + 1) % 1000 == 0:
            print(f"Indexed {i + 1} polygons...")
    print(f"Spatial index created in {time.time() - start_time:.2f} seconds")
    log_memory_usage()
    return idx

def generate_grid_chunks(minx, miny, maxx, maxy, spacing, chunk_size=1000000):
    print(f"Generating grid chunks with spacing {spacing} and chunk size {chunk_size}")
    x_chunks = np.arange(minx, maxx, spacing * chunk_size)
    total_chunks = len(x_chunks)
    print(f"Total number of chunks to process: {total_chunks}")
    for i, x_start in enumerate(x_chunks):
        print(f"Generating chunk {i + 1}/{total_chunks}")
        x_end = min(x_start + spacing * chunk_size, maxx)
        x = np.arange(x_start, x_end, spacing)
        y = np.arange(miny, maxy, spacing)
        xx, yy = np.meshgrid(x, y)
        yield xx.ravel(), yy.ravel()

def process_large_grid(gdf, easting, northing, eastings_dim, northings_dim, spacing=0.25):
    print(f"Initializing grid processing...")
    print(f"Grid dimensions: {eastings_dim}x{northings_dim}, spacing: {spacing}")
    print(f"Center point: Easting {easting}, Northing {northing}")
    start_time = time.time()
    
    minx, miny = easting - eastings_dim/2, northing - northings_dim/2
    maxx, maxy = easting + eastings_dim/2, northing + northings_dim/2
    print(f"Grid bounds: ({minx}, {miny}) to ({maxx}, {maxy})")
    
    polygon_index = create_polygon_index(gdf)
    
    interior_points = []
    total_points = 0
    chunks_processed = 0
    
    print("Starting to process grid points...")
    for chunk_x, chunk_y in generate_grid_chunks(minx, miny, maxx, maxy, spacing):
        chunk_start_time = time.time()
        chunk_points = np.column_stack((chunk_x, chunk_y))
        chunk_size = len(chunk_points)
        total_points += chunk_size
        chunks_processed += 1
        
        print(f"\nProcessing chunk {chunks_processed}")
        print(f"Chunk size: {chunk_size} points")
        print(f"Chunk bounds: ({chunk_points[:, 0].min()}, {chunk_points[:, 1].min()}) to ({chunk_points[:, 0].max()}, {chunk_points[:, 1].max()})")
        
        chunk_interior_points = []
        for i, point in enumerate(chunk_points):
            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1}/{chunk_size} points in current chunk...")
            potential_matches_idx = list(polygon_index.intersection(point))
            if any(gdf.iloc[idx].geometry.contains(Point(point)) for idx in potential_matches_idx):
                chunk_interior_points.append(point)
        
        interior_points.extend(chunk_interior_points)
        
        chunk_time = time.time() - chunk_start_time
        print(f"Chunk {chunks_processed} completed in {chunk_time:.2f} seconds.")
        print(f"Points in this chunk: {chunk_size}")
        print(f"Interior points in this chunk: {len(chunk_interior_points)}")
        print(f"Total points processed so far: {total_points}")
        print(f"Total interior points found so far: {len(interior_points)}")
        print(f"Current processing rate: {chunk_size / chunk_time:.0f} points/second")
        print(f"Percentage of points inside (this chunk): {len(chunk_interior_points) / chunk_size * 100:.2f}%")
        print(f"Overall percentage of points inside: {len(interior_points) / total_points * 100:.2f}%")
        log_memory_usage()
        print("--------------------")
    
    total_time = time.time() - start_time
    print("\nProcessing completed!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total chunks processed: {chunks_processed}")
    print(f"Total points processed: {total_points}")
    print(f"Total interior points found: {len(interior_points)}")
    print(f"Overall processing rate: {total_points / total_time:.0f} points/second")
    print(f"Final percentage of points inside: {len(interior_points) / total_points * 100:.2f}%")
    log_memory_usage()
    
    return np.array(interior_points)


# Example usage
if __name__ == "__main__":
    site_name = 'uni'  # Melbourne Connect site
    easting, northing = get_site_coordinates(site_name)
    eastings_dim = 1000
    northings_dim = 1000
    
    # Retrieve tree canopy data
    gdf = handle_road_segments(easting, northing, eastings_dim, northings_dim)
    
    # Process the grid
    print("\nStarting grid processing...")
    interior_grid = process_large_grid(gdf, easting, northing, eastings_dim, northings_dim)
    
    print(f"\nFinal Results:")
    print(f"Total number of interior points: {len(interior_grid)}")
    print(f"Bounding box of interior points:")
    print(f"  Min X: {interior_grid[:, 0].min()}, Max X: {interior_grid[:, 0].max()}")
    print(f"  Min Y: {interior_grid[:, 1].min()}, Max Y: {interior_grid[:, 1].max()}")
    
    print("\nGenerating visualization...")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.plot(ax=ax, alpha=0.5, edgecolor='black')
    sample_size = min(10000, len(interior_grid))
    sample_points = interior_grid[np.random.choice(len(interior_grid), sample_size, replace=False)]
    ax.scatter(sample_points[:, 0], sample_points[:, 1], c='red', s=1, alpha=0.5)
    plt.title(f'Sample of Interior Grid Points (showing {sample_size} out of {len(interior_grid)})')
    plt.tight_layout()
    print("Saving visualization to 'grid_visualization.png'...")
    plt.savefig('grid_visualization.png', dpi=300)
    plt.close()
    
    print("\nProgram completed successfully!")
    log_memory_usage()"""






def plot_outline_data(gdf, plotter, color='white', opacity=0.5, line_width=2, label=None):
    print("Creating 3D outline visualization...")
    all_points = []
    all_faces = []
    total_points = 0
    extrusion_height = 1  # 1 meter extrusion

    for _, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, (Polygon, MultiPolygon)):
            if isinstance(geom, MultiPolygon):
                polygons = geom.geoms
            else:
                polygons = [geom]

            for polygon in polygons:
                exterior_coords = np.array(polygon.exterior.coords)
                if np.allclose(exterior_coords[0], exterior_coords[-1]):
                    exterior_coords = exterior_coords[:-1]

                if len(exterior_coords) < 3:
                    continue

                bottom = np.column_stack((exterior_coords, np.zeros(len(exterior_coords))))
                top = np.column_stack((exterior_coords, np.full(len(exterior_coords), extrusion_height)))
                points = np.vstack((bottom, top))

                num_points = len(exterior_coords)
                faces = []
                for i in range(num_points):
                    faces.extend([4, total_points + i, total_points + (i+1)%num_points, 
                                  total_points + num_points + (i+1)%num_points, total_points + num_points + i])

                all_points.append(points)
                all_faces.extend(faces)
                total_points += len(points)

                # Add interior rings (holes) if any
                for interior in polygon.interiors:
                    interior_coords = np.array(interior.coords)
                    if np.allclose(interior_coords[0], interior_coords[-1]):
                        interior_coords = interior_coords[:-1]

                    bottom = np.column_stack((interior_coords, np.zeros(len(interior_coords))))
                    top = np.column_stack((interior_coords, np.full(len(interior_coords), extrusion_height)))
                    points = np.vstack((bottom, top))

                    num_points = len(interior_coords)
                    faces = []
                    for i in range(num_points):
                        faces.extend([4, total_points + i, total_points + (i+1)%num_points, 
                                      total_points + num_points + (i+1)%num_points, total_points + num_points + i])

                    all_points.append(points)
                    all_faces.extend(faces)
                    total_points += len(points)

        elif isinstance(geom, (LineString, MultiLineString)):
            if isinstance(geom, MultiLineString):
                lines = geom.geoms
            else:
                lines = [geom]

            for line in lines:
                coords = np.array(line.coords)
                bottom = np.column_stack((coords, np.zeros(len(coords))))
                top = np.column_stack((coords, np.full(len(coords), extrusion_height)))
                points = np.vstack((bottom, top))

                num_points = len(coords)
                faces = []
                for i in range(num_points - 1):
                    faces.extend([4, total_points + i, total_points + i + 1, 
                                  total_points + num_points + i + 1, total_points + num_points + i])

                all_points.append(points)
                all_faces.extend(faces)
                total_points += len(points)

    # Combine all points and faces into a single mesh
    combined_points = np.vstack(all_points)
    combined_mesh = pv.PolyData(combined_points, np.array(all_faces))

    # Add the mesh to the plotter
    plotter.add_mesh(combined_mesh, color=color, opacity=opacity, show_edges=True, line_width=line_width, label=label)

    
    
    print(f"Created 3D visualization for {len(gdf)} geometries")

def plot_point_data(gdf, plotter, color='green', point_size=5, label=None):
    points = np.array([[point.x, point.y, 0] for point in gdf.geometry])
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(point_cloud, color=color, point_size=point_size, render_points_as_spheres=True, label=label)
    

def plot_terrain_mesh(terrain_mesh, plotter, cmap='terrain', opacity=1, show_edges=False, label=None):
    plotter.add_mesh(terrain_mesh, scalars="Elevation", cmap=cmap, opacity=opacity, show_edges=show_edges, label=label)


def main2():
    # Set up the parameters
    site_name = 'uni'  # Melbourne Connect site
    easting, northing = get_site_coordinates(site_name)
    eastings_dim = 1000
    northings_dim = 1000

    print(f"Testing for site: {site_name}")
    print(f"Easting: {easting}, Northing: {northing}")
    
    try:
        # Retrieve contour data and create terrain mesh
        terrain_mesh, contours_gdf = handle_contours(easting, northing, eastings_dim, northings_dim)

        plotter = pv.Plotter()

        if terrain_mesh is not None and "Elevation" in terrain_mesh.point_data:
            plot_terrain_mesh(terrain_mesh, plotter, label='Terrain')
        else:
            print("Warning: Terrain mesh could not be created or lacks elevation data.")

        if contours_gdf is not None:
            plot_outline_data(contours_gdf, plotter, color='black', opacity=1, line_width=1, label='Contour Lines')

        # Plot the site center
        center_point = pv.PolyData(np.array([[easting, northing, terrain_mesh.points[:, 2].mean()]]))
        plotter.add_mesh(center_point, color='red', point_size=10, render_points_as_spheres=True, label='Site Center')
        
        # Add text for site center
        plotter.add_point_labels(center_point, ['Site Center'], font_size=10, point_size=1)
        
        # Set up the camera
        plotter.camera_position = [
            (easting + eastings_dim / 2, northing + northings_dim / 2, max(eastings_dim, northings_dim)),  # Camera position
            (easting, northing, terrain_mesh.points[:, 2].mean()),  # Focal point
            (0, 0, 1)  # View up direction
        ]
        
        # Add a colorbar
        plotter.add_scalar_bar('Elevation', vertical=True)
        
        # Add a legend
        plotter.add_legend()
        
        print("Displaying visualization...")
        plotter.show()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()




def create_3d_building_visualization(gdf):
    print("Creating 3D building visualization...")
    building_data = []
    all_points = []
    all_faces = []
    total_points = 0
    
    for idx, row in gdf.iterrows():
        try:
            geom = row.geometry
            if isinstance(geom, MultiPolygon):
                geom = max(geom, key=lambda x: x.area)
            
            if isinstance(geom, Polygon):
                exterior_coords = np.array(geom.exterior.coords)
                if np.allclose(exterior_coords[0], exterior_coords[-1]):
                    exterior_coords = exterior_coords[:-1]
                
                if len(exterior_coords) < 3:
                    continue
                
                height = row.get('structure_extrusion') or row.get('footprint_extrusion') or (row.get('structure_max_elevation', 0) - row.get('structure_min_elevation', 0))
                
                bottom = np.column_stack((exterior_coords, np.zeros(len(exterior_coords))))
                top = np.column_stack((exterior_coords, np.full(len(exterior_coords), height)))
                points = np.vstack((bottom, top))
                
                num_points = len(exterior_coords)
                faces = []
                faces.extend([num_points] + list(range(total_points + num_points, total_points + 2*num_points)))
                for i in range(num_points):
                    faces.extend([4, total_points + i, total_points + (i+1)%num_points, 
                                  total_points + num_points + (i+1)%num_points, total_points + num_points + i])
                
                all_points.append(points)
                all_faces.extend(faces)
                total_points += len(points)
                
                top_cap_data = analyze_top_cap(points, faces[:num_points+1], bottom)
                building_data.append((idx, height, top_cap_data))
        except Exception as e:
            print(f"Error processing building {idx}: {e}")
    
    # Combine all points and faces into a single mesh
    combined_points = np.vstack(all_points)
    combined_mesh = pv.PolyData(combined_points, np.array(all_faces))
    
    df = pd.DataFrame(building_data, columns=['building_id', 'height', 'top_cap_data'])
    df = pd.concat([df, df['top_cap_data'].apply(pd.Series)], axis=1)
    df = df.drop('top_cap_data', axis=1)
    
    print(f"Created 3D visualization for {len(building_data)} buildings")
    return combined_mesh, df

def analyze_top_cap(points, face, bottom_coords):
    top_coords = points[face[1:]]
    top_area = polygon_area(top_coords)
    bottom_area = polygon_area(bottom_coords)
    top_centroid = np.mean(top_coords, axis=0)
    bottom_centroid = np.mean(bottom_coords, axis=0)
    max_distance = max(np.linalg.norm(p1 - p2) for p1 in top_coords for p2 in top_coords)
    
    return {
        "top_area": top_area,
        "bottom_area": bottom_area,
        "area_ratio": top_area / bottom_area if bottom_area > 0 else float('inf'),
        "centroid_offset": np.linalg.norm(top_centroid[:2] - bottom_centroid[:2]),
        "max_distance": max_distance,
        "num_points": len(top_coords)
    }

def polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0
