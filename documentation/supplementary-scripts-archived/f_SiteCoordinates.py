import pandas as pd
import os
from shapely.geometry import box, Polygon
import numpy as np
import xml.etree.ElementTree as ET
from pyproj import Transformer
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union

def parse_kml(kml_path):
    tree = ET.parse(kml_path)
    root = tree.getroot()
    
    tiles = {}
    for placemark in root.findall(".//{http://www.opengis.net/kml/2.2}Placemark"):
        name = placemark.find("{http://www.opengis.net/kml/2.2}name").text
        coords_text = placemark.find(".//{http://www.opengis.net/kml/2.2}coordinates").text
        coords = [tuple(map(float, coord.split(','))) for coord in coords_text.strip().split()]
        polygon = Polygon(coords)
        tiles[name] = polygon
    
    return tiles

def convert_to_latlon(easting, northing):
    transformer = Transformer.from_crs("EPSG:28355", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

def convert_to_gda55(lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28355", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing

def find_nearest_tile(target, tiles):
    point = Point(target[1], target[0])  # lon, lat
    nearest_tile = None
    min_distance = float('inf')
    
    for name, polygon in tiles.items():
        distance = polygon.exterior.distance(point)
        if distance < min_distance:
            min_distance = distance
            nearest_tile = name
    
    if nearest_tile:
        print(f"Nearest tile found: {nearest_tile} (distance: {min_distance:.6f} degrees)")
    else:
        print("No tiles found")
    
    return nearest_tile

def get_site_coordinates(site_name):
    # Read the CSV file using pandas
    df = pd.read_csv('data/revised/csv/site_locations.csv')
    
    # Find the row corresponding to the given site_name
    site_row = df[df['Name'] == site_name]
    
    if site_row.empty:
        raise ValueError(f"Site '{site_name}' not found in the CSV file.")
    
    # Print the selected site information
    print(f"Selected site information:")
    print(site_row.to_string(index=False))
    
    # Get the Easting and Northing coordinates
    easting = float(site_row['Easting'].values[0])
    northing = float(site_row['Northing'].values[0])
    
    # Print the retrieved Easting and Northing
    print(f"Retrieved coordinates for site '{site_name}':")
    print(f"Easting: {easting}")
    print(f"Northing: {northing}")
    
    return easting, northing

def get_tile_names(kml_path, tiles_folder, center, x_dim, y_dim, subfolder=False):
    """
    Get the tile names within a bounding box centered on the given center point.

    Parameters:
    - kml_path: Path to the KML file.
    - tiles_folder: Path to the folder containing the tiles.
    - center: Tuple of (easting, northing) coordinates for the center point.
    - x_dim: Width of the bounding box in meters.
    - y_dim: Height of the bounding box in meters.
    - subfolder: Boolean indicating if tiles are stored in subfolders.

    Returns:
    - List of file paths for the tiles within the bounding box.
    """
    # Parse the KML file to get the tiles
    tiles = parse_kml(kml_path)
    
    # Convert center point to lat/lon
    lat, lon = convert_to_latlon(center[0], center[1])
    
    # Create the bounding box centered on the center point
    half_x_dim = x_dim / 2
    half_y_dim = y_dim / 2
    site_bbox = box(center[0] - half_x_dim, center[1] - half_y_dim, center[0] + half_x_dim, center[1] + half_y_dim)
    
    # Find tiles that intersect with the site bounding box
    intersecting_tiles = []
    for name, polygon in tiles.items():
        if polygon.intersects(site_bbox):
            intersecting_tiles.append(name)
    
    # Create file paths for the intersecting tiles
    tile_files = []
    for tile_name in intersecting_tiles:
        if subfolder:
            tile_path = os.path.join(tiles_folder, tile_name, f"{tile_name}.{tile_name.split('.')[-1]}")
        else:
            tile_path = os.path.join(tiles_folder, f"{tile_name}.{tile_name.split('.')[-1]}")
        
        if os.path.exists(tile_path):
            tile_files.append(tile_path)
        else:
            print(f"Warning: Tile file not found: {tile_path}")
    
    print(f"Found {len(tile_files)} tiles within the bounding box.")
    return tile_files

def find_tiles_within_bbox(easting, northing, tiles, x_dim_meters, y_dim_meters):
    """
    Find tiles that intersect with a bounding box centered on the given easting/northing.

    Parameters:
    - easting: Easting of the center point.
    - northing: Northing of the center point.
    - tiles: Dictionary of tile names and their polygons.
    - x_dim_meters: Width of the bounding box in meters.
    - y_dim_meters: Height of the bounding box in meters.

    Returns:
    - List of tile names that intersect with the bounding box.
    """
    # Convert center point to lat/lon
    lat, lon = convert_to_latlon(easting, northing)
    
    # Approximate conversion from meters to degrees (1 degree ~ 111,000 meters)
    x_dim_degrees = x_dim_meters / 111000
    y_dim_degrees = y_dim_meters / 111000
    
    half_x_dim = x_dim_degrees / 2
    half_y_dim = y_dim_degrees / 2
    bbox = box(lon - half_x_dim, lat - half_y_dim, lon + half_x_dim, lat + half_y_dim)
    
    intersecting_tiles = []
    for name, polygon in tiles.items():
        if polygon.intersects(bbox):
            intersecting_tiles.append(name)
    
    return intersecting_tiles

def confirm_coverage(easting, northing, selected_tiles, tiles, x_dim_meters, y_dim_meters):
    """
    Confirm that the selected tiles cover the specified site by comparing bounding boxes and plot the bounding boxes.

    Parameters:
    - easting: Easting of the center point.
    - northing: Northing of the center point.
    - selected_tiles: List of selected tile names.
    - tiles: Dictionary of all tile names and their polygons.
    - x_dim_meters: Width of the bounding box in meters.
    - y_dim_meters: Height of the bounding box in meters.

    Returns:
    - Boolean indicating whether the tiles cover the specified site.
    """
    # Convert site bounding box to GDA Zone 55
    half_x_dim = x_dim_meters / 2
    half_y_dim = y_dim_meters / 2
    site_bbox = box(easting - half_x_dim, northing - half_y_dim, easting + half_x_dim, northing + half_y_dim)
    
    # Plot the bounding boxes
    fig, ax = plt.subplots()
    
    # Plot the site bounding box
    x, y = site_bbox.exterior.xy
    ax.plot(x, y, color='blue', linewidth=2, label='Site Bounding Box')
    
    # Plot only the selected tile bounding boxes
    for tile_name in selected_tiles:
        if tile_name in tiles:
            polygon = tiles[tile_name]
            gda55_coords = [convert_to_gda55(coord[1], coord[0]) for coord in polygon.exterior.coords]
            tile_bbox = Polygon(gda55_coords)
            x, y = tile_bbox.exterior.xy
            ax.fill(x, y, color='grey', alpha=0.5)
    
    ax.set_title('Site and Selected Tile Bounding Boxes')
    ax.legend()
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.show()
    
    # Check if the site bounding box is within any of the selected tile bounding boxes
    for tile_name in selected_tiles:
        if tile_name in tiles:
            polygon = tiles[tile_name]
            gda55_coords = [convert_to_gda55(coord[1], coord[0]) for coord in polygon.exterior.coords]
            tile_bbox = Polygon(gda55_coords)
            if tile_bbox.contains(site_bbox):
                return True
    
    return False

def check_tile_existence(las_folder, selected_tiles):
    """
    Check the existence of selected tile files in the las_folder.

    Parameters:
    - las_folder: Path to the folder containing the LAS tiles.
    - selected_tiles: List of selected tile names (without extensions).

    Returns:
    - existing_files: List of existing tile file paths.
    - missing_files: List of tile names for missing files.
    """
    existing_files = []
    missing_files = []
    
    for tile_name in selected_tiles:
        las_file_path = os.path.join(las_folder, f"{tile_name}.las")  # Assuming LAS files have a .las extension
        if os.path.exists(las_file_path):
            existing_files.append(las_file_path)
        else:
            missing_files.append(tile_name)
    
    # Output results
    if missing_files:
        print(f"Missing {len(missing_files)} tile(s): {missing_files}")
    else:
        print("All selected tiles exist.")
    
    return existing_files, missing_files

def main():
    site_name = 'city'  # Example site name
    easting, northing = get_site_coordinates(site_name)
    print(f"Retrieved coordinates for site '{site_name}':")
    print(f"Easting: {easting}")
    print(f"Northing: {northing}")

    las_folder = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/processedLAS"
    kml_path = "/Users/alexholland/Coding/volumetric-scenarios-rhino-bim-gia/data/revised/LAS/Tile_Index.KML"

    tiles = parse_kml(kml_path)
    print(f"Parsed KML file. Found {len(tiles)} tiles.")

    x_dim_meters = 800  # Example width in meters
    y_dim_meters = 800  # Example height in meters
    selected_tiles = find_tiles_within_bbox(easting, northing, tiles, x_dim_meters, y_dim_meters)
    print(f"Found {len(selected_tiles)} tiles within the bounding box of {x_dim_meters} x {y_dim_meters} meters.")
    print(f"Selected tile names: {selected_tiles}")

    # Check tile existence
    existing_files, missing_files = check_tile_existence(las_folder, selected_tiles)

    # Confirm coverage
    confirm_coverage(easting, northing, selected_tiles, tiles, x_dim_meters, y_dim_meters)

import pyvista as pv
import numpy as np

def get_center_and_dims(polydata_list, eastings_offset=0, northings_offset=0):
    # Initialize with extreme values
    xmin_global = float('inf')
    xmax_global = float('-inf')
    ymin_global = float('inf')
    ymax_global = float('-inf')
    
    for i, polydata in enumerate(polydata_list):
        print(f"Processing PolyData object {i+1}...")

        # Get the bounds of the PolyData: (xmin, xmax, ymin, ymax, zmin, zmax)
        bounds = polydata.bounds
        xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]

        # Update global bounds
        xmin_global = min(xmin_global, xmin)
        xmax_global = max(xmax_global, xmax)
        ymin_global = min(ymin_global, ymin)
        ymax_global = max(ymax_global, ymax)
        
        print(f"Bounds for PolyData {i+1}: xmin = {xmin}, xmax = {xmax}, ymin = {ymin}, ymax = {ymax}")
    
    # After processing all polydata, calculate the global center and dimensions
    center_eastings = (xmin_global + xmax_global) / 2 + eastings_offset
    center_northings = (ymin_global + ymax_global) / 2 + northings_offset
    
    eastDim = (xmax_global - xmin_global)
    northDim = (ymax_global - ymin_global)

    print(f"Combined bounds: xmin = {xmin_global}, xmax = {xmax_global}, ymin = {ymin_global}, ymax = {ymax_global}")
    print(f"Combined Center: Eastings (x) = {center_eastings}, Northings (y) = {center_northings}")
    print(f"Combined Dimensions: EastDim = {eastDim}, NorthDim = {northDim}")
    
    # Return the combined center and dimensions
    return center_eastings, center_northings, eastDim, northDim

if __name__ == "__main__":
    main()
